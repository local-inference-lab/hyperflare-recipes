# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for on-the-fly ZeRO-3 attention weight gathering across BF16, MXFP8, and Block-FP8."""

from __future__ import annotations

import pytest
import torch

from hyperflare_recipes.components.block_fp8 import BlockFP8Weight
from hyperflare_recipes.recipes.glm53.weight_gather import (
    AttentionWeightGatherer,
    _mxfp8_scale_phys,
    full_kda_template,
    full_mla_template,
)
from hyperflare_recipes.components.weights import GroupedLinearWeights


class _StubGather:
    def __init__(self, source, output, *, runtime, session, coordinator, name):
        self.runtime, self.output, self.name = runtime, output, name
        self.single = False
        self.pcie = True
        self.plan = None

    def functional(self, *, session):
        self.output = None
        return self


def _make_full_kda_bf16(device: torch.device, seed: int = 7) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    return {
        "q": torch.randn((8192, 4096), dtype=torch.bfloat16, device=device, generator=g),
        "k": torch.randn((8192, 4096), dtype=torch.bfloat16, device=device, generator=g),
        "v": torch.randn((8192, 4096), dtype=torch.bfloat16, device=device, generator=g),
        "b": torch.randn((64, 4096), dtype=torch.bfloat16, device=device, generator=g),
        "f_a": torch.randn((128, 4096), dtype=torch.bfloat16, device=device, generator=g),
        "g_a_proj": torch.randn((128, 4096), dtype=torch.bfloat16, device=device, generator=g),
        "f_b_proj": torch.randn((8192, 128), dtype=torch.bfloat16, device=device, generator=g),
        "g_b_proj": torch.randn((8192, 128), dtype=torch.bfloat16, device=device, generator=g),
        "o_proj": torch.randn((4096, 8192), dtype=torch.bfloat16, device=device, generator=g),
        "o_norm.weight": torch.randn((128,), dtype=torch.bfloat16, device=device, generator=g),
        "q_conv": torch.randn((8192, 4), dtype=torch.bfloat16, device=device, generator=g),
        "k_conv": torch.randn((8192, 4), dtype=torch.bfloat16, device=device, generator=g),
        "v_conv": torch.randn((8192, 4), dtype=torch.bfloat16, device=device, generator=g),
        "dt_bias": torch.randn((8192,), dtype=torch.float32, device=device, generator=g),
        "A_log": torch.randn((64,), dtype=torch.float32, device=device, generator=g),
    }


def _shard_kda_bf16(full: dict[str, torch.Tensor], rank: int, world: int) -> dict[str, object]:
    ch = 8192 // world
    lh = 64 // world
    c0, c1 = rank * ch, (rank + 1) * ch
    h0, h1 = rank * lh, (rank + 1) * lh
    inp = torch.cat((full["q"][c0:c1], full["k"][c0:c1], full["v"][c0:c1], full["b"][h0:h1], full["f_a"]), dim=0)
    conv = torch.cat((full["q_conv"][c0:c1], full["k_conv"][c0:c1], full["v_conv"][c0:c1]), dim=0)
    return {
        "input_proj": inp.contiguous(),
        "g_a_proj": full["g_a_proj"],
        "f_b_proj": full["f_b_proj"][c0:c1].contiguous(),
        "g_b_proj": full["g_b_proj"][c0:c1].contiguous(),
        "o_proj": full["o_proj"][:, c0:c1].contiguous(),
        "o_norm.weight": full["o_norm.weight"],
        "conv1d.weight": conv.contiguous(),
        "dt_bias": full["dt_bias"][c0:c1].contiguous(),
        "A_log": full["A_log"][h0:h1].contiguous(),
    }


def _pack_mxfp8(tensor_bf16: torch.Tensor):
    from b12x.gemm.blockscaled._linear import pack_mxfp8_linear_weight

    n, k = tensor_bf16.shape
    vals = tensor_bf16.to(torch.float8_e4m3fn).contiguous()
    scales = (torch.arange(n * (k // 32), device=tensor_bf16.device) % 7 + 124).to(torch.uint8).reshape(n, k // 32)
    return pack_mxfp8_linear_weight(vals, scales.view(torch.float8_e8m0fnu))


def _install_simulated_gather(
    monkeypatch,
    gatherer: AttentionWeightGatherer,
    rank_weights_per_layer: list[list[dict[str, object]]],
):
    """Simulate multi-rank weight gather (both 1D and 2D col_row_bytes) in-process."""
    import hyperflare_recipes.components.collectives as _coll
    import hyperflare_recipes.recipes.glm53.weight_gather as wg

    world = gatherer.world
    rank_gatherers = []
    for r in range(world):
        rg = object.__new__(AttentionWeightGatherer)
        rg.layer_weights = tuple(layer_ranks[r] for layer_ranks in rank_weights_per_layer)
        rg.layer_types = gatherer.layer_types
        rg.world = world
        rg._kda_ch = gatherer._kda_ch
        rg._kda_lh = gatherer._kda_lh
        rg._kda_r_b = gatherer._kda_r_b
        rg._kda_r_fb = gatherer._kda_r_fb
        rg._kda_r_gb = gatherer._kda_r_gb
        rg._kda_r_conv = gatherer._kda_r_conv
        rg._kda_r_dt = gatherer._kda_r_dt
        rg._kda_r_alog = gatherer._kda_r_alog
        rg._kda_small_rows = gatherer._kda_small_rows
        rank_gatherers.append(rg)

    ptr_map: dict[tuple[int, tuple[int, ...], int], list[torch.Tensor]] = {}

    def register(tensors_per_rank: list[torch.Tensor]):
        t0 = tensors_per_rank[0]
        ptr_map[(t0.data_ptr(), tuple(t0.shape), t0.storage_offset())] = tensors_per_rank

    for layer_idx, (ltype, rank_weights_list) in enumerate(zip(gatherer.layer_types, rank_weights_per_layer, strict=True)):
        spec = gatherer._layer_ops[layer_idx]
        if ltype == "linear_attention":
            ch = gatherer._kda_ch
            if isinstance(rank_weights_list[0]["input_proj"], torch.Tensor):
                for s0, s1 in ((0, ch), (ch, 2 * ch), (2 * ch, 3 * ch), (0, 2 * ch), (0, 3 * ch)):
                    register([rw["input_proj"][s0:s1] for rw in rank_weights_list])
                register([rw["o_proj"] for rw in rank_weights_list])
            else:
                mt = ch // 128
                for s0, s1 in ((0, ch), (ch, 2 * ch), (2 * ch, 3 * ch), (0, 2 * ch), (0, 3 * ch)):
                    register([rw["input_proj"].parts[0].weight.values[s0:s1] for rw in rank_weights_list])
                for s0, s1 in ((0, mt), (mt, 2 * mt), (2 * mt, 3 * mt), (0, 3 * mt)):
                    register([_mxfp8_scale_phys(rw["input_proj"].parts[0])[0][s0:s1] for rw in rank_weights_list])
                register([rw["o_proj"].weight.values for rw in rank_weights_list])
                register([_mxfp8_scale_phys(rw["o_proj"])[0] for rw in rank_weights_list])
        else:
            if isinstance(rank_weights_list[0]["q_b_proj"], torch.Tensor):
                register([rw["q_b_proj"] for rw in rank_weights_list])
            else:
                register([rw["q_b_proj"].values for rw in rank_weights_list])
                register([rw["q_b_proj"].scales for rw in rank_weights_list])
            if isinstance(rank_weights_list[0]["kv_b_proj"], torch.Tensor):
                register([rw["kv_b_proj"] for rw in rank_weights_list])
            else:
                register([rw["kv_b_proj"].weight.values for rw in rank_weights_list])
                register([rw["kv_b_proj"].weight.scale_rows[0].view(torch.uint8) for rw in rank_weights_list])
            if isinstance(rank_weights_list[0]["o_proj"], torch.Tensor):
                if spec["mla_o_split"]:
                    register([rw["o_proj"][:2048] for rw in rank_weights_list])
                    register([rw["o_proj"][2048:] for rw in rank_weights_list])
                else:
                    register([rw["o_proj"] for rw in rank_weights_list])
            else:
                register([rw["o_proj"].values for rw in rank_weights_list])
                register([rw["o_proj"].scales for rw in rank_weights_list])

    small_counter = [0]

    def fake_gather(source, output, channel_token, dependencies, row_start, row_end, op_handle, col_row_bytes):
        del channel_token, dependencies
        op = _coll._OWNERS[op_handle]
        key = (source.data_ptr(), tuple(source.shape), source.storage_offset())
        if key in ptr_map:
            shards = ptr_map[key]
        else:
            li = small_counter[0] % len(rank_weights_per_layer)
            small_counter[0] += 1
            shards = [rg._pack_kda_small(rg.layer_weights[li]) for rg in rank_gatherers]
        dst = output if row_end < 0 else output[row_start:row_end]
        if col_row_bytes:
            # Emulate _RoceScatterGatherLaunch._gather_dst_offset 2D column-concat gather.
            shard_bytes = [s.contiguous().view(torch.uint8).reshape(-1, col_row_bytes) for s in shards]
            gathered_u8 = torch.stack(shard_bytes, dim=1).reshape(-1)
            dst.view(torch.uint8).reshape(-1).copy_(gathered_u8)
        else:
            gathered = torch.cat([s.view(torch.bfloat16).reshape(op.src_shape) for s in shards], dim=0)
            dst.view(torch.bfloat16).reshape(op.dst_shape).copy_(gathered)

    monkeypatch.setattr(wg, "ordered_gather_shards", fake_gather)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_full_kda_template_preserves_per_layer_replicated_weights():
    """Template must exclude per-layer replicated g_a_proj and o_norm.weight."""
    device = torch.device("cuda:0")
    layer0 = _shard_kda_bf16(_make_full_kda_bf16(device, seed=1), 0, 2)
    layer1 = _shard_kda_bf16(_make_full_kda_bf16(device, seed=2), 0, 2)

    tpl = full_kda_template(layer0, world_size=2, device=device)
    assert "g_a_proj" not in tpl
    assert "o_norm.weight" not in tpl

    merged1 = {**layer1, **tpl}
    assert torch.equal(merged1["g_a_proj"], layer1["g_a_proj"])
    assert torch.equal(merged1["o_norm.weight"], layer1["o_norm.weight"])
    assert not torch.equal(merged1["g_a_proj"], layer0["g_a_proj"])
    assert not torch.equal(merged1["o_norm.weight"], layer0["o_norm.weight"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("roce_2d,max_gb", [(False, 1 << 30), (True, 128 << 20), (True, 256 << 20)])
def test_kda_bf16_weight_gather_bit_exact(monkeypatch, world, roce_2d, max_gb):
    device = torch.device("cuda:0")
    full = _make_full_kda_bf16(device)
    rank_weights = [_shard_kda_bf16(full, r, world) for r in range(world)]

    tpl = full_kda_template(rank_weights[0], world_size=world, device=device)
    assert tpl["input_proj"].shape == (24768, 4096)
    assert tpl["o_proj"].shape == (4096, 8192)

    import hyperflare_recipes.recipes.glm53.weight_gather as _wg
    monkeypatch.setattr(_wg, "PreparedGatherShards", _StubGather)
    gatherer = AttentionWeightGatherer(
        [rank_weights[0]], ["linear_attention"], world_size=world,
        runtime=None, coordinator=None, session=None, name="test", device=device,
    )
    if roce_2d:
        gatherer.use_roce_2d = True
        gatherer.max_gather_bytes = max_gb
        gatherer._layer_ops[0] = gatherer._declare_kda_shapes(
            rank_weights[0], runtime=None, coordinator=None, session=None, name="test")
        assert gatherer._layer_ops[0]["qkv_coalesce"] == (3 if max_gb >= (192 << 20) else 2)
    _install_simulated_gather(monkeypatch, gatherer, [rank_weights])
    gathered = gatherer.gather_sync(0)

    expected_inp = torch.cat((full["q"], full["k"], full["v"], full["b"], full["f_a"]), dim=0)
    expected_conv = torch.cat((full["q_conv"], full["k_conv"], full["v_conv"]), dim=0)
    assert torch.equal(gathered.input_proj, expected_inp)
    assert torch.equal(gathered.f_b_proj, full["f_b_proj"])
    assert torch.equal(gathered.g_b_proj, full["g_b_proj"])
    assert torch.equal(gathered.o_proj, full["o_proj"])
    assert torch.equal(gathered.conv1d, expected_conv)
    assert torch.equal(gathered.dt_bias, full["dt_bias"])
    assert torch.equal(gathered.A_log, full["A_log"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("roce_2d,max_gb", [(False, 1 << 30), (True, 64 << 20), (True, 128 << 20)])
def test_kda_mxfp8_weight_gather_bit_exact(monkeypatch, world, roce_2d, max_gb):
    device = torch.device("cuda:0")
    full = _make_full_kda_bf16(device, seed=19)
    full_qkv = _pack_mxfp8(torch.cat((full["q"], full["k"], full["v"]), dim=0))
    full_o = _pack_mxfp8(full["o_proj"])

    ch, lh = 8192 // world, 64 // world
    rank_weights = []
    for r in range(world):
        c0, c1 = r * ch, (r + 1) * ch
        h0, h1 = r * lh, (r + 1) * lh
        qkv_r_vals = torch.cat((
            full_qkv.weight.values[c0:c1],
            full_qkv.weight.values[8192 + c0:8192 + c1],
            full_qkv.weight.values[16384 + c0:16384 + c1],
        ), dim=0).contiguous()
        full_qkv_sc = _mxfp8_scale_phys(full_qkv)[0]
        mt = ch // 128
        m0, m1 = r * mt, (r + 1) * mt
        qkv_r_sc = torch.cat((
            full_qkv_sc[m0:m1], full_qkv_sc[64 + m0:64 + m1], full_qkv_sc[128 + m0:128 + m1],
        ), dim=0).unsqueeze(0).contiguous()
        from hyperflare_recipes.recipes.glm53.weight_gather import _mxfp8_from_parts
        p0_r = _mxfp8_from_parts(qkv_r_vals, qkv_r_sc, in_features=4096, padded_in_features=4096, out_features=3 * ch)
        p1_r = torch.cat((full["b"][h0:h1], full["f_a"]), dim=0).contiguous()
        o_r_vals = full_o.weight.values[:, c0:c1].contiguous()
        full_o_sc = _mxfp8_scale_phys(full_o)
        kt = ch // 128
        o_r_sc = full_o_sc[:, :, r * kt:(r + 1) * kt].contiguous()
        o_r = _mxfp8_from_parts(o_r_vals, o_r_sc, in_features=ch, padded_in_features=ch, out_features=4096)
        conv_r = torch.cat((full["q_conv"][c0:c1], full["k_conv"][c0:c1], full["v_conv"][c0:c1]), dim=0).contiguous()
        rank_weights.append({
            "input_proj": GroupedLinearWeights(parts=(p0_r, p1_r), widths=(3 * ch, lh + 128)),
            "g_a_proj": full["g_a_proj"],
            "f_b_proj": full["f_b_proj"][c0:c1].contiguous(),
            "g_b_proj": full["g_b_proj"][c0:c1].contiguous(),
            "o_proj": o_r,
            "o_norm.weight": full["o_norm.weight"],
            "conv1d.weight": conv_r,
            "dt_bias": full["dt_bias"][c0:c1].contiguous(),
            "A_log": full["A_log"][h0:h1].contiguous(),
        })

    tpl = full_kda_template(rank_weights[0], world_size=world, device=device)
    assert isinstance(tpl["input_proj"], GroupedLinearWeights)

    import hyperflare_recipes.recipes.glm53.weight_gather as _wg
    monkeypatch.setattr(_wg, "PreparedGatherShards", _StubGather)
    gatherer = AttentionWeightGatherer(
        [rank_weights[0]], ["linear_attention"], world_size=world,
        runtime=None, coordinator=None, session=None, name="test", device=device,
    )
    if roce_2d:
        gatherer.use_roce_2d = True
        gatherer.max_gather_bytes = max_gb
        gatherer._layer_ops[0] = gatherer._declare_kda_shapes(
            rank_weights[0], runtime=None, coordinator=None, session=None, name="test")
        assert gatherer._layer_ops[0]["qkv_coalesce"] == (3 if max_gb >= (96 << 20) else 2)
        assert gatherer._layer_ops[0]["qkv_sc_bulk"] is not None
    _install_simulated_gather(monkeypatch, gatherer, [rank_weights])
    gathered = gatherer.gather_sync(0)

    assert torch.equal(gathered.input_proj.parts[0].weight.values.view(torch.uint8),
                       full_qkv.weight.values.view(torch.uint8))
    assert torch.equal(_mxfp8_scale_phys(gathered.input_proj.parts[0]), _mxfp8_scale_phys(full_qkv))
    assert torch.equal(gathered.input_proj.parts[1], torch.cat((full["b"], full["f_a"]), dim=0))
    assert torch.equal(gathered.o_proj.weight.values.view(torch.uint8), full_o.weight.values.view(torch.uint8))
    assert torch.equal(_mxfp8_scale_phys(gathered.o_proj), _mxfp8_scale_phys(full_o))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("split_o", [False, True])
@pytest.mark.parametrize("roce_2d", [False, True])
def test_mla_bf16_and_blockfp8_weight_gather_bit_exact(monkeypatch, world, split_o, roce_2d):
    device = torch.device("cuda:0")
    g = torch.Generator(device=device).manual_seed(31)

    # 1. BF16 MLA
    qb_full = torch.randn((16384, 1536), dtype=torch.bfloat16, device=device, generator=g)
    kvb_full = torch.randn((32768, 512), dtype=torch.bfloat16, device=device, generator=g)
    o_full = torch.randn((4096, 16384), dtype=torch.bfloat16, device=device, generator=g)
    rank_bf16 = []
    for r in range(world):
        q0, q1 = r * (16384 // world), (r + 1) * (16384 // world)
        k0, k1 = r * (32768 // world), (r + 1) * (32768 // world)
        rank_bf16.append({
            "q_b_proj": qb_full[q0:q1].contiguous(),
            "kv_b_proj": kvb_full[k0:k1].contiguous(),
            "o_proj": o_full[:, q0:q1].contiguous(),
        })
    import hyperflare_recipes.recipes.glm53.weight_gather as _wg
    monkeypatch.setattr(_wg, "PreparedGatherShards", _StubGather)
    gatherer = AttentionWeightGatherer(
        [rank_bf16[0]], ["indexed_attention"], world_size=world,
        runtime=None, coordinator=None, session=None, name="test_bf16", device=device,
    )
    if split_o or roce_2d:
        gatherer.use_roce_2d = roce_2d
        if split_o:
            gatherer.max_gather_bytes = 64 << 20
        gatherer._ops.clear()
        gatherer._gathers.clear()
        gatherer._layer_ops[0] = gatherer._declare_mla_shapes(
            rank_bf16[0], runtime=None, coordinator=None, session=None, name="test_bf16")
        assert gatherer._layer_ops[0]["mla_o_split"] == split_o
    _install_simulated_gather(monkeypatch, gatherer, [rank_bf16])
    got = gatherer.gather_sync(0)
    assert torch.equal(got.q_b_proj, qb_full)
    assert torch.equal(got.absorb, kvb_full.view(64, 512, 512)[:, :256])
    assert torch.equal(got.expand, kvb_full.view(64, 512, 512)[:, 256:])
    assert torch.equal(got.o_proj, o_full)

    # 2. Block-FP8 MLA (matches nvidia/GLM-5.3-Flash-FP4-Local)
    qb_v = qb_full.to(torch.float8_e4m3fn)
    qb_s = torch.randn((128, 12), dtype=torch.float32, device=device, generator=g)
    o_v = o_full.to(torch.float8_e4m3fn)
    o_s = torch.randn((32, 128), dtype=torch.float32, device=device, generator=g)
    rank_fp8 = []
    for r in range(world):
        q0, q1 = r * (16384 // world), (r + 1) * (16384 // world)
        qs0, qs1 = r * (128 // world), (r + 1) * (128 // world)
        k0, k1 = r * (32768 // world), (r + 1) * (32768 // world)
        rank_fp8.append({
            "q_b_proj": BlockFP8Weight(qb_v[q0:q1].contiguous(), qb_s[qs0:qs1].contiguous()),
            "kv_b_proj": kvb_full[k0:k1].contiguous(),
            "o_proj": BlockFP8Weight(o_v[:, q0:q1].contiguous(), o_s[:, qs0:qs1].contiguous()),
        })
    tpl = full_mla_template(rank_fp8[0], world_size=world, device=device)
    assert isinstance(tpl["q_b_proj"], BlockFP8Weight)
    assert isinstance(tpl["o_proj"], BlockFP8Weight)

    gatherer_fp8 = AttentionWeightGatherer(
        [rank_fp8[0]], ["indexed_attention"], world_size=world,
        runtime=None, coordinator=None, session=None, name="test_fp8", device=device,
    )
    gatherer_fp8.use_roce_2d = roce_2d
    _install_simulated_gather(monkeypatch, gatherer_fp8, [rank_fp8])
    got_fp8 = gatherer_fp8.gather_sync(0)
    assert torch.equal(got_fp8.q_b_proj.values.view(torch.uint8), qb_v.view(torch.uint8))
    assert torch.equal(got_fp8.q_b_proj.scales, qb_s)
    assert torch.equal(got_fp8.o_proj.values.view(torch.uint8), o_v.view(torch.uint8))
    assert torch.equal(got_fp8.o_proj.scales, o_s)
