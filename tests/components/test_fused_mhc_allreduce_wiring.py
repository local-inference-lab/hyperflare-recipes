# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU, FX-tracing, and mock multi-rank 0-ULP tests for PreparedFusedMhcAllReducePostYNorm."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from hyperflare_recipes.components.mhc import MHCWeights
from hyperflare_recipes.components.mhc_overlap import (
    PendingMHCSinkhorn,
    PreparedFusedMhcAllReducePostYNorm,
    pack_mxfp8_scale_mma,
    supports_fused_mhc_allreduce,
)
from hyperflare_recipes.recipes.deepseek41 import decoder


# ---------------------------------------------------------------------------
# Mock PCIe Oneshot runtime & Lamport sentinel channel
# ---------------------------------------------------------------------------


@dataclass
class _MockOneshotState:
    eager_tables: tuple[int, int] | None
    eager_buffer_bytes: int | None
    signal_table_address: int = 0x510000
    plain_payload_ready_region_packs: int = 64
    eager_slot: int = 0
    device_slot_selection: bool = False
    slot_bias: int = 0


class _MockOneshotExt:
    def __init__(self, state: _MockOneshotState):
        self._st = state

    def _state(self, ptr: int) -> _MockOneshotState:
        assert ptr == 1
        return self._st


class _MockOneshotChannel:
    def __init__(
        self,
        *,
        world_size: int = 4,
        rank: int = 0,
        max_size: int = 4 * 1024 * 1024,
        eager_buffer_bytes: int | None = 4 * 1024 * 1024,
        eager_tables: tuple[int, int] | None = (0x100000, 0x200000),
        plain_payload_ready_region_packs: int = 64,
        device: torch.device | str = "cpu",
    ):
        self.world_size = world_size
        self.rank = rank
        self.max_size = max_size
        self.device = torch.device(device)
        self._state = _MockOneshotState(
            eager_tables=eager_tables,
            eager_buffer_bytes=eager_buffer_bytes,
            plain_payload_ready_region_packs=plain_payload_ready_region_packs,
        )
        self._ext = _MockOneshotExt(self._state)
        self._ptr = 1


class _MockPCIeAllReduceRuntime:
    def __init__(
        self,
        *,
        algorithm: str = "oneshot",
        single_channel: bool = True,
        world_size: int = 4,
        rank: int = 0,
        max_size: int = 4 * 1024 * 1024,
        eager_buffer_bytes: int | None = 4 * 1024 * 1024,
        eager_tables: tuple[int, int] | None = (0x100000, 0x200000),
        plain_payload_ready_region_packs: int = 64,
        device: torch.device | str = "cpu",
    ):
        self.algorithm = algorithm
        self.single_channel = single_channel
        self.world_size = world_size
        self.rank = rank
        self.max_size = max_size
        self.device = torch.device(device)
        self.channel = _MockOneshotChannel(
            world_size=world_size,
            rank=rank,
            max_size=max_size,
            eager_buffer_bytes=eager_buffer_bytes,
            eager_tables=eager_tables,
            plain_payload_ready_region_packs=plain_payload_ready_region_packs,
            device=device,
        )

    def for_stream(self, stream=None, *, channel_id=None):
        return self.channel


def _mock_sentinel_params(channel, *, capturing: bool | None = None):
    state = channel._ext._state(channel._ptr)
    if state.eager_tables is None or state.eager_buffer_bytes is None:
        raise ValueError("missing eager buffers")
    if state.plain_payload_ready_region_packs <= 0:
        raise ValueError("missing Lamport sentinel storage")
    is_capturing = bool(capturing) if capturing is not None else False
    if is_capturing and not state.device_slot_selection:
        state.slot_bias = state.eager_slot & 1
        state.device_slot_selection = True
    if state.device_slot_selection:
        peer_tbl = int(state.eager_tables[0])
    else:
        slot = state.eager_slot % 2
        state.eager_slot += 1
        peer_tbl = int(state.eager_tables[slot])
    return (
        peer_tbl,
        int(state.signal_table_address),
        int(state.eager_buffer_bytes) // 16,
        int(state.plain_payload_ready_region_packs),
        bool(state.device_slot_selection),
        int(state.slot_bias) & 1 if state.device_slot_selection else 0,
    )


# ---------------------------------------------------------------------------
# Bit-exact CPU arithmetic reference matching ClusterMhcAllReducePostYNorm
# ---------------------------------------------------------------------------


def _encode_bf16_sentinel_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Encode +0.0 BF16 (0x0000) as -0.0 (0x8000), matching _encode_bf16_sentinel_word."""
    bits = tensor.contiguous().view(torch.uint16).to(torch.int32)
    encoded = torch.where(bits == 0, torch.full_like(bits, 0x8000), bits)
    return encoded.to(torch.uint16).view(torch.bfloat16)


def _canonical_allreduce_cpu(
    peer_xs: list[torch.Tensor],
    peer_addends: list[torch.Tensor] | None = None,
    *,
    reverse_order: bool = False,
) -> torch.Tensor:
    """Canonical rank-order 0..W-1 FP32 sum of BF16 payloads with single BF16 rounding seam."""
    order = range(len(peer_xs) - 1, -1, -1) if reverse_order else range(len(peer_xs))
    acc = None
    for r in order:
        val = peer_xs[r]
        if peer_addends is not None:
            # add.rn.bf16x2 rounds local + addend to BF16 before sentinel push.
            val = (val.float() + peer_addends[r].float()).to(torch.bfloat16)
        val = _encode_bf16_sentinel_cpu(val)
        vf = val.float()
        acc = vf.clone() if acc is None else (acc + vf)
    return acc.to(torch.bfloat16)


def _quantize_mxfp8_cpu(y_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact UE8M0 pow2_ceil + FP8 E4M3FN block-32 quantization matching ClusterMhcPostYNorm."""
    tokens, hidden = y_bf16.shape
    blocks = y_bf16.float().view(tokens, hidden // 32, 32)
    max_abs = blocks.abs().amax(dim=-1)
    ratio = max_abs * (1.0 / 448.0)
    bits = ratio.view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = torch.where(max_abs == 0.0, torch.full_like(exponent, 127), exponent)
    scale_u8 = exponent.clamp(0, 255).to(torch.uint8)
    inv_scale = ((254 - exponent).clamp(0, 255) << 23).view(torch.float32)
    q = (blocks * inv_scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).reshape(tokens, hidden), scale_u8


def _post_y_norm_cpu(
    x_reduced: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    norm_eps: float,
    emit_mxfp8: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Exact CPU reference for ClusterMhcPostYNorm."""
    xv = x_reduced.float()
    r = residual.float()
    p = prev_post.float()
    c = prev_comb.float()
    out_streams = []
    for s in range(4):
        stream = (
            p[:, s : s + 1] * xv
            + c[:, 0, s : s + 1] * r[:, 0, :]
            + c[:, 1, s : s + 1] * r[:, 1, :]
            + c[:, 2, s : s + 1] * r[:, 2, :]
            + c[:, 3, s : s + 1] * r[:, 3, :]
        ).to(torch.bfloat16)
        out_streams.append(stream)
    out = torch.stack(out_streams, dim=1)
    of = out.float()
    contracted = (
        pre_mix[:, 0:1] * of[:, 0, :]
        + pre_mix[:, 1:2] * of[:, 1, :]
        + pre_mix[:, 2:3] * of[:, 2, :]
        + pre_mix[:, 3:4] * of[:, 3, :]
    ).to(torch.bfloat16)
    yf = contracted.float()
    inv = torch.rsqrt(yf.square().mean(dim=-1, keepdim=True) + float(norm_eps))
    w = norm_weight.float()
    y = (yf * inv * w).to(torch.bfloat16)
    if emit_mxfp8:
        av, asc = _quantize_mxfp8_cpu(y)
        return out, y, av, asc
    return out, y, None, None


def _make_mhc_weights(hidden: int, norm_dtype: torch.dtype, seed: int = 17) -> MHCWeights:
    g = torch.Generator().manual_seed(seed)
    return MHCWeights(
        fn=torch.randn((24, 4 * hidden), dtype=torch.float32, generator=g) * 0.02,
        scale=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32),
        base=torch.randn((24,), dtype=torch.float32, generator=g) * 0.1,
        norm=(torch.randn((hidden,), dtype=norm_dtype, generator=g) * 0.2 + 1.0).contiguous(),
    )


def _install_mock_b12x_fused(
    monkeypatch,
    *,
    peer_state: dict,
    compiled_log: list | None = None,
    loaded_log: list | None = None,
    capturing_flag: list[bool] | None = None,
    reverse_rank_order: bool = False,
):
    """Patch b12x fused mHC-AR entry points with a deterministic CPU multi-rank emulator."""
    from b12x._lib import compile_plan
    from b12x.norm.mhc import _split_lagged

    def fake_supports(device, max_tokens, hidden_size, norm_dtype, world_size):
        return (
            1 <= int(max_tokens) <= 32
            and int(hidden_size) in (4096, 5120, 7168)
            and norm_dtype in (torch.bfloat16, torch.float32)
            and int(world_size) in (2, 3, 4, 6, 8)
        )

    def fake_compile(
        ordinal,
        hidden,
        world_size,
        rank,
        norm_eps,
        norm_dtype,
        *,
        cluster_size=None,
        emit_mxfp8=False,
        has_addend=False,
        device_slot_selection=True,
        slot_bias=0,
        release_dependents=False,
    ):
        spec = (
            ordinal,
            hidden,
            world_size,
            rank,
            norm_eps,
            norm_dtype,
            cluster_size,
            emit_mxfp8,
            has_addend,
            device_slot_selection,
            slot_bias,
            release_dependents,
        )
        if compiled_log is not None:
            compiled_log.append(spec)

        def runner(
            peer_table_address,
            signal_table_address,
            x,
            residual,
            prev_post,
            prev_comb,
            incoming,
            norm_weight,
            out,
            y,
            *,
            shard_packs,
            payload_ready_region_packs=0,
            input_addend=None,
            a_values=None,
            a_scales=None,
            stream=None,
        ):
            assert shard_packs >= x.shape[0] * (hidden // 8)
            assert payload_ready_region_packs > 0
            if has_addend:
                assert input_addend is not None
            else:
                assert input_addend is None
            if "schedule" in peer_state:
                idx = peer_state.get("cursor", 0) % len(peer_state["schedule"])
                peer_state["cursor"] = idx + 1
                sched_xs, sched_adds = peer_state["schedule"][idx]
                peer_xs = list(sched_xs)
                peer_adds = list(sched_adds) if has_addend else None
            else:
                peer_xs = list(peer_state["xs"])
                peer_adds = list(peer_state["addends"]) if has_addend else None
            peer_xs[rank] = x
            if has_addend:
                peer_adds[rank] = input_addend
            reduced = _canonical_allreduce_cpu(
                peer_xs, peer_adds, reverse_order=reverse_rank_order
            )
            o_ref, y_ref, av_ref, asc_ref = _post_y_norm_cpu(
                reduced,
                residual,
                prev_post,
                prev_comb,
                incoming,
                norm_weight,
                norm_eps=norm_eps,
                emit_mxfp8=emit_mxfp8,
            )
            out.copy_(o_ref)
            y.copy_(y_ref)
            if emit_mxfp8:
                assert a_values is not None and a_scales is not None
                a_values.copy_(av_ref)
                a_scales.copy_(asc_ref)
            peer_state.setdefault("calls", []).append(
                {
                    "peer_table_address": peer_table_address,
                    "signal_table_address": signal_table_address,
                    "device_slot_selection": device_slot_selection,
                    "slot_bias": slot_bias,
                    "has_addend": has_addend,
                    "emit_mxfp8": emit_mxfp8,
                    "release_dependents": release_dependents,
                    "cluster_size": cluster_size,
                    "hidden": hidden,
                    "tokens": int(residual.shape[0]),
                }
            )

        runner._spec = spec
        return runner

    def fake_params(channel, *, capturing=None):
        cap = (
            capturing_flag[0]
            if (capturing is None and capturing_flag is not None)
            else capturing
        )
        return _mock_sentinel_params(channel, capturing=cap)

    def fake_load(obj):
        if loaded_log is not None:
            loaded_log.append(getattr(obj, "_spec", obj))

    monkeypatch.setattr(_split_lagged, "supports_cluster_allreduce_post_y_norm", fake_supports, raising=False)
    monkeypatch.setattr(_split_lagged, "compile_cluster_allreduce_post_y_norm", fake_compile, raising=False)
    monkeypatch.setattr(_split_lagged, "oneshot_channel_sentinel_params", fake_params, raising=False)
    monkeypatch.setattr(compile_plan, "load_programs", fake_load)


# ---------------------------------------------------------------------------
# 1. Eligibility & Backwards-Compatible Fallback Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("tokens", [1, 8, 12, 16, 32])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_fits_accepts_supported_single_channel_oneshot_geometries(
    monkeypatch, hidden, tokens, world_size, norm_dtype
):
    _install_mock_b12x_fused(monkeypatch, peer_state={"xs": [], "addends": []})
    runtime = _MockPCIeAllReduceRuntime(world_size=world_size)
    assert supports_fused_mhc_allreduce(runtime, tokens, hidden, norm_dtype) is True
    assert PreparedFusedMhcAllReducePostYNorm.fits(runtime, tokens, hidden, norm_dtype) is True
    # Direct channel object also accepted
    assert PreparedFusedMhcAllReducePostYNorm.fits(runtime.channel, tokens, hidden, norm_dtype) is True


def test_fits_rejects_base_b12x_missing_symbols(monkeypatch):
    from b12x.norm.mhc import _split_lagged

    runtime = _MockPCIeAllReduceRuntime()
    for attr in (
        "supports_cluster_allreduce_post_y_norm",
        "compile_cluster_allreduce_post_y_norm",
        "oneshot_channel_sentinel_params",
    ):
        with monkeypatch.context() as mp:
            mp.setattr(_split_lagged, attr, None, raising=False)
            assert supports_fused_mhc_allreduce(runtime, 8, 5120, torch.bfloat16) is False


def test_fits_rejects_roce_dma_hierarchical_multichannel_and_invalid_shapes(monkeypatch):
    from b12x.comm import pcie, roce

    _install_mock_b12x_fused(monkeypatch, peer_state={"xs": [], "addends": []})

    assert supports_fused_mhc_allreduce(None, 8, 5120, torch.bfloat16) is False
    assert supports_fused_mhc_allreduce(object(), 8, 5120, torch.bfloat16) is False

    roce_rt = object.__new__(roce.AllReduce)
    assert supports_fused_mhc_allreduce(roce_rt, 8, 5120, torch.bfloat16) is False

    dma_rt = object.__new__(pcie.DmaAllReduce)
    assert supports_fused_mhc_allreduce(dma_rt, 8, 5120, torch.bfloat16) is False

    hier_rt = _MockPCIeAllReduceRuntime(algorithm="hierarchical", world_size=16)
    assert supports_fused_mhc_allreduce(hier_rt, 8, 5120, torch.bfloat16) is False

    multi_ch = _MockPCIeAllReduceRuntime(single_channel=False)
    assert supports_fused_mhc_allreduce(multi_ch, 8, 5120, torch.bfloat16) is False

    no_sentinel = _MockPCIeAllReduceRuntime(plain_payload_ready_region_packs=0)
    assert supports_fused_mhc_allreduce(no_sentinel, 8, 5120, torch.bfloat16) is False

    no_eager = _MockPCIeAllReduceRuntime(eager_tables=None)
    assert supports_fused_mhc_allreduce(no_eager, 8, 5120, torch.bfloat16) is False

    small_cap = _MockPCIeAllReduceRuntime(max_size=8 * 5120 * 2, eager_buffer_bytes=8 * 5120 * 2)
    assert supports_fused_mhc_allreduce(small_cap, 8, 5120, torch.bfloat16) is True
    assert supports_fused_mhc_allreduce(small_cap, 9, 5120, torch.bfloat16) is False

    good_rt = _MockPCIeAllReduceRuntime()
    assert supports_fused_mhc_allreduce(good_rt, 0, 5120, torch.bfloat16) is False
    assert supports_fused_mhc_allreduce(good_rt, 33, 5120, torch.bfloat16) is False
    assert supports_fused_mhc_allreduce(good_rt, 64, 5120, torch.bfloat16) is False
    assert supports_fused_mhc_allreduce(good_rt, 128, 5120, torch.bfloat16) is False
    assert supports_fused_mhc_allreduce(good_rt, 8, 3072, torch.bfloat16) is False
    assert supports_fused_mhc_allreduce(good_rt, 8, 5120, torch.float16) is False

    tp1_rt = _MockPCIeAllReduceRuntime(world_size=1)
    assert supports_fused_mhc_allreduce(tp1_rt, 8, 5120, torch.bfloat16) is False


# ---------------------------------------------------------------------------
# 2. Finalization, Lamport Epoch/Slot Tracking, and MXFP8 Scale Swizzle
# ---------------------------------------------------------------------------


def test_finalize_precompiles_all_slot_phases_and_tracks_live_epoch(monkeypatch):
    from b12x.moe.fused_moe import _binding_ops

    compiled_log: list = []
    loaded_log: list = []
    capturing = [False]
    tokens, hidden, world = 16, 5120, 4
    g = torch.Generator().manual_seed(11)
    peer_xs = [torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) for _ in range(world)]
    peer_state = {"xs": peer_xs, "addends": []}
    _install_mock_b12x_fused(
        monkeypatch,
        peer_state=peer_state,
        compiled_log=compiled_log,
        loaded_log=loaded_log,
        capturing_flag=capturing,
    )
    monkeypatch.setattr(_binding_ops, "supports_prequant_mxfp8", lambda plan, m: True)

    finalizers = []
    monkeypatch.setattr(
        "hyperflare_recipes.components.mhc_overlap.register_finalizer",
        lambda session, fn: finalizers.append(fn),
    )

    runtime = _MockPCIeAllReduceRuntime(world_size=world, rank=0)
    weights = _make_mhc_weights(hidden, torch.bfloat16)
    res = torch.randn((tokens, 4, hidden), dtype=torch.bfloat16, generator=g)
    pp = torch.rand((tokens, 4), dtype=torch.float32, generator=g)
    pc = torch.softmax(torch.randn((tokens, 4, 4), dtype=torch.float32, generator=g), dim=-1)
    pm = torch.softmax(torch.randn((tokens, 4), dtype=torch.float32, generator=g), dim=-1)

    fused = PreparedFusedMhcAllReducePostYNorm(
        (peer_xs[0], res, pp, pc),
        weights,
        runtime=runtime,
        session=object(),
        norm_eps=1e-6,
        has_addend=False,
        release_dependents=True,
        experts=SimpleNamespace(plan=object(), integrated_routing=False),
    ).functional(session=object())
    assert len(finalizers) == 1
    finalizers[0]()

    assert fused.emit_mxfp8 is True
    assert len(compiled_log) == 3
    assert len(loaded_log) == 3
    phases = [(spec[9], spec[10]) for spec in compiled_log]
    assert phases == [(False, 0), (True, 0), (True, 1)]
    assert all(spec[6] == 8 and spec[7] is True and spec[11] is True for spec in compiled_log)

    # 1 warmup call advances eager_slot 0 -> 1 (odd phase!)
    fused(peer_xs[0], res, pp, pc, pre_mix=pm)
    assert runtime.channel._state.eager_slot == 1
    assert peer_state["calls"][-1]["device_slot_selection"] is False
    assert peer_state["calls"][-1]["peer_table_address"] == 0x100000

    # Entering graph capture freezes slot_bias = 1 and reuses precompiled (True, 1) with 0 recompiles!
    capturing[0] = True
    fused(peer_xs[0], res, pp, pc, pre_mix=pm)
    assert len(compiled_log) == 3
    assert peer_state["calls"][-1]["device_slot_selection"] is True
    assert peer_state["calls"][-1]["slot_bias"] == 1
    assert peer_state["calls"][-1]["peer_table_address"] == 0x100000


@pytest.mark.parametrize("hidden", [4096, 5120])
@pytest.mark.parametrize("tokens", [1, 8, 12, 16, 32])
def test_pack_mxfp8_scale_mma_layout_and_neutral_padding(hidden, tokens):
    scale_rows = torch.randint(110, 140, (tokens, hidden // 32), dtype=torch.uint8)
    mma = pack_mxfp8_scale_mma(scale_rows, hidden)
    assert mma.shape == (1, 1, hidden // 128, 32, 4, 4)
    assert mma.dtype == torch.uint8
    # Unswizzle and check exact active rows + neutral 127 padding rows
    recovered = (
        mma.view(1, hidden // 128, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(128, hidden // 32)
    )
    assert torch.equal(recovered[:tokens], scale_rows)
    assert torch.all(recovered[tokens:] == 127)


# ---------------------------------------------------------------------------
# 3. 0-ULP Bit-Exact Multi-Rank Parity + make_fx Tracing + Negative Checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden", [4096, 5120])
@pytest.mark.parametrize("tokens", [1, 8, 12, 16, 32])
@pytest.mark.parametrize("has_addend", [False, True])
@pytest.mark.parametrize("emit_mxfp8", [False, True])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_zero_ulp_parity_and_make_fx_tracing(
    monkeypatch, hidden, tokens, has_addend, emit_mxfp8, norm_dtype
):
    world = 4
    g = torch.Generator().manual_seed(1000 + hidden + tokens * 7 + int(has_addend) * 31 + int(emit_mxfp8) * 61)
    peer_xs = [
        (torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.5).contiguous()
        for _ in range(world)
    ]
    # Inject adversarial +0.0 / -0.0 lanes to exercise Lamport sentinel encoding
    peer_xs[0][0, :16] = 0.0
    peer_xs[1][0, :8] = -0.0
    peer_addends = (
        [
            (torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.25).contiguous()
            for _ in range(world)
        ]
        if has_addend
        else None
    )
    if peer_addends is not None:
        peer_addends[0][0, :8] = 0.0
        peer_addends[2][0, 8:16] = -peer_xs[2][0, 8:16]

    residual = (torch.randn((tokens, 4, hidden), dtype=torch.bfloat16, generator=g) * 0.5).contiguous()
    prev_post = torch.rand((tokens, 4), dtype=torch.float32, generator=g).contiguous()
    prev_comb = torch.softmax(
        torch.randn((tokens, 4, 4), dtype=torch.float32, generator=g), dim=-1
    ).contiguous()
    pre_mix = torch.softmax(
        torch.randn((tokens, 4), dtype=torch.float32, generator=g), dim=-1
    ).contiguous()
    weights = _make_mhc_weights(hidden, norm_dtype, seed=hidden + tokens)

    peer_state = {"xs": peer_xs, "addends": peer_addends or []}
    _install_mock_b12x_fused(monkeypatch, peer_state=peer_state)
    monkeypatch.setattr(
        "hyperflare_recipes.components.mhc_overlap.register_finalizer",
        lambda session, fn: None,
    )

    runtime = _MockPCIeAllReduceRuntime(world_size=world, rank=2)
    fused = PreparedFusedMhcAllReducePostYNorm(
        (peer_xs[2], residual, prev_post, prev_comb),
        weights,
        runtime=runtime,
        session=object(),
        norm_eps=1e-6,
        has_addend=has_addend,
        emit_mxfp8=emit_mxfp8,
        release_dependents=True,
    ).functional(session=object())
    fused._finalize()

    # Reference: standalone OneshotAllReduce + ClusterMhcPostYNorm
    reduced_ref = _canonical_allreduce_cpu(peer_xs, peer_addends)
    out_ref, y_ref, av_ref, asc_ref = _post_y_norm_cpu(
        reduced_ref,
        residual,
        prev_post,
        prev_comb,
        pre_mix,
        weights.norm,
        norm_eps=1e-6,
        emit_mxfp8=emit_mxfp8,
    )

    # Eager call via custom op
    dep = torch.zeros(1, dtype=torch.float32)
    out, y, av, asc = fused(
        peer_xs[2],
        residual,
        prev_post,
        prev_comb,
        pre_mix=pre_mix,
        input_addend=None if not has_addend else peer_addends[2],
        dependencies=(dep,),
    )
    assert torch.equal(out.view(torch.uint16), out_ref.view(torch.uint16))
    assert torch.equal(y.view(torch.uint16), y_ref.view(torch.uint16))
    if emit_mxfp8:
        assert av is not None and asc is not None
        assert torch.equal(av, av_ref)
        assert torch.equal(asc, asc_ref)
    else:
        assert av is None and asc is None

    # Clean make_fx tracing on representative shapes
    if tokens in (1, 16, 32) and norm_dtype == torch.bfloat16:
        if has_addend:
            def traced_fn(x_, add_, res_, pp_, pc_, pm_):
                o_, y_, av_, asc_ = fused(
                    x_, res_, pp_, pc_, pre_mix=pm_, input_addend=add_, dependencies=(pm_,)
                )
                return (o_, y_, av_, asc_) if emit_mxfp8 else (o_, y_)

            gm = make_fx(traced_fn)(peer_xs[2], peer_addends[2], residual, prev_post, prev_comb, pre_mix)
            res_fx = gm(peer_xs[2], peer_addends[2], residual, prev_post, prev_comb, pre_mix)
        else:
            def traced_fn(x_, res_, pp_, pc_, pm_):
                o_, y_, av_, asc_ = fused(
                    x_, res_, pp_, pc_, pre_mix=pm_, dependencies=(pm_,)
                )
                return (o_, y_, av_, asc_) if emit_mxfp8 else (o_, y_)

            gm = make_fx(traced_fn)(peer_xs[2], residual, prev_post, prev_comb, pre_mix)
            res_fx = gm(peer_xs[2], residual, prev_post, prev_comb, pre_mix)

        op_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.flare.cluster_mhc_allreduce_post_y_norm.default
        ]
        assert len(op_nodes) == 1
        assert torch.equal(res_fx[0].view(torch.uint16), out_ref.view(torch.uint16))
        assert torch.equal(res_fx[1].view(torch.uint16), y_ref.view(torch.uint16))
        if emit_mxfp8:
            assert torch.equal(res_fx[2], av_ref)
            assert torch.equal(res_fx[3], asc_ref)


def test_negative_sensitivity_catches_noncanonical_order_addend_and_pre_mix(monkeypatch):
    tokens, hidden, world = 8, 5120, 4
    g = torch.Generator().manual_seed(909)
    # Construct values across FP32's 24-bit mantissa boundary (2^24 = 16777216.0, exact in BF16)
    # so rank-order 0->1->2->3 vs 3->2->1->0 produces a real IEEE-754 rounding difference:
    # forward: (((2^24 + 0.5) - 2^24) + 0.5) = 0.5; reverse: (((0.5 - 2^24) + 0.5) + 2^24) = 0.0!
    peer_xs = [
        torch.full((tokens, hidden), 16777216.0, dtype=torch.bfloat16),
        torch.full((tokens, hidden), 0.5, dtype=torch.bfloat16),
        torch.full((tokens, hidden), -16777216.0, dtype=torch.bfloat16),
        torch.full((tokens, hidden), 0.5, dtype=torch.bfloat16),
    ]
    peer_addends = [
        torch.zeros((tokens, hidden), dtype=torch.bfloat16) for _ in range(world)
    ]
    residual = (torch.randn((tokens, 4, hidden), dtype=torch.bfloat16, generator=g) * 0.5).contiguous()
    prev_post = torch.rand((tokens, 4), dtype=torch.float32, generator=g).contiguous()
    prev_comb = torch.softmax(torch.randn((tokens, 4, 4), dtype=torch.float32, generator=g), dim=-1).contiguous()
    pre_mix = torch.softmax(torch.randn((tokens, 4), dtype=torch.float32, generator=g), dim=-1).contiguous()
    weights = _make_mhc_weights(hidden, torch.bfloat16, seed=909)

    reduced_canon = _canonical_allreduce_cpu(peer_xs, peer_addends, reverse_order=False)
    out_ref, y_ref, av_ref, asc_ref = _post_y_norm_cpu(
        reduced_canon, residual, prev_post, prev_comb, pre_mix, weights.norm, norm_eps=1e-6, emit_mxfp8=True
    )

    # 1. Reversed rank summation order must fail bit-exact parity
    peer_state = {"xs": peer_xs, "addends": peer_addends}
    _install_mock_b12x_fused(monkeypatch, peer_state=peer_state, reverse_rank_order=True)
    monkeypatch.setattr(
        "hyperflare_recipes.components.mhc_overlap.register_finalizer",
        lambda session, fn: None,
    )
    runtime = _MockPCIeAllReduceRuntime(world_size=world, rank=0)
    fused_rev = PreparedFusedMhcAllReducePostYNorm(
        (peer_xs[0], residual, prev_post, prev_comb),
        weights,
        runtime=runtime,
        session=object(),
        norm_eps=1e-6,
        has_addend=True,
        emit_mxfp8=True,
    ).functional()
    fused_rev._finalize()
    out_rev, y_rev, _, _ = fused_rev(
        peer_xs[0], residual, prev_post, prev_comb, pre_mix=pre_mix, input_addend=peer_addends[0]
    )
    assert not torch.equal(out_rev.view(torch.uint16), out_ref.view(torch.uint16))

    # 2. Wrong pre_mix must fail bit-exact y/MXFP8 parity
    _install_mock_b12x_fused(monkeypatch, peer_state=peer_state, reverse_rank_order=False)
    fused_ok = PreparedFusedMhcAllReducePostYNorm(
        (peer_xs[0], residual, prev_post, prev_comb),
        weights,
        runtime=runtime,
        session=object(),
        norm_eps=1e-6,
        has_addend=True,
        emit_mxfp8=True,
    ).functional()
    fused_ok._finalize()
    wrong_pre_mix = pre_mix.roll(1, dims=-1)
    _, y_wrong, av_wrong, asc_wrong = fused_ok(
        peer_xs[0], residual, prev_post, prev_comb, pre_mix=wrong_pre_mix, input_addend=peer_addends[0]
    )
    assert not torch.equal(y_wrong.view(torch.uint16), y_ref.view(torch.uint16))
    assert not torch.equal(av_wrong, av_ref)

    # 3. Missing or unexpected input_addend raises ValueError explicitly
    with pytest.raises(ValueError, match="requires input_addend"):
        fused_ok(peer_xs[0], residual, prev_post, prev_comb, pre_mix=pre_mix)

    fused_no_add = PreparedFusedMhcAllReducePostYNorm(
        (peer_xs[0], residual, prev_post, prev_comb),
        weights,
        runtime=runtime,
        session=object(),
        norm_eps=1e-6,
        has_addend=False,
        emit_mxfp8=False,
    ).functional()
    with pytest.raises(ValueError, match="does not accept input_addend"):
        fused_no_add(
            peer_xs[0], residual, prev_post, prev_comb, pre_mix=pre_mix, input_addend=peer_addends[0]
        )


# ---------------------------------------------------------------------------
# 4. DeepSeek-V4.1 PreparedLayer + Multi-Layer Linking + End-to-End Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 33, 64, 128])
@pytest.mark.parametrize("mode", ["decode", "verify", "draft", "prefill"])
@pytest.mark.parametrize("carry", [False, True])
def test_deepseek41_layer_fused_mhc_ar_activation_and_fallback(monkeypatch, rows, mode, carry):
    from b12x.norm.mhc import _split_lagged

    _install_mock_b12x_fused(monkeypatch, peer_state={"xs": [], "addends": []})
    monkeypatch.setattr(
        _split_lagged,
        "supports_split_lagged",
        lambda device, max_tokens, hidden, dtype: 1 <= int(max_tokens) <= 128,
    )

    scratch = decoder.Scratch.allocate(rows, 1, 4, "meta")
    ar_names = []
    split_compile_flags = []

    class Component:
        plan = object()

        def __init__(self, runtime=None):
            self.runtime = runtime

        def functional(self, **kwargs):
            return self

    def make_ar(*args, **kwargs):
        ar_names.append(kwargs.get("name"))
        return Component(runtime=kwargs.get("runtime"))

    def make_split(*args, **kwargs):
        split_compile_flags.append(kwargs.get("compile_post_y", True))
        return Component()

    class FakeArena(decoder.ScratchArena):
        def __init__(self):
            self.overlap = True

    for name in (
        "PreparedMHC",
        "PreparedAttention",
        "UnquantizedProjection",
        "PreparedRouter",
        "PreparedRoutedExperts",
        "PreparedMLP",
        "PreparedMHCPost",
        "ConcurrentSharedExpert",
    ):
        monkeypatch.setattr(decoder, name, lambda *a, **kw: Component())
    monkeypatch.setattr(decoder, "PreparedAllReduce", make_ar)
    monkeypatch.setattr(decoder, "SplitLaggedMHC", make_split)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *a, **kw: None)
    monkeypatch.setattr(
        "hyperflare_recipes.components.mhc_overlap.register_finalizer",
        lambda *a, **kw: None,
    )

    text = dict(
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
    )
    residual_weights = {
        "hc_attn_fn": torch.empty((24, 4 * 5120), dtype=torch.float32, device="meta"),
        "hc_attn_scale": torch.empty((3,), dtype=torch.float32, device="meta"),
        "hc_attn_base": torch.empty((24,), dtype=torch.float32, device="meta"),
        "attn_norm.weight": torch.empty((5120,), dtype=torch.bfloat16, device="meta"),
        "hc_ffn_fn": torch.empty((24, 4 * 5120), dtype=torch.float32, device="meta"),
        "hc_ffn_scale": torch.empty((3,), dtype=torch.float32, device="meta"),
        "hc_ffn_base": torch.empty((24,), dtype=torch.float32, device="meta"),
        "ffn_norm.weight": torch.empty((5120,), dtype=torch.bfloat16, device="meta"),
    }
    weights = SimpleNamespace(
        engram=None,
        attention=None,
        residual=residual_weights,
        experts=SimpleNamespace(
            vision_correction=None, router=None, correction=None, experts=None, shared=None
        ),
    )
    runtime = _MockPCIeAllReduceRuntime(world_size=4, rank=0, device="meta")
    layer = decoder.PreparedLayer(
        3,
        weights,
        None,
        scratch,
        SimpleNamespace(token_ids=torch.empty(rows, device="meta")),
        SimpleNamespace(config=SimpleNamespace(text=text)),
        None,
        shard=SimpleNamespace(world_size=4),
        mode=mode,
        runtime=runtime,
        coordinator=None,
        session=object(),
        name="layer3",
        workspace=FakeArena(),
        carry=carry,
        defer_post=True,
    )

    expect_split = mode in ("decode", "verify", "draft") and rows <= 128
    expect_fused = expect_split and rows <= 32
    assert layer.split_mhc is expect_split
    assert (layer.ffn_fused_mhc is not None) is expect_fused
    assert (layer.attn_fused_mhc is not None) is (expect_fused and carry)
    if expect_fused:
        # attention_reduce is replaced by ffn_fused_mhc and preserves .runtime for CatalogLayerFixture
        assert layer.attention_reduce is layer.ffn_fused_mhc
        assert layer.attention_reduce.runtime is runtime
        assert "layer3.attention_reduce" not in ar_names
        assert "layer3.mlp_reduce" in ar_names
        # attention_hc compiles post_y (fallback for 4-tuple callers); ffn_hc skips unused post_y
        assert split_compile_flags == [True, False]
    else:
        assert "layer3.attention_reduce" in ar_names
        assert "layer3.mlp_reduce" in ar_names


def test_link_fused_mhc_layers_respects_engram_post_and_runtime_boundaries():
    rt_a = object()
    rt_b = object()

    def fake_layer(*, split=True, post=None, carry=True, engram=None, has_attn_fused=True, rt=rt_a):
        return SimpleNamespace(
            split_mhc=split,
            post=post,
            carry=carry,
            engram=engram,
            attn_fused_mhc=SimpleNamespace(runtime=rt) if has_attn_fused else None,
            fused_mhc_runtime=rt,
            defer_mlp_reduce=False,
        )

    layers = [
        fake_layer(carry=False, has_attn_fused=False, post=None, rt=rt_a),  # L0 -> L1 (links!)
        fake_layer(carry=True, has_attn_fused=True, post=object(), rt=rt_a),  # L1 has post -> L2 (no link)
        fake_layer(carry=False, engram=object(), has_attn_fused=False, post=None, rt=rt_a),  # L2 -> L3 (links!)
        fake_layer(carry=True, has_attn_fused=True, post=None, rt=rt_a),  # L3 -> L4 (different runtime -> no link)
        fake_layer(carry=True, has_attn_fused=True, post=object(), rt=rt_b),  # L4 final
    ]
    decoder._link_fused_mhc_layers(layers)
    assert [layer.defer_mlp_reduce for layer in layers] == [True, False, True, False, False]


def test_multilayer_deepseek41_fused_vs_unfused_zero_ulp_parity_and_fx(monkeypatch):
    """Run a 3-layer lagged-mHC chain (L0 -> L1 -> L2) and assert 0-ULP parity vs unfused."""
    tokens, hidden, world, rank = 8, 5120, 4, 1
    g = torch.Generator().manual_seed(2026)

    # Shared multi-rank peer delta tensors injected for every attention and FFN sublayer.
    peer_state = {"xs": [], "addends": []}
    _install_mock_b12x_fused(monkeypatch, peer_state=peer_state)
    monkeypatch.setattr(
        "hyperflare_recipes.components.mhc_overlap.register_finalizer",
        lambda session, fn: None,
    )

    layer_weights = [
        (
            _make_mhc_weights(hidden, torch.bfloat16, seed=10 + 2 * i),
            _make_mhc_weights(hidden, torch.bfloat16, seed=11 + 2 * i),
        )
        for i in range(3)
    ]
    # Deterministic per-rank deltas for attention output, routed output, and shared output per layer.
    attn_deltas = [
        [
            (torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.05).contiguous()
            for _ in range(world)
        ]
        for _ in range(3)
    ]
    routed_deltas = [
        [
            (torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.05).contiguous()
            for _ in range(world)
        ]
        for _ in range(3)
    ]
    shared_deltas = [
        [
            (torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.03).contiguous()
            for _ in range(world)
        ]
        for _ in range(3)
    ]

    def _mock_sinkhorn(residual_out: torch.Tensor, scale: float):
        # Deterministic FP32 coefficients derived strictly from residual_out.
        base = residual_out.float().mean(dim=-1)
        pre_out = torch.softmax(base * scale, dim=-1).contiguous()
        post_out = (2.0 * torch.sigmoid(base * (scale + 0.1))).contiguous()
        comb_logits = (base[:, :, None] + base[:, None, :]) * 0.25
        comb_out = torch.softmax(comb_logits, dim=-1).contiguous()
        return pre_out, post_out, comb_out

    class _MockSplitHC:
        def __init__(self, weights: MHCWeights, pre_only: bool, sink_scale: float):
            self.weights = weights
            self.pre_only = pre_only
            self.sink_scale = sink_scale

        def post_y_norm(self, *inputs, pre_mix):
            if self.pre_only:
                (res,) = inputs
                of = res.float()
                contracted = (
                    pre_mix[:, 0:1] * of[:, 0, :]
                    + pre_mix[:, 1:2] * of[:, 1, :]
                    + pre_mix[:, 2:3] * of[:, 2, :]
                    + pre_mix[:, 3:4] * of[:, 3, :]
                ).to(torch.bfloat16)
                yf = contracted.float()
                inv = torch.rsqrt(yf.square().mean(dim=-1, keepdim=True) + 1e-6)
                y = (yf * inv * self.weights.norm.float()).to(torch.bfloat16)
                return res.clone(), y, None, None
            x, res, pp, pc = inputs
            return _post_y_norm_cpu(
                x, res, pp, pc, pre_mix, self.weights.norm, norm_eps=1e-6, emit_mxfp8=True
            )

        def start_sinkhorn(self, residual_out, trigger, trigger2=None):
            pre_out, post_out, comb_out = _mock_sinkhorn(residual_out, self.sink_scale)
            return PendingMHCSinkhorn(
                residual_out,
                torch.empty(1, dtype=torch.float32),
                pre_out,
                post_out,
                comb_out,
            )

        def finish_sinkhorn(self, pending, dependency):
            return pending.post_out, pending.comb_out, pending.pre_out

    def _build_chain(*, fused: bool):
        runtime = _MockPCIeAllReduceRuntime(world_size=world, rank=rank)
        layers = []
        for idx in range(3):
            attn_w, ffn_w = layer_weights[idx]
            carry = idx > 0
            defer_post = idx < 2
            layer = object.__new__(decoder.PreparedLayer)
            layer.engram = None
            layer.concurrent_shared = None
            layer.has_image_mask = False
            layer.row_shard_dma = False
            layer.pipelined_attn = None
            layer.pipelined_attn_hc = False
            layer.pipelined_ffn = False
            layer.split_mhc = True
            layer.carry = carry
            layer.defer_mlp_reduce = False
            layer.fused_mhc_runtime = runtime if fused else None
            layer.attention_hc = _MockSplitHC(attn_w, pre_only=not carry, sink_scale=0.3 + 0.1 * idx)
            layer.ffn_hc = _MockSplitHC(ffn_w, pre_only=False, sink_scale=0.5 + 0.1 * idx)

            dummy_x = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
            dummy_res = torch.zeros((tokens, 4, hidden), dtype=torch.bfloat16)
            dummy_pp = torch.ones((tokens, 4), dtype=torch.float32)
            dummy_pc = torch.eye(4, dtype=torch.float32).expand(tokens, 4, 4).contiguous()

            if fused and carry:
                layer.attn_fused_mhc = PreparedFusedMhcAllReducePostYNorm(
                    (dummy_x, dummy_res, dummy_pp, dummy_pc),
                    attn_w,
                    runtime=runtime,
                    session=object(),
                    norm_eps=1e-6,
                    has_addend=True,
                    emit_mxfp8=False,
                    release_dependents=True,
                ).functional()
                layer.attn_fused_mhc._finalize()
            else:
                layer.attn_fused_mhc = None

            if fused:
                layer.ffn_fused_mhc = PreparedFusedMhcAllReducePostYNorm(
                    (dummy_x, dummy_res, dummy_pp, dummy_pc),
                    ffn_w,
                    runtime=runtime,
                    session=object(),
                    norm_eps=1e-6,
                    has_addend=False,
                    emit_mxfp8=True,
                    release_dependents=True,
                ).functional()
                layer.ffn_fused_mhc._finalize()
                layer.attention_reduce = layer.ffn_fused_mhc
            else:
                layer.ffn_fused_mhc = None

                def _unfused_attn_ar(src, *, dependencies=(), _idx=idx):
                    peer_xs = list(attn_deltas[_idx])
                    peer_xs[rank] = src
                    return _canonical_allreduce_cpu(peer_xs)

                layer.attention_reduce = _unfused_attn_ar

            def _attn_call(norm, metadata, selection, *, _idx=idx, _layer=layer, **kwargs):
                # Inject peer attention outputs right before collective
                peer_state["xs"] = list(attn_deltas[_idx])
                peer_state["addends"] = []
                _layer._last_attn_norm = norm
                out = (norm * 0.5 + attn_deltas[_idx][rank]).to(torch.bfloat16)
                return out, None, selection

            layer.attention = _attn_call
            layer.router_projection = lambda norm: norm[:, :8].float()

            def _router_call(logits, image_mask=None, *, _layer=layer):
                ids = logits.topk(2, dim=-1).indices.to(torch.int32)
                _layer._last_router_ids = ids
                return SimpleNamespace(
                    ids=ids,
                    weights=torch.softmax(logits.topk(2, dim=-1).values, dim=-1),
                    selected_logits=logits[:, :2],
                )

            layer.router = _router_call

            def _experts(
                norm,
                ids,
                weights_t,
                *,
                prequant_a_values=None,
                prequant_a_scales=None,
                _idx=idx,
                **kwargs,
            ):
                assert prequant_a_values is not None and prequant_a_scales is not None
                # Use both norm and prequant values deterministically so MXFP8 parity affects output!
                q_bias = prequant_a_values.to(torch.float32).mean(dim=-1, keepdim=True) * 1e-4
                return (norm * 0.25 + q_bias + routed_deltas[_idx][rank]).to(torch.bfloat16)

            layer.experts = _experts

            def _shared(norm, *, _idx=idx):
                peer_state["xs"] = list(routed_deltas[_idx])
                peer_state["addends"] = list(shared_deltas[_idx])
                return (norm * 0.125 + shared_deltas[_idx][rank]).to(torch.bfloat16)

            layer.shared = _shared

            class _UnfusedMlpAR:
                fused_addend = True

                def __init__(self, _idx):
                    self._idx = _idx

                def __call__(self, src, *, addend=None, dependencies=()):
                    peer_xs = list(routed_deltas[self._idx])
                    peer_xs[rank] = src
                    peer_adds = list(shared_deltas[self._idx])
                    peer_adds[rank] = addend
                    return _canonical_allreduce_cpu(peer_xs, peer_adds)

            layer.mlp_reduce = _UnfusedMlpAR(idx)
            if defer_post:
                layer.post = None
            else:
                def _post(x, res, pp, pc):
                    o, _, _, _ = _post_y_norm_cpu(
                        x, res, pp, pc,
                        torch.ones((tokens, 4), dtype=torch.float32),
                        torch.ones((hidden,), dtype=torch.bfloat16),
                        norm_eps=1e-6,
                        emit_mxfp8=False,
                    )
                    return o

                layer.post = _post
            layers.append(layer)
        if fused:
            decoder._link_fused_mhc_layers(layers)
        return layers

    unfused_layers = _build_chain(fused=False)
    fused_layers = _build_chain(fused=True)
    assert [layer.defer_mlp_reduce for layer in fused_layers] == [True, True, False]

    init_res = (torch.randn((tokens, 4, hidden), dtype=torch.bfloat16, generator=g) * 0.3).contiguous()
    init_mix = torch.zeros((tokens, 4), dtype=torch.float32)
    init_mix[:, 0] = 1.0
    selection = (torch.zeros((tokens, 1), dtype=torch.int32), None, None)

    def _run_chain(layers, res0, mix0):
        incoming = (res0,)
        pre_mix = mix0
        sel = selection
        norms = []
        rids = []
        for layer in layers:
            incoming, pre_mix, _, sel = layer(incoming, pre_mix, None, None, sel)
            norms.append(layer._last_attn_norm)
            rids.append(layer._last_router_ids)
        return incoming[0], pre_mix, *norms, *rids

    # Deterministic per-sublayer schedule for the 5 fused calls (L0.attn, L0.mlp, L1.attn, L1.mlp, L2.attn)
    # so both eager execution and replayed make_fx graphs receive identical peer payloads.
    peer_state["schedule"] = [
        (attn_deltas[0], []),
        (routed_deltas[0], shared_deltas[0]),
        (attn_deltas[1], []),
        (routed_deltas[1], shared_deltas[1]),
        (attn_deltas[2], []),
    ]
    peer_state["cursor"] = 0

    out_unfused = _run_chain(unfused_layers, init_res, init_mix)
    peer_state["cursor"] = 0
    out_fused = _run_chain(fused_layers, init_res, init_mix)

    # Exact 0-ULP bit-for-bit parity across all 3 layers!
    assert torch.equal(out_fused[0].view(torch.uint16), out_unfused[0].view(torch.uint16))
    assert torch.equal(out_fused[1], out_unfused[1])
    for i in range(3):
        assert torch.equal(out_fused[2 + i].view(torch.uint16), out_unfused[2 + i].view(torch.uint16))
        assert torch.equal(out_fused[5 + i], out_unfused[5 + i])

    # Trace the full 3-layer fused chain cleanly with make_fx:
    # 3 layers -> 3 attn->ffn + 2 ffn->next-attn = 5 fused custom_op nodes!
    peer_state["cursor"] = 0
    gm = make_fx(lambda r0, m0: _run_chain(fused_layers, r0, m0))(init_res, init_mix)
    fused_nodes = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and n.target is torch.ops.flare.cluster_mhc_allreduce_post_y_norm.default
    ]
    assert len(fused_nodes) == 5
    peer_state["cursor"] = 0
    fx_out = gm(init_res, init_mix)
    assert torch.equal(fx_out[0].view(torch.uint16), out_unfused[0].view(torch.uint16))
    assert torch.equal(fx_out[1], out_unfused[1])


# ---------------------------------------------------------------------------
# 5. Pre-Slot Geometry/Capacity Validation, Operand Checks, and Transport Key
# ---------------------------------------------------------------------------


def test_oversized_launch_rejected_before_advancing_eager_slot(monkeypatch):
    """An oversized call must raise before oneshot_channel_sentinel_params advances eager_slot."""
    tokens, hidden, world = 8, 5120, 4
    shard_bytes = tokens * hidden * 2  # 81,920 bytes: fits exactly 8 rows
    g = torch.Generator().manual_seed(55)
    peer_xs = [
        (torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.25).contiguous()
        for _ in range(world)
    ]
    peer_state = {"xs": peer_xs, "addends": []}
    _install_mock_b12x_fused(monkeypatch, peer_state=peer_state)
    monkeypatch.setattr(
        "hyperflare_recipes.components.mhc_overlap.register_finalizer",
        lambda session, fn: None,
    )

    runtime = _MockPCIeAllReduceRuntime(
        world_size=world,
        rank=0,
        max_size=shard_bytes,
        eager_buffer_bytes=shard_bytes,
    )
    weights = _make_mhc_weights(hidden, torch.bfloat16, seed=55)
    res = torch.randn((tokens, 4, hidden), dtype=torch.bfloat16, generator=g).contiguous()
    pp = torch.rand((tokens, 4), dtype=torch.float32, generator=g).contiguous()
    pc = torch.softmax(torch.randn((tokens, 4, 4), dtype=torch.float32, generator=g), dim=-1).contiguous()
    pm = torch.softmax(torch.randn((tokens, 4), dtype=torch.float32, generator=g), dim=-1).contiguous()

    fused = PreparedFusedMhcAllReducePostYNorm(
        (peer_xs[0], res, pp, pc),
        weights,
        runtime=runtime,
        session=object(),
        norm_eps=1e-6,
        has_addend=False,
        emit_mxfp8=False,
    ).functional()
    fused._finalize()

    # Valid call advances eager_slot 0 -> 1 using slot 0 (0x100000).
    fused(peer_xs[0], res, pp, pc, pre_mix=pm)
    assert runtime.channel._state.eager_slot == 1
    assert peer_state["calls"][-1]["peer_table_address"] == 0x100000

    # 9-row call exceeds prepared capacity and 81,920-byte sentinel shard; eager_slot must stay 1.
    x9 = torch.zeros((9, hidden), dtype=torch.bfloat16)
    res9 = torch.zeros((9, 4, hidden), dtype=torch.bfloat16)
    pp9 = torch.ones((9, 4), dtype=torch.float32)
    pc9 = torch.ones((9, 4, 4), dtype=torch.float32)
    pm9 = torch.ones((9, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="requires 1 <= tokens <="):
        fused(x9, res9, pp9, pc9, pre_mix=pm9)
    assert runtime.channel._state.eager_slot == 1

    # Shrinking live sentinel shard capacity below payload_bytes must also reject without touching eager_slot.
    runtime.channel._state.eager_buffer_bytes = 4 * hidden * 2
    with pytest.raises(ValueError, match="exceeds live sentinel shard capacity"):
        fused(peer_xs[0], res, pp, pc, pre_mix=pm)
    assert runtime.channel._state.eager_slot == 1

    # Restoring capacity allows the next valid call to consume slot 1 (0x200000), aligned with peers.
    runtime.channel._state.eager_buffer_bytes = shard_bytes
    fused(peer_xs[0], res, pp, pc, pre_mix=pm)
    assert runtime.channel._state.eager_slot == 2
    assert peer_state["calls"][-1]["peer_table_address"] == 0x200000


def test_runtime_operand_validation_rejects_bad_layout_dtype_and_alignment(monkeypatch):
    """_cluster_mhc_allreduce_post_y_norm rejects non-contiguous, mistyped, or misaligned operands before slot consumption."""
    tokens, hidden, world = 8, 5120, 4
    g = torch.Generator().manual_seed(77)
    peer_xs = [
        (torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.25).contiguous()
        for _ in range(world)
    ]
    peer_adds = [
        (torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.25).contiguous()
        for _ in range(world)
    ]
    peer_state = {"xs": peer_xs, "addends": peer_adds}
    _install_mock_b12x_fused(monkeypatch, peer_state=peer_state)
    monkeypatch.setattr(
        "hyperflare_recipes.components.mhc_overlap.register_finalizer",
        lambda session, fn: None,
    )

    runtime = _MockPCIeAllReduceRuntime(world_size=world, rank=0)
    weights = _make_mhc_weights(hidden, torch.bfloat16, seed=77)
    res = torch.randn((tokens, 4, hidden), dtype=torch.bfloat16, generator=g).contiguous()
    pp = torch.rand((tokens, 4), dtype=torch.float32, generator=g).contiguous()
    pc = torch.softmax(torch.randn((tokens, 4, 4), dtype=torch.float32, generator=g), dim=-1).contiguous()
    pm = torch.softmax(torch.randn((tokens, 4), dtype=torch.float32, generator=g), dim=-1).contiguous()

    fused = PreparedFusedMhcAllReducePostYNorm(
        (peer_xs[0], res, pp, pc),
        weights,
        runtime=runtime,
        session=object(),
        norm_eps=1e-6,
        has_addend=True,
        emit_mxfp8=True,
    ).functional()
    fused._finalize()

    # 1. Transposed BF16 x with matching [tokens, hidden] shape is non-contiguous.
    x_transposed = torch.randn((hidden, tokens), dtype=torch.bfloat16).T
    assert x_transposed.shape == (tokens, hidden) and not x_transposed.is_contiguous()
    with pytest.raises(ValueError, match="row-major contiguous"):
        fused(x_transposed, res, pp, pc, pre_mix=pm, input_addend=peer_adds[0])
    assert runtime.channel._state.eager_slot == 0

    # 2. FP32 input_addend is rejected instead of being misread as BF16.
    add_fp32 = torch.zeros((tokens, hidden), dtype=torch.float32)
    with pytest.raises(ValueError, match="input_addend requires dtype"):
        fused(peer_xs[0], res, pp, pc, pre_mix=pm, input_addend=add_fp32)
    assert runtime.channel._state.eager_slot == 0

    # 3. Contiguous but non-16-byte-aligned x is rejected.
    flat_x = torch.zeros(tokens * hidden + 1, dtype=torch.bfloat16)
    x_unaligned = flat_x[1:].view(tokens, hidden)
    assert x_unaligned.is_contiguous() and x_unaligned.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte aligned"):
        fused(x_unaligned, res, pp, pc, pre_mix=pm, input_addend=peer_adds[0])
    assert runtime.channel._state.eager_slot == 0

    # 4. Contiguous but non-16-byte-aligned input_addend is rejected.
    add_unaligned = flat_x[1:].view(tokens, hidden)
    with pytest.raises(ValueError, match="input_addend must be 16-byte aligned"):
        fused(peer_xs[0], res, pp, pc, pre_mix=pm, input_addend=add_unaligned)
    assert runtime.channel._state.eager_slot == 0

    # 5. Direct custom-op call with wrong output dtype or misaligned output is rejected.
    out_ok = torch.empty((tokens, 4, hidden), dtype=torch.bfloat16)
    y_ok = torch.empty((tokens, hidden), dtype=torch.bfloat16)
    av_ok = torch.empty((tokens, hidden), dtype=torch.uint8)
    as_ok = torch.empty((tokens, hidden // 32), dtype=torch.uint8)
    with pytest.raises(ValueError, match="norm_weight requires dtype"):
        torch.ops.flare.cluster_mhc_allreduce_post_y_norm(
            peer_xs[0],
            peer_adds[0],
            res,
            pp,
            pc,
            pm,
            weights.norm.float(),
            out_ok,
            y_ok,
            av_ok,
            as_ok,
            [],
            fused.handle,
        )
    assert runtime.channel._state.eager_slot == 0

    with pytest.raises(ValueError, match="prev_post requires dtype"):
        fused(peer_xs[0], res, pp.bfloat16(), pc, pre_mix=pm, input_addend=peer_adds[0])
    assert runtime.channel._state.eager_slot == 0

    assert len(peer_state.get("calls", [])) == 0


def test_transport_selection_forwarding_and_runner_cache_key_isolation(monkeypatch):
    """_transport forwards transport when supported and keeps (cs=8, oneshot) distinct from (cs=8, rsag)."""
    from b12x._lib import compile_plan
    from b12x.norm.mhc import _split_lagged
    from hyperflare_recipes.components import mhc_overlap

    before_runners = dict(mhc_overlap._SPLIT_MHC_RUNNERS)
    try:
        mhc_overlap._SPLIT_MHC_RUNNERS.clear()
        seen_transports = []

        def fake_supports(device, max_tokens, hidden_size, norm_dtype, world_size):
            return True

        def fake_compile_with_transport(
            ordinal,
            hidden,
            world_size,
            rank,
            norm_eps,
            norm_dtype,
            *,
            cluster_size=None,
            emit_mxfp8=False,
            has_addend=False,
            device_slot_selection=True,
            slot_bias=0,
            release_dependents=False,
            transport="auto",
        ):
            seen_transports.append((cluster_size, transport, device_slot_selection, slot_bias))
            return SimpleNamespace(tag=(cluster_size, transport, device_slot_selection, slot_bias))

        monkeypatch.setattr(_split_lagged, "supports_cluster_allreduce_post_y_norm", fake_supports, raising=False)
        monkeypatch.setattr(
            _split_lagged, "compile_cluster_allreduce_post_y_norm", fake_compile_with_transport, raising=False
        )
        monkeypatch.setattr(_split_lagged, "oneshot_channel_sentinel_params", _mock_sentinel_params, raising=False)
        monkeypatch.setattr(_split_lagged, "split_lagged_cluster_size", lambda m: 8 if m <= 16 else 4, raising=False)
        monkeypatch.setattr(
            _split_lagged,
            "cluster_allreduce_transport",
            lambda m, w: "oneshot" if (w <= 2 or m <= 4) else "rsag",
            raising=False,
        )
        monkeypatch.setattr(compile_plan, "load_programs", lambda r: None)
        monkeypatch.setattr(
            "hyperflare_recipes.components.mhc_overlap.register_finalizer",
            lambda session, fn: None,
        )

        hidden, world = 5120, 4
        runtime = _MockPCIeAllReduceRuntime(world_size=world, rank=0)
        weights = _make_mhc_weights(hidden, torch.bfloat16, seed=88)

        def make_fused(tokens):
            x = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
            res = torch.zeros((tokens, 4, hidden), dtype=torch.bfloat16)
            pp = torch.ones((tokens, 4), dtype=torch.float32)
            pc = torch.ones((tokens, 4, 4), dtype=torch.float32)
            f = PreparedFusedMhcAllReducePostYNorm(
                (x, res, pp, pc),
                weights,
                runtime=runtime,
                session=object(),
                norm_eps=1e-6,
                has_addend=False,
                emit_mxfp8=False,
            )
            f._finalize()
            return f

        f4 = make_fused(4)
        f8 = make_fused(8)
        assert f4._transport() == "oneshot"
        assert f8._transport() == "rsag"
        assert [t for _, t, _, _ in seen_transports] == [
            "oneshot", "oneshot", "oneshot", "rsag", "rsag", "rsag",
        ]
        # Both (cs=8, "oneshot") and (cs=8, "rsag") coexist in _SPLIT_MHC_RUNNERS (3 phases each = 6 entries).
        assert len(mhc_overlap._SPLIT_MHC_RUNNERS) == 6
        tags = {runner.tag[:2] for runner in mhc_overlap._SPLIT_MHC_RUNNERS.values()}
        assert tags == {(8, "oneshot"), (8, "rsag")}
    finally:
        mhc_overlap._SPLIT_MHC_RUNNERS.clear()
        mhc_overlap._SPLIT_MHC_RUNNERS.update(before_runners)
