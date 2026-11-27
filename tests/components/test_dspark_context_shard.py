# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Output-sharded mtp.0.main_proj loading, collective sizing, and 0-ULP PreparedContext parity."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from hyperflare_recipes.components import dspark
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.deepseek41 import bootstrap
from hyperflare_recipes.recipes.deepseek41.config import ModelConfig, QUALIFIED_ENCODER_SCHEDULE, TEXT_GEOMETRY
from hyperflare_recipes.recipes.deepseek41.pool import ContinuationPool
from hyperflare_recipes.recipes.deepseek41.weights import DSparkWeights, main_projection_partition
from ignition.components import collectives
from ignition.parameters import SpeculationParameters


def _deepseek41_config(*, hidden_size=5120):
    text = {
        **TEXT_GEOMETRY,
        **QUALIFIED_ENCODER_SCHEDULE,
        "hidden_size": hidden_size,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
        },
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
        "compress_ratios": [*QUALIFIED_ENCODER_SCHEDULE["compress_ratios"], 0, 0, 0],
    }
    quant = {
        "quant_method": "fp8",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "activation_scheme": "dynamic",
    }
    return ModelConfig(text, quant)


class _FakeLinearWeight:
    """Minimal packed-weight stand-in exposing ``out_features`` and exact BF16 matrix values."""

    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix.contiguous()
        self.out_features, self.in_features = self.matrix.shape


class _CpuLinearProjection:
    def __init__(self, source, weight, output, *, session, name, workspace, activation_mode="auto"):
        del session, workspace, activation_mode
        assert source.dtype == torch.bfloat16 and output.dtype == torch.bfloat16
        out_features = weight.shape[0] if isinstance(weight, torch.Tensor) else weight.out_features
        assert output.shape == (source.shape[0], out_features)
        self.weight = weight if isinstance(weight, torch.Tensor) else weight.matrix
        self.name = name

    def functional(self, *, session):
        del session
        return self

    def __call__(self, source):
        return torch.nn.functional.linear(source.float(), self.weight.float()).to(torch.bfloat16)


class _CpuRMSNorm:
    def __init__(self, source, weight, output, *, eps, session, name):
        del session, name
        assert source.shape == output.shape and weight.shape == (source.shape[1],)
        self.weight = weight
        self.eps = float(eps)

    def functional(self):
        return self

    def __call__(self, source):
        x = source.float()
        inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * inv * self.weight.float()).to(torch.bfloat16)


class _CpuRotary:
    def __init__(self, config, device):
        del config
        pair = torch.arange(32, dtype=torch.float32, device=device)
        self.frequencies = 1.0 / (10000.0 ** (2.0 * pair / 64.0))

    def apply(self, values, positions, *, compressed=False):
        del compressed
        angle = positions.to(torch.float32)[:, None] * self.frequencies[None, :]
        cos, sin = torch.cos(angle), torch.sin(angle)
        tail = values[:, -64:]
        a = tail[:, 0::2].float()
        b = tail[:, 1::2].float()
        rot_a = (a * cos - b * sin).to(values.dtype)
        rot_b = (a * sin + b * cos).to(values.dtype)
        rotated = torch.stack((rot_a, rot_b), dim=-1).reshape(values.shape[0], 64)
        values[:, -64:].copy_(rotated)


class _CpuSwaWriter:
    def __init__(self, values, window, slots, *, page_size, kind, session, name, cache_format):
        del values, slots, session, name, cache_format
        assert page_size == 128 and kind == "swa"
        self.window = window

    def functional(self, *, session):
        del session
        return self

    def __call__(self, values, slots):
        # `pool.window(index, draft=True)` is `[state_rows, 128 * 528]` strided across state rows.
        # Pack head + tail-RoPE BF16 bits (264 BF16 values = 528 bytes) into each slot record.
        record_bytes = self.window.shape[-1] // 128
        assert record_bytes == 528
        pages = self.window.view(self.window.shape[0], 128, record_bytes)
        payload = torch.cat((values[:, :200], values[:, -64:]), dim=-1).contiguous().view(torch.uint8)
        assert payload.shape[1] == record_bytes
        for row in range(values.shape[0]):
            slot = int(slots[row].item())
            if slot >= 0:
                pages[slot // 128, slot % 128].copy_(payload[row])


class _SimulatedGatherRuntime:
    """Multi-rank column AllGather simulator driving real PreparedChunkedAllGather / _ChunkedRoceAllGather."""

    def __init__(self, world_size, *, max_batch_size, max_gather_bytes=None, roce=False):
        self.world_size = world_size
        self.max_batch_size = max_batch_size
        self.max_gather_bytes = max_gather_bytes if max_gather_bytes is not None else (1 << 30)
        self.roce = roce
        self.records = []
        self._rank_inputs = {}
        self._active_rank = 0

    def set_rank_inputs(self, inputs_by_rank):
        self._rank_inputs = inputs_by_rank

    def set_active_rank(self, rank):
        self._active_rank = rank


@torch.library.custom_op("flare_test::sim_all_gather", mutates_args=("output",))
def _sim_all_gather_op(source: torch.Tensor, output: torch.Tensor, tag: str) -> None:
    runtime, begin, end = _SIM_GATHER_REGISTRY[tag]
    parts = [runtime._rank_inputs[r][begin:end] for r in range(runtime.world_size)]
    parts[runtime._active_rank] = source
    output.copy_(torch.cat(parts, dim=-1))


@_sim_all_gather_op.register_fake
def _fake_sim_all_gather_op(source, output, tag):
    del source, output, tag


_SIM_GATHER_REGISTRY = {}


class _SimPreparedAllGather:
    def __init__(self, source, output, *, runtime, session, coordinator, name, dependencies=(), has_gate=False):
        del session, coordinator, dependencies, has_gate
        self.runtime = runtime
        self.output = output
        self.name = name
        self.single = runtime is None
        if not self.single:
            assert source.ndim == 2 and output.ndim == 2
            assert source.dtype == torch.bfloat16 and output.dtype == torch.bfloat16
            assert output.shape == (source.shape[0], source.shape[1] * runtime.world_size)
            if runtime.roce:
                assert source.numel() * source.element_size() <= runtime.max_gather_bytes
            else:
                assert source.shape[0] <= runtime.max_batch_size
            runtime.records.append((name, tuple(source.shape), tuple(output.shape)))
            self._offset = int(source.storage_offset() // source.shape[1]) if source.numel() else 0
            self._rows = source.shape[0]
            self._tag = f"{id(self)}:{name}"
            _SIM_GATHER_REGISTRY[self._tag] = (runtime, self._offset, self._offset + self._rows)

    def functional(self, *, session):
        del session
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        return self

    def __call__(self, source, output=None):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        else:
            _sim_all_gather_op(source, output, self._tag)
        return output


def _patch_cpu_context_ops(monkeypatch):
    monkeypatch.setattr(dspark, "QuantizedProjection", _CpuLinearProjection)
    monkeypatch.setattr(dspark, "PreparedRMSNorm", _CpuRMSNorm)
    monkeypatch.setattr(dspark, "Rotary", _CpuRotary)
    monkeypatch.setattr(dspark, "PreparedCompressedCacheWriter", _CpuSwaWriter)
    monkeypatch.setattr(collectives, "PreparedAllGather", _SimPreparedAllGather)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: bool(getattr(runtime, "roce", False)))


def _make_synthetic_weights(config, *, hidden, world_size, rank, full_main, context_kv, main_norm, kv_norms, sharded):
    if sharded:
        part = main_projection_partition(hidden, TensorParallelShard(rank, world_size), block_size=32)
        assert part is not None
        r0, r1 = part
        main_w = _FakeLinearWeight(full_main[r0:r1])
    else:
        main_w = _FakeLinearWeight(full_main)
    draft = SimpleNamespace(
        main_projection=main_w,
        context_kv=_FakeLinearWeight(context_kv),
        main_norm=main_norm,
        layers=tuple(SimpleNamespace(attention={"kv_norm": norm}) for norm in kv_norms),
    )
    return SimpleNamespace(config=config, shard=TensorParallelShard(rank, world_size), draft=draft)


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    assert a.shape == b.shape and a.dtype == b.dtype
    view_dtype = torch.uint16 if a.element_size() == 2 else torch.uint8
    return torch.equal(a.contiguous().view(view_dtype), b.contiguous().view(view_dtype))


@pytest.mark.parametrize("world_size,expect_sharded", [
    (1, False),
    (2, True),
    (3, False),
    (4, True),
    (6, False),
    (8, True),
])
def test_main_projection_partition_and_dspark_weights_load(world_size, expect_sharded):
    config = _deepseek41_config(hidden_size=5120)
    torch.manual_seed(4100 + world_size)
    full_main = torch.randn((5120, 128), dtype=torch.bfloat16)
    loaded_shards = []

    for rank in range(world_size):
        shard = TensorParallelShard(rank, world_size)
        part = main_projection_partition(5120, shard, block_size=32)
        if expect_sharded:
            width = 5120 // world_size
            assert part == (rank * width, (rank + 1) * width)
        else:
            assert part is None

        recorded_calls = []

        class _FakeLoader:
            block_size = 32

            def __init__(self):
                self.config = config
                self.shard = shard

            def load_layer(self, index, *, draft=False):
                assert draft is True
                return SimpleNamespace(layer=index)

            def load_linear(self, prefix, *, rows=None, columns=None, temporary=False):
                recorded_calls.append(("load_linear", prefix, rows, columns, temporary))
                assert prefix == "mtp.0.main_proj"
                assert columns is None
                return full_main if rows is None else full_main[rows[0]:rows[1]].clone()

            def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None, pad=False, temporary=False):
                recorded_calls.append(("load_linear_group", tuple(prefixes), row_ranges, column_range))
                # context_kv must remain replicated (no row_ranges / column_range).
                assert tuple(prefixes) == ("mtp.0.attn.wkv", "mtp.1.attn.wkv", "mtp.2.attn.wkv")
                assert row_ranges is None and column_range is None
                return SimpleNamespace(prefixes=tuple(prefixes))

            def load_tensor(self, name, *, temporary=False):
                return SimpleNamespace(name=name, temporary=temporary)

            def load_padded_tensor(self, name, *, rows, temporary=False):
                return SimpleNamespace(name=name, rows=rows, temporary=temporary)

        draft_weights = DSparkWeights.load(_FakeLoader())
        if expect_sharded:
            assert draft_weights.main_projection.shape == (5120 // world_size, 128)
            loaded_shards.append(draft_weights.main_projection)
        else:
            assert draft_weights.main_projection.shape == (5120, 128)
            assert _bit_equal(draft_weights.main_projection, full_main)
        assert recorded_calls[0] == ("load_linear", "mtp.0.main_proj", part, None, False)
        assert recorded_calls[1] == (
            "load_linear_group", ("mtp.0.attn.wkv", "mtp.1.attn.wkv", "mtp.2.attn.wkv"), None, None
        )

    if expect_sharded:
        reconstructed = torch.cat(loaded_shards, dim=0)
        assert _bit_equal(reconstructed, full_main)


def test_context_gather_runtime_and_roce_sizing(monkeypatch):
    gather_calls = []
    roce_calls = []

    def fake_create_all_gather(group, device, *, max_rows, columns, ipc=None):
        token = SimpleNamespace(kind="pcie_gather", max_rows=max_rows, columns=columns, ipc=ipc)
        gather_calls.append((max_rows, columns))
        return token

    def fake_create_roce(group, device, *, max_tokens, pipe_tokens, hidden_size, gather_bytes, **kwargs):
        token = SimpleNamespace(kind="roce", gather_bytes=gather_bytes, max_tokens=max_tokens)
        roce_calls.append(gather_bytes)
        return token

    monkeypatch.setattr(bootstrap, "create_all_gather_runtime", fake_create_all_gather)
    monkeypatch.setattr(bootstrap, "create_roce_runtime", fake_create_roce)
    monkeypatch.setattr(bootstrap, "create_all_reduce_runtime", lambda *a, **kw: SimpleNamespace(kind="ar"))
    monkeypatch.setattr(bootstrap, "create_dma_all_reduce_runtime", lambda *a, **kw: SimpleNamespace(kind="dma"))
    monkeypatch.setattr(bootstrap, "create_result_broadcast_runtime", lambda *a, **kw: SimpleNamespace(kind="res"))

    storage = SimpleNamespace(peer_storage="ipc_handle")
    local_coord = SimpleNamespace(spans_hosts=False)
    roce_coord = SimpleNamespace(spans_hosts=True)

    base_params = bootstrap._default_parameters()

    # TP3 and TP6 keep main_proj replicated because 5120 is not divisible by 3*64 or 6*64.
    for devices in ((0, 1, 2), (0, 1, 2, 3, 4, 5)):
        cfg = bootstrap.PreparationConfig(devices=devices, parameters=base_params)
        assert bootstrap._context_gather_runtime(
            cfg, None, "cpu", coordinator=local_coord, core_storage=storage, collective="shared"
        ) is None

    # dspark=False disables context gather even on TP4.
    cfg_no_dspark = bootstrap.PreparationConfig(
        devices=(0, 1, 2, 3), parameters=replace(base_params, speculation=None)
    )
    assert bootstrap._context_gather_runtime(
        cfg_no_dspark, None, "cpu", coordinator=local_coord, core_storage=storage, collective="shared"
    ) is None

    # TP4 single-host allocates a 5120-column BF16 gather channel sized to max(128, decode_tokens).
    cfg_tp4 = bootstrap.PreparationConfig(
        devices=(0, 1, 2, 3), parameters=replace(base_params, max_concurrency=32)
    )
    ctx_rt = bootstrap._context_gather_runtime(
        cfg_tp4, None, "cpu", coordinator=local_coord, core_storage=storage, collective="shared"
    )
    assert ctx_rt.kind == "pcie_gather"
    assert ctx_rt.columns == 5120
    assert ctx_rt.max_rows == max(128, cfg_tp4.decode_tokens)
    assert ctx_rt.ipc == "ipc_handle"

    # TP4 multi-host reuses the shared RoCE runtime and sizes gather_bytes to cover context gather
    # even for the minimal max_active=1, speculative_widths=(1,) configuration.
    spec_width_1 = SpeculationParameters(
        draft_tokens=0, max_draft_tokens=7, token_capacity=128,
        packed_verification=False, draft_prefix_selection="cpu", graph_budget=None,
    )
    cfg_small_roce = bootstrap.PreparationConfig(
        devices=(0, 1, 2, 3),
        hosts=("h0", "h1", "h2", "h3"),
        parameters=replace(base_params, max_concurrency=1, speculation=spec_width_1),
    )
    roce_rt, *_ = bootstrap._collective_runtimes(
        cfg_small_roce, None, "cpu", coordinator=roce_coord, core_storage=storage, verification_samples=1
    )
    min_context_bytes = 128 * (5120 // 4) * 2
    assert roce_rt.gather_bytes >= min_context_bytes
    assert bootstrap._context_gather_runtime(
        cfg_small_roce, None, "cpu", coordinator=roce_coord, core_storage=storage, collective=roce_rt
    ) is roce_rt


@pytest.mark.parametrize("mode,tokens,max_batch_size,compact_prefill", [
    ("decode_1", 1, 128, False),
    ("speculative_8", 8, 128, False),
    ("speculative_32", 32, 128, False),
    ("speculative_128", 128, 128, False),
    ("compact_prefill_long", 128, 128, True),
    ("compact_prefill_short", 128, 128, True),
    ("scored_prefill_chunked", 300, 128, False),
])
def test_prepared_context_tp4_bit_exact_zero_ulp(monkeypatch, mode, tokens, max_batch_size, compact_prefill):
    _patch_cpu_context_ops(monkeypatch)
    torch.manual_seed(20261125 + tokens + (1000 if compact_prefill else 0))

    hidden = 512
    aux_width = 3 * hidden
    config = _deepseek41_config(hidden_size=hidden)
    full_main = (torch.randn((hidden, aux_width), dtype=torch.float32) * 0.05).to(torch.bfloat16)
    context_kv = (torch.randn((3 * 512, hidden), dtype=torch.float32) * 0.05).to(torch.bfloat16)
    main_norm = (1.0 + 0.1 * torch.randn(hidden, dtype=torch.float32)).to(torch.bfloat16)
    kv_norms = tuple((1.0 + 0.1 * torch.randn(512, dtype=torch.float32)).to(torch.bfloat16) for _ in range(3))

    if compact_prefill:
        total_positions = 512
        rows = 1
        metadata = Metadata.allocate(total_positions, rows, 4, "cpu")
        count_val = 300 if mode == "compact_prefill_long" else 37
        metadata.starts.fill_(113)
        metadata.lengths.fill_(count_val)
        metadata.state_rows.fill_(2)
        metadata.query_start_loc.copy_(torch.tensor([0, count_val], dtype=torch.int32))
        metadata.positions.fill_(-1)
        metadata.positions[:count_val].copy_(torch.arange(113, 113 + count_val, dtype=torch.int64))
        counts = torch.tensor([count_val], dtype=torch.int32)
    elif mode == "scored_prefill_chunked":
        rows = 1
        metadata = Metadata.allocate(tokens, rows, 4, "cpu")
        metadata.starts.fill_(50)
        metadata.lengths.fill_(tokens)
        metadata.state_rows.fill_(1)
        metadata.requests.fill_(0)
        metadata.positions.copy_(torch.arange(50, 50 + tokens, dtype=torch.int64))
        counts = torch.tensor([tokens], dtype=torch.int32)
    else:
        rows = min(4, tokens)
        metadata = Metadata.allocate(tokens, rows, 4, "cpu")
        metadata.starts.copy_(torch.arange(rows, dtype=torch.int64) * 40 + 7)
        metadata.state_rows.copy_(torch.arange(rows, dtype=torch.int32))
        metadata.positions.fill_(-1)
        metadata.requests.zero_()
        per_req = max(1, tokens // rows)
        counts_list = []
        cursor = 0
        for req in range(rows):
            n_tok = min(per_req, tokens - cursor)
            # Leave tail rows padded (-1) and mix partial / zero accepted counts.
            active_n = max(0, n_tok - (1 if n_tok > 2 and req % 2 == 1 else 0))
            for step in range(active_n):
                metadata.positions[cursor + step] = int(metadata.starts[req].item()) + step
                metadata.requests[cursor + step] = req
            accepted = max(0, active_n - (1 if active_n > 1 and req == rows - 1 else 0))
            counts_list.append(accepted)
            cursor += n_tok
        counts = torch.tensor(counts_list, dtype=torch.int32)

    auxiliary = torch.randn((tokens, aux_width), dtype=torch.float32).to(torch.bfloat16)

    # TP1 replicated reference.
    ref_pool = ContinuationPool.allocate(config, 6, 2, "cpu", dspark=True)
    ref_pool.backing.fill_(0x5A)
    ref_weights = _make_synthetic_weights(
        config, hidden=hidden, world_size=1, rank=0,
        full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=False,
    )
    ref_ctx = dspark.PreparedContext(
        ref_weights, auxiliary, metadata, ref_pool,
        session=None, name="ref", workspace=None, compact_prefill=compact_prefill,
    )
    assert ref_ctx.gather is None
    ref_projected = ref_ctx.project(auxiliary)
    ref_main = ref_ctx.norm(ref_projected)
    ref_kv = ref_ctx.context_projection(ref_main)
    ref_ctx(auxiliary, counts, metadata)

    # TP4 output-sharded execution across all 4 ranks.
    world = 4
    sim_runtime = _SimulatedGatherRuntime(world, max_batch_size=max_batch_size)
    rank_contexts = []
    rank_pools = []
    for rank in range(world):
        pool = ContinuationPool.allocate(config, 6, 2, "cpu", dspark=True)
        pool.backing.fill_(0x5A)
        weights = _make_synthetic_weights(
            config, hidden=hidden, world_size=world, rank=rank,
            full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=True,
        )
        ctx = dspark.PreparedContext(
            weights, auxiliary, metadata, pool,
            session=None, name=f"tp4.r{rank}", workspace=None,
            compact_prefill=compact_prefill, runtime=sim_runtime, coordinator=None,
        )
        assert ctx.gather is not None
        assert isinstance(ctx.project, dspark._ShardedMainProjection)
        rank_contexts.append(ctx)
        rank_pools.append(pool)

    # Verify chunking count for scored_prefill vs single gather for tokens <= max_batch_size.
    expected_chunks = (tokens + max_batch_size - 1) // max_batch_size
    assert len(rank_contexts[0].gather.parts) == expected_chunks

    # Compute every rank's local shard output first, then run each rank through the collective gather.
    local_outputs = {rank: rank_contexts[rank].project.local(auxiliary) for rank in range(world)}
    for rank in range(world):
        assert local_outputs[rank].shape == (tokens, hidden // world)
    sim_runtime.set_rank_inputs(local_outputs)

    for rank in range(world):
        sim_runtime.set_active_rank(rank)
        ctx = rank_contexts[rank]
        projected = ctx.project(auxiliary)
        main = ctx.norm(projected)
        kv = ctx.context_projection(main)
        ctx(auxiliary, counts, metadata)

        assert _bit_equal(projected, ref_projected)
        assert _bit_equal(main, ref_main)
        assert _bit_equal(kv, ref_kv)
        assert torch.equal(rank_pools[rank].backing, ref_pool.backing)

    # Confirm untouched sentinel slots remain 0x5A (no writes to inactive/unused state row 5).
    for layer_idx in range(3):
        assert torch.all(rank_pools[0].window(layer_idx, draft=True)[5] == 0x5A)


def test_prepared_context_roce_chunked_gather_bit_exact(monkeypatch):
    _patch_cpu_context_ops(monkeypatch)
    torch.manual_seed(9988)

    hidden = 512
    aux_width = 3 * hidden
    tokens = 260
    world = 4
    config = _deepseek41_config(hidden_size=hidden)
    full_main = (torch.randn((hidden, aux_width), dtype=torch.float32) * 0.05).to(torch.bfloat16)
    context_kv = (torch.randn((3 * 512, hidden), dtype=torch.float32) * 0.05).to(torch.bfloat16)
    main_norm = torch.ones(hidden, dtype=torch.bfloat16)
    kv_norms = tuple(torch.ones(512, dtype=torch.bfloat16) for _ in range(3))
    auxiliary = torch.randn((tokens, aux_width), dtype=torch.float32).to(torch.bfloat16)

    metadata = Metadata.allocate(tokens, 1, 4, "cpu")
    metadata.starts.fill_(10)
    metadata.lengths.fill_(tokens)
    metadata.state_rows.fill_(0)
    metadata.requests.fill_(0)
    metadata.positions.copy_(torch.arange(10, 10 + tokens, dtype=torch.int64))
    counts = torch.tensor([tokens], dtype=torch.int32)

    # Reference TP1.
    ref_pool = ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True)
    ref_pool.backing.fill_(0x33)
    ref_weights = _make_synthetic_weights(
        config, hidden=hidden, world_size=1, rank=0,
        full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=False,
    )
    ref_ctx = dspark.PreparedContext(ref_weights, auxiliary, metadata, ref_pool, session=None, name="ref", workspace=None)
    ref_ctx(auxiliary, counts, metadata)

    # Cap RoCE max_gather_bytes to 100 rows * (128 * 2 bytes) = 25600 bytes -> 3 chunks (100, 100, 60).
    row_bytes = (hidden // world) * 2
    roce_runtime = _SimulatedGatherRuntime(world, max_batch_size=1, max_gather_bytes=100 * row_bytes, roce=True)
    rank_contexts = []
    rank_pools = []
    for rank in range(world):
        pool = ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True)
        pool.backing.fill_(0x33)
        weights = _make_synthetic_weights(
            config, hidden=hidden, world_size=world, rank=rank,
            full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=True,
        )
        ctx = dspark.PreparedContext(
            weights, auxiliary, metadata, pool,
            session=None, name=f"roce.r{rank}", workspace=None, runtime=roce_runtime, coordinator=None,
        )
        assert isinstance(ctx.gather, dspark._ChunkedRoceAllGather)
        assert ctx.gather.bounds == ((0, 100), (100, 200), (200, 260))
        rank_contexts.append(ctx)
        rank_pools.append(pool)

    local_outputs = {r: rank_contexts[r].project.local(auxiliary) for r in range(world)}
    roce_runtime.set_rank_inputs(local_outputs)
    for rank in range(world):
        roce_runtime.set_active_rank(rank)
        rank_contexts[rank](auxiliary, counts, metadata)
        assert torch.equal(rank_pools[rank].backing, ref_pool.backing)

    # Capacity smaller than a single row must raise explicitly.
    tiny_roce = _SimulatedGatherRuntime(world, max_batch_size=1, max_gather_bytes=row_bytes - 16, roce=True)
    with pytest.raises(ValueError, match="exceeds the RoCE all-gather capacity"):
        dspark.PreparedContext(
            _make_synthetic_weights(
                config, hidden=hidden, world_size=world, rank=0,
                full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=True,
            ),
            auxiliary, metadata, ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True),
            session=None, name="tiny_roce", workspace=None, runtime=tiny_roce, coordinator=None,
        )


def test_prepared_context_backwards_compatibility_and_validation(monkeypatch):
    _patch_cpu_context_ops(monkeypatch)
    hidden = 512
    aux_width = 3 * hidden
    tokens = 8
    config = _deepseek41_config(hidden_size=hidden)
    full_main = torch.randn((hidden, aux_width), dtype=torch.bfloat16)
    context_kv = torch.randn((3 * 512, hidden), dtype=torch.bfloat16)
    main_norm = torch.ones(hidden, dtype=torch.bfloat16)
    kv_norms = tuple(torch.ones(512, dtype=torch.bfloat16) for _ in range(3))
    auxiliary = torch.randn((tokens, aux_width), dtype=torch.bfloat16)
    metadata = Metadata.allocate(tokens, 1, 2, "cpu")
    metadata.starts.fill_(0)
    metadata.state_rows.fill_(0)
    metadata.requests.fill_(0)
    metadata.positions.copy_(torch.arange(tokens, dtype=torch.int64))
    counts = torch.tensor([tokens], dtype=torch.int32)

    # 1) Unsharded weight with world_size=4 and runtime=None (and with non-None runtime) stays unsharded.
    for rt in (None, _SimulatedGatherRuntime(4, max_batch_size=128)):
        pool = ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True)
        weights = _make_synthetic_weights(
            config, hidden=hidden, world_size=4, rank=0,
            full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=False,
        )
        ctx = dspark.PreparedContext(
            weights, auxiliary, metadata, pool, session=None, name="unsharded", workspace=None, runtime=rt
        )
        assert ctx.gather is None
        ctx(auxiliary, counts, metadata)

    # 2) Legacy test weights without `.shard` attribute continue to work with runtime=None.
    legacy_weights = SimpleNamespace(
        config=config,
        draft=SimpleNamespace(
            main_projection=_FakeLinearWeight(full_main),
            context_kv=_FakeLinearWeight(context_kv),
            main_norm=main_norm,
            layers=tuple(SimpleNamespace(attention={"kv_norm": norm}) for norm in kv_norms),
        ),
    )
    legacy_ctx = dspark.PreparedContext(
        legacy_weights, auxiliary, metadata, ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True),
        session=None, name="legacy", workspace=None,
    )
    assert legacy_ctx.gather is None
    legacy_ctx(auxiliary, counts, metadata)

    # 3) Sharded weight with runtime=None must fail explicitly.
    sharded_weights = _make_synthetic_weights(
        config, hidden=hidden, world_size=4, rank=0,
        full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=True,
    )
    with pytest.raises(ValueError, match="requires a collective all-gather runtime"):
        dspark.PreparedContext(
            sharded_weights, auxiliary, metadata, ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True),
            session=None, name="missing_rt", workspace=None, runtime=None,
        )

    # 4) Mismatched output width must fail explicitly.
    bad_weights = SimpleNamespace(
        config=config,
        shard=TensorParallelShard(0, 4),
        draft=SimpleNamespace(
            main_projection=_FakeLinearWeight(full_main[:192]),
            context_kv=_FakeLinearWeight(context_kv),
            main_norm=main_norm,
            layers=tuple(SimpleNamespace(attention={"kv_norm": norm}) for norm in kv_norms),
        ),
    )
    with pytest.raises(ValueError, match="must equal hidden_size"):
        dspark.PreparedContext(
            bad_weights, auxiliary, metadata, ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True),
            session=None, name="bad_width", workspace=None, runtime=_SimulatedGatherRuntime(4, max_batch_size=128),
        )


def test_prepared_context_negative_sensitivity_corrupted_shard_breaks_parity(monkeypatch):
    _patch_cpu_context_ops(monkeypatch)
    torch.manual_seed(777)

    hidden = 512
    aux_width = 3 * hidden
    tokens = 16
    world = 4
    config = _deepseek41_config(hidden_size=hidden)
    full_main = (torch.randn((hidden, aux_width), dtype=torch.float32) * 0.05).to(torch.bfloat16)
    context_kv = (torch.randn((3 * 512, hidden), dtype=torch.float32) * 0.05).to(torch.bfloat16)
    main_norm = torch.ones(hidden, dtype=torch.bfloat16)
    kv_norms = tuple(torch.ones(512, dtype=torch.bfloat16) for _ in range(3))
    auxiliary = torch.randn((tokens, aux_width), dtype=torch.float32).to(torch.bfloat16)

    metadata = Metadata.allocate(tokens, 1, 2, "cpu")
    metadata.starts.fill_(0)
    metadata.state_rows.fill_(0)
    metadata.requests.fill_(0)
    metadata.positions.copy_(torch.arange(tokens, dtype=torch.int64))
    counts = torch.tensor([tokens], dtype=torch.int32)

    ref_pool = ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True)
    ref_pool.backing.zero_()
    ref_weights = _make_synthetic_weights(
        config, hidden=hidden, world_size=1, rank=0,
        full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=False,
    )
    ref_ctx = dspark.PreparedContext(ref_weights, auxiliary, metadata, ref_pool, session=None, name="ref", workspace=None)
    ref_projected = ref_ctx.project(auxiliary)
    ref_ctx(auxiliary, counts, metadata)

    # Corrupt rank 2's weight shard and verify parity fails across projected output and SWA cache.
    corrupted_main = full_main.clone()
    corrupted_main[2 * (hidden // world) + 5, 13] = corrupted_main[2 * (hidden // world) + 5, 13] + 1.0
    sim_runtime = _SimulatedGatherRuntime(world, max_batch_size=128)
    contexts = []
    pools = []
    for rank in range(world):
        pool = ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True)
        pool.backing.zero_()
        src = corrupted_main if rank == 2 else full_main
        weights = _make_synthetic_weights(
            config, hidden=hidden, world_size=world, rank=rank,
            full_main=src, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=True,
        )
        contexts.append(dspark.PreparedContext(
            weights, auxiliary, metadata, pool,
            session=None, name=f"corrupt.r{rank}", workspace=None, runtime=sim_runtime, coordinator=None,
        ))
        pools.append(pool)

    sim_runtime.set_rank_inputs({r: contexts[r].project.local(auxiliary) for r in range(world)})
    sim_runtime.set_active_rank(0)
    actual_projected = contexts[0].project(auxiliary)
    contexts[0](auxiliary, counts, metadata)
    assert not _bit_equal(actual_projected, ref_projected)
    assert not torch.equal(pools[0].backing, ref_pool.backing)


def test_prepared_context_make_fx_tracing(monkeypatch):
    _patch_cpu_context_ops(monkeypatch)
    torch.manual_seed(31415)

    hidden = 512
    aux_width = 3 * hidden
    tokens = 16
    world = 4
    config = _deepseek41_config(hidden_size=hidden)
    full_main = (torch.randn((hidden, aux_width), dtype=torch.float32) * 0.05).to(torch.bfloat16)
    context_kv = (torch.randn((3 * 512, hidden), dtype=torch.float32) * 0.05).to(torch.bfloat16)
    main_norm = torch.ones(hidden, dtype=torch.bfloat16)
    kv_norms = tuple(torch.ones(512, dtype=torch.bfloat16) for _ in range(3))
    auxiliary = torch.randn((tokens, aux_width), dtype=torch.float32).to(torch.bfloat16)

    metadata = Metadata.allocate(tokens, 1, 2, "cpu")
    metadata.starts.fill_(4)
    metadata.state_rows.fill_(0)
    metadata.requests.fill_(0)
    metadata.positions.copy_(torch.arange(4, 4 + tokens, dtype=torch.int64))
    counts = torch.tensor([tokens], dtype=torch.int32)

    # Replace writer with a tensor-only FX-traceable scatter so make_fx traces the full graph end-to-end.
    class _FxSwaWriter:
        def __init__(self, values, window, slots, **kwargs):
            del values, slots, kwargs
            self.window = window

        def functional(self, *, session):
            del session
            return self

        def __call__(self, values, slots):
            record_bytes = self.window.shape[-1] // 128
            pages = self.window.view(self.window.shape[0], 128, record_bytes)
            payload = torch.cat((values[:, :200], values[:, -64:]), dim=-1).contiguous().view(torch.uint8)
            valid = slots >= 0
            safe_slots = slots.clamp_min(0)
            row_idx = safe_slots // 128
            pos_idx = safe_slots % 128
            current = pages[row_idx, pos_idx]
            updated = torch.where(valid[:, None], payload, current)
            pages[row_idx, pos_idx] = updated

    monkeypatch.setattr(dspark, "PreparedCompressedCacheWriter", _FxSwaWriter)

    # Trace sharded TP4 context and verify custom op presence + bit-exact traced execution.
    sim_runtime = _SimulatedGatherRuntime(world, max_batch_size=128)
    sharded_pools = []
    sharded_ctxs = []
    for rank in range(world):
        pool = ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True)
        pool.backing.fill_(0x22)
        weights = _make_synthetic_weights(
            config, hidden=hidden, world_size=world, rank=rank,
            full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=True,
        )
        sharded_ctxs.append(dspark.PreparedContext(
            weights, auxiliary, metadata, pool,
            session=None, name=f"fx.r{rank}", workspace=None, runtime=sim_runtime, coordinator=None,
        ))
        sharded_pools.append(pool)

    sim_runtime.set_rank_inputs({r: sharded_ctxs[r].project.local(auxiliary) for r in range(world)})
    sim_runtime.set_active_rank(0)

    def sharded_fn(aux, cnt, positions, starts, requests, state_rows):
        m = SimpleNamespace(
            positions=positions, starts=starts, requests=requests, state_rows=state_rows,
            query_start_loc=metadata.query_start_loc,
        )
        sharded_ctxs[0](aux, cnt, m)
        return sharded_ctxs[0].project(aux)

    traced_sharded = make_fx(sharded_fn)(
        auxiliary, counts, metadata.positions, metadata.starts, metadata.requests, metadata.state_rows
    )
    op_targets = [str(node.target) for node in traced_sharded.graph.nodes if node.op == "call_function"]
    assert any("sim_all_gather" in target for target in op_targets)

    # Compare traced execution against unsharded eager TP1 reference.
    ref_pool = ContinuationPool.allocate(config, 2, 2, "cpu", dspark=True)
    ref_pool.backing.fill_(0x22)
    ref_weights = _make_synthetic_weights(
        config, hidden=hidden, world_size=1, rank=0,
        full_main=full_main, context_kv=context_kv, main_norm=main_norm, kv_norms=kv_norms, sharded=False,
    )
    ref_ctx = dspark.PreparedContext(
        ref_weights, auxiliary, metadata, ref_pool, session=None, name="fx.ref", workspace=None
    )

    def unsharded_fn(aux, cnt, positions, starts, requests, state_rows):
        m = SimpleNamespace(
            positions=positions, starts=starts, requests=requests, state_rows=state_rows,
            query_start_loc=metadata.query_start_loc,
        )
        ref_ctx(aux, cnt, m)
        return ref_ctx.project(aux)

    traced_unsharded = make_fx(unsharded_fn)(
        auxiliary, counts, metadata.positions, metadata.starts, metadata.requests, metadata.state_rows
    )
    unsharded_targets = [str(node.target) for node in traced_unsharded.graph.nodes if node.op == "call_function"]
    assert not any("sim_all_gather" in target for target in unsharded_targets)

    sharded_pools[0].backing.fill_(0x22)
    ref_pool.backing.fill_(0x22)
    out_sharded = traced_sharded(
        auxiliary, counts, metadata.positions, metadata.starts, metadata.requests, metadata.state_rows
    )
    out_unsharded = traced_unsharded(
        auxiliary, counts, metadata.positions, metadata.starts, metadata.requests, metadata.state_rows
    )
    assert _bit_equal(out_sharded, out_unsharded)
    assert torch.equal(sharded_pools[0].backing, ref_pool.backing)
