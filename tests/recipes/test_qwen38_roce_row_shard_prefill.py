# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 RoCE Sharded QSA and RoCE row-sharded prefill CPU unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import threading

import pytest
import torch

from b12x.comm import pcie
from hyperflare_recipes.components.dflash2 import PreparedContextCapture
from hyperflare_recipes.components.gated_delta import PreparedGDN
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.qwen38 import (
    attention as qwen_attention,
    bootstrap as qwen_bootstrap,
    decoder as qwen_decoder,
    qsa_sharded,
    recipe as qwen_recipe,
)
from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig
from ignition.planning import Gpu, Plan, UsageError


def _gb10_gpus(count: int, capability: tuple[int, int] = (12, 1)) -> tuple[Gpu, ...]:
    return tuple(
        Gpu(i, "NVIDIA GB10", f"GPU-{i}", 120 << 30, 110 << 30, capability)
        for i in range(count)
    )


def _make_qwen_plan(*, gpus: tuple[Gpu, ...], row_shard_prefill: bool) -> Plan:
    return Plan(
        recipe=qwen_recipe.RECIPE,
        checkpoint=qwen_recipe.RECIPE.checkpoints[0],
        gpus=gpus,
        concurrency=16,
        prefill_chunk=8192,
        speculation=qwen_recipe.RECIPE.speculation,
        cache_target=8192,
        cache_states=64,
        pages=1024,
        reserve_bytes=0,
        weights_bytes=1,
        graph_bytes=1,
        image_concurrency=2,
        row_shard_prefill=row_shard_prefill,
    )


def test_qwen38_recipe_config_plan_and_identity_row_shard_prefill(monkeypatch):
    assert qwen_recipe.RECIPE.row_shard_prefill_world_sizes == (2, 4)

    default_cfg = PreparationConfig()
    assert default_cfg.row_shard_prefill is False
    assert PreparationConfig(devices=(0, 1), row_shard_prefill=True).row_shard_prefill is True
    assert PreparationConfig(devices=(0, 1, 2, 3), row_shard_prefill=True).row_shard_prefill is True

    for invalid in (1, 0, "true", None):
        with pytest.raises(TypeError, match="row_shard_prefill must be a boolean"):
            PreparationConfig(devices=(0, 1), row_shard_prefill=invalid)

    with pytest.raises(ValueError, match="--row-shard-prefill requires tensor parallelism"):
        PreparationConfig(devices=(0,), row_shard_prefill=True)

    with pytest.raises(ValueError, match="--row-shard-prefill cannot be combined with --shard-hyperconnections"):
        PreparationConfig(devices=(0, 1), row_shard_prefill=True, shard_hyperconnections=True)

    for world in (2, 4):
        plan = _make_qwen_plan(gpus=_gb10_gpus(world), row_shard_prefill=True)
        assert plan.row_shard_prefill is True

    with pytest.raises(UsageError, match="--row-shard-prefill requires tensor parallelism"):
        _make_qwen_plan(gpus=_gb10_gpus(1), row_shard_prefill=True)

    with pytest.raises(UsageError, match="not qualified on SM103"):
        _make_qwen_plan(gpus=_gb10_gpus(2, capability=(10, 3)), row_shard_prefill=True)

    captured = {}

    def fake_prepare_group(snapshot, config, prepare_rank, **kwargs):
        captured["config"] = config
        return "prepared"

    import ignition.preparation.rank_group as rank_group

    monkeypatch.setattr(rank_group, "prepare_group", fake_prepare_group)
    tp2_plan = _make_qwen_plan(gpus=_gb10_gpus(2), row_shard_prefill=True)
    assert qwen_recipe.prepare(
        "/tmp/snapshot", tp2_plan, compile_forwards=True, kernel_cache=None, progress=None
    ) == "prepared"
    assert captured["config"].row_shard_prefill is True

    manifest = SimpleNamespace(headers_sha256="hdr", metadata_sha256={"config.json": "cfg"})
    cfg_default = PreparationConfig(devices=(0, 1), row_shard_prefill=False)
    cfg_sharded = PreparationConfig(devices=(0, 1), row_shard_prefill=True)
    id_default = qwen_bootstrap._identity(manifest, world_size=2, config=cfg_default)
    id_sharded = qwen_bootstrap._identity(manifest, world_size=2, config=cfg_sharded)
    assert id_default != id_sharded


class _FakeRoceRuntime:
    """Minimal RoCE runtime double matching ignition.components.collectives contracts."""

    def __init__(self, world_size: int = 2, *, pipe_bytes: int = 8192 * 2560 * 2,
                 max_gather_bytes: int = 32 << 20, abi: int = 11):
        self.world_size = world_size
        self.pipe_bytes = pipe_bytes
        self.max_gather_bytes = max_gather_bytes
        self.abi = abi
        if abi >= 11:
            self.should_reduce_scatter = self._should_reduce_scatter
            self.should_gather_shards = self._should_gather_shards

    def _should_reduce_scatter(self, tensor: torch.Tensor) -> bool:
        nbytes = tensor.numel() * tensor.element_size()
        row_bytes = tensor.numel() // max(tensor.shape[0], 1) * tensor.element_size()
        return (
            tensor.dtype is torch.bfloat16
            and tensor.is_contiguous()
            and tensor.shape[0] % self.world_size == 0
            and row_bytes % 16 == 0
            and 0 < nbytes <= self.pipe_bytes
        )

    def _should_gather_shards(self, tensor: torch.Tensor) -> bool:
        nbytes = tensor.numel() * tensor.element_size() * self.world_size
        row_bytes = tensor.numel() // max(tensor.shape[0], 1) * tensor.element_size()
        return (
            tensor.dtype is torch.bfloat16
            and tensor.is_contiguous()
            and row_bytes % 16 == 0
            and 0 < nbytes <= self.pipe_bytes
        )

    def should_allreduce(self, tensor: torch.Tensor) -> bool:
        return True


class _FakeDmaRuntime:
    def __init__(self, world_size: int = 2, *, max_bytes: int = 64 << 20, shard_capacity: int = 32 << 20):
        self.world_size = world_size
        self.max_bytes = max_bytes
        self.shard_capacity = shard_capacity
        self.should_reduce_scatter = lambda tensor: True
        self.should_all_gather = lambda tensor: True

    def should_allreduce(self, tensor: torch.Tensor) -> bool:
        return True


@pytest.mark.parametrize("world_size,max_active,prefill_chunk", [
    (2, 2, 8192),    # small vocab_rows -> qsa_prefill_gather_bytes (8 MiB) dominates gather_bytes!
    (2, 16, 8192),   # TP2 8K chunk
    (4, 2, 8192),    # TP4 small concurrency -> 4 MiB QSA prefill gather dominates
    (4, 16, 4096),   # TP4 4K chunk
])
def test_roce_sharded_qsa_bootstrap_gather_bytes_and_wiring(monkeypatch, world_size, max_active, prefill_chunk):
    """RoCE bootstrap sizes gather_bytes and wires QSA gather runtime/mode and row_shard_prefill into decoders."""
    import ignition.components.collectives as collectives_mod
    import ignition.components.readiness as readiness_mod
    import hyperflare_recipes.recipes.qwen38.vision_state as vision_state_mod
    import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_mod

    recorded = {"roce_kwargs": None, "decoder_calls": [], "namespace": None}
    fake_roce = _FakeRoceRuntime(world_size=world_size)

    def fake_create_roce_runtime(group, device, **kwargs):
        recorded["roce_kwargs"] = kwargs
        return fake_roce

    class FakeCoordinator:
        def __init__(self, group):
            self.spans_hosts = True

    class StopAfterDecoders(Exception):
        pass

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            pass

    class FakeScratch:
        sample_hidden = torch.zeros((max_active, 2560), dtype=torch.bfloat16)
        logits = torch.zeros((max_active, 32), dtype=torch.bfloat16)
        local_logits = None

        def rows(self, capacity, rows):
            return SimpleNamespace(embedding=torch.zeros((capacity, 2560), dtype=torch.bfloat16))

    class FakeInputs:
        vision_token_indices = torch.zeros(prefill_chunk, dtype=torch.int64)
        vision_feature_slot = torch.zeros(1, dtype=torch.int64)
        row_slot_ids = torch.zeros(1, dtype=torch.int64)
        row_generations = torch.zeros(1, dtype=torch.int64)
        query_lengths = torch.zeros(1, dtype=torch.int64)
        live_rows = torch.zeros(1, dtype=torch.int32)

        def buffers(self):
            return {"input_ids": torch.zeros(1, dtype=torch.int64)}

    class FakeDecoder:
        def __init__(self, *args, **kwargs):
            state_meta = args[2]
            recorded["decoder_calls"].append({
                "name": kwargs["name"],
                "mode": state_meta.mode,
                "capacity": state_meta.tokens,
                "qsa_gather_runtime": kwargs.get("qsa_gather_runtime"),
                "qsa_shard_mode": kwargs.get("qsa_shard_mode"),
                "row_shard_prefill": kwargs.get("row_shard_prefill"),
            })

        def functional(self, *, session):
            return self

        def release_outputs(self):
            pass

    monkeypatch.setattr(qwen_bootstrap, "CollectivePreparation", FakeCoordinator)
    monkeypatch.setattr(qwen_bootstrap, "create_roce_runtime", fake_create_roce_runtime)
    monkeypatch.setattr(qwen_bootstrap, "inspect_qwen38_checkpoint", lambda snapshot: SimpleNamespace(
        snapshot=snapshot,
        headers_sha256="hdr",
        metadata_sha256={"config.json": "cfg"},
        shard_paths=(),
    ))
    monkeypatch.setattr(qwen_bootstrap, "ModelWeights", SimpleNamespace(
        load=lambda *a, **kw: SimpleNamespace(
            shard=TensorParallelShard(0, world_size),
            tree=SimpleNamespace(
                config=SimpleNamespace(text_config=SimpleNamespace(), vision_config=SimpleNamespace()),
                lm_head=SimpleNamespace(weight=None),
            ),
            vision=SimpleNamespace(),
            allocated_bytes=0,
        )
    ))
    monkeypatch.setattr(qwen_bootstrap, "PLETable", SimpleNamespace(
        load=lambda *a, **kw: SimpleNamespace(close=lambda: None, buffers=lambda: {})
    ))
    monkeypatch.setattr(torch.cuda, "set_device", lambda dev: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda dev: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (64 << 30, 128 << 30))
    monkeypatch.setattr(collectives_mod, "packed_broadcast_layout", lambda vals: ((), 64))
    monkeypatch.setattr(qwen_bootstrap, "preparation_session", lambda **kw: (
        recorded.__setitem__("namespace", kw["namespace"]) or FakeSession()
    ))
    monkeypatch.setattr(qwen_bootstrap, "HybridStateLayout", SimpleNamespace(
        build=lambda **kw: SimpleNamespace(snapshot_capacity=lambda budget: 4, row_bytes=1024)
    ))
    monkeypatch.setattr(qwen_bootstrap, "HybridStatePool", SimpleNamespace(
        allocate=lambda *a, **kw: SimpleNamespace(
            gdn={0: SimpleNamespace(recurrent=torch.zeros(1))},
            backing=torch.zeros(1),
            reset_for_preparation=lambda: None,
            memory_capacity=torch.zeros(2, dtype=torch.int64),
        )
    ))
    monkeypatch.setattr(qwen_bootstrap, "register_benchmark_state", lambda *a, **kw: None)
    monkeypatch.setattr(qwen_bootstrap, "DeviceRequestState", SimpleNamespace(
        allocate=lambda *a, **kw: SimpleNamespace(grammar_mask=torch.zeros(1, dtype=torch.uint8))
    ))
    monkeypatch.setattr(qwen_bootstrap, "DecoderInputs", SimpleNamespace(allocate=lambda *a, **kw: FakeInputs()))
    monkeypatch.setattr(qwen_bootstrap, "DecoderScratch", SimpleNamespace(allocate=lambda *a, **kw: FakeScratch()))
    monkeypatch.setattr(qwen_bootstrap, "ScratchArena", lambda *a, **kw: SimpleNamespace(overlap=True))
    monkeypatch.setattr(qwen_bootstrap, "RotaryTables", SimpleNamespace(allocate=lambda *a, **kw: SimpleNamespace()))
    monkeypatch.setattr(qwen_bootstrap, "QSAState", SimpleNamespace(
        allocate=lambda *a, **kw: SimpleNamespace(
            main_k_cache=torch.zeros(16, dtype=torch.uint8),
            main_v_cache=torch.zeros(16, dtype=torch.uint8),
            compressed_k_cache=torch.zeros(16, dtype=torch.uint8),
        )
    ))
    monkeypatch.setattr(qwen_bootstrap, "minimum_across_ranks", lambda val, group: val)
    monkeypatch.setattr(qwen_bootstrap, "AdmissionBuffers", SimpleNamespace(
        allocate=lambda *a, **kw: SimpleNamespace(__dataclass_fields__={})
    ))
    monkeypatch.setattr(qwen_bootstrap, "StateTransferMetadata", SimpleNamespace(
        allocate=lambda *a, **kw: SimpleNamespace(__dataclass_fields__={})
    ))
    monkeypatch.setattr(readiness_mod, "ReadinessTimings", SimpleNamespace(allocate=lambda *a, **kw: SimpleNamespace()))
    monkeypatch.setattr(qwen_bootstrap, "CompletionOutputs", SimpleNamespace(
        allocate=lambda *a, **kw: SimpleNamespace(buffers=lambda: {})
    ))
    monkeypatch.setattr(vision_state_mod, "VisionFeatureState", SimpleNamespace(allocate=lambda *a, **kw: SimpleNamespace()))
    monkeypatch.setattr(vision_state_mod, "VisionInputs", SimpleNamespace(
        allocate=lambda *a, **kw: SimpleNamespace(__dataclass_fields__={})
    ))
    monkeypatch.setattr(vision_state_mod, "PreparedVisionHandoff", lambda *a, **kw: SimpleNamespace(
        metadata=SimpleNamespace(), prepare=lambda: None, functional=lambda: None
    ))
    monkeypatch.setattr(vision_state_mod, "PreparedVisionGeneration", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(vision_state_mod, "PreparedEmbeddingInjection", lambda *a, **kw: SimpleNamespace(prepare=lambda: None))
    monkeypatch.setattr(qwen_bootstrap, "VisionScratch", SimpleNamespace(
        allocate=lambda *a, **kw: SimpleNamespace(output=None, patches=None)
    ))
    monkeypatch.setattr(qwen4_mod, "Qwen4ExpVisionModel", lambda cfg: SimpleNamespace(
        eval=lambda: SimpleNamespace(requires_grad_=lambda flag: None)
    ))
    monkeypatch.setattr(qwen_bootstrap, "PreparedVision", lambda *a, **kw: SimpleNamespace(functional=lambda **fkw: None))
    monkeypatch.setattr(qwen_bootstrap, "PromptScoring", SimpleNamespace(allocate=lambda *a, **kw: None))
    monkeypatch.setattr(qwen_bootstrap, "PreparedDecoderState", lambda *a, **kw: SimpleNamespace(
        tokens=kw["tokens"], rows=kw["rows"], mode=kw["mode"]
    ))
    monkeypatch.setattr(qwen_bootstrap, "PreparedDecoder", FakeDecoder)
    monkeypatch.setattr(qwen_bootstrap, "PreparedResume", lambda *a, **kw: (_ for _ in ()).throw(StopAfterDecoders()))
    orig_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: orig_zeros(*a, **{**kw, "device": "cpu"}))

    cfg = PreparationConfig(
        devices=(0,) * world_size,
        hosts=tuple(f"h{i}" for i in range(world_size)),
        max_active=max_active,
        prefill_chunk_size=prefill_chunk,
        row_shard_prefill=True,
    )
    with pytest.raises(StopAfterDecoders):
        qwen_bootstrap.prepare_model(
            snapshot=SimpleNamespace(), config=cfg, rank=0, group=object()
        )

    assert recorded["roce_kwargs"] is not None
    expected_qsa_prefill = (prefill_chunk // world_size) * 1024 * 2
    expected_qsa_decode = min(128, cfg.request_capacity) * 2048 * 2
    expected_vocab = cfg.request_capacity * (248320 // world_size) * 2
    assert recorded["roce_kwargs"]["gather_bytes"] == max(expected_vocab, expected_qsa_prefill, expected_qsa_decode, 64)
    if max_active == 2 and prefill_chunk == 8192:
        # At small concurrency, QSA prefill gather (8 MiB TP2 / 4 MiB TP4) exceeds vocab gather!
        assert recorded["roce_kwargs"]["gather_bytes"] == expected_qsa_prefill
    assert recorded["namespace"]["row_shard_prefill"] is True

    calls = recorded["decoder_calls"]
    assert calls, "bootstrap must reach PreparedDecoder construction"
    eligible_prefill = [c for c in calls if c["mode"] == "prefill" and c["capacity"] >= 512 and c["capacity"] % (32 * world_size) == 0]
    ineligible_prefill = [c for c in calls if c["mode"] == "prefill" and (c["capacity"] < 512 or c["capacity"] % (32 * world_size) != 0)]
    eligible_decode = [c for c in calls if c["mode"] == "decode" and c["capacity"] <= 128]
    assert eligible_prefill, "expected at least one eligible prefill bucket (>=512 aligned)"
    assert ineligible_prefill, "expected at least one ineligible prefill bucket (<512)"
    assert eligible_decode, "expected at least one eligible decode bucket (<=128)"
    for call in eligible_prefill:
        assert call["qsa_gather_runtime"] is fake_roce
        assert call["qsa_shard_mode"] == "row"
        assert call["row_shard_prefill"] is True
    for call in ineligible_prefill:
        assert call["qsa_gather_runtime"] is None
        assert call["qsa_shard_mode"] is None
        assert call["row_shard_prefill"] is True
    for call in eligible_decode:
        assert call["qsa_gather_runtime"] is fake_roce
        assert call["qsa_shard_mode"] == "group"
        assert call["row_shard_prefill"] is False


def test_prepared_sharded_qsa_disables_overlap_on_roce_preserves_pcie(monkeypatch):
    """PreparedShardedQSA disables aux-stream overlap on RoCE (single-stream capture) while preserving PCIe."""
    import ignition.components.collectives as collectives_mod
    from ignition.preparation.scratch import ScratchArena

    roce_rt = _FakeRoceRuntime(world_size=2)
    pcie_rt = SimpleNamespace(world_size=2)
    monkeypatch.setattr(collectives_mod, "_roce", lambda rt: rt is roce_rt)

    class FakeArena(ScratchArena):
        def __init__(self):
            self.overlap = True

        @property
        def auxiliary_stream(self):
            return SimpleNamespace(cuda_stream=12345)

    class StopAfterOverlap(Exception):
        pass

    monkeypatch.setattr(qsa_sharded, "persistent_copy", lambda t: (_ for _ in ()).throw(StopAfterOverlap()))

    fake_attn = SimpleNamespace(
        tokens=8,
        q_heads=12,
        scratch=SimpleNamespace(query=torch.zeros((8, 12, 256), dtype=torch.bfloat16)),
        plan=SimpleNamespace(query=SimpleNamespace(max_speculative_tokens=7)),
        sparse=None,
        metadata=SimpleNamespace(main_block_table=torch.zeros((1, 4), dtype=torch.int32)),
        state=SimpleNamespace(main_k_cache=torch.zeros((1, 64, 1, 256), dtype=torch.bfloat16)),
    )
    arena = FakeArena()

    roce_obj = qsa_sharded.PreparedShardedQSA.__new__(qsa_sharded.PreparedShardedQSA)
    with pytest.raises(StopAfterOverlap):
        roce_obj.__init__(
            fake_attn, mode="group", rank=0, world_size=2,
            gather_runtime=roce_rt, coordinator=None, session=None, name="qsa", workspace=arena,
        )
    assert roce_obj.overlap is False
    assert roce_obj.stream is None
    assert roce_obj.stream_handle == 0

    pcie_obj = qsa_sharded.PreparedShardedQSA.__new__(qsa_sharded.PreparedShardedQSA)
    with pytest.raises(StopAfterOverlap):
        pcie_obj.__init__(
            fake_attn, mode="group", rank=0, world_size=2,
            gather_runtime=pcie_rt, coordinator=None, session=None, name="qsa", workspace=arena,
        )
    assert pcie_obj.overlap is True
    assert pcie_obj.stream_handle == 12345


def test_gdn_and_qsa_preencoded_input_contracts():
    """PreparedGDN and PreparedAttention forward preencoded MXFP8 bases to all input projections cleanly."""
    # 1) PreparedGDN functional runtime forwarding (fused_qkvz=True -> in_proj_qkvz + in_proj_ba).
    gdn = PreparedGDN.__new__(PreparedGDN)
    gdn._functional_requested = True
    gdn._functional_ready = True
    gdn.fused_qkvzba = False
    gdn.fused_qkvz = True
    gdn.parallel_projections = None
    gdn.world_size = 2
    gdn._has_partitions = False
    seen_gdn = {}

    def make_proj(name, out_cols):
        def _run(src, **kw):
            seen_gdn[name] = (src, kw.get("encoded_bases"))
            return torch.zeros((src.shape[0], out_cols), dtype=torch.bfloat16)
        return _run

    gdn.projections = {
        "in_proj_qkvz": make_proj("in_proj_qkvz", 8192),
        "in_proj_ba": make_proj("in_proj_ba", 48),
        "out_proj": lambda gated, **kw: torch.zeros((gated.shape[0], 2560), dtype=torch.bfloat16),
    }
    gdn._functional_sequence = lambda projected, z, a, b, **kw: (torch.zeros((projected.shape[0], 3072), dtype=torch.bfloat16), None)

    fake_src = torch.zeros((16, 2560), dtype=torch.float8_e4m3fn)
    fake_bases = (fake_src, torch.zeros((1, 16, 80), dtype=torch.uint8), torch.zeros((1, 1, 20, 32, 4, 4), dtype=torch.uint8))
    out = gdn(fake_src, metadata=object(), input_encoded_bases=fake_bases)
    assert out.shape == (16, 2560)
    assert seen_gdn["in_proj_qkvz"][1] is fake_bases
    assert seen_gdn["in_proj_ba"][1] is fake_bases

    # Reject preencoded bases when fused_qkvzba=True.
    gdn.fused_qkvzba = True
    with pytest.raises(ValueError, match="preencoded GDN inputs require unfused input projections"):
        gdn(fake_src, metadata=object(), input_encoded_bases=fake_bases)

    # 2) PreparedAttention pre_gather_runtime (row-split QSA) & serial __call__ with FP8 source + bases.
    seen_qsa = {}

    def make_qsa_proj(name, out_cols):
        def _run(src, **kw):
            seen_qsa[name] = (src, kw.get("encoded_bases"))
            return torch.zeros((src.shape[0], out_cols), dtype=torch.bfloat16)
        return _run

    attn = qwen_attention.PreparedAttention.__new__(qwen_attention.PreparedAttention)
    attn.cache_only = False
    attn.tokens = 16
    attn.q_heads = 12
    attn.kv_heads = 1
    attn.concurrent = None
    attn.sparse = None
    attn.constants = {"k_descale": torch.ones(1), "v_descale": torch.ones(1)}
    attn.state = SimpleNamespace(main_k_cache=None, main_v_cache=None)
    attn.writer = None
    attn.projections = {
        "qkv_proj": make_qsa_proj("qkv_proj", 12 * 512 + 2 * 256),
        "indexer.index_qk_proj": make_qsa_proj("indexer.index_qk_proj", 640),
        "o_proj": lambda gated: torch.zeros((gated.shape[0], 2560), dtype=torch.bfloat16),
    }
    attn._prepare_qkv_runtime = lambda proj, meta: (
        torch.zeros((16, 12, 256), dtype=torch.bfloat16),
        torch.zeros((16, 12, 256), dtype=torch.bfloat16),
        torch.zeros((16, 1, 256), dtype=torch.bfloat16),
        torch.zeros((16, 1, 256), dtype=torch.bfloat16),
    )

    orig_kv_run = qwen_attention.paged_kv_write.run
    qwen_attention.paged_kv_write.run = lambda *a, **kw: None
    try:
        meta = SimpleNamespace(
            slot_mapping=torch.zeros(16, dtype=torch.int64),
            main_block_table=torch.zeros((1, 4), dtype=torch.int32),
            compressed_block_table=torch.zeros((1, 4), dtype=torch.int32),
            raw_state_slot_ids=torch.zeros(1, dtype=torch.int64),
            request_ids=torch.zeros(16, dtype=torch.int64),
            query_positions=torch.arange(16, dtype=torch.int64),
            rope_positions=torch.zeros((16, 3), dtype=torch.int64),
            sequence_lengths=torch.tensor([16], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 16], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
            is_prefilling=torch.tensor([True]),
        )
        attn.workspace_bytes = 64
        captured_dtype = {}
        attn.sharded = SimpleNamespace(
            mode="row",
            overlap=False,
            select_row_runtime=lambda src, bindings, dynamic, wb: (
                captured_dtype.__setitem__("row", bindings["output"].dtype) or ([], torch.zeros((8, 1024), dtype=torch.bfloat16))
            ),
            finish_row_runtime=lambda *a, **kw: None,
        )
        out_row = attn(fake_src, meta, input_encoded_bases=fake_bases)
        assert out_row.shape == (16, 2560)
        assert captured_dtype["row"] is torch.bfloat16
        assert seen_qsa["qkv_proj"][1] is fake_bases
        assert seen_qsa["indexer.index_qk_proj"][1] is fake_bases

        # Also test non-sharded serial __call__ with FP8 source + bases.
        seen_qsa.clear()
        attn.sharded = None
        attn.functional_qsa = lambda src, bindings, dynamic, ws: captured_dtype.__setitem__("serial", bindings["output"].dtype)
        out_serial = attn(fake_src, meta, input_encoded_bases=fake_bases)
        assert out_serial.shape == (16, 2560)
        assert captured_dtype["serial"] is torch.bfloat16
        assert seen_qsa["qkv_proj"][1] is fake_bases
        assert seen_qsa["indexer.index_qk_proj"][1] is fake_bases
    finally:
        qwen_attention.paged_kv_write.run = orig_kv_run


def _build_roce_decoder_env(
    monkeypatch,
    *,
    tokens: int = 512,
    rows: int = 1,
    mode: str = "prefill",
    rank: int = 0,
    world_size: int = 2,
    hc_lowrank: int = 320,
    maintain_draft_state: bool = False,
    gdn_partitions=None,
    ple_partitions=None,
):
    import hyperflare_recipes.components.embedding as embedding_mod
    import hyperflare_recipes.components.gated_delta as gdn_mod
    import hyperflare_recipes.components.linear as linear_mod
    import hyperflare_recipes.components.vocabulary as vocab_mod
    import hyperflare_recipes.recipes.qwen38.hyperconnection as hc_mod
    import hyperflare_recipes.recipes.qwen38.moe as moe_mod
    import hyperflare_recipes.recipes.qwen38.ple as ple_mod
    import ignition.components.collectives as collectives_mod
    import ignition.components.sampling as sampling_mod

    records = {
        "hc_rows": [],
        "all_reduce": [],
        "reduce_scatter": [],
        "gather_shards": [],
        "chunked_gather": [],
        "combine_norm_rows": [],
        "final_mixer_rows": [],
    }

    class DummyOp:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            src = args[2] if len(args) >= 3 and isinstance(args[2], torch.Tensor) else None
            n = src.shape[0] if src is not None else tokens
            self.scratch = SimpleNamespace(output=torch.zeros((n, 2560), dtype=torch.bfloat16))
            self.output = torch.zeros((n, 2560), dtype=torch.bfloat16)
            self.plan = SimpleNamespace(name=kwargs.get("name", "op"))
            self.plans = {"combine": self.plan}
            self.down = SimpleNamespace(plan=self.plan)
            self.gate = SimpleNamespace(plan=self.plan)
            self.projections = {}
            self.writer = self.plan
            self.sharded = None
            self.router_projection = SimpleNamespace(plan=self.plan)
            self.routing = SimpleNamespace(plan=self.plan)
            self.experts = SimpleNamespace(plan=self.plan)
            self.shared = SimpleNamespace(
                up_projection=SimpleNamespace(plan=self.plan),
                activation_plan=self.plan,
                down_projection=SimpleNamespace(plan=self.plan),
                gate_projection=SimpleNamespace(plan=self.plan),
            )
            self.reduce = SimpleNamespace(plan=self.plan)

        def functional(self, *args, **kwargs):
            return self

        def prepare_combine_norm(self, *args, **kwargs):
            records["combine_norm_rows"].append((kwargs.get("name", ""), kwargs["priming_state"].shape[0]))
            self.plans["combine_norm"] = SimpleNamespace(name=kwargs.get("name", ""))

    class DummyHC(DummyOp):
        def __init__(self, module, hc_scratch, *, priming_state, **kwargs):
            super().__init__(module, hc_scratch, priming_state=priming_state, **kwargs)
            name = kwargs.get("name", "")
            if name.endswith(".final_mixer"):
                records["final_mixer_rows"].append(priming_state.shape[0])
            else:
                records["hc_rows"].append((name, priming_state.shape[0]))

    class DummyAllReduce(DummyOp):
        def __init__(self, source, output, **kwargs):
            super().__init__(source, output, **kwargs)
            records["all_reduce"].append((kwargs.get("name", ""), tuple(source.shape), tuple(output.shape)))

    class DummyReduceScatter(DummyOp):
        fits = staticmethod(collectives_mod.PreparedReduceScatter.fits)

        def __init__(self, source, output, **kwargs):
            super().__init__(source, output, **kwargs)
            records["reduce_scatter"].append((kwargs.get("name", ""), tuple(source.shape), tuple(output.shape)))

    class DummyGatherShards(DummyOp):
        fits = staticmethod(collectives_mod.PreparedGatherShards.fits)

        def __init__(self, source, output, **kwargs):
            super().__init__(source, output, **kwargs)
            records["gather_shards"].append((kwargs.get("name", ""), tuple(source.shape), tuple(output.shape)))

    class DummyChunkedGather(DummyOp):
        fits = staticmethod(collectives_mod.PreparedChunkedGatherShards.fits)

        def __init__(self, source, output, **kwargs):
            super().__init__(source, output, **kwargs)
            records["chunked_gather"].append((kwargs.get("name", ""), tuple(source.shape), tuple(output.shape)))

    class DummySampler:
        def __init__(self, *a, **kw):
            pass
        def prepare(self):
            pass
        def prepare_broadcast(self, *a, **kw):
            pass
        def functional(self, *a, **kw):
            return self

    monkeypatch.setattr(embedding_mod, "PreparedEmbedding", DummyOp)
    monkeypatch.setattr(hc_mod, "PreparedHyperconnection", DummyHC)
    monkeypatch.setattr(qwen_attention, "PreparedAttention", DummyOp)
    monkeypatch.setattr(gdn_mod, "PreparedGDN", DummyOp)
    monkeypatch.setattr(moe_mod, "PreparedQwenMoe", DummyOp)
    monkeypatch.setattr(ple_mod, "PreparedPLE", DummyOp)
    monkeypatch.setattr(collectives_mod, "PreparedAllReduce", DummyAllReduce)
    monkeypatch.setattr(collectives_mod, "ConcurrentAllReduce", DummyOp)
    monkeypatch.setattr(collectives_mod, "PreparedReduceScatter", DummyReduceScatter)
    monkeypatch.setattr(collectives_mod, "PreparedGatherShards", DummyGatherShards)
    monkeypatch.setattr(collectives_mod, "PreparedChunkedGatherShards", DummyChunkedGather)
    monkeypatch.setattr(vocab_mod, "PreparedVocabularyProjection", DummyOp)
    monkeypatch.setattr(linear_mod, "UnquantizedProjection", DummyOp)
    monkeypatch.setattr(sampling_mod, "PreparedCompletion", DummySampler)
    monkeypatch.setattr(sampling_mod, "PreparedSampling", DummySampler)
    monkeypatch.setattr(collectives_mod, "_roce", lambda rt: isinstance(rt, _FakeRoceRuntime))
    monkeypatch.setattr(collectives_mod, "_dma", lambda rt: isinstance(rt, _FakeDmaRuntime))
    monkeypatch.setattr(qwen_decoder, "_is_dma_runtime", lambda rt: isinstance(rt, _FakeDmaRuntime))

    def make_layer():
        ple = SimpleNamespace(key_proj=SimpleNamespace(weight=None), value_proj=SimpleNamespace(weight=None))
        return SimpleNamespace(
            attn_hyper_connection=SimpleNamespace(
                input_mix_weight_down=SimpleNamespace(weight=torch.empty((hc_lowrank, 10240)))
            ),
            mlp_hyper_connection=object(),
            self_attn=object(),
            linear_attn=object(),
            ple=ple,
        )

    language = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=None),
        layers=[make_layer() for _ in range(48)],
        hyper_connection_mixer=object(),
    )
    weights = SimpleNamespace(
        tree=SimpleNamespace(model=SimpleNamespace(language_model=language), lm_head=SimpleNamespace(weight=None)),
        layers=[SimpleNamespace(attention={}, moe=None) for _ in range(48)],
        shard=TensorParallelShard(rank, world_size),
    )

    @dataclass(frozen=True)
    class DummyHCScratch:
        block_input: torch.Tensor

    class DummyScratch:
        def __init__(self, n):
            self.embedding = torch.zeros((n, 2560), dtype=torch.bfloat16)
            self.residual = torch.zeros((n, 10240), dtype=torch.bfloat16)
            self.hidden = torch.zeros((n, 2560), dtype=torch.bfloat16)
            self.sample_hidden = torch.zeros((rows, 2560), dtype=torch.bfloat16)
            self.logits = torch.zeros((rows, 16), dtype=torch.bfloat16)
            self.local_logits = torch.zeros((rows, 8), dtype=torch.bfloat16)
            self.hyperconnection = DummyHCScratch(block_input=torch.zeros((n, 2560), dtype=torch.bfloat16))
            self.gdn = object()
            self.qsa = object()
            self.ple = object()
            self.moe = object()

        def rows(self, n, seqs):
            return DummyScratch(n)

    scratch = DummyScratch(tokens)
    inputs = SimpleNamespace(
        row_slot_ids=torch.zeros((4,), dtype=torch.int32),
        row_generations=torch.zeros((4,), dtype=torch.int64),
        live_rows=torch.zeros((1,), dtype=torch.int32),
        sample_enabled=torch.zeros((4,), dtype=torch.int32),
    )
    state = SimpleNamespace(
        tokens=tokens,
        rows=rows,
        mode=mode,
        maintain_draft_state=maintain_draft_state,
        inputs=inputs,
        requests=object(),
        ple=SimpleNamespace(token_ids=None),
        gdn=SimpleNamespace(num_tokens=None),
        qsa=SimpleNamespace(main_block_table=torch.full((1, 2), -1, dtype=torch.int32)),
        gdn_partitions=gdn_partitions,
        ple_partitions=ple_partitions,
        pool=SimpleNamespace(
            layout=SimpleNamespace(speculative_tokens=7 if maintain_draft_state else 0),
            gdn={i: None for i in range(48) if i % 4 != 3},
            ple_convolution=None,
        ),
        prepare_hidden=lambda *a: None,
        prepare_rope_delta=lambda *a: None,
        prepare_residual=lambda *a: None,
        functional=lambda: None,
    )
    qsa_states = {i: None for i in range(3, 48, 4)}
    return weights, scratch, state, qsa_states, records


@pytest.mark.parametrize("world_size,tokens,sample,maintain_draft", [
    (2, 512, False, False),
    (2, 8192, True, False),
    (2, 1024, True, True),
    (4, 512, False, False),
    (4, 8192, True, False),
])
def test_qwen38_decoder_roce_row_shard_enabled_shapes_and_collectives(
    monkeypatch, world_size, tokens, sample, maintain_draft
):
    weights, scratch, state, qsa_states, rec = _build_roce_decoder_env(
        monkeypatch, tokens=tokens, rank=world_size - 1, world_size=world_size,
        maintain_draft_state=maintain_draft,
    )
    roce_rt = _FakeRoceRuntime(world_size=world_size, pipe_bytes=tokens * 2560 * 2)
    dec = qwen_decoder.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name=f"prefill{tokens}",
        runtime=roce_rt, sample=sample, row_shard_prefill=True,
    )
    st = tokens // world_size
    assert dec.row_shard_roce is True
    assert dec.shard_tokens == st
    assert torch.equal(dec.shard_rows, torch.arange((world_size - 1) * st, world_size * st, dtype=torch.int64))

    # Layer 0 uses 2 full-row AllReduces and full [T] HC; Layers 1..47 use [st] HC + 47*(2 Gather + 2 RS).
    assert len(rec["all_reduce"]) == 2
    assert rec["all_reduce"][0][1] == (tokens, 2560)
    assert len(rec["reduce_scatter"]) == 47 * 2
    assert all(src == (tokens, 2560) and dst == (st, 2560) for _, src, dst in rec["reduce_scatter"])
    # GatherShards: 47 attn_gather + 47 moe_gather + (1 hidden_gather if sample else 0).
    assert len(rec["gather_shards"]) == 47 * 2 + int(sample)
    # ChunkedGatherShards for full residual only when maintain_draft_state or residual_output is set.
    assert len(rec["chunked_gather"]) == int(maintain_draft)
    # Layer 0 attn_hc + mlp_hc are full [T]; layers 1..47 attn_hc + mlp_hc are [st].
    assert rec["hc_rows"][:2] == [(f"prefill{tokens}.layer0.attn_hc", tokens), (f"prefill{tokens}.layer0.mlp_hc", tokens)]
    assert all(r == st for _, r in rec["hc_rows"][2:])
    assert rec["final_mixer_rows"] == ([st] if sample else [])

    # DFlash2 capture_context registers 5 context_gathers (layers 5, 15, 25, 35, 45).
    meta = SimpleNamespace(positions=torch.zeros(tokens, dtype=torch.int64), starts=torch.zeros(1, dtype=torch.int64))
    draft_cfg = SimpleNamespace(target_layers=(5, 15, 25, 35, 45), attention=SimpleNamespace(window=2048))
    dec.capture_context(meta, draft_cfg, compact_prefill=True)
    assert set(dec.context_gathers) == {5, 15, 25, 35, 45}
    assert len(rec["gather_shards"]) == 47 * 2 + int(sample) + 5

    # Functional conversion succeeds and refuses late capture_context without mutating decoder state.
    dec.functional(session=None)
    orig_layers = dec.context_layers
    orig_capture = dec.context_capture
    orig_aux = dec.auxiliary
    late_draft_cfg = SimpleNamespace(target_layers=(1, 2), attention=SimpleNamespace(window=128))
    with pytest.raises(RuntimeError, match="capture_context must precede functional conversion"):
        dec.capture_context(meta, late_draft_cfg, compact_prefill=False)
    assert dec.context_layers == orig_layers
    assert dec.context_capture is orig_capture
    assert dec.auxiliary is orig_aux


def test_qwen38_decoder_roce_row_shard_eligibility_rejections(monkeypatch):
    """All ineligible configurations cleanly preserve full-row execution with zero regressions."""
    cases = [
        ("opt_in_default_false", dict(row_shard_prefill=False), _FakeRoceRuntime(2)),
        ("tp1_single_rank", dict(world_size=1, row_shard_prefill=True), None),
        ("pcie_dma_runtime", dict(row_shard_prefill=True), _FakeDmaRuntime(2)),
        ("roce_abi_below_11", dict(row_shard_prefill=True), _FakeRoceRuntime(2, abi=10)),
        ("decode_mode", dict(tokens=4, rows=4, mode="decode", row_shard_prefill=True), _FakeRoceRuntime(2)),
        ("scored_prefill", dict(scoring=SimpleNamespace(prepare=lambda *a, **kw: object()), row_shard_prefill=True), _FakeRoceRuntime(2)),
        ("verification", dict(verification=True, row_shard_prefill=True), _FakeRoceRuntime(2)),
        ("sharded_hc", dict(hc_lowrank=160, row_shard_prefill=True), _FakeRoceRuntime(2)),
        ("layer_outputs_trace", dict(layer_outputs=torch.zeros((48, 512, 10240), dtype=torch.bfloat16), row_shard_prefill=True), _FakeRoceRuntime(2)),
        ("below_min_tokens", dict(tokens=256, row_shard_prefill=True), _FakeRoceRuntime(2)),
        ("unaligned_shard_128", dict(tokens=640, row_shard_prefill=True), _FakeRoceRuntime(2)),
        ("pipe_bytes_too_small", dict(tokens=512, row_shard_prefill=True), _FakeRoceRuntime(2, pipe_bytes=1024)),
        ("gdn_partitions", dict(gdn_partitions=object(), row_shard_prefill=True), _FakeRoceRuntime(2)),
    ]
    for label, opts, rt in cases:
        env_kw = {k: opts[k] for k in ("tokens", "rows", "mode", "world_size", "hc_lowrank", "gdn_partitions") if k in opts}
        weights, scratch, state, qsa_states, rec = _build_roce_decoder_env(monkeypatch, **env_kw)
        dec_kw = {k: opts[k] for k in ("row_shard_prefill", "scoring", "verification", "layer_outputs") if k in opts}
        dec = qwen_decoder.PreparedDecoder(
            weights, scratch, state, qsa_states, None, None, None,
            None, None, None, session=None, name=f"case_{label}",
            runtime=rt, **dec_kw,
        )
        assert dec.row_shard_roce is False, f"{label} should not enable row_shard_roce"
        assert not rec["reduce_scatter"], f"{label} should not construct ReduceScatter"
        assert not rec["gather_shards"], f"{label} should not construct GatherShards"
        if rt is not None:
            assert len(rec["all_reduce"]) == 96, f"{label} should keep 96 standard AllReduces"

    # Post-init residual_output assignment without maintain_draft_state raises cleanly in functional().
    weights, scratch, state, qsa_states, _ = _build_roce_decoder_env(monkeypatch, tokens=512, maintain_draft_state=False)
    dec = qwen_decoder.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="late_res",
        runtime=_FakeRoceRuntime(2), row_shard_prefill=True,
    )
    assert dec.row_shard_roce is True and dec.need_full_residual is False
    dec.residual_output = torch.zeros((512, 10240), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires residual_output or maintain_draft_state"):
        dec.functional(session=None)


def test_mxfp8_pack_unpack_and_scale_mma_swizzle_sensitivity():
    """Verify Qwen3.8 MXFP8 packed gather unpacking and exact MMA tile swizzle mapping."""
    tokens, hidden = 512, 2560
    scale_cols_u8 = hidden // 32
    vals_bf16_cols = hidden // 2

    dec = qwen_decoder.PreparedDecoder.__new__(qwen_decoder.PreparedDecoder)
    dec.attn_vals_bf16_cols = vals_bf16_cols

    vals = ((torch.arange(tokens * hidden, dtype=torch.int64) % 113) + 1).to(torch.uint8).view(torch.float8_e4m3fn).reshape(tokens, hidden)
    srows = ((torch.arange(tokens * scale_cols_u8, dtype=torch.int64) % 241) + 5).to(torch.uint8).reshape(tokens, scale_cols_u8)
    packed = torch.cat((vals.view(torch.bfloat16), srows.view(torch.bfloat16)), dim=1)

    unpacked_vals, (b_vals, b_srows, b_smma) = dec._unpack_attn_gathered(packed)
    assert torch.equal(unpacked_vals.view(torch.uint8), vals.view(torch.uint8))
    assert torch.equal(b_vals.view(torch.uint8), vals.view(torch.uint8))
    assert torch.equal(b_srows, srows.view(1, tokens, scale_cols_u8))
    assert b_smma.shape == (1, tokens // 128, hidden // 128, 32, 4, 4)

    for m_tile in range(tokens // 128):
        for k_tile in range(hidden // 128):
            for r_outer in range(4):
                for c_inner in range(4):
                    rows_idx = m_tile * 128 + r_outer * 32 + torch.arange(32)
                    col_idx = k_tile * 4 + c_inner
                    assert torch.equal(
                        b_smma[0, m_tile, k_tile, :, r_outer, c_inner],
                        srows[rows_idx, col_idx],
                    )

    # Single-byte sensitivity check: flipping one scale byte mutates only its exact MMA tile slot.
    mutated = packed.clone()
    mutated_srows = mutated[:, vals_bf16_cols:].contiguous().view(torch.uint8)
    # row 165 -> m_tile=1, r_outer=1 (32..63), r_inner=5; col 9 -> k_tile=2, c_inner=1
    mutated_srows[165, 9] ^= 0x01
    mutated[:, vals_bf16_cols:] = mutated_srows.view(torch.bfloat16)
    _, (_, _, mutated_smma) = dec._unpack_attn_gathered(mutated)
    diff = mutated_smma != b_smma
    assert int(diff.sum().item()) == 1
    assert bool(diff[0, 1, 2, 5, 1, 1].item())


class _SimulatedRoceFabric:
    """Thread-synchronized multi-rank RoCE collective simulator enforcing zero graph-ordering deps."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.slots: list[torch.Tensor | None] = [None] * world_size
        self.calls: list[list[tuple[str, tuple[int, ...], tuple]]] = [[] for _ in range(world_size)]

    def _exchange(self, rank: int, kind: str, tensor: torch.Tensor, dependencies: tuple):
        if dependencies:
            raise AssertionError(f"RoCE {kind} must never pass graph-ordering dependencies, got {dependencies}")
        self.calls[rank].append((kind, tuple(tensor.shape), dependencies))
        self.slots[rank] = tensor.detach().clone()
        self.barrier.wait(timeout=30)
        gathered = [self.slots[r].clone() for r in range(self.world_size)]
        self.barrier.wait(timeout=30)
        return gathered

    def all_reduce(self, rank: int, tensor: torch.Tensor, *, dependencies=()):
        parts = self._exchange(rank, "all_reduce", tensor, dependencies)
        return sum(p.float() for p in parts).bfloat16()

    def reduce_scatter(self, rank: int, tensor: torch.Tensor, *, dependencies=()):
        parts = self._exchange(rank, "reduce_scatter", tensor, dependencies)
        reduced = sum(p.float() for p in parts).bfloat16()
        st = reduced.shape[0] // self.world_size
        return reduced[rank * st:(rank + 1) * st].contiguous()

    def gather_shards(self, rank: int, tensor: torch.Tensor, *, dependencies=()):
        parts = self._exchange(rank, "gather_shards", tensor, dependencies)
        return torch.cat(parts, dim=0).contiguous()

    def chunked_gather_shards(self, rank: int, tensor: torch.Tensor, *, dependencies=()):
        parts = self._exchange(rank, "chunked_gather_shards", tensor, dependencies)
        return torch.cat(parts, dim=0).contiguous()


@pytest.mark.parametrize("world_size,sample,mxfp8_attn,need_full_residual,live_tokens", [
    (2, False, True, False, 512),   # TP2 DFlash2 prefill (non-terminal), MXFP8 attn gather, full chunk
    (2, True, True, False, 317),    # TP2 DFlash2 terminal_prefill, MXFP8 attn gather, partial tail spanning ranks!
    (2, True, False, True, 317),    # TP2 classic MTP terminal_prefill, BF16 attn gather, full residual gather
    (4, True, True, False, 317),    # TP4 DFlash2 terminal_prefill, MXFP8 attn gather, partial tail spanning 3 ranks!
])
def test_roce_row_shard_multi_rank_end_to_end_parity_and_zero_dependencies(
    world_size, sample, mxfp8_attn, need_full_residual, live_tokens
):
    """Multi-rank simulated _run_row_shard_roce matches full-row baseline bit-for-bit across all outputs."""
    tokens = 512
    st = tokens // world_size
    hidden = 2560
    num_layers = 8  # Layers 0..7 (includes layer 0 full-row, layer 1 causal PLE, GDN, QSA at 3 & 7, taps at 2 & 5)
    target_layers = (2, 5)
    window = 64

    gen = torch.Generator(device="cpu").manual_seed(42 + world_size)
    emb_table = (torch.randn((tokens, hidden), generator=gen) * 0.25).bfloat16()

    # Deterministic synthetic per-layer weights (shared across full-row reference and TP ranks).
    hc_mix_w = [(torch.randn((4,), generator=gen) * 0.1 + 0.25).bfloat16() for _ in range(num_layers * 2 + 1)]
    hc_inj_w = [(torch.randn((4,), generator=gen) * 0.05).bfloat16() for _ in range(num_layers * 2)]
    ple_w = (torch.randn((4,), generator=gen) * 0.1).bfloat16()  # causal 4-tap dilation-3 conv weights
    attn_proj = [(torch.randn((hidden, hidden), generator=gen) * (0.02 / world_size)).bfloat16() for _ in range(num_layers)]
    moe_proj = [(torch.randn((hidden, hidden), generator=gen) * (0.02 / world_size)).bfloat16() for _ in range(num_layers)]
    # Rank-specific partial projections summing to the full projection.
    attn_rank_scale = [torch.linspace(0.5 + 0.2 * r, 1.5 - 0.1 * r, hidden).bfloat16() for r in range(world_size)]
    attn_scale_sum = sum(s.float() for s in attn_rank_scale).bfloat16()
    lm_w = (torch.randn((hidden, 32), generator=gen) * 0.05).bfloat16()

    def make_hc(mix_w, inj_w=None):
        def mix_runtime(res, norm=None):
            r4 = res.view(-1, 4, hidden)
            mixed = (r4.float() * mix_w.float()[None, :, None]).sum(dim=1).bfloat16()
            inj = None if inj_w is None else inj_w[None, :].expand(res.shape[0], 4).contiguous()
            return mixed, inj

        def combine_runtime(res, block, inj):
            r4 = res.view(-1, 4, hidden).float()
            out = (r4 + block.float()[:, None, :] * inj.float()[:, :, None]).bfloat16().view(-1, 10240)
            return out

        def combine_norm_runtime(res, block, inj):
            out = combine_runtime(res, block, inj)
            return out, out

        return SimpleNamespace(
            mix_runtime=mix_runtime,
            combine_runtime=combine_runtime,
            combine_norm_runtime=combine_norm_runtime,
        )

    def run_ple(res, ple_meta, partitions=None):
        # Causal dilated convolution (kernel=4, dilation=3) across the FULL token axis!
        out = torch.zeros_like(res)
        for tap in range(4):
            shift = tap * 3
            if shift == 0:
                out.add_((res.float() * ple_w[tap].float()).bfloat16())
            else:
                out[shift:].add_((res[:-shift].float() * ple_w[tap].float()).bfloat16())
        return out

    def fake_mxfp8_quant(x_bf16: torch.Tensor):
        # Deterministic per-32 block quantization on CPU (matches min_amax=0.0 blockwise independence).
        rows_n, cols = x_bf16.shape
        x_blk = x_bf16.float().view(rows_n, cols // 32, 32)
        amax = x_blk.abs().amax(dim=-1)
        exp = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448.0)), torch.zeros_like(amax)).clamp(-127, 127)
        scale = torch.exp2(exp)
        srows = (exp + 127).to(torch.uint8)
        vals = (x_blk / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(rows_n, cols)
        return vals, srows

    def dequant_mxfp8_bases(encoded_bases):
        vals, srows, smma = encoded_bases
        rows_n, cols = vals.shape
        # Verify smma matches srows before computing.
        assert torch.equal(smma, qwen_decoder._mxfp8_scale_mma(srows.view(rows_n, cols // 32), cols))
        scale = torch.exp2(srows.view(rows_n, cols // 32).float() - 127.0)
        return (vals.float().view(rows_n, cols // 32, 32) * scale.unsqueeze(-1)).view(rows_n, cols).bfloat16()

    def make_attention(layer_idx: int, rank: int | None):
        is_qsa = layer_idx % 4 == 3
        w = attn_proj[layer_idx]
        r_scale = attn_scale_sum if rank is None else attn_rank_scale[rank]

        def _compute(src, encoded_bases):
            if mxfp8_attn and layer_idx >= 1:
                if encoded_bases is not None:
                    x = dequant_mxfp8_bases(encoded_bases)
                else:
                    v, sr = fake_mxfp8_quant(src)
                    sm = qwen_decoder._mxfp8_scale_mma(sr, hidden)
                    x = dequant_mxfp8_bases((v, sr.view(1, src.shape[0], hidden // 32), sm))
            else:
                x = src
            # Causal mix across tokens (e.g. GDN recurrence / QSA attention) so cross-token deps are exercised!
            y = (x.float() @ w.float()) * r_scale.float()[None, :]
            y[1:] = y[1:] + 0.1 * y[:-1]
            return y.bfloat16()

        if is_qsa:
            return lambda src, qsa_meta, *, input_encoded_bases=None: _compute(src, input_encoded_bases)
        return lambda src, *, records=None, metadata=None, partitions=None, input_encoded_bases=None: _compute(src, input_encoded_bases)

    def make_moe(layer_idx: int, rank: int | None):
        w = moe_proj[layer_idx]
        r_scale = attn_scale_sum if rank is None else attn_rank_scale[rank]
        return lambda src: ((src.float() @ w.float()) * r_scale.float()[None, :]).bfloat16()

    # Build capture metadata with partial tail `live_tokens`.
    positions = torch.where(torch.arange(tokens) < live_tokens, torch.arange(tokens), -1)
    cap_meta = SimpleNamespace(
        positions=positions,
        starts=torch.tensor([0], dtype=torch.int64),
        lengths=torch.tensor([live_tokens], dtype=torch.int32),
        query_start_loc=torch.tensor([0, live_tokens], dtype=torch.int32),
    )
    runtime = SimpleNamespace(
        ple=SimpleNamespace(token_ids=torch.arange(tokens, dtype=torch.int64)),
        gdn=SimpleNamespace(num_tokens=torch.tensor([live_tokens], dtype=torch.int32)),
        qsa=SimpleNamespace(),
        capture_metadata=cap_meta,
    )

    def build_decoder(rank: int | None, fabric: _SimulatedRoceFabric | None):
        dec = qwen_decoder.PreparedDecoder.__new__(qwen_decoder.PreparedDecoder)
        is_roce = rank is not None
        dec.sample = sample
        dec.pipelined_prefill = False
        dec.row_shard_roce = is_roce
        dec.need_full_residual = need_full_residual
        dec.shard_tokens = st if is_roce else 0
        dec.shard_rows = torch.arange(rank * st, (rank + 1) * st, dtype=torch.int64) if is_roce else None
        dec.record_layout = None
        dec.embedding_injection = None
        dec.layer_outputs = None
        dec.scoring = None
        dec.sampler = None
        dec.completion_recorder = None
        dec.reductions = []
        dec.pipelined_reductions = []
        dec.ple = SimpleNamespace(
            embedding_lookup=SimpleNamespace(publish=lambda ple: None),
            __call__=run_ple,
        )
        # Make ple callable directly.
        class _PLE:
            embedding_lookup = SimpleNamespace(publish=lambda ple: None)
            def __call__(self, res, ple_meta, partitions=None):
                return run_ple(res, ple_meta, partitions)
        dec.ple = _PLE()
        dec.embedding = lambda *, token_ids, num_rows: emb_table[token_ids]
        dec.layers = [
            (
                make_hc(hc_mix_w[2 * i], hc_inj_w[2 * i]),
                make_attention(i, rank),
                make_hc(hc_mix_w[2 * i + 1], hc_inj_w[2 * i + 1]),
                make_moe(i, rank),
            )
            for i in range(num_layers)
        ]
        dec.final_mixer = make_hc(hc_mix_w[-1], None) if sample else None
        dec.lm_head = (lambda x: (x.float() @ lm_w.float()).bfloat16()) if sample else None
        dec.context_layers = target_layers
        dec.context_capture = PreparedContextCapture(cap_meta, hidden * len(target_layers), window, compact_prefill=True)

        def finish_runtime(rt, hid, res):
            if hid is None:
                return None
            return hid[live_tokens - 1:live_tokens]

        dec.state = SimpleNamespace(finish_runtime=finish_runtime)

        if is_roce:
            dec.attn_vals_bf16_cols = hidden // 2
            dec.attn_local_shapes = ((st, hidden), (1, st, hidden // 32), (1, st // 128, hidden // 128, 32, 4, 4))

            def _prep_shard(norm_sh, qplan):
                vals, srows = fake_mxfp8_quant(norm_sh)
                return torch.cat((vals.view(torch.bfloat16), srows.view(torch.bfloat16)), dim=1)

            dec._prepare_attn_shard = _prep_shard
            dec.layer0_reductions = (
                lambda t, *, dependencies=(): fabric.all_reduce(rank, t, dependencies=dependencies),
                lambda t, *, dependencies=(): fabric.all_reduce(rank, t, dependencies=dependencies),
            )
            dec.roce_layer_ops = [None]
            dec.roce_has_mxfp8_gather = [False]
            for i in range(1, num_layers):
                dec.roce_layer_ops.append((
                    object() if mxfp8_attn else None,
                    lambda t, *, dependencies=(): fabric.gather_shards(rank, t, dependencies=dependencies),
                    lambda t, *, dependencies=(): fabric.reduce_scatter(rank, t, dependencies=dependencies),
                    lambda t, *, dependencies=(): fabric.gather_shards(rank, t, dependencies=dependencies),
                    lambda t, *, dependencies=(): fabric.reduce_scatter(rank, t, dependencies=dependencies),
                ))
                dec.roce_has_mxfp8_gather.append(mxfp8_attn)
            dec.hidden_gather = (
                (lambda t, *, dependencies=(): fabric.gather_shards(rank, t, dependencies=dependencies))
                if sample else None
            )
            dec.residual_gather = (
                (lambda t, *, dependencies=(): fabric.chunked_gather_shards(rank, t, dependencies=dependencies))
                if need_full_residual else None
            )
            dec.context_gathers = {
                layer: (lambda t, *, dependencies=(): fabric.gather_shards(rank, t, dependencies=dependencies))
                for layer in target_layers if layer >= 1
            }
        return dec

    # Run full-row reference (single rank representing exact unsharded mathematical baseline).
    ref_dec = build_decoder(rank=None, fabric=None)
    ref_sample, ref_logits, ref_aux, _ = ref_dec(runtime)

    # Run all `world_size` RoCE row-sharded ranks concurrently over the simulated fabric.
    fabric = _SimulatedRoceFabric(world_size)
    results: list[tuple | None] = [None] * world_size
    errors: list[tuple[int, BaseException]] = []
    errors_lock = threading.Lock()

    def worker(r: int):
        try:
            dec = build_decoder(rank=r, fabric=fabric)
            results[r] = dec(runtime)
        except BaseException as exc:  # noqa: BLE001
            with errors_lock:
                errors.append((r, exc))
            fabric.barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    alive = [i for i, t in enumerate(threads) if t.is_alive()]
    assert not alive, f"worker threads hung: {alive}; errors={errors}"
    assert not errors, errors

    # Verify numerical parity across EVERY rank AND strict collective call counts / zero dependencies!
    for r in range(world_size):
        r_sample, r_logits, r_aux, _ = results[r]
        if sample:
            assert torch.allclose(r_sample.float(), ref_sample.float(), atol=5e-2, rtol=1e-2)
            assert torch.allclose(r_logits.float(), ref_logits.float(), atol=5e-2, rtol=1e-2)
        else:
            assert r_sample is None and r_logits is None
        assert torch.allclose(r_aux.draft_features.float(), ref_aux.draft_features.float(), atol=5e-2, rtol=1e-2)
        if need_full_residual:
            assert r_aux.residual.shape == (tokens, 10240)
            assert torch.allclose(r_aux.residual.float(), ref_aux.residual.float(), atol=5e-2, rtol=1e-2)
        else:
            assert r_aux.residual.shape == (st, 10240)
            expected_shard = ref_aux.residual[r * st:(r + 1) * st]
            assert torch.allclose(r_aux.residual.float(), expected_shard.float(), atol=5e-2, rtol=1e-2)

        # Verify exact RoCE collective call sequence and zero dependencies on every single call.
        kinds = [kind for kind, _, deps in fabric.calls[r] if deps == ()]
        assert len(kinds) == len(fabric.calls[r])
        assert kinds.count("all_reduce") == 2  # Layer 0 only!
        assert kinds.count("reduce_scatter") == (num_layers - 1) * 2
        assert kinds.count("gather_shards") == (num_layers - 1) * 2 + len(target_layers) + int(sample)
        assert kinds.count("chunked_gather_shards") == int(need_full_residual)
        # Verify MXFP8 attention gather carried packed width 1320 (vs 2560 BF16).
        attn_gather_shapes = [
            shape for kind, shape, _ in fabric.calls[r]
            if kind == "gather_shards" and shape[1] == hidden // 2 + hidden // 64
        ]
        assert len(attn_gather_shapes) == (num_layers - 1 if mxfp8_attn else 0)


def test_qwen38_decoder_roce_mxfp8_attn_gather_init_and_fallback(monkeypatch):
    """Verify PreparedDecoder.__init__ declares packed MXFP8 attn gathers (min_amax=0.0) across all 47 layers."""
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from b12x.quantization import mxfp8
    import hyperflare_recipes.components.gated_delta as gdn_mod
    import hyperflare_recipes.components.sm103_projection as sm103_mod
    import ignition.preparation.tuning as tuning_mod

    weights, scratch, state, qsa_states, rec = _build_roce_decoder_env(
        monkeypatch, tokens=512, world_size=2,
    )
    # Provide real MXFP8LinearWeight stubs on layers 0..47, except layer 2 (BF16 fallback check).
    mxfp8_w = MXFP8LinearWeight(weight=None, in_features=2560, padded_in_features=2560, out_features=640)
    for idx in range(48):
        if idx % 4 == 3:
            weights.layers[idx].attention = {
                "qkv_proj": mxfp8_w,
                "indexer.index_qk_proj": mxfp8_w,
                "o_proj": mxfp8_w,
            }
        elif idx == 2:
            # Non-MXFP8 layer -> must fall back cleanly to BF16 gather while other layers pack MXFP8!
            weights.layers[idx].attention = {
                "in_proj_qkvz": torch.empty((1, 1)),
                "in_proj_ba": torch.empty((1, 1)),
                "out_proj": mxfp8_w,
            }
        else:
            weights.layers[idx].attention = {
                "in_proj_qkvz": mxfp8_w,
                "in_proj_ba": mxfp8_w,
                "out_proj": mxfp8_w,
            }

    seen_input_encoded = []
    seen_min_amax = []
    orig_gdn = gdn_mod.PreparedGDN
    orig_attn = qwen_attention.PreparedAttention

    class TrackingGDN(orig_gdn):
        def __init__(self, *args, **kwargs):
            seen_input_encoded.append((kwargs.get("name", ""), kwargs.get("input_encoded") is not None))
            super().__init__(*args, **kwargs)

    class TrackingAttn(orig_attn):
        def __init__(self, *args, **kwargs):
            seen_input_encoded.append((kwargs.get("name", ""), kwargs.get("input_encoded") is not None))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(gdn_mod, "PreparedGDN", TrackingGDN)
    monkeypatch.setattr(qwen_attention, "PreparedAttention", TrackingAttn)
    monkeypatch.setattr(sm103_mod, "is_sm103", lambda dev: False)

    # Simulate dev.type == "cuda" on CPU tensors for the __init__ guard check.
    fake_cuda_dev = SimpleNamespace(type="cuda", index=0)

    class CudaTagTensor(torch.Tensor):
        @property
        def device(self):
            return fake_cuda_dev

    orig_rows = scratch.rows

    def fake_rows(n, seqs):
        sub = orig_rows(n, seqs)
        sub.hyperconnection = type(sub.hyperconnection)(
            block_input=sub.hyperconnection.block_input.as_subclass(CudaTagTensor)
        )
        return sub

    scratch.rows = fake_rows
    orig_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        if kwargs.get("device") is fake_cuda_dev:
            kwargs["device"] = "cpu"
        return orig_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(mxfp8, "dense_rows_from_bases", lambda v, sr, sm: SimpleNamespace(values=v, scale_rows=sr, scale_mma=sm))

    def fake_query_from_call(src, v, sr, sm, *, min_amax=0.0):
        seen_min_amax.append(min_amax)
        return SimpleNamespace(min_amax=min_amax)

    monkeypatch.setattr(mxfp8, "query_from_call", fake_query_from_call)
    monkeypatch.setattr(mxfp8, "plan", lambda q: SimpleNamespace(request=lambda **kw: kw))
    monkeypatch.setattr(tuning_mod, "declare", lambda session, reqs: None)

    roce_rt = _FakeRoceRuntime(world_size=2, pipe_bytes=512 * 2560 * 2)
    dec = qwen_decoder.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="prefill512",
        runtime=roce_rt, sample=False, row_shard_prefill=True,
    )
    assert dec.row_shard_roce is True
    expected_flags = [idx >= 1 and idx != 2 for idx in range(48)]
    assert dec.roce_has_mxfp8_gather == expected_flags
    assert seen_min_amax == [0.0] * 46
    assert [has_enc for _, has_enc in seen_input_encoded] == expected_flags

    # Verify attn_gather shapes: 46 packed MXFP8 (256, 1320) -> (512, 1320), 1 BF16 fallback (256, 2560) -> (512, 2560).
    attn_gathers = [(name, src, dst) for name, src, dst in rec["gather_shards"] if name.endswith(".attn_gather")]
    assert len(attn_gathers) == 47
    packed = [g for g in attn_gathers if g[1] == (256, 1320) and g[2] == (512, 1320)]
    bf16 = [g for g in attn_gathers if g[1] == (256, 2560) and g[2] == (512, 2560)]
    assert len(packed) == 46
    assert len(bf16) == 1 and bf16[0][0] == "prefill512.layer2.attn_gather"


def test_qwen38_decoder_default_path_does_not_import_row_shard_collectives(monkeypatch):
    """Default PreparedDecoder construction (row_shard_prefill=False) works when row-shard symbols are absent."""
    import ignition.components.collectives as collectives_mod

    weights, scratch, state, qsa_states, rec = _build_roce_decoder_env(monkeypatch, tokens=512, world_size=2)
    for attr in (
        "PreparedChunkedGatherShards",
        "PreparedGatherShards",
        "PreparedReduceScatter",
        "row_shard_surfaces",
    ):
        monkeypatch.delattr(collectives_mod, attr, raising=False)

    dec = qwen_decoder.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="prefill512_default",
        runtime=_FakeRoceRuntime(2), row_shard_prefill=False,
    )
    assert dec.row_shard_roce is False
    assert len(rec["all_reduce"]) == 96
    assert not rec["reduce_scatter"]
    assert not rec["gather_shards"]


def test_prepared_attention_rejects_preencoded_input_with_overlapping_group_sharded_qsa(monkeypatch):
    """PreparedAttention rejects input_encoded at construction when PCIe group-mode QSA would overlap streams."""
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from b12x.quantization import mxfp8
    import ignition.components.collectives as collectives_mod
    from ignition.preparation.scratch import ScratchArena

    pcie_rt = SimpleNamespace(world_size=2)
    monkeypatch.setattr(collectives_mod, "_roce", lambda rt: False)
    monkeypatch.setattr(qwen_attention.sm103, "is_sm103", lambda dev: False)
    monkeypatch.setattr(qwen_attention.qsa, "Caps", lambda **kw: SimpleNamespace(**kw))

    projection_calls = []

    def fail_if_called(*args, **kwargs):
        projection_calls.append((args, kwargs))
        raise AssertionError("projection must not be constructed when overlap + input_encoded is rejected")

    monkeypatch.setattr(qwen_attention, "QuantizedProjection", fail_if_called)
    monkeypatch.setattr(qwen_attention, "UnquantizedProjection", fail_if_called)

    class FakeArena(ScratchArena):
        def __init__(self):
            self.overlap = True

    class FakeMXFP8Rows:
        pass

    monkeypatch.setattr(mxfp8, "MXFP8Rows", FakeMXFP8Rows)

    mxfp8_w = MXFP8LinearWeight(weight=None, in_features=2560, padded_in_features=2560, out_features=640)
    module = SimpleNamespace(
        q_norm=torch.nn.RMSNorm(256, eps=1e-6),
        k_norm=torch.nn.RMSNorm(256, eps=1e-6),
        config=SimpleNamespace(rms_norm_eps=1e-6),
        indexer=SimpleNamespace(
            q_layernorm=SimpleNamespace(weight=torch.ones(128)),
            k_layernorm=SimpleNamespace(weight=torch.ones(128)),
        ),
    )
    source = torch.zeros((8, 2560), dtype=torch.bfloat16)
    scratch = SimpleNamespace(
        query=torch.zeros((8, 12, 256), dtype=torch.bfloat16),
        key=torch.zeros((8, 1, 256), dtype=torch.bfloat16),
        output=torch.zeros((8, 2560), dtype=torch.bfloat16),
        projected_qkv=torch.zeros((8, 6656), dtype=torch.bfloat16),
        projected_index=torch.zeros((8, 640), dtype=torch.bfloat16),
        projected_v=torch.zeros((8, 256), dtype=torch.bfloat16),
        gated=torch.zeros((8, 3072), dtype=torch.bfloat16),
        attended=torch.zeros((8, 12, 256), dtype=torch.bfloat16),
        selected_positions=torch.full((8, 2051), -1, dtype=torch.int32),
    )
    state = SimpleNamespace(
        main_k_cache=torch.zeros((2, 64, 1, 256), dtype=torch.float8_e4m3fn),
        main_v_cache=torch.zeros((2, 64, 1, 256), dtype=torch.float8_e4m3fn),
        compressed_k_cache=torch.zeros((2, 16, 128), dtype=torch.bfloat16),
        raw_k_ring=torch.zeros((2, 8, 128), dtype=torch.bfloat16),
        raw_logical_positions=torch.full((2, 8), -1, dtype=torch.int64),
        raw_rope_positions=torch.full((2, 8, 3), -1, dtype=torch.int64),
        raw_interval_start_positions=torch.full((2,), -1, dtype=torch.int64),
    )
    metadata = SimpleNamespace(
        main_block_table=torch.full((1, 4), -1, dtype=torch.int32),
        compressed_block_table=torch.full((1, 4), -1, dtype=torch.int32),
        raw_state_slot_ids=torch.full((1,), -1, dtype=torch.int64),
        request_ids=torch.full((8,), -1, dtype=torch.int64),
        query_positions=torch.full((8,), -1, dtype=torch.int64),
        rope_positions=torch.full((8, 3), -1, dtype=torch.int64),
        sequence_lengths=torch.zeros((1,), dtype=torch.int32),
        query_start_loc=torch.zeros((2,), dtype=torch.int32),
        num_accepted_tokens=torch.zeros((1,), dtype=torch.int32),
        is_prefilling=torch.zeros((1,), dtype=torch.bool),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64),
    )
    rotary = SimpleNamespace(cos=torch.ones((256, 32)), sin=torch.zeros((256, 32)))
    weights = {"qkv_proj": mxfp8_w, "indexer.index_qk_proj": mxfp8_w, "o_proj": mxfp8_w}

    with pytest.raises(ValueError, match="preencoded QSA inputs require serial non-overlapping projections"):
        qwen_attention.PreparedAttention(
            module, weights, source, scratch, state, metadata, rotary,
            session=None, name="qsa_overlap_reject",
            workspace=FakeArena(), projection_workspace=FakeArena(),
            shard=TensorParallelShard(0, 2),
            qsa_gather_runtime=pcie_rt, qsa_shard_mode="group",
            input_encoded=FakeMXFP8Rows(),
        )
    assert projection_calls == []


