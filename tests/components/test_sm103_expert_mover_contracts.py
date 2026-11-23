# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contract checks for the SM103 GPU expert mover, predictor state, and dynamic tier."""

from dataclasses import dataclass
from pathlib import Path
import struct
import tomllib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from hyperflare_recipes.components.megamoe import (
    CONTRACT as MEGAMOE_CONTRACT,
    MegaMoeGeometry,
    MegaMoeQuery,
)
from hyperflare_recipes.components.megamoe_staging import (
    CONTRACT as STAGE_CONTRACT,
    StageQuery,
)
from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
from hyperflare_recipes.components.sm103_expert_mover import (
    LAYER_DESC_BYTES,
    LAYER_DESC_PREDICTOR_OFFSET,
    PREDICTOR_STATE_BYTES,
    QUEUE_HEADER_BYTES,
    QUEUE_REQUEST_BYTES,
    MoverMetrics,
    PredictorConfig,
    SM103ExpertMoverRuntime,
    build_persistent_worker,
    decode_mover_metrics,
    pack_layer_mover_desc,
    pack_predictor_layer_state,
    query_green_context_compute_sms,
)


def test_predictor_config_defaults_and_validation():
    default = PredictorConfig()
    assert default == PredictorConfig(
        decay_shift=6,
        promote_threshold=256,
        min_age_steps=64,
        interval_steps=8,
        cooldown_steps=16,
        max_enqueues=1,
        max_predict_rows=128,
    )
    assert PredictorConfig(
        decay_shift=0,
        promote_threshold=1,
        min_age_steps=0,
        interval_steps=1,
        cooldown_steps=1,
        max_enqueues=0,
        max_predict_rows=1,
    ).max_enqueues == 0
    assert PredictorConfig(decay_shift=30, min_age_steps=(1 << 31) - 1, max_enqueues=4).decay_shift == 30

    for kwargs in (
        {"decay_shift": -1},
        {"decay_shift": 31},
        {"decay_shift": True},
        {"promote_threshold": 0},
        {"promote_threshold": -8},
        {"min_age_steps": -1},
        {"min_age_steps": 1 << 31},
        {"interval_steps": 0},
        {"cooldown_steps": 0},
        {"max_enqueues": -1},
        {"max_enqueues": 5},
        {"max_predict_rows": 0},
        {"max_predict_rows": 128.0},
    ):
        with pytest.raises(ValueError):
            PredictorConfig(**kwargs)


def test_mover_metrics_decoding_and_hit_rate():
    empty = decode_mover_metrics(bytes(SM103ExpertMoverRuntime.METRICS_COUNT * 8))
    assert empty == MoverMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert empty.hit_rate == 0.0

    last_packed = (17 << 32) | 383
    raw = struct.pack("<8Q", 42, 800_000_000, 3, 9, last_packed, 750, 250, 14)
    decoded = decode_mover_metrics(raw)
    assert decoded == MoverMetrics(
        swaps_completed=42,
        bytes_copied=800_000_000,
        rcu_waits=3,
        skipped_already_resident=9,
        last_layer=17,
        last_expert=383,
        predict_hits=750,
        predict_misses=250,
        predict_enqueues=14,
    )
    assert decoded.hit_rate == pytest.approx(0.75)

    with pytest.raises(ValueError, match="metric bytes"):
        decode_mover_metrics(b"\x00" * 40)


def test_binary_descriptor_and_predictor_state_layouts():
    assert LAYER_DESC_BYTES == 192
    assert LAYER_DESC_PREDICTOR_OFFSET == 136
    assert QUEUE_HEADER_BYTES == 106944
    assert QUEUE_REQUEST_BYTES == 16
    assert PREDICTOR_STATE_BYTES == 128

    desc = pack_layer_mover_desc(
        host_plane_ptrs=(0x1000, 0x2000, 0x3000, 0x4000),
        hbm_plane_ptrs=(0x5000, 0x6000, 0x7000, 0x8000),
        plane_row_bytes=(12582912, 786432, 6291456, 393216),
        live_slot_map_ptr=0x9000,
        slot_owner_ptr=0xA000,
        layer_epoch_ptr=0xB000,
        num_experts=384,
        num_hbm_slots=290,
        num_host_experts=320,
        victim_cursor=5,
        predictor_state_ptr=0xC000,
    )
    assert len(desc) == LAYER_DESC_BYTES
    unpacked = struct.unpack_from("<4Q4Q4Q3Q4IQ", desc, 0)
    assert unpacked[:4] == (0x1000, 0x2000, 0x3000, 0x4000)
    assert unpacked[4:8] == (0x5000, 0x6000, 0x7000, 0x8000)
    assert unpacked[8:12] == (12582912, 786432, 6291456, 393216)
    assert unpacked[12:15] == (0x9000, 0xA000, 0xB000)
    assert unpacked[15:19] == (384, 290, 5, 320)
    assert unpacked[19] == 0xC000
    assert struct.unpack_from("<Q", desc, LAYER_DESC_PREDICTOR_OFFSET) == (0xC000,)
    assert desc[LAYER_DESC_PREDICTOR_OFFSET + 8:] == b"\x00" * (LAYER_DESC_BYTES - LAYER_DESC_PREDICTOR_OFFSET - 8)

    cfg = PredictorConfig(
        decay_shift=5,
        promote_threshold=512,
        min_age_steps=32,
        interval_steps=4,
        cooldown_steps=12,
        max_enqueues=2,
        max_predict_rows=64,
    )
    state = pack_predictor_layer_state(
        scores_ptr=0x1100,
        last_seen_ptr=0x2200,
        born_ptr=0x3300,
        cooldown_ptr=0x4400,
        init_clock=32,
        layer_idx=7,
        config=cfg,
        num_host_experts=320,
    )
    assert len(state) == PREDICTOR_STATE_BYTES
    assert struct.unpack_from("<4Q", state, 0) == (0x1100, 0x2200, 0x3300, 0x4400)
    # Offsets [32:40] and [52:64] are read and restored by reset_runtime_state().
    assert struct.unpack_from("<2I", state, 32) == (32, 0)
    assert struct.unpack_from("<3I", state, 52) == (32, 4, 12)
    assert struct.unpack_from("<11I", state, 32) == (32, 0, 7, 5, 512, 32, 4, 12, 2, 64, 320)
    assert state[76:] == b"\x00" * (PREDICTOR_STATE_BYTES - 76)


def test_persistent_worker_descriptor_contracts_and_proto_schema():
    @dataclass(frozen=True)
    class _CoreWorkerFiveFields:
        name: str
        graph: Any
        worker_sm_count: int
        stop_flag_buffer: str
        stop_flag_offset: int = 0

    @dataclass(frozen=True)
    class _CoreWorkerEightFields:
        name: str
        graph: Any
        worker_sm_count: int
        stop_flag_buffer: str
        stop_flag_offset: int = 0
        metrics_buffer: str = ""
        metrics_offset: int = 0
        metrics_count: int = 0

    sentinel_graph = object()
    w5 = build_persistent_worker(
        name="sm103_expert_mover",
        graph=sentinel_graph,
        worker_sm_count=8,
        control_buffer_name="sm103_mover_control",
        worker_cls=_CoreWorkerFiveFields,
    )
    assert w5 == _CoreWorkerFiveFields(
        name="sm103_expert_mover",
        graph=sentinel_graph,
        worker_sm_count=8,
        stop_flag_buffer="sm103_mover_control",
        stop_flag_offset=0,
    )

    w8 = build_persistent_worker(
        name="sm103_expert_mover",
        graph=sentinel_graph,
        worker_sm_count=8,
        control_buffer_name="sm103_mover_control",
        worker_cls=_CoreWorkerEightFields,
    )
    assert w8 == _CoreWorkerEightFields(
        name="sm103_expert_mover",
        graph=sentinel_graph,
        worker_sm_count=8,
        stop_flag_buffer="sm103_mover_control",
        stop_flag_offset=0,
        metrics_buffer="sm103_mover_control",
        metrics_offset=8,
        metrics_count=8,
    )
    # Control layout places stop_flag at [0..4) and metrics u64[8] at [8..72) inside CONTROL_BYTES=128.
    assert w8.stop_flag_offset % 4 == 0
    assert w8.metrics_offset % 8 == 0
    assert w8.stop_flag_offset + 4 <= w8.metrics_offset
    assert w8.metrics_offset + w8.metrics_count * 8 <= SM103ExpertMoverRuntime.CONTROL_BYTES

    from ignition.preparation import execution as core_execution

    core_worker_cls = getattr(core_execution, "PersistentWorker", None)
    if core_worker_cls is not None:
        core_worker = build_persistent_worker(
            name="sm103_expert_mover",
            graph=sentinel_graph,
            worker_sm_count=8,
            control_buffer_name="sm103_mover_control",
        )
        assert isinstance(core_worker, core_worker_cls)
        assert core_worker.name == "sm103_expert_mover"
        assert core_worker.worker_sm_count == 8
        assert core_worker.stop_flag_buffer == "sm103_mover_control"
        assert core_worker.stop_flag_offset == 0

    from ignition.preparation import runtime_core_pb2

    pw_fields = {f.name: f.number for f in runtime_core_pb2.PersistentWorker.DESCRIPTOR.fields}
    assert pw_fields["worker_sm_count"] == 1
    assert pw_fields["graph"] == 2
    assert pw_fields["stop_flag"] == 3
    if "metrics" in pw_fields:
        assert pw_fields["metrics"] == 4
        assert pw_fields["metrics_count"] == 5
        execution = runtime_core_pb2.Execution(
            persistent_workers=[
                runtime_core_pb2.PersistentWorker(
                    worker_sm_count=8,
                    graph="sm103_expert_mover",
                    stop_flag=runtime_core_pb2.Reference(allocation="sm103_mover_control", offset=0),
                    metrics=runtime_core_pb2.Reference(allocation="sm103_mover_control", offset=8),
                    metrics_count=SM103ExpertMoverRuntime.METRICS_COUNT,
                )
            ]
        )
        restored = runtime_core_pb2.Execution()
        restored.ParseFromString(execution.SerializeToString())
        assert len(restored.persistent_workers) == 1
        worker = restored.persistent_workers[0]
        assert worker.graph == "sm103_expert_mover"
        assert worker.stop_flag.offset == 0
        assert worker.metrics.allocation == "sm103_mover_control"
        assert worker.metrics.offset == 8
        assert worker.metrics_count == 8


def test_runtime_and_green_context_argument_validation():
    for bad_sms in (0, -1, True, 8.0):
        with pytest.raises(ValueError, match="positive integer"):
            query_green_context_compute_sms(0, bad_sms)

    storage = SimpleNamespace(device=0)
    with pytest.raises(ValueError, match="at least one dynamic layer"):
        SM103ExpertMoverRuntime([], storage=storage, kernel_cache=Path("/tmp/unused"))
    for bad_sms in (0, -4, True):
        with pytest.raises(ValueError, match="positive integer"):
            SM103ExpertMoverRuntime([(object(), object())], storage=storage, kernel_cache=Path("/tmp/unused"),
                                    worker_sm_count=bad_sms)
    for bad_cap in (0, 8, 24, 1000, True):
        with pytest.raises(ValueError, match="power of two >= 16"):
            SM103ExpertMoverRuntime([(object(), object())], storage=storage, kernel_cache=Path("/tmp/unused"),
                                    queue_capacity=bad_cap)
    for bad_name in ("", "   ", 123):
        with pytest.raises(ValueError, match="nonempty string"):
            SM103ExpertMoverRuntime([(object(), object())], storage=storage, kernel_cache=Path("/tmp/unused"),
                                    control_buffer_name=bad_name)


def test_dynamic_host_tier_and_query_validation():
    geom = MegaMoeGeometry(experts=8, capacity=64, top_k=2, hidden=128, intermediate=128, shared_experts=1)
    storage = SimpleNamespace(device=0)

    for kwargs in (
        {"hbm_slots": 1, "initial_hbm_experts": (0,)},
        {"hbm_slots": 9, "initial_hbm_experts": (0, 1)},
        {"hbm_slots": True, "initial_hbm_experts": (0,)},
        {"hbm_slots": 4, "initial_hbm_experts": (0, 1), "host_experts": 0},
        {"hbm_slots": 4, "initial_hbm_experts": (0, 1), "host_experts": 9},
        {"hbm_slots": 4, "initial_hbm_experts": (0, 1), "host_lane_clusters": -1},
        {"hbm_slots": 4, "initial_hbm_experts": ()},
        {"hbm_slots": 4, "initial_hbm_experts": (0, 1, 2, 3)},  # leaves no spare slot
        {"hbm_slots": 4, "initial_hbm_experts": (0, 0, 1)},  # duplicate expert
        {"hbm_slots": 4, "initial_hbm_experts": (0, 8)},  # out of range
        {"hbm_slots": 4, "initial_hbm_experts": (0, 6), "host_experts": 6},  # missing pinned expert 7
    ):
        with pytest.raises(ValueError):
            MegaMoeDynamicHostTier(geom, storage=storage, name="dyn_check", **kwargs)

    dyn_query = MegaMoeQuery(
        rows=16,
        experts=8,
        capacity=64,
        top_k=2,
        hidden=128,
        intermediate=128,
        shared_experts=1,
        x_sf_stride=(1, 64),
        shared_sf_stride=(1, 64),
        shared_block_m=16,
        clamp=7.0,
        host_rows=6,
        host_lane_clusters=0,
        host_shared_first=True,
        indirection=True,
        hbm_slots=4,
        compute_sm_count=144,
    )
    assert MegaMoeQuery(**dict(MEGAMOE_CONTRACT.encode_query(dyn_query))) == dyn_query

    stage_query = StageQuery(rows=6, words=(1024, 64, 512, 32), min_cold=2, dynamic=True)
    assert StageQuery(**dict(STAGE_CONTRACT.encode_query(stage_query))) == stage_query
    with pytest.raises(ValueError):
        StageQuery(rows=6, words=(1024, 64, 512, 32), min_cold=2, dynamic=1)


def test_pyproject_packages_mover_cuda_source():
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]
    assert "mover_kernels.cu" in package_data["hyperflare_recipes.components.sm103_mover_kernel"]
    cuda_source = repo_root / "hyperflare_recipes/components/sm103_mover_kernel/mover_kernels.cu"
    assert cuda_source.is_file()
    source_text = cuda_source.read_text(encoding="utf-8")
    assert f"sizeof(LayerMoverDesc) == {LAYER_DESC_BYTES}" in source_text
    assert f"sizeof(MoverQueueHeader) == {QUEUE_HEADER_BYTES}" in source_text
    assert f"sizeof(MoverRequest) == {QUEUE_REQUEST_BYTES}" in source_text
    assert f"sizeof(PredictorLayerState) == {PREDICTOR_STATE_BYTES}" in source_text
