# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation owner, inline on-GPU predictor state, and persistent graph builder for the SM103 GPU expert mover."""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, fields
from pathlib import Path
import struct
from typing import TYPE_CHECKING

import torch

from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
from hyperflare_recipes.components.megamoe_weights import MegaMoeWeights
from hyperflare_recipes.components.sm103_mover_kernel import prepare as prepare_mover
from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.runtime_core import capture_graph, record_export_graph
from ignition.preparation.tensor_metadata import record_numeric_tensor

if TYPE_CHECKING:
    from ignition.preparation.execution import PersistentWorker

LAYER_DESC_BYTES = 192
LAYER_DESC_PREDICTOR_OFFSET = 136
QUEUE_HEADER_BYTES = 106944
QUEUE_REQUEST_BYTES = 16
PREDICTOR_STATE_BYTES = 128


@dataclass(frozen=True)
class PredictorConfig:
    """Parameters for the inline on-GPU decayed-miss LRU routing predictor."""

    decay_shift: int = 6
    promote_threshold: int = 256
    min_age_steps: int = 64
    interval_steps: int = 8
    cooldown_steps: int = 16
    max_enqueues: int = 1
    max_predict_rows: int = 128

    def __post_init__(self):
        if any(type(v) is not int for v in vars(self).values()):
            raise ValueError("PredictorConfig fields must be integers")
        if not 0 <= self.decay_shift < 31:
            raise ValueError("decay_shift must be in [0, 30]")
        if self.promote_threshold <= 0:
            raise ValueError("promote_threshold must be positive")
        if not 0 <= self.min_age_steps < (1 << 31):
            raise ValueError("min_age_steps must be in [0, 2**31 - 1]")
        if self.interval_steps < 1:
            raise ValueError("interval_steps must be at least 1")
        if self.cooldown_steps < 1:
            raise ValueError("cooldown_steps must be at least 1")
        if not 0 <= self.max_enqueues <= 4:
            raise ValueError("max_enqueues must be in [0, 4]")
        if self.max_predict_rows < 1:
            raise ValueError("max_predict_rows must be positive")


@dataclass(frozen=True)
class PredictorLayerBinding:
    """Per-layer persistent HBM predictor state bound to a MegaMoeDynamicHostTier."""

    layer_idx: int
    state: torch.Tensor
    scores: torch.Tensor
    last_seen: torch.Tensor
    born: torch.Tensor
    cooldown: torch.Tensor
    queue_header: torch.Tensor
    queue_ring: torch.Tensor
    metrics_ptr: int
    metrics_count: int


@dataclass(frozen=True)
class MoverMetrics:
    swaps_completed: int
    bytes_copied: int
    rcu_waits: int
    skipped_already_resident: int
    last_layer: int
    last_expert: int
    predict_hits: int = 0
    predict_misses: int = 0
    predict_enqueues: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.predict_hits + self.predict_misses
        return 0.0 if total == 0 else self.predict_hits / total


def decode_mover_metrics(raw: bytes | bytearray | memoryview) -> MoverMetrics:
    """Decode the 8-counter uint64 mapped-host metrics layout written by the persistent mover."""
    if len(raw) != SM103ExpertMoverRuntime.METRICS_COUNT * 8:
        raise ValueError(
            f"expected {SM103ExpertMoverRuntime.METRICS_COUNT * 8} metric bytes, got {len(raw)}"
        )
    swaps, copied, rcu_waits, skipped, last_packed, p_hits, p_misses, p_enq = struct.unpack("<8Q", raw)
    return MoverMetrics(
        swaps_completed=swaps,
        bytes_copied=copied,
        rcu_waits=rcu_waits,
        skipped_already_resident=skipped,
        last_layer=(last_packed >> 32) & 0xFFFFFFFF,
        last_expert=last_packed & 0xFFFFFFFF,
        predict_hits=p_hits,
        predict_misses=p_misses,
        predict_enqueues=p_enq,
    )


def pack_layer_mover_desc(
    *,
    host_plane_ptrs: tuple[int, int, int, int],
    hbm_plane_ptrs: tuple[int, int, int, int],
    plane_row_bytes: tuple[int, int, int, int],
    live_slot_map_ptr: int,
    slot_owner_ptr: int,
    layer_epoch_ptr: int,
    num_experts: int,
    num_hbm_slots: int,
    num_host_experts: int,
    victim_cursor: int = 0,
    predictor_state_ptr: int = 0,
) -> bytes:
    """Pack one 192-byte LayerMoverDesc matching mover_kernels.cu."""
    entry = struct.pack(
        "<4Q4Q4Q3Q4IQ",
        *host_plane_ptrs,
        *hbm_plane_ptrs,
        *plane_row_bytes,
        live_slot_map_ptr,
        slot_owner_ptr,
        layer_epoch_ptr,
        num_experts,
        num_hbm_slots,
        victim_cursor,
        num_host_experts,
        predictor_state_ptr,
    )
    assert len(entry) == LAYER_DESC_PREDICTOR_OFFSET + 8, len(entry)
    return entry + b"\x00" * (LAYER_DESC_BYTES - len(entry))


def pack_predictor_layer_state(
    *,
    scores_ptr: int,
    last_seen_ptr: int,
    born_ptr: int,
    cooldown_ptr: int,
    init_clock: int,
    layer_idx: int,
    config: PredictorConfig,
    num_host_experts: int,
    last_enqueue_step: int = 0,
) -> bytes:
    """Pack one 128-byte PredictorLayerState matching mover_kernels.cu."""
    raw = struct.pack(
        "<4Q11I",
        scores_ptr,
        last_seen_ptr,
        born_ptr,
        cooldown_ptr,
        init_clock,
        last_enqueue_step,
        layer_idx,
        config.decay_shift,
        config.promote_threshold,
        config.min_age_steps,
        config.interval_steps,
        config.cooldown_steps,
        config.max_enqueues,
        config.max_predict_rows,
        num_host_experts,
    )
    assert len(raw) == 76, len(raw)
    return raw + b"\x00" * (PREDICTOR_STATE_BYTES - len(raw))


def build_persistent_worker(
    *,
    name: str,
    graph,
    worker_sm_count: int,
    control_buffer_name: str,
    metrics_count: int = 8,
    worker_cls=None,
) -> PersistentWorker:
    """Construct a core Ignition PersistentWorker for Execution.persistent_workers export.

    Populates the mapped-host counter metadata (``metrics_buffer``, ``metrics_offset``,
    ``metrics_count``) when the installed core ``PersistentWorker`` schema declares them.
    """
    if worker_cls is None:
        from ignition.preparation.execution import PersistentWorker as worker_cls

    kwargs = {
        "name": name,
        "graph": graph,
        "worker_sm_count": worker_sm_count,
        "stop_flag_buffer": control_buffer_name,
        "stop_flag_offset": SM103ExpertMoverRuntime.STOP_FLAG_OFFSET,
    }
    field_names = {f.name for f in fields(worker_cls)}
    if {"metrics_buffer", "metrics_offset", "metrics_count"} <= field_names:
        kwargs.update(
            metrics_buffer=control_buffer_name,
            metrics_offset=SM103ExpertMoverRuntime.METRICS_OFFSET,
            metrics_count=metrics_count,
        )
    return worker_cls(**kwargs)


def query_green_context_compute_sms(device_ordinal: int, worker_sm_count: int) -> tuple[int, int]:
    """Query CUDA driver SM green-context split so preparation compiles compute kernels for exact compute SMs."""
    if type(worker_sm_count) is not int or worker_sm_count <= 0:
        raise ValueError("worker_sm_count must be a positive integer")
    torch.cuda.init()
    lib = C.CDLL("libcuda.so.1")

    class _DevResource(C.Structure):
        _fields_ = [
            ("type", C.c_int),
            ("_internal_padding", C.c_ubyte * 92),
            ("sm_count", C.c_uint32),
            ("min_sm_partition_size", C.c_uint32),
            ("sm_coscheduled_alignment", C.c_uint32),
            ("flags", C.c_uint32),
            ("_tail", C.c_ubyte * 32),
        ]

    def call(name, *args):
        fn = getattr(lib, name)
        fn.restype = C.c_int
        rc = fn(*args)
        if rc != 0:
            msg = C.c_char_p()
            lib.cuGetErrorString(C.c_int(rc), C.byref(msg))
            raise RuntimeError(f"{name} failed with CUDA {rc}: {msg.value!r}")

    sm_res = _DevResource()
    w_res = _DevResource()
    c_res = _DevResource()
    nb = C.c_uint(1)
    call("cuDeviceGetDevResource", C.c_int(device_ordinal), C.byref(sm_res), C.c_int(1))
    call(
        "cuDevSmResourceSplitByCount",
        C.byref(w_res), C.byref(nb), C.byref(sm_res), C.byref(c_res),
        C.c_uint(0), C.c_uint(worker_sm_count),
    )
    w_sms, c_sms = int(w_res.sm_count), int(c_res.sm_count)
    if w_sms == 0 or c_sms == 0:
        raise ValueError(f"invalid green context SM split for worker_sm_count={worker_sm_count}: ({w_sms}, {c_sms})")
    return w_sms, c_sms


class SM103ExpertMoverRuntime:
    """Own GPU descriptor tables, lock-free prefetch ring, predictor state, and mapped-host controls.

    Prepares the persistent 1D-TMA mover kernel for capture as a dedicated
    ``Execution.persistent_workers`` graph running on a spatially partitioned
    CUDA green context alongside compute graphs.
    """

    METRICS_COUNT = 8
    CONTROL_BYTES = 128
    STOP_FLAG_OFFSET = 0
    METRICS_OFFSET = 8

    def __init__(
        self,
        layers: list[tuple[MegaMoeDynamicHostTier, MegaMoeWeights]] | tuple[tuple[MegaMoeDynamicHostTier, MegaMoeWeights], ...],
        *,
        storage,
        kernel_cache: Path,
        worker_sm_count: int = 8,
        queue_capacity: int = 1024,
        control_buffer_name: str = "sm103_mover_control",
    ):
        layers = tuple(layers)
        if not layers:
            raise ValueError("SM103ExpertMoverRuntime requires at least one dynamic layer")
        if type(worker_sm_count) is not int or worker_sm_count <= 0:
            raise ValueError("worker_sm_count must be a positive integer")
        if type(queue_capacity) is not int or queue_capacity < 16 or (queue_capacity & (queue_capacity - 1)) != 0:
            raise ValueError("queue_capacity must be a power of two >= 16")
        if not isinstance(control_buffer_name, str) or not control_buffer_name.strip():
            raise ValueError("control_buffer_name must be a nonempty string")

        self.layers = layers
        self.storage = storage
        self.device = torch.device("cuda", storage.device)
        self.worker_sm_count = worker_sm_count
        self.queue_capacity = queue_capacity
        self.control_buffer_name = control_buffer_name
        self.kernel_cache = Path(kernel_cache)
        self.binding = prepare_mover(kernel_cache)
        with torch.cuda.device(self.device):
            self.binding.configure_persistent_mover()

        desc_bytes = self.binding.layer_desc_bytes()
        hdr_bytes = self.binding.queue_header_bytes()
        req_bytes = self.binding.queue_request_bytes()
        pred_bytes = self.binding.predictor_state_bytes()
        if (
            desc_bytes != LAYER_DESC_BYTES
            or hdr_bytes != QUEUE_HEADER_BYTES
            or req_bytes != QUEUE_REQUEST_BYTES
            or pred_bytes != PREDICTOR_STATE_BYTES
        ):
            raise RuntimeError(
                f"unexpected mover struct sizes: desc={desc_bytes}, hdr={hdr_bytes}, req={req_bytes}, pred={pred_bytes}"
            )

        # MappedHost control block: [0..4) stop_flag (uint32), [8..8+8*8) metrics (uint64[8]).
        self.control_region = storage.host_allocate(control_buffer_name, (self.CONTROL_BYTES,), torch.uint8)
        self.control_region.host_view.zero_()
        self.stop_flag_ptr = int(self.control_region.device_view.data_ptr())
        self.metrics_ptr = self.stop_flag_ptr + self.METRICS_OFFSET

        # Pack immutable per-layer descriptors into persistent HBM storage.
        packed = bytearray()
        for tier, hbm_weights in layers:
            if not isinstance(tier, MegaMoeDynamicHostTier):
                raise TypeError("SM103ExpertMoverRuntime requires MegaMoeDynamicHostTier layers")
            hbm_planes = (*hbm_weights.gate_up, *hbm_weights.down)
            host_planes = tuple(tier.device_views)
            if len(hbm_planes) != 4 or len(host_planes) != 4:
                raise ValueError("each dynamic MegaMoE layer requires four weight/scale planes")
            for p_idx, (hp, dp, row_b) in enumerate(zip(host_planes, hbm_planes, tier.plane_row_bytes)):
                if dp.shape[0] != tier.hbm_slots or hp.shape[0] != tier.rows:
                    raise ValueError(f"plane {p_idx} row counts do not match tier geometry")
                if dp.stride(0) * dp.element_size() != row_b:
                    raise ValueError(f"plane {p_idx} HBM row stride differs from HOST_NUMA row stride")
            packed.extend(
                pack_layer_mover_desc(
                    host_plane_ptrs=tuple(int(t.data_ptr()) for t in host_planes),
                    hbm_plane_ptrs=tuple(int(t.data_ptr()) for t in hbm_planes),
                    plane_row_bytes=tuple(int(b) for b in tier.plane_row_bytes),
                    live_slot_map_ptr=int(tier.live_slot_map.data_ptr()),
                    slot_owner_ptr=int(tier.slot_owner.data_ptr()),
                    layer_epoch_ptr=int(tier.layer_epoch.data_ptr()),
                    num_experts=int(tier.experts),
                    num_hbm_slots=int(tier.hbm_slots),
                    num_host_experts=int(tier.rows),
                )
            )

        self.layer_descs = persistent_copy(torch.tensor(list(packed), dtype=torch.uint8, device=self.device))
        header_init = bytearray(hdr_bytes)
        struct.pack_into("<III", header_init, 0, 0, 0, queue_capacity - 1)
        self.queue_header = persistent_copy(torch.tensor(list(header_init), dtype=torch.uint8, device=self.device))
        self.queue_ring = persistent_zeros((queue_capacity * req_bytes,), dtype=torch.uint8, device=self.device)
        self.predictor_bindings: list[PredictorLayerBinding] = []
        # Omit self.layer_descs: it stores 64-bit host/HBM/table pointers that export_core must scan.
        for tensor in (self.queue_header, self.queue_ring, self.control_region.device_view):
            record_numeric_tensor(tensor)

    def _initial_born_tensor(self, tier: MegaMoeDynamicHostTier) -> torch.Tensor:
        # Bit 31 (0x80000000 == -2**31 in int32) marks initially resident movable experts born at step 0.
        resident = tier.live_slot_map[:tier.rows] >= 0
        return torch.where(
            resident,
            torch.full((tier.rows,), -(1 << 31), dtype=torch.int32, device=self.device),
            torch.zeros((tier.rows,), dtype=torch.int32, device=self.device),
        )

    def attach_predictors(self, config: PredictorConfig | None = None) -> tuple[PredictorLayerBinding, ...]:
        """Allocate per-layer GPU predictor tables and bind them to each MegaMoeDynamicHostTier."""
        if self.predictor_bindings:
            raise RuntimeError("predictors have already been attached to this mover runtime")
        cfg = PredictorConfig() if config is None else config
        bindings = []
        init_clock = max(1, cfg.cooldown_steps, cfg.min_age_steps, cfg.interval_steps)
        for layer_idx, (tier, _) in enumerate(self.layers):
            if tier.experts > 1024:
                raise ValueError(f"inline GPU predictor supports up to 1024 experts, got {tier.experts}")
            scores = persistent_zeros((tier.rows,), dtype=torch.int32, device=self.device)
            last_seen = persistent_zeros((tier.rows,), dtype=torch.int32, device=self.device)
            born = persistent_copy(self._initial_born_tensor(tier))
            cooldown = persistent_zeros((tier.rows,), dtype=torch.int32, device=self.device)
            packed_state = pack_predictor_layer_state(
                scores_ptr=int(scores.data_ptr()),
                last_seen_ptr=int(last_seen.data_ptr()),
                born_ptr=int(born.data_ptr()),
                cooldown_ptr=int(cooldown.data_ptr()),
                init_clock=init_clock,
                layer_idx=layer_idx,
                config=cfg,
                num_host_experts=int(tier.rows),
            )
            state = persistent_copy(torch.tensor(list(packed_state), dtype=torch.uint8, device=self.device))
            # Bind predictor_state pointer inside LayerMoverDesc[layer_idx] at byte offset 136
            # so the persistent worker can run predictor scoring off the compute stream.
            desc_off = layer_idx * LAYER_DESC_BYTES + LAYER_DESC_PREDICTOR_OFFSET
            self.layer_descs[desc_off:desc_off + 8].copy_(
                torch.tensor(list(struct.pack("<Q", int(state.data_ptr()))), dtype=torch.uint8, device=self.device)
            )
            # Omit state: it stores 64-bit pointers to scores, last_seen, born, and cooldown.
            for tensor in (scores, last_seen, born, cooldown):
                record_numeric_tensor(tensor)
            binding = PredictorLayerBinding(
                layer_idx=layer_idx,
                state=state,
                scores=scores,
                last_seen=last_seen,
                born=born,
                cooldown=cooldown,
                queue_header=self.queue_header,
                queue_ring=self.queue_ring,
                metrics_ptr=self.metrics_ptr,
                metrics_count=self.METRICS_COUNT,
            )
            tier.predictor_binding = binding
            bindings.append(binding)
        self.predictor_bindings = bindings
        return tuple(bindings)

    def enqueue_swap(self, layer_idx: int, expert_in: int, victim_expert: int = -1) -> None:
        self.binding.launch_enqueue_swap(
            self.queue_header, self.queue_ring, int(layer_idx), int(expert_in), int(victim_expert)
        )

    def enqueue_swap_batch(
        self, layers: torch.Tensor, experts_in: torch.Tensor, victims: torch.Tensor
    ) -> None:
        self.binding.launch_enqueue_swap_batch(
            self.queue_header, self.queue_ring, layers, experts_in, victims
        )

    def launch_worker(self, *, stream_handle: int = 0) -> None:
        self.binding.launch_persistent_mover(
            self.layer_descs,
            len(self.layers),
            self.queue_header,
            self.queue_ring,
            self.stop_flag_ptr,
            self.metrics_ptr,
            self.METRICS_COUNT,
            self.worker_sm_count,
            int(stream_handle),
        )

    def capture_worker_graph(
        self, *, name: str = "sm103_expert_mover", retain_graphs: bool = True, stream=None
    ) -> PersistentWorker:
        """Capture the persistent mover kernel into a dedicated CUDA graph without executing it."""
        graph = capture_graph(retain_graphs=retain_graphs)
        capture_stream = torch.cuda.Stream(device=self.device) if stream is None else stream
        with torch.cuda.device(self.device), torch.cuda.stream(capture_stream):
            with torch.cuda.graph(graph, pool=self.storage.graph_pool, stream=capture_stream):
                self.launch_worker(stream_handle=int(capture_stream.cuda_stream))
        if retain_graphs:
            record_export_graph(graph, self.storage, owners=(self,))
        return build_persistent_worker(
            name=name,
            graph=graph,
            worker_sm_count=self.worker_sm_count,
            control_buffer_name=self.control_buffer_name,
            metrics_count=self.METRICS_COUNT,
        )

    def read_metrics(self) -> MoverMetrics:
        raw = bytes(self.control_region.host_view[self.METRICS_OFFSET:self.METRICS_OFFSET + self.METRICS_COUNT * 8].tolist())
        return decode_mover_metrics(raw)

    def reset_metrics(self) -> None:
        self.control_region.host_view[self.METRICS_OFFSET:self.METRICS_OFFSET + self.METRICS_COUNT * 8].zero_()

    def reset_runtime_state(self) -> None:
        """Reset mutable queue, epoch, predictor, and metric state after preparation warmup."""
        header_init = bytearray(self.binding.queue_header_bytes())
        struct.pack_into("<III", header_init, 0, 0, 0, self.queue_capacity - 1)
        self.queue_header.copy_(torch.tensor(list(header_init), dtype=torch.uint8, device=self.device))
        self.queue_ring.zero_()
        self.control_region.host_view.zero_()
        for tier, _ in self.layers:
            tier.layer_epoch.zero_()
        if self.predictor_bindings:
            for (tier, _), binding in zip(self.layers, self.predictor_bindings, strict=True):
                binding.scores.zero_()
                binding.last_seen.zero_()
                binding.born.copy_(self._initial_born_tensor(tier))
                binding.cooldown.zero_()
                # Restore step_clock at [32:36] and last_enqueue_step = 0 at [36:40] from
                # (min_age_steps, interval_steps, cooldown_steps) at [52:64].
                min_age_steps, interval_steps, cooldown_steps = struct.unpack_from(
                    "<3I", bytes(binding.state[52:64].cpu().tolist()), 0
                )
                init_clock = max(1, cooldown_steps, min_age_steps, interval_steps)
                binding.state[32:40].copy_(
                    torch.tensor(list(struct.pack("<2I", init_clock, 0)), dtype=torch.uint8, device=self.device)
                )

    def reset_stop_flag(self) -> None:
        self.binding.store_stop_flag_release(self.stop_flag_ptr, 0)

    def signal_stop(self) -> None:
        self.binding.store_stop_flag_release(self.stop_flag_ptr, 1)

    def close(self) -> None:
        if not self.control_region.closed:
            self.control_region.close()
