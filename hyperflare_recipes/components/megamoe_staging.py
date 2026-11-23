# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured cold-expert staging, inspired by DerpSeek's prompt staging.

Prepared copy kernels read the canonical residency map on every replay. The
preceding invocation's distinct cold demand decides whether to stage; a skipped
copy binds the original host weights. Staging never changes permanent residency.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from hyperflare_recipes.components.native_operator import native_contract, native_plan
from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.triton import launch_triton, prepare_triton
from ignition.preparation.tuning import declare


@triton.jit
def _copy_cold(source, destination, mapping, canonical, decision,
               ROWS: tl.constexpr, WORDS: tl.constexpr, CTAS: tl.constexpr,
               MIN_COLD: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if tl.load(decision) >= MIN_COLD:
        for row in range(ROWS):
            expert = tl.load(canonical + row)
            if tl.load(mapping + expert) == row:
                base = row.to(tl.int64) * WORDS
                for offset in range(tl.program_id(0) * BLOCK, WORDS, CTAS * BLOCK):
                    column = offset + lane
                    value = tl.load(source + base + column, column < WORDS, 0,
                                    cache_modifier=".cg")
                    tl.store(destination + base + column, value, column < WORDS,
                             cache_modifier=".cs")


@triton.jit
def _snapshot_dynamic_cold(live_slot_map, staged_mask, decision,
                           ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    mask = row < ROWS
    cold = (tl.load(live_slot_map + row, mask, 0) < 0).to(tl.int32)
    tl.store(staged_mask + row, cold, mask)
    tl.store(decision, tl.sum(cold, 0))


@triton.jit
def _copy_cold_dynamic(source, destination, staged_mask, decision,
                       ROWS: tl.constexpr, WORDS: tl.constexpr, CTAS: tl.constexpr,
                       MIN_COLD: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if tl.load(decision) >= MIN_COLD:
        for row in range(ROWS):
            if tl.load(staged_mask + row) != 0:
                base = row.to(tl.int64) * WORDS
                for offset in range(tl.program_id(0) * BLOCK, WORDS, CTAS * BLOCK):
                    column = offset + lane
                    value = tl.load(source + base + column, column < WORDS, 0,
                                    cache_modifier=".cg")
                    tl.store(destination + base + column, value, column < WORDS,
                             cache_modifier=".cs")


@triton.jit
def _bind(source, gate, gate_scale, down, down_scale, output, decision,
          MIN_COLD: tl.constexpr, CELLS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    enabled = tl.load(decision) >= MIN_COLD
    value = tl.load(source + i, mask=i < 16, other=0).to(tl.uint32)
    for plane in tl.static_range(4):
        pointer = (gate, gate_scale, down, down_scale)[plane].to(tl.uint64)
        value = tl.where((i == 2 + plane * 2) & enabled, pointer.to(tl.uint32), value)
        value = tl.where((i == 3 + plane * 2) & enabled, (pointer >> 32).to(tl.uint32), value)
    # Staged rows read HBM; direct-host calls retain the prepared C2C lane.
    value = tl.where((i == 10) & enabled, 0, value)
    tl.store(output + i, value, mask=i < CELLS)


@triton.jit
def _observe(control, decision, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    count = tl.load(control + 16 + row, row < ROWS, 0)
    tl.store(decision, tl.sum((count > 0).to(tl.int32), 0))


@dataclass(frozen=True)
class StageQuery:
    rows: int
    words: tuple
    min_cold: int
    dynamic: bool = False

    def __post_init__(self):
        object.__setattr__(self, "words", tuple(self.words))
        if (type(self.rows) is not int or self.rows <= 0 or len(self.words) != 4
                or any(type(n) is not int or n <= 0 for n in self.words)
                or type(self.min_cold) is not int or not 1 <= self.min_cold <= self.rows
                or type(self.dynamic) is not bool):
            raise ValueError("expert staging requires four positive row extents and a valid cold-demand threshold")


CONTRACT = native_contract(component_id="flare.megamoe.cold_stage", query_type=StageQuery,
                           tactics=lambda query: (64, 128, 256))
_NATIVE_JITS.add(_copy_cold)
_NATIVE_JITS.add(_snapshot_dynamic_cold)
_NATIVE_JITS.add(_copy_cold_dynamic)


@program_cache(scope="preparation")
def compile_copies(payload, ordinal):
    query, ctas = StageQuery(**payload["query"]), payload["ctas"]
    with torch.cuda.device(ordinal):
        if query.dynamic:
            snap = _snapshot_dynamic_cold.warmup(
                torch.int32, torch.int32, torch.int32, query.rows, triton.next_power_of_2(query.rows),
                num_warps=4, grid=(1, 1, 1),
            )
            copies = tuple(_copy_cold_dynamic.warmup(
                torch.int32, torch.int32, torch.int32, torch.int32,
                query.rows, words, ctas, query.min_cold, 4096,
                num_warps=4, grid=(ctas, 1, 1),
            ) for words in query.words)
            return (snap, *copies)
        return tuple(_copy_cold.warmup(torch.int32, torch.int32, torch.int32,
            torch.int32, torch.int32, query.rows, words, ctas, query.min_cold, 4096,
            num_warps=4, grid=(ctas, 1, 1)) for words in query.words)


class _State:
    def __init__(self, query, ctas, device):
        self.query, self.ctas = query, ctas
        loaded = load_programs(compile_copies(
            dict(query=CONTRACT.encode_query(query), ctas=ctas), device.index))
        if query.dynamic:
            self.snapshot_kernel = loaded[0]
            self.kernels = loaded[1:]
        else:
            self.snapshot_kernel = None
            self.kernels = loaded

    def run(self, sources, destinations, mapping, canonical, decision):
        q = self.query
        for kernel, source, destination, words in zip(
                self.kernels, sources, destinations, q.words, strict=True):
            kernel[(self.ctas, 1, 1)](source.view(torch.int32), destination.view(torch.int32),
                mapping, canonical, decision, q.rows, words, self.ctas, q.min_cold, 4096)

    def run_dynamic(self, sources, destinations, live_slot_map, staged_mask, decision):
        q = self.query
        self.snapshot_kernel[(1, 1, 1)](live_slot_map, staged_mask, decision, q.rows, triton.next_power_of_2(q.rows))
        for kernel, source, destination, words in zip(
                self.kernels, sources, destinations, q.words, strict=True):
            kernel[(self.ctas, 1, 1)](source.view(torch.int32), destination.view(torch.int32),
                staged_mask, decision, q.rows, words, self.ctas, q.min_cold, 4096)


@torch.library.custom_op("flare::start_expert_stage", mutates_args=("destinations",))
def _start(anchor: torch.Tensor, sources: list[torch.Tensor],
           destinations: list[torch.Tensor], mapping: torch.Tensor, canonical: torch.Tensor,
           decision: torch.Tensor, stream: int, handle: int) -> None:
    main = torch.cuda.current_stream(anchor.device)
    auxiliary = torch.cuda.ExternalStream(stream, device=anchor.device) if stream else main
    if stream:
        auxiliary.wait_stream(main)
    plan = plan_from_handle(handle)
    with torch.cuda.stream(auxiliary):
        require_prepared(plan, plan.component_id, anchor.device).run(
            sources, destinations, mapping, canonical, decision)


@_start.register_fake
def _start_fake(anchor, sources, destinations, mapping, canonical, decision, stream, handle):
    pass


@torch.library.custom_op("flare::start_expert_stage_dynamic", mutates_args=("destinations", "staged_mask", "decision"))
def _start_dynamic(anchor: torch.Tensor, sources: list[torch.Tensor],
                   destinations: list[torch.Tensor], live_slot_map: torch.Tensor,
                   staged_mask: torch.Tensor, decision: torch.Tensor, stream: int, handle: int) -> None:
    main = torch.cuda.current_stream(anchor.device)
    auxiliary = torch.cuda.ExternalStream(stream, device=anchor.device) if stream else main
    if stream:
        auxiliary.wait_stream(main)
    plan = plan_from_handle(handle)
    with torch.cuda.stream(auxiliary):
        require_prepared(plan, plan.component_id, anchor.device).run_dynamic(
            sources, destinations, live_slot_map, staged_mask, decision)


@_start_dynamic.register_fake
def _start_dynamic_fake(anchor, sources, destinations, live_slot_map, staged_mask, decision, stream, handle):
    pass


@torch.library.custom_op("flare::join_expert_stage", mutates_args=("destinations",))
def _join(consumer: torch.Tensor, sources: list[torch.Tensor],
          destinations: list[torch.Tensor], mapping: torch.Tensor, canonical: torch.Tensor,
          decision: torch.Tensor, stream: int) -> None:
    if stream:
        torch.cuda.current_stream(consumer.device).wait_stream(
            torch.cuda.ExternalStream(stream, device=consumer.device))


@_join.register_fake
def _join_fake(consumer, sources, destinations, mapping, canonical, decision, stream):
    pass


@torch.library.custom_op("flare::join_expert_stage_dynamic", mutates_args=("destinations", "staged_mask"))
def _join_dynamic(consumer: torch.Tensor, sources: list[torch.Tensor],
                  destinations: list[torch.Tensor], live_slot_map: torch.Tensor,
                  staged_mask: torch.Tensor, decision: torch.Tensor, stream: int) -> None:
    if stream:
        torch.cuda.current_stream(consumer.device).wait_stream(
            torch.cuda.ExternalStream(stream, device=consumer.device))


@_join_dynamic.register_fake
def _join_dynamic_fake(consumer, sources, destinations, live_slot_map, staged_mask, decision, stream):
    pass


@torch.library.custom_op("flare::start_expert_stage_worker", mutates_args=("destinations", "staged_mask", "req_seq", "queue_header"))
def _start_worker(anchor: torch.Tensor, destinations: list[torch.Tensor],
                  staged_mask: torch.Tensor, req_seq: torch.Tensor,
                  queue_header: torch.Tensor, layer_idx: int, num_host_experts: int,
                  min_cold: int, kernel_cache: str) -> None:
    from hyperflare_recipes.components.sm103_mover_kernel import prepare as prepare_mover

    prepare_mover(Path(kernel_cache)).launch_stage_request(
        queue_header,
        layer_idx,
        destinations[0],
        destinations[1],
        destinations[2],
        destinations[3],
        staged_mask,
        req_seq,
        num_host_experts,
        min_cold,
    )


@_start_worker.register_fake
def _start_worker_fake(anchor, destinations, staged_mask, req_seq, queue_header,
                       layer_idx, num_host_experts, min_cold, kernel_cache):
    pass


@torch.library.custom_op("flare::join_expert_stage_worker", mutates_args=("destinations", "staged_mask", "req_seq"))
def _join_worker(consumer: torch.Tensor, destinations: list[torch.Tensor],
                 staged_mask: torch.Tensor, req_seq: torch.Tensor,
                 queue_header: torch.Tensor) -> None:
    pass


@_join_worker.register_fake
def _join_worker_fake(consumer, destinations, staged_mask, req_seq, queue_header):
    pass


@dataclass(frozen=True)
class PendingExpertStage:
    planes: list[torch.Tensor]
    staged_mask: torch.Tensor | None = None
    req_seq: torch.Tensor | None = None
    worker_queue_header: torch.Tensor | None = None


class PreparedExpertStage:
    """Look ahead on a caller-owned stream while retaining complete expert work.

    One instance owns one layer's previous cold-demand count, shared across its
    serialized graph variants. Zero history selects direct-host execution; export
    must clear preparation observations after warmup and capture have drained.
    Residency publication must exclude readers for the entire graph, including the
    copy. Reuse a stream only after its preceding consumer is enqueued, passing that
    consumer's output as the next start anchor. A None stream uses the calling
    capture stream for serial diagnosis. Stage scratch is forward-local.
    """

    def __init__(self, tier, stream, mapping, host_experts, *, session, workspace, name, min_cold=8,
                 worker_mover=None, layer_idx=None):
        host_experts = tuple(host_experts)
        if (mapping.dtype != torch.int32 or mapping.ndim != 1 or not mapping.is_contiguous()
                or mapping.device != tier.control.device
                or stream is not None and stream.device != mapping.device
                or len(host_experts) != tier.rows
                or len(set(host_experts)) != tier.rows
                or any(type(i) is not int or not 0 <= i < mapping.numel() for i in host_experts)):
            raise ValueError("expert staging requires canonical host IDs and a contiguous int32 residency map")
        self.tier, self.stream, self.mapping = tier, stream, mapping
        self.dynamic = bool(getattr(tier, "indirection", False))
        self.worker_mover = worker_mover if self.dynamic else None
        self.worker_layer_idx = layer_idx
        self.worker_kernel_cache = str(worker_mover.kernel_cache) if self.worker_mover is not None else ""
        if self.worker_mover is not None and (type(layer_idx) is not int or layer_idx < 0):
            raise ValueError("worker-offloaded expert staging requires a nonnegative mover layer_idx")
        self.stream_handle = 0 if stream is None else stream.cuda_stream
        self.layouts = tuple((tuple(t.shape), t.stride(), t.dtype) for t in tier.device_views)
        self.sources = [t.as_strided((t.numel(),), (1,)).view(torch.uint8) for t in tier.device_views]
        self.canonical = persistent_copy(torch.tensor(host_experts, device=mapping.device, dtype=torch.int32))
        record_numeric_tensor(self.canonical)
        self.decision = persistent_zeros((1,), device=mapping.device, dtype=torch.int32)
        self.query = q = StageQuery(
            tier.rows, tuple(t.numel() // (4 * tier.rows) for t in self.sources), min_cold, dynamic=self.dynamic
        )
        if self.worker_mover is not None:
            self.plan = None
            return
        self.plan = native_plan(contract=CONTRACT, query=q, device=mapping.device,
            scratch_bytes=lambda tactic: sum((source.numel() + 255) // 256 * 256 for source in self.sources),
            compile_jobs=lambda tactic, device: (CompileJob.create(
                "hyperflare_recipes.components.megamoe_staging:compile_copies",
                dict(query=CONTRACT.encode_query(q), ctas=tactic), device.ordinal),),
            materialize=lambda tactic: _State(q, tactic, mapping.device))

        def prepare(state):
            # Candidate trials execute serially. Borrow the shared preparation
            # arena rather than allocating four weight planes per candidate.
            destinations = workspace.views(*((tuple(source.shape), source.dtype) for source in self.sources))
            current_map = mapping.clone()
            decision = torch.full_like(self.decision, min_cold)
            if self.dynamic:
                staged_mask = torch.zeros((tier.rows,), dtype=torch.int32, device=mapping.device)
                def produce():
                    current_map.copy_(mapping)
                    staged_mask.zero_()
                    decision.zero_()
                return PreparedCall(
                    run=lambda: state.run_dynamic(self.sources, destinations, current_map, staged_mask, decision),
                    produce=produce, output=destinations,
                    owners=(tier, mapping, self.canonical),
                )
            def produce():
                current_map.copy_(mapping)
                decision.fill_(min_cold)
            return PreparedCall(run=lambda: state.run(self.sources, destinations, current_map, self.canonical, decision),
                                produce=produce, output=destinations,
                                owners=(tier, mapping, self.canonical))

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=prepare),))
        if not self.dynamic:
            example = torch.empty(1, dtype=torch.uint8, device=mapping.device)
            control = torch.empty_like(tier.control)
            self.cells, self.block = control.numel(), triton.next_power_of_2(control.numel())
            self.binder = prepare_triton(_bind, (tier.control, example, example, example,
                example, control, self.decision, min_cold, self.cells, self.block), (1, 1, 1)).compiled
            self.observer = prepare_triton(_observe, (control, self.decision,
                tier.rows, triton.next_power_of_2(tier.rows)), (1, 1, 1)).compiled

    def start(self, anchor):
        planes = [torch.empty_like(source) for source in self.sources]
        if self.worker_mover is not None:
            staged_mask = torch.empty((self.tier.rows,), dtype=torch.int32, device=anchor.device)
            req_seq = torch.empty((1,), dtype=torch.int32, device=anchor.device)
            _start_worker(anchor, planes, staged_mask, req_seq, self.worker_mover.queue_header,
                          self.worker_layer_idx, self.tier.rows, self.query.min_cold,
                          self.worker_kernel_cache)
            return PendingExpertStage(planes, staged_mask, req_seq, self.worker_mover.queue_header)
        if self.dynamic:
            staged_mask = torch.empty((self.tier.rows,), dtype=torch.int32, device=anchor.device)
            _start_dynamic(anchor, self.sources, planes, self.mapping, staged_mask, self.decision,
                           self.stream_handle, self.plan.handle)
            return PendingExpertStage(planes, staged_mask)
        _start(anchor, self.sources, planes, self.mapping, self.canonical, self.decision,
               self.stream_handle, self.plan.handle)
        return PendingExpertStage(planes)

    def finish(self, pending, consumer):
        if self.worker_mover is not None:
            _join_worker(consumer, pending.planes, pending.staged_mask, pending.req_seq, pending.worker_queue_header)
        elif self.dynamic:
            _join_dynamic(consumer, self.sources, pending.planes, self.mapping, pending.staged_mask,
                          self.decision, self.stream_handle)
        else:
            _join(consumer, self.sources, pending.planes, self.mapping, self.canonical,
                  self.decision, self.stream_handle)
        control = torch.empty_like(self.tier.control)
        if not self.dynamic:
            launch_triton(_bind, self.binder, (self.tier.control, *pending.planes, control,
                self.decision, self.query.min_cold, self.cells, self.block), (1, 1, 1))
        planes = [plane.view(dtype).as_strided(shape, stride)
                  for plane, (shape, stride, dtype) in zip(pending.planes, self.layouts, strict=True)]
        return planes, control

    def observe(self, control):
        if self.dynamic:
            return
        launch_triton(_observe, self.observer, (control, self.decision, self.tier.rows,
            triton.next_power_of_2(self.tier.rows)), (1, 1, 1))
