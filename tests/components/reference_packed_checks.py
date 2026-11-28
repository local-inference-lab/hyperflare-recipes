# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native packed reference rounds over two independently owned request states."""

from dataclasses import replace
import time
from types import SimpleNamespace

import torch

from ignition.components.completion import CompletionOutputs
from hyperflare_recipes.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.sampling import DeviceRequestState
from hyperflare_recipes.components.speculation import SpeculativeRngState
from ignition.components.state import save_hidden
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution
from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheStorage
from hyperflare_recipes.recipes.deepseek41.reference_engram import ReferenceEngramHistory
from hyperflare_recipes.recipes.deepseek41.reference_speculation import ReferenceSpeculation
from tests.components.causal_prefix_oracle import allocate


@torch.no_grad()
def check_reference_packed_rounds(model, delivery, frame, sequence, session, resident, ordinary):
    import flare_heliopause

    source_pool, _, source_requests, source_inputs, _, _, _ = ordinary
    page_count = next(iter(source_pool.pages.values())).shape[0]
    pool = ReferenceCacheStorage.allocate(source_pool.geometry, 4, page_count + 1, "cuda")
    pool.backing[1:3].copy_(source_pool.backing[1])
    for name, value in pool.pages.items():
        value[:page_count].copy_(source_pool.pages[name])
        value[page_count].copy_(source_pool.pages[name][int(source_inputs.page_tables[0, 0])])
    history = ReferenceEngramHistory.from_config(pool.region("engram_history"), model.cfg)
    requests = DeviceRequestState.allocate("cuda", vocab_size=model.cfg.vocab_size, capacity=2)
    for name in requests.__dataclass_fields__:
        getattr(requests, name).copy_(getattr(source_requests, name)[0])
    requests.seed.view(torch.int64)[1] += 21
    inputs = BatchInputs.allocate(192, 256, "cuda", capacity=2)
    inputs.row_slot_ids[:] = torch.tensor([1, 0], device="cuda")
    inputs.row_generations.fill_(1)
    inputs.live_rows.fill_(2)
    inputs.request_state_rows[:, 0] = torch.tensor([1, 2], device="cuda")
    inputs.page_tables[:] = source_inputs.page_tables[0]
    inputs.page_tables[1, 0] = page_count
    prefix_limits = torch.full((2,), 8, dtype=torch.int32, device="cuda")
    confidence = torch.zeros((2, 2, 7), dtype=torch.float32, device="cuda")
    completion = CompletionOutputs.allocate((2, 2, 32), device="cuda")
    rng = SpeculativeRngState.allocate("cuda", capacity=2)
    proposal_generation = torch.zeros(2, dtype=torch.int64, device="cuda")
    proposal_offset = torch.zeros_like(proposal_generation)
    batch = PreparedBatch(inputs, requests, Metadata.allocate(2, 2, 256, "cuda"), mode="decode",
        shard=SimpleNamespace(world_size=1), pool=pool, verification_width=8)
    operation = ReferenceSpeculation(model, batch, history, completion, frame, sequence,
        rng=rng, proposal_generation=proposal_generation, proposal_offset=proposal_offset, session=session,
        prefix_limits=prefix_limits, confidence_output=confidence, target_token_capacity=12)
    proposals = torch.empty((2, 8), dtype=torch.int64, device="cuda")
    logits = torch.empty((2, 8, model.cfg.vocab_size), dtype=torch.float32, device="cuda")
    state = (pool.backing, *pool.pages.values(),
        *(getattr(requests, name) for name in requests.__dataclass_fields__),
        *completion.buffers().values(), rng.generation, rng.target_offset, proposal_generation, proposal_offset)
    names = ("continuation", *pool.pages, *requests.__dataclass_fields__, *completion.buffers(),
             "target_rng_generation", "target_rng_offset", "proposal_generation", "proposal_offset")
    initial = tuple(value.clone() for value in state)

    def restore(values):
        for value, saved in zip(state, values, strict=True):
            value.copy_(saved)

    def forward():
        candidates, scores, *_ = operation.observe()
        proposals.copy_(candidates)
        logits.copy_(scores)

    compiled = compile_forward(forward)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled()
    torch.cuda.synchronize()
    restore(initial)
    graph = torch.cuda.CUDAGraph()
    with session.capture(), torch.cuda.graph(graph, stream=stream):
        compiled()
    restore(initial)
    fields = tuple(field for field in inputs.buffers() if field != "query_lengths") + ("speculative_prefix_limits",)
    execution = PreparedExecution(device=0, max_active=2, graphs={"packed": graph},
        signatures={"packed": GraphSignature("speculative", 2, 8, fields, draft_tokens=7,
            packed_token_capacity=12, confidence_width=7, speculation_enabled=True)},
        buffers={**inputs.buffers(), **completion.buffers(), **delivery.buffers(),
            "speculative_prefix_limits": prefix_limits, "proposal_confidence": confidence,
            "frame_index": frame, "submission_sequence": sequence},
        resources=(operation, compiled, pool, requests, history, rng, proposals, logits),
        export_events=delivery.export_events())
    manifest = execution.manifest()
    manifest["engram"] = resident.bind(manifest, execution.export_events)
    worker = flare_heliopause.GraphExecutor((execution, resident), manifest)
    try:
        cases = (([8, 8], 2, [1, 0], False), ([8, 8], 2, [0, 1], False),
                 ([2, 8], 2, [1, 0], False), ([8, 8], 1, [1, 0], False),
                 ([8, 8], 2, [1, 0], True), ([8, 8], 0, [1, 0], False))
        for ceilings, live, slots, finished_first in cases:
            restore(initial)
            prefix_limits[:] = torch.tensor(ceilings, device="cuda")
            inputs.row_slot_ids[:] = torch.tensor(slots, device="cuda")
            inputs.live_rows.fill_(live)
            if finished_first:
                requests.status[slots[0]] = 2
            active = [i < live and not (i == 0 and finished_first) for i in range(2)]
            before = tuple(value.clone() for value in state)
            confidence.fill_(-1.)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            fence = worker.submit(0, [("logprob_requested", 0, bytes(8)),
                                      ("logprob_token_id_counts", 0, bytes(8))])
            result, deadline = None, time.monotonic() + 60
            while result is None and time.monotonic() < deadline:
                result = worker.poll()
                if result is None:
                    time.sleep(.001)
            assert result is not None and result.fence == fence
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            lengths = inputs.query_lengths.clone()
            observed_confidence = confidence[int(frame[0])].clone()
            wanted, _ = allocate(observed_confidence.tolist(), slots, active, active, ceilings, 12, 2)
            assert lengths.tolist() == wanted and int(lengths.sum()) <= 12
            assert prefix_limits.tolist() == ceilings
            actual, actual_logits = proposals.clone(), logits.clone()
            after = tuple(value.clone() for value in state)
            restore(before)
            forward()
            torch.cuda.synchronize()
            torch.testing.assert_close(proposals, actual, rtol=0, atol=0)
            torch.testing.assert_close(confidence[int(frame[0])], observed_confidence, rtol=0, atol=0)
            for lane in range(live):
                extent = int(lengths[lane])
                torch.testing.assert_close(logits[lane, :extent], actual_logits[lane, :extent], rtol=0, atol=0)
            for name, value, saved in zip(names, state, after, strict=True):
                tolerance = 1e-6 if name == "completion_speculative_survival" else 0
                torch.testing.assert_close(value, saved, rtol=tolerance, atol=0, msg=lambda message: f"{name}: {message}")
            assert requests.error_code.eq(0).all()
            assert torch.equal(pool.backing[0], initial[0][0])
            if any(active):
                counts = completion.records[int(frame[0]), inputs.row_slot_ids.long(), 8].int().clone()
                active_tensor = torch.tensor(active, device="cuda")
                counts[~active_tensor] = 0
                assert bool(((counts[active_tensor] > 0) & (counts <= lengths)[active_tensor]).all())
                _check_retained_prefix(operation, actual, lengths, counts, before, after)
    finally:
        worker.close()
        torch.cuda.synchronize()
        graph.reset()


def _check_retained_prefix(operation, candidates, lengths, counts, before, after):
    """Mask rejected target rows at their original packed offsets.

    This preserves physical arithmetic shape and the native-delivered Engram
    rows while comparing cache writes with target-only accepted-prefix execution.
    """
    pool, batch = operation.pool, operation.batch
    caches = (pool.backing, *pool.pages.values())
    for value, saved in zip(caches, before[:len(caches)], strict=True):
        value.copy_(saved)
    slots = batch.inputs.row_slot_ids.long()
    starts = pool.region("consumed_tokens")[batch.inputs.request_state_rows[slots, 0], 0]
    packed = Metadata.pack(starts, lengths, batch.inputs.request_state_rows[slots, 0],
        batch.inputs.page_tables[slots], candidates, batch.inputs.live_rows,
        token_capacity=12, sample_count=16, max_query_tokens=8)
    relative = packed.positions - packed.starts[packed.requests]
    prefix = replace(packed, positions=torch.where(relative < counts[packed.requests], packed.positions, -1))
    prepared = operation.history(prefix)
    hidden = operation.model(operation.caches,
        operation.metadata(prefix, ngram_token_ids=prepared.ngrams)).hidden_states
    operation.history.commit(prefix, counts, prepared)
    selected = (packed.query_start_loc[:-1] + counts - 1).long().clamp_min(0)
    save_hidden(hidden[selected], pool.last_hidden, packed.state_rows, counts > 0)
    save_hidden((packed.starts + counts)[:, None], pool.region("consumed_tokens"), packed.state_rows, counts > 0)
    torch.cuda.synchronize()
    for index, (value, saved) in enumerate(zip(caches, after[:len(caches)], strict=True)):
        torch.testing.assert_close(value, saved, rtol=0, atol=0,
            msg=lambda message: f"packed prefix cache plane {index}, accepted={counts.tolist()}: {message}")
