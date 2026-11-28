# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference cache transfers and cached-prefix sampling with synthetic contents."""

import os
import time
from types import SimpleNamespace

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.completion import CompletionOutputs
from hyperflare_recipes.components.paged_batch import BatchInputs
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution
from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheLayout, ReferenceCacheStorage
from hyperflare_recipes.recipes.deepseek41.reference_resume import ReferenceResume
from tests.recipes.test_deepseek41_reference_cache import reference_config

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned SM103 GPU and DeepInfer")


@torch.no_grad()
def test_reference_complete_snapshot_and_fork(reference_config):
    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=8)
    pool = ReferenceCacheStorage.allocate(geometry, 8, 5, "cuda")
    commands = StateCommands.allocate("cuda", capacity=4)
    tables = torch.full((4, 3), -1, dtype=torch.int32, device="cuda")
    requests = DeviceRequestState.allocate("cuda", vocab_size=128, capacity=4,
        consumed_tokens=pool.region("consumed_tokens")[1:5, 0])
    rows = torch.arange(1, 5, device="cuda")[:, None].expand(4, 8).contiguous()
    operation = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(), page_tokens=128,
                                     state_rows=rows, requests=requests)
    operation.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation.validate()
        operation.apply()

    def replay(transfers, endpoint):
        commands.num_ops.fill_(len(transfers))
        for index, (source, destination, request) in enumerate(transfers):
            commands.source_slots[index] = source
            commands.destination_slots[index] = destination
            commands.request_slots[index] = request
            commands.expected_consumed[index] = endpoint
            commands.operation[index] = 0
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        assert not commands.errors.count_nonzero()

    try:
        for endpoint in (1, 127, 128, 129, 255, 256):
            pool.clear_for_preparation()
            pool.backing[1].random_(0, 256)
            pool.region("consumed_tokens")[1] = endpoint
            tables[0] = torch.tensor([3, 0, 4], device="cuda")
            tables[1].fill_(2)
            tables[2].fill_(1)
            for cache in pool.pages.values():
                cache.random_(0, 256)
            tail = int(tables[0, endpoint // 128]) if endpoint % 128 else None
            pages = {name: cache[tail].clone() for name, cache in pool.partial_pages()} if tail is not None else {}
            expected = pool.backing[1].clone()
            for name, value in pages.items():
                region = pool.layout.region(name)
                expected[region.offset:region.offset + region.nbytes].copy_(value)
            replay([(1, 5, 0)], endpoint)
            torch.testing.assert_close(pool.backing[5], expected, rtol=0, atol=0)
            pool.backing[1].fill_(255)
            for cache in pool.pages.values():
                cache.zero_()
            replay([(5, 2, 1), (5, 3, 2)], endpoint)
            for row in (2, 3, 5):
                torch.testing.assert_close(pool.backing[row], expected, rtol=0, atol=0)
            for name, cache in pool.partial_pages():
                if tail is not None:
                    torch.testing.assert_close(cache[1], pages[name], rtol=0, atol=0)
                    torch.testing.assert_close(cache[2], pages[name], rtol=0, atol=0)
                assert not cache[3:].count_nonzero()
            assert not pool.backing[0].count_nonzero()
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("rows", [1, 4])
@torch.no_grad()
def test_reference_native_resume_preserves_cache(reference_config, rows):
    import flare_heliopause
    from deepinfer.model_executor.layers.sampler_utils import get_logits

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=8)
    pool = ReferenceCacheStorage.allocate(geometry, 6, 2, "cuda")
    generator = torch.Generator(device="cuda").manual_seed(531)
    model = SimpleNamespace(cfg=SimpleNamespace(vocab_size=1024), lm_head=SimpleNamespace(
        weight=torch.randn((1024, 5120), dtype=torch.bfloat16, device="cuda", generator=generator) * .03125))
    requests = DeviceRequestState.allocate("cuda", vocab_size=1024, capacity=4)
    inputs = BatchInputs.allocate(192, 4, "cuda", capacity=4)
    inputs.row_slot_ids[:] = torch.tensor([2, 0, 3, 1], device="cuda")
    inputs.row_generations.fill_(7)
    inputs.live_rows.fill_(rows)
    inputs.request_state_rows[:, 0] = torch.tensor([4, 1, 3, 2], device="cuda")
    completion = CompletionOutputs.allocate((2, 4, 32), device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.zeros(1, dtype=torch.int64, device="cuda")
    admission = RequestAdmission(generation=7, seed=11, temperature=1., top_k=1, top_p=1., max_new_tokens=4)

    def reset():
        for slot in range(4):
            requests.admit(slot, admission, [5, 6, 7], consumed_tokens=3)
        for value in completion.buffers().values():
            value.zero_()

    worker = graph = None
    with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
        operation = ReferenceResume(model, requests, pool, inputs, completion, frame, sequence,
                                    rows=rows, session=session)
        compiled = compile_forward(operation)
        reset()
        pool.last_hidden.normal_(generator=generator)
        compiled()
        torch.cuda.synchronize()
        reset()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        execution = PreparedExecution(device=0, max_active=4, graphs={"resume": graph},
            signatures={"resume": GraphSignature("resume", rows, 1, tuple(inputs.buffers()))},
            buffers={**inputs.buffers(), **completion.buffers(), "frame_index": frame, "submission_sequence": sequence},
            resources=(model, pool, requests, inputs, operation, compiled))
        try:
            worker = flare_heliopause.GraphExecutor(execution, execution.manifest())
            for live in (rows, max(rows - 1, 0), rows):
                reset()
                inputs.live_rows.fill_(live)
                pool.backing.random_(0, 256)
                pool.last_hidden.normal_(generator=generator)
                for cache in pool.pages.values():
                    cache.random_(0, 256)
                slots = inputs.row_slot_ids[:live].long()
                hidden = pool.last_hidden[inputs.request_state_rows[slots, 0]]
                expected = get_logits(hidden, model.lm_head.weight, 1024).argmax(-1) if live else None
                cache_before = [pool.backing.clone(), *(value.clone() for value in pool.pages.values())]
                request_before = [getattr(requests, name).clone() for name in requests.__dataclass_fields__]
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                fence = worker.submit(0, [("logprob_requested", 0, bytes(16)),
                                          ("logprob_token_id_counts", 0, bytes(16))])
                result, deadline = None, time.monotonic() + 60
                while result is None and time.monotonic() < deadline:
                    result = worker.poll()
                    if result is None:
                        time.sleep(.001)
                assert result is not None and result.fence == fence
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                for actual, saved in zip((pool.backing, *pool.pages.values()), cache_before, strict=True):
                    torch.testing.assert_close(actual, saved, rtol=0, atol=0)
                inactive = torch.ones(4, dtype=torch.bool, device="cuda")
                inactive[slots] = False
                for name, saved in zip(requests.__dataclass_fields__, request_before, strict=True):
                    actual = getattr(requests, name)
                    if actual.dtype == torch.uint64:
                        actual, saved = actual.view(torch.int64), saved.view(torch.int64)
                    torch.testing.assert_close(actual[inactive], saved[inactive], rtol=0, atol=0)
                assert requests.consumed_tokens.eq(3).all()
                assert requests.generated_tokens[slots].eq(1).all()
                assert requests.rng_counter.view(torch.int64)[slots].eq(1).all()
                assert requests.error_code.eq(0).all()
                if live:
                    torch.testing.assert_close(requests.pending_token[slots], expected, rtol=0, atol=0)
                    assert completion.records[int(frame[0]), slots, 13].eq(3).all()
        finally:
            if worker is not None:
                worker.close()
            torch.cuda.synchronize()
            if graph is not None:
                graph.reset()
