# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Assemble all checkpoint layers with real tiered experts and native delivery."""

import os
import time
from pathlib import Path

import pytest
import torch

from hyperflare_recipes.components.expert_placement import ExpertPlacement
from hyperflare_recipes.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution
from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheLayout, ReferenceCacheStorage
from hyperflare_recipes.recipes.deepseek41.reference_generation import ReferenceGeneration
from ignition.preparation.tuning import RecordedPreparationSession
from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint
from hyperflare_recipes.recipes.deepseek41.reference_engram import ReferenceEngramDelivery, ReferenceEngramHistory
from hyperflare_recipes.recipes.deepseek41.reference_model import ReferenceExpertPreparation, ReferenceModel
from hyperflare_recipes.recipes.deepseek41.reference_parameters import ReferenceParameters

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or os.environ.get("FLARE_RUN_REFERENCE_MODEL_CUDA_TESTS") != "1",
    reason="requires an assigned GB300 and memory for all target/draft experts")


@torch.no_grad()
def test_complete_reference_native_generation(tmp_path, megamoe_core_storage):
    import flare_heliopause
    from deepinfer.model_executor.execution_settings import ExecutionSettings
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    path = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not path:
        pytest.skip("requires the DeepSeek V4.1 CSF checkpoint")
    if PG.world is None:
        PG.initialize_local()
    checkpoint = ReferenceCheckpoint(Path(path))
    parameters = ReferenceParameters(checkpoint)
    parameters.load_ordinary("cuda:0")
    cfg = parameters.config.gpt
    placements = {f"layers.{i}.ffn": ExpertPlacement(384, tuple(range(192, 384)), tuple(range(192)))
                  for i in range(cfg.num_hidden_layers)}
    placements.update({f"mtp.{i}.ffn": ExpertPlacement(128, tuple(range(64, 128)), tuple(range(64)))
                       for i in range(cfg.num_nextn_predict_layers)})
    experts = delivery = None
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint=str(checkpoint.source.snapshot),
                workload_schema="complete-reference-construction", numerical_policy="reference-checkpoint",
                compile_workers=16, autotune_effort="off") as session:
            experts = ReferenceExpertPreparation.load(checkpoint, placements,
                target_rows=(1, 8, 12, 192, 8192), draft_rows=(1, 7, 14), storage=megamoe_core_storage,
                kernel_cache=tmp_path, session=session)
            sequence = torch.zeros(1, dtype=torch.int64, device="cuda")
            frame = torch.zeros(1, dtype=torch.int32, device="cuda")
            delivery = ReferenceEngramDelivery.from_checkpoint(checkpoint, cfg, capacity=8192,
                sequence=sequence, frame=frame,
                host_allocator=megamoe_core_storage.host_allocate)
            execution = ExecutionSettings(draft_tokens=7, batch_invariant=False,
                yoco_prefill_truncation="off", index_memory_limit=64 << 20,
                sampler_topk=0, confidence_temperature=1.)
            model = ReferenceModel(parameters, experts, delivery, execution=execution, context=32768,
                                   request_capacity=2, prefill_tokens=8192, verification_tokens=12)
            session.tune()
            assert len(model.model.layers) == 40
            assert len(model.draft.layers) == 3
            assert len(model.vision_language_tuner.visual_encoder.blocks) == 32
            assert model.model.engram_runtime is delivery
            assert model.draft.execution is execution
            assert model.model.execution is execution
            assert model.draft.max_MTP_value == 7
            assert all(value.is_cuda for value in model.parameters())
            for name, module in experts.modules.items():
                assert module.weight.shape[0] == placements[name].experts
                assert module.placement.flatten().unique().numel() == placements[name].experts
                assert experts.weights[name].host_tier is not None
                assert all(op.plan.query.host_shared_first for op in module.operations.values())
            resident = flare_heliopause.EmbeddingTables(delivery.resident_catalog())
            ordinary = _native_generation(model, delivery, frame, sequence, session, resident,
                    prefill_capacity=1, chunks=(1,), generation_budget=12)
            from tests.components.reference_packed_checks import check_reference_packed_rounds
            check_reference_packed_rounds(model, delivery, frame, sequence, session, resident, ordinary)
            _native_speculation(model, delivery, frame, sequence, session, resident, *ordinary)
            _native_generation(model, delivery, frame, sequence, session, resident,
                    prefill_capacity=192, chunks=(129, 128, 3))
            conditioned = _native_generation(model, delivery, frame, sequence, session, resident,
                    prefill_capacity=192, chunks=(129, 3), image=True, generation_budget=12)
            _native_speculation(model, delivery, frame, sequence, session, resident, *conditioned)
            _native_generation(model, delivery, frame, sequence, session, resident,
                    prefill_capacity=8192, chunks=(8192, 129, 3))
            assert not torch.distributed.is_initialized()
            with pytest.raises(ValueError, match="unconsumed"):
                ReferenceModel(parameters, experts, delivery, execution=execution, context=32768,
                               request_capacity=1, prefill_tokens=8192, verification_tokens=8)
    finally:
        torch.cuda.synchronize()
        # The native storage owner also tracks regions allocated before an
        # interrupted layer load could be returned in the expert collection.
        for region in megamoe_core_storage.host_regions:
            region.close()


def _native_generation(model, delivery, frame, sequence, session, resident, *, prefill_capacity, chunks,
                       generation_budget=4, image=False):
    import flare_heliopause
    from types import SimpleNamespace
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState, RequestAdmission, FINISHED

    geometry = ReferenceCacheLayout.from_config(model.cfg, verification_width=8)
    prompt = [(1000 + 31 * i) % model.cfg.vocab_size for i in range(sum(chunks))]
    page_count = (len(prompt) + 4 + geometry.page_tokens - 1) // geometry.page_tokens
    pool = ReferenceCacheStorage.allocate(geometry, 2, page_count + 2, "cuda")
    history = ReferenceEngramHistory.from_config(pool.region("engram_history"), model.cfg)
    requests = DeviceRequestState.allocate("cuda", vocab_size=model.cfg.vocab_size, capacity=1)
    inputs = BatchInputs.allocate(prefill_capacity, 256, "cuda", capacity=1)
    inputs.row_slot_ids.fill_(0)
    inputs.row_generations.fill_(1)
    inputs.live_rows.fill_(1)
    inputs.request_state_rows[0, 0] = 1
    inputs.page_tables[0, :page_count] = torch.arange(2, page_count + 2, device="cuda")
    inputs.query_lengths.fill_(chunks[0])
    inputs.sample_enabled.fill_(int(len(chunks) == 1))
    inputs.token_ids[:chunks[0]] = torch.tensor(prompt[:chunks[0]], device="cuda")
    completion = CompletionOutputs.allocate((2, 1, 32), device="cuda")
    vision = image_forward = None
    if image:
        vision, image_forward, image_tokens, feature_indices = _image_conditioning(
            model, inputs, requests, completion, frame, sequence)
        image_start = chunks[0] - len(image_tokens)
        assert image_start >= 0
        prompt[image_start:chunks[0]] = image_tokens
        inputs.token_ids[:chunks[0]] = torch.tensor(prompt[:chunks[0]], device="cuda")
        vision.prefill.vision_token_indices[image_start:chunks[0]] = torch.tensor(feature_indices, device="cuda")
    output = torch.empty((1, model.cfg.hidden_size), dtype=torch.bfloat16, device="cuda")
    logits = torch.empty((1, model.cfg.vocab_size), dtype=torch.float32, device="cuda")
    admission = RequestAdmission(generation=1, seed=187, max_new_tokens=generation_budget,
                                temperature=1., top_k=0, top_p=1.)

    def reset():
        pool.clear_for_preparation()
        requests.admit(0, admission, [token for token in prompt if token < model.cfg.vocab_size])
        for value in completion.buffers().values():
            value.zero_()

    forwards, compiled, graphs = {}, {}, {}
    worker = None
    try:
        for mode in ("prefill", "decode"):
            tokens = prefill_capacity if mode == "prefill" else 1
            metadata = Metadata.allocate(tokens, 1, 256, "cuda", max_query_tokens=tokens)
            batch = PreparedBatch(inputs, requests, metadata, mode=mode,
                shard=SimpleNamespace(world_size=1), pool=pool,
                input_vocabulary=model.cfg.vocab_size + (4 if image else 0),
                validate_prefill=vision.validate_prefill if image else None)
            generation = ReferenceGeneration(model, batch, history, completion, frame, sequence,
                                             session=session, vision=vision)

            def forward(generation=generation):
                hidden, scores = generation.observe()
                output.copy_(hidden)
                logits.copy_(scores)

            forwards[mode] = forward
            compiled[mode] = compile_forward(forward)
        if image:
            compiled["vision"] = compile_forward(image_forward)
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        modes = ("vision", "prefill", "decode") if image else ("prefill", "decode")
        for mode in modes:
            reset()
            if mode == "decode":
                requests.pending_token[0] = 1000
            with torch.cuda.stream(capture_stream):
                compiled[mode]()
            torch.cuda.synchronize()
        graph_pool = torch.cuda.graph_pool_handle()
        for mode in modes:
            reset()
            if mode == "decode":
                requests.pending_token[0] = 1000
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph, stream=capture_stream, pool=graph_pool):
                compiled[mode]()
            graphs[mode] = graph
        reset()
        torch.cuda.synchronize()
        buffers = dict(**inputs.buffers(), **completion.buffers(), output=output, logits=logits,
                       frame_index=frame, submission_sequence=sequence, **delivery.buffers())
        signatures = {mode: GraphSignature(mode, 1, prefill_capacity if mode == "prefill" else 1,
                                           tuple(inputs.buffers())) for mode in ("prefill", "decode")}
        if image:
            buffers.update(vision.prefill.buffers())
            buffers.update(image_forward.encoder.inputs.buffers())
            signatures["vision"] = GraphSignature("vision", 1, 576, tuple(image_forward.encoder.inputs.buffers()))
        execution = PreparedExecution(device=0, graphs=graphs, buffers=buffers,
            signatures=signatures, max_active=1,
            resources=(model, pool, history, requests, inputs, completion, forwards, compiled),
            export_events=delivery.export_events())
        manifest = execution.manifest()
        manifest["engram"] = resident.bind(manifest, execution.export_events)
        worker = flare_heliopause.GraphExecutor((execution, resident), manifest)
        if image:
            vision.arena.fill_(float("nan"))
            vision.count.zero_()
            vision.generation.zero_()
            fence = worker.submit(list(graphs).index("vision"), [])
            result, deadline = None, time.monotonic() + 60
            while result is None and time.monotonic() < deadline:
                result = worker.poll()
                if result is None:
                    time.sleep(.001)
            assert result is not None and result.fence == fence
            assert int(vision.count[0]) == 64 and int(vision.generation[0]) == 1
            assert torch.isfinite(vision.arena).all()
            _compare_image_embedding(model, vision, inputs)
        state = (pool.backing, *pool.pages.values(),
                 *(getattr(requests, name) for name in requests.__dataclass_fields__),
                 *completion.buffers().values())
        emitted, consumed = [], []
        rounds = [("prefill", length, index == len(chunks) - 1) for index, length in enumerate(chunks)]
        rounds += [("decode", 1, True)] * (3 + int(generation_budget == 4))
        for step, (mode, length, sample) in enumerate(rounds):
            inputs.query_lengths.fill_(length)
            inputs.sample_enabled.fill_(int(sample))
            if mode == "prefill":
                inputs.token_ids.zero_()
                inputs.token_ids[:length] = torch.tensor(prompt[len(consumed):len(consumed) + length], device="cuda")
                if image and step > 0:
                    vision.prefill.vision_token_indices.fill_(-1)
            active = len(emitted) < admission.max_new_tokens
            if active:
                consumed += (prompt[len(consumed):len(consumed) + length] if mode == "prefill" else [emitted[-1]])
            before = tuple(value.clone() for value in state)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            fence = worker.submit(list(graphs).index(mode), [
                ("logprob_requested", 0, bytes(4)),
                ("logprob_token_id_counts", 0, bytes(4)),
            ])
            assert fence is not None
            result, deadline = None, time.monotonic() + 60
            while result is None and time.monotonic() < deadline:
                result = worker.poll()
                if result is None:
                    time.sleep(.001)
            assert result is not None and result.fence == fence
            assert result.readiness_wait_ns is not None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            if active:
                assert torch.isfinite(output).all() and torch.isfinite(logits).all()
            actual, actual_logits = output.clone(), logits.clone()
            after = tuple(value.clone() for value in state)
            for value, saved in zip(state, before, strict=True):
                value.copy_(saved)
            # Delivery has drained; both executions consume the same native rows
            # from identical request RNG, pending-token and continuation state.
            forwards[mode]()
            torch.cuda.synchronize()
            if active:
                torch.testing.assert_close(output, actual, rtol=0, atol=0)
                torch.testing.assert_close(logits, actual_logits, rtol=0, atol=0)
            for value, saved in zip(state, after, strict=True):
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
            assert torch.equal(pool.backing[0], before[0][0])
            expected_count = len(consumed)
            generated_count = len(emitted) + int(active and sample)
            assert int(requests.consumed_tokens[0]) == expected_count
            assert int(requests.generated_tokens[0]) == generated_count
            assert int(requests.rng_counter[0]) == generated_count
            assert int(requests.error_code[0]) == 0
            assert int(pool.region("consumed_tokens")[1, 0]) == expected_count
            record = completion.records[int(frame[0]), 0]
            assert int(record[6]) == 0
            assert int(record[2]) == expected_count
            assert int(record[3]) == generated_count
            assert int(record[8]) == int(active and sample)
            assert int(record[13]) == expected_count
            if active and sample:
                token = int(record[16])
                assert 0 <= token < model.cfg.vocab_size
                emitted.append(token)
            elif not active:
                assert int(requests.status[0]) == FINISHED
                assert torch.equal(pool.backing, before[0])
            expected_history = [token if token < model.cfg.vocab_size else -1 for token in consumed[-3:]]
            assert pool.region("engram_history")[1].tolist()[-min(3, len(consumed)):] == expected_history
        assert len(emitted) == 4
        expected_histogram = torch.bincount(torch.tensor([t for t in prompt + emitted if t < model.cfg.vocab_size], device="cuda"),
                                            minlength=model.cfg.vocab_size)
        torch.testing.assert_close(requests.histogram[0].long(), expected_histogram, rtol=0, atol=0)
        return pool, history, requests, inputs, completion, consumed, emitted
    finally:
        if worker is not None:
            worker.close()
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()


def _image_conditioning(model, inputs, requests, completion, frame, sequence):
    from hyperflare_recipes.components.deepseek_vision import (
        PrefillVisionInputs, PreparedVisionGeneration, VisionInputs, VisionState,
    )
    from hyperflare_recipes.recipes.deepseek41.reference_vision import ReferenceVision

    source = model.vision_language_tuner
    weights = {name: getattr(source.params, name) for name in ("image_start", "image_end", "image_newline")}
    state = VisionState(weights, inputs, PrefillVisionInputs.allocate(inputs.token_ids.numel(), "cuda"), requests,
        features=64, token_base=model.cfg.vocab_size,
        sentinel_names=("image_start", "image_start", "image_newline", "image_end"), feature_type=1, image_slots=1)
    state.feature_slot.zero_()
    pixels = VisionInputs.allocate("cuda", patches=576)
    pixels.vision_slot.zero_()
    pixels.vision_generation.fill_(1)
    pixels.vision_num_patches.fill_(576)
    pixels.vision_num_features.fill_(64)
    pixels.vision_cu_seqlens[1] = 576
    generator = torch.Generator(device="cuda").manual_seed(2187)
    pixels.vision_pixel_values.normal_(generator=generator)
    pixels.vision_position_ids[:] = torch.stack(torch.meshgrid(torch.arange(24, device="cuda"),
        torch.arange(24, device="cuda"), indexing="ij"), -1).view(-1, 2)
    pixels.vision_merge_indices[:] = torch.tensor([
        [(row * 3 + dy) * 24 + col * 3 + dx for dy in range(3) for dx in range(3)]
        for row in range(8) for col in range(8)], device="cuda")
    operation = PreparedVisionGeneration(ReferenceVision(source, pixels, capacity=576), state,
                                          completion, frame, sequence)
    tokens, indices = [state.token_base], [-1]
    for row in range(8):
        tokens.extend([state.token_base + 1] * 8 + [state.token_base + 2])
        indices.extend(list(range(row * 8, (row + 1) * 8)) + [-1])
    tokens.append(state.token_base + 3)
    indices.append(-1)
    return state, operation, tokens, indices


def _compare_image_embedding(model, vision, inputs):
    from types import SimpleNamespace
    from deepinfer.multimodal_tokens import TokenType

    ids = inputs.token_ids
    actual = model.model.embed_tokens(ids.int())
    expected = actual.clone()
    vision.inject(SimpleNamespace(token_ids=ids), actual)
    kinds = torch.full_like(ids, int(TokenType.TEXT), dtype=torch.int32)
    for offset, kind in enumerate((TokenType.IMAGE_START, TokenType.IMAGE, TokenType.IMAGE_NEW_LINE, TokenType.IMAGE_END)):
        kinds[ids == model.cfg.vocab_size + offset] = int(kind)
    model.vision_language_tuner._merge_image_embeds(expected, vision.arena[0], kinds)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _native_speculation(model, delivery, frame, sequence, session, resident,
                        pool, history, requests, inputs, completion, consumed, emitted):
    import flare_heliopause
    from types import SimpleNamespace
    from ignition.components.sampling import FINISHED
    from hyperflare_recipes.components.speculation import SpeculativeRngState
    from hyperflare_recipes.recipes.deepseek41.reference_speculation import ReferenceSpeculation

    prompt = consumed[:len(consumed) - len(emitted) + 1]
    inputs.query_lengths.fill_(8)
    batch = PreparedBatch(inputs, requests, Metadata.allocate(1, 1, 256, "cuda"),
        mode="decode", shard=SimpleNamespace(world_size=1), pool=pool, verification_width=8)
    rng = SpeculativeRngState.allocate("cuda", capacity=1)
    proposal_generation = torch.zeros(1, dtype=torch.int64, device="cuda")
    proposal_offset = torch.zeros_like(proposal_generation)
    operation = ReferenceSpeculation(model, batch, history, completion, frame, sequence,
        rng=rng, proposal_generation=proposal_generation, proposal_offset=proposal_offset, session=session)
    proposals = torch.empty((1, 8), dtype=torch.int64, device="cuda")
    logits = torch.empty((1, 8, model.cfg.vocab_size), dtype=torch.float32, device="cuda")
    draft_logits = torch.empty((1, 7, model.cfg.vocab_size), dtype=torch.float32, device="cuda")
    draft_probabilities = torch.empty_like(draft_logits)
    target_probabilities = torch.empty_like(logits)
    state = (pool.backing, *pool.pages.values(),
             *(getattr(requests, name) for name in requests.__dataclass_fields__),
             *completion.buffers().values(), rng.generation, rng.target_offset,
             proposal_generation, proposal_offset)
    state_names = ("continuation", *pool.pages,
                   *requests.__dataclass_fields__, *completion.buffers(),
                   "target_rng_generation", "target_rng_offset", "proposal_generation", "proposal_offset")
    initial = tuple(value.clone() for value in state)

    def restore(values):
        for value, saved in zip(state, values, strict=True):
            value.copy_(saved)

    def forward():
        candidates, scores, _, base, probabilities, target = operation.observe()
        proposals.copy_(candidates)
        logits.copy_(scores)
        draft_logits.copy_(base)
        draft_probabilities.copy_(probabilities)
        target_probabilities.copy_(target)

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
    torch.cuda.synchronize()
    execution = PreparedExecution(device=0, max_active=1, graphs={"speculate": graph},
        signatures={"speculate": GraphSignature("speculative", 1, 8, tuple(inputs.buffers()),
                                               draft_tokens=7, speculation_enabled=True)},
        buffers={**inputs.buffers(), **completion.buffers(), **delivery.buffers(),
                 "frame_index": frame, "submission_sequence": sequence},
        resources=(operation, compiled, pool, requests, history, rng, proposals, logits),
        export_events=delivery.export_events())
    manifest = execution.manifest()
    manifest["engram"] = resident.bind(manifest, execution.export_events)
    worker = flare_heliopause.GraphExecutor((execution, resident), manifest)
    try:
        for iteration in range(9):
            finished = int(requests.status[0]) == FINISHED
            remaining = int(requests.max_new_tokens[0] - requests.generated_tokens[0])
            inputs.query_lengths.fill_(max(1, min(8, remaining)))
            before = tuple(value.clone() for value in state)
            before_generated = int(requests.generated_tokens[0])
            before_rng = int(rng.target_offset[0])
            before_proposal_rng = int(proposal_offset[0])
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            fence = worker.submit(0, [("logprob_requested", 0, bytes(4)),
                                      ("logprob_token_id_counts", 0, bytes(4))])
            result, deadline = None, time.monotonic() + 60
            while result is None and time.monotonic() < deadline:
                result = worker.poll()
                if result is None:
                    time.sleep(.001)
            assert result is not None and result.fence == fence
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            actual_logits, actual_proposals = logits.clone(), proposals.clone()
            actual_draft_logits, actual_probabilities = draft_logits.clone(), draft_probabilities.clone()
            actual_target_probabilities = target_probabilities.clone()
            after = tuple(value.clone() for value in state)
            restore(before)
            forward()
            torch.cuda.synchronize()
            mismatches = [name for name, value, saved in zip(state_names, state, after, strict=True)
                          if not torch.equal(value, saved)]
            if mismatches:
                print(f"speculative iteration={iteration}, mismatched state={mismatches}", flush=True)
                print(f"native candidates={actual_proposals.tolist()}, direct candidates={proposals.tolist()}", flush=True)
            if not finished:
                extent = min(8, remaining)
                for name, value, saved in (("draft logits", draft_logits, actual_draft_logits),
                                          ("proposal probabilities", draft_probabilities, actual_probabilities)):
                    if not torch.equal(value, saved):
                        print(f"{name}: changed per position={(value != saved).sum(-1).tolist()}, "
                              f"maximum difference={(value - saved).abs().amax(-1).tolist()}", flush=True)
                torch.testing.assert_close(draft_logits, actual_draft_logits, rtol=0, atol=0)
                torch.testing.assert_close(draft_probabilities, actual_probabilities, rtol=0, atol=0)
                torch.testing.assert_close(target_probabilities, actual_target_probabilities, rtol=0, atol=0)
                torch.testing.assert_close(proposals, actual_proposals, rtol=0, atol=0)
                torch.testing.assert_close(logits[:, :extent], actual_logits[:, :extent], rtol=0, atol=0)
            for name, value, saved in zip(state_names, state, after, strict=True):
                # The shared sampler's survival estimate uses FP32 division and
                # serial products. Its existing compiled/direct bound is 1e-6;
                # model state, decisions, counters and transfers remain exact.
                tolerance = 1e-6 if name == "completion_speculative_survival" else 0
                if tolerance:
                    assert torch.equal(value < 0, saved < 0)
                    assert torch.equal(value[value < 0], saved[saved < 0])
                torch.testing.assert_close(value, saved, rtol=tolerance, atol=0,
                    msg=lambda message: f"iteration={iteration}, {name}: {message}")
            if not finished:
                indices = actual_proposals[:, 1:, None]
                q = actual_probabilities.gather(-1, indices).squeeze(-1).double()
                p = actual_target_probabilities[:, :-1].gather(-1, indices).squeeze(-1).double()
                expected_survival = torch.where(q > 0, (p / q).clamp(max=1), (p > 0).double()).cumprod(-1)
                for estimates in (completion.speculative_survival,
                                  after[state_names.index("completion_speculative_survival")]):
                    actual_survival = estimates[int(frame[0]), :, :7]
                    observed = actual_survival >= 0
                    torch.testing.assert_close(actual_survival[observed].double(), expected_survival[observed],
                                               rtol=1e-6, atol=0)
            if iteration == 0:
                from tests.components.reference_cache_checks import check_reference_cache_prefixes
                restore(before)
                check_reference_cache_prefixes(operation, actual_proposals)
                restore(after)
            record = completion.records[int(frame[0]), 0]
            assert int(record[6]) == 0
            count = int(record[8])
            if finished:
                assert count == 0
                assert int(rng.target_offset[0]) == before_rng
                assert int(proposal_offset[0]) == before_proposal_rng
                assert torch.equal(pool.backing, before[0])
                break
            assert 1 <= count <= min(8, remaining)
            assert int(record[9]) == min(7, remaining - 1)
            assert int(rng.target_offset[0]) == before_rng + operation.verification.draw_reservation
            assert int(proposal_offset[0]) == before_proposal_rng + 28
            consumed += actual_proposals[0, :count].tolist()
            emitted += record[16:16 + count].tolist()
            assert int(requests.generated_tokens[0]) == before_generated + count
            assert int(requests.consumed_tokens[0]) == len(consumed)
            assert int(record[13]) == len(consumed)
            assert int(pool.region("consumed_tokens")[1, 0]) == len(consumed)
            assert pool.region("engram_history")[1].tolist() == consumed[-3:]
            assert torch.equal(pool.backing[0], before[0][0])
        else:
            pytest.fail("speculative generation did not reach its output limit and drain")
        assert int(requests.generated_tokens[0]) == 12
        expected = torch.bincount(torch.tensor([token for token in prompt + emitted if token < model.cfg.vocab_size],
                                               device="cuda"), minlength=model.cfg.vocab_size)
        torch.testing.assert_close(requests.histogram[0].long(), expected, rtol=0, atol=0)
    finally:
        worker.close()
        torch.cuda.synchronize()
        graph.reset()
