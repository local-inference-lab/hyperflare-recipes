# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile the checkpoint's complete reference vision encoder and projector."""

import os
import time
from pathlib import Path

import pytest
import torch

from ignition.preparation.compilation import compile_forward
from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint
from hyperflare_recipes.recipes.deepseek41.reference_parameters import ReferenceParameters

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned SM103 GPU and checkpoint")


@pytest.fixture(scope="module")
def reference_parameters():
    pytest.importorskip("deepinfer")
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    path = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not path:
        pytest.skip("requires the DeepSeek V4.1 CSF checkpoint")
    if PG.world is None:
        PG.initialize_local()
    parameters = ReferenceParameters(ReferenceCheckpoint(Path(path)))
    parameters.load_ordinary("cuda:0")
    yield parameters
    torch.cuda.synchronize()


@pytest.fixture(scope="module")
def reference_vision(reference_parameters):
    return reference_parameters.model.vision_pipeline.aligner.eval()


@torch.no_grad()
def test_reference_vision_projection_uses_operand_device(reference_vision):
    import deep_gemm
    from torch.utils._python_dispatch import TorchDispatchMode
    from deepinfer.model_executor.models.vision_encoder.vit import _linear

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires the SM10x DeepGEMM projection path")
    weight = reference_vision.visual_encoder.blocks[0].mlp.linear1.weight
    source = torch.randn((9, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    calls = []

    class RecordOperations(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            calls.append(function)
            return function(*args, **(kwargs or {}))

    with RecordOperations():
        actual = _linear(source, weight, None)
    assert torch.ops.deepinfer.bf16_gemm_nt.default in calls
    assert torch.ops.deepinfer.vision_cublaslt.default not in calls
    expected = torch.empty_like(actual)
    deep_gemm.bf16_gemm_nt(source, weight, expected, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("height,width", [(24, 24), (25, 31)])
@torch.no_grad()
def test_complete_reference_vision_replay(reference_vision, height, width):
    from deepinfer.model_executor.layers.tilelang_ops.image_embed_ops import batch_downsample

    source = reference_vision
    tokens = height * width
    ratio = source.aligner.cfg.downsample_ratio
    features = ((height + ratio - 1) // ratio) * ((width + ratio - 1) // ratio)
    generator = torch.Generator(device="cuda").manual_seed(1348)
    images = torch.randn((tokens, 3, 14, 14), dtype=torch.bfloat16, device="cuda", generator=generator)
    positions = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"),
        torch.arange(width, device="cuda"), indexing="ij"), -1).view(-1, 2)
    crop = torch.tensor([[width, height]], dtype=torch.int64, device="cuda")
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    output = torch.empty((features, source.aligner.cfg.hidden_size), dtype=torch.bfloat16, device="cuda")

    def forward():
        encoded = source.visual_encoder(images, positions, cu, tokens)
        merged = batch_downsample(encoded, crop, ratio, source.aligner.cfg.projector_type, features)
        output.copy_(source.aligner.layers(merged))

    forward()
    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for _ in range(3):
            images.normal_(generator=generator)
            forward()
            expected = output.clone()
            assert torch.isfinite(expected).all()
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("height,width,capacity", [(15, 15, 576), (24, 24, 576), (25, 31, 1152), (93, 93, 9216)])
@torch.no_grad()
def test_reference_vision_native_publication(reference_parameters, reference_vision, height, width, capacity):
    import flare_heliopause
    from deepinfer.model_executor.layers.tilelang_ops.image_embed_ops import batch_downsample
    from ignition.components.completion import CompletionOutputs
    from hyperflare_recipes.components.deepseek_vision import (
        PrefillVisionInputs, PreparedVisionGeneration, VisionInputs, VisionState,
    )
    from hyperflare_recipes.components.paged_batch import BatchInputs
    from ignition.components.sampling import DeviceRequestState, ERROR, RUNNING
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from hyperflare_recipes.recipes.deepseek41.reference_vision import ReferenceVision

    count, features = height * width, ((height + 2) // 3) * ((width + 2) // 3)
    inputs = VisionInputs.allocate("cuda", patches=capacity)
    requests = DeviceRequestState.allocate("cuda", vocab_size=129280, capacity=4)
    text = BatchInputs.allocate(192, 8, "cuda", capacity=4)
    params = reference_parameters.model.module.module[0].params
    weights = {name: getattr(params, name) for name in ("image_start", "image_end", "image_newline")}
    state = VisionState(weights, text, PrefillVisionInputs.allocate(192, "cuda"), requests,
        features=2 * (capacity // 9), token_base=129280,
        sentinel_names=("image_start", "image_start", "image_newline", "image_end"),
        feature_type=1, image_slots=2)
    state.feature_slot.fill_(1)
    requests.generation[2] = 7
    requests.status[2] = RUNNING
    inputs.vision_slot.fill_(2)
    inputs.vision_generation.fill_(7)
    inputs.vision_num_patches.fill_(count)
    inputs.vision_num_features.fill_(features)
    inputs.vision_cu_seqlens[1] = count
    inputs.vision_pixel_values.fill_(float("nan"))
    inputs.vision_pixel_values[:count].normal_()
    inputs.vision_position_ids.fill_(-17)
    inputs.vision_position_ids[:count] = torch.stack(torch.meshgrid(
        torch.arange(height, device="cuda"), torch.arange(width, device="cuda"), indexing="ij"), -1).view(-1, 2)
    merge = [[(row * 3 + dy) * width + col * 3 + dx
              if row * 3 + dy < height and col * 3 + dx < width else -1
              for dy in range(3) for dx in range(3)]
             for row in range((height + 2) // 3) for col in range((width + 2) // 3)]
    inputs.vision_merge_indices.fill_(-19)
    inputs.vision_merge_indices[:features] = torch.tensor(merge, device="cuda")
    completion = CompletionOutputs.allocate((2, 4, 32), device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.zeros(1, dtype=torch.int64, device="cuda")
    encoder = ReferenceVision(reference_vision, inputs, capacity=capacity)
    operation = PreparedVisionGeneration(encoder, state, completion, frame, sequence)
    compiled = compile_forward(operation)
    compiled()
    torch.cuda.synchronize()
    shared_pool = torch.cuda.graph_pool_handle()
    poison_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(poison_graph, pool=shared_pool):
        poison_blocks = [torch.empty((capacity, 16, 64), dtype=torch.bfloat16, device="cuda") for _ in range(8)]
        for block in poison_blocks:
            block.fill_(float("nan"))
        del poison_blocks
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=shared_pool):
        compiled()
    state.arena.fill_(float("nan"))
    state.generation.zero_()
    state.count.zero_()
    execution = PreparedExecution(device=0, max_active=4, graphs={"vision": graph},
        signatures={"vision": GraphSignature("vision", 1, capacity, tuple(inputs.buffers()))},
        buffers={**inputs.buffers(), **completion.buffers(), "vision_feature_slot": state.feature_slot,
                 "frame_index": frame, "submission_sequence": sequence},
        resources=(reference_parameters, state, requests, operation, compiled))
    worker = None

    def replay():
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        fence = worker.submit(0, [])
        result, deadline = None, time.monotonic() + 60
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            if result is None:
                time.sleep(.001)
        assert result is not None and result.fence == fence
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        return completion.records[int(frame[0]), 2]

    try:
        worker = flare_heliopause.GraphExecutor(execution, execution.manifest())
        for offset in (0, features):
            inputs.vision_feature_offset.fill_(offset)
            inputs.vision_pixel_values[:count].normal_()
            pixels = torch.zeros((capacity, 3, 14, 14), dtype=torch.bfloat16, device="cuda")
            pixels[:count] = inputs.vision_pixel_values[:count].view(count, 3, 14, 14)
            positions = inputs.vision_position_ids.clamp_min(0)
            encoded = reference_vision.visual_encoder(pixels, positions, inputs.vision_cu_seqlens, capacity)
            crop = torch.tensor([[width, height]], dtype=torch.int64, device="cuda")
            merged = batch_downsample(encoded, crop, 3, "downsample_mlp_gelu", features)
            padded = torch.zeros((capacity // 9, merged.shape[1]), dtype=merged.dtype, device="cuda")
            padded[:features] = merged
            expected = reference_vision.aligner.layers(padded)[:features]
            before = state.arena.clone()
            poison_graph.replay()
            record = replay()
            torch.testing.assert_close(state.arena[1, offset:offset + features], expected, rtol=0, atol=0)
            torch.testing.assert_close(state.arena[0], before[0], rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(state.arena[1, :offset], before[1, :offset], rtol=0, atol=0, equal_nan=True)
            assert int(state.count[2]) == offset + features
            assert int(state.generation[2]) == 7 and int(record[6]) == 0
        before = state.arena.clone()
        inputs.vision_generation.fill_(6)
        record = replay()
        assert int(record[6]) != 0 and int(requests.status[2]) == RUNNING
        assert int(state.count[2]) == 2 * features and int(state.generation[2]) == 7
        torch.testing.assert_close(state.arena, before, rtol=0, atol=0, equal_nan=True)

        inputs.vision_generation.fill_(7)
        inputs.vision_feature_offset.zero_()
        inputs.vision_position_ids[0, 0] = encoder.position_limit
        record = replay()
        assert int(record[6]) != 0 and int(requests.status[2]) == ERROR
        assert int(state.count[2]) == 2 * features and int(state.generation[2]) == 7
        torch.testing.assert_close(state.arena, before, rtol=0, atol=0, equal_nan=True)
    finally:
        if worker is not None:
            worker.close()
        torch.cuda.synchronize()
        poison_graph.reset()
        graph.reset()
