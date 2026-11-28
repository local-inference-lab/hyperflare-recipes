# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay sequential full-vocabulary proposals with explicit RNG and Markov scores."""

import os

import pytest
import torch

from ignition.components.flashinfer_sampling import sample_probabilities
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from hyperflare_recipes.components.speculation import DRAFT_RNG_DOMAIN, PreparedSpeculativeDistributions
from ignition.preparation.compilation import compile_forward


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned SM103 GPU")


@pytest.mark.parametrize("markov", [False, True])
@torch.no_grad()
def test_reference_sequential_proposal_replay(markov):
    from deepinfer.model_executor.layers.linear import ReplicatedLinear
    from deepinfer.model_executor.layers.tilelang_ops.gather import gather

    vocabulary, rank, width = 8192, 128, 7
    generator = torch.Generator(device="cuda").manual_seed(187)
    state = DeviceRequestState.allocate("cuda", vocab_size=vocabulary, capacity=1)
    state.admit(0, RequestAdmission(generation=1, seed=187, max_new_tokens=32,
                temperature=1., top_k=0, top_p=1.), [1, 2, 3], consumed_tokens=3)
    state.pending_token[0] = 4
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    generations = torch.ones(1, dtype=torch.int64, device="cuda")
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    base = torch.randn((1, width, vocabulary), device="cuda", generator=generator)
    embedding = torch.randn((vocabulary, rank), dtype=torch.bfloat16, device="cuda", generator=generator)
    weight = torch.randn(embedding.shape, dtype=embedding.dtype, device="cuda", generator=generator) * .01
    projection = ReplicatedLinear(rank, vocabulary, use_fp32_output=True,
        weight=torch.nn.Parameter(weight, requires_grad=False))
    template = torch.zeros((1, width), dtype=torch.int64, device="cuda")
    samplers = tuple(PreparedSpeculativeDistributions(base[:, :1].contiguous(),
        state, slots, generations, live, template, prefix_offset=index) for index in range(width))
    for sampler in samplers:
        sampler.prepare()
        sampler.functional()
    offset = torch.zeros(1, dtype=torch.int64, device="cuda")
    output = torch.empty_like(template)
    probabilities = torch.empty_like(base)
    feedback = torch.empty((width, rank), dtype=torch.bfloat16, device="cuda")
    projected = torch.empty_like(base)

    def forward():
        prefix = torch.zeros_like(template)
        prior = state.pending_token.long()
        seed = state.seed.view(torch.int64) ^ DRAFT_RNG_DOMAIN
        retained = []
        for index, sampler in enumerate(samplers):
            scores = base[:, index].clone()
            if markov:
                embedded = gather(embedding, prior)
                feedback[index].copy_(embedded[0])
                scores = projection(embedded, out_acc=scores)
            projected[:, index].copy_(scores)
            scratch = {name: torch.empty(shape, dtype=dtype, device="cuda")
                       for name, (shape, dtype) in sampler.layout.items()}
            sampler.run(scratch, scores[:, None].contiguous(), candidates=prefix)
            token, _ = sample_probabilities(scratch["probabilities"][0], seed,
                                            offset + 4 * index, return_valid=True)
            prior = token.long()
            prefix[:, index].copy_(prior)
            retained.append(scratch["probabilities"][:, 0])
        output.copy_(prefix)
        probabilities.copy_(torch.stack(retained, 1))

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for step in range(4):
            offset.fill_(step * 28)
            forward()
            expected = output.clone(), probabilities.clone(), feedback.clone(), projected.clone()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            if markov:
                torch.testing.assert_close(feedback, expected[2], rtol=0, atol=0)
            torch.testing.assert_close(projected, expected[3], rtol=0, atol=0)
            torch.testing.assert_close(probabilities, expected[1], rtol=0, atol=0)
            torch.testing.assert_close(output, expected[0], rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()
