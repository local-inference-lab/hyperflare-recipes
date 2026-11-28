# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain reference confidence arithmetic through compilation and prefix allocation."""

import os

import pytest
import torch

from hyperflare_recipes.components.causal_prefix import PreparedCausalPrefixAllocation
from ignition.preparation.compilation import compile_forward
from tests.components.causal_prefix_oracle import allocate

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned GPU and DeepInfer")


@pytest.mark.parametrize("markov", [False, True])
@torch.no_grad()
def test_reference_confidence_and_budget_replay(markov):
    from deepinfer.model_executor.layers.tilelang_ops import ops
    from deepinfer.model_executor.layers.tilelang_ops.dflash import confidence_head_fused

    rows, depth, hidden_width, rank = 4, 7, 5120, 128 if markov else 0
    generator = torch.Generator(device="cuda").manual_seed(541)
    hidden = torch.randn((rows, depth, hidden_width), device="cuda", dtype=torch.bfloat16, generator=generator)
    embedded = torch.randn((rows, depth, rank), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(hidden_width + rank, device="cuda", dtype=torch.bfloat16, generator=generator) * .01
    confidence = torch.empty((rows, depth), dtype=torch.float32, device="cuda")
    expected = torch.empty_like(confidence)
    slots = torch.tensor([3, 0, 2, 1], dtype=torch.int64, device="cuda")
    submitted = torch.ones(rows, dtype=torch.bool, device="cuda")
    valid = torch.ones_like(submitted)
    limits = torch.tensor([8, 3, 5, 8], dtype=torch.int32, device="cuda")
    lengths = torch.empty_like(limits)
    admitted = torch.empty_like(valid)
    allocator = PreparedCausalPrefixAllocation(confidence, slots, submitted, valid, limits, lengths, admitted,
        token_capacity=14, request_capacity=4).functional()
    temperatures = (.5, 1., 2., .5, 1., 2., 1.)

    def forward():
        for index, temperature in enumerate(temperatures):
            confidence_head_fused(hidden[:, index], weight, temperature,
                embedded[:, index].contiguous() if markov else None, confidence[:, index])
        admitted.copy_(allocator(confidence, slots, submitted, valid, limits, lengths))

    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    kernel = ops.confidence_head_fused_tl(n_embed=hidden_width, markov_rank=rank, has_markov_emb=markov)
    try:
        for live in ((True, True, True, True), (False, True, True, False), (False,) * 4):
            hidden.normal_(generator=generator)
            embedded.normal_(generator=generator)
            submitted.copy_(torch.tensor(live, device="cuda"))
            for index, temperature in enumerate(temperatures):
                kernel(hidden[:, index], embedded[:, index].contiguous() if markov else None,
                       weight, expected[:, index], temperature)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(confidence, expected, rtol=0, atol=0)
            wanted_lengths, wanted_valid = allocate(expected.tolist(), slots.tolist(), list(live),
                valid.tolist(), limits.tolist(), 14, 4)
            assert lengths.tolist() == wanted_lengths
            assert admitted.tolist() == wanted_valid
    finally:
        torch.cuda.synchronize()
        graph.reset()
