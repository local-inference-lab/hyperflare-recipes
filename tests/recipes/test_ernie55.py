# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pytest
import torch

from hyperflare_recipes.recipes import resolve
from hyperflare_recipes.recipes.ernie55.recipe import RECIPE


def test_ernie55_recipe_registration_and_metadata() -> None:
    resolved = resolve("ernie-5.5-flash")
    assert resolved.recipe.name == "ernie55"
    assert resolved.recipe is RECIPE
    assert any(c.architecture == "Ernie5_5_MoeForCausalLM" for c in RECIPE.checkpoints)
    assert RECIPE.served_model_name == "ERNIE-5.5-Flash-Preview"
    assert RECIPE.prefill_chunk == 4096
    assert RECIPE.max_draft_tokens == 7


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA opt-in",
)
def test_multimax_inplace_matches_reference() -> None:
    from hyperflare_recipes.components.markov_sampling import apply_multimax_inplace

    device = torch.device("cuda", 0)
    ranges = (-2.0, 2.0, -4.0, 4.0)
    ts = (-0.25, 0.35, -0.05, 0.02)
    x = torch.linspace(-8.0, 8.0, 4096, device=device, dtype=torch.float32).reshape(2, 2048)
    x[0, 0] = float("-inf")

    expected = x.clone()
    r0, r1, r2, r3 = ranges
    t0, t1, t2, t3 = ts
    a = (r0 - expected).clamp(min=0.0)
    b = (expected - r1).clamp(min=0.0)
    c = (r2 - expected).clamp(min=0.0)
    d = (expected - r3).clamp(min=0.0)
    expected = expected + t0 * a + t1 * b + t2 * (c * c) + t3 * (d * d)
    expected[0, 0] = float("-inf")

    actual = x.clone()
    apply_multimax_inplace(actual, (ranges, ts))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA opt-in",
)
def test_ernie55_route_matches_reference_and_pads_safely() -> None:
    from hyperflare_recipes.recipes.ernie55.decoder import ernie55_route

    device = torch.device("cuda", 0)
    torch.manual_seed(17)
    T, E, K = 8, 512, 10
    logits = torch.randn((T, 2 * E), device=device, dtype=torch.bfloat16)
    bias = torch.randn((E,), device=device, dtype=torch.float32) * 0.1
    scale = torch.linspace(2.5, 3.5, E, device=device, dtype=torch.float32)
    positions = torch.tensor([0, 1, 2, 3, 4, 5, -1, -1], device=device, dtype=torch.int32)

    ids, weights = ernie55_route(logits, bias, scale, positions, top_k=K, renormalize=True)
    assert weights.shape == (T, K)
    assert ids.shape == (T, K)
    # Padded rows must have ids == -1 and weights == 0.0
    assert (ids[6:] == -1).all()
    assert (weights[6:] == 0.0).all()
    assert ((ids[:6] >= 0) & (ids[:6] < E)).all()

    scores = torch.sigmoid(logits[:6, :E].float()) + torch.sigmoid(logits[:6, E:].float())
    biased = scores + bias.unsqueeze(0)
    ref_ids = torch.topk(biased, k=K, dim=-1, sorted=False).indices
    ref_w = scores.gather(-1, ref_ids)
    ref_w = ref_w / (ref_w.sum(dim=-1, keepdim=True) + 1e-20)
    ref_w = ref_w * scale[ref_ids]

    for row in range(6):
        got_order = torch.argsort(ids[row])
        ref_order = torch.argsort(ref_ids[row])
        assert torch.equal(ids[row][got_order].long(), ref_ids[row][ref_order].long())
        torch.testing.assert_close(
            weights[row][got_order], ref_w[row][ref_order], rtol=1e-4, atol=1e-4
        )
