# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and CUDA tests for opt-in Qwen3.8 target lm_head MXFP8 quantization."""

import os

import pytest
import torch

from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig


def test_preparation_config_validates_target_head_quantization():
    from hyperflare_recipes.recipes.qwen38.recipe import RECIPE

    cfg_none = PreparationConfig()
    assert cfg_none.target_head_quantization is None
    cfg_mxfp8 = PreparationConfig(target_head_quantization="mxfp8")
    assert cfg_mxfp8.target_head_quantization == "mxfp8"
    assert RECIPE.target_head_quantizations == ("mxfp8",)
    with pytest.raises(ValueError, match="target head quantization"):
        PreparationConfig(target_head_quantization="nvfp4")


def test_non_sampling_decoder_keeps_lm_head_none_with_quantized_target_head(monkeypatch):
    from tests.recipes.test_qwen38_bootstrap import _build_stub_decoder_env

    decoder_mod, weights, scratch, state, qsa_states, _ = _build_stub_decoder_env(
        monkeypatch, tokens=512, rows=1, mode="prefill"
    )
    weights.lm_head_weight = object()
    nonterm = decoder_mod.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="prefill512", sample=False,
    )
    assert nonterm.lm_head is None


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires CUDA device and FLARE_RUN_CUDA_TESTS=1",
)
@pytest.mark.parametrize("rows", [1, 8, 32])
@torch.no_grad()
def test_mxfp8_target_head_projection_matches_bf16_and_saves_memory(rows):
    from ignition.preparation.persistent import allocated_tensor_bytes
    from ignition.preparation.scratch import ScratchArena
    from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
    from hyperflare_recipes.components.vocabulary import mxfp8_draft_head
    from hyperflare_recipes.recipes.qwen38.tuning import preparation_session

    device = torch.device("cuda:0")
    vocab, hidden = 4096, 2560
    g = torch.Generator(device=device).manual_seed(1234)
    weight = (torch.randn((vocab, hidden), generator=g, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
    x = torch.randn((rows, hidden), generator=g, device=device, dtype=torch.bfloat16).contiguous()
    # Add a realistic LM logit margin on one target token per row.
    targets = torch.arange(rows, device=device) * 97 + 13
    x = (x + 10.0 * weight[targets]).contiguous()

    before = allocated_tensor_bytes(device)
    mxfp8_weight = mxfp8_draft_head(weight)
    mxfp8_bytes = allocated_tensor_bytes(device) - before
    assert 0 < mxfp8_bytes < weight.nbytes * 0.60

    out_bf16 = torch.empty((rows, vocab), dtype=torch.bfloat16, device=device)
    out_mxfp8 = torch.empty((rows, vocab), dtype=torch.bfloat16, device=device)
    arena = ScratchArena(device)
    with preparation_session(device=device, checkpoint="test", autotune_effort="off") as session:
        p_bf16 = UnquantizedProjection(x, weight, out_bf16, session=session, name="bf16_head")
        p_mxfp8 = QuantizedProjection(
            x, mxfp8_weight, out_mxfp8, session=session, name="mxfp8_head",
            activation_mode="auto", workspace=arena,
        )
        p_bf16.functional(session=session)
        p_mxfp8.functional(session=session)
        session.register_scratch(arena)
        session.tune()
        ref = p_bf16(x).float()
        got = p_mxfp8(x).float()

    cos = torch.nn.functional.cosine_similarity(ref, got, dim=-1).min().item()
    assert cos > 0.997, f"cosine similarity too low: {cos}"
    assert torch.equal(ref.argmax(-1), got.argmax(-1)), "top-1 argmax diverged"
