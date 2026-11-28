# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Inkling-2-Mini online MXFP8 attention/lm_head scopes and fused softcap+mask logits."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.cli import parse_args, preparation
from hyperflare_recipes.recipes import ResolvedModel, recipes
from hyperflare_recipes.recipes.inkling2.recipe import MINI_RECIPE


def test_online_mxfp8_scopes_plan_savings():
    from tests.cli.test_planning import derive, gpu

    model = ResolvedModel(MINI_RECIPE, MINI_RECIPE.checkpoints[0])
    base = derive(
        model=MINI_RECIPE, gpus=(gpu(0, 31.84),), selected=(0,),
        parameters=preparation.resolve_parameters(
            model, parse_args(["export", model.checkpoint.alias, "--no-mtp"], recipes=recipes())
        ),
    )
    attn = derive(
        model=MINI_RECIPE, gpus=(gpu(0, 31.84),), selected=(0,),
        parameters=preparation.resolve_parameters(
            model,
            parse_args(["export", model.checkpoint.alias, "--no-mtp", "--online-mxfp8", "--online-mxfp8-scope", "attention"], recipes=recipes()),
        ),
    )
    all_scope = derive(
        model=MINI_RECIPE, gpus=(gpu(0, 31.84),), selected=(0,),
        parameters=preparation.resolve_parameters(
            model,
            parse_args(["export", model.checkpoint.alias, "--no-mtp", "--online-mxfp8", "--online-mxfp8-scope", "all"], recipes=recipes()),
        ),
    )
    assert base.weights_bytes - attn.weights_bytes == 609_429_504
    assert base.weights_bytes - all_scope.weights_bytes == 1_043_948_032
    assert all_scope.pages > attn.pages > base.pages


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA device")
@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("tokens", [1, 8, 64])
@torch.no_grad()
def test_prepared_logits_fused_softcap_and_mxfp8(mxfp8, tokens):
    from hyperflare_recipes.components.weights import quantize_bf16_linear_as_mxfp8
    from hyperflare_recipes.recipes.inkling2.vocabulary import PreparedLogits
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.manual_seed(42 + tokens)
    hidden_dim, vocab, valid_vocab = 2304, 4096, 4000
    weight_bf16 = (torch.randn(vocab, hidden_dim, device=device) * 0.05).to(torch.bfloat16)
    weight = quantize_bf16_linear_as_mxfp8(weight_bf16) if mxfp8 else weight_bf16
    text = SimpleNamespace(hidden=hidden_dim, softcap=30.0, valid_vocabulary=valid_vocab)
    arena = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint="test_logits", workload_schema="inkling2",
        numerical_policy="default", autotune_effort="off",
    ) as session:
        session.register_scratch(arena)
        op = PreparedLogits(weight, text, rank=0, tokens=tokens, session=session,
                            name=f"logits_{mxfp8}_{tokens}", workspace=arena)
        session.tune()
        arena.release_preparation()

    x = (torch.randn(tokens, hidden_dim, device=device) * 0.5).to(torch.bfloat16)
    actual = op(x)
    ref = 30.0 * torch.tanh((x.float() @ weight_bf16.float().t()) / 30.0)
    ref[:, valid_vocab:] = float("-inf")
    assert actual.dtype == torch.float32
    assert torch.isneginf(actual[:, valid_vocab:]).all()
    cos = torch.nn.functional.cosine_similarity(
        actual[:, :valid_vocab].flatten(), ref[:, :valid_vocab].flatten(), dim=0
    ).item()
    assert cos > (0.998 if mxfp8 else 0.99999)
