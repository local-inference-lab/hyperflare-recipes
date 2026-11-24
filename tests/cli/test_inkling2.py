# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling checkpoint resolution and declared preparation capacities."""

from dataclasses import replace
import json

import pytest

from ignition.cli import _parser, preparation
from hyperflare_recipes.recipes import ResolvedModel, resolve
from hyperflare_recipes.recipes.inkling2.recipe import RECIPE, MINI_RECIPE
from tests.cli.test_planning import derive, gpu


def test_mini_directory_resolves_loop_architecture(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Inkling2LoopForConditionalGeneration"], "model_type": "inkling2_loop"}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": MINI_RECIPE.checkpoints[0].stored_bytes}}))
    model = resolve(str(tmp_path))
    assert model.recipe is MINI_RECIPE
    assert model.recipe.tensor_parallel == (1,)
    assert model.recipe.speculation.draft_tokens == 7
    assert model.recipe.max_images == 4
    assert (model.recipe.max_concurrency, model.recipe.prefill_chunk,
            model.recipe.speculation.token_capacity) == (8, 8192, 64)


def test_varloop_alias_and_repository_share_mini_recipe():
    mini = resolve("inkling-2-mini")
    varloop = resolve("tml-d24-varloop-research")
    assert varloop.recipe is MINI_RECIPE
    assert resolve(varloop.checkpoint.repository).checkpoint is varloop.checkpoint
    assert resolve(f"{varloop.checkpoint.repository}@{varloop.checkpoint.revision}").checkpoint is varloop.checkpoint
    assert MINI_RECIPE.served_name(mini.checkpoint) == "inkling-2-mini"
    assert MINI_RECIPE.served_name(varloop.checkpoint) == "TML-d24-varloop-research"


def test_small_directory_retains_its_recipe(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Inkling2ForConditionalGeneration"], "model_type": "inkling2"}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": RECIPE.checkpoints[0].stored_bytes}}))
    assert resolve(str(tmp_path)).recipe is RECIPE


def test_inkling_defaults_cover_tp4_seven_drafts_and_vision():
    assert RECIPE.tensor_parallel == (4,)
    assert (RECIPE.context, RECIPE.concurrency, RECIPE.prefill_chunk) == (1048576, 8, 4096)
    assert RECIPE.max_concurrency == 32
    assert (RECIPE.max_images, RECIPE.image_bytes) == (4, 32 << 20)
    model = ResolvedModel(RECIPE, RECIPE.checkpoints[0])
    args = _parser().parse_args(["export", model.checkpoint.alias])
    speculation = preparation.speculation(args, model)
    assert (speculation.draft_tokens, speculation.token_capacity, speculation.graph_budget) == (7, 128, 64)
    assert speculation.packed_verification
    assert speculation.widths == tuple(range(1, 9))


def test_inkling_adapter_preserves_uncompiled_untuned_qualification(monkeypatch, tmp_path):
    from hyperflare_recipes.recipes.inkling2 import bootstrap

    devices = tuple(gpu(i, 95) for i in range(4))
    plan = replace(derive(model=RECIPE, gpus=devices, selected=(0, 1, 2, 3), concurrency=8),
                   autotune_effort="off")
    captured = []
    configuration = bootstrap.PreparationConfig

    class ConfigurationCaptured(Exception):
        pass

    def capture_config(**kwargs):
        captured.append(configuration(**kwargs))
        raise ConfigurationCaptured

    monkeypatch.setattr(bootstrap, "PreparationConfig", capture_config)
    with pytest.raises(ConfigurationCaptured):
        RECIPE.prepare(tmp_path, plan, compile_forwards=False, kernel_cache=None, progress=None)
    config = captured.pop()
    assert config.devices == (0, 1, 2, 3)
    assert config.autotune_effort == "off"
    assert not config.compile_decoder
    assert config.context == 1048576
    assert config.max_active == 8
    assert config.drafting and config.vision
    assert config.speculative_widths == tuple(range(1, 9))
    assert config.speculative_token_capacity == 128
    assert config.speculative_graph_budget == 64


@pytest.mark.parametrize("draft_head", [None, "nvfp4", "mxfp8"])
@pytest.mark.parametrize("markov_head", [None, "nvfp4"])
def test_mini_proposal_head_quantization_plans_and_prepares(monkeypatch, tmp_path, draft_head, markov_head):
    from hyperflare_recipes.recipes.inkling2 import bootstrap

    model = ResolvedModel(MINI_RECIPE, MINI_RECIPE.checkpoints[0])
    argv = ["export", model.checkpoint.alias]
    if draft_head is not None:
        argv += ["--draft-head-quantization", draft_head]
    if markov_head is not None:
        argv += ["--markov-head-quantization", markov_head]
    args = _parser().parse_args(argv)
    spec = preparation.speculation(args, model)
    plan = derive(model=MINI_RECIPE, gpus=(gpu(0, 95),), selected=(0,), concurrency=8, speculation=spec)
    assert plan.speculation.draft_head_quantization == draft_head
    assert plan.speculation.markov_head_quantization == markov_head

    captured = []
    configuration = bootstrap.PreparationConfig

    class ConfigurationCaptured(Exception):
        pass

    def capture_config(**kwargs):
        captured.append(configuration(**kwargs))
        raise ConfigurationCaptured

    monkeypatch.setattr(bootstrap, "PreparationConfig", capture_config)
    with pytest.raises(ConfigurationCaptured):
        MINI_RECIPE.prepare(tmp_path, plan, compile_forwards=False, kernel_cache=None, progress=None)
    config = captured.pop()
    assert config.draft_head_quantization == draft_head
    assert config.markov_head_quantization == markov_head


def test_mini_rejects_undeclared_proposal_head_quantization():
    from ignition.planning import UsageError
    from hyperflare_recipes.recipes.inkling2.bootstrap import PreparationConfig

    bad_markov = replace(MINI_RECIPE.speculation, markov_head_quantization="fp8")
    with pytest.raises(UsageError, match="markov-head-quantization"):
        derive(model=MINI_RECIPE, gpus=(gpu(0, 95),), selected=(0,), concurrency=8, speculation=bad_markov)
    small_draft = replace(RECIPE.speculation, draft_head_quantization="nvfp4")
    with pytest.raises(UsageError, match="draft-head-quantization"):
        derive(model=RECIPE, gpus=tuple(gpu(i, 95) for i in range(4)), selected=(0, 1, 2, 3),
               concurrency=8, speculation=small_draft)
    with pytest.raises(ValueError, match="requires drafting"):
        PreparationConfig(devices=(0,), drafting=False, draft_head_quantization="nvfp4")
    with pytest.raises(ValueError, match="requires drafting"):
        PreparationConfig(devices=(0,), drafting=False, markov_head_quantization="nvfp4")
