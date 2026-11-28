# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""YaRN rotary scaling selected by the checkpoint's own text configuration."""

from dataclasses import fields, replace
import hashlib
import json
import math
import pickle
from types import SimpleNamespace

import pytest
import torch

from ignition import planning
from ignition.declaration import Checkpoint
from ignition.errors import UsageError
from hyperflare_recipes.recipes.qwen38.attention import RotaryTables
from hyperflare_recipes.recipes.qwen38.bootstrap import _identity, _validate_rotary_context
from hyperflare_recipes.recipes.qwen38.checkpoint import CheckpointError, serving_context
from hyperflare_recipes.recipes.qwen38.recipe import Qwen38Checkpoint, RECIPE, inspect_replacement

NATIVE = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
          "rope_theta": 10000000, "rope_type": "default"}
# The block the Qwen3.8 Flash Next model card gives for a 1M-token context.
YARN = {**NATIVE, "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}


def _text(rope):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(head_dim=256, hidden_size=2560, num_attention_heads=24,
                              max_position_embeddings=262144, rope_parameters=dict(rope))


def test_yarn_tables_match_the_reference_rotary_embedding():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

    config = _text(YARN)
    tables = RotaryTables.allocate(config, 1048576, torch.device("cpu"))
    positions = torch.tensor([[0, 0, 0], [1, 2, 3], [262143, 262144, 262145], [1048575, 524288, 999999]])
    expected = Qwen4ExpTextRotaryEmbedding(config)(torch.empty(1, 4, 256, dtype=torch.bfloat16), positions.T[:, None])
    for got, want in zip(tables.gather(positions), expected):
        torch.testing.assert_close(got, want[0], atol=0, rtol=0)


def test_yarn_frequencies_and_attention_factor_are_the_published_values():
    tables = RotaryTables.allocate(_text(YARN), 2, torch.device("cpu"))
    factor = 0.1 * math.log(4.0) + 1.0
    assert round(factor, 4) == 1.1386
    torch.testing.assert_close(tables.cos[0], torch.full((32,), factor), atol=0, rtol=0)
    native = 1.0 / (10000000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    scaled = torch.atan2(tables.sin[1], tables.cos[1])
    # Rotations 32 and 1 of the 262,144-token range bound the ramp at columns 14 and 22.
    torch.testing.assert_close(scaled[:15], native[:15], atol=0, rtol=1e-6)
    torch.testing.assert_close(scaled[22:], native[22:] / 4.0, atol=0, rtol=1e-6)
    assert ((scaled[15:22] < native[15:22]) & (scaled[15:22] > native[15:22] / 4.0)).all()


def test_native_tables_are_unscaled():
    tables = RotaryTables.allocate(_text(NATIVE), 4096, torch.device("cpu"))
    inverse = 1.0 / (10000000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    frequency = torch.arange(4096, dtype=torch.float32)[:, None] * inverse
    assert torch.equal(tables.cos, torch.cos(frequency)) and torch.equal(tables.sin, torch.sin(frequency))
    # A configuration without rope_type is a default configuration.
    untyped = SimpleNamespace(head_dim=256, rope_parameters={k: v for k, v in NATIVE.items() if k != "rope_type"})
    assert torch.equal(RotaryTables.allocate(untyped, 4096, torch.device("cpu")).cos, tables.cos)


def test_unimplemented_rope_type_is_refused_by_the_tables():
    with pytest.raises(ValueError, match="longrope"):
        RotaryTables.allocate(SimpleNamespace(head_dim=256, rope_parameters={**NATIVE, "rope_type": "longrope"}),
                              64, torch.device("cpu"))


@pytest.mark.parametrize("rope,context", [
    (NATIVE, 262144), (YARN, 1048576), ({**YARN, "factor": 2.0}, 524288), ({**YARN, "factor": 2}, 524288),
    ({**YARN, "attention_factor": 1.0, "beta_fast": 32, "beta_slow": 1}, 1048576),
    ({**YARN, "mscale": 1.0, "mscale_all_dim": 0.0, "truncate": True}, 1048576),
])
def test_serving_context_follows_the_rope_configuration(rope, context):
    assert serving_context({"rope_parameters": rope}) == context


@pytest.mark.parametrize("rope,field", [
    ({**YARN, "rope_type": "longrope"}, "rope_type"),
    ({**YARN, "factor": 1000}, "factor"),
    ({**YARN, "factor": 8.0}, "factor"),
    ({**YARN, "factor": 1.0}, "factor"),
    ({**YARN, "factor": 2.5001}, "factor"),
    ({**YARN, "factor": float("inf")}, "factor"),
    ({**YARN, "factor": float("nan")}, "factor"),
    ({**YARN, "factor": "4"}, "factor"),
    ({key: value for key, value in YARN.items() if key != "factor"}, "factor"),
    ({**YARN, "original_max_position_embeddings": 131072}, "original_max_position_embeddings"),
    ({**YARN, "short_factor": [1.0]}, "short_factor"),
    ({**NATIVE, "factor": 4.0}, "factor"),
    ({**YARN, "rope_theta": 10000}, "rope_theta"),
    ({**YARN, "mrope_section": [16, 8, 8]}, "mrope_section"),
])
def test_unsupported_rope_configuration_names_its_field(rope, field):
    with pytest.raises(CheckpointError, match=rf"rope_parameters\.{field}"):
        serving_context({"rope_parameters": rope})


def test_explicit_attention_factor_reaches_the_tables():
    tables = RotaryTables.allocate(_text({**YARN, "attention_factor": 1.0}), 2, torch.device("cpu"))
    torch.testing.assert_close(tables.cos[0], torch.ones(32), atol=0, rtol=0)


def _directory(tmp_path, rope):
    tmp_path.joinpath("config.json").write_text(json.dumps({"text_config": {"rope_parameters": rope}}))
    return tmp_path


def test_checkpoint_directory_configuration_sets_the_context(tmp_path):
    native = RECIPE.checkpoints[0]
    assert native.context is None and not hasattr(native, "rope_parameters")
    assert inspect_replacement(_directory(tmp_path, NATIVE), native) is None
    scaled = inspect_replacement(_directory(tmp_path, YARN), native)
    assert isinstance(scaled, Qwen38Checkpoint)
    assert scaled.context == 1048576 and dict(scaled.rope_parameters) == YARN
    assert Checkpoint(**{f.name: (None if f.name == "context" else getattr(scaled, f.name))
                         for f in fields(Checkpoint)}) == native
    assert pickle.loads(pickle.dumps(scaled)) == scaled
    with pytest.raises(UsageError, match=r"rope_parameters\.factor"):
        inspect_replacement(_directory(tmp_path, {**YARN, "factor": 1000}), native)


def _gpu(capability):
    return planning.Gpu(0, "GPU", "GPU-0", 192 << 30, 188 << 30, capability)


def test_scaled_context_is_refused_on_sm103(tmp_path, monkeypatch):
    native = RECIPE.checkpoints[0]
    scaled = inspect_replacement(_directory(tmp_path, YARN), native)
    parameters = replace(RECIPE.parameters, max_concurrency=1)
    plan = planning.plan(RECIPE, scaled, (_gpu((12, 0)),), selected=(0,), parameters=parameters)
    assert plan.declared_context == 1048576 and plan.to_dict()["rope_parameters"] == YARN
    with pytest.raises(UsageError, match="a rope-scaled context is not measured on SM103"):
        planning.plan(RECIPE, scaled, (_gpu((10, 3)),), selected=(0,), parameters=parameters)
    native_plan = planning.plan(RECIPE, native, (_gpu((10, 3)),), selected=(0,), parameters=parameters)
    assert native_plan.declared_context == 262144 and "rope_parameters" not in native_plan.to_dict()

    # Direct preparation also enforces ceiling and SM103 scaled-context rejection.
    monkeypatch.setattr("hyperflare_recipes.components.sm103.is_sm103", lambda _device: True)
    _validate_rotary_context(_text(NATIVE), 262144, torch.device("cpu"))
    with pytest.raises(ValueError, match="exceeds the 262,144 tokens"):
        _validate_rotary_context(_text(NATIVE), 524288, torch.device("cpu"))
    with pytest.raises(ValueError, match="a rope-scaled context is not measured on SM103"):
        _validate_rotary_context(_text(YARN), 1048576, torch.device("cpu"))
    with pytest.raises(ValueError, match="a rope-scaled context is not measured on SM103"):
        _validate_rotary_context(_text(YARN), 262144, torch.device("cpu"))


def test_scaled_and_native_configurations_have_distinct_identities():
    def identity(rope):
        raw = json.dumps({"text_config": {"rope_parameters": rope}}).encode()
        return _identity(SimpleNamespace(headers_sha256="headers",
                                         metadata_sha256={"config.json": hashlib.sha256(raw).hexdigest()}))

    identities = {identity(NATIVE), identity(YARN), identity({**YARN, "factor": 2.0})}
    assert len(identities) == 3 and identity(NATIVE) == identity(dict(NATIVE))
