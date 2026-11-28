# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU preparation checks for the isolated DeepInfer integration dependency."""

import subprocess
import sys
import os

import pytest
import torch

pytest.importorskip("deepinfer")


def test_local_reference_geometry_does_not_import_deepep():
    # A fresh interpreter prevents an earlier test's imports from concealing
    # an accidental dependency on distributed initialization.
    subprocess.run([sys.executable, "-c", """
import importlib.abc
import sys

class RejectDeepEP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'deep_ep' or fullname.startswith('deep_ep.'):
            raise AssertionError('single-device preparation imported DeepEP')

sys.meta_path.insert(0, RejectDeepEP())
import torch
from deepinfer.model_executor.parallel_utils.parallel_state import ParallelProcessGroup
from deepinfer.model_executor.parallel_utils.userbuffer import _UserBufferManager
from deepinfer.model_executor.models.deepseek import DeepseekModel
from deepinfer.model_executor.models.deepseek_vl import DeepseekVLForCausalLM
groups = ParallelProcessGroup()
groups.initialize_local()
for group in (groups.expert_parallel_group(), groups.context_parallel_group(),
              groups.engram_parallel_group(), groups.data_parallel_group(), groups.world_group()):
    assert group.rank() == 0 and group.size() == 1
    assert group.barrier() is None
    assert groups.is_within_nvl_domain(group)
buffers = _UserBufferManager()
assert buffers.allocate_cp(groups.context_parallel_group(), 8192, 512) is None
with buffers.cp_session():
    pass
buffers.close()
assert not torch.distributed.is_initialized()
assert not torch.cuda.is_initialized()
"""], check=True)


def test_local_reference_geometry_rejects_reinitialization():
    from deepinfer.model_executor.parallel_utils.parallel_state import ParallelProcessGroup

    groups = ParallelProcessGroup()
    groups.initialize_local()
    original = groups.expert_parallel_group()
    with pytest.raises(RuntimeError, match="uninitialized process groups"):
        groups.initialize_local()
    assert groups.expert_parallel_group() is original


def test_vision_meta_construction_and_rotary_materialization():
    from deepinfer.model_executor.models.vision_encoder.vit import (
        MAX_GRID_SIZE, VisionRotaryEmbedding, VisionTransformer,
    )

    with torch.device("meta"):
        model = VisionTransformer(recomputing_every_layer=1)
    assert len(model.blocks) == 32
    assert all(param.is_meta for param in model.parameters())
    assert model.rope_table.is_meta
    assert model.rotary_pos_emb.inv_freq.is_meta

    model.materialize_rotary("cpu")
    rotary = VisionRotaryEmbedding(model.rotary_pos_emb.dim, model.rotary_pos_emb.theta)
    freqs = rotary(MAX_GRID_SIZE)
    expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    assert torch.equal(model.rotary_pos_emb.inv_freq, rotary.inv_freq)
    assert torch.equal(model.rope_table, expected)
    # Only derived constants materialize; checkpoint parameters remain unfilled.
    assert all(param.is_meta for param in model.parameters())
    assert "rope_table" not in model.state_dict()
    assert not torch.cuda.is_initialized()


def test_prepared_mlp_binding_preserves_meta_checkpoint_storage():
    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires the CSF checkpoint metadata for target and draft construction")
    subprocess.run([sys.executable, "-c", """
from pathlib import Path
import sys
import torch
from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint
from deepinfer.model_executor.parallel_utils.parallel_state import PG
from deepinfer.model_executor.models.v4_modules.builder import V41Builder
from deepinfer.model_executor.models.decoder_layer import _build_mlp_and_layernorms

PG.initialize_local()
cfg = ReferenceCheckpoint(Path(sys.argv[1])).model_config()
with torch.device('meta'):
    model = V41Builder(cfg).build_model(vision_device='meta')
    prepared = torch.nn.Linear(cfg.gpt.hidden_size, cfg.gpt.hidden_size, bias=False)
layers = model.module.module
target = layers[1]
draft = layers[cfg.gpt.num_hidden_layers + 2].block
for index, layer in ((0, target), (cfg.gpt.num_hidden_layers, draft)):
    before = tuple(layer.experts.parameters())
    mlp, attention_norm, expert_norm = _build_mlp_and_layernorms(layer, index, cfg.gpt, prepared)
    assert mlp is prepared
    assert tuple(layer.experts.parameters()) == before
    assert all(value.is_meta for value in before)
    assert attention_norm.weight.is_meta and expert_norm.weight.is_meta
    assert mlp(torch.empty(2, cfg.gpt.hidden_size, device='meta')).shape == (2, cfg.gpt.hidden_size)
assert not torch.cuda.is_initialized()
assert not torch.distributed.is_initialized()
assert 'deep_ep' not in sys.modules
""", snapshot], check=True)


def test_explicit_reference_model_metadata_avoids_engine_configuration(monkeypatch):
    from pathlib import Path
    from deepinfer.model_executor.execution_settings import ExecutionSettings
    from deepinfer.model_executor import model_metadata_utils
    from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint

    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires the CSF checkpoint configuration")
    config = ReferenceCheckpoint(Path(snapshot)).model_config().gpt
    settings = ExecutionSettings(7, False, "off", 64 << 20, 0, 1.)

    class RejectEnvironment:
        def __getattr__(self, name):
            raise AssertionError(f"prepared model metadata read engine configuration: {name}")

    monkeypatch.setattr(model_metadata_utils, "envs", RejectEnvironment())
    metadata = model_metadata_utils.ModelMetadata(config, execution=settings)
    assert metadata.execution is settings
    assert metadata.engram_enabled and metadata.hc_enabled
    assert metadata.prepare_decode is model_metadata_utils.prepare_decode_v4
    assert not torch.cuda.is_initialized()
