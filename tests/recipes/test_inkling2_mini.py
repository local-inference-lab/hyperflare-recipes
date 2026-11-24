# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mini checkpoint geometry, independent loop histories, and expert replay."""

from dataclasses import fields
import json
import os
from pathlib import Path

import pytest
import torch

from hyperflare_recipes.recipes.inkling2.config import ModelConfig


class ExpertForward:
    def __init__(self, operation, source, ids, routes):
        self.operation, self.source, self.ids, self.routes = operation, source, ids, routes
        self.output = torch.empty_like(source)

    def __call__(self):
        self.output.copy_(self.operation(self.source, self.ids, self.routes))


_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_config_dir(tmp_path):
    (tmp_path / "config.json").write_text((_FIXTURES / "inkling2_mini_config.json").read_text())
    (tmp_path / "hf_quant_config.json").write_text((_FIXTURES / "inkling2_mini_quant_config.json").read_text())
    return tmp_path


@pytest.fixture
def snapshot():
    path = os.environ.get("FLARE_INKLING2_MINI_SNAPSHOT")
    if path is None:
        pytest.skip("requires FLARE_INKLING2_MINI_SNAPSHOT")
    return Path(path)


def test_mini_visits_and_cache_storage(mini_config_dir):
    from hyperflare_recipes.recipes.inkling2.pool import continuation_layout, RankGeometry, page_specs
    config = ModelConfig.load(mini_config_dir)
    t = config.text
    assert len(t.visits) == 48
    assert t.visits[6:18] == tuple((i, 0) for i in range(6, 18))
    assert t.visits[18:30] == tuple((i, 1) for i in range(6, 18))
    assert t.visits[30:42] == tuple((i, 2) for i in range(6, 18))
    assert t.visits[42:] == tuple((i, 0) for i in range(18, 24))
    cache = t.cache_config
    geometry = RankGeometry.of(cache, 1)
    layout = continuation_layout(cache, geometry)
    histories = [layout.region(f"layer{i}.attention_convolution") for i in (6, 18, 30)]
    assert len({r.offset for r in histories}) == 3
    assert cache.global_layers == (5, 11, 17, 23, 29, 35, 41, 47)
    assert sum(s[0] for _, s in page_specs(cache, geometry)) == 2097152
    assert config.draft.taps == (5, 23, 41, 48)


def test_public_release_configuration(mini_config_dir):
    raw = json.loads((mini_config_dir / "config.json").read_text())
    precision = json.loads((mini_config_dir / "hf_quant_config.json").read_text())
    raw.update(architectures=["Inkling2LoopForConditionalGeneration"], dtype="bfloat16",
               quantization_config=precision)
    raw["text_config"]["model_type"] = "inkling2_loop_text"
    raw["vision_config"].update(model_type="inkling2_vision", text_hidden_size=2304)
    for flag in ("pre_norm_pre_affine", "final_norm_pre_affine"):
        raw["text_config"].pop(flag, None)
    model = ModelConfig.from_dict(raw)
    assert model.text.hidden == 2304 and len(model.text.visits) == 48
    assert model.quantization == precision


def test_public_precision_summary_loads_layer_map(mini_config_dir, tmp_path):
    target = tmp_path / "summary"
    target.mkdir()
    raw = json.loads((mini_config_dir / "config.json").read_text())
    raw["quantization_config"] = dict(quant_method="modelopt", quant_algo="NVFP4")
    (target / "config.json").write_text(json.dumps(raw))
    precision = (mini_config_dir / "hf_quant_config.json").read_text()
    (target / "hf_quant_config.json").write_text(precision)
    assert ModelConfig.load(target).quantization == json.loads(precision)


def test_varloop_release_configuration(mini_config_dir):
    # Adapt the reduced local Mini fixture to match the public varloop checkpoint's schema differences
    # (no top-level dtype, explicit sub-config model_type/text_hidden_size, and loop_config.is_varloop=True).
    raw = json.loads((mini_config_dir / "config.json").read_text())
    precision = json.loads((mini_config_dir / "hf_quant_config.json").read_text())
    raw.pop("dtype", None)
    raw.update(architectures=["Inkling2LoopForConditionalGeneration"], quantization_config=precision)
    raw["text_config"]["model_type"] = "inkling2_loop_text"
    raw["text_config"]["loop_config"]["is_varloop"] = True
    raw["vision_config"].update(model_type="inkling2_vision", text_hidden_size=2304)
    for flag in ("pre_norm_pre_affine", "final_norm_pre_affine"):
        raw["text_config"].pop(flag, None)
    model = ModelConfig.from_dict(raw)
    assert model.text.hidden == 2304 and len(model.text.visits) == 48
    assert model.quantization == precision


def test_mini_preparation_identity(mini_config_dir):
    from hyperflare_recipes.recipes.inkling2.config import LoopConfig, MiniTextConfig

    mini = ModelConfig.load(mini_config_dir)
    assert type(mini.text) is MiniTextConfig
    assert type(mini.text.loop) is LoopConfig
    assert tuple(f.name for f in fields(LoopConfig)) == ("core_start", "core_end", "num_passes", "gate_rank")
    raw = json.loads((mini_config_dir / "config.json").read_text())
    raw["quantization_config"] = json.loads((mini_config_dir / "hf_quant_config.json").read_text())
    raw["text_config"]["loop_config"]["is_varloop"] = True
    varloop = ModelConfig.from_dict(raw)
    assert varloop.text == mini.text
    assert varloop.text.loop == mini.text.loop


@pytest.mark.parametrize("field,value", [("num_passes", 0), ("core_start", 7), ("core_end", 30), ("is_varloop", "true")])
def test_invalid_loop_geometry(mini_config_dir, field, value):
    raw = json.loads((mini_config_dir / "config.json").read_text())
    raw["quantization_config"] = json.loads((mini_config_dir / "hf_quant_config.json").read_text())
    raw["text_config"]["loop_config"][field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA device")
@pytest.mark.parametrize("layer", [0, 20])
def test_expert_checkpoint_replay(snapshot, layer):
    from ignition import runtime_patches
    runtime_patches.apply()
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.components.situ_experts import PreparedSiTUExperts
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from hyperflare_recipes.recipes.inkling2.mini_weights import MiniWeightLoader
    from hyperflare_recipes.recipes.inkling2.mini_experts import load_experts
    with MiniWeightLoader(inspect_indexed_checkpoint(snapshot), "cuda:0", shard=TensorParallelShard(0, 1), vision=False) as loader:
        experts = load_experts(loader, layer, shard=loader.shard)
    operation = PreparedSiTUExperts(experts, tokens=4, top_k=8)
    torch.manual_seed(917)
    x = torch.randn(4, 2304, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(32, device="cuda", dtype=torch.int32).reshape(4, 8)
    ids[-1] = -1
    weights = torch.softmax(torch.randn(4, 8, device="cuda"), -1) * 8
    expected = operation(x, ids, weights)
    assert torch.isfinite(expected).all()
    assert torch.count_nonzero(expected[-1]) == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = operation(x, ids, weights)
    for _ in range(3):
        x.normal_()
        expected = operation(x, ids, weights)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA device")
@pytest.mark.parametrize("layer", [0, 20])
def test_expert_vllm_agreement(snapshot, layer):
    reference = os.environ.get("FLARE_INKLING2_MINI_REFERENCE")
    if not reference:
        pytest.skip("requires independently generated vLLM expert fixtures")
    from safetensors.torch import load_file
    from ignition import runtime_patches
    runtime_patches.apply()
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.components.situ_experts import PreparedSiTUExperts
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from hyperflare_recipes.recipes.inkling2.mini_weights import MiniWeightLoader
    from hyperflare_recipes.recipes.inkling2.mini_experts import load_experts
    tensors = load_file(reference, device="cuda")
    with MiniWeightLoader(inspect_indexed_checkpoint(snapshot), "cuda:0", shard=TensorParallelShard(0, 1), vision=False) as loader:
        experts = load_experts(loader, layer, shard=loader.shard)
    for tokens in (1, 4, 8, 32, 128, 4096, 8192):
        prefix = f"layer{layer}.{tokens}."
        operation = PreparedSiTUExperts(experts, tokens=tokens, top_k=8)
        actual = operation(tensors[prefix + "input"], tensors[prefix + "ids"].int(), tensors[prefix + "routes"])
        expected = tensors[prefix + "output"]
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        print(f"layer={layer} tokens={tokens} relative_l2={error.item():.6g}")
        # GB10 fixtures measure at most 3.93e-7; allow FP32 reduction-order variation.
        assert torch.isfinite(actual).all() and error <= 1e-6


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA device")
@pytest.mark.parametrize("layer", [0, 20])
@torch.no_grad()
def test_expert_compiled_catalog_restore(snapshot, tmp_path, layer):
    from ignition import runtime_patches
    runtime_patches.apply()
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.components.situ_experts import PreparedSiTUExperts
    from ignition.preparation.aot import ForwardCache
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.execution import GraphSignature
    from hyperflare_recipes.recipes.inkling2.mini_weights import MiniWeightLoader
    from hyperflare_recipes.recipes.inkling2.mini_experts import load_experts
    with MiniWeightLoader(inspect_indexed_checkpoint(snapshot), "cuda:0", shard=TensorParallelShard(0, 1), vision=False) as loader:
        weights = load_experts(loader, layer, shard=loader.shard)
    operation = PreparedSiTUExperts(weights, tokens=4, top_k=8)
    source = torch.randn(4, 2304, dtype=torch.bfloat16, device="cuda")
    ids = torch.arange(32, dtype=torch.int32, device="cuda").reshape(4, 8)
    ids[-1] = -1
    routes = torch.softmax(torch.randn(4, 8, device="cuda"), -1) * 8
    forward = ExpertForward(operation, source, ids, routes)
    cache = ForwardCache(namespace={"model": "mini-experts", "layer": layer}, device="cuda:0",
                         root=tmp_path, compile_workers=1)
    signature = {"decode": GraphSignature("decode", 4, 1)}
    for _ in range(2):
        compiled = cache.prepare_many({"decode": forward}, signature, materialize=lambda: None)["decode"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        source.normal_()
        expected = operation(source, ids, routes)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(forward.output, expected, rtol=0, atol=0)
        graph.reset()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA device")
@pytest.mark.parametrize("endpoint", [127, 128, 129, 1025, 4122, 8191, 8192, 8193])
@pytest.mark.parametrize("drafting", [False, True])
@torch.no_grad()
def test_loop_histories_snapshot_branch_restore(snapshot, endpoint, drafting):
    from ignition import runtime_patches
    runtime_patches.apply()
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
    from hyperflare_recipes.recipes.inkling2.pool import ContinuationPool, RankGeometry

    config = ModelConfig.load(snapshot)
    text = config.text.cache_config
    pool = ContinuationPool.allocate(text, RankGeometry.of(text, 1), 8, 5, "cuda:0",
                                     draft=config.draft if drafting else None)
    commands = StateCommands.allocate("cuda:0", capacity=4)
    tables = torch.full((4, endpoint // 128 + 1), 3, dtype=torch.int32, device="cuda")
    tables[1].fill_(2)
    tables[2].fill_(1)
    consumed = pool.region("consumed_tokens")
    requests = DeviceRequestState.allocate("cuda:0", vocab_size=128, capacity=4,
                                          consumed_tokens=consumed[1:5, 0])
    rows = torch.arange(1, 5, device="cuda")[:, None].expand(4, 8).contiguous()
    transfer = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(),
                                    page_tokens=128, state_rows=rows, requests=requests)
    transfer.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer.validate()
        transfer.apply()

    def replay(operations):
        commands.num_ops.fill_(len(operations))
        for index, (source, destination, slot) in enumerate(operations):
            commands.source_slots[index] = source
            commands.destination_slots[index] = destination
            commands.operation[index] = 0
            commands.expected_consumed[index] = endpoint
            commands.request_slots[index] = slot
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
            assert before[key] == after[key]
        assert not commands.errors.count_nonzero()

    # Distinct bytes cover every visit's rings and convolution histories.
    pool.backing[1].random_(0, 256)
    consumed[1] = endpoint
    for cache in pool.pages.values():
        cache.random_(0, 256)
    tails = {name: cache[3].clone() for name, cache in pool.partial_pages()} if endpoint % 128 else {}
    expected = pool.backing[1].clone()
    for name, value in tails.items():
        region = pool.layout.region(name)
        expected[region.offset:region.offset + region.nbytes].copy_(value)
    replay([(1, 5, 0)])
    assert torch.equal(pool.backing[5], expected)
    pool.backing[1].fill_(255)
    for cache in pool.pages.values():
        cache.zero_()
    replay([(5, 2, 1), (5, 3, 2)])
    for row in (2, 3, 5):
        assert torch.equal(pool.backing[row], expected)
    for name, cache in pool.partial_pages():
        if tails:
            assert torch.equal(cache[1], tails[name])
            assert torch.equal(cache[2], tails[name])
        assert not cache[3:].count_nonzero()
    graph.reset()
