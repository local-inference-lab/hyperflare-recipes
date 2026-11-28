# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 preparation admits complete explicit placements and counts physical copies."""

from dataclasses import replace
import json
import struct
import subprocess
import sys

import pytest

from hyperflare_recipes.components.expert_placement import ExpertPlacement, PrefillExpertStaging
from hyperflare_recipes.recipes.deepseek41.bootstrap import PreparationConfig, _collective_runtimes
from hyperflare_recipes.recipes.deepseek41.sm103 import SM103Config
from .test_deepseek41_pool import config
from .test_deepseek41_csf_config import configuration


@pytest.fixture
def model():
    base = config()
    return replace(base, text={**base.text, "n_routed_experts": 384, "moe_intermediate_size": 2304,
        "num_nextn_predict_layers": 3, "dspark_block_size": 8, "dspark_noise_token_id": 1,
        "dspark_target_layer_ids": [2, 10, 20], "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3,
        "vocab_size": 129280, "compress_ratios": base.text["compress_ratios"] + [0, 0, 0]})


@pytest.fixture
def options():
    target = ExpertPlacement(384, tuple(range(160, 384)), tuple(range(256)))
    draft = ExpertPlacement(128, (), tuple(range(128)))
    placements = tuple((f"layers.{i}.ffn", target) for i in range(40))
    placements += tuple((f"mtp.{i}.ffn", draft) for i in range(3))
    return SM103Config(placements)


def test_physical_accounting_includes_duplicate_host_and_hbm_copies(model, options):
    options.validate_model(model, dspark=True)
    per_expert = 3 * 5120 * 2304 * 17 // 32
    assert options.expert_bytes(model, dspark=True) == {
        "host": 40 * 224 * per_expert, "hbm": (40 * 256 + 3 * 128) * per_expert}
    assert options.placements[0][1].physical_rows == 480
    assert options.to_dict()["placements"]["layers.0.ffn"]["hbm"] == list(range(256))


@pytest.mark.parametrize("mutation", ["missing_target", "missing_draft", "wrong_count", "extra_draft"])
def test_incomplete_or_mismatched_model_coverage_fails_before_allocation(model, options, mutation):
    entries = dict(options.placements)
    if mutation == "missing_target":
        del entries["layers.0.ffn"]
    elif mutation == "missing_draft":
        del entries["mtp.0.ffn"]
    elif mutation == "wrong_count":
        entries["layers.0.ffn"] = ExpertPlacement(383, (), tuple(range(383)))
    with pytest.raises(ValueError, match="every enabled"):
        replace(options, placements=tuple(entries.items())).validate_model(model, dspark=mutation != "extra_draft")


def test_tp1_requires_explicit_local_sm103_configuration(options, tmp_path):
    with pytest.raises(ValueError, match="distinct GPUs"):
        PreparationConfig(devices=(0,))
    with pytest.raises(ValueError, match="explicit placement"):
        PreparationConfig(devices=(0,), sm103=options)
    with pytest.raises(ValueError, match="distinct GPUs"):
        PreparationConfig(devices=(0, 1, 2, 3), sm103=options, kernel_cache=tmp_path)
    with pytest.raises(ValueError, match="one local GPU"):
        PreparationConfig(devices=(0,), hosts=("gracie",), sm103=options, kernel_cache=tmp_path)
    with pytest.raises(ValueError, match="expanded native expert scales"):
        PreparationConfig(devices=(0,), sm103=options, kernel_cache=tmp_path, expert_scale_compression="csf")
    configuration = PreparationConfig(devices=(0,), sm103=options, kernel_cache=tmp_path)
    assert configuration.draft_tokens == 5
    assert _collective_runtimes(configuration, None, None, coordinator=None,
        core_storage=None, verification_samples=configuration.decode_tokens) == (None,) * 5


@pytest.mark.parametrize("experts,host,hbm", [(4, (0, 1), (2,)), (4, (0, 1), (1, 2, 4)),
    (4, (0, 0, 1), (2, 3)), (4, (), (0, 1, 2, True)), (4, (0, 1, 2, 3), ())])
def test_invalid_physical_expert_rows_are_rejected(experts, host, hbm):
    with pytest.raises(ValueError, match="complete canonical coverage"):
        ExpertPlacement(experts, host, hbm)


@pytest.fixture
def snapshot(tmp_path, configuration):
    """Small synthetic payloads exercise header accounting, not model execution."""
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(configuration))
    tensors = {
        "layers.0.attn.wq.weight": ("F8_E4M3", 1024),
        "layers.0.attn.wq.scale": ("U8", 32),
        "layers.0.ffn.experts.0.w1.weight": ("U8", 16),
        "layers.0.engram.embed.weight": ("U8", 32),
        "mtp.0.attn.wkv.weight": ("F8_E4M3", 1024),
    }
    offset, header = 0, {}
    for name, (dtype, size) in tensors.items():
        header[name] = dict(dtype=dtype, shape=[size], data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return path


@pytest.mark.parametrize("staging", [None, PrefillExpertStaging(1024, 8)])
def test_configuration_roundtrip_preserves_row_order(options, tmp_path, staging):
    options = replace(options, prefill_staging=staging)
    value = options.to_dict()
    path = tmp_path / "sm103.json"
    path.write_text(json.dumps(value))
    assert SM103Config.load(path) == options


def test_staging_capacity_and_host_demand_are_admitted_explicitly(options, model, tmp_path):
    options = replace(options, prefill_staging=PrefillExpertStaging(1024, 8))
    with pytest.raises(ValueError, match="threshold exceeds"):
        PreparationConfig(devices=(0,), sm103=options, kernel_cache=tmp_path, prefill_chunk_size=256)
    assert PreparationConfig(devices=(0,), sm103=options, kernel_cache=tmp_path,
                             prefill_chunk_size=8192).sm103.prefill_staging == options.prefill_staging
    with pytest.raises(ValueError, match="cold-demand threshold"):
        replace(options, prefill_staging=PrefillExpertStaging(1024, 385)).staging_bytes(model)


@pytest.mark.parametrize("staging", [True, {}, {"min_rows": 1024},
    {"min_rows": 1024, "min_cold_experts": 8, "unknown": 1},
    {"min_rows": True, "min_cold_experts": 8}, {"min_rows": 1024, "min_cold_experts": 0}])
def test_staging_json_rejects_incomplete_or_invalid_settings(options, tmp_path, staging):
    path = tmp_path / "sm103.json"
    path.write_text(json.dumps({**options.to_dict(), "prefill_staging": staging}))
    with pytest.raises(ValueError):
        SM103Config.load(path)


def test_single_rank_recipe_passes_placement_without_starting_peers(options, snapshot, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ignition import planning
    from hyperflare_recipes.recipes.deepseek41 import bootstrap
    from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE, usage
    from ignition.declaration import RankUsage
    from ignition.preparation import rank_group

    recipe, checkpoint = options.planning_inputs(RECIPE, RECIPE.checkpoints[0], snapshot, dspark=True)
    gpu = planning.Gpu(0, "GB300", "GPU-test", 256 << 30, 250 << 30, (10, 3))
    base_plan = planning.plan(recipe, checkpoint, (gpu,), selected=(0,), concurrency=1,
        reserve_bytes=0, speculation=recipe.speculation, prefill_chunk=256)
    if "sm103" in planning.Plan.__dataclass_fields__:
        plan = replace(base_plan, sm103=options)
    else:
        class _SM103Plan:
            def __init__(self, inner, sm103):
                self._inner = inner
                self.sm103 = sm103

            def __getattr__(self, name):
                return getattr(self._inner, name)

        plan = _SM103Plan(base_plan, options)
    prepared = SimpleNamespace(memory_report=dict(pages=64, free_device_bytes=1 << 30))
    storage = object()
    calls = []

    def prepare_rank(path, config, *, rank, group, progress, core_storage):
        calls.append((path, config, rank, group, core_storage))
        return prepared

    monkeypatch.setattr(bootstrap, "prepare_rank", prepare_rank)
    monkeypatch.setattr(rank_group, "prepare_group", lambda *a, **kw: pytest.fail("TP1 started peer orchestration"))
    assert recipe.prepare(snapshot, plan, compile_forwards=True, kernel_cache=tmp_path,
                          progress=None, core_storage=storage) is prepared
    assert len(calls) == 1
    path, config, rank, group, owner = calls[0]
    assert path == snapshot and rank == 0 and group is None and owner is storage
    assert config.sm103 == options and config.devices == (0,) and config.dspark
    assert usage(prepared) == (RankUsage(64, 1 << 30),)
    peers = SimpleNamespace(memory_report={"ranks": [prepared.memory_report] * 3})
    assert usage(peers) == (RankUsage(64, 1 << 30),) * 3


@pytest.mark.parametrize("change", ["version", "unknown", "bool_cluster", "duplicate", "incomplete"])
def test_invalid_json_configuration_is_rejected(options, tmp_path, change):
    value = options.to_dict()
    if change == "version":
        value["version"] = True
    elif change == "unknown":
        value["ignored"] = 1
    elif change == "bool_cluster":
        value["host_lane_clusters"] = True
    elif change == "incomplete":
        value["placements"]["layers.0.ffn"]["host"] = []
    text = json.dumps(value)
    if change == "duplicate":
        text = '{"version": 1, ' + text[1:]
    path = tmp_path / "sm103.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        SM103Config.load(path)


@pytest.mark.parametrize("drafting", [False, True])
@pytest.mark.parametrize("staging", [False, True])
def test_planning_counts_native_experts_and_physical_cache_layout(options, snapshot, drafting, staging):
    from hyperflare_recipes.recipes.deepseek41.config import ModelConfig
    from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE
    from hyperflare_recipes.recipes.deepseek41.sm103 import _reference_cache_costs

    if not drafting:
        options = replace(options, placements=options.placements[:40])
    if staging:
        options = replace(options, prefill_staging=PrefillExpertStaging())
    recipe, checkpoint = options.planning_inputs(RECIPE, RECIPE.checkpoints[0], snapshot, dspark=drafting)
    model = ModelConfig.load(snapshot)
    tier = options.expert_bytes(model, dspark=drafting)
    # FP8 projections reserve expanded scales and round up to allocation alignment.
    assert checkpoint.device_bytes == tier["hbm"] + 1536 + 512 + (1536 + 1056 if drafting else 0)
    assert checkpoint.mapped_host_bytes == tier["host"]
    assert checkpoint.expansion_bytes == 0
    assert not checkpoint.scale_compressed_device_bytes
    costs = recipe.ranks[1] if drafting else recipe.plain_ranks[1]
    width = RECIPE.max_draft_tokens + 1 if drafting else 1
    expected_row, expected_page = _reference_cache_costs(model, verification_width=width, dspark=drafting)
    assert costs.state_row_bytes == expected_row
    assert costs.page_bytes == expected_page
    assert costs.graph_bytes == (8 << 30) + (3 * 224 * 18_800_640 if staging else 0)
    assert 1 not in RECIPE.ranks  # Explicit placement must not change ordinary TP admission.


@pytest.mark.parametrize("staging", [False, True])
def test_sm103_planning_inputs_do_not_import_torch(options, snapshot, tmp_path, staging):
    if staging:
        options = replace(options, prefill_staging=PrefillExpertStaging(min_rows=256))
    path = tmp_path / "sm103.json"
    path.write_text(json.dumps(options.to_dict()))
    # A separate process checks the planning boundary before CUDA or torch setup.
    script = '''
import sys
from pathlib import Path
from ignition import planning
from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE
from hyperflare_recipes.recipes.deepseek41.sm103 import SM103Config
snapshot, placement = map(Path, sys.argv[1:])
loaded = SM103Config.load(placement)
recipe, checkpoint = loaded.planning_inputs(RECIPE, RECIPE.checkpoints[0], snapshot, dspark=True)
gpus = (planning.Gpu(0, "RTX", "GPU-rtx", 96 << 30, 95 << 30, (12, 0)),
        planning.Gpu(1, "GB300", "GPU-gb300", 256 << 30, 250 << 30, (10, 3)))
plan = planning.plan(recipe, checkpoint, gpus, selected=(1,), concurrency=1,
                     reserve_bytes=0, speculation=recipe.speculation, prefill_chunk=256)
assert tuple(g.index for g in plan.gpus) == (1,)
assert plan.speculation == RECIPE.speculation
assert plan.expert_scale_compression is None
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script, str(snapshot), str(path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_execution_configuration_and_planning_costs(options, snapshot, tmp_path):
    from ignition import planning
    from ignition.components.capacity import request_capacity
    from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE
    from hyperflare_recipes.recipes.deepseek41.sm103 import _reference_cache_costs
    from hyperflare_recipes.recipes.deepseek41.config import ModelConfig

    reference = SM103Config(options.placements, host_lane_clusters=4, execution="reference", context=262144)
    path = tmp_path / "sm103-reference.json"
    path.write_text(json.dumps(reference.to_dict()))
    assert SM103Config.load(path) == reference
    assert "flashmla_library" not in reference.to_dict()
    staged_reference = replace(reference, prefill_staging=PrefillExpertStaging(1024, 8))
    staged_path = tmp_path / "sm103-reference-staged.json"
    staged_path.write_text(json.dumps(staged_reference.to_dict()))
    assert SM103Config.load(staged_path) == staged_reference
    assert staged_reference.to_dict()["prefill_staging"] == {"min_rows": 1024, "min_cold_experts": 8}
    with pytest.raises(ValueError, match="must be 'reference'"):
        SM103Config(options.placements, execution="hybrid")
    hybrid_path = tmp_path / "sm103-hybrid.json"
    hybrid_path.write_text(json.dumps({**reference.to_dict(), "flashmla_library": "libflashmla.so"}))
    with pytest.raises(ValueError, match="do not match reference execution"):
        SM103Config.load(hybrid_path)
    with pytest.raises(ValueError, match="128-aligned"):
        SM103Config(options.placements, execution="reference", context=300)

    recipe, checkpoint = reference.planning_inputs(RECIPE, RECIPE.checkpoints[0], snapshot,
                                                   dspark=True, draft_tokens=7)
    model = ModelConfig.load(snapshot)
    expected_spec, expected_page = _reference_cache_costs(model, verification_width=8, dspark=True)
    expected_plain, _ = _reference_cache_costs(model, verification_width=1, dspark=False)
    assert recipe.ranks[1].state_row_bytes == expected_spec == 3_311_616
    assert recipe.plain_ranks[1].state_row_bytes == expected_plain == 2_855_424
    assert recipe.ranks[1].page_bytes == expected_page == 113_920
    assert recipe.verification_rows == 0
    assert recipe.context == 262144
    if hasattr(RECIPE, "packed_speculative_depths"):
        assert not RECIPE.packed_speculative_depths
        assert recipe.packed_speculative_depths
    tp_cfg = PreparationConfig(speculative_widths=tuple(range(1, 9)), speculative_graph_budget=12)
    assert len(tp_cfg.packed_speculative_shapes) == 12
    sm103_cfg = PreparationConfig(devices=(0,), sm103=reference, kernel_cache=tmp_path,
                                  speculative_widths=tuple(range(1, 9)), speculative_graph_budget=48)
    assert len(sm103_cfg.packed_speculative_depth_shapes) == 48
    gpu = planning.Gpu(0, "GB300", "GPU-gb300", 256 << 30, 250 << 30, (10, 3))
    plan = planning.plan(recipe, checkpoint, (gpu,), selected=(0,), concurrency=16,
                         reserve_bytes=0, speculation=recipe.speculation, prefill_chunk=8192)
    assert plan.live_rows == 1 + request_capacity(16)
    assert plan.context == 262144


def test_reference_sm103_admits_quantized_draft_and_markov_heads(options, tmp_path):
    reference = SM103Config(options.placements, host_lane_clusters=4, execution="reference", context=262144)
    for draft_q in (None, "nvfp4", "mxfp8"):
        for markov_q in (None, "fp8"):
            cfg = PreparationConfig(
                devices=(0,), sm103=reference, kernel_cache=tmp_path,
                draft_head_quantization=draft_q, markov_head_quantization=markov_q,
            )
            assert cfg.draft_head_quantization == draft_q
            assert cfg.markov_head_quantization == markov_q
    with pytest.raises(ValueError, match="Markov head quantization"):
        PreparationConfig(devices=(0,), sm103=reference, kernel_cache=tmp_path,
                          markov_head_quantization="nvfp4")
    with pytest.raises(ValueError, match="vocabulary-parallel"):
        PreparationConfig(devices=(0,), sm103=reference, kernel_cache=tmp_path,
                          vocabulary_parallel_sampling=True)


def test_contract_prepare_reads_reference_pool_geometry_ratios():
    from types import SimpleNamespace
    import torch
    from ignition.components.state import StateLayout
    from ignition.preparation.execution import GraphSignature
    from hyperflare_recipes.recipes.deepseek41 import contract

    layout = StateLayout.from_specs((("consumed_tokens", (1,), torch.int64),))
    page_l2 = torch.zeros((2, 64, 1, 288), dtype=torch.uint8)
    page_l20 = torch.zeros((2, 128, 1, 288), dtype=torch.uint8)
    layers = [SimpleNamespace(ratio=0)] * 21
    layers[2] = SimpleNamespace(ratio=2)
    layers[20] = SimpleNamespace(ratio=1)
    pool = SimpleNamespace(
        pages={"layer2.compressed_kv": page_l2, "layer20.compressed_kv": page_l20},
        geometry=SimpleNamespace(layers=tuple(layers)),
        layout=layout,
    )
    buffers = {"cache.layer2.compressed_kv": page_l2, "cache.layer20.compressed_kv": page_l20}
    signatures = {"decode1": GraphSignature("decode", 1, 1, ("request_state_rows",), speculation_enabled=False)}
    result = contract.prepare(signatures, buffers, pool, page_tokens=128, draft_lag=None)
    assert result["version"] == 3
    by_buffer = {r["storage"]["buffer"]: r["tokens_per_record"] for r in result["cache"]["regions"]}
    assert by_buffer["cache.layer2.compressed_kv"] == 2
    assert by_buffer["cache.layer20.compressed_kv"] == 1


def test_merge_image_features_concatenates_valid_groups_and_zeroes_padding():
    import torch
    from hyperflare_recipes.components.deepseek_vision import merge_image_features

    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    indices = torch.tensor([[0, 2, -1], [1, -1, 0]], dtype=torch.int32)
    merged = merge_image_features(features, indices)
    expected = torch.tensor([
        [1.0, 5.0, 0.0, 2.0, 6.0, 0.0],
        [3.0, 0.0, 1.0, 4.0, 0.0, 2.0],
    ])
    assert torch.equal(merged, expected)


@pytest.mark.parametrize(
    ("filename", "early_hbm", "late_hbm", "host_per_tiered"),
    [
        ("sm103-gb300-384-315-301.json", 315, 301, 216),
        ("sm103-gb300-largekvc-384-296-294.json", 296, 294, 216),
    ],
)
def test_bundled_sm103_gb300_placement_assets_load_and_validate(
    model, filename, early_hbm, late_hbm, host_per_tiered
):
    from pathlib import Path

    assets_dir = Path(__file__).resolve().parents[2] / "hyperflare_recipes/recipes/deepseek41/assets"
    loaded = SM103Config.load(assets_dir / filename)
    loaded.validate_model(model, dspark=True)
    assert loaded.execution == "reference"
    assert loaded.context == 1048576
    assert loaded.host_lane_clusters == 4
    assert loaded.prefill_staging == PrefillExpertStaging(min_rows=1024, min_cold_experts=8)
    assert loaded.dynamic_mover is not None
    assert loaded.dynamic_mover.spare_slots == 1

    placements = dict(loaded.placements)
    assert len(placements["layers.0.ffn"].hbm) == 384
    assert len(placements["layers.0.ffn"].host) == 0
    for layer_idx in range(1, 20):
        placement = placements[f"layers.{layer_idx}.ffn"]
        assert len(placement.hbm) == early_hbm
        assert len(placement.host) == host_per_tiered
    for layer_idx in range(20, 40):
        placement = placements[f"layers.{layer_idx}.ffn"]
        assert len(placement.hbm) == late_hbm
        assert len(placement.host) == host_per_tiered
    for mtp_idx in range(3):
        placement = placements[f"mtp.{mtp_idx}.ffn"]
        assert len(placement.hbm) == 128
        assert len(placement.host) == 0
