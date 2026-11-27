# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Kimi EXL3 replacement inspection, ExampleArena, and full 65,536-patch vision catalog."""

from dataclasses import replace
import json

import pytest
import torch

from ignition import planning, registry
from ignition.errors import UsageError
from hyperflare_recipes.parameters import ExpertScaleCompressionParameters
from hyperflare_recipes.recipes.kimi3.examples import ExampleArena, declaration_tensor
from hyperflare_recipes.recipes.kimi3.recipe import _exl3_scope_savings_for
from hyperflare_recipes.recipes.kimi3.vision import PATCH_CAPACITIES
from hyperflare_recipes.recipes.kimi31.recipe import RECIPE as KIMI31


def _write_exl3_dir(directory, *, weight_file_bytes=796_967_521_440, bad_manifest=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({
        "architectures": ["KimiK31ForConditionalGeneration"],
        "text_config": {
            "quantization_config": {
                "quant_method": "exl3",
                "exl3": {
                    "manifest": "exl3-manifest.json",
                    "weight_file_bytes": weight_file_bytes,
                },
            },
            "hidden_size": 7168,
            "num_hidden_layers": 93,
            "num_attention_heads": 64,
            "q_lora_rank": 1536,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "first_k_dense_replace": 1,
            "num_shared_experts": 1,
            "moe_intermediate_size": 1024,
            "linear_attn_config": {
                "num_heads": 64,
                "kda_layers": list(range(69)),
            },
        },
    }))
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 300_000_000_000},
        "weight_map": {"model.embed_tokens.weight": "model-00001-of-00094.safetensors"},
    }))
    if bad_manifest:
        (directory / "exl3-manifest.json").write_text(json.dumps({"schema": "bad-v0"}))
    else:
        (directory / "exl3-manifest.json").write_text(json.dumps({
            "schema": "exl3-v1",
            "kind": "exl3-manifest",
            "layers": {"1": {"file": "exl3-experts-layer-001.safetensors"}},
        }))


def test_kimi31_exl3_replacement_inspection_and_csf_rejection(tmp_path):
    ckpt_dir = tmp_path / "kimi31-exl3"
    _write_exl3_dir(ckpt_dir)

    resolved = registry.resolve(str(ckpt_dir), recipes=(KIMI31,))
    assert resolved.recipe is KIMI31
    assert resolved.checkpoint.stored_bytes == 796_967_521_440
    assert resolved.checkpoint.scale_compressed_device_bytes == {}
    savings = _exl3_scope_savings_for(resolved.checkpoint)
    assert set(savings) == {"kda-input", "kda-all", "all"}
    assert savings["kda-all"] > savings["kda-input"]

    gpus = tuple(
        planning.Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i:04d}", 96 << 30, 95 << 30, (12, 0))
        for i in range(8)
    )
    base_params = KIMI31.parameters
    # EXL3 checkpoints reject --expert-scale-compression csf cleanly.
    csf_params = replace(
        base_params,
        max_concurrency=4,
        speculation=replace(base_params.speculation, draft_tokens=7, method="dspark"),
        expert_scale_compression=ExpertScaleCompressionParameters(encoding="csf"),
    )
    with pytest.raises(UsageError, match="--expert-scale-compression csf"):
        planning.fit(KIMI31, resolved.checkpoint, gpus, parameters=csf_params, reserve_bytes=0)

    # TP8/DCP8 EXL3 planning succeeds with kda-all MXFP8 and DSpark draft-body MXFP8.
    kda_all_params = replace(
        base_params,
        max_concurrency=4,
        speculation=replace(base_params.speculation, draft_tokens=7, method="dspark"),
        online_mxfp8=replace(base_params.online_mxfp8, enabled=True, scope="kda-all"),
        draft_quantization=replace(base_params.draft_quantization, draft_body_quantization="mxfp8"),
        context_parallel=replace(base_params.context_parallel, enabled=True, size=8),
    )
    plan = planning.fit(KIMI31, resolved.checkpoint, gpus, parameters=kda_all_params, reserve_bytes=0)
    assert plan.world_size == 8 and plan.pages > 0

    kda_input_params = replace(
        kda_all_params,
        online_mxfp8=replace(base_params.online_mxfp8, enabled=True, scope="kda-input"),
    )
    plan_input = planning.fit(KIMI31, resolved.checkpoint, gpus, parameters=kda_input_params, reserve_bytes=0)
    assert plan.weights_bytes < plan_input.weights_bytes

    # Malformed EXL3 manifest raises UsageError rather than bare ValueError.
    bad_dir = tmp_path / "kimi31-exl3-bad"
    _write_exl3_dir(bad_dir, bad_manifest=True)
    with pytest.raises(UsageError, match="unsupported EXL3 expert manifest"):
        registry.resolve(str(bad_dir), recipes=(KIMI31,))


def test_example_arena_disjoint_namespaces_validation_and_fallback():
    arena = ExampleArena(
        "cpu",
        64 << 10,
        min_bytes=1024,
        namespace_bases={
            "vision": (0, 64 << 10),
            "common": (0, 32 << 10),
            "kda": (32 << 10, 64 << 10),
            "dsa": (32 << 10, 64 << 10),
        },
    )
    examples = {"__arena__": arena}

    v = declaration_tensor(examples, "vision.hidden", (1024,), torch.bfloat16, "cpu")
    c = declaration_tensor(examples, "misc.buf", (1024,), torch.bfloat16, "cpu")
    k = declaration_tensor(examples, "kda.q", (1024,), torch.bfloat16, "cpu")
    d = declaration_tensor(examples, "dsa.q", (1024,), torch.bfloat16, "cpu")

    # vision aliases common at offset 0; kda and dsa share the upper half disjoint from common.
    assert v.data_ptr() == arena.slab.data_ptr()
    assert c.data_ptr() == arena.slab.data_ptr()
    assert k.data_ptr() == arena.slab.data_ptr() + (32 << 10)
    assert d.data_ptr() == arena.slab.data_ptr() + (32 << 10)
    assert k.data_ptr() >= c.data_ptr() + c.numel() * c.element_size()

    # Allocations exceeding namespace span fall back cleanly to standalone tensors.
    overflow = declaration_tensor(examples, "kda.huge", (32 << 10,), torch.bfloat16, "cpu")
    assert not (arena.slab.data_ptr() <= overflow.data_ptr() < arena.slab.data_ptr() + arena.slab.nbytes)

    # Undeclared namespaces are rejected explicitly rather than silently aliasing offset 0.
    with pytest.raises(KeyError, match="undeclared ExampleArena namespace"):
        arena.allocate("unknown", (1024,), torch.bfloat16)

    arena.release()
    assert arena.slab is None
    with pytest.raises(RuntimeError, match="already been released"):
        arena.allocate("kda", (1024,), torch.bfloat16)


def test_vision_patch_capacities_include_full_65536():
    assert PATCH_CAPACITIES == (256, 1024, 4096, 16384, 65536)


def test_vision_tp8_and_tp16_scratch_and_weight_sharding(monkeypatch):
    from types import SimpleNamespace
    from hyperflare_recipes.recipes.kimi3 import vision as vision_mod

    for tp_size, expected_heads, expected_mlp in ((1, 12, 4096), (8, 2, 512), (16, 1, 256)):
        scratch = vision_mod.VisionScratch.allocate(
            256, "cpu", tp_size=tp_size, local_heads=expected_heads,
        )
        assert scratch.qkv.shape == (256, 3 * expected_heads * 128)
        assert scratch.query.shape == (256, expected_heads, 128)
        assert scratch.attended.shape == (256, expected_heads * 128)
        assert scratch.mlp_up.shape == (256, expected_mlp)
        assert scratch.activated.shape == (64, expected_mlp)

    wqkv_full = torch.arange(3 * 12 * 128 * 1024, dtype=torch.bfloat16).reshape(3 * 12 * 128, 1024)
    wo_full = torch.arange(1024 * 12 * 128, dtype=torch.bfloat16).reshape(1024, 12 * 128)
    fc0_full = torch.arange(4096 * 1024, dtype=torch.bfloat16).reshape(4096, 1024)
    fc1_full = torch.arange(1024 * 4096, dtype=torch.bfloat16).reshape(1024, 4096)
    norm_w = torch.ones(1024, dtype=torch.bfloat16)
    fake_tensors = {
        "vision_tower.encoder.blocks.0.wqkv.weight": wqkv_full,
        "vision_tower.encoder.blocks.0.wo.weight": wo_full,
        "vision_tower.encoder.blocks.0.mlp.fc0.weight": fc0_full,
        "vision_tower.encoder.blocks.0.mlp.fc1.weight": fc1_full,
        "vision_tower.encoder.blocks.0.norm0.weight": norm_w,
    }

    class FakeLoader:
        def __init__(self, manifest, device, *, prefixes=(), progress=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def _selected_names(self):
            return list(fake_tensors)

        def load_tensor(self, name, *, extent=None, temporary=False):
            tensor = fake_tensors[name]
            return tensor[extent].clone() if extent is not None else tensor.clone()

    monkeypatch.setattr(vision_mod, "DirectWeightLoader", FakeLoader)
    monkeypatch.setattr(vision_mod, "persistent_copy", lambda tensor: tensor.clone())

    def make_shard(rank, world_size):
        def partition(size, alignment=1):
            width = size // world_size
            return (rank * width, (rank + 1) * width)

        return SimpleNamespace(rank=rank, world_size=world_size, partition=partition)

    # TP16 active head rank (rank 0 owns head 0) vs zero-padded head rank (rank 12 owns padded head).
    w0 = vision_mod.VisionWeights.load(None, "cpu", shard=make_shard(0, 16), tp_enabled=True, online_mxfp8=False)
    w12 = vision_mod.VisionWeights.load(None, "cpu", shard=make_shard(12, 16), tp_enabled=True, online_mxfp8=False)
    assert w0.tp_size == 16 and w0.local_heads == 1
    assert w12.tp_size == 16 and w12.local_heads == 1
    assert w0.tensors["vision_tower.encoder.blocks.0.wqkv.weight"].shape == (3 * 128, 1024)
    assert w0.tensors["vision_tower.encoder.blocks.0.wo.weight"].shape == (1024, 128)
    assert w0.tensors["vision_tower.encoder.blocks.0.mlp.fc0.weight"].shape == (256, 1024)
    assert w0.tensors["vision_tower.encoder.blocks.0.mlp.fc1.weight"].shape == (1024, 256)
    torch.testing.assert_close(
        w0.tensors["vision_tower.encoder.blocks.0.wqkv.weight"],
        wqkv_full.view(3, 12, 128, 1024)[:, 0:1, :, :].reshape(3 * 128, 1024),
    )
    assert torch.count_nonzero(w12.tensors["vision_tower.encoder.blocks.0.wqkv.weight"]) == 0
    assert torch.count_nonzero(w12.tensors["vision_tower.encoder.blocks.0.wo.weight"]) == 0
    torch.testing.assert_close(
        w12.tensors["vision_tower.encoder.blocks.0.mlp.fc0.weight"],
        fc0_full[12 * 256 : 13 * 256, :],
    )


def test_query_sharded_dsa_threshold_2560():
    from types import SimpleNamespace
    from hyperflare_recipes.recipes.kimi31.query_selection import PreparedQuerySelection

    metadata = SimpleNamespace(starts=torch.zeros(1, dtype=torch.int64))
    # Below 2560 rows is rejected by the row-count threshold guard.
    with pytest.raises(ValueError, match="query-sharded indexing requires"):
        PreparedQuerySelection(
            torch.zeros((2048, 4096), dtype=torch.bfloat16),
            torch.zeros((2048, 32), dtype=torch.bfloat16),
            torch.zeros((2, 64 * 132), dtype=torch.uint8),
            metadata,
            SimpleNamespace(world=8, index=0),
            None,
            {},
            session=None,
            name="test",
            workspace=None,
        )
    # At 2560 rows, the row-count threshold passes and reaches index column validation.
    context = SimpleNamespace(
        index_staging=SimpleNamespace(tensor=torch.zeros((64, 64 * 132), dtype=torch.uint8), addresses=torch.zeros(8, dtype=torch.int64)),
        index_records=torch.zeros((4, 64 * 132), dtype=torch.uint8),
        index_columns=torch.zeros((2,), dtype=torch.int32),  # deliberate mismatch to stop before CUDA ops
        selections=object(),
    )
    with pytest.raises(ValueError, match="one int32 column per global index page"):
        PreparedQuerySelection(
            torch.zeros((2560, 4096), dtype=torch.bfloat16),
            torch.zeros((2560, 32), dtype=torch.bfloat16),
            torch.zeros((2, 64 * 132), dtype=torch.uint8),
            metadata,
            SimpleNamespace(world=8, index=0),
            context,
            {},
            session=None,
            name="test",
            workspace=None,
        )
