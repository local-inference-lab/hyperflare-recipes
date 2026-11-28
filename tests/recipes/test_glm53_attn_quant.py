# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GLM-5.3 MXFP8 attention quantization, zero-copy KDA segments, and MLA staging."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
from safetensors.torch import save_file
import torch


def _load_quantize_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "quantize_glm53_attention.py"
    spec = importlib.util.spec_from_file_location("quantize_glm53_attention", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target_attention_prefixes_counts_and_flags():
    mod = _load_quantize_module()
    base = mod.target_attention_prefixes(45, include_mtp=False, include_kv_b_proj=False)
    # 34 KDA layers (0..44 where l%4 != 3) * 4 + 11 MLA layers (3,7,...,43) * 4 = 180
    assert len(base) == 180
    assert all(not p.endswith(".kv_b_proj") for p in base)
    assert all(".layers.45." not in p for p in base)
    # Gates and indexer projections must never be targeted.
    for forbidden in (".b_proj", ".f_a_proj", ".f_b_proj", ".g_a_proj", ".g_b_proj", ".indexer."):
        assert all(forbidden not in p for p in base)

    with_kvb = mod.target_attention_prefixes(45, include_mtp=False, include_kv_b_proj=True)
    assert len(with_kvb) == 180 + 11
    with_mtp = mod.target_attention_prefixes(45, include_mtp=True, include_kv_b_proj=True)
    assert len(with_mtp) == 180 + 11 + 5


def test_quantize_mxfp8_numerical_roundtrip_and_zero_blocks():
    mod = _load_quantize_module()
    torch.manual_seed(7)
    w = (torch.randn(128, 256, dtype=torch.bfloat16) * 0.05)
    w[0, :32] = 0.0
    q = mod.quantize_mxfp8(w, label="test.w")
    assert q["weight"].dtype == torch.float8_e4m3fn
    assert q["weight"].shape == (128, 256)
    assert q["weight_scale"].dtype == torch.uint8
    assert q["weight_scale"].shape == (128, 8)
    deq = mod.dequantize_mxfp8(q)
    assert torch.isfinite(deq).all()
    assert torch.all(deq[0, :32] == 0.0)
    rel_err = (deq - w.float()).norm() / w.float().norm()
    assert rel_err.item() < 0.04


def test_quantize_mxfp8_rejects_bad_inputs():
    mod = _load_quantize_module()
    with pytest.raises(ValueError, match="2D BF16"):
        mod.quantize_mxfp8(torch.randn(64, 64, dtype=torch.float32))
    with pytest.raises(ValueError, match="divisible by 32"):
        mod.quantize_mxfp8(torch.randn(64, 48, dtype=torch.bfloat16))
    bad = torch.zeros(32, 32, dtype=torch.bfloat16)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        mod.quantize_mxfp8(bad)


def test_quantize_checkpoint_end_to_end_and_hardlinks(tmp_path):
    mod = _load_quantize_module()
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    # Tiny 4-layer checkpoint: layers 0,1,2 KDA + layer 3 MLA + one untouched MoE shard.
    attn_tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.0.self_attn.b_proj.weight": torch.randn(16, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.3.self_attn.q_a_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.3.self_attn.q_b_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.3.self_attn.o_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
        "model.language_model.layers.3.self_attn.kv_b_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.02,
    }
    moe_tensors = {
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": torch.zeros(32, 32, dtype=torch.uint8),
    }
    save_file(attn_tensors, str(src / "shard-attn.safetensors"))
    save_file(moe_tensors, str(src / "shard-moe.safetensors"))
    (src / "tokenizer.json").write_text("{}\n")

    weight_map = {k: "shard-attn.safetensors" for k in attn_tensors}
    weight_map.update({k: "shard-moe.safetensors" for k in moe_tensors})
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 100000}, "weight_map": weight_map})
    )
    (src / "config.json").write_text(
        json.dumps(
            {
                "text_config": {"num_hidden_layers": 4},
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {"model.language_model.layers.3.mlp.experts": {"quant_algo": "NVFP4"}},
                },
            }
        )
    )

    summary = mod.quantize_checkpoint(src, dst, device="cpu")
    assert summary["converted_count"] == 8
    assert summary["rewritten_shards"] == 1
    assert summary["bytes_saved"] > 0

    # Untouched MoE shard is hardlinked (same inode).
    assert (src / "shard-moe.safetensors").stat().st_ino == (dst / "shard-moe.safetensors").stat().st_ino
    assert (src / "shard-attn.safetensors").stat().st_ino != (dst / "shard-attn.safetensors").stat().st_ino

    out_cfg = json.loads((dst / "config.json").read_text())
    qlayers = out_cfg["quantization_config"]["quantized_layers"]
    assert qlayers["model.language_model.layers.0.self_attn.q_proj"] == {"quant_algo": "MXFP8", "group_size": 32}
    assert "model.language_model.layers.0.self_attn.b_proj" not in qlayers
    assert "model.language_model.layers.3.self_attn.kv_b_proj" not in qlayers

    out_idx = json.loads((dst / "model.safetensors.index.json").read_text())
    assert "model.language_model.layers.0.self_attn.q_proj.weight_scale" in out_idx["weight_map"]
    assert "model.language_model.layers.0.self_attn.b_proj.weight_scale" not in out_idx["weight_map"]


def test_prepared_projection_group_zero_copy_output_and_validation():
    from hyperflare_recipes.components.linear import PreparedProjectionGroup

    class _FakeProj:
        def __init__(self, width: int, fill: float):
            self.width = width
            self.fill = fill

        def __call__(self, source, *, weight=None):
            val = self.fill if weight is None else float(weight)
            return torch.full((source.shape[0], self.width), val, dtype=source.dtype, device=source.device)

    grp = object.__new__(PreparedProjectionGroup)
    grp.output = None
    grp.output_layout = ((4, 10), (10, 1), torch.bfloat16)
    grp.widths = (6, 4)
    grp.projections = [_FakeProj(6, 1.0), _FakeProj(4, 2.0)]

    src = torch.zeros((4, 8), dtype=torch.bfloat16)
    backing = torch.zeros((8, 10), dtype=torch.bfloat16)
    out_slice = backing[:4]
    res = grp(src, output=out_slice)
    assert res is out_slice
    assert torch.all(backing[:4, :6] == 1.0)
    assert torch.all(backing[:4, 6:] == 2.0)
    assert torch.all(backing[4:] == 0.0)

    # Weight override works with caller-supplied output slice.
    out_slice2 = backing[4:]
    res2 = grp(src, output=out_slice2, weight=types.SimpleNamespace(parts=(3.0, 4.0)))
    assert res2 is out_slice2
    assert torch.all(backing[4:, :6] == 3.0)
    assert torch.all(backing[4:, 6:] == 4.0)

    # Mismatched shape or stride is rejected explicitly.
    with pytest.raises(ValueError, match="prepared shape, strides, dtype, and device"):
        grp(src, output=torch.zeros((4, 12), dtype=torch.bfloat16)[:, :10])


def test_kda_zero_copy_segment_for_grouped_mxfp8():
    from hyperflare_recipes.components.linear import PreparedProjectionGroup, UnquantizedProjection
    from hyperflare_recipes.recipes.glm53.kda import PreparedGlmKDA

    kda = object.__new__(PreparedGlmKDA)
    kda.projections = {
        "input_proj": object.__new__(PreparedProjectionGroup),
        "f_b_proj": object.__new__(UnquantizedProjection),
    }
    assert kda.zero_copy_segment is True

    kda.projections["f_b_proj"] = object()
    assert kda.zero_copy_segment is False
