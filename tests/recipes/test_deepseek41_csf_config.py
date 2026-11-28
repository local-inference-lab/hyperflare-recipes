# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless scale storage preserves the recipe's native arithmetic admission."""

import pytest

from hyperflare_recipes.recipes.deepseek41.config import ModelConfig


@pytest.fixture
def configuration():
    return {
        "model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
        "quantization_config": dict(quant_method="mxfp4_csf", format_version=1,
            weight_scale_encoding="csf", activation_scheme="dynamic", weight_block_size=[32, 32],
            scale_fmt="ue8m0", expert_dtype="fp4"),
        "text_config": dict(hidden_size=5120, num_hidden_layers=40, num_attention_heads=64,
            num_key_value_heads=1, head_dim=512, qk_rope_head_dim=64, q_lora_rank=1280,
            o_lora_rank=1024, o_groups=8, vocab_size=129280, moe_intermediate_size=2304,
            n_routed_experts=384, n_shared_experts=1, num_experts_per_tok=6,
            scoring_func="sqrtsoftplus", index_n_heads=32, index_head_dim=128, index_topk=512,
            candidate_topk_blocks=2048, candidate_block_size=8, hc_mult=4, sliding_window=128,
            engram_max_ngram_size=4, engram_n_heads=8, engram_head_dim=256,
            compress_ratios=[0] * 2 + [2] * 18 + [1] * 20 + [0] * 3,
            kv_source_layer_ids=[2, 8, 14, 20], index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
            candidate_source_layer_id=20, num_nextn_predict_layers=3, dspark_block_size=5,
            dspark_noise_token_id=128799, dspark_target_layer_ids=[37, 38, 39],
            dspark_markov_rank=256, dspark_n_routed_experts=128, dspark_num_experts_per_tok=3),
    }


def test_native_and_lossless_csf_keep_identical_model_geometry(configuration):
    compressed = ModelConfig.from_dict(configuration)
    configuration["quantization_config"] = dict(quant_method="fp8", activation_scheme="dynamic",
        weight_block_size=[32, 32], scale_fmt="ue8m0", expert_dtype="fp4")
    native = ModelConfig.from_dict(configuration)
    assert compressed.text == native.text
    assert compressed.sources == native.sources
    assert compressed.draft == native.draft


@pytest.mark.parametrize("field,value", [("format_version", 0), ("format_version", 2),
    ("format_version", True), ("weight_scale_encoding", None), ("weight_scale_encoding", "int8"),
    ("quant_method", "nvfp4"), ("weight_block_size", [128, 128]), ("scale_fmt", "e4m3"),
    ("expert_dtype", "fp8"), ("activation_scheme", "static")])
def test_invalid_csf_contract_is_rejected(configuration, field, value):
    configuration["quantization_config"][field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(configuration)
