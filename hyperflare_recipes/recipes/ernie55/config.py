# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Geometry, hybrid HCA/DSA layout, and bundled DSpark contracts for ERNIE 5.5 Flash Preview."""

import json
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "baidu/ERNIE-5.5-Flash-Preview"
REVISION = "ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe"


@dataclass(frozen=True)
class Compressor:
    layer: int
    ratio: int
    head_dim: int
    index: bool = False

    @property
    def prefix(self):
        return f"layer{self.layer}.{'index_' if self.index else ''}compressor"

    @property
    def overlap(self):
        return self.ratio == 4

    @property
    def state_shape(self):
        factor = 2 if self.overlap else 1
        return self.ratio * factor, self.head_dim * factor


@dataclass(frozen=True)
class DSparkConfig:
    layers: int
    block_size: int
    noise_token: int
    # Zero-based target decoder layer indices whose post-mHC stream mean feeds DSpark.
    # Checkpoint `dspark_target_layer_ids: [40, 41, 42]` are 0-based (final 3 layers of 43).
    target_layers: tuple[int, ...]
    markov_rank: int
    experts: int
    top_k: int


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if config.get("model_type") != "ernie5_5_moe" or config.get("architectures") != [
            "Ernie5_5_MoeForCausalLM"
        ]:
            raise ValueError("ERNIE 5.5 Flash Preview requires Ernie5_5_MoeForCausalLM")
        for name, expected in {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "first_k_dense_replace": 1,
            "intermediate_size": 14336,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_nope_head_dim": 448,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hybrid_mla_num_attention_heads": 64,
            "hybrid_mla_q_lora_rank": 1024,
            "hybrid_mla_kv_lora_rank": 512,
            "hybrid_mla_qk_nope_head_dim": 192,
            "hybrid_mla_qk_rope_head_dim": 64,
            "hybrid_mla_v_head_dim": 256,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "vocab_size": 201216,
            "moe_intermediate_size": 2048,
            "moe_latent_size": 2048,
            "n_routed_experts": 512,
            "n_shared_experts": 1,
            "num_experts_per_tok": 10,
            "scoring_func": "sigmoid",
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "sliding_window": 128,
            "multimax": "lm_head",
            "dspark_markov_rank": 256,
            "dspark_block_size": 7,
            "dspark_noise_token_id": 1024,
        }.items():
            if config.get(name) != expected:
                raise ValueError(
                    f"ERNIE 5.5 Flash Preview {name} requires {expected}, got {config.get(name)}"
                )
        if not config.get("moe_split_feature_routing") or not config.get("routed_scaling_factor_learnable"):
            raise ValueError("ERNIE 5.5 Flash Preview requires split-feature routing with learnable expert scales")
        if not config.get("use_gated_attn") or not config.get("use_vha_attention") or not config.get("latent_moe_use_norm"):
            raise ValueError("ERNIE 5.5 Flash Preview requires gated attention, VHA post-mix, and latent MoE norm")
        quant = config["quantization_config"]
        if (
            quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != [128, 128]
            or quant.get("scale_fmt") != "ue8m0"
            or quant.get("expert_dtype") != "mxfp4"
            or quant.get("activation_scheme") != "dynamic"
        ):
            raise ValueError("ERNIE 5.5 Flash Preview requires block-128 FP8 dense weights and native MXFP4 experts")
        ratios = config.get("compress_ratios", [])
        if (
            len(ratios) != config["num_hidden_layers"]
            or any(type(r) is not int or r not in (128, -2) for r in ratios)
        ):
            raise ValueError("ERNIE 5.5 Flash Preview requires 43 layers with C128 HCA or -2 DSA compress_ratios")
        targets = config.get("dspark_target_layer_ids", [])
        if (
            not targets
            or any(type(i) is not int or not 0 <= i < config["num_hidden_layers"] for i in targets)
            or targets != sorted(set(targets))
        ):
            raise ValueError("ERNIE 5.5 Flash Preview requires sorted 0-based dspark_target_layer_ids")
        return cls(config, quant)

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def vocab_size(self):
        return self.text["vocab_size"]

    def is_dsa_layer(self, layer):
        return self.text["compress_ratios"][layer] <= 0

    @property
    def hca_layers(self):
        return tuple(i for i in range(self.layers) if not self.is_dsa_layer(i))

    @property
    def dsa_layers(self):
        return tuple(i for i in range(self.layers) if self.is_dsa_layer(i))

    @property
    def draft(self):
        c = self.text
        # Checkpoint stores 0-based layer outputs [40, 41, 42] (final 3 layers of 43).
        targets = tuple(c["dspark_target_layer_ids"])
        return DSparkConfig(
            layers=len(targets),
            block_size=c["dspark_block_size"],
            noise_token=c["dspark_noise_token_id"],
            target_layers=targets,
            markov_rank=c["dspark_markov_rank"],
            experts=c["n_routed_experts"],
            top_k=c["num_experts_per_tok"],
        )

    @property
    def compressors(self):
        return tuple(
            Compressor(layer, 128, self.text["head_dim"])
            for layer in range(self.layers)
            if not self.is_dsa_layer(layer)
        )
