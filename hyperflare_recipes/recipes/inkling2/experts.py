# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrated ModelOpt NVFP4 expert shards for Inkling2."""

import torch


def _expert_scale_vectors(sources, prefixes):
    vectors = ([], [], [], [])
    for prefix in prefixes:
        scales = {}
        for projection in ("up_proj", "gate_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                if name not in sources:
                    raise ValueError(f"{name} is required for calibrated NVFP4 activations")
                value = sources[name]
                if (value.device.type != "cpu" or value.dtype != torch.float32
                        or value.numel() != 1 or not bool((torch.isfinite(value) & (value > 0)).all())):
                    raise ValueError(f"{name} requires a positive finite CPU FP32 scalar")
                scales[projection, suffix] = value.item()
        for suffix in ("weight_scale_2", "input_scale"):
            if scales["up_proj", suffix] != scales["gate_proj", suffix]:
                raise ValueError(f"{prefix}: fused gate/up {suffix} must be identical")
        for values, key in zip(vectors, (("up_proj", "weight_scale_2"), ("down_proj", "weight_scale_2"),
                                        ("up_proj", "input_scale"), ("down_proj", "input_scale")), strict=True):
            values.append(scales[key])
    result = tuple(torch.tensor(values, dtype=torch.float32) for values in vectors)
    if not bool((result[2] == result[2][0]).all()):
        raise ValueError("Inkling2 routed experts require one shared input activation scale per layer")
    result = (*result[:2], result[2].reciprocal(), result[3].reciprocal())
    if any(not bool(torch.isfinite(value).all()) for value in result):
        raise ValueError("Inkling2 expert activation reciprocal scales must remain finite in FP32")
    return result


def load_experts(loader, layer_index, *, shard, a16_max_tokens=64, expert_scale_compression=None):
    """Load one layer's routed experts in its checkpoint precision and layout."""
    if f"model.llm.layers.{layer_index}.mlp.experts.w13_weight" in loader._sources:
        from .mini_experts import load_experts as load_mini_experts
        return load_mini_experts(loader, layer_index, shard=shard,
                                 expert_scale_compression=expert_scale_compression)
    if type(layer_index) is not int or not 0 <= layer_index < 48:
        raise ValueError("Inkling2 expert layer index must be in [0, 48)")
    prefixes = tuple(f"model.language_model.layers.{layer_index}.mlp.experts.{expert}" for expert in range(384))
    return loader.load_fp4_experts(
        prefixes, hidden_size=4608, intermediate_size=2304,
        intermediate_range=shard.partition(2304, alignment=64),
        format="nvfp4", activation_mode="a4", activation="situ", w13_layout="w13",
        a16_max_tokens=a16_max_tokens,
        scale_vectors=_expert_scale_vectors(loader._sources, prefixes),
        input_scale_name="input_scale",
    )
