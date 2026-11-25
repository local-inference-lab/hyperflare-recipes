# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mini's stored expert tensors and required activation quantization."""

from dataclasses import dataclass

import torch

from ignition.preparation.rebuild_manifest import weight_container


def expert_format(precision, layer):
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    formats = []
    for projection in ("w13", "w2"):
        name = prefix + projection + "_weight"
        groups = [g for g in precision["model"]["parameter_groups"] if name in g["tensors"]]
        if len(groups) != 1:
            raise ValueError(f"{name} requires exactly one precision declaration")
        group = groups[0]
        fmt = group["compute_dtype"]
        expected = (dict(amax_type="per_token", block_size=16, format="nvfp4_e2m1", rounding="nearest",
                         scale_dtype="float8_e4m3fn", scale_selection_rule="fouroversix_mae") if fmt == "nvfp4" else
                    dict(block_size=32, format="mxfp8_e4m3", rounding="nearest", scale_dtype="ue8m0"))
        if fmt not in ("nvfp4", "mxfp8") or group.get("required_activation_quantization") != expected:
            raise ValueError(f"{name} has unsupported activation quantization")
        formats.append(fmt)
    if formats[0] != formats[1]:
        raise ValueError("Mini expert projections must use the same precision")
    return formats[0]


@weight_container(("w13", "s13", "g13", "w2", "s2", "g2", "s13_exc", "s13_slab_offsets", "s2_exc", "s2_slab_offsets"))
@dataclass(frozen=True)
class MiniExperts:
    w13: torch.Tensor
    s13: torch.Tensor
    g13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    g2: torch.Tensor
    nvfp4: bool
    s13_exc: torch.Tensor | None = None
    s13_slab_offsets: torch.Tensor | None = None
    s2_exc: torch.Tensor | None = None
    s2_slab_offsets: torch.Tensor | None = None


def load_experts(loader, layer, *, shard, expert_scale_compression=None):
    from ignition.preparation.persistent import persistent_copy

    # Per-token maxima cover the complete intermediate vector. Partitioning
    # intermediate channels would change the quantizer without a global maximum.
    if shard.world_size != 1:
        raise ValueError("Mini expert loading requires one rank to preserve per-token activation scales")
    if expert_scale_compression not in (None, "csf"):
        raise ValueError(f"unsupported Inkling-2-Mini expert scale compression: {expert_scale_compression!r}")
    fmt = expert_format(loader.config.quantization, layer)
    c = loader.config.text
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    pack, block = (2, 16) if fmt == "nvfp4" else (1, 32)
    use_csf = fmt == "nvfp4" and expert_scale_compression == "csf"
    values = []
    csf_extras = []
    for projection, n, k in (("w13", 2 * c.intermediate, c.hidden), ("w2", c.hidden, c.intermediate)):
        weight = loader.load_tensor(prefix + projection + "_weight")
        scale = loader.load_tensor(prefix + projection + "_weight_scale", temporary=use_csf)
        if weight.shape != (c.experts, n, k // pack) or scale.shape != (c.experts, n, k // block):
            raise ValueError(f"{prefix}{projection} has invalid expert storage geometry")
        if weight.dtype != (torch.uint8 if fmt == "nvfp4" else torch.float8_e4m3fn):
            raise ValueError(f"{prefix}{projection} has invalid weight storage dtype")
        if scale.dtype != (torch.float8_e4m3fn if fmt == "nvfp4" else torch.uint8):
            raise ValueError(f"{prefix}{projection} has invalid scale storage dtype")
        if use_csf:
            from b12x._lib.quant.csf_encode import encode_csf_source
            from b12x.moe import situ_moe

            csf = encode_csf_source(scale.view(torch.uint8), quantization="nvfp4")
            exc, slab_offsets = situ_moe.partition_csf_slab_offsets(
                csf.components[1], csf.starts, c.experts, n, k // block
            )
            scale = persistent_copy(csf.components[0])
            csf_extras.extend((persistent_copy(exc), persistent_copy(slab_offsets)))
        if fmt == "nvfp4":
            amax = loader.load_tensor(prefix + projection + "_weight_amax", temporary=True)
            denominator = loader.load_tensor(prefix + projection + "_weight_outer_scale_denominator", temporary=True)
            if (amax.dtype != torch.float32 or denominator.dtype != torch.float32
                    or amax.shape != (c.experts,) or denominator.numel() != 1
                    or not bool(torch.isfinite(amax).all() & (amax > 0).all()
                                & torch.isfinite(denominator).all() & (denominator > 0).all())):
                raise ValueError("Mini NVFP4 weight outer scales must be finite and positive")
            # Retain both operands: the reference rounds (x_amax * w_amax)
            # before multiplying by reciprocal(1536 * denominator).
            gain = persistent_copy(torch.stack((amax, denominator.expand_as(amax))))
        else:
            gain = persistent_copy(torch.ones(c.experts, device=loader.device, dtype=torch.float32))
        values.extend((weight, scale, gain))
    if use_csf:
        return MiniExperts(*values, nvfp4=True,
                           s13_exc=csf_extras[0], s13_slab_offsets=csf_extras[1],
                           s2_exc=csf_extras[2], s2_slab_offsets=csf_extras[3])
    return MiniExperts(*values, nvfp4=fmt == "nvfp4")
