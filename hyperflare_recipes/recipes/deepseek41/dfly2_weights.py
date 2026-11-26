# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DFly2-B6 draft weights with optional online MXFP8 conversion."""

from dataclasses import dataclass, replace

import torch

from hyperflare_recipes.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
from hyperflare_recipes.components.weights import (
    DirectWeightLoader,
    MLPWeights,
    quantize_bf16_linear_as_mxfp8,
)
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.persistent import allocated_tensor_bytes, persistent_cat, persistent_empty
from ignition.preparation.rebuild_manifest import weight_container

from .weights import ExpertWeights, LayerWeights


def _pack_wo_mxfp8(wo_a: torch.Tensor, wo_b: torch.Tensor, *, groups: int, rank: int, group_width: int, hidden: int):
    """Quantize BF16 grouped WO-A/WO-B weights into native b12x MXFP8 WOProjection layout."""
    from b12x.gemm._shared.wo_mxfp8 import WOProjectionMXFP8Weights, quantize_mxfp8_rows_torch

    wo_a_grd = wo_a.view(groups, rank, group_width)
    wo_a_src = wo_a_grd.permute(1, 2, 0).contiguous()
    if groups == 1:
        wo_a_src = wo_a_src[:, :, 0]
    packed_a = quantize_mxfp8_rows_torch(wo_a_src, retain_scale_rows=True, allocate=persistent_empty)
    packed_b = quantize_mxfp8_rows_torch(wo_b.contiguous(), retain_scale_rows=True, allocate=persistent_empty)
    return WOProjectionMXFP8Weights(
        wo_a=packed_a,
        wo_b=packed_b,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )


def _pad_rows_bf16(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    if tensor.shape[0] == target_rows:
        return tensor
    if tensor.shape[0] > target_rows:
        raise ValueError(f"cannot pad {tensor.shape[0]} rows down to {target_rows}")
    pad = torch.zeros((target_rows - tensor.shape[0], tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=0)


def _pad_cols_bf16(tensor: torch.Tensor, target_cols: int) -> torch.Tensor:
    if tensor.shape[1] == target_cols:
        return tensor
    if tensor.shape[1] > target_cols:
        raise ValueError(f"cannot pad {tensor.shape[1]} cols down to {target_cols}")
    pad = torch.zeros((tensor.shape[0], target_cols - tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=1)


def _slice_padded_2d(loader, name: str, *, rows=None, cols=None, temporary: bool = True) -> torch.Tensor:
    """Load a 2D BF16 tensor slice with zero padding beyond checkpoint extents."""
    src_shape = loader._sources[name].shape
    full_r, full_c = int(src_shape[0]), int(src_shape[1])
    r0, r1 = (0, full_r) if rows is None else rows
    c0, c1 = (0, full_c) if cols is None else cols
    rr0, rr1 = min(r0, full_r), min(r1, full_r)
    cc0, cc1 = min(c0, full_c), min(c1, full_c)
    if rr0 == r0 and rr1 == r1 and cc0 == c0 and cc1 == c1:
        return loader.load_tensor(name, extent=(slice(r0, r1), slice(c0, c1)), temporary=temporary)
    alloc = torch.empty if temporary else persistent_empty
    out = alloc((r1 - r0, c1 - c0), dtype=torch.bfloat16, device=loader.device)
    out.zero_()
    if rr1 > rr0 and cc1 > cc0:
        part = loader.load_tensor(name, extent=(slice(rr0, rr1), slice(cc0, cc1)), temporary=True)
        out[: rr1 - rr0, : cc1 - cc0].copy_(part)
    return out


def _slice_padded_1d(loader, name: str, *, rows, dtype=None, temporary: bool = False) -> torch.Tensor:
    src_shape = loader._sources[name].shape
    full_r = int(src_shape[0])
    r0, r1 = rows
    rr0, rr1 = min(r0, full_r), min(r1, full_r)
    if rr0 == r0 and rr1 == r1:
        return loader.load_tensor(name, dtype=dtype, extent=slice(r0, r1), temporary=temporary)
    part = loader.load_tensor(name, dtype=dtype, extent=slice(rr0, rr1), temporary=True) if rr1 > rr0 else None
    out_dtype = dtype or (part.dtype if part is not None else torch.float32)
    alloc = torch.empty if temporary else persistent_empty
    out = alloc((r1 - r0,), dtype=out_dtype, device=loader.device)
    out.zero_()
    if part is not None:
        out[: rr1 - rr0].copy_(part)
    return out


def _slice_padded_2d_f32(loader, name: str, *, rows, temporary: bool = False) -> torch.Tensor:
    src_shape = loader._sources[name].shape
    full_r, full_c = int(src_shape[0]), int(src_shape[1])
    r0, r1 = rows
    rr0, rr1 = min(r0, full_r), min(r1, full_r)
    if rr0 == r0 and rr1 == r1:
        return loader.load_tensor(name, extent=(slice(r0, r1), slice(None)), temporary=temporary)
    alloc = torch.empty if temporary else persistent_empty
    out = alloc((r1 - r0, full_c), dtype=torch.float32, device=loader.device)
    out.zero_()
    if rr1 > rr0:
        part = loader.load_tensor(name, extent=(slice(rr0, rr1), slice(None)), temporary=True)
        out[: rr1 - rr0].copy_(part)
    return out


@weight_container(("layers", "mtp_fusion", "norm", "correction", "draft_head"))
@dataclass(frozen=True)
class DFly2Weights:
    layers: tuple[LayerWeights, ...]
    mtp_fusion: dict
    norm: torch.Tensor
    correction: dict
    draft_head: object | None = None
    confidence_head: torch.Tensor | None = None
    allocated_bytes: int = 0

    @classmethod
    def load(
        cls,
        snapshot,
        config,
        device,
        *,
        shard,
        target_head,
        progress=None,
        expert_scale_compression=None,
        draft_head_quantization=None,
        draft_body_quantization=None,
    ):
        if draft_body_quantization not in (None, "mxfp8"):
            raise ValueError("DFly2 draft body quantization must be None or 'mxfp8'")
        use_mxfp8 = draft_body_quantization == "mxfp8"
        manifest = inspect_indexed_checkpoint(snapshot)
        text = config.text
        draft_cfg = config.draft
        hidden = text["hidden_size"]
        first_g, last_g = shard.padded_partition(text["o_groups"])
        local_groups = last_g - first_g
        heads_per_group = text["num_attention_heads"] // text["o_groups"]
        local_heads = local_groups * heads_per_group
        head_dim = text["head_dim"]
        o_rank = text["o_lora_rank"]
        q_rank = text["q_lora_rank"]
        head_rows = (first_g * heads_per_group * head_dim, last_g * heads_per_group * head_dim)
        head_ids = (first_g * heads_per_group, last_g * heads_per_group)
        group_rows = (first_g * o_rank, last_g * o_rank)
        inter_range = shard.partition(text["moe_intermediate_size"], alignment=32)
        corr_inter_range = shard.padded_partition(draft_cfg.correction_intermediate, alignment=128)
        local_corr_inter = corr_inter_range[1] - corr_inter_range[0]

        def pack_linear(weight: torch.Tensor):
            if use_mxfp8:
                return quantize_bf16_linear_as_mxfp8(weight, retain_scale_rows=True)
            return persistent_empty(weight.shape, dtype=weight.dtype, device=weight.device).copy_(weight)

        before = allocated_tensor_bytes(device)
        scale_scratch = {}
        layers = []
        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load DFly2 draft weights") as loader:
            for i in range(draft_cfg.layers):
                p = f"model.layers.{i}"
                attn = p + ".attn"
                wq_a = _slice_padded_2d(loader, attn + ".wq_a.weight", temporary=True)
                wkv = _slice_padded_2d(loader, attn + ".wkv.weight", temporary=True)
                qkv_a = pack_linear(torch.cat((wq_a, wkv), dim=0))
                wq_b = pack_linear(_slice_padded_2d(loader, attn + ".wq_b.weight", rows=head_rows, temporary=True))
                wo_a = _slice_padded_2d(loader, attn + ".wo_a.weight", rows=group_rows, temporary=True)
                wo_b = _slice_padded_2d(loader, attn + ".wo_b.weight", cols=group_rows, temporary=True)
                if use_mxfp8:
                    output_w = _pack_wo_mxfp8(
                        wo_a, wo_b, groups=local_groups, rank=o_rank,
                        group_width=heads_per_group * head_dim, hidden=hidden,
                    )
                else:
                    output_w = {
                        "wo_a": persistent_empty(wo_a.shape, dtype=wo_a.dtype, device=device).copy_(wo_a),
                        "wo_b": persistent_empty(wo_b.shape, dtype=wo_b.dtype, device=device).copy_(wo_b),
                        "groups": local_groups,
                    }
                attn_dict = {
                    "qkv_a": qkv_a,
                    "q_b": wq_b,
                    "output": output_w,
                    "q_norm": loader.load_tensor(attn + ".q_norm.weight"),
                    "kv_norm": loader.load_tensor(attn + ".kv_norm.weight"),
                    "sink": _slice_padded_1d(loader, attn + ".attn_sink", rows=head_ids),
                }

                ffn = p + ".ffn"
                experts = loader.load_mxfp4_experts(
                    [f"{ffn}.experts.{e}" for e in range(draft_cfg.experts)],
                    hidden_size=hidden,
                    intermediate_size=text["moe_intermediate_size"],
                    intermediate_range=inter_range,
                    swiglu_limit=text["swiglu_limit"],
                    scale_name="scale",
                    projection_names=("w3", "w1", "w2"),
                    scale_compression=expert_scale_compression,
                    scale_scratch=scale_scratch,
                )
                sw1 = _slice_padded_2d(loader, ffn + ".shared_experts.w1.weight", rows=inter_range, temporary=True)
                sw3 = _slice_padded_2d(loader, ffn + ".shared_experts.w3.weight", rows=inter_range, temporary=True)
                sw2 = _slice_padded_2d(loader, ffn + ".shared_experts.w2.weight", cols=inter_range, temporary=True)
                shared = MLPWeights(pack_linear(torch.cat((sw1, sw3), dim=0)), pack_linear(sw2))
                routed = ExpertWeights(
                    loader.load_tensor(ffn + ".gate.weight"),
                    loader.load_tensor(ffn + ".gate.bias"),
                    experts,
                    shared,
                    loader.load_tensor(ffn + ".gate.bias_vl")
                    if config.vision is not None and f"{ffn}.gate.bias_vl" in loader._sources
                    else None,
                )
                residual = {
                    name: loader.load_tensor(f"{p}.{name}")
                    for name in (
                        "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                        "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
                        "attn_norm.weight", "ffn_norm.weight",
                    )
                }
                layers.append(LayerWeights(attn_dict, routed, residual, None))

            mtp_fusion = {
                "enorm": loader.load_tensor("model.mtp_fusion.enorm.weight"),
                "hnorm": loader.load_tensor("model.mtp_fusion.hnorm.weight"),
                "eh_proj": pack_linear(_slice_padded_2d(loader, "model.mtp_fusion.eh_proj.weight", temporary=True)),
                "gate": loader.load_tensor("model.mtp_fusion.gate"),
            }
            final_norm = loader.load_tensor("model.final_norm.weight")

            cp = "model.hidden_correction"
            ca = cp + ".attention"
            # Pre-split all [*, 2*hidden] correction projections into h_norm and e_norm halves
            # and fuse them into two stacked matrices: h_proj runs ONCE across all 6 block slots,
            # e_proj runs ONCE per sequential step (5x fewer GEMMs, 2x less weight traffic!).
            gate_w = _slice_padded_2d(loader, f"{cp}.gate_proj.weight", rows=corr_inter_range, temporary=True)
            up_w = _slice_padded_2d(loader, f"{cp}.up_proj.weight", rows=corr_inter_range, temporary=True)
            qin_w = _slice_padded_2d(loader, f"{ca}.q_in.weight", temporary=True)
            kvin_w = _slice_padded_2d(loader, f"{ca}.kv_in.weight", temporary=True)
            bg_w = _slice_padded_2d(loader, f"{ca}.block_gate.weight", rows=head_ids, temporary=True)

            raw_fused_rows = 2 * local_corr_inter + q_rank + head_dim + local_heads
            fused_rows = (raw_fused_rows + 127) // 128 * 128 if use_mxfp8 else raw_fused_rows
            h_cat = _pad_rows_bf16(
                torch.cat((gate_w[:, :hidden], up_w[:, :hidden], qin_w[:, :hidden], kvin_w[:, :hidden], bg_w[:, :hidden]), dim=0),
                fused_rows,
            )
            e_cat = _pad_rows_bf16(
                torch.cat((gate_w[:, hidden:], up_w[:, hidden:], qin_w[:, hidden:], kvin_w[:, hidden:], bg_w[:, hidden:]), dim=0),
                fused_rows,
            )
            down_w = _slice_padded_2d(loader, f"{cp}.down_proj.weight", cols=corr_inter_range, temporary=True)
            c_wqb = _slice_padded_2d(loader, f"{ca}.wq_b.weight", rows=head_rows, temporary=True)
            c_woa = _slice_padded_2d(loader, f"{ca}.wo_a.weight", rows=group_rows, temporary=True)
            c_wob = _slice_padded_2d(loader, f"{ca}.wo_b.weight", cols=group_rows, temporary=True)
            if use_mxfp8:
                c_output = _pack_wo_mxfp8(
                    c_woa, c_wob, groups=local_groups, rank=o_rank,
                    group_width=heads_per_group * head_dim, hidden=hidden,
                )
            else:
                c_output = {
                    "wo_a": persistent_empty(c_woa.shape, dtype=c_woa.dtype, device=device).copy_(c_woa),
                    "wo_b": persistent_empty(c_wob.shape, dtype=c_wob.dtype, device=device).copy_(c_wob),
                    "groups": local_groups,
                }
            correction = {
                "hidden_norm": loader.load_tensor(f"{cp}.hidden_norm.weight"),
                "embed_norm": loader.load_tensor(f"{cp}.embed_norm.weight"),
                "h_proj": pack_linear(h_cat),
                "e_proj": pack_linear(e_cat),
                "fused_rows": fused_rows,
                "local_corr_inter": local_corr_inter,
                "local_heads": local_heads,
                "down_proj": pack_linear(down_w),
                "q_norm": loader.load_tensor(f"{ca}.q_norm.weight"),
                "wq_b": pack_linear(c_wqb),
                "kv_norm": loader.load_tensor(f"{ca}.kv_norm.weight"),
                "attn_sink": _slice_padded_1d(loader, f"{ca}.attn_sink", rows=head_ids),
                "output": c_output,
                "block_sink": _slice_padded_1d(loader, f"{ca}.block_sink", rows=head_ids),
                "block_rel_bias": _slice_padded_2d_f32(loader, f"{ca}.block_rel_bias", rows=head_ids),
                "block_gate_bias": _slice_padded_2d_f32(loader, f"{ca}.block_gate_bias", rows=head_ids),
            }

        draft_head = None
        if draft_head_quantization == "nvfp4":
            draft_head = nvfp4_draft_head(target_head)
        elif draft_head_quantization == "mxfp8":
            draft_head = mxfp8_draft_head(target_head)
        return cls(
            layers=tuple(layers),
            mtp_fusion=mtp_fusion,
            norm=final_norm,
            correction=correction,
            draft_head=draft_head,
            confidence_head=None,
            allocated_bytes=allocated_tensor_bytes(device) - before,
        )
