#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Offline MXFP8 attention weight quantizer for GLM-5.3-Flash checkpoints.

Converts BF16 attention projections in a ModelOpt MIXED_PRECISION GLM-5.3-Flash checkpoint
to MXFP8 (E4M3 values + per-32 UE8M0 block scales, ``quant_algo="MXFP8", group_size=32``)
while preserving fused projection grouping (KDA ``q_proj/k_proj/v_proj`` and MLA
``q_a_proj/kv_a_proj_with_mqa``).

Target projections:
- KDA layers (0..44 where ``layer % 4 != 3``): ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``
- MLA layers (0..44 where ``layer % 4 == 3``, plus layer 45 when ``--include-mtp`` is set):
  ``q_a_proj``, ``kv_a_proj_with_mqa``, ``q_b_proj``, ``o_proj`` (plus ``kv_b_proj`` when
  ``--include-kv-b-proj`` is passed)

KDA gates (``b_proj``, ``f_a_proj``, ``f_b_proj``, ``g_a_proj``, ``g_b_proj``), indexer
projections (``indexer.wq_b``, ``indexer.wk``, ``indexer.weights_proj``), norms, convolutions,
dense/shared MLPs, and already-quantized routed MoE experts are left untouched. Unmodified
safetensors shards are hardlinked (with automatic copy fallback) so only shards containing
converted attention projections are rewritten.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
import torch


TEXT_PREFIX = "model.language_model"
FP8_E4M3_MAX = 448.0

KDA_TARGET_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLA_TARGET_PROJECTIONS = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")


def target_attention_prefixes(
    num_hidden_layers: int = 45,
    *,
    include_mtp: bool = False,
    include_kv_b_proj: bool = False,
) -> tuple[str, ...]:
    """Return ordered GLM-5.3 attention linear prefixes eligible for MXFP8 conversion."""
    prefixes: list[str] = []
    for layer in range(num_hidden_layers):
        attn = f"{TEXT_PREFIX}.layers.{layer}.self_attn"
        if layer % 4 != 3:
            names = KDA_TARGET_PROJECTIONS
        else:
            names = MLA_TARGET_PROJECTIONS + (("kv_b_proj",) if include_kv_b_proj else ())
        for name in names:
            prefixes.append(f"{attn}.{name}")
    if include_mtp:
        attn = f"{TEXT_PREFIX}.layers.{num_hidden_layers}.self_attn"
        names = MLA_TARGET_PROJECTIONS + (("kv_b_proj",) if include_kv_b_proj else ())
        for name in names:
            prefixes.append(f"{attn}.{name}")
    return tuple(prefixes)


def quantization_entry() -> dict[str, object]:
    """Return the ModelOpt ``quantized_layers`` metadata entry for MXFP8."""
    return {"quant_algo": "MXFP8", "group_size": 32}


@torch.no_grad()
def quantize_mxfp8(weight: torch.Tensor, *, label: str = "weight") -> dict[str, torch.Tensor]:
    """Quantize a BF16 [N, K] weight to MXFP8 E4M3 values + UE8M0 [N, K/32] uint8 scales."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError(f"{label}: expected 2D BF16 weight, got dtype={weight.dtype}, shape={tuple(weight.shape)}")
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if rows <= 0 or cols <= 0 or cols % 32 != 0:
        raise ValueError(f"{label}: shape {(rows, cols)} must be positive with K divisible by 32")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"{label}: BF16 weight contains non-finite values")
    blocked = weight.float().reshape(rows, cols // 32, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(max_abs > 0, max_abs / FP8_E4M3_MAX, torch.ones_like(max_abs))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127.0, 127.0)
    scale_u8 = (exponent + 127.0).to(torch.uint8)
    scale_f32 = torch.exp2(exponent)
    q = (blocked / scale_f32.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    values = q.to(torch.float8_e4m3fn).reshape(rows, cols).contiguous()
    return {"weight": values, "weight_scale": scale_u8.contiguous()}


def dequantize_mxfp8(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Dequantize MXFP8 ``{weight, weight_scale}`` back to FP32 for verification."""
    values = tensors["weight"].float()
    scales = torch.exp2(tensors["weight_scale"].to(torch.float32) - 127.0)
    return (values.reshape(values.shape[0], -1, 32) * scales.unsqueeze(-1)).reshape_as(values)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _update_quantization_config(
    config: dict[str, object],
    converted_prefixes: list[str],
) -> dict[str, object]:
    """Return a deep copy of ``config`` with MXFP8 ``quantized_layers`` entries added."""
    updated = json.loads(json.dumps(config))
    qc = updated.get("quantization_config")
    if not isinstance(qc, dict):
        qc = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {}}
        updated["quantization_config"] = qc
    if qc.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError(f"expected MIXED_PRECISION checkpoint quantization_config, got {qc.get('quant_algo')!r}")
    layers = qc.setdefault("quantized_layers", {})
    if not isinstance(layers, dict):
        raise ValueError("quantization_config.quantized_layers must be a JSON object")
    entry = quantization_entry()
    for prefix in converted_prefixes:
        layers[prefix] = dict(entry)
    return updated


def quantize_checkpoint(
    src_dir: Path | str,
    dst_dir: Path | str,
    *,
    include_mtp: bool = False,
    include_kv_b_proj: bool = False,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a new GLM-5.3 checkpoint directory with MXFP8 attention projections."""
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()
    if src == dst:
        raise ValueError("output checkpoint directory must differ from source directory")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {dst} (pass --overwrite to replace)")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    config_path = src / "config.json"
    index_path = src / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = dict(index["weight_map"])

    text_cfg = config.get("text_config", config)
    num_hidden_layers = int(text_cfg.get("num_hidden_layers", 45))
    targets = target_attention_prefixes(
        num_hidden_layers,
        include_mtp=include_mtp,
        include_kv_b_proj=include_kv_b_proj,
    )

    quant_layers = config.get("quantization_config", {}).get("quantized_layers", {})
    shard_targets: dict[str, list[str]] = defaultdict(list)
    converted: list[str] = []
    skipped_already_quantized: list[str] = []
    missing: list[str] = []
    for prefix in targets:
        wkey = f"{prefix}.weight"
        if wkey not in weight_map:
            missing.append(prefix)
            continue
        if prefix in quant_layers:
            skipped_already_quantized.append(prefix)
            continue
        shard_targets[weight_map[wkey]].append(prefix)
        converted.append(prefix)

    # Link or copy all non-config, non-index, non-modified-shard top-level files.
    modified_shards = set(shard_targets)
    for item in sorted(src.iterdir()):
        if item.name in ("config.json", "model.safetensors.index.json", ".cache") or item.name in modified_shards:
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, symlinks=True)
        else:
            _link_or_copy(item, dst / item.name)

    new_weight_map = dict(weight_map)
    bytes_before = 0
    bytes_after = 0
    total_size = int(index.get("metadata", {}).get("total_size", 0))
    quant_device = torch.device(device)

    for shard_name in sorted(modified_shards):
        prefixes_in_shard = set(shard_targets[shard_name])
        weight_keys_to_convert = {f"{p}.weight": p for p in prefixes_in_shard}
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(src / shard_name), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key not in weight_keys_to_convert:
                    out_tensors[key] = tensor
                    continue
                prefix = weight_keys_to_convert[key]
                before = tensor.numel() * tensor.element_size()
                qdict = quantize_mxfp8(tensor.to(quant_device), label=prefix)
                after = 0
                for suffix, qtensor in qdict.items():
                    qcpu = qtensor.cpu().contiguous()
                    out_key = f"{prefix}.{suffix}"
                    out_tensors[out_key] = qcpu
                    new_weight_map[out_key] = shard_name
                    after += qcpu.numel() * qcpu.element_size()
                bytes_before += before
                bytes_after += after
        save_file(out_tensors, str(dst / shard_name), metadata=metadata)

    new_index = dict(index)
    new_index["weight_map"] = new_weight_map
    if total_size > 0:
        new_meta = dict(index.get("metadata", {}))
        new_meta["total_size"] = total_size - bytes_before + bytes_after
        new_index["metadata"] = new_meta
    (dst / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2) + "\n")

    new_config = _update_quantization_config(config, converted)
    (dst / "config.json").write_text(json.dumps(new_config, indent=2) + "\n")

    return {
        "mode": "mxfp8",
        "src": str(src),
        "dst": str(dst),
        "converted_count": len(converted),
        "skipped_already_quantized": len(skipped_already_quantized),
        "missing": len(missing),
        "rewritten_shards": len(modified_shards),
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "bytes_saved": bytes_before - bytes_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize GLM-5.3 BF16 attention projections to MXFP8.")
    parser.add_argument("src", type=Path, help="Source GLM-5.3 checkpoint directory")
    parser.add_argument("dst", type=Path, help="Destination checkpoint directory")
    parser.add_argument("--include-mtp", action="store_true", help="Also convert layer 45 MTP MLA projections")
    parser.add_argument("--include-kv-b-proj", action="store_true", help="Also convert MLA kv_b_proj to MXFP8")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination directory if it exists")
    args = parser.parse_args(argv)

    summary = quantize_checkpoint(
        args.src,
        args.dst,
        include_mtp=args.include_mtp,
        include_kv_b_proj=args.include_kv_b_proj,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
