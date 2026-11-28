# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 checkpoint validation before b12x weight preparation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType

from ignition.preparation.checkpoint import (
    CheckpointError,
    IndexedCheckpoint,
    TensorMetadata,
    _inspect_indexed_checkpoint,
    _parse_json,
)
from ignition.preparation.quantization import expert_format, linear_format, quantized_layers

from . import checkpoint_schema as qwen38


@dataclass(frozen=True, slots=True)
class CheckpointManifest(IndexedCheckpoint):
    metadata_sha256: Mapping[str, str]
    headers_sha256: str
    generation_config: Mapping

    def to_dict(self) -> dict:
        sizes = Counter()
        counts = Counter()
        csf_bytes = Counter()
        for tensor in self.tensors.values():
            sizes[tensor.component] += tensor.nbytes
            counts[tensor.component] += 1
            if ".csf32_" in tensor.name or ".nvfp4_csf_" in tensor.name:
                csf_bytes[tensor.name.rsplit(".", 1)[1]] += tensor.nbytes
        gpu_bytes = sum(sizes[key] for key in ("decoder_experts", "decoder_other", "vision"))
        return {
            "model_id": qwen38.MODEL_ID,
            "reference_revision": qwen38.REVISION,
            "architecture": qwen38.ARCHITECTURE,
            "snapshot": str(self.snapshot),
            "validation": {
                "tensor_payloads_read": False,
                "tensor_payloads_verified": False,
                "checks": [
                    "config", "generation_config", "quantization_map", "tensor_names",
                    "shapes", "dtypes", "file_ranges", "index_coverage", "header_fingerprints",
                    "frontend_asset_fingerprints",
                ],
                "unverified": [
                    "tensor_values", "scale_values", "csf_exception_contents",
                    "ple_hash_constants", "inference_correctness",
                ],
            },
            "metadata_sha256": dict(self.metadata_sha256),
            "frontend_assets": {
                name: {"path": str(self.snapshot / name), "sha256": self.metadata_sha256[name]}
                for name in qwen38.FRONTEND_SHA256
            },
            "headers_sha256": self.headers_sha256,
            "tensor_count": len(self.tensors),
            "shard_count": len(self.shards),
            "payload_bytes": sum(sizes.values()),
            "gpu_stored_bytes": gpu_bytes,
            "ngram_stored_bytes": sizes["ple_tables"],
            "excluded_stored_bytes": sizes["mtp"],
            "components": {
                key: {"tensor_count": counts[key], "stored_bytes": sizes[key]}
                for key in sorted(sizes)
            },
            "geometry": {
                "text": dict(qwen38.TEXT_GEOMETRY),
                "vision": dict(qwen38.VISION_GEOMETRY),
                "gdn_layers": [i for i in range(48) if i % 4 != 3],
                "sparse_attention_layers": list(range(3, 48, 4)),
            },
            "quantization": {
                "module_counts": dict(Counter(
                    entry["quant_algo"] for entry in quantized_layers(
                        _parse_json((self.snapshot / "config.json").read_bytes(), "config.json")
                    ).values()
                )),
                "excluded_mtp_aliases": [qwen38.EXCLUDED_QUANTIZATION_ALIAS],
                "csf_stored_bytes": dict(csf_bytes),
            },
            "ple": {
                "prefix": qwen38.PLE_PREFIX,
                "table_planes": 128,
                "rows_per_plane": 2500012,
                "packed_width": 80,
                "scale_width": 10,
                "logical_embedding_width": 160,
            },
            "generation_config": dict(self.generation_config),
            "shards": [
                {
                    "path": shard.path, "file_size": shard.file_size,
                    "header_size": shard.header_size, "header_sha256": shard.header_sha256,
                    "tensor_count": shard.tensor_count,
                }
                for shard in self.shards
            ],
        }


def serving_context(text: dict) -> int:
    """The context a text configuration's rotary parameters serve: native, or YaRN `factor` times native."""
    native = qwen38.TEXT_GEOMETRY["max_position_embeddings"]
    rope = text.get("rope_parameters")
    if not isinstance(rope, dict):
        raise CheckpointError("text_config.rope_parameters: missing")
    for field, value in qwen38.ROPE_GEOMETRY.items():
        if rope.get(field) != value:
            raise CheckpointError(f"text_config.rope_parameters.{field}: expected {value!r}, got {rope.get(field)!r}")
    kind = rope.get("rope_type")
    scaling = set(rope) - set(qwen38.ROPE_GEOMETRY) - {"rope_type"}
    if kind == "default":
        if scaling:
            raise CheckpointError(f"text_config.rope_parameters.{min(scaling)}: not a default rope parameter")
        return native
    if kind != "yarn":
        raise CheckpointError(f"text_config.rope_parameters.rope_type: expected 'default' or 'yarn', got {kind!r}")
    if scaling - qwen38.YARN_KEYS:
        raise CheckpointError(
            f"text_config.rope_parameters.{min(scaling - qwen38.YARN_KEYS)}: not a YaRN rope parameter")
    if rope.get("original_max_position_embeddings") != native:
        raise CheckpointError("text_config.rope_parameters.original_max_position_embeddings: expected "
                              f"{native!r}, got {rope.get('original_max_position_embeddings')!r}")
    factor = rope.get("factor")
    context = factor * native if type(factor) in (int, float) and math.isfinite(factor) else 0
    if context != int(context) or context % 64 or not native < context <= qwen38.MAX_SCALED_CONTEXT:
        raise CheckpointError(
            f"text_config.rope_parameters.factor: {factor!r} must extend {native:,} tokens to a 64-aligned"
            f" context of at most {qwen38.MAX_SCALED_CONTEXT:,}")
    return int(context)


def _validate_config(config: dict) -> None:
    if (
        config.get("architectures") != [qwen38.ARCHITECTURE]
        or config.get("model_type") != "qwen4_exp"
    ):
        raise CheckpointError(f"expected architecture {qwen38.ARCHITECTURE}")
    for section, expected in (
        ("text_config", qwen38.TEXT_GEOMETRY), ("vision_config", qwen38.VISION_GEOMETRY)
    ):
        actual = config.get(section)
        if not isinstance(actual, dict):
            raise CheckpointError(f"missing {section}")
        for field, value in expected.items():
            if actual.get(field) != value:
                raise CheckpointError(
                    f"{section}.{field}: expected {value!r}, got {actual.get(field)!r}"
                )
    serving_context(config["text_config"])
    expected_layers = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)]
    if config["text_config"].get("layer_types") != expected_layers:
        raise CheckpointError(
            "text_config.layer_types: expected 36 GDN and 12 interleaved sparse-attention layers"
        )
    try:
        actual = quantized_layers(config)
        expected = qwen38.expected_quantization()
        for name, recipe in expected.items():
            if name.endswith(".experts"):
                fmt, mode = expert_format(actual, name)
                if name.startswith("mtp.") and fmt == "nvfp4_csf":
                    raise CheckpointError("Qwen MTP experts require ordinary NVFP4 or MXFP4 storage")
            elif name == f"{qwen38.PLE_PREFIX}.ngram_embedding":
                if actual.get(name) != recipe:
                    raise CheckpointError(
                        f"quantized_layers.{name}: expected {recipe!r}, got {actual.get(name)!r}"
                    )
            else:
                storage = linear_format(actual, name)
                if storage not in ("mxfp8", "mxfp4", "nvfp4"):
                    raise CheckpointError(
                        f"{name}: Qwen dense projections do not support {storage} storage"
                    )
    except ValueError as exc:
        raise CheckpointError(str(exc)) from exc
    extra = actual.keys() - expected.keys()
    if extra:
        raise CheckpointError(f"unmapped quantization entry: {min(extra)}")


def _validate_tensors(tensors: Mapping[str, TensorMetadata], quantization=None) -> None:
    unmatched = set(tensors)
    for name, dtype, shape in qwen38.expected_tensors(quantization):
        tensor = tensors.get(name)
        if tensor is None:
            raise CheckpointError(f"missing required tensor: {name}")
        if (
            tensor.dtype != dtype or len(tensor.shape) != len(shape)
            or any(expected is not None and actual != expected
                   for actual, expected in zip(tensor.shape, shape))
        ):
            raise CheckpointError(
                f"{name}: expected {dtype} {shape}, got {tensor.dtype} {tensor.shape}"
            )
        if shape == (None,) and tensor.shape[0] > 102400:
            raise CheckpointError(f"{name}: CSF exception count exceeds the scale element count")
        unmatched.remove(name)
    for name in tuple(unmatched):
        if not name.endswith(".input_scale") or quantization is None:
            continue
        entry = quantization.get(name.removesuffix(".input_scale"), {})
        if entry.get("quant_algo") != "NVFP4" or entry.get("weight_scale_encoding") is not None:
            continue
        tensor = tensors[name]
        if tensor.dtype != "F32" or tensor.shape not in ((), (1,)):
            raise CheckpointError(f"{name}: expected one F32 activation scale")
        unmatched.remove(name)
    if unmatched:
        raise CheckpointError(f"unmapped tensor: {min(unmatched)}")


def _inspect_frontend_assets(snapshot: Path) -> dict[str, str]:
    fingerprints = {}
    for name, expected in qwen38.FRONTEND_SHA256.items():
        with (snapshot / name).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected:
            raise CheckpointError(
                f"{name}: SHA256 does not match reference revision {qwen38.REVISION}"
            )
        fingerprints[name] = digest
    return fingerprints


def inspect_qwen38_checkpoint(snapshot: str | Path) -> CheckpointManifest:
    """Validate model geometry and declared storage; tensor payloads are not read.

    The returned shard_paths and index_path can be passed to b12x's weight
    session. Exclude mtp.* while routing the remaining tensors into the recipe.
    File paths remain caller-owned and must not change between inspection and
    loading. The loader must perform its own range and source-lifetime checks.
    """
    snapshot = Path(snapshot).expanduser().absolute()
    metadata = _inspect_frontend_assets(snapshot)

    def read_json(name):
        raw = (snapshot / name).read_bytes()
        metadata[name] = hashlib.sha256(raw).hexdigest()
        return _parse_json(raw, name)

    config = read_json("config.json")
    _validate_config(config)
    index = read_json("model.safetensors.index.json")
    generation = read_json("generation_config.json")
    indexed = _inspect_indexed_checkpoint(snapshot, index)
    tensors, shards = indexed.tensors, indexed.shards
    _validate_tensors(tensors, quantized_layers(config))
    header_digest = hashlib.sha256()
    for shard in shards:
        header_digest.update(shard.path.encode() + b"\0" + bytes.fromhex(shard.header_sha256))
    return CheckpointManifest(
        snapshot, MappingProxyType(tensors), tuple(shards), MappingProxyType(metadata),
        header_digest.hexdigest(), MappingProxyType(generation),
    )
