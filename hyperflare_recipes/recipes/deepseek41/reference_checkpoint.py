# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Logical native tensor reads for preparing DeepInfer from an MXFP4-CSF source.

This adapter does not allocate expert residency or create an expanded checkpoint.
The source manifest retains physical compressed tensors; ``tensors`` describes
the losslessly decoded operands consumed by the reference model's loader.
It is not yet connected to a serving recipe.
"""

from dataclasses import dataclass
import json
from types import MappingProxyType

import numpy as np
import torch
from safetensors import safe_open

from ignition.preparation.checkpoint import CheckpointError, inspect_indexed_checkpoint


FIXED = ".mxfp4_csf_fixed"
EXCEPTIONS = ".mxfp4_csf_exceptions"


@dataclass(frozen=True)
class LogicalTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str


class ReferenceCheckpoint:
    """Expose native and decoded tensors without changing their checkpoint identity.

    Reads return owned CPU tensors. GPU placement, native scale packing and
    asynchronous upload lifetimes belong to the preparation caller. One scale
    matrix is expanded per call; no expanded scales are cached on this object.
    """

    def __init__(self, snapshot):
        self.source = inspect_indexed_checkpoint(snapshot)
        self.tensors = {name: LogicalTensor(name, tensor.shape, tensor.dtype)
                        for name, tensor in self.source.tensors.items()}
        self.compressed_scales = set()
        for name, fixed in self.source.tensors.items():
            if not name.endswith(FIXED):
                continue
            logical = name.removesuffix(FIXED)
            other = self.source.tensors.get(logical + EXCEPTIONS)
            weight = self.source.tensors.get(logical.removesuffix(".scale") + ".weight")
            if (not logical.endswith(".scale") or logical in self.source.tensors
                    or other is None or weight is None or weight.dtype not in ("I8", "U8")
                    or len(weight.shape) != 2):
                raise CheckpointError(f"incomplete or ambiguous MXFP4-CSF tensor: {logical}")
            rows, packed_columns = weight.shape
            columns = packed_columns // 16
            if (rows <= 0 or rows % 16 or packed_columns <= 0 or packed_columns % 16
                    or rows * columns > 1 << 24
                    or fixed.dtype != "U8" or fixed.shape != (rows * (1 + (columns + 7) // 8),)
                    or other.dtype != "U32" or len(other.shape) != 1
                    or other.shape[0] > rows * columns):
                raise CheckpointError(f"invalid MXFP4-CSF tensor geometry: {logical}")
            self.compressed_scales.add(logical)
            del self.tensors[name]
            del self.tensors[logical + EXCEPTIONS]
            self.tensors[logical] = LogicalTensor(logical, (rows, columns), "F8_E8M0")
        if any(name.endswith(EXCEPTIONS) for name in self.tensors):
            raise CheckpointError("MXFP4-CSF exception tensor lacks its fixed tensor")
        if any(".csf32_" in name for name in self.tensors):
            raise CheckpointError("the reference checkpoint reader requires native or row-base MXFP4-CSF scales")
        self.tensors = MappingProxyType(self.tensors)
        self.compressed_scales = frozenset(self.compressed_scales)

    def _physical(self, name):
        metadata = self.source.tensors[name]
        with safe_open(self.source.snapshot / metadata.shard, framework="pt", device="cpu") as reader:
            view = reader.get_slice(name)
            if tuple(view.get_shape()) != metadata.shape or view.get_dtype() != metadata.dtype:
                raise CheckpointError(f"checkpoint tensor changed after inspection: {name}")
            return reader.get_tensor(name).clone()

    def read(self, name):
        """Read exact source bytes, decoding a scale only when requested."""
        if name not in self.tensors:
            raise KeyError(name)
        if name not in self.compressed_scales:
            return self._physical(name)
        rows, columns = self.tensors[name].shape
        fixed = self._physical(name + FIXED).numpy().reshape(rows // 16, -1)
        bases = fixed[:, :16].reshape(rows, 1)
        if np.any(bases > 254):
            raise CheckpointError(f"MXFP4-CSF row base would overflow: {name}")
        selectors = np.unpackbits(fixed[:, 16:].reshape(rows, -1), axis=1,
                                  bitorder="little")[:, :columns]
        decoded = bases + selectors
        records = self._physical(name + EXCEPTIONS).numpy()
        positions = records & 0xFFFFFF
        if (np.any(positions >= rows * columns)
                or np.any(positions[1:] <= positions[:-1])):
            raise CheckpointError(f"MXFP4-CSF exceptions must be sorted, unique and in range: {name}")
        decoded.reshape(-1)[positions] = records >> 24
        return torch.from_numpy(decoded).view(torch.float8_e8m0fnu)

    def read_concat(self, names):
        names = tuple(names)
        if not names:
            raise ValueError("reference checkpoint concatenation needs at least one tensor")
        parts = [self.read(name) for name in names]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def model_config(self):
        """Build reference geometry for the operands returned by this reader.

        Source configuration and physical tensor metadata remain unchanged.
        This requires the reference parser's dictionary entry point; it does
        not make DeepInfer's native safetensors loader accept compressed files.
        """
        from .config import ModelConfig
        from deepinfer.hugging_face.config import model_config_from_dict

        raw = json.loads((self.source.metadata_path / "config.json").read_text())
        model = ModelConfig.from_dict(raw)
        if self.compressed_scales and model.quantization["quant_method"] != "mxfp4_csf":
            raise CheckpointError("compressed scale tensors require an MXFP4-CSF source declaration")
        # The reference constructor needs the decoded arithmetic format, not
        # the source codec. This descriptor is never written as a checkpoint.
        raw["quantization_config"] = dict(quant_method="fp8", expert_dtype="fp4",
            activation_scheme="dynamic", weight_block_size=[32, 32], scale_fmt="ue8m0")
        return model_config_from_dict(raw, hf_dir=self.source.metadata_path)
