# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The reference loader sees native scale bytes without an expanded checkpoint."""

import json

import pytest
import torch
from safetensors.torch import save_file

from ignition.preparation.checkpoint import CheckpointError
from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint, FIXED, EXCEPTIONS


NAME = "layers.0.ffn.experts.0.w1"


def source_tensors():
    # Deliberately include a partial selector byte and two groups of 16 rows.
    rows, columns = 32, 10
    expected = torch.empty((rows, columns), dtype=torch.uint8)
    fixed = []
    for start in range(0, rows, 16):
        fixed.extend(120 + row % 3 for row in range(start, start + 16))
        for row in range(start, start + 16):
            for first in range(0, columns, 8):
                byte = 0
                for column in range(first, min(columns, first + 8)):
                    selector = (row + column) % 2
                    expected[row, column] = 120 + row % 3 + selector
                    byte |= selector << (column - first)
                fixed.append(byte)
    overrides = [(0, 0), (columns * 16 - 1, 254), (rows * columns - 1, 255)]
    for position, value in overrides:
        expected.view(-1)[position] = value
    tensors = {
        NAME + ".weight": torch.arange(rows * columns * 16).to(torch.int8).reshape(rows, columns * 16),
        NAME + ".scale" + FIXED: torch.tensor(fixed, dtype=torch.uint8),
        NAME + ".scale" + EXCEPTIONS: torch.tensor([p | (v << 24) for p, v in overrides], dtype=torch.uint32),
        "layers.0.hc_attn_fn": torch.tensor([1.0000001192092896, -0.3333333432674408]),
    }
    return tensors, expected


def write_checkpoint(path, tensors):
    save_file(tensors, str(path / "weights.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))


def test_logical_scales_are_exact_and_reads_do_not_retain_expansion(tmp_path):
    tensors, expected = source_tensors()
    write_checkpoint(tmp_path, tensors)
    reader = ReferenceCheckpoint(tmp_path)
    scale = NAME + ".scale"
    assert reader.tensors[scale].shape == expected.shape
    assert reader.tensors[scale].dtype == "F8_E8M0"
    assert scale not in reader.source.tensors
    assert scale + FIXED not in reader.tensors
    assert not hasattr(reader.tensors[scale], "offset")
    actual = reader.read(scale)
    assert torch.equal(actual.view(torch.uint8), expected)
    actual.view(torch.uint8).zero_()
    assert torch.equal(reader.read(scale).view(torch.uint8), expected)
    assert torch.equal(reader.read(NAME + ".weight"), tensors[NAME + ".weight"])
    assert torch.equal(reader.read("layers.0.hc_attn_fn"), tensors["layers.0.hc_attn_fn"])
    assert torch.equal(reader.read_concat([scale, scale]).view(torch.uint8), torch.cat([expected, expected]))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.safetensors.index.json", "weights.safetensors"]


def test_native_scale_and_compressed_scale_have_identical_logical_values(tmp_path):
    tensors, expected = source_tensors()
    del tensors[NAME + ".scale" + FIXED], tensors[NAME + ".scale" + EXCEPTIONS]
    tensors[NAME + ".scale"] = expected.view(torch.float8_e8m0fnu)
    write_checkpoint(tmp_path, tensors)
    reader = ReferenceCheckpoint(tmp_path)
    assert not reader.compressed_scales
    assert torch.equal(reader.read(NAME + ".scale").view(torch.uint8), expected)


@pytest.mark.parametrize("defect", ["missing_fixed", "missing_exceptions", "ambiguous", "wrong_geometry", "wrong_dtype"])
def test_invalid_encoded_inventory_is_rejected(tmp_path, defect):
    tensors, expected = source_tensors()
    fixed, records = NAME + ".scale" + FIXED, NAME + ".scale" + EXCEPTIONS
    if defect == "missing_fixed":
        del tensors[fixed]
    elif defect == "missing_exceptions":
        del tensors[records]
    elif defect == "ambiguous":
        tensors[NAME + ".scale"] = expected.view(torch.float8_e8m0fnu)
    elif defect == "wrong_geometry":
        tensors[fixed] = tensors[fixed][:-1].clone()
    else:
        tensors[records] = tensors[records].to(torch.int64)
    write_checkpoint(tmp_path, tensors)
    with pytest.raises(CheckpointError, match="MXFP4-CSF"):
        ReferenceCheckpoint(tmp_path)


@pytest.mark.parametrize("defect", ["base_overflow", "duplicate", "unsorted", "out_of_range"])
def test_invalid_payload_is_rejected_before_indexing(tmp_path, defect):
    tensors, _ = source_tensors()
    if defect == "base_overflow":
        tensors[NAME + ".scale" + FIXED][0] = 255
    else:
        positions = {"duplicate": [0, 0], "unsorted": [2, 1], "out_of_range": [320]}[defect]
        tensors[NAME + ".scale" + EXCEPTIONS] = torch.tensor(positions, dtype=torch.uint32)
    write_checkpoint(tmp_path, tensors)
    reader = ReferenceCheckpoint(tmp_path)
    with pytest.raises(CheckpointError, match="MXFP4-CSF"):
        reader.read(NAME + ".scale")


def test_changed_tensor_header_is_rejected(tmp_path):
    tensors, _ = source_tensors()
    write_checkpoint(tmp_path, tensors)
    reader = ReferenceCheckpoint(tmp_path)
    tensors[NAME + ".weight"] = tensors[NAME + ".weight"].reshape(-1)
    write_checkpoint(tmp_path, tensors)
    with pytest.raises(CheckpointError, match="changed after inspection"):
        reader.read(NAME + ".weight")
