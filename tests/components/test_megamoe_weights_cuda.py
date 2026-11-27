# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native MegaMoE source planes preserve checkpoint bytes and expert identity."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from hyperflare_recipes.components.megamoe_weights import (
    load_mxfp4_planes, pack_mxfp4_planes, load_shared_fp8_planes, pack_shared_fp8_planes,
)
from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def encode(scales):
    """Independent one-bit row offsets plus explicit byte exceptions."""
    rows, columns = scales.shape
    bits = (scales == 121).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    fixed = np.concatenate((np.full((rows // 16, 16), 120, np.uint8),
                            packed.reshape(rows // 16, -1)), axis=1).reshape(-1)
    positions = np.flatnonzero((scales != 120) & (scales != 121)).astype(np.uint32)
    exceptions = positions | (scales.reshape(-1)[positions].astype(np.uint32) << 24)
    return torch.from_numpy(fixed), torch.from_numpy(exceptions)


def decode(reader, name, rows, columns):
    fixed = reader.get_tensor(name + ".mxfp4_csf_fixed").numpy().reshape(rows // 16, -1)
    base = fixed[:, :16].reshape(rows, 1)
    bits = np.unpackbits(fixed[:, 16:].reshape(rows, -1), axis=1, bitorder="little")[:, :columns]
    values = (base + bits).copy()
    exceptions = reader.get_tensor(name + ".mxfp4_csf_exceptions").numpy()
    values.reshape(-1)[exceptions & 0xFFFFFF] = exceptions >> 24
    return torch.from_numpy(values)


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(807)
    tensors, expected = {}, {}
    for expert in range(2):
        for projection in ("w1", "w3", "w2"):
            rows, columns = (128, 256) if projection == "w2" else (256, 128)
            name = f"experts.{expert}.{projection}"
            weights = torch.from_numpy(rng.integers(0, 256, (rows, columns // 2), dtype=np.uint8))
            scales = rng.integers(120, 122, (rows, columns // 32), dtype=np.uint8)
            scales.reshape(-1)[[0, scales.size // 2, scales.size - 1]] = [110, 130, 125]
            tensors[name + ".weight"] = weights
            fixed, exceptions = encode(scales)
            tensors[name + ".scale.mxfp4_csf_fixed"] = fixed
            tensors[name + ".scale.mxfp4_csf_exceptions"] = exceptions
            expected[name] = (weights, torch.from_numpy(scales))
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    return tmp_path, expected


def assert_planes(planes, prefixes, expected, first, last):
    for index, prefix in enumerate(prefixes):
        gate, up, down = (expected[prefix + "." + projection] for projection in ("w1", "w3", "w2"))
        for actual, wanted in (
            (planes.w13[index], torch.cat((gate[0][first:last], up[0][first:last]))),
            (planes.s13[index], torch.cat((gate[1][first:last], up[1][first:last]))),
            (planes.w2[index], down[0][:, first // 2:last // 2]),
            (planes.s2[index], down[1][:, first // 32:last // 32]),
        ):
            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)


def undo_gate_up(value):
    groups, rows, columns = value.shape
    tiled = value.reshape(groups, rows // 16, 2, 8, columns)
    return torch.cat((tiled[:, :, 0].reshape(groups, rows // 2, columns),
                      tiled[:, :, 1].reshape(groups, rows // 2, columns)), dim=1)


def undo_scale_rows(value):
    groups, rows, columns = value.shape
    return value.reshape(groups, rows // 128, 32, 4, columns).transpose(2, 3).reshape(
        groups, rows, columns).contiguous().view(torch.uint8)


@pytest.mark.parametrize("interval", [(0, 256), (64, 192)])
def test_csf_and_native_scales_preserve_weight_bytes_and_partitions(source, interval):
    root, expected = source
    # Reverse checkpoint order to exercise canonical expert-to-row assignment.
    prefixes = ("experts.1", "experts.0")
    with DirectWeightLoader(inspect_indexed_checkpoint(root), "cuda:0") as loader:
        planes = load_mxfp4_planes(loader, prefixes, hidden_size=128, intermediate_size=256,
                                  intermediate_range=interval)
    assert_planes(planes, prefixes, expected, *interval)
    native = {name + suffix: values[index] for name, values in expected.items()
              for index, suffix in enumerate((".weight", ".scale"))}
    save_file(native, str(root / "weights.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.nbytes for t in native.values())},
        "weight_map": {name: "weights.safetensors" for name in native}}))
    with DirectWeightLoader(inspect_indexed_checkpoint(root), "cuda:0") as loader:
        ordinary = load_mxfp4_planes(loader, prefixes, hidden_size=128, intermediate_size=256,
                                    intermediate_range=interval)
    assert_planes(ordinary, prefixes, expected, *interval)


@pytest.mark.parametrize("defect, message", [
    ("missing", "incomplete MXFP4 scale encoding"),
    ("ambiguous", "exactly one scale representation"),
    ("mixed", "one scale encoding per expert group"),
])
def test_invalid_scale_groups_are_rejected(source, defect, message):
    root, expected = source
    tensors = load_file(root / "weights.safetensors")
    name = "experts.0.w1.scale"
    if defect in ("missing", "mixed"):
        del tensors[name + ".mxfp4_csf_exceptions"]
    if defect == "mixed":
        del tensors[name + ".mxfp4_csf_fixed"]
    if defect in ("ambiguous", "mixed"):
        tensors[name] = expected["experts.0.w1"][1]
    save_file(tensors, str(root / "weights.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": {key: "weights.safetensors" for key in tensors}}))
    with DirectWeightLoader(inspect_indexed_checkpoint(root), "cuda:0") as loader:
        with pytest.raises(ValueError, match=message):
            load_mxfp4_planes(loader, ("experts.0", "experts.1"), hidden_size=128, intermediate_size=256)


def test_kernel_weight_layout_roundtrips_native_checkpoint_bytes(source, tmp_path):
    pytest.importorskip("deep_gemm", reason="requires the megamoe preparation extra")
    from hyperflare_recipes.components.megamoe_kernel import initialize

    initialize(tmp_path / "kernels")
    root, _ = source
    with DirectWeightLoader(inspect_indexed_checkpoint(root), "cuda:0") as loader:
        planes = load_mxfp4_planes(loader, ("experts.0", "experts.1"),
                                  hidden_size=128, intermediate_size=256)
    packed = pack_mxfp4_planes(planes)

    for actual, expected in (
        (undo_gate_up(packed.gate_up[0].cpu().view(torch.uint8)), planes.w13),
        (undo_gate_up(undo_scale_rows(packed.gate_up[1].cpu())), planes.s13),
        (packed.down[0].cpu().view(torch.uint8), planes.w2),
        (undo_scale_rows(packed.down[1].cpu()), planes.s2),
    ):
        torch.testing.assert_close(actual, expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("layer", [0, 20, 39])
def test_checkpoint_planes_match_independent_cpu_decode(layer):
    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires the DeepSeek V4.1 Flash lossless-CSF checkpoint")
    root = Path(snapshot)
    manifest = inspect_indexed_checkpoint(root)
    prefixes = tuple(f"layers.{layer}.ffn.experts.{expert}" for expert in (383, 0))
    expected = {}
    for prefix in prefixes:
        for projection in ("w1", "w3", "w2"):
            name = prefix + "." + projection
            rows, columns = (5120, 2304) if projection == "w2" else (2304, 5120)
            with safe_open(root / manifest.tensors[name + ".weight"].shard, framework="pt") as reader:
                expected[name] = (reader.get_tensor(name + ".weight").view(torch.uint8),
                                  decode(reader, name + ".scale", rows, columns // 32))
    with DirectWeightLoader(manifest, "cuda:0", prefixes=tuple(p + "." for p in prefixes)) as loader:
        planes = load_mxfp4_planes(loader, prefixes, hidden_size=5120, intermediate_size=2304)
    assert_planes(planes, prefixes, expected, 0, 2304)


@pytest.mark.parametrize("layer", [0, 39])
def test_shared_checkpoint_keeps_fp8_weights_and_exact_scale_values(layer, tmp_path):
    pytest.importorskip("deep_gemm", reason="requires the megamoe preparation extra")
    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires the DeepSeek V4.1 Flash lossless-CSF checkpoint")
    from hyperflare_recipes.components.megamoe_kernel import initialize

    initialize(tmp_path / "kernels")
    root = Path(snapshot)
    manifest = inspect_indexed_checkpoint(root)
    prefix = f"layers.{layer}.ffn.shared_experts"
    with DirectWeightLoader(manifest, "cuda:0", prefixes=(prefix + ".",)) as loader:
        planes = load_shared_fp8_planes(loader, prefix, hidden_size=5120,
                                        intermediate_size=2304, block_size=(32, 32))
    packed = pack_shared_fp8_planes(planes)
    actual = (
        undo_gate_up(packed.gate_up[0].cpu().view(torch.uint8)[None])[0],
        undo_gate_up(undo_scale_rows(packed.gate_up[1].cpu()[None]))[0],
        packed.down[0].cpu().view(torch.uint8),
        undo_scale_rows(packed.down[1].cpu()[None])[0],
    )
    expected = []
    for projection in ("w1", "w3", "w2"):
        name = prefix + "." + projection
        with safe_open(root / manifest.tensors[name + ".weight"].shard, framework="pt") as reader:
            expected.append((reader.get_tensor(name + ".weight").view(torch.uint8),
                             reader.get_tensor(name + ".scale").view(torch.uint8).repeat_interleave(32, dim=0)))
    wanted = (torch.cat((expected[0][0], expected[1][0])), torch.cat((expected[0][1], expected[1][1])),
              expected[2][0], expected[2][1])
    for value, reference in zip(actual, wanted, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
