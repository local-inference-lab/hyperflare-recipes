# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless source planes for MXFP4 MegaMoE preparation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Mxfp4Planes:
    """Packed gate/up and down weights with row-major native UE8M0 scales.

    These are preparation intermediates. MegaMoE's weight transform consumes
    gate rows followed by up rows; its final interleaving is a separate step.
    No activation quantization or expert execution occurs here.
    """

    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor


@dataclass(frozen=True)
class MegaMoeWeights:
    """MegaMoE's eight-row gate/up interleave and packed UTCCP scale layout."""

    gate_up: tuple[torch.Tensor, torch.Tensor]
    down: tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class SharedFp8Planes:
    """FP8 shared-expert matrices with equivalent native scales for each row."""

    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor


def _pack_scales(value, rows, columns, count):
    import deep_gemm

    fp32 = (value.to(torch.int32) << 23).view(torch.float32)
    return deep_gemm.transform_sf_into_required_layout(fp32, rows, columns, (1, 32), count)


def pack_mxfp4_planes(planes, *, temporary=False):
    """Apply DeepGEMM's reference MXFP4 transform without requantizing weights.

    Matches vLLM's SM100 MegaMoE preparation: reinterpret native exponent bytes
    as FP32 powers of two, pack the native scale layout, then interleave gate
    and up rows. Call only during preparation. By default, returned planes use
    persistent storage for capture and export. ``temporary=True`` leaves them
    in disposable storage for copying into final tier rows.
    """
    import deep_gemm
    from ignition.preparation.persistent import persistent_copy

    count, doubled, packed_hidden = planes.w13.shape
    hidden, intermediate = packed_hidden * 2, doubled // 2
    expected = ((count, doubled, hidden // 32), (count, hidden, intermediate // 2),
                (count, hidden, intermediate // 32))
    if (doubled % 2 or intermediate % 128 or hidden % 128
            or any(t.shape != shape for t, shape in zip((planes.s13, planes.w2, planes.s2), expected))
            or any(t.dtype != torch.uint8 or t.device != planes.w13.device or not t.is_contiguous()
                   for t in (planes.w13, planes.s13, planes.w2, planes.s2))):
        raise ValueError("MegaMoE packing requires matched contiguous MXFP4 planes and 128-aligned widths")

    gate_up, down = deep_gemm.transform_weights_for_mega_moe(
        (planes.w13.view(torch.int8), _pack_scales(planes.s13, 2 * intermediate, hidden, count)),
        (planes.w2.view(torch.int8), _pack_scales(planes.s2, hidden, intermediate, count)),
    )
    if temporary:
        return MegaMoeWeights(gate_up, down)
    return MegaMoeWeights(tuple(persistent_copy(t) for t in gate_up),
                          tuple(persistent_copy(t) for t in down))


def load_mxfp4_planes(loader, prefixes, *, hidden_size, intermediate_size,
                     intermediate_range=None, projection_names=("w1", "w3", "w2"),
                     scale_name="scale"):
    """Expand checkpoint CSF once into exact native scale bytes for preparation.

    ``projection_names`` is gate, up, down. Packed FP4 bytes are only sliced
    and copied. Native and CSF source groups produce the same row-major planes.
    Temporary GPU storage is owned by the result and the caller must retain it
    until the consuming weight transform has completed.
    """
    from b12x._lib.quant._csf_prepare import expand_source_scales
    from hyperflare_recipes.components.weights import _aligned_range

    loader._require_open()
    prefixes = tuple(prefixes)
    if (not prefixes or len(set(prefixes)) != len(prefixes)
            or hidden_size <= 0 or hidden_size % 128):
        raise ValueError("MegaMoE source planes require unique experts and 128-aligned hidden width")
    if len(projection_names) != 3 or len(set(projection_names)) != 3:
        raise ValueError("MegaMoE source projections must name distinct gate, up, and down weights")
    first, last = _aligned_range(intermediate_range, intermediate_size, 32)
    local, count = last - first, len(prefixes)
    gate, up, down = projection_names
    names = [f"{p}.{projection}.{scale_name}" for p in prefixes for projection in projection_names]
    encodings = []
    for name in names:
        native = name in loader._sources
        choices = []
        for suffixes in (("mxfp4_csf_fixed", "mxfp4_csf_exceptions"),
                         ("csf32_bases", "csf32_codes", "csf32_page_bases", "csf32_offsets", "csf32_exceptions")):
            present = [f"{name}.{suffix}" in loader._sources for suffix in suffixes]
            if any(present):
                if not all(present):
                    raise ValueError(f"incomplete MXFP4 scale encoding: {name}")
                choices.append(suffixes[0])
        if len(choices) + int(native) != 1:
            raise ValueError(f"MXFP4 source needs exactly one scale representation: {name}")
        encodings.append(choices[0] if choices else "native")
    if len(set(encodings)) != 1:
        raise ValueError("MegaMoE source planes require one scale encoding per expert group")

    def empty(shape):
        return torch.empty(shape, dtype=torch.uint8, device=loader.device)

    planes = Mxfp4Planes(empty((count, 2 * local, hidden_size // 2)),
                         empty((count, 2 * local, hidden_size // 32)),
                         empty((count, hidden_size, local // 2)),
                         empty((count, hidden_size, local // 32)))
    for index, prefix in enumerate(prefixes):
        for projection, target, scale_target, rows, columns in (
            (gate, planes.w13[index, :local], planes.s13[index, :local], intermediate_size, hidden_size),
            (up, planes.w13[index, local:], planes.s13[index, local:], intermediate_size, hidden_size),
            (down, planes.w2[index], planes.s2[index], hidden_size, intermediate_size),
        ):
            name = f"{prefix}.{projection}"
            source = loader._sources[f"{name}.weight"]
            if source.dtype not in (torch.uint8, torch.int8) or source.shape != (rows, columns // 2):
                raise ValueError(f"invalid packed MXFP4 source geometry: {name}")
            source = source.view(torch.uint8)
            loader._copy(target, source[:, first // 2:last // 2] if projection == down else source[first:last])
            if encodings[0] == "native":
                scales = loader._sources[f"{name}.{scale_name}"]
                if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu) or scales.shape != (rows, columns // 32):
                    raise ValueError(f"invalid native MXFP4 scale geometry: {name}")
                scales = scales.view(torch.uint8)
                loader._copy(scale_target, scales[:, first // 32:last // 32]
                             if projection == down else scales[first:last])
    sources = ()
    if encodings[0] != "native":
        sources = (
            loader._queue_csf_sources(
                [f"{p}.{projection}.{scale_name}" for p in prefixes for projection in (gate, up)],
                rows=intermediate_size, columns=hidden_size // 32, quantization="mxfp4",
                temporary=True, join=2, row_range=(first, last)),
            loader._queue_csf_sources(
                [f"{p}.{down}.{scale_name}" for p in prefixes], rows=hidden_size,
                columns=intermediate_size // 32, quantization="mxfp4", temporary=True,
                column_range=(first // 32, last // 32)),
        )
    loader._session.flush()
    for source, target in zip(sources, (planes.s13, planes.s2)):
        expand_source_scales(source, target, swizzled=False)
    return planes


def load_shared_fp8_planes(loader, prefix, *, hidden_size, intermediate_size,
                           block_size, projection_names=("w1", "w3", "w2"), scale_name="scale"):
    """Retain shared FP8 weights and broadcast block scales to native 1x32 groups.

    Broadcasting the checkpoint's scale bytes preserves their numerical values;
    no shared weight is converted to FP4 to enable the fused kernel.
    """
    loader._require_open()
    block_m, block_k = block_size
    if (block_m <= 0 or block_k <= 0 or block_k % 32 or hidden_size <= 0
            or intermediate_size <= 0 or hidden_size % 128 or intermediate_size % 128
            or len(projection_names) != 3 or len(set(projection_names)) != 3):
        raise ValueError("shared MegaMoE weights require 128-aligned widths and native block-FP8 scales")
    weights, scales = [], []
    for index, projection in enumerate(projection_names):
        rows, columns = (hidden_size, intermediate_size) if index == 2 else (intermediate_size, hidden_size)
        name = prefix + "." + projection
        weight = loader.load_tensor(name + ".weight", temporary=True)
        scale = loader.load_tensor(name + "." + scale_name, temporary=True)
        if (weight.dtype != torch.float8_e4m3fn or weight.shape != (rows, columns)
                or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
                or scale.shape != ((rows + block_m - 1) // block_m, (columns + block_k - 1) // block_k)):
            raise ValueError(f"shared FP8 checkpoint geometry or scale format mismatch: {name}")
        weights.append(weight)
        scales.append(scale.view(torch.uint8).repeat_interleave(block_m, dim=0)[:rows]
                      .repeat_interleave(block_k // 32, dim=1)[:, :columns // 32].contiguous())
    return SharedFp8Planes(torch.cat(weights[:2]), torch.cat(scales[:2]), weights[2], scales[2])


def pack_shared_fp8_planes(planes):
    """Pack shared FP8 matrices for the same MegaMoE launch as routed MXFP4 experts."""
    import deep_gemm
    from ignition.preparation.persistent import persistent_copy

    rows, hidden = planes.w13.shape
    intermediate = rows // 2
    if (rows % 2 or intermediate % 128 or hidden % 128
            or planes.w2.shape != (hidden, intermediate)
            or planes.s13.shape != (rows, hidden // 32) or planes.s2.shape != (hidden, intermediate // 32)
            or any(t.dtype != torch.float8_e4m3fn for t in (planes.w13, planes.w2))
            or any(t.dtype != torch.uint8 for t in (planes.s13, planes.s2))
            or any(t.device != planes.w13.device or not t.is_contiguous()
                   for t in (planes.w13, planes.s13, planes.w2, planes.s2))):
        raise ValueError("shared MegaMoE packing requires matched FP8 matrices and native 1x32 scale bytes")
    gate_up, down = deep_gemm.transform_weights_for_mega_moe(
        (planes.w13, _pack_scales(planes.s13[None], rows, hidden, 1).squeeze(0)),
        (planes.w2, _pack_scales(planes.s2[None], hidden, intermediate, 1).squeeze(0)),
    )
    return MegaMoeWeights(tuple(persistent_copy(t) for t in gate_up),
                          tuple(persistent_copy(t) for t in down))


def mxfp4_weight_layouts(count, hidden, intermediate):
    """Physical value/scale layouts for transformed routed weights in either tier."""
    if (any(type(v) is not int or v <= 0 for v in (count, hidden, intermediate))
            or hidden % 128 or intermediate % 128):
        raise ValueError("MegaMoE weight storage requires positive rows and 128-aligned widths")
    result = []
    for rows, columns in ((2 * intermediate, hidden), (hidden, intermediate)):
        result.append(((count, rows, columns // 2), torch.int8,
                       (rows * (columns // 2), columns // 2, 1)))
        result.append(((count, rows, columns // 128), torch.int32,
                       (rows * (columns // 128), 1, rows)))
    return tuple(result)


def allocate_mxfp4_weights(count, hidden, intermediate, *, device):
    """Allocate only the physical HBM expert rows, ready for bounded loading."""
    from ignition.preparation.persistent import persistent_empty
    from ignition.preparation.tensor_metadata import record_numeric_tensor

    values = tuple(persistent_empty(shape, dtype, device, stride=stride)
                   for shape, dtype, stride in mxfp4_weight_layouts(count, hidden, intermediate))
    for value in values:
        record_numeric_tensor(value)
    return MegaMoeWeights(values[:2], values[2:])


def load_mxfp4_weights_into(loader, prefixes, destination, *, hidden_size, intermediate_size,
                            batch_experts=8, intermediate_range=None,
                            projection_names=("w1", "w3", "w2"), scale_name="scale"):
    """Load and transform bounded batches directly into final physical tier rows.

    Destination tensors may be HBM tensors or registered CPU host views. Scale
    layout and native bytes are preserved. The caller owns placement order;
    this function neither chooses experts nor changes routing mappings.
    """
    from hyperflare_recipes.components.weights import _aligned_range

    loader._require_open()
    prefixes = tuple(prefixes)
    if (not prefixes or len(set(prefixes)) != len(prefixes)
            or type(batch_experts) is not int or batch_experts <= 0):
        raise ValueError("MegaMoE bounded loading requires unique expert prefixes and a positive batch size")
    first, last = _aligned_range(intermediate_range, intermediate_size, 32)
    targets = (*destination.gate_up, *destination.down)
    layouts = mxfp4_weight_layouts(len(prefixes), hidden_size, last - first)
    for target, (shape, dtype, stride) in zip(targets, layouts, strict=True):
        if target.shape != shape or target.dtype != dtype or target.stride() != stride:
            raise ValueError("MegaMoE bounded loading requires destination geometry and packed scale layout to match")
        if target.device.type != "cpu" and target.device != loader.device:
            raise ValueError("MegaMoE destinations must be host views or use the loader's CUDA device")
    if len({t.device for t in targets}) != 1:
        raise ValueError("MegaMoE physical weight planes must occupy one tier")
    for start in range(0, len(prefixes), batch_experts):
        stop = min(start + batch_experts, len(prefixes))
        planes = load_mxfp4_planes(loader, prefixes[start:stop], hidden_size=hidden_size,
            intermediate_size=intermediate_size, intermediate_range=(first, last),
            projection_names=projection_names, scale_name=scale_name)
        packed = pack_mxfp4_planes(planes, temporary=True)
        for target, source in zip(targets, (*packed.gate_up, *packed.down), strict=True):
            target[start:stop].copy_(source)
        # Release this batch before allocating the next source/transform planes.
        del source, packed, planes
    return destination
