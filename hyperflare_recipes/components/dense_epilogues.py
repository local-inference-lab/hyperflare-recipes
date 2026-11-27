# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Selective dense MXFP8 epilogue helpers (SwiGLU+quant, K=64 packing, residual, Q-RoPE).

All b12x epilogue extensions are probed dynamically so hyperflare-recipes stays
compatible with older b12x revisions that lack these entry points or options.
Each fusion is gated to the row-count regimes where it improves latency on
SM120/SM121 while preserving bit-exact BF16 output parity.
"""

from __future__ import annotations

import contextlib
import inspect
import math
import weakref
from dataclasses import dataclass

import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared
from ignition.preparation.scratch import ScratchArena, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer

from hyperflare_recipes.components import sm103_projection
from hyperflare_recipes.components.native_operator import native_contract, native_plan
from hyperflare_recipes.components.rotary import _rotate_grouped

# Measured SM120 row-count cutoffs for DenseGemmKernel tail-64 Q-RoPE.
# Beyond these thresholds the epilogue proxy-fence and shared-memory read-modify-write
# cost more than a standalone Triton grouped rotary kernel.
DENSE_ROPE_MAX_ROWS_QB = 1024
DENSE_ROPE_MAX_ROWS_INDEX_Q = 128

# In auto activation mode, M <= 28 uses MXFP8 A16 TC-GEMV without quantizing activations.
# From M >= 29, dense MXFP8 execution quantizes activations and benefits from fused SwiGLU+quant.
SWIGLU_QUANT_MIN_AUTO_ROWS = 29

FUSED_SWIGLU_DOWN_COMPONENT = "flare.b12x.swiglu_mxfp8_down"

# Keyed by (plan_handle, head_dim, rope_dim, rope_split, rope_inverse).
_DENSE_ROPE_RUNNERS: dict[
    tuple[int, int, int, bool, bool], tuple[object, torch.Tensor, object]
] = {}


def _has_parameters(fn, names: tuple[str, ...]) -> bool:
    if fn is None or not callable(fn):
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in names)


def supports_swiglu_quantize_rows() -> bool:
    """True when b12x exposes the fused SwiGLU -> BF16 round -> MXFP8 row quantizer."""
    try:
        from b12x._lib.quant import mxfp8_rows
        from b12x.quantization import mxfp8
    except ImportError:
        return False
    return callable(getattr(mxfp8, "swiglu_quantize_rows", None)) and callable(
        getattr(mxfp8_rows, "_get_compiled_swiglu_mxfp8_rows_quant", None)
    )


def supports_k_alignment_64() -> bool:
    """True when b12x supports K-alignment 64 for MXFP8 weights and dense rows."""
    try:
        from b12x.gemm.blockscaled._linear import pack_mxfp8_linear_weight, pack_weight
        from b12x.quantization import mxfp8
    except ImportError:
        return False
    return (
        _has_parameters(pack_mxfp8_linear_weight, ("k_alignment",))
        and _has_parameters(pack_weight, ("k_alignment",))
        and _has_parameters(
            getattr(mxfp8, "dense_rows_storage_shapes", None), ("k_alignment",)
        )
        and _has_parameters(
            getattr(mxfp8, "dense_rows_from_bases", None), ("k_alignment",)
        )
    )


def supports_dense_residual_epilogue() -> bool:
    """True when b12x DenseGemmKernel supports an opt-in BF16 residual addend."""
    try:
        from b12x._lib.dense_gemm.compile import _get_compiled_dense_gemm
        from b12x._lib.dense_gemm.execution import dense_gemm
    except ImportError:
        return False
    return _has_parameters(dense_gemm, ("residual",)) and _has_parameters(
        _get_compiled_dense_gemm, ("has_residual",)
    )


def supports_dense_rope_epilogue() -> bool:
    """True when b12x DenseGemmKernel supports opt-in tail-64 Q-RoPE."""
    try:
        from b12x._lib.dense_gemm.compile import _get_compiled_dense_gemm
        from b12x._lib.dense_gemm.execution import dense_gemm
    except ImportError:
        return False
    return _has_parameters(
        dense_gemm,
        ("rope_cos_sin", "rope_head_dim", "rope_dim", "rope_split", "rope_inverse"),
    ) and _has_parameters(
        _get_compiled_dense_gemm,
        ("rope_head_dim", "rope_dim", "rope_split", "rope_inverse"),
    )


def supports_wo_quantize_c_multirow() -> bool:
    """True when b12x wo_projection generalizes fused wo_a quantize_c beyond M <= 8."""
    try:
        from b12x.gemm.wo_projection import _preparation as wo_prep
    except ImportError:
        return False
    try:
        source = inspect.getsource(wo_prep.plan)
    except (OSError, TypeError):
        return False
    return "_quantized_c" in source


def should_fuse_dense_rope(rows: int, *, head_dim: int, rope_dim: int = 64) -> bool:
    """Gate DenseGemmKernel tail-64 Q-RoPE strictly to measured winning regimes."""
    if type(rows) is not int or rows <= 0 or int(rope_dim) != 64:
        return False
    if int(head_dim) == 512:
        return rows <= DENSE_ROPE_MAX_ROWS_QB
    if int(head_dim) == 128:
        return rows <= DENSE_ROPE_MAX_ROWS_INDEX_Q
    return False


def should_pack_unpadded_k64(in_features: int, device) -> bool:
    """True when a 64-aligned non-128-aligned MXFP8 weight should pack with k_alignment=64."""
    return (
        in_features > 0
        and in_features % 64 == 0
        and in_features % 128 != 0
        and supports_k_alignment_64()
        and supports_swiglu_quantize_rows()
        and not sm103_projection.is_sm103(device)
    )


def dense_rows_storage_shapes_aligned(
    rows: int, columns: int, *, k_alignment: int = 128
) -> tuple[tuple[int, ...], ...]:
    """Return MXFP8 dense-row physical base shapes for 64- or 128-aligned K."""
    from b12x.quantization import mxfp8

    if k_alignment == 128:
        return mxfp8.dense_rows_storage_shapes(rows, columns)
    if _has_parameters(mxfp8.dense_rows_storage_shapes, ("k_alignment",)):
        return mxfp8.dense_rows_storage_shapes(rows, columns, k_alignment=k_alignment)
    if k_alignment != 64 or rows <= 0 or columns <= 0 or columns % 64:
        raise ValueError("dense MXFP8 rows require positive M and 64-aligned K")
    return (
        (rows, columns),
        (1, rows, columns // 32),
        (1, (rows + 127) // 128, (columns + 127) // 128, 32, 4, 4),
    )


def _scale_mma_physical_base(scale_mma: torch.Tensor) -> torch.Tensor:
    """Return contiguous uint8 physical storage backing a canonical 6D MMA scale view."""
    if scale_mma.device.type == "cuda":
        from b12x.quantization import mxfp8

        return mxfp8.dense_scale_storage(scale_mma)
    row_tiles, k_tiles = int(scale_mma.shape[2]), int(scale_mma.shape[4])
    physical_shape = (1, row_tiles, k_tiles, 32, 4, 4)
    physical_stride = (row_tiles * k_tiles * 512, k_tiles * 512, 512, 16, 4, 1)
    return scale_mma.view(torch.uint8).as_strided(physical_shape, physical_stride)


def _scale_mma_view_from_physical(scale_mma_base: torch.Tensor) -> torch.Tensor:
    """Build the canonical 6D UE8M0 MMA view over physical [1, row_tiles, k_tiles, 32, 4, 4] storage."""
    return scale_mma_base.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)


def _cuda_stream_context(stream, device: torch.device):
    if stream is None:
        return contextlib.nullcontext()
    if isinstance(stream, torch.cuda.Stream):
        return torch.cuda.stream(stream)
    if isinstance(stream, int) and device.type == "cuda":
        return torch.cuda.stream(torch.cuda.ExternalStream(stream, device=device))
    return contextlib.nullcontext()


def _stream_to_int(stream) -> int | None:
    if stream is None:
        return None
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    return int(stream)


def _compile_dense_gemm_from_lowering(
    lowering,
    device_ordinal: int,
    *,
    has_residual: bool = False,
    rope_head_dim: int = 0,
    rope_dim: int = 0,
    rope_split: bool = False,
    rope_inverse: bool = False,
):
    from b12x._lib.compile_plan import load_programs
    from b12x._lib.dense_gemm.compile import _get_compiled_dense_gemm
    from b12x._lib.utils.dtypes import get_cutlass_dtype

    extra_kwargs = {"has_residual": True} if has_residual else {}
    with torch.cuda.device(device_ordinal):
        runner = _get_compiled_dense_gemm(
            n=lowering.n,
            k=lowering.k,
            l=lowering.l,
            c_l=lowering.kernel_c_l,
            a_major="k",
            b_major="k",
            c_major="n",
            ab_dtype=get_cutlass_dtype(lowering.ab_dtype),
            sf_dtype=get_cutlass_dtype(lowering.sf_dtype),
            c_dtype=get_cutlass_dtype(lowering.c_dtype),
            alpha_dtype=get_cutlass_dtype(lowering.alpha_dtype),
            sf_vec_size=lowering.sf_vec_size,
            mma_k=lowering.mma_k,
            tile_k=lowering.tile_k,
            mma_tiler_mn=lowering.mma_tiler_mn,
            cluster_shape_mn=lowering.cluster_shape_mn,
            policy=lowering.policy,
            sm_count=lowering.sm_count,
            sm_version="sm_120",
            load_path=lowering.load_path,
            swap_ab=lowering.swap_ab,
            sfb_k_reuse=False if lowering.plain_fp8 else lowering.sfb_k_reuse,
            b_tile_major=False if lowering.plain_fp8 else lowering.b_tile_major,
            quantize_c=False,
            alpha_is_one=True if rope_head_dim else lowering.alpha_is_one,
            direct_sfa_live16=False,
            direct_m1_wo_a_inputs=False,
            plain_fp8=lowering.plain_fp8,
            block_fp8=lowering.block_fp8,
            target_occupancy_override=lowering.target_occupancy_override,
            programmatic_serialization=lowering.programmatic_serialization,
            evict_first=lowering.evict_first,
            rope_head_dim=int(rope_head_dim),
            rope_dim=int(rope_dim),
            rope_split=bool(rope_split),
            rope_inverse=bool(rope_inverse),
            device_ordinal=device_ordinal,
            **extra_kwargs,
        )
    load_programs(runner)
    return runner


def _extract_dense_rope_target(plan):
    """Extract (lowering, rows, device, alpha_one) if `plan` executes MXFP8 DenseGemm."""
    prepared = getattr(plan, "_prepared", None)
    state = getattr(prepared, "state", prepared)
    if state is None:
        return None
    core = getattr(state, "core", None)
    if core is not None and getattr(core, "lowering", None) is not None:
        lowering = core.lowering
        return lowering, int(lowering.m), core.device, core.alpha_one
    config = getattr(state, "config", None)
    query = getattr(state, "query", None)
    dense = getattr(state, "dense", None)
    if (
        config is not None
        and query is not None
        and dense is not None
        and getattr(config, "mode", None) == "quantized"
        and getattr(query, "recipe", None) == "mxfp8"
        and getattr(dense, "lowering", None) is not None
    ):
        return dense.lowering, int(query.num_tokens), state.device, dense.alpha_one
    return None


def _evict_dense_rope_runner(
    key: tuple[int, int, int, bool, bool], expected_entry: object
) -> None:
    if _DENSE_ROPE_RUNNERS.get(key) is expected_entry:
        _DENSE_ROPE_RUNNERS.pop(key, None)


def _resolve_dense_rope_runner(
    plan_handle: int,
    *,
    head_dim: int,
    rope_dim: int,
    rope_split: bool,
    rope_inverse: bool = False,
):
    key = (
        int(plan_handle),
        int(head_dim),
        int(rope_dim),
        bool(rope_split),
        bool(rope_inverse),
    )
    entry = _DENSE_ROPE_RUNNERS.get(key)
    try:
        plan = plan_from_handle(plan_handle)
    except (KeyError, ValueError, RuntimeError):
        plan = None
    if plan is not None:
        target = _extract_dense_rope_target(plan)
        if target is None:
            _DENSE_ROPE_RUNNERS.pop(key, None)
            return None
        current_lowering = target[0]
        if entry is None or entry[2] != current_lowering:
            if register_dense_gemm_rope(
                plan,
                head_dim=head_dim,
                rope_dim=rope_dim,
                rope_split=rope_split,
                rope_inverse=rope_inverse,
            ):
                entry = _DENSE_ROPE_RUNNERS.get(key)
            else:
                _DENSE_ROPE_RUNNERS.pop(key, None)
                return None
    return entry


def register_dense_gemm_rope(
    plan,
    *,
    head_dim: int,
    rope_dim: int = 64,
    rope_split: bool = True,
    rope_inverse: bool = False,
) -> bool:
    """Precompile a fused tail-64 Q-RoPE DenseGemm runner when `plan` is in a winning regime."""
    head_dim, rope_dim = int(head_dim), int(rope_dim)
    handle = getattr(plan, "handle", None)
    key = (
        None
        if handle is None
        else (int(handle), head_dim, rope_dim, bool(rope_split), bool(rope_inverse))
    )
    if not supports_dense_rope_epilogue():
        if key is not None:
            _DENSE_ROPE_RUNNERS.pop(key, None)
        return False
    target = _extract_dense_rope_target(plan)
    if target is None:
        if key is not None:
            _DENSE_ROPE_RUNNERS.pop(key, None)
        return False
    lowering, rows, device, alpha_one = target
    if not should_fuse_dense_rope(rows, head_dim=head_dim, rope_dim=rope_dim):
        if key is not None:
            _DENSE_ROPE_RUNNERS.pop(key, None)
        return False
    if (
        lowering.ab_dtype != "float8_e4m3fn"
        or lowering.sf_dtype != "float8_e8m0fnu"
        or lowering.c_dtype != "bfloat16"
        or lowering.sf_vec_size != 32
        or lowering.l != 1
        or lowering.swap_ab
        or lowering.plain_fp8
        or lowering.block_fp8
        or lowering.is_mxfp6
        or lowering.policy.split_k_slices != 1
        or lowering.n % head_dim != 0
    ):
        if key is not None:
            _DENSE_ROPE_RUNNERS.pop(key, None)
        return False
    assert key is not None
    existing = _DENSE_ROPE_RUNNERS.get(key)
    if existing is not None and existing[2] == lowering:
        return True
    runner = _compile_dense_gemm_from_lowering(
        lowering,
        int(device.index or 0),
        rope_head_dim=head_dim,
        rope_dim=rope_dim,
        rope_split=rope_split,
        rope_inverse=rope_inverse,
    )
    if alpha_one is None:
        from b12x._lib.dense_gemm.buffers import _cached_alpha_one

        alpha_one = _cached_alpha_one(device)
    entry = (runner, alpha_one, lowering)
    _DENSE_ROPE_RUNNERS[key] = entry
    try:
        weakref.finalize(plan, _evict_dense_rope_runner, key, entry)
    except TypeError:
        pass
    return True


def has_dense_gemm_rope(
    plan_handle: int,
    *,
    head_dim: int,
    rope_dim: int = 64,
    rope_split: bool = True,
    rope_inverse: bool = False,
) -> bool:
    return (
        int(plan_handle),
        int(head_dim),
        int(rope_dim),
        bool(rope_split),
        bool(rope_inverse),
    ) in _DENSE_ROPE_RUNNERS


def _angles_valid_for_dense_rope(
    angles: torch.Tensor, rows: int, rope_dim: int
) -> bool:
    return (
        angles.dtype == torch.float32
        and angles.is_contiguous()
        and (angles.numel() == 0 or angles.data_ptr() % 16 == 0)
        and angles.shape in ((rows, rope_dim), (rows, rope_dim // 2, 2))
    )


def run_encoded_dense_rope(
    values_base: torch.Tensor,
    mma_base: torch.Tensor,
    weight_values: torch.Tensor,
    weight_mma_base: torch.Tensor,
    output: torch.Tensor,
    plan_handle: int,
    angles: torch.Tensor,
    head_dim: int,
    rope_dim: int = 64,
    *,
    rope_split: bool = True,
    stream=None,
) -> bool:
    """Launch precompiled DenseGemm with fused Q-RoPE over preencoded MXFP8 bases."""
    entry = _resolve_dense_rope_runner(
        plan_handle, head_dim=head_dim, rope_dim=rope_dim, rope_split=rope_split
    )
    if entry is None:
        return False
    rows, k = int(values_base.shape[0]), int(values_base.shape[1])
    n = int(weight_values.shape[0])
    if (
        output.dtype != torch.bfloat16
        or not output.is_contiguous()
        or output.shape != (rows, n)
        or not _angles_valid_for_dense_rope(angles, rows, int(rope_dim))
    ):
        return False
    runner, alpha_one, _ = entry
    sfa = _scale_mma_view_from_physical(mma_base)
    sfb = _scale_mma_view_from_physical(weight_mma_base)
    runner(
        a_tensor_gpu=values_base.view(rows, k, 1),
        b_tensor_gpu=weight_values.view(n, k, 1),
        sfa_tensor_gpu=sfa,
        sfb_tensor_gpu=sfb,
        c_tensor_gpu=output.view(rows, n, 1),
        alpha_tensor_gpu=alpha_one,
        stream_int=_stream_to_int(stream),
        rope_cos_sin_gpu=angles,
    )
    return True


def run_quantized_dense_rope(
    source: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scales: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    plan_handle: int,
    angles: torch.Tensor,
    head_dim: int,
    rope_dim: int = 64,
    *,
    rope_split: bool = True,
    stream=None,
) -> bool:
    """Quantize BF16 rows using the prepared blockscaled quantizer and launch fused Q-RoPE DenseGemm."""
    entry = _resolve_dense_rope_runner(
        plan_handle, head_dim=head_dim, rope_dim=rope_dim, rope_split=rope_split
    )
    if entry is None:
        return False
    rows = int(source.shape[0])
    if (
        source.ndim != 2
        or source.dtype != torch.bfloat16
        or not source.is_contiguous()
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
        or not _angles_valid_for_dense_rope(angles, rows, int(rope_dim))
    ):
        return False
    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "gemm.blockscaled_precision", source.device)
    q = state.query
    if (
        state.config.mode != "quantized"
        or q.recipe != "mxfp8"
        or state.offsets is None
        or "quantize" not in state.programs
        or rows > q.num_tokens
        or workspace.numel() < state.required_workspace
    ):
        return False
    import triton
    from b12x._lib.intrinsics.reference import as_grouped_scale_view_mx
    from b12x.gemm.blockscaled._preparation import _pad_activation_scale_rows

    runner, alpha_one, lowering = entry
    sf_start, alpha_start, _, _ = state.offsets
    stored_k = q.padded_in_features
    packed_values = workspace[: rows * stored_k].view(torch.float8_e4m3fn)
    scale_size = triton.cdiv(rows, 128) * triton.cdiv(stored_k // 32, 4) * 512
    packed_scales = workspace[sf_start : sf_start + scale_size]
    alpha = workspace[alpha_start : alpha_start + 4].view(torch.float32)
    grid = (rows, triton.cdiv(stored_k // 32, 16), 1)
    state.programs["quantize"][grid](
        source,
        packed_values,
        packed_scales,
        alpha,
        alpha,
        alpha,
        rows,
        q.in_features,
        stored_k,
        False,
        False,
        32,
        16,
        state.config.fuse_swiglu,
        _pad_activation_scale_rows(q, lowering),
    )
    sfa = as_grouped_scale_view_mx(packed_scales.view(1, -1), rows, stored_k)
    sfb = (
        as_grouped_scale_view_mx(weight_scales.view(1, -1), q.out_features, stored_k)
        if weight_scales.ndim != 6
        else weight_scales.view(torch.float8_e8m0fnu)
    )
    runner(
        a_tensor_gpu=packed_values.view(rows, stored_k, 1),
        b_tensor_gpu=weight_values.view(q.out_features, stored_k, 1),
        sfa_tensor_gpu=sfa,
        sfb_tensor_gpu=sfb,
        c_tensor_gpu=output.view(rows, q.out_features, 1),
        alpha_tensor_gpu=alpha_one,
        stream_int=_stream_to_int(stream),
        rope_cos_sin_gpu=angles,
    )
    return True


@torch.library.custom_op(
    "flare::mxfp8_encoded_projection_rope",
    mutates_args=("output", "workspace"),
)
def _mxfp8_encoded_projection_rope(
    values_base: torch.Tensor,
    mma_base: torch.Tensor,
    weight_values: torch.Tensor,
    weight_mma_base: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    plan_handle: int,
    angles: torch.Tensor,
    head_dim: int,
    rope_dim: int,
    rope_split: bool,
) -> None:
    stream = (
        torch.cuda.current_stream(values_base.device)
        if values_base.device.type == "cuda"
        else None
    )
    if run_encoded_dense_rope(
        values_base,
        mma_base,
        weight_values,
        weight_mma_base,
        output,
        plan_handle,
        angles,
        head_dim,
        rope_dim,
        rope_split=rope_split,
        stream=stream,
    ):
        return
    from b12x import gemm

    gemm.mxfp8_mm_bases(
        values_base,
        mma_base,
        weight_values,
        weight_mma_base,
        output,
        workspace,
        plan=plan_from_handle(plan_handle),
    )
    heads = output.shape[-1] // head_dim
    view = output.view(output.shape[0], heads, head_dim)
    per_cta = 8 if heads % 8 == 0 else 4
    _rotate_grouped[(view.shape[0], (heads + per_cta - 1) // per_cta)](
        view,
        angles,
        view.stride(0),
        view.stride(1),
        head_dim - rope_dim,
        heads,
        1,
        rope_split,
        per_cta,
        num_warps=1,
    )


@_mxfp8_encoded_projection_rope.register_fake
def _fake_mxfp8_encoded_projection_rope(
    values_base,
    mma_base,
    weight_values,
    weight_mma_base,
    output,
    workspace,
    plan_handle,
    angles,
    head_dim,
    rope_dim,
    rope_split,
):
    pass


def run_encoded_projection_rope(
    projection,
    values_base: torch.Tensor,
    mma_base: torch.Tensor,
    angles: torch.Tensor,
    *,
    head_dim: int,
    rope_dim: int = 64,
    rope_split: bool = True,
) -> torch.Tensor:
    """Run a functional preencoded MXFP8 projection with fused Q-RoPE when registered."""
    shape, stride, dtype = projection.output_layout
    output = torch.empty_strided(shape, stride, dtype=dtype, device=values_base.device)
    workspace = torch.empty(
        projection.workspace_nbytes, dtype=torch.uint8, device=values_base.device
    )
    _mxfp8_encoded_projection_rope(
        values_base,
        mma_base,
        projection.parts[0],
        projection.parts[1],
        output,
        workspace,
        projection.plan.handle,
        angles,
        int(head_dim),
        int(rope_dim),
        bool(rope_split),
    )
    return output


def _can_fuse_residual(residual: torch.Tensor, output: torch.Tensor) -> bool:
    return (
        residual.is_contiguous()
        and residual.dtype == output.dtype
        and residual.shape == output.shape
        and residual.device == output.device
    )


def _resolve_fused_swiglu_down_policy(
    rows: int, out_features: int, padded_k: int, sm_count: int
):
    import cutlass
    from b12x._lib.dense_gemm.config import _dense_gemm_policy_for
    from b12x._lib.dense_gemm.policy import (
        _select_default_dense_gemm_plan,
        _select_mxfp8_tile_k,
        _validate_mxfp8_bk64_plan,
    )

    default_plan = _select_default_dense_gemm_plan(
        rows,
        out_features,
        padded_k,
        sm_count,
        is_mxfp8=True,
        expected_m=rows,
    )
    tile_k = _select_mxfp8_tile_k(rows, out_features, padded_k, rows, sm_count)
    if default_plan.mma_tiler_mn[0] != 128 and padded_k % 128 == 0:
        tile_k = 128
    if _has_parameters(_validate_mxfp8_bk64_plan, ("allow_short_m",)):
        _validate_mxfp8_bk64_plan(
            tile_k,
            default_plan.mma_tiler_mn,
            default_plan.swap_ab,
            allow_short_m=(padded_k % 128 != 0),
        )
    else:
        _validate_mxfp8_bk64_plan(
            tile_k, default_plan.mma_tiler_mn, default_plan.swap_ab
        )
    policy = _dense_gemm_policy_for(
        m=rows,
        n=out_features,
        k=padded_k,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=default_plan.mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=sm_count,
        tile_k=tile_k,
        expected_m=rows,
        generalize_mxfp8_split_k=False,
        allow_bf16_atomic_reduction=False,
    )
    return default_plan, tile_k, policy


def _device_sm_count(device: torch.device, session=None) -> int:
    if device.type == "cuda":
        from b12x._lib.utils.device import get_num_sm

        return int(get_num_sm(device))
    identity = getattr(session, "device_identity", None)
    if identity is not None and getattr(identity, "sm_count", None) is not None:
        return int(identity.sm_count)
    return 170


def supports_fused_swiglu_down(
    intermediate: torch.Tensor,
    down_weight,
    output: torch.Tensor,
    *,
    activation_mode: str,
    down_bias,
    session,
) -> bool:
    """True when PreparedMLP should fuse SwiGLU + MXFP8 quantization with its MXFP8 down projection."""
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    if not supports_swiglu_quantize_rows():
        return False
    if down_bias is not None or not isinstance(down_weight, MXFP8LinearWeight):
        return False
    if sm103_projection.is_sm103(intermediate.device):
        return False
    if (
        intermediate.ndim != 2
        or output.ndim != 2
        or intermediate.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or not intermediate.is_contiguous()
        or not output.is_contiguous()
    ):
        return False
    rows, width = int(intermediate.shape[0]), int(intermediate.shape[1])
    out_features = int(output.shape[1])
    padded_k = int(down_weight.padded_in_features)
    if (
        rows <= 0
        or width != int(down_weight.in_features)
        or out_features != int(down_weight.out_features)
        or padded_k < width
        or padded_k % 64 != 0
        or out_features % 8 != 0
    ):
        return False
    if padded_k % 128 != 0 and not supports_k_alignment_64():
        return False
    if activation_mode == "a16":
        return False
    native = activation_mode == "auto" and getattr(
        session, "native_dense_activations", False
    )
    quantized_mode = activation_mode == "quantized" or native
    if not quantized_mode and rows < SWIGLU_QUANT_MIN_AUTO_ROWS:
        return False
    try:
        _, _, policy = _resolve_fused_swiglu_down_policy(
            rows, out_features, padded_k, _device_sm_count(intermediate.device, session)
        )
    except (ImportError, ValueError):
        return False
    return policy.split_k_slices == 1


@dataclass(frozen=True)
class FusedSwiGLUDownQuery:
    rows: int
    width: int
    padded_k: int
    out_features: int
    limit_finite: bool
    limit: float
    round_silu: bool
    clamp_after_silu: bool
    swiglu_alpha: float
    swiglu_beta: float
    residual_capable: bool


FUSED_SWIGLU_DOWN_CONTRACT = native_contract(
    component_id=FUSED_SWIGLU_DOWN_COMPONENT,
    query_type=FusedSwiGLUDownQuery,
    tactics=lambda query: ("cutedsl",),
)


@dataclass(frozen=True)
class _FusedSwiGLUDownState:
    query: FusedSwiGLUDownQuery
    device: torch.device
    swiglu_quant: object
    dense: object
    dense_residual: object | None
    alpha_one: torch.Tensor

    def run(
        self,
        gate_up: torch.Tensor,
        values: torch.Tensor,
        scale_rows: torch.Tensor,
        scale_mma: torch.Tensor,
        weight_values: torch.Tensor,
        weight_mma_base: torch.Tensor,
        output: torch.Tensor,
        *,
        residual: torch.Tensor | None = None,
        stream=None,
    ) -> None:
        rows = int(gate_up.shape[0])
        k = self.query.padded_k
        n = self.query.out_features
        sfa = _scale_mma_view_from_physical(scale_mma)
        sfb = _scale_mma_view_from_physical(weight_mma_base)
        stream_int = _stream_to_int(stream)
        with _cuda_stream_context(stream, gate_up.device):
            self.swiglu_quant(
                gate_up,
                values,
                scale_rows.view(rows, -1),
                scale_mma.view(-1),
            )
            if residual is not None:
                if self.dense_residual is None or not _can_fuse_residual(
                    residual, output
                ):
                    self.dense(
                        a_tensor_gpu=values.view(rows, k, 1),
                        b_tensor_gpu=weight_values.view(n, k, 1),
                        sfa_tensor_gpu=sfa,
                        sfb_tensor_gpu=sfb,
                        c_tensor_gpu=output.view(rows, n, 1),
                        alpha_tensor_gpu=self.alpha_one,
                        stream_int=stream_int,
                    )
                    output.add_(residual)
                    return
                self.dense_residual(
                    a_tensor_gpu=values.view(rows, k, 1),
                    b_tensor_gpu=weight_values.view(n, k, 1),
                    sfa_tensor_gpu=sfa,
                    sfb_tensor_gpu=sfb,
                    c_tensor_gpu=output.view(rows, n, 1),
                    alpha_tensor_gpu=self.alpha_one,
                    stream_int=stream_int,
                    residual_gpu=residual.view(rows, n, 1),
                )
                return
            self.dense(
                a_tensor_gpu=values.view(rows, k, 1),
                b_tensor_gpu=weight_values.view(n, k, 1),
                sfa_tensor_gpu=sfa,
                sfb_tensor_gpu=sfb,
                c_tensor_gpu=output.view(rows, n, 1),
                alpha_tensor_gpu=self.alpha_one,
                stream_int=stream_int,
            )


def compile_fused_swiglu_down(
    query_payload: dict[str, object],
    device_ordinal: int,
    sm_count: int,
) -> dict[str, object]:
    """Compile the SwiGLU+MXFP8 quantizer and DenseGemm programs for one fused MLP down plan."""
    query = FusedSwiGLUDownQuery(**dict(query_payload))
    ordinal = int(device_ordinal)
    sm_count = int(sm_count)
    default_plan, tile_k, policy = _resolve_fused_swiglu_down_policy(
        query.rows, query.out_features, query.padded_k, sm_count
    )
    if policy.split_k_slices != 1:
        raise ValueError(
            f"fused SwiGLU+MXFP8 down projection requires split_k_slices == 1, got {policy.split_k_slices}"
        )
    import cutlass
    from b12x._lib.dense_gemm.compile import _get_compiled_dense_gemm
    from b12x._lib.quant.mxfp8_rows import (
        _get_compiled_swiglu_mxfp8_rows_quant,
        mxfp8_rows_quant_launch_options,
    )

    subgroup_width, threads = mxfp8_rows_quant_launch_options(query.rows, "linear")
    limit = float(query.limit) if query.limit_finite else float("inf")
    with torch.cuda.device(ordinal):
        swiglu_quant = _get_compiled_swiglu_mxfp8_rows_quant(
            query.width,
            query.padded_k,
            subgroup_width,
            threads,
            limit,
            query.round_silu,
            query.clamp_after_silu,
            query.swiglu_alpha,
            query.swiglu_beta,
            0.0,
            False,
            device_ordinal=ordinal,
            sm_count=sm_count,
        )
        common = {
            "n": query.out_features,
            "k": query.padded_k,
            "l": 1,
            "c_l": 1,
            "a_major": "k",
            "b_major": "k",
            "c_major": "n",
            "ab_dtype": cutlass.Float8E4M3FN,
            "sf_dtype": cutlass.Float8E8M0FNU,
            "c_dtype": cutlass.BFloat16,
            "alpha_dtype": cutlass.Float32,
            "sf_vec_size": 32,
            "mma_k": 32,
            "tile_k": tile_k,
            "mma_tiler_mn": default_plan.mma_tiler_mn,
            "cluster_shape_mn": (1, 1),
            "policy": policy,
            "sm_count": sm_count,
            "sm_version": "sm_120",
            "load_path": default_plan.load_path,
            "swap_ab": default_plan.swap_ab,
            "sfb_k_reuse": False,
            "b_tile_major": False,
            "alpha_is_one": True,
            "device_ordinal": ordinal,
        }
        dense = _get_compiled_dense_gemm(**common)
        dense_residual = None
        if (
            query.residual_capable
            and supports_dense_residual_epilogue()
            and not default_plan.swap_ab
            and policy.split_k_slices == 1
        ):
            dense_residual = _get_compiled_dense_gemm(**common, has_residual=True)
    programs: dict[str, object] = {
        "swiglu_quant": swiglu_quant,
        "dense": dense,
    }
    if dense_residual is not None:
        programs["dense_residual"] = dense_residual
    return programs


def _materialize_fused_swiglu_down(
    query: FusedSwiGLUDownQuery, device: torch.device
) -> _FusedSwiGLUDownState:
    from b12x._lib.dense_gemm.buffers import _cached_alpha_one
    from b12x._lib.utils.device import get_num_sm

    ordinal = int(device.index or 0)
    sm_count = get_num_sm(device)
    payload = FUSED_SWIGLU_DOWN_CONTRACT.encode_query(query)
    programs = compile_fused_swiglu_down(payload, ordinal, sm_count)
    swiglu_quant = programs["swiglu_quant"]
    dense = programs["dense"]
    dense_residual = programs.get("dense_residual")
    with torch.cuda.device(ordinal):
        alpha_one = _cached_alpha_one(device)
    load_programs(swiglu_quant)
    load_programs(dense)
    if dense_residual is not None:
        load_programs(dense_residual)
    state = _FusedSwiGLUDownState(
        query=query,
        device=device,
        swiglu_quant=swiglu_quant,
        dense=dense,
        dense_residual=dense_residual,
        alpha_one=alpha_one,
    )
    carriers = (
        (swiglu_quant, dense)
        if dense_residual is None
        else (swiglu_quant, dense, dense_residual)
    )
    return attach_programs(state, *carriers)


def is_fused_swiglu_down_plan(plan_handle: int) -> bool:
    try:
        return plan_from_handle(plan_handle).component_id == FUSED_SWIGLU_DOWN_COMPONENT
    except (KeyError, ValueError, RuntimeError):
        return False


def run_fused_swiglu_down(
    gate_up: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    weight_values: torch.Tensor,
    weight_mma_base: torch.Tensor,
    output: torch.Tensor,
    plan_handle: int,
    *,
    residual: torch.Tensor | None = None,
    stream=None,
) -> None:
    """Execute a prepared fused SwiGLU+quant -> MXFP8 DenseGemm (+ optional residual)."""
    state = require_prepared(
        plan_from_handle(plan_handle),
        FUSED_SWIGLU_DOWN_COMPONENT,
        gate_up.device if gate_up.device.type == "cuda" else None,
    )
    if stream is None and gate_up.device.type == "cuda":
        stream = torch.cuda.current_stream(gate_up.device)
    state.run(
        gate_up,
        values,
        scale_rows,
        scale_mma,
        weight_values,
        weight_mma_base,
        output,
        residual=residual,
        stream=stream,
    )


@torch.library.custom_op(
    "flare::swiglu_mxfp8_down",
    mutates_args=("values", "scale_rows", "scale_mma", "output"),
)
def _swiglu_mxfp8_down_op(
    gate_up: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    weight_values: torch.Tensor,
    weight_mma_base: torch.Tensor,
    output: torch.Tensor,
    plan_handle: int,
) -> None:
    run_fused_swiglu_down(
        gate_up,
        values,
        scale_rows,
        scale_mma,
        weight_values,
        weight_mma_base,
        output,
        plan_handle,
    )


@_swiglu_mxfp8_down_op.register_fake
def _fake_swiglu_mxfp8_down_op(
    gate_up,
    values,
    scale_rows,
    scale_mma,
    weight_values,
    weight_mma_base,
    output,
    plan_handle,
):
    pass


@torch.library.custom_op(
    "flare::swiglu_mxfp8_down_residual",
    mutates_args=("values", "scale_rows", "scale_mma", "output"),
)
def _swiglu_mxfp8_down_residual_op(
    gate_up: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    weight_values: torch.Tensor,
    weight_mma_base: torch.Tensor,
    residual: torch.Tensor,
    output: torch.Tensor,
    plan_handle: int,
) -> None:
    run_fused_swiglu_down(
        gate_up,
        values,
        scale_rows,
        scale_mma,
        weight_values,
        weight_mma_base,
        output,
        plan_handle,
        residual=residual,
    )


@_swiglu_mxfp8_down_residual_op.register_fake
def _fake_swiglu_mxfp8_down_residual_op(
    gate_up,
    values,
    scale_rows,
    scale_mma,
    weight_values,
    weight_mma_base,
    residual,
    output,
    plan_handle,
):
    pass


def allocate_mxfp8_activation_bases(
    encoded_layout: tuple[tuple[tuple[int, ...], torch.dtype], ...],
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate contiguous MXFP8 activation bases; fill padded MMA scale rows with UE8M0 1.0."""
    (val_shape, val_dtype), (row_shape, row_dtype), (mma_shape, mma_dtype) = (
        encoded_layout
    )
    values = torch.empty(val_shape, dtype=val_dtype, device=device)
    scale_rows = torch.empty(row_shape, dtype=row_dtype, device=device)
    if rows % 128 == 0:
        scale_mma = torch.empty(mma_shape, dtype=mma_dtype, device=device)
    else:
        scale_mma = torch.full(mma_shape, 127, dtype=mma_dtype, device=device)
    return values, scale_rows, scale_mma


def _mxfp8_activation_bases_nbytes(
    encoded_layout: tuple[tuple[tuple[int, ...], torch.dtype], ...],
) -> int:
    """Return the 256-byte-aligned byte capacity needed for the three MXFP8 activation bases."""
    return sum(
        (math.prod(shape) * dtype.itemsize + 255) // 256 * 256
        for shape, dtype in encoded_layout
    )


def mxfp8_bases_from_byte_buffer(
    storage: torch.Tensor,
    encoded_layout: tuple[tuple[tuple[int, ...], torch.dtype], ...],
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice 256-byte-aligned MXFP8 activation bases from a contiguous uint8 byte buffer."""
    views = []
    offset = 0
    for shape, dtype in encoded_layout:
        size = math.prod(shape) * dtype.itemsize
        views.append(storage.narrow(0, offset, size).view(dtype).reshape(shape))
        offset += (size + 255) // 256 * 256
    values, scale_rows, scale_mma = views
    if rows % 128 != 0 and storage.device.type != "meta":
        scale_mma.fill_(127)
    return values, scale_rows, scale_mma


def _mxfp8_bases_from_workspace(
    workspace,
    encoded_layout: tuple[tuple[tuple[int, ...], torch.dtype], ...],
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice declaration or bound MXFP8 activation bases from preparation workspace when available."""
    if isinstance(workspace, ScratchArena):
        values, scale_rows, scale_mma = workspace.views(*encoded_layout)
        if rows % 128 != 0 and device.type != "meta":
            scale_mma.fill_(127)
        return values, scale_rows, scale_mma
    if (
        isinstance(workspace, torch.Tensor)
        and workspace.dtype == torch.uint8
        and workspace.ndim == 1
        and workspace.is_contiguous()
        and workspace.device == device
        and workspace.nbytes >= _mxfp8_activation_bases_nbytes(encoded_layout)
    ):
        return mxfp8_bases_from_byte_buffer(workspace, encoded_layout, rows)
    return allocate_mxfp8_activation_bases(encoded_layout, rows, device)


class PreparedFusedSwiGLUDown:
    """Fused SwiGLU -> BF16 round -> MXFP8 quantization -> DenseGemm down projection.

    Replaces separate hyperconnection.run_swiglu + activation quantization + down
    projection when the down weight is MXFP8 and the row count is in the quantized
    regime. Supports unpadded K%64==0 weights (k_alignment=64) and an optional
    BF16 residual addend fused into the GEMM epilogue.
    """

    is_fused_swiglu_down = True

    def __init__(
        self,
        gate_up: torch.Tensor,
        weight,
        output: torch.Tensor,
        *,
        session,
        name: str,
        workspace: ScratchArena | torch.Tensor | None = None,
        limit: float = float("inf"),
        round_silu: bool = True,
        clamp_after_silu: bool = False,
        swiglu_alpha: float = 1.0,
        swiglu_beta: float = 0.0,
    ) -> None:
        self.weight = weight
        self.output = output
        self.bias = None
        self.workspace = workspace
        self.preparation_arena = (
            workspace if isinstance(workspace, ScratchArena) else None
        )
        self.quantizer_plan = None
        self.encoded = None
        self.rows = int(gate_up.shape[0])
        self.width = int(gate_up.shape[1]) // 2
        self.padded_k = int(weight.padded_in_features)
        self.out_features = int(weight.out_features)
        self.k_alignment = 64 if self.padded_k % 128 != 0 else 128
        self.supports_residual = supports_dense_residual_epilogue()
        packed = weight.weight
        self.parts = (packed.values, _scale_mma_physical_base(packed.scale_mma), None)
        for tensor in (packed.values, packed.scale_rows, packed.scale_mma):
            if tensor is not None:
                record_numeric_tensor(tensor)

        val_shape, row_shape, mma_shape = dense_rows_storage_shapes_aligned(
            self.rows, self.padded_k, k_alignment=self.k_alignment
        )
        self.encoded_layout = (
            (val_shape, torch.float8_e4m3fn),
            (row_shape, torch.uint8),
            (mma_shape, torch.uint8),
        )
        device = gate_up.device
        bases_nbytes = _mxfp8_activation_bases_nbytes(self.encoded_layout)
        self.encoded_bases = _mxfp8_bases_from_workspace(
            workspace, self.encoded_layout, self.rows, device
        )
        limit_finite = limit != float("inf")
        query = FusedSwiGLUDownQuery(
            rows=self.rows,
            width=self.width,
            padded_k=self.padded_k,
            out_features=self.out_features,
            limit_finite=limit_finite,
            limit=float(limit) if limit_finite else 0.0,
            round_silu=bool(round_silu),
            clamp_after_silu=bool(clamp_after_silu),
            swiglu_alpha=float(swiglu_alpha),
            swiglu_beta=float(swiglu_beta),
            residual_capable=self.supports_residual,
        )
        payload = FUSED_SWIGLU_DOWN_CONTRACT.encode_query(query)
        self.plan = native_plan(
            contract=FUSED_SWIGLU_DOWN_CONTRACT,
            query=query,
            device=device,
            scratch_bytes=lambda tactic: bases_nbytes,
            materialize=lambda tactic: _materialize_fused_swiglu_down(query, device),
            compile_jobs=lambda tactic, detected: (
                CompileJob.create(
                    "hyperflare_recipes.components.dense_epilogues:compile_fused_swiglu_down",
                    payload,
                    CompileJob.DEVICE_ORDINAL,
                    detected.identity.sm_count,
                ),
            ),
        )
        encoded_layout = self.encoded_layout
        rows = self.rows
        self._decl_refs: list[object] = [gate_up, output, workspace, self.parts]
        decl_refs = self._decl_refs

        def _prepare_call(state):
            ref_gate_up, ref_output, ref_workspace, ref_parts = decl_refs
            if ref_gate_up is None:
                raise RuntimeError(
                    "fused SwiGLU down declaration examples were already released"
                )
            active_bases = _mxfp8_bases_from_workspace(
                ref_workspace, encoded_layout, rows, device
            )
            return PreparedCall(
                run=lambda: state.run(
                    ref_gate_up,
                    active_bases[0],
                    active_bases[1],
                    active_bases[2],
                    ref_parts[0],
                    ref_parts[1],
                    ref_output,
                ),
                owners=prime_owners(
                    ref_workspace,
                    ref_gate_up,
                    ref_output,
                    *active_bases,
                    ref_parts[0],
                    ref_parts[1],
                ),
            )

        declare(session, (self.plan.request(name=name, prepare_call=_prepare_call),))
        self.output_layout = None

        def _finalize_preparation() -> None:
            decl_refs[:] = [None, None, None, None]
            if self.output is None:
                self.encoded_bases = None
                self.workspace = None
                self.preparation_arena = None
            elif isinstance(self.workspace, ScratchArena):
                self.encoded_bases = _mxfp8_bases_from_workspace(
                    self.workspace, self.encoded_layout, self.rows, device
                )

        register_finalizer(
            session, _finalize_preparation, workspace=self.preparation_arena
        )

    def functional(self, *, session):
        if self.output is not None:
            self.output_layout = (
                tuple(self.output.shape),
                self.output.stride(),
                self.output.dtype,
            )
            self.output = None
            self.encoded_bases = None
            self.workspace = None
            register_finalizer(
                session, self._finalize_functional, workspace=self.preparation_arena
            )
        return self

    def _finalize_functional(self) -> None:
        self._decl_refs[:] = [None, None, None, None]
        self.encoded_bases = None
        self.workspace = None
        self.preparation_arena = None

    def __call__(
        self, gate_up: torch.Tensor, *, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(
                shape, stride, dtype=dtype, device=gate_up.device
            )
            values, scale_rows, scale_mma = allocate_mxfp8_activation_bases(
                self.encoded_layout, int(gate_up.shape[0]), gate_up.device
            )
        else:
            values, scale_rows, scale_mma = self.encoded_bases
        if (
            residual is not None
            and self.supports_residual
            and _can_fuse_residual(residual, output)
        ):
            _swiglu_mxfp8_down_residual_op(
                gate_up,
                values,
                scale_rows,
                scale_mma,
                self.parts[0],
                self.parts[1],
                residual,
                output,
                self.plan.handle,
            )
            return output
        _swiglu_mxfp8_down_op(
            gate_up,
            values,
            scale_rows,
            scale_mma,
            self.parts[0],
            self.parts[1],
            output,
            self.plan.handle,
        )
        if residual is not None:
            return output + residual
        return output
