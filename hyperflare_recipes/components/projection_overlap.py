# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture a chain of dense projections alongside independent main-stream work."""

import torch

from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.rotary import _rotate_grouped
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer


_TC_GEMV_ROPE_RUNNERS: dict[tuple[int, int, int], tuple[object, object]] = {}


def register_tc_gemv_rope(plan, *, head_dim: int, rope_dim: int = 64) -> bool:
    """Precompile a fused-RoPE MXFP8 TC-GEMV runner if `plan` selected tc_gemv."""
    prepared = getattr(plan, "_prepared", None)
    state = getattr(prepared, "state", prepared)
    if state is None:
        return False
    config = getattr(state, "config", None)
    query = getattr(state, "query", None)
    if (config is None or query is None or config.mode != "a16"
            or getattr(config, "a16_backend", None) != "tc_gemv"
            or getattr(query, "recipe", None) != "mxfp8"):
        return False
    if head_dim < 128 or (head_dim & (head_dim - 1)) != 0 or rope_dim != 64:
        return False
    try:
        from b12x._lib.compile_plan import load_programs
        from b12x.gemm.blockscaled._mxfp8_tc_gemv import compile_mxfp8_tc_gemv_rope
    except ImportError:
        return False
    device = state.device
    runner = compile_mxfp8_tc_gemv_rope(
        int(device.index or 0),
        query.out_features,
        query.in_features,
        query.padded_in_features,
        query.num_tokens,
        config.split_k,
        config.mma_warps or 8,
        config.tile_n or 16,
        head_dim=head_dim,
        rope_dim=rope_dim,
    )
    load_programs(runner)
    _TC_GEMV_ROPE_RUNNERS[(plan.handle, int(head_dim), int(rope_dim))] = (runner, state.tc_flags)
    return True


def _run_mxfp8_with_rope(source, values, scales, gain, activation_scale, scale_kind,
                         output, workspace, plan_handle, angles, head_dim, rope_dim, stream):
    entry = _TC_GEMV_ROPE_RUNNERS.get((plan_handle, head_dim, rope_dim))
    if entry is None:
        from b12x.preparation.types import plan_from_handle

        try:
            plan = plan_from_handle(plan_handle)
        except Exception:
            plan = None
        if plan is not None and register_tc_gemv_rope(plan, head_dim=head_dim, rope_dim=rope_dim):
            entry = _TC_GEMV_ROPE_RUNNERS.get((plan_handle, head_dim, rope_dim))
    if (entry is not None and source.stride(-1) == 1 and output.is_contiguous()
            and angles.dtype == torch.float32 and angles.is_contiguous() and angles.shape[-1] == 64):
        runner, tc_flags = entry
        stream_handle = stream.cuda_stream if hasattr(stream, "cuda_stream") else stream
        runner(source, values, scales, output, angles,
               workspace if workspace.numel() > 0 else None, tc_flags, stream_handle)
        return
    from hyperflare_recipes.components.sm103_projection import packed_out_workspace

    packed_out_workspace(source, values, scales, gain, activation_scale, output, workspace,
                         ("none", "multiplier", "reciprocal")[scale_kind], None, plan_handle, "any")
    heads = output.shape[-1] // head_dim
    view = output.view(output.shape[0], heads, head_dim)
    per_cta = 8 if heads % 8 == 0 else 4
    _rotate_grouped[(view.shape[0], (heads + per_cta - 1) // per_cta)](
        view, angles, view.stride(0), view.stride(1), head_dim - rope_dim, heads,
        1, True, per_cta, num_warps=1)


@torch.library.custom_op("flare::mxfp8_projection_rope", mutates_args=("output", "workspace"))
def _mxfp8_projection_rope(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                           gain: torch.Tensor | None, activation_scale: torch.Tensor | None,
                           scale_kind: int, output: torch.Tensor, workspace: torch.Tensor,
                           plan_handle: int, angles: torch.Tensor, head_dim: int, rope_dim: int) -> None:
    _run_mxfp8_with_rope(source, values, scales, gain, activation_scale, scale_kind,
                         output, workspace, plan_handle, angles, head_dim, rope_dim,
                         torch.cuda.current_stream(source.device))


@_mxfp8_projection_rope.register_fake
def _fake_mxfp8_projection_rope(source, values, scales, gain, activation_scale, scale_kind,
                                output, workspace, plan_handle, angles, head_dim, rope_dim):
    pass


def run_mxfp8_projection_rope(projection: QuantizedProjection, source: torch.Tensor,
                              angles: torch.Tensor, *, head_dim: int, rope_dim: int = 64) -> torch.Tensor:
    """Run a functional QuantizedProjection with fused RoPE when tc_gemv was selected."""
    from b12x.gemm.blockscaled._a16 import _weight_parts

    values, scales, gain, fp4 = _weight_parts(projection.weight)
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    kind = ("none", "multiplier", "reciprocal").index(projection.weight.global_scale_kind if fp4 else "none")
    shape, stride, dtype = projection.output_layout
    output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
    workspace = torch.empty(projection.workspace_nbytes, dtype=torch.uint8, device=source.device)
    _mxfp8_projection_rope(source, values, scales, gain, projection.activation_scale, kind,
                           output, workspace, projection.plan.handle, angles, head_dim, rope_dim)
    return output


@torch.library.custom_op("flare::start_projection_chain", mutates_args=("outputs", "workspace"))
def _start(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor | None],
           gains: list[torch.Tensor | None], activation_scales: list[torch.Tensor | None], scale_kinds: list[int],
           outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], stream: int) -> None:
    from hyperflare_recipes.components.sm103_projection import bf16_out, packed_out_workspace

    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        value = source
        for weight, scale, gain, activation_scale, kind, output, plan in zip(
                weights, scales, gains, activation_scales, scale_kinds, outputs, plans, strict=True):
            if scale is None:
                bf16_out(value, weight, output, plan)
            else:
                packed_out_workspace(value, weight, scale, gain, activation_scale, output, workspace,
                                      ("none", "multiplier", "reciprocal")[kind], None, plan, "any")
            value = output


@_start.register_fake
def _fake_start(source, weights, scales, gains, activation_scales, scale_kinds, outputs, workspace, plans, stream):
    pass


@torch.library.custom_op("flare::join_projection_chain", mutates_args=("outputs", "workspace"))
def _join(source: torch.Tensor, dependency: torch.Tensor, outputs: list[torch.Tensor],
          workspace: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join.register_fake
def _fake_join(source, dependency, outputs, workspace, stream):
    pass


@torch.library.custom_op("flare::start_encoded_projection", mutates_args=(
    "output", "workspace", "values_base", "row_base", "mma_base"))
def _start_encoded(source: torch.Tensor, weight_values: torch.Tensor, weight_mma_base: torch.Tensor,
                   output: torch.Tensor, values_base: torch.Tensor, row_base: torch.Tensor,
                   mma_base: torch.Tensor, workspace: torch.Tensor, quantizer: int, dense: int,
                   stream: int) -> None:
    from b12x import gemm
    from b12x.preparation.types import plan_from_handle
    from b12x.quantization import mxfp8

    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        mxfp8.quantize_dense_rows(source, values_base, row_base, mma_base,
                                  plan=plan_from_handle(quantizer))
        gemm.mxfp8_mm_bases(values_base, mma_base, weight_values, weight_mma_base,
                            output, workspace, plan=plan_from_handle(dense))


@_start_encoded.register_fake
def _fake_start_encoded(source, weight_values, weight_mma_base, output, values_base, row_base,
                        mma_base, workspace, quantizer, dense, stream):
    pass


@torch.library.custom_op("flare::start_preencoded_projection", mutates_args=("output", "workspace"))
def _start_preencoded(source: torch.Tensor, weight_values: torch.Tensor, weight_mma_base: torch.Tensor,
                      output: torch.Tensor, values_base: torch.Tensor, row_base: torch.Tensor,
                      mma_base: torch.Tensor, workspace: torch.Tensor, dense: int, stream: int) -> None:
    from b12x import gemm
    from b12x.preparation.types import plan_from_handle

    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        gemm.mxfp8_mm_bases(values_base, mma_base, weight_values, weight_mma_base,
                            output, workspace, plan=plan_from_handle(dense))


@_start_preencoded.register_fake
def _fake_start_preencoded(source, weight_values, weight_mma_base, output, values_base,
                           row_base, mma_base, workspace, dense, stream):
    pass


@torch.library.custom_op("flare::join_encoded_projection", mutates_args=("output", "workspace"))
def _join_encoded(source: torch.Tensor, dependency: torch.Tensor, output: torch.Tensor,
                  values_base: torch.Tensor, row_base: torch.Tensor, mma_base: torch.Tensor,
                  workspace: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_encoded.register_fake
def _fake_join_encoded(source, dependency, output, values_base, row_base, mma_base, workspace, stream):
    pass


class ConcurrentProjectionChain:
    """Run dependent projections on an auxiliary stream with one scratch suffix.

    Each projection consumes its predecessor's complete output. The recipe
    declares every concurrent main-stream plan, starts the chain, and joins
    before consuming its output or reusing source and scratch storage.
    """

    def __init__(self, projections, *, parallel_plans, session, workspace: ScratchArena):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        self.projections, self.parallel_plans = tuple(projections), tuple(parallel_plans)
        if not isinstance(workspace, ScratchArena) or not self.projections or not self.parallel_plans:
            raise ValueError("projection overlap requires an arena, a projection chain, and concurrent plans")
        self.arena = workspace
        self.encoded_projection = (self.projections[0] if len(self.projections) == 1
            and isinstance(self.projections[0], QuantizedProjection)
            and self.projections[0].encoded is not None else None)
        if self.encoded_projection is not None:
            projection = self.encoded_projection
            if projection.bias is not None:
                raise ValueError("overlapped MXFP8 projection cannot add bias")
            for main in self.parallel_plans:
                if projection.quantizer_plan is not None:
                    workspace.register_parallel(main, projection.quantizer_plan)
                workspace.register_parallel(main, projection.plan)
            self.stream = workspace.auxiliary_stream
            self.stream_handle = self.stream.cuda_stream
            register_finalizer(session, self.refresh, workspace=workspace)
            return
        self.weights, self.scales, self.outputs, self.plans = [], [], [], []
        self.gains, self.activation_scales, self.scale_kinds = [], [], []
        for projection in self.projections:
            if not isinstance(projection, (QuantizedProjection, UnquantizedProjection)) or projection.bias is not None:
                raise ValueError("projection overlap requires unbiased BF16 or blockscaled projections")
            if isinstance(projection, QuantizedProjection):
                values, scales, gain, fp4 = _weight_parts(projection.weight)
                if scales.dtype == torch.float8_e8m0fnu:
                    scales = scales.view(torch.uint8)
                width = projection.weight.in_features
                activation_scale = projection.activation_scale
                kind = projection.weight.global_scale_kind if fp4 else "none"
            else:
                values, scales, gain = projection.weight, None, None
                width, activation_scale, kind = values.shape[1], None, "none"
            if self.outputs and self.outputs[-1].shape[1] != width:
                raise ValueError("projection chain output width must match the following input width")
            self.weights.append(values)
            self.scales.append(scales)
            self.gains.append(gain)
            self.activation_scales.append(activation_scale)
            self.scale_kinds.append(("none", "multiplier", "reciprocal").index(kind))
            self.outputs.append(projection.output)
            self.plans.append(projection.plan.handle)
            for main in self.parallel_plans:
                workspace.register_parallel(main, projection.plan)
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        if self.arena is None:
            return
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        if self.encoded_projection is not None:
            projection = self.encoded_projection
            side_plans = ((projection.quantizer_plan, projection.plan)
                          if projection.quantizer_plan is not None else (projection.plan,))
            side_bytes = max(plan.memory_requirements().scratch_nbytes for plan in side_plans)
        else:
            side_bytes = max(p.plan.memory_requirements().scratch_nbytes for p in self.projections)
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def functional(self, *, session):
        self.output_layouts = tuple((tuple(p.output.shape), tuple(p.output.stride()), p.output.dtype)
                                   for p in self.projections)
        projection = self.encoded_projection
        self.encoded_layouts = None
        if projection is not None:
            self.encoded_layouts = tuple((tuple(t.shape), t.dtype) for t in projection.encoded_bases)
            self.encoded_weights = projection.parts[:2]
            self.quantizer_handle = None if projection.quantizer_plan is None else projection.quantizer_plan.handle
            self.dense_handle = projection.plan.handle
        self.side_plans = tuple(p.plan for p in self.projections)
        if projection is not None and projection.quantizer_plan is not None:
            self.side_plans += (projection.quantizer_plan,)
        self.plan_owners = tuple(p.functional(session=session) for p in self.projections)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.side_plans = tuple(p.plan for p in self.plan_owners)
        if self.encoded_layouts is not None:
            projection = self.plan_owners[0]
            self.dense_handle = projection.plan.handle
            self.quantizer_handle = None if projection.quantizer_plan is None else projection.quantizer_plan.handle
            if projection.quantizer_plan is not None:
                self.side_plans += (projection.quantizer_plan,)
        else:
            self.plans = [p.plan.handle for p in self.plan_owners]
        self.workspace_nbytes = max(plan.memory_requirements().scratch_nbytes for plan in self.side_plans)
        self.projections = self.parallel_plans = self.encoded_projection = self.arena = self.workspace = None
        self.outputs = None

    def start(self, source, *, encoded_bases=None):
        if self.projections is None:
            outputs = [torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
                       for shape, stride, dtype in self.output_layouts]
            workspace = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
            if self.encoded_layouts is not None:
                if self.quantizer_handle is None:
                    if encoded_bases is None:
                        raise ValueError("preencoded projection overlap requires runtime encoded bases")
                    _start_preencoded(source, *self.encoded_weights, outputs[0], *encoded_bases,
                                      workspace, self.dense_handle, self.stream_handle)
                else:
                    encoded_bases = tuple(torch.empty(shape, dtype=dtype, device=source.device)
                                          for shape, dtype in self.encoded_layouts)
                    _start_encoded(source, *self.encoded_weights, outputs[0], *encoded_bases, workspace,
                                   self.quantizer_handle, self.dense_handle, self.stream_handle)
            else:
                _start(source, self.weights, self.scales, self.gains, self.activation_scales,
                       self.scale_kinds, outputs, workspace, self.plans, self.stream_handle)
            return source, outputs, workspace, encoded_bases
        if self.encoded_projection is not None:
            projection = self.encoded_projection
            if projection.quantizer_plan is None:
                _start_preencoded(source, projection.parts[0], projection.parts[1], projection.output,
                    *projection.encoded_bases, self.workspace, projection.plan.handle, self.stream_handle)
            else:
                _start_encoded(source, projection.parts[0], projection.parts[1], projection.output,
                    *projection.encoded_bases, self.workspace, projection.quantizer_plan.handle,
                    projection.plan.handle, self.stream_handle)
            return
        _start(source, self.weights, self.scales, self.gains, self.activation_scales, self.scale_kinds,
               self.outputs, self.workspace, self.plans, self.stream_handle)

    def finish(self, source, dependency):
        if self.projections is None:
            source, outputs, workspace, bases = source
            if bases is not None:
                _join_encoded(source, dependency, outputs[0], *bases, workspace, self.stream_handle)
            else:
                _join(source, dependency, outputs, workspace, self.stream_handle)
            return outputs[-1]
        if self.encoded_projection is not None:
            projection = self.encoded_projection
            _join_encoded(source, dependency, projection.output, *projection.encoded_bases,
                          self.workspace, self.stream_handle)
            return projection.output
        _join(source, dependency, self.outputs, self.workspace, self.stream_handle)
        return self.outputs[-1]


def _run_encoded_with_rope(values_base, mma_base, weight_values, weight_mma_base,
                           output, workspace, dense_handle, angles, head_dim, rope_dim):
    from b12x import gemm
    from b12x.preparation.types import plan_from_handle

    gemm.mxfp8_mm_bases(values_base, mma_base, weight_values, weight_mma_base,
                        output, workspace, plan=plan_from_handle(dense_handle))
    heads = output.shape[-1] // head_dim
    view = output.view(output.shape[0], heads, head_dim)
    per_cta = 8 if heads % 8 == 0 else 4
    _rotate_grouped[(view.shape[0], (heads + per_cta - 1) // per_cta)](
        view, angles, view.stride(0), view.stride(1), head_dim - rope_dim, heads,
        1, True, per_cta, num_warps=1)


@torch.library.custom_op("flare::start_attention_query_rope", mutates_args=("q_out", "workspace"))
def _start_attn_q_rope(query_low: torch.Tensor, q_w: torch.Tensor, q_s: torch.Tensor,
                       q_gain: torch.Tensor | None, q_act_scale: torch.Tensor | None,
                       q_scale_kind: int, q_out: torch.Tensor, layer_angles: torch.Tensor,
                       workspace: torch.Tensor, q_plan: int, head_dim: int, rope_dim: int,
                       stream: int) -> None:
    auxiliary = torch.cuda.ExternalStream(stream, device=query_low.device)
    auxiliary.wait_stream(torch.cuda.current_stream(query_low.device))
    with torch.cuda.stream(auxiliary):
        _run_mxfp8_with_rope(query_low, q_w, q_s, q_gain, q_act_scale, q_scale_kind,
                             q_out, workspace, q_plan, layer_angles, head_dim, rope_dim, auxiliary)


@_start_attn_q_rope.register_fake
def _fake_start_attn_q_rope(query_low, q_w, q_s, q_gain, q_act_scale, q_scale_kind,
                            q_out, layer_angles, workspace, q_plan, head_dim, rope_dim, stream):
    pass


@torch.library.custom_op("flare::start_attention_preencoded_rope", mutates_args=("q_out", "workspace"))
def _start_attn_preencoded_rope(query_low: torch.Tensor, q_w: torch.Tensor, q_s: torch.Tensor,
                                q_out: torch.Tensor, values_base: torch.Tensor, row_base: torch.Tensor,
                                mma_base: torch.Tensor, layer_angles: torch.Tensor,
                                workspace: torch.Tensor, q_plan: int, head_dim: int, rope_dim: int,
                                stream: int) -> None:
    auxiliary = torch.cuda.ExternalStream(stream, device=query_low.device)
    auxiliary.wait_stream(torch.cuda.current_stream(query_low.device))
    with torch.cuda.stream(auxiliary):
        _run_encoded_with_rope(values_base, mma_base, q_w, q_s, q_out, workspace,
                               q_plan, layer_angles, head_dim, rope_dim)


@_start_attn_preencoded_rope.register_fake
def _fake_start_attn_preencoded_rope(query_low, q_w, q_s, q_out, values_base, row_base, mma_base,
                                     layer_angles, workspace, q_plan, head_dim, rope_dim, stream):
    pass


@torch.library.custom_op("flare::join_attention_query_rope", mutates_args=("q_out", "workspace"))
def _join_attn_q_rope(query_low: torch.Tensor, layer_angles: torch.Tensor,
                      dependency: torch.Tensor, q_out: torch.Tensor,
                      workspace: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(query_low.device).wait_stream(
        torch.cuda.ExternalStream(stream, device=query_low.device))


@_join_attn_q_rope.register_fake
def _fake_join_attn_q_rope(query_low, layer_angles, dependency, q_out, workspace, stream):
    pass


@torch.library.custom_op("flare::join_attention_preencoded_rope", mutates_args=("q_out", "workspace"))
def _join_attn_preencoded_rope(query_low: torch.Tensor, layer_angles: torch.Tensor,
                               dependency: torch.Tensor, q_out: torch.Tensor,
                               values_base: torch.Tensor, row_base: torch.Tensor,
                               mma_base: torch.Tensor, workspace: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(query_low.device).wait_stream(
        torch.cuda.ExternalStream(stream, device=query_low.device))


@_join_attn_preencoded_rope.register_fake
def _fake_join_attn_preencoded_rope(query_low, layer_angles, dependency, q_out,
                                    values_base, row_base, mma_base, workspace, stream):
    pass


@torch.library.custom_op(
    "flare::start_attention_aux_indexed",
    mutates_args=("q_out", "iq_out", "iw_out", "packed_list", "scales_list", "workspace"),
)
def _start_attn_aux_indexed(
    query_low: torch.Tensor, source: torch.Tensor,
    layer_angles: torch.Tensor, index_angles: torch.Tensor,
    q_w: torch.Tensor, q_s: torch.Tensor, q_gain: torch.Tensor | None,
    q_act_scale: torch.Tensor | None, q_scale_kind: int, q_plan: int,
    q_values_base: torch.Tensor | None, q_mma_base: torch.Tensor | None,
    iq_w: torch.Tensor, iq_s: torch.Tensor, iq_gain: torch.Tensor | None,
    iq_act_scale: torch.Tensor | None, iq_scale_kind: int, iq_plan: int,
    iw_w: torch.Tensor, iw_plan: int,
    q_out: torch.Tensor, iq_out: torch.Tensor, iw_out: torch.Tensor,
    packed_list: list[torch.Tensor], scales_list: list[torch.Tensor],
    selector_handles: list[int], selector_starts: list[int], selector_ends: list[int],
    workspace: torch.Tensor, q_head_dim: int, iq_head_dim: int, rope_dim: int,
    iw_scale: float, stream: int,
) -> None:
    from b12x.attention import dsa_indexer as dsa
    from b12x.preparation.types import plan_from_handle
    from hyperflare_recipes.components.sm103_projection import bf16_out

    auxiliary = torch.cuda.ExternalStream(stream, device=query_low.device)
    auxiliary.wait_stream(torch.cuda.current_stream(query_low.device))
    with torch.cuda.stream(auxiliary):
        if q_values_base is not None:
            _run_encoded_with_rope(q_values_base, q_mma_base, q_w, q_s, q_out, workspace,
                                   q_plan, layer_angles, q_head_dim, rope_dim)
        else:
            _run_mxfp8_with_rope(query_low, q_w, q_s, q_gain, q_act_scale, q_scale_kind,
                                 q_out, workspace, q_plan, layer_angles, q_head_dim, rope_dim, auxiliary)
        _run_mxfp8_with_rope(query_low, iq_w, iq_s, iq_gain, iq_act_scale, iq_scale_kind,
                             iq_out, workspace, iq_plan, index_angles, iq_head_dim, rope_dim, auxiliary)
        bf16_out(source, iw_w, iw_out, iw_plan)
        iw_out.mul_(iw_scale)
        iq_view = iq_out.view(query_low.shape[0], 32, iq_head_dim)
        for handle, first, last, packed, scales in zip(
                selector_handles, selector_starts, selector_ends, packed_list, scales_list, strict=True):
            dsa.quantize_q_mxfp4(plan_from_handle(handle), iq_view[first:last],
                                 q_mxfp4=packed, q_scales=scales)


@_start_attn_aux_indexed.register_fake
def _fake_start_attn_aux_indexed(
    query_low, source, layer_angles, index_angles,
    q_w, q_s, q_gain, q_act_scale, q_scale_kind, q_plan, q_values_base, q_mma_base,
    iq_w, iq_s, iq_gain, iq_act_scale, iq_scale_kind, iq_plan, iw_w, iw_plan,
    q_out, iq_out, iw_out, packed_list, scales_list,
    selector_handles, selector_starts, selector_ends, workspace,
    q_head_dim, iq_head_dim, rope_dim, iw_scale, stream,
):
    pass


@torch.library.custom_op(
    "flare::join_attention_aux_indexed",
    mutates_args=("q_out", "iq_out", "iw_out", "packed_list", "scales_list", "workspace"),
)
def _join_attn_aux_indexed(
    query_low: torch.Tensor, source: torch.Tensor,
    layer_angles: torch.Tensor, index_angles: torch.Tensor,
    dep_window: torch.Tensor, dep_global: torch.Tensor,
    q_values_base: torch.Tensor | None, q_row_base: torch.Tensor | None, q_mma_base: torch.Tensor | None,
    q_out: torch.Tensor, iq_out: torch.Tensor, iw_out: torch.Tensor,
    packed_list: list[torch.Tensor], scales_list: list[torch.Tensor],
    workspace: torch.Tensor, stream: int,
) -> None:
    torch.cuda.current_stream(query_low.device).wait_stream(
        torch.cuda.ExternalStream(stream, device=query_low.device))


@_join_attn_aux_indexed.register_fake
def _fake_join_attn_aux_indexed(
    query_low, source, layer_angles, index_angles, dep_window, dep_global,
    q_values_base, q_row_base, q_mma_base, q_out, iq_out, iw_out,
    packed_list, scales_list, workspace, stream,
):
    pass


class ConcurrentAttentionAux:
    """Overlap q_b (+ fused RoPE) and optional indexer query/weight/quantize on auxiliary stream."""

    def __init__(self, q_b: QuantizedProjection, *, parallel_plans, session, workspace: ScratchArena,
                 index_query: QuantizedProjection | None = None,
                 index_weight: UnquantizedProjection | None = None,
                 selectors=(), selector_rows: int = 128,
                 q_head_dim: int = 512, iq_head_dim: int = 128, rope_dim: int = 64):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        if not isinstance(workspace, ScratchArena) or not parallel_plans:
            raise ValueError("ConcurrentAttentionAux requires a ScratchArena and parallel plans")
        if q_b.bias is not None:
            raise ValueError("overlapped q_b cannot add bias")
        self.arena = workspace
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        self.parallel_plans = tuple(parallel_plans)
        self.q_head_dim = int(q_head_dim)
        self.iq_head_dim = int(iq_head_dim)
        self.rope_dim = int(rope_dim)
        self.q_b = q_b
        self.q_preencoded = q_b.encoded is not None and q_b.quantizer_plan is None
        if q_b.encoded is not None and q_b.quantizer_plan is not None:
            raise ValueError("ConcurrentAttentionAux expects preencoded or a16 q_b")
        if self.q_preencoded:
            self.q_weights = q_b.parts[:2]
            self.q_gain = self.q_act_scale = None
            self.q_scale_kind = 0
        else:
            vals, sc, gain, fp4 = _weight_parts(q_b.weight)
            if sc.dtype == torch.float8_e8m0fnu:
                sc = sc.view(torch.uint8)
            self.q_weights = (vals, sc)
            self.q_gain = gain
            self.q_act_scale = q_b.activation_scale
            self.q_scale_kind = ("none", "multiplier", "reciprocal").index(
                q_b.weight.global_scale_kind if fp4 else "none")
        self.has_indexer = index_query is not None
        self.index_query = index_query
        self.index_weight = index_weight
        self.selectors = tuple(selectors)
        self.selector_rows = int(selector_rows)
        side_plans = [q_b.plan]
        if self.has_indexer:
            if index_query.encoded is not None or index_query.bias is not None or index_weight.bias is not None:
                raise ValueError("overlapped indexer requires unbiased a16 index_query and index_weight")
            vals, sc, gain, fp4 = _weight_parts(index_query.weight)
            if sc.dtype == torch.float8_e8m0fnu:
                sc = sc.view(torch.uint8)
            self.iq_weights = (vals, sc)
            self.iq_gain = gain
            self.iq_act_scale = index_query.activation_scale
            self.iq_scale_kind = ("none", "multiplier", "reciprocal").index(
                index_query.weight.global_scale_kind if fp4 else "none")
            self.iw_weight = index_weight.weight
            side_plans.extend((index_query.plan, index_weight.plan))
        for main in self.parallel_plans:
            for side in side_plans:
                workspace.register_parallel(main, side)
        self.side_plans = tuple(side_plans)
        register_finalizer(session, self._refresh, workspace=workspace)

    def _refresh(self):
        if self.arena is None:
            return
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        side_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.side_plans)
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def functional(self, *, session):
        self.q_layout = (tuple(self.q_b.output.shape), tuple(self.q_b.output.stride()), self.q_b.output.dtype)
        self.q_owner = self.q_b.functional(session=session)
        if self.has_indexer:
            self.iq_layout = (tuple(self.index_query.output.shape),
                              tuple(self.index_query.output.stride()), self.index_query.output.dtype)
            self.iw_layout = (tuple(self.index_weight.output.shape),
                              tuple(self.index_weight.output.stride()), self.index_weight.output.dtype)
            self.iq_owner = self.index_query.functional(session=session)
            self.iw_owner = self.index_weight.functional(session=session)
            total_rows = self.iq_layout[0][0]
            self.selector_slices = tuple(
                (i * self.selector_rows, min(total_rows, (i + 1) * self.selector_rows))
                for i in range(len(self.selectors)))
            self.selector_owners = tuple(sel.functional(session=session) for sel in self.selectors)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.q_handle = self.q_owner.plan.handle
        side_plans = [self.q_owner.plan]
        if not self.q_preencoded:
            register_tc_gemv_rope(self.q_owner.plan, head_dim=self.q_head_dim, rope_dim=self.rope_dim)
        if self.has_indexer:
            self.iq_handle = self.iq_owner.plan.handle
            self.iw_handle = self.iw_owner.plan.handle
            register_tc_gemv_rope(self.iq_owner.plan, head_dim=self.iq_head_dim, rope_dim=self.rope_dim)
            side_plans.extend((self.iq_owner.plan, self.iw_owner.plan))
            self.selector_handles = [sel.plan.handle for sel in self.selector_owners]
            self.selector_starts = [a for a, _ in self.selector_slices]
            self.selector_ends = [b for _, b in self.selector_slices]
            self.quant_layouts = tuple(
                (sel.layouts["q_mxfp4"], sel.layouts["q_scales"]) for sel in self.selector_owners)
        self.side_plans = tuple(side_plans)
        self.workspace_nbytes = max(plan.memory_requirements().scratch_nbytes for plan in self.side_plans)
        self.q_b = self.index_query = self.index_weight = self.selectors = None
        self.parallel_plans = self.arena = self.workspace = None

    def start(self, query_low, source, layer_angles, *, index_angles=None, encoded_bases=None):
        device = query_low.device
        q_out = torch.empty_strided(self.q_layout[0], self.q_layout[1], dtype=self.q_layout[2], device=device)
        workspace = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=device)
        if self.has_indexer:
            iq_out = torch.empty_strided(self.iq_layout[0], self.iq_layout[1], dtype=self.iq_layout[2], device=device)
            iw_out = torch.empty_strided(self.iw_layout[0], self.iw_layout[1], dtype=self.iw_layout[2], device=device)
            packed_list = [torch.empty_strided(pl[0], pl[1], dtype=pl[2], device=device) for pl, _ in self.quant_layouts]
            scales_list = [torch.empty_strided(sl[0], sl[1], dtype=sl[2], device=device) for _, sl in self.quant_layouts]
            q_vb = None if encoded_bases is None else encoded_bases[0]
            q_mb = None if encoded_bases is None else encoded_bases[2]
            _start_attn_aux_indexed(
                query_low, source, layer_angles, index_angles,
                self.q_weights[0], self.q_weights[1], self.q_gain, self.q_act_scale,
                self.q_scale_kind, self.q_handle, q_vb, q_mb,
                self.iq_weights[0], self.iq_weights[1], self.iq_gain, self.iq_act_scale,
                self.iq_scale_kind, self.iq_handle,
                self.iw_weight, self.iw_handle,
                q_out, iq_out, iw_out, packed_list, scales_list,
                self.selector_handles, self.selector_starts, self.selector_ends,
                workspace, self.q_head_dim, self.iq_head_dim, self.rope_dim,
                float(128 ** -.5 * 32 ** -.5), self.stream_handle)
            return (query_low, source, layer_angles, index_angles, encoded_bases,
                    q_out, iq_out, iw_out, packed_list, scales_list, workspace)
        if self.q_preencoded:
            if encoded_bases is None:
                raise ValueError("preencoded attention overlap requires runtime encoded bases")
            _start_attn_preencoded_rope(query_low, self.q_weights[0], self.q_weights[1], q_out,
                                        *encoded_bases, layer_angles, workspace, self.q_handle,
                                        self.q_head_dim, self.rope_dim, self.stream_handle)
        else:
            _start_attn_q_rope(query_low, self.q_weights[0], self.q_weights[1], self.q_gain,
                               self.q_act_scale, self.q_scale_kind, q_out, layer_angles,
                               workspace, self.q_handle, self.q_head_dim, self.rope_dim, self.stream_handle)
        return (query_low, layer_angles, encoded_bases, q_out, workspace)

    def finish(self, pending, dependency, *, global_dependency=None):
        if self.has_indexer:
            (query_low, source, layer_angles, index_angles, encoded_bases,
             q_out, iq_out, iw_out, packed_list, scales_list, workspace) = pending
            q_vb = None if encoded_bases is None else encoded_bases[0]
            q_rb = None if encoded_bases is None else encoded_bases[1]
            q_mb = None if encoded_bases is None else encoded_bases[2]
            _join_attn_aux_indexed(
                query_low, source, layer_angles, index_angles,
                dependency, dependency if global_dependency is None else global_dependency,
                q_vb, q_rb, q_mb, q_out, iq_out, iw_out, packed_list, scales_list,
                workspace, self.stream_handle)
            return q_out, iq_out, iw_out, tuple(zip(packed_list, scales_list, strict=True))
        query_low, layer_angles, encoded_bases, q_out, workspace = pending
        if self.q_preencoded:
            _join_attn_preencoded_rope(query_low, layer_angles, dependency, q_out,
                                       *encoded_bases, workspace, self.stream_handle)
        else:
            _join_attn_q_rope(query_low, layer_angles, dependency, q_out, workspace, self.stream_handle)
        return q_out
