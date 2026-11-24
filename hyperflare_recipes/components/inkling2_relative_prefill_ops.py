# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared SM120 4D-TMA relative prefill attention with fused gate-sink."""

from __future__ import annotations

from dataclasses import replace

import torch
from b12x.preparation import PreparedCall, plan_from_handle

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import is_sm120

_BOUND_OPERAND = object()


def _relative_prefill_module():
    try:
        from b12x.attention import relative_prefill
    except ImportError:
        return None
    return relative_prefill


def _scratch(specs, workspace):
    specs = tuple(specs)
    if not specs:
        return []
    if workspace is None:
        return [torch.empty(spec.shape, dtype=spec.dtype, device=spec.device) for spec in specs]
    values = planned_scratch(specs, workspace)
    return list(values) if isinstance(values, tuple) else [values]


def _operand_aligned(tensor: torch.Tensor, align: int) -> bool:
    if tensor.stride(-1) != 1 or tensor.data_ptr() % align:
        return False
    elem = tensor.element_size()
    return all((stride * elem) % align == 0 for stride in tensor.stride()[:-1])


@torch.library.custom_op(
    "flare::inkling2_relative_prefill_attention",
    mutates_args=("gated", "scratch"),
)
def _execute_relative_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    relative_logits: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    gated: torch.Tensor,
    scratch: list[torch.Tensor],
    softmax_scale: float,
    handle: int,
) -> None:
    from b12x.attention import relative_prefill

    plan = plan_from_handle(handle)
    binding = relative_prefill.bind(
        plan,
        scratch=scratch,
        q=query,
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=page_table,
        lengths=lengths,
        cu_seqlens_q=cu_seqlens_q,
        bias=relative_logits,
        gate=gate,
        sink=sink,
        gated=gated,
        softmax_scale=softmax_scale,
    )
    relative_prefill.run(binding)


@_execute_relative_prefill.register_fake
def _fake_relative_prefill(
    query,
    key_cache,
    value_cache,
    page_table,
    lengths,
    cu_seqlens_q,
    relative_logits,
    gate,
    sink,
    gated,
    scratch,
    softmax_scale,
    handle,
):
    pass


def supports_relative_prefill(
    *,
    query,
    key_cache,
    value_cache,
    relative_logits,
    gate,
    sink,
    gated,
    window: int,
) -> bool:
    """Whether SM120 4D-TMA relative prefill attention supports this layer."""
    if not is_sm120(query.device) or query.dtype != torch.bfloat16:
        return False
    relative_prefill = _relative_prefill_module()
    if relative_prefill is None:
        return False
    if key_cache.dtype != torch.float8_e4m3fn or value_cache.dtype != torch.float8_e4m3fn:
        return False
    if key_cache.ndim != 4 or value_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        return False
    if relative_logits.dtype != torch.bfloat16 or relative_logits.stride(-1) != 1:
        return False
    if gate.dtype != torch.bfloat16 or sink.dtype != torch.bfloat16 or gated.dtype != torch.bfloat16:
        return False
    if sink.stride(-1) != 1:
        return False
    if not (
        _operand_aligned(query, 16)
        and _operand_aligned(key_cache, 16)
        and _operand_aligned(value_cache, 16)
        and _operand_aligned(gate, 8)
        and _operand_aligned(gated, 8)
    ):
        return False
    is_window = window > 0
    if is_window:
        if key_cache.shape[1] < window + int(query.shape[0]) or key_cache.shape[1] % 64 != 0:
            return False
        page_size = 64
    else:
        page_size = int(key_cache.shape[1])
        if page_size <= 0 or page_size % 64 != 0:
            return False
    return relative_prefill.supports(
        num_q_heads=int(query.shape[1]),
        num_kv_heads=int(key_cache.shape[2]),
        head_dim=int(query.shape[2]),
        page_size=page_size,
        rel_extent=int(relative_logits.shape[2]),
        max_total_q=int(query.shape[0]),
        window_left=(window - 1) if is_window else -1,
        device=query.device,
    )


class PreparedRelativePrefillAttention:
    """Fused SM120 4D-TMA relative-position prefill attention and gate-sink."""

    def __init__(
        self,
        *,
        query,
        key_cache,
        value_cache,
        page_table,
        lengths,
        query_start_loc,
        relative_logits,
        gate,
        sink,
        gated,
        window: int,
        softmax_scale: float,
        session,
        workspace,
        name: str,
    ):
        from b12x.attention import relative_prefill

        device = query.device
        batch = int(lengths.numel())
        is_window = window > 0
        page_size = 64 if is_window else int(key_cache.shape[1])
        self.applies_gate_sink = True
        self.softmax_scale = float(softmax_scale)
        self.workspace = workspace
        self.is_window = is_window
        self.window = int(window)

        self.dummy_pt = torch.zeros((1, 1), dtype=torch.int32, device=device)
        pt = self.dummy_pt if is_window else page_table
        sink_2d = sink.view(-1, int(query.shape[1])) if sink.ndim == 1 else sink

        self.arguments = (
            query,
            key_cache,
            value_cache,
            pt,
            lengths,
            query_start_loc,
            relative_logits,
            gate,
            sink_2d,
            gated,
        )
        self.cache = (key_cache, value_cache)
        self.output_layout = (tuple(gated.shape), tuple(gated.stride()), gated.dtype)
        self.scratch_layout = None

        self.plan = relative_prefill.plan(
            relative_prefill.Caps(
                device=device,
                num_q_heads=int(query.shape[1]),
                num_kv_heads=int(key_cache.shape[2]),
                head_dim=int(query.shape[2]),
                page_size=page_size,
                rel_extent=int(relative_logits.shape[2]),
                max_batch=batch,
                max_total_q=int(query.shape[0]),
                window_left=(window - 1) if is_window else -1,
            )
        )
        self.scratch = []

        def benchmark(state):
            from b12x.attention.relative_prefill._preparation import scratch_specs

            def inputs():
                b_q = torch.zeros_like(query)
                total_q = int(query.shape[0])
                active = min(batch, total_q)
                per = max(1, total_q // active)
                b_cu = torch.zeros((batch + 1,), dtype=torch.int32, device=device)
                b_cu[1 : active + 1] = (
                    torch.arange(1, active + 1, dtype=torch.int32, device=device) * per
                )
                if active < batch:
                    b_cu[active + 1 :] = active * per
                if is_window:
                    staged_cap = int(key_cache.shape[1])
                    b_kc = torch.zeros(
                        (batch, staged_cap, key_cache.shape[2], key_cache.shape[3]),
                        dtype=torch.float8_e4m3fn,
                        device=device,
                    )
                    b_vc = torch.zeros_like(b_kc)
                    b_pt = self.dummy_pt
                    hist = min(window, max(0, staged_cap - per))
                    b_lengths = torch.zeros((batch,), dtype=torch.int32, device=device)
                    b_lengths[:active] = hist + per
                else:
                    b_pages = max(2, (per + page_size - 1) // page_size)
                    b_kc = torch.zeros(
                        (b_pages, page_size, key_cache.shape[2], key_cache.shape[3]),
                        dtype=torch.float8_e4m3fn,
                        device=device,
                    )
                    b_vc = torch.zeros_like(b_kc)
                    b_pt = (
                        torch.arange(b_pages, dtype=torch.int32, device=device)
                        .view(1, -1)
                        .expand(batch, -1)
                        .contiguous()
                    )
                    b_lengths = torch.zeros((batch,), dtype=torch.int32, device=device)
                    b_lengths[:active] = per
                b_bias = torch.zeros_like(relative_logits)
                b_gate = torch.zeros_like(gate)
                b_sink = torch.zeros_like(sink_2d)
                b_gated = torch.empty_like(gated)
                return dict(
                    q=b_q,
                    k_cache=b_kc,
                    v_cache=b_vc,
                    page_table=b_pt,
                    lengths=b_lengths,
                    cu_seqlens_q=b_cu,
                    bias=b_bias,
                    gate=b_gate,
                    sink=b_sink,
                    gated=b_gated,
                )

            operands = benchmark_resource(session, ("inkling2_relative_prefill", name), inputs)
            specs = scratch_specs(state.query, state.config, device)
            sc = _scratch(specs, workspace)
            binding = state.bind(
                plan=self.plan, scratch=sc, softmax_scale=self.softmax_scale, **operands
            )
            return PreparedCall(
                run=lambda: state.run(binding),
                output=operands["gated"],
                owners=(*prime_owners(workspace, sc, binding), operands),
            )

        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=lambda state: replace(benchmark(state), owners=()),
                    benchmark_call=benchmark,
                ),
            ),
        )
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = _scratch(self.plan.scratch_specs(), self.workspace)

    def functional(self, *, session, k_cache=_BOUND_OPERAND, v_cache=_BOUND_OPERAND):
        """Consume forward-local inputs and allocate output and split scratch locally."""
        kc = self.cache[0] if k_cache is _BOUND_OPERAND else k_cache
        vc = self.cache[1] if v_cache is _BOUND_OPERAND else v_cache
        self.cache = (kc, vc)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.arguments = None
        self.scratch = []

    def __call__(
        self,
        query=None,
        *,
        key_cache=None,
        value_cache=None,
        page_table=None,
        lengths=None,
        cu_seqlens_q=None,
        relative_logits=None,
        gate=None,
        sink=None,
    ):
        if self.arguments is not None and query is None:
            if not self.scratch and self.plan.scratch_specs():
                self.rebind()
            _execute_relative_prefill(
                *self.arguments,
                self.scratch,
                self.softmax_scale,
                self.plan.handle,
            )
            return self.arguments[9]
        if self.scratch_layout is None:
            raise RuntimeError("functional relative prefill attention requires finalized preparation")
        if query is None:
            raise ValueError("functional relative prefill attention requires query")
        kc = self.cache[0] if key_cache is None else key_cache
        vc = self.cache[1] if value_cache is None else value_cache
        if kc is None or vc is None:
            raise ValueError("functional relative prefill attention requires key_cache and value_cache")
        pt = self.dummy_pt if self.is_window else page_table
        sink_2d = sink.view(-1, int(query.shape[1])) if sink.ndim == 1 else sink
        shape, stride, dtype = self.output_layout
        gated = torch.empty_strided(shape, stride, dtype=dtype, device=query.device)
        scratch = [torch.empty(s_shape, dtype=s_dtype, device=query.device) for s_shape, s_dtype in self.scratch_layout]
        _execute_relative_prefill(
            query,
            kc,
            vc,
            pt,
            lengths,
            cu_seqlens_q,
            relative_logits,
            gate,
            sink_2d,
            gated,
            scratch,
            self.softmax_scale,
            self.plan.handle,
        )
        return gated
