# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared SM120 TMA relative decode/verify attention and fused log-scaling tau."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall, plan_from_handle
from b12x.preparation.types import require_prepared

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import compile_triton, is_sm120, native_contract, native_plan
from .selected_attention import metadata_tensor, tensor_facts


def _relative_decode_module():
    try:
        from b12x.attention import relative_decode
    except ImportError:
        return None
    return relative_decode


def _scratch(specs, workspace):
    specs = tuple(specs)
    if not specs:
        return []
    if workspace is None:
        return [torch.empty(spec.shape, dtype=spec.dtype, device=spec.device) for spec in specs]
    values = planned_scratch(specs, workspace)
    return list(values) if isinstance(values, tuple) else [values]


# ---------------------------------------------------------------------------
# One-launch log-scaling tau for full-attention layers.


@triton.jit
def _log_scaling_tau_kernel(
    Positions,
    Tau,
    N,
    INV_FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    pos = tl.maximum(tl.load(Positions + offs, mask=mask, other=0), 0).to(tl.float32)
    scaled = tl.maximum((pos + 1.0) * INV_FLOOR, 1.0)
    tau = 1.0 + ALPHA * tl.extra.cuda.libdevice.log(scaled)
    tl.store(Tau + offs, tau, mask=mask)


@dataclass(frozen=True)
class TauQuery:
    operands: tuple
    inv_floor: float
    alpha: float


TAU_CONTRACT = native_contract(
    component_id="flare.inkling2_log_scaling_tau",
    query_type=TauQuery,
    tactics=lambda q: ((128,),),
)


@program_cache(scope="preparation")
def compile_tau(payload, tactic, ordinal):
    query = TauQuery(**dict(payload))
    (block,) = tactic
    pos, tau = tuple(map(metadata_tensor, query.operands))
    with torch.cuda.device(ordinal), compile_only_launches():
        return compile_triton(
            _log_scaling_tau_kernel,
            (triton.cdiv(pos.numel(), block), 1, 1),
            pos,
            tau,
            pos.numel(),
            query.inv_floor,
            query.alpha,
            block,
            num_warps=2,
        )


@dataclass(frozen=True)
class TauState:
    query: TauQuery
    tactic: tuple
    program: object

    def run(self, positions, tau):
        (block,) = self.tactic
        n = positions.numel()
        self.program[(triton.cdiv(n, block), 1, 1)](
            positions, tau, n, self.query.inv_floor, self.query.alpha, block
        )


@torch.library.custom_op("flare::inkling2_log_scaling_tau", mutates_args=("tau",))
def _execute_tau(positions: torch.Tensor, tau: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, positions.device).run(positions, tau)


@_execute_tau.register_fake
def _fake_tau(positions, tau, handle):
    pass


class PreparedLogScalingTau:
    """Compute ``1 + alpha * log(max((max(pos, 0) + 1) / floor, 1))`` in one launch."""

    def __init__(self, positions, tau, *, floor, alpha, session, name):
        if (
            positions.ndim != 1
            or tau.ndim != 1
            or positions.shape != tau.shape
            or tau.dtype != torch.float32
            or not positions.is_contiguous()
            or not tau.is_contiguous()
            or positions.device != tau.device
            or positions.device.type != "cuda"
        ):
            raise ValueError("log-scaling tau requires matching contiguous 1D CUDA positions and float32 tau")
        self.positions = positions
        self.tau = tau
        self.output_shape = tuple(tau.shape)
        query = TauQuery((tensor_facts(positions), tensor_facts(tau)), float(1.0 / floor), float(alpha))
        payload = TAU_CONTRACT.encode_query(query)

        def materialize(tactic):
            program = compile_tau(payload, tactic, positions.device.index)
            load_programs(program)
            return attach_programs(TauState(query, tactic, program), program)

        self.plan = native_plan(
            contract=TAU_CONTRACT,
            query=query,
            device=positions.device,
            scratch_bytes=lambda t: 0,
            materialize=materialize,
            compile_jobs=lambda t, d: (
                CompileJob.create(
                    "hyperflare_recipes.components.inkling2_relative_decode_ops:compile_tau",
                    payload,
                    t,
                    d.ordinal,
                ),
            ),
        )

        def benchmark(state):
            def inputs():
                p = torch.zeros_like(positions)
                t = torch.empty_like(tau)
                return p, t

            p, t = benchmark_resource(session, ("inkling2_log_scaling_tau", name), inputs)
            return PreparedCall(run=lambda: state.run(p, t), output=t, owners=(p, t))

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

    def functional(self, *, session=None):
        """Return forward-local float32 tau tensors without retaining preparation storage."""
        del session
        self.positions = self.tau = None
        return self

    def __call__(self, positions=None):
        if self.positions is None:
            if positions is None:
                raise ValueError("functional log-scaling tau requires positions")
            tau = torch.empty(self.output_shape, dtype=torch.float32, device=positions.device)
            _execute_tau(positions, tau, self.plan.handle)
            return tau
        _execute_tau(self.positions, self.tau, self.plan.handle)
        return self.tau


# ---------------------------------------------------------------------------
# Direct SM120 TMA relative decode/verify attention with fused KV write & gate-sink.


@torch.library.custom_op(
    "flare::inkling2_relative_decode_attention",
    mutates_args=("key_cache", "value_cache", "staged_key", "staged_value", "gated", "scratch"),
)
def _execute_relative_decode(
    query: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    staged_key: torch.Tensor,
    staged_value: torch.Tensor,
    page_table: torch.Tensor,
    state_rows: torch.Tensor,
    starts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    relative_logits: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    gated: torch.Tensor,
    scratch: list[torch.Tensor],
    softmax_scale: float,
    handle: int,
) -> None:
    from b12x.attention import relative_decode

    plan = plan_from_handle(handle)
    binding = relative_decode.bind(
        plan,
        scratch=scratch,
        q=query,
        new_k=new_key,
        new_v=new_value,
        k_cache=key_cache,
        v_cache=value_cache,
        staged_k=staged_key,
        staged_v=staged_value,
        page_table=page_table,
        state_rows=state_rows,
        starts=starts,
        cu_seqlens_q=cu_seqlens_q,
        bias=relative_logits,
        gate=gate,
        sink=sink,
        gated=gated,
        softmax_scale=softmax_scale,
    )
    relative_decode.run(binding)


@_execute_relative_decode.register_fake
def _fake_relative_decode(
    query,
    new_key,
    new_value,
    key_cache,
    value_cache,
    staged_key,
    staged_value,
    page_table,
    state_rows,
    starts,
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


def _operand_aligned(tensor: torch.Tensor, align: int) -> bool:
    if tensor.stride(-1) != 1 or tensor.data_ptr() % align:
        return False
    elem = tensor.element_size()
    return all((stride * elem) % align == 0 for stride in tensor.stride()[:-1])


def supports_relative_decode(
    *,
    query,
    key_cache,
    value_cache,
    is_window: bool,
    commit_ring: bool,
    window: int,
    rel_extent: int,
    max_q_per_req: int,
    new_key=None,
    new_value=None,
    gate=None,
    gated=None,
) -> bool:
    """Whether SM120 TMA direct relative decode/verify attention supports this layer."""
    if not is_sm120(query.device):
        return False
    relative_decode = _relative_decode_module()
    if relative_decode is None:
        return False
    if key_cache.dtype != torch.float8_e4m3fn or value_cache.dtype != torch.float8_e4m3fn:
        return False
    if key_cache.ndim != 4 or value_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        return False
    if not (
        _operand_aligned(query, 16)
        and _operand_aligned(key_cache, 16)
        and _operand_aligned(value_cache, 16)
    ):
        return False
    if new_key is not None and not _operand_aligned(new_key, 16):
        return False
    if new_value is not None and not _operand_aligned(new_value, 16):
        return False
    if gate is not None and not _operand_aligned(gate, 8):
        return False
    if gated is not None and not _operand_aligned(gated, 8):
        return False
    page_or_window = int(key_cache.shape[1])
    if is_window and (page_or_window < window or page_or_window % 16 != 0):
        return False
    return relative_decode.supports(
        num_q_heads=int(query.shape[1]),
        num_kv_heads=int(key_cache.shape[2]),
        head_dim=int(query.shape[2]),
        is_window=is_window,
        commit_ring=commit_ring,
        page_or_window=page_or_window,
        window_left=(window - 1) if is_window else -1,
        rel_extent=int(rel_extent),
        max_q_per_req=int(max_q_per_req),
        device=query.device,
    )


class VerifyWindowStagingView:
    """Compact verification window-staging carrier for uncommitted windowed decode."""

    def __init__(self, staging, *, requests: int, window: int, max_query_tokens: int, kv_heads: int, dim: int):
        capacity = window + max_query_tokens
        size = 2 * requests * capacity * kv_heads * dim
        if staging.dtype != torch.uint8 or not staging.is_contiguous() or staging.numel() < size:
            raise ValueError("window staging storage is too small for verification rows")
        self.window = int(window)
        self.staged = staging[:size].view(torch.float8_e4m3fn).view(
            2, requests, capacity, kv_heads, dim
        )

    @property
    def staged_key(self) -> torch.Tensor:
        return self.staged[0]

    @property
    def staged_value(self) -> torch.Tensor:
        return self.staged[1]

    def shares_schedule(self, _other) -> bool:
        return False


class PreparedRelativeDecodeAttention:
    """Fused SM120 TMA relative-position decode/verify attention, KV write, and gate-sink."""

    def __init__(
        self,
        *,
        query,
        new_key,
        new_value,
        key_cache,
        value_cache,
        metadata,
        relative_logits,
        gate,
        sink,
        gated,
        is_window: bool,
        commit_ring: bool,
        window: int,
        softmax_scale: float,
        max_q_per_req: int,
        staged_key=None,
        staged_value=None,
        session,
        workspace,
        name: str,
    ):
        from b12x.attention import relative_decode

        device = query.device
        batch = int(metadata.starts.numel())
        page_or_window = int(key_cache.shape[1])
        self.applies_gate_sink = True
        self.softmax_scale = float(softmax_scale)
        self.workspace = workspace
        self.is_window = bool(is_window)
        self.commit_ring = bool(commit_ring)
        self.batch = batch
        self.page_or_window = page_or_window
        self.max_q_per_req = int(max_q_per_req)
        self.kv_heads = int(key_cache.shape[2])
        self.head_dim = int(key_cache.shape[3])

        self.dummy_u8 = torch.zeros((1, 1, self.kv_heads, self.head_dim), dtype=torch.uint8, device=device)
        self.dummy_pt = torch.zeros((1, 1), dtype=torch.int32, device=device)
        self.dummy_sr = torch.zeros((1,), dtype=torch.int64, device=device)

        sk = staged_key if (is_window and not commit_ring) else self.dummy_u8
        sv = staged_value if (is_window and not commit_ring) else self.dummy_u8
        pt = self.dummy_pt if is_window else metadata.page_table
        sr = metadata.state_rows if is_window else self.dummy_sr
        sink_2d = sink.view(-1, int(query.shape[1])) if sink.ndim == 1 else sink

        self.arguments = (
            query,
            new_key,
            new_value,
            key_cache,
            value_cache,
            sk,
            sv,
            pt,
            sr,
            metadata.starts,
            metadata.query_start_loc,
            relative_logits,
            gate,
            sink_2d,
            gated,
        )
        self.cache = (key_cache, value_cache)
        self.output_layout = (tuple(gated.shape), tuple(gated.stride()), gated.dtype)
        self.scratch_layout = None

        self.plan = relative_decode.plan(
            relative_decode.Caps(
                device=device,
                num_q_heads=int(query.shape[1]),
                num_kv_heads=self.kv_heads,
                head_dim=int(query.shape[2]),
                is_window=is_window,
                commit_ring=commit_ring,
                page_or_window=page_or_window,
                window_left=(window - 1) if is_window else -1,
                rel_extent=int(relative_logits.shape[2]),
                max_batch=batch,
                max_q_per_req=self.max_q_per_req,
                max_total_q=int(query.shape[0]),
            )
        )
        self.scratch = []

        def benchmark(state):
            from b12x.attention.relative_decode._preparation import scratch_specs

            def inputs():
                b_q = torch.zeros_like(query)
                b_nk = torch.zeros_like(new_key)
                b_nv = torch.zeros_like(new_value)
                cache_blocks = max(batch + 1, 2) if is_window else max(batch * 2, 2)
                b_kc = torch.zeros(
                    (cache_blocks, page_or_window, self.kv_heads, self.head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                b_vc = torch.zeros_like(b_kc)
                b_sk = (
                    torch.zeros(
                        (batch, page_or_window + self.max_q_per_req, self.kv_heads, self.head_dim),
                        dtype=torch.float8_e4m3fn,
                        device=device,
                    )
                    if (is_window and not commit_ring)
                    else self.dummy_u8
                )
                b_sv = torch.zeros_like(b_sk) if (is_window and not commit_ring) else self.dummy_u8
                b_pt = torch.zeros((batch, 2), dtype=torch.int32, device=device)
                b_sr = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
                b_starts = torch.full((batch,), min(page_or_window, 128), dtype=torch.int64, device=device)
                per = max(1, min(self.max_q_per_req, int(query.shape[0]) // batch))
                b_cu = torch.arange(0, (batch + 1) * per, per, dtype=torch.int32, device=device)
                b_bias = torch.zeros_like(relative_logits)
                b_gate = torch.zeros_like(gate)
                b_sink = torch.zeros_like(sink_2d)
                b_gated = torch.empty_like(gated)
                return dict(
                    q=b_q,
                    new_k=b_nk,
                    new_v=b_nv,
                    k_cache=b_kc,
                    v_cache=b_vc,
                    staged_k=b_sk,
                    staged_v=b_sv,
                    page_table=b_pt,
                    state_rows=b_sr,
                    starts=b_starts,
                    cu_seqlens_q=b_cu,
                    bias=b_bias,
                    gate=b_gate,
                    sink=b_sink,
                    gated=b_gated,
                )

            operands = benchmark_resource(session, ("inkling2_relative_decode", name), inputs)
            specs = scratch_specs(state.query, state.config, device)
            sc = _scratch(specs, workspace)
            binding = state.bind(plan=self.plan, scratch=sc, softmax_scale=self.softmax_scale, **operands)
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

    def functional(self, *, session):
        """Consume forward-local inputs and allocate output and split scratch locally."""
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
        new_key=None,
        new_value=None,
        key_cache=None,
        value_cache=None,
        staged_key=None,
        staged_value=None,
        page_table=None,
        state_rows=None,
        starts=None,
        cu_seqlens_q=None,
        relative_logits=None,
        gate=None,
        sink=None,
    ):
        if self.arguments is not None and query is None:
            if not self.scratch and self.plan.scratch_specs():
                self.rebind()
            _execute_relative_decode(
                *self.arguments,
                self.scratch,
                self.softmax_scale,
                self.plan.handle,
            )
            return self.arguments[14]
        if self.scratch_layout is None:
            raise RuntimeError("functional relative decode attention requires finalized preparation")
        if query is None or new_key is None or new_value is None:
            raise ValueError("functional relative decode attention requires query, new_key, and new_value")
        kc = self.cache[0] if key_cache is None else key_cache
        vc = self.cache[1] if value_cache is None else value_cache
        if self.is_window and not self.commit_ring:
            cap = self.page_or_window + self.max_q_per_req
            sk = (
                torch.empty((self.batch, cap, self.kv_heads, self.head_dim), dtype=torch.float8_e4m3fn, device=query.device)
                if staged_key is None
                else staged_key
            )
            sv = (
                torch.empty((self.batch, cap, self.kv_heads, self.head_dim), dtype=torch.float8_e4m3fn, device=query.device)
                if staged_value is None
                else staged_value
            )
        else:
            sk, sv = self.dummy_u8, self.dummy_u8
        pt = self.dummy_pt if self.is_window else page_table
        sr = state_rows if self.is_window else self.dummy_sr
        sink_2d = sink.view(-1, int(query.shape[1])) if sink.ndim == 1 else sink
        shape, stride, dtype = self.output_layout
        gated = torch.empty_strided(shape, stride, dtype=dtype, device=query.device)
        scratch = [torch.empty(s_shape, dtype=s_dtype, device=query.device) for s_shape, s_dtype in self.scratch_layout]
        _execute_relative_decode(
            query,
            new_key,
            new_value,
            kc,
            vc,
            sk,
            sv,
            pt,
            sr,
            starts,
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
