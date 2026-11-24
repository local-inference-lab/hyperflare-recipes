# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared SM120 split-K BF16 router projection and fused top-8 routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from b12x._lib.compile_plan import attach_programs, compile_only_launches, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare

from .native_operator import compile_triton, is_sm120, native_contract, native_plan
from .selected_attention import metadata_tensor, tensor_facts


def _has_splitk_router_kernel() -> bool:
    try:
        from b12x.moe._inkling2_splitk_router import SplitKRouterGemvKernel  # noqa: F401
    except ImportError:
        return False
    return True


def _splitk_router_splits(k: int) -> int:
    """Match ``TensorCoreGemvKernel`` warp count for deterministic split-K reduction."""
    chunks = k // 64
    splits = min(8, chunks)
    while chunks % splits:
        splits -= 1
    return splits


@dataclass(frozen=True)
class SplitKRouteOutput:
    """Router outputs matching ``RouterScratch`` fields plus FP32 logits."""

    logits: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor


@triton.jit
def _reduce_and_route(
    Partials,
    Logits,
    Bias,
    Ids,
    Weights,
    NumValidTokens,
    logit_stride,
    id_stride,
    weight_stride,
    route_scale,
    T,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_VALID_TOKENS: tl.constexpr,
):
    token = tl.program_id(0)
    experts = tl.arange(0, BLOCK_E)
    valid = experts < E
    logits = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        logits = logits + tl.load(Partials + (s * T + token) * E + experts, valid, other=0.0)
    tl.store(Logits + token * logit_stride + experts, logits, valid)
    bias = tl.load(Bias + experts, valid, other=0.0)
    selection = tl.where(valid, tl.sigmoid(logits) + bias, -float("inf"))
    log_sig = tl.minimum(logits, 0.0) - libdevice.log1p(libdevice.exp(-tl.abs(logits)))
    slots = tl.arange(0, 8)
    ids_vec = tl.zeros((8,), dtype=tl.int32)
    p0 = tl.full((), 0.0, tl.float32)
    p1 = tl.full((), 0.0, tl.float32)
    p2 = tl.full((), 0.0, tl.float32)
    p3 = tl.full((), 0.0, tl.float32)
    p4 = tl.full((), 0.0, tl.float32)
    p5 = tl.full((), 0.0, tl.float32)
    p6 = tl.full((), 0.0, tl.float32)
    p7 = tl.full((), 0.0, tl.float32)
    for slot in tl.static_range(8):
        chosen = tl.argmax(selection, 0, tie_break_left=True)
        val = tl.sum(tl.where(experts == chosen, log_sig, 0.0), 0)
        ids_vec = tl.where(slots == slot, chosen.to(tl.int32), ids_vec)
        if slot == 0:
            p0 = val
        if slot == 1:
            p1 = val
        if slot == 2:
            p2 = val
        if slot == 3:
            p3 = val
        if slot == 4:
            p4 = val
        if slot == 5:
            p5 = val
        if slot == 6:
            p6 = val
        if slot == 7:
            p7 = val
        selection = tl.where(experts == chosen, -float("inf"), selection)
    probabilities = tl.where(
        slots == 0,
        p0,
        tl.where(
            slots == 1,
            p1,
            tl.where(
                slots == 2,
                p2,
                tl.where(
                    slots == 3,
                    p3,
                    tl.where(
                        slots == 4,
                        p4,
                        tl.where(slots == 5, p5, tl.where(slots == 6, p6, p7)),
                    ),
                ),
            ),
        ),
    )
    maximum = tl.max(probabilities, 0)
    total = (
        (libdevice.exp(p0 - maximum) + libdevice.exp(p2 - maximum))
        + (libdevice.exp(p1 - maximum) + libdevice.exp(p3 - maximum))
    ) + (
        (libdevice.exp(p4 - maximum) + libdevice.exp(p6 - maximum))
        + (libdevice.exp(p5 - maximum) + libdevice.exp(p7 - maximum))
    )
    normalized = libdevice.exp(probabilities - (maximum + libdevice.log(total))) * route_scale
    if HAS_VALID_TOKENS:
        live = token < tl.load(NumValidTokens)
        ids_vec = tl.where(live, ids_vec, tl.full((8,), -1, dtype=tl.int32))
        normalized = tl.where(live, normalized, tl.zeros((8,), dtype=tl.float32))
    tl.store(Ids + token * id_stride + slots, ids_vec)
    tl.store(Weights + token * weight_stride + slots, normalized)


@dataclass(frozen=True)
class SplitKRouterQuery:
    operands: tuple
    tokens: int
    experts: int
    hidden: int
    splits: int
    route_scale: float
    has_valid_tokens: bool = False


SPLITK_ROUTER_CONTRACT = native_contract(
    component_id="flare.inkling2_splitk_router",
    query_type=SplitKRouterQuery,
    tactics=lambda q: ((triton.next_power_of_2(q.experts), 1 if q.experts <= 256 else 4),),
    semantic_version=2,
)


@program_cache(scope="preparation")
def compile_splitk_router(payload, tactic, ordinal):
    query = SplitKRouterQuery(**dict(payload))
    tensors = tuple(map(metadata_tensor, query.operands))
    _source, _weight, bias, partials, logits, ids, weights, valid_tokens = tensors
    block, warps = tactic

    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x._lib.utils.device import current_cuda_stream
    from b12x._lib.utils.pointers import make_ptr
    from b12x.moe._inkling2_splitk_router import SplitKRouterGemvKernel

    gemv_kernel = SplitKRouterGemvKernel(query.experts, query.hidden, query.splits)
    p_bf16 = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    p_f32 = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16)
    with torch.cuda.device(ordinal):
        gemv_raw = b12x_compile(
            gemv_kernel,
            p_bf16,
            p_bf16,
            p_f32,
            Int32(1),
            Int64(query.hidden),
            Int64(query.hidden),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "flare.inkling2_splitk_router_gemv",
                2,
                (query.experts, query.hidden, query.splits),
            ),
        )
    with torch.cuda.device(ordinal), compile_only_launches():
        reduce_prog = compile_triton(
            _reduce_and_route,
            (query.tokens, 1, 1),
            partials,
            logits,
            bias,
            ids,
            weights,
            valid_tokens,
            logits.stride(0),
            ids.stride(0),
            weights.stride(0),
            query.route_scale,
            query.tokens,
            query.experts,
            block,
            query.splits,
            query.has_valid_tokens,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return dict(gemv=gemv_raw, reduce=reduce_prog)


@dataclass(frozen=True)
class SplitKRouterState:
    tactic: tuple
    programs: dict
    query: SplitKRouterQuery
    dummy_valid_tokens: torch.Tensor

    def run(self, tensors):
        import cutlass.cute as cute
        from cutlass import BFloat16, Float32, Int32, Int64
        from b12x._lib.compiler import run_compiled
        from b12x._lib.utils.device import current_cuda_stream
        from b12x._lib.utils.pointers import make_ptr

        source, weight, bias, partials, logits, ids, weights, valid_tokens = tensors
        if valid_tokens is None:
            valid_tokens = self.dummy_valid_tokens
        run_compiled(
            self.programs["gemv"],
            (
                make_ptr(BFloat16, source.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(Float32, partials.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                Int32(source.shape[0]),
                Int64(source.stride(0)),
                Int64(weight.stride(0)),
                current_cuda_stream(),
            ),
        )
        self.programs["reduce"][(self.query.tokens, 1, 1)](
            partials,
            logits,
            bias,
            ids,
            weights,
            valid_tokens,
            logits.stride(0),
            ids.stride(0),
            weights.stride(0),
            self.query.route_scale,
            source.shape[0],
        )


@torch.library.custom_op(
    "flare::inkling2_splitk_router",
    mutates_args=("partials", "logits", "ids", "weights"),
)
def _execute_splitk_router(
    source: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    partials: torch.Tensor,
    logits: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    num_valid_tokens: torch.Tensor | None,
    handle: int,
) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, weights.device).run(
        (source, weight, bias, partials, logits, ids, weights, num_valid_tokens)
    )


@_execute_splitk_router.register_fake
def _fake_splitk_router(source, weight, bias, partials, logits, ids, weights, num_valid_tokens, handle):
    pass


def splitk_router_eligible(source, weight, bias, logits, top_k) -> bool:
    """Whether SM120 split-K router projection and fused top-8 routing supports this layer."""
    if not is_sm120(source.device) or not _has_splitk_router_kernel():
        return False
    tokens, hidden = source.shape
    experts = weight.shape[0]
    if logits is not None and (logits.dtype != torch.float32 or not logits.is_contiguous()):
        return False
    return (
        top_k == 8
        and 1 <= tokens <= 8
        and experts % 16 == 0
        and hidden % 64 == 0
        and _splitk_router_splits(hidden) >= 2
        and source.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and bias.dtype == torch.float32
        and source.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and source.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


class PreparedInkling2SplitKRouter:
    """Split-K BF16 router projection with deterministic reduction and top-8 routing."""

    def __init__(
        self,
        source,
        weight,
        bias,
        logits,
        ids,
        weights,
        *,
        session,
        name,
        route_scale,
        num_valid_tokens=None,
    ):
        tokens, hidden = source.shape
        experts = weight.shape[0]
        splits = _splitk_router_splits(hidden)
        if num_valid_tokens is not None and (
            num_valid_tokens.dtype != torch.int32
            or num_valid_tokens.numel() != 1
            or num_valid_tokens.device != source.device
        ):
            raise ValueError("split-K router num_valid_tokens must be a 1-element int32 CUDA tensor")
        for tensor in (weight, bias):
            record_numeric_tensor(tensor)
        self.weight = weight
        self.bias = bias
        self.tokens = tokens
        self.experts = experts
        self.splits = splits
        self.device = source.device
        self.has_valid_tokens = num_valid_tokens is not None
        self.dummy_valid_tokens = torch.full((1,), tokens, dtype=torch.int32, device=source.device)
        valid_arg = self.dummy_valid_tokens if num_valid_tokens is None else num_valid_tokens
        self.partials = torch.empty((splits, tokens, experts), dtype=torch.float32, device=source.device)
        self.arguments = (source, weight, bias, self.partials, logits, ids, weights, valid_arg)
        self.query = SplitKRouterQuery(
            tuple(tensor_facts(t) for t in self.arguments),
            tokens,
            experts,
            hidden,
            splits,
            float(route_scale),
            has_valid_tokens=self.has_valid_tokens,
        )
        payload = SPLITK_ROUTER_CONTRACT.encode_query(self.query)

        def materialize(tactic):
            programs = compile_splitk_router(payload, tactic, source.device.index)
            load_programs(programs)
            dummy = torch.full((1,), tokens, dtype=torch.int32, device=source.device)
            return attach_programs(SplitKRouterState(tactic, programs, self.query, dummy), *programs.values())

        self.plan = native_plan(
            contract=SPLITK_ROUTER_CONTRACT,
            query=self.query,
            device=source.device,
            scratch_bytes=lambda tactic: 0,
            materialize=materialize,
            compile_jobs=lambda tactic, device: (
                CompileJob.create(
                    "hyperflare_recipes.components.inkling2_router_ops:compile_splitk_router",
                    payload,
                    tactic,
                    device.ordinal,
                ),
            ),
        )
        declaration = self.arguments
        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(declaration), owners=declaration
                    ),
                ),
            ),
        )

    def functional(self, *, session=None):
        """Consume forward-local inputs and allocate partials and route outputs locally."""
        del session
        self.arguments = None
        self.partials = None
        return self

    def __call__(self, source=None, *, num_valid_tokens=None):
        if self.arguments is not None and source is None:
            _execute_splitk_router(*self.arguments, self.plan.handle)
            return self.arguments[4], self.arguments[5], self.arguments[6]
        if source is None:
            raise ValueError("functional split-K router requires source activations")
        if self.has_valid_tokens and num_valid_tokens is None:
            raise ValueError("functional split-K router requires num_valid_tokens")
        if not self.has_valid_tokens and num_valid_tokens is not None:
            raise ValueError("split-K router was prepared without num_valid_tokens")
        partials = torch.empty((self.splits, self.tokens, self.experts), dtype=torch.float32, device=source.device)
        logits = torch.empty((self.tokens, self.experts), dtype=torch.float32, device=source.device)
        ids = torch.empty((self.tokens, 8), dtype=torch.int32, device=source.device)
        weights = torch.empty((self.tokens, 8), dtype=torch.float32, device=source.device)
        _execute_splitk_router(
            source,
            self.weight,
            self.bias,
            partials,
            logits,
            ids,
            weights,
            num_valid_tokens,
            self.plan.handle,
        )
        return SplitKRouteOutput(logits=logits, ids=ids, weights=weights)
