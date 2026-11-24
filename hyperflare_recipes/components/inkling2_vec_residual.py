# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared SM120 vectorized attention-residual mix with fused RMSNorm."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare

from .native_operator import is_sm120, native_contract, native_plan


def _has_vec_residual_kernel() -> bool:
    try:
        from b12x.norm.inkling2_residual import VecResidualMixKernel  # noqa: F401
    except ImportError:
        return False
    return True


def supports_vec_residual_mix(tokens: int, hidden: int, sources: int, device) -> bool:
    """Whether SM120 ``VecResidualMixKernel`` supports this residual mixture geometry."""
    return (
        is_sm120(device)
        and _has_vec_residual_kernel()
        and 1 <= tokens <= 32
        and hidden > 0
        and hidden % 256 == 0
        and hidden // 8 <= 1024
        and 1 <= sources <= 8
    )


@dataclass(frozen=True)
class VecResidualMixQuery:
    hidden: int
    sources: int
    has_addend: bool
    normalize: bool
    eps: float
    norm_eps: float


VEC_RESIDUAL_MIX_CONTRACT = native_contract(
    component_id="flare.inkling2_vec_residual_mix",
    query_type=VecResidualMixQuery,
    tactics=lambda q: (("vec",),),
    semantic_version=1,
)


@program_cache(scope="preparation")
def compile_vec_residual_mix(payload, tactic, ordinal):
    del tactic
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x._lib.utils.device import current_cuda_stream
    from b12x._lib.utils.pointers import make_ptr
    from b12x.norm.inkling2_residual import VecResidualMixKernel

    query = VecResidualMixQuery(**dict(payload))
    kernel = VecResidualMixKernel(
        query.hidden,
        query.sources,
        query.has_addend,
        query.normalize,
        query.eps,
        query.norm_eps,
    )
    p_bf16 = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    p_f32 = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16)
    key = (query.hidden, query.sources, query.has_addend, query.normalize, query.eps, query.norm_eps)
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            p_bf16,
            p_bf16,
            p_bf16,
            p_f32,
            p_f32,
            p_bf16,
            Int64(query.hidden),
            Int64(query.hidden),
            Int64(query.hidden),
            Int64(query.hidden),
            Int64(query.hidden),
            Int64(query.hidden),
            Int32(1),
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("flare.inkling2_vec_residual_mix", 3, key),
        )
    return dict(vec=raw)


@dataclass(frozen=True)
class VecResidualMixState:
    query: VecResidualMixQuery
    programs: dict

    def run(self, blocks, last, addend, queries, weight, output):
        import cutlass.cute as cute
        from cutlass import BFloat16, Float32, Int32, Int64
        from b12x._lib.compiler import run_compiled
        from b12x._lib.utils.device import current_cuda_stream
        from b12x._lib.utils.pointers import make_ptr

        tokens = int(last.shape[0])
        query_rows = int(queries.shape[0])
        run_compiled(
            self.programs["vec"],
            (
                make_ptr(BFloat16, blocks.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(BFloat16, last.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(BFloat16, addend.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(Float32, queries.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(Float32, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(BFloat16, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                Int64(blocks.stride(0)),
                Int64(blocks.stride(1)),
                Int64(last.stride(0)),
                Int64(addend.stride(0)),
                Int64(output.stride(0)),
                Int64(output.stride(1)),
                Int32(tokens),
                Int32(query_rows),
                current_cuda_stream(),
            ),
        )


@torch.library.custom_op("flare::inkling2_vec_residual_mix", mutates_args=("output",))
def _execute_vec_residual_mix(
    blocks: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor,
    queries: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    handle: int,
) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, output.device).run(
        blocks, last, addend, queries, weight, output
    )


@_execute_vec_residual_mix.register_fake
def _fake_vec_residual_mix(blocks, last, addend, queries, weight, output, handle):
    pass


class PreparedVecResidualMix:
    """Single-pass SM120 vectorized residual mix with fused output RMSNorm."""

    def __init__(self, source, queries, norms, *, sources, eps, addend, session, name):
        tokens, hidden = source.shape
        streams = len(norms)
        if not supports_vec_residual_mix(tokens, hidden, sources, source.device):
            raise ValueError("vectorized residual mix requires SM120 with tokens <= 32, hidden % 256 == 0, and 1 <= sources <= 8")
        self.tokens = tokens
        self.hidden = hidden
        self.sources = sources
        self.streams = streams
        self.addend = bool(addend)
        self.queries = queries.contiguous()
        self.weight = persistent_copy(torch.stack([n.float() for n in norms], dim=0).contiguous())
        record_numeric_tensor(self.queries)
        record_numeric_tensor(self.weight)

        self.query = VecResidualMixQuery(
            hidden=hidden,
            sources=sources,
            has_addend=self.addend,
            normalize=True,
            eps=float(eps),
            norm_eps=float(eps),
        )
        payload = VEC_RESIDUAL_MIX_CONTRACT.encode_query(self.query)

        def materialize(tactic):
            programs = compile_vec_residual_mix(payload, tactic, source.device.index)
            load_programs(programs)
            return attach_programs(VecResidualMixState(self.query, programs), *programs.values())

        if session is None:
            self.plan = None
            self._eager_state = materialize(("vec",))
            return

        self._eager_state = None
        self.plan = native_plan(
            contract=VEC_RESIDUAL_MIX_CONTRACT,
            query=self.query,
            device=source.device,
            scratch_bytes=lambda tactic: 0,
            materialize=materialize,
            compile_jobs=lambda tactic, device: (
                CompileJob.create(
                    "hyperflare_recipes.components.inkling2_vec_residual:compile_vec_residual_mix",
                    payload,
                    tactic,
                    device.ordinal,
                ),
            ),
        )
        blocks_ex = torch.zeros((max(1, sources - 1), tokens, hidden), dtype=torch.bfloat16, device=source.device)
        out_ex = torch.empty((streams, tokens, hidden), dtype=torch.bfloat16, device=source.device)
        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(blocks_ex, source, source, self.queries, self.weight, out_ex),
                        owners=(blocks_ex, source, self.queries, self.weight, out_ex),
                    ),
                ),
            ),
        )

    def __call__(self, sources, *, addend=None):
        last = sources[-1]
        if self.sources == 1:
            blocks = last.unsqueeze(0)
        elif self.sources == 2:
            blocks = sources[0].unsqueeze(0)
        else:
            blocks = torch.stack(sources[:-1], dim=0)
        extra = last if addend is None else addend
        output = torch.empty((self.streams, self.tokens, self.hidden), dtype=torch.bfloat16, device=last.device)
        if self.plan is None:
            self._eager_state.run(blocks, last, extra, self.queries, self.weight, output)
        else:
            _execute_vec_residual_mix(blocks, last, extra, self.queries, self.weight, output, self.plan.handle)
        return tuple(output[i] for i in range(self.streams))
