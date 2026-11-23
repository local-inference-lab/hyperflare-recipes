# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 overlapped shifted-mHC seam and direct F8_128x4 RMSNorm kernels.

For shifted mHC (``hc_shifted == True``), the next sub-layer's pre-mix weights
are already produced by the preceding sub-layer. During small-batch decode
(``M <= 64``), the critical path only needs the single-stream pre-mixed input
normalized and quantized into the consumer's scale layout (``F8_128x4`` for
SM103 FlashInfer projections or MegaMoE routed/shared scales for experts).

``_mhc_seam_1pass`` produces ``y_bf16``, ``y_fp8``, and consumer scales on the
main stream in one pass, while ``deep_gemm.mega_mhc`` updates the 4-stream
residual trunk and Sinkhorn coefficients concurrently on an auxiliary stream
hidden behind Attention or MegaMoE. Every input and output tensor of the
auxiliary launch is passed to the matching stream join op so TorchInductor and
``ScratchArena`` preserve live lifetimes until the auxiliary stream drains.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.tuning import declare

from .native_operator import native_contract, native_plan

MHC_SEAM_COMPONENT = "flare.sm103.mhc_seam"
RMSNORM_F8_COMPONENT = "flare.sm103.rmsnorm_f8_128x4"
MHC_SEAM_MAX_ROWS = 64


@triton.jit
def _mhc_seam_1pass(
    X, Res, PrevPre, PostMix, Comb, WNorm,
    YBf16, YFp8, YSf, YSharedSf,
    r_sf_stride_m, s_sf_stride_k,
    eps, out_scale,
    H: tl.constexpr,
    G_PAD: tl.constexpr,
    MODE: tl.constexpr,            # 0 = F8_128x4 uint8 (int32 view), 1 = MegaMoE routed + shared int32
    SHARED_BLOCK_M: tl.constexpr,  # 0 if no shared_sf, else MegaMoE shared_sf_block_m
):
    m = tl.program_id(0).to(tl.int64)
    p0 = tl.load(PrevPre + m * 4 + 0)
    p1 = tl.load(PrevPre + m * 4 + 1)
    p2 = tl.load(PrevPre + m * 4 + 2)
    p3 = tl.load(PrevPre + m * 4 + 3)
    po0 = tl.load(PostMix + m * 4 + 0)
    po1 = tl.load(PostMix + m * 4 + 1)
    po2 = tl.load(PostMix + m * 4 + 2)
    po3 = tl.load(PostMix + m * 4 + 3)
    c00 = tl.load(Comb + m * 16 + 0)
    c01 = tl.load(Comb + m * 16 + 1)
    c02 = tl.load(Comb + m * 16 + 2)
    c03 = tl.load(Comb + m * 16 + 3)
    c10 = tl.load(Comb + m * 16 + 4)
    c11 = tl.load(Comb + m * 16 + 5)
    c12 = tl.load(Comb + m * 16 + 6)
    c13 = tl.load(Comb + m * 16 + 7)
    c20 = tl.load(Comb + m * 16 + 8)
    c21 = tl.load(Comb + m * 16 + 9)
    c22 = tl.load(Comb + m * 16 + 10)
    c23 = tl.load(Comb + m * 16 + 11)
    c30 = tl.load(Comb + m * 16 + 12)
    c31 = tl.load(Comb + m * 16 + 13)
    c32 = tl.load(Comb + m * 16 + 14)
    c33 = tl.load(Comb + m * 16 + 15)

    g = tl.arange(0, G_PAD)
    e = tl.arange(0, 32)
    col = g[:, None] * 32 + e[None, :]
    mask = col < H

    xv = tl.load(X + m * H + col, mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(Res + m * (4 * H) + 0 * H + col, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(Res + m * (4 * H) + 1 * H + col, mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(Res + m * (4 * H) + 2 * H + col, mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(Res + m * (4 * H) + 3 * H + col, mask=mask, other=0.0).to(tl.float32)

    nr0 = (po0 * xv + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3).to(tl.bfloat16).to(tl.float32)
    nr1 = (po1 * xv + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3).to(tl.bfloat16).to(tl.float32)
    nr2 = (po2 * xv + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3).to(tl.bfloat16).to(tl.float32)
    nr3 = (po3 * xv + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3).to(tl.bfloat16).to(tl.float32)

    hin = (p0 * nr0 + p1 * nr1 + p2 * nr2 + p3 * nr3).to(tl.bfloat16).to(tl.float32)
    hin = tl.where(mask, hin, 0.0)
    sum_sq = tl.sum(hin * hin)
    rstd = tl.math.rsqrt(sum_sq * (1.0 / H) + eps) * out_scale

    w = tl.load(WNorm + col, mask=mask, other=0.0).to(tl.float32)
    y = (hin * rstd * w).to(tl.bfloat16)
    tl.store(YBf16 + m * H + col, y, mask=mask)

    yf = y.to(tl.float32)
    amax = tl.max(tl.abs(yf), axis=1)
    raw_sf = tl.maximum(amax * (1.0 / 448.0), 1e-30)
    bits = raw_sf.to(tl.int32, bitcast=True)
    exp_u8 = ((bits + 0x7FFFFF) >> 23) & 0xFF
    scale_f = (exp_u8 << 23).to(tl.float32, bitcast=True)
    q = (yf / scale_f[:, None]).to(tl.float8e4nv)
    tl.store(YFp8 + m * H + col, q, mask=mask)

    exp4 = tl.reshape(exp_u8, (G_PAD // 4, 4))
    shifts = (tl.arange(0, 4) * 8)[None, :]
    packed = tl.sum(exp4 << shifts, axis=1)
    kt = tl.arange(0, G_PAD // 4)
    kt_mask = kt < (H // 128)
    if MODE == 0:
        dst_off = kt * 128 + (m % 32) * 4 + (m // 32)
        tl.store(YSf + dst_off, packed, mask=kt_mask)
    else:
        tl.store(YSf + m * r_sf_stride_m + kt, packed, mask=kt_mask)
        if SHARED_BLOCK_M > 0:
            blk = m // SHARED_BLOCK_M
            in_blk = m % SHARED_BLOCK_M
            dst_row = blk * 128 + (in_blk % 32) * 4 + (in_blk // 32)
            tl.store(YSharedSf + dst_row + kt * s_sf_stride_k, packed, mask=kt_mask)


@triton.jit
def _rmsnorm_f8_128x4(
    X, W, YFp8, YSf, YBf16,
    x_stride_m,
    eps, out_scale,
    H: tl.constexpr,
    G_PAD: tl.constexpr,
    WRITE_BF16: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, G_PAD)
    e = tl.arange(0, 32)
    col = g[:, None] * 32 + e[None, :]
    mask = col < H
    xv = tl.load(X + m * x_stride_m + col, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(xv * xv)
    rstd = tl.math.rsqrt(sum_sq * (1.0 / H) + eps) * out_scale
    w = tl.load(W + col, mask=mask, other=0.0).to(tl.float32)
    y = (xv * rstd * w).to(tl.bfloat16)
    if WRITE_BF16:
        tl.store(YBf16 + m * H + col, y, mask=mask)
    yf = y.to(tl.float32)
    amax = tl.max(tl.abs(yf), axis=1)
    raw_sf = tl.maximum(amax * (1.0 / 448.0), 1e-30)
    bits = raw_sf.to(tl.int32, bitcast=True)
    exp_u8 = ((bits + 0x7FFFFF) >> 23) & 0xFF
    scale_f = (exp_u8 << 23).to(tl.float32, bitcast=True)
    q = (yf / scale_f[:, None]).to(tl.float8e4nv)
    tl.store(YFp8 + m * H + col, q, mask=mask)
    exp4 = tl.reshape(exp_u8, (G_PAD // 4, 4))
    shifts = (tl.arange(0, 4) * 8)[None, :]
    packed = tl.sum(exp4 << shifts, axis=1)
    kt = tl.arange(0, G_PAD // 4)
    kt_mask = kt < (H // 128)
    dst_off = kt * 128 + (m % 32) * 4 + (m // 32)
    tl.store(YSf + dst_off, packed, mask=kt_mask)


_NATIVE_JITS.update((_mhc_seam_1pass, _rmsnorm_f8_128x4))


@dataclass(frozen=True)
class MHCSeamQuery:
    rows: int
    hidden_size: int
    mode: int            # 0 = F8_128x4, 1 = MegaMoE
    shared_block_m: int  # 0 when mode == 0 or no shared_sf

    def __post_init__(self):
        if (type(self.rows) is not int or not 1 <= self.rows <= MHC_SEAM_MAX_ROWS
                or type(self.hidden_size) is not int or self.hidden_size <= 0 or self.hidden_size % 128
                or self.mode not in (0, 1)
                or type(self.shared_block_m) is not int or self.shared_block_m < 0):
            raise ValueError("SM103 mHC seam requires 1..64 rows and 128-aligned hidden size")


MHC_SEAM_CONTRACT = native_contract(
    component_id=MHC_SEAM_COMPONENT,
    query_type=MHCSeamQuery,
    tactics=lambda query: (16,),
)


@program_cache(scope="preparation")
def compile_mhc_seam(payload, ordinal):
    query = MHCSeamQuery(**payload["query"])
    num_warps = payload["num_warps"]
    g_pad = triton.next_power_of_2(query.hidden_size // 32)
    with torch.cuda.device(ordinal):
        return (_mhc_seam_1pass.warmup(
            torch.bfloat16, torch.bfloat16, torch.float32, torch.float32, torch.float32, torch.bfloat16,
            torch.bfloat16, torch.float8_e4m3fn, torch.int32, torch.int32,
            40, 1024, 1e-20, 1.0,
            H=query.hidden_size, G_PAD=g_pad, MODE=query.mode, SHARED_BLOCK_M=query.shared_block_m,
            grid=(query.rows, 1, 1), num_warps=num_warps,
        ),)


class PreparedMHCSeam:
    def __init__(self, query: MHCSeamQuery, num_warps: int, device):
        self.query = query
        self.g_pad = triton.next_power_of_2(query.hidden_size // 32)
        self.device = torch.device(device)
        self.kernel, = load_programs(compile_mhc_seam(
            dict(query=MHC_SEAM_CONTRACT.encode_query(query), num_warps=num_warps), self.device.index))

    def run(self, x, res, prev_pre, post_mix, comb, w_norm, y_bf16, y_fp8, y_sf, y_shared_sf, eps, out_scale):
        q = self.query
        r_stride = y_sf.stride(0) if q.mode == 1 else 0
        s_stride = y_shared_sf.stride(1) if (q.mode == 1 and q.shared_block_m > 0) else 0
        shared_ptr = y_shared_sf if (q.mode == 1 and q.shared_block_m > 0) else y_sf
        self.kernel[(q.rows, 1, 1)](
            x, res, prev_pre, post_mix, comb, w_norm,
            y_bf16, y_fp8, y_sf, shared_ptr,
            r_stride, s_stride, eps, out_scale,
            q.hidden_size, self.g_pad, q.mode, q.shared_block_m,
        )


def mhc_seam_plan(query: MHCSeamQuery, device):
    resolved = torch.device(device)
    return native_plan(
        contract=MHC_SEAM_CONTRACT,
        query=query,
        device=resolved,
        scratch_bytes=lambda tactic: 0,
        compile_jobs=lambda tactic, detected: (CompileJob.create(
            "hyperflare_recipes.components.sm103_mhc_overlap:compile_mhc_seam",
            dict(query=MHC_SEAM_CONTRACT.encode_query(query), num_warps=tactic), detected.ordinal),),
        materialize=lambda tactic: PreparedMHCSeam(query, tactic, resolved),
    )


@torch.library.custom_op(
    "flare::sm103_mhc_seam_and_aux",
    mutates_args=(
        "y_bf16", "y_fp8", "y_sf", "y_shared_sf",
        "new_residual", "new_prev_mix", "new_post_mix", "new_comb_res_mix", "y_bf16_dummy",
    ),
)
def _mhc_seam_and_aux(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    hc_mult: int,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_weight: torch.Tensor,
    rmsnorm_eps: float,
    rmsnorm_scale: float,
    y_bf16: torch.Tensor,
    y_fp8: torch.Tensor,
    y_sf: torch.Tensor,
    y_shared_sf: torch.Tensor | None,
    new_residual: torch.Tensor,
    new_prev_mix: torch.Tensor,
    new_post_mix: torch.Tensor,
    new_comb_res_mix: torch.Tensor,
    y_bf16_dummy: torch.Tensor,
    aux_stream: int,
    plan_handle: int,
) -> None:
    import deep_gemm

    main = torch.cuda.current_stream(x.device)
    auxiliary = torch.cuda.ExternalStream(aux_stream, device=x.device)
    # Fork auxiliary stream before seam so seam (main) and deep_gemm.mega_mhc (aux) run concurrently.
    auxiliary.wait_stream(main)
    require_prepared(plan_from_handle(plan_handle), MHC_SEAM_COMPONENT, x.device).run(
        x, residual, prev_pre_mix, post_mix, comb_res_mix, rmsnorm_weight,
        y_bf16, y_fp8, y_sf, y_shared_sf, rmsnorm_eps, rmsnorm_scale,
    )
    with torch.cuda.stream(auxiliary):
        deep_gemm.mega_mhc(
            x, residual, prev_pre_mix, post_mix, comb_res_mix,
            fn=fn, mix_scales=mix_scales, mix_bases=mix_bases, hc_mult=hc_mult,
            hc_norm_eps=hc_norm_eps, hc_pre_eps=hc_pre_eps, hc_post_scale=hc_post_scale,
            sinkhorn_eps=sinkhorn_eps, num_sinkhorn_iters=num_sinkhorn_iters,
            rmsnorm_weight=rmsnorm_weight, rmsnorm_eps=rmsnorm_eps, rmsnorm_scale=rmsnorm_scale,
            new_residual=new_residual, new_prev_mix=new_prev_mix,
            new_post_mix=new_post_mix, new_comb_res_mix=new_comb_res_mix,
            y_bf16=y_bf16_dummy, y_fp8=None, y_gemm_sf=None,
            y_routed_sf=None, y_shared_sf=None, shared_sf_block_m=0,
        )


@_mhc_seam_and_aux.register_fake
def _mhc_seam_and_aux_fake(
    x, residual, prev_pre_mix, post_mix, comb_res_mix, fn, mix_scales, mix_bases,
    hc_mult, hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters,
    rmsnorm_weight, rmsnorm_eps, rmsnorm_scale, y_bf16, y_fp8, y_sf, y_shared_sf,
    new_residual, new_prev_mix, new_post_mix, new_comb_res_mix, y_bf16_dummy,
    aux_stream, plan_handle,
):
    pass


@torch.library.custom_op(
    "flare::sm103_mhc_aux_join",
    mutates_args=("new_residual", "new_prev_mix", "new_post_mix", "new_comb_res_mix", "y_bf16_dummy"),
)
def _mhc_aux_join(
    anchor: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    new_residual: torch.Tensor,
    new_prev_mix: torch.Tensor,
    new_post_mix: torch.Tensor,
    new_comb_res_mix: torch.Tensor,
    y_bf16_dummy: torch.Tensor,
    aux_stream: int,
) -> None:
    # Keep all auxiliary-read inputs and auxiliary-written outputs live until main_stream joins aux_stream.
    del x, residual, prev_pre_mix, post_mix, comb_res_mix
    del new_residual, new_prev_mix, new_post_mix, new_comb_res_mix, y_bf16_dummy
    torch.cuda.current_stream(anchor.device).wait_stream(
        torch.cuda.ExternalStream(aux_stream, device=anchor.device))


@_mhc_aux_join.register_fake
def _mhc_aux_join_fake(
    anchor, x, residual, prev_pre_mix, post_mix, comb_res_mix,
    new_residual, new_prev_mix, new_post_mix, new_comb_res_mix, y_bf16_dummy, aux_stream,
):
    pass


def join_pending_mhc(anchor: torch.Tensor, lazy_data) -> None:
    """Synchronize main_stream with any pending auxiliary mHC update attached to ``lazy_data``."""
    pending = getattr(lazy_data, "_aux_pending", None)
    if pending is not None:
        _mhc_aux_join(anchor, *pending)
        lazy_data._aux_pending = None


@dataclass(frozen=True)
class RMSNormF8Query:
    rows: int
    hidden_size: int
    return_bf16: bool = False

    def __post_init__(self):
        if (type(self.rows) is not int or not 1 <= self.rows <= MHC_SEAM_MAX_ROWS
                or type(self.hidden_size) is not int or self.hidden_size <= 0 or self.hidden_size % 128
                or type(self.return_bf16) is not bool):
            raise ValueError("SM103 RMSNorm F8_128x4 requires 1..64 rows and 128-aligned hidden size")


RMSNORM_F8_CONTRACT = native_contract(
    component_id=RMSNORM_F8_COMPONENT,
    query_type=RMSNormF8Query,
    tactics=lambda query: (16 if query.hidden_size >= 4096 else 4,),
    semantic_version=2,
)


@program_cache(scope="preparation")
def compile_rmsnorm_f8_128x4(payload, ordinal):
    query = RMSNormF8Query(**payload["query"])
    num_warps = payload["num_warps"]
    g_pad = triton.next_power_of_2(query.hidden_size // 32)
    with torch.cuda.device(ordinal):
        return (_rmsnorm_f8_128x4.warmup(
            torch.bfloat16, torch.bfloat16, torch.float8_e4m3fn, torch.int32, torch.bfloat16,
            query.hidden_size, 1e-20, 1.0,
            H=query.hidden_size, G_PAD=g_pad, WRITE_BF16=1 if query.return_bf16 else 0,
            grid=(query.rows, 1, 1), num_warps=num_warps,
        ),)


class PreparedRMSNormF8:
    def __init__(self, query: RMSNormF8Query, num_warps: int, device):
        self.query = query
        self.g_pad = triton.next_power_of_2(query.hidden_size // 32)
        self.device = torch.device(device)
        self.write_bf16 = 1 if query.return_bf16 else 0
        self.kernel, = load_programs(compile_rmsnorm_f8_128x4(
            dict(query=RMSNORM_F8_CONTRACT.encode_query(query), num_warps=num_warps), self.device.index))

    def run(self, x, weight, y_fp8, y_sf, y_bf16, eps, out_scale):
        bf16_ptr = y_bf16 if y_bf16 is not None else x
        self.kernel[(self.query.rows, 1, 1)](
            x, weight, y_fp8, y_sf, bf16_ptr,
            x.stride(0), eps, out_scale,
            self.query.hidden_size, self.g_pad, self.write_bf16,
        )


def rmsnorm_f8_plan(query: RMSNormF8Query, device):
    resolved = torch.device(device)
    return native_plan(
        contract=RMSNORM_F8_CONTRACT,
        query=query,
        device=resolved,
        scratch_bytes=lambda tactic: 0,
        compile_jobs=lambda tactic, detected: (CompileJob.create(
            "hyperflare_recipes.components.sm103_mhc_overlap:compile_rmsnorm_f8_128x4",
            dict(query=RMSNORM_F8_CONTRACT.encode_query(query), num_warps=tactic), detected.ordinal),),
        materialize=lambda tactic: PreparedRMSNormF8(query, tactic, resolved),
    )


@torch.library.custom_op("flare::sm103_rmsnorm_f8_128x4", mutates_args=("y_fp8", "y_sf"))
def _run_rmsnorm_f8(
    x: torch.Tensor, weight: torch.Tensor, y_fp8: torch.Tensor, y_sf: torch.Tensor,
    eps: float, out_scale: float, plan_handle: int,
) -> None:
    require_prepared(plan_from_handle(plan_handle), RMSNORM_F8_COMPONENT, x.device).run(
        x, weight, y_fp8, y_sf, None, eps, out_scale)


@_run_rmsnorm_f8.register_fake
def _run_rmsnorm_f8_fake(x, weight, y_fp8, y_sf, eps, out_scale, plan_handle):
    pass


@torch.library.custom_op("flare::sm103_rmsnorm_f8_128x4_bf16", mutates_args=("y_fp8", "y_sf", "y_bf16"))
def _run_rmsnorm_f8_bf16(
    x: torch.Tensor, weight: torch.Tensor, y_fp8: torch.Tensor, y_sf: torch.Tensor,
    y_bf16: torch.Tensor, eps: float, out_scale: float, plan_handle: int,
) -> None:
    require_prepared(plan_from_handle(plan_handle), RMSNORM_F8_COMPONENT, x.device).run(
        x, weight, y_fp8, y_sf, y_bf16, eps, out_scale)


@_run_rmsnorm_f8_bf16.register_fake
def _run_rmsnorm_f8_bf16_fake(x, weight, y_fp8, y_sf, y_bf16, eps, out_scale, plan_handle):
    pass


class FastRMSNormF8(nn.Module):
    """Wrap DeepInfer RMSNorm to emit F8_128x4 directly for small decode rows."""

    def __init__(self, base, consumer, rows_list, *, session, name: str, return_bf16: bool = False):
        super().__init__()
        self.base = base
        self.consumer = consumer
        self.hidden_size = base.weight.numel()
        self.variance_epsilon = base.variance_epsilon
        self.use_fp8 = base.use_fp8
        self.out_scale = base.out_scale
        self.return_bf16 = return_bf16
        device = base.weight.device
        self._plans = {}
        requests = []
        for rows in sorted({r for r in rows_list if 1 <= r <= MHC_SEAM_MAX_ROWS}):
            query = RMSNormF8Query(rows, self.hidden_size, return_bf16=return_bf16)
            op_plan = rmsnorm_f8_plan(query, device)
            self._plans[rows] = op_plan

            def make_prepare(rows=rows, h=self.hidden_size, ret_bf16=return_bf16):
                def prepare(state):
                    x = torch.zeros((rows, h), dtype=torch.bfloat16, device=device)
                    y_fp8 = torch.empty((rows, h), dtype=torch.float8_e4m3fn, device=device)
                    y_sf = torch.empty((128 * (h // 128),), dtype=torch.int32, device=device)
                    y_bf16 = torch.empty((rows, h), dtype=torch.bfloat16, device=device) if ret_bf16 else None
                    return PreparedCall(
                        run=lambda: state.run(x, base.weight, y_fp8, y_sf, y_bf16,
                                              self.variance_epsilon, self.out_scale),
                        output=y_fp8,
                        owners=(base.weight,),
                    )
                return prepare

            requests.append(op_plan.request(name=f"{name}.m{rows}", prepare_call=make_prepare()))
        if requests:
            declare(session, tuple(requests))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x, residual=None, out=None, return_bf16=False, **kwargs):
        if (residual is None and out is None and bool(return_bf16) == self.return_bf16
                and not kwargs and x.ndim == 2 and x.dtype == torch.bfloat16
                and x.stride(1) == 1 and self.base.weight.dtype == torch.bfloat16):
            rows = x.shape[0]
            op_plan = self._plans.get(rows)
            if op_plan is not None and self.consumer.prefers_f8_128x4_scales(rows):
                y_fp8 = torch.empty((rows, self.hidden_size), dtype=torch.float8_e4m3fn, device=x.device)
                y_sf = torch.empty((128 * (self.hidden_size // 128),), dtype=torch.int32, device=x.device)
                if self.return_bf16:
                    y_bf16 = torch.empty((rows, self.hidden_size), dtype=torch.bfloat16, device=x.device)
                    _run_rmsnorm_f8_bf16(x, self.base.weight, y_fp8, y_sf, y_bf16,
                                         self.variance_epsilon, self.out_scale, op_plan.handle)
                    return (y_fp8, y_sf.view(torch.uint8)), y_bf16
                _run_rmsnorm_f8(x, self.base.weight, y_fp8, y_sf,
                                self.variance_epsilon, self.out_scale, op_plan.handle)
                return (y_fp8, y_sf.view(torch.uint8))
        return self.base(x, residual=residual, out=out, return_bf16=return_bf16, **kwargs)


def _declare_mhc_seam_plans(layer, rows_list, *, hidden_size: int, mode: int, shared_block_m_by_rows: dict,
                            session, name: str):
    """Declare Triton seam plans and warm up deep_gemm.mega_mhc(..., y_fp8=None) outside capture."""
    import deep_gemm

    device = layer.hc_attn.device
    plans = {}
    requests = []
    for rows in sorted({r for r in rows_list if 1 <= r <= MHC_SEAM_MAX_ROWS}):
        sbm = int(shared_block_m_by_rows.get(rows, 0)) if mode == 1 else 0
        query = MHCSeamQuery(rows, hidden_size, mode, sbm)
        op_plan = mhc_seam_plan(query, device)
        plans[rows] = (op_plan, sbm)

        def make_prepare(rows=rows, h=hidden_size, mode=mode, sbm=sbm, layer=layer):
            def prepare(state):
                x = torch.zeros((rows, h), dtype=torch.bfloat16, device=device)
                res = torch.zeros((rows, layer.hc_mult, h), dtype=torch.bfloat16, device=device)
                prev_pre = torch.zeros((rows, layer.hc_mult, 1), dtype=torch.float32, device=device)
                post_mix = torch.zeros((rows, layer.hc_mult, 1), dtype=torch.float32, device=device)
                comb = torch.zeros((rows, layer.hc_mult, layer.hc_mult), dtype=torch.float32, device=device)
                y_bf16 = torch.empty((rows, h), dtype=torch.bfloat16, device=device)
                y_fp8 = torch.empty((rows, h), dtype=torch.float8_e4m3fn, device=device)
                if mode == 0:
                    y_sf = torch.empty((128 * (h // 128),), dtype=torch.int32, device=device)
                    y_shared_sf = None
                else:
                    y_sf = torch.empty((rows, h // 128), dtype=torch.int32, device=device)
                    if sbm > 0:
                        blocks = (rows + sbm - 1) // sbm
                        y_shared_sf = torch.empty((blocks * 128, h // 128), dtype=torch.int32, device=device)
                    else:
                        y_shared_sf = None
                new_res = torch.empty_like(res)
                new_prev = torch.empty_like(prev_pre)
                new_post = torch.empty_like(post_mix)
                new_comb = torch.empty_like(comb)
                y_dummy = torch.empty_like(y_bf16)
                hc_mod = layer._attn_hc_modules() if mode == 0 else layer._ffn_hc_modules()

                def run_both():
                    state.run(x, res, prev_pre, post_mix, comb, hc_mod.layer_norm.weight,
                              y_bf16, y_fp8, y_sf, y_shared_sf,
                              hc_mod.layer_norm.variance_epsilon, hc_mod.layer_norm.out_scale)
                    deep_gemm.mega_mhc(
                        x, res, prev_pre, post_mix, comb,
                        fn=hc_mod.fn, mix_scales=hc_mod.scale, mix_bases=hc_mod.base,
                        hc_mult=layer.hc_mult, hc_norm_eps=layer.hc_norm_eps,
                        hc_pre_eps=layer.hc_pre_eps, hc_post_scale=layer.hc_post_mult_value,
                        sinkhorn_eps=layer.hc_sinkhorn_eps, num_sinkhorn_iters=layer.hc_sinkhorn_repeat,
                        rmsnorm_weight=hc_mod.layer_norm.weight,
                        rmsnorm_eps=hc_mod.layer_norm.variance_epsilon,
                        rmsnorm_scale=hc_mod.layer_norm.out_scale,
                        new_residual=new_res, new_prev_mix=new_prev,
                        new_post_mix=new_post, new_comb_res_mix=new_comb,
                        y_bf16=y_dummy, y_fp8=None, y_gemm_sf=None,
                        y_routed_sf=None, y_shared_sf=None, shared_sf_block_m=0,
                    )

                return PreparedCall(run=run_both, output=y_fp8, owners=(hc_mod.fn, hc_mod.layer_norm.weight))
            return prepare

        requests.append(op_plan.request(name=f"{name}.m{rows}", prepare_call=make_prepare()))
    if requests:
        declare(session, tuple(requests))
    return plans


def _run_overlapped_mhc(layer, x, prev_lazy_data, hc_modules, *, plan_handle: int,
                        aux_handle: int, y_fp8, y_sf, y_shared_sf):
    from deepinfer.model_executor.models.v4_modules.hc_lazy import HCLazyPostData

    rows, hidden = x.shape
    device = x.device
    residual = prev_lazy_data.residual.squeeze(0)
    post_mix = prev_lazy_data.post_layer_mix.squeeze(0)
    comb_res_mix = prev_lazy_data.comb_res_mix.squeeze(0)
    prev_pre_mix = prev_lazy_data.pre_layer_mix.squeeze(0)
    prev_lazy_data.clear()

    y_bf16 = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
    new_residual = torch.empty((rows, layer.hc_mult, hidden), dtype=torch.bfloat16, device=device)
    new_prev_mix = torch.empty((rows, layer.hc_mult, 1), dtype=torch.float32, device=device)
    new_post_mix = torch.empty((rows, layer.hc_mult, 1), dtype=torch.float32, device=device)
    new_comb_res_mix = torch.empty((rows, layer.hc_mult, layer.hc_mult), dtype=torch.float32, device=device)
    y_bf16_dummy = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)

    _mhc_seam_and_aux(
        x, residual, prev_pre_mix, post_mix, comb_res_mix,
        hc_modules.fn, hc_modules.scale, hc_modules.base,
        layer.hc_mult, layer.hc_norm_eps, layer.hc_pre_eps,
        layer.hc_post_mult_value, layer.hc_sinkhorn_eps, layer.hc_sinkhorn_repeat,
        hc_modules.layer_norm.weight, hc_modules.layer_norm.variance_epsilon,
        hc_modules.layer_norm.out_scale,
        y_bf16, y_fp8, y_sf, y_shared_sf,
        new_residual, new_prev_mix, new_post_mix, new_comb_res_mix, y_bf16_dummy,
        aux_handle, plan_handle,
    )
    lazy_data = HCLazyPostData(
        new_residual.unsqueeze(0),
        new_post_mix.unsqueeze(0),
        new_comb_res_mix.unsqueeze(0),
        new_prev_mix.unsqueeze(0),
    )
    lazy_data._aux_pending = (
        x, residual, prev_pre_mix, post_mix, comb_res_mix,
        new_residual, new_prev_mix, new_post_mix, new_comb_res_mix, y_bf16_dummy,
        aux_handle,
    )
    return y_bf16, lazy_data


@torch.library.custom_op("flare::sm103_dflash_store_aux", mutates_args=("dflash_input",))
def _dflash_store_aux(
    hidden_states: torch.Tensor,
    dflash_input: torch.Tensor,
    index: int,
    aux_stream: int,
) -> None:
    from deepinfer.model_executor.layers.tilelang_ops.dflash import dflash_input_store

    main = torch.cuda.current_stream(hidden_states.device)
    auxiliary = torch.cuda.ExternalStream(aux_stream, device=hidden_states.device)
    auxiliary.wait_stream(main)
    n_embed = hidden_states.shape[-1]
    with torch.cuda.stream(auxiliary):
        dflash_input_store(hidden_states, dflash_input[:, index * n_embed : (index + 1) * n_embed])


@_dflash_store_aux.register_fake
def _dflash_store_aux_fake(hidden_states, dflash_input, index, aux_stream):
    pass


@torch.library.custom_op("flare::sm103_dflash_store_join", mutates_args=("dflash_input",))
def _dflash_store_join(
    anchor: torch.Tensor,
    hidden_states: torch.Tensor,
    dflash_input: torch.Tensor,
    aux_stream: int,
) -> None:
    del hidden_states, dflash_input
    torch.cuda.current_stream(anchor.device).wait_stream(
        torch.cuda.ExternalStream(aux_stream, device=anchor.device))


@_dflash_store_join.register_fake
def _dflash_store_join_fake(anchor, hidden_states, dflash_input, aux_stream):
    pass


class _AuxStoreDFlashInput:
    """Picklable DeepseekDecoderLayer._store_dflash_input wrapper ordered after aux mega_mhc."""

    def __init__(self, layer, aux_handle: int):
        self.layer = layer
        self.aux_handle = aux_handle

    def __call__(self, hidden_states, dflash_input):
        if getattr(self.layer, "_aux_mhc_active", False):
            _dflash_store_aux(hidden_states, dflash_input, int(self.layer.dflash_input_index), self.aux_handle)
            self.layer._aux_dflash_pending = (hidden_states, dflash_input, self.aux_handle)
            return
        type(self.layer)._store_dflash_input(self.layer, hidden_states, dflash_input)


class _OverlappedAttnNorm:
    """Picklable replacement for HCLayerMixin.attn_norm_forward."""

    def __init__(self, layer, attn_plans: dict, aux_handle: int):
        self.layer = layer
        self.attn_plans = attn_plans
        self.aux_handle = aux_handle

    def __call__(self, hidden_states, residual, hc_post_intervention=None):
        layer = self.layer
        layer._aux_mhc_active = False
        layer._aux_dflash_pending = None
        if residual is not None:
            join_pending_mhc(hidden_states, residual)
        if (hc_post_intervention is None and hidden_states.ndim == 2
                and residual is not None and getattr(residual, "post_layer_mix", None) is not None
                and getattr(residual, "pre_layer_mix", None) is not None):
            rows = hidden_states.shape[0]
            entry = self.attn_plans.get(rows)
            if entry is not None:
                op_plan, _ = entry
                consumer = (getattr(layer.self_attn, "q_a_kv_swa_proj", None)
                            or getattr(layer.self_attn, "q_a_kv_proj", None))
                if consumer is not None and consumer.prefers_f8_128x4_scales(rows):
                    h = hidden_states.shape[1]
                    y_fp8 = torch.empty((rows, h), dtype=torch.float8_e4m3fn, device=hidden_states.device)
                    y_sf = torch.empty((128 * (h // 128),), dtype=torch.int32, device=hidden_states.device)
                    y_bf16, lazy_data = _run_overlapped_mhc(
                        layer, hidden_states, residual, layer._attn_hc_modules(),
                        plan_handle=op_plan.handle, aux_handle=self.aux_handle,
                        y_fp8=y_fp8, y_sf=y_sf, y_shared_sf=None,
                    )
                    layer._aux_mhc_active = True
                    return (y_fp8, y_sf.view(torch.uint8)), y_bf16, lazy_data
        return type(layer).attn_norm_forward(layer, hidden_states, residual, hc_post_intervention)


class _OverlappedFFNNorm:
    """Picklable replacement for HCLayerMixin.ffn_norm_forward."""

    def __init__(self, layer, ffn_plans: dict, aux_handle: int):
        self.layer = layer
        self.ffn_plans = ffn_plans
        self.aux_handle = aux_handle

    def __call__(self, hidden_states, residual):
        from deepinfer.model_executor.models.v4_modules.hc import _fused_cast_kwargs

        layer = self.layer
        layer._aux_mhc_active = False
        if residual is not None:
            join_pending_mhc(hidden_states, residual)
        pending_dflash = getattr(layer, "_aux_dflash_pending", None)
        if pending_dflash is not None:
            _dflash_store_join(hidden_states, *pending_dflash)
            layer._aux_dflash_pending = None
        if (hidden_states.ndim == 2 and residual is not None
                and getattr(residual, "post_layer_mix", None) is not None
                and getattr(residual, "pre_layer_mix", None) is not None):
            rows = hidden_states.shape[0]
            entry = self.ffn_plans.get(rows)
            if entry is not None:
                op_plan, expected_sbm = entry
                cast_kwargs = _fused_cast_kwargs(layer.mlp, rows)
                if cast_kwargs and int(cast_kwargs.get("extra_sf_block_m", 0)) == expected_sbm:
                    y_fp8, y_routed_sf = cast_kwargs["out"]
                    y_shared_sf = cast_kwargs.get("extra_sf_out")
                    y_bf16, lazy_data = _run_overlapped_mhc(
                        layer, hidden_states, residual, layer._ffn_hc_modules(),
                        plan_handle=op_plan.handle, aux_handle=self.aux_handle,
                        y_fp8=y_fp8, y_sf=y_routed_sf, y_shared_sf=y_shared_sf,
                    )
                    return (y_fp8, y_routed_sf), y_bf16, lazy_data
        return type(layer).ffn_norm_forward(layer, hidden_states, residual)


class _JoinAfterForward:
    """Picklable forward wrapper that drains any pending auxiliary mHC update."""

    def __init__(self, layer, *, is_draft: bool = False):
        self.layer = layer
        self.is_draft = is_draft

    def rebind(self, layer):
        return _JoinAfterForward(layer, is_draft=self.is_draft)

    def __call__(self, *args, **kwargs):
        fn = type(self.layer).forward_drafts if self.is_draft else type(self.layer).forward
        out, res = fn(self.layer, *args, **kwargs)
        if res is not None:
            join_pending_mhc(out, res)
        return out, res


def bind_mhc_overlap_layer(
    layer,
    rows_list,
    *,
    aux_stream: torch.cuda.Stream,
    session,
    name: str,
    allow_attn_overlap: bool,
    allow_ffn_overlap: bool,
    join_after_forward: bool = False,
    is_draft: bool = False,
):
    """Wrap shifted-mHC norm seams on one decoder layer to overlap mega_mhc on ``aux_stream``."""
    if not (getattr(layer, "hc_enabled", False) and getattr(layer, "hc_shifted", False)
            and getattr(layer, "use_mega_mhc", False) and layer.hc_mult == 4):
        return
    rows_small = tuple(sorted({r for r in rows_list if 1 <= r <= MHC_SEAM_MAX_ROWS}))
    if not rows_small:
        return
    hidden_size = layer.attn_layernorm.weight.numel()
    aux_handle = aux_stream.cuda_stream

    attn_plans = {}
    attn_consumer = getattr(layer.self_attn, "q_a_kv_swa_proj", None) or getattr(layer.self_attn, "q_a_kv_proj", None)
    if (allow_attn_overlap and attn_consumer is not None
            and hasattr(attn_consumer, "prefers_f8_128x4_scales")
            and layer.attn_layernorm.use_fp8):
        attn_plans = _declare_mhc_seam_plans(
            layer, rows_small, hidden_size=hidden_size, mode=0,
            shared_block_m_by_rows={}, session=session, name=f"{name}.attn_seam",
        )

    ffn_plans = {}
    if allow_ffn_overlap and hasattr(layer.mlp, "fused_norm_cast_kwargs"):
        shared_bms = {}
        eligible_ffn = []
        for rows in rows_small:
            ops = getattr(layer.mlp, "operations", None)
            if ops is not None and rows in ops:
                _, _, _, _, shared_sf, block_m = ops[rows].encoded_inputs()
                shared_bms[rows] = int(block_m) if shared_sf is not None else 0
                eligible_ffn.append(rows)
        if eligible_ffn:
            ffn_plans = _declare_mhc_seam_plans(
                layer, eligible_ffn, hidden_size=hidden_size, mode=1,
                shared_block_m_by_rows=shared_bms, session=session, name=f"{name}.ffn_seam",
            )

    layer.attn_norm_forward = _OverlappedAttnNorm(layer, attn_plans, aux_handle)
    layer.ffn_norm_forward = _OverlappedFFNNorm(layer, ffn_plans, aux_handle)
    if attn_plans and getattr(layer, "dflash_input_index", None) is not None:
        layer._store_dflash_input = _AuxStoreDFlashInput(layer, aux_handle)

    if join_after_forward:
        if is_draft:
            layer.forward_drafts = _JoinAfterForward(layer, is_draft=True)
        else:
            layer.forward = _JoinAfterForward(layer, is_draft=False)


@torch.library.custom_op("flare::sm103_swa_kv_store_aux", mutates_args=("swa_cache",))
def _swa_kv_store_aux(
    kv_swa: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_eps: float,
    rope_dim: int,
    is_neox_style: bool,
    rope_offset: int,
    num_per_channels: int,
    aux_stream: int,
) -> None:
    from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (
        CastArgs, KVCacheOutArgs, NormArgs, RopeArgs,
        fused_norm_rope_hadamard_cast_store_kvcache,
    )

    main = torch.cuda.current_stream(kv_swa.device)
    auxiliary = torch.cuda.ExternalStream(aux_stream, device=kv_swa.device)
    auxiliary.wait_stream(main)
    with torch.cuda.stream(auxiliary):
        fused_norm_rope_hadamard_cast_store_kvcache(
            kv_swa,
            norm_args=NormArgs(norm_weight, norm_eps),
            rope_args=RopeArgs(positions, cos_sin_cache, rope_dim, is_neox_style, offset=rope_offset),
            cast_args=CastArgs("e4m3", num_per_channels, round_sf=True, sf_format="ue8m0"),
            out_args=KVCacheOutArgs(swa_cache, slot_mapping),
        )


@_swa_kv_store_aux.register_fake
def _swa_kv_store_aux_fake(
    kv_swa, norm_weight, positions, cos_sin_cache, swa_cache, slot_mapping,
    norm_eps, rope_dim, is_neox_style, rope_offset, num_per_channels, aux_stream,
):
    pass


@torch.library.custom_op("flare::sm103_swa_kv_join", mutates_args=("swa_cache",))
def _swa_kv_join(
    anchor: torch.Tensor,
    kv_swa: torch.Tensor,
    swa_cache: torch.Tensor,
    aux_stream: int,
) -> None:
    del kv_swa, swa_cache
    torch.cuda.current_stream(anchor.device).wait_stream(
        torch.cuda.ExternalStream(aux_stream, device=anchor.device))


@_swa_kv_join.register_fake
def _swa_kv_join_fake(anchor, kv_swa, swa_cache, aux_stream):
    pass


def join_pending_swa_kv(anchor: torch.Tensor, input_metadata) -> None:
    """Drain any auxiliary SWA KV cache write attached to ``input_metadata``."""
    pending = getattr(input_metadata, "_aux_swa_pending", None)
    if pending is not None:
        _swa_kv_join(anchor, *pending)
        input_metadata._aux_swa_pending = None


class _OverlappedPrepareSwaKv:
    """Picklable decode wrapper overlapping SWA KV cache writes behind q_a_norm and q_b_proj."""

    def __init__(self, attn, rows: frozenset[int], aux_stream_handle: int):
        self.attn = attn
        self.rows = rows
        self.aux_stream_handle = aux_stream_handle

    def __call__(self, kv_swa, swa_cache, input_metadata, use_cp):
        attn = self.attn
        if (kv_swa is not None and swa_cache is not None
                and not input_metadata.is_prompt and not use_cp
                and kv_swa.shape[0] in self.rows):
            _swa_kv_store_aux(
                kv_swa, attn.kv_swa_norm.weight, input_metadata.positions,
                attn.rotary_emb.cos_sin_cache, swa_cache, input_metadata.slot_mapping_store_swa,
                float(attn.kv_swa_norm.variance_epsilon), int(attn.qk_rope_head_dim),
                bool(attn.rotary_emb.is_neox_style), int(attn.qk_head_dim - attn.qk_rope_head_dim),
                int(attn.swa_cache_num_per_channels), self.aux_stream_handle,
            )
            input_metadata._aux_swa_pending = (kv_swa, swa_cache, self.aux_stream_handle)
            return None, None
        return type(attn)._prepare_swa_kv(attn, kv_swa, swa_cache, input_metadata, use_cp)


class _JoinSwaDecodeAttention:
    """Picklable decode-attention wrapper that drains a pending auxiliary SWA KV write."""

    def __init__(self, attn, inner=None):
        self.attn = attn
        self.inner = inner

    def __call__(self, q_b, kv_ca_index, kv_swa_index, kv_cache, input_metadata):
        join_pending_swa_kv(q_b, input_metadata)
        if self.inner is not None:
            return self.inner(q_b, kv_ca_index, kv_swa_index, kv_cache, input_metadata)
        return type(self.attn)._decode_attention(self.attn, q_b, kv_ca_index, kv_swa_index, kv_cache, input_metadata)


class _OverlappedDFlashForwardDrafts:
    """Picklable DFlashAttention.forward_drafts wrapper overlapping draft KV store behind q_b_proj."""

    def __init__(self, attn, rows: frozenset[int], aux_stream_handle: int):
        self.attn = attn
        self.rows = rows
        self.aux_stream_handle = aux_stream_handle

    def __call__(self, draft_hiddens, kv_cache, input_metadata):
        from deepinfer.model_executor.layers.attention_backend import decode_mla_mqa_sparse

        attn = self.attn
        rows = draft_hiddens[0].shape[0] if isinstance(draft_hiddens, tuple) else draft_hiddens.shape[0]
        if rows not in self.rows or kv_cache.swa_cache is None:
            return type(attn).forward_drafts(attn, draft_hiddens, kv_cache, input_metadata)
        draft_positions = input_metadata.positions
        q_a, kv_a = attn.q_a_kv_proj(draft_hiddens).split([attn.q_lora_rank, attn.qk_head_dim], dim=-1)
        _swa_kv_store_aux(
            kv_a, attn.kv_norm.weight, draft_positions,
            attn.rotary_emb.cos_sin_cache, kv_cache.swa_cache,
            input_metadata.dflash_newx_slot_mapping_store,
            float(attn.kv_norm.variance_epsilon), int(attn.qk_rope_head_dim),
            bool(attn.rotary_emb.is_neox_style), int(attn.qk_head_dim - attn.qk_rope_head_dim),
            int(attn.swa_cache_num_per_channels), self.aux_stream_handle,
        )
        q = attn.q_b_proj(attn.q_a_norm(q_a))
        q = q.unflatten(-1, (attn.num_heads, attn.qk_head_dim))
        q = attn._q_b_norm_rope(q, draft_positions)
        _swa_kv_join(q, kv_a, kv_cache.swa_cache, self.aux_stream_handle)
        fused_kwargs = attn._fused_norm_rope_cast_kwargs(draft_positions)
        context, _ = decode_mla_mqa_sparse(
            q,
            kv_cache.swa_cache,
            attn.qk_head_dim,
            input_metadata.dflash_kv_index,
            input_metadata.flash_mla_sched_meta_dict["DFlash"],
            softmax_scale=attn.softmax_scale,
            attn_sink=attn.attn_sink_params,
            batch_invariant=attn.execution.batch_invariant,
            key_cache_extra=None,
            indices_in_kvcache_extra=None,
            index_length_extra=None,
            fused_norm_rope_cast_kwargs=fused_kwargs,
        )
        return attn._wv_and_o_proj(context, draft_positions)


def bind_swa_kv_overlap(attn, rows_list, *, aux_stream: torch.cuda.Stream, is_draft: bool = False):
    """Overlap decode SWA KV cache writes on ``aux_stream`` behind ``q_a_norm`` and ``q_b_proj``."""
    rows = frozenset(int(r) for r in rows_list if 1 <= int(r) <= MHC_SEAM_MAX_ROWS)
    if not rows:
        return
    aux_handle = aux_stream.cuda_stream
    if is_draft:
        if hasattr(attn, "forward_drafts") and hasattr(attn, "kv_norm"):
            attn.forward_drafts = _OverlappedDFlashForwardDrafts(attn, rows, aux_handle)
        return
    if getattr(attn, "use_swa", False) and hasattr(attn, "_prepare_swa_kv") and hasattr(attn, "kv_swa_norm"):
        attn._prepare_swa_kv = _OverlappedPrepareSwaKv(attn, rows, aux_handle)
        prev_decode = attn.__dict__.get("_decode_attention")
        attn._decode_attention = _JoinSwaDecodeAttention(attn, prev_decode)


@torch.library.custom_op(
    "flare::sm103_indexer_kv_prep_aux",
    mutates_args=("index_weights_out", "pre_rope_kv_out", "ca_cache", "ca_state", "ica_cache"),
)
def _indexer_kv_prep_aux(
    hidden_states_bf16: torch.Tensor,
    shared_hidden_bf16: torch.Tensor | None,
    cur_weight: torch.Tensor,
    shared_weight: torch.Tensor | None,
    kv_norm_weight: torch.Tensor | None,
    k_proj_weight: torch.Tensor | None,
    k_norm_weight: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    positions_c: torch.Tensor | None,
    slot_mapping_store_ca: torch.Tensor | None,
    state_block_idx: torch.Tensor | None,
    valid_token_mask: torch.Tensor | None,
    context_lens: torch.Tensor | None,
    index_weights_out: torch.Tensor,
    pre_rope_kv_out: torch.Tensor | None,
    ca_cache: torch.Tensor | None,
    ca_state: torch.Tensor | None,
    ica_cache: torch.Tensor | None,
    kv_dim: int,
    index_weights_dim: int,
    cur_fp32_output: bool,
    shared_fp32_output: bool,
    kv_compress_ratio: int,
    verification_width: int,
    kv_norm_eps: float,
    k_norm_eps: float,
    rotary_dim: int,
    is_neox_style: bool,
    is_kv_source: bool,
    store_index_k: bool,
    aux_stream: int,
) -> None:
    import deep_gemm
    from deepinfer.model_executor.layers.tilelang_ops.cache_common import (
        FP4_INDEX_KV_FORMAT, get_kv_cache_format,
    )
    from deepinfer.model_executor.layers.tilelang_ops.compress import compress_and_update_state_decode
    from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (
        CastArgs, KVCacheOutArgs, NormArgs, RopeArgs,
        fused_norm_rope_hadamard_cast_store_kvcache,
    )

    device = hidden_states_bf16.device
    main = torch.cuda.current_stream(device)
    auxiliary = torch.cuda.ExternalStream(aux_stream, device=device)
    auxiliary.wait_stream(main)
    with torch.cuda.stream(auxiliary):
        rows = hidden_states_bf16.shape[0]
        cur_dtype = torch.float32 if cur_fp32_output else torch.bfloat16
        cur_n = cur_weight.shape[0]
        align = 32 // cur_dtype.itemsize
        phys_n = (cur_n + align - 1) // align * align
        cur_storage = torch.empty((rows, phys_n), dtype=cur_dtype, device=device)
        cur_out = cur_storage[:, :cur_n]
        deep_gemm.cublaslt_gemm_nt(hidden_states_bf16, cur_weight, cur_out)
        if kv_dim > 0:
            projected_kv = cur_out[:, :kv_dim]
            index_weights_out.copy_(cur_out[:, kv_dim : kv_dim + index_weights_dim].bfloat16())
        else:
            projected_kv = None
            index_weights_out.copy_(cur_out[:, :index_weights_dim].bfloat16())

        if shared_weight is not None and shared_hidden_bf16 is not None:
            sh_dtype = torch.float32 if shared_fp32_output else torch.bfloat16
            sh_n = shared_weight.shape[0]
            sh_align = 32 // sh_dtype.itemsize
            sh_phys = (sh_n + sh_align - 1) // sh_align * sh_align
            sh_storage = torch.empty((rows, sh_phys), dtype=sh_dtype, device=device)
            projected_kv = sh_storage[:, :sh_n]
            deep_gemm.cublaslt_gemm_nt(shared_hidden_bf16, shared_weight, projected_kv)

        if is_kv_source and projected_kv is not None and ca_cache is not None and pre_rope_kv_out is not None:
            if kv_compress_ratio == 1:
                kv_compressed = projected_kv.unsqueeze(1)
            else:
                score, latent = projected_kv.chunk(2, dim=-1)
                state_ids = state_block_idx
                if valid_token_mask is not None:
                    state_ids = torch.where(valid_token_mask, state_ids, -1)
                kv_compressed = compress_and_update_state_decode(
                    score, latent, ca_state, state_ids, context_lens,
                    max_seqlen_q=verification_width, compress_ratio=kv_compress_ratio,
                ).unsqueeze(1)
            cache_format = get_kv_cache_format(ca_cache)
            fused_norm_rope_hadamard_cast_store_kvcache(
                kv_compressed,
                norm_args=NormArgs(kv_norm_weight, kv_norm_eps),
                rope_args=RopeArgs(
                    positions_c, cos_sin_cache, rotary_dim, is_neox_style,
                    offset=kv_compressed.shape[-1] - rotary_dim,
                ),
                cast_args=CastArgs.from_cache_format(cache_format),
                out_args=KVCacheOutArgs(
                    ca_cache, slot_mapping_store_ca,
                    out=pre_rope_kv_out.unsqueeze(1), out_stage="norm", layout=cache_format,
                ),
            )

        if store_index_k and pre_rope_kv_out is not None and ica_cache is not None:
            k_n = k_proj_weight.shape[0]
            k_align = 32 // torch.bfloat16.itemsize
            k_phys = (k_n + k_align - 1) // k_align * k_align
            k_storage = torch.empty((pre_rope_kv_out.shape[0], k_phys), dtype=torch.bfloat16, device=device)
            index_k = k_storage[:, :k_n]
            deep_gemm.cublaslt_gemm_nt(pre_rope_kv_out, k_proj_weight, index_k)
            fused_norm_rope_hadamard_cast_store_kvcache(
                index_k,
                norm_args=NormArgs(k_norm_weight, k_norm_eps),
                rope_args=RopeArgs(
                    positions_c, cos_sin_cache, rotary_dim, is_neox_style,
                    offset=index_k.shape[-1] - rotary_dim,
                ),
                cast_args=CastArgs.from_cache_format(FP4_INDEX_KV_FORMAT),
                out_args=KVCacheOutArgs(ica_cache, slot_mapping_store_ca, layout=FP4_INDEX_KV_FORMAT),
            )


@_indexer_kv_prep_aux.register_fake
def _indexer_kv_prep_aux_fake(
    hidden_states_bf16, shared_hidden_bf16, cur_weight, shared_weight, kv_norm_weight,
    k_proj_weight, k_norm_weight, cos_sin_cache, positions_c, slot_mapping_store_ca,
    state_block_idx, valid_token_mask, context_lens, index_weights_out, pre_rope_kv_out,
    ca_cache, ca_state, ica_cache, kv_dim, index_weights_dim, cur_fp32_output,
    shared_fp32_output, kv_compress_ratio, verification_width, kv_norm_eps, k_norm_eps,
    rotary_dim, is_neox_style, is_kv_source, store_index_k, aux_stream,
):
    pass


@torch.library.custom_op(
    "flare::sm103_indexer_kv_prep_join",
    mutates_args=("index_weights_out", "pre_rope_kv_out", "ca_cache", "ca_state", "ica_cache"),
)
def _indexer_kv_prep_join(
    anchor: torch.Tensor,
    hidden_states_bf16: torch.Tensor,
    shared_hidden_bf16: torch.Tensor | None,
    positions_c: torch.Tensor | None,
    slot_mapping_store_ca: torch.Tensor | None,
    state_block_idx: torch.Tensor | None,
    valid_token_mask: torch.Tensor | None,
    context_lens: torch.Tensor | None,
    index_weights_out: torch.Tensor,
    pre_rope_kv_out: torch.Tensor | None,
    ca_cache: torch.Tensor | None,
    ca_state: torch.Tensor | None,
    ica_cache: torch.Tensor | None,
    aux_stream: int,
) -> None:
    del hidden_states_bf16, shared_hidden_bf16, positions_c, slot_mapping_store_ca
    del state_block_idx, valid_token_mask, context_lens
    del index_weights_out, pre_rope_kv_out, ca_cache, ca_state, ica_cache
    torch.cuda.current_stream(anchor.device).wait_stream(
        torch.cuda.ExternalStream(aux_stream, device=anchor.device))


@_indexer_kv_prep_join.register_fake
def _indexer_kv_prep_join_fake(
    anchor, hidden_states_bf16, shared_hidden_bf16, positions_c, slot_mapping_store_ca,
    state_block_idx, valid_token_mask, context_lens, index_weights_out, pre_rope_kv_out,
    ca_cache, ca_state, ica_cache, aux_stream,
):
    pass


class _OverlappedIndexerForward:
    """Picklable V4Attention.forward wrapper overlapping indexer KV prep behind Q/SWA projections."""

    def __init__(self, attn, rows: frozenset[int], aux_kv_handle: int, kv_dim: int, index_weights_dim: int):
        self.attn = attn
        self.rows = rows
        self.aux_kv_handle = aux_kv_handle
        self.kv_dim = kv_dim
        self.index_weights_dim = index_weights_dim

    def __call__(self, hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime):
        from deepinfer.model_executor.utils import Movable

        attn = self.attn
        hidden_states = hidden_states.move() if isinstance(hidden_states, Movable) else hidden_states
        hidden_states_bf16 = hidden_states_bf16.move() if isinstance(hidden_states_bf16, Movable) else hidden_states_bf16
        rows = hidden_states[0].shape[0] if isinstance(hidden_states, tuple) else hidden_states.shape[0]
        if (input_metadata.is_prompt
                or cross_layer_runtime.consumer_full_metadata is not None
                or rows not in self.rows):
            return type(attn).forward(attn, Movable(hidden_states), kv_cache, input_metadata,
                                      Movable(hidden_states_bf16), cross_layer_runtime)
        shared_hidden_bf16 = cross_layer_runtime.shared_hidden
        block_size = kv_cache.ca_cache.shape[1] * attn.kv_compress_ratio if kv_cache.ca_cache is not None else None
        input_metadata.compute_attn_info(
            attn.max_model_len, attn.window_size, block_size,
            swa_block_size=None if kv_cache.swa_cache is None else kv_cache.swa_cache.shape[1],
        )
        shared_hidden_bf16 = attn._resolve_shared_hidden(hidden_states_bf16, shared_hidden_bf16)

        device = hidden_states_bf16.device
        index_weights_out = torch.empty((rows, self.index_weights_dim), dtype=torch.bfloat16, device=device)
        ratio_key = attn.ratio_key
        needs_ca = attn.is_kv_source or attn.is_index_k_source
        positions_c = input_metadata.positions_c[ratio_key] if needs_ca else None
        slot_mapping_store_ca = input_metadata.slot_mapping_store_ca[ratio_key] if needs_ca else None
        if attn.is_kv_source:
            pre_rope_kv_out = torch.empty((positions_c.shape[0], attn.qk_head_dim), dtype=torch.bfloat16, device=device)
        elif attn.is_index_k_source:
            pre_rope_kv_out = cross_layer_runtime.pre_rope_kv
        else:
            pre_rope_kv_out = None

        store_index_k = bool(attn.index_attn.store_index_k)
        k_proj_weight = attn.index_attn.index_k_module.linear.weight if store_index_k else None
        k_norm_weight = attn.index_attn.index_k_module.norm.weight if store_index_k else None
        k_norm_eps = float(attn.index_attn.index_k_module.norm.variance_epsilon) if store_index_k else 0.0
        kv_norm_weight = attn.PackedWkv.norm.weight if attn.is_kv_source else None
        kv_norm_eps = float(attn.PackedWkv.norm.variance_epsilon) if attn.is_kv_source else 0.0
        shared_weight = attn.shared_hidden_kv_proj.weight if attn.shared_hidden_kv_proj is not None else None
        shared_fp32 = bool(attn.shared_hidden_kv_proj.use_fp32_output) if attn.shared_hidden_kv_proj is not None else False
        state_block_idx = input_metadata.state_block_idx if (attn.is_kv_source and attn.kv_compress_ratio > 1) else None
        valid_token_mask = input_metadata.valid_token_mask if (attn.is_kv_source and attn.kv_compress_ratio > 1) else None
        context_lens = input_metadata.context_lens if (attn.is_kv_source and attn.kv_compress_ratio > 1) else None
        ca_cache = kv_cache.ca_cache if attn.is_kv_source else None
        ca_state = kv_cache.ca_state if attn.is_kv_source else None
        ica_cache = kv_cache.ica_cache if store_index_k else None

        _indexer_kv_prep_aux(
            hidden_states_bf16, shared_hidden_bf16,
            attn.current_hidden_proj.weight, shared_weight,
            kv_norm_weight, k_proj_weight, k_norm_weight,
            attn.rotary_emb.cos_sin_cache, positions_c, slot_mapping_store_ca,
            state_block_idx, valid_token_mask, context_lens,
            index_weights_out, pre_rope_kv_out, ca_cache, ca_state, ica_cache,
            self.kv_dim, self.index_weights_dim,
            bool(attn.current_hidden_proj.use_fp32_output), shared_fp32,
            int(attn.kv_compress_ratio), int(input_metadata.verification_width),
            kv_norm_eps, k_norm_eps,
            int(attn.rotary_emb.rotary_dim), bool(attn.rotary_emb.is_neox_style),
            bool(attn.is_kv_source), store_index_k,
            self.aux_kv_handle,
        )

        q_a, kv_swa = attn._project_q_and_swa(hidden_states, shared_hidden_bf16)
        del hidden_states
        attn._prepare_swa_kv(kv_swa, kv_cache.swa_cache, input_metadata, False)
        q_b, index_q_b = attn._project_q_b(q_a, input_metadata.positions)
        del q_a

        _indexer_kv_prep_join(
            index_q_b, hidden_states_bf16, shared_hidden_bf16,
            positions_c, slot_mapping_store_ca,
            state_block_idx, valid_token_mask, context_lens,
            index_weights_out, pre_rope_kv_out, ca_cache, ca_state, ica_cache,
            self.aux_kv_handle,
        )
        del hidden_states_bf16, shared_hidden_bf16
        if attn.is_kv_source:
            cross_layer_runtime.pre_rope_kv = pre_rope_kv_out

        kv_ca_index = attn.index_attn(
            index_weights_out,
            index_q_b,
            None,
            kv_cache=kv_cache.ica_cache,
            state_cache=kv_cache.ica_state,
            input_metadata=input_metadata,
            store_kv=False,
            cross_layer_runtime=cross_layer_runtime,
        )
        cross_layer_runtime.topk_indices = kv_ca_index
        del index_weights_out, index_q_b

        kv_swa_index = input_metadata.kv_swa_index if attn.use_swa else None
        context = attn._decode_attention(q_b, kv_ca_index, kv_swa_index, kv_cache, input_metadata)
        del q_b, kv_ca_index
        return attn._wv_and_o_proj(context, input_metadata.positions)


def bind_indexer_kv_overlap(attn, rows_list, *, aux_kv_stream: torch.cuda.Stream):
    """Overlap decode indexer/compressed KV preparation on ``aux_kv_stream`` behind Q/SWA projections."""
    if not (getattr(attn, "has_local_index", False) and getattr(attn, "has_packed_kv", False)
            and getattr(attn, "current_hidden_proj", None) is not None
            and getattr(attn, "index_attn", None) is not None):
        return
    if attn.current_hidden_proj.weight.dtype != torch.bfloat16:
        return
    if attn.shared_hidden_kv_proj is not None and attn.shared_hidden_kv_proj.weight.dtype != torch.bfloat16:
        return
    if attn.index_attn.store_index_k and attn.index_attn.index_k_module.linear.weight.dtype != torch.bfloat16:
        return
    layout = tuple(attn.current_hidden_proj_layout)
    if len(layout) == 2 and layout[0][0] == "kv" and layout[1][0] == "index_weights":
        kv_dim, idx_dim = int(layout[0][1]), int(layout[1][1])
    elif len(layout) == 1 and layout[0][0] == "index_weights":
        kv_dim, idx_dim = 0, int(layout[0][1])
    else:
        return
    rows = frozenset(int(r) for r in rows_list if 1 <= int(r) <= MHC_SEAM_MAX_ROWS)
    if not rows:
        return
    attn.forward = _OverlappedIndexerForward(attn, rows, aux_kv_stream.cuda_stream, kv_dim, idx_dim)
