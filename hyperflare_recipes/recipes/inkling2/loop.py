# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Zero-materialization cross-pass message routing and fused loop message gates.

At the start of each repeated loop pass, the previous pass supplies two residual
blocks `(s0, s1)`. Normalizing them once yields BF16 `(n0, n1)` and FP32 `n1 - n0`.
For every core layer `l` and stream `s in {attn, k, v, mlp}`, the two-source
attention-residual weight is algebraically:

    p1[t, l, s] = sigmoid(dot(n1[t] - n0[t], q[l, s]))
    msg[t, l, s] = (1 - p1) * n0[t] + p1 * n1[t]

All `4 * core_layers` scalar probabilities compute in one small matrix multiply
and remain `[T, 4 * core_layers]` in memory. No `[T, H]` message tensor is
materialized to DRAM: the fused gate kernels reconstruct `msg` in registers from
`(n0, n1, p1)`.

For decode/verify (`T <= 256`), `forget_gain` uses BF16 `F.linear` followed by a
fused mix+RMSNorm Triton kernel (and a single fused Q/K/V launch on sliding loop
layers). For prefill (`T > 256`), a 2D Tensor-Core tiled Triton kernel fuses
`Z @ [ForgetT, GainT]`, `tanh`, cross-pass mixing, and sum-of-squares accumulation
without materializing `[T, 2H]` forget/gain tensors to DRAM.

Controller semantics follow Inkling2MessageController in vLLM PR #69028 at
b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0:
https://github.com/vllm-project/vllm/blob/b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0/vllm/models/inkling2/nvidia/loop.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ignition.preparation.examples import declaration_tensor
from ignition.preparation.rebuild_manifest import weight_container
from ignition.preparation.triton import launch_triton, prepare_triton

STREAMS = ("attn", "k", "v", "mlp")
_DECODE_FUSE_MAX_TOKENS = 16
_PREFILL_FUSE_MIN_TOKENS = 257


@triton.jit
def _prepare_cross_pass_sources(
    S0, S1, N0, N1, Diff,
    HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    """RMS-normalize the two previous-pass residual blocks once; emit BF16 `(n0, n1)` and FP32 `n1 - n0`."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < HIDDEN
    s0 = tl.load(S0 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    n0 = s0 * tl.rsqrt(tl.sum(s0 * s0, 0) / HIDDEN + EPS)
    s1 = tl.load(S1 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    n1 = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / HIDDEN + EPS)
    tl.store(N0 + row * HIDDEN + cols, n0.to(tl.bfloat16), valid)
    tl.store(N1 + row * HIDDEN + cols, n1.to(tl.bfloat16), valid)
    tl.store(Diff + row * HIDDEN + cols, n1 - n0, valid)


@triton.jit
def _norm_cat_from_cross_pass(
    Hidden, N0, N1, Probs, Out, stream_idx,
    NUM_STREAMS: tl.constexpr,
    HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    """Reconstruct `msg = (1-p1)*n0 + p1*n1` in registers, normalize both inputs, and store BF16 `[2H]`."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < HIDDEN
    h = tl.load(Hidden + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    h_inv = tl.rsqrt(tl.sum(h * h, 0) / HIDDEN + EPS)
    tl.store(Out + row * (2 * HIDDEN) + cols, (h * h_inv).to(tl.bfloat16), valid)

    p1 = tl.load(Probs + row * NUM_STREAMS + stream_idx).to(tl.float32)
    n0 = tl.load(N0 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    n1 = tl.load(N1 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    m = (1.0 - p1) * n0 + p1 * n1
    m_inv = tl.rsqrt(tl.sum(m * m, 0) / HIDDEN + EPS)
    tl.store(Out + row * (2 * HIDDEN) + HIDDEN + cols, (m * m_inv).to(tl.bfloat16), valid)


@triton.jit
def _norm_cat_qkv_shared_hidden(
    Hidden, N0, N1, Probs, OutQ, OutK, OutV,
    BASE_STREAM: tl.constexpr, NUM_STREAMS: tl.constexpr,
    HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    """Fused Q/K/V norm-cat for sliding loop layers where Q, K, and V share one `Hidden` tensor."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < HIDDEN
    h = tl.load(Hidden + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    nh = (h * tl.rsqrt(tl.sum(h * h, 0) / HIDDEN + EPS)).to(tl.bfloat16)
    n0 = tl.load(N0 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    n1 = tl.load(N1 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    base = row * (2 * HIDDEN)
    prob_base = row * NUM_STREAMS + BASE_STREAM

    pq = tl.load(Probs + prob_base).to(tl.float32)
    mq = (1.0 - pq) * n0 + pq * n1
    nmq = (mq * tl.rsqrt(tl.sum(mq * mq, 0) / HIDDEN + EPS)).to(tl.bfloat16)
    tl.store(OutQ + base + cols, nh, valid)
    tl.store(OutQ + base + HIDDEN + cols, nmq, valid)

    pk = tl.load(Probs + prob_base + 1).to(tl.float32)
    mk = (1.0 - pk) * n0 + pk * n1
    nmk = (mk * tl.rsqrt(tl.sum(mk * mk, 0) / HIDDEN + EPS)).to(tl.bfloat16)
    tl.store(OutK + base + cols, nh, valid)
    tl.store(OutK + base + HIDDEN + cols, nmk, valid)

    pv = tl.load(Probs + prob_base + 2).to(tl.float32)
    mv = (1.0 - pv) * n0 + pv * n1
    nmv = (mv * tl.rsqrt(tl.sum(mv * mv, 0) / HIDDEN + EPS)).to(tl.bfloat16)
    tl.store(OutV + base + cols, nh, valid)
    tl.store(OutV + base + HIDDEN + cols, nmv, valid)


@triton.jit
def _gate_mix_rms_norm_from_cross_pass(
    Hidden, N0, N1, Probs, ForgetGain, NormWeight, Out, stream_idx,
    NUM_STREAMS: tl.constexpr,
    HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    """Apply `(1 - tanh(f)) * hidden + tanh(g) * msg` and weighted RMSNorm directly to BF16."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < HIDDEN
    h = tl.load(Hidden + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    p1 = tl.load(Probs + row * NUM_STREAMS + stream_idx).to(tl.float32)
    n0 = tl.load(N0 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    n1 = tl.load(N1 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    m = (1.0 - p1) * n0 + p1 * n1

    fg_base = ForgetGain + row * (2 * HIDDEN)
    forget = tl.extra.cuda.libdevice.tanh(tl.load(fg_base + cols, valid, other=0.0).to(tl.float32))
    gain = tl.extra.cuda.libdevice.tanh(tl.load(fg_base + HIDDEN + cols, valid, other=0.0).to(tl.float32))
    mixed = (1.0 - forget) * h + gain * m
    inv = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    gamma = tl.load(NormWeight + cols, valid, other=0.0).to(tl.float32)
    tl.store(Out + row * HIDDEN + cols, (mixed * inv * gamma).to(tl.bfloat16), valid)


@triton.jit
def _gate_mix_rms_norm_qkv_shared(
    Hidden, N0, N1, Probs, FG_Q, FG_K, FG_V, NormWeight, OutQ, OutK, OutV,
    BASE_STREAM: tl.constexpr, NUM_STREAMS: tl.constexpr,
    HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    """Fused Q/K/V gate-mix+RMSNorm reading `Hidden`, `N0`, `N1`, and `NormWeight` once."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < HIDDEN
    h = tl.load(Hidden + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    n0 = tl.load(N0 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    n1 = tl.load(N1 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    gamma = tl.load(NormWeight + cols, valid, other=0.0).to(tl.float32)
    pbase = row * NUM_STREAMS + BASE_STREAM
    fg_off = row * (2 * HIDDEN)
    out_off = row * HIDDEN

    pq = tl.load(Probs + pbase).to(tl.float32)
    mq = (1.0 - pq) * n0 + pq * n1
    fq = tl.extra.cuda.libdevice.tanh(tl.load(FG_Q + fg_off + cols, valid, other=0.0).to(tl.float32))
    gq = tl.extra.cuda.libdevice.tanh(tl.load(FG_Q + fg_off + HIDDEN + cols, valid, other=0.0).to(tl.float32))
    xq = (1.0 - fq) * h + gq * mq
    inv_q = tl.rsqrt(tl.sum(xq * xq, 0) / HIDDEN + EPS)
    tl.store(OutQ + out_off + cols, (xq * inv_q * gamma).to(tl.bfloat16), valid)

    pk = tl.load(Probs + pbase + 1).to(tl.float32)
    mk = (1.0 - pk) * n0 + pk * n1
    fk = tl.extra.cuda.libdevice.tanh(tl.load(FG_K + fg_off + cols, valid, other=0.0).to(tl.float32))
    gk = tl.extra.cuda.libdevice.tanh(tl.load(FG_K + fg_off + HIDDEN + cols, valid, other=0.0).to(tl.float32))
    xk = (1.0 - fk) * h + gk * mk
    inv_k = tl.rsqrt(tl.sum(xk * xk, 0) / HIDDEN + EPS)
    tl.store(OutK + out_off + cols, (xk * inv_k * gamma).to(tl.bfloat16), valid)

    pv = tl.load(Probs + pbase + 2).to(tl.float32)
    mv = (1.0 - pv) * n0 + pv * n1
    fv = tl.extra.cuda.libdevice.tanh(tl.load(FG_V + fg_off + cols, valid, other=0.0).to(tl.float32))
    gv = tl.extra.cuda.libdevice.tanh(tl.load(FG_V + fg_off + HIDDEN + cols, valid, other=0.0).to(tl.float32))
    xv = (1.0 - fv) * h + gv * mv
    inv_v = tl.rsqrt(tl.sum(xv * xv, 0) / HIDDEN + EPS)
    tl.store(OutV + out_off + cols, (xv * inv_v * gamma).to(tl.bfloat16), valid)


@triton.jit
def _fused_fg_mix_to_buf(
    Hidden, N0, N1, Probs, Z, ForgetT, GainT, MixedOut, RowSumSq,
    STREAM_IDX: tl.constexpr, NUM_STREAMS: tl.constexpr,
    T: tl.constexpr, HIDDEN: tl.constexpr, RANK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """2D Tensor-Core tiled fusion of `Z @ [ForgetT, GainT]`, tanh gating, cross-pass mix, and sum-of-squares."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < T
    cmask = cols < HIDDEN
    mask = rmask[:, None] & cmask[None, :]

    r_offs = tl.arange(0, RANK)
    z = tl.load(Z + rows[:, None] * RANK + r_offs[None, :], mask=rmask[:, None], other=0.0)
    wf = tl.load(ForgetT + r_offs[:, None] * HIDDEN + cols[None, :], mask=cmask[None, :], other=0.0)
    wg = tl.load(GainT + r_offs[:, None] * HIDDEN + cols[None, :], mask=cmask[None, :], other=0.0)
    f = tl.extra.cuda.libdevice.tanh(tl.dot(z, wf, out_dtype=tl.float32))
    g = tl.extra.cuda.libdevice.tanh(tl.dot(z, wg, out_dtype=tl.float32))

    p1 = tl.load(Probs + rows * NUM_STREAMS + STREAM_IDX, mask=rmask, other=0.0).to(tl.float32)[:, None]
    h = tl.load(Hidden + rows[:, None] * HIDDEN + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    s0 = tl.load(N0 + rows[:, None] * HIDDEN + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(N1 + rows[:, None] * HIDDEN + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    m = (1.0 - p1) * s0 + p1 * s1
    mixed = (1.0 - f) * h + g * m
    tl.store(MixedOut + rows[:, None] * HIDDEN + cols[None, :], mixed.to(tl.bfloat16), mask=mask)
    tl.atomic_add(RowSumSq + rows, tl.sum(mixed * mixed, axis=1), mask=rmask)


@triton.jit
def _scale_rmsnorm(
    Mixed, RowSumSq, NormW, Out,
    HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < HIDDEN
    x = tl.load(Mixed + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    ss = tl.load(RowSumSq + row).to(tl.float32)
    inv = tl.rsqrt(ss / HIDDEN + EPS)
    gamma = tl.load(NormW + cols, valid, other=0.0).to(tl.float32)
    tl.store(Out + row * HIDDEN + cols, (x * inv * gamma).to(tl.bfloat16), valid)


@triton.jit
def _norm_cat_qkv_packed(
    Hidden, N0, N1, Probs, Out3, base_stream,
    NUM_STREAMS: tl.constexpr,
    T: tl.constexpr, HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    """Fused Q/K/V norm-cat writing directly into contiguous `[3, T, 2H]` for batched decode gates."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < HIDDEN
    h = tl.load(Hidden + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    nh = (h * tl.rsqrt(tl.sum(h * h, 0) / HIDDEN + EPS)).to(tl.bfloat16)
    n0 = tl.load(N0 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    n1 = tl.load(N1 + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    stride_s = T * (2 * HIDDEN)
    base = row * (2 * HIDDEN)
    prob_base = row * NUM_STREAMS + base_stream

    pq = tl.load(Probs + prob_base).to(tl.float32)
    mq = (1.0 - pq) * n0 + pq * n1
    nmq = (mq * tl.rsqrt(tl.sum(mq * mq, 0) / HIDDEN + EPS)).to(tl.bfloat16)
    tl.store(Out3 + base + cols, nh, valid)
    tl.store(Out3 + base + HIDDEN + cols, nmq, valid)

    pk = tl.load(Probs + prob_base + 1).to(tl.float32)
    mk = (1.0 - pk) * n0 + pk * n1
    nmk = (mk * tl.rsqrt(tl.sum(mk * mk, 0) / HIDDEN + EPS)).to(tl.bfloat16)
    tl.store(Out3 + stride_s + base + cols, nh, valid)
    tl.store(Out3 + stride_s + base + HIDDEN + cols, nmk, valid)

    pv = tl.load(Probs + prob_base + 2).to(tl.float32)
    mv = (1.0 - pv) * n0 + pv * n1
    nmv = (mv * tl.rsqrt(tl.sum(mv * mv, 0) / HIDDEN + EPS)).to(tl.bfloat16)
    tl.store(Out3 + 2 * stride_s + base + cols, nh, valid)
    tl.store(Out3 + 2 * stride_s + base + HIDDEN + cols, nmv, valid)


@triton.jit
def _down_decode_single(
    Cat, Down, Z,
    K: tl.constexpr, R: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    pid_r = tl.program_id(1)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    cat_base = Cat + t * K
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(cat_base + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(Down + r_offs[:, None] * K + k_offs[None, :],
                    mask=r_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(Z + t * R + r_offs, acc.to(tl.bfloat16), mask=r_mask)


@triton.jit
def _down_decode_qkv(
    Cat3, DownQ, DownK, DownV, Z3,
    T: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid0 = tl.program_id(0)
    s = pid0 // T
    t = pid0 % T
    pid_r = tl.program_id(1)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    cat_base = Cat3 + (s * T + t).to(tl.int64) * K
    w_ptr = DownQ if s == 0 else (DownK if s == 1 else DownV)
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(cat_base + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + r_offs[:, None] * K + k_offs[None, :],
                    mask=r_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(Z3 + (s * T + t).to(tl.int64) * R + r_offs, acc.to(tl.bfloat16), mask=r_mask)


@triton.jit
def _fg_mix_decode_single(
    Hidden, N0, N1, Probs, Z, ForgetT, GainT, MixedOut, stream_idx,
    NUM_STREAMS: tl.constexpr,
    HIDDEN: tl.constexpr, RANK: tl.constexpr, BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < HIDDEN
    r_offs = tl.arange(0, RANK)
    z = tl.load(Z + t * RANK + r_offs).to(tl.float32)
    wf = tl.load(ForgetT + r_offs[:, None] * HIDDEN + cols[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
    wg = tl.load(GainT + r_offs[:, None] * HIDDEN + cols[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
    f = tl.extra.cuda.libdevice.tanh(tl.sum(z[:, None] * wf, axis=0))
    g = tl.extra.cuda.libdevice.tanh(tl.sum(z[:, None] * wg, axis=0))
    p1 = tl.load(Probs + t * NUM_STREAMS + stream_idx).to(tl.float32)
    h = tl.load(Hidden + t * HIDDEN + cols, mask=cmask, other=0.0).to(tl.float32)
    s0 = tl.load(N0 + t * HIDDEN + cols, mask=cmask, other=0.0).to(tl.float32)
    s1 = tl.load(N1 + t * HIDDEN + cols, mask=cmask, other=0.0).to(tl.float32)
    mixed = (1.0 - f) * h + g * ((1.0 - p1) * s0 + p1 * s1)
    tl.store(MixedOut + t * HIDDEN + cols, mixed.to(tl.bfloat16), mask=cmask)


@triton.jit
def _fg_mix_decode_qkv(
    Hidden, N0, N1, Probs, Z3, FQ, FK, FV, GQ, GK, GV, Mixed3, base_stream,
    NUM_STREAMS: tl.constexpr,
    T: tl.constexpr, HIDDEN: tl.constexpr, RANK: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid0 = tl.program_id(0)
    s = pid0 // T
    t = pid0 % T
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < HIDDEN
    r_offs = tl.arange(0, RANK)
    z = tl.load(Z3 + (s * T + t).to(tl.int64) * RANK + r_offs).to(tl.float32)
    wf_ptr = FQ if s == 0 else (FK if s == 1 else FV)
    wg_ptr = GQ if s == 0 else (GK if s == 1 else GV)
    wf = tl.load(wf_ptr + r_offs[:, None] * HIDDEN + cols[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
    wg = tl.load(wg_ptr + r_offs[:, None] * HIDDEN + cols[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
    f = tl.extra.cuda.libdevice.tanh(tl.sum(z[:, None] * wf, axis=0))
    g = tl.extra.cuda.libdevice.tanh(tl.sum(z[:, None] * wg, axis=0))
    p1 = tl.load(Probs + t.to(tl.int64) * NUM_STREAMS + base_stream + s).to(tl.float32)
    h = tl.load(Hidden + t.to(tl.int64) * HIDDEN + cols, mask=cmask, other=0.0).to(tl.float32)
    s0 = tl.load(N0 + t.to(tl.int64) * HIDDEN + cols, mask=cmask, other=0.0).to(tl.float32)
    s1 = tl.load(N1 + t.to(tl.int64) * HIDDEN + cols, mask=cmask, other=0.0).to(tl.float32)
    mixed = (1.0 - f) * h + g * ((1.0 - p1) * s0 + p1 * s1)
    tl.store(Mixed3 + (s * T + t).to(tl.int64) * HIDDEN + cols, mixed.to(tl.bfloat16), mask=cmask)


@triton.jit
def _rmsnorm_rows_det(
    Mixed, NormW, Out,
    HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < HIDDEN
    x = tl.load(Mixed + row * HIDDEN + cols, valid, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / HIDDEN + EPS)
    gamma = tl.load(NormW + cols, valid, other=0.0).to(tl.float32)
    tl.store(Out + row * HIDDEN + cols, (x * inv * gamma).to(tl.bfloat16), valid)


@weight_container(("down", "forget_gain", "forget_t", "gain_t"))
@dataclass(frozen=True)
class GateWeights:
    """Packed BF16 controller weights: `down [R, 2H]`, `forget_gain [2H, R]`, and transposed `[R, H]`."""

    down: torch.Tensor
    forget_gain: torch.Tensor
    forget_t: torch.Tensor
    gain_t: torch.Tensor


@weight_container(("q", "k", "v", "mlp"))
@dataclass(frozen=True)
class LayerControllers:
    q: GateWeights
    k: GateWeights
    v: GateWeights
    mlp: GateWeights


class PreparedCrossPassRouter:
    """Normalizes the two previous-pass blocks once and computes all `4 * core_layers` scalar weights."""

    def __init__(self, queries_all: torch.Tensor | None = None, *, tokens: int, hidden: int, eps: float,
                 queries_all_t: torch.Tensor | None = None, examples=None):
        if queries_all_t is not None:
            if queries_all_t.ndim != 2 or queries_all_t.dtype != torch.float32 or queries_all_t.shape[0] != hidden:
                raise ValueError("transposed cross-pass queries must be FP32 [hidden, 4 * core_layers]")
            self.queries_all_t = queries_all_t
            self.num_streams = queries_all_t.shape[1]
            device = queries_all_t.device
        else:
            if queries_all is None or queries_all.ndim != 2 or queries_all.dtype != torch.float32 or queries_all.shape[1] != hidden:
                raise ValueError("cross-pass queries must be FP32 [4 * core_layers, hidden]")
            self.queries_all_t = queries_all.t().contiguous()
            self.num_streams = queries_all.shape[0]
            device = queries_all.device
        self.tokens, self.hidden, self.eps = tokens, hidden, eps
        declarations = {} if examples is None else examples

        def example(role, shape, dtype=torch.float32):
            return declaration_tensor(declarations, "inkling.cross_pass." + role, shape, dtype, device)

        s0 = example("s0", (tokens, hidden), torch.bfloat16)
        s1 = example("s1", (tokens, hidden), torch.bfloat16)
        n0 = example("n0_bf16", (tokens, hidden), torch.bfloat16)
        n1 = example("n1_bf16", (tokens, hidden), torch.bfloat16)
        diff = example("diff", (tokens, hidden), torch.float32)
        block = triton.next_power_of_2(hidden)
        self._prep_opts = dict(HIDDEN=hidden, EPS=eps, BLOCK=block, num_warps=8, enable_fp_fusion=False)
        self._prep_compiled = prepare_triton(
            _prepare_cross_pass_sources, (s0, s1, n0, n1, diff), (tokens, 1, 1), **self._prep_opts
        ).compiled

    def __call__(self, s0: torch.Tensor, s1: torch.Tensor):
        device = s0.device
        n0 = torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
        n1 = torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
        diff = torch.empty((self.tokens, self.hidden), dtype=torch.float32, device=device)
        launch_triton(
            _prepare_cross_pass_sources, self._prep_compiled,
            (s0, s1, n0, n1, diff), (self.tokens, 1, 1), **self._prep_opts,
        )
        probs = torch.sigmoid(diff @ self.queries_all_t)
        return n0, n1, probs


class PreparedMessageGate:
    """Fused cross-pass message gate and downstream weighted RMSNorm."""

    def __init__(self, weights: GateWeights, norm_weight: torch.Tensor, *,
                 stream_idx: int, num_streams: int, tokens: int, hidden: int, eps: float,
                 examples=None):
        device = weights.down.device
        rank = weights.down.shape[0]
        if weights.down.shape != (rank, 2 * hidden) or weights.forget_gain.shape != (2 * hidden, rank):
            raise ValueError("loop message gate weights do not match hidden size and gate rank")
        if norm_weight.shape != (hidden,) or norm_weight.dtype != torch.bfloat16:
            raise ValueError("loop message gate requires a BF16 downstream RMSNorm weight")
        self.tokens, self.hidden, self.rank, self.eps = tokens, hidden, rank, eps
        self.stream_idx, self.num_streams = stream_idx, num_streams
        self.down_w = weights.down
        self.fg_w = weights.forget_gain
        self.forget_t, self.gain_t = weights.forget_t, weights.gain_t
        self.norm_weight = norm_weight
        self.use_tiled_fg = tokens >= _PREFILL_FUSE_MIN_TOKENS
        # Single-stream cuBLAS hits a small-M cliff on SM120 at T in [2..7] (24.8 us vs 14.5 us),
        # while at T=1 and T>=8 cuBLAS beats the 3-launch single-stream Triton path (9.2-12.5 us vs 14.5 us).
        self.use_decode_triton = 2 <= tokens <= 7 and rank > 0 and (rank & (rank - 1)) == 0
        if self.use_tiled_fg and (rank <= 0 or (rank & (rank - 1)) != 0):
            raise ValueError(f"fused prefill message gate requires a power-of-two rank, got {rank}")
        declarations = {} if examples is None else examples

        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(declarations, "inkling.gate." + role, shape, dtype, device)

        fp32_h = example("fp32_h", (tokens, hidden), torch.float32)
        bf16_n0 = example("bf16_n0", (tokens, hidden), torch.bfloat16)
        bf16_n1 = example("bf16_n1", (tokens, hidden), torch.bfloat16)
        probs = example("probs", (tokens, num_streams), torch.float32)
        cat_bf16 = example("cat_bf16", (tokens, 2 * hidden), torch.bfloat16)
        z_bf16 = example("z_bf16", (tokens, rank), torch.bfloat16)
        out_bf16 = example("out_bf16", (tokens, hidden), torch.bfloat16)

        block = triton.next_power_of_2(hidden)
        self._norm_cat_opts = dict(
            NUM_STREAMS=num_streams,
            HIDDEN=hidden, EPS=eps, BLOCK=block, num_warps=8, enable_fp_fusion=False,
        )
        self._norm_cat_compiled = prepare_triton(
            _norm_cat_from_cross_pass, (fp32_h, bf16_n0, bf16_n1, probs, cat_bf16, stream_idx),
            (tokens, 1, 1), **self._norm_cat_opts,
        ).compiled

        self._down_decode_compiled = self._fg_decode_compiled = self._norm_decode_compiled = None
        if self.use_tiled_fg:
            mixed_bf16 = example("mixed_bf16", (tokens, hidden), torch.bfloat16)
            row_ss = example("row_ss", (tokens,), torch.float32)
            bm, bn = 64, 128
            self._tiled_grid = (triton.cdiv(tokens, bm), triton.cdiv(hidden, bn), 1)
            self._tiled_opts = dict(
                STREAM_IDX=stream_idx, NUM_STREAMS=num_streams,
                T=tokens, HIDDEN=hidden, RANK=rank,
                BLOCK_M=bm, BLOCK_N=bn, num_warps=4, num_stages=2,
            )
            self._tiled_compiled = prepare_triton(
                _fused_fg_mix_to_buf,
                (fp32_h, bf16_n0, bf16_n1, probs, z_bf16, self.forget_t, self.gain_t, mixed_bf16, row_ss),
                self._tiled_grid, **self._tiled_opts,
            ).compiled
            self._scale_opts = dict(HIDDEN=hidden, EPS=eps, BLOCK=block, num_warps=8, enable_fp_fusion=False)
            self._scale_compiled = prepare_triton(
                _scale_rmsnorm, (mixed_bf16, row_ss, norm_weight, out_bf16),
                (tokens, 1, 1), **self._scale_opts,
            ).compiled
            self._mix_norm_compiled = None
        elif self.use_decode_triton:
            mixed_bf16 = example("mixed_bf16", (tokens, hidden), torch.bfloat16)
            self._down_decode_grid = (tokens, triton.cdiv(rank, 16), 1)
            self._down_decode_opts = dict(K=2 * hidden, R=rank, BLOCK_K=256, BLOCK_R=16, num_warps=4)
            self._down_decode_compiled = prepare_triton(
                _down_decode_single, (cat_bf16, self.down_w, z_bf16),
                self._down_decode_grid, **self._down_decode_opts,
            ).compiled
            self._fg_decode_grid = (tokens, triton.cdiv(hidden, 128), 1)
            self._fg_decode_opts = dict(
                NUM_STREAMS=num_streams, HIDDEN=hidden, RANK=rank, BLOCK_N=128, num_warps=4,
            )
            self._fg_decode_compiled = prepare_triton(
                _fg_mix_decode_single,
                (fp32_h, bf16_n0, bf16_n1, probs, z_bf16, self.forget_t, self.gain_t, mixed_bf16, stream_idx),
                self._fg_decode_grid, **self._fg_decode_opts,
            ).compiled
            self._norm_decode_opts = dict(HIDDEN=hidden, EPS=eps, BLOCK=block, num_warps=8, enable_fp_fusion=False)
            self._norm_decode_compiled = prepare_triton(
                _rmsnorm_rows_det, (mixed_bf16, norm_weight, out_bf16),
                (tokens, 1, 1), **self._norm_decode_opts,
            ).compiled
            self._mix_norm_compiled = self._tiled_compiled = self._scale_compiled = None
        else:
            fg_bf16 = example("fg_bf16", (tokens, 2 * hidden), torch.bfloat16)
            self._mix_norm_opts = dict(
                NUM_STREAMS=num_streams,
                HIDDEN=hidden, EPS=eps, BLOCK=block, num_warps=8, enable_fp_fusion=False,
            )
            self._mix_norm_compiled = prepare_triton(
                _gate_mix_rms_norm_from_cross_pass,
                (fp32_h, bf16_n0, bf16_n1, probs, fg_bf16, norm_weight, out_bf16, stream_idx),
                (tokens, 1, 1), **self._mix_norm_opts,
            ).compiled
            self._tiled_compiled = None
            self._scale_compiled = None

    def compute_fg(self, cat_bf16: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(cat_bf16, self.down_w), self.fg_w)

    def apply_with_cat(self, hidden: torch.Tensor, cat_bf16: torch.Tensor,
                       n0: torch.Tensor, n1: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        device = hidden.device
        out = torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
        if self.use_decode_triton:
            z = torch.empty((self.tokens, self.rank), dtype=torch.bfloat16, device=device)
            launch_triton(
                _down_decode_single, self._down_decode_compiled,
                (cat_bf16, self.down_w, z), self._down_decode_grid, **self._down_decode_opts,
            )
            mixed = torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
            launch_triton(
                _fg_mix_decode_single, self._fg_decode_compiled,
                (hidden, n0, n1, probs, z, self.forget_t, self.gain_t, mixed, self.stream_idx),
                self._fg_decode_grid, **self._fg_decode_opts,
            )
            launch_triton(
                _rmsnorm_rows_det, self._norm_decode_compiled,
                (mixed, self.norm_weight, out), (self.tokens, 1, 1), **self._norm_decode_opts,
            )
            return out
        z = F.linear(cat_bf16, self.down_w)
        if self.use_tiled_fg:
            mixed = torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
            row_ss = torch.zeros((self.tokens,), dtype=torch.float32, device=device)
            launch_triton(
                _fused_fg_mix_to_buf, self._tiled_compiled,
                (hidden, n0, n1, probs, z, self.forget_t, self.gain_t, mixed, row_ss),
                self._tiled_grid, **self._tiled_opts,
            )
            launch_triton(
                _scale_rmsnorm, self._scale_compiled,
                (mixed, row_ss, self.norm_weight, out),
                (self.tokens, 1, 1), **self._scale_opts,
            )
            return out
        fg = F.linear(z, self.fg_w)
        launch_triton(
            _gate_mix_rms_norm_from_cross_pass, self._mix_norm_compiled,
            (hidden, n0, n1, probs, fg, self.norm_weight, out, self.stream_idx),
            (self.tokens, 1, 1), **self._mix_norm_opts,
        )
        return out

    def __call__(self, hidden: torch.Tensor, n0: torch.Tensor, n1: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        cat_bf16 = torch.empty((self.tokens, 2 * self.hidden), dtype=torch.bfloat16, device=hidden.device)
        launch_triton(
            _norm_cat_from_cross_pass, self._norm_cat_compiled,
            (hidden, n0, n1, probs, cat_bf16, self.stream_idx), (self.tokens, 1, 1), **self._norm_cat_opts,
        )
        return self.apply_with_cat(hidden, cat_bf16, n0, n1, probs)


class PreparedSharedQKVNormCat:
    """Computes Q, K, and V `[nh, nm]` inputs and fused decode gate-mix in minimal Triton launches."""

    def __init__(self, norm_weight: torch.Tensor, *, base_stream: int, num_streams: int,
                 tokens: int, hidden: int, eps: float, device, examples=None,
                 gates: tuple[GateWeights, GateWeights, GateWeights] | None = None):
        self.tokens, self.hidden = tokens, hidden
        self.base_stream = base_stream
        self.norm_weight = norm_weight
        self.use_fused_mix = tokens < _PREFILL_FUSE_MIN_TOKENS
        rank = gates[0].down.shape[0] if gates is not None else 0
        self.rank = rank
        self.use_fused_decode = (
            gates is not None and tokens <= _DECODE_FUSE_MAX_TOKENS
            and rank > 0 and (rank & (rank - 1)) == 0
        )
        self._qkv_gates = gates
        declarations = {} if examples is None else examples

        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(declarations, "inkling.qkv_cat." + role, shape, dtype, device)

        fp32_h = example("fp32_h", (tokens, hidden), torch.float32)
        bf16_n0 = example("bf16_n0", (tokens, hidden), torch.bfloat16)
        bf16_n1 = example("bf16_n1", (tokens, hidden), torch.bfloat16)
        probs = example("probs", (tokens, num_streams), torch.float32)
        block = triton.next_power_of_2(hidden)
        self._opts = dict(
            BASE_STREAM=base_stream, NUM_STREAMS=num_streams,
            HIDDEN=hidden, EPS=eps, BLOCK=block, num_warps=8, enable_fp_fusion=False,
        )
        self._compiled = self._mix_compiled = None
        if not self.use_fused_decode:
            out_q = example("out_q", (tokens, 2 * hidden), torch.bfloat16)
            out_k = example("out_k", (tokens, 2 * hidden), torch.bfloat16)
            out_v = example("out_v", (tokens, 2 * hidden), torch.bfloat16)
            self._compiled = prepare_triton(
                _norm_cat_qkv_shared_hidden, (fp32_h, bf16_n0, bf16_n1, probs, out_q, out_k, out_v),
                (tokens, 1, 1), **self._opts,
            ).compiled
            if self.use_fused_mix:
                fg_q = example("fg_q", (tokens, 2 * hidden), torch.bfloat16)
                fg_k = example("fg_k", (tokens, 2 * hidden), torch.bfloat16)
                fg_v = example("fg_v", (tokens, 2 * hidden), torch.bfloat16)
                oq = example("mix_oq", (tokens, hidden), torch.bfloat16)
                ok = example("mix_ok", (tokens, hidden), torch.bfloat16)
                ov = example("mix_ov", (tokens, hidden), torch.bfloat16)
                self._mix_compiled = prepare_triton(
                    _gate_mix_rms_norm_qkv_shared,
                    (fp32_h, bf16_n0, bf16_n1, probs, fg_q, fg_k, fg_v, norm_weight, oq, ok, ov),
                    (tokens, 1, 1), **self._opts,
                ).compiled

        self._packed_cat_compiled = self._qkv_down_compiled = None
        self._qkv_fg_compiled = self._qkv_norm_compiled = None
        if self.use_fused_decode:
            cat3 = example("cat3", (3, tokens, 2 * hidden), torch.bfloat16)
            z3 = example("z3", (3, tokens, rank), torch.bfloat16)
            m3 = example("m3", (3, tokens, hidden), torch.bfloat16)
            o3 = example("o3", (3, tokens, hidden), torch.bfloat16)
            gq, gk, gv = gates
            self._packed_cat_opts = dict(
                NUM_STREAMS=num_streams,
                T=tokens, HIDDEN=hidden, EPS=eps, BLOCK=block, num_warps=8, enable_fp_fusion=False,
            )
            self._packed_cat_compiled = prepare_triton(
                _norm_cat_qkv_packed, (fp32_h, bf16_n0, bf16_n1, probs, cat3, base_stream),
                (tokens, 1, 1), **self._packed_cat_opts,
            ).compiled
            self._qkv_down_grid = (3 * tokens, triton.cdiv(rank, 16), 1)
            self._qkv_down_opts = dict(T=tokens, K=2 * hidden, R=rank, BLOCK_K=256, BLOCK_R=16, num_warps=4)
            self._qkv_down_compiled = prepare_triton(
                _down_decode_qkv, (cat3, gq.down, gk.down, gv.down, z3),
                self._qkv_down_grid, **self._qkv_down_opts,
            ).compiled
            self._qkv_fg_grid = (3 * tokens, triton.cdiv(hidden, 128), 1)
            self._qkv_fg_opts = dict(
                NUM_STREAMS=num_streams,
                T=tokens, HIDDEN=hidden, RANK=rank, BLOCK_N=128, num_warps=4,
            )
            self._qkv_fg_compiled = prepare_triton(
                _fg_mix_decode_qkv,
                (fp32_h, bf16_n0, bf16_n1, probs, z3,
                 gq.forget_t, gk.forget_t, gv.forget_t, gq.gain_t, gk.gain_t, gv.gain_t, m3, base_stream),
                self._qkv_fg_grid, **self._qkv_fg_opts,
            ).compiled
            self._qkv_norm_opts = dict(HIDDEN=hidden, EPS=eps, BLOCK=block, num_warps=8, enable_fp_fusion=False)
            self._qkv_norm_compiled = prepare_triton(
                _rmsnorm_rows_det, (m3, norm_weight, o3),
                (3 * tokens, 1, 1), **self._qkv_norm_opts,
            ).compiled

    def apply_fused_decode(self, hidden: torch.Tensor, n0: torch.Tensor, n1: torch.Tensor, probs: torch.Tensor):
        device = hidden.device
        gq, gk, gv = self._qkv_gates
        cat3 = torch.empty((3, self.tokens, 2 * self.hidden), dtype=torch.bfloat16, device=device)
        launch_triton(
            _norm_cat_qkv_packed, self._packed_cat_compiled,
            (hidden, n0, n1, probs, cat3, self.base_stream), (self.tokens, 1, 1), **self._packed_cat_opts,
        )
        z3 = torch.empty((3, self.tokens, self.rank), dtype=torch.bfloat16, device=device)
        launch_triton(
            _down_decode_qkv, self._qkv_down_compiled,
            (cat3, gq.down, gk.down, gv.down, z3), self._qkv_down_grid, **self._qkv_down_opts,
        )
        m3 = torch.empty((3, self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
        launch_triton(
            _fg_mix_decode_qkv, self._qkv_fg_compiled,
            (hidden, n0, n1, probs, z3,
             gq.forget_t, gk.forget_t, gv.forget_t, gq.gain_t, gk.gain_t, gv.gain_t, m3, self.base_stream),
            self._qkv_fg_grid, **self._qkv_fg_opts,
        )
        o3 = torch.empty((3, self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
        launch_triton(
            _rmsnorm_rows_det, self._qkv_norm_compiled,
            (m3, self.norm_weight, o3), (3 * self.tokens, 1, 1), **self._qkv_norm_opts,
        )
        return o3[0], o3[1], o3[2]

    def __call__(self, hidden: torch.Tensor, n0: torch.Tensor, n1: torch.Tensor, probs: torch.Tensor):
        device = hidden.device
        out_q = torch.empty((self.tokens, 2 * self.hidden), dtype=torch.bfloat16, device=device)
        out_k = torch.empty((self.tokens, 2 * self.hidden), dtype=torch.bfloat16, device=device)
        out_v = torch.empty((self.tokens, 2 * self.hidden), dtype=torch.bfloat16, device=device)
        launch_triton(
            _norm_cat_qkv_shared_hidden, self._compiled,
            (hidden, n0, n1, probs, out_q, out_k, out_v), (self.tokens, 1, 1), **self._opts,
        )
        return out_q, out_k, out_v

    def mix_qkv(self, hidden: torch.Tensor, n0: torch.Tensor, n1: torch.Tensor, probs: torch.Tensor,
                fg_q: torch.Tensor, fg_k: torch.Tensor, fg_v: torch.Tensor):
        device = hidden.device
        oq = torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
        ok = torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
        ov = torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16, device=device)
        launch_triton(
            _gate_mix_rms_norm_qkv_shared, self._mix_compiled,
            (hidden, n0, n1, probs, fg_q, fg_k, fg_v, self.norm_weight, oq, ok, ov),
            (self.tokens, 1, 1), **self._opts,
        )
        return oq, ok, ov
