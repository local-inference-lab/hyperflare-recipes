# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hybrid HCA C128 + DSA 656B/FlashIndexer attention with VHA PostMix and gated output for ERNIE 5.5."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from b12x.attention import sparse_mla
from b12x.attention.dsa_indexer._glm_pool_kernels import _write_pool

from hyperflare_recipes.components.attention_output import PreparedGroupedOutputProjection
from hyperflare_recipes.components.compressed_mla import (
    PreparedCompressedCacheWriter,
    PreparedCompressedMLA,
)
from hyperflare_recipes.components.fp8_indexer import PreparedFP8Indexer
from hyperflare_recipes.components.gated_compression import PreparedGatedCompressor
from hyperflare_recipes.components.linear import (
    BatchedUnquantizedProjection,
    QuantizedProjection,
    UnquantizedProjection,
)
from hyperflare_recipes.components.projection_overlap import ConcurrentProjectionChain
from hyperflare_recipes.components.rotary import CompressedRotary as Rotary, PreparedRotaryAngles
from hyperflare_recipes.components.sparse_mla import PreparedSparseMLA
from hyperflare_recipes.components.window_indices import PreparedWindowIndices
from ignition.components.collectives import ConcurrentAllReduce, PreparedAllReduce
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer

from .config import Compressor
from .pool import DSA_SUBPAGES_PER_PAGE, PAGE_TOKENS, latent_page_bytes

# Bound FlashIndexer prefill workspace while amortizing launches across 4096-token chunks.
INDEX_ROWS = 4096
FLASHINDEXER_TOPK_HEADS = 8
DSA_TOPK = 2048
DSA_WINDOW = 128
DSA_TOTAL_WIDTH = DSA_TOPK + DSA_WINDOW  # 2176
VHA_RANK = 16


# ---------------------------------------------------------------------------
# Fused Triton Kernels (HCA window transfer, norms, RoPE, VHA, DSA, FlashIndexer)
# ---------------------------------------------------------------------------

@triton.jit
def _transfer_window(
    Source,
    Destination,
    StateRows,
    Starts,
    Counts,
    SOURCE_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
    STAGED_TOKENS: tl.constexpr,
    COMMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, item = tl.program_id(0), tl.program_id(1)
    row = tl.load(StateRows + request)
    offset = tl.arange(0, BLOCK)
    if COMMIT:
        length = tl.load(Counts + request)
        token = tl.maximum(0, length - 128) + item
        live = (row > 0) & (token < length)
        source_row, source_token = request, 128 + token
        dest_row, dest_token = row, (tl.load(Starts + request) + token) % 128
        source_extent: tl.constexpr = STAGED_TOKENS
        dest_extent: tl.constexpr = 128
    else:
        live = row > 0
        source_row, source_token = row, item
        dest_row, dest_token = request, item
        source_extent: tl.constexpr = 128
        dest_extent: tl.constexpr = STAGED_TOKENS
    source_offset = tl.where(
        offset < 576, source_token * 576 + offset, source_extent * 576 + source_token * 8 + offset - 576
    )
    dest_offset = tl.where(
        offset < 576, dest_token * 576 + offset, dest_extent * 576 + dest_token * 8 + offset - 576
    )
    value = tl.load(Source + source_row * SOURCE_STRIDE + source_offset, live & (offset < 584), other=0)
    tl.store(Destination + dest_row * DEST_STRIDE + dest_offset, value, live & (offset < 584))


def transfer_window(source, destination, metadata, staged_tokens, *, counts=None):
    torch.library.wrap_triton(_transfer_window)[(metadata.starts.numel(), 128)](
        source,
        destination,
        metadata.state_rows,
        metadata.starts,
        metadata.lengths if counts is None else counts,
        source.stride(0),
        destination.stride(0),
        staged_tokens,
        counts is not None,
        1024,
    )


@triton.jit
def _hca_q_kv_norm_rope_kernel(X, QW, KW, ANGLES, Q_OUT, KV_OUT, EPS: tl.constexpr):
    """Fused HCA Q RMSNorm (1024) + KV RMSNorm (512) + KV interleaved RoPE (trailing 64)."""
    row = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    if task == 0:
        qc = tl.arange(0, 1024)
        qx = tl.load(X + row * 1536 + qc).to(tl.float32)
        qw = tl.load(QW + qc).to(tl.float32)
        qinv = tl.rsqrt(tl.sum(qx * qx, 0) * (1.0 / 1024.0) + EPS)
        tl.store(Q_OUT + row * 1024 + qc, (qx * qinv * qw).to(tl.bfloat16))
    else:
        kc = tl.arange(0, 512)
        kx = tl.load(X + row * 1536 + 1024 + kc).to(tl.float32)
        kw = tl.load(KW + kc).to(tl.float32)
        kinv = tl.rsqrt(tl.sum(kx * kx, 0) * (1.0 / 512.0) + EPS)
        tl.store(KV_OUT + row * 512 + kc, (kx * kinv * kw).to(tl.bfloat16), mask=kc < 448)
        pair = tl.arange(0, 32)
        xa = tl.load(X + row * 1536 + 1024 + 448 + 2 * pair).to(tl.float32)
        xb = tl.load(X + row * 1536 + 1024 + 448 + 2 * pair + 1).to(tl.float32)
        wa = tl.load(KW + 448 + 2 * pair).to(tl.float32)
        wb = tl.load(KW + 448 + 2 * pair + 1).to(tl.float32)
        a = (xa * kinv * wa).to(tl.bfloat16).to(tl.float32)
        b = (xb * kinv * wb).to(tl.bfloat16).to(tl.float32)
        cos = tl.load(ANGLES + row * 64 + pair)
        sin = tl.load(ANGLES + row * 64 + 32 + pair)
        tl.store(KV_OUT + row * 512 + 448 + 2 * pair, (a * cos - b * sin).to(tl.bfloat16))
        tl.store(KV_OUT + row * 512 + 448 + 2 * pair + 1, (a * sin + b * cos).to(tl.bfloat16))


def _hca_q_kv_norm_rope(projected, qw, kw, angles, eps):
    rows = projected.shape[0]
    query_low = torch.empty((rows, 1024), dtype=torch.bfloat16, device=projected.device)
    window_values = torch.empty((rows, 512), dtype=torch.bfloat16, device=projected.device)
    torch.library.wrap_triton(_hca_q_kv_norm_rope_kernel)[(rows, 2)](
        projected, qw, kw, angles, query_low, window_values, eps, num_warps=4
    )
    return query_low, window_values


@triton.jit
def _hca_normalize_rotate_kernel(Query, Angles, ROW_STRIDE, HEAD_STRIDE, EPS, WIDTH: tl.constexpr):
    """Parameter-free per-head RMSNorm (BF16-step match to DeepSeek V4 HCA) + trailing-64 interleaved RoPE."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    base = Query + row * ROW_STRIDE + head * HEAD_STRIDE
    offsets = tl.arange(0, WIDTH)
    value = tl.load(base + offsets).to(tl.float32)
    mean = tl.sum(value * value, axis=0) * (1.0 / WIDTH)
    scale = tl.rsqrt(mean + EPS)
    tl.store(base + offsets, (value * scale).to(tl.bfloat16), mask=offsets < WIDTH - 64)
    pair = tl.arange(0, 32)
    cos = tl.load(Angles + row * 64 + pair)
    sin = tl.load(Angles + row * 64 + 32 + pair)
    tail = base + WIDTH - 64 + 2 * pair
    a = (tl.load(tail).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    b = (tl.load(tail + 1).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    tl.store(tail, (a * cos - b * sin).to(tl.bfloat16))
    tl.store(tail + 1, (a * sin + b * cos).to(tl.bfloat16))


def _hca_normalize_rotate(query, angles, eps):
    torch.library.wrap_triton(_hca_normalize_rotate_kernel)[(query.shape[0], query.shape[1])](
        query, angles, query.stride(0), query.stride(1), eps, query.shape[2], num_warps=4
    )


@triton.jit
def _vha_project_z_kernel(
    O_ptr,
    U_ptr,
    Z_ptr,
    o_stride_t,
    o_stride_h,
    u_stride_h,
    z_stride_t,
    z_stride_r,
    LOCAL_HEADS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute z = U^T @ o: [T, LOCAL_HEADS, D] -> [T, RANK, D] in FP32 accumulation."""
    tok = tl.program_id(0).to(tl.int64)
    d_block = tl.program_id(1)
    heads = tl.arange(0, LOCAL_HEADS)
    ranks = tl.arange(0, RANK)
    cols = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    # u_t: [RANK, LOCAL_HEADS] BF16
    u_t = tl.load(U_ptr + heads[None, :] * u_stride_h + ranks[:, None])
    # o_tile: [LOCAL_HEADS, BLOCK_D] BF16
    o_tile = tl.load(O_ptr + tok * o_stride_t + heads[:, None] * o_stride_h + cols[None, :])
    z = tl.dot(u_t, o_tile, out_dtype=tl.float32)
    tl.store(
        Z_ptr + tok * z_stride_t + ranks[:, None] * z_stride_r + cols[None, :],
        z.to(Z_ptr.dtype.element_ty),
    )


def _vha_project_z(o, u, *, out_dtype=torch.float32):
    tokens, local_heads, head_dim = o.shape
    rank = u.shape[1]
    z = torch.empty((tokens, rank, head_dim), dtype=out_dtype, device=o.device)
    block_d = 128
    torch.library.wrap_triton(_vha_project_z_kernel)[(tokens, head_dim // block_d)](
        o,
        u,
        z,
        o.stride(0),
        o.stride(1),
        u.stride(0),
        z.stride(0),
        z.stride(1),
        LOCAL_HEADS=local_heads,
        RANK=rank,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return z


@triton.jit
def _vha_add_mix_inplace_kernel(
    O_ptr,
    V_ptr,
    Z_ptr,
    o_stride_t,
    o_stride_h,
    v_stride_h,
    z_stride_t,
    z_stride_r,
    LOCAL_HEADS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Z_IS_FP32: tl.constexpr,
):
    """Compute o = (o + V @ z).to(bf16) in-place with late O load for high SM occupancy."""
    tok = tl.program_id(0).to(tl.int64)
    d_block = tl.program_id(1)
    heads = tl.arange(0, LOCAL_HEADS)
    ranks = tl.arange(0, RANK)
    cols = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    v_tile = tl.load(V_ptr + heads[:, None] * v_stride_h + ranks[None, :])
    z_tile = tl.load(Z_ptr + tok * z_stride_t + ranks[:, None] * z_stride_r + cols[None, :])
    if Z_IS_FP32:
        mix = tl.dot(v_tile.to(tl.float32), z_tile, input_precision="ieee")
    else:
        mix = tl.dot(v_tile, z_tile, out_dtype=tl.float32)
    o_ptrs = O_ptr + tok * o_stride_t + heads[:, None] * o_stride_h + cols[None, :]
    o_tile = tl.load(o_ptrs).to(tl.float32)
    tl.store(o_ptrs, (o_tile + mix).to(tl.bfloat16))


def _vha_add_mix(o, v, z):
    tokens, local_heads, head_dim = o.shape
    rank = v.shape[1]
    block_d = 64
    torch.library.wrap_triton(_vha_add_mix_inplace_kernel)[(tokens, head_dim // block_d)](
        o,
        v,
        z,
        o.stride(0),
        o.stride(1),
        v.stride(0),
        z.stride(0),
        z.stride(1),
        LOCAL_HEADS=local_heads,
        RANK=rank,
        BLOCK_D=block_d,
        Z_IS_FP32=z.dtype == torch.float32,
        num_warps=4,
    )
    return o


@triton.jit
def _vha_add_mix_sigmoid_mul_kernel(
    O_ptr,
    V_ptr,
    Z_ptr,
    G_ptr,
    OUT_ptr,
    o_stride_t,
    o_stride_h,
    v_stride_h,
    z_stride_t,
    z_stride_r,
    g_stride_t,
    out_stride_t,
    LOCAL_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Z_IS_FP32: tl.constexpr,
):
    """Fused DSA ((o + V @ z).to(bf16) * sigmoid(gate).to(bf16)).to(bf16) -> [T, LOCAL_HEADS * HEAD_DIM]."""
    tok = tl.program_id(0).to(tl.int64)
    d_block = tl.program_id(1)
    heads = tl.arange(0, LOCAL_HEADS)
    ranks = tl.arange(0, RANK)
    cols = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    v_tile = tl.load(V_ptr + heads[:, None] * v_stride_h + ranks[None, :])
    z_tile = tl.load(Z_ptr + tok * z_stride_t + ranks[:, None] * z_stride_r + cols[None, :])
    if Z_IS_FP32:
        mix = tl.dot(v_tile.to(tl.float32), z_tile, input_precision="ieee")
    else:
        mix = tl.dot(v_tile, z_tile, out_dtype=tl.float32)
    o_tile = tl.load(O_ptr + tok * o_stride_t + heads[:, None] * o_stride_h + cols[None, :]).to(tl.float32)
    mixed = (o_tile + mix).to(tl.bfloat16).to(tl.float32)
    flat_cols = heads[:, None] * HEAD_DIM + cols[None, :]
    g_tile = tl.load(G_ptr + tok * g_stride_t + flat_cols).to(tl.float32)
    s_tile = tl.sigmoid(g_tile).to(tl.bfloat16).to(tl.float32)
    out = (mixed * s_tile).to(tl.bfloat16)
    tl.store(OUT_ptr + tok * out_stride_t + flat_cols, out)


def _vha_add_mix_sigmoid_mul(o, v, z, gate):
    tokens, local_heads, head_dim = o.shape
    rank = v.shape[1]
    out = torch.empty((tokens, local_heads * head_dim), dtype=torch.bfloat16, device=o.device)
    block_d = 64
    torch.library.wrap_triton(_vha_add_mix_sigmoid_mul_kernel)[(tokens, head_dim // block_d)](
        o,
        v,
        z,
        gate,
        out,
        o.stride(0),
        o.stride(1),
        v.stride(0),
        z.stride(0),
        z.stride(1),
        gate.stride(0),
        out.stride(0),
        LOCAL_HEADS=local_heads,
        HEAD_DIM=head_dim,
        RANK=rank,
        BLOCK_D=block_d,
        Z_IS_FP32=z.dtype == torch.float32,
        num_warps=4,
    )
    return out


@triton.jit
def _pool_half_c128_kernel(
    ValuesA,
    ScoresA,
    ValuesB,
    ScoresB,
    Ape,
    StateValues,
    StateScores,
    Requests,
    Positions,
    Starts,
    Begins,
    StateRows,
    Output,
    SS: tl.constexpr,
    TS_A: tl.constexpr,
    TS_B: tl.constexpr,
    HALF_TOKENS: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """C128 gated pooling over one half-chunk [ROW_OFFSET : ROW_OFFSET + HALF_TOKENS] with zero cat."""
    local_tid = tl.program_id(0)
    token = ROW_OFFSET + local_tid
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(Requests + token)
    position = tl.load(Positions + token)
    start = tl.load(Starts + request)
    begin = tl.load(Begins + request)
    state = tl.load(StateRows + request)
    emitted = (position >= 0) & ((position + 1) % 128 == 0) & (state > 0)
    if emitted:
        offsets = tl.arange(0, 128)
        absolute = position + 1 - 128 + offsets
        valid = (absolute[:, None] >= 0) & (dim[None, :] < HEAD)
        is_local = absolute[:, None] >= start
        local_row = begin + absolute[:, None] - start
        in_a = is_local & (local_row < HALF_TOKENS)
        in_b = is_local & (local_row >= HALF_TOKENS)
        src_a = local_row * TS_A + dim[None, :]
        src_b = (local_row - HALF_TOKENS) * TS_B + dim[None, :]
        history = state * SS + (absolute[:, None] % 128) * HEAD + dim[None, :]
        values = tl.load(ValuesA + src_a, valid & in_a, other=0.0)
        if ROW_OFFSET > 0:
            values += tl.load(ValuesB + src_b, valid & in_b, other=0.0)
        values += tl.load(StateValues + history, valid & ~is_local, other=0.0)
        scores = tl.load(ScoresA + src_a, valid & in_a, other=0.0)
        if ROW_OFFSET > 0:
            scores += tl.load(ScoresB + src_b, valid & in_b, other=0.0)
        scores += tl.load(StateScores + history, valid & ~is_local, other=0.0)
        scores += tl.load(Ape + (absolute[:, None] % 128) * HEAD + dim[None, :], valid, other=0.0)
        scores = tl.where(valid, scores, -float("inf"))
        maximum = tl.max(scores, 0)
        probabilities = tl.exp(scores - maximum[None, :])
        pooled = tl.sum(values * probabilities, 0) / tl.sum(probabilities, 0)
        tl.store(Output + local_tid * HEAD + dim, pooled, dim < HEAD)
    else:
        tl.store(Output + local_tid * HEAD + dim, 0.0, dim < HEAD)


def _pool_half_c128(values_a, scores_a, values_b, scores_b, ape, state_values, state_scores, metadata, *, row_offset):
    half_tokens, head = values_a.shape
    output = torch.empty((half_tokens, head), dtype=torch.bfloat16, device=values_a.device)
    vb = values_a if values_b is None else values_b
    sb = scores_a if scores_b is None else scores_b
    torch.library.wrap_triton(_pool_half_c128_kernel)[(half_tokens, triton.cdiv(head, 32))](
        values_a,
        scores_a,
        vb,
        sb,
        ape,
        state_values,
        state_scores,
        metadata.requests,
        metadata.positions,
        metadata.starts,
        metadata.query_start_loc,
        metadata.state_rows,
        output,
        state_values.stride(0),
        values_a.stride(0),
        vb.stride(0),
        HALF_TOKENS=half_tokens,
        ROW_OFFSET=row_offset,
        HEAD=head,
        BLOCK=32,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _commit_half_c128_kernel(
    ValuesA,
    ScoresA,
    ValuesB,
    ScoresB,
    StateValues,
    StateScores,
    Starts,
    Begins,
    Counts,
    StateRows,
    SS: tl.constexpr,
    TS_A: tl.constexpr,
    TS_B: tl.constexpr,
    HALF_TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Commit trailing 128 C128 compressor rows from split (Half A, Half B) projections with zero cat."""
    request, distance = tl.program_id(0), tl.program_id(1)
    dim = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    start, count = tl.load(Starts + request), tl.load(Counts + request)
    begin, state = tl.load(Begins + request), tl.load(StateRows + request)
    end = start + count
    absolute = end - 1 - distance
    ring_row = absolute % 128
    valid = (state > 0) & (count > 0) & (absolute >= start) & (dim < WIDTH)
    local_row = begin + absolute - start
    in_a = valid & (local_row < HALF_TOKENS)
    in_b = valid & (local_row >= HALF_TOKENS)
    values = tl.load(ValuesA + local_row * TS_A + dim, in_a, other=0.0) + tl.load(
        ValuesB + (local_row - HALF_TOKENS) * TS_B + dim, in_b, other=0.0
    )
    scores = tl.load(ScoresA + local_row * TS_A + dim, in_a, other=0.0) + tl.load(
        ScoresB + (local_row - HALF_TOKENS) * TS_B + dim, in_b, other=0.0
    )
    destination = state * SS + ring_row * WIDTH + dim
    tl.store(StateValues + destination, values, valid)
    tl.store(StateScores + destination, scores, valid)


def _commit_half_c128(values_a, scores_a, values_b, scores_b, state_values, state_scores, metadata, counts):
    half_tokens, width = values_a.shape
    torch.library.wrap_triton(_commit_half_c128_kernel)[(counts.numel(), 128, triton.cdiv(width, 256))](
        values_a,
        scores_a,
        values_b,
        scores_b,
        state_values,
        state_scores,
        metadata.starts,
        metadata.query_start_loc,
        counts,
        metadata.state_rows,
        state_values.stride(0),
        values_a.stride(0),
        values_b.stride(0),
        HALF_TOKENS=half_tokens,
        WIDTH=width,
        BLOCK=256,
    )


@triton.jit
def _dsa_q_kv_norm_rope_write_kernel(
    X_ptr,
    QW_ptr,
    KW_ptr,
    ANGLES_ptr,
    Q_OUT_ptr,
    CACHE_ptr,
    SLOTS_ptr,
    cache_block_stride,
    cache_entry_stride,
    EPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """Fused DSA q_a_layernorm(1024) + kv_a_layernorm(512) + k_pe RoPE(64) + 656B GLM_NSA cache write."""
    row = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    if task == 0:
        qc = tl.arange(0, 1024)
        qx = tl.load(X_ptr + row * 1600 + qc).to(tl.float32)
        qw = tl.load(QW_ptr + qc).to(tl.float32)
        qinv = tl.rsqrt(tl.sum(qx * qx, 0) * (1.0 / 1024.0) + EPS)
        tl.store(Q_OUT_ptr + row * 1024 + qc, (qx * qinv * qw).to(tl.bfloat16))
    else:
        slot = tl.load(SLOTS_ptr + row).to(tl.int64)
        if slot < 0:
            return
        kc = tl.arange(0, 512)
        kx = tl.load(X_ptr + row * 1600 + 1024 + kc).to(tl.float32)
        kw = tl.load(KW_ptr + kc).to(tl.float32)
        kinv = tl.rsqrt(tl.sum(kx * kx, 0) * (1.0 / 512.0) + EPS)
        block_idx = slot // PAGE_SIZE
        block_off = slot - block_idx * PAGE_SIZE
        dst = CACHE_ptr + block_idx * cache_block_stride + block_off * cache_entry_stride
        # 4 groups of 128 dims -> 512 E4M3 bytes + 4 FP32 scales at [512:528]
        c128 = tl.arange(0, 128)
        for g in tl.static_range(4):
            g_off = g * 128 + c128
            gx = tl.load(X_ptr + row * 1600 + 1024 + g_off).to(tl.float32)
            gw = tl.load(KW_ptr + g_off).to(tl.float32)
            vals = (gx * kinv * gw).to(tl.bfloat16).to(tl.float32)
            amax = tl.max(tl.abs(vals), axis=0)
            scale = tl.where(amax > 0.0, amax / 448.0, 1.0)
            q = tl.clamp(vals / scale, -448.0, 448.0).to(tl.float8e4nv)
            tl.store(dst.to(tl.pointer_type(tl.float8e4nv)) + g_off, q)
            tl.store((dst + 512 + g * 4).to(tl.pointer_type(tl.float32)), scale)
        # 64 BF16 interleaved-RoPE k_pe -> 128 bytes at [528:656]
        pair = tl.arange(0, 32)
        a = tl.load(X_ptr + row * 1600 + 1536 + 2 * pair).to(tl.float32)
        b = tl.load(X_ptr + row * 1600 + 1536 + 2 * pair + 1).to(tl.float32)
        cos = tl.load(ANGLES_ptr + row * 64 + pair)
        sin = tl.load(ANGLES_ptr + row * 64 + 32 + pair)
        pe_dst = (dst + 528).to(tl.pointer_type(tl.bfloat16))
        tl.store(pe_dst + 2 * pair, (a * cos - b * sin).to(tl.bfloat16))
        tl.store(pe_dst + 2 * pair + 1, (a * sin + b * cos).to(tl.bfloat16))


def _dsa_q_kv_norm_rope_write(projected, qw, kw, angles, kv_cache, slots, eps):
    rows = projected.shape[0]
    q_c = torch.empty((rows, 1024), dtype=torch.bfloat16, device=projected.device)
    torch.library.wrap_triton(_dsa_q_kv_norm_rope_write_kernel)[(rows, 2)](
        projected,
        qw,
        kw,
        angles,
        q_c,
        kv_cache,
        slots,
        kv_cache.stride(0),
        kv_cache.stride(1),
        EPS=eps,
        PAGE_SIZE=kv_cache.shape[1],
        num_warps=4,
    )
    return q_c


@triton.jit
def _dsa_indexer_k_norm_neox_rope_write_kernel(
    KW_ptr,
    W_ptr,
    B_ptr,
    ANGLES_ptr,
    CACHE_FP8_ptr,
    CACHE_FP32_ptr,
    SLOTS_ptr,
    kw_stride,
    cache_stride,
    EPS: tl.constexpr,
):
    """Fused DSA Indexer LayerNorm(128) + leading-64 NeoX RoPE + FWHT-128 + UE8M0 FP8 cache write."""
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(SLOTS_ptr + row).to(tl.int64)
    if slot < 0:
        return
    dim = tl.arange(0, 128)
    k = tl.load(KW_ptr + row * kw_stride + dim).to(tl.float32)
    mean = tl.sum(k, axis=0) * (1.0 / 128.0)
    centered = k - mean
    var = tl.sum(centered * centered, axis=0) * (1.0 / 128.0)
    w = tl.load(W_ptr + dim).to(tl.float32)
    b = tl.load(B_ptr + dim).to(tl.float32)
    normed = (centered * tl.rsqrt(var + EPS) * w + b).to(tl.bfloat16).to(tl.float32)
    # NeoX RoPE on leading 64 dims: pairs (d, d+32) for d in [0, 32).
    partner_idx = tl.where(dim < 32, dim + 32, tl.where(dim < 64, dim - 32, dim))
    partner = tl.gather(normed, partner_idx, axis=0)
    freq_idx = dim & 31
    cos = tl.where(dim < 64, tl.load(ANGLES_ptr + row * 64 + freq_idx), 1.0)
    sin = tl.where(dim < 64, tl.load(ANGLES_ptr + row * 64 + 32 + freq_idx), 0.0)
    signed_sin = tl.where(dim < 32, -sin, sin)
    rot = (normed * cos + partner * signed_sin).to(tl.bfloat16)
    _write_pool(
        CACHE_FP8_ptr,
        CACHE_FP32_ptr,
        slot,
        rot,
        dim,
        dim < 128,
        slot >= 0,
        cache_stride,
        64,
        128,
        448.0,
        HADAMARD=True,
        POWER_OF_TWO_SCALE=True,
    )


def _dsa_indexer_k_norm_neox_rope_write(kw, k_norm_w, k_norm_b, angles, index_cache, slots, eps=1e-6):
    torch.library.wrap_triton(_dsa_indexer_k_norm_neox_rope_write_kernel)[(kw.shape[0],)](
        kw,
        k_norm_w,
        k_norm_b,
        angles,
        index_cache.view(torch.float8_e4m3fn),
        index_cache.view(torch.float32),
        slots,
        kw.stride(0),
        index_cache.stride(0),
        EPS=eps,
        num_warps=4,
    )


@triton.jit
def _neox_rope_leading64_grouped_kernel(
    Q_ptr,
    ANGLES_ptr,
    row_stride,
    head_stride,
    HEADS: tl.constexpr,
    HEADS_PER_CTA: tl.constexpr,
):
    """In-place leading-64 NeoX RoPE on [T, H, 128] indexer queries."""
    row = tl.program_id(0).to(tl.int64)
    h_block = tl.program_id(1)
    d = tl.arange(0, 32)
    cos = tl.load(ANGLES_ptr + row * 64 + d)
    sin = tl.load(ANGLES_ptr + row * 64 + 32 + d)
    row_base = Q_ptr + row * row_stride
    h0 = h_block * HEADS_PER_CTA
    for hi in tl.static_range(HEADS_PER_CTA):
        h = h0 + hi
        if h < HEADS:
            base = row_base + h * head_stride
            x1 = tl.load(base + d).to(tl.float32)
            x2 = tl.load(base + 32 + d).to(tl.float32)
            tl.store(base + d, (x1 * cos - x2 * sin).to(tl.bfloat16))
            tl.store(base + 32 + d, (x2 * cos + x1 * sin).to(tl.bfloat16))


def _neox_rope_leading64(q, angles):
    heads = q.shape[1]
    per_cta = 8 if heads % 8 == 0 else 4
    torch.library.wrap_triton(_neox_rope_leading64_grouped_kernel)[
        (q.shape[0], triton.cdiv(heads, per_cta))
    ](q, angles, q.stride(0), q.stride(1), HEADS=heads, HEADS_PER_CTA=per_cta, num_warps=1)


@triton.jit
def _flashindexer_select_heads_neox_rope_kernel(
    hidden_ptr,
    gate_w_ptr,
    bias_ptr,
    kw_ptr,
    q_raw_ptr,
    angles_ptr,
    q_sel_ptr,
    w_sel_ptr,
    T,
    kw_stride,
    HIDDEN: tl.constexpr,
    NH: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused FlashIndexer gate GEMM + top-8 selection + BF16 Q gather + leading-64 NeoX RoPE."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < T
    offs_m64 = offs_m.to(tl.int64)
    offs_n = tl.arange(0, NH)

    # (a) Gate GEMM: gates[BM, 64] = hidden[BM, 4096] @ gate_w^T[4096, 64]
    acc = tl.zeros((BLOCK_M, NH), dtype=tl.float32)
    for k0 in range(0, HIDDEN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        h_tile = tl.load(
            hidden_ptr + offs_m64[:, None] * HIDDEN + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        w_tile = tl.load(gate_w_ptr + offs_n[None, :] * HIDDEN + offs_k[:, None])
        acc = tl.dot(h_tile, w_tile, acc)

    gates = acc
    scores = gates + tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
    w_vec = tl.load(
        kw_ptr + offs_m64[:, None] * kw_stride + 128 + offs_n[None, :],
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    d32 = tl.arange(0, 32)
    d64 = tl.arange(0, 64)
    cos = tl.load(angles_ptr + offs_m64[:, None] * 64 + d32[None, :], mask=m_mask[:, None], other=1.0)
    sin = tl.load(angles_ptr + offs_m64[:, None] * 64 + 32 + d32[None, :], mask=m_mask[:, None], other=0.0)

    NEG_INF = -float("inf")
    cur = scores
    for i in tl.static_range(TOPK):
        idx = tl.argmax(cur, axis=1).to(tl.int32)
        sel_mask = offs_n[None, :] == idx[:, None]
        cur = tl.where(sel_mask, NEG_INF, cur)
        gate_at = tl.sum(tl.where(sel_mask, gates, 0.0), axis=1)
        w_at = tl.sum(tl.where(sel_mask, w_vec, 0.0), axis=1)
        tl.store(w_sel_ptr + offs_m64 * TOPK + i, w_at * gate_at, mask=m_mask)
        # Gather Q head [BM, 128] and apply leading-64 NeoX RoPE inline!
        src_base = q_raw_ptr + (offs_m64 * NH + idx.to(tl.int64))[:, None] * 128
        dst_base = q_sel_ptr + (offs_m64 * TOPK + i)[:, None] * 128
        x1 = tl.load(src_base + d32[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(src_base + 32 + d32[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        tail = tl.load(src_base + 64 + d64[None, :], mask=m_mask[:, None], other=0.0)
        tl.store(dst_base + d32[None, :], (x1 * cos - x2 * sin).to(tl.bfloat16), mask=m_mask[:, None])
        tl.store(dst_base + 32 + d32[None, :], (x2 * cos + x1 * sin).to(tl.bfloat16), mask=m_mask[:, None])
        tl.store(dst_base + 64 + d64[None, :], tail, mask=m_mask[:, None])


def _flashindexer_select_heads_neox_rope(hidden, flash_gate, flash_bias, kw, q_raw, angles):
    tokens = hidden.shape[0]
    q_sel = torch.empty((tokens, FLASHINDEXER_TOPK_HEADS, 128), dtype=torch.bfloat16, device=hidden.device)
    w_sel = torch.empty((tokens, FLASHINDEXER_TOPK_HEADS), dtype=torch.float32, device=hidden.device)
    block_m = 32
    torch.library.wrap_triton(_flashindexer_select_heads_neox_rope_kernel)[(triton.cdiv(tokens, block_m),)](
        hidden,
        flash_gate,
        flash_bias,
        kw,
        q_raw,
        angles,
        q_sel,
        w_sel,
        tokens,
        kw.stride(0),
        HIDDEN=hidden.shape[1],
        NH=64,
        TOPK=FLASHINDEXER_TOPK_HEADS,
        BLOCK_M=block_m,
        BLOCK_K=64,
        num_warps=4,
    )
    return q_sel, w_sel


@triton.jit
def _dsa_pack_q_rope_kernel(
    Q_NOPE_ABS_ptr,
    Q_RAW_ptr,
    ANGLES_ptr,
    Q_OUT_ptr,
    abs_stride_t,
    abs_stride_h,
    raw_stride_t,
    raw_stride_h,
    out_stride_t,
    out_stride_h,
    HEADS: tl.constexpr,
):
    """Copy absorbed q_nope [T, H, 512] + interleaved-RoPE q_pe [T, H, 64] -> q_full [T, H, 576]."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    c512 = tl.arange(0, 512)
    nope = tl.load(Q_NOPE_ABS_ptr + row * abs_stride_t + head * abs_stride_h + c512)
    dst = Q_OUT_ptr + row * out_stride_t + head * out_stride_h
    tl.store(dst + c512, nope)
    pair = tl.arange(0, 32)
    src_pe = Q_RAW_ptr + row * raw_stride_t + head * raw_stride_h + 192
    a = tl.load(src_pe + 2 * pair).to(tl.float32)
    b = tl.load(src_pe + 2 * pair + 1).to(tl.float32)
    cos = tl.load(ANGLES_ptr + row * 64 + pair)
    sin = tl.load(ANGLES_ptr + row * 64 + 32 + pair)
    tl.store(dst + 512 + 2 * pair, (a * cos - b * sin).to(tl.bfloat16))
    tl.store(dst + 512 + 2 * pair + 1, (a * sin + b * cos).to(tl.bfloat16))


def _dsa_pack_q_rope(q_absorbed, q_raw, angles):
    tokens, heads, _ = q_absorbed.shape
    q_full = torch.empty((tokens, heads, 576), dtype=torch.bfloat16, device=q_absorbed.device)
    torch.library.wrap_triton(_dsa_pack_q_rope_kernel)[(tokens, heads)](
        q_absorbed,
        q_raw,
        angles,
        q_full,
        q_absorbed.stride(0),
        q_absorbed.stride(1),
        q_raw.stride(0),
        q_raw.stride(1),
        q_full.stride(0),
        q_full.stride(1),
        HEADS=heads,
        num_warps=4,
    )
    return q_full


@triton.jit
def _merge_swa_topk_physical_kernel(
    TOPK_ptr,
    POS_ptr,
    PT_ptr,
    OUT_ptr,
    LEN_ptr,
    topk_stride,
    pt_stride,
    out_stride,
    num_blocks,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Merge logical top-k [2048] + trailing SWA [128] and map to physical 64-token DSA slots."""
    row = tl.program_id(0).to(tl.int64)
    pos = tl.load(POS_ptr + row).to(tl.int32)
    live = pos >= 0
    seq_len = tl.where(live, pos + 1, 0)
    idx_len = tl.minimum(tl.maximum(seq_len - WINDOW, 0), TOPK)
    swa_len = tl.minimum(seq_len, WINDOW)
    total = idx_len + swa_len
    tl.store(LEN_ptr + row, total)
    pt_base = PT_ptr + row * pt_stride
    topk_base = TOPK_ptr + row * topk_stride
    out_base = OUT_ptr + row * out_stride
    for start in tl.static_range(0, WIDTH, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        in_w = cols < WIDTH
        is_topk = cols < idx_len
        is_swa = (~is_topk) & (cols < total)
        topk_val = tl.load(topk_base + cols, mask=is_topk, other=-1)
        swa_val = seq_len - swa_len + cols - idx_len
        logical = tl.where(is_topk, topk_val, tl.where(is_swa, swa_val, -1))
        valid = live & (logical >= 0) & (logical < seq_len)
        page_idx = tl.where(valid, logical >> 7, 0)
        page_in_bounds = valid & (page_idx < num_blocks)
        model_page = tl.load(pt_base + page_idx, mask=page_in_bounds, other=-1)
        phys = tl.where(page_in_bounds & (model_page >= 0), (model_page << 7) + (logical & 127), -1)
        tl.store(out_base + cols, phys, mask=in_w)


def _merge_swa_topk_physical(topk_logical, positions, token_page_table):
    rows = topk_logical.shape[0]
    physical = torch.empty((rows, DSA_TOTAL_WIDTH), dtype=torch.int32, device=topk_logical.device)
    selected_lengths = torch.empty((rows,), dtype=torch.int32, device=topk_logical.device)
    torch.library.wrap_triton(_merge_swa_topk_physical_kernel)[(rows,)](
        topk_logical,
        positions,
        token_page_table,
        physical,
        selected_lengths,
        topk_logical.stride(0),
        token_page_table.stride(0),
        physical.stride(0),
        token_page_table.shape[1],
        WINDOW=DSA_WINDOW,
        TOPK=DSA_TOPK,
        WIDTH=DSA_TOTAL_WIDTH,
        BLOCK=256,
        num_warps=4,
    )
    return physical, selected_lengths


@triton.jit
def _sigmoid_mul_kernel(X_ptr, G_ptr, OUT_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT_ptr + offs, (x * s).to(tl.bfloat16), mask=mask)


def _sigmoid_mul(x, gate):
    out = torch.empty_like(x)
    n = x.numel()
    block = 1024
    torch.library.wrap_triton(_sigmoid_mul_kernel)[(triton.cdiv(n, block),)](
        x, gate, out, n, BLOCK=block, num_warps=4
    )
    return out


# ---------------------------------------------------------------------------
# Shared Attention Scratch & Indices (HCA C128 + DSA 656B/Index)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionScratch:
    # HCA buffers
    projected: torch.Tensor
    query_low: torch.Tensor
    query: torch.Tensor
    window_values: torch.Tensor
    o_gate_hca: torch.Tensor
    attended_hca: torch.Tensor
    vha_z_hca: torch.Tensor
    vha_z_hca_reduced: torch.Tensor
    output: torch.Tensor
    dense_indices: torch.Tensor
    # DSA buffers
    dsa_projected: torch.Tensor
    dsa_query: torch.Tensor
    dsa_q_absorbed: torch.Tensor
    dsa_q_full: torch.Tensor
    dsa_o_gate: torch.Tensor
    dsa_attended_lat: torch.Tensor
    dsa_expanded: torch.Tensor
    vha_z_dsa: torch.Tensor
    vha_z_dsa_reduced: torch.Tensor
    dsa_index_q: torch.Tensor
    dsa_index_q_sel: torch.Tensor
    dsa_index_kw: torch.Tensor
    dsa_index_w_sel: torch.Tensor
    dsa_topk_logical: torch.Tensor
    dsa_physical: torch.Tensor
    dsa_selected_lengths: torch.Tensor

    @classmethod
    def allocate(
        cls,
        tokens,
        world,
        device,
        *,
        blocks=8192,
        include_dsa=True,
        vha_dtype=torch.bfloat16,
    ):
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        heads = 64 // world
        groups = 8 // world
        dsa_tokens = tokens if include_dsa else 1
        return cls(
            projected=zeros((tokens, 1536)),
            query_low=zeros((tokens, 1024)),
            query=zeros((tokens, heads, 512)),
            window_values=zeros((tokens, 512)),
            o_gate_hca=zeros((tokens, groups * 1024)),
            attended_hca=zeros((tokens, heads, 512)),
            vha_z_hca=zeros((tokens, VHA_RANK, 512), vha_dtype),
            vha_z_hca_reduced=zeros((tokens, VHA_RANK, 512), vha_dtype),
            output=zeros((tokens, 4096)),
            dense_indices=zeros((tokens, blocks * (PAGE_TOKENS // 128)), torch.int32).fill_(-1),
            dsa_projected=zeros((dsa_tokens, 1600)),
            dsa_query=zeros((dsa_tokens, heads, 256)),
            dsa_q_absorbed=zeros((dsa_tokens, heads, 512)),
            dsa_q_full=zeros((dsa_tokens, heads, 576)),
            dsa_o_gate=zeros((dsa_tokens, heads * 256)),
            dsa_attended_lat=zeros((dsa_tokens, heads, 512)),
            dsa_expanded=zeros((dsa_tokens, heads, 256)),
            vha_z_dsa=zeros((dsa_tokens, VHA_RANK, 256), vha_dtype),
            vha_z_dsa_reduced=zeros((dsa_tokens, VHA_RANK, 256), vha_dtype),
            dsa_index_q=zeros((dsa_tokens, 64, 128)),
            dsa_index_q_sel=zeros((dsa_tokens, FLASHINDEXER_TOPK_HEADS, 128)),
            dsa_index_kw=zeros((dsa_tokens, 192)),
            dsa_index_w_sel=zeros((dsa_tokens, FLASHINDEXER_TOPK_HEADS), torch.float32),
            dsa_topk_logical=zeros((dsa_tokens, DSA_TOPK), torch.int32).fill_(-1),
            dsa_physical=zeros((dsa_tokens, DSA_TOTAL_WIDTH), torch.int32).fill_(-1),
            dsa_selected_lengths=zeros((dsa_tokens,), torch.int32),
        )

    def rows(self, tokens):
        def cut(t):
            return t[: min(tokens, t.shape[0])]

        return AttentionScratch(*(cut(getattr(self, f.name)) for f in self.__dataclass_fields__.values()))


class CompressionGeometry(NamedTuple):
    slots: torch.Tensor
    lengths: torch.Tensor
    index_lengths: torch.Tensor
    active_width: torch.Tensor
    position: torch.Tensor
    emitted: torch.Tensor
    emitted_slots: torch.Tensor


class DSAGeometry(NamedTuple):
    slots: torch.Tensor
    page_table: torch.Tensor
    index_lengths: torch.Tensor
    active_width: torch.Tensor
    cache_lengths: torch.Tensor


class AttentionGeometry(NamedTuple):
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lengths: torch.Tensor
    ratios: dict[int, CompressionGeometry]
    dense: torch.Tensor | None
    angles: dict[tuple[bool, int], torch.Tensor]
    dsa: DSAGeometry | None


class AttentionIndices(PreparedWindowIndices):
    """Window, C128 compression, DSA 64-token page tables, and split rotary angles."""

    def __init__(self, metadata, *, mode, tokens, include_dsa=True):
        page_size = 128 if mode == "decode" else 128 + metadata.max_query_tokens
        width = page_size if mode == "draft" else 128
        super().__init__(metadata, mode=mode, tokens=tokens, page_size=page_size, width=width)
        self.ratios, self.dense, self.angles = {}, None, {}
        self.include_dsa = include_dsa
        device = metadata.positions.device
        self.token_offsets = persistent_copy(torch.arange(tokens, dtype=torch.int64, device=device))
        record_numeric_tensor(self.token_offsets)
        self.dsa_geometry = None
        if include_dsa:
            blocks = metadata.page_table.shape[1]
            pt_rows = 1 if mode == "prefill" else tokens
            self.dsa_geometry = DSAGeometry(
                slots=torch.full((tokens,), -1, dtype=torch.int64, device=device),
                page_table=torch.full((pt_rows, blocks * DSA_SUBPAGES_PER_PAGE), -1, dtype=torch.int32, device=device),
                index_lengths=torch.zeros((tokens,), dtype=torch.int32, device=device),
                active_width=torch.zeros((1,), dtype=torch.int32, device=device),
                cache_lengths=torch.zeros((tokens,), dtype=torch.int32, device=device),
            )

    def rotary_angles(self, rotary, *, compressed, ratio=0):
        key = (bool(compressed), int(ratio))
        if key not in self.angles:
            positions = self.metadata.positions if ratio == 0 else self.compressed(ratio).position
            self.angles[key] = PreparedRotaryAngles(positions, rotary.frequencies[bool(compressed)], split=True)
        return self.angles[key]

    def compressed(self, ratio):
        if ratio not in self.ratios:
            device, tokens = self.window_slots.device, self.window_slots.shape[0]

            def zeros(dtype):
                return torch.zeros((tokens,), dtype=dtype, device=device)

            self.ratios[ratio] = CompressionGeometry(
                slots=zeros(torch.int64),
                lengths=zeros(torch.int32),
                index_lengths=zeros(torch.int32),
                active_width=torch.zeros(1, dtype=torch.int32, device=device),
                position=zeros(torch.int64),
                emitted=zeros(torch.bool),
                emitted_slots=zeros(torch.int64),
            )
        return self.ratios[ratio]

    def dense_pages(self, target):
        if self.dense is None:
            self.dense = target
        return self.dense

    def functional(self):
        self.compression_ratios = tuple(self.ratios)
        self.has_dense = self.dense is not None
        self.ratios = self.dense = self.dsa_geometry = None
        for operation in self.angles.values():
            operation.functional()
        super().functional()
        return self

    def __call__(self, metadata):
        base = super().__call__(metadata)
        m = metadata
        live, position = m.positions >= 0, m.positions.clamp_min(0)
        ratios = {}
        pages = None
        if self.compression_ratios or self.include_dsa:
            pages = m.page_table[m.requests, (position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)]
        for ratio in self.compression_ratios:
            lengths = torch.where(live, (position + 1) // ratio, 0).to(torch.int32)
            slots = torch.where(live, pages.long() * (PAGE_TOKENS // ratio) + (position % PAGE_TOKENS) // ratio, -1)
            emitted = live & ((m.positions + 1) % ratio == 0)
            ratios[ratio] = CompressionGeometry(
                slots=slots,
                lengths=lengths,
                index_lengths=lengths,
                active_width=lengths.amax().view(1),
                position=((position + 1) // ratio * ratio - ratio).clamp_min(0),
                emitted=emitted,
                emitted_slots=torch.where(emitted, slots, -1),
            )
        dense = None
        if self.has_dense:
            tp = m.token_page_table
            # 128-token model page = 1 C128 record -> flat record index == model page index!
            dense = torch.where(tp >= 0, tp, -1).to(torch.int32)
        dsa = None
        if self.include_dsa:
            dsa_slots = torch.where(live, pages.long() * PAGE_TOKENS + (position % PAGE_TOKENS), -1)
            src_pt = m.page_table[:1] if self.mode == "prefill" else m.token_page_table
            sub = torch.arange(DSA_SUBPAGES_PER_PAGE, dtype=torch.int32, device=src_pt.device)
            dsa_pt = torch.where(
                src_pt[:, :, None] >= 0, src_pt[:, :, None] * DSA_SUBPAGES_PER_PAGE + sub, -1
            ).flatten(1).contiguous()
            idx_lens = torch.where(live, (position + 1 - DSA_WINDOW).clamp_min(0), 0).to(torch.int32)
            cache_lens = torch.where(live, position + 1, 0).to(torch.int32)
            dsa = DSAGeometry(
                slots=dsa_slots,
                page_table=dsa_pt,
                index_lengths=idx_lens,
                active_width=idx_lens.amax().view(1),
                cache_lengths=cache_lens,
            )
        angles = {
            key: operation(m.positions if key[1] == 0 else ratios[key[1]].position)
            for key, operation in self.angles.items()
        }
        return AttentionGeometry(
            window_slots=base.window_slots,
            window_indices=base.window_indices,
            window_lengths=base.window_lengths,
            ratios=ratios,
            dense=dense,
            angles=angles,
            dsa=dsa,
        )


# ---------------------------------------------------------------------------
# Prepared HCA Attention (37 target layers + 3 DSpark draft layers)
# ---------------------------------------------------------------------------

class PreparedHCAAttention:
    def __init__(
        self,
        layer,
        weights,
        source,
        scratch: AttentionScratch,
        metadata,
        pool,
        rotary,
        *,
        mode,
        world,
        runtime,
        coordinator,
        session,
        name,
        workspace,
        indices,
        mla_compute_mode="auto",
        examples=None,
        pipelined=False,
    ):
        self.layer, self.weights, self.pool, self.rotary, self.mode = layer, weights, pool, rotary, mode
        self.world = world
        self.draft = mode == "draft"
        self.ratio = 0 if self.draft else 128
        self.eps = pool.config.text["rms_norm_eps"]
        self.pipelined = bool(pipelined)
        tokens, rows, device = source.shape[0], metadata.starts.numel(), source.device
        self.tokens = tokens
        self.half_tokens = tokens // 2 if self.pipelined else tokens
        self.heads = scratch.query.shape[1]
        self.window = pool.window(layer, draft=self.draft)
        self.window_page_size = indices.window_page_size
        self.window_shape = (rows, latent_page_bytes(self.window_page_size))
        self.window_cache = (
            self.window
            if mode == "decode"
            else declaration_tensor(examples, "attention.window", self.window_shape, torch.uint8, device)
        )
        self.q_norm_weight = weights["q_norm"]
        self.kv_norm_weight = weights["kv_norm"]
        self.vha_u = weights["vha_u"]
        self.vha_v = weights["vha_v"]
        for t in (self.q_norm_weight, self.kv_norm_weight, self.vha_u, self.vha_v):
            record_numeric_tensor(t)
        self.vha_dtype = scratch.vha_z_hca.dtype
        self.angles = indices.rotary_angles(rotary, compressed=not self.draft, ratio=0)
        self.compressed_angles = (
            indices.rotary_angles(rotary, compressed=True, ratio=128) if self.ratio else None
        )
        self.token_offsets = indices.token_offsets
        shared = indices.compressed(128) if self.ratio else None

        # Input projections (half_tokens when pipelined for full-half HCA overlap, else full tokens)
        h = self.half_tokens
        self.qkv = QuantizedProjection(
            source[:h],
            weights["qkv_a"],
            scratch.projected[:h],
            session=session,
            name=name + ".qkv_a",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.qkv",
        )
        self.qb = QuantizedProjection(
            scratch.query_low[:h],
            weights["q_b"],
            scratch.query[:h].view(h, -1),
            session=session,
            name=name + ".q_b",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.query",
        )
        self.o_gate = QuantizedProjection(
            scratch.query_low[:h],
            weights["o_gate"],
            scratch.o_gate_hca[:h],
            session=session,
            name=name + ".o_gate",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.o_gate",
        )
        self.window_writer = PreparedCompressedCacheWriter(
            scratch.window_values[:h],
            self.window_cache,
            indices.window_slots[:h],
            page_size=self.window_page_size,
            kind="swa",
            session=session,
            name=name + ".window_write",
            cache_format="deepseek_v4",
        )
        self.compressor = self.compression = self.compressed_writer = None
        self.compressor_norm = self.compressor_ape = None
        self.compressor_state_values = self.compressor_state_scores = None
        indexed = {}
        if self.ratio:
            spec = Compressor(layer, 128, 512)
            width = spec.state_shape[1]
            compressed_buf = declaration_tensor(
                examples, "attention.compression", (h, 2 * width), torch.float32, device
            )
            self.compression = UnquantizedProjection(
                source[:h], weights["compression"], compressed_buf, session=session, name=name + ".compression"
            )
            if self.pipelined:
                from hyperflare_recipes.components.rms_norm import PreparedRMSNorm

                self.compressor_ape = weights["compressor"]["ape"]
                record_numeric_tensor(self.compressor_ape)
                self.compressor_state_values = pool.reserved_region(f"{spec.prefix}.values")
                self.compressor_state_scores = pool.reserved_region(f"{spec.prefix}.scores")
                pooled_buf = declaration_tensor(
                    examples, "attention.compressor.pooled", (h, spec.head_dim), torch.bfloat16, device
                )
                latent_buf = declaration_tensor(
                    examples, "attention.compressor.output", (h, spec.head_dim), torch.bfloat16, device
                )
                self.compressor_norm = PreparedRMSNorm(
                    pooled_buf,
                    weights["compressor"]["norm"],
                    latent_buf,
                    eps=self.eps,
                    session=session,
                    name=name + ".compress.norm",
                )
                self.compressed_writer = PreparedCompressedCacheWriter(
                    latent_buf,
                    pool.pages[f"layer{layer}.latent_page"],
                    shared.emitted_slots[:h],
                    page_size=PAGE_TOKENS // 128,
                    kind="indexed",
                    session=session,
                    name=name + ".compressed_write",
                    cache_format="deepseek_v4",
                )
            else:
                self.compressor = PreparedGatedCompressor(
                    spec,
                    weights["compressor"],
                    source,
                    metadata,
                    pool,
                    shared.slots,
                    session=session,
                    name=name + ".compress",
                    deferred=mode == "verify",
                    examples=examples,
                    example_role="attention.compressor",
                    emitted=shared.emitted,
                    emitted_slots=shared.emitted_slots,
                    projected=(compressed_buf[:, :width], compressed_buf[:, width : 2 * width]),
                )
                self.compressed_writer = PreparedCompressedCacheWriter(
                    self.compressor.output,
                    pool.pages[f"layer{layer}.latent_page"],
                    self.compressor.slots,
                    page_size=PAGE_TOKENS // 128,
                    kind="indexed",
                    session=session,
                    name=name + ".compressed_write",
                    cache_format="deepseek_v4",
                )
            selected = indices.dense_pages(scratch.dense_indices[:, : metadata.page_table.shape[1]])
            indexed = {
                "indexed": pool.pages[f"layer{layer}.latent_page"],
                "indexed_indices": selected[:h],
                "indexed_lengths": shared.index_lengths[:h],
                "indexed_page_size": PAGE_TOKENS // 128,
            }

        mla_mode = "decode" if mode in ("decode", "verify", "draft") else "extend"
        self.attention = PreparedCompressedMLA(
            scratch.query[:h],
            self.window_cache,
            indices.window_indices[:h],
            indices.window_lengths[:h],
            weights["sink"],
            scratch.attended_hca[:h],
            window_page_size=self.window_page_size,
            mode=mla_mode,
            decode_row_capacity=h if mla_mode == "decode" else None,
            session=session,
            name=name + ".attention",
            workspace=workspace,
            cache_format="deepseek_v4",
            compute_mode=mla_compute_mode,
            **indexed,
        )
        # VHA AllReduce: [h, 16, 512] -> flat [h, 8192]
        if self.pipelined:
            self.vha_reduce_0 = PreparedAllReduce(
                scratch.vha_z_hca[:h].view(h, -1),
                scratch.vha_z_hca_reduced[:h].view(h, -1),
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".vha_reduce.0",
            )
            self.vha_reduce_1 = PreparedAllReduce(
                scratch.vha_z_hca[h:].view(h, -1),
                scratch.vha_z_hca_reduced[h:].view(h, -1),
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".vha_reduce.1",
            )
            self.vha_reduce = None
        else:
            self.vha_reduce = PreparedAllReduce(
                scratch.vha_z_hca.view(tokens, -1),
                scratch.vha_z_hca_reduced.view(tokens, -1),
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".vha_reduce",
            )
            self.vha_reduce_0 = self.vha_reduce_1 = None

        self.output_projection = PreparedGroupedOutputProjection(
            scratch.attended_hca[:h],
            weights["output"],
            scratch.output[:h],
            session=session,
            name=name + ".output",
            workspace=workspace,
            inv_rope=True,
            heads_per_group=self.heads // weights["output"].groups,
            nope_dim=448,
            rope_dim=64,
            positions=self.token_offsets[:h],
            cos_sin_cache=self.angles.values[:h],
            gate=scratch.o_gate_hca[:h],
        )

        # Overlap o_gate on aux stream AFTER q_b (behind latency-bound window/compressor/MLA/VHA AR)
        # so q_b runs at uncontended GDDR7 bandwidth!
        self.concurrent_gate = None
        if (
            not self.pipelined
            and mode in ("decode", "verify", "draft")
            and isinstance(workspace, ScratchArena)
            and workspace.overlap
        ):
            parallel_plans = [self.window_writer.plan, self.attention.plan]
            if self.compression is not None:
                parallel_plans.extend(
                    p
                    for p in (
                        self.compression.plan,
                        self.compressor.normalize.plan,
                        self.compressed_writer.plan,
                    )
                    if p is not None
                )
            self.concurrent_gate = ConcurrentProjectionChain(
                (self.o_gate,),
                parallel_plans=parallel_plans,
                session=session,
                workspace=workspace,
            )

        self.cr_vha_0 = self.cr_vha_1 = None
        if self.pipelined and not self.vha_reduce_0.single:
            # cr_vha_0 overlaps with ALL of Half B prefix + Half B attention (o_gate deferred into cr_vha_1!).
            vha0_parallel = tuple(
                p
                for p in (
                    self.qkv.plan,
                    getattr(self.qkv, "quantizer_plan", None),
                    self.qb.plan,
                    getattr(self.qb, "quantizer_plan", None),
                    self.window_writer.plan,
                    None if self.compression is None else self.compression.plan,
                    None if self.compressor_norm is None else self.compressor_norm.plan,
                    None if self.compressed_writer is None else self.compressed_writer.plan,
                    self.attention.plan,
                )
                if p is not None
            )
            vha1_parallel = tuple(
                p
                for p in (
                    self.o_gate.plan,
                    getattr(self.o_gate, "quantizer_plan", None),
                    self.output_projection.plan,
                )
                if p is not None
            )
            self.cr_vha_0 = ConcurrentAllReduce(
                self.vha_reduce_0,
                workspace=workspace,
                session=session,
                parallel_plans=vha0_parallel,
            )
            self.cr_vha_1 = ConcurrentAllReduce(
                self.vha_reduce_1,
                workspace=workspace,
                session=session,
                parallel_plans=vha1_parallel,
            )

        # Functionalize all subcomponents
        self.qkv.functional(session=session)
        self.qb.functional(session=session)
        if self.concurrent_gate is not None:
            self.concurrent_gate.functional(session=session)
        else:
            self.o_gate.functional(session=session)
        self.window_writer.functional(session=session, cache=None if mode != "decode" else self.window)
        if self.compression is not None:
            self.compression.functional(session=session)
            if self.compressor is not None:
                self.compressor.functional(session=session)
            if self.compressor_norm is not None:
                self.compressor_norm = self.compressor_norm.functional()
            self.compressed_writer.functional(session=session)
        self.attention.functional(session=session, window=None if mode != "decode" else self.window)
        self.output_projection.functional(session=session)
        if self.pipelined:
            if self.cr_vha_0 is not None:
                self.cr_vha_0.functional(session=session)
                self.cr_vha_1.functional(session=session)
            else:
                self.vha_reduce_0.functional(session=session)
                self.vha_reduce_1.functional(session=session)
        else:
            self.vha_reduce.functional(session=session)
        register_finalizer(session, self._release_examples)

    def _release_examples(self):
        self.window_cache = None
        if self.concurrent_gate is not None:
            self.o_gate = None

    def plans(self):
        result = [
            self.qkv.plan,
            getattr(self.qkv, "quantizer_plan", None),
            self.qb.plan,
            getattr(self.qb, "quantizer_plan", None),
            self.window_writer.plan,
            self.attention.plan,
            self.output_projection.plan,
        ]
        if self.o_gate is not None:
            result.extend((self.o_gate.plan, getattr(self.o_gate, "quantizer_plan", None)))
        if self.compression is not None:
            result.extend(
                (
                    self.compression.plan,
                    self.compressor.normalize.plan if self.compressor is not None else self.compressor_norm.plan,
                    self.compressed_writer.plan,
                )
            )
        return tuple(p for p in result if p is not None)

    def _run_prefix(self, source, metadata, indices):
        m = metadata
        window = (
            self.window
            if self.mode == "decode"
            else torch.empty(self.window_shape, dtype=torch.uint8, device=source.device)
        )
        if self.mode != "decode":
            transfer_window(self.window, window, m, self.window_page_size)
        angles = indices.angles[(not self.draft, 0)]
        projected = self.qkv(source)
        query_low, window_values = _hca_q_kv_norm_rope(
            projected, self.q_norm_weight, self.kv_norm_weight, angles, self.eps
        )
        query = self.qb(query_low).view(source.shape[0], self.heads, 512)
        gate_pending = self.concurrent_gate.start(query_low) if self.concurrent_gate is not None else None
        _hca_normalize_rotate(query, angles, self.eps)
        self.window_writer(window_values, indices.window_slots, cache=window)
        compressed_records = None
        if self.ratio:
            shared = indices.ratios[128]
            comp_proj = self.compression(source)
            width = self.compressor.spec.state_shape[1]
            latent, values, scores, _, slots = self.compressor(
                source,
                m,
                shared.slots,
                projected=(comp_proj[:, :width], comp_proj[:, width : 2 * width]),
                emitted=shared.emitted,
                emitted_slots=shared.emitted_slots,
            )
            compressed_records = (values, scores)
            self.compressed_angles.apply(latent, indices.angles[(True, 128)])
            self.compressed_writer(latent, slots)
        gate = self.o_gate(query_low) if gate_pending is None else None
        return query, gate, gate_pending, window, compressed_records, angles

    def __call__(self, source, metadata, indices):
        query, gate, gate_pending, window, compressed_records, angles = self._run_prefix(source, metadata, indices)
        shared = indices.ratios[128] if self.ratio else None
        attended = self.attention(
            query,
            indices.window_indices,
            indices.window_lengths,
            indexed_indices=indices.dense if self.ratio else None,
            indexed_lengths=None if shared is None else shared.index_lengths,
            window=window,
        )
        z = _vha_project_z(attended, self.vha_u, out_dtype=self.vha_dtype)
        z_red = self.vha_reduce(z.view(source.shape[0], -1)).view_as(z)
        mixed = _vha_add_mix(attended, self.vha_v, z_red)
        if gate_pending is not None:
            gate = self.concurrent_gate.finish(gate_pending, mixed)
        output = self.output_projection(
            mixed,
            positions=self.token_offsets,
            cos_sin_cache=angles,
            gate=gate,
        )
        records = (window, compressed_records)
        if self.mode == "prefill":
            self.commit(metadata.lengths, records, metadata)
        return output, records

    def init_window_cache(self, metadata, device):
        window = torch.empty(self.window_shape, dtype=torch.uint8, device=device)
        transfer_window(self.window, window, metadata, self.window_page_size)
        return window

    def run_half_front(self, source_half, metadata, indices, *, window, half_idx, comp_a=None):
        """Run one half-chunk [h, 4096] through QKV, Q_b, window write, C128 compress, MLA, and VHA z.

        o_gate is deferred into the VHA AR(B) window so both o_gate(A) and o_gate(B) hide behind
        the 32 MB VHA PCIe DMA transfer instead of adding to Half A/B front latency!
        """
        h = self.half_tokens
        sl = slice(0, h) if half_idx == 0 else slice(h, 2 * h)
        angles_h = indices.angles[(True, 0)][sl]
        projected = self.qkv(source_half)
        query_low, window_values = _hca_q_kv_norm_rope(
            projected, self.q_norm_weight, self.kv_norm_weight, angles_h, self.eps
        )
        query = self.qb(query_low).view(h, self.heads, 512)
        _hca_normalize_rotate(query, angles_h, self.eps)
        self.window_writer(window_values, indices.window_slots[sl], cache=window)
        shared = indices.ratios[128]
        comp_proj = self.compression(source_half)
        vals_h, scores_h = comp_proj[:, :512], comp_proj[:, 512:1024]
        if half_idx == 0:
            pooled = _pool_half_c128(
                vals_h,
                scores_h,
                None,
                None,
                self.compressor_ape,
                self.compressor_state_values,
                self.compressor_state_scores,
                metadata,
                row_offset=0,
            )
        else:
            vals_a, scores_a = comp_a
            pooled = _pool_half_c128(
                vals_a,
                scores_a,
                vals_h,
                scores_h,
                self.compressor_ape,
                self.compressor_state_values,
                self.compressor_state_scores,
                metadata,
                row_offset=h,
            )
            _commit_half_c128(
                vals_a,
                scores_a,
                vals_h,
                scores_h,
                self.compressor_state_values,
                self.compressor_state_scores,
                metadata,
                metadata.lengths,
            )
        latent = self.compressor_norm(pooled)
        self.compressed_angles.apply(latent, indices.angles[(True, 128)][sl])
        self.compressed_writer(latent, shared.emitted_slots[sl])
        att = self.attention(
            query,
            indices.window_indices[sl],
            indices.window_lengths[sl],
            indexed_indices=indices.dense[sl],
            indexed_lengths=shared.index_lengths[sl],
            window=window,
        )
        z = _vha_project_z(att, self.vha_u, out_dtype=self.vha_dtype)
        return att, z, query_low, (vals_h, scores_h), angles_h

    def run_pipelined_full(
        self,
        norm_a,
        prepare_norm_b,
        metadata,
        indices,
        *,
        cr_att_0,
        att_reduce_0=None,
        prev_pending=None,
        prev_finish_cr=None,
    ):
        """Full-half 2-microbatch DMA/SM overlapped HCA prefill (overlaps prev FFN AR(B) with ALL of Half A!)."""
        h = self.half_tokens
        window = self.init_window_cache(metadata, norm_a.device)
        # Half A: QKV + Q_b + window + C128 compress + MLA + VHA_z (all overlapped with prev FFN AR(B)!)
        att_a, z_a, qlow_a, comp_a, angles_a = self.run_half_front(
            norm_a, metadata, indices, window=window, half_idx=0
        )
        vha_deps = ()
        prev_res = None
        if prev_pending is not None:
            prev_res = prev_finish_cr.finish(prev_pending, z_a)
            vha_deps = (prev_res,)
        # Overlap VHA AR(A) with Half B (prev fc2_latent + attn_hc(B) + QKV + Q_b + comp + MLA + VHA_z)!
        p_vha_a = self.cr_vha_0.start(z_a.view(h, -1), dependencies=vha_deps) if self.cr_vha_0 is not None else None
        norm_b, b_hc_state = prepare_norm_b(prev_res)
        att_b, z_b, qlow_b, _, angles_b = self.run_half_front(
            norm_b, metadata, indices, window=window, half_idx=1, comp_a=comp_a
        )
        z_a_red = (
            self.cr_vha_0.finish(p_vha_a, z_b).view_as(z_a)
            if p_vha_a is not None
            else self.vha_reduce_0(z_a.view(h, -1)).view_as(z_a)
        )
        # Overlap VHA AR(B) with o_gate(A) + Half A VHA mix + Wo(A) + o_gate(B)!
        p_vha_b = self.cr_vha_1.start(z_b.view(h, -1), dependencies=(z_a_red,)) if self.cr_vha_1 is not None else None
        gate_a = self.o_gate(qlow_a)
        mixed_a = _vha_add_mix(att_a, self.vha_v, z_a_red)
        out_a = self.output_projection(
            mixed_a, positions=self.token_offsets[:h], cos_sin_cache=angles_a, gate=gate_a
        )
        gate_b = self.o_gate(qlow_b)
        z_b_red = (
            self.cr_vha_1.finish(p_vha_b, gate_b).view_as(z_b)
            if p_vha_b is not None
            else self.vha_reduce_1(z_b.view(h, -1)).view_as(z_b)
        )
        # Overlap Attn AR(A) with Half B VHA mix + Wo(B) + window commit!
        p_att_a = cr_att_0.start(out_a, dependencies=(z_b_red,)) if cr_att_0 is not None else None
        mixed_b = _vha_add_mix(att_b, self.vha_v, z_b_red)
        out_b = self.output_projection(
            mixed_b, positions=self.token_offsets[:h], cos_sin_cache=angles_b, gate=gate_b
        )
        records = (window, None)
        if self.mode == "prefill":
            transfer_window(window, self.window, metadata, self.window_page_size, counts=metadata.lengths)
        reduced_a = (
            cr_att_0.finish(p_att_a, out_b)
            if p_att_a is not None
            else att_reduce_0(out_a)
        )
        return reduced_a, out_b, records, b_hc_state

    def commit(self, counts, records, metadata):
        window, compressed = records
        if self.mode != "decode":
            transfer_window(window, self.window, metadata, self.window_page_size, counts=counts)
        if self.mode == "verify" and self.compressor is not None:
            self.compressor.commit(counts, compressed, metadata)


# ---------------------------------------------------------------------------
# Prepared DSA Attention (6 layers: 7, 14, 21, 28, 35, 42)
# ---------------------------------------------------------------------------

class PreparedDSAAttention:
    def __init__(
        self,
        layer,
        weights,
        source,
        scratch: AttentionScratch,
        metadata,
        pool,
        rotary,
        *,
        mode,
        world,
        runtime,
        coordinator,
        session,
        name,
        workspace,
        indices,
        max_cache_pages=None,
        examples=None,
        pipelined=False,
    ):
        self.layer, self.weights, self.pool, self.rotary, self.mode = layer, weights, pool, rotary, mode
        self.world = world
        self.eps = pool.config.text["rms_norm_eps"]
        self.pipelined = bool(pipelined)
        tokens, device = source.shape[0], source.device
        self.tokens = tokens
        self.half_tokens = tokens // 2 if self.pipelined else tokens
        self.heads = scratch.dsa_query.shape[1]
        self.kv_cache = pool.dsa_kv_cache(layer)
        self.index_cache = pool.dsa_index_cache(layer)
        self.q_norm_weight = weights["q_norm"]
        self.kv_norm_weight = weights["kv_norm"]
        self.vha_u = weights["vha_u"]
        self.vha_v = weights["vha_v"]
        self.index_k_norm_w = weights["index_k_norm_w"]
        self.index_k_norm_b = weights["index_k_norm_b"]
        self.flash_gate = weights.get("flash_gate")
        self.flash_bias = weights.get("flash_bias")
        self.use_flashindexer = (mode == "prefill") and (self.flash_gate is not None)
        for t in (
            self.q_norm_weight,
            self.kv_norm_weight,
            self.vha_u,
            self.vha_v,
            self.index_k_norm_w,
            self.index_k_norm_b,
            self.flash_gate,
            self.flash_bias,
        ):
            if t is not None:
                record_numeric_tensor(t)
        self.vha_dtype = scratch.vha_z_dsa.dtype
        self.angles = indices.rotary_angles(rotary, compressed=False, ratio=0)
        dsa_geom = indices.dsa_geometry

        # Projections (full tokens)
        self.qkv = QuantizedProjection(
            source,
            weights["qkv_a"],
            scratch.dsa_projected,
            session=session,
            name=name + ".qkv_a",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="dsa.qkv",
        )
        self.qb = QuantizedProjection(
            scratch.query_low,
            weights["q_b"],
            scratch.dsa_query.view(tokens, -1),
            session=session,
            name=name + ".q_b",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="dsa.q_b",
        )
        gate_rows = self.half_tokens if self.pipelined else tokens
        self.o_gate = QuantizedProjection(
            scratch.query_low[:gate_rows],
            weights["o_gate"],
            scratch.dsa_o_gate[:gate_rows],
            session=session,
            name=name + ".o_gate",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="dsa.o_gate",
        )
        self.index_q_proj = QuantizedProjection(
            scratch.query_low,
            weights["index_q"],
            scratch.dsa_index_q.view(tokens, -1),
            session=session,
            name=name + ".index_q",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="dsa.index_q",
        )
        self.index_kw_proj = UnquantizedProjection(
            source,
            weights["index_kw"],
            scratch.dsa_index_kw,
            session=session,
            name=name + ".index_kw",
        )
        # Batched W_UK absorption & W_UV expansion
        h = self.half_tokens
        self.absorb = BatchedUnquantizedProjection(
            scratch.dsa_query[:h, :, :192].transpose(0, 1),
            weights["w_uk"],
            scratch.dsa_q_absorbed[:h].transpose(0, 1),
            major="n",
            session=session,
            name=name + ".absorb",
        )
        self.expand = BatchedUnquantizedProjection(
            scratch.dsa_attended_lat[:h].transpose(0, 1),
            weights["w_uv"],
            scratch.dsa_expanded[:h].transpose(0, 1),
            major="k",
            session=session,
            name=name + ".expand",
        )
        self.o_proj = QuantizedProjection(
            scratch.dsa_expanded[:h].view(h, -1),
            weights["o_proj"],
            scratch.output[:h],
            session=session,
            name=name + ".o_proj",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="dsa.o_proj",
        )
        # FP8 Indexer (8-head FlashIndexer in prefill; 64-head in decode/verify)
        self.selectors = []
        idx_mode = "prefill" if mode == "prefill" else "decode"
        step = INDEX_ROWS if mode == "prefill" else tokens
        idx_pt_full = dsa_geom.page_table.expand(tokens, -1) if mode == "prefill" else dsa_geom.page_table
        for first in range(0, tokens, step):
            last = min(tokens, first + step)
            q_ex = (
                scratch.dsa_index_q_sel[first:last]
                if self.use_flashindexer
                else scratch.dsa_index_q[first:last]
            )
            w_ex = (
                scratch.dsa_index_w_sel[first:last]
                if self.use_flashindexer
                else scratch.dsa_index_kw[first:last, 128:]
            )
            self.selectors.append(
                PreparedFP8Indexer(
                    q_ex,
                    w_ex,
                    self.index_cache,
                    idx_pt_full[first:last],
                    dsa_geom.index_lengths[first:last],
                    dsa_geom.active_width,
                    scratch.dsa_topk_logical[first:last],
                    mode=idx_mode,
                    session=session,
                    name=f"{name}.index{first}",
                    workspace=workspace,
                    topk=DSA_TOPK,
                    head_weight_scale=64.0 ** -0.5,
                    hadamard=True,
                    key_power_of_two=True,
                    prefill_records_per_call=131072 if mode == "prefill" else None,
                    examples=examples,
                    example_role=f"dsa.indexer.{'fi8' if self.use_flashindexer else 'h64'}",
                )
            )
        # Sparse MLA (GLM_NSA 656B cache, topk=2176, sm_scale=256**-0.5, attention_sink)
        max_kv_rows = max(self.kv_cache.shape[0], DSA_SUBPAGES_PER_PAGE * (max_cache_pages or 0)) * 64
        caps = sparse_mla.Caps(
            device=device,
            num_q_heads=self.heads,
            max_q_rows=h,
            max_width=DSA_TOTAL_WIDTH,
            softmax_scale=256.0 ** -0.5,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            head_dim=576,
            v_head_dim=512,
            model_type=int(sparse_mla.ModelType.GLM_NSA),
            mode="decode" if mode in ("decode", "verify") else "extend",
            page_size=64,
            max_batch=h,
            max_kv_rows=max_kv_rows,
            max_page_table_width=dsa_geom.page_table.shape[1],
            has_attention_sink=True,
        )
        self.attention = PreparedSparseMLA(
            scratch.dsa_q_full[:h],
            self.kv_cache,
            scratch.dsa_physical[:h],
            dsa_geom.cache_lengths[:h],
            scratch.dsa_selected_lengths[:h],
            scratch.dsa_attended_lat[:h],
            caps=caps,
            session=session,
            name=name + ".sparse_mla",
            workspace=workspace,
            attention_sink=weights["sink"],
        )
        # VHA AllReduce: [h, 16, 256] -> flat [h, 4096]
        if self.pipelined:
            self.vha_reduce_0 = PreparedAllReduce(
                scratch.vha_z_dsa[:h].view(h, -1),
                scratch.vha_z_dsa_reduced[:h].view(h, -1),
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".vha_reduce.0",
            )
            self.vha_reduce_1 = PreparedAllReduce(
                scratch.vha_z_dsa[h:].view(h, -1),
                scratch.vha_z_dsa_reduced[h:].view(h, -1),
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".vha_reduce.1",
            )
            self.vha_reduce = None
        else:
            self.vha_reduce = PreparedAllReduce(
                scratch.vha_z_dsa.view(tokens, -1),
                scratch.vha_z_dsa_reduced.view(tokens, -1),
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".vha_reduce",
            )
            self.vha_reduce_0 = self.vha_reduce_1 = None

        # Overlap DSA o_gate (1024->8192) AFTER q_b/index_q behind latency-bound indexer + sparse_mla + VHA AR!
        self.concurrent_gate = None
        if (
            not self.pipelined
            and mode in ("decode", "verify")
            and isinstance(workspace, ScratchArena)
            and workspace.overlap
        ):
            parallel_plans = [
                p
                for p in (
                    *(sel.query_plan for sel in self.selectors),
                    *(sel.selection.plan for sel in self.selectors),
                    self.attention.plan,
                )
                if p is not None
            ]
            self.concurrent_gate = ConcurrentProjectionChain(
                (self.o_gate,),
                parallel_plans=parallel_plans,
                session=session,
                workspace=workspace,
            )

        self.cr_vha_0 = self.cr_vha_1 = None
        if self.pipelined and not self.vha_reduce_0.single:
            self.cr_vha_0 = ConcurrentAllReduce(
                self.vha_reduce_0,
                workspace=workspace,
                session=session,
                parallel_plans=(self.attention.plan,),
            )
            self.cr_vha_1 = ConcurrentAllReduce(
                self.vha_reduce_1,
                workspace=workspace,
                session=session,
                parallel_plans=tuple(
                    p
                    for p in (
                        self.o_gate.plan,
                        getattr(self.o_gate, "quantizer_plan", None),
                        self.o_proj.plan,
                        getattr(self.o_proj, "quantizer_plan", None),
                    )
                    if p is not None
                ),
            )

        # Functionalize all subcomponents
        for proj in (self.qkv, self.qb, self.index_q_proj, self.index_kw_proj, self.absorb, self.expand, self.o_proj):
            proj.functional(session=session)
        if self.concurrent_gate is not None:
            self.concurrent_gate.functional(session=session)
        else:
            self.o_gate.functional(session=session)
        for sel in self.selectors:
            sel.functional(session=session)
        self.attention.functional(session=session)
        if self.pipelined:
            if self.cr_vha_0 is not None:
                self.cr_vha_0.functional(session=session)
                self.cr_vha_1.functional(session=session)
            else:
                self.vha_reduce_0.functional(session=session)
                self.vha_reduce_1.functional(session=session)
        else:
            self.vha_reduce.functional(session=session)
        register_finalizer(session, self._release_examples)

    def _release_examples(self):
        if self.concurrent_gate is not None:
            self.o_gate = None

    def plans(self):
        result = [
            self.qkv.plan,
            getattr(self.qkv, "quantizer_plan", None),
            self.qb.plan,
            getattr(self.qb, "quantizer_plan", None),
            self.index_q_proj.plan,
            getattr(self.index_q_proj, "quantizer_plan", None),
            self.index_kw_proj.plan,
            self.attention.plan,
            self.o_proj.plan,
            getattr(self.o_proj, "quantizer_plan", None),
            *(sel.query_plan for sel in self.selectors),
            *(sel.selection.plan for sel in self.selectors),
        ]
        if self.o_gate is not None:
            result.extend((self.o_gate.plan, getattr(self.o_gate, "quantizer_plan", None)))
        return tuple(p for p in result if p is not None)

    def _run_prefix(self, source, metadata, indices):
        dsa = indices.dsa
        angles = indices.angles[(False, 0)]
        projected = self.qkv(source)
        q_c = _dsa_q_kv_norm_rope_write(
            projected, self.q_norm_weight, self.kv_norm_weight, angles, self.kv_cache, dsa.slots, self.eps
        )
        q_raw = self.qb(q_c).view(source.shape[0], self.heads, 256)
        # Indexer
        kw = self.index_kw_proj(source)
        _dsa_indexer_k_norm_neox_rope_write(
            kw, self.index_k_norm_w, self.index_k_norm_b, angles, self.index_cache, dsa.slots, 1e-6
        )
        idx_q_raw = self.index_q_proj(q_c).view(source.shape[0], 64, 128)
        gate_pending = self.concurrent_gate.start(q_c) if self.concurrent_gate is not None else None
        gate = None if (gate_pending is not None or self.pipelined) else self.o_gate(q_c)
        if self.use_flashindexer:
            idx_q, idx_w = _flashindexer_select_heads_neox_rope(
                source, self.flash_gate, self.flash_bias, kw, idx_q_raw, angles
            )
        else:
            _neox_rope_leading64(idx_q_raw, angles)
            idx_q = idx_q_raw
            idx_w = kw[:, 128:].contiguous()
        step = INDEX_ROWS if self.mode == "prefill" else source.shape[0]
        pt_full = dsa.page_table.expand(source.shape[0], -1) if self.mode == "prefill" else dsa.page_table
        if len(self.selectors) == 1:
            topk_logical = self.selectors[0](
                idx_q, idx_w, pt_full, dsa.index_lengths, dsa.active_width
            )
        else:
            topk_logical = torch.cat(
                [
                    sel(
                        idx_q[first : first + step],
                        idx_w[first : first + step],
                        pt_full[first : first + step],
                        dsa.index_lengths[first : first + step],
                        dsa.active_width,
                    )
                    for first, sel in zip(range(0, source.shape[0], step), self.selectors, strict=True)
                ],
                dim=0,
            )
        physical, selected_lengths = _merge_swa_topk_physical(
            topk_logical, metadata.positions, metadata.token_page_table
        )
        return q_raw, (q_c if self.pipelined else gate), gate_pending, physical, selected_lengths, angles

    def __call__(self, source, metadata, indices):
        dsa = indices.dsa
        q_raw, gate, gate_pending, physical, selected_lengths, angles = self._run_prefix(source, metadata, indices)
        q_abs = self.absorb(q_raw[:, :, :192].transpose(0, 1)).transpose(0, 1)
        q_full = _dsa_pack_q_rope(q_abs, q_raw, angles)
        attended_lat = self.attention(q_full, physical, dsa.cache_lengths, selected_lengths)
        expanded = self.expand(attended_lat.transpose(0, 1)).transpose(0, 1)
        z = _vha_project_z(expanded, self.vha_u, out_dtype=self.vha_dtype)
        z_red = self.vha_reduce(z.view(source.shape[0], -1)).view_as(z)
        if gate_pending is not None:
            gate = self.concurrent_gate.finish(gate_pending, z_red)
        gated = _vha_add_mix_sigmoid_mul(expanded, self.vha_v, z_red, gate)
        output = self.o_proj(gated)
        return output, None

    def run_pipelined(self, source, metadata, indices, *, cr_att_0, att_reduce_0=None, prev_pending=None, prev_finish_cr=None):
        """2-microbatch DMA/SM overlapped DSA prefill."""
        h = self.half_tokens
        sl_a, sl_b = slice(0, h), slice(h, 2 * h)
        dsa = indices.dsa
        q_raw, q_c, _, physical, selected_lengths, angles = self._run_prefix(source, metadata, indices)
        # Half A: absorb -> pack+rope -> sparse MLA -> expand -> VHA project (zero contiguous copies!)
        q_abs_a = self.absorb(q_raw[sl_a, :, :192].transpose(0, 1)).transpose(0, 1)
        q_full_a = _dsa_pack_q_rope(q_abs_a, q_raw[sl_a], angles[sl_a])
        att_a = self.attention(q_full_a, physical[sl_a], dsa.cache_lengths[sl_a], selected_lengths[sl_a])
        exp_a = self.expand(att_a.transpose(0, 1)).transpose(0, 1)
        z_a = _vha_project_z(exp_a, self.vha_u, out_dtype=self.vha_dtype)
        vha_deps = ()
        if prev_pending is not None:
            prev_res = prev_finish_cr.finish(prev_pending, z_a)
            vha_deps = (prev_res,)
        else:
            prev_res = None
        # Overlap VHA AR(A) with Half B absorb + sparse MLA + expand + VHA project(B)!
        p_vha_a = self.cr_vha_0.start(z_a.view(h, -1), dependencies=vha_deps) if self.cr_vha_0 is not None else None
        q_abs_b = self.absorb(q_raw[sl_b, :, :192].transpose(0, 1)).transpose(0, 1)
        q_full_b = _dsa_pack_q_rope(q_abs_b, q_raw[sl_b], angles[sl_b])
        att_b = self.attention(q_full_b, physical[sl_b], dsa.cache_lengths[sl_b], selected_lengths[sl_b])
        exp_b = self.expand(att_b.transpose(0, 1)).transpose(0, 1)
        z_b = _vha_project_z(exp_b, self.vha_u, out_dtype=self.vha_dtype)
        z_a_red = (
            self.cr_vha_0.finish(p_vha_a, z_b).view_as(z_a)
            if p_vha_a is not None
            else self.vha_reduce_0(z_a.view(h, -1)).view_as(z_a)
        )
        # Overlap VHA AR(B) with o_gate(A) + Half A fused VHA mix+sigmoid_mul + o_proj(A) + o_gate(B)!
        p_vha_b = self.cr_vha_1.start(z_b.view(h, -1), dependencies=(z_a_red,)) if self.cr_vha_1 is not None else None
        gate_a = self.o_gate(q_c[sl_a])
        out_a = self.o_proj(_vha_add_mix_sigmoid_mul(exp_a, self.vha_v, z_a_red, gate_a))
        gate_b = self.o_gate(q_c[sl_b])
        z_b_red = (
            self.cr_vha_1.finish(p_vha_b, gate_b).view_as(z_b)
            if p_vha_b is not None
            else self.vha_reduce_1(z_b.view(h, -1)).view_as(z_b)
        )
        # Overlap Attn AR(A) with Half B fused VHA mix+sigmoid_mul + o_proj(B)!
        p_att_a = cr_att_0.start(out_a, dependencies=(z_b_red,)) if cr_att_0 is not None else None
        out_b = self.o_proj(_vha_add_mix_sigmoid_mul(exp_b, self.vha_v, z_b_red, gate_b))
        reduced_a = (
            cr_att_0.finish(p_att_a, out_b)
            if p_att_a is not None
            else att_reduce_0(out_a)
        )
        return reduced_a, out_b, None, prev_res

    def commit(self, counts, records, metadata):
        pass
