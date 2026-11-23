# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Prepared block-scaled experts with route-weighted SiTU intermediates.

The route weight precedes the intermediate BF16 rounding and quantization.
NVFP4 uses the four-or-six MAE rule with a per-row amax/1536 outer scale;
MXFP8 uses nearest E4M3 values and upward-rounded UE8M0 block scales.

SiTU, scale rounding, and SM120 expert arithmetic are adapted from vLLM
PR #69028 at b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0:
https://github.com/vllm-project/vllm/blob/b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0/vllm/models/inkling2/nvidia/ops/activation.py
https://github.com/vllm-project/vllm/blob/b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0/vllm/models/inkling2/nvidia/ops/mxfp8.py
https://github.com/vllm-project/vllm/blob/b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0/vllm/models/inkling2/nvidia/ops/moe_sm120.py
Flare prepares persistent buffers and captures these operations through b12x.
"""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.runtime.jit import MockTensor
from b12x.moe.fused_moe._token_outer import _clear, _pack, _tasks

from ignition.preparation.triton import prepare_triton, launch_triton
from hyperflare_recipes.components.four_six_quantization import _quantize_block


@triton.jit
def _quantize(X, Q, S, Outer, K: tl.constexpr, FP4: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, B)
    x = tl.load(X + row * K + cols, cols < K, 0).to(tl.float32)
    if FP4:
        amax = tl.max(tl.abs(x), 0)
        encode = tl.minimum(tl.div_rn(1536., amax), 3.4028234663852886e38)
        outer = amax * (1. / 1536.)
        q, s = _quantize_block(x, encode, tl.div_rn(1., encode), outer, B)
        tl.store(Q + row * (K // 2) + tl.arange(0, B // 2), q, tl.arange(0, B // 2) < K // 2)
        tl.store(S + row * (K // 16) + tl.arange(0, B // 16), s, tl.arange(0, B // 16) < K // 16)
    else:
        x = tl.reshape(x, (B // 32, 32))
        raw = tl.max(tl.abs(x), 1) * (1. / 448.)
        minimum = tl.full((), 2. ** -127, tl.float32)
        exponent = ((raw.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000) >> 23
        exponent = tl.where(raw <= minimum, 0, tl.minimum(exponent, 254))
        scale = tl.maximum((exponent << 23).to(tl.float32, bitcast=True), minimum)
        inverse = tl.where(exponent == 0, 2. ** 127, libdevice.rcp_rn(scale))
        q = tl.minimum(tl.maximum(x * inverse[:, None], -448.), 448.).to(tl.float8e4nv)
        tl.store(Q + row * K + cols, tl.reshape(q, (B,)), cols < K)
        groups = tl.arange(0, B // 32)
        tl.store(S + row * (K // 32) + groups, exponent, groups < K // 32)
        outer = 1.
    tl.store(Outer + row, amax if FP4 else outer)


@triton.jit
def _expand(scales):
    r: tl.constexpr = scales.shape[0]
    g: tl.constexpr = scales.shape[1]
    exponents = scales.to(tl.uint32)
    bits = tl.where(exponents == 0, 0x00400000, exponents << 23)
    return tl.reshape(tl.broadcast_to(bits.to(tl.float32, bitcast=True)[:, :, None], (r, g, 32)), (r, g * 32))


@triton.jit
def _gemm(A, AS, AO, W, WS, WG, Rows, Counts, TaskExpert, TaskBegin, Y,
          K: tl.constexpr, N: tl.constexpr, TopK: tl.constexpr, Capacity: tl.constexpr,
          FIRST: tl.constexpr, FP4: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    task = tl.program_id(0)
    expert = tl.load(TaskExpert + task).to(tl.int64)
    if expert >= 0:
        r = tl.load(TaskBegin + task) + tl.arange(0, BM)
        live = r < tl.load(Counts + expert)
        route = tl.load(Rows + expert * Capacity + r, live, 0).to(tl.int64)
        ar = route // TopK if FIRST else route
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        if FP4:
            for base in range(tl.cdiv(K, BK)):
                k = base * (BK // 2) + tl.arange(0, BK // 2)
                g = base * (BK // 16) + tl.arange(0, BK // 16)
                a = tl.load(A + ar[:, None] * (K // 2) + k[None, :], live[:, None] & (k[None, :] < K // 2), 0)
                sa = tl.load(AS + ar[:, None] * (K // 16) + g[None, :], live[:, None] & (g[None, :] < K // 16), 0.0)
                w = tl.load(W + (expert * N + n[None, :]) * (K // 2) + k[:, None], (n[None, :] < N) & (k[:, None] < K // 2), 0)
                sw = tl.load(WS + (expert * N + n[:, None]) * (K // 16) + g[None, :], (n[:, None] < N) & (g[None, :] < K // 16), 0.0)
                acc = tl.dot_scaled(a, sa, "e2m1", w, sw, "e2m1", acc)
        else:
            for base in range(tl.cdiv(K, BK)):
                k = base * BK + tl.arange(0, BK)
                g = base * (BK // 32) + tl.arange(0, BK // 32)
                a = tl.load(A + ar[:, None] * K + k[None, :], live[:, None] & (k[None, :] < K), 0.0)
                sa = tl.load(AS + ar[:, None] * (K // 32) + g[None, :], live[:, None] & (g[None, :] < K // 32), 0)
                w = tl.load(W + (expert * N + n[:, None]) * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0.0)
                sw = tl.load(WS + (expert * N + n[:, None]) * (K // 32) + g[None, :], (n[:, None] < N) & (g[None, :] < K // 32), 0)
                a = (a.to(tl.float32) * _expand(sa)).to(tl.bfloat16)
                w = (w.to(tl.float32) * _expand(sw)).to(tl.bfloat16)
                acc = tl.dot(a, tl.trans(w), acc)
        if FP4:
            gain = (tl.load(AO + ar, live, 0) * tl.load(WG + expert)) * libdevice.rcp_rn(1536. * tl.load(WG + E + expert))
            acc *= gain[:, None]
        tl.store(Y + route[:, None] * N + n[None, :], acc, live[:, None] & (n[None, :] < N))


@triton.jit
def _activate(Pair, Weights, Ids, Y, I: tl.constexpr, B: tl.constexpr):
    route = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, B)
    live = (cols < I) & (tl.load(Ids + route) >= 0)
    gate = tl.load(Pair + route * (2 * I) + 2 * cols, live, 0).to(tl.float32)
    up = tl.load(Pair + route * (2 * I) + 2 * cols + 1, live, 0).to(tl.float32)
    sigmoid = tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 t, t; "
        "add.rn.f32 t, t, 0f3F800000; rcp.approx.ftz.f32 $0, t; }",
        "=f,f", [gate], dtype=tl.float32, is_pure=True, pack=1)
    gate = 4. * tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f",
        [tl.div_rn(gate, 4.)], dtype=tl.float32, is_pure=True, pack=1)
    up = 25. * tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f",
        [tl.div_rn(up, 25.)], dtype=tl.float32, is_pure=True, pack=1)
    value = (gate * sigmoid) * up
    value *= tl.load(Weights + route)
    tl.store(Y + route * I + cols, value, cols < I)


def _gemm_tile(tokens, nvfp4):
    """Select bit-exact GEMM tile geometry (BM, BN, BK, num_warps) for token count and format."""
    if nvfp4:
        if tokens >= 64:
            return 64, 128, 256, 8
        if tokens >= 4:
            return 16, 128, 256, 4
        return 16, 64, 256, 4
    if tokens >= 4096:
        return 64, 128, 128, 8
    if tokens >= 4:
        return 16, 128, 256, 4
    return 16, 64, 256, 4


class PreparedSiTUExperts:
    def __init__(self, weights, *, tokens, top_k, output_dtype=torch.bfloat16):
        self.weights, self.tokens, self.top_k = weights, tokens, top_k
        self.output_dtype = output_dtype
        self.hidden, self.intermediate = weights.w2.shape[1], weights.w13.shape[1] // 2
        self.experts, self.rows = weights.w2.shape[0], tokens * top_k
        self.bm, self.bn, self.bk, self.num_warps = _gemm_tile(tokens, weights.nvfp4)
        self.tasks = triton.cdiv(self.rows, self.bm) + self.experts
        self.programs = {}
        self._declare()

    def _call(self, name, kernel, args, grid, **options):
        grid = (*grid, *((1,) * (3 - len(grid))))
        # Compiled forwards trace the kernel, never the process-local launch handle.
        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            launch_triton(kernel, None, args, grid, **options)
        elif name not in self.programs:
            self.programs[name] = prepare_triton(kernel, args, grid, **options).compiled
        else:
            launch_triton(kernel, self.programs[name], args, grid, **options)

    def _declare(self):
        # Mock tensors compile each specialization without retaining example storage.
        mock = lambda dtype: MockTensor(dtype, (1,))
        self._execute(mock(torch.bfloat16), mock(torch.int32), mock(torch.float32), mock=mock)

    def _execute(self, source, ids, weights, mock=None):
        w, h, i, r = self.weights, self.hidden, self.intermediate, self.rows
        fp4 = w.nvfp4
        if mock is None and w.expert_offset is not None:
            local_ids = ids - w.expert_offset
            owned = (local_ids >= 0) & (local_ids < self.experts)
            ids = local_ids.masked_fill(~owned, -1)
        def empty(shape, dtype):
            return mock(dtype) if mock else torch.empty(shape, dtype=dtype, device=source.device)
        counts = empty((self.experts,), torch.int32)
        rows = empty((self.experts, r), torch.int32)
        valid = empty((r,), torch.uint8)
        te, tb = (empty((self.tasks,), torch.int32) for _ in range(2))
        self._call("clear", _clear, (counts, te), (triton.cdiv(max(self.experts, self.tasks), 256),),
                   E=self.experts, Tasks=self.tasks, B=256)
        self._call("pack", _pack, (ids, counts, rows, valid, r), (triton.cdiv(r, 256),),
                   E=self.experts, Capacity=r, B=256)
        self._call("tasks", _tasks, (counts, te, tb), (self.experts,), E=self.experts, BM=self.bm,
                   BE=triton.next_power_of_2(self.experts), BT=triton.next_power_of_2(triton.cdiv(r, self.bm)))
        value = source
        for first, n, k, matrix, scales, gain in ((True, 2*i, h, w.w13, w.s13, w.g13),
                                                 (False, h, i, w.w2, w.s2, w.g2)):
            label = "first" if first else "second"
            m = self.tokens if first else r
            q = empty((m, k // 2 if fp4 else k), torch.uint8 if fp4 else torch.float8_e4m3fn)
            s = empty((m, k // (16 if fp4 else 32)), torch.float8_e4m3fn if fp4 else torch.uint8)
            outer = empty((m,), torch.float32)
            self._call(label + "_quant", _quantize, (value, q, s, outer), (m,),
                       K=k, FP4=fp4, B=triton.next_power_of_2(k), enable_fp_fusion=False)
            output = empty((r, n), torch.bfloat16)
            self._call(label + "_gemm", _gemm,
                       (q, s, outer, matrix, scales, gain, rows, counts, te, tb, output),
                       (self.tasks, triton.cdiv(n, self.bn)), K=k, N=n, TopK=self.top_k, Capacity=r,
                       FIRST=first, FP4=fp4, E=self.experts, BM=self.bm, BN=self.bn, BK=self.bk,
                       num_warps=self.num_warps, enable_fp_fusion=False)
            if first:
                value = empty((r, i), torch.bfloat16)
                self._call("activate", _activate, (output, weights, ids, value), (r,),
                           I=i, B=triton.next_power_of_2(i), enable_fp_fusion=False)
        if mock:
            return None
        output = output.view(self.tokens, self.top_k, h).float()
        summed = torch.where((ids >= 0)[:, :, None], output, 0).sum(1)
        return summed if self.output_dtype == torch.float32 else summed.to(self.output_dtype)

    def __call__(self, source, ids, weights):
        return self._execute(source, ids, weights)
