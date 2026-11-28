# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed causal depthwise convolutions with explicit raw-history snapshots."""

import torch
import triton
import triton.language as tl

from ignition.preparation.tensor_metadata import record_numeric_tensor


@triton.jit
def _forward(X, Weight, Dynamic, History, Requests, Positions, QueryStarts, Slots, Y, Previous,
             TOKENS: tl.constexpr, SEQS: tl.constexpr, CHANNELS: tl.constexpr, WIDTH: tl.constexpr,
             XS: tl.constexpr, WS0: tl.constexpr, WS1: tl.constexpr,
             DS0: tl.constexpr, DS1: tl.constexpr, DS2: tl.constexpr,
             HS0: tl.constexpr, HS1: tl.constexpr, DYNAMIC: tl.constexpr,
             GROUP: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    column_live = channel < CHANNELS
    if row < SEQS:
        slot = tl.load(Slots + row).to(tl.int64)
        for lag in tl.static_range(WIDTH - 1):
            value = tl.load(History + slot * HS0 + lag * HS1 + channel,
                            column_live, other=0)
            tl.store(Previous + (row * (WIDTH - 1) + lag) * CHANNELS + channel,
                     value, column_live)
    if row < TOKENS:
        request = tl.load(Requests + row).to(tl.int32)
        position = tl.load(Positions + row)
        live = (position >= 0) & (request >= 0) & (request < SEQS)
        begin = tl.load(QueryStarts + request, live, other=0)
        slot = tl.load(Slots + request, live, other=0).to(tl.int64)
        current = tl.load(X + row.to(tl.int64) * XS + channel, column_live & live, other=0).to(tl.float32)
        total = tl.full((BLOCK,), 0., tl.float32)
        for tap in tl.static_range(WIDTH):
            lag = tap if DYNAMIC else WIDTH - 1 - tap
            relative = row - begin - lag
            from_batch = relative >= 0
            valid = live & (position - lag >= 0)
            value = tl.load(X + (row.to(tl.int64) - lag) * XS + channel,
                            column_live & valid & from_batch, other=0).to(tl.float32)
            old = tl.load(History + slot * HS0 + (WIDTH - 1 + relative) * HS1 + channel,
                          column_live & valid & ~from_batch, other=0).to(tl.float32)
            value = tl.where(from_batch, value, old)
            if DYNAMIC:
                weight = tl.load(Weight + tap * WS0 + channel * WS1, column_live, other=0).to(tl.float32)
                delta = tl.load(Dynamic + row.to(tl.int64) * DS0 + (channel // GROUP) * DS1 + tap * DS2,
                                column_live & live, other=0).to(tl.float32)
                weight = weight + delta
            else:
                weight = tl.load(Weight + channel * WS0 + tap * WS1, column_live, other=0).to(tl.float32)
            product = value * weight
            if tap == 0:
                total = product
            else:
                total = total + product
        result = (total + current) * SCALE
        tl.store(Y + row.to(tl.int64) * CHANNELS + channel, tl.where(live, result, 0.), column_live)


@triton.jit
def _commit(Previous, X, Starts, QueryStarts, Slots, Counts, History,
            TOKENS: tl.constexpr, CHANNELS: tl.constexpr, LENGTH: tl.constexpr,
            XS: tl.constexpr, HS0: tl.constexpr, HS1: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(Counts + request)
    slot = tl.load(Slots + request).to(tl.int64)
    live = (count > 0) & (slot >= 0)
    start = tl.load(Starts + request)
    begin = tl.load(QueryStarts + request)
    for lag in tl.static_range(LENGTH):
        relative = count - LENGTH + lag
        token = begin + relative
        from_batch = relative >= 0
        value = tl.load(X + token.to(tl.int64) * XS + channel,
                        (channel < CHANNELS) & live & from_batch & (token < TOKENS), other=0)
        old = tl.load(Previous + (request * LENGTH + LENGTH + relative) * CHANNELS + channel,
                      (channel < CHANNELS) & live & ~from_batch, other=0)
        value = tl.where(start + relative >= 0, tl.where(from_batch, value, old), 0.)
        tl.store(History + slot * HS0 + lag * HS1 + channel, value, (channel < CHANNELS) & live)


@triton.jit
def _forward_and_commit_decode(
    X, Weight, Dynamic, History, Starts, QueryStarts, Slots, Lengths, Y,
    TOKENS: tl.constexpr, SEQS: tl.constexpr, CHANNELS: tl.constexpr, WIDTH: tl.constexpr,
    XS: tl.constexpr, WS0: tl.constexpr, WS1: tl.constexpr,
    DS0: tl.constexpr, DS1: tl.constexpr, DS2: tl.constexpr,
    HS0: tl.constexpr, HS1: tl.constexpr, DYNAMIC: tl.constexpr,
    GROUP: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr,
):
    """Single-launch decode convolution with in-place history shift (at most one token per request)."""
    req = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_live = channel < CHANNELS

    total_live = tl.load(QueryStarts + SEQS)
    if req >= total_live and req < TOKENS:
        tl.store(Y + req.to(tl.int64) * CHANNELS + channel, tl.zeros((BLOCK,), dtype=tl.bfloat16), col_live)

    length = tl.load(Lengths + req)
    slot = tl.load(Slots + req).to(tl.int64)
    live = (length > 0) & (slot >= 0)
    if live:
        start = tl.load(Starts + req)
        tok = tl.load(QueryStarts + req).to(tl.int64)
        cur_bf16 = tl.load(X + tok * XS + channel, col_live, other=0)
        cur = cur_bf16.to(tl.float32)

        h_base = History + slot * HS0 + channel
        total = tl.full((BLOCK,), 0., tl.float32)
        for tap in tl.static_range(WIDTH):
            lag = tap if DYNAMIC else WIDTH - 1 - tap
            valid = (start - lag) >= 0
            if lag == 0:
                val = cur
            else:
                val = tl.load(h_base + (WIDTH - 1 - lag) * HS1, col_live & valid, other=0).to(tl.float32)
            if DYNAMIC:
                w = tl.load(Weight + tap * WS0 + channel * WS1, col_live, other=0).to(tl.float32)
                delta = tl.load(Dynamic + tok * DS0 + (channel // GROUP) * DS1 + tap * DS2, col_live, other=0).to(tl.float32)
                w = w + delta
            else:
                w = tl.load(Weight + channel * WS0 + tap * WS1, col_live, other=0).to(tl.float32)
            prod = val * w
            if tap == 0:
                total = prod
            else:
                total = total + prod
        res = (total + cur) * SCALE
        tl.store(Y + tok * CHANNELS + channel, res.to(tl.bfloat16), col_live)

        # Shift history left by one in ascending order so each read precedes its slot's overwrite.
        for lag in tl.static_range(WIDTH - 2):
            older = tl.load(h_base + (lag + 1) * HS1, col_live, other=0)
            tl.store(h_base + lag * HS1, tl.where(start - (WIDTH - 2 - lag) >= 0, older, 0), col_live)
        tl.store(h_base + (WIDTH - 2) * HS1, tl.where(start >= 0, cur_bf16, 0), col_live)


class PreparedCausalConvolution:
    """Residual-scaled convolution and accepted-prefix raw-history commits.

    Static weights are ``[channels, 1, W]`` or ``[channels, W]`` in oldest-first
    order. With positive ``dynamic_heads``, weights are ``[W, channels]`` in
    current-first order, and calls supply BF16 ``[tokens, heads, W]`` additive
    coefficients. Multiplication and ordered accumulation use FP32 without FMA;
    only the output rounds to BF16. The residual scales are ``1/sqrt(W+1)`` and
    ``1/sqrt(2W+1)`` respectively.

    History is BF16 ``[slots, W-1, channels]``, oldest first, with a possibly
    padded slot stride. Forward returns an independent history snapshot and
    leaves persistent history unchanged. Retain that snapshot and the raw input
    until ``commit`` consumes the accepted counts. Active destination slots must
    be unique. Complete-forward warmup resolves both kernels before capture.
    """

    def __init__(self, weight, history, *, dynamic_heads=0):
        if (type(dynamic_heads) is not int or dynamic_heads < 0 or weight.device.type != "cuda"
                or history.device != weight.device or weight.dtype not in (torch.bfloat16, torch.float32)
                or history.dtype != torch.bfloat16 or history.ndim != 3
                or min(history.shape) <= 0 or history.stride(2) != 1 or history.stride(1) != history.shape[2]
                or history.stride(0) < history.shape[1] * history.shape[2]):
            raise ValueError("causal convolution requires CUDA FP32/BF16 weights and dense BF16 history rows")
        self.channels = history.shape[2]
        self.width = history.shape[1] + 1
        if dynamic_heads:
            if weight.shape != (self.width, self.channels) or self.channels % dynamic_heads:
                raise ValueError("dynamic convolution weights and head groups must match history geometry")
        else:
            if weight.shape not in ((self.channels, self.width), (self.channels, 1, self.width)):
                raise ValueError("static convolution weights must match history channels and width")
            weight = weight.view(self.channels, self.width)
        self.weight, self.history, self.dynamic_heads = weight, history, dynamic_heads
        self.scale = (self.width * (2 if dynamic_heads else 1) + 1) ** -.5
        record_numeric_tensor(weight)

    def _source(self, source):
        if (source.ndim != 2 or source.shape[1] != self.channels or source.dtype != torch.bfloat16
                or source.device != self.history.device or source.stride(1) != 1 or source.stride(0) < self.channels):
            raise ValueError("causal convolution input requires BF16 token rows with contiguous channels")

    def __call__(self, source, metadata, *, dynamic=None):
        self._source(source)
        if self.dynamic_heads:
            if (dynamic is None or dynamic.shape != (source.shape[0], self.dynamic_heads, self.width)
                    or dynamic.dtype != torch.bfloat16 or dynamic.device != source.device):
                raise ValueError("dynamic convolution requires matching BF16 per-head tap coefficients")
            ds = dynamic.stride()
        else:
            if dynamic is not None:
                raise ValueError("static convolution does not consume dynamic tap coefficients")
            ds = (0, 0, 0)
        tokens, requests = source.shape[0], metadata.state_rows.numel()
        output = torch.empty_like(source, memory_format=torch.contiguous_format)
        previous = torch.empty((requests, self.width - 1, self.channels), dtype=source.dtype, device=source.device)
        torch.library.wrap_triton(_forward)[(max(tokens, requests), triton.cdiv(self.channels, 128))](
            source, self.weight, dynamic, self.history, metadata.requests, metadata.positions,
            metadata.query_start_loc, metadata.state_rows, output, previous,
            tokens, requests, self.channels, self.width, source.stride(0), *self.weight.stride(),
            *ds, self.history.stride(0), self.history.stride(1), bool(self.dynamic_heads),
            self.channels // self.dynamic_heads if self.dynamic_heads else 1, self.scale, 128,
            enable_fp_fusion=False)
        return output, previous

    def commit(self, previous, source, metadata, *, counts=None):
        """Commit raw inputs through each request's accepted count; zero counts preserve bytes."""
        self._source(source)
        if (previous.shape != (metadata.state_rows.numel(), self.width - 1, self.channels)
                or previous.dtype != torch.bfloat16 or previous.device != source.device or not previous.is_contiguous()):
            raise ValueError("convolution commit requires its matching contiguous raw-history snapshot")
        counts = metadata.lengths if counts is None else counts
        if (counts.shape != metadata.state_rows.shape or counts.device != source.device
                or counts.dtype not in (torch.int32, torch.int64) or not counts.is_contiguous()):
            raise ValueError("convolution commit counts must match the request rows")
        torch.library.wrap_triton(_commit)[(counts.numel(), triton.cdiv(self.channels, 128))](
            previous, source, metadata.starts, metadata.query_start_loc, metadata.state_rows, counts, self.history,
            source.shape[0], self.channels, self.width - 1, source.stride(0),
            self.history.stride(0), self.history.stride(1), 128, enable_fp_fusion=False)

    def forward_and_commit(self, source, metadata, *, dynamic=None):
        """Fused single-launch forward and in-place history commit for single-token-per-request decode."""
        self._source(source)
        if self.dynamic_heads:
            if (dynamic is None or dynamic.shape != (source.shape[0], self.dynamic_heads, self.width)
                    or dynamic.dtype != torch.bfloat16 or dynamic.device != source.device):
                raise ValueError("dynamic convolution requires matching BF16 per-head tap coefficients")
            ds = dynamic.stride()
        else:
            if dynamic is not None:
                raise ValueError("static convolution does not consume dynamic tap coefficients")
            ds = (0, 0, 0)
        tokens, requests = source.shape[0], metadata.state_rows.numel()
        if tokens != requests:
            raise ValueError("fused decode convolution requires one token capacity per request row")
        output = torch.empty_like(source, memory_format=torch.contiguous_format)
        torch.library.wrap_triton(_forward_and_commit_decode)[(requests, triton.cdiv(self.channels, 128))](
            source, self.weight, dynamic, self.history, metadata.starts, metadata.query_start_loc,
            metadata.state_rows, metadata.lengths, output,
            tokens, requests, self.channels, self.width, source.stride(0), *self.weight.stride(),
            *ds, self.history.stride(0), self.history.stride(1), bool(self.dynamic_heads),
            self.channels // self.dynamic_heads if self.dynamic_heads else 1, self.scale, 128,
            enable_fp_fusion=False)
        return output
