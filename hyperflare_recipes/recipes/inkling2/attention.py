# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Whole-head attention shards with FP8 histories, relative bias and causal convolutions.

Prefill stages the complete query chunk after its preceding sliding window.
Verification stages candidate window entries without changing committed rings
or convolution histories. Accepted counts commit those histories afterward.
Full-attention candidates write private request pages. Attention uses BF16
queries and relative-bias operands, FP8 K/V and FP32 row sinks. The gated
BF16 result projects to FP32 partial sums for cross-rank reduction before the
replicated BF16 output convolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.causal_convolution import PreparedCausalConvolution
from hyperflare_recipes.components.head_gate import PreparedHeadGate
from hyperflare_recipes.components.inkling2_relative_decode_ops import (
    PreparedLogScalingTau,
    PreparedRelativeDecodeAttention,
    supports_relative_decode,
)
from hyperflare_recipes.components.inkling2_relative_prefill_ops import (
    PreparedRelativePrefillAttention,
    supports_relative_prefill,
)
from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.native_operator import is_sm120
from hyperflare_recipes.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from ignition.preparation.persistent import persistent_empty
from ignition.preparation.triton import launch_triton, prepare_triton

from .arithmetic import l2_clip, log_scaling
from .pool import KV_DTYPE, PAGE_TOKENS
from ignition.preparation.rebuild_manifest import weight_container

RELATIVE_RANK = 16
_SM120_MAX_PREFILL_BIAS_BYTES = 512 << 20


def _ring_4d(rings, ring_len, kv_heads, dim):
    """4D `[rows, ring, kv_heads, dim]` FP8 view of a local layer's ring storage."""
    if rings is None or rings.stride(1) != 1:
        raise ValueError("ring view requires contiguous per-row ring bytes")
    return rings.view(KV_DTYPE).as_strided(
        (rings.shape[0], ring_len, kv_heads, dim),
        (rings.stride(0), kv_heads * dim, dim, 1),
    )


@dataclass(frozen=True)
class Columns:
    """Column ranges of one rank's fused attention projections, by stream."""

    streams: tuple[tuple[tuple[str, int], ...], ...]

    @classmethod
    def of(cls, local, heads, kv_heads, dim, width, separate_inputs=False):
        query = (("query", heads * dim), ("gate", heads * dim))
        relative = (("relative", heads * RELATIVE_RANK), ("sink", heads))
        if local and separate_inputs:
            return cls(((*query, *relative), (("key", kv_heads * dim),), (("value", kv_heads * dim),)))
        if local:
            return cls(((*query, ("key", kv_heads * dim), ("value", kv_heads * dim), *relative),))
        return cls(((*query, *relative), (("key", kv_heads * dim), ("key_lags", kv_heads * width)),
                    (("value", kv_heads * dim), ("value_lags", kv_heads * width))))

    def widths(self):
        """Fused output widths, padded to 8 columns so that every BF16 row starts 16-byte aligned."""
        return tuple(-(-sum(width for _, width in stream) // 8) * 8 for stream in self.streams)

    def split(self, outputs):
        parts = {}
        for stream, output in zip(self.streams, outputs, strict=True):
            offset = 0
            for name, width in stream:
                parts[name] = output[:, offset:offset + width]
                offset += width
        return parts


@weight_container(('projections', 'query_clip_scale', 'relative_clip_scale', 'relative_projection', 'relative_rows', 'key_convolution', 'value_convolution', 'output'))
@dataclass(frozen=True)
class AttentionWeights:
    """One rank's attention tensors of one layer in serving layouts.

    `projections` hold one fused BF16 weight per input stream with the rows of
    `Columns`. Local layers carry static `[channels, 1, W]` K/V convolution taps;
    global layers carry `[W, channels]` static lag taps. `relative_projection` is
    the checkpoint's `[16, extent]` relative-logit projection and
    `relative_rows` its contiguous `[extent, 16]` transpose.
    """

    projections: tuple[torch.Tensor, ...]
    query_clip_scale: torch.Tensor
    relative_clip_scale: torch.Tensor
    relative_projection: torch.Tensor
    relative_rows: torch.Tensor
    key_convolution: torch.Tensor
    value_convolution: torch.Tensor
    output: torch.Tensor


def fuse_projections(placed, columns: Columns, *, local):
    """Concatenate a rank's projection rows in `Columns` order; padding rows are zero."""
    from ignition.preparation.persistent import persistent_zeros

    names = {"query": "query", "gate": "output_gate", "key": "key", "value": "value", "relative": "relative_query",
             "sink": "sink", "key_lags": "key_convolution_dynamic", "value_lags": "value_convolution_dynamic"}
    fused = []
    for stream, width in zip(columns.streams, columns.widths(), strict=True):
        hidden = placed["query"].shape[1]
        weight = persistent_zeros((width, hidden), dtype=torch.bfloat16, device=placed["query"].device)
        offset = 0
        for name, rows in stream:
            source = placed[names[name]]
            if source.shape != (rows, hidden):
                raise ValueError(f"{names[name]} has {tuple(source.shape)} rows, the fused projection requires {rows}")
            weight[offset:offset + rows].copy_(source)
            offset += rows
        fused.append(weight)
    return tuple(fused)


def attention_weights(placed, *, local, heads, kv_heads, dim, width, separate_inputs=False):
    """Serving attention tensors from one rank's placed checkpoint roles (`weights.checkpoint_layout`)."""
    from ignition.preparation.persistent import persistent_copy

    columns = Columns.of(local, heads, kv_heads, dim, width, separate_inputs)
    projection = placed["relative_projection"]
    if local:
        key, value = placed["key_convolution"], placed["value_convolution"]
    else:
        key, value = placed["key_convolution_static"], placed["value_convolution_static"]
    return AttentionWeights(fuse_projections(placed, columns, local=local), placed["query_clip_scale"],
        placed["relative_clip_scale"], projection, persistent_copy(projection.t().contiguous()), key, value,
        placed["output"])


@dataclass(frozen=True)
class LayerCache:
    """A layer's rank-local storage: K/V (paged or ring pages) and K/V convolution histories.

    Local layers also expose each continuation row's K and V rings as bytes
    (`key_rings`, `value_rings`, `[rows, ring bytes]`), which prefill reads and
    rewrites one request row at a time.
    """

    key: torch.Tensor
    value: torch.Tensor
    key_history: torch.Tensor
    value_history: torch.Tensor
    ring_row_pages: int = 0
    key_rings: torch.Tensor | None = None
    value_rings: torch.Tensor | None = None

    @classmethod
    def of(cls, pool, layer):
        if pool.text.local(layer):
            return cls(pool.ring_pages(layer, "key"), pool.ring_pages(layer, "value"),
                       pool.history(layer, "key_convolution"), pool.history(layer, "value_convolution"),
                       pool.ring_row_pages, pool.ring_rows(layer, "key"), pool.ring_rows(layer, "value"))
        return cls(pool.cache(layer, "key"), pool.cache(layer, "value"), pool.history(layer, "key_convolution"),
                   pool.history(layer, "value_convolution"))


def ring_slots(positions, rows, ring, row_pages):
    """Flat cache slots of positions in their requests' rings."""
    slot = positions.clamp_min(0).remainder(ring)
    page = rows * row_pages + slot // PAGE_TOKENS
    return torch.where(positions >= 0, page * PAGE_TOKENS + slot % PAGE_TOKENS, -1)


def rotated_ring(metadata, ring, window, row_pages):
    """Page tables and lengths that list each decoding request's window in position order.

    Returns int32 `[requests, ring pages]` pages, starting with the page that holds
    position `max(p - window + 1, 0)` of the request's current position `p`, and
    each request's key count from that page's first position through `p`.
    """
    m = metadata
    count = ring // PAGE_TOKENS
    current = (m.starts + m.lengths - 1).clamp_min(0)
    first = (current - window + 1).clamp_min(0) // PAGE_TOKENS
    columns = torch.arange(count, device=current.device)
    pages = m.state_rows[:, None] * row_pages + (first[:, None] + columns).remainder(count)
    lengths = torch.where(m.lengths > 0, current + 1 - first * PAGE_TOKENS, 0)
    return pages.int(), lengths.int()


class PreparedAttention(torch.nn.Module):
    """A rank's attention for prefill, decode or speculative verification."""

    def __init__(self, layer, weights: AttentionWeights, text, cache: LayerCache, metadata, *, mode, tokens,
                 session, workspace, name, examples=None, tuning_page_count=None):
        super().__init__()
        if mode not in ("prefill", "decode", "verify"):
            raise ValueError(f"unsupported Inkling attention mode: {mode}")
        self.layer, self.mode, self.text, self.cache = layer, mode, text, cache
        self.local = text.local(layer)
        self.heads, self.kv_heads, self.dim = weights.query_clip_scale.shape[0], cache.key.shape[2], text.head_dim
        self.window, self.ring = text.window, text.window + PAGE_TOKENS
        self.columns = Columns.of(self.local, self.heads, self.kv_heads, self.dim, text.convolution_width,
                                  separate_inputs=len(weights.projections) == 3)
        self.weights = weights
        self.key_convolution = PreparedCausalConvolution(weights.key_convolution, cache.key_history,
            dynamic_heads=0 if self.local else self.kv_heads)
        self.value_convolution = PreparedCausalConvolution(weights.value_convolution, cache.value_history,
            dynamic_heads=0 if self.local else self.kv_heads)
        device = weights.output.device
        examples = {} if examples is None else examples

        def example(role, shape, dtype=torch.bfloat16):
            from ignition.preparation.examples import declaration_tensor
            return declaration_tensor(examples, role, shape, dtype, device)

        source = example("inkling.attention.source", (tokens, text.hidden))
        self.projections = tuple(UnquantizedProjection(source, weight,
            example(f"inkling.attention.projection{index}", (tokens, width)), session=session,
            name=f"{name}.projection{index}").functional(session=session)
            for index, (weight, width) in enumerate(zip(weights.projections, self.columns.widths(), strict=True)))
        attended = example("inkling.attention.attended", (tokens, self.heads * self.dim))
        self.output_projection = UnquantizedProjection(attended, weights.output,
            example("inkling.attention.partial", (tokens, text.hidden), torch.float32),
            session=session, name=f"{name}.output").functional(session=session)
        query = example("inkling.attention.query", (tokens, self.heads, self.dim))
        output = example("inkling.attention.output", (tokens, self.heads, self.dim))
        sink = example("inkling.attention.sink", (tokens, self.heads), torch.float32)
        key = example("inkling.attention.key", (tokens, self.kv_heads, self.dim))
        value = example("inkling.attention.value", (tokens, self.kv_heads, self.dim))
        gain_key = ("inkling.attention.value_norm_gain", self.dim, device)
        gain = examples.get(gain_key)
        if gain is None:
            gain = examples[gain_key] = persistent_empty((self.dim,), torch.bfloat16, device).fill_(1)
        self.value_norm = PreparedRMSNorm(value.view(-1, self.dim), gain,
            example("inkling.attention.normalized_value", (tokens * self.kv_heads, self.dim)),
            eps=text.eps, session=session, name=f"{name}.value_norm").functional()
        projection = example("inkling.attention.projection0", (tokens, self.columns.widths()[0]))
        channels = self.heads * self.dim
        self.writer = None
        self._stage_compiled = self._retain_compiled = None
        self._stage_opts = self._retain_opts = None
        self._stage_grid = self._retain_grid = None
        self._stage_right_compiled = self._stage_right_opts = self._stage_right_grid = None
        self.sm120_staged_cap = 0
        self.sm120 = is_sm120(device)
        self.use_sm120_decode = False
        self.use_sm120_prefill = False
        self.relative_projection = None
        self.tau_op = None
        if self.sm120:
            requests = metadata.lengths.numel()
            if not self.local and session is not None:
                tau_ex = example("inkling.attention.tau", (tokens,), torch.float32)
                self.tau_op = PreparedLogScalingTau(
                    metadata.positions, tau_ex,
                    floor=text.log_scaling_floor, alpha=text.log_scaling_alpha,
                    session=session, name=f"{name}.tau",
                ).functional(session=session)
            extent = weights.relative_projection.shape[1]
            gate_ex = projection[:, channels:2 * channels].view(tokens, self.heads, self.dim)
            if mode in ("decode", "verify"):
                rel_ex = example("inkling.attention.relative_bias", (tokens, self.heads, extent))
                rel_src = example("inkling.attention.relative_source", (tokens * self.heads, RELATIVE_RANK))
                self.relative_projection = UnquantizedProjection(
                    rel_src, weights.relative_rows, rel_ex.view(tokens * self.heads, extent),
                    session=session, name=f"{name}.relative",
                ).functional(session=session)
                decode_max_q = metadata.max_query_tokens
                decode_key_cache = _ring_4d(cache.key_rings, self.ring, self.kv_heads, self.dim) if self.local else cache.key
                decode_val_cache = _ring_4d(cache.value_rings, self.ring, self.kv_heads, self.dim) if self.local else cache.value
                if (
                    1 <= decode_max_q <= 16
                    and tokens >= requests
                    and supports_relative_decode(
                        query=query,
                        key_cache=decode_key_cache,
                        value_cache=decode_val_cache,
                        is_window=self.local,
                        commit_ring=(self.local and mode == "decode"),
                        window=self.window,
                        rel_extent=extent,
                        max_q_per_req=decode_max_q,
                        new_key=key,
                        new_value=value,
                        gate=gate_ex,
                        gated=output,
                    )
                ):
                    sink_bf16_ex = example("inkling.attention.sink_bf16", (tokens, self.heads), torch.bfloat16)
                    self.use_sm120_decode = True
                    self.gate = None
                    staged_k_ex = staged_v_ex = None
                    if self.local and mode == "verify":
                        row_elems = self.kv_heads * self.dim
                        block_e = triton.next_power_of_2(row_elems)
                        row_stride = cache.key_rings.stride(0)
                        if (
                            cache.value_rings.stride(0) != row_stride
                            or cache.key_rings.stride(1) != 1
                            or cache.value_rings.stride(1) != 1
                        ):
                            raise ValueError("ring staging requires matching row strides and contiguous ring elements")
                        retain_w = min(metadata.max_query_tokens, self.ring)
                        counts_ex = example("inkling.attention.retain_counts", (requests,), torch.int32)
                        self._retain_opts = dict(
                            RING_LEN=self.ring, RING_ROW_STRIDE=row_stride, ROW_ELEMS=row_elems, MAX_TOKENS=tokens,
                            BLOCK_E=block_e, num_warps=4,
                        )
                        self._retain_grid = (requests, retain_w, 1)
                        self._retain_compiled = prepare_triton(
                            _retain_kv_kernel,
                            (cache.key_rings.view(KV_DTYPE), cache.value_rings.view(KV_DTYPE), key, value,
                             metadata.state_rows, metadata.starts, counts_ex, metadata.query_start_loc),
                            self._retain_grid, **self._retain_opts,
                        ).compiled
                        staged_k_ex = example(
                            "inkling.attention.verify_staged_key",
                            (requests, self.ring + decode_max_q, self.kv_heads, self.dim),
                            KV_DTYPE,
                        )
                        staged_v_ex = example(
                            "inkling.attention.verify_staged_value",
                            (requests, self.ring + decode_max_q, self.kv_heads, self.dim),
                            KV_DTYPE,
                        )
                    self.attention = PreparedRelativeDecodeAttention(
                        query=query,
                        new_key=key,
                        new_value=value,
                        key_cache=decode_key_cache,
                        value_cache=decode_val_cache,
                        metadata=metadata,
                        relative_logits=rel_ex,
                        gate=gate_ex,
                        sink=sink_bf16_ex,
                        gated=output,
                        is_window=self.local,
                        commit_ring=(self.local and mode == "decode"),
                        window=self.window,
                        softmax_scale=self.dim ** -0.5,
                        max_q_per_req=decode_max_q,
                        staged_key=staged_k_ex,
                        staged_value=staged_v_ex,
                        session=session,
                        workspace=workspace,
                        name=f"{name}.attention",
                    ).functional(session=session)
                    return
            elif mode == "prefill" and tokens * self.heads * extent * 2 <= _SM120_MAX_PREFILL_BIAS_BYTES:
                from hyperflare_recipes.components.inkling2_relative_prefill_ops import _relative_prefill_module

                if _relative_prefill_module() is not None:
                    rel_ex = example("inkling.attention.relative_bias", (tokens, self.heads, extent))
                    sink_bf16_ex = example("inkling.attention.sink_bf16", (tokens, self.heads), torch.bfloat16)
                    if self.local:
                        staged_cap = -(-(self.window + max(metadata.max_query_tokens, tokens)) // 64) * 64
                        prefill_key_cache = example(
                            "inkling.attention.sm120_staged_key",
                            (requests, staged_cap, self.kv_heads, self.dim),
                            KV_DTYPE,
                        )
                        prefill_val_cache = example(
                            "inkling.attention.sm120_staged_value",
                            (requests, staged_cap, self.kv_heads, self.dim),
                            KV_DTYPE,
                        )
                        prefill_window = self.window
                    else:
                        staged_cap = 0
                        prefill_key_cache, prefill_val_cache = cache.key, cache.value
                        prefill_window = -1
                    if supports_relative_prefill(
                        query=query,
                        key_cache=prefill_key_cache,
                        value_cache=prefill_val_cache,
                        relative_logits=rel_ex,
                        gate=gate_ex,
                        sink=sink_bf16_ex,
                        gated=output,
                        window=prefill_window,
                    ):
                        self.use_sm120_prefill = True
                        self.gate = None
                        rel_src = example("inkling.attention.relative_source", (tokens * self.heads, RELATIVE_RANK))
                        self.relative_projection = UnquantizedProjection(
                            rel_src, weights.relative_rows, rel_ex.view(tokens * self.heads, extent),
                            session=session, name=f"{name}.relative",
                        ).functional(session=session)
                        lengths = example("inkling.attention.lengths", (requests,), torch.int32)
                        if self.local:
                            self.sm120_staged_cap = staged_cap
                            row_elems = self.kv_heads * self.dim
                            block_e = triton.next_power_of_2(row_elems)
                            row_stride = cache.key_rings.stride(0)
                            if (
                                cache.value_rings.stride(0) != row_stride
                                or cache.key_rings.stride(1) != 1
                                or cache.value_rings.stride(1) != 1
                            ):
                                raise ValueError("ring staging requires matching row strides and contiguous ring elements")
                            self._stage_right_opts = dict(
                                RING_LEN=self.ring, RING_ROW_STRIDE=row_stride, WINDOW=self.window, CAPACITY=staged_cap,
                                ROW_ELEMS=row_elems, MAX_TOKENS=tokens, BLOCK_E=block_e, num_warps=4,
                            )
                            self._stage_right_grid = (requests, staged_cap, 1)
                            self._stage_right_compiled = prepare_triton(
                                _stage_kv_right_aligned_kernel,
                                (cache.key_rings.view(KV_DTYPE), cache.value_rings.view(KV_DTYPE), key, value,
                                 metadata.state_rows, metadata.starts, metadata.lengths, metadata.query_start_loc,
                                 prefill_key_cache, prefill_val_cache, lengths),
                                self._stage_right_grid, **self._stage_right_opts,
                            ).compiled
                            retain_w = min(metadata.max_query_tokens, self.ring)
                            counts_ex = example("inkling.attention.retain_counts", (requests,), torch.int32)
                            self._retain_opts = dict(
                                RING_LEN=self.ring, RING_ROW_STRIDE=row_stride, ROW_ELEMS=row_elems, MAX_TOKENS=tokens,
                                BLOCK_E=block_e, num_warps=4,
                            )
                            self._retain_grid = (requests, retain_w, 1)
                            self._retain_compiled = prepare_triton(
                                _retain_kv_kernel,
                                (cache.key_rings.view(KV_DTYPE), cache.value_rings.view(KV_DTYPE), key, value,
                                 metadata.state_rows, metadata.starts, counts_ex, metadata.query_start_loc),
                                self._retain_grid, **self._retain_opts,
                            ).compiled
                        else:
                            slots = example("inkling.attention.slots", (tokens,), torch.int64)
                            self.writer = PreparedPagedKVWriter(
                                key, value, cache.key, cache.value, slots,
                                session=session, name=f"{name}.cache_write",
                            ).functional()
                        self.attention = PreparedRelativePrefillAttention(
                            query=query,
                            key_cache=prefill_key_cache,
                            value_cache=prefill_val_cache,
                            page_table=None if self.local else metadata.page_table,
                            lengths=lengths,
                            query_start_loc=metadata.query_start_loc,
                            relative_logits=rel_ex,
                            gate=gate_ex,
                            sink=sink_bf16_ex,
                            gated=output,
                            window=prefill_window,
                            softmax_scale=self.dim ** -0.5,
                            session=session,
                            workspace=workspace,
                            name=f"{name}.attention",
                        ).functional(
                            session=session,
                            k_cache=None if self.local else cache.key,
                            v_cache=None if self.local else cache.value,
                        )
                        return
        self.gate = PreparedHeadGate(attended.view(tokens, channels, 1), projection[:, channels:2 * channels],
            session=session, name=f"{name}.gate").functional(session=session)
        slots = example("inkling.attention.slots", (tokens,), torch.int64)
        requests = metadata.lengths.numel()
        if self.local and mode in ("prefill", "verify"):
            self.staged_pages = -(-(self.window - 1 + metadata.max_query_tokens) // PAGE_TOKENS)
            count = requests * self.staged_pages
            cache_key = example("inkling.attention.staged_key", (count, PAGE_TOKENS, self.kv_heads, self.dim), KV_DTYPE)
            cache_value = example("inkling.attention.staged_value", (count, PAGE_TOKENS, self.kv_heads, self.dim), KV_DTYPE)
            pages = torch.arange(count, dtype=torch.int32, device=device).view(requests, self.staged_pages)
            self.staged_table = pages
            cap = self.staged_pages * PAGE_TOKENS
            row_elems = self.kv_heads * self.dim
            block_e = triton.next_power_of_2(row_elems)
            out_len = example("inkling.attention.staged_lengths", (requests,), torch.int32)
            row_stride = cache.key_rings.stride(0)
            if cache.value_rings.stride(0) != row_stride or cache.key_rings.stride(1) != 1 or cache.value_rings.stride(1) != 1:
                raise ValueError("ring staging requires matching row strides and contiguous ring elements")
            self._stage_opts = dict(
                RING_LEN=self.ring, RING_ROW_STRIDE=row_stride, WINDOW=self.window, CAPACITY=cap,
                ROW_ELEMS=row_elems, MAX_TOKENS=tokens, BLOCK_E=block_e, num_warps=4,
            )
            self._stage_grid = (requests, cap, 1)
            self._stage_compiled = prepare_triton(
                _stage_kv_kernel,
                (cache.key_rings.view(KV_DTYPE), cache.value_rings.view(KV_DTYPE), key, value,
                 metadata.state_rows, metadata.starts, metadata.lengths, metadata.query_start_loc,
                 cache_key, cache_value, out_len),
                self._stage_grid, **self._stage_opts,
            ).compiled
            retain_w = min(metadata.max_query_tokens, self.ring)
            counts_ex = example("inkling.attention.retain_counts", (requests,), torch.int32)
            self._retain_opts = dict(
                RING_LEN=self.ring, RING_ROW_STRIDE=row_stride, ROW_ELEMS=row_elems, MAX_TOKENS=tokens,
                BLOCK_E=block_e, num_warps=4,
            )
            self._retain_grid = (requests, retain_w, 1)
            self._retain_compiled = prepare_triton(
                _retain_kv_kernel,
                (cache.key_rings.view(KV_DTYPE), cache.value_rings.view(KV_DTYPE), key, value,
                 metadata.state_rows, metadata.starts, counts_ex, metadata.query_start_loc),
                self._retain_grid, **self._retain_opts,
            ).compiled
        else:
            cache_key, cache_value = cache.key, cache.value
            pages = (example("inkling.attention.ring_table", (requests, self.ring // PAGE_TOKENS), torch.int32)
                     if self.local else metadata.page_table)
            self.writer = PreparedPagedKVWriter(key, value, cache_key, cache_value, slots,
                session=session, name=f"{name}.cache_write").functional()
        lengths = example("inkling.attention.lengths", (requests,), torch.int32)
        if mode == "prefill":
            bias_arguments = dict(relative_bias_factors=example("inkling.attention.relative_factor",
                (tokens, self.heads, RELATIVE_RANK)), relative_bias_projection=weights.relative_rows)
        else:
            bias_arguments = dict(relative_bias=example("inkling.attention.relative_bias",
                (tokens, self.heads, weights.relative_projection.shape[1])))
        self.attention = PreparedPagedAttention(query, cache_key, cache_value, pages, lengths,
            metadata.query_start_loc, output, sink=sink, mode="extend" if mode == "prefill" else mode,
            width=metadata.max_query_tokens, window=self.window - 1 if self.local else -1,
            tuning_page_count=min(tuning_page_count or cache_key.shape[0], cache_key.shape[0]),
            session=session, workspace=workspace, name=f"{name}.attention", **bias_arguments)
        self.attention = self.attention.functional(session=session,
            k_cache=None if self.local and mode in ("prefill", "verify") else cache.key,
            v_cache=None if self.local and mode in ("prefill", "verify") else cache.value,
            cache_seqlens=None, page_table=None, cu_seqlens_q=None, sink=None,
            **({"relative_bias_factors": None, "relative_bias_projection": None} if mode == "prefill"
               else {"relative_bias": None}))

    def project(self, streams, metadata):
        """Fused projections, convolved and clipped K/V and queries, sinks and gates of `streams`."""
        t, m = self.text, metadata
        inputs = streams if len(streams) == len(self.projections) else streams * len(self.projections)
        parts = self.columns.split([projection(source) for projection, source in zip(self.projections, inputs,
                                                                                    strict=True)])
        tokens = m.positions.shape[0]
        heads, kv_heads, dim = self.heads, self.kv_heads, self.dim
        key_raw, value_raw = parts["key"], parts["value"]
        dynamic_shape = (tokens, kv_heads, t.convolution_width)
        key, key_previous = self.key_convolution(key_raw, m,
            dynamic=None if self.local else parts["key_lags"].reshape(dynamic_shape))
        value, value_previous = self.value_convolution(value_raw, m,
            dynamic=None if self.local else parts["value_lags"].reshape(dynamic_shape))
        if self.local:
            tau = None
        elif self.tau_op is not None:
            tau = self.tau_op(m.positions)
        else:
            tau = log_scaling(m.positions, t.log_scaling_floor, t.log_scaling_alpha)
        if self.mode != "verify":
            self.key_convolution.commit(key_previous, key_raw, m)
            self.value_convolution.commit(value_previous, value_raw, m)
        query = l2_clip(parts["query"].reshape(tokens, heads, dim), t.qk_clip, scale=self.weights.query_clip_scale,
                        tau=tau)
        key = l2_clip(key.reshape(tokens, kv_heads, dim), t.qk_clip)
        value = self.value_norm(value.view(-1, dim)).view(tokens, kv_heads, dim)
        relative = l2_clip(parts["relative"].reshape(tokens, heads, RELATIVE_RANK), t.relative_clip,
                           scale=self.weights.relative_clip_scale, tau=tau)
        sink = (
            parts["sink"]
            if (self.use_sm120_decode or self.use_sm120_prefill)
            else parts["sink"].to(dtype=torch.float32, memory_format=torch.contiguous_format)
        )
        return dict(query=query, key=key, value=value, relative=relative, sink=sink,
                    gate=parts["gate"].reshape(tokens, heads, dim), key_raw=key_raw, value_raw=value_raw,
                    key_previous=key_previous, value_previous=value_previous)

    def attend(self, streams, metadata):
        """The gated attention `[tokens, heads * dim]` and its partial output projection."""
        m = metadata
        values = self.project(streams, m)
        if self.mode == "prefill":
            attended = self._prefill(values, m)
        elif self.mode == "verify":
            attended = self._verify(values, m)
        else:
            attended = self._decode(values, m)
        if self.gate is not None:
            attended = self.gate(attended.view(attended.shape[0], -1, 1),
                values["gate"].reshape(attended.shape[0], -1)).flatten(1)
        else:
            attended = attended.flatten(1)
        return attended, self.output_projection(attended), values

    def forward(self, streams, metadata):
        return self.attend(streams, metadata)[1]

    def _stage_kv(self, key, value, m):
        requests = m.lengths.numel()
        count = requests * self.staged_pages
        staged_key = torch.empty((count, PAGE_TOKENS, self.kv_heads, self.dim), dtype=KV_DTYPE, device=key.device)
        staged_value = torch.empty((count, PAGE_TOKENS, self.kv_heads, self.dim), dtype=KV_DTYPE, device=key.device)
        lengths = torch.empty((requests,), dtype=torch.int32, device=key.device)
        launch_triton(
            _stage_kv_kernel, self._stage_compiled,
            (self.cache.key_rings.view(KV_DTYPE), self.cache.value_rings.view(KV_DTYPE), key, value,
             m.state_rows, m.starts, m.lengths, m.query_start_loc, staged_key, staged_value, lengths),
            self._stage_grid, **self._stage_opts,
        )
        return staged_key, staged_value, lengths

    def _stage_kv_right_aligned(self, key, value, m):
        requests = m.lengths.numel()
        staged_key = torch.empty((requests, self.sm120_staged_cap, self.kv_heads, self.dim), dtype=KV_DTYPE, device=key.device)
        staged_value = torch.empty((requests, self.sm120_staged_cap, self.kv_heads, self.dim), dtype=KV_DTYPE, device=key.device)
        lengths = torch.empty((requests,), dtype=torch.int32, device=key.device)
        launch_triton(
            _stage_kv_right_aligned_kernel, self._stage_right_compiled,
            (self.cache.key_rings.view(KV_DTYPE), self.cache.value_rings.view(KV_DTYPE), key, value,
             m.state_rows, m.starts, m.lengths, m.query_start_loc, staged_key, staged_value, lengths),
            self._stage_right_grid, **self._stage_right_opts,
        )
        return staged_key, staged_value, lengths

    def _retain_kv(self, key, value, counts, m):
        launch_triton(
            _retain_kv_kernel, self._retain_compiled,
            (self.cache.key_rings.view(KV_DTYPE), self.cache.value_rings.view(KV_DTYPE), key, value,
             m.state_rows, m.starts, counts, m.query_start_loc),
            self._retain_grid, **self._retain_opts,
        )

    def _relative_bias(self, relative):
        tokens = relative.shape[0]
        if self.relative_projection is not None:
            return self.relative_projection(relative.flatten(0, 1)).view(tokens, self.heads, -1)
        return (relative.reshape(tokens * self.heads, RELATIVE_RANK) @ self.weights.relative_projection).view(
            tokens, self.heads, -1
        )

    def _prefill(self, v, m):
        key, value = v["key"], v["value"]
        if self.use_sm120_prefill:
            bias = self._relative_bias(v["relative"])
            if self.local:
                staged_key, staged_value, staged = self._stage_kv_right_aligned(key, value, m)
                self._retain_kv(key, value, m.lengths, m)
                return self.attention(
                    v["query"], key_cache=staged_key, value_cache=staged_value,
                    lengths=staged, cu_seqlens_q=m.query_start_loc,
                    relative_logits=bias, gate=v["gate"], sink=v["sink"],
                )
            self.writer(key, value, _page_slots(m))
            lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int()
            return self.attention(
                v["query"], page_table=m.page_table, lengths=lengths,
                cu_seqlens_q=m.query_start_loc, relative_logits=bias,
                gate=v["gate"], sink=v["sink"],
            )
        if self.local:
            staged_key, staged_value, staged = self._stage_kv(key, value, m)
            self._retain_kv(key, value, m.lengths, m)
            return self.attention(v["query"], sink=v["sink"], relative_bias_factors=v["relative"],
                                  relative_bias_projection=self.weights.relative_rows,
                                  k_cache=staged_key, v_cache=staged_value, cache_seqlens=staged,
                                  page_table=self.staged_table, cu_seqlens_q=m.query_start_loc)
        self.writer(key, value, _page_slots(m))
        lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int()
        return self.attention(v["query"], sink=v["sink"], relative_bias_factors=v["relative"],
                                  relative_bias_projection=self.weights.relative_rows,
                              cache_seqlens=lengths, page_table=m.page_table, cu_seqlens_q=m.query_start_loc)

    def _decode(self, v, m):
        if self.use_sm120_decode:
            bias = self._relative_bias(v["relative"])
            return self.attention(
                v["query"], new_key=v["key"], new_value=v["value"],
                page_table=None if self.local else m.page_table,
                state_rows=m.state_rows if self.local else None,
                starts=m.starts, cu_seqlens_q=m.query_start_loc,
                relative_logits=bias, gate=v["gate"], sink=v["sink"],
            )
        if self.local:
            slots = ring_slots(m.positions, m.state_rows[m.requests], self.ring, self.cache.ring_row_pages)
            pages, lengths = rotated_ring(m, self.ring, self.window, self.cache.ring_row_pages)
        else:
            slots = _page_slots(m)
            pages = m.page_table
            lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int()
        self.writer(v["key"], v["value"], slots)
        bias = self._relative_bias(v["relative"])
        attended = self.attention(v["query"], sink=v["sink"], relative_bias=bias,
                                  cache_seqlens=lengths, page_table=pages, cu_seqlens_q=m.query_start_loc)
        return attended

    def _verify(self, v, m):
        if self.use_sm120_decode:
            bias = self._relative_bias(v["relative"])
            return self.attention(
                v["query"], new_key=v["key"], new_value=v["value"],
                page_table=None if self.local else m.page_table,
                state_rows=m.state_rows if self.local else None,
                starts=m.starts, cu_seqlens_q=m.query_start_loc,
                relative_logits=bias, gate=v["gate"], sink=v["sink"],
            )
        if self.local:
            cache_key, cache_value, lengths = self._stage_kv(v["key"], v["value"], m)
            pages = self.staged_table
            kwargs = dict(k_cache=cache_key, v_cache=cache_value)
        else:
            self.writer(v["key"], v["value"], _page_slots(m))
            lengths, pages = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int(), m.page_table
            kwargs = {}
        bias = self._relative_bias(v["relative"])
        return self.attention(v["query"], sink=v["sink"], relative_bias=bias,
            cache_seqlens=lengths, page_table=pages, cu_seqlens_q=m.query_start_loc, **kwargs)

    def commit(self, values, counts, metadata):
        self.key_convolution.commit(values["key_previous"], values["key_raw"], metadata, counts=counts)
        self.value_convolution.commit(values["value_previous"], values["value_raw"], metadata, counts=counts)
        if self.local:
            self._retain_kv(values["key"], values["value"], counts.int(), metadata)


@triton.jit
def _stage_kv_kernel(
    KeyRings, ValRings, KeyVals, ValVals,
    StateRows, Starts, Lengths, QueryStartLoc,
    OutKey, OutVal, OutLengths,
    RING_LEN: tl.constexpr, RING_ROW_STRIDE: tl.constexpr, WINDOW: tl.constexpr, CAPACITY: tl.constexpr,
    ROW_ELEMS: tl.constexpr, MAX_TOKENS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Fused K/V sliding-window staging: copies raw FP8 ring bytes directly without BF16 round-trip."""
    b = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1).to(tl.int64)
    start = tl.load(Starts + b).to(tl.int64)
    length = tl.load(Lengths + b).to(tl.int64)
    first = tl.maximum(start - WINDOW + 1, 0)
    if j == 0:
        out_len = tl.where(length > 0, start - first + length, 0).to(tl.int32)
        tl.store(OutLengths + b, out_len)

    pos = first + j
    out_offset = (b * CAPACITY + j) * ROW_ELEMS
    cols = tl.arange(0, BLOCK_E)
    mask = cols < ROW_ELEMS

    if length <= 0 or pos >= start + length:
        z = tl.zeros((BLOCK_E,), dtype=tl.float8e4nv)
        tl.store(OutKey + out_offset + cols, z, mask=mask)
        tl.store(OutVal + out_offset + cols, z, mask=mask)
    elif pos < start:
        srow = tl.load(StateRows + b).to(tl.int64)
        slot = pos % RING_LEN
        ring_offset = srow * RING_ROW_STRIDE + slot * ROW_ELEMS
        k = tl.load(KeyRings + ring_offset + cols, mask=mask)
        v = tl.load(ValRings + ring_offset + cols, mask=mask)
        tl.store(OutKey + out_offset + cols, k, mask=mask)
        tl.store(OutVal + out_offset + cols, v, mask=mask)
    else:
        qstart = tl.load(QueryStartLoc + b).to(tl.int64)
        idx = tl.minimum(tl.maximum(qstart + pos - start, 0), MAX_TOKENS - 1)
        val_offset = idx * ROW_ELEMS
        k_bf = tl.load(KeyVals + val_offset + cols, mask=mask, other=0.0).to(tl.float32)
        v_bf = tl.load(ValVals + val_offset + cols, mask=mask, other=0.0).to(tl.float32)
        k_fp8 = tl.minimum(tl.maximum(k_bf, -448.0), 448.0).to(tl.float8e4nv)
        v_fp8 = tl.minimum(tl.maximum(v_bf, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(OutKey + out_offset + cols, k_fp8, mask=mask)
        tl.store(OutVal + out_offset + cols, v_fp8, mask=mask)


@triton.jit
def _stage_kv_right_aligned_kernel(
    KeyRings, ValRings, KeyVals, ValVals,
    StateRows, Starts, Lengths, QueryStartLoc,
    OutKey, OutVal, OutLengths,
    RING_LEN: tl.constexpr, RING_ROW_STRIDE: tl.constexpr, WINDOW: tl.constexpr, CAPACITY: tl.constexpr,
    ROW_ELEMS: tl.constexpr, MAX_TOKENS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Fused K/V right-aligned sliding-window staging for SM120 relative prefill attention."""
    b = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1).to(tl.int64)
    start = tl.load(Starts + b).to(tl.int64)
    length = tl.load(Lengths + b).to(tl.int64)
    if j == 0:
        hist = tl.minimum(start, WINDOW)
        out_len = tl.where(length > 0, hist + length, 0).to(tl.int32)
        tl.store(OutLengths + b, out_len)

    pos = start - WINDOW + j
    out_offset = (b * CAPACITY + j) * ROW_ELEMS
    cols = tl.arange(0, BLOCK_E)
    mask = cols < ROW_ELEMS

    if length <= 0 or pos < 0 or pos >= start + length:
        z = tl.zeros((BLOCK_E,), dtype=tl.float8e4nv)
        tl.store(OutKey + out_offset + cols, z, mask=mask)
        tl.store(OutVal + out_offset + cols, z, mask=mask)
    elif pos < start:
        srow = tl.load(StateRows + b).to(tl.int64)
        slot = pos % RING_LEN
        ring_offset = srow * RING_ROW_STRIDE + slot * ROW_ELEMS
        k = tl.load(KeyRings + ring_offset + cols, mask=mask)
        v = tl.load(ValRings + ring_offset + cols, mask=mask)
        tl.store(OutKey + out_offset + cols, k, mask=mask)
        tl.store(OutVal + out_offset + cols, v, mask=mask)
    else:
        qstart = tl.load(QueryStartLoc + b).to(tl.int64)
        idx = tl.minimum(tl.maximum(qstart + pos - start, 0), MAX_TOKENS - 1)
        val_offset = idx * ROW_ELEMS
        k_bf = tl.load(KeyVals + val_offset + cols, mask=mask, other=0.0).to(tl.float32)
        v_bf = tl.load(ValVals + val_offset + cols, mask=mask, other=0.0).to(tl.float32)
        k_fp8 = tl.minimum(tl.maximum(k_bf, -448.0), 448.0).to(tl.float8e4nv)
        v_fp8 = tl.minimum(tl.maximum(v_bf, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(OutKey + out_offset + cols, k_fp8, mask=mask)
        tl.store(OutVal + out_offset + cols, v_fp8, mask=mask)


@triton.jit
def _retain_kv_kernel(
    KeyRings, ValRings, KeyVals, ValVals,
    StateRows, Starts, Counts, QueryStartLoc,
    RING_LEN: tl.constexpr, RING_ROW_STRIDE: tl.constexpr, ROW_ELEMS: tl.constexpr, MAX_TOKENS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Fused K/V sliding-window ring commit: writes only the newly accepted tail tokens in place."""
    b = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1).to(tl.int64)
    count = tl.load(Counts + b).to(tl.int64)
    offset = tl.maximum(count - RING_LEN, 0)
    k_idx = offset + j
    if k_idx >= count:
        return
    start = tl.load(Starts + b).to(tl.int64)
    srow = tl.load(StateRows + b).to(tl.int64)
    qstart = tl.load(QueryStartLoc + b).to(tl.int64)
    pos = start + k_idx
    slot = pos % RING_LEN
    tok_idx = tl.minimum(tl.maximum(qstart + k_idx, 0), MAX_TOKENS - 1)

    cols = tl.arange(0, BLOCK_E)
    mask = cols < ROW_ELEMS
    k_bf = tl.load(KeyVals + tok_idx * ROW_ELEMS + cols, mask=mask, other=0.0).to(tl.float32)
    v_bf = tl.load(ValVals + tok_idx * ROW_ELEMS + cols, mask=mask, other=0.0).to(tl.float32)
    k_fp8 = tl.minimum(tl.maximum(k_bf, -448.0), 448.0).to(tl.float8e4nv)
    v_fp8 = tl.minimum(tl.maximum(v_bf, -448.0), 448.0).to(tl.float8e4nv)
    ring_offset = srow * RING_ROW_STRIDE + slot * ROW_ELEMS
    tl.store(KeyRings + ring_offset + cols, k_fp8, mask=mask)
    tl.store(ValRings + ring_offset + cols, v_fp8, mask=mask)


def _page_slots(m):
    page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
    return torch.where((m.positions >= 0) & (page >= 0), page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1)
