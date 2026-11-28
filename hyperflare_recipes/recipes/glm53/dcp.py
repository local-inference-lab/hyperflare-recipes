# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Decode context parallel placement of GLM latent records and C4 index pools.

Ranks own interleaved four-token groups: global position t belongs to rank
(t // 4) % W at local index (t // 4W) * 4 + t % 4. A C4 pool is one such group,
so pool p lives on rank p % W as local pool p // W. Every rank addresses its
local records through the shared page tables: a page holds 256 local records,
which cover 256 * W global positions. Index tails, recurrent state and sampling
stay replicated; attention gathers query heads, attends over local records and
combines partial outputs by their log-sum-exp.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.collectives import PreparedVarlenGather

GROUP = 4
PAGE_RECORDS = 256
POOL_TOPK = 512


@dataclass(frozen=True)
class ContextShards:
    """One rank's share of the context. Compiled forwards are shared by every rank, so the rank
    they use is a device tensor bound per rank rather than a traced constant."""

    rank: int = 0
    world: int = 1
    index: torch.Tensor | None = None

    def __post_init__(self):
        if self.world not in (1, 2, 4) or not 0 <= self.rank < self.world:
            raise ValueError("GLM context shards require 1, 2 or 4 ranks")
        if self.world > 1 and (self.index is None or self.index.shape != (1,) or self.index.dtype != torch.int64):
            raise ValueError("GLM context shards require the rank as a one-element int64 device tensor")

    @classmethod
    def on(cls, rank, world, device):
        return cls(rank, world, torch.tensor([rank], dtype=torch.int64, device=device))

    @property
    def owner(self):
        return 0 if self.index is None else self.index

    @property
    def page_tokens(self):
        return PAGE_RECORDS * self.world

    def token_slots(self, positions, tables):
        """Local physical record of each global position on this rank, -1 where another rank owns it.

        `tables` holds the page-table row of every position; negative positions are inactive.
        """
        owner = (positions // GROUP) % self.world
        local = (positions // (GROUP * self.world)) * GROUP + positions % GROUP
        column = (local // PAGE_RECORDS).clamp(0, tables.shape[-1] - 1)
        page = tables.gather(-1, column[..., None].long()).squeeze(-1).long()
        live = (positions >= 0) & (owner == self.owner) & (page >= 0)
        return torch.where(live, page * PAGE_RECORDS + local % PAGE_RECORDS, -1)

    def local_tokens(self, lengths):
        """This rank's share of the first `lengths` global positions."""
        span = GROUP * self.world
        return (lengths // span) * GROUP + (lengths % span - self.owner * GROUP).clamp(0, GROUP)

    def local_pools(self, pools):
        """This rank's share of the first `pools` global C4 pools."""
        return pools // self.world + (pools % self.world - self.owner).clamp(0, 1)


@triton.jit
def _orderable(scores):
    bits = scores.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 2147483647, bits)


@triton.jit
def _candidate_keys(score, pool):
    # Higher scores first, then lower pool ids; missing candidates below every present one.
    keys = (_orderable(score).to(tl.int64) << 32) | (2147483647 - pool)
    return tl.where(pool >= 0, keys, -9223372036854775808)


@triton.jit
def _pack_candidates(Pools, Scores, Packed, Rank, WORLD: tl.constexpr, TOPK: tl.constexpr):
    # (score, global pool id) pairs in descending key order; ids are exact in FP32 below 2^24 pools.
    row = tl.program_id(0)
    rank = tl.load(Rank)
    index = tl.arange(0, TOPK)
    local = tl.load(Pools + row * TOPK + index)
    score = tl.load(Scores + row * TOPK + index)
    keys = tl.sort(_candidate_keys(score, tl.where(local >= 0, local * WORLD + rank, -1).to(tl.int64)),
                   descending=True)
    present = keys != -9223372036854775808
    # The orderable transform is its own inverse.
    score = _orderable((keys >> 32).to(tl.int32)).to(tl.float32, bitcast=True)
    pool = 2147483647 - (keys & 4294967295)
    tl.store(Packed + (row * TOPK + index) * 2, tl.where(present, score, -float("inf")))
    tl.store(Packed + (row * TOPK + index) * 2 + 1, tl.where(present, pool, -1).to(tl.float32))


@triton.jit
def _merge_row(Gathered, Pools, row, CANDIDATES: tl.constexpr, TOPK: tl.constexpr):
    # The TOPK highest keys of every rank's candidates; each rank's list arrives in descending key order.
    if CANDIDATES == 2 * TOPK:
        # Two descending lists: pairing the i-th of one with the (TOPK-1-i)-th of the other and keeping the
        # higher key selects exactly the top TOPK, as in one step of a bitonic merge.
        index = tl.arange(0, TOPK)
        first = row * CANDIDATES + index
        second = row * CANDIDATES + 2 * TOPK - 1 - index
        keys = tl.maximum(
            _candidate_keys(tl.load(Gathered + first * 2), tl.load(Gathered + first * 2 + 1).to(tl.int64)),
            _candidate_keys(tl.load(Gathered + second * 2), tl.load(Gathered + second * 2 + 1).to(tl.int64)))
    else:
        index = tl.arange(0, CANDIDATES)
        keys = _candidate_keys(tl.load(Gathered + (row * CANDIDATES + index) * 2),
                               tl.load(Gathered + (row * CANDIDATES + index) * 2 + 1).to(tl.int64))
        keys = tl.sort(keys, descending=True)
    # Expansion reads the first min(pools, TOPK) entries, so present pools are compacted to the front. Two ranks
    # with fewer than TOPK pools between them leave the second rank's pools at the back of the pairing.
    rank = tl.arange(0, keys.shape[0])
    present = keys != -9223372036854775808
    slot = tl.cumsum(present.to(tl.int32), 0) - 1
    tl.store(Pools + row * TOPK + rank, tl.full(keys.shape, -1, tl.int32), mask=rank < TOPK)
    tl.debug_barrier()
    tl.store(Pools + row * TOPK + slot, (2147483647 - (keys & 4294967295)).to(tl.int32), mask=present & (slot < TOPK))


@triton.jit
def _merge_candidates(Gathered, Pools, CANDIDATES: tl.constexpr, TOPK: tl.constexpr):
    _merge_row(Gathered, Pools, tl.program_id(0), CANDIDATES, TOPK)


@triton.jit
def _merge_expand_local(Gathered, Pools, Positions, Requests, Tables, Output, Counts, table_stride, columns,
                        Rank, WORLD: tl.constexpr, CANDIDATES: tl.constexpr, TOPK: tl.constexpr,
                        WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    # Merge one row's global pools, then expand them and its incomplete pool into this rank's records.
    row = tl.program_id(0)
    _merge_row(Gathered, Pools, row, CANDIDATES, TOPK)
    tl.debug_barrier()
    _expand_local_row(Pools, Positions, Requests, Tables, Output, Counts, table_stride, columns, row, Rank,
                      WORLD, TOPK, WIDTH, BLOCK)


@triton.jit
def _expand_local_row(Pools, Positions, Requests, Tables, Output, Counts, table_stride, columns, row,
                      Rank, WORLD: tl.constexpr, TOPK: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    # Selected pools and the incomplete tail as this rank's records, compacted to the front.
    rank = tl.load(Rank)
    column = tl.arange(0, BLOCK)
    length = tl.load(Positions + row).to(tl.int64) + 1
    complete = length // 4
    selected = tl.minimum(complete, TOPK)
    history = column < selected * 4
    pool = tl.load(Pools + row * TOPK + column // 4, mask=history, other=-1).to(tl.int64)
    tail = column - selected * 4
    in_tail = (tail >= 0) & (tail < length - complete * 4)
    token = tl.where(history, tl.where(pool >= 0, pool * 4 + column % 4, -1),
                     tl.where(in_tail, complete * 4 + tail, -1))
    owner = (token // 4) % WORLD
    local = (token // (4 * WORLD)) * 4 + token % 4
    request = tl.load(Requests + row).to(tl.int64)
    block = local // 256
    live = (token >= 0) & (owner == rank) & (block < columns) & (column < WIDTH)
    page = tl.load(Tables + request * table_stride + block, mask=live, other=-1).to(tl.int64)
    live &= page >= 0
    position = tl.cumsum(live.to(tl.int32), 0) - 1
    count = tl.sum(live.to(tl.int32), 0)
    tl.store(Output + row * WIDTH + column, tl.full((BLOCK,), -1, tl.int32), mask=column < WIDTH)
    tl.debug_barrier()
    tl.store(Output + row * WIDTH + position, (page * 256 + local % 256).to(tl.int32), mask=live)
    tl.store(Counts + row, count)


@triton.jit
def _expand_tokens(Pools, Positions, Output, Counts, TOPK: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    # Selected pools and the incomplete tail as global positions, compacted to the front.
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    length = tl.load(Positions + row).to(tl.int64) + 1
    complete = length // 4
    selected = tl.minimum(complete, TOPK)
    history = column < selected * 4
    pool = tl.load(Pools + row * TOPK + column // 4, mask=history, other=-1).to(tl.int64)
    tail = column - selected * 4
    in_tail = (tail >= 0) & (tail < length - complete * 4)
    token = tl.where(history, tl.where(pool >= 0, pool * 4 + column % 4, -1),
                     tl.where(in_tail, complete * 4 + tail, -1))
    live = (token >= 0) & (column < WIDTH)
    position = tl.cumsum(live.to(tl.int32), 0) - 1
    tl.store(Output + row * WIDTH + column, tl.full((BLOCK,), -1, tl.int32), mask=column < WIDTH)
    tl.debug_barrier()
    tl.store(Output + row * WIDTH + position, token.to(tl.int32), mask=live)
    tl.store(Counts + row, tl.sum(live.to(tl.int32), 0))


@triton.jit
def _pull_context(Addresses, Table, End, Slab, columns, WORLD: tl.constexpr, TOKENS: tl.constexpr,
                  WORDS: tl.constexpr, WORD_BLOCK: tl.constexpr):
    # Copy the latent records of global positions below End into the slab at their positions, reading
    # each from its owner's cache: this rank's own memory or a peer's through CUDA IPC.
    first = tl.program_id(0) * TOKENS
    end = tl.load(End).to(tl.int64)
    if first < end:
        word = tl.arange(0, WORD_BLOCK)
        for offset in tl.range(0, TOKENS):
            token = first + offset
            if token < end:
                owner = (token // 4) % WORLD
                local = (token // (4 * WORLD)) * 4 + token % 4
                page = tl.load(Table + tl.minimum(local // 256, columns - 1)).to(tl.int64)
                source = tl.load(Addresses + owner).to(tl.pointer_type(tl.int32))
                record = (page * 256 + local % 256) * WORDS
                values = tl.load(source + record + word, mask=word < WORDS)
                tl.store(Slab + token * WORDS + word, values, mask=word < WORDS)


@triton.jit
def _pack_records(Cache, Table, End, Records, Rank, columns, first, row_words, WORLD: tl.constexpr,
                  TOKENS: tl.constexpr, WORDS: tl.constexpr, WORD_BLOCK: tl.constexpr):
    # Copy this rank's records of local positions from `first`, below End's share, into its row.
    rank = tl.load(Rank).to(tl.int64)
    end = tl.load(End).to(tl.int64)
    span = 4 * WORLD
    count = (end // span) * 4 + tl.minimum(tl.maximum(end % span - rank * 4, 0), 4)
    start = tl.program_id(0) * TOKENS
    if first + start < count:
        word = tl.arange(0, WORD_BLOCK)
        for offset in tl.range(0, TOKENS):
            local = first + start + offset
            if local < count:
                page = tl.load(Table + tl.minimum(local // 256, columns - 1)).to(tl.int64)
                values = tl.load(Cache + (page * 256 + local % 256) * WORDS + word, mask=word < WORDS)
                tl.store(Records + rank * row_words + (start + offset) * WORDS + word, values, mask=word < WORDS)


@triton.jit
def _scatter_records(Records, End, Slab, first, row_words, WORLD: tl.constexpr, TOKENS: tl.constexpr,
                     WORDS: tl.constexpr, WORD_BLOCK: tl.constexpr):
    # Place every rank's records of local positions from `first` at their global positions below End.
    owner = tl.program_id(1).to(tl.int64)
    start = tl.program_id(0) * TOKENS
    end = tl.load(End).to(tl.int64)
    if (first + start) * WORLD < end:
        word = tl.arange(0, WORD_BLOCK)
        for offset in tl.range(0, TOKENS):
            local = first + start + offset
            token = (local // 4) * 4 * WORLD + owner * 4 + local % 4
            if token < end:
                values = tl.load(Records + owner * row_words + (start + offset) * WORDS + word, mask=word < WORDS)
                tl.store(Slab + token * WORDS + word, values, mask=word < WORDS)


@triton.jit
def _pack_index_pools(Cache, Table, End, Records, Rank, columns, row_words,
                      WORLD: tl.constexpr, POOLS: tl.constexpr):
    # Pack this rank's visible C4 pools (128B FP8 key + 4B FP32 scale = 33 int32 words) into its row.
    # Each 64-pool index page holds 2048 key int32 words followed by 64 scale int32 words (2112 total).
    rank = tl.load(Rank).to(tl.int64)
    total_pools = tl.load(End).to(tl.int64) // 4
    count = total_pools // WORLD + tl.minimum(tl.maximum(total_pools % WORLD - rank, 0), 1)
    start = tl.program_id(0) * POOLS
    if start < count:
        w = tl.arange(0, 32)
        for offset in tl.range(0, POOLS):
            lp = start + offset
            if lp < count:
                page = tl.load(Table + tl.minimum(lp // 64, columns - 1)).to(tl.int64)
                off = lp % 64
                page_base = page * 2112
                key_words = tl.load(Cache + page_base + off * 32 + w)
                scale_word = tl.load(Cache + page_base + 2048 + off)
                dst = Records + rank * row_words + lp * 33
                tl.store(dst + w, key_words)
                tl.store(dst + 32, scale_word)


@triton.jit
def _scatter_index_pools(Records, End, GlobalCache, row_words,
                         WORLD: tl.constexpr, POOLS: tl.constexpr):
    # Scatter every rank's gathered C4 pools into contiguous global 64-pool index pages.
    owner = tl.program_id(1).to(tl.int64)
    start = tl.program_id(0) * POOLS
    total_pools = tl.load(End).to(tl.int64) // 4
    if start * WORLD + owner < total_pools:
        w = tl.arange(0, 32)
        for offset in tl.range(0, POOLS):
            lp = start + offset
            pool = lp * WORLD + owner
            if pool < total_pools:
                src = Records + owner * row_words + lp * 33
                key_words = tl.load(src + w)
                scale_word = tl.load(src + 32)
                gpage = pool // 64
                goff = pool % 64
                gbase = gpage * 2112
                tl.store(GlobalCache + gbase + goff * 32 + w, key_words)
                tl.store(GlobalCache + gbase + 2048 + goff, scale_word)


def _launch(kernel, grid, args, **meta):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **meta)
    else:
        kernel[grid](*args, **meta)


def pack_candidates(pools, scores, packed, shards):
    """Write this rank's local top pools as (score, global pool) pairs into packed [rows, TOPK, 2] FP32."""
    _launch(_pack_candidates, (pools.shape[0],), (pools, scores, packed, shards.index),
            WORLD=shards.world, TOPK=POOL_TOPK, num_warps=8)


def merge_candidates(gathered, pools):
    """Select the global top pools from every rank's gathered [rows, W * TOPK, 2] candidates."""
    candidates = gathered.shape[1]
    _launch(_merge_candidates, (pools.shape[0],), (gathered, pools),
            CANDIDATES=candidates, TOPK=POOL_TOPK, num_warps=8 if candidates == 2 * POOL_TOPK else 16)


def merge_expand_local(gathered, pools, positions, requests, tables, output, counts, shards):
    """Select the global top pools from every rank's gathered [rows, W * TOPK, 2] candidates, then expand them
    and each row's incomplete pool into compacted local physical records."""
    width = output.shape[1]
    _launch(_merge_expand_local, (pools.shape[0],), (gathered, pools, positions, requests, tables, output, counts,
            tables.stride(0), tables.shape[1], shards.index), WORLD=shards.world, CANDIDATES=gathered.shape[1],
            TOPK=POOL_TOPK, WIDTH=width, BLOCK=triton.next_power_of_2(width), num_warps=8)


def expand_tokens(pools, positions, output, counts):
    """Expand global pools and each row's incomplete pool into compacted global positions."""
    width = output.shape[1]
    _launch(_expand_tokens, (pools.shape[0],), (pools, positions, output, counts),
            TOPK=POOL_TOPK, WIDTH=width, BLOCK=triton.next_power_of_2(width), num_warps=8)


_PULL_TOKENS = 16


def pull_context(addresses, table, end, slab, shards):
    """Gather every rank's records of the positions before `end` into the slab, indexed by global position."""
    rows = slab.shape[0]
    _launch(_pull_context, (triton.cdiv(rows, _PULL_TOKENS),), (addresses, table, end, slab, table.shape[0]),
            WORLD=shards.world, TOKENS=_PULL_TOKENS, WORDS=132, WORD_BLOCK=256, num_warps=4)


# Latent records each rank contributes per prefill gather window; transport staging scales
# with the window rather than the context.
WINDOW_RECORDS = 65536


def window_counts(end, world, first, window):
    """Bytes of records each rank owns among local positions [first, first + window) before `end`, [world] int64."""
    span = 4 * world
    ranks = torch.arange(world, device=end.device)
    owned = (end // span) * 4 + (end % span - ranks * 4).clamp(0, 4)
    return (owned - first).clamp(0, window) * 528


def pack_records(cache, table, end, records, shards, first):
    """Copy this rank's records of local positions from `first` into its row of `records` [world, window * 528].

    `cache` is this rank's latent cache as [records, 132] int32 words. Rows hold whole records in local order,
    whatever their element type.
    """
    words = records.view(torch.uint8).view(torch.int32)
    window = words.shape[1] // 132
    _launch(_pack_records, (triton.cdiv(window, _PULL_TOKENS),), (cache, table, end, words, shards.index,
            table.shape[0], first, words.stride(0)), WORLD=shards.world, TOKENS=_PULL_TOKENS, WORDS=132,
            WORD_BLOCK=256, num_warps=4)


def scatter_records(records, end, slab, shards, first):
    """Place every rank's gathered records of local positions from `first` into the slab [positions, 132] int32."""
    words = records.view(torch.uint8).view(torch.int32)
    window = words.shape[1] // 132
    _launch(_scatter_records, (triton.cdiv(window, _PULL_TOKENS), shards.world), (words, end, slab, first,
            words.stride(0)), WORLD=shards.world, TOKENS=_PULL_TOKENS, WORDS=132, WORD_BLOCK=256, num_warps=4)


_POOL_TOKENS = 16


def pool_gather_counts(end, world, max_local_pools):
    """16-byte-aligned byte counts of C4 pools each rank owns below `end`, [world] int64."""
    total_pools = end // 4
    ranks = torch.arange(world, device=end.device)
    owned = total_pools // world + (total_pools % world - ranks).clamp(0, 1)
    # Four 132-byte pools = 528 bytes = 33 * 16 bytes, satisfying variable-length gather alignment.
    aligned = ((owned + 3) // 4 * 4).clamp(0, max_local_pools)
    return aligned * 132


def pack_index_pools(cache, table, end, records, shards):
    """Copy this rank's visible C4 pools from paged index cache into its row of `records` [world, pools * 132]."""
    words = records.view(torch.uint8).view(torch.int32)
    max_local_pools = words.shape[1] // 33
    _launch(_pack_index_pools, (triton.cdiv(max_local_pools, _POOL_TOKENS),),
            (cache.view(torch.int32), table, end, words, shards.index, table.shape[0], words.stride(0)),
            WORLD=shards.world, POOLS=_POOL_TOKENS, num_warps=4)


def scatter_index_pools(records, end, global_cache, shards):
    """Scatter gathered C4 pools into contiguous global 64-pool index pages [global_pages, 64, 132]."""
    words = records.view(torch.uint8).view(torch.int32)
    max_local_pools = words.shape[1] // 33
    _launch(_scatter_index_pools, (triton.cdiv(max_local_pools, _POOL_TOKENS), shards.world),
            (words, end, global_cache.view(torch.int32), words.stride(0)),
            WORLD=shards.world, POOLS=_POOL_TOKENS, num_warps=4)


class PreparedPoolGather:
    """Gather every rank's visible C4 pooled keys into contiguous global index pages.

    Each C4 pool is 132 bytes (128 FP8 key bytes + 4 FP32 scale bytes), so gathering all
    visible pools moves 1-4 MiB at 32K-128K context instead of exchanging 64 MiB of top-k
    candidates per indexed layer.
    """

    def __init__(self, state, pool_records, pool_counts, global_cache, context, *, session, name):
        if context.records is None:
            raise ValueError("C4 pool gather requires a variable-length gather runtime")
        self.state, self.context = state, context
        self.max_local_pools = pool_records.shape[1] // 132
        self.records_shape = tuple(pool_records.shape)
        self.global_cache_shape = tuple(global_cache.shape)
        self.gather = PreparedVarlenGather(pool_records, pool_counts,
            runtime=context.records, session=session, coordinator=context.coordinator,
            name=f"{name}.gather_pools")

    def functional(self, *, session):
        self.gather.functional(session=session)
        return self

    def __call__(self, table, end):
        shards = self.context.shards
        device = table.device
        records = torch.empty(self.records_shape, dtype=torch.uint8, device=device)
        counts = pool_gather_counts(end, shards.world, self.max_local_pools)
        pack_index_pools(self.state.cache, table, end, records, shards)
        self.gather(output=records, counts=counts)
        global_cache = torch.empty(self.global_cache_shape, dtype=torch.uint8, device=device)
        scatter_index_pools(records, end, global_cache, shards)
        return global_cache


class PreparedContextGather:
    """Produce the global latent slab of one prefill layer.

    `implementation` is "ring" (pack this rank's records, gather them in windows, scatter them into the
    slab) or "pull" (read every peer's cache directly). Contexts across hosts always use the ring. On one
    host the implementation is "pull" until preparation's race binds the measured winner with `bind`.
    The `implementation` argument of `__call__` lets the race trial run either transport on one operation;
    serving never passes it.
    """

    IMPLEMENTATIONS = ("pull", "ring")

    def __init__(self, state, scratch, context, *, session, name):
        if context.remote and context.records is None:
            raise ValueError("a context across hosts gathers through its records runtime")
        self.state, self.context = state, context
        self.implementation = "ring" if context.remote else "pull"
        self.positions = scratch.slab.shape[0]
        self.window = scratch.records.shape[1] // 528
        self.windows = ()
        if context.records is not None:
            if not self.window:
                raise ValueError("context gather windows require scratch records")
            self.windows = tuple((first, PreparedVarlenGather(scratch.records, scratch.record_counts,
                runtime=context.records, session=session, coordinator=context.coordinator,
                name=f"{name}.gather_records{first}"))
                for first in range(0, self.positions // context.shards.world, self.window))
        if context.selection_tuning is not None:
            context.selection_tuning.register(self)

    @property
    def uses_records(self):
        """Whether a forward must allocate the records its ring windows pack into."""
        return self.implementation == "ring"

    def bind(self, implementation):
        """Select the transport of every later call; the ring needs the windows this gather prepared."""
        if implementation not in self.IMPLEMENTATIONS or implementation == "ring" and not self.windows:
            raise ValueError("context gather was not prepared for this implementation")
        self.implementation = implementation

    def plans(self, implementation):
        """The prepared kernel plans the transport runs: one gather per window for the ring, none for peer pull."""
        return tuple((f"window{first}", gather.plan) for first, gather in self.windows) if implementation == "ring" else ()

    def functional(self, *, session):
        for _, gather in self.windows:
            gather.functional(session=session)

    def __call__(self, s, table, end, *, implementation=None):
        shards = self.context.shards
        if (self.implementation if implementation is None else implementation) == "ring":
            cache = self.state.cache.view(torch.int32).view(-1, 132)
            for first, gather in self.windows:
                pack_records(cache, table, end, s.records, shards, first)
                s.record_counts.copy_(window_counts(end, shards.world, first, self.window))
                gather(output=s.records, counts=s.record_counts)
                scatter_records(s.records, end, s.slab.view(torch.int32), shards, first)
        else:
            # The indexer's candidate gather orders this layer's peer cache writes before these reads.
            pull_context(self.state.peers, table, end, s.slab.view(torch.int32), shards)


# Rows per collective slice: bounds exchange staging and gathered-head scratch for chunked prefill.
SLICE_ROWS = 512
# Prefill attention under context parallelism gathers the whole context into a local slab, then
# absorbs, attends with this rank's heads only, and expands in query slices that bound its scratch.
PREFILL_SLICE_ROWS = 2048


@dataclass(frozen=True)
class ContextParallel:
    """Decode context parallel resources shared by every sparse-attention layer of a rank."""

    shards: ContextShards
    heads: object
    candidates: object
    coordinator: object
    # Ranks on separate hosts exchange through one RoCEnante runtime instead of CUDA IPC.
    remote: bool = False
    # Gathers the latent-record windows of prefill context: the RoCEnante runtime across hosts, or a
    # variable-length all-gather runtime on one host when preparation races the ring against peer pull.
    records: object = None
    # Chooses between peer pull and the ring at preparation time; absent when no race runs.
    selection_tuning: object = None

    def slices(self, rows):
        return [(start, min(start + SLICE_ROWS, rows)) for start in range(0, rows, SLICE_ROWS)]
