# SPDX-License-Identifier: Apache-2.0

"""Copy interleaved FP8 index keys and FP32 scale bits into global planar pages."""

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.collectives import PreparedAllReduce

PAGE_BYTES = 64 * 132
PAGE_WORDS = PAGE_BYTES // 4


@triton.jit
def _pack(Cache, Table, End, Rank, Completed, Stage, Signal,
          COLUMNS: tl.constexpr, CACHE_PAGES: tl.constexpr, WORLD: tl.constexpr,
          MAXIMUM: tl.constexpr):
    word = tl.arange(0, 32)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    local_end = tl.maximum(end - tl.load(Rank) + WORLD - 1, 0) // WORLD
    first = tl.program_id(0).to(tl.int64) * 4
    if tl.load(Completed) == 0:
        while first < local_end:
            row = first + tl.arange(0, 4).to(tl.int64)
            live = row < local_end
            page = tl.load(Table + row // 128, live & (row // 128 < COLUMNS), other=-1).to(tl.int64)
            valid = live & (page >= 0) & (2 * page + 1 < CACHE_PAGES)
            source = (2 * page + row % 128 // 64) * 2112
            destination = row // 64 * 2112
            keys = tl.load(Cache + source[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                           valid[:, None], other=0)
            scales = tl.load(Cache + source + 2048 + row % 64, valid, other=0)
            tl.store(Stage + destination[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                     keys, live[:, None])
            tl.store(Stage + destination + 2048 + row % 64, scales, live)
            first += tl.num_programs(0) * 4
    if tl.program_id(0) == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _unpack(Addresses, End, Ready, Output, Signal, WORLD: tl.constexpr, MAXIMUM: tl.constexpr):
    word = tl.arange(0, 32)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    first = tl.program_id(0).to(tl.int64) * 8
    if tl.load(Ready) == 0:
        while first < end:
            token = first + tl.arange(0, 8).to(tl.int64)
            live = token < end
            local = token // WORLD
            source = tl.load(Addresses + token % WORLD).to(tl.pointer_type(tl.int32))
            source += local // 64 * 2112
            destination = token // 64 * 2112
            keys = tl.load(source[:, None] + (local % 64)[:, None] * 32 + word[None, :],
                           live[:, None], other=0)
            scales = tl.load(source + 2048 + local % 64, live, other=0)
            tl.store(Output + destination[:, None] + (token % 64)[:, None] * 32 + word[None, :],
                     keys, live[:, None])
            tl.store(Output + destination + 2048 + token % 64, scales, live)
            first += tl.num_programs(0) * 8
    if tl.program_id(0) == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _push_direct_or_pack(Cache, Table, End, Rank, Completed, Stage, Addresses, Output, Signal,
                         COLUMNS: tl.constexpr, CACHE_PAGES: tl.constexpr, WORLD: tl.constexpr,
                         MAXIMUM: tl.constexpr, FAST_LIMIT: tl.constexpr, SLOT_PAGES: tl.constexpr):
    word = tl.arange(0, 32)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    local_end = tl.maximum(end - rank + WORLD - 1, 0) // WORLD
    target_idx = tl.program_id(1)
    if tl.load(Completed) == 0:
        if end <= FAST_LIMIT:
            peer = rank ^ (target_idx + 1)
            dst_base = tl.load(Addresses + peer).to(tl.pointer_type(tl.int32)) + rank * SLOT_PAGES * 2112
            first = tl.program_id(0).to(tl.int64) * 8
            while first < local_end:
                row = first + tl.arange(0, 8).to(tl.int64)
                live = row < local_end
                page = tl.load(Table + row // 128, live & (row // 128 < COLUMNS), other=-1).to(tl.int64)
                valid = live & (page >= 0) & (2 * page + 1 < CACHE_PAGES)
                source = (2 * page + row % 128 // 64) * 2112
                dst_page = (row // 64) * 2112
                keys = tl.load(Cache + source[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                               valid[:, None], other=0)
                scales = tl.load(Cache + source + 2048 + row % 64, valid, other=0)
                tl.store(dst_base + dst_page[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(dst_base + dst_page + 2048 + row % 64, scales, live)
                if target_idx == 0:
                    gtok = row * WORLD + rank
                    gdst = (gtok // 64) * 2112
                    tl.store(Output + gdst[:, None] + (gtok % 64)[:, None] * 32 + word[None, :],
                             keys, live[:, None])
                    tl.store(Output + gdst + 2048 + gtok % 64, scales, live)
                first += tl.num_programs(0) * 8
        elif target_idx == 0:
            first = tl.program_id(0).to(tl.int64) * 4
            while first < local_end:
                row = first + tl.arange(0, 4).to(tl.int64)
                live = row < local_end
                page = tl.load(Table + row // 128, live & (row // 128 < COLUMNS), other=-1).to(tl.int64)
                valid = live & (page >= 0) & (2 * page + 1 < CACHE_PAGES)
                source = (2 * page + row % 128 // 64) * 2112
                destination = row // 64 * 2112
                keys = tl.load(Cache + source[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                               valid[:, None], other=0)
                scales = tl.load(Cache + source + 2048 + row % 64, valid, other=0)
                tl.store(Stage + destination[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(Stage + destination + 2048 + row % 64, scales, live)
                first += tl.num_programs(0) * 4
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _unpack_direct_or_mrd(Stage, Addresses, End, Rank, Ready, Output, Signal,
                          WORLD: tl.constexpr, MAXIMUM: tl.constexpr,
                          FAST_LIMIT: tl.constexpr, SLOT_PAGES: tl.constexpr):
    word = tl.arange(0, 32)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    target_idx = tl.program_id(1)
    if tl.load(Ready) == 0:
        if end <= FAST_LIMIT:
            owner = rank ^ (target_idx + 1)
            owner_end = tl.maximum(end - owner + WORLD - 1, 0) // WORLD
            src_base = Stage + owner * SLOT_PAGES * 2112
            first = tl.program_id(0).to(tl.int64) * 8
            while first < owner_end:
                row = first + tl.arange(0, 8).to(tl.int64)
                live = row < owner_end
                spage = (row // 64) * 2112
                keys = tl.load(src_base + spage[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                               live[:, None], other=0)
                scales = tl.load(src_base + spage + 2048 + row % 64, live, other=0)
                gtok = row * WORLD + owner
                gdst = (gtok // 64) * 2112
                tl.store(Output + gdst[:, None] + (gtok % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(Output + gdst + 2048 + gtok % 64, scales, live)
                first += tl.num_programs(0) * 8
        else:
            gid = tl.program_id(0).to(tl.int64) + target_idx.to(tl.int64) * tl.num_programs(0).to(tl.int64)
            stride = tl.num_programs(0).to(tl.int64) * (WORLD - 1)
            first = gid * 8
            while first < end:
                token = first + tl.arange(0, 8).to(tl.int64)
                live = token < end
                local = token // WORLD
                source = tl.load(Addresses + token % WORLD).to(tl.pointer_type(tl.int32))
                source += local // 64 * 2112
                destination = token // 64 * 2112
                keys = tl.load(source[:, None] + (local % 64)[:, None] * 32 + word[None, :],
                               live[:, None], other=0)
                scales = tl.load(source + 2048 + local % 64, live, other=0)
                tl.store(Output + destination[:, None] + (token % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(Output + destination + 2048 + token % 64, scales, live)
                first += stride * 8
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _twohop_hop1_or_pack(Cache, Table, End, Rank, Completed, Stage, Addresses, Output, Signal,
                         COLUMNS: tl.constexpr, CACHE_PAGES: tl.constexpr, WORLD: tl.constexpr,
                         ISLANDS: tl.constexpr, MAXIMUM: tl.constexpr,
                         FAST_LIMIT: tl.constexpr, SLOT_PAGES: tl.constexpr):
    """Switch-aware 2-hop posted-write all-push (Hop 1) for planar FP8 index pages."""
    word = tl.arange(0, 32)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    local_end = tl.maximum(end - rank + WORLD - 1, 0) // WORLD
    target_idx = tl.program_id(1)
    cross_targets: tl.constexpr = ISLANDS - 1
    if tl.load(Completed) == 0:
        if end <= FAST_LIMIT:
            d = tl.where(target_idx < cross_targets, (target_idx + 1) * 4, target_idx - cross_targets + 1)
            peer = rank ^ d
            dst_base = tl.load(Addresses + peer).to(tl.pointer_type(tl.int32)) + rank * SLOT_PAGES * 2112
            first = tl.program_id(0).to(tl.int64) * 8
            while first < local_end:
                row = first + tl.arange(0, 8).to(tl.int64)
                live = row < local_end
                page = tl.load(Table + row // 128, live & (row // 128 < COLUMNS), other=-1).to(tl.int64)
                valid = live & (page >= 0) & (2 * page + 1 < CACHE_PAGES)
                source = (2 * page + row % 128 // 64) * 2112
                dst_page = (row // 64) * 2112
                keys = tl.load(Cache + source[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                               valid[:, None], other=0)
                scales = tl.load(Cache + source + 2048 + row % 64, valid, other=0)
                tl.store(dst_base + dst_page[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(dst_base + dst_page + 2048 + row % 64, scales, live)
                if target_idx == 0:
                    gtok = row * WORLD + rank
                    gdst = (gtok // 64) * 2112
                    tl.store(Output + gdst[:, None] + (gtok % 64)[:, None] * 32 + word[None, :],
                             keys, live[:, None])
                    tl.store(Output + gdst + 2048 + gtok % 64, scales, live)
                first += tl.num_programs(0) * 8
        elif target_idx == 0:
            first = tl.program_id(0).to(tl.int64) * 4
            while first < local_end:
                row = first + tl.arange(0, 4).to(tl.int64)
                live = row < local_end
                page = tl.load(Table + row // 128, live & (row // 128 < COLUMNS), other=-1).to(tl.int64)
                valid = live & (page >= 0) & (2 * page + 1 < CACHE_PAGES)
                source = (2 * page + row % 128 // 64) * 2112
                destination = row // 64 * 2112
                keys = tl.load(Cache + source[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                               valid[:, None], other=0)
                scales = tl.load(Cache + source + 2048 + row % 64, valid, other=0)
                tl.store(Stage + destination[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(Stage + destination + 2048 + row % 64, scales, live)
                first += tl.num_programs(0) * 4
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _twohop_hop2_or_unpack(Stage, Addresses, End, Rank, Ready, Output, Signal,
                           WORLD: tl.constexpr, ISLANDS: tl.constexpr, MAXIMUM: tl.constexpr,
                           FAST_LIMIT: tl.constexpr, SLOT_PAGES: tl.constexpr):
    """Hop 2 (intra-switch only): forward received cross-switch twin pages to local peers and unpack local/twin slots."""
    word = tl.arange(0, 32)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    target_idx = tl.program_id(1)
    if tl.load(Ready) == 0:
        if end <= FAST_LIMIT:
            d_local = target_idx + 1
            local_peer = rank ^ d_local
            peer_addr = tl.load(Addresses + local_peer).to(tl.pointer_type(tl.int32))
            for c in tl.static_range(1, ISLANDS):
                twin = rank ^ (c * 4)
                twin_end = tl.maximum(end - twin + WORLD - 1, 0) // WORLD
                twin_base = Stage + twin * SLOT_PAGES * 2112
                dst_base = peer_addr + twin * SLOT_PAGES * 2112
                first = tl.program_id(0).to(tl.int64) * 8
                while first < twin_end:
                    row = first + tl.arange(0, 8).to(tl.int64)
                    live = row < twin_end
                    spage = (row // 64) * 2112
                    keys = tl.load(twin_base + spage[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                                   live[:, None], other=0)
                    scales = tl.load(twin_base + spage + 2048 + row % 64, live, other=0)
                    tl.store(dst_base + spage[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                             keys, live[:, None])
                    tl.store(dst_base + spage + 2048 + row % 64, scales, live)
                    if target_idx == 0:
                        gtok = row * WORLD + twin
                        gdst = (gtok // 64) * 2112
                        tl.store(Output + gdst[:, None] + (gtok % 64)[:, None] * 32 + word[None, :],
                                 keys, live[:, None])
                        tl.store(Output + gdst + 2048 + gtok % 64, scales, live)
                    first += tl.num_programs(0) * 8
            peer_end = tl.maximum(end - local_peer + WORLD - 1, 0) // WORLD
            peer_base = Stage + local_peer * SLOT_PAGES * 2112
            first = tl.program_id(0).to(tl.int64) * 8
            while first < peer_end:
                row = first + tl.arange(0, 8).to(tl.int64)
                live = row < peer_end
                spage = (row // 64) * 2112
                keys = tl.load(peer_base + spage[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                               live[:, None], other=0)
                scales = tl.load(peer_base + spage + 2048 + row % 64, live, other=0)
                gtok = row * WORLD + local_peer
                gdst = (gtok // 64) * 2112
                tl.store(Output + gdst[:, None] + (gtok % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(Output + gdst + 2048 + gtok % 64, scales, live)
                first += tl.num_programs(0) * 8
        else:
            gid = tl.program_id(0).to(tl.int64) + target_idx.to(tl.int64) * tl.num_programs(0).to(tl.int64)
            stride = tl.num_programs(0).to(tl.int64) * 3
            first = gid * 8
            while first < end:
                token = first + tl.arange(0, 8).to(tl.int64)
                live = token < end
                local = token // WORLD
                source = tl.load(Addresses + token % WORLD).to(tl.pointer_type(tl.int32))
                source += local // 64 * 2112
                destination = token // 64 * 2112
                keys = tl.load(source[:, None] + (local % 64)[:, None] * 32 + word[None, :],
                               live[:, None], other=0)
                scales = tl.load(source + 2048 + local % 64, live, other=0)
                tl.store(Output + destination[:, None] + (token % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(Output + destination + 2048 + token % 64, scales, live)
                first += stride * 8
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _twohop_unpack_tail(Stage, End, Rank, Ready, Output, Signal,
                        WORLD: tl.constexpr, ISLANDS: tl.constexpr, MAXIMUM: tl.constexpr,
                        FAST_LIMIT: tl.constexpr, SLOT_PAGES: tl.constexpr):
    """Pure local HBM unpack of the remaining remote cross-switch pages received in Hop 2."""
    word = tl.arange(0, 32)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    target_idx = tl.program_id(1)
    if tl.load(Ready) == 0 and end <= FAST_LIMIT:
        d_local = target_idx + 1
        for c in tl.static_range(1, ISLANDS):
            owner = rank ^ (c * 4) ^ d_local
            owner_end = tl.maximum(end - owner + WORLD - 1, 0) // WORLD
            src_base = Stage + owner * SLOT_PAGES * 2112
            first = tl.program_id(0).to(tl.int64) * 8
            while first < owner_end:
                row = first + tl.arange(0, 8).to(tl.int64)
                live = row < owner_end
                spage = (row // 64) * 2112
                keys = tl.load(src_base + spage[:, None] + (row % 64)[:, None] * 32 + word[None, :],
                               live[:, None], other=0)
                scales = tl.load(src_base + spage + 2048 + row % 64, live, other=0)
                gtok = row * WORLD + owner
                gdst = (gtok // 64) * 2112
                tl.store(Output + gdst[:, None] + (gtok % 64)[:, None] * 32 + word[None, :],
                         keys, live[:, None])
                tl.store(Output + gdst + 2048 + gtok % 64, scales, live)
                first += tl.num_programs(0) * 8
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


def validate_planar_history(cache, table, end, rank, stage, addresses, output, signals):
    """Check planar storage extents, addressing metadata, and mutation aliases."""
    if any(value.ndim != 2 or value.shape[1] != PAGE_BYTES or value.dtype != torch.uint8
           for value in (cache, stage, output)):
        raise ValueError("index history requires uint8 planar pages of 64 keys and scales")
    world, maximum = addresses.numel(), output.shape[0] * 64
    if (min(cache.shape[0], stage.shape[0], maximum, world) <= 0 or cache.shape[0] % 2
            or maximum >= 2**31 or maximum % (world * 128)
            or stage.shape[0] * world != output.shape[0]):
        raise ValueError("index history requires complete interleaved 128-record model pages")
    if (table.ndim != 1 or table.dtype != torch.int32 or table.numel() * world * 128 < maximum
            or end.shape != (1,) or end.dtype != torch.int64
            or rank.shape != (1,) or rank.dtype != torch.int64
            or addresses.ndim != 1 or addresses.dtype != torch.int64):
        raise ValueError("index history requires a shared page table and int64 addressing")
    if len(signals) != 4 or any(value.shape != (1, 8) or value.dtype != torch.bfloat16 for value in signals):
        raise ValueError("index history requires four BF16 collective signal rows")
    operands = (cache, table, end, rank, stage, addresses, output, *signals)
    if any(value.device != cache.device or not value.is_contiguous() for value in operands):
        raise ValueError("index history operands must be contiguous on one device")
    if any(value.storage_offset() % 4 for value in (cache, stage, output)):
        raise ValueError("index history pages require word-aligned offsets")
    spans = [(value.untyped_storage()._cdata, value.storage_offset() * value.element_size(),
              value.numel() * value.element_size()) for value in operands]
    for index, (storage, begin, size) in enumerate(spans):
        if any(storage == other and begin < offset + extent and offset < begin + size
               for other, offset, extent in spans[:index]):
            raise ValueError("index history source, metadata, staging and outputs must not overlap")


def global_index_pages(model_pages, columns, output, *, world, cache_pages):
    """Map each shared model page's validity onto global-position index pages."""
    if (world <= 0 or cache_pages <= 0 or columns.ndim != 1 or columns.dtype != torch.int32
            or model_pages.ndim != 1 or model_pages.dtype != torch.int32
            or output.shape != (1, columns.numel()) or output.dtype != torch.int32
            or columns.numel() > model_pages.numel() * 2 * world):
        raise ValueError("global index pages require matching int32 page-table extents")
    physical = model_pages[(columns // (2 * world)).long()]
    output[0].copy_(torch.where((physical >= 0) & (physical < cache_pages), columns, -1))


class PreparedPlanarIndexHistory:
    """Publish exact owner-stored keys/scales and release staging after all reads.

    The publication and consumption collectives depend on signals written by
    their preceding copies. Serial layers share storage; a following pack reads
    the previous consumption result before overwriting any peer-visible byte.
    """

    def __init__(self, cache, table, end, rank, stage, addresses, output, signals, *,
                 runtime, session, coordinator, name):
        validate_planar_history(cache, table, end, rank, stage, addresses, output, signals)
        self.cache, self.table, self.end, self.rank = cache, table, end, rank
        self.stage, self.addresses, self.output = stage, addresses, output
        self.packed, self.ready, self.copied, self.completed = signals
        self.world, self.maximum = addresses.numel(), output.shape[0] * 64
        self.publish = PreparedAllReduce(self.packed, self.ready, runtime=runtime, session=session,
            coordinator=coordinator, name=name + ".ready")
        self.consume = PreparedAllReduce(self.copied, self.completed, runtime=runtime, session=session,
            coordinator=coordinator, name=name + ".consumed")

    def functional(self):
        self.output_layout = tuple(self.output.shape), self.output.dtype
        self.table = self.end = self.output = None
        return self

    def __call__(self, table=None, end=None):
        if table is None:
            table, end, output = self.table, self.end, self.output
        else:
            shape, dtype = self.output_layout
            output = torch.empty(shape, dtype=dtype, device=self.cache.device)
        stage_pages = self.stage.shape[0]
        fast_limit = stage_pages * 64
        if self.world in (8, 16) and stage_pages % self.world == 0:
            islands = self.world // 4
            slot_pages = stage_pages // self.world
            blocks = min(64, max(1, slot_pages * 8))
            torch.library.wrap_triton(_twohop_hop1_or_pack)[(blocks, islands + 2)](
                self.cache.view(torch.int32), table, end, self.rank, self.completed,
                self.stage.view(torch.int32), self.addresses, output.view(torch.int32), self.packed,
                COLUMNS=table.numel(), CACHE_PAGES=self.cache.shape[0], WORLD=self.world,
                ISLANDS=islands, MAXIMUM=self.maximum, FAST_LIMIT=fast_limit,
                SLOT_PAGES=slot_pages, num_warps=4)
            self.publish(self.packed)
            torch.library.wrap_triton(_twohop_hop2_or_unpack)[(blocks, 3)](
                self.stage.view(torch.int32), self.addresses, end, self.rank, self.ready,
                output.view(torch.int32), self.packed, WORLD=self.world, ISLANDS=islands,
                MAXIMUM=self.maximum, FAST_LIMIT=fast_limit, SLOT_PAGES=slot_pages, num_warps=4)
            self.publish(self.packed)
            torch.library.wrap_triton(_twohop_unpack_tail)[(blocks, 3)](
                self.stage.view(torch.int32), end, self.rank, self.ready,
                output.view(torch.int32), self.copied, WORLD=self.world, ISLANDS=islands,
                MAXIMUM=self.maximum, FAST_LIMIT=fast_limit, SLOT_PAGES=slot_pages, num_warps=4)
            self.consume(self.copied)
            return output
        if self.world == 4 and stage_pages % self.world == 0:
            slot_pages = stage_pages // self.world
            blocks = min(64, max(1, slot_pages * 8))
            torch.library.wrap_triton(_push_direct_or_pack)[(blocks, 3)](
                self.cache.view(torch.int32), table, end, self.rank, self.completed,
                self.stage.view(torch.int32), self.addresses, output.view(torch.int32), self.packed,
                COLUMNS=table.numel(), CACHE_PAGES=self.cache.shape[0], WORLD=self.world,
                MAXIMUM=self.maximum, FAST_LIMIT=fast_limit, SLOT_PAGES=slot_pages, num_warps=4)
            self.publish(self.packed)
            torch.library.wrap_triton(_unpack_direct_or_mrd)[(blocks, 3)](
                self.stage.view(torch.int32), self.addresses, end, self.rank, self.ready,
                output.view(torch.int32), self.copied, WORLD=self.world, MAXIMUM=self.maximum,
                FAST_LIMIT=fast_limit, SLOT_PAGES=slot_pages, num_warps=4)
            self.consume(self.copied)
            return output
        torch.library.wrap_triton(_pack)[(min(1024, stage_pages * 16),)](
            self.cache.view(torch.int32), table, end, self.rank, self.completed,
            self.stage.view(torch.int32), self.packed, COLUMNS=table.numel(),
            CACHE_PAGES=self.cache.shape[0], WORLD=self.world, MAXIMUM=self.maximum, num_warps=4)
        self.publish(self.packed)
        torch.library.wrap_triton(_unpack)[(min(1024, output.shape[0] * 8),)](
            self.addresses, end, self.ready, output.view(torch.int32), self.copied,
            WORLD=self.world, MAXIMUM=self.maximum, num_warps=4)
        self.consume(self.copied)
        return output
