# SPDX-License-Identifier: Apache-2.0

"""Copy interleaved paged records through fixed caller-owned peer buffers."""

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.collectives import PreparedAllReduce


@triton.jit
def _pack(Cache, Table, End, Rank, Completed, Stage, Signal,
          COLUMNS: tl.constexpr, CACHE_PAGES: tl.constexpr, PAGE: tl.constexpr,
          WORLD: tl.constexpr, MAXIMUM: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    local_end = tl.maximum(end - tl.load(Rank) + WORLD - 1, 0) // WORLD
    first = tl.program_id(0).to(tl.int64) * 4
    if tl.load(Completed) == 0:
        while first < local_end:
            row = (first + tl.arange(0, 4)).to(tl.int64)
            live = row < local_end
            column = row // PAGE
            page = tl.load(Table + column, mask=live & (column < COLUMNS), other=-1).to(tl.int64)
            valid = live & (page >= 0) & (page < CACHE_PAGES)
            data = tl.load(Cache + (page[:, None] * PAGE + row[:, None] % PAGE) * WORDS + words[None, :],
                           mask=valid[:, None] & (words[None, :] < WORDS), other=0)
            tl.store(Stage + row[:, None] * WORDS + words[None, :], data,
                     mask=live[:, None] & (words[None, :] < WORDS))
            first += tl.num_programs(0) * 4
    if tl.program_id(0) == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _unpack(Addresses, End, Ready, Output, Signal,
            WORLD: tl.constexpr, MAXIMUM: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    first = tl.program_id(0).to(tl.int64) * 8
    if tl.load(Ready) == 0:
        while first < end:
            row = (first + tl.arange(0, 8)).to(tl.int64)
            live = row < end
            source = tl.load(Addresses + row % WORLD).to(tl.pointer_type(tl.int32))
            data = tl.load(source[:, None] + (row[:, None] // WORLD) * WORDS + words[None, :],
                           mask=live[:, None] & (words[None, :] < WORDS), other=0)
            tl.store(Output + row[:, None] * WORDS + words[None, :], data,
                     mask=live[:, None] & (words[None, :] < WORDS))
            first += tl.num_programs(0) * 8
    if tl.program_id(0) == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _push_direct_or_pack(Cache, Table, End, Rank, Completed, Stage, Addresses, Output, Signal,
                         COLUMNS: tl.constexpr, CACHE_PAGES: tl.constexpr, PAGE: tl.constexpr,
                         WORLD: tl.constexpr, MAXIMUM: tl.constexpr, FAST_LIMIT: tl.constexpr,
                         SLOT_ROWS: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    """Single-switch (DCP4) posted-write all-push for active lengths <= FAST_LIMIT, else local pack."""
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    local_end = tl.maximum(end - rank + WORLD - 1, 0) // WORLD
    target_idx = tl.program_id(1)
    if tl.load(Completed) == 0:
        if end <= FAST_LIMIT:
            peer = rank ^ (target_idx + 1)
            dst_base = tl.load(Addresses + peer).to(tl.pointer_type(tl.int32))
            slot_off = rank * SLOT_ROWS * WORDS
            first = tl.program_id(0).to(tl.int64) * 4
            while first < local_end:
                row = (first + tl.arange(0, 4)).to(tl.int64)
                live = row < local_end
                column = row // PAGE
                page = tl.load(Table + column, mask=live & (column < COLUMNS), other=-1).to(tl.int64)
                valid = live & (page >= 0) & (page < CACHE_PAGES)
                data = tl.load(Cache + (page[:, None] * PAGE + row[:, None] % PAGE) * WORDS + words[None, :],
                               mask=valid[:, None] & (words[None, :] < WORDS), other=0)
                tl.store(dst_base + slot_off + row[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                if target_idx == 0:
                    grow = row * WORLD + rank
                    tl.store(Output + grow[:, None] * WORDS + words[None, :], data,
                             mask=live[:, None] & (words[None, :] < WORDS))
                first += tl.num_programs(0) * 4
        elif target_idx == 0:
            first = tl.program_id(0).to(tl.int64) * 4
            while first < local_end:
                row = (first + tl.arange(0, 4)).to(tl.int64)
                live = row < local_end
                column = row // PAGE
                page = tl.load(Table + column, mask=live & (column < COLUMNS), other=-1).to(tl.int64)
                valid = live & (page >= 0) & (page < CACHE_PAGES)
                data = tl.load(Cache + (page[:, None] * PAGE + row[:, None] % PAGE) * WORDS + words[None, :],
                               mask=valid[:, None] & (words[None, :] < WORDS), other=0)
                tl.store(Stage + row[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                first += tl.num_programs(0) * 4
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _unpack_direct_or_mrd(Stage, Addresses, End, Rank, Ready, Output, Signal,
                          WORLD: tl.constexpr, MAXIMUM: tl.constexpr, FAST_LIMIT: tl.constexpr,
                          SLOT_ROWS: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    target_idx = tl.program_id(1)
    if tl.load(Ready) == 0:
        if end <= FAST_LIMIT:
            owner = rank ^ (target_idx + 1)
            owner_end = tl.maximum(end - owner + WORLD - 1, 0) // WORLD
            slot_off = owner * SLOT_ROWS * WORDS
            first = tl.program_id(0).to(tl.int64) * 4
            while first < owner_end:
                row = (first + tl.arange(0, 4)).to(tl.int64)
                live = row < owner_end
                data = tl.load(Stage + slot_off + row[:, None] * WORDS + words[None, :],
                               mask=live[:, None] & (words[None, :] < WORDS), other=0)
                grow = row * WORLD + owner
                tl.store(Output + grow[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                first += tl.num_programs(0) * 4
        else:
            gid = tl.program_id(0).to(tl.int64) + target_idx.to(tl.int64) * tl.num_programs(0).to(tl.int64)
            stride = tl.num_programs(0).to(tl.int64) * (WORLD - 1)
            first = gid * 8
            while first < end:
                row = (first + tl.arange(0, 8)).to(tl.int64)
                live = row < end
                source = tl.load(Addresses + row % WORLD).to(tl.pointer_type(tl.int32))
                data = tl.load(source[:, None] + (row[:, None] // WORLD) * WORDS + words[None, :],
                               mask=live[:, None] & (words[None, :] < WORDS), other=0)
                tl.store(Output + row[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                first += stride * 8
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _twohop_hop1_or_pack(Cache, Table, End, Rank, Completed, Stage, Addresses, Output, Signal,
                         COLUMNS: tl.constexpr, CACHE_PAGES: tl.constexpr, PAGE: tl.constexpr,
                         WORLD: tl.constexpr, ISLANDS: tl.constexpr, MAXIMUM: tl.constexpr,
                         FAST_LIMIT: tl.constexpr, SLOT_ROWS: tl.constexpr,
                         WORDS: tl.constexpr, BLOCK: tl.constexpr):
    """Switch-aware 2-hop posted-write all-push (Hop 1) for active lengths <= FAST_LIMIT.

    Each rank writes its own slice to Output and pushes one copy to each cross-switch twin
    (rank ^ (c * 4)) and each intra-switch peer (rank ^ {1,2,3}), cutting cross-switch traffic 4x.
    When end > FAST_LIMIT, falls back to packing local records into Stage for full-context unpack.
    """
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    local_end = tl.maximum(end - rank + WORLD - 1, 0) // WORLD
    target_idx = tl.program_id(1)
    cross_targets: tl.constexpr = ISLANDS - 1
    if tl.load(Completed) == 0:
        if end <= FAST_LIMIT:
            d = tl.where(target_idx < cross_targets, (target_idx + 1) * 4, target_idx - cross_targets + 1)
            peer = rank ^ d
            dst_base = tl.load(Addresses + peer).to(tl.pointer_type(tl.int32))
            slot_off = rank * SLOT_ROWS * WORDS
            first = tl.program_id(0).to(tl.int64) * 4
            while first < local_end:
                row = (first + tl.arange(0, 4)).to(tl.int64)
                live = row < local_end
                column = row // PAGE
                page = tl.load(Table + column, mask=live & (column < COLUMNS), other=-1).to(tl.int64)
                valid = live & (page >= 0) & (page < CACHE_PAGES)
                data = tl.load(Cache + (page[:, None] * PAGE + row[:, None] % PAGE) * WORDS + words[None, :],
                               mask=valid[:, None] & (words[None, :] < WORDS), other=0)
                tl.store(dst_base + slot_off + row[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                if target_idx == 0:
                    grow = row * WORLD + rank
                    tl.store(Output + grow[:, None] * WORDS + words[None, :], data,
                             mask=live[:, None] & (words[None, :] < WORDS))
                first += tl.num_programs(0) * 4
        elif target_idx == 0:
            first = tl.program_id(0).to(tl.int64) * 4
            while first < local_end:
                row = (first + tl.arange(0, 4)).to(tl.int64)
                live = row < local_end
                column = row // PAGE
                page = tl.load(Table + column, mask=live & (column < COLUMNS), other=-1).to(tl.int64)
                valid = live & (page >= 0) & (page < CACHE_PAGES)
                data = tl.load(Cache + (page[:, None] * PAGE + row[:, None] % PAGE) * WORDS + words[None, :],
                               mask=valid[:, None] & (words[None, :] < WORDS), other=0)
                tl.store(Stage + row[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                first += tl.num_programs(0) * 4
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _twohop_hop2_or_unpack(Stage, Addresses, End, Rank, Ready, Output, Signal,
                           WORLD: tl.constexpr, ISLANDS: tl.constexpr, MAXIMUM: tl.constexpr,
                           FAST_LIMIT: tl.constexpr, SLOT_ROWS: tl.constexpr,
                           WORDS: tl.constexpr, BLOCK: tl.constexpr):
    """Hop 2 (intra-switch only): forward received cross-switch twin slices to local peers and unpack local/twin slots."""
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    target_idx = tl.program_id(1)
    if tl.load(Ready) == 0:
        if end <= FAST_LIMIT:
            d_local = target_idx + 1
            local_peer = rank ^ d_local
            dst_base = tl.load(Addresses + local_peer).to(tl.pointer_type(tl.int32))
            for c in tl.static_range(1, ISLANDS):
                twin = rank ^ (c * 4)
                twin_end = tl.maximum(end - twin + WORLD - 1, 0) // WORLD
                twin_off = twin * SLOT_ROWS * WORDS
                first = tl.program_id(0).to(tl.int64) * 4
                while first < twin_end:
                    row = (first + tl.arange(0, 4)).to(tl.int64)
                    live = row < twin_end
                    data = tl.load(Stage + twin_off + row[:, None] * WORDS + words[None, :],
                                   mask=live[:, None] & (words[None, :] < WORDS), other=0)
                    tl.store(dst_base + twin_off + row[:, None] * WORDS + words[None, :], data,
                             mask=live[:, None] & (words[None, :] < WORDS))
                    if target_idx == 0:
                        grow = row * WORLD + twin
                        tl.store(Output + grow[:, None] * WORDS + words[None, :], data,
                                 mask=live[:, None] & (words[None, :] < WORDS))
                    first += tl.num_programs(0) * 4
            peer_end = tl.maximum(end - local_peer + WORLD - 1, 0) // WORLD
            peer_off = local_peer * SLOT_ROWS * WORDS
            first = tl.program_id(0).to(tl.int64) * 4
            while first < peer_end:
                row = (first + tl.arange(0, 4)).to(tl.int64)
                live = row < peer_end
                data = tl.load(Stage + peer_off + row[:, None] * WORDS + words[None, :],
                               mask=live[:, None] & (words[None, :] < WORDS), other=0)
                grow = row * WORLD + local_peer
                tl.store(Output + grow[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                first += tl.num_programs(0) * 4
        else:
            gid = tl.program_id(0).to(tl.int64) + target_idx.to(tl.int64) * tl.num_programs(0).to(tl.int64)
            stride = tl.num_programs(0).to(tl.int64) * 3
            first = gid * 8
            while first < end:
                row = (first + tl.arange(0, 8)).to(tl.int64)
                live = row < end
                source = tl.load(Addresses + row % WORLD).to(tl.pointer_type(tl.int32))
                data = tl.load(source[:, None] + (row[:, None] // WORLD) * WORDS + words[None, :],
                               mask=live[:, None] & (words[None, :] < WORDS), other=0)
                tl.store(Output + row[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                first += stride * 8
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _twohop_unpack_tail(Stage, End, Rank, Ready, Output, Signal,
                        WORLD: tl.constexpr, ISLANDS: tl.constexpr, MAXIMUM: tl.constexpr,
                        FAST_LIMIT: tl.constexpr, SLOT_ROWS: tl.constexpr,
                        WORDS: tl.constexpr, BLOCK: tl.constexpr):
    """Pure local HBM unpack of the remaining remote cross-switch slices received in Hop 2."""
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    rank = tl.load(Rank).to(tl.int64)
    target_idx = tl.program_id(1)
    if tl.load(Ready) == 0 and end <= FAST_LIMIT:
        d_local = target_idx + 1
        for c in tl.static_range(1, ISLANDS):
            owner = rank ^ (c * 4) ^ d_local
            owner_end = tl.maximum(end - owner + WORLD - 1, 0) // WORLD
            slot_off = owner * SLOT_ROWS * WORDS
            first = tl.program_id(0).to(tl.int64) * 4
            while first < owner_end:
                row = (first + tl.arange(0, 4)).to(tl.int64)
                live = row < owner_end
                data = tl.load(Stage + slot_off + row[:, None] * WORDS + words[None, :],
                               mask=live[:, None] & (words[None, :] < WORDS), other=0)
                grow = row * WORLD + owner
                tl.store(Output + grow[:, None] * WORDS + words[None, :], data,
                         mask=live[:, None] & (words[None, :] < WORDS))
                first += tl.num_programs(0) * 4
    if tl.program_id(0) == 0 and target_idx == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


def validate_history(cache, table, end, rank, stage, addresses, output, signals):
    """Check transfer geometry without inspecting device values."""
    if cache.ndim != 3 or output.ndim != 2 or stage.ndim != 2:
        raise ValueError("history transfer requires paged source and matrix staging/output")
    pages, page, width = cache.shape
    maximum, record = output.shape
    world = addresses.numel()
    if (min(pages, page, width, maximum, world) <= 0 or width != record or width % 16
            or maximum >= 2**31 or maximum % world or stage.shape != (maximum // world, width)):
        raise ValueError("history transfer requires whole interleaved records and 16-byte widths")
    if cache.dtype not in (torch.uint8, torch.float8_e4m3fn) or stage.dtype != torch.uint8 or output.dtype != cache.dtype:
        raise TypeError("history transfer preserves uint8 or E4M3 bytes through uint8 staging")
    if any(value.storage_offset() % 4 for value in (cache, stage, output)):
        raise ValueError("history records require word-aligned storage offsets")
    if (table.ndim != 1 or table.numel() * page < maximum // world or table.dtype != torch.int32
            or end.shape != (1,) or end.dtype != torch.int64 or rank.shape != (1,) or rank.dtype != torch.int64
            or addresses.ndim != 1 or addresses.dtype != torch.int64):
        raise ValueError("history transfer requires a complete int32 page table and int64 addressing")
    operands = (cache, table, end, rank, stage, addresses, output, *signals)
    if any(value.device != cache.device or not value.is_contiguous() for value in operands):
        raise ValueError("history transfer operands must be contiguous on one device")
    if len(signals) != 4 or any(value.shape != (1, 8) or value.dtype != torch.bfloat16 for value in signals):
        raise ValueError("history transfer requires four BF16 collective signal rows")
    spans = [(value.untyped_storage()._cdata, value.storage_offset() * value.element_size(),
              value.numel() * value.element_size()) for value in (cache, stage, output, *signals)]
    for index, (storage, begin, size) in enumerate(spans):
        if any(storage == other and begin < offset + extent and offset < begin + size
               for other, offset, extent in spans[:index]):
            raise ValueError("history source, destination and signals must not overlap")


class PreparedInterleavedHistory:
    """Publish local history, copy peer bytes, then release the shared stage.

    Signal inputs are written by the copy kernels. Their collective outputs are
    copy inputs, so compilation preserves both publication and reuse boundaries.
    The caller retains all storage and maps every address before preparation.
    """

    def __init__(self, cache, table, end, rank, stage, addresses, output, signals, *,
                 runtime, session, coordinator, name):
        validate_history(cache, table, end, rank, stage, addresses, output, signals)
        self.cache, self.table, self.end, self.rank = cache, table, end, rank
        self.stage, self.addresses, self.output = stage, addresses, output
        self.packed, self.ready, self.copied, self.completed = signals
        self.world, self.maximum = addresses.numel(), output.shape[0]
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
        words = self.cache.shape[2] // 4
        block = triton.next_power_of_2(words)
        stage_rows = self.stage.shape[0]
        if self.world in (8, 16) and stage_rows % self.world == 0:
            islands = self.world // 4
            slot_rows = stage_rows // self.world
            blocks = min(64, max(1, triton.cdiv(slot_rows, 4)))
            torch.library.wrap_triton(_twohop_hop1_or_pack)[(blocks, islands + 2)](
                self.cache.view(torch.int32), table, end, self.rank, self.completed,
                self.stage.view(torch.int32), self.addresses, output.view(torch.int32), self.packed,
                COLUMNS=table.numel(), CACHE_PAGES=self.cache.shape[0], PAGE=self.cache.shape[1],
                WORLD=self.world, ISLANDS=islands, MAXIMUM=self.maximum, FAST_LIMIT=stage_rows,
                SLOT_ROWS=slot_rows, WORDS=words, BLOCK=block, num_warps=4)
            self.publish(self.packed)
            torch.library.wrap_triton(_twohop_hop2_or_unpack)[(blocks, 3)](
                self.stage.view(torch.int32), self.addresses, end, self.rank, self.ready,
                output.view(torch.int32), self.packed, WORLD=self.world, ISLANDS=islands,
                MAXIMUM=self.maximum, FAST_LIMIT=stage_rows, SLOT_ROWS=slot_rows,
                WORDS=words, BLOCK=block, num_warps=4)
            self.publish(self.packed)
            torch.library.wrap_triton(_twohop_unpack_tail)[(blocks, 3)](
                self.stage.view(torch.int32), end, self.rank, self.ready,
                output.view(torch.int32), self.copied, WORLD=self.world, ISLANDS=islands,
                MAXIMUM=self.maximum, FAST_LIMIT=stage_rows, SLOT_ROWS=slot_rows,
                WORDS=words, BLOCK=block, num_warps=4)
            self.consume(self.copied)
            return output
        if self.world == 4 and stage_rows % self.world == 0:
            slot_rows = stage_rows // self.world
            blocks = min(64, max(1, triton.cdiv(slot_rows, 4)))
            torch.library.wrap_triton(_push_direct_or_pack)[(blocks, 3)](
                self.cache.view(torch.int32), table, end, self.rank, self.completed,
                self.stage.view(torch.int32), self.addresses, output.view(torch.int32), self.packed,
                COLUMNS=table.numel(), CACHE_PAGES=self.cache.shape[0], PAGE=self.cache.shape[1],
                WORLD=self.world, MAXIMUM=self.maximum, FAST_LIMIT=stage_rows,
                SLOT_ROWS=slot_rows, WORDS=words, BLOCK=block, num_warps=4)
            self.publish(self.packed)
            torch.library.wrap_triton(_unpack_direct_or_mrd)[(blocks, 3)](
                self.stage.view(torch.int32), self.addresses, end, self.rank, self.ready,
                output.view(torch.int32), self.copied, WORLD=self.world, MAXIMUM=self.maximum,
                FAST_LIMIT=stage_rows, SLOT_ROWS=slot_rows, WORDS=words, BLOCK=block, num_warps=4)
            self.consume(self.copied)
            return output
        # Bound empty-CTA work by the launch grid, not the prepared history
        # capacity. Grid-stride copies stop at the device-visible prefix length.
        torch.library.wrap_triton(_pack)[(min(1024, triton.cdiv(stage_rows, 4)),)](
            self.cache.view(torch.int32), table, end, self.rank, self.completed,
            self.stage.view(torch.int32), self.packed, COLUMNS=table.numel(),
            CACHE_PAGES=self.cache.shape[0], PAGE=self.cache.shape[1], WORLD=self.world,
            MAXIMUM=self.maximum, WORDS=words, BLOCK=block, num_warps=4)
        self.publish(self.packed)
        torch.library.wrap_triton(_unpack)[(min(1024, triton.cdiv(self.maximum, 8)),)](
            self.addresses, end, self.ready, output.view(torch.int32), self.copied,
            WORLD=self.world, MAXIMUM=self.maximum, WORDS=words,
            BLOCK=block, num_warps=4)
        self.consume(self.copied)
        return output
