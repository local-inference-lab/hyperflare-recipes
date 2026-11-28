# SPDX-License-Identifier: Apache-2.0
"""Compiled history copies preserve active prefixes with invocation-local outputs."""

import os

import pytest
import torch

class _HistoryCopies:
    def __init__(self, latent, index, index_history, latent_history, table, end):
        self.index_history, self.latent_history = index_history, latent_history
        self.table, self.end = table, end
        self.saved_index = torch.empty_like(index)
        self.saved_latent = torch.empty_like(latent)

    def __call__(self):
        self.saved_index.copy_(self.index_history(self.table, self.end))
        self.saved_latent.copy_(self.latent_history(self.table, self.end))


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA GPU")
@torch.no_grad()
def test_compiled_serial_history_copies_preserve_bytes_with_changed_pages_and_tails():
    from hyperflare_recipes.components.interleaved_history import PreparedInterleavedHistory
    from hyperflare_recipes.components.planar_index_history import PreparedPlanarIndexHistory
    from ignition.preparation.compilation import compile_forward

    device = torch.device("cuda:0")
    maximum, pages = 4096, 36
    latent = torch.empty((maximum, 576), dtype=torch.uint8, device=device)
    index = torch.empty((maximum // 64, 8448), dtype=torch.uint8, device=device)
    latent_cache = torch.randint(0, 256, (pages, 128, 576), dtype=torch.uint8, device=device)
    index_cache = torch.randint(0, 256, (2 * pages, 8448), dtype=torch.uint8, device=device)
    table = torch.arange(maximum // 128, dtype=torch.int32, device=device)
    end = torch.zeros(1, dtype=torch.int64, device=device)
    rank = torch.zeros_like(end)
    latent_stage = torch.empty((maximum, 576), dtype=torch.uint8, device=device)
    index_stage = torch.empty_like(index)
    latent_addresses = torch.tensor([latent_stage.data_ptr()], dtype=torch.int64, device=device)
    index_addresses = torch.tensor([index_stage.data_ptr()], dtype=torch.int64, device=device)

    def signals():
        return tuple(torch.zeros((1, 8), dtype=torch.bfloat16, device=device) for _ in range(4))

    index_history = PreparedPlanarIndexHistory(index_cache, table, end, rank, index_stage,
        index_addresses, index, signals(), runtime=None, session=None, coordinator=None, name="index")
    latent_history = PreparedInterleavedHistory(latent_cache, table, end, rank, latent_stage,
        latent_addresses, latent.view(torch.uint8), signals(), runtime=None, session=None,
        coordinator=None, name="latent")
    index_history.functional()
    latent_history.functional()
    forward = _HistoryCopies(latent, index, index_history, latent_history, table, end)
    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    source_index, source_latent = index_cache.cpu(), latent_cache.cpu()
    try:
        for step, count in enumerate((4096, 1, 63, 129, 2048, 0, 4096)):
            mapping = (torch.arange(maximum // 128) * 5 + step) % pages
            table.copy_(mapping)
            end.fill_(count)
            expected_index = torch.full_like(source_index[:maximum // 64], 173)
            for token in range(count):
                src = source_index[2 * mapping[token // 128] + token % 128 // 64]
                dst = expected_index[token // 64]
                row = token % 64
                dst[row * 128:(row + 1) * 128].copy_(src[row * 128:(row + 1) * 128])
                dst[8192 + row * 4:8196 + row * 4].copy_(src[8192 + row * 4:8196 + row * 4])
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            observed_index = forward.saved_index.cpu()
            for token in range(count):
                page, row = divmod(token, 64)
                assert torch.equal(observed_index[page, row * 128:(row + 1) * 128],
                                   expected_index[page, row * 128:(row + 1) * 128])
                assert torch.equal(observed_index[page, 8192 + row * 4:8196 + row * 4],
                                   expected_index[page, 8192 + row * 4:8196 + row * 4])
            expected_latent = torch.full((maximum, 576), 173, dtype=torch.uint8)
            expected_latent[:count].copy_(source_latent[mapping].view(maximum, 576)[:count])
            assert torch.equal(forward.saved_latent.view(torch.uint8).cpu()[:count], expected_latent[:count])
    finally:
        graph.reset()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA GPU")
@pytest.mark.parametrize("world", [4, 8, 16])
@torch.no_grad()
def test_multi_rank_push_and_fallback_history_kernels_match_reference_bit_exact(world):
    """Exercise single-switch (world=4) and two-hop (world=8, 16) posted-write push and >FAST_LIMIT fallback."""
    from hyperflare_recipes.components import interleaved_history as ih
    from hyperflare_recipes.components import planar_index_history as pih
    import triton

    device = torch.device("cuda:0")
    pages = 8
    # Choose maximum so stage_rows = maximum // world is divisible by world (and >= 64 * world for planar index).
    # For world=16: stage_pages=16 -> stage_rows=1024 -> maximum=16384; fast_limit=1024.
    stage_pages = max(world, 16)
    stage_rows = stage_pages * 64
    maximum = stage_rows * world
    cols = maximum // (world * 128)
    slot_rows = stage_rows // world
    slot_pages = stage_pages // world

    gen = torch.Generator(device=device).manual_seed(42 + world)
    lat_caches = [torch.randint(0, 256, (pages, 128, 576), dtype=torch.uint8, device=device, generator=gen)
                  for _ in range(world)]
    idx_caches = [torch.randint(0, 256, (2 * pages, 8448), dtype=torch.uint8, device=device, generator=gen)
                  for _ in range(world)]
    lat_stages = [torch.zeros((stage_rows, 576), dtype=torch.uint8, device=device) for _ in range(world)]
    idx_stages = [torch.zeros((stage_pages, 8448), dtype=torch.uint8, device=device) for _ in range(world)]
    lat_addrs = torch.tensor([t.data_ptr() for t in lat_stages], dtype=torch.int64, device=device)
    idx_addrs = torch.tensor([t.data_ptr() for t in idx_stages], dtype=torch.int64, device=device)
    lat_outs = [torch.empty((maximum, 576), dtype=torch.uint8, device=device) for _ in range(world)]
    idx_outs = [torch.empty((stage_pages * world, 8448), dtype=torch.uint8, device=device) for _ in range(world)]
    table = ((torch.arange(cols, dtype=torch.int32, device=device) * 3 + 1) % pages).to(torch.int32)
    ranks = [torch.tensor([r], dtype=torch.int64, device=device) for r in range(world)]
    zero_sig = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
    sink_sig = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
    end = torch.zeros(1, dtype=torch.int64, device=device)

    lat_ref = torch.empty((maximum, 576), dtype=torch.uint8, device=device)
    idx_ref = torch.empty((stage_pages * world, 8448), dtype=torch.uint8, device=device)

    # Test edge lengths below, at, and above fast_limit (stage_rows), plus zero and full maximum.
    counts = (0, 1, world - 1, world, 63, 64, 129, stage_rows - 1, stage_rows, stage_rows + 1, maximum)
    words = 576 // 4
    block = triton.next_power_of_2(words)
    lat_blocks = min(64, max(1, triton.cdiv(slot_rows, 4)))
    idx_blocks = min(64, max(1, slot_pages * 8))

    for count in counts:
        end.fill_(count)
        # Reference baseline pack + unpack
        for r in range(world):
            ih._pack[(min(1024, triton.cdiv(stage_rows, 4)),)](
                lat_caches[r].view(torch.int32), table, end, ranks[r], zero_sig,
                lat_stages[r].view(torch.int32), sink_sig, COLUMNS=cols, CACHE_PAGES=pages,
                PAGE=128, WORLD=world, MAXIMUM=maximum, WORDS=words, BLOCK=block, num_warps=4)
            pih._pack[(min(1024, stage_pages * 16),)](
                idx_caches[r].view(torch.int32), table, end, ranks[r], zero_sig,
                idx_stages[r].view(torch.int32), sink_sig, COLUMNS=cols, CACHE_PAGES=2 * pages,
                WORLD=world, MAXIMUM=maximum, num_warps=4)
        lat_ref.fill_(173)
        idx_ref.fill_(173)
        ih._unpack[(min(1024, triton.cdiv(maximum, 8)),)](
            lat_addrs, end, zero_sig, lat_ref.view(torch.int32), sink_sig,
            WORLD=world, MAXIMUM=maximum, WORDS=words, BLOCK=block, num_warps=4)
        pih._unpack[(min(1024, idx_ref.shape[0] * 8),)](
            idx_addrs, end, zero_sig, idx_ref.view(torch.int32), sink_sig,
            WORLD=world, MAXIMUM=maximum, num_warps=4)

        # Candidate posted-write push / fallback across all ranks
        for r in range(world):
            lat_stages[r].fill_(201)
            idx_stages[r].fill_(201)
            lat_outs[r].fill_(173)
            idx_outs[r].fill_(173)
        if world == 4:
            for r in range(world):
                ih._push_direct_or_pack[(lat_blocks, 3)](
                    lat_caches[r].view(torch.int32), table, end, ranks[r], zero_sig,
                    lat_stages[r].view(torch.int32), lat_addrs, lat_outs[r].view(torch.int32), sink_sig,
                    COLUMNS=cols, CACHE_PAGES=pages, PAGE=128, WORLD=world, MAXIMUM=maximum,
                    FAST_LIMIT=stage_rows, SLOT_ROWS=slot_rows, WORDS=words, BLOCK=block, num_warps=4)
                pih._push_direct_or_pack[(idx_blocks, 3)](
                    idx_caches[r].view(torch.int32), table, end, ranks[r], zero_sig,
                    idx_stages[r].view(torch.int32), idx_addrs, idx_outs[r].view(torch.int32), sink_sig,
                    COLUMNS=cols, CACHE_PAGES=2 * pages, WORLD=world, MAXIMUM=maximum,
                    FAST_LIMIT=stage_rows, SLOT_PAGES=slot_pages, num_warps=4)
            for r in range(world):
                ih._unpack_direct_or_mrd[(lat_blocks, 3)](
                    lat_stages[r].view(torch.int32), lat_addrs, end, ranks[r], zero_sig,
                    lat_outs[r].view(torch.int32), sink_sig, WORLD=world, MAXIMUM=maximum,
                    FAST_LIMIT=stage_rows, SLOT_ROWS=slot_rows, WORDS=words, BLOCK=block, num_warps=4)
                pih._unpack_direct_or_mrd[(idx_blocks, 3)](
                    idx_stages[r].view(torch.int32), idx_addrs, end, ranks[r], zero_sig,
                    idx_outs[r].view(torch.int32), sink_sig, WORLD=world, MAXIMUM=maximum,
                    FAST_LIMIT=stage_rows, SLOT_PAGES=slot_pages, num_warps=4)
        else:
            islands = world // 4
            for r in range(world):
                ih._twohop_hop1_or_pack[(lat_blocks, islands + 2)](
                    lat_caches[r].view(torch.int32), table, end, ranks[r], zero_sig,
                    lat_stages[r].view(torch.int32), lat_addrs, lat_outs[r].view(torch.int32), sink_sig,
                    COLUMNS=cols, CACHE_PAGES=pages, PAGE=128, WORLD=world, ISLANDS=islands,
                    MAXIMUM=maximum, FAST_LIMIT=stage_rows, SLOT_ROWS=slot_rows, WORDS=words,
                    BLOCK=block, num_warps=4)
                pih._twohop_hop1_or_pack[(idx_blocks, islands + 2)](
                    idx_caches[r].view(torch.int32), table, end, ranks[r], zero_sig,
                    idx_stages[r].view(torch.int32), idx_addrs, idx_outs[r].view(torch.int32), sink_sig,
                    COLUMNS=cols, CACHE_PAGES=2 * pages, WORLD=world, ISLANDS=islands,
                    MAXIMUM=maximum, FAST_LIMIT=stage_rows, SLOT_PAGES=slot_pages, num_warps=4)
            for r in range(world):
                ih._twohop_hop2_or_unpack[(lat_blocks, 3)](
                    lat_stages[r].view(torch.int32), lat_addrs, end, ranks[r], zero_sig,
                    lat_outs[r].view(torch.int32), sink_sig, WORLD=world, ISLANDS=islands,
                    MAXIMUM=maximum, FAST_LIMIT=stage_rows, SLOT_ROWS=slot_rows, WORDS=words,
                    BLOCK=block, num_warps=4)
                pih._twohop_hop2_or_unpack[(idx_blocks, 3)](
                    idx_stages[r].view(torch.int32), idx_addrs, end, ranks[r], zero_sig,
                    idx_outs[r].view(torch.int32), sink_sig, WORLD=world, ISLANDS=islands,
                    MAXIMUM=maximum, FAST_LIMIT=stage_rows, SLOT_PAGES=slot_pages, num_warps=4)
            for r in range(world):
                ih._twohop_unpack_tail[(lat_blocks, 3)](
                    lat_stages[r].view(torch.int32), end, ranks[r], zero_sig,
                    lat_outs[r].view(torch.int32), sink_sig, WORLD=world, ISLANDS=islands,
                    MAXIMUM=maximum, FAST_LIMIT=stage_rows, SLOT_ROWS=slot_rows, WORDS=words,
                    BLOCK=block, num_warps=4)
                pih._twohop_unpack_tail[(idx_blocks, 3)](
                    idx_stages[r].view(torch.int32), end, ranks[r], zero_sig,
                    idx_outs[r].view(torch.int32), sink_sig, WORLD=world, ISLANDS=islands,
                    MAXIMUM=maximum, FAST_LIMIT=stage_rows, SLOT_PAGES=slot_pages, num_warps=4)
        torch.cuda.synchronize()
        full_pages, rem = divmod(count, 64)
        for r in range(world):
            assert torch.equal(lat_outs[r][:count], lat_ref[:count]), f"latent mismatch world={world} rank={r} count={count}"
            assert torch.equal(idx_outs[r][:full_pages], idx_ref[:full_pages]), f"index page mismatch world={world} rank={r} count={count}"
            if rem:
                assert torch.equal(idx_outs[r][full_pages, :rem * 128], idx_ref[full_pages, :rem * 128])
                assert torch.equal(idx_outs[r][full_pages, 8192:8192 + rem * 4], idx_ref[full_pages, 8192:8192 + rem * 4])
