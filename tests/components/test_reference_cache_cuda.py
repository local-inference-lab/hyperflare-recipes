# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured reference compressor state across prefill chunks and draft rejection."""

import os

import pytest
import torch

from ignition.preparation.compilation import compile_forward
from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheLayout, ReferenceCacheStorage
from tests.recipes.test_deepseek41_reference_cache import reference_config

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.fixture(scope="module")
def reference_compressor(reference_config):
    from deepinfer.model_executor.models.v4_modules.mla_attn import LoraRopeCompressFlashMHAAutoRecomp
    from deepinfer.model_executor.models.v4_modules.compress import Compressor

    cfg = reference_config + reference_config.attn_module_list[2].cfg
    cfg.layer_id = 2
    with torch.device("meta"):
        source = LoraRopeCompressFlashMHAAutoRecomp(cfg)
        compressor = Compressor(source.PackedWkv, cfg, source.rotary_emb)
    return compressor


@pytest.mark.parametrize("width", [1, 6, 8])
@torch.no_grad()
def test_reference_decode_padding_preserves_state(reference_config, reference_compressor, width):
    from deepinfer.model_executor.input_metadata import InputMetadataV4
    from deepinfer.model_executor.layers.tilelang_ops.compress import compress_and_update_state_decode

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=width)
    pool = ReferenceCacheStorage.allocate(geometry, 4, 2, "cuda")
    state = pool.cache_views()[2].ca_state
    projected = torch.randn((3 * width, 1024), device="cuda")
    positions = torch.cat([torch.arange(start, start + width, device="cuda", dtype=torch.int32)
                           for start in (11, 11, 101)])
    slots = torch.tensor([1, 1, 2], dtype=torch.int32, device="cuda").repeat_interleave(width)
    valid = torch.zeros(3 * width, dtype=torch.bool, device="cuda")
    metadata = InputMetadataV4(False, torch.zeros_like(positions), positions - 1,
        state_block_idx=slots, context_lens=positions, valid_token_mask=valid, verification_width=width)
    output = torch.empty((3 * width, 512), dtype=torch.bfloat16, device="cuda")
    metadata.verification_width = 0
    with pytest.raises(ValueError, match="verification width"):
        reference_compressor.forward_pre_norm(projected, state, metadata)
    metadata.verification_width = width
    with pytest.raises(AssertionError, match="cannot hold"):
        reference_compressor.forward_pre_norm(projected, state[:, :width], metadata)
    compiled = compile_forward(lambda: output.copy_(reference_compressor.forward_pre_norm(projected, state, metadata)))
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            compiled()
        for enabled in ((True, False, True), (False, False, False), (False, True, True)):
            valid.copy_(torch.tensor(enabled, dtype=torch.bool, device="cuda").repeat_interleave(width))
            projected.normal_()
            projected[~valid] = float("nan")
            state.normal_()
            before = pool.backing.clone()
            expected_state = state.clone()
            expected = torch.zeros_like(output)
            for request, active in enumerate(enabled):
                if active:
                    rows = slice(request * width, (request + 1) * width)
                    score, latent = projected[rows].chunk(2, dim=-1)
                    expected[rows] = compress_and_update_state_decode(score, latent, expected_state,
                        slots[rows], positions[rows], width, 2)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
            for row in (0, 3):
                assert torch.equal(pool.backing[row], before[row])
            if not any(enabled):
                assert torch.equal(pool.backing, before)
    finally:
        graph.reset()


@torch.no_grad()
def test_reference_prefill_padding_preserves_state(reference_config, reference_compressor):
    from deepinfer.model_executor.input_metadata import InputMetadataV4
    from deepinfer.model_executor.layers.tilelang_ops.compress import compress_and_update_state_prefill

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=8)
    pool = ReferenceCacheStorage.allocate(geometry, 4, 2, "cuda")
    state = pool.cache_views()[2].ca_state
    projected = torch.randn((258, 1024), device="cuda")
    slots = torch.tensor([1, 1, 2], dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, 129, 129, 258], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 129, 129, 261], dtype=torch.int32, device="cuda")
    cu_c = torch.tensor([0, 64, 65, 130], dtype=torch.int32, device="cuda")
    mapping = torch.repeat_interleave(torch.arange(3, dtype=torch.int32, device="cuda"),
                                     torch.tensor([64, 1, 65], device="cuda"))
    metadata = InputMetadataV4(True, torch.zeros(258, dtype=torch.int32, device="cuda"),
        torch.zeros(258, dtype=torch.int32, device="cuda"), state_block_idx=slots,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, cu_compressed_seqlens_q={"2": cu_c},
        compressed_batch_mapping={"2": mapping})
    output = torch.empty((130, 512), dtype=torch.bfloat16, device="cuda")
    compiled = compile_forward(lambda: output.copy_(reference_compressor.forward_pre_norm(projected, state, metadata)))
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            compiled()
        for all_inactive in (False, True, False):
            slots.copy_(torch.tensor([-1, -1, -1] if all_inactive else [1, 1, 2], device="cuda", dtype=torch.int32))
            state.normal_()
            projected.normal_()
            if all_inactive:
                projected.fill_(float("nan"))
            before = pool.backing.clone()
            expected_state = state.clone()
            expected = torch.zeros_like(output)
            if not all_inactive:
                for request, begin, end, start, count, out_start in ((1, 0, 129, 0, 64, 0), (2, 129, 258, 3, 65, 65)):
                    score, latent = projected[begin:end].chunk(2, dim=-1)
                    expected[out_start:out_start + count] = compress_and_update_state_prefill(
                        score, latent, expected_state, torch.tensor([request], device="cuda", dtype=torch.int32),
                        torch.zeros(count, device="cuda", dtype=torch.int32),
                        torch.tensor([0, end - begin], device="cuda", dtype=torch.int32),
                        torch.tensor([0, start + end - begin], device="cuda", dtype=torch.int32),
                        torch.tensor([0, count], device="cuda", dtype=torch.int32), 2)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
            assert torch.equal(pool.backing[0], before[0]) and torch.equal(pool.backing[3], before[3])
            if all_inactive:
                assert torch.equal(pool.backing, before)
    finally:
        graph.reset()


@pytest.mark.parametrize("width", [1, 6, 8])
@torch.no_grad()
def test_reference_decode_indices_use_prepared_ring(reference_config, width):
    from deepinfer.model_executor.input_metadata import InputMetadataV4
    from deepinfer.model_executor.layers.tilelang_ops.get_kv_index import get_kv_swa_index_decode_ref

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=width)
    ring = geometry.window_tokens
    offsets = torch.zeros((16, 2), dtype=torch.int32, device="cuda")
    tokens = torch.zeros(16, dtype=torch.int32, device="cuda")
    output = torch.empty((16, 128), dtype=torch.int32, device="cuda")

    def metadata():
        return InputMetadataV4(False, tokens, tokens, kv_swa_index_offset_len=offsets,
                               kv_ca_index={}, kv_ca_index_len={})

    def forward():
        inputs = metadata()
        inputs.compute_attn_info(8192, 128, swa_block_size=ring)
        output.copy_(inputs.kv_swa_index)

    with pytest.raises(ValueError, match="ring size"):
        metadata().compute_attn_info(8192, 128)
    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            compiled()
        for step in range(4):
            offsets[:, 0] = (torch.arange(16, device="cuda") % 3) * ring + (ring - 3 + step) % ring
            offsets[:, 1] = 128
            offsets[::3, 1] = 0
            offsets[1::3, 1] = step + 1
            graph.replay()
            torch.cuda.synchronize()
            expected = get_kv_swa_index_decode_ref(offsets, 128, ring)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        graph.reset()


@pytest.mark.parametrize("drafts", [5, 7])
@torch.no_grad()
def test_reference_draft_indices_use_prepared_ring(reference_config, drafts):
    from deepinfer.model_executor.input_metadata import InputMetadataV4
    from deepinfer.model_executor.layers.tilelang_ops.get_kv_index import get_fused_dflash_get_input_ref

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=drafts + 1)
    ring = geometry.window_tokens
    positions = torch.tensor([1, 127, 128, 8191], dtype=torch.int32, device="cuda")
    slots = torch.arange(1, 5, dtype=torch.int32, device="cuda") * ring + positions % ring
    first = torch.tensor([0, 1, -1, 3], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([0, 3, 0, 1], dtype=torch.int32, device="cuda")
    inputs = InputMetadataV4(False, torch.zeros_like(positions), positions,
        valid_token_mask=torch.ones(4, dtype=torch.bool, device="cuda"),
        slot_mapping_store_swa=slots, block_tables=torch.zeros((4, 1), dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="proposed tokens"):
        inputs.build_dflash_input(first, accepted, drafts, 128, swa_block_size=ring - 1)
    result = inputs.build_dflash_input(first, accepted, drafts, 128, swa_block_size=ring)
    expected = get_fused_dflash_get_input_ref(slots, positions, first, accepted, ring, drafts, 128,
                                           has_valid_token_mask=True)
    actual = (result.dflash_newx_slot_mapping_store, result.positions, result.input_ids,
              result.valid_token_mask, result.gate_force_random, result.dflash_kv_index)
    for value, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert value is None
        else:
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def forward():
        result = inputs.build_dflash_input(first, accepted, drafts, 128, swa_block_size=ring)
        values = (result.dflash_newx_slot_mapping_store, result.positions, result.input_ids,
                  result.valid_token_mask, result.gate_force_random, result.dflash_kv_index)
        for output, value in zip(actual, values, strict=True):
            if output is not None:
                output.copy_(value)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for _ in range(3):
            positions.add_(3)
            slots.copy_(torch.arange(1, 5, dtype=torch.int32, device="cuda") * ring + positions % ring)
            expected = get_fused_dflash_get_input_ref(slots, positions, first, accepted, ring, drafts, 128,
                                                    has_valid_token_mask=True)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            for value, reference in zip(actual, expected, strict=True):
                if reference is not None:
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("width", [1, 6, 8])
@torch.no_grad()
def test_reference_compressor_continuation(reference_config, width):
    from deepinfer.model_executor.layers.tilelang_ops.compress import (
        compress_and_update_state_prefill, compress_and_update_state_decode,
    )

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=width)
    pool = ReferenceCacheStorage.allocate(geometry, 3, 2, "cuda")
    state = pool.cache_views()[2].ca_state
    prefix = 129 + 8192
    generator = torch.Generator(device="cuda").manual_seed(915)
    scores = torch.randn((prefix + 2 * width, 512), device="cuda", generator=generator)
    values = torch.randn_like(scores)
    graphs, owners = [], []

    def prefill(score, latent, cache, start):
        count = score.shape[0]
        compressed = (start + count) // 2 - start // 2
        args = (torch.tensor([1], dtype=torch.int32, device="cuda"),
                torch.zeros(max(1, compressed), dtype=torch.int32, device="cuda"),
                torch.tensor([0, count], dtype=torch.int32, device="cuda"),
                torch.tensor([0, start + count], dtype=torch.int32, device="cuda"),
                torch.tensor([0, max(1, compressed)], dtype=torch.int32, device="cuda"))
        return lambda: compress_and_update_state_prefill(score, latent, cache, *args, 2)

    def capture(fn, output):
        compiled = compile_forward(lambda: output.copy_(fn()))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        graphs.append(graph)
        owners.append(compiled)
        return graph

    try:
        oracle_state = torch.zeros_like(state)
        full = prefill(scores[:prefix], values[:prefix], oracle_state, 0)()
        oracle_tail = oracle_state[1].clone()
        first = torch.empty((64, 512), dtype=torch.bfloat16, device="cuda")
        second = torch.empty((4096, 512), dtype=torch.bfloat16, device="cuda")
        first_graph = capture(prefill(scores[:129], values[:129], state, 0), first)
        second_graph = capture(prefill(scores[129:prefix], values[129:prefix], state, 129), second)
        pool.backing.zero_()
        first_graph.replay()
        pool.backing[2].copy_(pool.backing[1])
        second_graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(first, full[:64], rtol=0, atol=0)
        torch.testing.assert_close(second, full[64:], rtol=0, atol=0)
        torch.testing.assert_close(state[1], oracle_tail, rtol=0, atol=0)
        # Restore an odd chunk boundary, including its pending compression pair.
        pool.backing[1].copy_(pool.backing[2])
        second_graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(second, full[64:], rtol=0, atol=0)
        torch.testing.assert_close(state[1], oracle_tail, rtol=0, atol=0)

        score = torch.empty((width, 512), device="cuda")
        latent = torch.empty_like(score)
        contexts = torch.arange(prefix + 1, prefix + width + 1, dtype=torch.int32, device="cuda")
        slots = torch.ones(width, dtype=torch.int32, device="cuda")
        decoded = torch.empty((width, 512), dtype=torch.bfloat16, device="cuda")
        score.copy_(scores[prefix:prefix + width])
        latent.copy_(values[prefix:prefix + width])
        decode_graph = capture(lambda: compress_and_update_state_decode(
            score, latent, state, slots, contexts, width, 2), decoded)
        for accepted in sorted({0, width // 2, width}):
            state[1].copy_(oracle_tail)
            score.copy_(scores[prefix:prefix + width])
            latent.copy_(values[prefix:prefix + width])
            contexts.copy_(torch.arange(prefix + 1, prefix + width + 1, device="cuda", dtype=torch.int32))
            decode_graph.replay()
            score.copy_(scores[prefix + width:prefix + 2 * width])
            latent.copy_(values[prefix + width:prefix + 2 * width])
            contexts.copy_(torch.arange(prefix + accepted + 1, prefix + accepted + width + 1,
                                       device="cuda", dtype=torch.int32))
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            decode_graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            logical_score = torch.cat((scores[:prefix + accepted], score))
            logical_value = torch.cat((values[:prefix + accepted], latent))
            oracle_state.zero_()
            expected = prefill(logical_score, logical_value, oracle_state, 0)()
            emitted = contexts % 2 == 0
            torch.testing.assert_close(decoded[emitted], expected[contexts[emitted].long() // 2 - 1], rtol=0, atol=0)
            assert not decoded[~emitted].count_nonzero()
            torch.testing.assert_close(state[1], oracle_state[1], rtol=0, atol=0)
            assert not pool.backing[0].count_nonzero()
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("width", [1, 8])
@torch.no_grad()
def test_reference_transaction_restores_encoded_cache_sections(reference_config, width):
    from deepinfer.model_executor.layers.tilelang_ops.cache_common import get_kv_cache_format
    from deepinfer.model_executor.layers.tilelang_ops.store_kv_cache import store_kv_cache
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheTransaction

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=8)
    pool = ReferenceCacheStorage.allocate(geometry, 5, 13, "cuda")
    pages = torch.arange(1, 13, device="cuda", dtype=torch.int32).view(4, 3)
    lengths = torch.tensor([width, width, min(width, 3), 0], device="cuda", dtype=torch.int32)
    metadata = Metadata.pack(torch.tensor([127, 134, 255, 0], device="cuda"), lengths,
        torch.tensor([1, 2, 3, 0], device="cuda"), pages,
        torch.zeros((4, width), device="cuda", dtype=torch.int64),
        torch.tensor([4], device="cuda", dtype=torch.int32), token_capacity=4 * width,
        max_query_tokens=width)
    transaction = ReferenceCacheTransaction(pool)
    counts = torch.zeros_like(lengths)
    position = metadata.positions.clamp_min(0)
    physical = metadata.token_page_table.gather(1, (position // geometry.page_tokens)[:, None]).flatten()
    stores = []
    generator = torch.Generator(device="cuda").manual_seed(923)
    views = pool.cache_views()
    for layer in (2, 20):
        ratio = geometry.layers[layer].ratio
        slots = (physical * (geometry.page_tokens // ratio)
                 + position % geometry.page_tokens // ratio).int()
        valid = (metadata.positions >= 0) & ((position + 1) % ratio == 0)
        for cache in (views[layer].ca_cache, views[layer].ica_cache):
            layout = get_kv_cache_format(cache)
            values = torch.randn((4 * width, layout.num_elements), device="cuda",
                                 dtype=torch.bfloat16, generator=generator)
            stores.append((cache, values, slots, valid))

    def store(accepted):
        prefix = metadata.positions - metadata.starts[metadata.requests] < accepted[metadata.requests]
        for cache, values, slots, valid in stores:
            store_kv_cache(values, cache, torch.where(valid & prefix, slots, -1))

    def forward():
        saved = transaction.begin(metadata)
        store(lengths)
        transaction.commit(metadata, counts, saved)

    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    tensors = (pool.backing, *pool.pages.values())
    try:
        for accepted in range(width + 1):
            counts.copy_(lengths.clamp(max=accepted))
            for value in tensors:
                value.random_(0, 256)
            initial = tuple(value.clone() for value in tensors)
            store(counts)
            expected = tuple(value.clone() for value in tensors)
            for value, saved in zip(tensors, initial, strict=True):
                value.copy_(saved)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            for value, saved in zip(tensors, expected, strict=True):
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("width", [1, 8])
@torch.no_grad()
def test_reference_transaction_restores_every_rejected_cache_write(reference_config, width):
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheTransaction

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=8)
    pool = ReferenceCacheStorage.allocate(geometry, 5, 13, "cuda")
    transaction = ReferenceCacheTransaction(pool)
    starts = [127, 134, 255, 0]
    lengths = [width, width, min(width, 3), 0]
    rows = [1, 2, 3, 0]
    pages = torch.arange(1, 13, device="cuda", dtype=torch.int32).view(4, 3)
    metadata = Metadata.pack(torch.tensor(starts, device="cuda"),
        torch.tensor(lengths, device="cuda", dtype=torch.int32), torch.tensor(rows, device="cuda"),
        pages, torch.zeros((4, width), device="cuda", dtype=torch.int64),
        torch.tensor([4], device="cuda", dtype=torch.int32), token_capacity=4 * width,
        max_query_tokens=width)
    counts = torch.zeros(4, dtype=torch.int32, device="cuda")
    writes = []
    for region in transaction.rings:
        request_rows, ring_rows, owners, columns = [], [], [], []
        for request, length in enumerate(lengths):
            for column in range(length):
                request_rows.append(rows[request])
                ring_rows.append((starts[request] + column) % region.shape[1])
                owners.append(request)
                columns.append(column)
        values = torch.arange(len(owners) * region[0, 0].numel(), device="cuda").reshape(
            len(owners), *region.shape[2:]).to(region.dtype)
        indices = tuple(torch.tensor(value, device="cuda") for value in (request_rows, ring_rows))
        writes.append((region, indices, values, owners, columns))
    for region, ratio, stride, offset in transaction.pages:
        physical, owners, columns = [], [], []
        for request, length in enumerate(lengths):
            for column in range(length):
                position = starts[request] + column
                if (position + 1) % ratio == 0:
                    page = 1 + 3 * request + position // geometry.page_tokens
                    physical.append(page * stride + offset + position % geometry.page_tokens // ratio)
                    owners.append(request)
                    columns.append(column)
        values = torch.arange(len(owners) * region.shape[1], device="cuda").reshape(
            len(owners), region.shape[1]).to(torch.uint8)
        writes.append((region, (torch.tensor(physical, device="cuda"),), values, owners, columns))

    def forward():
        saved = transaction.begin(metadata)
        for region, indices, values, _, _ in writes:
            region.index_put_(indices, values)
        transaction.commit(metadata, counts, saved)

    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    storage = (pool.backing, *pool.pages.values())
    try:
        for accepted in range(width + 1):
            retained = [min(accepted, length) for length in lengths]
            counts.copy_(torch.tensor(retained, device="cuda", dtype=torch.int32))
            for value in storage:
                value.random_(0, 256)
            before = tuple(value.clone() for value in storage)
            for region, indices, values, owners, columns in writes:
                selected = torch.tensor([column < retained[owner]
                                         for owner, column in zip(owners, columns, strict=True)], device="cuda")
                region.index_put_(tuple(index[selected] for index in indices), values[selected])
            expected = tuple(value.clone() for value in storage)
            for value, saved in zip(storage, before, strict=True):
                value.copy_(saved)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            for value, saved in zip(storage, expected, strict=True):
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()
