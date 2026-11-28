# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare captured metadata with DeepInfer's native input construction."""

import os
from dataclasses import replace

import pytest
import torch

from hyperflare_recipes.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheLayout, ReferenceCacheStorage
from hyperflare_recipes.recipes.deepseek41.reference_metadata import ReferenceDecodeMetadata, ReferencePrefillMetadata
from tests.recipes.test_deepseek41_reference_cache import reference_config

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def reference_inputs(starts, lengths, states, pages, tokens, builder, *, prefill=False, yoco_keep_tokens=None):
    from deepinfer._C.engine import (
        PrepareArgs, SequenceData, SequenceMetadata, YocoPrefillTruncationMode,
        prepare_decode_v4, prepare_prompt_v4,
    )
    from deepinfer.sampling_params import SamplingParams

    sequences = []
    for index, (start, length, state, table, ids) in enumerate(zip(starts, lengths, states, pages, tokens, strict=True)):
        data = SequenceData([0] * start + (ids[:length] if prefill else [ids[0]]))
        sequences.append(SequenceMetadata(str(index), index, prefill, data, data,
            None, SamplingParams(), table, length, start + length, ids[0], [], ids[1:], state_id=state))
    args = PrepareArgs()
    args.seq_metadata_list = sequences
    args.is_prompt = prefill
    args.block_size = builder.geometry.page_tokens
    args.max_num_blocks = len(pages[0])
    args.window_size = builder.window
    args.swa_block_size = builder.geometry.window_tokens
    args.dense_compress_ratios = builder.dense_ratios
    args.sparse_compress_ratios = builder.sparse_ratios
    if yoco_keep_tokens is not None:
        args.yoco_prefill_truncation_mode = YocoPrefillTruncationMode.COMPACT
        args.yoco_prefill_keep_tokens = yoco_keep_tokens
        args.max_backtracking_tokens = builder.geometry.verification_width - 1
    return (prepare_prompt_v4 if prefill else prepare_decode_v4)(args)


def tensors(metadata):
    for field in ("input_ids", "positions", "valid_token_mask", "state_block_idx", "context_lens",
                  "query_indices", "block_tables", "slot_mapping_store_swa", "kv_swa_index_offset_len",
                  "positions_c", "slot_mapping_store_ca", "kv_ca_index_offset_len", "klen"):
        value = getattr(metadata, field)
        if isinstance(value, dict):
            for key, tensor in value.items():
                yield field + "." + key, tensor
        else:
            yield field, value


@pytest.mark.parametrize("width", [1, 6, 8])
@torch.no_grad()
def test_reference_decode_metadata_matches_native(reference_config, width):
    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=width)
    builder = ReferenceDecodeMetadata(reference_config, geometry)
    batch, blocks = 4, 66
    source = Metadata.allocate(batch * width + 3, batch, blocks, "cuda", max_query_tokens=width)
    with pytest.raises(ValueError, match="verification width"):
        builder(replace(source, max_query_tokens=width + 1))
    pages = torch.arange(batch * blocks, dtype=torch.int32, device="cuda").reshape(batch, blocks).flip(1).contiguous()
    states = torch.tensor([3, 1, 4, 2], device="cuda")
    ids = torch.arange(batch * width, dtype=torch.int64, device="cuda").reshape(batch, width) + 11
    starts = torch.tensor([0, 127, 134, 8192 - width], device="cuda")
    lengths = torch.full((batch,), width, dtype=torch.int32, device="cuda")
    source.set(starts, lengths, states, pages, ids, torch.tensor([batch], device="cuda"))
    buffers = {name: torch.empty_like(value) for name, value in tensors(builder(source))}

    def run():
        for name, value in tensors(builder(source)):
            buffers[name].copy_(value)

    compiled = compile_forward(run)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for step in range(3):
            lengths.copy_(torch.tensor([width, max(1, width - step), 1, width], device="cuda"))
            starts[:3].add_(step)
            source.set(starts, lengths, states, pages, ids, torch.tensor([batch], device="cuda"))
            expected = reference_inputs(starts.tolist(), lengths.tolist(), states.tolist(),
                                        pages.tolist(), ids.tolist(), builder)
            live = int(lengths.sum())
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
            for name, actual in buffers.items():
                if name == "block_tables":
                    # The native constructor leaves unused page-table columns uninitialized.
                    for row in range(live):
                        count = (int(expected["context_lens"][row]) + 127) // 128
                        assert torch.equal(actual[row, :count].cpu(), expected[name][row, :count]), name
                else:
                    field, _, key = name.partition(".")
                    reference = expected[field][key] if key else expected[field]
                    assert torch.equal(actual[:live].cpu(), reference), name
            assert not buffers["valid_token_mask"][live:].any()
            assert (buffers["slot_mapping_store_swa"][live:] == -1).all()
            assert (buffers["state_block_idx"][live:] == -1).all()
            for ratio in builder.ratios:
                assert (buffers[f"slot_mapping_store_ca.{ratio}"][live:] == -1).all()
        source.positions.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert not buffers["valid_token_mask"].any()
        assert not buffers["context_lens"].any()
        assert (buffers["slot_mapping_store_swa"] == -1).all()
        assert (buffers["state_block_idx"] == -1).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()


def prefill_tensors(metadata):
    for field in ("input_ids", "positions", "valid_token_mask", "state_block_idx",
                  "cu_seqlens_q", "cu_seqlens_k", "slot_mapping_store_swa",
                  "kv_swa_full_index_pivots", "positions_c", "slot_mapping_store_ca",
                  "cu_compressed_seqlens_q", "compressed_batch_mapping",
                  "load_segments_swa", "load_segments_ca", "kv_ca_index_offset_len", "ks", "ke"):
        value = getattr(metadata, field)
        if isinstance(value, dict):
            for key, tensor in value.items():
                yield field + "." + key, tensor
        else:
            yield field, value


@pytest.mark.parametrize("capacity", [129, 8192])
@torch.no_grad()
def test_reference_prefill_metadata_matches_native(reference_config, capacity):
    from deepinfer.model_executor.layers.tilelang_ops.cache_common import V41_FORMAT, V41_FP4_FORMAT
    from deepinfer.model_executor.layers.tilelang_ops.compress import compress_and_update_state_prefill
    from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache
    from deepinfer.model_executor.layers.tilelang_ops.store_kv_cache import store_kv_cache

    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=8)
    builder = ReferencePrefillMetadata(reference_config, geometry)
    blocks = 130
    source = Metadata.allocate(capacity, 1, blocks, "cuda", max_query_tokens=capacity)
    with pytest.raises(ValueError, match="one admitted request"):
        builder(Metadata.allocate(capacity, 2, blocks, "cuda", max_query_tokens=capacity))
    starts = torch.zeros(1, dtype=torch.int64, device="cuda")
    lengths = torch.tensor([capacity], dtype=torch.int32, device="cuda")
    states = torch.tensor([2], device="cuda")
    pages = (torch.arange(blocks, dtype=torch.int32, device="cuda").flip(0) + 3)[None].contiguous()
    ids = torch.arange(capacity, device="cuda")[None] + 17
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    source.set(starts, lengths, states, pages, ids, live)
    metadata = builder(source)
    buffers = {name: torch.empty_like(value) for name, value in prefill_tensors(metadata)}

    def prepare():
        for name, value in prefill_tensors(builder(source)):
            buffers[name].copy_(value)

    compiled = compile_forward(prepare)
    compiled()
    # The same reference gather kernel consumes fixed descriptors and the C++
    # builder's compact descriptors. Physical rows have distinct random values.
    storage = ReferenceCacheStorage.allocate(geometry, 3, blocks + 3, "cuda")
    views = storage.cache_views()
    caches = {"swa": views[0].swa_cache, "1": views[20].ca_cache, "2": views[2].ca_cache}
    for key, cache in caches.items():
        count, block = cache.shape[:2]
        layout = V41_FORMAT if key == "swa" else V41_FP4_FORMAT
        values = torch.randn(count * block, layout.num_elements, dtype=torch.bfloat16, device="cuda")
        store_kv_cache(values, cache, torch.arange(count * block, dtype=torch.int32, device="cuda"))
    state = torch.randn(3, 9, 2, 512, device="cuda")
    score = torch.randn(capacity, 512, device="cuda")
    latent = torch.randn_like(score)
    gathered = {}

    def gather():
        gathered["swa"] = load_paged_kv_cache(caches["swa"], buffers["load_segments_swa"], 128).bf16_data
        for ratio in builder.ratios:
            key = str(ratio)
            gathered[key] = load_paged_kv_cache(caches[key], buffers[f"load_segments_ca.{key}"],
                                               metadata.total_k_compressed[key]).bf16_data
        gathered["compressed"] = compress_and_update_state_prefill(score, latent, state,
            buffers["state_block_idx"], buffers["compressed_batch_mapping.2"],
            buffers["cu_seqlens_q"], buffers["cu_seqlens_k"], buffers["cu_compressed_seqlens_q.2"], 2)

    gather()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
        gather()
    try:
        for start, length in ((0, capacity), (1, capacity), (127, capacity - 1),
                              (135, 1), (8191, capacity), (8192, 1), (0, 0)):
            starts.fill_(start)
            lengths.fill_(length)
            source.set(starts, lengths, states, pages, ids, live)
            expected = reference_inputs([start], [length], [2], pages.tolist(), ids.tolist(), builder, prefill=True)
            score.normal_()
            latent.normal_()
            score[length:].fill_(float("nan"))
            latent[length:].fill_(float("nan"))
            expected_state = state.clone()
            expected_compressed = compress_and_update_state_prefill(score, latent, expected_state,
                torch.tensor([2 if length else -1], dtype=torch.int32, device="cuda"),
                torch.zeros_like(expected["positions_c"]["2"], device="cuda"),
                expected["cu_seqlens_q"].cuda(), expected["cu_seqlens_k"].cuda(),
                expected["cu_compressed_seqlens_q"]["2"].cuda(), 2)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
            assert torch.equal(state, expected_state)
            count = expected_compressed.shape[0]
            assert torch.equal(gathered["compressed"][:count], expected_compressed)
            assert not gathered["compressed"][count:].count_nonzero()
            for name, actual in buffers.items():
                if name.startswith(("load_segments", "compressed_batch_mapping")):
                    continue
                if name == "valid_token_mask":
                    assert actual[:length].all() and not actual[length:].any()
                    continue
                field, _, key = name.partition(".")
                reference = expected[field][key] if key else expected[field]
                if name == "state_block_idx" and length == 0:
                    assert (actual == -1).all()
                else:
                    assert torch.equal(actual[:reference.shape[0]].cpu(), reference), (name, start, length)
            for key, cache in caches.items():
                segments = expected["load_segments_swa"] if key == "swa" else expected["load_segments_ca"][key]
                count = 128 if key == "swa" else expected["total_k_compressed"][key]
                reference = load_paged_kv_cache(cache, segments.cuda(), count).bf16_data
                assert torch.equal(gathered[key][:count], reference), (key, start, length)
                assert not gathered[key][count:].count_nonzero()
            assert not buffers["valid_token_mask"][length:].any()
            assert (buffers["slot_mapping_store_swa"][length:] == -1).all()
            for ratio in builder.ratios:
                count = (start + length) // ratio - start // ratio
                assert (buffers[f"slot_mapping_store_ca.{ratio}"][count:] == -1).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("capacity", [256, 8192])
@torch.no_grad()
def test_reference_prefill_yoco_compaction_matches_native(reference_config, capacity):
    keep = 144
    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=8)
    with pytest.raises(ValueError, match="sliding-window ring capacity"):
        ReferencePrefillMetadata(reference_config, geometry, yoco_keep_tokens=134)
    builder = ReferencePrefillMetadata(reference_config, geometry, yoco_keep_tokens=keep)
    small_source = Metadata.allocate(128, 1, 4, "cuda", max_query_tokens=128)
    assert not builder(small_source).yoco_compaction_enabled

    blocks = 130
    source = Metadata.allocate(capacity, 1, blocks, "cuda", max_query_tokens=capacity)
    starts = torch.zeros(1, dtype=torch.int64, device="cuda")
    lengths = torch.tensor([capacity], dtype=torch.int32, device="cuda")
    states = torch.tensor([2], device="cuda")
    pages = (torch.arange(blocks, dtype=torch.int32, device="cuda").flip(0) + 3)[None].contiguous()
    ids = torch.arange(capacity, device="cuda")[None] + 17
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    images = (torch.arange(capacity, device="cuda") % 7 == 0)
    ngrams = torch.arange(capacity * 4, dtype=torch.int32, device="cuda").reshape(capacity, 4)
    source.set(starts, lengths, states, pages, ids, live)
    sample = builder(source, ngram_token_ids=ngrams, images_seq_mask=images)
    assert sample.yoco_compaction_enabled
    assert sample.yoco_consumer_total_q == keep
    consumer_view = sample.get_consumer_view()
    assert consumer_view.total_q == keep

    def yoco_tensors(meta):
        view = meta.get_consumer_view()
        for field in ("yoco_consumer_query_indices", "yoco_consumer_cu_seqlens_q",
                      "yoco_consumer_positions", "yoco_consumer_swa_index_pivots",
                      "yoco_consumer_slot_mapping_store_swa"):
            yield field, getattr(meta, field)
        for field in ("yoco_consumer_ks", "yoco_consumer_ke", "yoco_consumer_kv_ca_index_offset_len"):
            for key, tensor in getattr(meta, field).items():
                yield f"{field}.{key}", tensor
        yield "view.valid_token_mask", view.valid_token_mask
        yield "view.images_seq_mask", view.images_seq_mask
        yield "view.valid_token_count", view.valid_token_count
        yield "view.ngram_token_ids", view.ngram_token_ids

    buffers = {name: torch.empty_like(value) for name, value in yoco_tensors(sample)}

    def prepare():
        meta = builder(source, ngram_token_ids=ngrams, images_seq_mask=images & (source.positions >= 0))
        for name, value in yoco_tensors(meta):
            buffers[name].copy_(value)

    compiled = compile_forward(prepare)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for start, length in ((0, capacity), (8192, capacity), (8192, keep), (135, keep + 5),
                              (0, keep), (0, 50), (0, 1), (0, 0)):
            starts.fill_(start)
            lengths.fill_(length)
            source.set(starts, lengths, states, pages, ids, live)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
            kept = min(length, keep)
            assert int(buffers["view.valid_token_count"].item()) == kept
            assert buffers["view.valid_token_mask"][:kept].all()
            assert not buffers["view.valid_token_mask"][kept:].any()
            assert not buffers["view.images_seq_mask"][kept:].any()
            assert (buffers["yoco_consumer_query_indices"][kept:] == -1).all()
            assert (buffers["yoco_consumer_slot_mapping_store_swa"][kept:] == -1).all()
            if length == 0:
                assert torch.equal(buffers["yoco_consumer_cu_seqlens_q"].cpu(), torch.tensor([0, 0], dtype=torch.int32))
                continue
            expected = reference_inputs([start], [length], [2], pages.tolist(), ids.tolist(),
                                        builder, prefill=True, yoco_keep_tokens=keep)
            assert int(expected["yoco_consumer_total_q"]) == kept
            assert torch.equal(buffers["yoco_consumer_cu_seqlens_q"].cpu(), expected["yoco_consumer_cu_seqlens_q"])
            assert torch.equal(buffers["yoco_consumer_query_indices"][:kept].cpu(), expected["yoco_consumer_query_indices"])
            assert torch.equal(buffers["yoco_consumer_positions"][:kept].cpu(), expected["yoco_consumer_positions"])
            assert torch.equal(buffers["yoco_consumer_slot_mapping_store_swa"][:kept].cpu(),
                               expected["yoco_consumer_slot_mapping_store_swa"])
            expected_pivots = expected["yoco_consumer_swa_index_pivots"].clone()
            # sec2_start_c uses total_q_c; for 1 request both start at window_size (128).
            assert torch.equal(buffers["yoco_consumer_swa_index_pivots"][:kept].cpu(), expected_pivots)
            if length >= keep:
                # Every kept query attends solely within the chunk's kept suffix and overwrites all 135 ring slots.
                assert (buffers["yoco_consumer_swa_index_pivots"][:kept, 1] == 0).all()
                written_slots = buffers["yoco_consumer_slot_mapping_store_swa"][:kept]
                valid_slots = written_slots[written_slots >= 0]
                assert valid_slots.unique().numel() == geometry.window_tokens
            for field in ("yoco_consumer_ks", "yoco_consumer_ke", "yoco_consumer_kv_ca_index_offset_len"):
                for key, expected_tensor in expected[field].items():
                    assert torch.equal(buffers[f"{field}.{key}"][:kept].cpu(), expected_tensor), (field, key, start, length)
    finally:
        torch.cuda.synchronize()
        graph.reset()
