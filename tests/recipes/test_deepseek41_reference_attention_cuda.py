# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Construct the complete checkpoint attention tree and replay reference attention."""

import os
from pathlib import Path

import pytest
import torch

from ignition.preparation.compilation import compile_forward
from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint
from hyperflare_recipes.recipes.deepseek41.reference_parameters import ReferenceParameters

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.fixture(scope="module")
def reference_attention_tree():
    pytest.importorskip("deepinfer")
    from deepinfer import AttrDict
    from deepinfer.engine.engine_type import EngineType
    from deepinfer.model_executor.execution_settings import ExecutionSettings
    from deepinfer.model_executor.models.v4_modules.attn import V4Attention, DFlashAttention
    from deepinfer.model_executor.models.v4_modules.cross_layer import build_shared_layer_topology
    from deepinfer.model_executor.models.v4_modules.utils import create_deepseek_decode_layer, layer_group_key
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires the DeepSeek V4.1 CSF checkpoint")
    if PG.world is None:
        PG.initialize_local()
    parameters = ReferenceParameters(ReferenceCheckpoint(Path(snapshot)))
    with torch.no_grad():
        parameters.load_ordinary("cuda:0")
    cfg = AttrDict(parameters.config.gpt, runtime_max_model_len=32768,
                   max_num_seqs=32, max_num_batched_tokens=8192,
                   max_num_decoding_tokens=256, max_num_prefilling_tokens=8192,
                   engine_type=EngineType.NORMAL)
    execution = ExecutionSettings(draft_tokens=7, batch_invariant=False,
        yoco_prefill_truncation="off", index_memory_limit=64 << 20,
        sampler_topk=0, confidence_temperature=1.)

    def reject_environment():
        raise AssertionError("prepared attention attempted to select execution from the environment")

    patch = pytest.MonkeyPatch()
    patch.setattr(ExecutionSettings, "from_engine_environment", staticmethod(reject_environment))
    try:
        with torch.no_grad():
            sources = parameters.model.module.module
            topology = build_shared_layer_topology(cfg, 43)
            layers = [V4Attention(sources[i + 1].self_attn, i, cfg, topology, execution=execution)
                      for i in range(40)]
            layers.extend(DFlashAttention(sources[42 + i].block.self_attn, 40 + i, cfg, execution=execution)
                          for i in range(3))
            assert len(layers) == 43
            for layer in layers:
                create_deepseek_decode_layer(layer_group_key(layer, None), layer.layer_idx)
                assert layer.execution is execution
                assert all(value.is_cuda for value in layer.parameters())
                assert layer.rotary_emb.cos_sin_cache.is_cuda
                if (indexer := getattr(layer, "index_attn", None)) is not None:
                    assert indexer.compute_chunk_size > 0
            assert not torch.distributed.is_initialized()
        yield layers, cfg
    finally:
        torch.cuda.synchronize()
        patch.undo()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sparse,rows,prefill", [
    (False, 1, False), (False, 8, False), (False, 129, True), (False, 8192, True),
    (True, 1, False), (True, 8, False), (True, 129, True), (True, 8192, True),
])
@torch.no_grad()
def test_complete_reference_attention_tree(reference_attention_tree, compiled, sparse, rows, prefill):
    layers, cfg = reference_attention_tree
    if sparse:
        _replay_sparse_attention(layers[2:40], cfg, width=rows, prefill=prefill, compiled=compiled)
    else:
        _replay_attention(layers[0], cfg, rows=rows, prefill=prefill, compiled=compiled)


def _replay_attention(layer, cfg, *, rows, prefill, compiled):
    from deepinfer.model_executor.input_metadata import InputMetadataV4
    from deepinfer.model_executor.models.v4_modules.cross_layer import CrossLayerRuntime
    from deepinfer.model_executor.utils import Movable
    from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheLayout, ReferenceCacheStorage

    geometry = ReferenceCacheLayout.from_config(cfg, verification_width=8)
    storage = ReferenceCacheStorage.allocate(geometry, 2, 2, "cuda:0")
    cache = storage.cache_views()[0]
    x = torch.randn(rows, cfg.hidden_size, dtype=torch.bfloat16, device="cuda") * .1
    output = torch.empty_like(x)
    positions = torch.arange(rows, dtype=torch.int32, device="cuda")
    slots = geometry.window_tokens + positions % geometry.window_tokens
    if prefill:
        slots = torch.where(positions >= rows - geometry.window_tokens, slots, -1)
    metadata = InputMetadataV4(is_prompt=prefill,
        input_ids=torch.ones(rows, dtype=torch.int64, device="cuda"),
        positions=positions.clone(),
        valid_token_mask=torch.ones(rows, dtype=torch.bool, device="cuda"),
        verification_width=0 if prefill else 8, num_reqs=1,
        state_block_idx=torch.ones(1 if prefill else rows, dtype=torch.int32, device="cuda"),
        context_lens=positions + 1,
        query_indices=torch.zeros(rows, dtype=torch.int32, device="cuda"),
        slot_mapping_store_swa=slots,
        kv_ca_index_offset_len={})
    if prefill:
        metadata.total_q = rows
        metadata.cu_seqlens_q = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
        metadata.cu_seqlens_k = metadata.cu_seqlens_q.clone()
        metadata.cu_compressed_seqlens_q = {}
        metadata.positions_c = {}
        metadata.ke = {}
        metadata.total_k_swa = 128
        metadata.load_segments_swa = torch.tensor([[-1, 0, 128, 0]], dtype=torch.int32, device="cuda")
        metadata.kv_swa_full_index_pivots = torch.stack((
            torch.zeros_like(positions), torch.zeros_like(positions),
            128 + (positions - 127).clamp_min(0), (positions + 1).clamp_max(128)), dim=1)
    else:
        metadata.kv_swa_index_offset_len = torch.stack((
            torch.full_like(positions, geometry.window_tokens), positions + 1), dim=1)

    def run():
        for field in (*metadata.get_clean_list(), "kv_swa_full_index", "compressed_batch_mapping", "klen"):
            setattr(metadata, field, None)
        output.copy_(layer(Movable(x), cache, metadata, Movable(x), CrossLayerRuntime()))

    captured = compile_forward(run) if compiled else run
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        captured()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured()
    try:
        cache.swa_cache.zero_()
        for step in range(3):
            x.neg_()
            if prefill:
                cache.swa_cache.zero_()
            else:
                metadata.positions.copy_(positions + step * rows)
                metadata.context_lens.copy_(metadata.positions + 1)
                metadata.slot_mapping_store_swa.copy_(geometry.window_tokens + metadata.positions)
                metadata.kv_swa_index_offset_len[:, 1].copy_(metadata.context_lens)
            before = cache.swa_cache.clone()
            run()
            expected, expected_cache = output.clone(), cache.swa_cache.clone()
            cache.swa_cache.copy_(before)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            assert torch.equal(cache.swa_cache, expected_cache)
            assert not storage.backing[0].count_nonzero()
    finally:
        torch.cuda.synchronize()
        graph.reset()


def _replay_sparse_attention(layers, cfg, *, width, prefill, compiled):
    from deepinfer.model_executor.models.v4_modules.cross_layer import CrossLayerRuntime
    from deepinfer.model_executor.utils import Movable
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheLayout, ReferenceCacheStorage
    from hyperflare_recipes.recipes.deepseek41.reference_metadata import ReferenceDecodeMetadata, ReferencePrefillMetadata

    geometry = ReferenceCacheLayout.from_config(cfg, verification_width=8)
    blocks = cfg.runtime_max_model_len // geometry.page_tokens if prefill else 1
    capacity = width if prefill else width + 1
    storage = ReferenceCacheStorage.allocate(geometry, 2, blocks + 2, "cuda:0")
    caches = storage.cache_views()
    builder = (ReferencePrefillMetadata if prefill else ReferenceDecodeMetadata)(cfg, geometry)
    source = Metadata.allocate(capacity, 1, blocks, "cuda", max_query_tokens=width)
    starts = torch.zeros(1, dtype=torch.int64, device="cuda")
    lengths = torch.tensor([width], dtype=torch.int32, device="cuda")
    states = torch.ones(1, dtype=torch.int64, device="cuda")
    pages = (torch.arange(blocks, dtype=torch.int32, device="cuda").flip(0) + 2)[None].contiguous()
    tokens = torch.ones((1, width), dtype=torch.int64, device="cuda")
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    source.set(starts, lengths, states, pages, tokens, live)
    x = torch.randn(capacity, cfg.hidden_size, dtype=torch.bfloat16, device="cuda") * .1
    outputs = torch.empty(len(layers), capacity, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")

    def run():
        metadata = builder(source)
        shared = CrossLayerRuntime()
        for index, layer in enumerate(layers):
            if layer.is_first_yoco_consumer_layer:
                shared.shared_hidden = x
            outputs[index].copy_(layer(Movable(x), caches[layer.layer_idx], metadata, Movable(x), shared))

    captured = compile_forward(run) if compiled else run
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        captured()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured()
    try:
        storage.clear_for_preparation()
        start = 0
        for step in range(3):
            length = (width, width - 1, 1)[step] if prefill else width
            starts.fill_(start)
            lengths.fill_(length)
            source.set(starts, lengths, states, pages, tokens, live)
            x.normal_().mul_(.1)
            x[length:].fill_(float("nan"))
            owners = (storage.backing, *storage.pages.values())
            before = [value.clone() for value in owners]
            run()
            expected = outputs.clone()
            expected_state = [value.clone() for value in owners]
            for value, saved in zip(owners, before, strict=True):
                value.copy_(saved)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
            assert torch.isfinite(outputs[:, :length]).all()
            torch.testing.assert_close(outputs[:, :length], expected[:, :length], atol=0, rtol=0)
            for value, saved in zip(owners, expected_state, strict=True):
                assert torch.equal(value, saved)
            assert not storage.backing[0].count_nonzero()
            start += length
        lengths.zero_()
        source.set(starts, lengths, states, pages, tokens, live)
        x.fill_(float("nan"))
        before = [value.clone() for value in owners]
        graph.replay()
        torch.cuda.synchronize()
        for value, saved in zip(owners, before, strict=True):
            assert torch.equal(value, saved)
    finally:
        torch.cuda.synchronize()
        graph.reset()
