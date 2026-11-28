# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference hashing, native FP8 delivery and gating through captured execution.

The native delivery fixture uses a small synthetic hash space and real checkpoint
table rows, projections and gate weights. Full checkpoint hash geometry is checked
separately without loading the complete embedding tables.
"""

import os
from pathlib import Path
import time

import pytest
import torch
from safetensors import safe_open

from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.runtime_core import FileSource
from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint
from hyperflare_recipes.recipes.deepseek41.reference_engram import ReferenceEngramDelivery

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.fixture(scope="module")
def checkpoint_engram():
    pytest.importorskip("deepinfer")
    from deepinfer.engine.engine_type import EngineType
    from deepinfer.model_executor.models.v4_modules.builder import V41Builder
    from deepinfer.model_executor.models.v4_modules.engram import Engram
    from deepinfer.model_executor.models.v4_modules.engram_parameters import EngramEmbedding
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires the DeepSeek V4.1 CSF checkpoint")
    checkpoint = ReferenceCheckpoint(Path(snapshot))
    cfg = V41Builder(checkpoint.model_config()).gpt_cfg
    cfg.engine_type, cfg.max_num_prefilling_tokens = EngineType.PROMPT, 8192
    PG.initialize_local()
    layers, projections, samples = [], [], {}
    for layer in cfg.engram.engram_layer_ids:
        with torch.device("meta"):
            source = EngramEmbedding(cfg, layer)
        prefix = f"layers.{layer}.engram."
        weight = torch.nn.Parameter(checkpoint.read(prefix + "wkv.weight").cuda(), requires_grad=False)
        weight.sf = checkpoint.read(prefix + "wkv.scale").float().cuda()
        source.fused_kv_projection.weight = weight
        source.weight_hidden = torch.nn.Parameter(checkpoint.read(prefix + "q_weight").cuda(), requires_grad=False)
        source.weight_embed = torch.nn.Parameter(checkpoint.read(prefix + "k_weight").cuda(), requires_grad=False)
        module = Engram(source, layer, cfg)
        layers.append(module)
        projections.append(module.fused_kv_projection)
        planes = []
        for suffix in ("weight", "scale"):
            name = prefix + "embed." + suffix
            path = checkpoint.source.snapshot / checkpoint.source.tensors[name].shard
            with safe_open(path, framework="pt", device="cpu") as reader:
                planes.append(reader.get_slice(name)[:512].view(torch.uint8).clone())
        samples[layer] = planes
    return checkpoint, cfg, layers, projections, samples


@torch.no_grad()
def test_checkpoint_hash_geometry_matches_reference(checkpoint_engram):
    import tile_kernels
    from deepinfer.model_executor.models.v4_modules.engram import hash_ngrams
    from deepinfer.model_executor.models.v4_modules.engram_metadata import NgramHashMapping

    _, cfg, _, _, _ = checkpoint_engram
    mapping = NgramHashMapping(cfg.engram)
    for rows in (1, 129, 8192):
        tokens = torch.arange(rows * 4, dtype=torch.int32, device="cuda").view(rows, 4) % 99092
        images = torch.arange(rows, device="cuda") % 5 == 2
        tokens[::7, 2:] = -1
        valid = torch.arange(rows, device="cuda") % 3 != 1
        args = (tokens, mapping.ngram_multipliers, mapping.ngram_vocab_sizes, mapping.ngram_offsets, images)
        expected = tile_kernels.torch.engram.engram_hash_ref(*args)
        expected.masked_fill_(~valid[None, :, None], -1)
        compiled = compile_forward(lambda: hash_ngrams(*args, valid))
        torch.testing.assert_close(compiled(), expected, rtol=0, atol=0)


@torch.no_grad()
def test_checkpoint_table_binding_is_compact(checkpoint_engram, megamoe_core_storage):
    checkpoint, cfg, layers, _, _ = checkpoint_engram
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.zeros(1, dtype=torch.int64, device="cuda")
    delivery = ReferenceEngramDelivery.from_checkpoint(checkpoint, cfg, capacity=1,
        sequence=sequence, frame=frame, host_allocator=megamoe_core_storage.host_allocate)
    try:
        assert tuple(delivery.sources) == tuple(layer.layer_id for layer in layers)
        for layer, pair in delivery.sources.items():
            for source, suffix in zip(pair, ("weight", "scale"), strict=True):
                tensor = checkpoint.source.tensors[f"layers.{layer}.engram.embed.{suffix}"]
                assert source.path == checkpoint.source.snapshot / tensor.shard
                assert source.offset == tensor.offset
                assert source.nbytes == tensor.shape[0] * tensor.shape[1]
        assert sum(value.nbytes for value in delivery.buffers().values()) < 65536
        assert all(layer.ngram_embedding.weight.is_meta for layer in layers)
        with pytest.raises(ValueError, match="table order"):
            delivery.bind(layers[::-1])
        with pytest.raises(ValueError, match="geometry"):
            delivery.fetch(torch.zeros((2, 2, 24), dtype=torch.int32, device="cuda"))
    finally:
        for box in delivery.mailboxes.values():
            box.close()


@pytest.mark.parametrize("tokens", [1, 192, 8192])
@torch.no_grad()
def test_native_reference_engram(checkpoint_engram, tokens, tmp_path, megamoe_core_storage):
    import flare_heliopause
    import tile_kernels
    from deepinfer.model_executor.models.v4_modules.engram import hash_ngrams

    _, cfg, layers, projections, samples = checkpoint_engram
    device = torch.device("cuda:0")
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    sources = {}
    for layer, planes in samples.items():
        pair = []
        for index, plane in enumerate(planes):
            path = tmp_path / f"{layer}-{index}"
            path.write_bytes(bytes(37) + plane.numpy().tobytes())
            pair.append(FileSource(path, 37, plane.nbytes))
        sources[layer] = tuple(pair)
    delivery = ReferenceEngramDelivery(sources, vocabulary=cfg.engram.tokenizer_vocab_size, heads=24, width=256,
        capacity=8192, sequence=sequence, frame=frame, host_allocator=megamoe_core_storage.host_allocate)
    # A smaller graph must never fetch stale requests beyond its token capacity.
    for _, _, requests in delivery.planes.values():
        requests.fill_(1000000)
    # Each parametrized case binds its own event and compact row planes.
    for layer, projection in zip(layers, projections, strict=True):
        layer.fused_kv_projection = projection
    delivery.bind(layers)
    ngram_tokens = torch.zeros((tokens, 4), dtype=torch.int32, device=device)
    images = torch.zeros(tokens, dtype=torch.bool, device=device)
    valid = torch.zeros_like(images)
    source = torch.randn((tokens, 4, 5120), dtype=torch.bfloat16, device=device) * .1
    output = torch.empty_like(source)
    records = torch.zeros((2, 1, 32), dtype=torch.int64, device=device)
    multipliers = torch.tensor([[13, 29, 37, 41], [17, 31, 43, 53]], dtype=torch.int64, device=device)
    vocab = torch.full((2, 3, 8), 13, dtype=torch.int32, device=device)
    offsets = (torch.arange(24, dtype=torch.int32, device=device) * 13).expand(2, 24).contiguous()

    def forward():
        hashes = hash_ngrams(ngram_tokens, multipliers, vocab, offsets, images, valid)
        hooks = delivery.fetch(hashes)
        hidden = source.clone()
        for layer in layers:
            hidden = layer(hidden, hooks[layer.layer_id], images)
        output.copy_(hidden)
        records.zero_()
        records[:, 0, 0] = 1
        records[:, 0, 1].copy_(sequence[0].expand(2))
        records[:, 0, 4] = -1
        records[:, 0, 5] = 1
        records[:, 0, 12] = -1
        records[:, 0, 15] = 1
        records[:, 0, 16:] = -1

    graph, worker = None, None
    try:
        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        execution = PreparedExecution(device=0, graphs={"prefill": graph}, buffers=dict(
            ngram_tokens=ngram_tokens, images=images, valid=valid, output=output,
            frame_index=frame, submission_sequence=sequence, completion_records=records, **delivery.buffers()),
            signatures={"prefill": GraphSignature("prefill", 1, tokens, ("ngram_tokens", "images", "valid"))},
            max_active=1, resources=(delivery, layers, compiled), export_events=delivery.export_events())
        resident = flare_heliopause.EmbeddingTables(delivery.resident_catalog())
        manifest = execution.manifest()
        manifest["engram"] = resident.bind(manifest, execution.export_events)
        worker = flare_heliopause.GraphExecutor((execution, resident), manifest)
        for pair in sources.values():
            for plane in pair:
                plane.path.unlink()
        for generation in range(4):
            ids = (torch.arange(tokens * 4, dtype=torch.int32).view(tokens, 4) * 19 + generation) % 99092
            ids[::7, 2:] = -1
            active = torch.arange(tokens) % 3 != generation % 3
            image = torch.arange(tokens) % 5 == generation
            if generation == 2:
                active.zero_()
            if generation == 3:
                image.fill_(True)
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            fence = worker.submit(0, [(name, 0, value.numpy().tobytes()) for name, value in
                (("ngram_tokens", ids), ("images", image), ("valid", active))])
            assert fence is not None
            completion, deadline = None, time.monotonic() + 30
            while completion is None and time.monotonic() < deadline:
                completion = worker.poll()
                if completion is None:
                    time.sleep(.001)
            assert completion is not None and completion.fence == fence
            assert completion.readiness_wait_ns is not None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            expected_hashes = tile_kernels.engram.engram_hash(ids.cuda(), multipliers, vocab, offsets, image.cuda())
            expected_hashes.masked_fill_(~active.cuda()[None, :, None], -1)
            expected = source.clone()
            for index, (layer, projection) in enumerate(zip(layers, projections, strict=True)):
                hashes = expected_hashes[index].cpu().long()
                assert torch.equal(delivery.planes[layer.layer_id][2][:tokens].cpu(), hashes)
                values, scales = (plane[hashes.clamp_min(0)].clone() for plane in samples[layer.layer_id])
                values[hashes < 0] = 0
                scales[hashes < 0] = 0
                values = values.view(torch.float8_e4m3fn).flatten(1).cuda()
                sf = scales.flatten(1).view(torch.int32).cuda()
                packed = torch.empty((sf.shape[1], (tokens + 3) // 4 * 4), dtype=torch.int32, device=device).mT[:tokens]
                packed.copy_(sf)
                kv = projection((values, packed)).view(tokens, 5, 5120)
                kv.masked_fill_(~(hashes >= 0).any(dim=1).cuda()[:, None, None], 0)
                expected = tile_kernels.engram.engram_gate_fwd(expected, kv, layer.weight_fused,
                    layer.eps, layer.clamp_value, save_for_backward=False, image_token_mask=image.cuda(), inplace=True)[0]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(output[~active.cuda() | image.cuda()],
                                       source[~active.cuda() | image.cuda()], rtol=0, atol=0)
    finally:
        if worker is not None:
            worker.close()
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        for box in delivery.mailboxes.values():
            box.close()
