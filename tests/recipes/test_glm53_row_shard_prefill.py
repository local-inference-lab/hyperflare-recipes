# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for GLM prefill row ownership, 2-microbatch pipelining, and row-shard MLA."""

from types import SimpleNamespace

import pytest
import torch
from b12x.comm import pcie

from hyperflare_recipes.components.mhc import MHCScratch
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.recurrent import DeltaMetadata
from hyperflare_recipes.recipes.glm53 import dcp, recipe, stack


class DmaRuntime:
    def __init__(self, *, scatter=True, gather=True, surfaces=True):
        self.world_size = 4
        self.max_bytes = 64 << 20
        if surfaces:
            self.should_reduce_scatter = lambda source: scatter
            self.should_all_gather = lambda source: gather


class FakeArena:
    def __init__(self, overlap=True):
        self.overlap = overlap


def _branch(full):
    return SimpleNamespace(
        output=torch.empty_like(full),
        index=SimpleNamespace(page_table=torch.empty(1, 1)),
        rows=lambda n: SimpleNamespace(output=full[:n]),
    )


def prepare(monkeypatch, tokens, runtime, *, enabled=True, mode="prefill", rank=0, world=4,
            feedback=None, scoring=None, vision=None, workspace=None, row_shard_attention=False):
    monkeypatch.setattr(pcie, "DmaAllReduce", DmaRuntime)
    if workspace is not None:
        monkeypatch.setattr(stack, "ScratchArena", FakeArena)
    full = torch.empty((tokens, 4096), dtype=torch.bfloat16)
    hc = MHCScratch.allocate(tokens, 4096, "cpu")
    branch = _branch(full)
    scratch = stack.DecoderScratch(embedding=full, residual=torch.empty_like(hc.residual),
        selected_residual=hc.residual[:1], hidden=full[:1], local_logits=None, logits=None,
        attention_hc=hc, mlp_hc=MHCScratch.allocate(tokens, 4096, "cpu"), kda=branch, mla=branch,
        dense=branch, moe=branch, attention_output=torch.empty_like(full), mlp_output=torch.empty_like(full))
    module = SimpleNamespace(block_type="linear_attention")
    language = SimpleNamespace(layers=[module] * 45, embed_tokens=SimpleNamespace(weight=None),
        hc_head=lambda value: value[:, :, 0], norm=SimpleNamespace(weight=None, variance_epsilon=1e-6))
    weights = SimpleNamespace(tree=SimpleNamespace(model=SimpleNamespace(language_model=language)),
        layers=[SimpleNamespace(attention=None, mlp=None, residual=None)] * 45, head=None,
        shard=TensorParallelShard(rank, world), replicated_attention=None)
    recurrent = DeltaMetadata.allocate(1, torch.device("cpu"))
    recurrent.num_seqs.fill_(1)
    recurrent.num_tokens.fill_(tokens)
    recurrent.query_start_loc[1] = tokens
    metadata = SimpleNamespace(token_ids=torch.arange(tokens), sample_indices=torch.zeros(1, dtype=torch.int64),
        recurrent=recurrent, indexed=None)
    monkeypatch.setattr(stack, "PreparedEmbedding", lambda *a, **kw:
        lambda *, token_ids, num_rows: token_ids[:, None].clone())
    monkeypatch.setattr(stack, "PreparedVocabularyProjection", lambda *a, **kw: lambda hidden: hidden)

    class Layer:
        def __init__(self, module, attn, mlp, weights, source, output, scratch, *args,
                     gathered_input, carry, pipelined=False, **kw):
            self.module, self.gathered_input, self.pipelined = module, gathered_input, pipelined
            if pipelined:
                half = tokens // 2
                rows = half // world
                assert source.shape == ((rows, 4, 4096) if carry is None else (rows, 4096))
                assert output.shape == (rows, 4, 4096)
                assert gathered_input.shape == (half, 4096)
                assert scratch.attention.output.shape == scratch.mlp.output.shape == (half, 4096)
            else:
                rows = tokens if gathered_input is None else tokens // world
                assert source.shape == ((rows, 4, 4096) if carry is None else (rows, 4096))
                assert output.shape == (rows, 4, 4096)
                for local in (scratch.attention_hc, scratch.mlp_hc):
                    assert local.normalized.shape == (rows, 4096)
                    if gathered_input is not None:
                        assert gathered_input.shape == (tokens, 4096)
                        assert gathered_input.untyped_storage().data_ptr() != local.normalized.untyped_storage().data_ptr()
                assert scratch.attention.output.shape == scratch.mlp.output.shape == (tokens, 4096)

        def __call__(self, value, metadata, carry, **kwargs):
            return value, carry

        def forward_pipelined(self, incoming, metadata, **kwargs):
            return incoming, None

    monkeypatch.setattr(stack, "PreparedGlmLayer", Layer)
    return stack.PreparedDecoder(weights, scratch, SimpleNamespace(layers=[None] * 45), metadata,
        mode=mode, runtime=runtime, vocabulary_runtime=None, coordinator=None, session=None,
        workspace=workspace, name="text", row_shard_prefill=enabled,
        row_shard_attention=row_shard_attention, feedback=feedback, scoring=scoring, vision=vision)


@pytest.mark.parametrize("tokens,runtime,options", [
    (1024, DmaRuntime(), {"enabled": False}), (1024, object(), {}), (1024, DmaRuntime(surfaces=False), {}),
    (512, DmaRuntime(), {}), (1025, DmaRuntime(), {}), (1, DmaRuntime(), {}), (1028, DmaRuntime(), {}),
    (1024, DmaRuntime(), {"scoring": SimpleNamespace(prepare=lambda *a, **kw: None)}),
    (1024, DmaRuntime(scatter=False), {}), (1024, DmaRuntime(gather=False), {}),
    (1024, DmaRuntime(), {"mode": "decode"}), (1024, DmaRuntime(), {"world": 2}),
])
def test_ineligible_decoder_keeps_full_rows(monkeypatch, tokens, runtime, options):
    decoder = prepare(monkeypatch, tokens, runtime, **options)
    assert decoder.shard_rows is decoder.final_gather is None
    assert not decoder.pipelined
    assert all(layer.gathered_input is None for layer in decoder.layers)


@pytest.mark.parametrize("output", ["sample", "feedback"])
def test_global_sample_after_local_head_and_norm(monkeypatch, output):
    tokens, length = 1024, 773
    rows = tokens // 4
    labels = torch.arange(tokens)[:, None]
    labels[length:] = 0
    expected = labels * 2
    expected[length - 1] += 2
    peers = expected.split(rows)

    def inject(metadata, embedded):
        assert embedded.shape == (tokens, 1)
        embedded[length - 1] += 1

    vision = SimpleNamespace(inject=inject)

    def norm(source, weight, target, eps, **kwargs):
        assert source.shape == (rows, 1)
        target.copy_(source * 2)

    class Gather(stack.PreparedGatherShards):
        def __init__(self, source, destination, *, runtime, name, **kwargs):
            assert name == "text.final_gather"
            assert source.shape == (rows, 4096) and destination.shape == (tokens, 4096)
            self.rank = runtime.rank

        def __call__(self, local, dependencies=()):
            assert torch.equal(local, peers[self.rank])
            return torch.cat((*peers[:self.rank], local, *peers[self.rank + 1:]))

    monkeypatch.setattr(stack, "rms_norm", norm)
    monkeypatch.setattr(stack, "PreparedGatherShards", Gather)
    for rank in range(4):
        runtime = DmaRuntime()
        runtime.rank = rank
        decoder = prepare(monkeypatch, tokens, runtime, rank=rank,
            feedback=True if output == "feedback" else None, vision=vision)
        assert not decoder.pipelined
        assert torch.equal(decoder.shard_rows, torch.arange(rank * rows, (rank + 1) * rows))
        metadata = decoder.metadata
        metadata.token_ids.copy_(labels[:, 0])
        metadata.recurrent.num_tokens.fill_(length)
        metadata.sample_indices.fill_(length - 1)
        hidden, logits, auxiliary, _ = decoder(metadata)
        assert torch.equal(hidden, expected[length - 1:length]) and torch.equal(logits, hidden)
        if output == "sample":
            assert auxiliary is None
        else:
            assert torch.equal(auxiliary, expected)


@pytest.mark.parametrize("tokens,workspace,expected_pipelined", [
    (1024, FakeArena(overlap=True), False),
    (2048, FakeArena(overlap=False), False),
    (2048, None, False),
    (2048, FakeArena(overlap=True), True),
    (4096, FakeArena(overlap=True), True),
    (8192, FakeArena(overlap=True), True),
])
def test_pipelined_eligibility(monkeypatch, tokens, workspace, expected_pipelined):
    class DummyGather(stack.PreparedGatherShards):
        def __init__(self, *a, **kw):
            self.name = kw.get("name")

    monkeypatch.setattr(stack, "PreparedGatherShards", DummyGather)
    decoder = prepare(monkeypatch, tokens, DmaRuntime(), workspace=workspace)
    assert decoder.pipelined is expected_pipelined
    if expected_pipelined:
        assert decoder.half_tokens == tokens // 2
        assert decoder.final_gather_a is not None and decoder.final_gather_b is not None
        assert all(layer.pipelined for layer in decoder.layers)


@pytest.mark.parametrize("output", ["sample", "feedback"])
def test_pipelined_global_sample_reconstruction(monkeypatch, output):
    tokens, length = 2048, 1773
    half = tokens // 2
    rows = half // 4
    labels = torch.arange(tokens)[:, None]
    labels[length:] = 0
    expected = labels * 2
    expected[length - 1] += 2
    peers_a = expected[:half].split(rows)
    peers_b = expected[half:].split(rows)

    def inject(metadata, embedded):
        assert embedded.shape == (tokens, 1)
        embedded[length - 1] += 1

    vision = SimpleNamespace(inject=inject)

    def norm(source, weight, target, eps, **kwargs):
        assert source.shape == (rows, 1)
        target.copy_(source * 2)

    class Gather(stack.PreparedGatherShards):
        def __init__(self, source, destination, *, runtime, name, **kwargs):
            assert name in ("text.final_gather.0", "text.final_gather.1")
            assert source.shape == (rows, 4096) and destination.shape == (half, 4096)
            self.rank = runtime.rank
            self.peers = peers_a if name.endswith(".0") else peers_b

        def __call__(self, local, dependencies=()):
            assert torch.equal(local, self.peers[self.rank])
            return torch.cat((*self.peers[:self.rank], local, *self.peers[self.rank + 1:]))

    monkeypatch.setattr(stack, "rms_norm", norm)
    monkeypatch.setattr(stack, "PreparedGatherShards", Gather)
    for rank in range(4):
        runtime = DmaRuntime()
        runtime.rank = rank
        decoder = prepare(monkeypatch, tokens, runtime, rank=rank, workspace=FakeArena(overlap=True),
            feedback=True if output == "feedback" else None, vision=vision)
        assert decoder.pipelined
        assert torch.equal(decoder.shard_rows_a, torch.arange(rank * rows, (rank + 1) * rows))
        assert torch.equal(decoder.shard_rows_b, torch.arange(half + rank * rows, half + (rank + 1) * rows))
        metadata = decoder.metadata
        metadata.token_ids.copy_(labels[:, 0])
        metadata.recurrent.num_tokens.fill_(length)
        metadata.sample_indices.fill_(length - 1)
        hidden, logits, auxiliary, _ = decoder(metadata)
        assert torch.equal(hidden, expected[length - 1:length]) and torch.equal(logits, hidden)
        if output == "sample":
            assert auxiliary is None
        else:
            assert torch.equal(auxiliary, expected)


@pytest.mark.parametrize("live,fresh,ckpt,exp_a,exp_b", [
    (8192, True, 0, (1, 4096, True, 0, 7, 7, 0, 0), (1, 4096, False, 7, 7, 7, 0, 0)),
    (8192, False, 2048, (1, 4096, False, 7, 7, 7, 2048, 9), (1, 4096, False, 7, 7, 7, 0, 0)),
    (8192, False, 4096, (1, 4096, False, 7, 7, 7, 4096, 9), (1, 4096, False, 7, 7, 7, 0, 0)),
    (8192, False, 6144, (1, 4096, False, 7, 7, 7, 0, 0), (1, 4096, False, 7, 7, 7, 2048, 9)),
    (3000, True, 1024, (1, 3000, True, 0, 7, 7, 1024, 9), (0, 0, False, 0, 0, -1, 0, 0)),
    (5000, False, 4500, (1, 4096, False, 7, 7, 7, 0, 0), (1, 904, False, 7, 7, 7, 404, 9)),
    (0, False, 0, (0, 0, False, 7, 7, -1, 0, 0), (0, 0, False, 0, 0, -1, 0, 0)),
])
def test_split_recurrent_metadata(live, fresh, ckpt, exp_a, exp_b):
    m = DeltaMetadata.allocate(1, torch.device("cpu"))
    m.num_seqs.fill_(1 if live > 0 else 0)
    m.num_tokens.fill_(live)
    m.query_start_loc[1] = live
    m.convolution_slots.fill_(7)
    m.fresh.fill_(fresh)
    m.initial_state_indices.fill_(0 if fresh else 7)
    m.final_state_indices.fill_(7)
    m.checkpoint_offsets.fill_(ckpt)
    m.checkpoint_state_indices.fill_(9 if ckpt else 0)
    a, b = stack._split_recurrent_metadata(m, 4096)
    for half_meta, (seqs, toks, fr, init_s, fin_s, conv_s, c_off, c_slot) in ((a, exp_a), (b, exp_b)):
        assert half_meta.num_seqs.item() == seqs
        assert half_meta.num_tokens.item() == toks
        assert half_meta.query_start_loc.tolist() == [0, toks]
        assert bool(half_meta.fresh.item()) is fr
        assert half_meta.initial_state_indices.item() == init_s
        assert half_meta.final_state_indices.item() == fin_s
        assert half_meta.convolution_slots.item() == conv_s
        assert half_meta.checkpoint_offsets.item() == c_off
        assert half_meta.checkpoint_state_indices.item() == c_slot


def test_replicated_attention_bytes_and_pool_gather_counts():
    assert recipe.replicated_attention_bytes(1) == 0
    expected_mla = 11 * (16384 * 1536 + 32768 * 512 + 4096 * 16384) * 2
    expected_kda = 34 * (
        (4 * 8192 * 4096 + 64 * 4096 + 128 * 4096 + 2 * 8192 * 128 + 3 * 8192 * 4) * 2
        + (8192 + 64) * 4
    )
    assert recipe.replicated_attention_bytes(4) == expected_mla + expected_kda
    for end_val in (0, 1, 3, 4, 7, 16, 17, 1023, 131072):
        counts = dcp.pool_gather_counts(torch.tensor([end_val], dtype=torch.int64), 4, 8192)
        assert counts.shape == (4,) and counts.dtype == torch.int64
        assert (counts % 16 == 0).all()
        assert (counts <= 8192 * 132).all()
        total_pools = end_val // 4
        for r in range(4):
            owned = total_pools // 4 + min(max(total_pools % 4 - r, 0), 1)
            assert counts[r].item() >= owned * 132


def test_contiguous_shard_reconstruction_and_deinterleave():
    tokens = 2048
    world = 4
    rows = tokens // (2 * world)
    expected = torch.arange(tokens, dtype=torch.int64)[:, None].expand(-1, 4)
    a_parts = [expected[r * 2 * rows : r * 2 * rows + rows] for r in range(world)]
    b_parts = [expected[r * 2 * rows + rows : (r + 1) * 2 * rows] for r in range(world)]
    norm_a = torch.cat(a_parts, dim=0)
    norm_b = torch.cat(b_parts, dim=0)
    reconstructed = torch.stack(
        (norm_a.view(world, rows, -1), norm_b.view(world, rows, -1)), dim=1
    ).reshape(tokens, -1)
    assert torch.equal(reconstructed, expected)





def test_quarter_mlp_prefill_threshold():
    assert stack.QUARTER_MLP_PREFILL_TOKENS == 10240
    for tok in (10240, 12288, 14336, 16384):
        assert tok >= stack.QUARTER_MLP_PREFILL_TOKENS
        assert tok % (4 * 128 * 4) == 0
    for tok in (2048, 4096, 6144, 8192):
        assert tok < stack.QUARTER_MLP_PREFILL_TOKENS


def test_glm53_parameters_gate_row_shard_prefill_and_attention():
    from dataclasses import replace

    from hyperflare_recipes.recipes.glm53.bootstrap import PreparationConfig
    from ignition.errors import UsageError
    from ignition.parameters import ParameterContext
    from ignition.planning import Gpu

    sm120 = tuple(
        Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i}", 96 << 30, 90 << 30, (12, 0))
        for i in range(4)
    )
    assert recipe.RECIPE.parameters.row_shard_prefill_world_sizes == (4,)
    assert PreparationConfig().row_shard_prefill is False
    assert PreparationConfig().row_shard_attention is False

    enabled = replace(recipe.RECIPE.parameters, row_shard_prefill=True, row_shard_attention=True)
    cfg = PreparationConfig(parameters=enabled, devices=(0, 1, 2, 3))
    assert cfg.row_shard_prefill is True and cfg.row_shard_attention is True

    ctx_tp4 = ParameterContext(recipe=recipe.RECIPE, checkpoint=recipe.RECIPE.checkpoints[1], gpus=sm120, world_size=4)
    enabled.validate(ctx_tp4)
    assert enabled.filter_world_sizes((1, 2, 4), ctx_tp4) == (4,)
    assert enabled.adjust_weights_bytes(1000, ctx_tp4) == 1000 + recipe.replicated_attention_bytes(4)

    ctx_tp2 = ParameterContext(recipe=recipe.RECIPE, checkpoint=recipe.RECIPE.checkpoints[1], gpus=sm120[:2], world_size=2)
    with pytest.raises(UsageError, match="not supported at TP2"):
        enabled.validate(ctx_tp2)

    attn_only = replace(recipe.RECIPE.parameters, row_shard_prefill=False, row_shard_attention=True)
    with pytest.raises(UsageError, match="--row-shard-attention requires --row-shard-prefill"):
        attn_only.validate(ctx_tp4)
    with pytest.raises(ValueError, match="row_shard_attention requires row_shard_prefill"):
        PreparationConfig(parameters=attn_only, devices=(0, 1, 2, 3))
