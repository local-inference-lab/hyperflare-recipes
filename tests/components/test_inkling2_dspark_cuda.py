# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared recurrent proposals retain conditional distributions across CUDA replay."""

import os
import json
from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.recurrent_proposal import PreparedRecurrentProposalHead, corrected_logits, recurrent_gate
from ignition.components.sampling import DeviceRequestState, RUNNING
from hyperflare_recipes.components.speculation import DRAFT_RNG_DOMAIN
from hyperflare_recipes.components.vocabulary_sampling import PreparedVocabularySoftmax
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.preparation.scratch import ScratchArena
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.inkling2.dspark import DraftAttention, DraftScratch, PreparedContext, PreparedDraft
from hyperflare_recipes.components.weights import MLPWeights
from hyperflare_recipes.recipes.inkling2.dspark_state import mix_features, rotary
from hyperflare_recipes.recipes.inkling2.dspark_weights import DraftWeights
from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from tests.components.test_markov_sampling_cuda import philox


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("batch,depth", [(1, 7), (4, 7), (4, 1)])
@torch.no_grad()
def test_sequential_recurrent_draws_record_actual_q_and_reset_state(batch, depth):
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(422)
    hidden, rank, vocabulary, local = 128, 256, 513, 576
    def values(*shape, denominator=64):
        return (torch.randint(-2, 3, shape, generator=generator).float() / denominator).to(device, torch.bfloat16)
    weight = values(3 * rank, 2 * rank + 2 * hidden)
    recurrent = SimpleNamespace(embedding=values(vocabulary, rank), state_weight=weight[:, :2 * rank].contiguous(),
        context_weight=weight[:, 2 * rank:].contiguous(), bias=values(3 * rank).float(),
        head=values(local, rank), head_bias=values(local), softcap=30.)
    features, context = values(batch, depth, hidden, denominator=8), values(batch, hidden, denominator=8)
    backing = values(batch, 8, local, denominator=2)
    base = backing[:, 1:depth + 1]
    scores = torch.empty_like(backing)[:, 1:depth + 1]
    q = torch.empty(batch, depth, local, dtype=torch.float32, device=device)
    valid = torch.empty(batch, depth, dtype=torch.bool, device=device)
    tokens = torch.zeros(batch, depth, dtype=torch.int64, device=device)
    pending = torch.arange(batch, dtype=torch.int64, device=device)
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    generations = torch.full((batch,), 7, dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=1 << (batch - 1).bit_length())
    requests.generation.fill_(7)
    requests.status.fill_(RUNNING)
    requests.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device=device))
    requests.top_k.fill_(-1)
    seeds = torch.arange(batch, device=device, dtype=torch.int64) ^ DRAFT_RNG_DOMAIN
    offsets = torch.full((batch,), 2**32 + 28, dtype=torch.int64, device=device)
    observed_states = torch.empty(batch, depth, rank, dtype=torch.bfloat16, device=device)
    with RecordedPreparationSession(device=device, checkpoint="synthetic-recurrent-head",
            workload_schema="conditional-proposals", numerical_policy="bf16-checkpoint",
            autotune_effort="off", compile_workers=2) as session:
        head = PreparedRecurrentProposalHead(recurrent, pending, session=session, name="head")
        sampler = PreparedVocabularySoftmax(base, q, valid, requests, slots, generations, live, tokens,
            start=0, runtime=None, coordinator=None, session=session, name="q",
            pending=pending, seeds=seeds, offsets=offsets, tokens=tokens)
        sampler.prepare()
        sampler.functional(session=session)
        session.tune(show_progress=False)

        def forward():
            state = head.begin(context)
            previous = pending
            for index in range(depth):
                state, logits = head.step(state, previous, features[:, index].contiguous(), context, base[:, index])
                observed_states[:, index].copy_(state)
                scores[:, index].copy_(logits)
                sampler.run_step(index, scores, tokens, pending=pending, seeds=seeds, offsets=offsets,
                    probabilities=q, valid=valid, tokens=tokens)
                previous = tokens[:, index].clone()

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        for iteration in range(3):
            offsets.add_(28)
            if batch > 1:
                requests.status[batch - 1] = 0 if iteration == 1 else RUNNING
            q.fill_(float("nan"))
            observed_states.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            active = batch - (batch > 1 and iteration == 1)
            assert valid[:active].all()
            state = torch.zeros(batch, rank, dtype=torch.bfloat16, device=device)
            previous = pending
            for index in range(depth):
                dynamic = torch.cat((state, recurrent.embedding[previous]), -1)
                fixed = torch.cat((features[:, index], context), -1)
                raw = (dynamic.double() @ recurrent.state_weight.double().T
                    + fixed.double() @ recurrent.context_weight.double().T + recurrent.bias.double()).float()
                concatenated = torch.cat((dynamic, fixed), -1).double() @ weight.double().T + recurrent.bias.double()
                torch.testing.assert_close(raw.double(), concatenated, rtol=2e-7, atol=2e-7)
                state, latent = recurrent_gate(raw, state)
                assert torch.equal(observed_states[:active, index], state[:active])
                correction = (latent.double() @ recurrent.head.double().T + recurrent.head_bias.double()).bfloat16()
                expected_scores = corrected_logits(base[:, index], correction, 30.)
                assert torch.equal(scores[:active, index], expected_scores[:active])
                expected_q = expected_scores[:, :vocabulary].float().softmax(-1)
                torch.testing.assert_close(q[:active, index, :vocabulary], expected_q[:active], rtol=3e-6, atol=1e-9)
                assert not q[:, index, vocabulary:].count_nonzero()
                for row in range(active):
                    seed, draw = int(seeds[row]), int(offsets[row]) + 4 * index
                    bits = torch.tensor([philox(seed, (token, draw & 0xffffffff, draw >> 32, 0x47554d42))[0]
                        for token in range(vocabulary)], dtype=torch.float64)
                    uniform = ((bits + .5) * 2**-32).float().clamp(max=1 - 2**-24).double()
                    noise = -(-torch.log1p(-uniform)).log()
                    expected_token = (expected_scores[row, :vocabulary].double().cpu() + noise).argmax()
                    assert int(tokens[row, index]) == int(expected_token)
                previous = tokens[:, index]
            if active < batch:
                assert not valid[-1].any()
                assert torch.equal(q[-1, :, 0], torch.ones(depth, device=device))
                assert not q[-1, :, 1:].count_nonzero()
        graph.reset()


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.no_grad()
def test_accepted_context_replaces_proposals_and_preserves_zero_count_rows(kv_dtype):
    device, hidden, heads, dimension, window = torch.device("cuda:0"), 128, 2, 128, 1024
    generator = torch.Generator().manual_seed(31)
    def values(*shape, dtype=torch.bfloat16):
        return (torch.randint(-2, 3, shape, generator=generator).float() / 8).to(device, dtype)
    layers = tuple(SimpleNamespace(context_kv=values(2 * heads * dimension, hidden),
        context_key_norm=torch.ones(dimension, dtype=torch.bfloat16, device=device),
        context_norm=torch.ones(hidden, dtype=torch.bfloat16, device=device)) for _ in range(5))
    draft = SimpleNamespace(layers=5, window=window, head_dim=dimension, kv_heads=heads, theta=1e6,
        local_layers=(0, 1, 3, 4), text={"rms_norm_eps": 1e-6})
    alpha = torch.randn(6, 8, generator=generator).to(device).softmax(-1).bfloat16()
    weights = SimpleNamespace(config=SimpleNamespace(draft=draft, text={"hidden_size": hidden}),
        shard=TensorParallelShard(0, 1), draft=SimpleNamespace(layers=layers, alpha=alpha,
            recurrent=SimpleNamespace(context_norm=torch.ones(hidden, dtype=torch.bfloat16, device=device))))
    regions = {f"draft{index}.{kind}": values(3, window, heads, dimension, dtype=kv_dtype)
        for index in (0, 1, 3, 4) for kind in ("key", "value")}
    regions["draft.rnn_context"] = values(3, hidden)
    key, value = values(20, 128, heads, dimension, dtype=kv_dtype), values(20, 128, heads, dimension, dtype=kv_dtype)
    pool = SimpleNamespace(reserved_region=regions.__getitem__, key=lambda *args, **kwargs: key,
        value=lambda *args, **kwargs: value)
    m = Metadata.pack(torch.tensor([1022, 127], device=device), torch.tensor([8, 8], device=device),
        torch.tensor([1, 2], device=device), torch.arange(20, device=device, dtype=torch.int32).view(2, 10),
        torch.zeros(2, 8, dtype=torch.int64, device=device), torch.tensor([2], dtype=torch.int32, device=device),
        token_capacity=16, sample_count=16)
    auxiliary, counts = values(16, 8, hidden), torch.tensor([3, 0], device=device, dtype=torch.int32)
    snapshots = {name: tensor.clone() for name, tensor in regions.items()}
    key_before, value_before = key.clone(), value.clone()
    with RecordedPreparationSession(device=device, checkpoint=f"synthetic-dspark-context-{kv_dtype}",
            workload_schema="accepted-context", numerical_policy="bf16-checkpoint",
            autotune_effort="off", compile_workers=2) as session:
        operation = PreparedContext(weights, auxiliary, m, pool, session=session, name="context")
        session.tune(show_progress=False)
        operation(auxiliary, counts, m)
        torch.cuda.synchronize()
        inverse = 1. / (1e6 ** (torch.arange(0, dimension, 2, device=device).float() / dimension))
        rtol = 0.08 if kv_dtype == torch.float8_e4m3fn else 0.008
        for index, layer in enumerate(layers):
            mixed = mix_features(auxiliary, alpha[index], layer.context_norm, 1e-6)
            projected = (mixed.double() @ layer.context_kv.double().T).bfloat16()
            k, v = projected.chunk(2, -1)
            k = k.view(16, heads, dimension).float()
            k = (k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
            k = rotary(k, m.positions, inverse)
            v = v.view(16, heads, dimension)
            if kv_dtype == torch.float8_e4m3fn:
                k = k.clamp(-448.0, 448.0).to(kv_dtype)
                v = v.clamp(-448.0, 448.0).to(kv_dtype)
            if index in draft.local_layers:
                for kind, expected in (("key", k), ("value", v)):
                    name = f"draft{index}.{kind}"
                    reference = snapshots[name].clone()
                    reference[1, torch.tensor([1022, 1023, 0], device=device)] = expected[:3]
                    torch.testing.assert_close(regions[name].float(), reference.float(), rtol=rtol, atol=0.)
                    assert torch.equal(regions[name][0].view(torch.uint8), snapshots[name][0].view(torch.uint8))
                    assert torch.equal(regions[name][2].view(torch.uint8), snapshots[name][2].view(torch.uint8))
            else:
                for actual, before, expected in ((key, key_before, k), (value, value_before, v)):
                    reference = before.clone()
                    reference.view(-1, heads, dimension)[1022:1025] = expected[:3]
                    torch.testing.assert_close(actual.float(), reference.float(), rtol=rtol, atol=0.)
        expected_context = snapshots["draft.rnn_context"].clone()
        expected_context[1] = mix_features(auxiliary[2:3], alpha[-1], weights.draft.recurrent.context_norm, 1e-6)[0]
        assert torch.equal(regions["draft.rnn_context"], expected_context)
        retained = {name: tensor.clone() for name, tensor in regions.items()}
        key_before, value_before = key.clone(), value.clone()
        counts.zero_()
        operation(auxiliary, counts, m)
        torch.cuda.synchronize()
        assert all(torch.equal(regions[name].view(torch.uint8), before.view(torch.uint8)) for name, before in retained.items())
        assert (torch.equal(key.view(torch.uint8), key_before.view(torch.uint8))
                and torch.equal(value.view(torch.uint8), value_before.view(torch.uint8)))


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("layer,start,page_ids,leading_inactive", [(0, 1027, tuple(range(9)), False),
    (2, 1027, tuple(range(9)), False), (2, 127, (2,), False), (2, 255, (2, 4), False),
    (2, 127, (2,), True)])
@torch.no_grad()
def test_draft_attention_preserves_committed_kv_with_short_target_page_tables(kv_dtype, layer, start, page_ids, leading_inactive):
    device, hidden, heads, kv_heads, dimension = torch.device("cuda:0"), 128, 12, 2, 128
    generator = torch.Generator().manual_seed(231)
    def values(*shape, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator) / 16).to(device, dtype)
    d = SimpleNamespace(heads=heads, kv_heads=kv_heads, head_dim=dimension, window=1024, block_size=8,
        local_layers=(0, 1, 3, 4), theta=1e6, text={"rms_norm_eps": 1e-6, "intermediate_size": 256})
    config = SimpleNamespace(text={"hidden_size": hidden}, draft=d, vocabulary=576)
    weights = SimpleNamespace(qkv=values((heads + 2 * kv_heads) * dimension, hidden),
        query_norm=torch.ones(dimension, dtype=torch.bfloat16, device=device),
        key_norm=torch.ones(dimension, dtype=torch.bfloat16, device=device), output=values(hidden, heads * dimension))
    regions = {f"draft0.{kind}": values(3, 1024, kv_heads, dimension, dtype=kv_dtype) for kind in ("key", "value")}
    key, value = values(10, 128, kv_heads, dimension, dtype=kv_dtype), values(10, 128, kv_heads, dimension, dtype=kv_dtype)
    pool = SimpleNamespace(reserved_region=regions.__getitem__, key=lambda *args, **kwargs: key,
        value=lambda *args, **kwargs: value)
    starts, lengths, states, tables = ([0, start], [0, 8], [0, 1], [[-1] * len(page_ids), page_ids]) if leading_inactive else (
        [start], [8], [1], [page_ids])
    batch = len(starts)
    m = Metadata.pack(torch.tensor(starts, device=device), torch.tensor(lengths, device=device),
        torch.tensor(states, device=device), torch.tensor(tables, dtype=torch.int32, device=device),
        torch.zeros(batch, 8, dtype=torch.int64, device=device), torch.tensor([batch], dtype=torch.int32, device=device),
        token_capacity=batch * 8, sample_count=batch * 8)
    scratch = DraftScratch.allocate(config, TensorParallelShard(0, 1), batch * 8, 8, device)
    source = values(batch * 8, hidden)
    committed = {name: tensor.clone() for name, tensor in regions.items()}
    committed_key, committed_value = key.clone(), value.clone()
    with RecordedPreparationSession(device=device, checkpoint=f"synthetic-dspark-attention-{kv_dtype}",
            workload_schema="noncausal-block", numerical_policy="bf16-checkpoint",
            autotune_effort="off", compile_workers=2) as session:
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        operation = DraftAttention(layer, weights, config, scratch, m, pool, shard=TensorParallelShard(0, 1),
            runtime=None, coordinator=None, session=session, name="attention", workspace=arena)
        session.tune(show_progress=False)
        actual = operation(source, m)
        qkv = (source[:8].double() @ weights.qkv.double().T).bfloat16()
        q, k, v = qkv.split((heads * dimension, kv_heads * dimension, kv_heads * dimension), -1)
        def normalize(x, count):
            x = x.view(8, count, dimension).float()
            return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        q = rotary(normalize(q, heads), m.positions[:8], operation.inverse)
        k = rotary(normalize(k, kv_heads), m.positions[:8], operation.inverse)
        v = v.reshape(8, kv_heads, dimension)
        if kv_dtype == torch.float8_e4m3fn:
            k = k.clamp(-448.0, 448.0).to(kv_dtype)
            v = v.clamp(-448.0, 448.0).to(kv_dtype)
        if layer == 0:
            positions = torch.arange(max(0, start - d.window), start, device=device) % d.window
            context_k = torch.cat((committed["draft0.key"][1, positions], k), 0)
            context_v = torch.cat((committed["draft0.value"][1, positions], v), 0)
            assert all(torch.equal(regions[name].view(torch.uint8), tensor.view(torch.uint8)) for name, tensor in committed.items())
        else:
            pages = torch.tensor(page_ids, device=device)
            context_k = torch.cat((committed_key[pages].flatten(0, 1)[:start], k), 0)
            context_v = torch.cat((committed_value[pages].flatten(0, 1)[:start], v), 0)
        assert (torch.equal(key.view(torch.uint8), committed_key.view(torch.uint8))
                and torch.equal(value.view(torch.uint8), committed_value.view(torch.uint8)))
        expanded_k = context_k.repeat_interleave(heads // kv_heads, 1).double()
        expanded_v = context_v.repeat_interleave(heads // kv_heads, 1).double()
        scores = torch.einsum("qhd,khd->hqk", q.double(), expanded_k) / dimension**.5
        attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), expanded_v).bfloat16()
        expected = (attended.flatten(1).double() @ weights.output.double().T).bfloat16()
        torch.testing.assert_close(actual[:8], expected, rtol=.02, atol=2**-12)
        assert not actual[8:].count_nonzero()
        masked = scores.clone()
        query = torch.arange(8, device=device)
        keys = torch.arange(context_k.shape[0], device=device)
        masked.masked_fill_(keys[None, None] > context_k.shape[0] - 8 + query[None, :, None], -torch.inf)
        causal = torch.einsum("hqk,khd->qhd", masked.softmax(-1), expanded_v).bfloat16()
        assert not torch.equal(causal[0], attended[0])
        replayed = torch.empty_like(actual)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            replayed.copy_(operation(source, m))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.equal(replayed, actual)
        positions, begins, active_lengths = m.positions.clone(), m.query_start_loc.clone(), m.lengths.clone()
        m.lengths.zero_()
        m.query_start_loc.zero_()
        m.positions.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert not replayed.count_nonzero()
        assert (torch.equal(key.view(torch.uint8), committed_key.view(torch.uint8))
                and torch.equal(value.view(torch.uint8), committed_value.view(torch.uint8)))
        assert all(torch.equal(regions[name].view(torch.uint8), tensor.view(torch.uint8)) for name, tensor in committed.items())
        m.lengths.copy_(active_lengths)
        m.query_start_loc.copy_(begins)
        m.positions.copy_(positions)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replayed, actual)
        graph.reset()


@torch.no_grad()
def test_five_layer_block_uses_learned_masks_and_replays_without_allocating():
    device, hidden, heads, kv_heads, dimension, vocabulary, rank = torch.device("cuda:0"), 128, 12, 2, 128, 576, 256
    generator = torch.Generator().manual_seed(341)
    def values(*shape):
        return (torch.randn(shape, generator=generator) / 32).to(device, torch.bfloat16)
    def ones(width):
        return torch.ones(width, dtype=torch.bfloat16, device=device)
    layers = tuple(SimpleNamespace(qkv=values((heads + 2 * kv_heads) * dimension, hidden),
        output=values(hidden, heads * dimension), query_norm=ones(dimension), key_norm=ones(dimension),
        input_norm=ones(hidden), post_attention_norm=ones(hidden),
        mlp=MLPWeights(values(512, hidden), values(hidden, 256))) for _ in range(5))
    d = SimpleNamespace(heads=heads, kv_heads=kv_heads, head_dim=dimension, window=1024, block_size=8,
        local_layers=(0, 1, 3, 4), theta=1e6, text={"rms_norm_eps": 1e-6, "intermediate_size": 256})
    config = SimpleNamespace(text={"hidden_size": hidden}, draft=d, vocabulary=vocabulary)
    weights = SimpleNamespace(config=config, shard=TensorParallelShard(0, 1), embedding=values(vocabulary, hidden),
        embedding_norm=ones(hidden), head=values(vocabulary, hidden), draft=SimpleNamespace(layers=layers,
            mask_embedding=values(hidden), norm=ones(hidden), recurrent=SimpleNamespace(embedding=values(vocabulary, rank),
                context_weight=values(3 * rank, 2 * hidden), state_weight=values(3 * rank, 2 * rank),
                bias=values(3 * rank).float(), head=values(vocabulary, rank), head_bias=values(vocabulary), softcap=30.)))
    regions = {f"draft{index}.{kind}": values(3, 1024, kv_heads, dimension)
        for index in (0, 1, 3, 4) for kind in ("key", "value")}
    regions["draft.rnn_context"] = values(3, hidden)
    key, value = values(2, 128, kv_heads, dimension), values(2, 128, kv_heads, dimension)
    pool = SimpleNamespace(reserved_region=regions.__getitem__, key=lambda *args, **kwargs: key,
        value=lambda *args, **kwargs: value)
    m = Metadata.pack(torch.tensor([3], device=device), torch.tensor([8], device=device),
        torch.tensor([1], device=device), torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        torch.arange(8, dtype=torch.int64, device=device)[None], torch.tensor([1], dtype=torch.int32, device=device),
        token_capacity=8, sample_count=8)
    scratch = DraftScratch.allocate(config, weights.shard, 8, 8, device)
    before = {name: tensor.clone() for name, tensor in regions.items()}
    with RecordedPreparationSession(device=device, checkpoint="synthetic-dspark-block",
            workload_schema="five-layer-block", numerical_policy="bf16-checkpoint",
            autotune_effort="off", compile_workers=2) as session:
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        operation = PreparedDraft(weights, scratch, m, pool, runtime=None, coordinator=None,
            session=session, name="draft", workspace=arena)
        session.tune(show_progress=False)
        output = torch.empty(8, vocabulary, dtype=torch.bfloat16, device=device)
        def forward():
            logits, _, features = operation(m)
            output.copy_(logits)
            return features
        features = forward()
        assert features["hidden"].shape == (8, hidden)
        assert torch.equal(features["recurrent_context"], regions["draft.rnn_context"][1:2])
        torch.cuda.synchronize()
        expected = output.clone()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        m.token_ids[1:].add_(71)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.equal(output, expected)
        assert torch.isfinite(output).all()
        assert all(torch.equal(regions[name], tensor) for name, tensor in before.items())
        m.token_ids[0].add_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.equal(output, expected)
        graph.reset()


class _SparseDraftContextForward:
    def __init__(self, operation, metadata):
        self.operation, self.metadata = operation, metadata

    def __call__(self):
        logits, _, features = self.operation(self.metadata)
        return logits, features["recurrent_context"]


@torch.no_grad()
@pytest.mark.parametrize("variant", ["small", "mini"])
def test_checkpoint_draft_aot_reads_only_selected_mapped_context_rows(tmp_path, variant):
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.core_storage import CoreStorage
    from hyperflare_recipes.recipes.inkling2.config import ModelConfig
    from hyperflare_recipes.recipes.inkling2.pool import ContinuationPool, RankGeometry, continuation_layout, page_specs

    variable = "FLARE_INKLING2_MINI_SNAPSHOT" if variant == "mini" else "FLARE_INKLING2_SNAPSHOT"
    snapshot = os.environ.get(variable)
    if snapshot is None:
        pytest.skip(f"requires {variable}")
    device, batch, block = torch.device("cuda:0"), 8, 8
    torch.cuda.set_device(device)
    config, shard = ModelConfig.load(snapshot), TensorParallelShard(0, 1 if variant == "mini" else 4)
    manifest = inspect_indexed_checkpoint(snapshot)
    if variant == "mini":
        from hyperflare_recipes.recipes.inkling2.mini_weights import MiniWeightLoader
        context = MiniWeightLoader(manifest, device, shard=shard, vision=False)
    else:
        context = DirectWeightLoader(manifest, device)
    first, last = shard.padded_partition(config.vocabulary, alignment=64)
    with context as loader:
        head = loader.load_tensor("lm_head.weight", extent=(slice(first, last), slice(None)))
        embedding = loader.load_tensor("model.language_model.embed_tokens.weight",
            extent=(slice(first, last), slice(None)))
        embedding_norm = loader.load_tensor("model.language_model.embed_norm.weight")
    weights = SimpleNamespace(config=config, shard=shard, head=head, embedding=embedding,
        embedding_norm=embedding_norm, draft=DraftWeights.load(snapshot, config, device, shard, target_head=head))
    text = config.text.cache_config
    geometry = RankGeometry.of(text, shard.world_size)
    total = torch.cuda.get_device_properties(device).total_memory
    page_bytes = sum(shape[0] for _, shape in page_specs(text, geometry, config.draft))
    row_bytes = continuation_layout(text, geometry, config.draft).row_bytes
    storage = CoreStorage(device=0)
    pool = ContinuationPool.allocate(text, geometry, batch + 1, batch, device, storage=storage,
        maximum_pages=total // page_bytes, maximum_rows=total // row_bytes, backed_slots=batch + 1,
        draft=config.draft)
    pool.clear_for_preparation()
    context = pool.reserved_region("draft.rnn_context")
    assert context.shape[0] > batch + 1
    markers = (torch.arange((batch + 1) * config.hidden, device=device).reshape(batch + 1, -1) % 127).bfloat16()
    context[:batch + 1].copy_(markers)
    before = pool.backing.clone()
    metadata = Metadata.allocate(batch * block, batch, 1, device, samples=batch * block, max_query_tokens=block)
    scratch = DraftScratch.allocate(config, shard, batch * block, block, device)
    with RecordedPreparationSession(device=device, checkpoint="inkling2-sparse-draft-context",
            workload_schema="selected-context-rows", numerical_policy="bf16-checkpoint",
            compile_workers=2) as session:
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        operation = PreparedDraft(weights, scratch, metadata, pool, runtime=None, coordinator=None,
            session=session, name="draft", workspace=arena)
        session.tune(show_progress=False)
        owner = _SparseDraftContextForward(operation, metadata)
        eager_logits, eager_context = owner()
        torch.testing.assert_close(eager_context, markers[0].expand(batch, -1), rtol=0, atol=0)
        compiled = aot_forward(owner)
        logits, selected = compiled(owner)
        torch.cuda.synchronize()
        torch.testing.assert_close(logits, eager_logits, rtol=0, atol=0)
        torch.testing.assert_close(selected, eager_context, rtol=0, atol=0)
        path = tmp_path / "draft.bin"
        compiled.save_compiled_function(str(path))
        with path.open("rb") as file:
            restored = torch.compiler.load_compiled_function(file)
        restored(owner)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            logits, selected = restored(owner)
        session.freeze()
        starts = torch.full((batch,), 127, dtype=torch.int64, device=device)
        lengths = torch.zeros(batch, dtype=torch.int32, device=device)
        rows = torch.arange(batch, 0, -1, dtype=torch.int64, device=device)
        pages = torch.arange(batch, dtype=torch.int32, device=device)[:, None]
        tokens = torch.zeros((batch, block), dtype=torch.int64, device=device)
        live = torch.zeros(1, dtype=torch.int32, device=device)
        for active in (8, 4, 0):
            markers.add_(1)
            context[:batch + 1].copy_(markers)
            before.copy_(pool.backing)
            lengths.zero_()
            lengths[::batch // max(active, 1)] = block if active else 0
            live.fill_(batch if active else 0)
            metadata.set(starts, lengths, rows, pages, tokens, live)
            eager_logits, eager_context = owner()
            expected = markers[torch.where(lengths > 0, rows, 0)]
            torch.testing.assert_close(eager_context, expected, rtol=0, atol=0)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(logits, eager_logits, rtol=0, atol=0)
            torch.testing.assert_close(selected, expected, rtol=0, atol=0)
            assert torch.equal(pool.backing, before)
        graph.reset()


@torch.no_grad()
def test_checkpoint_sidecar_preserves_fp8_values_scales_and_feature_coefficients():
    from b12x.gemm._shared.wo_mxfp8 import materialize_mxfp8_scale_rows
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from safetensors import safe_open

    snapshot = os.environ.get("FLARE_INKLING2_SNAPSHOT")
    if snapshot is None:
        pytest.skip("requires FLARE_INKLING2_SNAPSHOT")
    raw = json.loads(open(snapshot + "/config.json").read())
    m = raw["mtp_config"]
    d = SimpleNamespace(**m, heads=m["num_attention_heads"], kv_heads=m["num_key_value_heads"],
        layers=m["num_hidden_layers"], auxiliary_layers=m["target_layer_ids"], text=m)
    config = SimpleNamespace(text=raw["text_config"], draft=d, vocabulary=201024)
    shard, device = TensorParallelShard(0, 4), torch.device("cuda:0")
    with DirectWeightLoader(inspect_indexed_checkpoint(snapshot), device) as loader:
        head = loader.load_tensor("lm_head.weight", extent=(slice(0, 50304), slice(None)))
    weights = DraftWeights.load(snapshot, config, device, shard, target_head=head)
    assert len(weights.layers) == 5 and weights.recurrent.head.shape == (50304, 256)
    assert weights.alpha.dtype == torch.bfloat16
    with safe_open(snapshot + "/mtp.safetensors", framework="pt", device="cpu") as source:
        expected_alpha = source.get_tensor("fuse_weight").float().softmax(-1).bfloat16()
        assert torch.equal(weights.alpha.cpu(), expected_alpha)
        for index, layer in enumerate(weights.layers):
            packed = layer.mlp.gate_up
            assert isinstance(packed, MXFP8LinearWeight)
            matrix = packed.weight.values.squeeze(-1).view(torch.uint8)
            scales = materialize_mxfp8_scale_rows(packed.weight).view(torch.uint8).squeeze(0)
            for offset, role in ((0, "gate"), (4608, "up")):
                name = f"layers.{index}.mlp.{role}_proj"
                expected = source.get_slice(name + ".weight")[:1].view(torch.uint8)
                expected_scales = source.get_slice(name + ".weight_scale")[:1].view(torch.uint8)
                assert torch.equal(matrix[offset:offset + 1].cpu(), expected)
                assert torch.equal(scales[offset:offset + 1].cpu(), expected_scales)
