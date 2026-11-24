# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Decoder suffix metadata preserves live positions, cache pages, and replay bounds."""

import pytest
import torch
from types import SimpleNamespace

from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix


@pytest.mark.parametrize("capacity", [256, 512, 4096])
@pytest.mark.parametrize("start", [0, 127, 4095])
def test_decoder_suffix_alternates_full_partial_and_inactive_chunks(capacity, start):
    source = Metadata.allocate(capacity, 1, 40, "cpu")
    packing = PreparedDecoderSuffix(source).functional()
    values = torch.arange(capacity * 12).reshape(capacity, 3, 4)
    output = torch.empty((128, 3, 4), dtype=values.dtype)
    pages = torch.arange(40).flip(0).view(1, -1)
    source.page_table.copy_(pages)
    source.starts.fill_(start)
    source.query_start_loc[0].zero_()
    source.token_ids.copy_(torch.arange(capacity) + 1000)
    for count in (capacity, 129, 128, 1, 127, 0, capacity - 1):
        source.lengths.fill_(count)
        source.query_start_loc[1] = count
        source.state_rows.fill_(7 if count else 0)
        source.live_counts.copy_(torch.tensor([count, 1 if count else 0]))
        source.sample_indices.fill_(max(0, count - 1))
        source.positions.fill_(-1)
        source.positions[:count] = torch.arange(start, start + count)
        output.fill_(-999)
        packed = packing.begin(source)
        output.copy_(packed.gather(values))
        target = packed.metadata
        retained = list(range(max(0, count - 128), count))
        kept = len(retained)
        assert target.positions.tolist() == [start + index for index in retained] + [-1] * (128 - kept)
        assert target.token_ids.tolist() == [1000 + index for index in retained] + [0] * (128 - kept)
        assert target.starts.item() == start + max(0, count - 128)
        assert target.lengths.item() == kept
        assert target.query_start_loc.tolist() == [0, kept]
        assert target.sample_indices.item() == max(0, kept - 1)
        assert target.state_rows.item() == (7 if count else 0)
        assert target.live_counts.tolist() == [kept, 1 if count else 0]
        assert target.requests.eq(0).all()
        assert torch.equal(target.page_table, pages)
        assert torch.equal(target.token_page_table, pages.expand(128, -1))
        expected = torch.zeros_like(output)
        if retained:
            expected[:kept] = values[retained]
        assert torch.equal(output, expected)
        assert packed.replay_start.item() == (start + retained[0] if count > 128 else 0)


@pytest.mark.parametrize("rows,tokens", [(2, 256), (1, 128)])
def test_decoder_suffix_rejects_unsupported_packing_geometry(rows, tokens):
    with pytest.raises(ValueError, match="one prefill chunk larger"):
        PreparedDecoderSuffix(Metadata.allocate(tokens, rows, 2, "cpu"))


def test_decoder_feedback_at_replay_boundary_is_rejected_before_component_construction(monkeypatch):
    from hyperflare_recipes.recipes.deepseek41 import decoder

    monkeypatch.setattr(decoder, "PreparedTableRequests", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder.Scratch, "allocate", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "PreparedEmbedding",
                        lambda *args, **kwargs: pytest.fail("unsupported feedback reached component construction"))
    weights = SimpleNamespace(config=SimpleNamespace(decoder_start=20,
        sources=(SimpleNamespace(layer=20),), draft=SimpleNamespace(target_layers=(20,))),
        draft=object(), layers=tuple(SimpleNamespace(engram=None) for _ in range(40)),
        shard=SimpleNamespace(world_size=4), embedding=None)
    scratch = SimpleNamespace(embedding=torch.zeros(1), hidden=torch.zeros((1, 5120)),
        local_logits=torch.zeros((1, 1)), logits=torch.zeros((1, 1)))
    with pytest.raises(ValueError, match="draft target layers after"):
        decoder.PreparedDecoder(weights, None, scratch, Metadata.allocate(256, 1, 2, "cpu"), None,
            mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None,
            session=None, name="boundary", workspace=None)


def test_encoder_only_decoder_skips_decoder_layers_and_head(monkeypatch):
    from hyperflare_recipes.recipes.deepseek41 import decoder

    built_layers = []
    boundary_calls = []
    committed = []

    class FakeTableRequests:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, metadata):
            return "raw-history"

        def commit(self, lengths, raw_history, metadata):
            committed.append((lengths.tolist(), raw_history))

    class FakeEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def functional(self, **kwargs):
            return self

        def __call__(self, *, token_ids, num_rows):
            return torch.zeros((token_ids.numel(), 5120), dtype=torch.bfloat16)

    class FakeIndices:
        def __init__(self, *args, **kwargs):
            pass

        def functional(self):
            return self

        def __call__(self, metadata):
            return "indices"

    class FakeLayer:
        def __init__(self, index, *args, **kwargs):
            self.index = index
            self.has_image_mask = False
            self.pipelined_ffn = False
            self.pipelined_attn = None
            self.boundary = None
            self.engram = None
            built_layers.append((index, kwargs["carry"], kwargs["defer_post"]))

        def __call__(self, incoming, pre_mix, metadata, indices, selection, **kwargs):
            return incoming, pre_mix, f"rec{self.index}", selection

    class FakeBoundary:
        def __init__(self, index, layer, source, target, packing, pool, rotary, **kwargs):
            self.index = index
            self.packing = packing
            self.carry = kwargs["carry"]

        def __call__(self, incoming, pre_mix, metadata, indices, indexer=None):
            boundary_calls.append((self.index, self.packing, self.carry, indexer))
            return None, None, None

    monkeypatch.setattr(decoder, "PreparedTableRequests", FakeTableRequests)
    monkeypatch.setattr(decoder, "PreparedEmbedding", FakeEmbedding)
    monkeypatch.setattr(decoder, "Rotary", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "AttentionIndices", FakeIndices)
    monkeypatch.setattr(decoder, "PreparedLayer", FakeLayer)
    monkeypatch.setattr(decoder, "PreparedDecoderBoundary", FakeBoundary)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder.PreparedDecoder, "_prepare_head",
                        lambda *args, **kwargs: pytest.fail("encoder-only decoder must not build LM head"))

    layers = tuple(SimpleNamespace(engram=None, attention={}) for _ in range(40))
    weights = SimpleNamespace(
        config=SimpleNamespace(
            decoder_start=20,
            sources=(SimpleNamespace(layer=2), SimpleNamespace(layer=20)),
            text={"compress_ratios": [0, 0, *([2] * 18), *([1] * 20)]},
            draft=SimpleNamespace(target_layers=(35,)),
        ),
        draft=object(),
        layers=layers,
        shard=SimpleNamespace(world_size=4),
        embedding=torch.zeros((16, 5120)),
    )
    scratch = SimpleNamespace(
        embedding=torch.zeros((256, 5120)),
        attention=SimpleNamespace(output=torch.zeros((256, 5120))),
        hidden=torch.zeros((1, 5120)),
        local_logits=torch.zeros((1, 1), dtype=torch.float32),
        logits=torch.zeros((1, 1), dtype=torch.float32),
    )
    metadata = Metadata.allocate(256, 1, 2, "cpu")
    metadata.lengths.fill_(256)

    dec = decoder.PreparedDecoder(
        weights, None, scratch, metadata, None,
        mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None,
        session=None, name="prefill256", workspace=None, encoder_only=True,
    )
    assert dec.encoder_only is True
    assert len(dec.layers) == 20
    assert [idx for idx, _, _ in built_layers] == list(range(20))
    assert dec.packing is None
    assert dec.suffix_window_metadata is None
    assert dec.auxiliary is None
    assert dec.head is None
    assert dec.encoder_boundary is not None
    assert dec.encoder_boundary.index == 20
    assert dec.encoder_boundary.packing is None
    assert dec.encoder_boundary.carry is True

    hidden, logits, auxiliary, state = dec(metadata)
    assert (hidden, logits, auxiliary) == (None, None, None)
    assert boundary_calls == [(20, None, True, None)]
    assert committed == [([256], "raw-history")]
    assert len(state[1]) == 20

    with pytest.raises(ValueError, match="encoder-only prefill requires"):
        decoder.PreparedDecoder(
            weights, None, scratch, metadata, None,
            mode="decode", runtime=None, vocabulary_runtime=None, coordinator=None,
            session=None, name="bad", workspace=None, encoder_only=True,
        )


def test_prepared_generation_record_only_skips_sampling_and_context(monkeypatch):
    from hyperflare_recipes.components import paged_generation as pg

    events = []

    class FakeAllReduce:
        def __init__(self, *args, **kwargs):
            pass

        def functional(self, **kwargs):
            return self

        def __call__(self, value):
            events.append("validate_reduce")
            return value

    class FakeSampling:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self):
            events.append("sampling_prepare")

        def prepare_broadcast(self, **kwargs):
            pytest.fail("record-only generation must not prepare sampling broadcast")

        def functional(self, **kwargs):
            events.append("sampling_functional")

        def run(self, logits):
            pytest.fail("record-only generation must not run sampling")

        def record_completion(self):
            events.append("record_completion")

    monkeypatch.setattr(pg, "PreparedAllReduce", FakeAllReduce)
    monkeypatch.setattr(pg, "PreparedSampling", FakeSampling)
    monkeypatch.setattr(pg, "save_hidden",
                        lambda *args, **kwargs: pytest.fail("record-only generation must not save hidden"))

    meta = Metadata.allocate(256, 1, 2, "cpu")
    class FakeDecoder:
        def __init__(self):
            self.scratch = SimpleNamespace(logits=torch.zeros((1, 16), dtype=torch.float32))
            self.weights = SimpleNamespace(draft=object())
            self.auxiliary = None
            self.pool = None

        def release_outputs(self):
            events.append("release_outputs")

        def __call__(self, m):
            events.append("decoder")
            return None, None, None, ()

    class FakeBatch:
        rows = 1
        validation_input = torch.zeros(1)
        validation_sum = torch.zeros(1)
        requests = object()
        metadata = meta
        inputs = SimpleNamespace(
            row_slot_ids=torch.zeros(1, dtype=torch.int32),
            row_generations=torch.zeros(1, dtype=torch.int64),
            live_rows=torch.ones(1, dtype=torch.int32),
            sample_enabled=torch.zeros(1, dtype=torch.int32),
        )

        def functional(self):
            events.append("batch_functional")

        def validate(self):
            events.append("batch_validate")
            return torch.zeros(1), True

        def begin(self, reduced, *, present):
            events.append("batch_begin")
            return meta

        def finish(self, m):
            events.append("batch_finish")

    gen = pg.PreparedGeneration(
        FakeDecoder(), FakeBatch(), None, None, None,
        runtime=None, coordinator=None, session=None, name="prefill256.generation",
        workspace=None,
        context_factory=lambda *args, **kwargs: pytest.fail("record-only generation must not build draft context"),
        sample=False,
    )
    assert gen.sample is False
    assert gen.compilation_variant == "record"
    assert gen.context is None
    gen()
    assert events == [
        "sampling_prepare", "sampling_functional", "release_outputs", "batch_functional",
        "batch_validate", "validate_reduce", "batch_begin", "decoder", "batch_finish", "record_completion",
    ]


def test_prepared_prefill_rejects_decoder_without_terminal_keyword(monkeypatch):
    from hyperflare_recipes.components import paged_generation

    class LegacyDecoder:
        def __call__(self, metadata):
            return None

        def release_outputs(self):
            pass

    batch = SimpleNamespace(
        inputs=SimpleNamespace(row_slot_ids=torch.zeros(1), row_generations=torch.zeros(1), live_rows=torch.zeros(1)),
        rows=1,
        requests=object(),
        functional=lambda: None,
    )
    with pytest.raises(TypeError, match="terminal=False"):
        paged_generation.PreparedPrefill(LegacyDecoder(), batch, None, None, None, validation=lambda v: v)


def test_prepared_prefill_invokes_decoder_with_terminal_false_and_completes(monkeypatch):
    from hyperflare_recipes.components import paged_generation

    events = []

    class FakeCompletion:
        def __init__(self, *args):
            self.prepared = False

        def prepare(self):
            self.prepared = True

        def __call__(self):
            events.append("complete")

    monkeypatch.setattr(paged_generation, "PreparedCompletion", FakeCompletion)

    class BoundaryDecoder:
        def __init__(self):
            self.released = False

        def release_outputs(self):
            self.released = True

        def __call__(self, metadata, *, terminal=True):
            events.append(("decoder", metadata, terminal))

    class FakeBatch:
        def __init__(self):
            self.inputs = SimpleNamespace(
                row_slot_ids=torch.zeros(1), row_generations=torch.zeros(1), live_rows=torch.zeros(1)
            )
            self.rows = 1
            self.requests = object()
            self.functional_calls = 0

        def functional(self):
            self.functional_calls += 1

        def validate(self):
            events.append("validate")
            return "v", "p"

        def begin(self, totals, *, present):
            events.append(("begin", totals, present))
            return "meta"

        def finish(self, metadata):
            events.append(("finish", metadata))

    decoder = BoundaryDecoder()
    batch = FakeBatch()
    prefill = paged_generation.PreparedPrefill(
        decoder, batch, None, None, None, validation=lambda value: ("reduced", value)
    )
    assert decoder.released and batch.functional_calls == 1 and prefill.completion.prepared
    prefill()
    assert events == [
        "validate",
        ("begin", ("reduced", "v"), "p"),
        ("decoder", "meta", False),
        ("finish", "meta"),
        "complete",
    ]


@pytest.mark.parametrize("with_boundary", [True, False])
def test_deepseek_decoder_non_terminal_stops_at_global_kv_boundary(with_boundary):
    from hyperflare_recipes.recipes.deepseek41.decoder import PreparedDecoder

    events = []
    decoder = object.__new__(PreparedDecoder)
    decoder.mode = "prefill"
    decoder.encoder_only = False
    decoder.scoring = None
    decoder.decoder_start = 1
    decoder.vision = None
    decoder.staging_shape = None
    decoder.auxiliary_shape = (4, 0)
    decoder.aux_layers = ()
    decoder.compact_head = False
    decoder.engram_consumers = {}
    decoder.engram_overlaps = {}
    decoder.engram_projects = {}
    decoder.weights = SimpleNamespace(embedding=torch.zeros((16, 4)))

    class FakeTableRequests:
        def __call__(self, metadata):
            events.append("table_prepare")
            return "raw_history"

        def commit(self, lengths, raw_history, metadata):
            events.append(("table_commit", raw_history))

    decoder.table_requests = FakeTableRequests()
    decoder.embedding = lambda *, token_ids, num_rows: torch.ones((token_ids.numel(), 4), dtype=torch.bfloat16)

    class FakeIndices:
        def compressed(self, ratio):
            return ("compressed", ratio)

    decoder.window_metadata = lambda metadata: FakeIndices()

    class EncoderLayer:
        boundary = None
        has_image_mask = False

        def __call__(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None,
                     **kwargs):
            events.append("encoder_layer")
            return incoming, pre_mix, "enc_rec", selection

    class BoundaryHelper:
        def produce_global_kv(self, incoming, pre_mix, metadata, indices, indexer):
            events.append(("boundary_global_kv", indexer, indices.compressed(1)))
            return None, pre_mix

    class FakeGlobalKV:
        ratio = 1

        def __call__(self, normalized, indexer, metadata, geometry):
            events.append(("layer_global_kv", indexer, geometry))

    class DecoderStartLayer:
        has_image_mask = False

        def __init__(self):
            self.boundary = BoundaryHelper() if with_boundary else None
            self.attention = SimpleNamespace(selectors=["indexer0"], global_kv=FakeGlobalKV())

        def attention_hc(self, *incoming, pre_mix, pre_out):
            events.append("layer_mhc")
            return None, None, None, "normalized"

        def __call__(self, *args, **kwargs):
            pytest.fail("non-terminal prefill must not execute decoder attention/FFN")

    class TrailingDecoderLayer:
        boundary = None
        has_image_mask = False

        def __call__(self, *args, **kwargs):
            pytest.fail("non-terminal prefill must not execute layers after decoder_start")

    decoder.layers = [EncoderLayer(), DecoderStartLayer(), TrailingDecoderLayer()]
    decoder.head = lambda *args, **kwargs: pytest.fail("non-terminal prefill must not project logits")
    metadata = Metadata.allocate(4, 1, 2, "cpu")
    hidden, logits, auxiliary, (_, records, raw) = decoder(metadata, terminal=False)
    assert (hidden, logits, auxiliary, records, raw) == (None, None, None, ("enc_rec",), "raw_history")
    expected_boundary = (
        [("boundary_global_kv", "indexer0", ("compressed", 1))]
        if with_boundary
        else ["layer_mhc", ("layer_global_kv", "indexer0", ("compressed", 1))]
    )
    assert events == ["table_prepare", "encoder_layer", *expected_boundary, ("table_commit", "raw_history")]


@pytest.mark.parametrize("mode,scoring,decoder_start", [
    ("decode", None, 20),
    ("prefill", object(), 20),
    ("prefill", None, None),
])
def test_deepseek_decoder_rejects_invalid_non_terminal_configurations(mode, scoring, decoder_start):
    from hyperflare_recipes.recipes.deepseek41.decoder import PreparedDecoder

    decoder = object.__new__(PreparedDecoder)
    decoder.mode = mode
    decoder.scoring = scoring
    decoder.decoder_start = decoder_start
    with pytest.raises(ValueError, match="non-terminal prefill"):
        decoder(Metadata.allocate(4, 1, 2, "cpu"), terminal=False)


def test_deepseek_deferred_prefill_config_and_gating():
    from hyperflare_recipes.recipes.deepseek41 import bootstrap

    default_config = bootstrap.PreparationConfig(prefill_chunk_size=4096)
    assert default_config.deferred_prefill is False
    assert not bootstrap.deferred_prefill_enabled(default_config, 128, decoder_start=20)

    enabled_config = bootstrap.PreparationConfig(prefill_chunk_size=4096, deferred_prefill=True)
    assert bootstrap.deferred_prefill_enabled(enabled_config, 128, decoder_start=20)
    assert not bootstrap.deferred_prefill_enabled(enabled_config, 128, decoder_start=None)

    short_chunk_config = bootstrap.PreparationConfig(prefill_chunk_size=128, deferred_prefill=True)
    assert not bootstrap.deferred_prefill_enabled(short_chunk_config, 128, decoder_start=20)

    with pytest.raises(TypeError, match="deferred_prefill must be a boolean"):
        bootstrap.PreparationConfig(deferred_prefill=1)


def test_deepseek_contract_emits_version_4_and_min_tokens_with_terminal_prefill(monkeypatch):
    from ignition.preparation import contract as c
    from ignition.preparation import runtime_core_pb2 as p
    from ignition.preparation.execution import GraphSignature
    from hyperflare_recipes.recipes.deepseek41 import contract

    pool = SimpleNamespace(
        pages={},
        config=SimpleNamespace(text={"sliding_window": 128, "compress_ratios": []}),
        layout=SimpleNamespace(row_bytes=64, region=lambda name: SimpleNamespace(offset=0, nbytes=8)),
    )
    prefill_fields = tuple(c.text_inputs("prefill"))
    prefill_only = {
        "prefill256": GraphSignature("prefill", 1, 256, prefill_fields, speculation_enabled=False),
    }
    v3_contract = contract.prepare(prefill_only, {}, pool, page_tokens=128, draft_lag=None, rows=None)
    assert v3_contract["version"] == 3
    assert not any(graph["kind"] == p.TERMINAL_PREFILL for graph in v3_contract["graphs"])

    with_terminal = {
        **prefill_only,
        "terminal_prefill256": GraphSignature("terminal_prefill", 1, 256, prefill_fields, speculation_enabled=False),
    }
    if "terminal_prefill_min_tokens" not in p.GraphContract.DESCRIPTOR.fields_by_name:
        v3_terminal = contract.prepare(with_terminal, {}, pool, page_tokens=128, draft_lag=None, rows=None)
        assert v3_terminal["version"] == 3
        terminal_graphs = [graph for graph in v3_terminal["graphs"] if graph["kind"] == p.TERMINAL_PREFILL]
        assert len(terminal_graphs) == 1
        assert "terminal_prefill_min_tokens" not in terminal_graphs[0]
        fake_graph_contract = SimpleNamespace(
            DESCRIPTOR=SimpleNamespace(fields_by_name={"terminal_prefill_min_tokens": object()})
        )
        monkeypatch.setattr(p, "GraphContract", fake_graph_contract)

    v4_contract = contract.prepare(with_terminal, {}, pool, page_tokens=128, draft_lag=None, rows=None)
    assert v4_contract["version"] == 4
    terminal_graphs = [graph for graph in v4_contract["graphs"] if graph["kind"] == p.TERMINAL_PREFILL]
    assert len(terminal_graphs) == 1
    assert terminal_graphs[0]["terminal_prefill_min_tokens"] == 129

    overridden = contract.prepare(
        with_terminal, {}, pool, page_tokens=128, draft_lag=None, rows=None, terminal_prefill_min_tokens=160
    )
    assert overridden["version"] == 4
    assert [g["terminal_prefill_min_tokens"] for g in overridden["graphs"] if g["kind"] == p.TERMINAL_PREFILL] == [160]

    v3_schema = SimpleNamespace(DESCRIPTOR=SimpleNamespace(fields_by_name={}))
    monkeypatch.setattr(p, "GraphContract", v3_schema)
    v3_fallback = contract.prepare(with_terminal, {}, pool, page_tokens=128, draft_lag=None, rows=None)
    assert v3_fallback["version"] == 3
    assert all("terminal_prefill_min_tokens" not in g for g in v3_fallback["graphs"])


def test_deepseek_contextual_selection_includes_terminal_prefill():
    from ignition.preparation.execution import GraphSignature
    from hyperflare_recipes.recipes.deepseek41.contextual_selection import catalog_contexts

    layer = SimpleNamespace(
        attention=SimpleNamespace(norm_quant=object(), global_kv=None, mode="prefill", ratio=1,
                                  selectors=(), concurrent_query=None),
        engram=None,
        boundary=None,
        metadata=Metadata.allocate(256, 1, 2, "cpu"),
        carry=False,
        post=None,
    )
    decoder = SimpleNamespace(layers=[layer])
    forwards = {"terminal_prefill256": SimpleNamespace(decoder=decoder)}
    signatures = {"terminal_prefill256": GraphSignature("terminal_prefill", 1, 256, ())}
    contexts = list(catalog_contexts(forwards, signatures))
    assert len(contexts) == 1
    assert contexts[0][0] == "terminal_prefill256"
    assert contexts[0][1].kind == "terminal_prefill"
    assert contexts[0][2] is decoder
    assert contexts[0][3] == 0
