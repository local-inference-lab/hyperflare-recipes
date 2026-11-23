# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Split-attention prefill token halves geometry, carry materialization, and selection propagation."""

from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.recipes.deepseek41.contextual_selection import catalog_contexts
from hyperflare_recipes.recipes.deepseek41.decoder_halves import (
    ChunkHalves,
    PendingHalvesCarry,
    PreparedHalvesLayer,
    halves_size,
)


@pytest.mark.parametrize("chunk_rows,expected", [
    (128, None),
    (4096, None),
    (8190, None),
    (8320, None),
    (8192, 4096),
    (16384, 8192),
])
def test_halves_size_requires_128_aligned_halves_of_at_least_4096(chunk_rows, expected):
    assert halves_size(chunk_rows) == expected


@pytest.mark.parametrize("live_tokens,expected_kept,expected_rows", [
    (8192, (4096, 4096), (3, 3)),
    (5000, (4096, 904), (3, 3)),
    (1000, (1000, 0), (3, 0)),
])
def test_chunk_halves_begin_slices_metadata_for_full_and_partial_chunks(
    monkeypatch, live_tokens, expected_kept, expected_rows
):
    from hyperflare_recipes.recipes.deepseek41 import decoder_halves

    monkeypatch.setattr(decoder_halves, "AttentionIndices", lambda *args, **kwargs: _IdentityIndices())
    source = Metadata.allocate(8192, 1, 16, "cpu", samples=1, max_query_tokens=8192)
    source.token_ids.copy_(torch.arange(8192, dtype=torch.int64))
    source.positions.copy_(torch.arange(100, 100 + 8192, dtype=torch.int32))
    source.starts.fill_(100)
    source.lengths.fill_(live_tokens)
    source.state_rows.fill_(3)
    source.live_counts[0] = live_tokens

    halves = ChunkHalves(source, 4096, ratios=(1, 2))
    halves.functional()
    assert halves.metadata is None

    metas, geoms = halves.begin(source)
    assert len(metas) == 2 and len(geoms) == 2
    for index, (meta, kept, row) in enumerate(zip(metas, expected_kept, expected_rows, strict=True)):
        first = index * 4096
        assert meta.starts.item() == 100 + first
        assert meta.lengths.item() == kept
        assert meta.query_start_loc.tolist() == [0, kept]
        assert meta.state_rows.item() == row
        assert meta.live_counts[0].item() == kept
        assert meta.sample_indices.item() == max(kept - 1, 0)
        assert torch.equal(meta.token_ids, source.token_ids[first:first + 4096])
        assert torch.equal(meta.positions, source.positions[first:first + 4096])


class _IdentityIndices:
    def functional(self):
        return self

    def __call__(self, meta):
        return meta


def test_pending_halves_carry_materialize_joins_and_concatenates():
    joined = []

    class FakeReduce:
        def __init__(self, tag):
            self.tag = tag

        def join_side(self, pending, *, dependencies=()):
            joined.append(self.tag)
            return pending

    mlp_0 = torch.full((4, 8), 1.0)
    mlp_1 = torch.full((4, 8), 2.0)
    state_0 = (torch.full((4, 4, 8), 3.0), torch.full((4, 4), 4.0), torch.full((4, 4), 5.0))
    state_1 = (torch.full((4, 4, 8), 6.0), torch.full((4, 4), 7.0), torch.full((4, 4), 8.0))
    carry = PendingHalvesCarry(
        pending_mlp=(mlp_0, mlp_1),
        ffn_state=(state_0, state_1),
        mlp_reduce=(FakeReduce("half0"), FakeReduce("half1")),
    )
    mlp, res, post, comb = carry.materialize()
    assert joined == ["half0", "half1"]
    assert torch.equal(mlp[:4], mlp_0) and torch.equal(mlp[4:], mlp_1)
    assert torch.equal(res[:4], state_0[0]) and torch.equal(res[4:], state_1[0])
    assert torch.equal(post[:4], state_0[1]) and torch.equal(post[4:], state_1[1])
    assert torch.equal(comb[:4], state_0[2]) and torch.equal(comb[4:], state_1[2])


@pytest.mark.parametrize("has_selectors", [False, True])
def test_prepared_halves_layer_propagates_selected_indices_from_both_halves(has_selectors):
    h = 4
    layer = object.__new__(PreparedHalvesLayer)
    layer.size = h
    layer.sides = ("side_a", "side_b")
    layer.engram = None
    layer.has_image_mask = False
    layer.boundary = None

    def make_hc():
        def run(x, *rest, pre_mix, pre_out):
            return x, pre_mix, pre_mix, x
        return run

    class FakeAttention:
        def __init__(self, fill_value):
            self.selectors = [object()] if has_selectors else []
            self.fill_value = fill_value

        def _run(self, norm, meta, sel, *, indices, window_cache):
            updated = torch.full_like(sel[0], self.fill_value) if has_selectors else sel[0]
            return norm, None, (updated, None, None), {}

    class FakeReduce:
        def start_side(self, source, side, *, after=None, dependencies=()):
            return SimpleNamespace(source=source, output=source, side=side)

        def join_side(self, pending, *, dependencies=()):
            return pending.output

    layer.attention_hc = [make_hc(), make_hc()]
    layer.attention = [FakeAttention(11), FakeAttention(22)]
    layer.attention_reduce = [FakeReduce(), FakeReduce()]
    layer.ffn_hc = [make_hc(), make_hc()]
    layer.mlp_reduce = [FakeReduce(), FakeReduce()]
    layer.post = [None, None]
    layer.router_projection = lambda whole: whole
    layer.router = lambda logits, image_mask=None: SimpleNamespace(ids=None, weights=None, selected_logits=None)
    layer.experts = lambda whole, ids, weights, **kwargs: whole
    layer.shared = lambda whole: torch.zeros_like(whole)

    incoming = (torch.ones((2 * h, 8), dtype=torch.bfloat16),)
    pre_mix = torch.ones((2 * h, 4), dtype=torch.float32)
    initial_selected = torch.full((2 * h, 16), -1, dtype=torch.int32)
    selection = (initial_selected, None, None)
    meta = SimpleNamespace(token_ids=torch.zeros(2 * h, dtype=torch.int64))
    next_incoming, _, _, next_selection = layer(
        incoming, pre_mix, meta, None, selection, halves_state=((None, None), (None, None))
    )
    assert isinstance(next_incoming, PendingHalvesCarry)
    if has_selectors:
        assert torch.equal(next_selection[0][:h], torch.full((h, 16), 11, dtype=torch.int32))
        assert torch.equal(next_selection[0][h:], torch.full((h, 16), 22, dtype=torch.int32))
    else:
        assert next_selection is selection


def test_catalog_contexts_skips_list_attention_halves_layers():
    halves_layer = SimpleNamespace(attention=[object(), object()], engram=None, boundary=None)
    regular_attention = SimpleNamespace(
        norm_quant=object(), global_kv=None, mode="prefill", ratio=1, selectors=(), concurrent_query=None
    )
    regular_meta = SimpleNamespace(
        positions=torch.zeros(128, dtype=torch.int32),
        starts=torch.zeros(1, dtype=torch.int32),
        max_query_tokens=128,
    )
    regular_layer = SimpleNamespace(
        attention=regular_attention, engram=None, boundary=None,
        metadata=regular_meta, carry=False, post=None,
    )
    decoder = SimpleNamespace(layers=[halves_layer, regular_layer])
    forwards = {"prefill8192": SimpleNamespace(decoder=decoder)}
    signatures = {"prefill8192": SimpleNamespace(kind="prefill")}
    contexts = list(catalog_contexts(forwards, signatures))
    assert len(contexts) == 1
    assert contexts[0][0] == "prefill8192" and contexts[0][3] == 1
