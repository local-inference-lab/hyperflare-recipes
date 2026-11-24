# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared attention retains canonical computation across serialization."""

import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from hyperflare_recipes.recipes.sdxl.attention import PreparedAttention, bind_attention_processors


@pytest.mark.parametrize("spatial,cross", [(False, False), (False, True), (True, False)])
def test_attention_dispatch_preserves_canonical_output_and_serialization(spatial, cross):
    torch.manual_seed(42)
    attention = Attention(query_dim=16, cross_attention_dim=24 if cross else None,
                          heads=2, dim_head=8, processor=AttnProcessor2_0(),
                          residual_connection=spatial, norm_num_groups=4 if spatial else None).eval()
    hidden = torch.randn((2, 16, 4, 4) if spatial else (2, 7, 16))
    kwargs = {"encoder_hidden_states": torch.randn(2, 5, 24)} if cross else {"temb": None}
    with torch.no_grad():
        expected = attention(hidden, **kwargs)
        prepared = PreparedAttention(attention)
        assert prepared.attention is attention
        assert torch.equal(prepared(hidden, **kwargs), expected)
        restored = pickle.loads(pickle.dumps(prepared))
        assert torch.equal(restored(hidden, **kwargs), expected)


def test_attention_dispatch_rejects_unprepared_processor_arguments():
    attention = Attention(query_dim=16, heads=2, dim_head=8, processor=AttnProcessor2_0()).eval()
    with pytest.raises(ValueError, match="extra processor arguments"):
        PreparedAttention(attention)(torch.randn(1, 7, 16), scale=1.0)


def test_attention_binding_retains_parameter_ownership():
    attention = Attention(query_dim=16, heads=2, dim_head=8, processor=AttnProcessor2_0()).eval()
    model = torch.nn.Sequential(torch.nn.ModuleList([attention]))
    parameters = tuple(model.parameters())
    assert bind_attention_processors(model) == 1
    assert isinstance(model[0][0], PreparedAttention)
    assert tuple(model.parameters()) == parameters
    assert model[0][0].attention is attention
