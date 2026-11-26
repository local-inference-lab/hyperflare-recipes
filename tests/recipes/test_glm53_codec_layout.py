# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GLM 5.3 lossless KV codec layout declarations."""

from types import SimpleNamespace

from google.protobuf.json_format import ParseDict
import pytest
import torch

from hyperflare_recipes.recipes.glm53.contract import checkpoint_encoding, prepare
from hyperflare_recipes.recipes.glm53.pool import continuation_layout
from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as wire


_LAYERS = ("linear_attention", "linear_attention", "linear_attention", "indexed_attention")


def _export_glm53_codec_layout(world_size, *, mtp):
    layout = continuation_layout(_LAYERS, world_size, mtp=mtp)
    pool = SimpleNamespace(layout=layout)
    buffers = {}
    cache_regions = []
    mla_layers = [3, 4] if mtp else [3]
    for layer in mla_layers:
        latent = torch.zeros((2, 4, 64, 528), dtype=torch.uint8)
        index = torch.zeros((2, 64, 132), dtype=torch.uint8)
        cache_regions.append(c.history_region(buffers, f"cache.{layer}.latent", latent, wire.PAGED_LATENT))
        cache_regions.append(c.history_region(buffers, f"cache.{layer}.index", index, wire.PAGED_LATENT))
    contract_dict = prepare(
        signatures={},
        buffers=buffers,
        pool=pool,
        page_tokens=256 * world_size,
        draft_lag=1 if mtp else None,
        regions=cache_regions,
    )
    contract = ParseDict(contract_dict, wire.ModelContract())
    return layout, contract.cache.codec_layout, mla_layers


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("mtp", [False, True])
def test_glm53_exports_codec_layout_across_tp_and_mtp(world_size, mtp):
    layout, codec, mla_layers = _export_glm53_codec_layout(world_size, mtp=mtp)
    assert codec.revision == 1

    by_offset = {(f.buffer, f.offset): (f.length, f.encoding) for f in codec.fields}
    for region in layout.regions:
        length, encoding = by_offset[("state_pool", region.offset)]
        assert length == region.nbytes
        if region.name.endswith(".recurrent"):
            assert encoding == 1
            assert length % 65_536 == 0
        elif region.name.endswith(".convolution"):
            assert encoding == (20 if mtp else 2)
        elif region.name.endswith(".index_tail") or region.name == "last_hidden":
            assert encoding == 2
        elif region.name.endswith(".latent_page"):
            assert encoding == 40
            assert length % 135_168 == 0
        elif region.name.endswith(".index_page"):
            assert encoding == 41
            assert length % 8_448 == 0
        elif region.name == "consumed_tokens":
            assert encoding == 0
        else:
            pytest.fail(f"unexpected region {region.name}")

    for layer in mla_layers:
        assert by_offset[(f"cache.{layer}.latent", 0)] == (4 * 64 * 528, 40)
        assert by_offset[(f"cache.{layer}.index", 0)] == (64 * 132, 41)


def test_checkpoint_encoding_defaults_unrecognized_fields_to_bytes():
    assert checkpoint_encoding("cache.other.blob", torch.uint8, (4, 64, 528)) == 0
    assert checkpoint_encoding("cache.other.blob", torch.uint8, (64, 132)) == 0
    assert checkpoint_encoding("consumed_tokens", torch.int64, (1,)) == 0
