# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Kimi K3 and Kimi K3.1 lossless KV codec layout declarations."""

from types import SimpleNamespace

from google.protobuf.json_format import ParseDict
import pytest
import torch

from hyperflare_recipes.recipes.kimi3.config import DSAConfig, DSparkConfig, ModelConfig, PAGE_TOKENS
from hyperflare_recipes.recipes.kimi3.contract import checkpoint_encoding, prepare
from hyperflare_recipes.recipes.kimi3.dcp import RECORD_BYTES
from hyperflare_recipes.recipes.kimi3.pool import continuation_layout, page_specs
from hyperflare_recipes.recipes.kimi31.dflash_config import DFlash2Config
from hyperflare_recipes.recipes.kimi31.dspark_config import DSparkConfig as FullMLADSparkConfig
from ignition.preparation import runtime_core_pb2 as wire


def _minimal_config(*, dsa=False, draft=None):
    text = dict(
        hidden_size=7168,
        num_hidden_layers=4,
        linear_attn_config=dict(num_heads=96, kda_layers=[1, 2, 3]),
    )
    dsa_cfg = DSAConfig(heads=32, head_dim=128, top_k=2048, layers=(4,)) if dsa else None
    return ModelConfig(text=text, vision={}, draft=draft, dsa=dsa_cfg)


def _export_codec_layout(config, *, speculative_tokens=0):
    layout = continuation_layout(config, world_size=16, speculative_tokens=speculative_tokens)
    pages = {name: torch.zeros((2, *shape), dtype=torch.uint8) for name, shape in page_specs(config)}
    pool = SimpleNamespace(layout=layout, pages=pages)
    buffers = {}
    contract_dict = prepare(
        signatures={},
        buffers=buffers,
        pool=pool,
        vocabulary=163840,
        page_tokens=PAGE_TOKENS,
        draft_lag=0 if config.draft is not None else None,
        vision=False,
    )
    contract = ParseDict(contract_dict, wire.ModelContract())
    return layout, contract.cache.codec_layout


def test_kimi3_plain_exports_latent576_and_kda_encodings():
    config = _minimal_config(dsa=False, draft=None)
    layout, codec = _export_codec_layout(config, speculative_tokens=0)
    assert codec.revision == 1

    by_offset = {(f.buffer, f.offset): (f.length, f.encoding) for f in codec.fields}
    for region in layout.regions:
        length, encoding = by_offset[("state_pool", region.offset)]
        assert length == region.nbytes
        if region.name.endswith(".recurrent"):
            assert encoding == 1
        elif region.name.endswith(".convolution"):
            assert encoding == 2
        elif region.name == "layer3.latent":
            assert encoding == 48
        elif region.name == "last_hidden":
            assert encoding == 2
        elif region.name == "consumed_tokens":
            assert encoding == 0
        else:
            pytest.fail(f"unexpected region {region.name}")

    assert by_offset[("cache.layer3.latent", 0)] == (PAGE_TOKENS * RECORD_BYTES, 48)


def test_kimi31_full_context_bf16_dspark_separates_latent576_indexer_and_bf16_draft():
    draft = FullMLADSparkConfig(raw=dict(num_hidden_layers=2, sliding_window=None))
    config = _minimal_config(dsa=True, draft=draft)
    layout, codec = _export_codec_layout(config, speculative_tokens=7)

    by_offset = {(f.buffer, f.offset): (f.length, f.encoding) for f in codec.fields}
    for region in layout.regions:
        length, encoding = by_offset[("state_pool", region.offset)]
        assert length == region.nbytes
        if region.name.endswith(".recurrent"):
            assert encoding == 1
        elif region.name.endswith(".convolution"):
            assert encoding == 20  # 3 + 7 = 10 bf16 lanes -> Convolution20
        elif region.name == "layer3.latent":
            assert encoding == 48
        elif region.name == "layer3.indexer":
            assert encoding == 41
        elif region.name.startswith("draft") and region.name.endswith(".latent"):
            # Full-context DSpark stores BF16 latent records in uint8 backing -> Bfloat16 (2), never 48.
            assert encoding == 2
        elif region.name == "last_hidden":
            assert encoding == 2
        elif region.name == "consumed_tokens":
            assert encoding == 0
        else:
            pytest.fail(f"unexpected region {region.name}")

    assert by_offset[("cache.layer3.latent", 0)] == (PAGE_TOKENS * RECORD_BYTES, 48)
    assert by_offset[("cache.layer3.indexer", 0)] == (PAGE_TOKENS * 132, 41)
    assert by_offset[("cache.draft0.latent", 0)] == (PAGE_TOKENS * RECORD_BYTES * 2, 2)
    assert by_offset[("cache.draft1.latent", 0)] == (PAGE_TOKENS * RECORD_BYTES * 2, 2)


def test_kimi31_dflash2_e4m3_draft_and_qwen3_dspark_window_encodings():
    dflash = DFlash2Config(raw=dict(num_hidden_layers=2, sliding_window=512))
    config = _minimal_config(dsa=True, draft=dflash)
    layout, codec = _export_codec_layout(config, speculative_tokens=7)
    by_offset = {(f.buffer, f.offset): (f.length, f.encoding) for f in codec.fields}
    assert by_offset[("state_pool", layout.region("draft0.latent").offset)] == (512 * RECORD_BYTES, 48)
    assert by_offset[("state_pool", layout.region("draft1.latent").offset)] == (512 * RECORD_BYTES, 48)
    assert by_offset[("cache.layer3.latent", 0)] == (PAGE_TOKENS * RECORD_BYTES, 48)
    assert by_offset[("cache.layer3.indexer", 0)] == (PAGE_TOKENS * 132, 41)

    qwen_dspark = DSparkConfig(raw=dict(transformer_layer_config=dict(
        num_hidden_layers=1, num_key_value_heads=16, head_dim=128, sliding_window=256)))
    config_q = _minimal_config(dsa=False, draft=qwen_dspark)
    layout_q, codec_q = _export_codec_layout(config_q, speculative_tokens=7)
    by_offset_q = {(f.buffer, f.offset): (f.length, f.encoding) for f in codec_q.fields}
    assert by_offset_q[("state_pool", layout_q.region("draft0.key").offset)][1] == 2
    assert by_offset_q[("state_pool", layout_q.region("draft0.value").offset)][1] == 2


def test_checkpoint_encoding_does_not_match_arbitrary_73728_byte_buffer():
    # Matching byte count alone must not select Latent576 (48).
    assert checkpoint_encoding("cache.other.blob", torch.uint8, (73728,)) == 0
    assert checkpoint_encoding("draft0.latent", torch.uint8, (73728,)) == 0
    assert checkpoint_encoding("draft0.latent", torch.float8_e4m3fn, (64, 576)) == 0
    assert checkpoint_encoding("cache.layer0.latent", torch.uint8, (73728,)) == 48
