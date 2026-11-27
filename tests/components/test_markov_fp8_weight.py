# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP8 Markov checkpoint conversion retains explicit row reconstruction scales."""

import pytest
import torch

from hyperflare_recipes.components.markov_sampling import RowScaledFP8MarkovWeight, fp8_markov_head


def test_markov_fp8_rows_follow_amax_scale_and_preserve_checkpoint_weight():
    weight = torch.stack((torch.linspace(-1, 1, 64), torch.zeros(64),
                          torch.arange(64).float() / 8)).bfloat16()
    saved = weight.clone()
    packed = fp8_markov_head(weight, rows_per_chunk=1)
    expected_scale = weight.float().abs().amax(dim=1).clamp_min(1e-12) / 448.0
    expected_values = (weight.float() / expected_scale[:, None]).to(torch.float8_e4m3fn)
    assert packed.out_features == 3 and packed.in_features == 64
    assert torch.equal(packed.scale, expected_scale)
    assert torch.equal(packed.values.view(torch.uint8), expected_values.view(torch.uint8))
    assert torch.equal(weight, saved)


def test_markov_fp8_rows_reject_invalid_weights_and_storage():
    with pytest.raises(ValueError, match="contiguous BF16"):
        fp8_markov_head(torch.zeros((2, 64), dtype=torch.float32))
    with pytest.raises(ValueError, match="finite checkpoint"):
        fp8_markov_head(torch.full((2, 64), float("nan"), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="row scales"):
        RowScaledFP8MarkovWeight(torch.empty((2, 64), dtype=torch.float8_e4m3fn),
                                torch.empty((2,), dtype=torch.bfloat16))
    for shape in ((0, 64), (2, 0), (2, 32)):
        with pytest.raises(ValueError, match="contiguous BF16"):
            fp8_markov_head(torch.empty(shape, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="row-scaled FP8"):
            RowScaledFP8MarkovWeight(torch.empty(shape, dtype=torch.float8_e4m3fn),
                                    torch.empty((shape[0],), dtype=torch.float32))


def test_markov_fp8_weight_views_and_numeric_owners_survive_preparation(monkeypatch):
    from hyperflare_recipes.components import markov_sampling
    from ignition.preparation.rebuild_manifest import rebuild_weights_metadata
    from ignition.preparation.tensor_metadata import NumericStorage

    packed = fp8_markov_head(torch.ones((2, 64), dtype=torch.bfloat16))
    source = torch.zeros((1, 64), dtype=torch.bfloat16)
    output = torch.empty((1, 2), dtype=torch.float32)
    monkeypatch.setattr(markov_sampling, "register_finalizer", lambda session, callback: None)
    with NumericStorage() as numeric:
        projection = markov_sampling.PreparedRowScaledFP8MarkovProjection(source, packed, output,
            session=object(), name="markov")
    assert projection.weight is packed
    ranges = numeric.ranges()
    for tensor in (packed.values, packed.scale):
        assert any(start <= tensor.data_ptr() and end >= tensor.data_ptr() + tensor.nbytes
                   for start, end in ranges)
    catalog = rebuild_weights_metadata({"markov": packed})
    views = {item["name"]: item for item in catalog["tensors"]}
    assert views["markov.values"]["dtype"] == "float8_e4m3fn"
    assert views["markov.scale"]["dtype"] == "float32"
    assert not catalog["unavailable"]


def test_vocabulary_score_tiles_markov_transition_bias_cpu_interpreter():
    import os
    from pathlib import Path
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _score_tiles_markov_cases():
    from hyperflare_recipes.components import vocabulary_sampling as v

    torch.manual_seed(0)
    vocab, rank, block = 32, 64, 32
    logits = torch.randn((1, 1, vocab), dtype=torch.float32)
    embedding = torch.randn((vocab, rank), dtype=torch.bfloat16)
    head_bf16 = torch.randn((vocab, rank), dtype=torch.bfloat16)
    head_fp8 = fp8_markov_head(head_bf16)
    pending = torch.tensor([7], dtype=torch.int64)
    drafts = torch.zeros((1, 1), dtype=torch.int64)
    histogram = torch.zeros((1, vocab), dtype=torch.int32)
    slots = torch.zeros((1,), dtype=torch.int32)
    generations = torch.ones((1,), dtype=torch.int64)
    live_rows = torch.ones((1,), dtype=torch.int32)
    generation = torch.ones((1,), dtype=torch.int64)
    status = torch.ones((1,), dtype=torch.int32)
    params = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    top_k = torch.zeros((1,), dtype=torch.int32)
    grammar = torch.zeros((1, 1), dtype=torch.int32)
    grammar_enabled = torch.zeros((1,), dtype=torch.int32)
    seeds = torch.zeros((1,), dtype=torch.int64)
    offsets = torch.zeros((1,), dtype=torch.int64)
    limits = torch.ones((1,), dtype=torch.int32)
    vocab_offset = torch.zeros((1,), dtype=torch.int32)

    for is_fp8, head_vals, head_scale in (
        (False, head_bf16, head_bf16),
        (True, head_fp8.values, head_fp8.scale),
    ):
        probs = torch.empty((1, 1, vocab), dtype=torch.float32)
        partials = torch.empty((1, 6, 1), dtype=torch.float32)
        v._score_tiles[(1, 1)](
            vocab_offset,
            logits,
            probs,
            partials,
            embedding,
            head_vals,
            head_scale,
            pending,
            drafts,
            histogram,
            slots,
            generations,
            live_rows,
            generation,
            status,
            params,
            top_k,
            grammar,
            grammar_enabled,
            seeds,
            offsets,
            limits,
            logits.stride(0),
            logits.stride(1),
            histogram.stride(0),
            grammar.stride(0),
            LOCAL=vocab,
            VOCAB=vocab,
            WIDTH=1,
            DRAFTS=1,
            PREFIX=0,
            BLOCKS=1,
            CAPACITY=1,
            BATCH=1,
            RUNNING=1,
            MARKOV=True,
            RANK=rank,
            FP8=is_fp8,
            SAMPLE=False,
            STEP=0,
            BLOCK=block,
            HAS_LIMITS=False,
        )
        ref_head = head_vals.float() * head_scale[:, None].float() if is_fp8 else head_vals.float()
        expected = logits[0, 0] + ref_head @ embedding[7].float()
        torch.testing.assert_close(probs[0, 0], expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    _score_tiles_markov_cases()
