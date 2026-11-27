# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared PixArt composition preserves canonical encoder and denoiser semantics."""

import argparse
import pickle
from types import SimpleNamespace

import pytest

from hyperflare_recipes.recipes.pixart.prepare import (
    DecodeForward, DenoiseForward, PreparationNamespace, PreparedCaptions,
    T5Conditioning, component_precision, inference_steps, remove_output_capture_hooks,
)


@pytest.mark.parametrize("value", ["0", "1", "-2", "1.5", "bad"])
def test_cli_rejects_unsupported_step_counts(value):
    with pytest.raises(argparse.ArgumentTypeError, match="at least two"):
        inference_steps(value)


def test_schedule_identity_and_decoder_use_the_prepared_step_count():
    assert inference_steps("20") == 20
    assert PreparationNamespace("revision", "source", 4).to_dict() != PreparationNamespace("revision", "source", 20).to_dict()
    buffers = {"latent_after_step_19": object(), "decoded_image": object()}
    pipe = SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=0.18215)))
    decoder = DecodeForward(pipe, buffers, 20)
    assert decoder.input is buffers["latent_after_step_19"]
    assert decoder.output is buffers["decoded_image"]


def test_cached_captions_preserve_canonical_cleaning_without_timed_reprocessing():
    pytest.importorskip("torch")
    pytest.importorskip("bs4")
    pytest.importorskip("ftfy")
    from diffusers import PixArtAlphaPipeline

    pipe = PixArtAlphaPipeline.__new__(PixArtAlphaPipeline)
    prompt = "A ROBOT <b>reads</b> a book &amp; smiles."
    expected = pipe._text_preprocessing([prompt], clean_caption=True)
    cached = PreparedCaptions(pipe._text_preprocessing, prompt)
    assert cached(prompt, clean_caption=True) == expected
    assert cached(["", prompt], clean_caption=True) == ["", expected[0]]
    with pytest.raises(ValueError, match="canonical caption cleaning"):
        cached(prompt, clean_caption=False)


def test_t5_conditioning_preserves_padding_masks_and_serialization():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(42)
    config = T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=2,
                      num_heads=4, dropout_rate=0.0, use_cache=False)
    model = T5EncoderModel(config).eval()
    ids = torch.tensor([[4, 5, 6, 1, 0, 0, 0, 0]])
    masks = [torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]]), torch.ones_like(ids)]
    with torch.no_grad():
        expected = [model(ids, attention_mask=mask)[0] for mask in masks]
        remove_output_capture_hooks(model)
        prepared = T5Conditioning(model)
        restored = pickle.loads(pickle.dumps(prepared))
        for mask, reference in zip(masks, expected):
            assert torch.equal(prepared(ids, mask), reference)
            assert torch.equal(restored(ids, mask), reference)
        assert not torch.equal(expected[0], expected[1])


def test_denoiser_matches_canonical_guidance_and_scheduler_history():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers import DPMSolverMultistepScheduler, PixArtTransformer2DModel
    from hyperflare_recipes.recipes.sdxl.attention import bind_attention_processors

    torch.manual_seed(43)
    transformer = PixArtTransformer2DModel(num_layers=1, num_attention_heads=2,
                    attention_head_dim=8, in_channels=4, out_channels=8,
                    cross_attention_dim=16, caption_channels=32, sample_size=8,
                    patch_size=2, use_additional_conditions=False).eval()
    scheduler = DPMSolverMultistepScheduler()
    scheduler.set_timesteps(4)
    initial = torch.randn(1, 4, 8, 8)
    sequence = torch.randn(2, 8, 32)
    mask = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    expected = []
    sample = initial.clone()
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            prediction = transformer(torch.cat([sample] * 2), encoder_hidden_states=sequence,
                encoder_attention_mask=mask, timestep=timestep[None].expand(2),
                added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0]
            unconditional, conditional = prediction.chunk(2)
            prediction = (unconditional + 4.5 * (conditional - unconditional)).chunk(2, dim=1)[0]
            sample = scheduler.step(prediction, timestep, sample, return_dict=False)[0]
            expected.append((sample.clone(), scheduler.model_outputs[-1].clone()))
        scheduler.set_timesteps(4)
        bind_attention_processors(transformer)
        buffers = dict(initialized_latent=initial, text_embeddings=sequence, attention_mask=mask,
                       guidance=torch.tensor([4.5]))
        for index in range(4):
            buffers[f"latent_after_step_{index}"] = torch.empty_like(initial)
            buffers[f"solver_prediction_{index}"] = torch.empty_like(initial)
        pipe = SimpleNamespace(transformer=transformer, scheduler=scheduler)
        forwards = [DenoiseForward(pipe, buffers, index) for index in range(4)]
        for index, forward in enumerate(forwards):
            forward()
            assert torch.equal(buffers[f"latent_after_step_{index}"], expected[index][0]), index
            assert torch.equal(buffers[f"solver_prediction_{index}"], expected[index][1]), index


def test_precision_metadata_records_mixed_t5_weights():
    torch = pytest.importorskip("torch")
    class Component(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wi = torch.nn.Linear(2, 4, bias=False, dtype=torch.float16)
            self.wo = torch.nn.Linear(4, 2, bias=False, dtype=torch.float32)
    pipe = SimpleNamespace(text_encoder=torch.nn.Sequential(Component()),
                           transformer=torch.nn.Linear(2, 2, dtype=torch.float16),
                           vae=torch.nn.Linear(2, 2, dtype=torch.float32))
    result = component_precision(pipe)
    assert result["text_encoder"]["parameter_elements_by_dtype"] == {"torch.float16": 8, "torch.float32": 8}
    assert result["text_encoder"]["feed_forward_output_projection_dtypes"] == {"0.wo.weight": "torch.float32"}


def test_recipe_source_changes_preparation_cache_identity():
    first = PreparationNamespace("revision", "source-a").to_dict()
    second = PreparationNamespace("revision", "source-b").to_dict()
    assert first != second
    assert {key: value for key, value in first.items() if key != "source_identity"} == {
        key: value for key, value in second.items() if key != "source_identity"}
