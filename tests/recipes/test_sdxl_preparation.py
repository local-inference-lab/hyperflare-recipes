# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation declares tensor geometry separately from runtime input files."""

from types import SimpleNamespace
import argparse
import sys

import pytest

from hyperflare_recipes.recipes.sdxl.prepare import (
    DecodeForward, GRAPH_ORDER, PreparationNamespace, canonical_reference,
    graph_order, positive_steps, remove_output_capture_hooks, tensor_binding, tensor_contract, tensor_request,
)


def tensor(address=1040, *, contiguous=True, elements=8):
    return SimpleNamespace(data_ptr=lambda: address, numel=lambda: elements,
                           element_size=lambda: 2, shape=(2, 4), dtype="torch.float16",
                           is_contiguous=lambda: contiguous)


def test_preparation_binding_retains_address_extent_and_geometry():
    assert tensor_binding("latent", tensor()) == dict(
        name="latent", address=1040, nbytes=16, dtype="F16", shape=[2, 4])


@pytest.mark.parametrize("options", [{"contiguous": False}, {"elements": 0}])
def test_preparation_rejects_unrepresentable_tensor_views(options):
    with pytest.raises(ValueError, match="nonempty and contiguous"):
        tensor_binding("latent", tensor(**options))


@pytest.mark.parametrize("steps", [4, 30])
def test_prepared_contract_owns_schedule_and_views_while_request_only_names_files(steps):
    inputs = {"initial_noise": tensor()}
    buffers = {**inputs, "initialized_latent": tensor(1100)}
    buffers.update({f"latent_after_step_{index}": tensor(1200 + index * 16) for index in range(steps)})
    buffers["decoded_image"] = tensor(4096)
    contract = tensor_contract(graph_order(steps), buffers, inputs)
    assert contract["version"] == 1
    assert contract["graphs"] == list(graph_order(steps))
    assert contract["inputs"] == [tensor_binding("initial_noise", inputs["initial_noise"])]
    assert [item["name"] for item in contract["outputs"]] == list(buffers)[1:]
    assert contract["outputs"][-2]["name"] == f"latent_after_step_{steps - 1}"
    assert tensor_request(inputs) == {"inputs": [{"name": "initial_noise", "file": "inputs/initial_noise.bin"}]}


def test_prepare_removes_only_transformers_output_observation_hooks():
    def capture(module, args, output):
        return None

    capture.__module__ = "transformers.utils.output_capturing"
    capture.__qualname__ = "install_output_capuring_hook.<locals>.output_capturing_hook"

    def application_hook(module, args, output):
        return output

    layer = SimpleNamespace(_forward_hooks={1: capture, 2: application_hook},
                            _forward_hooks_with_kwargs={1: True, 2: True},
                            _forward_hooks_always_called={1: True})
    model = SimpleNamespace(modules=lambda: [layer])
    assert remove_output_capture_hooks(model) == 1
    assert layer._forward_hooks == {2: application_hook}
    assert layer._forward_hooks_with_kwargs == {2: True}
    assert layer._forward_hooks_always_called == {}
    assert remove_output_capture_hooks(model) == 0


@pytest.mark.parametrize("steps", [4, 30])
def test_schedule_changes_graph_order_cache_identity_and_decoder_input(steps):
    order = graph_order(steps)
    assert order[:2] == ("text", "init")
    assert order[2:-1] == tuple(f"step_{index}" for index in range(steps))
    assert order[-1] == "decode"
    assert len(order) == steps + 3
    buffers = {f"latent_after_step_{index}": object() for index in range(steps)}
    buffers["decoded_image"] = object()
    vae = SimpleNamespace(post_quant_conv=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="F32")])),
                          config=SimpleNamespace(scaling_factor=0.13025))
    decoder = DecodeForward(SimpleNamespace(vae=vae), buffers, steps)
    assert decoder.input is buffers[f"latent_after_step_{steps - 1}"]
    assert decoder.output is buffers["decoded_image"]
    assert PreparationNamespace("revision", "source", steps).to_dict()["steps"] == steps
    assert PreparationNamespace("revision", "source", 4).to_dict() != PreparationNamespace("revision", "source", 30).to_dict()


def test_four_step_defaults_are_preserved():
    assert graph_order() == GRAPH_ORDER == ("text", "init", "step_0", "step_1", "step_2", "step_3", "decode")
    assert PreparationNamespace("revision", "source").to_dict()["steps"] == 4
    assert positive_steps("30") == 30


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_cli_rejects_invalid_step_counts(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        positive_steps(value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_graph_order_rejects_invalid_step_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        graph_order(value)


@pytest.mark.parametrize("steps", [4, 30])
def test_canonical_pipeline_receives_requested_step_count(monkeypatch, steps):
    class Pipeline:
        def __init__(self):
            self.prepare_latents = object()
            self.vae = SimpleNamespace(decode=object())
            self.image_processor = SimpleNamespace(postprocess=object())

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images="raw image")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    pipe = Pipeline()
    original = pipe.image_processor.postprocess
    assert canonical_reference(pipe, "prompt", {"initial_noise": "noise"}, capture=False, steps=steps) == "raw image"
    assert pipe.kwargs["num_inference_steps"] == steps
    assert pipe.kwargs["latents"] == "noise"
    assert pipe.image_processor.postprocess is original


def test_recipe_source_changes_preparation_cache_identity():
    first = PreparationNamespace("revision", "source-a").to_dict()
    second = PreparationNamespace("revision", "source-b").to_dict()
    assert first != second
    assert {key: value for key, value in first.items() if key != "source_identity"} == {
        key: value for key, value in second.items() if key != "source_identity"}
