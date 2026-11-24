# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare PixArt-Alpha 512 diffusion graphs and canonical Diffusers references.

The fixed profile computes T5 embeddings, classifier-free transformer predictions,
DPM-Solver++ updates and the raw VAE output. Native replay receives token IDs,
padding masks and initial noise; tokenization remains preparation work.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

from hyperflare_recipes.recipes.sdxl.prepare import (
    CASES, TensorSignature, graph_order, recipe_source_identity, remove_output_capture_hooks,
    save_captured_outputs, save_tensor, tensor_contract, tensor_request, verify_eager_forwards, write_json,
)

MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
MODEL_REVISION = "50f702106901db6d0f8b67eb88e814c56ded2692"
NEGATIVE_PROMPT = ""
DEFAULT_STEPS = 20
SEQUENCE_LENGTH = 120
GUIDANCE = 4.5


def inference_steps(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("PixArt steps must be an integer of at least two") from error
    if result < 2:
        raise argparse.ArgumentTypeError("PixArt steps must be an integer of at least two")
    return result


def inputs_for_case(pipe, prompt, seed):
    import torch

    text = pipe._text_preprocessing([NEGATIVE_PROMPT, prompt], clean_caption=True)
    tokens = pipe.tokenizer(text, padding="max_length", max_length=SEQUENCE_LENGTH,
                            truncation=True, add_special_tokens=True, return_tensors="pt")
    return dict(token_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                initial_noise=torch.randn((1, 4, 64, 64), dtype=pipe.text_encoder.dtype,
                                          generator=torch.Generator().manual_seed(seed)),
                guidance=torch.tensor([GUIDANCE], dtype=torch.float32))


class PreparedCaptions:
    """Cache canonical caption cleaning before token-based inference timing."""

    def __init__(self, preprocess, prompt):
        self.cache = {text: preprocess([text], clean_caption=True)[0]
                      for text in (NEGATIVE_PROMPT, prompt)}

    def __call__(self, text, clean_caption=False):
        if not clean_caption:
            raise ValueError("prepared PixArt captions require canonical caption cleaning")
        texts = [text] if isinstance(text, str) else text
        return [self.cache[item] for item in texts]


class PreparedTokenizer:
    """Supply cached IDs and padding masks to the canonical prompt encoder."""

    def __init__(self, pipe, prompt, inputs):
        import torch

        self.tokenizer = pipe.tokenizer
        self.cache = {}
        texts = pipe._text_preprocessing([NEGATIVE_PROMPT, prompt], clean_caption=True)
        for row, text in enumerate(texts):
            options = dict(padding="max_length", max_length=SEQUENCE_LENGTH,
                           truncation=True, add_special_tokens=True, return_tensors="pt")
            if row == 0:
                options["return_attention_mask"] = True
            encoded = self.tokenizer([text], **options)
            for field, name in (("input_ids", "token_ids"), ("attention_mask", "attention_mask")):
                value = inputs[name][row:row + 1]
                if not torch.equal(getattr(encoded, field), value.cpu()):
                    raise ValueError("canonical PixArt tokenization differs from the supplied input tensors")
                setattr(encoded, field, value)
            self.cache[self.key([text], options)] = encoded
            if row == 1:
                options = dict(padding="longest", return_tensors="pt")
                encoded = self.tokenizer([text], **options).to(inputs["token_ids"].device)
                self.cache[self.key([text], options)] = encoded

    @staticmethod
    def key(text, options):
        return (tuple([text] if isinstance(text, str) else text), tuple(sorted(options.items())))

    def __call__(self, text, **options):
        return self.cache[self.key(text, options)]

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def canonical_reference(pipe, prompt, inputs, *, capture, steps=DEFAULT_STEPS):
    """Observe the unchanged pipeline's computation outside its timed runs."""
    import torch

    observed, encoded, handles = {}, [], []
    original_prepare, original_decode = pipe.prepare_latents, pipe.vae.decode
    original_step, original_postprocess = pipe.scheduler.step, pipe.image_processor.postprocess
    index = 0

    def remember(name, value):
        observed[name] = value.detach().cpu().clone()

    def encoder_hook(module, args, output):
        encoded.append(output[0].to(dtype=module.dtype).detach().cpu().clone())

    def prepare_latents(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        remember("initialized_latent", result)
        return result

    def step(*args, **kwargs):
        nonlocal index
        result = original_step(*args, **kwargs)
        remember(f"latent_after_step_{index}", result[0])
        remember(f"solver_prediction_{index}", pipe.scheduler.model_outputs[-1])
        index += 1
        return result

    def decode(*args, **kwargs):
        result = original_decode(*args, **kwargs)
        remember("decoded_image", result[0])
        return result

    pipe.image_processor.postprocess = lambda image, **kwargs: image
    if capture:
        pipe.prepare_latents, pipe.vae.decode, pipe.scheduler.step = prepare_latents, decode, step
        handles.append(pipe.text_encoder.register_forward_hook(encoder_hook))
    try:
        result = pipe(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, latents=inputs["initial_noise"],
                      height=512, width=512, num_inference_steps=steps, guidance_scale=GUIDANCE,
                      max_sequence_length=SEQUENCE_LENGTH, clean_caption=True,
                      use_resolution_binning=True, output_type="pt").images
        if capture:
            if len(encoded) != 2 or index != steps:
                raise ValueError("PixArt must encode both prompts and execute the complete solver schedule")
            observed["text_embeddings"] = torch.cat((encoded[1], encoded[0]))
        return observed if capture else result
    finally:
        pipe.prepare_latents, pipe.vae.decode = original_prepare, original_decode
        pipe.scheduler.step, pipe.image_processor.postprocess = original_step, original_postprocess
        for handle in handles:
            handle.remove()


def save_references(pipe, root, *, steps=DEFAULT_STEPS):
    import torch

    cases = {}
    for name, prompt, seed in CASES:
        print(f"REFERENCE={name}", flush=True)
        directory = root / name
        directory.mkdir(parents=True)
        host_inputs = inputs_for_case(pipe, prompt, seed)
        input_metadata = [dict(name=key, **save_tensor(directory, Path("inputs") / f"{key}.bin", value))
                          for key, value in host_inputs.items()]
        inputs = {key: value.to("cuda") for key, value in host_inputs.items()}
        original_tokenizer, original_preprocess = pipe.tokenizer, pipe._text_preprocessing
        pipe.tokenizer = PreparedTokenizer(pipe, prompt, inputs)
        pipe._text_preprocessing = PreparedCaptions(original_preprocess, prompt)
        try:
            first = canonical_reference(pipe, prompt, inputs, capture=True, steps=steps)
            second = canonical_reference(pipe, prompt, inputs, capture=True, steps=steps)
            outputs = []
            for key, value in first.items():
                item = dict(name=key, **save_tensor(directory, Path("reference") / f"{key}.bin", value))
                repeated = save_tensor(directory, Path("repeat") / f"{key}.bin", second[key])
                item.update(repeat_file=repeated["file"], repeat_sha256=repeated["sha256"])
                outputs.append(item)
            for _ in range(3):
                canonical_reference(pipe, prompt, inputs, capture=False, steps=steps)
            wall_ms, gpu_ms = [], []
            for _ in range(5):
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                torch.cuda.synchronize()
                started = time.perf_counter()
                begin.record()
                result = canonical_reference(pipe, prompt, inputs, capture=False, steps=steps)
                end.record()
                end.synchronize()
                wall_ms.append((time.perf_counter() - started) * 1000)
                gpu_ms.append(begin.elapsed_time(end))
                del result
            write_json(directory / "reference.json", dict(
                model=MODEL_ID, revision=MODEL_REVISION, prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                seed=seed, steps=steps, inputs=input_metadata, outputs=outputs,
                profile=profile(steps), component_precision=component_precision(pipe), timing=dict(wall_ms=wall_ms, gpu_ms=gpu_ms, warmups=3,
                boundary="GPU token IDs, padding masks and noise to raw decoded VAE tensor",
                includes=f"T5 encoder, {steps} DPM-Solver++ transformer updates and VAE; no caption cleaning, tokenization, observation hooks or image postprocessing",
                vae_precision="canonical PixArt decoder precision; no SDXL-style VAE weight conversion")))
            cases[name] = host_inputs
            del first, second, inputs
        finally:
            pipe.tokenizer, pipe._text_preprocessing = original_tokenizer, original_preprocess
    return cases


def component_precision(pipe):
    """Record the actual loaded parameter precision, including retained T5 outputs."""
    result = {}
    for name in ("text_encoder", "transformer", "vae"):
        counts = {}
        retained = {}
        for parameter_name, parameter in getattr(pipe, name).named_parameters():
            dtype = str(parameter.dtype)
            counts[dtype] = counts.get(dtype, 0) + parameter.numel()
            if name == "text_encoder" and ".wo." in parameter_name:
                retained[parameter_name] = dtype
        result[name] = dict(parameter_elements_by_dtype=counts)
        if retained:
            result[name]["feed_forward_output_projection_dtypes"] = retained
    return result


def profile(steps):
    return dict(height=512, width=512, batch=1, steps=steps, guidance=GUIDANCE,
                sequence_length=SEQUENCE_LENGTH, clean_caption=True, resolution_binning=True,
                scheduler="DPMSolverMultistepScheduler", algorithm="dpmsolver++",
                solver_order=2, solver_type="midpoint", prediction_type="epsilon")


@dataclass(frozen=True)
class PreparationNamespace:
    revision: str
    source_identity: str
    steps: int = DEFAULT_STEPS

    def to_dict(self):
        return dict(recipe="pixart-alpha-fixed-tensor", revision=self.revision,
                    source_identity=self.source_identity, **profile(self.steps))


class T5Conditioning:
    """Retain canonical T5 blocks while omitting observation-only decorators."""

    def __init__(self, model):
        if model.encoder.is_decoder or model.encoder.config.use_cache:
            raise ValueError("PixArt text preparation requires the T5 encoder without a cache")
        self.config = model.encoder.config
        self.embeddings = model.encoder.embed_tokens
        self.blocks = model.encoder.block
        self.dropout = model.encoder.dropout
        self.final_layer_norm = model.encoder.final_layer_norm
        self.dtype = model.dtype

    def __call__(self, token_ids, attention_mask):
        from transformers.masking_utils import create_bidirectional_mask

        hidden = self.embeddings(token_ids)
        attention_mask = create_bidirectional_mask(config=self.config, inputs_embeds=hidden,
                                                   attention_mask=attention_mask)
        position_bias = None
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden, position_bias, _ = block(hidden, attention_mask, position_bias,
                                             None, None, None, past_key_values=None)
        hidden = self.final_layer_norm(hidden)
        return self.dropout(hidden).to(dtype=self.dtype)


class TextForward:
    def __init__(self, pipe, buffers):
        remove_output_capture_hooks(pipe.text_encoder)
        self.encoder = T5Conditioning(pipe.text_encoder)
        self.ids, self.mask = buffers["token_ids"], buffers["attention_mask"]
        self.output = buffers["text_embeddings"]

    def __call__(self):
        import torch

        positive = self.encoder(self.ids[1:2], self.mask[1:2])
        negative = self.encoder(self.ids[0:1], self.mask[0:1])
        self.output.copy_(torch.cat((negative, positive)))


class InitialForward:
    def __init__(self, buffers, sigma):
        self.noise, self.output, self.sigma = buffers["initial_noise"], buffers["initialized_latent"], sigma

    def __call__(self):
        self.output.copy_(self.noise * self.sigma)


class DenoiseForward:
    def __init__(self, pipe, buffers, index):
        from hyperflare_recipes.recipes.pixart.scheduler import PreparedDPMStep

        self.transformer = pipe.transformer
        self.input = buffers["initialized_latent" if index == 0 else f"latent_after_step_{index - 1}"]
        self.output = buffers[f"latent_after_step_{index}"]
        self.converted = buffers[f"solver_prediction_{index}"]
        self.previous_converted = buffers[f"solver_prediction_{index - 1}"] if index else None
        self.sequence, self.mask = buffers["text_embeddings"], buffers["attention_mask"]
        self.guidance = buffers["guidance"]
        self.timestep = pipe.scheduler.timesteps[index:index + 1].expand(2)
        self.solver = PreparedDPMStep(pipe.scheduler, index)

    def __call__(self):
        import torch

        sample = torch.cat((self.input, self.input))
        prediction = self.transformer(sample, encoder_hidden_states=self.sequence,
                encoder_attention_mask=self.mask, timestep=self.timestep,
                added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0]
        unconditional, conditional = prediction.chunk(2)
        prediction = unconditional + self.guidance[0] * (conditional - unconditional)
        prediction = prediction.chunk(2, dim=1)[0]
        converted, previous = self.solver(prediction, self.input, self.previous_converted)
        self.converted.copy_(converted)
        self.output.copy_(previous)


class DecodeForward:
    def __init__(self, pipe, buffers, steps):
        self.vae = pipe.vae
        self.input, self.output = buffers[f"latent_after_step_{steps - 1}"], buffers["decoded_image"]
        self.scaling_factor = pipe.vae.config.scaling_factor

    def __call__(self):
        self.output.copy_(self.vae.decode(self.input / self.scaling_factor, return_dict=False)[0])


def allocate_outputs(buffers, reference, *, device):
    import torch

    dtypes = {"F16": torch.float16, "F32": torch.float32}
    for item in reference["outputs"]:
        buffers[item["name"]] = torch.empty(item["shape"], dtype=dtypes[item["dtype"]], device=device)


def prepare_graphs(pipe, storage, images, cases, args):
    import torch
    from hyperflare_recipes.recipes.sdxl.attention import bind_attention_processors
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.core_file import combine
    from ignition.preparation.core_snapshot import export_options
    from ignition.preparation.protection import Publisher
    from ignition.preparation.runtime_core import Allocation, RuntimeCore
    from ignition.preparation.tensor_metadata import live_metadata
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    order = graph_order(args.steps)
    buffers = {name: value.to("cuda") for name, value in cases["case_a"].items()}
    reference = json.loads((args.output / "case_a" / "reference.json").read_text())
    allocate_outputs(buffers, reference, device="cuda")
    pipe.scheduler.set_timesteps(args.steps, device="cuda")
    timesteps, sigmas = pipe.scheduler.timesteps, pipe.scheduler.sigmas
    bind_attention_processors(pipe.transformer)
    bind_attention_processors(pipe.vae)
    functions = dict(text=TextForward(pipe, buffers),
                     init=InitialForward(buffers, float(pipe.scheduler.init_noise_sigma)))
    for index in range(args.steps):
        functions[f"step_{index}"] = DenoiseForward(pipe, buffers, index)
    functions["decode"] = DecodeForward(pipe, buffers, args.steps)
    verify_eager_forwards(functions, buffers, cases, args.output, steps=args.steps)
    for name, value in cases["case_a"].items():
        buffers[name].copy_(value)
    signatures = {name: TensorSignature() for name in functions}
    namespace = PreparationNamespace(MODEL_REVISION, recipe_source_identity(), args.steps)
    session = SimpleNamespace(capture=nullcontext, namespace=namespace,
                              requested_compile_workers=args.compile_workers, headroom_bytes=8 << 30,
                              lock_gpu_clocks=False, autotune=True)
    graphs, compiled = capture_forwards(
        functions, signatures, session=session, device=torch.device("cuda", 0), stream=torch.cuda.Stream(),
        pool=storage.graph_pool.id, warmup_storage=storage, retain_graphs=True,
        cache_root=args.output / "kernel-cache", compile_kinds=("tensor",),
        report_catalog=False, progress=print)
    save_captured_outputs(graphs, buffers, cases, args.output, steps=args.steps)
    allocations = [Allocation(f"device-{index}", mapping.address, mapping.nbytes,
                              alignment=storage.granularity, preserve_address=True)
                   for index, mapping in enumerate(sorted(storage.mappings.values(), key=lambda value: value.address))]
    package = RuntimeCore(allocations=allocations, modules={}, kernels=(), recorder=images.native,
                          owners=(pipe, storage, images, functions, buffers, compiled, graphs))
    numeric_bounds = [*allocations, *(Allocation(f"recorded-{address}", address, size)
                      for address, (size, kind) in images.allocations.items() if kind == "device")]
    zero_ranges = [(segment["address"], segment["total_size"])
                   for segment in storage.graph_pool.snapshot(include_traces=False)]
    inactive = storage.unused_ranges()
    zero_ranges.extend((start, end - start) for start, end in inactive)
    numeric, parameter_layouts = live_metadata(numeric_bounds, storage.numeric_storage, buffers,
                                               zero_ranges=zero_ranges)
    sources = args.output / "rank.sources.json"
    publisher = Publisher(Ed25519PrivateKey.generate(), "local-development", "local-development", development=True)
    with export_options(args.output / "core-cache"):
        package.write(sources, graphs, execution=None, sources=True, zero_ranges=zero_ranges,
                      discover_allocations=True, numeric=numeric, parameter_layouts=parameter_layouts,
                      inactive=inactive, granularity=storage.granularity, roots=(),
                      tensor_contract=tensor_contract(order, buffers, cases["case_a"]))
        for case, inputs in cases.items():
            write_json(args.output / case / "request.json", tensor_request(inputs))
        result = combine(args.output / "core", [sources], [], publisher=publisher, form="directory")
    write_json(args.output / "publisher.json", [dict(name="PixArt local qualification", namespace=publisher.namespace,
                public_key=list(publisher.public_key), development=True)])
    write_json(args.output / "publisher-info.json", dict(namespace=publisher.namespace,
                fingerprint=publisher.fingerprint.hex(), development=True))
    write_json(args.output / "preparation.json", dict(model=MODEL_ID, revision=MODEL_REVISION,
                graphs=list(order), export_counts=package.export_counts,
                core=str(args.output / "core"), publication=vars(result),
                timesteps=timesteps.cpu().tolist(), sigmas=sigmas.cpu().tolist(),
                scheduler_step_sha256=hashlib.sha256(inspect.getsource(type(pipe.scheduler).step).encode()).hexdigest(),
                profile=profile(args.steps), component_precision=component_precision(pipe)))
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=2)
    parser.add_argument("--steps", type=inference_steps, default=DEFAULT_STEPS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch
    from diffusers import DPMSolverMultistepScheduler, PixArtAlphaPipeline
    from diffusers.utils import is_bs4_available, is_ftfy_available
    if not is_bs4_available() or not is_ftfy_available():
        raise ValueError("canonical PixArt caption cleaning requires beautifulsoup4 and ftfy")
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.module_images import ModuleImages

    with ModuleImages() as images:
        torch.cuda.set_device(0)
        storage = CoreStorage()
        with storage, torch.no_grad():
            pipe = PixArtAlphaPipeline.from_pretrained(str(args.snapshot), torch_dtype=torch.float16,
                    use_safetensors=True, local_files_only=True)
            pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            if (not isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
                    or pipe.transformer.config.sample_size != 64 or pipe.transformer.config.in_channels != 4
                    or pipe.transformer.config.out_channels != 8 or pipe.vae_scale_factor != 8
                    or pipe.text_encoder.config.d_model != 4096):
                raise ValueError("checkpoint does not match the fixed PixArt-Alpha 512 profile")
            cases = save_references(pipe, args.output, steps=args.steps)
            prepare_graphs(pipe, storage, images, cases, args)


if __name__ == "__main__":
    # Compiler workers resolve prepared forward types by their importable module.
    from hyperflare_recipes.recipes.pixart.prepare import main as prepare_main

    prepare_main()
