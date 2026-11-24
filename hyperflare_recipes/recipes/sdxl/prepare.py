# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export SDXL GPU graphs and independent Diffusers reference tensors.

The prepared profile uses SDXL Base 1.0, one 1024-square image, classifier-free
guidance and a fixed number of Euler updates. Tokenization remains preparation work. Native
execution consumes token IDs and noise, and computes text encodings, latents
and the raw VAE output without Python.
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


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
NEGATIVE_PROMPT = "blurry, low quality"
CASES = (
    ("case_a", "A red sailboat on a quiet lake, mountains in the distance, daylight", 42),
    ("case_b", "A blue robot reading a book in a sunny garden, detailed illustration", 43),
)
DEFAULT_STEPS = 4
DTYPES = {"torch.float16": "F16", "torch.float32": "F32", "torch.int64": "I64"}


def graph_order(steps=DEFAULT_STEPS):
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    return ("text", "init", *(f"step_{index}" for index in range(steps)), "decode")


GRAPH_ORDER = graph_order()


def positive_steps(value):
    try:
        steps = int(value)
        graph_order(steps)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("steps must be a positive integer") from error
    return steps


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def save_tensor(root, relative, tensor):
    import torch

    packed = tensor.detach().cpu().contiguous()
    if packed.is_floating_point() and not torch.isfinite(packed).all():
        raise ValueError(f"nonfinite values in {relative}")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = packed.view(torch.uint8).numpy().tobytes()
    path.write_bytes(data)
    return dict(dtype=DTYPES[str(packed.dtype)], shape=list(packed.shape),
                file=str(relative), nbytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def inputs_for_case(pipe, prompt, seed):
    import torch

    tokens = [tokenizer([NEGATIVE_PROMPT, prompt], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids
              for tokenizer in (pipe.tokenizer, pipe.tokenizer_2)]
    return {
        "token_ids_1": tokens[0], "token_ids_2": tokens[1],
        "initial_noise": torch.randn((1, 4, 128, 128), dtype=torch.float16,
                                     generator=torch.Generator().manual_seed(seed)),
        "guidance": torch.tensor([5.0], dtype=torch.float32),
        "size_crop": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * 2, dtype=torch.float16),
    }


class PreparedTokenizer:
    """Supply pretokenized GPU inputs to the unchanged Diffusers prompt encoder."""

    def __init__(self, tokenizer, prompt, token_ids):
        import torch

        self.tokenizer = tokenizer
        self.cache = {}
        self.tokens = {text: tokenizer.tokenize(text) for text in (NEGATIVE_PROMPT, prompt)}
        for row, text in enumerate((NEGATIVE_PROMPT, prompt)):
            for options in (dict(padding="max_length", max_length=77, truncation=True, return_tensors="pt"),
                            dict(padding="longest", return_tensors="pt")):
                encoded = tokenizer(text, **options)
                if options["padding"] == "max_length":
                    if not torch.equal(encoded.input_ids, token_ids[row:row + 1].cpu()):
                        raise ValueError("canonical tokenization differs from the supplied token IDs")
                    encoded.input_ids = token_ids[row:row + 1]
                else:
                    encoded.input_ids = encoded.input_ids.to("cuda")
                self.cache[self.key(text, options)] = encoded

    @staticmethod
    def key(text, options):
        return (tuple([text] if isinstance(text, str) else text), tuple(sorted(options.items())))

    def __call__(self, text, **options):
        return self.cache[self.key(text, options)]

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def tokenize(self, text, **kwargs):
        if kwargs:
            raise ValueError("prepared tokenizer does not accept tokenization overrides")
        return self.tokens[text]


def canonical_reference(pipe, prompt, inputs, *, capture, steps=DEFAULT_STEPS):
    """Run the upstream pipeline; hooks only observe tensors in untimed runs."""
    import torch

    observed, handles = {}, []
    encoder_outputs = [[], []]
    original_prepare, original_decode = pipe.prepare_latents, pipe.vae.decode
    original_postprocess = pipe.image_processor.postprocess

    def remember(name, value):
        observed[name] = value.detach().cpu().clone()

    def encoder_hook(index):
        def hook(module, args, output):
            encoder_outputs[index].append((output.hidden_states[-2].detach().cpu().clone(),
                                           output[0].detach().cpu().clone()))
        return hook

    def prepare_latents(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        remember("initialized_latent", result)
        return result

    def decode(*args, **kwargs):
        result = original_decode(*args, **kwargs)
        remember("decoded_image", result[0] if isinstance(result, tuple) else result.sample)
        return result

    def after_step(pipeline, index, timestep, values):
        remember(f"latent_after_step_{index}", values["latents"])
        return values

    # Compare the raw decoded tensor endpoint, excluding image normalization,
    # clipping, watermarking and conversion to a PIL image from both timings.
    pipe.image_processor.postprocess = lambda image, **kwargs: image
    if capture:
        pipe.prepare_latents, pipe.vae.decode = prepare_latents, decode
        handles = [model.register_forward_hook(encoder_hook(index))
                   for index, model in enumerate((pipe.text_encoder, pipe.text_encoder_2))]
    try:
        kwargs = dict(prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                      latents=inputs["initial_noise"], height=1024, width=1024,
                      num_inference_steps=steps, guidance_scale=5.0, output_type="pt")
        if capture:
            kwargs.update(callback_on_step_end=after_step,
                          callback_on_step_end_tensor_inputs=["latents"])
        result = pipe(**kwargs).images
        if capture:
            if any(len(values) != 2 for values in encoder_outputs):
                raise ValueError("SDXL must encode positive and negative prompts once in each text encoder")
            positive = torch.cat([values[0][0] for values in encoder_outputs], dim=-1)
            negative = torch.cat([values[1][0] for values in encoder_outputs], dim=-1)
            observed["text_embeddings"] = torch.cat((negative, positive))
            observed["pooled_embeddings"] = torch.cat((encoder_outputs[1][1][1], encoder_outputs[1][0][1]))
        return observed if capture else result
    finally:
        pipe.prepare_latents, pipe.vae.decode = original_prepare, original_decode
        pipe.image_processor.postprocess = original_postprocess
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
        original_tokenizers = pipe.tokenizer, pipe.tokenizer_2
        pipe.tokenizer = PreparedTokenizer(original_tokenizers[0], prompt, inputs["token_ids_1"])
        pipe.tokenizer_2 = PreparedTokenizer(original_tokenizers[1], prompt, inputs["token_ids_2"])
        pipe.vae.to(dtype=torch.float16)
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
        torch.cuda.synchronize()
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
        metadata = dict(model=MODEL_ID, revision=MODEL_REVISION, prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT, seed=seed, steps=steps, inputs=input_metadata, outputs=outputs,
                        timing=dict(wall_ms=wall_ms, gpu_ms=gpu_ms, warmups=3,
                                    boundary="GPU token IDs and noise to raw decoded VAE tensor",
                                    includes=f"both text encoders, {steps} Euler updates, VAE; no tokenization, observation hooks or image postprocessing",
                                    vae_precision="stock pipeline converts VAE weights to float32 and back per inference; native execution uses weights converted during preparation"))
        write_json(directory / "reference.json", metadata)
        cases[name] = host_inputs
        pipe.tokenizer, pipe.tokenizer_2 = original_tokenizers
        del first, second, inputs
    return cases


@dataclass(frozen=True)
class TensorSignature:
    kind: str = "tensor"


@dataclass(frozen=True)
class PreparationNamespace:
    revision: str
    source_identity: str
    steps: int = DEFAULT_STEPS

    def to_dict(self):
        return dict(recipe="sdxl-fixed-tensor", revision=self.revision, source_identity=self.source_identity,
                    height=1024, width=1024, steps=self.steps)


def recipe_source_identity():
    from ignition.preparation.source_identity import source_identity

    return source_identity(extra_roots=(("hyperflare_recipes", Path(__file__).resolve().parents[2]),))


def remove_output_capture_hooks(model):
    """Detach Transformers' observation hooks before serializing prepared modules."""
    removed = 0
    for module in model.modules():
        for identifier, hook in tuple(module._forward_hooks.items()):
            if (getattr(hook, "__module__", None) == "transformers.utils.output_capturing"
                    and getattr(hook, "__qualname__", "").startswith("install_output_capuring_hook.<locals>.")):
                del module._forward_hooks[identifier]
                module._forward_hooks_with_kwargs.pop(identifier, None)
                module._forward_hooks_always_called.pop(identifier, None)
                removed += 1
    return removed


class CLIPConditioning:
    """Expose CLIP's penultimate state without process-local observation hooks."""

    def __init__(self, model, token_ids, *, projection=None):
        from transformers.masking_utils import create_causal_mask

        self.embeddings = model.embeddings
        self.layers = model.encoder.layers
        self.final_layer_norm = model.final_layer_norm
        self.eos_token_id = model.eos_token_id
        self.projection = projection
        sample = self.embeddings(input_ids=token_ids)
        self.attention_mask = create_causal_mask(config=model.config, inputs_embeds=sample,
                                                 attention_mask=None, past_key_values=None)

    def __call__(self, token_ids):
        import torch

        hidden = self.embeddings(input_ids=token_ids)
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                penultimate = hidden
            hidden = layer(hidden, self.attention_mask, is_causal=True)
        hidden = self.final_layer_norm(hidden)
        ids = token_ids.to(dtype=torch.int, device=hidden.device)
        positions = ids.argmax(dim=-1) if self.eos_token_id == 2 else (ids == self.eos_token_id).int().argmax(dim=-1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        if self.projection is not None:
            pooled = self.projection(pooled)
        return penultimate, pooled


class TextForward:
    def __init__(self, pipe, buffers):
        remove_output_capture_hooks(pipe.text_encoder)
        remove_output_capture_hooks(pipe.text_encoder_2)
        self.ids_1, self.ids_2 = buffers["token_ids_1"], buffers["token_ids_2"]
        self.encoder_1 = CLIPConditioning(pipe.text_encoder, self.ids_1[0:1])
        self.encoder_2 = CLIPConditioning(pipe.text_encoder_2.text_model, self.ids_2[0:1],
                                           projection=pipe.text_encoder_2.text_projection)
        self.sequence, self.pooled = buffers["text_embeddings"], buffers["pooled_embeddings"]

    def __call__(self):
        import torch

        # The upstream pipeline encodes positive and negative prompts separately.
        positive_1 = self.encoder_1(self.ids_1[1:2])
        positive_2 = self.encoder_2(self.ids_2[1:2])
        negative_1 = self.encoder_1(self.ids_1[0:1])
        negative_2 = self.encoder_2(self.ids_2[0:1])
        positive = torch.cat((positive_1[0], positive_2[0]), dim=-1)
        negative = torch.cat((negative_1[0], negative_2[0]), dim=-1)
        self.sequence.copy_(torch.cat((negative, positive)))
        self.pooled.copy_(torch.cat((negative_2[1], positive_2[1])))


class InitialForward:
    def __init__(self, buffers, sigma):
        self.noise, self.output, self.sigma = buffers["initial_noise"], buffers["initialized_latent"], sigma

    def __call__(self):
        self.output.copy_(self.noise * self.sigma)


class EulerForward:
    def __init__(self, pipe, buffers, index, timestep, sigma, scale, delta):
        self.unet = pipe.unet
        self.input = buffers["initialized_latent" if index == 0 else f"latent_after_step_{index - 1}"]
        self.output = buffers[f"latent_after_step_{index}"]
        self.sequence, self.pooled = buffers["text_embeddings"], buffers["pooled_embeddings"]
        self.size_crop, self.guidance = buffers["size_crop"], buffers["guidance"]
        self.timestep, self.sigma, self.scale, self.delta = timestep, sigma, scale, delta

    def __call__(self):
        import torch

        sample = torch.cat((self.input, self.input))
        sample = sample / self.scale
        prediction = self.unet(sample, self.timestep, encoder_hidden_states=self.sequence,
                               added_cond_kwargs={"text_embeds": self.pooled, "time_ids": self.size_crop},
                               return_dict=False)[0]
        unconditional, conditional = prediction.chunk(2)
        prediction = unconditional + self.guidance[0] * (conditional - unconditional)
        sample = self.input.to(torch.float32)
        predicted_original = sample - self.sigma * prediction
        derivative = (sample - predicted_original) / self.sigma
        previous = sample + derivative * self.delta
        self.output.copy_(previous.to(prediction.dtype))


class DecodeForward:
    def __init__(self, pipe, buffers, steps=DEFAULT_STEPS):
        self.vae = pipe.vae
        self.input, self.output = buffers[f"latent_after_step_{steps - 1}"], buffers["decoded_image"]
        self.dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
        self.scaling_factor = pipe.vae.config.scaling_factor

    def __call__(self):
        latents = self.input.to(self.dtype) / self.scaling_factor
        self.output.copy_(self.vae.decode(latents, return_dict=False)[0])


def tensor_binding(name, tensor):
    if not tensor.is_contiguous() or tensor.numel() == 0:
        raise ValueError(f"prepared tensor binding {name} must be nonempty and contiguous")
    return dict(name=name, address=tensor.data_ptr(), nbytes=tensor.numel() * tensor.element_size(),
                dtype=DTYPES[str(tensor.dtype)], shape=list(tensor.shape))


def tensor_contract(order, buffers, inputs):
    return dict(version=1, graphs=list(order),
                inputs=[tensor_binding(name, buffers[name]) for name in inputs],
                outputs=[tensor_binding(name, value) for name, value in buffers.items() if name not in inputs])


def tensor_request(inputs):
    return dict(inputs=[dict(name=name, file=f"inputs/{name}.bin") for name in inputs])


def verify_eager_forwards(functions, buffers, cases, root, *, steps=DEFAULT_STEPS):
    """Check preparation composition against independently observed pipeline values."""
    import torch

    for case, inputs in cases.items():
        for name, value in inputs.items():
            buffers[name].copy_(value)
        for name in graph_order(steps):
            functions[name]()
        expected = json.loads((root / case / "reference.json").read_text())
        checks = []
        for item in expected["outputs"]:
            actual = buffers[item["name"]].detach().cpu()
            payload = bytearray((root / case / item["file"]).read_bytes())
            reference = torch.frombuffer(payload, dtype=actual.dtype).reshape(item["shape"])
            exact = torch.equal(actual, reference)
            checks.append(dict(name=item["name"], exact=exact,
                               maximum_absolute_error=float((actual.float() - reference.float()).abs().max())))
        write_json(root / case / "eager_composition.json", dict(checks=checks))
        if not all(item["exact"] for item in checks):
            raise ValueError(f"prepared eager composition differs from canonical Diffusers: {checks}")


def save_captured_outputs(graphs, buffers, cases, root, *, steps=DEFAULT_STEPS):
    """Record compiled replay results independently of native export and loading."""
    import torch

    for case, inputs in cases.items():
        for name, value in inputs.items():
            buffers[name].copy_(value)
        for name in graph_order(steps):
            graphs[name].replay()
        torch.cuda.synchronize()
        reference = json.loads((root / case / "reference.json").read_text())
        outputs = []
        for item in reference["outputs"]:
            actual = buffers[item["name"]].detach().cpu()
            recorded = save_tensor(root / case, Path("captured") / f"{item['name']}.bin", actual)
            payload = bytearray((root / case / item["file"]).read_bytes())
            expected = torch.frombuffer(payload, dtype=actual.dtype).reshape(item["shape"])
            outputs.append(dict(name=item["name"], **recorded, exact=torch.equal(actual, expected),
                                maximum_absolute_error=float((actual.float() - expected.float()).abs().max())))
        write_json(root / case / "captured.json", dict(outputs=outputs))
    for name, value in cases["case_a"].items():
        buffers[name].copy_(value)
    torch.cuda.synchronize()


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
    for name in ("initialized_latent", *(f"latent_after_step_{index}" for index in range(args.steps))):
        buffers[name] = torch.empty((1, 4, 128, 128), dtype=torch.float16, device="cuda")
    buffers["text_embeddings"] = torch.empty((2, 77, 2048), dtype=torch.float16, device="cuda")
    buffers["pooled_embeddings"] = torch.empty((2, 1280), dtype=torch.float16, device="cuda")
    buffers["decoded_image"] = torch.empty((1, 3, 1024, 1024), dtype=torch.float32, device="cuda")
    pipe.scheduler.set_timesteps(args.steps, device="cuda")
    sigmas = pipe.scheduler.sigmas.to("cuda")
    timesteps = pipe.scheduler.timesteps
    initial_sigma = float(pipe.scheduler.init_noise_sigma)
    pipe.upcast_vae()
    bind_attention_processors(pipe.unet)
    bind_attention_processors(pipe.vae)
    functions = dict(text=TextForward(pipe, buffers), init=InitialForward(buffers, initial_sigma))
    for index in range(args.steps):
        sigma = pipe.scheduler.sigmas[index]
        scale = float((sigma ** 2 + 1) ** 0.5)
        delta = float(pipe.scheduler.sigmas[index + 1] - sigma)
        functions[f"step_{index}"] = EulerForward(pipe, buffers, index, timesteps[index], float(sigma), scale, delta)
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
    write_json(args.output / "publisher.json", [dict(name="SDXL local qualification", namespace=publisher.namespace,
                public_key=list(publisher.public_key), development=True)])
    write_json(args.output / "publisher-info.json", dict(namespace=publisher.namespace,
                fingerprint=publisher.fingerprint.hex(), development=True))
    write_json(args.output / "preparation.json", dict(model=MODEL_ID, revision=MODEL_REVISION,
                graphs=list(order), export_counts=package.export_counts,
                core=str(args.output / "core"), publication=vars(result),
                timesteps=timesteps.cpu().tolist(), sigmas=sigmas.cpu().tolist(),
                scheduler_step_sha256=hashlib.sha256(inspect.getsource(type(pipe.scheduler).step).encode()).hexdigest(),
                profile=dict(height=1024, width=1024, batch=1, steps=args.steps, guidance=5.0)))
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=2)
    parser.add_argument("--steps", type=positive_steps, default=DEFAULT_STEPS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.module_images import ModuleImages

    with ModuleImages() as images:
        torch.cuda.set_device(0)
        storage = CoreStorage()
        with storage, torch.no_grad():
            pipe = StableDiffusionXLPipeline.from_pretrained(str(args.snapshot), torch_dtype=torch.float16,
                    variant="fp16", use_safetensors=True, local_files_only=True, add_watermarker=False)
            pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            if (not isinstance(pipe.scheduler, EulerDiscreteScheduler)
                    or pipe.scheduler.config.prediction_type != "epsilon"
                    or pipe.vae_scale_factor != 8 or not pipe.vae.config.force_upcast):
                raise ValueError("checkpoint does not match the fixed SDXL Euler profile")
            cases = save_references(pipe, args.output, steps=args.steps)
            prepare_graphs(pipe, storage, images, cases, args)


if __name__ == "__main__":
    # Compiler workers resolve prepared forward types by their importable module.
    from hyperflare_recipes.recipes.sdxl.prepare import main as prepare_main

    prepare_main()
