# Diffusion preparation and comparison

The [SDXL](../hyperflare_recipes/recipes/sdxl/prepare.py) and
[PixArt-Alpha](../hyperflare_recipes/recipes/pixart/prepare.py) module commands
prepare fixed image-generation profiles for the native tensor runner. Each core
contains text encoding, denoising, scheduler updates and VAE decoding. Its signed
tensor contract owns graph order and input/output geometry; request files supply
only named input files. The commands generate canonical Diffusers references and
retain each intermediate output for exact comparison.

These commands use the tensor-core support in
[Flare PR #314](https://github.com/NVIDIA/hyperflare-inference/pull/314).
The lockfile selects Ignition and Heliopause from core revision
[`80fb66a2`](https://github.com/schopde-nvidia/hyperflare-inference/commit/80fb66a21709932a4de313f84b324a2c0b9b1ad7).
Build the Flare binary from that revision. The model computation and native admission contract are
specified in [Flare's diffusion design](https://github.com/schopde-nvidia/hyperflare-inference/blob/80fb66a21709932a4de313f84b324a2c0b9b1ad7/docs/diffusion-design.md).

## Qualified profiles

The implementation at core revision
[`7ddce700`](https://github.com/NVIDIA/hyperflare-inference/commit/7ddce700be0b82a001da94f87e2137cd496b2123)
passed exact canonical/captured/native comparisons for SDXL at 1024×1024 with
four and thirty Euler steps and PixArt-Alpha at 512×512 with twenty DPM-Solver++
steps, using the two prompts and seeds 42 and 43 declared by these recipes.
The GPU was an RTX6000D (SM120), with Torch 2.14.1+cu132, Diffusers 0.43.0 and
Transformers revision `1d9654e78a23fb167ef539cbd699dfbd3e4830cf`.
That qualification does not cover the package namespace and Transformers 5.23.0
environment used here. No serving-throughput or latency improvement is claimed.

## Prepare and compare fixed profiles

Use the [core preparation environment](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/model-preparation.md#environment-setup),
including the CUDA toolkit and CUPTI. Run these commands from the recipes checkout
with one visible SM120 GPU. Preparation was exercised with 96 GiB of host memory
and two compiler workers. Keep checkpoint downloads, compilation caches, and exported cores
outside the checkout, with enough disk space for several copies of the weights.

Install the diffusion dependencies and matching core packages. Set `flare_source`
to a checkout of core revision `80fb66a21709932a4de313f84b324a2c0b9b1ad7`.

```sh
flare_source=/absolute/path/to/hyperflare-inference
uv sync --locked --extra diffusion --extra test
uv pip check --python .venv/bin/python
cargo build --release --manifest-path "$flare_source/flare-cli/Cargo.toml"
diffusion_work=$(mktemp -d)

sdxl_snapshot=$(.venv/bin/hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --revision 462165984030d82259a11f4367a4eed129e94a7b \
  --include '*.json' '*.txt' '*.fp16.safetensors')
pixart_snapshot=$(.venv/bin/hf download PixArt-alpha/PixArt-XL-2-512x512 \
  --revision 50f702106901db6d0f8b67eb88e814c56ded2692 \
  --include '*.json' '*.txt' '*.model' '*.safetensors')

.venv/bin/python -m hyperflare_recipes.recipes.sdxl.prepare --snapshot "$sdxl_snapshot" \
  --output "$diffusion_work/sdxl-30" --steps 30 --compile-workers 2
.venv/bin/python -m hyperflare_recipes.recipes.pixart.prepare --snapshot "$pixart_snapshot" \
  --output "$diffusion_work/pixart-20" --steps 20 --compile-workers 2
```

Each command runs the two canonical references, enforces exact eager composition,
captures compiled outputs, and publishes a core with both request files. For
the four-step SDXL profile, use `--steps 4` and a separate output directory.
The recipes generate a development publisher; the following native commands
accept the fingerprint recorded by that local preparation. Existing output
directories are rejected, so use fresh native destinations when repeating runs.

```sh
: > "$diffusion_work/flare.toml"
comparison_status=0
for prepared in "$diffusion_work/sdxl-30" "$diffusion_work/pixart-20"; do
  fingerprint=$(.venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' \
    "$prepared/publisher-info.json")
  for case in case_a case_b; do
    "$flare_source/flare-cli/target/release/flare" --config "$diffusion_work/flare.toml" \
      run "$prepared/core" --gpus 0 --offline --verify-weights \
      --publisher-keys "$prepared/publisher.json" \
      --trust-store "$diffusion_work/trust.json" \
      --dangerously-approve-key "$fingerprint" \
      --dangerously-accept-publisher local-development \
      --tensor-request "$prepared/$case/request.json" \
      --tensor-output "$prepared/$case/native" \
      --tensor-warmup 3 --tensor-repetitions 5 || exit $?
    .venv/bin/python -m qualify.sdxl_compare \
      --reference "$prepared/$case/reference.json" \
      --captured "$prepared/$case/captured.json" \
      --native "$prepared/$case/native/report.json" \
      --output "$prepared/$case/comparison.json" --images || comparison_status=1
  done
done
test "$comparison_status" -eq 0
```

The comparison writes per-tensor errors, repeatability, raw timing samples,
and `pytorch.png` and `flare.png`. Native/captured acceptance requires identical
bytes as well as finite values, including identical signed zeros. Canonical
acceptance uses the explicitly recorded numerical bounds, which default to
zero. A failed canonical comparison returns nonzero and withholds timing
ratios even when native replay is byte-identical to the captured graphs.
