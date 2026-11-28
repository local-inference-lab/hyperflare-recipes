# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a miniature reference SM103 core and serve it in fresh flare run processes."""

from concurrent.futures import ThreadPoolExecutor
import base64
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import urllib.request

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest
import torch
from safetensors.torch import save_file

from hyperflare_recipes.components.expert_placement import ExpertPlacement
from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
from hyperflare_recipes.components.megamoe_tiers import MegaMoeHostTier, MegaMoeLayerWeights
from hyperflare_recipes.components.megamoe_weights import MegaMoeWeights, mxfp4_weight_layouts
from ignition.preparation.core_snapshot import export_options
from ignition.preparation.core_storage import CoreStorage
from ignition.preparation.model_core import export_core
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.persistent import persistent_copy, persistent_empty
from ignition.preparation.protection import Publisher
from ignition.preparation.runtime_core import FileSource
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import RecordedPreparationSession
from hyperflare_recipes.recipes.deepseek41.bootstrap import PreparationConfig, _assemble_reference_prepared_model
from hyperflare_recipes.recipes.deepseek41.reference_engram import ReferenceEngramDelivery
from hyperflare_recipes.recipes.deepseek41.reference_experts import ReferenceExperts
from hyperflare_recipes.recipes.deepseek41.reference_model import ReferenceExpertPreparation, ReferenceModel
from hyperflare_recipes.recipes.deepseek41.sm103 import SM103Config

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or os.environ.get("FLARE_RUN_REFERENCE_CORE_EXPORT_CUDA_TESTS") != "1",
    reason="requires an assigned SM103 GPU, DeepSeek V4.1 metadata, and release flare CLI",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _png_data_url(width: int = 84, height: int = 84) -> str:
    from PIL import Image

    image = Image.new("RGB", (width, height), color=(48, 112, 192))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _materialize_synthetic_ordinary(model, cfg, device):
    from deepinfer.hugging_face.parameters import model_params, set_ue8m0_sf
    from deepinfer.model_executor.models.v4_modules.containers import materialize_hf_parameter

    n_layer = cfg.gpt.num_hidden_layers
    tied_sources = {
        "emb.tok_emb.weight": "module.module.0.emb.tok_emb.weight",
        "shared_head.head.weight": f"module.module.{n_layer + 1}.shared_head.head.weight",
    }
    generator = torch.Generator(device=device).manual_seed(20261119)
    params = model_params(model)
    aliases, replacements, fp8_params = {}, {}, []
    for name, value in params.items():
        if getattr(value, "is_engram", False) or getattr(value, "mxfp4", False):
            continue
        if ".experts." in name and "shared_experts" not in name:
            continue
        parts = name.split(".", 3)
        if (
            len(parts) == 4
            and parts[:2] == ["module", "module"]
            and parts[2].isdigit()
            and int(parts[2]) > n_layer + 1
            and parts[3] in tied_sources
        ):
            aliases[name] = tied_sources[parts[3]]
            continue
        materialized = materialize_hf_parameter(value, device=device)
        if materialized.dtype == torch.float8_e4m3fn:
            src = (torch.randn(materialized.shape, device=device, dtype=torch.bfloat16, generator=generator) * 0.02)
            materialized.data.copy_(src.to(torch.float8_e4m3fn))
            if hasattr(materialized, "sf"):
                materialized.sf = torch.full(
                    materialized.sf.shape, 1.0 / 64.0, dtype=torch.float32, device=device
                ).contiguous()
                fp8_params.append(materialized)
        elif materialized.dtype.is_floating_point:
            if "norm" in name or name.endswith(".alpha"):
                materialized.data.fill_(1.0)
            else:
                materialized.data.normal_(0.0, 0.02, generator=generator)
        else:
            materialized.data.zero_()
        replacements[name] = materialized

    for name, source in aliases.items():
        original = params[name]
        owner = replacements[source]
        alias = torch.nn.Parameter(owner.data, requires_grad=False)
        for key, val in original.__dict__.items():
            if key != "sf":
                setattr(alias, key, val)
        if hasattr(owner, "sf"):
            alias.sf = owner.sf
        replacements[name] = alias

    for name, value in replacements.items():
        parent, leaf = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        if leaf in module._buffers:
            module._buffers[leaf] = value
        else:
            setattr(module, leaf, value)

    if fp8_params:
        set_ue8m0_sf(fp8_params)
    model.vision_pipeline.aligner.visual_encoder.materialize_rotary(device)
    for tensor in replacements.values():
        record_numeric_tensor(tensor)
        if hasattr(tensor, "sf") and isinstance(tensor.sf, torch.Tensor) and not tensor.sf.is_meta:
            record_numeric_tensor(tensor.sf)
        for extra in getattr(tensor, "ue8m0_sf", ()):
            record_numeric_tensor(extra)


def _allocate_synthetic_mxfp4(rows: int, hidden: int, intermediate: int, device):
    layouts = mxfp4_weight_layouts(rows, hidden, intermediate)
    tensors = []
    for shape, dtype, stride in layouts:
        tensor = persistent_empty(shape, dtype, device, stride=stride)
        if dtype == torch.int32:
            # Neutral UE8M0 exponent 120 across all 4 bytes of each int32 word.
            tensor.fill_(0x78787878)
        else:
            tensor.fill_(0x11)
        record_numeric_tensor(tensor)
        tensors.append(tensor)
    return MegaMoeWeights(tuple(tensors[:2]), tuple(tensors[2:]))


def _allocate_synthetic_shared_fp8(hidden: int, intermediate: int, device):
    def pair(rows: int, cols: int):
        values = persistent_empty((rows, cols), torch.float8_e4m3fn, device)
        values.zero_()
        sf_cols = cols // 128
        scales = persistent_empty((rows, sf_cols), torch.int32, device, stride=(1, rows))
        scales.fill_(0x78787878)
        record_numeric_tensor(values)
        record_numeric_tensor(scales)
        return values, scales

    return MegaMoeWeights(pair(2 * intermediate, hidden), pair(hidden, intermediate))


def _build_mini_reference_experts(placements, *, target_rows, draft_rows, storage, kernel_cache, session, device):
    hidden, intermediate = 5120, 2304
    generator = torch.Generator(device=device).manual_seed(7711)
    workspaces, modules, owners = {}, {}, {}
    for name, placement in placements.items():
        draft = name.startswith("mtp.")
        rows = draft_rows if draft else target_rows
        top_k = 3 if draft else 6
        geometry = MegaMoeGeometry(placement.physical_rows, max(rows), top_k, hidden, intermediate, 1)
        if geometry not in workspaces:
            workspaces[geometry] = MegaMoeWorkspace(geometry, device=device, kernel_cache=kernel_cache)
        ws = workspaces[geometry]
        routed = _allocate_synthetic_mxfp4(len(placement.hbm), hidden, intermediate, device)
        shared = _allocate_synthetic_shared_fp8(hidden, intermediate, device)
        tier = None
        if placement.host:
            tier = MegaMoeHostTier(ws.geometry, rows=len(placement.host), storage=storage, name=name,
                                   host_lane_clusters=4)
            for dst in tier.host_views:
                if dst.dtype == torch.int32:
                    dst.fill_(0x78787878)
                else:
                    dst.fill_(0x11)
        physical = placement.host + placement.hbm
        mapping = [0] * placement.experts
        for row, canonical in enumerate(physical):
            mapping[canonical] = row
        expert_map = persistent_copy(torch.tensor(mapping, dtype=torch.int32, device=device))
        record_numeric_tensor(expert_map)
        layer_weights = MegaMoeLayerWeights(routed, shared, ws, tier, expert_map, placement.host, placement.hbm)
        owners[name] = layer_weights
        gate_w = persistent_copy(
            torch.randn((placement.experts, hidden), dtype=torch.bfloat16, device=device, generator=generator) * 0.05
        )
        gate_b = persistent_copy(torch.linspace(-0.2, 0.2, placement.experts, dtype=torch.float32, device=device))
        gate_vl = persistent_copy(torch.linspace(0.2, -0.2, placement.experts, dtype=torch.float32, device=device))
        modules[name] = ReferenceExperts.prepare(
            layer_weights, rows, weight=gate_w, bias=gate_b, image_bias=gate_vl,
            scaling=1.5, scoring_func="sqrtsoftplus", clamp=10.0, session=session, name=name,
        )
    return ReferenceExpertPreparation(modules, owners)


@torch.no_grad()
def test_miniature_reference_core_export_and_fresh_process_flare_run(tmp_path):
    from deepinfer.hugging_face.config import model_config_from_dict
    from deepinfer.model_executor.execution_settings import ExecutionSettings
    from deepinfer.model_executor.models.v4_modules.builder import V41Builder
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    source_snapshot = Path(os.environ["FLARE_DEEPSEEK41_CSF_SNAPSHOT"]).resolve(strict=True)
    repo_root = Path(__file__).resolve().parents[2]
    flare_bin = repo_root / "flare-cli" / "target" / "release" / "flare"
    assert flare_bin.is_file(), f"missing release flare binary at {flare_bin}"

    if PG.world is None:
        PG.initialize_local()

    meta_dir = tmp_path / "mini_checkpoint"
    meta_dir.mkdir()
    for asset in source_snapshot.iterdir():
        if asset.name != "config.json" and not asset.name.endswith(".safetensors") and asset.name != "model.safetensors.index.json":
            (meta_dir / asset.name).symlink_to(asset)

    raw = json.loads((source_snapshot / "config.json").read_text())
    text = raw["text_config"]
    text.update(
        num_hidden_layers=5,
        compress_ratios=[0, 0, 2, 2, 1, 0],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4],
        candidate_source_layer_id=4,
        candidate_topk_blocks=16,
        index_topk=64,
        engram_layer_ids=[1],
        engram_vocab_size=64,
        engram_num_embeddings=[0],
        n_routed_experts=8,
        num_experts_per_tok=6,
        num_nextn_predict_layers=1,
        dspark_target_layer_ids=[1, 2, 4],
        dspark_n_routed_experts=8,
        dspark_num_experts_per_tok=3,
    )
    raw["vision_config"]["num_hidden_layers"] = 2
    (meta_dir / "config.json").write_text(json.dumps(raw, indent=2))

    decoded_raw = dict(raw, quantization_config=dict(
        quant_method="fp8", expert_dtype="fp4", activation_scheme="dynamic",
        weight_block_size=[32, 32], scale_fmt="ue8m0",
    ))
    parsed_cfg = model_config_from_dict(decoded_raw, hf_dir=meta_dir)
    hash_meta = ReferenceEngramDelivery.export_hash_metadata(parsed_cfg.gpt)
    layer1_hash = hash_meta["table_hashes"][1]
    engram_rows = layer1_hash["offsets"][-1] + layer1_hash["primes"][-1]
    engram_width = 256
    engram_file = meta_dir / "engram.safetensors"
    save_file(
        {
            "layers.1.engram.embed.weight": torch.zeros((engram_rows, engram_width), dtype=torch.uint8),
            "layers.1.engram.embed.scale": torch.full((engram_rows, engram_width // 32), 120, dtype=torch.uint8),
        },
        str(engram_file),
    )
    with engram_file.open("rb") as fh:
        header_len, = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        data_base = 8 + header_len
    w_off = data_base + header["layers.1.engram.embed.weight"]["data_offsets"][0]
    s_off = data_base + header["layers.1.engram.embed.scale"]["data_offsets"][0]

    # 8 canonical target experts: 2 pinned HBM (0, 1), 6 Grace-backed (2..7),
    # 2 movable HBM cache rows (2, 3) + 1 final host-backed HBM row (4) freed for exchange.
    target_placement = ExpertPlacement(8, (2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4))
    draft_placement = ExpertPlacement(8, (), tuple(range(8)))
    placements = tuple((f"layers.{i}.ffn", target_placement) for i in range(5)) + (("mtp.0.ffn", draft_placement),)
    sm103 = SM103Config(placements, host_lane_clusters=4, execution="reference", context=1024)
    cache_override = os.environ.get("FLARE_TEST_REFERENCE_MINI_CACHE_DIR")
    kernel_cache = Path(cache_override) if cache_override else tmp_path / "kernel_cache"
    kernel_cache.mkdir(parents=True, exist_ok=True)

    prep_cfg = PreparationConfig(
        devices=(0,),
        context=1024,
        prefill_chunk_size=1024,
        max_active=2,
        image_concurrency=1,
        snapshot_budget_bytes=64 << 20,
        headroom_bytes=512 << 20,
        kv_pages=64,
        compile_workers=12,
        autotune_effort="off",
        compile_decoder=True,
        retain_graphs=True,
        kernel_cache=kernel_cache,
        dspark=True,
        draft_head_quantization="nvfp4",
        markov_head_quantization="fp8",
        speculative_widths=(1, 2, 3, 4),
        speculative_token_capacity=8,
        speculative_graph_budget=16,
        sm103=sm103,
    )

    class _ParameterCarrier:
        def __init__(self, tree, cfg_obj, metadata_root):
            self.model = tree
            self.config = cfg_obj
            self.checkpoint = type("CP", (), {"source": type("Src", (), {"metadata_path": metadata_root})()})()
            self.loaded = True
            self.assembled = False

    core_dir = tmp_path / "exported_core"
    cache_dir = tmp_path / "core_cache"
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    pub = Publisher(
        Ed25519PrivateKey.from_private_bytes(bytes([42]) * 32),
        "local-development",
        "local-development",
        development=True,
    )

    recorder = ModuleImages()
    with export_options(cache_dir), recorder:
        torch.cuda.set_device(0)
        storage = CoreStorage(world=1)
        with storage:
            session = RecordedPreparationSession(
                compile_workers=12, autotune_effort="off", device=torch.device("cuda", 0),
                checkpoint="mini-reference-gate2", workload_schema="deepseek41-reference-mini",
                numerical_policy="reference-checkpoint", headroom_bytes=prep_cfg.headroom_bytes,
                benchmark_storage=storage,
            )
            session.__enter__()
            with storage.transient():
                with torch.device("meta"):
                    raw_tree = V41Builder(parsed_cfg).build_model(vision_device="meta")
                _materialize_synthetic_ordinary(raw_tree, parsed_cfg, torch.device("cuda", 0))
                modes = (False, True)
                target_rows = tuple(sorted(
                    set(prep_cfg.prefill_capacities)
                    | set(prep_cfg.batch_capacities)
                    | {b for b, _ in prep_cfg.speculative_shapes}
                    | {t for _, _, t in prep_cfg.packed_speculative_depth_shapes}
                ))
                draft_rows = tuple(sorted({b * (w - 1) for b, w, _ in prep_cfg.packed_speculative_depth_shapes}))
                experts = _build_mini_reference_experts(
                    dict(placements), target_rows=target_rows, draft_rows=draft_rows,
                    storage=storage, kernel_cache=kernel_cache, session=session,
                    device=torch.device("cuda", 0),
                )
                sequence = torch.zeros(1, dtype=torch.int64, device="cuda:0")
                frame = torch.zeros(1, dtype=torch.int32, device="cuda:0")
                delivery = ReferenceEngramDelivery(
                    {1: (FileSource(engram_file, w_off, engram_rows * engram_width),
                         FileSource(engram_file, s_off, engram_rows * (engram_width // 32)))},
                    vocabulary=parsed_cfg.gpt.vocab_size,
                    heads=24,
                    width=engram_width,
                    capacity=max(prep_cfg.prefill_chunk_size, prep_cfg.decode_tokens),
                    sequence=sequence,
                    frame=frame,
                    host_allocator=storage.host_allocate,
                    **hash_meta,
                )
                execution = ExecutionSettings(
                    draft_tokens=3, batch_invariant=False, yoco_prefill_truncation="off",
                    index_memory_limit=16 << 20, sampler_topk=0, confidence_temperature=1.0,
                )
                carrier = _ParameterCarrier(raw_tree, parsed_cfg, meta_dir)
                ref_model = ReferenceModel(
                    carrier, experts, delivery, execution=execution, context=prep_cfg.context,
                    request_capacity=prep_cfg.request_capacity, prefill_tokens=prep_cfg.prefill_chunk_size,
                    verification_tokens=prep_cfg.request_capacity * 4,
                )
                del raw_tree, carrier

            prepared = _assemble_reference_prepared_model(
                ref_model, config=prep_cfg, metadata_path=meta_dir,
                snapshot_name="mini-deepseek41-reference", identity="0" * 64,
                session=session, core_storage=storage, sequence=sequence, frame=frame,
                weights_bytes=1 << 20, tier_bytes={"hbm": 1 << 20, "host": 1 << 20},
                has_vision=True,
            )

        with prepared:
            from ignition.preparation.runtime_core_pb2 import RebuildManifest

            rebuild = dict(
                version=1, recipe="deepseek41", source_identity="0" * 64,
                checkpoint_repository="deepseek-ai/DeepSeek-V4.1-Flash-1017",
                checkpoint_revision="", local_checkpoint=True, tensor_parallel=1,
                max_concurrency=2, prefill_chunk=128, image_concurrency=1,
                context_parallel=False, context_parallel_size=0, stream_overlap=True,
                mixed_prefill=False, mla_compute_mode="bf16", roce_ring=False, roce_chain=False,
                online_mxfp8=False, online_mxfp8_scope="all", language_model_only=False,
                compiled_forwards=True, reserve_bytes=0, collective_tuning="static",
                speculation=dict(
                    draft_tokens=3, mtp_only=False, adaptive=True, token_capacity=8,
                    packed_verification=True, draft_prefix_selection="gpu",
                    vocabulary_parallel_sampling=False, graph_budget=16,
                ),
            )
            if "sm103_preparation_json" in RebuildManifest.DESCRIPTOR.fields_by_name:
                rebuild["sm103_preparation_json"] = json.dumps(
                    sm103.to_dict(), separators=(",", ":"), sort_keys=True
                )
            options = dict(
                model_id="mini-deepseek41-reference",
                served_names=["DeepSeek-V4.1-Flash"],
                reserve_bytes=0,
                queue_capacity=8,
                frontend_workers=2,
                max_body_bytes=32 << 20,
                context_hint="miniature test core",
                capacity_hint="--max-concurrency sets the request limit",
                rebuild_manifest=rebuild,
            )
            export_core(prepared, recorder, storage, core_dir, options=options,
                        publisher=pub, staging=staging_dir)

    # Release preparation CUDA allocations so fresh flare run subprocesses own the GPU cleanly.
    import gc

    del prepared, ref_model, experts, delivery, sequence, frame, session
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    _verify_exported_mini_core(core_dir, cache_dir, pub, flare_bin, tmp_path)


def _verify_exported_mini_core(core_dir: Path, cache_dir: Path, pub: Publisher, flare_bin: Path, tmp_path: Path):
    # Verify primary shard contains metadata/frontend headers only (no weight or rank execution payloads).
    primary = core_dir / "model.safetensors"
    assert primary.is_file()
    with primary.open("rb") as fh:
        hlen, = struct.unpack("<Q", fh.read(8))
        primary_header = json.loads(fh.read(hlen))
    tensor_keys = [k for k in primary_header if k != "__metadata__"]
    assert "flare.manifest" in tensor_keys, tensor_keys
    assert all(k.startswith(("flare.", "frontend.", "align.")) for k in tensor_keys), tensor_keys

    pub_keys_path = tmp_path / "publisher-keys.json"
    pub_keys_path.write_text(json.dumps([{
        "public_key": list(pub.public_key),
        "name": "Local development signer",
        "namespace": "local-development",
        "development": True,
    }]))
    trust_path = tmp_path / "trust.toml"
    trust_path.write_text(
        f'[[trusted]]\nfingerprint = "{pub.fingerprint.hex()}"\n'
        f'namespace = "local-development"\npublic_key = "{pub.public_key.hex()}"\n'
        f'name = "Local development signer"\nidentity = "profile"\n'
    )
    run_cfg_path = tmp_path / "flare.toml"
    run_cfg_path.write_text(f'[paths]\ncore_cache = "{cache_dir}"\n')

    inspected = json.loads(subprocess.check_output([
        str(flare_bin), "inspect", str(core_dir), "--json", "--graphs",
        "--publisher-keys", str(pub_keys_path), "--trust-store", str(trust_path),
    ], text=True))
    assert inspected["core"]["signatures_verified"] is True
    core_id = inspected["core"]["core_id"]
    graph_names = {g["name"] for g in inspected["ranks"][0]["execution_graphs"]}
    assert "prefill128" in graph_names
    assert "decode2" in graph_names
    assert "vision576" in graph_names

    def run_flare_server(extra_args):
        port = _free_port()
        mem_args = () if "--memory-limit" in extra_args else ("--memory-limit", "10G")
        cmd = [
            str(flare_bin), "run", str(core_dir),
            "--config", str(run_cfg_path),
            "--publisher-keys", str(pub_keys_path),
            "--trust-store", str(trust_path),
            "--gpus", "0",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--embedding-table-storage", "ram",
            "--verify-weights",
            *mem_args,
            *extra_args,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, _ = proc.communicate()
                raise RuntimeError(f"flare run exited early ({proc.returncode}):\n{out}")
            try:
                with urllib.request.urlopen(url + "/v1/models", timeout=2) as resp:
                    if resp.status == 200:
                        return proc, url
            except Exception:
                time.sleep(0.2)
        proc.kill()
        out, _ = proc.communicate()
        raise TimeoutError(f"flare run did not become ready:\n{out}")

    def stop_flare_server(proc):
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
        assert proc.returncode == 0, f"clean shutdown failed ({proc.returncode}):\n{out}"

    def chat(url, content, *, max_tokens=12, seed=42):
        body = dict(
            model="DeepSeek-V4.1-Flash",
            messages=[dict(role="user", content=content)],
            max_tokens=max_tokens,
            seed=seed,
            temperature=0.0,
            enable_thinking=False,
            return_token_ids=True,
        )
        req = urllib.request.Request(
            url + "/v1/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {err.code}: {detail}") from err

    def get_metrics(url):
        with urllib.request.urlopen(url + "/metrics", timeout=10) as resp:
            text_out = resp.read().decode()
        return {parts[0]: float(parts[1]) for line in text_out.splitlines()
                if not line.startswith("#") and len(parts := line.split()) == 2}

    # --- Subprocess Run A: Ordinary (--mtp-draft-tokens 0), text + image + prefix reuse + compacted prefill ---
    proc, url = run_flare_server(["--mtp-draft-tokens", "0"])
    try:
        long_prefix = " ".join(f"TokenSequenceBlock{i}" for i in range(40))
        r1 = chat(url, long_prefix + " First question?", max_tokens=8, seed=11)
        r2 = chat(url, long_prefix + " First question?", max_tokens=8, seed=11)
        assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0

        compacted_prompt = " ".join(f"record {j} value {(j * 17) % 97}" for j in range(80))
        res_compacted = chat(url, compacted_prompt, max_tokens=8, seed=499)
        assert res_compacted["usage"]["prompt_tokens"] > 144
        assert len(res_compacted["choices"][0]["token_ids"]) == 8

        img_res = chat(url, [
            {"type": "image_url", "image_url": {"url": _png_data_url(84, 84)}},
            {"type": "text", "text": "Describe this image."},
        ], max_tokens=6, seed=19)
        assert len(img_res["choices"][0]["token_ids"]) == 6
        m = get_metrics(url)
        assert m.get("flare_requests_failed_total", 0.0) == 0.0
    finally:
        stop_flare_server(proc)

    # --- Subprocess Run B: Adaptive speculative (--mtp-draft-tokens 3), concurrent C2 ---
    proc, url = run_flare_server(["--mtp-draft-tokens", "3", "--mtp-adaptive-verification"])
    try:
        with ThreadPoolExecutor(max_workers=2) as pool_exec:
            futures = [pool_exec.submit(chat, url, f"Concurrent request {i}", max_tokens=16, seed=100 + i)
                       for i in range(2)]
            results = [f.result() for f in futures]
        assert all(len(r["choices"][0]["token_ids"]) == 16 for r in results)
        m = get_metrics(url)
        assert m.get("flare_requests_failed_total", 0.0) == 0.0
        offered = sum(v for k, v in m.items() if k.startswith("flare_speculative_offered_drafts_total"))
        assert offered > 0, m
    finally:
        stop_flare_server(proc)
