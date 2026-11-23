# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contract checks for SM103 decode MLA, BF16 router, mHC overlap, and low-latency FP8 projections."""

import fnmatch
import importlib
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components import (
    mega_attention,
    mega_attention_kernel,
    sm103_mhc_overlap,
    sm103_projection,
    sm103_router,
)


def _resolve_compile_target(target: str):
    module_name, _, attr_name = target.partition(":")
    assert module_name.startswith("hyperflare_recipes.components."), target
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name)
    assert callable(fn)
    return fn


def test_mega_attention_workspace_and_kernel_argument_validation():
    for bad_cap in (0, -1, 77, True, 8.0):
        with pytest.raises(ValueError, match="capacity from 1 through 76"):
            mega_attention.MegaAttentionWorkspace(bad_cap, device="cpu")

    for bad_sms in (-1, True, 144.0):
        with pytest.raises(ValueError, match="non-negative integer"):
            mega_attention.MegaAttentionWorkspace(8, device="cpu", compute_sm_count=bad_sms)
        with pytest.raises(ValueError, match="non-negative integer"):
            mega_attention_kernel.prepare(Path("/tmp/unused"), num_sms=bad_sms)

    q = mega_attention.MegaAttentionQuery(
        rows=8,
        mode=-1,
        nsub=0,
        scale_log2=0.0625,
        token_interleaved=True,
        max_ctas=144,
    )
    encoded = mega_attention.CONTRACT.encode_query(q)
    assert mega_attention.MegaAttentionQuery(**dict(encoded)) == q
    assert mega_attention.CONTRACT.semantic_version == 2


def test_mega_attention_operand_validation_and_outputs():
    def make_operands(rows: int, swa_slots: int = 128, comp_slots: int | None = 512,
                      *, token_interleaved: bool = True, swa_page_stride: int | None = None):
        q = torch.empty((rows, 64, 512), dtype=torch.bfloat16)
        pos = torch.zeros((rows,), dtype=torch.int32)
        cos_sin = torch.zeros((16, 64), dtype=torch.float32)
        sink = torch.zeros((64,), dtype=torch.float32)
        if swa_page_stride is None:
            window = torch.empty((2, 128, 528), dtype=torch.uint8)
        else:
            storage = torch.empty((2 * swa_page_stride,), dtype=torch.uint8)
            window = storage.as_strided((2, 128, 528), (swa_page_stride, 528, 1))
        indices = torch.zeros((rows, swa_slots), dtype=torch.int32)
        lengths = torch.full((rows,), swa_slots, dtype=torch.int32)
        if comp_slots is None:
            comp = ci = cl = None
        else:
            comp = torch.empty((4, 64, 288), dtype=torch.uint8)
            ci = torch.zeros((rows, comp_slots), dtype=torch.int32)
            cl = torch.full((rows,), comp_slots, dtype=torch.int32)
        return q, pos, cos_sin, sink, window, indices, lengths, comp, ci, cl

    ws = SimpleNamespace(device=torch.device("cpu"), capacity=76)

    # Window-only mode (-2) supports up to 24 rows.
    _, mode = mega_attention._validate_operands(*make_operands(8, comp_slots=None), ws, token_interleaved=True)
    assert mode == -2
    with pytest.raises(ValueError, match="at most 24 rows"):
        mega_attention._validate_operands(*make_operands(25, comp_slots=None), ws, token_interleaved=True)

    # Compressed mode selects -1 (<= 16 rows) and -3 (> 16 rows, window <= 128 slots).
    _, mode = mega_attention._validate_operands(*make_operands(16, swa_slots=128, comp_slots=512), ws, token_interleaved=True)
    assert mode == -1
    _, mode = mega_attention._validate_operands(*make_operands(32, swa_slots=128, comp_slots=512), ws, token_interleaved=True)
    assert mode == -3
    with pytest.raises(ValueError, match="window of at most 128 slots"):
        mega_attention._validate_operands(*make_operands(32, swa_slots=192, comp_slots=384), ws, token_interleaved=True)

    # Total slots capped at 640 (ten 64-key blocks).
    with pytest.raises(ValueError, match="at most ten 64-key blocks"):
        mega_attention._validate_operands(*make_operands(8, swa_slots=129, comp_slots=512), ws, token_interleaved=True)

    # Non-interleaved cache requires 512-byte page stride; token-interleaved relaxes to 16-byte.
    # 128 * 528 = 67584 is divisible by 512; adding 16 bytes makes it 16-byte aligned only.
    ops_16 = make_operands(8, comp_slots=None, swa_page_stride=128 * 528 + 16)
    _, mode = mega_attention._validate_operands(*ops_16, ws, token_interleaved=True)
    assert mode == -2
    with pytest.raises(ValueError, match="512-byte aligned page stride"):
        mega_attention._validate_operands(*ops_16, ws, token_interleaved=False)

    # Partial compressed arguments are rejected.
    q, pos, cs, sink, win, idx, lens, comp, ci, _ = make_operands(8, comp_slots=64)
    with pytest.raises(ValueError, match="supplied together"):
        mega_attention._validate_operands(q, pos, cs, sink, win, idx, lens, comp, ci, None, ws)

    # Output tensor layout and scale stride permutation.
    out, scales = mega_attention._outputs(5, torch.device("cpu"))
    assert out.shape == (5, 8, 4096) and out.dtype == torch.float8_e4m3fn
    assert scales.shape == (5, 8, 32) and scales.dtype == torch.int32
    assert scales.stride() == (1, 32 * 8, 8)


def test_sm103_router_geometry_query_and_workspace_contracts():
    assert sm103_router.supports_sm103_router(384, 5120, 6)
    assert sm103_router.supports_sm103_router(128, 5120, 3)
    for bad in ((256, 5120, 6), (384, 4096, 6), (384, 5120, 8), (128, 5120, 6)):
        assert not sm103_router.supports_sm103_router(*bad)

    q = sm103_router.SM103RouterQuery(rows=8, experts=384, hidden=5120, top_k=6)
    assert sm103_router.SM103RouterQuery(**dict(sm103_router.CONTRACT.encode_query(q))) == q

    for kwargs in (
        {"rows": 0, "experts": 384, "hidden": 5120, "top_k": 6},
        {"rows": 65, "experts": 384, "hidden": 5120, "top_k": 6},
        {"rows": True, "experts": 384, "hidden": 5120, "top_k": 6},
        {"rows": 8, "experts": 256, "hidden": 5120, "top_k": 6},
    ):
        with pytest.raises(ValueError, match="unsupported SM103 BF16 router"):
            sm103_router.SM103RouterQuery(**kwargs)

    for bad_rows in (0, -1, 65, True):
        with pytest.raises(ValueError, match="1..64 rows"):
            sm103_router.SM103RouterWorkspace(max_rows=bad_rows, device="cpu")
    for kwargs in (
        {"max_experts": 0},
        {"max_experts": True},
        {"splits": 0},
        {"splits": -1},
    ):
        with pytest.raises(ValueError, match="positive expert and split counts"):
            sm103_router.SM103RouterWorkspace(device="cpu", **kwargs)


def test_sm103_mhc_overlap_queries_compile_targets_and_binding_guards():
    seam_q = sm103_mhc_overlap.MHCSeamQuery(rows=8, hidden_size=5120, mode=1, shared_block_m=16)
    assert sm103_mhc_overlap.MHCSeamQuery(**dict(sm103_mhc_overlap.MHC_SEAM_CONTRACT.encode_query(seam_q))) == seam_q
    for kwargs in (
        {"rows": 0, "hidden_size": 5120, "mode": 0, "shared_block_m": 0},
        {"rows": 65, "hidden_size": 5120, "mode": 0, "shared_block_m": 0},
        {"rows": True, "hidden_size": 5120, "mode": 0, "shared_block_m": 0},
        {"rows": 8, "hidden_size": 5100, "mode": 0, "shared_block_m": 0},
        {"rows": 8, "hidden_size": 5120, "mode": 2, "shared_block_m": 0},
        {"rows": 8, "hidden_size": 5120, "mode": 1, "shared_block_m": -1},
    ):
        with pytest.raises(ValueError, match="SM103 mHC seam"):
            sm103_mhc_overlap.MHCSeamQuery(**kwargs)

    norm_big = sm103_mhc_overlap.RMSNormF8Query(rows=8, hidden_size=5120, return_bf16=True)
    norm_small = sm103_mhc_overlap.RMSNormF8Query(rows=8, hidden_size=2048, return_bf16=False)
    assert sm103_mhc_overlap.RMSNormF8Query(**dict(sm103_mhc_overlap.RMSNORM_F8_CONTRACT.encode_query(norm_big))) == norm_big
    assert sm103_mhc_overlap.RMSNORM_F8_CONTRACT.parameters(norm_big, None)["tactic"] == (16,)
    assert sm103_mhc_overlap.RMSNORM_F8_CONTRACT.parameters(norm_small, None)["tactic"] == (4,)
    for kwargs in (
        {"rows": 0, "hidden_size": 5120},
        {"rows": 65, "hidden_size": 5120},
        {"rows": 8, "hidden_size": 192},
        {"rows": 8, "hidden_size": 5120, "return_bf16": 1},
    ):
        with pytest.raises(ValueError, match="SM103 RMSNorm F8_128x4"):
            sm103_mhc_overlap.RMSNormF8Query(**kwargs)

    detected = SimpleNamespace(ordinal=0)
    seam_plan = sm103_mhc_overlap.mhc_seam_plan(seam_q, torch.device("cpu"))
    seam_jobs = seam_plan._compile_jobs(SimpleNamespace(tactic=16), detected)
    assert len(seam_jobs) == 1
    assert _resolve_compile_target(seam_jobs[0].factory) is sm103_mhc_overlap.compile_mhc_seam

    norm_plan = sm103_mhc_overlap.rmsnorm_f8_plan(norm_big, torch.device("cpu"))
    norm_jobs = norm_plan._compile_jobs(SimpleNamespace(tactic=16), detected)
    assert len(norm_jobs) == 1
    assert _resolve_compile_target(norm_jobs[0].factory) is sm103_mhc_overlap.compile_rmsnorm_f8_128x4

    # No-op joins when no auxiliary launch is pending.
    anchor = torch.empty((1,), dtype=torch.bfloat16)
    sm103_mhc_overlap.join_pending_mhc(anchor, SimpleNamespace())
    sm103_mhc_overlap.join_pending_swa_kv(anchor, SimpleNamespace())

    # Binding guards skip ineligible layers/attentions without touching attributes.
    fake_stream = SimpleNamespace(cuda_stream=1234)
    ineligible_layer = SimpleNamespace(hc_enabled=True, hc_shifted=False, use_mega_mhc=True, hc_mult=4)
    sm103_mhc_overlap.bind_mhc_overlap_layer(
        ineligible_layer, (1, 8), aux_stream=fake_stream, session=None, name="noop",
        allow_attn_overlap=True, allow_ffn_overlap=True,
    )
    assert not hasattr(ineligible_layer, "attn_norm_forward")

    ineligible_attn = SimpleNamespace(
        has_local_index=True,
        has_packed_kv=True,
        current_hidden_proj=SimpleNamespace(weight=torch.empty((32, 128), dtype=torch.float8_e4m3fn)),
        shared_hidden_kv_proj=None,
        index_attn=SimpleNamespace(store_index_k=False),
        current_hidden_proj_layout=(("index_weights", 32),),
    )
    sm103_mhc_overlap.bind_indexer_kv_overlap(ineligible_attn, (1, 8), aux_kv_stream=fake_stream)
    assert not hasattr(ineligible_attn, "forward")


def test_sm103_low_latency_fp8_query_swizzle_and_compile_targets():
    q = sm103_projection.LowLatencyFP8Query(
        rows=8, in_features=5120, out_features=2048, pre_swizzled_input=True, bf16_input=True,
    )
    assert sm103_projection.LowLatencyFP8Query(**dict(sm103_projection.LOW_LATENCY_FP8_CONTRACT.encode_query(q))) == q

    for kwargs in (
        {"rows": 0, "in_features": 128, "out_features": 128},
        {"rows": 65, "in_features": 128, "out_features": 128},
        {"rows": True, "in_features": 128, "out_features": 128},
        {"rows": 8, "in_features": 192, "out_features": 128},
        {"rows": 8, "in_features": 128, "out_features": 192},
        {"rows": 8, "in_features": 128, "out_features": 128, "pre_swizzled_input": 1},
        {"rows": 8, "in_features": 128, "out_features": 128, "bf16_input": 0},
    ):
        with pytest.raises(ValueError, match="SM103 low-latency FP8 projection"):
            sm103_projection.LowLatencyFP8Query(**kwargs)

    # Verify deepgemm_weight_sf_to_f8_128x4 exact byte permutation.
    out_features, in_features = 256, 256
    k_blk = in_features // 32  # 8 groups of 32, packed as 2 int32s per row
    raw_u8 = torch.arange(out_features * k_blk, dtype=torch.int64).remainder(251).to(torch.uint8)
    w_sf_int32 = raw_u8.view(torch.int32).reshape(out_features, k_blk // 4).contiguous()
    swizzled = sm103_projection.deepgemm_weight_sf_to_f8_128x4(w_sf_int32, out_features, in_features)
    assert swizzled.dtype == torch.uint8 and swizzled.shape == (out_features * k_blk,)
    raw_2d = raw_u8.reshape(out_features, k_blk)
    swizzled_5d = swizzled.view(out_features // 128, k_blk // 4, 32, 4, 4)
    for n_tile in range(out_features // 128):
        for k_tile in range(k_blk // 4):
            for r32 in range(32):
                for q4 in range(4):
                    for g4 in range(4):
                        row = n_tile * 128 + q4 * 32 + r32
                        grp = k_tile * 4 + g4
                        assert swizzled_5d[n_tile, k_tile, r32, q4, g4].item() == raw_2d[row, grp].item()

    with pytest.raises(ValueError, match="128-aligned"):
        sm103_projection.deepgemm_weight_sf_to_f8_128x4(w_sf_int32.to(torch.uint8), 256, 256)
    with pytest.raises(ValueError, match="128-aligned"):
        sm103_projection.deepgemm_weight_sf_to_f8_128x4(w_sf_int32, 192, 256)

    # Verify CompileJob targets resolve cleanly into hyperflare_recipes.components.sm103_projection.
    detected = SimpleNamespace(ordinal=0)
    packed_q = sm103_projection.PackedProjectionQuery(
        recipe="mxfp8", rows=8, in_features=128, padded_in_features=128,
        out_features=128, global_scale_kind="none", workspace_form="provided",
    )
    packed_plan = sm103_projection.plan(packed_q)
    packed_jobs = packed_plan._compile_jobs(None, detected)
    assert len(packed_jobs) == 1
    assert _resolve_compile_target(packed_jobs[0].factory) is sm103_projection.compile_programs

    ll_plan = sm103_projection.low_latency_fp8_plan(q, torch.device("cpu"))
    assert ll_plan._compile_jobs(SimpleNamespace(tactic=("deep_gemm", None)), detected) == ()
    ll_jobs = ll_plan._compile_jobs(SimpleNamespace(tactic=("cutedsl_low_latency", (128, 6, 1, 2))), detected)
    assert len(ll_jobs) == 2
    assert _resolve_compile_target(ll_jobs[0].factory) is sm103_projection.compile_low_latency_fp8_swizzle
    assert _resolve_compile_target(ll_jobs[1].factory) is sm103_projection.compile_low_latency_fp8_bf16_quant

    q_no_bf16 = sm103_projection.LowLatencyFP8Query(rows=8, in_features=5120, out_features=2048, bf16_input=False)
    ll_plan_no_bf16 = sm103_projection.low_latency_fp8_plan(q_no_bf16, torch.device("cpu"))
    ll_jobs_no_bf16 = ll_plan_no_bf16._compile_jobs(SimpleNamespace(tactic=("cute-dsl", -1)), detected)
    assert len(ll_jobs_no_bf16) == 1
    assert _resolve_compile_target(ll_jobs_no_bf16[0].factory) is sm103_projection.compile_low_latency_fp8_swizzle


def test_pyproject_packages_decode_cuda_sources_and_notices():
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    license_files = set(data["project"]["license-files"])
    package_data = data["tool"]["setuptools"]["package-data"]

    for rel in (
        "hyperflare_recipes/components/mega_attention_kernel/LICENSE.ds41",
        "hyperflare_recipes/components/mega_attention_kernel/NOTICE",
        "hyperflare_recipes/components/mega_attention_kernel/NOTICE.upstream",
    ):
        assert rel in license_files
        assert (repo_root / rel).is_file()

    for pkg, rel_dir in (
        ("hyperflare_recipes.components.mega_attention_kernel", "hyperflare_recipes/components/mega_attention_kernel"),
        ("hyperflare_recipes.components.sm103_router_kernel", "hyperflare_recipes/components/sm103_router_kernel"),
    ):
        patterns = package_data[pkg]
        root = repo_root / rel_dir
        expected_files = [
            str(p.relative_to(root))
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".cu", ".cuh") or (p.is_file() and p.name.startswith(("LICENSE", "NOTICE")))
        ]
        assert expected_files
        for rel_file in expected_files:
            assert any(fnmatch.fnmatch(rel_file, pat) for pat in patterns), (
                f"{pkg} package-data {patterns!r} does not match {rel_file!r}"
            )

    router_cu = (repo_root / "hyperflare_recipes/components/sm103_router_kernel/csrc/sm103_router.cu").read_text(
        encoding="utf-8"
    )
    assert router_cu.count("griddepcontrol.launch_dependents;") == 2
    assert "griddepcontrol.wait;" in router_cu
    assert "cudaLaunchAttributeProgrammaticStreamSerialization" in router_cu

    w1_attn_cu = (repo_root / "hyperflare_recipes/components/mega_attention_kernel/csrc/w1_attn.cu").read_text(
        encoding="utf-8"
    )
    assert "void set_num_sms(int n)" in w1_attn_cu
    assert "int get_num_sms()" in w1_attn_cu
