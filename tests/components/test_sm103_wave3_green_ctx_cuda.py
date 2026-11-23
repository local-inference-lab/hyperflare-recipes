# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 hardware verification for fused decode MLA, BF16 router, mHC seam, and FP8 projections under Green Contexts."""

import math
import os
from pathlib import Path
import time

import pytest
import torch

from tests.components.test_sm103_gpu_expert_mover_cuda import (
    _GreenCtxPair,
    _make_random_mxfp4_weights,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned SM103 CUDA device (FLARE_RUN_CUDA_TESTS=1)",
)


def _require_sm103():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 GB300 required")


def _kernel_cache() -> Path:
    cache = Path("/tmp/flare-test-sm103-decode-cache")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def test_mega_attention_cooperative_green_context_with_live_worker(megamoe_core_storage):
    """Verify MegaAttention cooperative launches respect set_num_sms inside a 144-SM Green Context with an 8-SM worker."""
    _require_sm103()
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.mega_attention import MegaAttentionDecoder, MegaAttentionWorkspace
    from hyperflare_recipes.components.mega_attention_kernel import prepare as prepare_mega_attn
    from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
    from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
    from hyperflare_recipes.components.megamoe_weights import allocate_mxfp4_weights
    from hyperflare_recipes.components.sm103_expert_mover import SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    kernel_cache = _kernel_cache()
    device_sms = torch.cuda.get_device_properties(device).multi_processor_count

    gc = _GreenCtxPair(0, worker_sm_count=8)
    compute_sms = gc.compute_sms
    assert gc.worker_sms == 8 and compute_sms < device_sms

    lib = prepare_mega_attn(kernel_cache)
    lib.set_num_sms(compute_sms)
    assert lib.get_num_sms() == compute_sms
    lib.set_num_sms(0)
    assert lib.get_num_sms() == device_sms

    # Prepare one small dynamic mover layer so the 8-SM persistent worker runs concurrently on w_gctx.
    gen = torch.Generator(device="cpu").manual_seed(20261121)
    all_w = _make_random_mxfp4_weights(16, 1024, 512, device, gen)
    geom = MegaMoeGeometry(16, 16, 2, 1024, 512, shared_experts=0)
    moe_ws = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
    initial_hbm = tuple(range(8))
    tier = MegaMoeDynamicHostTier(
        moe_ws.geometry,
        hbm_slots=10,
        initial_hbm_experts=initial_hbm,
        storage=megamoe_core_storage,
        name="mega_attn_gc_worker_tier",
        host_experts=16,
    )
    for dst_cpu, src_gpu in zip(tier.host_views, (*all_w.gate_up, *all_w.down)):
        dst_cpu.copy_(src_gpu.cpu())
    hbm_w = allocate_mxfp4_weights(10, 1024, 512, device=device)
    for dst, src in zip((*hbm_w.gate_up, *hbm_w.down), (*all_w.gate_up, *all_w.down)):
        for slot, expert in enumerate(initial_hbm):
            dst[slot].copy_(src[expert])
        dst[len(initial_hbm):].zero_()

    mover = SM103ExpertMoverRuntime(
        [(tier, hbm_w)],
        storage=megamoe_core_storage,
        kernel_cache=kernel_cache,
        worker_sm_count=gc.worker_sms,
        queue_capacity=64,
        control_buffer_name="mega_attn_gc_worker_ctl",
    )
    try:
        gc.set_primary()
        scale = 1.0 / math.sqrt(512.0)
        swa_width, comp_width = 128, 512
        q8 = (torch.randn((8, 64, 512), generator=gen) * 0.25).to(torch.bfloat16).to(device)
        pos8 = torch.arange(8, dtype=torch.int32, device=device)
        cos_sin = torch.zeros((16, 64), dtype=torch.float32, device=device)
        cos_sin[:, :32] = 1.0
        sink = torch.zeros((64,), dtype=torch.float32, device=device)
        swa_cache = torch.randint(0, 64, (2, 128, 528), dtype=torch.uint8, generator=gen).to(device)
        comp_cache = torch.randint(0, 64, (8, 64, 288), dtype=torch.uint8, generator=gen).to(device)
        swa_idx8 = torch.arange(swa_width, dtype=torch.int32, device=device).unsqueeze(0).expand(8, -1).contiguous()
        comp_idx8 = torch.arange(comp_width, dtype=torch.int32, device=device).unsqueeze(0).expand(8, -1).contiguous()
        comp_lens8 = torch.full((8,), comp_width, dtype=torch.int32, device=device)

        # Reference output on full device SMs (compute_sm_count=0).
        lib.set_num_sms(0)
        ws_ref = MegaAttentionWorkspace(16, device=device, compute_sm_count=0)
        with PreparationSession(device=device, autotune=False) as sess_ref:
            dec_ref = MegaAttentionDecoder(
                (8,), scale=scale, workspace=ws_ref, kernel_cache=kernel_cache,
                session=sess_ref, name="mega_attn_ref", compressed=True,
                swa_width=swa_width, comp_width=comp_width, token_interleaved=True,
                compute_sm_count=0,
            )
        ref_out, ref_sf = dec_ref(q8, pos8, cos_sin, sink, swa_cache, swa_idx8, comp_cache, comp_idx8, comp_lens8)
        torch.cuda.synchronize(device)
        ref_out_cpu = ref_out.cpu()
        ref_sf_cpu = ref_sf.cpu()

        # Green-context clamped decoder (compute_sm_count=compute_sms) + direct 16-row cooperative launch
        # (16 * 10 + 16 * 8 = 288 tasks -> clamped to compute_sms CTAs instead of device_sms CTAs).
        ws_gc = MegaAttentionWorkspace(16, device=device, compute_sm_count=compute_sms)
        with PreparationSession(device=device, autotune=False) as sess_gc:
            dec_gc = MegaAttentionDecoder(
                (8,), scale=scale, workspace=ws_gc, kernel_cache=kernel_cache,
                session=sess_gc, name="mega_attn_gc", compressed=True,
                swa_width=swa_width, comp_width=comp_width, token_interleaved=True,
                compute_sm_count=compute_sms,
            )
        assert lib.get_num_sms() == compute_sms

        q16 = q8.repeat(2, 1, 1).contiguous()
        pos16 = torch.arange(16, dtype=torch.int32, device=device)
        swa_idx16 = swa_idx8.repeat(2, 1).contiguous()
        swa_lens16 = torch.full((16,), swa_width, dtype=torch.int32, device=device)
        comp_idx16 = comp_idx8.repeat(2, 1).contiguous()
        comp_lens16 = torch.full((16,), comp_width, dtype=torch.int32, device=device)
        gc_out_buf = torch.empty((8, 8, 4096), dtype=torch.float8_e4m3fn, device=device)
        gc_sf_buf = torch.empty((8, 8, 32), dtype=torch.int32, device=device)
        out16 = torch.empty((16, 8, 4096), dtype=torch.float8_e4m3fn, device=device)
        sf16 = torch.empty((8, 32, 16), dtype=torch.int32, device=device).permute(2, 0, 1)

        # Capture persistent worker graph, enqueue graph, and compute graph on primary ctx before launching.
        cap_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(cap_stream):
            o_warm, s_warm = dec_gc(q8, pos8, cos_sin, sink, swa_cache, swa_idx8, comp_cache, comp_idx8, comp_lens8)
            gc_out_buf.copy_(o_warm)
            gc_sf_buf.copy_(s_warm)
            lib.w1_attn_decode(
                q16, pos16, cos_sin, sink, scale * math.log2(math.e),
                swa_cache, swa_idx16, swa_lens16, comp_cache, comp_idx16, comp_lens16,
                out16, sf16, *ws_gc.tensors, None, -1, 0, True, 0, None, True,
            )
        cap_stream.synchronize()

        compute_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(compute_graph, pool=megamoe_core_storage.graph_pool, stream=cap_stream):
                o_cap, s_cap = dec_gc(q8, pos8, cos_sin, sink, swa_cache, swa_idx8, comp_cache, comp_idx8, comp_lens8)
                gc_out_buf.copy_(o_cap)
                gc_sf_buf.copy_(s_cap)
                lib.w1_attn_decode(
                    q16, pos16, cos_sin, sink, scale * math.log2(math.e),
                    swa_cache, swa_idx16, swa_lens16, comp_cache, comp_idx16, comp_lens16,
                    out16, sf16, *ws_gc.tensors, None, -1, 0, True, 0, None, True,
                )
        compute_graph.instantiate()
        compute_exec = int(compute_graph.raw_cuda_graph_exec())

        enqueue_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(enqueue_graph, pool=megamoe_core_storage.graph_pool, stream=cap_stream):
                mover.enqueue_swap(0, 8, 0)
        enqueue_graph.instantiate()
        enqueue_exec = int(enqueue_graph.raw_cuda_graph_exec())

        worker_decl = mover.capture_worker_graph(name="mega_attn_worker", retain_graphs=True, stream=cap_stream)
        worker_exec = int(worker_decl.graph.raw_cuda_graph_exec())
        mover.reset_stop_flag()
        torch.cuda.synchronize(device)

        # Launch 8-SM persistent worker on w_gctx, enqueue swap, and replay cooperative MegaAttention on 144-SM c_gctx.
        gc.launch_graph(worker_exec, worker=True)
        gc.launch_graph(enqueue_exec, worker=False)
        for _ in range(4):
            gc.launch_graph(compute_exec, worker=False)
        gc.sync_compute()

        gc.set_primary()
        assert torch.equal(gc_out_buf.cpu().view(torch.uint8), ref_out_cpu.view(torch.uint8))
        assert torch.equal(gc_sf_buf.cpu(), ref_sf_cpu)
        assert torch.equal(out16[:8].cpu().view(torch.uint8), ref_out_cpu.view(torch.uint8))
        assert torch.equal(sf16[:8].cpu(), ref_sf_cpu)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if mover.read_metrics().swaps_completed >= 1:
                break
            time.sleep(0.002)
        assert mover.read_metrics().swaps_completed >= 1
    finally:
        mover.signal_stop()
        gc.sync_worker()
        gc.set_primary()
        lib.set_num_sms(0)
        mover.close()
        for region in tier.regions:
            region.close()
        gc.close()


def test_sm103_router_correctness_including_dynamic_placement():
    """Verify SM103 BF16 router matches FP32 reference across target (384,6) and draft (128,3) geometries."""
    _require_sm103()
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.sm103_router import (
        SM103RouterWorkspace,
        declare_sm103_router_plans,
        run_sm103_router,
    )

    device = torch.device("cuda", 0)
    kernel_cache = _kernel_cache()
    gen = torch.Generator(device="cpu").manual_seed(20261122)

    ws = SM103RouterWorkspace(max_rows=64, max_experts=384, device=device)
    for experts, hidden, top_k in ((384, 5120, 6), (128, 5120, 3)):
        rows_list = (1, 8, 16, 32, 64)
        with PreparationSession(device=device, autotune=False) as session:
            plans, handles = declare_sm103_router_plans(
                rows_list, experts=experts, hidden=hidden, top_k=top_k,
                workspace=ws, kernel_cache=kernel_cache, device=device,
                session=session, name=f"router_{experts}_{top_k}",
            )

        w = (torch.randn((experts, hidden), generator=gen) * 0.05).to(torch.bfloat16).to(device)
        bias = (torch.randn((experts,), generator=gen) * 0.1).to(torch.float32).to(device)
        image_bias = (torch.randn((experts,), generator=gen) * 0.1).to(torch.float32).to(device)
        perm = torch.randperm(experts, generator=gen, dtype=torch.int32).to(device)
        placement_2d = perm.view(experts, 1).contiguous()
        scaling = 2.5

        for rows in rows_list:
            x = (torch.randn((rows, hidden), generator=gen) * 0.2).to(torch.bfloat16).to(device)
            valid = torch.ones((rows,), dtype=torch.bool, device=device)
            if rows > 1:
                valid[-1] = False
            image_mask = torch.zeros((rows,), dtype=torch.bool, device=device)
            image_mask[0] = True
            ids = torch.empty((rows, top_k), dtype=torch.int64, device=device)
            probs = torch.empty((rows, top_k), dtype=torch.float32, device=device)
            routes = torch.empty((rows, top_k), dtype=torch.int64, device=device)

            run_sm103_router(
                x, w, bias, image_bias, valid, image_mask, placement_2d,
                ws.part_buf, ids, probs, scaling, handles[rows], routes=routes,
            )
            torch.cuda.synchronize(device)

            logits = x.float() @ w.float().T
            u = torch.sqrt(torch.nn.functional.softplus(logits))
            b_sel = torch.where(image_mask[:, None], image_bias[None, :], bias[None, :])
            scores = u + b_sel
            ref_routes = torch.topk(scores, top_k, dim=-1).indices
            ref_u = torch.gather(u, 1, ref_routes)
            ref_probs = ref_u / (ref_u.sum(dim=-1, keepdim=True) + 1e-20) * scaling
            ref_ids = perm[ref_routes].to(torch.int64)
            ref_ids = torch.where(valid[:, None], ref_ids, -1)
            ref_routes = torch.where(valid[:, None], ref_routes, -1)
            ref_probs = torch.where(valid[:, None], ref_probs, 0.0)

            assert torch.equal(routes, ref_routes), f"routes mismatch for E={experts}, M={rows}"
            assert torch.equal(ids, ref_ids), f"mapped ids mismatch for E={experts}, M={rows}"
            torch.testing.assert_close(probs, ref_probs, rtol=1e-3, atol=1e-4)


def test_sm103_mhc_overlap_and_low_latency_fp8_projection(megamoe_core_storage):
    """Verify FastRMSNormF8, MHCSeam, and LowLatencyFP8Linear (pre-swizzled & BF16 inputs) on SM103."""
    _require_sm103()
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.sm103_mhc_overlap import (
        MHCSeamQuery,
        PreparedMHCSeam,
        PreparedRMSNormF8,
        RMSNormF8Query,
    )
    from hyperflare_recipes.components.sm103_projection import LowLatencyFP8Linear

    device = torch.device("cuda", 0)
    gen = torch.Generator(device="cpu").manual_seed(20261123)
    hidden = 5120
    eps = 1e-6

    # 1. RMSNorm F8_128x4 kernel parity on M in (1, 8, 32, 64).
    w_norm = (torch.randn((hidden,), generator=gen) * 0.5 + 1.0).to(torch.bfloat16).to(device)
    for rows in (1, 8, 32, 64):
        q = RMSNormF8Query(rows, hidden, return_bf16=True)
        op = PreparedRMSNormF8(q, 16, device)
        x = (torch.randn((rows, hidden), generator=gen) * 0.5).to(torch.bfloat16).to(device)
        y_fp8 = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device)
        y_sf = torch.empty((128 * (hidden // 128),), dtype=torch.int32, device=device)
        y_bf16 = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
        op.run(x, w_norm, y_fp8, y_sf, y_bf16, eps, 1.0)
        torch.cuda.synchronize(device)

        xf = x.float()
        rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        ref_bf16 = (xf * rstd * w_norm.float()).to(torch.bfloat16)
        torch.testing.assert_close(y_bf16, ref_bf16, rtol=1e-2, atol=1e-2)

    # 2. MHCSeam 1-pass kernel parity (mode 0 = F8_128x4, mode 1 = MegaMoE routed + shared).
    for mode, shared_bm in ((0, 0), (1, 16)):
        rows = 8
        q = MHCSeamQuery(rows, hidden, mode=mode, shared_block_m=shared_bm)
        op = PreparedMHCSeam(q, 16, device)
        x = (torch.randn((rows, hidden), generator=gen) * 0.25).to(torch.bfloat16).to(device)
        res = (torch.randn((rows, 4, hidden), generator=gen) * 0.25).to(torch.bfloat16).to(device)
        prev_pre = torch.full((rows, 4, 1), 0.25, dtype=torch.float32, device=device)
        post_mix = torch.full((rows, 4, 1), 0.5, dtype=torch.float32, device=device)
        comb = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).expand(rows, -1, -1).contiguous()
        y_bf16 = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
        y_fp8 = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device)
        if mode == 0:
            y_sf = torch.empty((128 * (hidden // 128),), dtype=torch.int32, device=device)
            y_shared_sf = None
        else:
            y_sf = torch.empty((rows, hidden // 128), dtype=torch.int32, device=device)
            y_shared_sf = torch.empty((128, hidden // 128), dtype=torch.int32, device=device)
        op.run(x, res, prev_pre, post_mix, comb, w_norm, y_bf16, y_fp8, y_sf, y_shared_sf, eps, 1.0)
        torch.cuda.synchronize(device)

        nr = (post_mix * x.float().unsqueeze(1) + torch.einsum("bij,bih->bjh", comb, res.float())).to(torch.bfloat16).float()
        hin = (prev_pre * nr).sum(dim=1).to(torch.bfloat16).float()
        rstd = torch.rsqrt(hin.pow(2).mean(dim=-1, keepdim=True) + eps)
        ref_bf16 = (hin * rstd * w_norm.float()).to(torch.bfloat16)
        torch.testing.assert_close(y_bf16, ref_bf16, rtol=1e-2, atol=1e-2)

    # 3. LowLatencyFP8Linear with bf16_input=True and pre_swizzled_input=True across M in (1, 8, 16, 64)
    # captured and replayed inside a 144-SM Green Context.
    from deepinfer.model_executor.layers.linear import ReplicatedLinear, cast_fp8

    gc = _GreenCtxPair(0, worker_sm_count=8)
    try:
        gc.set_primary()
        out_features = 2048
        w_f32 = (torch.randn((out_features, hidden), generator=gen) * 0.05).to(device)
        w_fp8, w_sf = cast_fp8(w_f32.to(torch.bfloat16))
        w_param = torch.nn.Parameter(w_fp8, requires_grad=False)
        w_param.sf = w_sf
        base = ReplicatedLinear(hidden, out_features, use_fp32_output=False, weight=w_param, bias=None)

        rows_list = (1, 8, 16, 64)
        with PreparationSession(device=device, autotune=False) as session:
            ll_bf16 = LowLatencyFP8Linear(
                base, rows_list, session=session, name="ll_fp8_bf16",
                pre_swizzled_input=True, bf16_input=True,
            )

        cap_stream = torch.cuda.Stream(device=device)
        for rows in rows_list:
            x = (torch.randn((rows, hidden), generator=gen) * 0.2).to(torch.bfloat16).to(device)
            out_buf = torch.empty((rows, out_features), dtype=torch.bfloat16, device=device)
            with torch.cuda.stream(cap_stream):
                out_buf.copy_(ll_bf16(x))
            cap_stream.synchronize()

            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.stream(cap_stream):
                with torch.cuda.graph(g, pool=megamoe_core_storage.graph_pool, stream=cap_stream):
                    out_buf.copy_(ll_bf16(x))
            g.instantiate()
            gc.launch_graph(int(g.raw_cuda_graph_exec()), worker=False)
            gc.sync_compute()
            gc.set_primary()

            ref = base(x)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(out_buf.float(), ref.float(), rtol=2e-2, atol=2e-2)
    finally:
        gc.set_primary()
        gc.close()


def test_sm103_cublaslt_gemm_nt_and_mhc_replay_under_green_context(megamoe_core_storage):
    """Verify aux_kv_stream cublaslt_gemm_nt (under set_num_sms) and multi-call mega_mhc graph replay."""
    _require_sm103()
    import deep_gemm

    device = torch.device("cuda", 0)
    gc = _GreenCtxPair(0, worker_sm_count=8)
    prev_sms = deep_gemm.get_num_sms()
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(gc.compute_sms)
        gen = torch.Generator(device="cpu").manual_seed(20261124)

        # 1. Verify cublaslt_gemm_nt across aux_kv_stream BF16/FP32 output shapes inside a 144-SM Green Context.
        cap_stream = torch.cuda.Stream(device=device)
        for out_dtype in (torch.bfloat16, torch.float32):
            for m, n, k in ((8, 32, 5120), (8, 512, 5120), (8, 128, 512)):
                a = (torch.randn((m, k), generator=gen) * 0.2).to(torch.bfloat16).to(device)
                b = (torch.randn((n, k), generator=gen) * 0.2).to(torch.bfloat16).to(device)
                out = torch.empty((m, n), dtype=out_dtype, device=device)
                with torch.cuda.stream(cap_stream):
                    deep_gemm.cublaslt_gemm_nt(a, b, out)
                cap_stream.synchronize()

                g = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.stream(cap_stream):
                    with torch.cuda.graph(g, pool=megamoe_core_storage.graph_pool, stream=cap_stream):
                        deep_gemm.cublaslt_gemm_nt(a, b, out)
                g.instantiate()
                out.zero_()
                gc.launch_graph(int(g.raw_cuda_graph_exec()), worker=False)
                gc.sync_compute()
                gc.set_primary()

                ref = (a.float() @ b.float().T).to(out_dtype)
                torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=5e-2)

        # 2. Verify multi-call shifted mega_mhc graph replay (matching aux_stream overlap layout where every
        #    graph executes >= 2 mega_mhc calls per barrier) produces zero NaNs across 200 replays.
        m, h, hc = 8, 5120, 4
        x = (torch.randn((m, h), generator=gen) * 0.1).to(torch.bfloat16).to(device)
        res = (torch.randn((m, hc, h), generator=gen) * 0.1).to(torch.bfloat16).to(device)
        prev_mix = torch.full((m, hc, 1), 0.25, dtype=torch.float32, device=device)
        post_mix = torch.full((m, hc, 1), 0.5, dtype=torch.float32, device=device)
        comb = torch.eye(hc, dtype=torch.float32, device=device).unsqueeze(0).expand(m, hc, hc).contiguous()
        fn = (torch.randn((hc * (hc + 2), hc * h), generator=gen) * 0.01).to(device)
        scales = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device)
        bases = torch.zeros((hc * (hc + 2),), dtype=torch.float32, device=device)
        w = torch.ones((h,), dtype=torch.bfloat16, device=device)
        nres = torch.empty_like(res)
        nprev = torch.empty_like(prev_mix)
        npost = torch.empty_like(post_mix)
        ncomb = torch.empty_like(comb)
        y_bf16 = torch.empty((m, h), dtype=torch.bfloat16, device=device)
        y_fp8 = torch.empty((m, h), dtype=torch.float8_e4m3fn, device=device)
        y_sf = torch.empty((h // 128, ((m + 3) // 4) * 4), dtype=torch.int32, device=device).t()[:m]

        with torch.cuda.stream(cap_stream):
            deep_gemm.mega_mhc(
                x, res, prev_mix, post_mix, comb,
                fn, scales, bases, hc, 1e-20, 1e-6, 2.0, 1e-6, 20,
                w, 1e-20, 1.0,
                nres, nprev, npost, ncomb,
                y_bf16=y_bf16, y_fp8=y_fp8, y_gemm_sf=y_sf,
            )
        cap_stream.synchronize()

        nan_count = torch.zeros((1,), dtype=torch.int64, device=device)
        mhc_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(mhc_graph, pool=megamoe_core_storage.graph_pool, stream=cap_stream):
                for _ in range(2):
                    deep_gemm.mega_mhc(
                        x, res, prev_mix, post_mix, comb,
                        fn, scales, bases, hc, 1e-20, 1e-6, 2.0, 1e-6, 20,
                        w, 1e-20, 1.0,
                        nres, nprev, npost, ncomb,
                        y_bf16=y_bf16, y_fp8=y_fp8, y_gemm_sf=y_sf,
                    )
                nan_count += torch.isnan(y_bf16).sum()
        mhc_graph.instantiate()
        exec_h = int(mhc_graph.raw_cuda_graph_exec())
        nan_count.zero_()
        for _ in range(200):
            gc.launch_graph(exec_h, worker=False)
        gc.sync_compute()
        gc.set_primary()
        assert nan_count.item() == 0
    finally:
        deep_gemm.set_num_sms(prev_sms)
        gc.set_primary()
        gc.close()
