# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 hardware qualification for the inline on-GPU routing predictor + 8-SM 1D-TMA expert mover."""

import ctypes as C
import os
from pathlib import Path
import struct
import time

import pytest
import torch

from tests.components.test_sm103_gpu_expert_mover_cuda import (
    _GreenCtxPair,
    _make_random_mxfp4_weights,
    _make_random_shared_fp8,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned SM103 CUDA device (FLARE_RUN_CUDA_TESTS=1)",
)


@pytest.fixture(scope="module")
def sm103_pred_storage(megamoe_core_storage):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 GB300 required")
    return megamoe_core_storage


def test_sm103_inline_gpu_predictor_adapts_across_regime_shifts_bit_exact(sm103_pred_storage):
    """Verify inline on-GPU predictor promotes cold working sets across topic shifts with bit-exact MegaMoE output."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
    from hyperflare_recipes.components.megamoe_weights import allocate_mxfp4_weights
    from hyperflare_recipes.components.sm103_expert_mover import (
        PredictorConfig,
        SM103ExpertMoverRuntime,
        query_green_context_compute_sms,
    )

    device = torch.device("cuda", 0)
    worker_sms, compute_sms = query_green_context_compute_sms(0, 8)
    gc = _GreenCtxPair(0, worker_sms)
    assert gc.worker_sms == worker_sms and gc.compute_sms == compute_sms

    E = 64
    S_ACTIVE = 16
    K_SPARE = 2
    HBM_SLOTS = S_ACTIVE + K_SPARE  # 18 slots
    HIDDEN = 1024
    INTER = 512
    TOP_K = 4
    CAPACITY = 128
    ROWS = 16
    CLAMP = 7.0

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    storage = sm103_pred_storage
    mover = None
    tier = None
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        gen = torch.Generator(device="cpu").manual_seed(20261121)
        geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
        ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        ws_dyn = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

        full_weights = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
        shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

        # Start with experts 0..15 resident in HBM; 16..63 are cold in HOST_NUMA.
        tier = MegaMoeDynamicHostTier(
            ws_dyn.geometry,
            hbm_slots=HBM_SLOTS,
            initial_hbm_experts=tuple(range(S_ACTIVE)),
            storage=storage,
            name="pred_layer0",
        )
        tier.copy_weights(
            type(full_weights)(
                (full_weights.gate_up[0].cpu(), full_weights.gate_up[1].cpu()),
                (full_weights.down[0].cpu(), full_weights.down[1].cpu()),
            )
        )
        hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
        for dst, src in zip((*hbm_pool.gate_up, *hbm_pool.down), (*full_weights.gate_up, *full_weights.down)):
            dst[:S_ACTIVE].copy_(src[:S_ACTIVE])
            dst[S_ACTIVE:].zero_()

        mover = SM103ExpertMoverRuntime(
            [(tier, hbm_pool)],
            storage=storage,
            kernel_cache=kernel_cache,
            worker_sm_count=worker_sms,
            queue_capacity=1024,
        )
        # Fast adaptation settings: 256 = 1-miss promote threshold, 4 enqueues/step, decay_shift=2,
        # min_age_steps=4, and interval_steps=1 for rapid regime shift within 24 steps.
        mover.attach_predictors(
            PredictorConfig(
                decay_shift=2,
                promote_threshold=256,
                min_age_steps=4,
                interval_steps=1,
                cooldown_steps=1,
                max_enqueues=4,
                max_predict_rows=64,
            )
        )

        with PreparationSession(device=device, autotune=False) as session:
            ex = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
            probs = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)
            ref_op = PreparedMegaMoe(
                ex, ids, probs, pad, full_weights, shared_weights,
                workspace=ws_ref, clamp=CLAMP, session=session, name="pred_ref",
                host_shared_first=True,
            )
            dyn_op = PreparedMegaMoe(
                ex, ids, probs, pad, hbm_pool, shared_weights,
                workspace=ws_dyn, clamp=CLAMP, session=session, name="pred_dyn",
                host_tier=tier, host_shared_first=True, compute_sm_count=compute_sms,
            )

        def make_regime_batches(expert_start: int, count: int):
            out = []
            for _ in range(count):
                x = (torch.randn((ROWS, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16)
                perm = torch.stack([torch.randperm(S_ACTIVE, generator=gen)[:TOP_K] + expert_start for _ in range(ROWS)])
                ids_cpu = perm.to(torch.int32).contiguous()
                probs_cpu = torch.softmax(torch.randn((ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32)
                pad_cpu = torch.zeros((ROWS,), dtype=torch.bool)
                gold = ref_op(x.to(device), ids_cpu.to(device), probs_cpu.to(device), pad_cpu.to(device)).cpu()
                out.append((x, ids_cpu, probs_cpu, pad_cpu, gold))
            torch.cuda.synchronize()
            return out

        regime_a = make_regime_batches(32, 24)  # Cold experts 32..47 (0% initial HBM hit rate)
        regime_b = make_regime_batches(48, 24)  # Abrupt topic shift to cold experts 48..63.

        # Capture compute graph (with inline predictor in sm103_layer_enter_predict_kernel) and mover graph.
        x_buf = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        ids_buf = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
        probs_buf = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
        pad_buf = torch.zeros((ROWS,), dtype=torch.bool, device=device)
        out_buf = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        torch.cuda.synchronize()

        cap_stream = torch.cuda.Stream(device=device)
        comp_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(comp_graph, pool=storage.graph_pool, stream=cap_stream):
                out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        comp_graph.instantiate()

        worker_decl = mover.capture_worker_graph(name="sm103_pred_mover", retain_graphs=True, stream=cap_stream)
        mover.reset_runtime_state()
        torch.cuda.synchronize()

        host_out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16)

        def run_step(x_cpu, ids_cpu, probs_cpu, pad_cpu, gold_cpu=None):
            gc.set_compute()
            for dst, src in ((x_buf, x_cpu), (ids_buf, ids_cpu), (probs_buf, probs_cpu), (pad_buf, pad_cpu)):
                gc._call(
                    "cuMemcpyHtoDAsync_v2",
                    C.c_uint64(dst.data_ptr()),
                    C.c_void_p(src.data_ptr()),
                    C.c_size_t(src.numel() * src.element_size()),
                    gc.c_stream,
                )
            gc.launch_graph(int(comp_graph.raw_cuda_graph_exec()), worker=False)
            gc._call(
                "cuMemcpyDtoHAsync_v2",
                C.c_void_p(host_out.data_ptr()),
                C.c_uint64(out_buf.data_ptr()),
                C.c_size_t(host_out.numel() * host_out.element_size()),
                gc.c_stream,
            )
            gc.sync_compute()
            if gold_cpu is not None:
                assert torch.equal(host_out, gold_cpu), (
                    f"inline predictor live swap mismatch: max_diff={(host_out.float() - gold_cpu.float()).abs().max().item()}"
                )

        gc.launch_graph(int(worker_decl.graph.raw_cuda_graph_exec()), worker=True)
        try:
            # --- Regime A: Cold experts 32..47 (starts at 0% HBM hit rate) ---
            for step_i, batch in enumerate(regime_a):
                run_step(*batch)
                # Give background 8-SM mover realistic inter-step decode window (~2ms) during warmup steps
                if step_i < 10:
                    time.sleep(0.002)
            m_before_tail_a = mover.read_metrics()
            for batch in regime_a[-6:]:
                run_step(*batch)
            m_after_a = mover.read_metrics()
            tail_hits_a = m_after_a.predict_hits - m_before_tail_a.predict_hits
            tail_misses_a = m_after_a.predict_misses - m_before_tail_a.predict_misses
            tail_rate_a = tail_hits_a / max(1, tail_hits_a + tail_misses_a)
            print(
                f"\n[Predictor Regime A (32..47)] tail_hit_rate={tail_rate_a*100:.1f}% "
                f"({tail_hits_a}/{tail_hits_a+tail_misses_a}), swaps={m_after_a.swaps_completed}, "
                f"enqueues={m_after_a.predict_enqueues}"
            )
            assert tail_rate_a >= 0.95, f"expected Regime A tail hit rate >= 95%, got {tail_rate_a:.3f} ({m_after_a})"

            # --- Regime B: Abrupt shift to cold experts 48..63 (drops back to 0% HBM hit rate) ---
            for step_i, batch in enumerate(regime_b):
                run_step(*batch)
                if step_i < 12:
                    time.sleep(0.002)
            m_before_tail_b = mover.read_metrics()
            for batch in regime_b[-6:]:
                run_step(*batch)
            m_after_b = mover.read_metrics()
            tail_hits_b = m_after_b.predict_hits - m_before_tail_b.predict_hits
            tail_misses_b = m_after_b.predict_misses - m_before_tail_b.predict_misses
            tail_rate_b = tail_hits_b / max(1, tail_hits_b + tail_misses_b)
            print(
                f"[Predictor Regime B (48..63)] tail_hit_rate={tail_rate_b*100:.1f}% "
                f"({tail_hits_b}/{tail_hits_b+tail_misses_b}), swaps={m_after_b.swaps_completed}, "
                f"enqueues={m_after_b.predict_enqueues}"
            )
            assert tail_rate_b >= 0.95, f"expected Regime B tail hit rate >= 95%, got {tail_rate_b:.3f} ({m_after_b})"

            # --- Principle 7: Inactive/padded rows perform ZERO predictor state mutations ---
            m_before_pad = mover.read_metrics()
            all_pad_x = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16)
            all_pad_ids = torch.zeros((ROWS, TOP_K), dtype=torch.int32)
            all_pad_probs = torch.zeros((ROWS, TOP_K), dtype=torch.float32)
            all_pad_mask = torch.ones((ROWS,), dtype=torch.bool)
            for _ in range(5):
                run_step(all_pad_x, all_pad_ids, all_pad_probs, all_pad_mask)
            m_after_pad = mover.read_metrics()
            assert m_after_pad.predict_hits == m_before_pad.predict_hits
            assert m_after_pad.predict_misses == m_before_pad.predict_misses
            assert m_after_pad.predict_enqueues == m_before_pad.predict_enqueues
        finally:
            gc.sync_compute()
            mover.signal_stop()
            gc.sync_worker()
            gc.set_primary()
            worker_decl.graph.reset()
            comp_graph.reset()

        assert int(tier.layer_epoch.item()) % 2 == 0
    finally:
        if mover is not None:
            mover.close()
        if tier is not None:
            for r in tier.regions:
                r.close()
        gc.close()


def test_sm103_inline_predictor_mixed_active_and_padded_rows_invariant(sm103_pred_storage):
    """Verify mixed active + padded rows leave bit-exact predictor scores, recency, cooldown, and counters."""
    from hyperflare_recipes.components.megamoe import MegaMoeGeometry
    from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
    from hyperflare_recipes.components.megamoe_weights import allocate_mxfp4_weights
    from hyperflare_recipes.components.sm103_expert_mover import PredictorConfig, SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    E, HBM_SLOTS, TOP_K = 384, 130, 6
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, TOP_K, 128, 128, shared_experts=1)
    tier_a = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(128)),
        storage=sm103_pred_storage, name="pred_mix_a",
    )
    tier_b = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(128)),
        storage=sm103_pred_storage, name="pred_mix_b",
    )
    hbm_a = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    hbm_b = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    mover_a = SM103ExpertMoverRuntime(
        [(tier_a, hbm_a)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8,
        control_buffer_name="mix_ctrl_a",
    )
    mover_b = SM103ExpertMoverRuntime(
        [(tier_b, hbm_b)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8,
        control_buffer_name="mix_ctrl_b",
    )
    try:
        cfg = PredictorConfig(
            decay_shift=6,
            promote_threshold=256,
            min_age_steps=4,
            interval_steps=2,
            cooldown_steps=2,
            max_enqueues=2,
        )
        (bind_a,) = mover_a.attach_predictors(cfg)
        (bind_b,) = mover_b.attach_predictors(cfg)

        gen = torch.Generator(device="cpu").manual_seed(20261121)
        active_rows = 8
        padded_rows = 8
        total_rows = active_rows + padded_rows
        for _ in range(10):
            active_ids = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(active_rows)]).to(torch.int64)
            mixed_ids = torch.full((total_rows, TOP_K), -1, dtype=torch.int64)
            # Interleave active rows at even indices and padded (-1) rows at odd indices.
            mixed_ids[0::2] = active_ids

            mover_a.binding.launch_layer_enter_predict(
                tier_a.live_slot_map, tier_a.control, tier_a.layer_epoch, active_ids.to(device).contiguous(),
                active_rows, TOP_K, E, bind_a.state, bind_a.queue_header, bind_a.queue_ring,
                bind_a.metrics_ptr, bind_a.metrics_count,
            )
            mover_b.binding.launch_layer_enter_predict(
                tier_b.live_slot_map, tier_b.control, tier_b.layer_epoch, mixed_ids.to(device).contiguous(),
                total_rows, TOP_K, E, bind_b.state, bind_b.queue_header, bind_b.queue_ring,
                bind_b.metrics_ptr, bind_b.metrics_count,
            )
        torch.cuda.synchronize()

        assert torch.equal(bind_a.scores, bind_b.scores), "padded rows mutated predictor scores"
        assert torch.equal(bind_a.last_seen, bind_b.last_seen), "padded rows mutated predictor last_seen"
        assert torch.equal(bind_a.born, bind_b.born), "padded rows mutated predictor born"
        assert torch.equal(bind_a.cooldown, bind_b.cooldown), "padded rows mutated predictor cooldown"
        assert torch.equal(bind_a.state[32:40], bind_b.state[32:40]), "padded rows mutated step_clock or last_enqueue_step"
        # Compare HBM shadow counters in queue_header[48:72] (predict_hits, predict_misses, predict_enqueues).
        assert torch.equal(bind_a.queue_header[48:72], bind_b.queue_header[48:72]), "padded rows mutated HBM counters"
        assert int(bind_a.scores.sum().item()) > 0, "expected active rows to produce nonzero cold-miss scores"
    finally:
        mover_a.close()
        mover_b.close()
        for r in (*tier_a.regions, *tier_b.regions):
            r.close()


def _read_queue_tail_and_requests(queue_header: torch.Tensor, queue_ring: torch.Tensor) -> list[tuple[int, int, int, int]]:
    head, tail, mask = struct.unpack_from("<III", bytes(queue_header[:12].cpu().tolist()), 0)
    ring_bytes = bytes(queue_ring.cpu().tolist())
    reqs = []
    for idx in range(head, tail):
        off = (idx & mask) * 16
        reqs.append(struct.unpack_from("<IiiI", ring_bytes, off))
    return reqs


def _read_predictor_clocks(state: torch.Tensor) -> tuple[int, int]:
    return struct.unpack_from("<2I", bytes(state[32:40].cpu().tolist()), 0)


def _read_born_u32(born: torch.Tensor) -> list[int]:
    return [int(v) & 0xFFFFFFFF for v in born.cpu().tolist()]


def test_sm103_predictor_min_age_protection_and_physical_install_birth(sm103_pred_storage):
    """Verify min_age_steps protects newly installed experts and born updates only on live_slot_map transitions."""
    from hyperflare_recipes.components.megamoe import MegaMoeGeometry
    from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
    from hyperflare_recipes.components.megamoe_weights import allocate_mxfp4_weights
    from hyperflare_recipes.components.sm103_expert_mover import PredictorConfig, SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    E, HBM_SLOTS, TOP_K = 16, 5, 4
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, TOP_K, 128, 128, shared_experts=1)
    tier = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=(0, 1, 2, 3),
        storage=sm103_pred_storage, name="pred_age_tier",
    )
    hbm = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    mover = SM103ExpertMoverRuntime(
        [(tier, hbm)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8,
        control_buffer_name="age_ctrl",
    )
    try:
        cfg = PredictorConfig(
            decay_shift=6, promote_threshold=256, min_age_steps=5,
            interval_steps=1, cooldown_steps=1, max_enqueues=1,
        )
        (bind,) = mover.attach_predictors(cfg)

        born_init = _read_born_u32(bind.born)
        assert born_init[:4] == [0x80000000] * 4
        assert born_init[4:] == [0] * (E - 4)
        step0, last_enq0 = _read_predictor_clocks(bind.state)
        assert step0 == 5 and last_enq0 == 0

        def step_ids(expert_ids):
            t = torch.tensor([expert_ids], dtype=torch.int64, device=device)
            mover.binding.launch_layer_enter_predict(
                tier.live_slot_map, tier.control, tier.layer_epoch, t,
                1, TOP_K, E, bind.state, bind.queue_header, bind.queue_ring,
                bind.metrics_ptr, bind.metrics_count,
            )
            torch.cuda.synchronize()

        # Step 1: cold expert 4 misses while residents 1, 2, 3 are hit -> enqueues (in=4, victim=0).
        step_ids([1, 2, 3, 4])
        reqs = _read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)
        assert len(reqs) == 1
        assert reqs[0][:3] == (0, 4, 0)
        # An unexecuted enqueue MUST NOT mark expert 4 as born/resident or clear expert 0's resident bit.
        born_after_enq = _read_born_u32(bind.born)
        assert born_after_enq[4] == 0
        assert born_after_enq[0] == 0x80000000
        assert int(bind.scores[4].item()) == 256

        # Step 2: simulate physical swap completion by mover (evict 0, install 4 into slot 0).
        tier.live_slot_map[0] = -1
        tier.live_slot_map[4] = 0
        step_ids([1, 2, 3, 4])
        step2, _ = _read_predictor_clocks(bind.state)
        born_after_install = _read_born_u32(bind.born)
        assert born_after_install[0] == 0
        assert born_after_install[4] == (0x80000000 | step2)
        assert int(bind.last_seen[4].item()) == step2
        assert int(bind.scores[4].item()) == 0

        # Steps 3..6: hit 1, 2, 3 and cold expert 6 (expert 4 is the ONLY unhit resident, age = 1..4 < 5).
        for expected_age in range(1, 5):
            step_ids([1, 2, 3, 6])
            cur_step, _ = _read_predictor_clocks(bind.state)
            assert cur_step - step2 == expected_age
            assert len(_read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)) == 1, (
                f"newly installed expert 4 was evicted prematurely at age {expected_age} < min_age_steps=5"
            )

        # Step 7: expert 4 reaches age == 5 >= min_age_steps -> now evictable in favor of cold expert 6.
        step_ids([1, 2, 3, 6])
        reqs = _read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)
        assert len(reqs) == 2
        assert reqs[1][:3] == (0, 6, 4)

        # Verify reset_runtime_state() syncs born to current live_slot_map and resets clocks.
        mover.reset_runtime_state()
        torch.cuda.synchronize()
        born_reset = _read_born_u32(bind.born)
        assert born_reset[0] == 0
        assert born_reset[1] == 0x80000000 and born_reset[2] == 0x80000000
        assert born_reset[3] == 0x80000000 and born_reset[4] == 0x80000000
        assert _read_predictor_clocks(bind.state) == (5, 0)
        assert int(bind.scores.sum().item()) == 0
        assert int(bind.last_seen.sum().item()) == 0
        assert int(bind.cooldown.sum().item()) == 0
    finally:
        mover.close()
        for r in tier.regions:
            r.close()


def test_sm103_predictor_pure_lru_victim_ordering(sm103_pred_storage):
    """Verify victim selection uses pure LRU (oldest last_seen) regardless of historical hit count."""
    from hyperflare_recipes.components.megamoe import MegaMoeGeometry
    from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
    from hyperflare_recipes.components.megamoe_weights import allocate_mxfp4_weights
    from hyperflare_recipes.components.sm103_expert_mover import PredictorConfig, SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    E, HBM_SLOTS, TOP_K = 16, 5, 4
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, TOP_K, 128, 128, shared_experts=1)
    tier = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=(0, 1, 2, 3),
        storage=sm103_pred_storage, name="pred_lru_tier",
    )
    hbm = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    mover = SM103ExpertMoverRuntime(
        [(tier, hbm)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8,
        control_buffer_name="lru_ctrl",
    )
    try:
        cfg = PredictorConfig(
            decay_shift=6, promote_threshold=256, min_age_steps=0,
            interval_steps=1, cooldown_steps=1, max_enqueues=1,
        )
        (bind,) = mover.attach_predictors(cfg)

        def step_rows(rows):
            t = torch.tensor(rows, dtype=torch.int64, device=device)
            mover.binding.launch_layer_enter_predict(
                tier.live_slot_map, tier.control, tier.layer_epoch, t,
                len(rows), TOP_K, E, bind.state, bind.queue_header, bind.queue_ring,
                bind.metrics_ptr, bind.metrics_count,
            )
            torch.cuda.synchronize()

        # Give expert 2 high hit frequency first, then touch 0, 3, 1 later so LRU order is 2 < 0 < 3 < 1.
        step_rows([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
        step_rows([[0, -1, -1, -1]])
        step_rows([[3, -1, -1, -1]])
        step_rows([[1, -1, -1, -1]])

        # All residents are unhit on this step while cold expert 8 misses -> oldest last_seen is expert 2.
        step_rows([[8, -1, -1, -1]])
        reqs = _read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)
        assert len(reqs) == 1
        assert reqs[0][:3] == (0, 8, 2)

        # Touch expert 2 to make it MRU while cold expert 9 misses -> next oldest last_seen is expert 0.
        step_rows([[2, 9, -1, -1]])
        reqs = _read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)
        assert len(reqs) == 2
        assert reqs[1][:3] == (0, 9, 0)

        # Verify resident scores remain 0 (cold-miss-only scoring).
        assert bind.scores[:4].cpu().tolist() == [0, 0, 0, 0]
    finally:
        mover.close()
        for r in tier.regions:
            r.close()


def test_sm103_predictor_interval_steps_pacing(sm103_pred_storage):
    """Verify interval_steps gates per-layer enqueue cadence while scores and last_seen update every active step."""
    from hyperflare_recipes.components.megamoe import MegaMoeGeometry
    from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
    from hyperflare_recipes.components.megamoe_weights import allocate_mxfp4_weights
    from hyperflare_recipes.components.sm103_expert_mover import PredictorConfig, SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    E, HBM_SLOTS, TOP_K = 16, 5, 4
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, TOP_K, 128, 128, shared_experts=1)
    tier = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=(0, 1, 2, 3),
        storage=sm103_pred_storage, name="pred_interval_tier",
    )
    hbm = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    mover = SM103ExpertMoverRuntime(
        [(tier, hbm)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8,
        control_buffer_name="interval_ctrl",
    )
    try:
        cfg = PredictorConfig(
            decay_shift=6, promote_threshold=256, min_age_steps=0,
            interval_steps=4, cooldown_steps=1, max_enqueues=1,
        )
        (bind,) = mover.attach_predictors(cfg)

        def step_ids(expert_ids):
            t = torch.tensor([expert_ids], dtype=torch.int64, device=device)
            mover.binding.launch_layer_enter_predict(
                tier.live_slot_map, tier.control, tier.layer_epoch, t,
                1, TOP_K, E, bind.state, bind.queue_header, bind.queue_ring,
                bind.metrics_ptr, bind.metrics_count,
            )
            torch.cuda.synchronize()

        # Step 1: initial enqueue is immediately allowed (init_clock=4, step_now=5, last_enqueue_step=0).
        step_ids([1, 4, -1, -1])
        step1, last_enq1 = _read_predictor_clocks(bind.state)
        assert step1 == 5 and last_enq1 == 5
        assert len(_read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)) == 1

        # Steps 2, 3, 4 (delta = 1, 2, 3 < interval_steps=4): enqueue is gated, but last_seen and scores update.
        step_ids([2, 5, -1, -1])
        step2, last_enq2 = _read_predictor_clocks(bind.state)
        assert step2 == 6 and last_enq2 == 5
        assert int(bind.last_seen[2].item()) == 6
        assert int(bind.scores[5].item()) == 256
        assert len(_read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)) == 1

        step_ids([3, 6, -1, -1])
        step3, last_enq3 = _read_predictor_clocks(bind.state)
        assert step3 == 7 and last_enq3 == 5
        assert int(bind.last_seen[3].item()) == 7
        assert int(bind.scores[6].item()) == 256
        # Cold expert 5 decayed by 1/64 on step 3: 256 - (256 >> 6) = 252.
        assert int(bind.scores[5].item()) == 252
        assert len(_read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)) == 1

        step_ids([1, 7, -1, -1])
        step4, last_enq4 = _read_predictor_clocks(bind.state)
        assert step4 == 8 and last_enq4 == 5
        assert int(bind.last_seen[1].item()) == 8
        assert int(bind.scores[7].item()) == 256
        assert len(_read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)) == 1

        # Step 5 (delta = 9 - 5 = 4 >= interval_steps=4): enqueue opens again and picks cold expert 8 with LRU victim 0.
        step_ids([1, 8, -1, -1])
        step5, last_enq5 = _read_predictor_clocks(bind.state)
        assert step5 == 9 and last_enq5 == 9
        reqs = _read_queue_tail_and_requests(bind.queue_header, bind.queue_ring)
        assert len(reqs) == 2
        assert reqs[1][:3] == (0, 8, 0)
    finally:
        mover.close()
        for r in tier.regions:
            r.close()


def test_sm103_inline_predictor_kernel_latency_deepseek41_shape(sm103_pred_storage):
    """Measure sm103_layer_enter_predict_kernel latency on DeepSeek V4.1 Flash shape (E=384, top_k=6) across rows=1..8192."""
    from hyperflare_recipes.components.megamoe import MegaMoeGeometry
    from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier
    from hyperflare_recipes.components.megamoe_weights import allocate_mxfp4_weights
    from hyperflare_recipes.components.sm103_expert_mover import PredictorConfig, SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    E, HBM_SLOTS, TOP_K = 384, 290, 6
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, TOP_K, 128, 128, shared_experts=1)
    tier = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(288)),
        storage=sm103_pred_storage, name="pred_lat_layer0",
    )
    hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    mover = SM103ExpertMoverRuntime(
        [(tier, hbm_pool)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8
    )
    try:
        (binding,) = mover.attach_predictors(PredictorConfig(max_enqueues=1, max_predict_rows=256))
        gen = torch.Generator(device="cpu").manual_seed(99)
        cap_s = torch.cuda.Stream(device=device)
        iters = 200

        def measure_graph(fn):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(cap_s):
                with torch.cuda.graph(g, stream=cap_s):
                    for _ in range(iters):
                        fn()
            g.replay()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            us = (s.elapsed_time(e) * 1000.0) / iters
            g.reset()
            return us

        # Warmup
        warm_ids = torch.zeros((16, TOP_K), dtype=torch.int64, device=device)
        for _ in range(10):
            mover.binding.launch_layer_enter_predict(
                tier.live_slot_map, tier.control, tier.layer_epoch, warm_ids,
                16, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                binding.metrics_ptr, binding.metrics_count,
            )
        torch.cuda.synchronize()
        mover.reset_runtime_state()
        torch.cuda.synchronize()

        base_us = measure_graph(
            lambda: mover.binding.launch_layer_enter(tier.live_slot_map, tier.control, tier.layer_epoch, E)
        )
        results = {}
        for rows in (1, 16, 64, 256, 1024, 8192):
            ids_hit = torch.stack([torch.randperm(288, generator=gen)[:TOP_K] for _ in range(rows)]).to(torch.int64).to(device).contiguous()
            ids_miss = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(rows)]).to(torch.int64).to(device).contiguous()
            hit_us = measure_graph(
                lambda r=rows, t=ids_hit: mover.binding.launch_layer_enter_predict(
                    tier.live_slot_map, tier.control, tier.layer_epoch, t,
                    r, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                    binding.metrics_ptr, binding.metrics_count,
                )
            )
            miss_us = measure_graph(
                lambda r=rows, t=ids_miss: mover.binding.launch_layer_enter_predict(
                    tier.live_slot_map, tier.control, tier.layer_epoch, t,
                    r, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                    binding.metrics_ptr, binding.metrics_count,
                )
            )
            results[rows] = (hit_us, miss_us)

        summary = ", ".join(
            f"rows={r}: hit={h:.2f}us miss={m:.2f}us" for r, (h, m) in results.items()
        )
        print(f"\n[Predictor Latency (Graph) E={E} top_k={TOP_K}] base_enter={base_us:.2f} us | {summary}")
        assert results[16][0] < 8.0, f"rows=16 steady hit too slow: {results[16][0]:.2f} us"
        assert results[16][1] < 12.0, f"rows=16 active miss too slow: {results[16][1]:.2f} us"
        # Prefill chunks above max_predict_rows=256 skip route tallying and pay only Dekker snapshot latency.
        assert results[8192][1] < 6.0, f"rows=8192 prefill bypass too slow: {results[8192][1]:.2f} us"
    finally:
        mover.close()
        for r in tier.regions:
            r.close()
