# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP2 CUDA parity and AOT cross-rank guard tests for sharded QSA selection."""

from __future__ import annotations

from functools import partial
import os
import socket
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _requires_tp2():
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1 and FLARE_RUN_TP_CUDA_TESTS=1")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")


def _make_dummy_qsa_module(device: torch.device):
    q_norm = torch.nn.RMSNorm(256, eps=1e-6, device=device, dtype=torch.bfloat16).requires_grad_(False)
    k_norm = torch.nn.RMSNorm(256, eps=1e-6, device=device, dtype=torch.bfloat16).requires_grad_(False)
    module = SimpleNamespace(
        config=SimpleNamespace(rms_norm_eps=1e-6),
        q_norm=q_norm,
        k_norm=k_norm,
        indexer=SimpleNamespace(
            q_layernorm=SimpleNamespace(weight=torch.zeros(128, dtype=torch.bfloat16, device=device)),
            k_layernorm=SimpleNamespace(weight=torch.zeros(128, dtype=torch.bfloat16, device=device)),
        ),
    )
    return module


def _make_weights(device: torch.device, q_heads: int, kv_heads: int, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def mat(out_f: int, in_f: int) -> torch.Tensor:
        return (torch.randn((out_f, in_f), generator=gen) * 0.02).bfloat16().to(device)

    return {
        "qkv_proj": mat(q_heads * 512 + 2 * kv_heads * 256, 2560),
        "indexer.index_qk_proj": mat(640, 2560),
        "o_proj": mat(2560, q_heads * 256),
    }


def _make_rotary(context: int, device: torch.device):
    from hyperflare_recipes.recipes.qwen38.attention import RotaryTables

    cfg = SimpleNamespace(
        head_dim=256,
        rope_parameters={"partial_rotary_factor": 0.25, "mrope_section": (11, 11, 10), "rope_theta": 10000000},
    )
    return RotaryTables.allocate(cfg, context, device)


def _tp2_sharded_qsa_worker(rank: int, world: int, port: int, mode: str, cache_dir: str, error_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_dir, f"inductor_{rank}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.group.WORLD

    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.recipes.qwen38.attention import (
        PreparedAttention,
        QSAMetadata,
        QSAScratch,
        QSAState,
    )
    from ignition.preparation.aot import ForwardCache
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    gather_rt = session = None
    try:
        coordinator = CollectivePreparation(group)
        shard = TensorParallelShard(rank, world)
        context = 8192
        page_size = 64
        num_pages = 512
        if mode == "row":
            tokens, batch = 512, 1
            gather_rt = create_all_gather_runtime(group, device, max_rows=tokens // world, columns=1024 * world)
        else:
            tokens, batch = 32, 4
            gather_rt = create_all_gather_runtime(group, device, max_rows=tokens, columns=2048 * world)

        session = RecordedPreparationSession(
            device=device,
            checkpoint=f"test_sharded_qsa_{mode}",
            workload_schema=f"test_sharded_qsa_{mode}",
            numerical_policy="default",
            coordinator=coordinator,
            autotune_effort="off",
        )
        session.__enter__()

        module = _make_dummy_qsa_module(device)
        weights = _make_weights(device, q_heads=12, kv_heads=1, seed=41)
        rotary = _make_rotary(context, device)
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)

        # Prepare BOTH unsharded reference and sharded candidate using identical weights & initial 1-page pools.
        source_ex = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device)
        scratch_ref = QSAScratch.allocate(tokens, device, world_size=world)
        scratch_shd = QSAScratch.allocate(tokens, device, world_size=world)
        state_ref = QSAState.allocate(pages=1, slots=batch + 1, device=device, draft_tokens=7, kv_heads=1)
        state_shd = QSAState.allocate(pages=1, slots=batch + 1, device=device, draft_tokens=7, kv_heads=1)
        meta_ref = QSAMetadata.allocate(tokens=tokens, batch=batch, context=context, device=device)
        meta_shd = QSAMetadata.allocate(tokens=tokens, batch=batch, context=context, device=device)

        ref_attn = PreparedAttention(
            module, weights, source_ex, scratch_ref, state_ref, meta_ref, rotary,
            session=session, name=f"ref_{mode}", draft_tokens=7,
            workspace=arena, projection_workspace=arena, max_cache_pages=num_pages,
        ).functional(session=session)
        shd_attn = PreparedAttention(
            module, weights, source_ex, scratch_shd, state_shd, meta_shd, rotary,
            session=session, name=f"shd_{mode}", draft_tokens=7,
            workspace=arena, projection_workspace=arena, max_cache_pages=num_pages,
            shard=shard, qsa_gather_runtime=gather_rt, qsa_shard_mode=mode, coordinator=coordinator,
        ).functional(session=session)

        session.tune(show_progress=False)

        # Exercise page rebinding from 1-page preparation pool to final multi-page pool.
        final_ref = state_ref.resize_pages_for_preparation(num_pages)
        final_shd = state_shd.resize_pages_for_preparation(num_pages)
        ref_attn.rebind_pages_for_preparation(final_ref)
        shd_attn.rebind_pages_for_preparation(final_shd)

        # Compile sharded forward through ForwardCache on rank 0 and bind+warm on both ranks
        # via prepare_many(coordinator=...) to verify AOT cross-rank sharing without rank guards.
        run_source = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device)
        run_meta = QSAMetadata.allocate(tokens=tokens, batch=batch, context=context, device=device)
        shd_fwd = partial(shd_attn, run_source, run_meta)
        fwd_cache = ForwardCache(
            namespace=f"test_sharded_qsa_{mode}",
            device=device,
            root=os.path.join(cache_dir, f"aot_{mode}"),
            compile_workers=1,
        )
        compiled_shd = fwd_cache.prepare_many(
            {f"shd_{mode}": shd_fwd},
            {f"shd_{mode}": (mode, tokens, batch)},
            coordinator=coordinator,
        )[f"shd_{mode}"]

        # Build test scenarios covering short context, exact budget boundary, long context, ties, and inactive rows.
        if mode == "row":
            scenarios = [
                ("short_120", 120, 120, False),
                ("boundary_2050", 2050, 512, False),
                ("long_6000", 6000, 512, False),
                ("long_6000_partial", 6000, 384, False),
                ("ties_6000", 6000, 512, True),
            ]
        else:
            scenarios = [
                ("mixed_short_and_boundary", [1, 63, 64, 2048], False),
                ("mixed_boundary_and_long", [2047, 2049, 4096, 7680], False),
                ("inactive_and_long", [0, 4096, 6144, 7936], False),
                ("exact_ties_across_ranks", [4096, 5120, 6144, 7168], True),
            ]

        stream = torch.cuda.Stream(device=device)
        for scenario in scenarios:
            gen = torch.Generator(device="cpu").manual_seed(900 + len(scenario[0]))
            # Identical KV & compressed cache state across ref & shd on both ranks.
            k_init = torch.randn(final_ref.main_k_cache.shape, generator=gen).to(torch.float8_e4m3fn).to(device)
            v_init = torch.randn(final_ref.main_v_cache.shape, generator=gen).to(torch.float8_e4m3fn).to(device)
            if scenario[-1]:
                # Exact tie test: constant compressed keys force identical FP32 scores across every group.
                c_init = torch.ones(final_ref.compressed_k_cache.shape, dtype=torch.bfloat16, device=device) * 0.25
            else:
                c_init = (torch.randn(final_ref.compressed_k_cache.shape, generator=gen) * 0.5).bfloat16().to(device)
            for st in (final_ref, final_shd):
                st.main_k_cache.copy_(k_init)
                st.main_v_cache.copy_(v_init)
                st.compressed_k_cache.copy_(c_init)
                st.raw_k_ring.zero_()
                st.raw_logical_positions.fill_(-1)
                st.raw_rope_positions.fill_(-1)
                st.raw_interval_start_positions.fill_(-1)

            run_source.copy_((torch.randn((tokens, 2560), generator=gen) * 0.5).bfloat16().to(device))
            run_meta.main_block_table.fill_(-1)
            run_meta.raw_state_slot_ids.fill_(-1)
            run_meta.request_ids.fill_(-1)
            run_meta.query_positions.fill_(-1)
            run_meta.rope_positions.fill_(-1)
            run_meta.sequence_lengths.zero_()
            run_meta.query_start_loc.zero_()
            run_meta.num_accepted_tokens.fill_(1)
            run_meta.is_prefilling.zero_()
            run_meta.slot_mapping.fill_(-1)

            if mode == "row":
                _, seqlen, live_tokens, _ = scenario
                pages = (seqlen + page_size - 1) // page_size
                run_meta.main_block_table[0, :pages] = torch.arange(pages, dtype=torch.int32, device=device)
                run_meta.raw_state_slot_ids[0] = 1
                start_pos = seqlen - live_tokens
                run_meta.request_ids[:live_tokens] = 0
                pos = torch.arange(start_pos, seqlen, dtype=torch.int64, device=device)
                run_meta.query_positions[:live_tokens] = pos
                run_meta.rope_positions[:live_tokens] = pos[:, None].expand(-1, 3)
                run_meta.sequence_lengths[0] = seqlen
                run_meta.query_start_loc[1] = live_tokens
                run_meta.is_prefilling[0] = True
                run_meta.slot_mapping[:live_tokens] = pos
            else:
                _, seqlens, _ = scenario
                width = tokens // batch
                offset = 0
                next_page = 0
                for b_idx, slen in enumerate(seqlens):
                    if slen == 0:
                        run_meta.query_start_loc[b_idx + 1] = offset
                        continue
                    pages = (slen + page_size - 1) // page_size
                    run_meta.main_block_table[b_idx, :pages] = torch.arange(
                        next_page, next_page + pages, dtype=torch.int32, device=device
                    )
                    next_page += pages
                    run_meta.raw_state_slot_ids[b_idx] = b_idx + 1
                    live_w = min(width, slen)
                    run_meta.request_ids[offset:offset + live_w] = b_idx
                    pos = torch.arange(slen - live_w, slen, dtype=torch.int64, device=device)
                    run_meta.query_positions[offset:offset + live_w] = pos
                    run_meta.rope_positions[offset:offset + live_w] = pos[:, None].expand(-1, 3)
                    run_meta.sequence_lengths[b_idx] = slen
                    run_meta.slot_mapping[offset:offset + live_w] = (
                        run_meta.main_block_table[b_idx, pos // page_size].long() * page_size + pos % page_size
                    )
                    offset += live_w
                    run_meta.query_start_loc[b_idx + 1] = offset

            dist.barrier(group)
            with torch.cuda.stream(stream):
                ref_out = ref_attn(run_source, run_meta)
                shd_out = compiled_shd()
            stream.synchronize()

            # Replicated selector caches must remain bitwise identical to unsharded reference.
            assert torch.equal(final_shd.compressed_k_cache, final_ref.compressed_k_cache), scenario[0]
            assert torch.equal(final_shd.raw_k_ring, final_ref.raw_k_ring), scenario[0]
            assert torch.equal(final_shd.raw_logical_positions, final_ref.raw_logical_positions), scenario[0]
            assert torch.equal(final_shd.raw_rope_positions, final_ref.raw_rope_positions), scenario[0]
            assert torch.equal(final_shd.raw_interval_start_positions, final_ref.raw_interval_start_positions), scenario[0]

            if mode == "row":
                assert torch.equal(shd_out, ref_out), (
                    f"{scenario[0]}: max diff {(shd_out.float() - ref_out.float()).abs().max().item()}"
                )
            else:
                # Group-split selects the exact same set of groups/positions (sorted ascending by group ID),
                # so sparse GQA output matches within FP32 softmax reduction rounding (<= 1-2 BF16 ULPs).
                max_diff = (shd_out.float() - ref_out.float()).abs().max().item()
                assert max_diff <= 2e-2, f"{scenario[0]}: max diff {max_diff}"
                # Also verify outputs are bitwise identical across TP2 ranks (given identical weights & caches).
                peer = torch.empty_like(shd_out.cpu())
                local_cpu = shd_out.cpu()
                gathered = [torch.empty_like(local_cpu) for _ in range(world)]
                dist.all_gather(gathered, local_cpu, group=group)
                assert torch.equal(gathered[0], gathered[1]), f"{scenario[0]}: rank outputs diverged"

            # Verify CUDA graph capture and replay on the last scenario.
            if scenario is scenarios[-1]:
                dist.barrier(group)
                with torch.cuda.stream(stream):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        g_out = compiled_shd()
                    stream.synchronize()
                    dist.barrier(group)
                    graph.replay()
                    stream.synchronize()
                    graph.reset()
                assert torch.equal(g_out, shd_out), f"{scenario[0]}: CUDA graph replay differed"
    except BaseException as exc:  # noqa: BLE001
        error_queue.put((rank, repr(exc)))
        raise
    finally:
        if session is not None:
            session.close()
        if gather_rt is not None:
            gather_rt.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["row", "group"])
def test_qwen38_sharded_qsa_matches_unsharded_and_shares_aot_binary_tp2(mode):
    """Sharded QSA selection matches unsharded reference and shares one AOT forward across TP2 ranks."""
    _requires_tp2()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        mp.spawn(_tp2_sharded_qsa_worker, args=(2, port, mode, tmpdir, errors), nprocs=2, join=True)
        assert errors.empty(), errors.get()


def test_qwen38_sharded_qsa_group_exact_tie_breaking_positions():
    """Group-split merge selects the exact same top-512 global group set as single-rank stable selection."""
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1")
    from hyperflare_recipes.recipes.qwen38.qsa_sharded import (
        COMPRESSED_PAGE_SIZE,
        GROUP_BUDGET,
        _merge_group_candidates_2_kernel,
        _pack_group_candidates_kernel,
    )

    device = torch.device("cuda:0")
    rows, world = 6, 2
    total_groups = 1200
    gen = torch.Generator(device="cpu").manual_seed(123)
    # Construct scores with heavy ties right at the 512th cutoff across even and odd pages.
    scores = torch.randint(0, 8, (rows, total_groups), generator=gen).float().to(device)
    scores[0] = 1.0  # pure tie: lowest 512 global group IDs (0..511) must win!

    packed_ranks = []
    for r in range(world):
        # Gather rank r's pages (page % world == r)
        local_gids = [g for g in range(total_groups) if (g // COMPRESSED_PAGE_SIZE) % world == r]
        local_scores_full = scores[:, local_gids]
        # Stable top-512 on rank r (higher score first, lower local index first)
        order = torch.argsort(
            (-local_scores_full.double()) * 1e6 + torch.arange(len(local_gids), device=device)[None, :].double(),
            dim=-1,
        )[:, :GROUP_BUDGET]
        l_ids = order.to(torch.int32).contiguous()
        l_sc = local_scores_full.gather(-1, order).contiguous()
        packed = torch.empty((rows, GROUP_BUDGET * 2), dtype=torch.int32, device=device)
        _pack_group_candidates_kernel[(rows,)](
            l_ids, l_sc, packed, torch.tensor([r], dtype=torch.int64, device=device),
            PAGE_GROUPS=COMPRESSED_PAGE_SIZE, WORLD=world, TOPK=GROUP_BUDGET, num_warps=8,
        )
        packed_ranks.append(packed)

    gathered = torch.cat(packed_ranks, dim=-1).contiguous()
    out_ids = torch.empty((rows, GROUP_BUDGET), dtype=torch.int32, device=device)
    _merge_group_candidates_2_kernel[(rows,)](gathered, out_ids, TOPK=GROUP_BUDGET, num_warps=8)
    torch.cuda.synchronize()

    for row in range(rows):
        ref_order = torch.argsort(
            (-scores[row].double()) * 1e6 + torch.arange(total_groups, device=device).double()
        )[:GROUP_BUDGET]
        ref_sorted = torch.sort(ref_order.to(torch.int32)).values
        assert torch.equal(out_ids[row], ref_sorted), f"row {row} tie-breaking mismatch"
