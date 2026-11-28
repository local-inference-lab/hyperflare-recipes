# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU numerical reference tests for Inkling 2 Triton ring, residual, and loop kernels."""

import os

import pytest
import torch
import triton

from ignition.preparation.triton import launch_triton, prepare_triton
from hyperflare_recipes.recipes.inkling2.attention import _retain_kv_kernel, _stage_kv_kernel
from hyperflare_recipes.recipes.inkling2.loop import (
    GateWeights,
    PreparedCrossPassRouter,
    PreparedMessageGate,
    PreparedSharedQKVNormCat,
)
from hyperflare_recipes.recipes.inkling2.pool import KV_DTYPE, PAGE_TOKENS
from hyperflare_recipes.recipes.inkling2.residual import PreparedResidualMix

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires assigned CUDA device",
)


def _reference_stage(rings, values, state_rows, starts, lengths, query_start_loc, *, window, pages):
    rows, capacity = starts.numel(), pages * PAGE_TOKENS
    ring = rings[state_rows.long()].view(KV_DTYPE).view(rows, -1, *values.shape[1:])
    first = (starts - window + 1).clamp_min(0)
    positions = first[:, None] + torch.arange(capacity, device=values.device)
    previous = ring[torch.arange(rows, device=values.device)[:, None], positions.remainder(ring.shape[1])].to(values.dtype)
    index = (query_start_loc[:-1, None] + positions - starts[:, None]).clamp(0, values.shape[0] - 1)
    staged = torch.where((positions < starts[:, None])[:, :, None, None], previous, values[index])
    out_lengths = torch.where(lengths > 0, starts - first + lengths, 0).int()
    valid = (lengths[:, None] > 0) & (positions < (starts + lengths)[:, None])
    staged = torch.where(valid[:, :, None, None], staged, 0)
    return staged.clamp(-448, 448).to(KV_DTYPE).view(rows * pages, PAGE_TOKENS, *values.shape[1:]), out_lengths


def _reference_retain(rings, values, state_rows, starts, counts, query_start_loc):
    out = rings.clone()
    rows = starts.numel()
    ring = out[state_rows.long()].view(KV_DTYPE).view(rows, -1, *values.shape[1:])
    capacity = ring.shape[1]
    first = starts + counts - capacity
    positions = first[:, None] + (torch.arange(capacity, device=values.device) - first[:, None]).remainder(capacity)
    indices = (query_start_loc[:-1, None] + positions - starts[:, None]).clamp(0, values.shape[0] - 1)
    current = values[indices].clamp(-448, 448)
    keep = ((positions < starts[:, None]) | (positions < 0))[:, :, None, None]
    retained = torch.where(keep, ring.to(values.dtype), current).to(KV_DTYPE)
    retained_u8 = retained.view(torch.uint8).view(rows, -1)
    for b in range(rows):
        if int(counts[b].item()) > 0:
            out[int(state_rows[b].item())].copy_(retained_u8[b])
    return out


@torch.no_grad()
@pytest.mark.parametrize("kv_heads", [8, 2])
@pytest.mark.parametrize("mode", ["prefill_short", "prefill_wrap", "verify_mixed"])
def test_fused_kv_ring_stage_and_retain_match_reference(kv_heads, mode):
    device = torch.device("cuda:0")
    torch.manual_seed(17)
    window, ring_len = 1024, 1024 + PAGE_TOKENS
    dim = 128
    row_elems = kv_heads * dim
    state_slots = 6
    key_rings = (torch.randn(state_slots, ring_len, kv_heads, dim, device=device) * 4.0).clamp(-448, 448).to(KV_DTYPE)
    val_rings = (torch.randn(state_slots, ring_len, kv_heads, dim, device=device) * 4.0).clamp(-448, 448).to(KV_DTYPE)
    key_rings_u8 = key_rings.view(torch.uint8).view(state_slots, -1).contiguous()
    val_rings_u8 = val_rings.view(torch.uint8).view(state_slots, -1).contiguous()

    if mode == "prefill_short":
        requests, max_q, tokens = 1, 512, 512
        state_rows = torch.tensor([2], dtype=torch.int32, device=device)
        starts = torch.tensor([0], dtype=torch.int32, device=device)
        lengths = torch.tensor([317], dtype=torch.int32, device=device)
        counts = lengths.clone()
        qstart = torch.tensor([0, 317], dtype=torch.int32, device=device)
    elif mode == "prefill_wrap":
        requests, max_q, tokens = 1, 2048, 2048
        state_rows = torch.tensor([4], dtype=torch.int32, device=device)
        starts = torch.tensor([1537], dtype=torch.int32, device=device)
        lengths = torch.tensor([1400], dtype=torch.int32, device=device)
        counts = lengths.clone()
        qstart = torch.tensor([0, 1400], dtype=torch.int32, device=device)
    else:
        requests, max_q, tokens = 4, 8, 32
        state_rows = torch.tensor([5, 1, 3, 0], dtype=torch.int32, device=device)
        # Includes start=0, wrap-around start, tail start, and an inactive row (length=0).
        starts = torch.tensor([0, 1021, 2300, 1500], dtype=torch.int32, device=device)
        lengths = torch.tensor([5, 8, 3, 0], dtype=torch.int32, device=device)
        counts = torch.tensor([3, 8, 1, 0], dtype=torch.int32, device=device)
        qstart = torch.tensor([0, 5, 13, 16, 16], dtype=torch.int32, device=device)

    keys = (torch.randn(tokens, kv_heads, dim, device=device, dtype=torch.bfloat16) * 20.0)
    vals = (torch.randn(tokens, kv_heads, dim, device=device, dtype=torch.bfloat16) * 20.0)
    # Include saturation values beyond +-448 to verify clamping matches reference.
    keys[0, 0, 0] = 600.0
    vals[0, 0, 1] = -600.0

    staged_pages = -(-(window - 1 + max_q) // PAGE_TOKENS)
    cap = staged_pages * PAGE_TOKENS
    out_k = torch.empty((requests * staged_pages, PAGE_TOKENS, kv_heads, dim), dtype=KV_DTYPE, device=device)
    out_v = torch.empty_like(out_k)
    out_len = torch.empty((requests,), dtype=torch.int32, device=device)
    row_stride = key_rings_u8.stride(0)
    stage_opts = dict(
        RING_LEN=ring_len, RING_ROW_STRIDE=row_stride, WINDOW=window, CAPACITY=cap,
        ROW_ELEMS=row_elems, MAX_TOKENS=tokens, BLOCK_E=triton.next_power_of_2(row_elems), num_warps=4,
    )
    stage_args = (key_rings_u8.view(KV_DTYPE), val_rings_u8.view(KV_DTYPE), keys, vals,
                  state_rows, starts, lengths, qstart, out_k, out_v, out_len)
    stage_compiled = prepare_triton(_stage_kv_kernel, stage_args, (requests, cap, 1), **stage_opts).compiled
    launch_triton(_stage_kv_kernel, stage_compiled, stage_args, (requests, cap, 1), **stage_opts)

    ref_k, ref_len = _reference_stage(key_rings_u8, keys, state_rows, starts, lengths, qstart,
                                      window=window, pages=staged_pages)
    ref_v, _ = _reference_stage(val_rings_u8, vals, state_rows, starts, lengths, qstart,
                                window=window, pages=staged_pages)
    assert torch.equal(out_len, ref_len)
    assert torch.equal(out_k.view(torch.uint8), ref_k.view(torch.uint8))
    assert torch.equal(out_v.view(torch.uint8), ref_v.view(torch.uint8))

    ref_retained_k = _reference_retain(key_rings_u8, keys, state_rows, starts, counts, qstart)
    ref_retained_v = _reference_retain(val_rings_u8, vals, state_rows, starts, counts, qstart)
    retain_w = min(max_q, ring_len)
    retain_opts = dict(
        RING_LEN=ring_len, RING_ROW_STRIDE=row_stride, ROW_ELEMS=row_elems, MAX_TOKENS=tokens,
        BLOCK_E=triton.next_power_of_2(row_elems), num_warps=4,
    )
    retain_args = (key_rings_u8.view(KV_DTYPE), val_rings_u8.view(KV_DTYPE), keys, vals,
                   state_rows, starts, counts, qstart)
    retain_compiled = prepare_triton(_retain_kv_kernel, retain_args, (requests, retain_w, 1), **retain_opts).compiled
    launch_triton(_retain_kv_kernel, retain_compiled, retain_args, (requests, retain_w, 1), **retain_opts)
    assert torch.equal(key_rings_u8, ref_retained_k)
    assert torch.equal(val_rings_u8, ref_retained_v)


@torch.no_grad()
@pytest.mark.parametrize("hidden", [2304, 4608])
@pytest.mark.parametrize("sources,streams,addend", [(1, 1, False), (4, 3, False), (7, 1, True), (9, 3, True)])
@pytest.mark.parametrize("tokens", [1, 17, 512])
def test_fused_mixture_norm_matches_pytorch_reference(hidden, sources, streams, addend, tokens):
    device = torch.device("cuda:0")
    torch.manual_seed(29)
    eps = 1e-6
    inputs = tuple(torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) for _ in range(sources))
    add = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) if addend else None
    queries = torch.randn(streams, hidden, device=device, dtype=torch.float32) * (hidden ** -0.5)
    norms = tuple(torch.randn(hidden, device=device, dtype=torch.bfloat16) for _ in range(streams))

    op = PreparedResidualMix(inputs[0], queries, norms, sources=sources, eps=eps,
                             session=None, name="ref", addend=addend)
    actual_mix = op.mix(inputs, addend=add)
    actual_out = op(inputs, addend=add)

    # Pure-PyTorch FP32 reference.
    vals = [x.float() for x in inputs]
    if addend:
        vals[-1] = vals[-1] + add.float()
    normed = [v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) for v in vals]
    stacked = torch.stack(normed, dim=1)  # [T, S, H]
    for s in range(streams):
        scores = torch.einsum("tsh,h->ts", stacked, queries[s])
        probs = torch.softmax(scores, dim=-1)
        ref_mix = torch.einsum("ts,tsh->th", probs, stacked)
        ref_norm = ref_mix * torch.rsqrt(ref_mix.square().mean(-1, keepdim=True) + eps)
        ref_out = (ref_norm * norms[s].float()).to(torch.bfloat16)
        assert (actual_mix[s] - ref_mix).abs().max().item() <= 1e-4
        torch.testing.assert_close(actual_out[s], ref_out, rtol=1e-2, atol=1e-2)
        rel_l2 = (actual_out[s].float() - ref_out.float()).norm() / ref_out.float().norm()
        assert rel_l2.item() <= 1e-4
        cos = torch.nn.functional.cosine_similarity(actual_out[s].float().flatten(), ref_out.float().flatten(), dim=0)
        assert cos.item() >= 0.99999


def _reference_controller(dest, s0, s1, query, down, gain, forget, norm, eps):
    """Exact two-source attention-residual message + vLLM Inkling2MessageController reference."""
    n0 = s0.float() * torch.rsqrt(s0.float().square().mean(-1, keepdim=True) + eps)
    n1 = s1.float() * torch.rsqrt(s1.float().square().mean(-1, keepdim=True) + eps)
    scores = torch.stack((n0 @ query, n1 @ query), dim=-1)
    probs = torch.softmax(scores, dim=-1)
    message = probs[:, :1] * n0 + probs[:, 1:] * n1

    pair = torch.stack((dest, message), dim=1)
    pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
    hidden = torch.nn.functional.linear(pair.to(torch.bfloat16).flatten(1), down)
    f = torch.nn.functional.linear(hidden, forget).float()
    g = torch.nn.functional.linear(hidden, gain).float()
    blend = (1.0 - torch.tanh(f)) * dest + torch.tanh(g) * message
    normalized = blend * torch.rsqrt(blend.square().mean(-1, keepdim=True) + eps)
    return (normalized * norm.float()).to(torch.bfloat16)


@torch.no_grad()
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 64, 512])
def test_cross_pass_router_and_fused_message_gates_match_reference(tokens):
    device = torch.device("cuda:0")
    torch.manual_seed(43)
    hidden, rank, eps = 2304, 64, 1e-6
    num_streams = 4

    s0 = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    s1 = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    queries = torch.randn(num_streams, hidden, device=device, dtype=torch.float32) * (hidden ** -0.5)
    dests = [torch.randn(tokens, hidden, device=device, dtype=torch.float32) for _ in range(num_streams)]
    shared_dest = dests[0]
    shared_norm = torch.randn(hidden, device=device, dtype=torch.bfloat16)
    norms = [torch.randn(hidden, device=device, dtype=torch.bfloat16) for _ in range(num_streams)]

    raw_weights = []
    gates = []
    for idx in range(num_streams):
        down = (torch.randn(rank, 2 * hidden, device=device) * 0.02).to(torch.bfloat16)
        gain = (torch.randn(hidden, rank, device=device) * 0.02).to(torch.bfloat16)
        forget = (torch.randn(hidden, rank, device=device) * 0.02).to(torch.bfloat16)
        raw_weights.append((down, gain, forget))
        gw = GateWeights(
            down=down,
            forget_gain=torch.cat((forget, gain), dim=0).contiguous(),
            forget_t=forget.t().contiguous(),
            gain_t=gain.t().contiguous(),
        )
        gates.append(PreparedMessageGate(
            gw, norms[idx], stream_idx=idx, num_streams=num_streams,
            tokens=tokens, hidden=hidden, eps=eps,
        ))

    router = PreparedCrossPassRouter(queries, tokens=tokens, hidden=hidden, eps=eps)
    n0, n1, probs = router(s0, s1)

    # Independent per-stream path (global loop layers and MLP controller; exercises decode and tiled prefill paths).
    for idx in range(num_streams):
        actual = gates[idx](dests[idx], n0, n1, probs)
        ref = _reference_controller(dests[idx], s0, s1, queries[idx], *raw_weights[idx], norms[idx], eps)
        cos = torch.nn.functional.cosine_similarity(actual.float().flatten(), ref.float().flatten(), dim=0)
        rel_l2 = (actual.float() - ref.float()).norm() / ref.float().norm()
        assert cos.item() >= 0.9999, f"tokens={tokens} stream={idx} cos={cos.item()}"
        assert rel_l2.item() <= 1.5e-2, f"tokens={tokens} stream={idx} rel_l2={rel_l2.item()}"

    # Shared-hidden Q/K/V path (sliding loop layers; exercises _norm_cat_qkv_shared_hidden and _gate_mix_rms_norm_qkv_shared).
    qkv_weights = tuple(
        GateWeights(
            down=raw_weights[idx][0],
            forget_gain=torch.cat((raw_weights[idx][2], raw_weights[idx][1]), dim=0).contiguous(),
            forget_t=raw_weights[idx][2].t().contiguous(),
            gain_t=raw_weights[idx][1].t().contiguous(),
        )
        for idx in range(3)
    )
    shared_gates = [
        PreparedMessageGate(
            qkv_weights[idx],
            shared_norm, stream_idx=idx, num_streams=num_streams,
            tokens=tokens, hidden=hidden, eps=eps,
        )
        for idx in range(3)
    ]
    qkv_cat = PreparedSharedQKVNormCat(shared_norm, base_stream=0, num_streams=num_streams,
                                       tokens=tokens, hidden=hidden, eps=eps, device=device,
                                       gates=qkv_weights)
    if qkv_cat.use_fused_decode:
        outs = qkv_cat.apply_fused_decode(shared_dest, n0, n1, probs)
    else:
        cat_q, cat_k, cat_v = qkv_cat(shared_dest, n0, n1, probs)
        if qkv_cat.use_fused_mix:
            outs = qkv_cat.mix_qkv(shared_dest, n0, n1, probs,
                                   shared_gates[0].compute_fg(cat_q),
                                   shared_gates[1].compute_fg(cat_k),
                                   shared_gates[2].compute_fg(cat_v))
        else:
            outs = (
                shared_gates[0].apply_with_cat(shared_dest, cat_q, n0, n1, probs),
                shared_gates[1].apply_with_cat(shared_dest, cat_k, n0, n1, probs),
                shared_gates[2].apply_with_cat(shared_dest, cat_v, n0, n1, probs),
            )
    for idx in range(3):
        ref = _reference_controller(shared_dest, s0, s1, queries[idx], *raw_weights[idx], shared_norm, eps)
        cos = torch.nn.functional.cosine_similarity(outs[idx].float().flatten(), ref.float().flatten(), dim=0)
        rel_l2 = (outs[idx].float() - ref.float()).norm() / ref.float().norm()
        assert cos.item() >= 0.9999, f"shared tokens={tokens} stream={idx} cos={cos.item()}"
        assert rel_l2.item() <= 1.5e-2, f"shared tokens={tokens} stream={idx} rel_l2={rel_l2.item()}"

    if qkv_cat.use_fused_decode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_outs = qkv_cat.apply_fused_decode(shared_dest, n0, n1, probs)
            g_mlp = gates[3](dests[3], n0, n1, probs)
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        for idx in range(3):
            assert torch.equal(g_outs[idx], outs[idx])
        assert torch.equal(g_mlp, gates[3](dests[3], n0, n1, probs))
        graph.reset()
