# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 operator integration and capability-gating tests for Inkling 2."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components import (
    inkling2_relative_decode_ops as decode_ops,
    inkling2_relative_prefill_ops as prefill_ops,
    inkling2_router_ops as router_ops,
    inkling2_vec_residual as vec_res_ops,
)
from hyperflare_recipes.components.native_operator import is_sm120


# ---------------------------------------------------------------------------
# CPU capability-gating and fallback tests (always run, no GPU required).


def test_sm120_capability_gating_rejects_non_sm120(monkeypatch):
    assert not is_sm120(torch.device("cpu"))

    class FakeTensor:
        def __init__(self, shape, dtype, stride=None, ptr=256):
            self.shape = shape
            self.dtype = dtype
            self._stride = stride or (shape[-1], 1) if len(shape) == 2 else (1,)
            self._ptr = ptr
            self.device = torch.device("cuda:0")
            self.ndim = len(shape)

        def stride(self, dim=None):
            return self._stride if dim is None else self._stride[dim]

        def data_ptr(self):
            return self._ptr

        def element_size(self):
            return 2 if self.dtype == torch.bfloat16 else (4 if self.dtype == torch.float32 else 1)

        def is_contiguous(self):
            return True

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=None: (12, 1))
    assert not is_sm120(torch.device("cuda:0"))

    q = FakeTensor((4, 24, 128), torch.bfloat16, stride=(24 * 128, 128, 1))
    kc = FakeTensor((2, 1152, 4, 128), torch.float8_e4m3fn, stride=(1152 * 4 * 128, 4 * 128, 128, 1))
    bias = FakeTensor((4, 24, 1024), torch.bfloat16, stride=(24 * 1024, 1024, 1))
    sink = FakeTensor((4, 24), torch.bfloat16, stride=(24, 1))
    assert not decode_ops.supports_relative_decode(
        query=q, key_cache=kc, value_cache=kc, is_window=True, commit_ring=True,
        window=1024, rel_extent=1024, max_q_per_req=1,
    )
    assert not prefill_ops.supports_relative_prefill(
        query=q, key_cache=kc, value_cache=kc, relative_logits=bias,
        gate=q, sink=sink, gated=q, window=-1,
    )
    src = FakeTensor((4, 2304), torch.bfloat16, stride=(2304, 1))
    w = FakeTensor((128, 2304), torch.bfloat16, stride=(2304, 1))
    b = FakeTensor((128,), torch.float32, stride=(1,))
    logits = FakeTensor((4, 128), torch.float32, stride=(128, 1))
    assert not router_ops.splitk_router_eligible(src, w, b, logits, top_k=8)
    assert not vec_res_ops.supports_vec_residual_mix(4, 2304, 4, torch.device("cuda:0"))


def test_sm120_missing_b12x_symbol_falls_back_cleanly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=None: (12, 0))
    monkeypatch.setattr(decode_ops, "_relative_decode_module", lambda: None)
    monkeypatch.setattr(prefill_ops, "_relative_prefill_module", lambda: None)
    monkeypatch.setattr(router_ops, "_has_splitk_router_kernel", lambda: False)
    monkeypatch.setattr(vec_res_ops, "_has_vec_residual_kernel", lambda: False)

    fake = SimpleNamespace(
        device=torch.device("cuda:0"),
        shape=(4, 2304),
        dtype=torch.bfloat16,
        ndim=2,
        stride=lambda d=None: (2304, 1) if d is None else (2304, 1)[d],
        data_ptr=lambda: 256,
        element_size=lambda: 2,
        is_contiguous=lambda: True,
    )
    assert is_sm120(torch.device("cuda:0"))
    assert not decode_ops.supports_relative_decode(
        query=fake, key_cache=fake, value_cache=fake, is_window=True, commit_ring=True,
        window=1024, rel_extent=1024, max_q_per_req=1,
    )
    assert not prefill_ops.supports_relative_prefill(
        query=fake, key_cache=fake, value_cache=fake, relative_logits=fake,
        gate=fake, sink=fake, gated=fake, window=-1,
    )
    assert not router_ops.splitk_router_eligible(fake, fake, fake, fake, top_k=8)
    assert not vec_res_ops.supports_vec_residual_mix(4, 2304, 4, torch.device("cuda:0"))


# ---------------------------------------------------------------------------
# SM120 CUDA operator integration tests.


def _require_sm120_cuda():
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires assigned CUDA device (FLARE_RUN_CUDA_TESTS=1)")
    if not torch.cuda.is_available() or not is_sm120(torch.device("cuda:0")):
        pytest.skip("requires SM120 CUDA device")


def _make_session(device, arena=None):
    from ignition.preparation.tuning import RecordedPreparationSession

    session = RecordedPreparationSession(
        device=device,
        checkpoint="test-sm120",
        workload_schema="test",
        numerical_policy="test",
        autotune_effort="off",
        compile_workers=2,
    )
    if arena is not None:
        session.register_scratch(arena)
    return session


@torch.no_grad()
@pytest.mark.parametrize("tokens,valid_count", [(1, 1), (4, 4), (8, 5), (8, 0)])
def test_splitk_router_matches_prepared_router_and_masks_padding(tokens, valid_count):
    _require_sm120_cuda()
    from hyperflare_recipes.components.linear import UnquantizedProjection
    from hyperflare_recipes.components.moe import PreparedRouter, RouterScratch

    device = torch.device("cuda:0")
    torch.manual_seed(101 + tokens * 13 + valid_count)
    hidden, experts, top_k, route_scale = 2304, 128, 8, 2.5

    source = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device) * 0.25
    weight = torch.randn((experts, hidden), dtype=torch.bfloat16, device=device) * 0.25
    bias = torch.randn((experts,), dtype=torch.float32, device=device) * 0.1
    valid_tokens = torch.tensor([valid_count], dtype=torch.int32, device=device)
    logits_ex = torch.empty((tokens, experts), dtype=torch.float32, device=device)
    routes_ex = RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k)

    assert router_ops.splitk_router_eligible(source, weight, bias, logits_ex, top_k)

    with _make_session(device) as session:
        splitk = router_ops.PreparedInkling2SplitKRouter(
            source, weight, bias, logits_ex, routes_ex.ids, routes_ex.weights,
            session=session, name="splitk", route_scale=route_scale, num_valid_tokens=valid_tokens,
        ).functional(session=session)
        ref_proj = UnquantizedProjection(
            source, weight, torch.empty_like(logits_ex), session=session, name="ref_proj",
        ).functional(session=session)
        ref_router = PreparedRouter(
            logits_ex, RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k),
            session=session, name="ref_router", score_func="sigmoid", renormalize=True,
            routed_scaling_factor=route_scale, correction_bias=bias, num_valid_tokens=valid_tokens,
        ).functional(session=session)
        session.tune()

        actual = splitk(source, num_valid_tokens=valid_tokens)
        expected = ref_router(ref_proj(source), num_valid_tokens=valid_tokens)

    if valid_count > 0:
        assert torch.equal(actual.ids[:valid_count], expected.ids[:valid_count])
        torch.testing.assert_close(actual.weights[:valid_count], expected.weights[:valid_count], rtol=1e-4, atol=1e-4)
    if valid_count < tokens:
        assert (actual.ids[valid_count:] == -1).all()
        assert (actual.weights[valid_count:] == 0.0).all()
        assert torch.equal(actual.ids[valid_count:], expected.ids[valid_count:])
        assert torch.equal(actual.weights[valid_count:], expected.weights[valid_count:])


@torch.no_grad()
@pytest.mark.parametrize("tokens", [1, 4, 8, 32])
@pytest.mark.parametrize("sources,streams,addend", [(1, 1, False), (2, 3, False), (5, 1, True), (7, 3, True)])
def test_vec_residual_mix_via_prepared_residual_mix_matches_triton(monkeypatch, tokens, sources, streams, addend):
    _require_sm120_cuda()
    from hyperflare_recipes.recipes.inkling2.residual import PreparedResidualMix

    device = torch.device("cuda:0")
    torch.manual_seed(211 + tokens + sources * 7)
    hidden, eps = 2304, 1e-6
    inputs = tuple(torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device) for _ in range(sources))
    add = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device) if addend else None
    queries = torch.randn((streams, hidden), dtype=torch.float32, device=device) * (hidden ** -0.5)
    norms = tuple(torch.randn((hidden,), dtype=torch.bfloat16, device=device) for _ in range(streams))

    with _make_session(device) as session:
        sm120_op = PreparedResidualMix(
            inputs[0], queries, norms, sources=sources, eps=eps,
            session=session, name="sm120_mix", addend=addend,
        )
        assert sm120_op._vec_mix is not None
        session.tune()
        actual = sm120_op(inputs, addend=add)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=None: (12, 1))
    ref_op = PreparedResidualMix(
        inputs[0], queries, norms, sources=sources, eps=eps,
        session=None, name="triton_mix", addend=addend,
    )
    assert ref_op._vec_mix is None
    expected = ref_op(inputs, addend=add)

    for s in range(streams):
        torch.testing.assert_close(actual[s], expected[s], rtol=1e-2, atol=1e-2)
        rel_l2 = (actual[s].float() - expected[s].float()).norm() / expected[s].float().norm()
        assert rel_l2.item() <= 1e-3


@torch.no_grad()
@pytest.mark.parametrize("tokens", [1, 4, 8])
def test_small_k16_gemm_via_unquantized_projection(tokens):
    _require_sm120_cuda()
    from b12x.gemm import bf16_gemv
    from hyperflare_recipes.components.linear import UnquantizedProjection

    device = torch.device("cuda:0")
    torch.manual_seed(307 + tokens)
    rows = tokens * 32
    source = torch.randn((rows, 16), dtype=torch.bfloat16, device=device) * 0.5
    weight = torch.randn((1024, 16), dtype=torch.bfloat16, device=device) * 0.5
    out_ex = torch.empty((rows, 1024), dtype=torch.bfloat16, device=device)

    query = bf16_gemv.query_from_call(source, weight, out=out_ex)
    assert bf16_gemv._tuning._tc_eligible(query)

    with _make_session(device) as session:
        proj = UnquantizedProjection(source, weight, out_ex, session=session, name="small_k16").functional(session=session)
        session.tune()
        actual = proj(source)

    expected = (source.float() @ weight.float().T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _make_attention_fixtures(device, *, local: bool, heads: int = 24, kv_heads: int = 4, dim: int = 128,
                             hidden: int = 2304, window: int = 1024, width: int = 4, extent: int = 1024, seed: int = 7):
    from hyperflare_recipes.recipes.inkling2.attention import AttentionWeights, Columns

    torch.manual_seed(seed)
    text = SimpleNamespace(
        hidden=hidden,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=dim,
        window=window,
        convolution_width=width,
        eps=1e-6,
        qk_clip=6.0,
        relative_clip=6.0,
        log_scaling_floor=1024,
        log_scaling_alpha=0.1,
        layers=2,
        global_layers=(1,),
        local=lambda layer: layer == 0,
    )
    cols = Columns.of(local, heads, kv_heads, dim, width, separate_inputs=False)
    projections = tuple(
        torch.randn((w, hidden), dtype=torch.bfloat16, device=device) * 0.05
        for w in cols.widths()
    )
    rel_proj = torch.randn((16, extent), dtype=torch.bfloat16, device=device) * 0.05
    if local:
        k_conv = torch.randn((kv_heads * dim, 1, width), dtype=torch.bfloat16, device=device) * 0.05
        v_conv = torch.randn((kv_heads * dim, 1, width), dtype=torch.bfloat16, device=device) * 0.05
    else:
        k_conv = torch.randn((width, kv_heads * dim), dtype=torch.bfloat16, device=device) * 0.05
        v_conv = torch.randn((width, kv_heads * dim), dtype=torch.bfloat16, device=device) * 0.05
    weights = AttentionWeights(
        projections=projections,
        query_clip_scale=torch.ones((heads, dim), dtype=torch.bfloat16, device=device),
        relative_clip_scale=torch.ones((heads, 16), dtype=torch.bfloat16, device=device),
        relative_projection=rel_proj,
        relative_rows=rel_proj.t().contiguous(),
        key_convolution=k_conv,
        value_convolution=v_conv,
        output=torch.randn((hidden, heads * dim), dtype=torch.bfloat16, device=device) * 0.05,
    )
    return text, weights


def _clone_pool_state(src_pool, dst_pool):
    dst_pool.backing.copy_(src_pool.backing)
    for name, src_pages in src_pool.pages.items():
        dst_pool.pages[name].copy_(src_pages)


@torch.no_grad()
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("mode", ["decode", "verify"])
def test_prepared_attention_sm120_decode_and_verify_match_reference(monkeypatch, local, mode):
    _require_sm120_cuda()
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.inkling2.attention import LayerCache, PreparedAttention
    from hyperflare_recipes.recipes.inkling2.pool import ContinuationPool, KV_DTYPE, PAGE_TOKENS, RankGeometry
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    layer = 0 if local else 1
    text, weights = _make_attention_fixtures(device, local=local, seed=401 + int(local) * 17 + int(mode == "verify") * 31)
    geometry = RankGeometry.of(text, 1)

    requests = 2
    max_q = 1 if mode == "decode" else 4
    tokens = requests * max_q
    blocks = 16
    pool_sm120 = ContinuationPool.allocate(text, geometry, slots=6, pages=32, device=device)
    pool_ref = ContinuationPool.allocate(text, geometry, slots=6, pages=32, device=device)

    # Populate prior history (including ring wrap-around past 1152 for local layers).
    torch.manual_seed(509)
    if local:
        for kind in ("key", "value"):
            ring = pool_sm120.reserved_region(f"layer{layer}.{kind}")
            ring.copy_((torch.randn(ring.shape, device=device) * 0.5).clamp(-448, 448).to(KV_DTYPE))
    else:
        for kind in ("key", "value"):
            pages = pool_sm120.cache(layer, kind)
            pages.copy_((torch.randn(pages.shape, device=device) * 0.5).clamp(-448, 448).to(KV_DTYPE))
    for kind in ("key_convolution", "value_convolution"):
        hist = pool_sm120.history(layer, kind)
        hist.copy_(torch.randn(hist.shape, dtype=torch.bfloat16, device=device) * 0.25)
    _clone_pool_state(pool_sm120, pool_ref)

    meta = Metadata.allocate(tokens, requests, blocks, device, max_query_tokens=max_q)
    starts = torch.tensor([1300 if local else 140, 20], dtype=torch.int64, device=device)
    lengths = torch.tensor([1, 0] if mode == "decode" else [3, 2], dtype=torch.int32, device=device)
    state_rows = torch.tensor([2, 4], dtype=torch.int64, device=device)
    page_table = torch.full((requests, blocks), -1, dtype=torch.int32, device=device)
    page_table[0, :12] = torch.arange(0, 12, dtype=torch.int32, device=device)
    page_table[1, :4] = torch.arange(12, 16, dtype=torch.int32, device=device)
    tok_ids = torch.zeros((requests, max_q), dtype=torch.int64, device=device)
    meta.set(starts, lengths, state_rows, page_table, tok_ids, torch.tensor([2], dtype=torch.int32, device=device))

    streams = (torch.randn((tokens, text.hidden), dtype=torch.bfloat16, device=device) * 0.5,)
    if not local:
        streams = (streams[0], streams[0], streams[0])

    arena_sm120 = ScratchArena(device)
    with _make_session(device, arena_sm120) as session:
        attn_sm120 = PreparedAttention(
            layer, weights, text, LayerCache.of(pool_sm120, layer), meta,
            mode=mode, tokens=tokens, session=session, workspace=arena_sm120, name="sm120_attn",
        )
        assert attn_sm120.use_sm120_decode
        assert attn_sm120.gate is None
        assert attn_sm120.writer is None
        assert attn_sm120.relative_projection is not None
        assert (attn_sm120.tau_op is not None) == (not local)
        session.tune()
        actual_gated, actual_partial, actual_vals = attn_sm120.attend(streams, meta)
        if mode == "verify":
            accept = torch.tensor([2, 1], dtype=torch.int32, device=device)
            attn_sm120.commit(actual_vals, accept, meta)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=None: (12, 1))
    arena_ref = ScratchArena(device)
    with _make_session(device, arena_ref) as session:
        attn_ref = PreparedAttention(
            layer, weights, text, LayerCache.of(pool_ref, layer), meta,
            mode=mode, tokens=tokens, session=session, workspace=arena_ref, name="ref_attn",
        )
        assert not attn_ref.use_sm120_decode
        assert attn_ref.gate is not None
        session.tune()
        ref_gated, ref_partial, ref_vals = attn_ref.attend(streams, meta)
        if mode == "verify":
            accept = torch.tensor([2, 1], dtype=torch.int32, device=device)
            attn_ref.commit(ref_vals, accept, meta)

    live = int(lengths.sum().item())
    torch.testing.assert_close(actual_gated[:live], ref_gated[:live], rtol=2e-2, atol=2e-2)
    rel_l2 = (actual_partial[:live] - ref_partial[:live]).norm() / ref_partial[:live].norm()
    assert rel_l2.item() <= 1e-2
    if live < tokens:
        assert torch.count_nonzero(actual_gated[live:]) == 0
    if local:
        for kind in ("key", "value"):
            assert torch.equal(
                pool_sm120.ring_rows(layer, kind),
                pool_ref.ring_rows(layer, kind),
            )
    else:
        for kind in ("key", "value"):
            assert torch.equal(
                pool_sm120.cache(layer, kind).view(torch.uint8),
                pool_ref.cache(layer, kind).view(torch.uint8),
            )


@torch.no_grad()
@pytest.mark.parametrize(
    "local,start_pos",
    [(True, 0), (True, 256), (True, 1300), (False, 0), (False, 128)],
)
def test_prepared_attention_sm120_prefill_matches_reference(monkeypatch, local, start_pos):
    _require_sm120_cuda()
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.inkling2.attention import LayerCache, PreparedAttention
    from hyperflare_recipes.recipes.inkling2.pool import ContinuationPool, KV_DTYPE, RankGeometry
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    layer = 0 if local else 1
    text, weights = _make_attention_fixtures(device, local=local, seed=607 + int(local) * 19 + start_pos)
    geometry = RankGeometry.of(text, 1)

    requests, tokens, blocks = 1, 256, 16
    live_len = 197
    pool_sm120 = ContinuationPool.allocate(text, geometry, slots=4, pages=32, device=device)
    pool_ref = ContinuationPool.allocate(text, geometry, slots=4, pages=32, device=device)

    torch.manual_seed(701)
    if local:
        for kind in ("key", "value"):
            ring = pool_sm120.reserved_region(f"layer{layer}.{kind}")
            ring.copy_((torch.randn(ring.shape, device=device) * 0.5).clamp(-448, 448).to(KV_DTYPE))
    else:
        for kind in ("key", "value"):
            pages = pool_sm120.cache(layer, kind)
            pages.copy_((torch.randn(pages.shape, device=device) * 0.5).clamp(-448, 448).to(KV_DTYPE))
    for kind in ("key_convolution", "value_convolution"):
        hist = pool_sm120.history(layer, kind)
        hist.copy_(torch.randn(hist.shape, dtype=torch.bfloat16, device=device) * 0.25)
    _clone_pool_state(pool_sm120, pool_ref)

    meta = Metadata.allocate(tokens, requests, blocks, device, max_query_tokens=tokens)
    starts = torch.tensor([start_pos], dtype=torch.int64, device=device)
    lengths = torch.tensor([live_len], dtype=torch.int32, device=device)
    state_rows = torch.tensor([1], dtype=torch.int64, device=device)
    page_table = torch.arange(blocks, dtype=torch.int32, device=device).unsqueeze(0)
    tok_ids = torch.zeros((requests, tokens), dtype=torch.int64, device=device)
    meta.set(starts, lengths, state_rows, page_table, tok_ids, torch.tensor([1], dtype=torch.int32, device=device))

    streams = (torch.randn((tokens, text.hidden), dtype=torch.bfloat16, device=device) * 0.5,)
    if not local:
        streams = (streams[0], streams[0], streams[0])

    arena_sm120 = ScratchArena(device)
    with _make_session(device, arena_sm120) as session:
        attn_sm120 = PreparedAttention(
            layer, weights, text, LayerCache.of(pool_sm120, layer), meta,
            mode="prefill", tokens=tokens, session=session, workspace=arena_sm120, name="sm120_prefill",
        )
        assert attn_sm120.use_sm120_prefill
        assert attn_sm120.gate is None
        assert attn_sm120.relative_projection is not None
        session.tune()
        actual_gated, actual_partial, _ = attn_sm120.attend(streams, meta)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=None: (12, 1))
    arena_ref = ScratchArena(device)
    with _make_session(device, arena_ref) as session:
        attn_ref = PreparedAttention(
            layer, weights, text, LayerCache.of(pool_ref, layer), meta,
            mode="prefill", tokens=tokens, session=session, workspace=arena_ref, name="ref_prefill",
        )
        assert not attn_ref.use_sm120_prefill
        assert attn_ref.gate is not None
        session.tune()
        ref_gated, ref_partial, _ = attn_ref.attend(streams, meta)

    torch.testing.assert_close(actual_gated[:live_len], ref_gated[:live_len], rtol=2e-2, atol=2e-2)
    rel_l2 = (actual_partial[:live_len] - ref_partial[:live_len]).norm() / ref_partial[:live_len].norm()
    assert rel_l2.item() <= 1e-2
    if live_len < tokens:
        assert torch.count_nonzero(actual_gated[live_len:]) == 0
    if local:
        for kind in ("key", "value"):
            assert torch.equal(
                pool_sm120.ring_rows(layer, kind),
                pool_ref.ring_rows(layer, kind),
            )
    else:
        for kind in ("key", "value"):
            assert torch.equal(
                pool_sm120.cache(layer, kind).view(torch.uint8),
                pool_ref.cache(layer, kind).view(torch.uint8),
            )


@torch.no_grad()
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("mode", ["decode", "verify", "prefill"])
def test_sm121_parity_zero_extra_examples_and_single_router_scratch(monkeypatch, local, mode):
    """Monkeypatched SM121 (12, 1) must bypass every SM120 path with zero extra examples or scratch."""
    _require_sm120_cuda()
    from hyperflare_recipes.components.moe import RouterScratch
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.inkling2.attention import LayerCache, PreparedAttention
    from hyperflare_recipes.recipes.inkling2.mini_decoder import PreparedLoopLayer
    from hyperflare_recipes.recipes.inkling2.pool import ContinuationPool, RankGeometry
    from hyperflare_recipes.recipes.inkling2.residual import PreparedResidualMix
    from ignition.preparation.examples import declaration_tensor
    from ignition.preparation.scratch import ScratchArena

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=None: (12, 1))
    device = torch.device("cuda:0")
    layer = 0 if local else 1
    text, weights = _make_attention_fixtures(device, local=local, seed=811 + int(local) * 7)
    geometry = RankGeometry.of(text, 1)
    requests = 1 if mode == "prefill" else 2
    max_q = 64 if mode == "prefill" else (1 if mode == "decode" else 4)
    tokens = requests * max_q
    pool = ContinuationPool.allocate(text, geometry, slots=4, pages=16, device=device)
    meta = Metadata.allocate(tokens, requests, 8, device, max_query_tokens=max_q)

    examples = {}
    arena = ScratchArena(device)
    with _make_session(device, arena) as session:
        attn = PreparedAttention(
            layer, weights, text, LayerCache.of(pool, layer), meta,
            mode=mode, tokens=tokens, session=session, workspace=arena,
            name="sm121_attn", examples=examples,
        )
        assert not attn.sm120
        assert not attn.use_sm120_decode
        assert not attn.use_sm120_prefill
        assert attn.tau_op is None
        assert attn.relative_projection is None
        assert attn.gate is not None

        sm120_only_roles = {
            "inkling.attention.tau",
            "inkling.attention.sink_bf16",
            "inkling.attention.relative_source",
            "inkling.attention.verify_staged_key",
            "inkling.attention.verify_staged_value",
            "inkling.attention.sm120_staged_key",
            "inkling.attention.sm120_staged_value",
        }
        declared_roles = {key[0] for key in examples if isinstance(key, tuple) and key}
        assert declared_roles.isdisjoint(sm120_only_roles)
        if mode == "prefill":
            assert "inkling.attention.relative_bias" not in declared_roles
            assert "inkling.attention.relative_factor" in declared_roles
        recorded_names = list(session._recorded_requests)
        gate_idx = recorded_names.index("sm121_attn.gate")
        attn_idx = recorded_names.index("sm121_attn.attention")
        assert gate_idx < attn_idx
        if "sm121_attn.cache_write" in recorded_names:
            assert gate_idx < recorded_names.index("sm121_attn.cache_write") < attn_idx

        # Residual mix bypasses VecResidualMixKernel on SM121.
        src = torch.zeros((min(tokens, 8), text.hidden), dtype=torch.bfloat16, device=device)
        queries = torch.zeros((1, text.hidden), dtype=torch.float32, device=device)
        norms = (torch.ones((text.hidden,), dtype=torch.bfloat16, device=device),)
        mix = PreparedResidualMix(src, queries, norms, sources=2, eps=1e-6, session=session, name="sm121_mix")
        assert mix._vec_mix is None
        assert mix.fused_compiled is not None

        # PreparedLoopLayer._build_router allocates exactly one RouterScratch on SM121.
        alloc_calls = 0
        orig_allocate = RouterScratch.allocate

        def counted_allocate(*args, **kwargs):
            nonlocal alloc_calls
            alloc_calls += 1
            return orig_allocate(*args, **kwargs)

        monkeypatch.setattr(RouterScratch, "allocate", classmethod(lambda cls, *a, **kw: counted_allocate(*a, **kw)))
        dummy_layer = object.__new__(PreparedLoopLayer)
        router_text = SimpleNamespace(experts=128, experts_per_token=8, route_scale=2.5)
        router_weights = SimpleNamespace(
            router=torch.zeros((128, text.hidden), dtype=torch.bfloat16, device=device),
            correction=torch.zeros((128,), dtype=torch.float32, device=device),
        )
        router_examples = {}

        def router_example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(router_examples, "inkling.mlp." + role, shape, dtype, device)

        dummy_layer._build_router(
            src, router_weights, router_text, meta,
            tokens=src.shape[0], session=session, name="sm121_layer", example=router_example,
        )
        assert dummy_layer.splitk_router is None
        assert dummy_layer.router is not None
        assert dummy_layer.router_projection is not None
        assert alloc_calls == 1

