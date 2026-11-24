# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA tests for register_tc_gemv_rope and SplitLaggedMHC overlap."""

from __future__ import annotations

import pytest
import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x.gemm import blockscaled
from b12x.norm import mhc
from b12x.norm.mhc import _impl as mhc_impl
from b12x.norm.mhc._split_lagged import compile_cluster_post_y_norm
from b12x.preparation import PreparationSession, PreparedCall

from hyperflare_recipes.components.mhc import MHCScratch, MHCWeights
from hyperflare_recipes.components.mhc_overlap import SplitLaggedMHC
from hyperflare_recipes.components.projection_overlap import (
    _TC_GEMV_ROPE_RUNNERS,
    _run_mxfp8_with_rope,
    register_tc_gemv_rope,
)
from hyperflare_recipes.components.rotary import PreparedRotaryAngles, _rotate_grouped
from ignition.preparation.scratch import ScratchArena


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120+ required")


@pytest.mark.parametrize("head_dim", [512, 128])
@pytest.mark.parametrize("tokens,split_k", [(1, 1), (4, 1), (8, 1), (8, 2)])
def test_register_tc_gemv_rope_matches_unfused(head_dim: int, tokens: int, split_k: int):
    _require_sm120()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(101 + head_dim + tokens + split_k)
    in_features, out_features = 1280, 4096
    w_bf16 = (
        torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
        * 0.05
    )
    vals = torch.empty((out_features, in_features), dtype=torch.uint8, device=device)
    sc = torch.empty(
        (out_features, in_features // 32), dtype=torch.uint8, device=device
    )
    mma = torch.empty(
        ((out_features + 127) // 128 * ((in_features + 127) // 128) * 512,),
        dtype=torch.uint8,
        device=device,
    )
    quantize_mxfp8_rows_cute(w_bf16, vals, sc, mma, expected_m=out_features)
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16, device=device) * 0.5
    out_ref = torch.empty((tokens, out_features), dtype=torch.bfloat16, device=device)
    out_fused = torch.empty_like(out_ref)

    query = blockscaled.BlockscaledQuery(
        recipe="mxfp8",
        num_tokens=tokens,
        in_features=in_features,
        padded_in_features=in_features,
        out_features=out_features,
        output_mode="provided",
        workspace_form="provided",
    )
    cfg = blockscaled.BlockscaledConfig(
        mode="a16",
        a16_backend="tc_gemv",
        tile_k=64,
        split_k=split_k,
        mma_warps=4,
        tile_n=16,
    )
    plan = blockscaled.plan(query, override=cfg)
    session = PreparationSession(compile_workers=0)

    def prepare(state):
        ws = torch.empty(
            max(state.required_workspace, 1), dtype=torch.uint8, device=device
        )
        return PreparedCall(
            run=lambda: state.run(
                x,
                vals.view(torch.float8_e4m3fn),
                mma.view(torch.uint8),
                None,
                out=out_ref,
                workspace=ws,
            )
        )

    session.prepare(
        (
            plan.request(
                name=f"rope_{head_dim}_{tokens}_{split_k}", prepare_call=prepare
            ),
        )
    )
    assert register_tc_gemv_rope(plan, head_dim=head_dim, rope_dim=64)
    assert (plan.handle, head_dim, 64) in _TC_GEMV_ROPE_RUNNERS

    pos = torch.arange(tokens, device=device, dtype=torch.int32) + 17
    freqs = (
        1.0 / (10000.0 ** (torch.arange(32, device=device, dtype=torch.float32) / 32.0))
    ).contiguous()
    angles = PreparedRotaryAngles(pos, freqs, split=True)()

    state = plan._prepared.state
    ws = torch.empty(max(state.required_workspace, 1), dtype=torch.uint8, device=device)
    state.run(
        x,
        vals.view(torch.float8_e4m3fn),
        mma.view(torch.uint8),
        None,
        out=out_ref,
        workspace=ws,
    )
    heads = out_features // head_dim
    view = out_ref.view(tokens, heads, head_dim)
    per_cta = 8 if heads % 8 == 0 else 4
    _rotate_grouped[(tokens, (heads + per_cta - 1) // per_cta)](
        view,
        angles,
        view.stride(0),
        view.stride(1),
        head_dim - 64,
        heads,
        1,
        True,
        per_cta,
        num_warps=1,
    )

    _run_mxfp8_with_rope(
        x,
        vals.view(torch.float8_e4m3fn),
        mma.view(torch.uint8),
        None,
        None,
        0,
        out_fused,
        ws,
        plan.handle,
        angles,
        head_dim,
        64,
        torch.cuda.current_stream(device),
    )
    torch.cuda.synchronize()
    fused_view = out_fused.view(tokens, heads, head_dim)
    ref_view = out_ref.view(tokens, heads, head_dim)
    assert torch.equal(fused_view[..., :-64], ref_view[..., :-64])
    torch.testing.assert_close(fused_view[..., -64:], ref_view[..., -64:], rtol=1e-2, atol=1e-3)
    ulp = (fused_view[..., -64:].view(torch.int16).to(torch.int32)
           - ref_view[..., -64:].view(torch.int16).to(torch.int32)).abs()
    assert int(ulp.max().item()) <= 1


def test_split_lagged_mhc_eager_and_compiled_match_reference():
    _require_sm120()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(202)
    tokens, hidden = 8, 5120
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=device) * 0.5
    prev_post = torch.rand(tokens, 4, dtype=torch.float32, device=device)
    prev_comb = torch.softmax(
        torch.randn(tokens, 4, 4, dtype=torch.float32, device=device), dim=-1
    )
    fn = torch.randn(24, 4 * hidden, dtype=torch.float32, device=device) * 0.02
    pre_mix = torch.softmax(
        torch.randn(tokens, 4, dtype=torch.float32, device=device), dim=-1
    )
    scale = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device)
    bias = torch.randn(24, dtype=torch.float32, device=device) * 0.1
    norm_weight = (
        torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    ).contiguous()

    inv = {
        "operation": "post_pre",
        "output_mode": "provided",
        "has_norm_weight": True,
        "norm_weight_dtype": "bfloat16",
        "expanded_residual": False,
        "rms_eps": 1e-20,
        "hc_eps": 1e-6,
        "sinkhorn_iters": 20,
        "norm_eps": 1e-20,
        "lagged_mix": True,
    }
    plan = mhc.plan(
        mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden), invocation=inv
    )
    sess = PreparationSession(compile_workers=0)
    sc_buf = MHCScratch.allocate(tokens, hidden, device)
    pre_ref = torch.empty_like(pre_mix)

    def prepare(state):
        ws = [
            torch.empty(sp.nbytes, dtype=sp.dtype, device=device)
            for sp in state.scratch_specs()
        ]
        binding = state.bind(
            scratch=ws, tokens=tokens, pre_out=pre_ref, **sc_buf.bind_args()
        )
        return PreparedCall(
            run=lambda: mhc_impl._b12x_mhc_post_pre_impl(
                x,
                residual,
                prev_post,
                prev_comb,
                fn,
                scale,
                bias,
                rms_eps=1e-20,
                hc_eps=1e-6,
                sinkhorn_iters=20,
                norm_weight=norm_weight,
                norm_eps=1e-20,
                pre_mix=pre_mix,
                binding=binding,
                _state=state,
            )
        )

    sess.prepare((plan.request(name="ref", prepare_call=prepare),))
    state = plan._prepared.state
    ws = [
        torch.empty(sp.nbytes, dtype=sp.dtype, device=device)
        for sp in state.scratch_specs()
    ]
    binding = state.bind(
        scratch=ws, tokens=tokens, pre_out=pre_ref, **sc_buf.bind_args()
    )
    out_ref, post_ref, comb_ref, y_ref = mhc_impl._b12x_mhc_post_pre_impl(
        x,
        residual,
        prev_post,
        prev_comb,
        fn,
        scale,
        bias,
        rms_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=1e-20,
        pre_mix=pre_mix,
        binding=binding,
        _state=state,
    )

    arena = ScratchArena(device, overlap=True)
    sl = SplitLaggedMHC(
        (x, residual, prev_post, prev_comb),
        MHCWeights(fn, scale, bias, norm_weight),
        session=PreparationSession(compile_workers=0),
        workspace=arena,
        rms_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        norm_eps=1e-20,
    )
    sl._finalize()
    sl.emit_mxfp8 = True
    load_programs(
        compile_cluster_post_y_norm(
            int(device.index or 0),
            hidden,
            False,
            1e-20,
            torch.bfloat16,
            emit_mxfp8=True,
        )
    )

    def step(x_, r_, pp_, pc_, pm_):
        out_, y_, av_, as_ = sl.post_y_norm(x_, r_, pp_, pc_, pre_mix=pm_)
        pend = sl.start_sinkhorn(out_, y_)
        dep = y_ + 1.0
        post_, comb_, pre_ = sl.finish_sinkhorn(pend, dep)
        return out_, y_, av_, as_, post_, comb_, pre_

    eager = step(x, residual, prev_post, prev_comb, pre_mix)
    compiled = torch.compile(step, fullgraph=True)(
        x, residual, prev_post, prev_comb, pre_mix
    )
    torch.cuda.synchronize()

    for a, b in zip(eager, compiled, strict=True):
        assert torch.equal(a, b)
    out_new, y_new, _, _, post_new, comb_new, pre_new = eager
    torch.testing.assert_close(out_new.float(), out_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_new.float(), y_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(pre_new, pre_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(post_new, post_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(comb_new, comb_ref, atol=1e-5, rtol=1e-5)
