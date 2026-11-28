# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity tests for the fused strided Inkling-2 pre-attention L2-clip kernel."""

from __future__ import annotations

import os
import pytest
import torch


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA device")
@pytest.mark.parametrize("has_scale,has_tau", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("tokens,heads,dim", [(1, 24, 128), (16, 24, 128), (64, 24, 16), (256, 8, 128)])
@torch.no_grad()
def test_fused_l2_clip_matches_reference(has_scale, has_tau, tokens, heads, dim):
    import triton
    from hyperflare_recipes.recipes.inkling2.arithmetic import l2_clip
    from hyperflare_recipes.recipes.inkling2.attention import _l2_clip_kernel
    from ignition.preparation.triton import launch_triton, prepare_triton

    device = torch.device("cuda:0")
    torch.manual_seed(101 + tokens + heads + dim)
    stride_cols = heads * dim + 64
    backing = torch.randn(tokens, stride_cols, dtype=torch.bfloat16, device=device) * 4.0
    src = backing[:, 32:32 + heads * dim]
    scale = (
        torch.randn(heads, dim, dtype=torch.bfloat16, device=device) * 0.5 + 1.0
        if has_scale
        else torch.ones(heads, dim, dtype=torch.bfloat16, device=device)
    )
    tau = (
        torch.rand(tokens, dtype=torch.float32, device=device) * 0.5 + 1.0
        if has_tau
        else torch.ones(tokens, dtype=torch.float32, device=device)
    )
    out = torch.empty((tokens, heads, dim), dtype=torch.bfloat16, device=device)
    opts = dict(
        HEADS=heads, DIM=dim, BLOCK_D=triton.next_power_of_2(dim),
        ROW_STRIDE=src.stride(0), THRESHOLD=10.0, EPS=1e-6,
        HAS_SCALE=has_scale, HAS_TAU=has_tau,
        num_warps=4 if dim >= 64 else 1,
    )
    compiled = prepare_triton(_l2_clip_kernel, (src, scale, tau, out), (tokens, heads, 1), **opts).compiled
    launch_triton(_l2_clip_kernel, compiled, (src, scale, tau, out), (tokens, heads, 1), **opts)
    ref = l2_clip(src.reshape(tokens, heads, dim), 10.0,
                  scale=scale if has_scale else None,
                  tau=tau if has_tau else None)
    max_err = (out.float() - ref.float()).abs().max().item()
    assert max_err <= 1e-2, f"max_err={max_err}"
