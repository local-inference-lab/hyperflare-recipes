# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 staging preserves routing and handles changing graph inputs and tails."""

import os

import pytest
import torch

from hyperflare_recipes.components.megamoe_kernel.inputs import prepare_megamoe_inputs


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def reference(x):
    groups = x.cpu().float().reshape(x.shape[0], -1, 32)
    scale = groups.abs().amax(-1).clamp_min(1e-4) / 448.0
    bits = scale.view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7fffff) != 0)).clamp(1, 254)
    values = (groups / (exponent << 23).view(torch.float32)[..., None]).to(torch.float8_e4m3fn)
    shifts = torch.arange(4, dtype=torch.int64) * 8
    packed = (exponent.to(torch.int64).reshape(x.shape[0], -1, 4) << shifts).sum(-1).to(torch.int32)
    return values.reshape(x.shape).view(torch.uint8), packed


@pytest.mark.parametrize("rows, hidden, block_m", [(1, 128, 32), (7, 128, 64),
                                                   (64, 5120, 64), (129, 5120, 128)])
def test_staging_graph_replay_updates_quantization_routes_and_padding(rows, hidden, block_m):
    # Noncontiguous inputs and MN-major scale destinations exercise the strides
    # used by MegaMoE's workspace, including row tails and unwritten shared rows.
    x = torch.empty((rows, hidden * 2), dtype=torch.bfloat16, device="cuda")[:, ::2]
    ids = torch.empty((rows, 12), dtype=torch.int32, device="cuda")[:, ::2]
    weights = torch.empty((rows, 12), dtype=torch.float32, device="cuda")[:, ::2]
    padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
    output = torch.empty_like(x, memory_format=torch.contiguous_format).to(torch.float8_e4m3fn)
    sf = torch.empty((hidden // 128, rows), dtype=torch.int32, device="cuda").T
    ids_out = torch.empty((rows, 6), dtype=torch.int64, device="cuda")
    weights_out = torch.empty_like(ids_out, dtype=torch.float32)
    shared_rows = ((rows + block_m - 1) // block_m) * 128 + 128
    shared = torch.full((hidden // 128, shared_rows), -999, dtype=torch.int32, device="cuda").T

    def fill(step):
        generator = torch.Generator(device="cuda").manual_seed(197 + step)
        x.copy_(torch.randn(x.shape, generator=generator, device="cuda") * (0.01 if step == 0 else 3))
        x[0, :32] = 0
        ids.copy_(torch.randint(0, 384, ids.shape, generator=generator, device="cuda"))
        weights.copy_(torch.rand(weights.shape, generator=generator, device="cuda"))
        padding.copy_(torch.arange(rows, device="cuda") % 3 == step)
        if step == 2:
            padding.fill_(True)

    def stage():
        prepare_megamoe_inputs(x, weights, ids, output, sf, ids_out, weights_out,
                              padding, shared, block_m)

    fill(0)
    stage()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage()
    for step in range(3):
        fill(step)
        output.view(torch.uint8).fill_(255)
        sf.fill_(-999)
        ids_out.fill_(-999)
        weights_out.fill_(float("nan"))
        shared.fill_(-999)
        graph.replay()
        torch.cuda.synchronize()
        expected_values, expected_sf = reference(x)
        torch.testing.assert_close(output.cpu().view(torch.uint8), expected_values, rtol=0, atol=0)
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(ids_out.cpu(), ids.cpu().long().masked_fill(padding.cpu()[:, None], -1))
        torch.testing.assert_close(weights_out.cpu(), weights.cpu().masked_fill(padding.cpu()[:, None], 0),
                                   rtol=0, atol=0)
        expected_shared = torch.full(shared.shape, -999, dtype=torch.int32)
        for row in range(rows):
            within = row % block_m
            destination = (row // block_m) * 128 + (within % 32) * 4 + within // 32
            expected_shared[destination] = expected_sf[row]
        torch.testing.assert_close(shared.cpu(), expected_shared, rtol=0, atol=0)


@pytest.mark.parametrize("defect, message", [("route_rows", "one nonempty top-k row"),
                                             ("scale_storage", "x_sf requires"),
                                             ("padding", "boolean vector"),
                                             ("shared_rows", "insufficient rows")])
def test_staging_rejects_unsafe_buffer_geometry(defect, message):
    x = torch.ones((7, 128), device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros((7, 6), device="cuda", dtype=torch.int32)
    weights = torch.ones((7, 6), device="cuda", dtype=torch.float32)
    sf = torch.empty((7, 1), device="cuda", dtype=torch.int32)
    padding = torch.zeros(7, device="cuda", dtype=torch.bool)
    shared = torch.empty((128, 1), device="cuda", dtype=torch.int32)
    if defect == "route_rows":
        ids = ids[:6]
    elif defect == "scale_storage":
        sf = sf.float()
    elif defect == "padding":
        padding = padding[:6]
    elif defect == "shared_rows":
        shared = shared[:127]
    with pytest.raises(ValueError, match=message):
        prepare_megamoe_inputs(x, weights, ids, torch.empty_like(x, dtype=torch.float8_e4m3fn), sf,
                              torch.empty((7, 6), device="cuda", dtype=torch.int64),
                              torch.empty_like(weights), padding, shared, 64)
