# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile reference gather, projection and mHC with synthetic SM103 operands."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.mark.parametrize("rows", [1, 8, 129, 8192])
@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("repeat_direct", [False, True])
@torch.no_grad()
def test_reference_mhc_complete_compilation(rows, shifted, repeat_direct):
    from deepinfer.model_executor.layers.layernorm import RMSNorm
    from deepinfer.model_executor.layers.linear import bf16_gemm_nt
    from deepinfer.model_executor.layers.tilelang_ops.gather import gather
    from deepinfer.model_executor.models.v4_modules.hc import HCLayerMixin

    generator = torch.Generator(device="cuda").manual_seed(187)
    hidden, mult = 5120, 4

    def parameter(*shape):
        return torch.nn.Parameter(torch.randn(shape, generator=generator, device="cuda") * .01,
                                  requires_grad=False)

    source = SimpleNamespace(hc_mult=mult, hc_post_mult_value=2., hc_sinkhorn_repeat=20,
        hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_shifted=shifted,
        hc_attn=parameter(24), hc_ffn=parameter(24), hc_attn_fn=parameter(24, mult * hidden),
        hc_ffn_fn=parameter(24, mult * hidden), hc_attn_scale=parameter(3), hc_ffn_scale=parameter(3),
        hc_attn_norm=SimpleNamespace(eps=1e-6), hc_ffn_norm=SimpleNamespace(eps=1e-6))
    layer = HCLayerMixin(source)
    layer.hc_enabled = True
    layer.attn_layernorm = RMSNorm(hidden, weight=torch.ones(hidden, dtype=torch.bfloat16, device="cuda"), use_fp8=True)
    layer.ffn_layernorm = RMSNorm(hidden, weight=torch.ones(hidden, dtype=torch.bfloat16, device="cuda"), use_fp8=True)
    layer.mlp = torch.nn.Identity()
    table = torch.randn((32, hidden), dtype=torch.bfloat16, device="cuda", generator=generator)
    indices = torch.arange(rows, device="cuda", dtype=torch.int32) % 32
    weight = torch.randn((128, hidden), dtype=torch.bfloat16, device="cuda", generator=generator) * .01
    output = torch.empty((rows, 128), dtype=torch.bfloat16, device="cuda")
    normalized = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    trunk = torch.empty((rows, mult, hidden), dtype=torch.bfloat16, device="cuda")
    quantized = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device="cuda")

    def forward():
        x = gather(table, indices)
        _, bf16, residual = layer.attn_norm_forward(x, None)
        encoded, bf16, residual = layer.ffn_norm_forward(bf16, residual)
        normalized.copy_(bf16)
        quantized.copy_(encoded[0])
        trunk.copy_(layer.hc_post(bf16, residual))
        bf16_gemm_nt(bf16, weight, output)

    compiled = compile_forward(forward)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        compiled()
    try:
        for step in range(12):
            indices.add_(3).remainder_(34).sub_(1)
            table.neg_()
            forward()
            expected = tuple(value.view(torch.uint8).clone() for value in (output, normalized, trunk, quantized))
            if repeat_direct:
                forward()
                for name, actual, reference in zip(("projection", "normalized", "residual", "fp8"),
                        (output, normalized, trunk, quantized), expected, strict=True):
                    difference = (actual.view(torch.uint8) != reference).count_nonzero().item()
                    assert difference == 0, f"direct repeat differs: {name}, step={step}, bytes={difference}"
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            def differences():
                return {name: (actual.view(torch.uint8) != reference).count_nonzero().item()
                        for name, actual, reference in zip(("projection", "normalized", "residual", "fp8"),
                            (output, normalized, trunk, quantized), expected, strict=True)}

            mismatch = differences()
            if any(mismatch.values()):
                graph.replay()
                torch.cuda.synchronize()
                repeated_graph = differences()
                forward()
                torch.cuda.synchronize()
                repeated_direct = differences()
                pytest.fail(f"step={step}, compiled={mismatch}, repeated_graph={repeated_graph}, "
                            f"repeated_direct={repeated_direct}")
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("rows", [1, 129])
@pytest.mark.parametrize("normalized", [False, True])
@torch.no_grad()
def test_reference_projection_compilation_and_accumulator(rows, normalized):
    import deep_gemm
    import tile_kernels
    from deepinfer.model_executor.layers.linear import get_fp8_cast_params
    from deepinfer.model_executor.layers.layernorm import RMSNorm
    from deepinfer.model_executor.layers.linear import ReplicatedLinear

    generator = torch.Generator(device="cuda").manual_seed(93)
    x = torch.randn((rows, 5120), dtype=torch.bfloat16, device="cuda", generator=generator)
    raw = torch.randn((128, 5120), dtype=torch.bfloat16, device="cuda", generator=generator) * .01
    fp8_weight = torch.nn.Parameter(raw.to(torch.float8_e4m3fn), requires_grad=False)
    fp8_weight.sf = torch.ones((4, 160), device="cuda")
    fp8 = ReplicatedLinear(5120, 128, weight=fp8_weight)
    bf16 = ReplicatedLinear(5120, 128, weight=torch.nn.Parameter(raw, requires_grad=False))
    norm = RMSNorm(5120, weight=torch.ones(5120, dtype=torch.bfloat16, device="cuda"), use_fp8=True)
    initial = torch.randn((rows, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    output = torch.empty_like(initial)

    def forward():
        output.copy_(initial)
        bf16(x, out_acc=output)
        fp8(norm(x) if normalized else x, out_acc=output)

    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for _ in range(3):
            x.neg_()
            initial.add_(.125)
            expected = initial.clone()
            deep_gemm.bf16_gemm_nt(x, bf16.weight, expected, expected)
            encoded = norm(x) if normalized else tile_kernels.quant.per_token_cast(x, **get_fp8_cast_params())
            deep_gemm.fp8_fp4_gemm_nt(encoded, (fp8.weight, fp8.weight.sf), expected, expected,
                                     recipe=(1, 1, 32))
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("rows", [1, 8, 129, 8192])
@pytest.mark.parametrize("format_name", ["V41_FORMAT", "V41_FP4_FORMAT", "FP4_INDEX_KV_FORMAT"])
@torch.no_grad()
def test_reference_fused_transform_compilation(rows, format_name):
    from deepinfer.model_executor.layers.tilelang_ops import cache_common
    from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (
        CastArgs, HadamardArgs, KVCacheOutArgs, NormArgs, RopeArgs,
        fused_norm_rope_hadamard_cast_store_kvcache,
    )

    layout = getattr(cache_common, format_name)
    dimension = layout.num_elements
    x = torch.randn((rows, dimension), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(dimension, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(((rows + 127) // 128 + 2, 128, 1, layout.bytes_per_token),
                        dtype=torch.uint8, device="cuda")
    slots = torch.arange(rows, dtype=torch.int32, device="cuda") + 256
    positions = torch.arange(rows, dtype=torch.int32, device="cuda")
    angles = positions[:, None].float() * torch.arange(32, device="cuda")[None] * .001
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
    bf16 = torch.empty_like(x)
    rope = None if dimension == 128 else RopeArgs(positions, cos_sin, 64, False)
    hadamard = HadamardArgs(dimension ** -.5) if dimension == 128 else None

    def forward():
        fused_norm_rope_hadamard_cast_store_kvcache(x, NormArgs(weight), rope,
            hadamard, CastArgs.from_cache_format(layout),
            KVCacheOutArgs(cache, slots, bf16, layout=layout))

    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for step in range(3):
            active = torch.arange(rows, device="cuda") % 3 != step
            slots.copy_(torch.where(active, torch.arange(rows, device="cuda") + 256, -1))
            x.normal_()
            cache.fill_(53)
            forward()
            expected, expected_bf16 = cache.clone(), bf16.clone()
            cache.fill_(53)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            assert torch.equal(cache, expected)
            assert torch.equal(bf16, expected_bf16)
            assert (cache[:2] == 53).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("rows", [1, 8, 129, 8192])
@pytest.mark.parametrize("hc_mult", [1, 4])
@torch.no_grad()
def test_reference_draft_feature_store_compilation(rows, hc_mult):
    pytest.importorskip("deepinfer")
    from deepinfer.model_executor.layers.tilelang_ops.dflash import dflash_input_store

    hidden = torch.randn((rows, hc_mult, 5120), dtype=torch.bfloat16, device="cuda")
    destination = torch.full((rows, 3 * 5120), 53., dtype=torch.bfloat16, device="cuda")

    def run():
        dflash_input_store(hidden, destination[:, 5120:10240])

    compiled = compile_forward(run)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compiled()
    try:
        for _ in range(3):
            hidden.normal_()
            destination.fill_(53.)
            run()
            expected = destination.clone()
            destination.fill_(53.)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
            assert torch.equal(destination, expected)
            assert (destination[:, :5120] == 53).all()
            assert (destination[:, 10240:] == 53).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()
