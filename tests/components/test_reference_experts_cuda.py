# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference normalization, text/image routing and prepared HBM/Grace experts."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
from hyperflare_recipes.components.megamoe_tiers import MegaMoeHostTier, MegaMoeLayerWeights
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession
from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint
from hyperflare_recipes.recipes.deepseek41.reference_experts import ReferenceExperts
from tests.components.test_megamoe_cuda import Rank, experts
from tests.components.test_megamoe_tiers_cuda import select

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.mark.parametrize("host_rows", [0, 4, 8], ids=["hbm", "hbm-grace", "hbm-copies"])
def test_reference_producer_router_and_experts(experts, tmp_path, megamoe_core_storage, host_rows):
    pytest.importorskip("deepinfer")
    from deepinfer.model_executor.layers.layernorm import RMSNorm

    routed, shared = experts
    checkpoint = ReferenceCheckpoint(Path(os.environ["FLARE_DEEPSEEK41_CSF_SNAPSHOT"]))
    chosen = torch.tensor([0, 1, 2, 3, 380, 381, 382, 383])
    weight = checkpoint.read("layers.0.ffn.gate.weight")[chosen].cuda()
    bias = checkpoint.read("layers.0.ffn.gate.bias")[chosen].float().cuda()
    image_bias = checkpoint.read("layers.0.ffn.gate.bias_vl")[chosen].float().cuda()
    physical_rows = 12 if host_rows == 8 else 8
    workspace = MegaMoeWorkspace(MegaMoeGeometry(physical_rows, 8192, 6, 5120, 2304, 1),
        device="cuda:0", kernel_cache=tmp_path)
    dg = workspace.library
    reference = dg.SymmBuffer(Rank(), physical_rows, 8192, 6, 5120, 2304, num_shared_experts=1)
    tier = None
    if host_rows:
        tier = MegaMoeHostTier(workspace.geometry, rows=host_rows, storage=megamoe_core_storage,
                              name="reference_producer")
        tier.copy_weights(select(routed, slice(0, host_rows)))
    resident = select(routed, slice(min(host_rows, 4), 8)) if host_rows else routed
    reference_weights = (select(routed, torch.tensor([*range(8), *range(4, 8)], device="cuda"))
                         if host_rows == 8 else routed)
    generator = torch.Generator(device="cuda").manual_seed(923)
    norm = RMSNorm(5120, weight=torch.ones(5120, dtype=torch.bfloat16, device="cuda"), use_fp8=True)
    placement = torch.arange(8, dtype=torch.int32, device="cuda").view(8, 1)
    if host_rows == 8:
        placement[4:].add_(4)
    cases, graphs, functions = {}, [], []
    capture_stream = torch.cuda.Stream()
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint="reference-csf-expert-sample",
                workload_schema="reference-normalization-router-experts", numerical_policy="reference-bitwise",
                compile_workers=8, autotune_effort="off") as session:
            for rows in (1, 8, 129, 8192):
                x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16, generator=generator) * .1
                padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
                metadata = SimpleNamespace(valid_token_mask=~padding,
                                           images_seq_mask=torch.zeros_like(padding))
                cases[rows] = x, metadata, torch.empty_like(x), torch.empty_like(x)
            weights = MegaMoeLayerWeights(resident, shared, workspace, tier, placement.flatten())
            module = ReferenceExperts.prepare(weights, cases, weight=weight, bias=bias, image_bias=image_bias,
                scaling=1.5, scoring_func="sqrtsoftplus", clamp=10., session=session, name="reference")
            operations = module.operations
            session.tune()
            for rows, (x, metadata, actual, expected) in cases.items():
                kwargs = module.fused_norm_cast_kwargs(rows)
                reference_kwargs = dict(kwargs, out=(reference.x[:rows], reference.x_sf[:rows]),
                                        extra_sf_out=reference.shared_l1_acts_sf[:rows])

                def consume(x=x, metadata=metadata, actual=actual, kwargs=kwargs):
                    encoded, bf16 = norm(x, return_bf16=True, **kwargs)
                    actual.copy_(module(encoded, bf16, metadata))

                consume = compile_forward(consume)
                functions.append(consume)

                def forward(rows=rows, x=x, metadata=metadata, actual=actual, expected=expected,
                            consume=consume, reference_kwargs=reference_kwargs):
                    _, bf16 = norm(x, return_bf16=True, **reference_kwargs)
                    dg.bf16_mega_gate(bf16, weight, num_topk=6, use_shared_as_routed=False,
                        num_shared_experts=0, routed_scaling_factor=1.5, ep_rank=0,
                        scoring_func="sqrtsoftplus", mask=metadata.valid_token_mask,
                        bias=bias, image_bias=image_bias, image_token_mask=metadata.images_seq_mask,
                        to_physical_map=placement, logical_count=module.counts,
                        out=(reference.topk_idx[:rows], reference.topk_weights[:rows]))
                    dg.fp8_fp4_mega_moe(expected, reference_weights.gate_up, reference_weights.down, reference,
                        shared_l1_weights=shared.gate_up, shared_l2_weights=shared.down,
                        activation="swiglu", activation_clamp=10., fast_math=True)
                    consume()

                # MegaGate owns per-stream score barriers. Warm the stream that
                # will be captured so capture never allocates those barriers.
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    forward()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph, stream=capture_stream):
                    forward()
                graphs.append(graph)
            session.freeze()
            for step in range(3):
                for (rows, (x, metadata, actual, expected)), graph in zip(cases.items(), graphs, strict=True):
                    x.neg_()
                    tokens = torch.arange(rows, device="cuda")
                    metadata.valid_token_mask.copy_(tokens % 3 != step)
                    metadata.images_seq_mask.copy_(tokens % 2 == step % 2)
                    actual.fill_(float("nan"))
                    expected.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                    encoded, scales, ids, probabilities, _, _ = operations[rows].encoded_inputs()
                    torch.testing.assert_close(encoded.view(torch.uint8), reference.x[:rows].view(torch.uint8), rtol=0, atol=0)
                    torch.testing.assert_close(scales, reference.x_sf[:rows], rtol=0, atol=0)
                    torch.testing.assert_close(workspace.views[5], reference.shared_l1_acts_sf, rtol=0, atol=0)
                    torch.testing.assert_close(ids, reference.topk_idx[:rows], rtol=0, atol=0)
                    torch.testing.assert_close(probabilities, reference.topk_weights[:rows], rtol=0, atol=0)
                    assert (ids[~metadata.valid_token_mask] == -1).all()
                    if not torch.equal(actual, expected):
                        saved_actual, saved_expected = actual.clone(), expected.clone()
                        def difference(a, b):
                            return dict(elements=int((a != b).sum()),
                                        max_abs=float((a.float() - b.float()).abs().max()))
                        repeatability = []
                        for _ in range(3):
                            graph.replay()
                            torch.cuda.synchronize()
                            repeatability.append(dict(actual=difference(actual, saved_actual),
                                reference=difference(expected, saved_expected),
                                cross=difference(actual, expected)))
                        print(dict(rows=rows, step=step, host_rows=host_rows,
                                   initial=difference(saved_actual, saved_expected), repeats=repeatability))
                        torch.testing.assert_close(saved_actual, saved_expected, rtol=0, atol=0)
            with pytest.raises(ValueError, match="no prepared operation"):
                module.fused_norm_cast_kwargs(2)
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        if tier is not None:
            for region in tier.regions:
                region.close()
