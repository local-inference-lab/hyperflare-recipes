# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Already-encoded expert inputs retain HBM/Grace arithmetic under graph replay."""

import os

import pytest
import torch

from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
from hyperflare_recipes.components.megamoe_kernel.inputs import prepare_megamoe_inputs
from hyperflare_recipes.components.megamoe_staging import PreparedExpertStage
from hyperflare_recipes.components.megamoe_tiers import MegaMoeHostTier
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from tests.components.test_megamoe_cuda import experts
from tests.components.test_megamoe_tiers_cuda import select

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.mark.parametrize("host_rows", [0, 4], ids=["hbm", "hbm-grace"])
def test_encoded_consumer_matches_staged_inputs(experts, tmp_path, megamoe_core_storage, host_rows):
    routed, shared = experts
    workspace = MegaMoeWorkspace(MegaMoeGeometry(8, 256, 6, 5120, 2304, 1),
        device="cuda:0", kernel_cache=tmp_path)
    tier = None
    graphs = []
    functions = []
    if host_rows:
        tier = MegaMoeHostTier(workspace.geometry, rows=host_rows, storage=megamoe_core_storage,
            name="reference_encoded_experts")
        tier.copy_weights(select(routed, slice(0, host_rows)))
        routed = select(routed, slice(host_rows, 8))
    generator = torch.Generator(device="cuda").manual_seed(438)
    cases = []
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint="encoded-megamoe-real-experts",
                workload_schema="tp1-encoded-inputs", numerical_policy="reference-bitwise",
                compile_workers=8, autotune_effort="off") as session:
            for rows in (1, 8, 256):
                x = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16, generator=generator) * .1
                ids = torch.stack([torch.randperm(8, device="cuda", generator=generator)[:6] for _ in range(rows)]).int()
                probabilities = torch.softmax(torch.randn((rows, 6), device="cuda", generator=generator), -1)
                padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
                op = PreparedMegaMoe(x, ids, probabilities, padding, routed, shared,
                    workspace=workspace, host_tier=tier, clamp=10., session=session, name=f"encoded-{rows}")
                actual, expected = torch.empty_like(x), torch.empty_like(x)
                cases.append((op, x, ids, probabilities, padding, actual, expected))
            session.tune()
            for op, x, ids, probabilities, padding, actual, expected in cases:
                encoded, scales, out_ids, out_probabilities, shared_scales, block_m = op.encoded_inputs()
                # The staging helper takes physical padded rows; the reference
                # normalization producer takes logical rows with the same stride.
                shared_scales = workspace.views[5]

                def forward(op=op, x=x, ids=ids, probabilities=probabilities, padding=padding,
                            actual=actual, expected=expected, encoded=encoded, scales=scales,
                            out_ids=out_ids, out_probabilities=out_probabilities,
                            shared_scales=shared_scales, block_m=block_m):
                    expected.copy_(op(x, ids, probabilities, padding))
                    prepare_megamoe_inputs(x, probabilities, ids, encoded, scales,
                        out_ids, out_probabilities, padding, shared_scales, block_m)
                    actual.copy_(op.encoded(x))

                forward = compile_forward(forward)
                functions.append(forward)
                forward()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    forward()
                graphs.append(graph)
            session.freeze()
            for step in range(3):
                for (op, x, ids, probabilities, padding, actual, expected), graph in zip(cases, graphs, strict=True):
                    x.neg_()
                    ids.copy_((ids + 1) % 8)
                    padding.copy_(torch.arange(x.shape[0], device="cuda") % 3 == step)
                    actual.fill_(float("nan"))
                    expected.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with pytest.raises(ValueError, match="producer shape"):
                cases[-1][0].encoded(cases[-1][1][:1])
            op, x, *_ = cases[0]
            encoded, scales, *_ = op.encoded_inputs()
            with pytest.raises(ValueError, match="must alias"):
                op.encoded(x, (encoded.clone(), scales))
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        if tier is not None:
            for region in tier.regions:
                region.close()


def test_encoded_consumer_supports_cold_expert_staging(experts, tmp_path, megamoe_core_storage):
    routed, shared = experts
    workspace = MegaMoeWorkspace(MegaMoeGeometry(8, 1024, 6, 5120, 2304, 1),
        device="cuda:0", kernel_cache=tmp_path)
    tier = MegaMoeHostTier(workspace.geometry, rows=4, storage=megamoe_core_storage,
        name="reference_encoded_staged_experts")
    control_tier = MegaMoeHostTier(workspace.geometry, rows=4, storage=megamoe_core_storage,
        name="reference_encoded_unstaged_experts")
    tier.copy_weights(select(routed, slice(0, 4)))
    control_tier.copy_weights(select(routed, slice(0, 4)))
    resident = select(routed, slice(4, 8))
    stream = torch.cuda.Stream()
    mapping = torch.arange(8, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(1031)
    graph = None
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint="encoded-megamoe-staged",
                workload_schema="tp1-encoded-staged", numerical_policy="reference-bitwise",
                compile_workers=4, autotune_effort="off") as session:
            arena = ScratchArena("cuda:0")
            session.register_scratch(arena)
            stager = PreparedExpertStage(tier, stream, mapping, range(4),
                session=session, workspace=arena, name="encoded-stage", min_cold=2)
            rows = 1024
            x = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16, generator=generator) * .1
            ids = (torch.arange(rows * 6, device="cuda") % 8).reshape(rows, 6).int()
            probabilities = torch.softmax(torch.randn((rows, 6), device="cuda", generator=generator), -1)
            padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
            op = PreparedMegaMoe(x, ids, probabilities, padding, resident, shared,
                workspace=workspace, host_tier=tier, clamp=10., session=session, name="encoded-staged-1024")
            control_op = PreparedMegaMoe(x, ids, probabilities, padding, resident, shared,
                workspace=workspace, host_tier=control_tier, clamp=10., session=session, name="encoded-unstaged-1024")
            actual, expected = torch.empty_like(x), torch.empty_like(x)
            session.tune()
            arena.release_preparation()
            encoded, scales, out_ids, out_probabilities, _, block_m = op.encoded_inputs()
            shared_scales = workspace.views[5]

            def forward():
                expected.copy_(control_op(x, ids, probabilities, padding))
                pending = stager.start(x)
                prepare_megamoe_inputs(x, probabilities, ids, encoded, scales,
                    out_ids, out_probabilities, padding, shared_scales, block_m)
                actual.copy_(op.encoded(x, staged=(stager, pending)))

            fn = compile_forward(forward)
            fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                fn()
            session.freeze()
            stager.decision.zero_()
            header = tier.control.clone()
            for step in range(4):
                x.neg_()
                ids.copy_((torch.arange(ids.numel(), device="cuda").reshape_as(ids) + step)
                          % (4 if step % 2 else 8) + (4 if step % 2 else 0))
                padding.copy_(torch.arange(rows, device="cuda") % 3 == step % 3)
                actual.fill_(float("nan"))
                expected.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(tier.control, header, rtol=0, atol=0)
                expected_cold = int((ids[~padding].unique() < 4).sum().item())
                assert stager.decision.item() == expected_cold
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        for region in (*tier.regions, *control_tier.regions):
            region.close()
