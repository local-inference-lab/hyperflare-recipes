# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured lookahead copies preserve checkpoint experts across stream reuse."""

import os

import pytest
import torch

from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
from hyperflare_recipes.components.megamoe_staging import PreparedExpertStage
from hyperflare_recipes.components.megamoe_tiers import MegaMoeHostTier
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from tests.components.test_megamoe_cuda import experts
from tests.components.test_megamoe_tiers_cuda import select

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.mark.parametrize("effort,overlap", [("off", True), ("default", True), ("off", False)],
                         ids=["heuristic", "tuned", "serial"])
def test_compiled_staging_preserves_checkpoint_outputs(experts, tmp_path, megamoe_core_storage, effort, overlap):
    routed, shared = experts
    workspace = MegaMoeWorkspace(MegaMoeGeometry(8, 8192, 6, 5120, 2304, 1),
                                device="cuda:0", kernel_cache=tmp_path)
    streams = [torch.cuda.Stream() for _ in range(2)] if overlap else [None, None]
    tiers, references, residents, stagers, graphs, functions = [], [], [], [], [], []
    for layer in range(3):
        weights = select(routed, [(i + layer) % 8 for i in range(8)])
        tier = MegaMoeHostTier(workspace.geometry, rows=4, storage=megamoe_core_storage,
                              name=f"staged{layer}", host_lane_clusters=4)
        tier.copy_weights(select(weights, slice(0, 4)))
        tiers.append(tier)
        residents.append(select(weights, slice(4, 8)))
        references.append(weights)
    generator = torch.Generator(device="cuda").manual_seed(58103)
    catalog = []
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint="deepseek41-eight-experts",
                workload_schema="captured-expert-staging", numerical_policy="reference-bitwise",
                compile_workers=2, autotune_effort=effort) as session:
            arena = ScratchArena("cuda:0")
            session.register_scratch(arena)
            mapping = torch.arange(8, device="cuda", dtype=torch.int32)
            stagers = [PreparedExpertStage(tier, streams[layer % 2], mapping, range(4),
                session=session, workspace=arena, name=f"stage-{layer}", min_cold=2) for layer, tier in enumerate(tiers)]
            for rows in (256, 1024, 8192):
                x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16, generator=generator) * .1
                ids = (torch.arange(rows * 6, device="cuda") % 8).reshape(rows, 6).int()
                probabilities = torch.softmax(torch.randn(rows, 6, device="cuda", generator=generator), -1)
                padding = torch.zeros(rows, device="cuda", dtype=torch.bool)
                args = x, ids, probabilities, padding
                actual = [torch.empty_like(x) for _ in tiers]
                expected = [torch.empty_like(x) for _ in tiers]
                operations, controls = [], []
                for layer, tier in enumerate(tiers):
                    operations.append(PreparedMegaMoe(*args, residents[layer], shared,
                        workspace=workspace, host_tier=tier, clamp=10., session=session,
                        name=f"staged-{layer}-{rows}"))
                    controls.append(PreparedMegaMoe(*args, references[layer], shared,
                        workspace=workspace, clamp=10., session=session, name=f"reference-{layer}-{rows}"))
                catalog.append((args, operations, controls, actual, expected))
            before_tuning = torch.cuda.memory_allocated()
            session.tune()
            torch.cuda.synchronize()
            arena.release_preparation()
            # A selected plan must not retain a full copy destination allocated
            # for tuning; model preparation has one such plan per host geometry.
            assert torch.cuda.memory_allocated() - before_tuning < sum(t.numel() for t in stagers[0].sources)
            for args, operations, controls, actual, expected in catalog:
                def forward(args=args, operations=operations, controls=controls,
                            actual=actual, expected=expected):
                    pending = [stagers[i].start(args[0]) for i in range(2)]
                    first = operations[0](*args, staged=(stagers[0], pending[0]))
                    last = stagers[2].start(first)
                    actual[0].copy_(first)
                    actual[1].copy_(operations[1](*args, staged=(stagers[1], pending[1])))
                    actual[2].copy_(operations[2](*args, staged=(stagers[2], last)))
                    for control, destination in zip(controls, expected, strict=True):
                        destination.copy_(control(*args))
                fn = compile_forward(forward)
                fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    fn()
                graphs.append(graph)
                functions.append(fn)
            session.freeze()
            headers = [tier.control.clone() for tier in tiers]
            for step in range(6):
                for (args, _, _, actual, expected), graph in reversed(list(zip(catalog, graphs))):
                    x, ids, _, padding = args
                    x.neg_()
                    ids.copy_((torch.arange(ids.numel(), device="cuda").reshape_as(ids) + step)
                              % (4 if step % 2 else 8) + (4 if step % 2 else 0))
                    padding.copy_(torch.arange(x.shape[0], device="cuda") % 3 == step % 3)
                    for destination in actual + expected:
                        destination.fill_(float("nan"))
                    torch.cuda.synchronize()
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                    for a, b in zip(actual, expected, strict=True):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        assert torch.isfinite(a).all()
                    for tier, header in zip(tiers, headers, strict=True):
                        torch.testing.assert_close(tier.control, header, rtol=0, atol=0)
                    expected_cold = ids[~padding].unique()
                    expected_cold = (expected_cold < 4).sum().item()
                    assert all(stager.decision.item() == expected_cold for stager in stagers)
            for tier, reference in zip(tiers, references, strict=True):
                for backing, source in zip(tier.host_views, (*reference.gate_up, *reference.down), strict=True):
                    assert torch.equal(backing, source[:4].cpu())
            # Poisoned destinations prove that skipped and resident rows are not
            # copied, even when residency changes without recapturing this graph.
            from hyperflare_recipes.components.megamoe_staging import _start, _join

            stager = stagers[0]
            planes = [torch.empty_like(source) for source in stager.sources]
            def copy_forward():
                _start(args[0], stager.sources, planes, mapping, stager.canonical,
                       stager.decision, stager.stream_handle, stager.plan.handle)
                _join(args[0], stager.sources, planes, mapping, stager.canonical,
                      stager.decision, stager.stream_handle)
            fn = compile_forward(copy_forward)
            fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs.append(graph)
            functions.append(fn)
            for enabled, cold in [(False, (0, 1, 2, 3)), (True, (0, 2)), (True, (1, 3)), (False, ())]:
                stager.decision.fill_(2 if enabled else 0)
                mapping.copy_(torch.tensor([i if i in cold else 4 + i for i in range(4)]
                                           + list(range(4, 8)), device="cuda", dtype=torch.int32))
                for plane in planes:
                    plane.fill_(165)
                graph.replay()
                torch.cuda.synchronize()
                for source, plane in zip(stager.sources, planes, strict=True):
                    expected = torch.full_like(source, 165).reshape(4, -1)
                    if enabled:
                        expected[list(cold)] = source.reshape(4, -1)[list(cold)]
                    torch.testing.assert_close(plane, expected.flatten(), rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        for tier in tiers:
            for region in tier.regions:
                region.close()
