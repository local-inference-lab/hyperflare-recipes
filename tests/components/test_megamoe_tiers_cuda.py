# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-first MegaMoE rows preserve arithmetic, counts and captured dependencies."""

import os

import pytest
import torch

from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
from hyperflare_recipes.components.megamoe_tiers import MegaMoeHostTier
from hyperflare_recipes.components.megamoe_weights import MegaMoeWeights, Mxfp4Planes, pack_mxfp4_planes
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession
from tests.components.test_megamoe_cuda import experts

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def select(weights, rows):
    return MegaMoeWeights(tuple(t[rows].clone() for t in weights.gate_up),
                          tuple(t[rows].clone() for t in weights.down))


@pytest.mark.parametrize("checkpoint", [True, False], ids=["checkpoint", "synthetic-geometry"])
def test_compiled_tiers_match_hbm_with_changing_routes(request, tmp_path, checkpoint, megamoe_core_storage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    if checkpoint:
        routed, shared = request.getfixturevalue("experts")
        geometry = MegaMoeGeometry(8, 256, 6, 5120, 2304, 1)
        host_rows, cases = 4, (1, 8, 64, 256)
    else:
        geometry = MegaMoeGeometry(12, 192, 2, 1024, 512, 0)
        host_rows, cases, shared = 5, (1, 16, 192), None
        g = torch.Generator(device="cuda").manual_seed(813)
        def values(shape):
            return torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8, generator=g)
        def scales(shape):
            return torch.randint(118, 124, shape, device="cuda", dtype=torch.uint8, generator=g)
        routed = pack_mxfp4_planes(Mxfp4Planes(values((12, 1024, 512)), scales((12, 1024, 32)),
            values((12, 1024, 256)), scales((12, 1024, 16))))
    storage = megamoe_core_storage
    graphs = []
    workspace = MegaMoeWorkspace(geometry,
        device="cuda:0", kernel_cache=tmp_path)
    tier = MegaMoeHostTier(workspace.geometry, rows=host_rows, storage=storage, name="routed", host_lane_clusters=4)
    tier.copy_weights(select(routed, slice(0, 2)))
    tier.copy_weights(select(routed, slice(2, host_rows)), start=2)
    hbm = select(routed, slice(host_rows, geometry.experts))
    assert hbm.gate_up[0].shape[0] + tier.weights.gate_up[0].shape[0] == geometry.experts
    with pytest.raises(ValueError, match="exceeds"):
        tier.copy_weights(routed)
    with pytest.raises(ValueError, match="leave at least"):
        MegaMoeHostTier(workspace.geometry, rows=host_rows, storage=storage, name="invalid", host_lane_clusters=76)
    generator = torch.Generator(device="cuda").manual_seed(971)
    catalog = []
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint="megamoe-tiered-real-experts",
                workload_schema="tp1-host-tier", numerical_policy="reference-bitwise",
                compile_workers=2, autotune_effort="off") as session:
            for rows in cases:
                x = torch.randn((rows, geometry.hidden), dtype=torch.bfloat16, device="cuda", generator=generator) * .1
                ids = torch.stack([torch.randperm(geometry.experts, generator=generator, device="cuda")[:geometry.top_k] for _ in range(rows)]).int()
                p = torch.softmax(torch.randn(rows, geometry.top_k, device="cuda", generator=generator), -1)
                mask = torch.zeros(rows, dtype=torch.bool, device="cuda")
                args = x, ids, p, mask
                actual, expected = torch.empty_like(x), torch.empty_like(x)
                op = PreparedMegaMoe(*args, hbm, shared, workspace=workspace, host_tier=tier,
                                     clamp=10., session=session, name=f"tiered-{rows}")
                ref = PreparedMegaMoe(*args, routed, shared, workspace=workspace,
                                      clamp=10., session=session, name=f"hbm-{rows}")
                catalog.append((args, op, ref, actual, expected))
            session.tune()
            functions = []
            for args, op, ref, actual, expected in catalog:
                def forward(args=args, op=op, ref=ref, actual=actual, expected=expected):
                    actual.copy_(op(*args))
                    expected.copy_(ref(*args))
                fn = compile_forward(forward)
                fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    fn()
                functions.append(fn)
                graphs.append(graph)
            session.freeze()
            for lane in (0, 4):
                tier.control[10] = lane
                for step in range(3):
                    for (args, _, _, actual, expected), graph in zip(catalog, graphs):
                        x, ids, p, mask = args
                        x.neg_()
                        ids.copy_((ids + 1) % geometry.experts)
                        mask.copy_(torch.arange(x.shape[0], device="cuda") % 3 == step)
                        before = tier.control[16:].clone()
                        actual.fill_(float("nan"))
                        expected.fill_(float("nan"))
                        torch.cuda.synchronize()
                        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                        graph.replay()
                        torch.cuda.synchronize()
                        assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        counts = torch.bincount(ids[~mask].long().flatten(), minlength=geometry.experts)
                        torch.testing.assert_close((tier.control[16:] - before).long(), counts, rtol=0, atol=0)
                        assert torch.equal(tier.control[12:14], torch.zeros(2, device="cuda", dtype=torch.int32))
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        for region in tier.regions:
            region.close()
