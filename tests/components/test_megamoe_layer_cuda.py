# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""A checkpoint layer retains canonical routing through host/cache row assignment."""

import json
import os
from pathlib import Path

import pytest
import torch

from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
from hyperflare_recipes.components.megamoe_tiers import MegaMoeHostTier
from hyperflare_recipes.components.megamoe_weights import (
    allocate_mxfp4_weights, load_mxfp4_weights_into,
    load_shared_fp8_planes, pack_shared_fp8_planes,
)
from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def test_checkpoint_layer_matches_hbm_with_static_and_cached_copies(tmp_path, megamoe_core_storage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    rowmap = os.environ.get("FLARE_MEGAMOE_REFERENCE_ROWMAP")
    if not snapshot or not rowmap:
        pytest.skip("requires the lossless-CSF checkpoint and DerpSeek exact rowmap")
    placement = json.loads(Path(rowmap).read_text())["layers"]["0"]
    hot, cache, host = (placement[key] for key in ("hot", "cache", "host"))
    assert all(len(rows) == len(set(rows)) for rows in (hot, cache, host))
    assert set(hot).isdisjoint(host) and set(hot) | set(host) == set(range(384))
    assert set(cache) <= set(host)
    assert (len(hot), len(cache), len(host)) == (224, 96, 160)
    physical = host + hot + cache
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    storage = megamoe_core_storage
    graphs = []
    hbm = allocate_mxfp4_weights(len(hot) + len(cache), 5120, 2304, device="cuda:0")
    reference = allocate_mxfp4_weights(384, 5120, 2304, device="cuda:0")
    geometry = MegaMoeGeometry(len(physical), 256, 6, 5120, 2304, 1)
    workspace = MegaMoeWorkspace(geometry, device="cuda:0", kernel_cache=tmp_path)
    tier = MegaMoeHostTier(workspace.geometry, rows=len(host), storage=storage, name="layer0")
    try:
        retained = storage._persistent_arena.used
        with DirectWeightLoader(manifest, "cuda:0", prefixes=("layers.0.ffn.",)) as loader:
            for experts, destination in ((host, tier.cpu_weights), (hot + cache, hbm),
                                          (range(384), reference)):
                load_mxfp4_weights_into(loader,
                    [f"layers.0.ffn.experts.{e}" for e in experts], destination,
                    hidden_size=5120, intermediate_size=2304, batch_experts=8)
                assert storage._persistent_arena.used == retained
            shared = pack_shared_fp8_planes(load_shared_fp8_planes(loader,
                "layers.0.ffn.shared_experts", hidden_size=5120,
                intermediate_size=2304, block_size=(32, 32)))
        # Check every physical copy before exercising the compiled row lookup.
        for experts, destination in ((host, tier.cpu_weights), (hot + cache, hbm)):
            for plane, expected in zip((*destination.gate_up, *destination.down),
                                       (*reference.gate_up, *reference.down)):
                for row, canonical in enumerate(experts):
                    torch.testing.assert_close(plane[row].cpu(), expected[canonical].cpu(), rtol=0, atol=0)
        lookup = torch.empty(384, dtype=torch.int32, device="cuda")
        for row, canonical in enumerate(physical):
            lookup[canonical] = row
        ref_workspace = MegaMoeWorkspace(MegaMoeGeometry(384, 256, 6, 5120, 2304, 1),
                                         device="cuda:0", kernel_cache=tmp_path)
        generator = torch.Generator(device="cuda").manual_seed(712)
        catalog, functions = [], []
        with RecordedPreparationSession(device="cuda:0", checkpoint="deepseek41-layer0",
                workload_schema="host-static-cache", numerical_policy="reference-bitwise",
                compile_workers=2, autotune_effort="off") as session:
            for rows in (1, 8, 64, 256):
                x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16, generator=generator) * .1
                ids = (torch.arange(rows * 6, device="cuda").reshape(rows, 6) % 384).int()
                p = torch.softmax(torch.randn(rows, 6, device="cuda", generator=generator), -1)
                mask = torch.zeros(rows, device="cuda", dtype=torch.bool)
                actual, expected = torch.empty_like(x), torch.empty_like(x)
                op = PreparedMegaMoe(x, ids, p, mask, hbm, shared, workspace=workspace,
                    host_tier=tier, clamp=10., session=session, name=f"tiered-{rows}")
                ref = PreparedMegaMoe(x, ids, p, mask, reference, shared, workspace=ref_workspace,
                    clamp=10., session=session, name=f"hbm-{rows}")
                catalog.append((x, ids, p, mask, actual, expected, op, ref))
            session.tune()
            for x, ids, p, mask, actual, expected, op, ref in catalog:
                def forward(x=x, ids=ids, p=p, mask=mask, actual=actual, expected=expected, op=op, ref=ref):
                    mapped = lookup[ids.long()]
                    actual.copy_(op(x, mapped, p, mask))
                    expected.copy_(ref(x, ids, p, mask))
                fn = compile_forward(forward)
                fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    fn()
                graphs.append(graph)
                functions.append(fn)
            session.freeze()
            for lane in (0, 4):
                tier.control[10] = lane
                for step in range(3):
                    # Publish only between drained replays, alternating cached
                    # experts between their identical host and HBM copies.
                    for index, canonical in enumerate(cache):
                        lookup[canonical] = (host.index(canonical) if step == 1
                                             else len(host) + len(hot) + index)
                    for (x, ids, p, mask, actual, expected, _, _), graph in zip(catalog, graphs):
                        x.neg_()
                        ids.copy_((ids + 7) % 384)
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
                        counts = torch.bincount(lookup[ids[~mask].long()].long().flatten(),
                                                minlength=len(physical))
                        torch.testing.assert_close((tier.control[16:] - before).long(), counts, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        for region in tier.regions:
            region.close()
