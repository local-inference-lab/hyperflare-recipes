# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared MegaMoE keeps routing, shared experts and workspace dependencies in graphs."""

import os
from pathlib import Path
import weakref

import pytest
import torch

from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
from hyperflare_recipes.components.megamoe_kernel.inputs import prepare_megamoe_inputs
from hyperflare_recipes.components.megamoe_weights import (
    load_mxfp4_planes, load_shared_fp8_planes, pack_mxfp4_planes, pack_shared_fp8_planes,
)
from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.fixture(scope="module")
def experts():
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires a lossless-CSF DeepSeek V4.1 checkpoint")
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    prefixes = tuple(f"layers.0.ffn.experts.{e}" for e in (0, 1, 2, 3, 380, 381, 382, 383))
    with DirectWeightLoader(manifest, "cuda:0", prefixes=tuple(p + "." for p in prefixes)) as loader:
        routed = pack_mxfp4_planes(load_mxfp4_planes(loader, prefixes, hidden_size=5120, intermediate_size=2304))
    with DirectWeightLoader(manifest, "cuda:0", prefixes=("layers.0.ffn.shared_experts.",)) as loader:
        shared = pack_shared_fp8_planes(load_shared_fp8_planes(loader, "layers.0.ffn.shared_experts",
            hidden_size=5120, intermediate_size=2304, block_size=(32, 32)))
    return routed, shared


class Rank:
    def size(self):
        return 1

    def rank(self):
        return 0

    def barrier(self):
        pass


@pytest.mark.parametrize("with_shared", [False, True])
def test_compiled_catalog_preserves_shared_workspace_and_matches_reference(experts, tmp_path, with_shared):
    routed, shared = experts
    shared = shared if with_shared else None
    workspace = MegaMoeWorkspace(MegaMoeGeometry(8, 256, 6, 5120, 2304, int(with_shared)),
        device="cuda:0", kernel_cache=tmp_path)
    dg = workspace.library
    reference_buffer = dg.SymmBuffer(Rank(), 8, 256, 6, 5120, 2304, num_shared_experts=int(with_shared))
    generator = torch.Generator(device="cuda").manual_seed(971)
    operands, operations, functions, graphs = {}, {}, {}, {}
    with RecordedPreparationSession(device="cuda:0", checkpoint="megamoe-real-experts",
            workload_schema="tp1-shared", numerical_policy="mxfp8-mxfp4",
            compile_workers=2, autotune_effort="off") as session:
        for rows in (1, 8, 64, 256):
            source = torch.randn((rows, 5120), dtype=torch.bfloat16, device="cuda", generator=generator) * .1
            ids = torch.stack([torch.randperm(8, generator=generator, device="cuda")[:6] for _ in range(rows)]).int()
            weights = torch.softmax(torch.randn((rows, 6), device="cuda", generator=generator), -1)
            padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
            output = torch.empty_like(source)
            operands[rows] = source, ids, weights, padding, output
            operations[rows] = PreparedMegaMoe(source, ids, weights, padding, routed, shared,
                workspace=workspace, clamp=10., session=session, name=f"experts{rows}")
        session.tune()
        for rows, operation in operations.items():
            source, ids, weights, padding, output = operands[rows]

            def forward(source=source, ids=ids, weights=weights, padding=padding, output=output, operation=operation):
                # Two serialized layers share protocol storage. Returning the
                # first layer's output prevents compilation from discarding it.
                first = operation(source, ids, weights, padding)
                second = operation(-source, ids, weights, padding)
                output.copy_(first + second)

            functions[rows] = compile_forward(forward)
            functions[rows]()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                functions[rows]()
            graphs[rows] = graph
        session.freeze()
        matrices = (*routed.gate_up, *routed.down, *((*shared.gate_up, *shared.down) if with_shared else ()))
        saved_weights = [tensor.clone() for tensor in matrices]
        for rows in (256, 1, 64, 8, 1, 256):
            source, ids, weights, padding, output = operands[rows]
            source.neg_()
            ids.copy_((ids + 1) % 8)
            padding.copy_(torch.arange(rows, device="cuda") % 3 == 0)
            expected_parts = []
            for value in (source, -source):
                expected = torch.empty_like(source)
                block_m = dg.get_block_m_for_mega_moe(1, 8, reference_buffer.num_max_tokens_per_rank,
                    rows, 6, "fp8xfp4") if with_shared else None
                prepare_megamoe_inputs(value, weights, ids, reference_buffer.x[:rows], reference_buffer.x_sf[:rows],
                    reference_buffer.topk_idx[:rows], reference_buffer.topk_weights[:rows], padding,
                    reference_buffer.shared_l1_acts_sf, block_m)
                dg.fp8_fp4_mega_moe(expected, routed.gate_up, routed.down, reference_buffer,
                    shared_l1_weights=shared.gate_up if with_shared else None,
                    shared_l2_weights=shared.down if with_shared else None,
                    activation_clamp=10., fast_math=True)
                expected_parts.append(expected)
            expected = expected_parts[0] + expected_parts[1]
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graphs[rows].replay()
            torch.cuda.synchronize()
            assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        for actual, saved in zip(matrices, saved_weights):
            torch.testing.assert_close(actual.contiguous().view(torch.uint8),
                                       saved.contiguous().view(torch.uint8), rtol=0, atol=0)
        for graph in graphs.values():
            graph.reset()


def test_prepared_call_rejects_changed_input_geometry(experts, tmp_path):
    routed, shared = experts
    workspace = MegaMoeWorkspace(MegaMoeGeometry(8, 8, 6, 5120, 2304, 1),
        device="cuda:0", kernel_cache=tmp_path)
    source = torch.zeros((8, 5120), device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(6, dtype=torch.int32, device="cuda").repeat(8, 1)
    weights = torch.full((8, 6), 1 / 6, device="cuda")
    padding = torch.zeros(8, dtype=torch.bool, device="cuda")
    with RecordedPreparationSession(device="cuda:0", checkpoint="megamoe-real-experts",
            workload_schema="tp1-shared", numerical_policy="mxfp8-mxfp4",
            compile_workers=2, autotune_effort="off") as session:
        operation = PreparedMegaMoe(source, ids, weights, padding, routed, shared,
            workspace=workspace, clamp=10., session=session, name="experts")
        for operands in ((source[:1], ids[:1], weights[:1], padding[:1]),
                         (source, ids.long(), weights, padding),
                         (source, ids, weights.bfloat16(), padding),
                         (source, ids, weights, padding[:7])):
            with pytest.raises(ValueError, match="prepared contiguous"):
                operation(*operands)
        declarations = [weakref.ref(value) for value in (source, ids, weights, padding)]
        session.tune()
        del operands, source, ids, weights, padding
        assert all(reference() is None for reference in declarations)
