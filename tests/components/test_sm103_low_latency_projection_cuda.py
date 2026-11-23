# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Low-row MXFP8 candidates preserve packed scales and allocation-free graph replay."""
from dataclasses import replace
import os

import pytest
import torch
from b12x.gemm import blockscaled
from b12x.gemm.blockscaled._a16 import _weight_parts
from b12x.preparation import PreparedCall
from b12x.preparation.tuning import BackendConfig
from b12x.preparation.types import require_prepared

from hyperflare_recipes.components import sm103_projection as projection
from ignition.preparation.capture import capture_forwards
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.tuning import RecordedPreparationSession, declare


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


class ProjectionForward:
    def __init__(self, source, packed, scales, output, workspace, plan, functional):
        self.source, self.packed, self.scales = source, packed, scales
        self.output, self.workspace, self.plan = output, workspace, plan
        self.functional = functional

    def __call__(self):
        if self.functional:
            destination = torch.empty_like(self.output)
            scratch = torch.empty_like(self.workspace)
        else:
            destination, scratch = self.output, self.workspace
        torch.ops.flare.sm103_packed_out(self.source, self.packed, self.scales, None,
                                        destination, scratch, self.plan.handle)
        if self.functional:
            self.output.copy_(destination)


@pytest.mark.parametrize("rows,width,features", [
    (1, 128, 256), (5, 512, 256), (8, 256, 128), (8, 128, 136)])
@pytest.mark.parametrize("backend", ["cute-dsl:-1", "cutedsl_low_latency:-1"])
@pytest.mark.parametrize("functional", [False, True])
@torch.no_grad()
def test_low_latency_packed_projection_replays_exact_binary_operands(
        rows, width, features, backend, functional, tmp_path):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(41017)
    values = (torch.randint(-4, 5, (features, width), device=device, generator=generator).float() / 8)
    scale_exponents = torch.randint(-2, 3, (features, width // 32), device=device, generator=generator)
    weight = blockscaled.pack_weight(values.to(torch.float8_e4m3fn),
        (scale_exponents + 127).to(torch.uint8), recipe="mxfp8")
    packed, scales, _, _ = _weight_parts(weight)
    scales = scales.view(torch.uint8)
    source = (torch.randint(-4, 5, (rows, width), device=device, generator=generator).to(torch.bfloat16) / 8)
    output = torch.empty((rows, features), dtype=torch.bfloat16, device=device)
    query = projection.query_for(source, weight, output, workspace_form="provided")
    workspace = torch.empty(projection.workspace_bytes(query), dtype=torch.uint8, device=device)
    plan = replace(projection.plan(query), override=BackendConfig(backend=backend))
    expected_weight = values.cpu().double() * (2.0 ** scale_exponents.cpu().double()).repeat_interleave(32, 1)
    with RecordedPreparationSession(device=device, checkpoint="binary-mxfp8",
            workload_schema="low-row-projection", numerical_policy="exact-fp64-reference",
            compile_workers=2, autotune_effort="off") as session:
        declare(session, (plan.request(name="projection", prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, packed, scales, None, out=output, workspace=workspace),
            output=output)),))
        session.tune()
        require_prepared(plan, projection.COMPONENT)

        project = ProjectionForward(source, packed, scales, output, workspace, plan, functional)

        if functional:
            graphs, compiled = capture_forwards({"projection": project},
                {"projection": GraphSignature("decode", 1, rows)}, session=session,
                device=device, stream=torch.cuda.Stream(), pool=torch.cuda.graph_pool_handle(),
                cache_root=tmp_path)
            graph, forward = graphs["projection"], compiled["projection"]
            assert not forward.cache_hit
        else:
            forward = compile_forward(project)
            forward()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                forward()
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"projection": graph}, buffers={"source": source},
            resources=(source, output, weight, workspace, plan, forward, session))
        worker = bundle.start_worker()
        try:
            for _ in range(3):
                source.neg_()
                expected = (source.cpu().double() @ expected_weight.T).to(torch.bfloat16)
                output.fill_(float("nan"))
                workspace.fill_(165)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(bundle.graph_indices["projection"])
                result = worker.poll(timeout_ms=120000)
                assert result is not None and result.ticket == ticket and result.error is None
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for counter in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                    assert before[counter] == after[counter]
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        finally:
            worker.close()
            graph.reset()
