# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Selected BF16 projections compile in workers without consulting the rank's plan registry."""

import os

import pytest
import torch

from hyperflare_recipes.components.linear import UnquantizedProjection
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


class ProjectionForward:
    def __init__(self, source, weight, output, session, name):
        self.source, self.weight, self.output = source, weight, output
        self.projection = UnquantizedProjection(source, weight, output,
            session=session, name=name).functional(session=session)

    def __call__(self):
        self.output.copy_(self.projection(self.source))


def test_dense_and_gemv_worker_compilation_preserves_outputs(tmp_path):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    generator = torch.Generator(device="cuda").manual_seed(1041)
    forwards, signatures = {}, {}
    with RecordedPreparationSession(device="cuda:0", checkpoint="synthetic-binary-fractions",
            workload_schema="dense-and-gemv", numerical_policy="exact-fp64-reference",
            compile_workers=2, autotune_effort="off") as session:
        for rows, columns, dense in ((128, 384, True), (1, 4096, True), (1, 384, False)):
            for dtype in (torch.bfloat16, torch.float32):
                name = f"projection-{rows}-{columns}-{dtype}"
                source = torch.randint(-4, 5, (rows, 256), device="cuda", generator=generator).to(torch.bfloat16) / 8
                weight = torch.randint(-4, 5, (columns, 256), device="cuda", generator=generator).to(torch.bfloat16) / 8
                output = torch.empty((rows, columns), device="cuda", dtype=dtype)
                forwards[name] = ProjectionForward(source, weight, output, session, name)
                assert forwards[name].projection.dense == dense
                signatures[name] = GraphSignature("prefill" if rows > 1 else "decode", 1, rows)
        session.tune()
        graphs, compiled = capture_forwards(forwards, signatures, session=session,
            device=torch.device("cuda:0"), stream=torch.cuda.Stream(),
            pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path)
        try:
            assert all(not forward.cache_hit for forward in compiled.values())
            session.freeze()
            for _ in range(3):
                for name, forward in forwards.items():
                    forward.source.neg_()
                    expected = (forward.source.cpu().double() @ forward.weight.cpu().double().T).to(forward.output.dtype)
                    forward.output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graphs[name].replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                    torch.testing.assert_close(forward.output.cpu(), expected, rtol=0, atol=0)
        finally:
            for graph in graphs.values():
                graph.reset()
