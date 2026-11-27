# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native table delivery feeds the reference projection without BF16 conversion."""

import os
from pathlib import Path
import time

import pytest
import torch
from safetensors import safe_open

from hyperflare_recipes.components.delivered_fp8_projection import (
    DeliveredFP8Projection, DeliveredFP8Rows,
)
from ignition.components.ngram_delivery import publish_requests
from ignition.components.readiness import ReadinessTimings
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.fixture(scope="module")
def reference_projection():
    pytest.importorskip("deepinfer")
    from deepinfer.model_executor.layers.linear import ReplicatedLinear

    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires the DeepSeek V4.1 CSF checkpoint")
    manifest = inspect_indexed_checkpoint(Path(snapshot))

    def read_tensor(name):
        path = manifest.snapshot / manifest.tensors[name].shard
        with safe_open(path, framework="pt", device="cpu") as reader:
            return reader.get_tensor(name).clone()

    weight = torch.nn.Parameter(read_tensor("layers.1.engram.wkv.weight").cuda(), requires_grad=False)
    weight.sf = read_tensor("layers.1.engram.wkv.scale").float().cuda()
    projection = ReplicatedLinear(weight.shape[1], weight.shape[0], weight=weight)
    samples = []
    for suffix in ("weight", "scale"):
        name = f"layers.1.engram.embed.{suffix}"
        path = manifest.snapshot / manifest.tensors[name].shard
        with safe_open(path, framework="pt", device="cpu") as reader:
            samples.append(reader.get_slice(name)[:512].view(torch.uint8).clone())
    return projection, samples


@pytest.mark.parametrize("tokens", [1, 192, 8192])
@torch.no_grad()
def test_native_delivery_matches_reference(reference_projection, tokens, tmp_path, megamoe_core_storage):
    import flare_heliopause

    reference, (table_values, table_scales) = reference_projection
    device = torch.device("cuda:0")
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    timings = ReadinessTimings(frame, 1)
    ready = torch.cuda.Event(external=True)
    ready.record()
    hashes = torch.full((tokens, 24), -1, dtype=torch.int64, device=device)
    values = torch.zeros((tokens, 6144), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.zeros((tokens, 192), dtype=torch.uint8, device=device)
    valid = torch.zeros(tokens, dtype=torch.bool, device=device)
    mailbox = megamoe_core_storage.host_allocate(f"reference_rows_{tokens}", (tokens * 24 + 1,), torch.int64)
    mailbox.host_view.zero_()
    inputs = DeliveredFP8Rows(values, scales, valid, hashes, mailbox.device_view)
    operation = DeliveredFP8Projection((reference.weight, reference.weight.sf),
        ready=ready, timings=timings, interval=0)
    output = torch.empty((tokens, reference.output_size), dtype=torch.bfloat16, device=device)
    records = torch.zeros((2, 1, 32), dtype=torch.int64, device=device)
    planes = []
    for name, table in (("weights", table_values), ("scales", table_scales)):
        path = tmp_path / name
        path.write_bytes(bytes(37) + table.numpy().tobytes())
        planes.append(dict(segments=[dict(path=str(path), offset=37, nbytes=table.nbytes)], row_bytes=table.shape[1]))
    catalog = dict(vocabulary=32, tables=[dict(planes=planes, heads=24,
        weights="row_weights", scales="row_scales", hashes="row_hashes",
        event="rows_ready", request="row_requests")])

    def forward():
        publish_requests(hashes, mailbox.device_view, sequence)
        output.copy_(operation(inputs))
        records.zero_()
        records[:, 0, 0] = 1
        records[:, 0, 1].copy_(sequence[0].expand(2))
        records[:, 0, 4] = -1
        records[:, 0, 5] = 1
        records[:, 0, 12] = -1
        records[:, 0, 15] = 1
        records[:, 0, 16:] = -1

    graph, worker = None, None
    try:
        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        torch.cuda.synchronize()
        execution = PreparedExecution(device=0, graphs={"prefill": graph}, buffers=dict(
            row_weights=values.view(torch.uint8).view(tokens * 24, 256),
            row_scales=scales.view(tokens * 24, 8), row_hashes=hashes, row_requests=mailbox.device_view,
            valid_rows=valid, output=output, frame_index=frame, submission_sequence=sequence,
            completion_records=records, **timings.buffers()),
            signatures={"prefill": GraphSignature("prefill", 1, tokens, ("row_hashes", "valid_rows"))},
            max_active=1, resources=(operation, reference, compiled, inputs, mailbox),
            export_events={"rows_ready": ready.cuda_event})
        resident = flare_heliopause.EmbeddingTables(catalog)
        manifest = execution.manifest()
        manifest["engram"] = resident.bind(manifest, execution.export_events)
        worker = flare_heliopause.GraphExecutor((execution, resident), manifest)
        for plane in planes:
            Path(plane["segments"][0]["path"]).unlink()
        for generation in range(1, 5):
            ids = (torch.arange(tokens * 24).reshape(tokens, 24) * 19 + generation) % 512
            active = torch.arange(tokens) % 3 != generation % 3
            if generation == 3:
                active.zero_()
            ids[~active] = -1
            ids[::2, ::5] = -1
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            fence = worker.submit(0, [("row_hashes", 0, ids.numpy().tobytes()),
                                      ("valid_rows", 0, active.numpy().tobytes())])
            assert fence is not None
            deadline = time.monotonic() + 30
            completion = worker.poll()
            while completion is None and time.monotonic() < deadline:
                time.sleep(.001)
                completion = worker.poll()
            assert completion is not None and completion.fence == fence
            assert completion.readiness_wait_ns is not None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            expected_values = table_values[ids.clamp_min(0)].clone()
            expected_scales = table_scales[ids.clamp_min(0)].clone()
            expected_values[ids < 0] = 0
            expected_scales[ids < 0] = 0
            expected_values = expected_values.view(torch.float8_e4m3fn).flatten(1).cuda()
            sf = expected_scales.flatten(1).view(torch.int32).cuda()
            packed = torch.empty((sf.shape[1], (tokens + 3) // 4 * 4), dtype=torch.int32, device=device).mT[:tokens]
            packed.copy_(sf)
            expected = reference((expected_values, packed))
            expected.masked_fill_(~active.cuda()[:, None], 0)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        if worker is not None:
            worker.close()
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        mailbox.close()
