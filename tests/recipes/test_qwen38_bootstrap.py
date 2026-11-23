# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU ownership tests for prepared-model handoff and confirmed-drain cleanup."""

from __future__ import annotations

import gc
import sys
import weakref
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.execution import GraphSignature
from ignition.preparation.model import PreparedModel


class Resource:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def close(self):
        self.events.append(self.name)

    def reset(self):
        self.events.append(self.name)


def model_fixture(monkeypatch, *, synchronize=None, server=None):
    events = []
    weights = Resource("weights", events)
    weights_ref = weakref.ref(weights)
    model = PreparedModel(
        device=0,
        graphs={"decode": Resource("graph", events)},
        buffers={"frame_index": object(), "submission_sequence": object(),
                 "completion_records": torch.empty((2, 4, 32), dtype=torch.int64)},
        signatures={"decode": GraphSignature("decode", 1, 1)},
        resources=(weights,),
        server_config={},
        memory_report={},
        session=Resource("session", events),
        close_resources=(Resource("table", events),),
    )
    monkeypatch.setattr(model, "manifest", lambda: {"test_manifest": True})
    monkeypatch.setattr("torch.cuda.synchronize", synchronize or (lambda device: events.append("synchronize")))
    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace(Server=server))
    return model, events, weights_ref


def thread_call(call, *, entered=None):
    errors = []

    def run():
        if entered is not None:
            entered.set()
        try:
            call()
        except BaseException as error:  # noqa: BLE001 - Propagate worker failures to the test thread.
            errors.append(error)

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def join(thread):
    thread.join(timeout=3)
    assert not thread.is_alive(), "prepared-model handoff deadlocked"


def test_close_excludes_concurrent_native_start(monkeypatch):
    closing, release, attempted, constructed = Event(), Event(), Event(), Event()

    def synchronize(device):
        closing.set()
        assert release.wait(3)

    def server(*args):
        constructed.set()
        raise AssertionError("native server must not start while or after the owner closes")

    model, events, weights = model_fixture(monkeypatch, synchronize=synchronize, server=server)

    def start():
        assert closing.wait(3)
        attempted.set()
        model.start_server()

    def release_close():
        assert attempted.wait(3)
        try:
            assert not constructed.wait(0.05)
            assert not model.closed and not events and weights() is not None
        finally:
            release.set()

    starter, start_errors = thread_call(start)
    observer, observer_errors = thread_call(release_close)
    model.close()
    join(observer)
    join(starter)
    assert not observer_errors
    assert len(start_errors) == 1 and "closed" in str(start_errors[0])
    assert not constructed.is_set()
    assert events == ["graph", "session", "table"]
    assert model.closed and not model.resources and not model.buffers and not model.graphs
    gc.collect()
    assert weights() is None


def test_native_start_excludes_close_until_owner_handoff_finishes(monkeypatch):
    constructing, release, attempted, drained = Event(), Event(), Event(), Event()

    class Runtime:
        drained = False

        def close(self):
            self.drained = True
            drained.set()

    def server(*args):
        constructing.set()
        assert release.wait(3)
        return Runtime()

    model, events, _ = model_fixture(monkeypatch, server=server)
    starter, start_errors = thread_call(model.start_server)
    assert constructing.wait(3)

    def release_startup():
        assert attempted.wait(3)
        try:
            assert not drained.wait(0.05)
            assert events == ["synchronize"]
        finally:
            release.set()

    observer, observer_errors = thread_call(release_startup)
    attempted.set()
    model.close()
    join(starter)
    join(observer)
    assert not start_errors and not observer_errors
    assert drained.is_set() and model.closed
    assert events == ["synchronize", "synchronize", "graph", "session", "table"]
    model.close()
    assert events.count("graph") == 1


def test_wrong_thread_close_preserves_runtime_and_all_owners(monkeypatch):
    class Runtime:
        drained = False

        def close(self):
            raise AssertionError("wrong-thread cleanup must not stop native serving")

    model, events, weights = model_fixture(monkeypatch)
    runtime = Runtime()
    model.runtime, model._consumed = runtime, True
    closer, errors = thread_call(model.close)
    join(closer)
    assert len(errors) == 1 and "preparation thread" in str(errors[0])
    assert model.runtime is runtime and not model.closed
    assert model.graphs and model.buffers and model.resources and weights() is not None
    assert not events


@pytest.mark.parametrize("raises", [False, True])
def test_unconfirmed_native_drain_retains_every_prepared_resource(monkeypatch, raises):
    class Runtime:
        drained = False

        def close(self):
            if raises:
                raise RuntimeError("CUDA stream drain failed")

    model, events, weights = model_fixture(monkeypatch)
    runtime = Runtime()
    model.runtime, model._consumed = runtime, True
    with pytest.raises(RuntimeError, match="drain"):
        model.close()
    gc.collect()
    assert not model.closed and model.runtime is runtime
    assert model.graphs and model.buffers and model.resources
    assert weights() is not None and not events


def test_failed_startup_retention_blocks_cleanup_and_second_handoff(monkeypatch):
    error = RuntimeError("native startup failed without confirmed drain")
    error.resources_retained = True

    def server(*args):
        raise error

    model, events, weights = model_fixture(monkeypatch, server=server)
    with pytest.raises(RuntimeError) as raised:
        model.start_server()
    assert raised.value is error and model.consumed
    with pytest.raises(RuntimeError, match="remain retained"):
        model.close()
    with pytest.raises(RuntimeError, match="consumed or closed"):
        model.start_server()
    assert not model.closed and weights() is not None
    assert model.graphs and model.buffers and model.resources
    assert events == ["synchronize"]


def test_confirmed_drain_releases_resources_even_when_runtime_reports_error(monkeypatch):
    failure = RuntimeError("model worker reported a device completion error")

    class Runtime:
        drained = True

        def close(self):
            raise failure

    model, events, weights = model_fixture(monkeypatch)
    model.runtime, model._consumed = Runtime(), True
    with pytest.raises(RuntimeError) as raised:
        model.close()
    assert raised.value is failure
    assert model.closed and model.runtime is None
    assert not model.resources and not model.buffers and not model.graphs
    assert events == ["synchronize", "graph", "session", "table"]
    gc.collect()
    assert weights() is None
    model.close()
    assert events.count("graph") == 1


def test_context_manager_preserves_primary_failure_when_cleanup_is_undrained(monkeypatch):
    model, events, weights = model_fixture(monkeypatch)
    model._startup_undrained = True
    primary = ValueError("preparation consumer failed")
    with pytest.raises(ValueError) as raised, model:
        raise primary
    assert raised.value is primary
    assert any("cleanup failed" in note and "remain retained" in note for note in primary.__notes__)
    assert not events and weights() is not None and not model.closed


def test_drained_owner_can_retry_incomplete_resource_cleanup(monkeypatch):
    class Runtime:
        drained = True

        def close(self):
            pass

    class Session(Resource):
        calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("preparation resource cleanup failed once")
            super().close()

    model, events, weights = model_fixture(monkeypatch)
    model.session = Session("session", events)
    model.runtime, model._consumed = Runtime(), True
    with pytest.raises(RuntimeError, match="failed once"):
        model.close()
    assert not model.closed and model.resources and weights() is not None
    assert "table" not in events
    model.close()
    assert model.closed and model.runtime is None
    assert events.count("table") == 1
    assert not model.resources and not model.graphs and not model.buffers
    gc.collect()
    assert weights() is None


@pytest.mark.parametrize("fails", [False, True])
def test_peer_resources_release_only_after_native_rank_drains(monkeypatch, fails):
    def serve_rank(owner, socket, *, rank, world):
        owner._consumed = True
        assert (socket, rank, world) == ("peer socket", 1, 2)
        if fails:
            raise RuntimeError("peer drain failed")

    monkeypatch.setattr("ignition.preparation.execution.PreparedExecution.serve_rank", serve_rank)
    model, events, weights = model_fixture(monkeypatch)
    if fails:
        with pytest.raises(RuntimeError, match="peer drain failed"):
            model.serve_rank("peer socket", rank=1, world=2)
        with pytest.raises(RuntimeError, match="must retain"):
            model.close()
        assert not events and not model.closed and weights() is not None
    else:
        model.serve_rank("peer socket", rank=1, world=2)
        model.close()
        assert model.closed and events == ["synchronize", "graph", "session", "table"]
        gc.collect()
        assert weights() is None


def test_snapshot_budget_covers_the_selected_speculative_layout():
    from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig
    from hyperflare_recipes.recipes.qwen38.state import HybridStateLayout

    ordinary = HybridStateLayout.build().row_bytes
    speculative = HybridStateLayout.build(speculative_tokens=7).row_bytes
    assert speculative > ordinary
    PreparationConfig(snapshot_budget_bytes=ordinary)
    with pytest.raises(ValueError, match="at least one state row"):
        PreparationConfig(mtp=True, snapshot_budget_bytes=ordinary)
    config = PreparationConfig(mtp=True,
                               snapshot_budget_bytes=speculative)
    assert config.snapshot_budget_bytes == speculative


def test_moe_decode_a16_requires_separate_graph_roles_and_aligned_experts():
    from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig

    assert PreparationConfig(force_moe_decode_a16=True).force_moe_decode_a16
    assert PreparationConfig(devices=(0, 1), force_moe_decode_a16=True).force_moe_decode_a16
    with pytest.raises(ValueError, match="separate prefill and decode graphs"):
        PreparationConfig(force_moe_decode_a16=True, mixed_prefill=True)
    with pytest.raises(ValueError, match="intermediate width divisible by 64"):
        PreparationConfig(devices=(0, 1, 2, 3), force_moe_decode_a16=True)


def test_prepared_model_retains_persistent_storage_until_confirmed_drain(monkeypatch):
    from ignition.preparation.persistent import _ACTIVE_STORAGE

    storage = Resource("persistent storage", [])
    token = _ACTIVE_STORAGE.set(storage)
    try:
        model, events, _ = model_fixture(monkeypatch)
    finally:
        _ACTIVE_STORAGE.reset(token)
    storage.events = events
    model._startup_undrained = True
    with pytest.raises(RuntimeError, match="resources remain retained"):
        model.close()
    assert not events
    model._startup_undrained = False
    model.close()
    assert events == ["synchronize", "graph", "session", "table", "persistent storage"]
    model.close()
    assert events.count("persistent storage") == 1


def test_qwen38_decoder_routes_ple_barrier_to_barrier_runtime(monkeypatch):
    """PreparedDecoder keeps PLE's 16-byte readiness barrier on barrier_runtime when layer reductions use DMA."""
    import hyperflare_recipes.components.embedding as embedding_mod
    import hyperflare_recipes.components.gated_delta as gdn_mod
    import hyperflare_recipes.components.vocabulary as vocab_mod
    import hyperflare_recipes.recipes.qwen38.attention as attention_mod
    import hyperflare_recipes.recipes.qwen38.decoder as decoder_mod
    import hyperflare_recipes.recipes.qwen38.hyperconnection as hc_mod
    import hyperflare_recipes.recipes.qwen38.moe as moe_mod
    import hyperflare_recipes.recipes.qwen38.ple as ple_mod
    import ignition.components.collectives as collectives_mod

    seen = {"hc": [], "reduce": [], "ple": [], "attn": []}

    class DummyOp:
        def __init__(self, *args, **kwargs):
            self.scratch = SimpleNamespace(output=torch.zeros((1, 1)))
            self.output = torch.zeros((1, 1))

        def prepare_combine_norm(self, *args, **kwargs):
            pass

    def fake_hc(*args, **kwargs):
        seen["hc"].append(kwargs.get("runtime"))
        return DummyOp()

    def fake_reduce(*args, **kwargs):
        seen["reduce"].append(kwargs.get("runtime"))
        return DummyOp()

    def fake_ple(*args, **kwargs):
        seen["ple"].append(kwargs.get("runtime"))
        return DummyOp()

    def fake_attn(*args, **kwargs):
        seen["attn"].append((kwargs.get("qsa_gather_runtime"), kwargs.get("qsa_shard_mode")))
        return DummyOp()

    monkeypatch.setattr(embedding_mod, "PreparedEmbedding", DummyOp)
    monkeypatch.setattr(hc_mod, "PreparedHyperconnection", fake_hc)
    monkeypatch.setattr(attention_mod, "PreparedAttention", fake_attn)
    monkeypatch.setattr(gdn_mod, "PreparedGDN", DummyOp)
    monkeypatch.setattr(moe_mod, "PreparedQwenMoe", DummyOp)
    monkeypatch.setattr(ple_mod, "PreparedPLE", fake_ple)
    monkeypatch.setattr(collectives_mod, "PreparedAllReduce", fake_reduce)
    monkeypatch.setattr(vocab_mod, "PreparedVocabularyProjection", DummyOp)

    def make_layer():
        ple = SimpleNamespace(
            key_proj=SimpleNamespace(weight=None),
            value_proj=SimpleNamespace(weight=None),
        )
        return SimpleNamespace(
            attn_hyper_connection=object(),
            mlp_hyper_connection=object(),
            self_attn=object(),
            linear_attn=object(),
            ple=ple,
        )

    layers = [make_layer() for _ in range(48)]
    language = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=None),
        layers=layers,
        hyper_connection_mixer=object(),
    )
    weights = SimpleNamespace(
        tree=SimpleNamespace(model=SimpleNamespace(language_model=language), lm_head=SimpleNamespace(weight=None)),
        layers=[SimpleNamespace(attention={}, moe=None) for _ in range(48)],
    )
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class DummyHCScratch:
        block_input: torch.Tensor

    scratch = SimpleNamespace(
        embedding=torch.zeros((4, 2560)),
        residual=torch.zeros((4, 10240)),
        hidden=torch.zeros((4, 2560)),
        sample_hidden=torch.zeros((1, 2560)),
        logits=torch.zeros((1, 16)),
        local_logits=torch.zeros((1, 8)),
        hyperconnection=DummyHCScratch(block_input=torch.zeros((4, 2560))),
        gdn=object(),
        qsa=object(),
        ple=object(),
        moe=object(),
    )
    state = SimpleNamespace(
        tokens=4,
        rows=1,
        mode="prefill",
        ple=SimpleNamespace(token_ids=None),
        gdn=SimpleNamespace(num_tokens=None),
        qsa=None,
        pool=SimpleNamespace(
            layout=SimpleNamespace(speculative_tokens=0),
            gdn={i: None for i in range(48) if i % 4 != 3},
            ple_convolution=None,
        ),
        prepare_hidden=lambda *args: None,
        prepare_residual=lambda *args: None,
    )
    qsa_states = {i: None for i in range(3, 48, 4)}
    dma_rt = object()
    pcie_rt = object()
    vocab_rt = object()
    qsa_gather_rt = object()

    decoder_mod.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="prefill1024", verification=True,
        runtime=dma_rt, barrier_runtime=pcie_rt, vocabulary_runtime=vocab_rt,
        qsa_gather_runtime=qsa_gather_rt, qsa_shard_mode="row",
    )
    assert seen["ple"] == [pcie_rt]
    assert len(seen["reduce"]) == 96 and all(rt is dma_rt for rt in seen["reduce"])
    assert len(seen["hc"]) == 97 and all(rt is dma_rt for rt in seen["hc"])
    assert seen["attn"] == [(qsa_gather_rt, "row")] * 12

    for bucket in seen.values():
        bucket.clear()
    decoder_mod.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="prefill512", verification=True,
        runtime=pcie_rt, vocabulary_runtime=vocab_rt,
    )
    assert seen["ple"] == [pcie_rt]
    assert len(seen["reduce"]) == 96 and all(rt is pcie_rt for rt in seen["reduce"])
    assert seen["attn"] == [(None, None)] * 12


def _tp2_dma_and_barrier_worker(rank, world, port, error_queue):
    import os
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.readiness import PreparedReadinessWait, wait_for_readiness
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
        create_dma_all_reduce_runtime,
    )
    from ignition.preparation.tuning import RecordedPreparationSession

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.group.WORLD
    pcie_rt = dma_rt = session = None
    try:
        coordinator = CollectivePreparation(group)
        pcie_rt = create_all_reduce_runtime(
            group, device, max_tokens=1024, hidden_size=2560, collective_tuning="static"
        )
        dma_rt = create_dma_all_reduce_runtime(group, device, max_tokens=1024, hidden_size=2560)
        session = RecordedPreparationSession(
            device=device,
            checkpoint="test_qwen38_dma_and_barrier",
            workload_schema="test_qwen38_dma_and_barrier",
            numerical_policy="default",
            coordinator=coordinator,
            autotune_effort="off",
        )
        session.__enter__()

        from ignition.components.readiness import ReadinessTimings

        frame = torch.zeros(1, dtype=torch.int32, device=device)
        timings = ReadinessTimings.allocate(frame, 1)
        # Verify DmaAllReduce rejects the 16-byte PLE readiness barrier while PCIeAllReduce accepts it.
        with pytest.raises(ValueError, match="alignment"):
            PreparedReadinessWait(timings, 0, runtime=dma_rt, session=session, coordinator=coordinator, name="bad")
        wait = PreparedReadinessWait(
            timings, 0, runtime=pcie_rt, session=session, coordinator=coordinator, name="ple.readiness"
        )
        source = torch.empty((1024, 2560), dtype=torch.bfloat16, device=device)
        output = torch.empty_like(source)
        reduce = PreparedAllReduce(
            source, output, runtime=dma_rt, session=session, coordinator=coordinator, name="layer.reduce"
        )
        session.tune(show_progress=False)

        torch.manual_seed(700 + rank)
        source.copy_(torch.randn_like(source))
        local_f32 = source.float().cpu()
        gathered = [torch.empty_like(local_f32) for _ in range(world)]
        dist.all_gather(gathered, local_f32, group=group)
        expected = sum(gathered).bfloat16().to(device)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        stream = torch.cuda.Stream(device=device)
        b_in, b_out = wait.operands(source)
        with torch.cuda.stream(stream):
            wait_for_readiness(source, event.cuda_event, timings.timestamps, frame, 0, b_in, b_out, wait.handle)
            reduce(source)
            stream.synchronize()
            dist.barrier(group)
            graph = torch.cuda.CUDAGraph()
            with pcie_rt.capture(stream=stream), torch.cuda.graph(graph, stream=stream):
                wait_for_readiness(source, event.cuda_event, timings.timestamps, frame, 0, b_in, b_out, wait.handle)
                reduce(source)
            stream.synchronize()
            output.zero_()
            dist.barrier(group)
            graph.replay()
            stream.synchronize()
            graph.reset()
        assert torch.equal(output, expected)
    except BaseException as exc:  # noqa: BLE001
        error_queue.put((rank, repr(exc)))
        raise
    finally:
        if session is not None:
            session.close()
        if dma_rt is not None:
            dma_rt.close()
        if pcie_rt is not None:
            pcie_rt.close()
        dist.destroy_process_group()


def test_qwen38_prefill_dma_all_reduce_coexists_with_ple_readiness_barrier_tp2():
    """TP2 CUDA graph combines a PCIeAllReduce PLE readiness barrier with DmaAllReduce prefill reductions."""
    import os
    import socket
    import torch.multiprocessing as mp

    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1 and FLARE_RUN_TP_CUDA_TESTS=1")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    mp.spawn(_tp2_dma_and_barrier_worker, args=(2, port, errors), nprocs=2, join=True)
    assert errors.empty(), errors.get()
