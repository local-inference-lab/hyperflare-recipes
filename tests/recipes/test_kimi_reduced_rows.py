# SPDX-License-Identifier: Apache-2.0
"""Map reduced attention and MLP outputs to the residual update that consumes them."""

import pytest
from hyperflare_recipes.recipes.kimi3.row_reductions import hidden_row_choices, RowGeometry


def test_layer_consumers_write_the_numbered_block_and_dspark_tap():
    class Catalog:
        def register(self, reduction, geometry):
            return geometry

    taps = (1, 13, 26, 40, 52)
    pairs = hidden_row_choices(Catalog(), [(None, None, None, None)] * 93, 12, taps, 8)
    assert len(pairs) == 93
    assert pairs[0][0] == RowGeometry(7168, sources=1, mode=2, blocks=8)
    assert pairs[0][1] == RowGeometry(7168, sources=1, mode=1, blocks=8, taps=5)
    assert pairs[11][1] == RowGeometry(7168, sources=1, write=1, mode=1, blocks=8)
    assert pairs[12][0] == RowGeometry(7168, sources=2, mode=2, blocks=8)
    assert pairs[-1][1] == RowGeometry(7168, sources=8, mode=1, blocks=8)
    assert sum(g.taps > 0 for _, g in pairs) == len(taps)


@pytest.mark.parametrize("geometry", [RowGeometry(3584), RowGeometry(7168, sources=1, mode=1, blocks=8, taps=5)])
def test_compiler_workers_receive_operands_and_handles_without_cuda_resources(geometry):
    from types import SimpleNamespace
    from cuda.bindings.driver import CUstream
    from ignition.preparation.forward_metadata import dumps, loads
    from hyperflare_recipes.recipes.kimi3.row_reductions import (
        RowReductionChoice,
        _BINDINGS,
        _PreparedTrial,
        _ReductionFixture,
    )
    import torch

    control = SimpleNamespace(handle=41, plan=None, stream=CUstream(0))
    choice = RowReductionChoice(SimpleNamespace(parts=[control]), geometry)
    _BINDINGS[choice].layout = SimpleNamespace(
        owner=SimpleNamespace(device=torch.device("cpu"), rank=0, stream=CUstream(0))
    )
    fixture = _ReductionFixture(choice, 1e-5)
    for fused in (True, False):
        traced = loads(dumps(_PreparedTrial(fixture, fused)))
        assert traced.choice.control.handle == 41
        assert traced.source.shape == (4, geometry.width)
        from torch.fx.experimental.proxy_tensor import make_fx

        graph = make_fx(traced, tracing_mode="fake", _allow_non_fake_inputs=True)()
        names = [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]
        if fused:
            expected = (
                "flare.kimi3_reduce_norm_columns.default"
                if geometry.width == 3584
                else "flare.kimi3_reduce_residual_mix.default"
            )
            assert expected in names and "flare.pcie_all_reduce.default" not in names
        else:
            assert "flare.pcie_all_reduce.default" in names and "triton_kernel_wrapper_mutation" in names


def test_row_reduction_tuning_batches_capture_forwards_and_defers_patterns(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from hyperflare_recipes.recipes.kimi3 import row_reductions

    class FakeGraph:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class FakeCompiled:
        def __init__(self, name, fn):
            self.name = name
            self.fn = fn
            self.retired = 0
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return self.fn()

        def retire_after_capture(self):
            self.retired += 1

    geometries = (
        RowGeometry(3584, rows=4),
        RowGeometry(7168, sources=1, mode=2, blocks=8, rows=4),
        RowGeometry(7168, sources=4, mode=1, blocks=8, taps=5, rows=8, addend=True),
    )
    choices = []
    grouped = []
    for idx, geom in enumerate(geometries):
        control = SimpleNamespace(handle=10 + idx, plan=f"plan-{idx}", stream=None)
        choice = row_reductions.RowReductionChoice(SimpleNamespace(parts=[control]), geom)
        row_reductions._BINDINGS[choice].layout = SimpleNamespace(
            owner=SimpleNamespace(device=torch.device("cpu"), rank=0, stream=None)
        )
        choices.append(choice)
        grouped.append(((geom, (f"ctrl-{idx}",)), [choice]))

    capture_calls = []
    all_graphs = []
    all_compiled = []

    fixtures_seen = []
    orig_fixture = row_reductions._ReductionFixture

    class TrackingFixture(orig_fixture):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            fixtures_seen.append(self)

    monkeypatch.setattr(row_reductions, "_ReductionFixture", TrackingFixture)

    def fake_capture(forwards, signatures, **kwargs):
        capture_calls.append((list(forwards), {k: v.token_capacity for k, v in signatures.items()}))
        # Deferred patterns keep pattern tensors off device during batched capture.
        assert all(len(fixture.patterns) == 0 for fixture in fixtures_seen)
        for fn in forwards.values():
            assert fn.source.isfinite().all()
        graphs = {}
        compiled = {}
        for name, fn in forwards.items():
            g = FakeGraph()
            c = FakeCompiled(name, lambda f=fn: f.out.zero_())
            graphs[name] = g
            compiled[name] = c
            all_graphs.append(g)
            all_compiled.append(c)
        return graphs, compiled

    raced_ops = []
    race_sessions_opened = []
    race_sessions_seen = []
    clock_locks_entered = 0
    shared_race_session = object()

    def fake_race_session(sess, *, coordinator, cache_policy):
        race_sessions_opened.append((sess, coordinator, cache_policy))
        return nullcontext(shared_race_session)

    def fake_clock_lock():
        nonlocal clock_locks_entered
        clock_locks_entered += 1
        return nullcontext()

    def fake_race(candidates, *, operation, geometry, resident_bytes, race_session=None, **kwargs):
        assert set(candidates) == {"separate", "fused"}
        assert resident_bytes > 0
        # Only the active group holds materialized patterns during its race.
        assert sum(1 for fixture in fixtures_seen if fixture.patterns) == 1
        race_sessions_seen.append(race_session)
        raced_ops.append((operation, geometry["rows"], resident_bytes))
        return {"implementation": "fused", "source": "tuned", "rejected_candidates": 0}

    monkeypatch.setattr(row_reductions, "capture_forwards", fake_capture)
    monkeypatch.setattr(row_reductions, "composition_race_session", fake_race_session)
    monkeypatch.setattr(row_reductions, "race_compositions", fake_race)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    coordinator = SimpleNamespace(ranks=(0, 1), barrier=lambda name, ranks: None)
    catalog = row_reductions.RowReductionTuning(coordinator, torch.device("cpu"), 1e-5, None)
    monkeypatch.setattr(catalog, "groups", lambda: grouped)
    session = SimpleNamespace(autotune=True, autotune_effort="default", _tuning_clocks=fake_clock_lock)

    report = catalog.tune(session)
    assert len(capture_calls) == 1
    assert len(capture_calls[0][0]) == 2 * len(geometries)
    assert list(capture_calls[0][1].values()) == [4, 4, 4, 4, 8, 8]
    assert len(report) == len(geometries)
    assert [entry["geometry"]["rows"] for entry in report] == [4, 4, 8]
    assert race_sessions_opened == [(session, coordinator, "warm")]
    assert race_sessions_seen == [shared_race_session] * len(geometries)
    assert clock_locks_entered == 1
    assert all(choice.fused for choice in choices)
    assert all(g.reset_calls == 1 for g in all_graphs)
    assert all(c.retired == 1 for c in all_compiled)
    assert all(not fixture.patterns and not fixture.expected for fixture in fixtures_seen)


def test_batch_groups_splits_deterministically_at_budget():
    from hyperflare_recipes.recipes.kimi3.row_reductions import _batch_groups, _fixture_live_bytes

    g_small = RowGeometry(3584, rows=4)
    g_large = RowGeometry(7168, sources=4, mode=1, blocks=8, rows=64, addend=True)
    entries = [((g_small, ("a",)), []), ((g_large, ("b",)), []), ((g_large, ("c",)), [])]
    budget = _fixture_live_bytes(g_small) + _fixture_live_bytes(g_large)
    batches = _batch_groups(entries, budget_bytes=budget)
    assert len(batches) == 2
    assert len(batches[0]) == 2 and len(batches[1]) == 1
    assert _batch_groups([], budget_bytes=budget) == []
