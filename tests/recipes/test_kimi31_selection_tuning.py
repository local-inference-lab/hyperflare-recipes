# SPDX-License-Identifier: Apache-2.0
"""Batch-level composition race sessions and clock locks for Kimi sparse selection tuning."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from hyperflare_recipes.recipes.kimi31 import selection_tuning


def test_candidate_selection_tuning_shares_one_race_session_and_clock_lock(monkeypatch):
    class FakeGraph:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class FakeCompiled:
        def __init__(self, fn):
            self.fn = fn
            self.retired = 0

        def __call__(self):
            return self.fn()

        def retire_after_capture(self):
            self.retired += 1

    class FakeFixture:
        def __init__(self, rows, context, reduction, session, alternatives):
            self.rows = rows
            self.values = {}
            self.output = torch.zeros((rows, 4), dtype=torch.int32)
            self.selection = SimpleNamespace(
                alternatives=alternatives,
                plans=lambda kind: (("exchange", f"plan-{rows}-{kind}"),),
            )
            self.forwards = {kind: (lambda: None) for kind in alternatives}
            self.patterns = [index for index in range(4)]
            self.expected = []
            self.step = 0

        def produce(self, index):
            self.step = index

        def validate(self):
            assert torch.equal(self.output, self.expected[self.step])

        @property
        def nbytes(self):
            return 1024

    all_graphs = []
    all_compiled = []

    def fake_capture(forwards, signatures, **kwargs):
        graphs, compiled = {}, {}
        for name in forwards:
            g = FakeGraph()
            c = FakeCompiled(lambda: None)
            graphs[name] = g
            compiled[name] = c
            all_graphs.append(g)
            all_compiled.append(c)
        return graphs, compiled

    race_sessions_opened = []
    race_sessions_seen = []
    shared_race_session = object()

    def fake_race_session(sess, *, coordinator, cache_policy):
        race_sessions_opened.append((sess, coordinator, cache_policy))
        return nullcontext(shared_race_session)

    def fake_race(candidates, *, race_session=None, **kwargs):
        race_sessions_seen.append(race_session)
        return {"implementation": "owner", "source": "cached", "rejected_candidates": 0}

    clock_locks = []

    class FakeTrials:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def tune(self, show_progress=False):
            pass

        def _tuning_clocks(self):
            clock_locks.append(1)
            return nullcontext()

    fake_trials = FakeTrials()
    monkeypatch.setattr(selection_tuning, "_Fixture", FakeFixture)
    monkeypatch.setattr(selection_tuning, "capture_forwards", fake_capture)
    monkeypatch.setattr(selection_tuning, "plan_identity", lambda plan: plan)
    monkeypatch.setattr(selection_tuning, "composition_race_session", fake_race_session)
    monkeypatch.setattr(selection_tuning, "race_compositions", fake_race)
    monkeypatch.setattr(selection_tuning, "trial_session", lambda *args, **kwargs: fake_trials)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    coordinator = SimpleNamespace(ranks=(0, 1), barrier=lambda name, ranks: None)
    context = SimpleNamespace(
        shards=SimpleNamespace(index=torch.zeros(1, device="cpu"), world=16, rank=0),
        coordinator=coordinator,
    )
    registry = selection_tuning.CandidateSelectionTuning(context, reduction=None)

    bound = {}
    for rows in (1, 4, 8):
        target = SimpleNamespace(
            rows=rows,
            alternatives=("replicated", "owner"),
            plans=lambda kind, r=rows: (("exchange", f"plan-{r}-{kind}"),),
            bind=lambda impl, r=rows: bound.setdefault(r, impl),
        )
        registry.register(target)

    report = registry.tune(SimpleNamespace())
    assert [entry["rows"] for entry in report] == [1, 4, 8]
    assert bound == {1: "owner", 4: "owner", 8: "owner"}
    assert race_sessions_opened == [(fake_trials, coordinator, "warm")]
    assert race_sessions_seen == [shared_race_session] * 3
    assert len(clock_locks) == 1
    assert all(g.reset_calls == 1 for g in all_graphs)
    assert all(c.retired == 1 for c in all_compiled)
