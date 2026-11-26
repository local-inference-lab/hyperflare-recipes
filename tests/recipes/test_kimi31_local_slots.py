# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from types import SimpleNamespace
import sys

from hyperflare_recipes.recipes.kimi31.slot_tuning import slot_inputs


@pytest.mark.parametrize("available,supported,agreed,expected", (
    (False, True, True, False), (True, False, True, False),
    (True, True, False, False), (True, True, True, True)))
def test_all_ranks_require_the_optional_slot_provider(monkeypatch, available, supported, agreed, expected):
    from hyperflare_recipes.recipes.kimi31 import local_slots

    observed = []
    def agree(value):
        observed.append(value)
        return agreed

    context = SimpleNamespace(candidates=object(), coordinator=SimpleNamespace(same_on_every_rank=agree))
    monkeypatch.setattr(local_slots, "find_spec", lambda _: object() if available else None)
    monkeypatch.setitem(sys.modules, "b12x.comm.pcie.interleaved_topk",
                        SimpleNamespace(supports_channel=lambda _: supported))
    assert local_slots.supports_local_slots(context) == expected
    assert observed == [available and supported]


def test_compiler_metadata_keeps_only_the_recording_rank_handle():
    from ignition.preparation import forward_metadata
    from hyperflare_recipes.recipes.kimi31 import local_slots

    class OnlyOnRecorder:
        def __reduce__(self):
            raise AssertionError("compiler metadata reached a recording rank's provider")

    choice = local_slots.PreparedLocalSlots.__new__(local_slots.PreparedLocalSlots)
    choice.handle, choice.enabled = 173, True
    choice.geometry = local_slots.SlotGeometry(130, 130, 0)
    local_slots._PLANS[choice] = OnlyOnRecorder()
    worker = forward_metadata.loads(forward_metadata.dumps(choice))
    assert (worker.geometry, worker.handle, worker.enabled) == (choice.geometry, 173, True)
    assert worker not in local_slots._PLANS
    assert isinstance(choice.plan, OnlyOnRecorder)


@pytest.mark.parametrize("length", (0, 512, 2048, 8192, 65536, 262144))
@pytest.mark.parametrize("rank", (0, 7, 15))
def test_slot_trials_preserve_the_visible_local_indexer_prefix(rank, length):
    indices, scores, positions = slot_inputs(rank, 401, length)
    assert indices.shape == scores.shape == (4, 2048)
    assert indices.dtype == torch.int32 and scores.dtype == torch.float32
    for row, position in enumerate(positions.tolist()):
        visible = max(0, (position + 16 - rank) // 16)
        capacity = min(visible, 2048)
        assert (indices[row, capacity:] == -1).all()
        assert torch.isneginf(scores[row, capacity:]).all()
        valid = indices[row] >= 0
        assert (indices[row, valid] * 16 + rank <= position).all()
        assert torch.unique(indices[row, valid]).numel() == valid.sum()
    if length > 2048:
        assert torch.isnan(scores).any() and torch.isposinf(scores).any()
        assert (positions[-1] < 0).item()


def test_tune_local_slots_batches_capture_forwards_with_unique_names(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from hyperflare_recipes.recipes.kimi31 import local_slots, slot_tuning

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

    class FakeTrial:
        def __init__(self, context, reduction, geometry, control, session, name, *, defer_patterns=False):
            self.context, self.geometry, self.control = context, geometry, control
            self.inputs = (
                torch.zeros((geometry.rows, 4), dtype=torch.int32),
                torch.zeros((geometry.rows, 4), dtype=torch.float32),
                torch.zeros((geometry.rows,), dtype=torch.int64),
                torch.zeros((geometry.rows, geometry.table_columns), dtype=torch.int32),
            )
            self.output = (
                torch.zeros((geometry.rows, 4), dtype=torch.int32),
                torch.zeros((geometry.rows, 4), dtype=torch.int32),
                torch.zeros((geometry.rows,), dtype=torch.int32),
            )
            self.merge = SimpleNamespace(plans=lambda impl: (("merge", f"m-{geometry}-{impl}"),))
            self.local = SimpleNamespace(plan=f"l-{geometry}", handle=1)
            self.values = {}
            self.patterns = [] if defer_patterns else [self._mk(i) for i in range(4)]
            self.expected = []
            self.index = 0
            self.prepared_count = 0
            self.released_count = 0

        def _mk(self, index):
            return (
                torch.full((self.geometry.rows, 4), index, dtype=torch.int32),
                torch.zeros((self.geometry.rows, 4), dtype=torch.float32),
                torch.zeros((self.geometry.rows,), dtype=torch.int64),
            )

        def prepare_patterns(self):
            self.prepared_count += 1
            if not self.patterns:
                self.patterns = [self._mk(i) for i in range(4)]

        def release_patterns(self):
            self.released_count += 1
            self.patterns.clear()
            self.expected.clear()

        def produce(self, index):
            self.index = index
            for dst, src in zip(self.inputs[:3], self.patterns[index], strict=True):
                dst.copy_(src)

        def reset(self):
            for tensor in self.output:
                tensor.fill_(-131)

        def validate(self):
            for observed, expected in zip(self.output, self.expected[self.index], strict=True):
                assert torch.equal(observed, expected)

        def plans(self, kind):
            return self.merge.plans(self.control) if kind == "global" else (("local-slots", self.local.plan),)

        def resident_bytes(self):
            return 1024 * (1 + len(self.patterns))

    g1 = local_slots.SlotGeometry(cache_pages=130, table_columns=64, table_stride=0, rows=4)
    g2 = local_slots.SlotGeometry(cache_pages=130, table_columns=64, table_stride=64, rows=4)
    g3 = local_slots.SlotGeometry(cache_pages=130, table_columns=64, table_stride=64, rows=8)
    bindings = {}
    for geom, ctrl in ((g1, "replicated"), (g2, "owner"), (g3, "owner")):
        local_obj = SimpleNamespace(plan=f"l-{geom}", enabled=False)
        merge_obj = SimpleNamespace(implementation=ctrl, plans=lambda impl, g=geom: (("merge", f"m-{g}-{impl}"),))
        bindings[geom] = [(local_obj, merge_obj)]

    capture_calls = []
    all_graphs = []
    all_compiled = []

    trials_seen = []

    class RecordingFakeTrial(FakeTrial):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trials_seen.append(self)

    def fake_capture(forwards, signatures, **kwargs):
        capture_calls.append((list(forwards), {k: v.token_capacity for k, v in signatures.items()}))
        assert len(set(forwards)) == len(forwards)
        assert all(len(trial.patterns) == 0 for trial in trials_seen)
        graphs, compiled = {}, {}
        for name, fwd in forwards.items():
            g = FakeGraph()
            c = FakeCompiled(lambda out=fwd.output: tuple(t.zero_() for t in out))
            graphs[name] = g
            compiled[name] = c
            all_graphs.append(g)
            all_compiled.append(c)
        return graphs, compiled

    def fake_race(candidates, **kwargs):
        assert sum(1 for trial in trials_seen if trial.patterns) == 1
        return {"implementation": "local", "source": "tuned", "rejected_candidates": 0}

    monkeypatch.setattr(slot_tuning, "SlotTrial", RecordingFakeTrial)
    monkeypatch.setattr(slot_tuning, "capture_forwards", fake_capture)
    monkeypatch.setattr(slot_tuning, "plan_identity", lambda plan: plan)
    monkeypatch.setattr(slot_tuning, "race_compositions", fake_race)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    class FakeTrials:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def tune(self, show_progress=False):
            pass

        def _tuning_clocks(self):
            return nullcontext()

    monkeypatch.setattr(slot_tuning, "trial_session", lambda *args, **kwargs: FakeTrials())

    coordinator = SimpleNamespace(ranks=(0, 1), barrier=lambda name, ranks: None)
    context = SimpleNamespace(
        shards=SimpleNamespace(index=torch.zeros(1, device="cpu"), world=16, rank=0),
        coordinator=coordinator,
    )
    registry = SimpleNamespace(context=context, reduction=None, slot_groups=bindings, cache_root=None)
    session = SimpleNamespace()

    results = slot_tuning.tune_local_slots(registry, session)
    assert len(capture_calls) == 1
    assert len(capture_calls[0][0]) == 6
    assert list(capture_calls[0][1].values()) == [4, 4, 4, 4, 8, 8]
    assert len(results) == 3
    assert all(local_obj.enabled for group in bindings.values() for local_obj, _ in group)
    assert all(g.reset_calls == 1 for g in all_graphs)
    assert all(c.retired == 1 for c in all_compiled)
    assert all(trial.prepared_count == 1 and trial.released_count == 1 and not trial.patterns for trial in trials_seen)
