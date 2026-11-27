# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU parity, sizing, fallback, and FX-tracing tests for DeepSeek V4.1 4x local-head Engram staging."""

from types import SimpleNamespace

import pytest
import torch
from b12x.comm import pcie
from b12x.sequence import engram as b12x_engram
from b12x.sequence.engram.reference import hash_reference
from torch.fx.experimental.proxy_tensor import make_fx

from hyperflare_recipes.components.embedding_partition import EmbeddingHeadPartition
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.deepseek41 import engram as ds_engram
from ignition.components import engram as core_engram
from ignition.components.engram import EngramInputs
from ignition.preparation.runtime_core import FileSource


class _FakeDmaRuntime:
    def __init__(self, world_size: int = 4, *, max_bytes: int = 256 << 20, shard_capacity: int = 64 << 20):
        self.world_size = world_size
        self.max_bytes = max_bytes
        self.shard_capacity = shard_capacity

    def should_allreduce(self, tensor):
        return True

    def should_all_gather(self, tensor):
        return True


def _dequantize_staged_cpu(weight: torch.Tensor, scales: torch.Tensor, hashes: torch.Tensor,
                           count: torch.Tensor, rows: int) -> torch.Tensor:
    """Bit-exact CPU model of staged FP8 E4M3 * E8M0 -> BF16 dequantization."""
    tokens, heads = hashes.shape
    nt = int(count.item())
    if not 0 <= nt <= tokens:
        return torch.zeros((tokens, heads * 256), dtype=torch.bfloat16)
    q = weight.view(torch.float8_e4m3fn).float().view(tokens, heads, 256)
    exp = scales.to(torch.int32).view(tokens, heads, 8)
    scale = (exp << 23).view(torch.float32)
    scale = torch.where(exp == 0, torch.full_like(scale, 2.0**-127), scale)
    scale = torch.where(exp == 255, torch.full_like(scale, float("nan")), scale)
    deq = (q * scale.repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    local = (torch.arange(tokens)[:, None] < nt) & (hashes >= 0) & (hashes < rows)
    return torch.where(local[:, :, None], deq, torch.zeros_like(deq)).reshape(tokens, heads * 256)


def _simulate_helios_gather(plane_weights: torch.Tensor, plane_scales: torch.Tensor,
                            mailbox: torch.Tensor, num_entries: int):
    """Exact replica of Helios reader::Reader::gather: valid row -> copy, -1 -> zero."""
    ids = mailbox[:num_entries]
    w_out = torch.zeros((num_entries, 256), dtype=torch.uint8)
    s_out = torch.zeros((num_entries, 8), dtype=torch.uint8)
    valid = ids >= 0
    if valid.any():
        w_out[valid] = plane_weights[ids[valid]]
        s_out[valid] = plane_scales[ids[valid]]
    return w_out, s_out


def _make_tables_pair(tmp_path, geometry, world: int, rank: int, tokens: int, *, compact: bool):
    rows = geometry.num_embeddings[0]
    staged_heads = 24 // world if (compact and world > 1) else 24
    timings = SimpleNamespace(
        timestamps=torch.zeros((2, 3), dtype=torch.int64),
        frame=torch.zeros((1,), dtype=torch.int32),
    )
    partition = EmbeddingHeadPartition.for_rank(geometry.offsets[0], rows, rank, world)
    inputs = {
        1: EngramInputs(
            weights=torch.zeros((tokens * staged_heads, 256), dtype=torch.uint8),
            scales=torch.zeros((tokens * staged_heads, 8), dtype=torch.uint8),
            hashes=torch.full((tokens, staged_heads), -1, dtype=torch.int64),
            ready=SimpleNamespace(cuda_event=777),
            timings=timings,
            interval=0,
        )
    }
    mailbox_tensor = torch.zeros((tokens * staged_heads + 1,), dtype=torch.int64)
    mailboxes = {1: SimpleNamespace(host_view=mailbox_tensor, device_view=mailbox_tensor)}
    sources = {
        1: (
            FileSource(tmp_path / "w.bin", 0, rows * 256),
            FileSource(tmp_path / "s.bin", 0, rows * 8),
        )
    }
    row_mapping = {1: torch.tensor(partition.row_mapping, dtype=torch.int64)}
    columns = torch.tensor(
        [head * 256 + col for head in range(rank, 24, world) for col in range(256)],
        dtype=torch.int64,
    )
    return ds_engram.Tables(
        geometry, tuple(range(32)), sources, inputs, mailboxes,
        torch.tensor([9], dtype=torch.int64), TensorParallelShard(rank, world),
        row_mapping, columns,
    ), partition


@pytest.mark.parametrize("world", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("tokens", [1, 8, 128, 512, 16384])
def test_tables_prepare_compacts_buffers_mailboxes_and_resident_catalog(monkeypatch, tmp_path, world, tokens):
    geometry = b12x_engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    w_path, s_path = tmp_path / "w.bin", tmp_path / "s.bin"
    w_path.write_bytes(bytes(rows * 256))
    s_path.write_bytes(bytes(rows * 8))

    manifest = SimpleNamespace(
        metadata_path=tmp_path,
        snapshot=tmp_path,
        tensors={
            "layers.1.engram.embed.weight": SimpleNamespace(shape=(rows, 256), dtype="F8_E4M3", shard="w.bin", offset=0),
            "layers.1.engram.embed.scale": SimpleNamespace(shape=(rows, 8), dtype="F8_E8M0", shard="s.bin", offset=0),
        },
    )
    config = SimpleNamespace(text={
        "engram_compressed_vocab_size": 32,
        "engram_layer_ids": [1],
        "engram_vocab_size": 101,
        "engram_num_embeddings": [rows],
    })

    import transformers
    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", lambda **kw: object())
    monkeypatch.setattr(ds_engram.engram, "build_compressed_token_map", lambda tok: (list(range(32)), 32))
    monkeypatch.setattr(ds_engram, "ReadinessTimings", SimpleNamespace(
        allocate=lambda frame, count: SimpleNamespace(
            timestamps=torch.zeros((2, 3 * count), dtype=torch.int64), frame=frame,
            buffers=lambda: {"readiness_timestamps": torch.zeros((2, 3 * count), dtype=torch.int64)},
        )
    ))

    class FakeEvent:
        def __init__(self, *, external=False):
            self.cuda_event = 11

        def record(self, stream):
            pass

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "stream")

    allocated_mailboxes = {}

    def host_alloc(name, shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        allocated_mailboxes[name] = tensor
        return SimpleNamespace(host_view=tensor, device_view=tensor)

    tables = ds_engram.Tables.prepare(
        manifest, config, tokens=tokens, device="cpu",
        sequence=torch.zeros(1, dtype=torch.int64), frame=torch.zeros(1, dtype=torch.int32),
        host_allocator=host_alloc, shard=TensorParallelShard(0, world),
    )

    compact_supported = ds_engram._supports_compact_heads()
    expected_heads = (24 // world) if compact_supported else 24
    assert tables.staged_heads == expected_heads
    assert tables.inputs[1].weights.shape == (tokens * expected_heads, 256)
    assert tables.inputs[1].scales.shape == (tokens * expected_heads, 8)
    assert tables.inputs[1].hashes.shape == (tokens, expected_heads)
    assert allocated_mailboxes["engram_1_requests"].shape == (tokens * expected_heads + 1,)
    assert tables.resident_catalog()["tables"][0]["heads"] == expected_heads
    # Canonical 24-head export metadata stays unchanged.
    assert len(tables.export_metadata()["tables"][0]["hash"]["orders"]) == 24
    if compact_supported and world == 4 and tokens == 128:
        assert tables.inputs[1].weights.nbytes + tables.inputs[1].scales.nbytes == 202_752
    if compact_supported and world == 4 and tokens == 16384:
        assert tables.inputs[1].weights.nbytes + tables.inputs[1].scales.nbytes == 25_952_256


def test_base_ignition_fallback_retains_24_head_staging(monkeypatch, tmp_path):
    """When core ignition lacks compact local-head support, recipe falls back cleanly to 24 heads."""
    geometry = b12x_engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    (tmp_path / "w.bin").write_bytes(bytes(rows * 256))
    (tmp_path / "s.bin").write_bytes(bytes(rows * 8))

    class LegacyEngramInputs:
        @classmethod
        def allocate(cls, tokens, device, *, timings, interval=0):
            return EngramInputs(
                weights=torch.zeros((tokens * 24, 256), dtype=torch.uint8, device=device),
                scales=torch.zeros((tokens * 24, 8), dtype=torch.uint8, device=device),
                hashes=torch.full((tokens, 24), -1, dtype=torch.int64, device=device),
                ready=SimpleNamespace(cuda_event=5),
                timings=timings,
                interval=interval,
            )

    monkeypatch.setattr(ds_engram, "EngramInputs", LegacyEngramInputs)
    assert ds_engram._supports_compact_heads() is False

    manifest = SimpleNamespace(
        metadata_path=tmp_path,
        snapshot=tmp_path,
        tensors={
            "layers.1.engram.embed.weight": SimpleNamespace(shape=(rows, 256), dtype="F8_E4M3", shard="w.bin", offset=0),
            "layers.1.engram.embed.scale": SimpleNamespace(shape=(rows, 8), dtype="F8_E8M0", shard="s.bin", offset=0),
        },
    )
    config = SimpleNamespace(text={
        "engram_compressed_vocab_size": 32,
        "engram_layer_ids": [1],
        "engram_vocab_size": 101,
        "engram_num_embeddings": [rows],
    })
    import transformers
    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", lambda **kw: object())
    monkeypatch.setattr(ds_engram.engram, "build_compressed_token_map", lambda tok: (list(range(32)), 32))
    monkeypatch.setattr(ds_engram, "ReadinessTimings", SimpleNamespace(
        allocate=lambda frame, count: SimpleNamespace(timestamps=torch.zeros((2, 3), dtype=torch.int64), frame=frame)
    ))

    tables = ds_engram.Tables.prepare(
        manifest, config, tokens=16, device="cpu",
        sequence=torch.zeros(1, dtype=torch.int64), frame=torch.zeros(1, dtype=torch.int32),
        host_allocator=lambda name, shape, dtype: SimpleNamespace(
            host_view=torch.zeros(shape, dtype=dtype), device_view=torch.zeros(shape, dtype=dtype),
        ),
        shard=TensorParallelShard(0, 4),
    )
    assert tables.staged_heads == 24
    assert tables.inputs[1].hashes.shape == (16, 24)
    assert tables.mailboxes[1].host_view.shape == (16 * 24 + 1,)


@pytest.mark.skipif(
    not ds_engram._supports_compact_heads(),
    reason="requires core ignition compact local-head Engram support",
)
@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens,use_all_gather", [
    (1, False),
    (8, False),
    (128, False),
    (512, False),
    (512, True),
])
def test_end_to_end_compact_engram_zero_ulp_parity(monkeypatch, tmp_path, world, tokens, use_all_gather):
    if world == 1 and use_all_gather:
        pytest.skip("all_gather path requires world > 1")

    geometry = b12x_engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    (tmp_path / "w.bin").write_bytes(b"\x00" * (rows * 256))
    (tmp_path / "s.bin").write_bytes(b"\x00" * (rows * 8))

    gen = torch.Generator().manual_seed(1234 + world * 31 + tokens)
    full_weights = (torch.randint(-7, 8, (rows, 256), generator=gen).float().to(torch.float8_e4m3fn)).view(torch.uint8)
    full_scales = torch.randint(123, 132, (rows, 8), dtype=torch.uint8, generator=gen)

    monkeypatch.setattr(pcie, "DmaAllReduce", _FakeDmaRuntime)
    monkeypatch.setattr(core_engram.engram, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(core_engram.engram, "plan", lambda caps, **kw: SimpleNamespace(
        handle=77, request=lambda **rkw: SimpleNamespace(**rkw)))
    monkeypatch.setattr(core_engram, "declare", lambda session, reqs: None)
    monkeypatch.setattr(ds_engram, "declare", lambda session, reqs: None)
    monkeypatch.setattr(ds_engram.hyperconnection, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(ds_engram.hyperconnection, "plan", lambda caps, **kw: SimpleNamespace(
        request=lambda **rkw: SimpleNamespace(**rkw)))

    class FakeWait:
        def __init__(self, timings, interval, **kw):
            self.timings, self.interval, self.handle = timings, interval, 0

        def functional(self, *, session):
            return self

        def operands(self, source):
            b = torch.zeros((1, 8), dtype=torch.bfloat16)
            return b, torch.empty_like(b)

    monkeypatch.setattr(core_engram, "PreparedReadinessWait", FakeWait)

    def fake_hash_op(handle, token_ids, mask, starts, slots, history, num_seqs, num_tokens,
                     out_hashes, compressed, request_ids, prepared_tokens):
        live_tokens = int(num_tokens.item())
        live_seqs = int(num_seqs.item())
        out_hashes.fill_(-1)
        if live_tokens > 0 and live_seqs > 0:
            active_starts = starts[:live_seqs + 1].tolist()
            active_starts[-1] = min(active_starts[-1], live_tokens)
            ref = hash_reference(
                token_ids[:live_tokens].tolist(),
                mask[:live_tokens].tolist(),
                active_starts,
                slots[:live_seqs].tolist(),
                history.tolist(),
                tuple(range(32)),
                geometry,
                layer_id=1,
            )
            out_hashes[:ref.shape[0]].copy_(ref)

    def fake_publish(ids, mailbox, sequence):
        n = ids.numel()
        mailbox[:n].copy_(ids.reshape(-1))
        mailbox[-1] = sequence[0]

    def fake_lookup(source, weight, scales, hashes, count, out, *rest):
        out.copy_(_dequantize_staged_cpu(weight, scales, hashes, count, rows))

    def fake_lookup_compact(source, weight, scales, hashes, count, out, timings, frame,
                            b_in, b_out, interval, barrier, table_rows, event):
        out.copy_(_dequantize_staged_cpu(weight, scales, hashes, count, table_rows))

    monkeypatch.setattr(core_engram, "hash_op", fake_hash_op)
    monkeypatch.setattr(core_engram, "publish_requests", fake_publish)
    monkeypatch.setattr(core_engram, "_lookup", fake_lookup)
    monkeypatch.setattr(core_engram, "_lookup_compact", fake_lookup_compact)
    monkeypatch.setattr(ds_engram, "gather_rows", lambda region, state_rows, dst: dst.copy_(region[state_rows]))

    class FakeProj:
        def __init__(self, source, weight, output, **kw):
            self.weight = weight

        def functional(self, **kw):
            return self

        def __call__(self, x):
            return x

    class FakeReduce:
        def __init__(self, *args, **kw):
            pass

        def functional(self, **kw):
            return self

        def __call__(self, x):
            return x

    class FakeAllGather:
        def __init__(self, src, dst, **kw):
            pass

        def functional(self, **kw):
            return self

    monkeypatch.setattr(ds_engram, "QuantizedProjection", FakeProj)
    monkeypatch.setattr(ds_engram, "PreparedChunkedAllReduce", FakeReduce)
    monkeypatch.setattr(ds_engram, "PreparedAllGather", FakeAllGather)

    seqs = min(4, tokens)
    live_tokens = max(1, tokens - (tokens // 5))
    meta = Metadata.allocate(tokens, seqs, 2, "cpu", max_query_tokens=tokens)
    meta.token_ids.copy_((torch.arange(tokens, dtype=torch.int64) * 11 + 5) % 32)
    meta.live_counts[0] = live_tokens
    meta.live_counts[1] = seqs
    meta.starts.fill_(3)
    meta.state_rows.copy_(torch.arange(seqs, dtype=torch.int64))
    bounds = [0] + [live_tokens * (i + 1) // seqs for i in range(seqs)]
    meta.query_start_loc.copy_(torch.tensor(bounds, dtype=torch.int32))
    history_region = (torch.arange(seqs * 3, dtype=torch.int64).reshape(seqs, 3) * 3 + 2) % 32
    pool = SimpleNamespace(
        config=SimpleNamespace(text={"rms_norm_eps": 1e-6}),
        region=lambda name: history_region,
    )
    runtime = _FakeDmaRuntime(world_size=world) if use_all_gather else (object() if world > 1 else None)
    weights = {"projection": object(), "weight": torch.ones(5120, dtype=torch.float32)}
    if use_all_gather:
        weights["projection_full"] = object()
    source = torch.zeros((tokens, 4, 5120), dtype=torch.bfloat16)
    output = torch.zeros_like(source)

    for rank in range(world):
        legacy_tables, partition = _make_tables_pair(tmp_path, geometry, world, rank, tokens, compact=False)
        compact_tables, _ = _make_tables_pair(tmp_path, geometry, world, rank, tokens, compact=True)

        # Build this rank's contiguous partitioned table planes (exact Helios storage layout).
        rank_w = torch.cat([full_weights[a:b] for a, b in partition.source_rows], dim=0)
        rank_s = torch.cat([full_scales[a:b] for a, b in partition.source_rows], dim=0)

        # Legacy 24-head path.
        legacy_reqs = ds_engram.PreparedTableRequests(legacy_tables, meta, pool, session=None, name="leg_req")
        legacy_reqs(meta)
        lw, ls = _simulate_helios_gather(rank_w, rank_s, legacy_tables.mailboxes[1].host_view, tokens * 24)
        legacy_tables.inputs[1].weights.copy_(lw)
        legacy_tables.inputs[1].scales.copy_(ls)
        legacy_scratch = ds_engram.EngramScratch.allocate(tokens, "cpu", world=world)
        legacy_engram = ds_engram.PreparedEngram(
            1, weights, legacy_tables, source, output, meta, pool,
            scratch=legacy_scratch, session=None, name="leg_eng", workspace=None, runtime=runtime,
        )
        assert legacy_engram.compact_lookup is False
        legacy_local = legacy_engram.lookup_local(source, meta)

        # Compact local-head path (with 0-byte gathered scratch when not using all_gather!).
        compact_reqs = ds_engram.PreparedTableRequests(compact_tables, meta, pool, session=None, name="cmp_req")
        compact_reqs(meta)
        local_heads = 24 // world
        cw, cs = _simulate_helios_gather(rank_w, rank_s, compact_tables.mailboxes[1].host_view, tokens * local_heads)
        compact_tables.inputs[1].weights.copy_(cw)
        compact_tables.inputs[1].scales.copy_(cs)
        compact_scratch = ds_engram.EngramScratch.allocate(
            tokens, "cpu", world=world, needs_gathered=use_all_gather,
        )
        if world > 1 and not use_all_gather:
            assert compact_scratch.gathered.nbytes == 0
        compact_engram = ds_engram.PreparedEngram(
            1, weights, compact_tables, source, output, meta, pool,
            scratch=compact_scratch, session=None, name="cmp_eng", workspace=None, runtime=runtime,
        )
        assert compact_engram.compact_lookup is (world > 1)
        assert compact_engram.use_all_gather is use_all_gather
        compact_local = compact_engram.lookup_local(source, meta)

        # 0-ULP bit-exact equality!
        assert compact_local.shape == legacy_local.shape == (tokens, 6144 // world)
        assert torch.equal(compact_local.view(torch.uint16), legacy_local.view(torch.uint16))

        # Negative sensitivity: flipping one staged weight byte must break equality.
        cw_bad = cw.clone()
        cw_bad[0, 0] ^= 0x10
        compact_tables.inputs[1].weights.copy_(cw_bad)
        bad_local = compact_engram.lookup_local(source, meta)
        assert not torch.equal(bad_local.view(torch.uint16), legacy_local.view(torch.uint16))


@pytest.mark.skipif(
    not ds_engram._supports_compact_heads(),
    reason="requires core ignition compact local-head Engram support",
)
def test_compact_lookup_local_make_fx_has_no_index_select(monkeypatch, tmp_path):
    geometry = b12x_engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    (tmp_path / "w.bin").write_bytes(bytes(rows * 256))
    (tmp_path / "s.bin").write_bytes(bytes(rows * 8))
    tokens = 128
    world = 4

    monkeypatch.setattr(core_engram.engram, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(core_engram.engram, "plan", lambda caps, **kw: SimpleNamespace(
        handle=77, request=lambda **rkw: SimpleNamespace(**rkw)))
    monkeypatch.setattr(core_engram, "declare", lambda session, reqs: None)
    monkeypatch.setattr(ds_engram, "declare", lambda session, reqs: None)
    monkeypatch.setattr(ds_engram.hyperconnection, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(ds_engram.hyperconnection, "plan", lambda caps, **kw: SimpleNamespace(
        request=lambda **rkw: SimpleNamespace(**rkw)))

    class FakeWait:
        def __init__(self, timings, interval, **kw):
            self.timings, self.interval, self.handle = timings, interval, 0

        def functional(self, *, session):
            return self

        def operands(self, source):
            b = torch.zeros((1, 8), dtype=torch.bfloat16)
            return b, torch.empty_like(b)

    monkeypatch.setattr(core_engram, "PreparedReadinessWait", FakeWait)
    monkeypatch.setattr(ds_engram, "QuantizedProjection",
                        lambda *a, **kw: SimpleNamespace(functional=lambda **fkw: None))
    monkeypatch.setattr(ds_engram, "PreparedChunkedAllReduce",
                        lambda *a, **kw: SimpleNamespace(functional=lambda **fkw: None))

    tables, _ = _make_tables_pair(tmp_path, geometry, world, 0, tokens, compact=True)
    meta = Metadata.allocate(tokens, 1, 2, "cpu", max_query_tokens=tokens)
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    scratch = ds_engram.EngramScratch.allocate(tokens, "cpu", world=world, needs_gathered=False)
    source = torch.zeros((tokens, 4, 5120), dtype=torch.bfloat16)
    output = torch.zeros_like(source)
    weights = {"projection": object(), "weight": torch.ones(5120, dtype=torch.float32)}
    engram_op = ds_engram.PreparedEngram(
        1, weights, tables, source, output, meta, pool,
        scratch=scratch, session=None, name="fx_eng", workspace=None, runtime=object(),
    )

    gm = make_fx(
        lambda s, live: engram_op.lookup_local(s, SimpleNamespace(live_counts=live)),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(source, meta.live_counts)
    ops = [str(node.target) for node in gm.graph.nodes if node.op == "call_function"]
    assert any("staged_engram_compact" in op for op in ops)
    assert not any("index_select" in op for op in ops)
