# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 arithmetic and complete continuation geometry."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.inkling2.arithmetic import l2_clip, residual_mix, short_convolution
from hyperflare_recipes.recipes.inkling2.config import ModelConfig
from hyperflare_recipes.recipes.inkling2.pool import RankGeometry, continuation_layout, page_specs


@pytest.fixture
def configuration():
    path = Path(os.environ.get('FLARE_INKLING2_SNAPSHOT', '/data/models/Inkling-2-Small-NVFP4')) / 'config.json'
    if not path.is_file():
        pytest.skip('requires the Inkling 2 checkpoint configuration')
    return json.loads(path.read_text())


def test_target_and_draft_state_geometry(configuration):
    config = ModelConfig.from_dict(configuration)
    geometry = RankGeometry.of(config.text, 4)
    layout = continuation_layout(config.text, geometry, config.draft)
    assert (geometry.heads, geometry.kv_heads, geometry.ring) == (12, 2, 1152)
    assert config.text.global_layers == (5, 11, 17, 23, 29, 35, 41, 47)
    assert sum(shape[0] for _, shape in page_specs(config.text, geometry, config.draft)) == 589824
    assert layout.row_bytes == 29130752
    assert len([r for r in layout.regions if r.name.startswith('layer') and r.name.endswith('convolution')]) == 192
    for layer in (0, 1, 3, 4):
        assert layout.region(f'draft{layer}.key').shape == (1024, 2, 128)
        assert layout.region(f'draft{layer}.key').dtype == torch.float8_e4m3fn
    assert layout.region('draft.rnn_context').shape == (4608,)
    assert TensorParallelShard(3, 4).padded_partition(config.text.vocabulary, alignment=64) == (150912, 201216)


@pytest.mark.parametrize('key,value', [('mlp_gate_softcap', 5.0), ('mlp_up_softcap', 24.0),
    ('hidden_size', 4096), ('intermediate_size', 2048), ('n_routed_experts', 256)])
def test_unsupported_expert_geometry_is_rejected(configuration, key, value):
    configuration['text_config'][key] = value
    with pytest.raises(ValueError, match='SiTU caps 4/25'):
        ModelConfig.from_dict(configuration)


def _metadata(lengths, starts, device='cpu'):
    rows, tokens = len(lengths), sum(lengths)
    m = Metadata.allocate(tokens, rows, 1, device, max_query_tokens=max(lengths))
    m.set(torch.tensor(starts, device=device), torch.tensor(lengths, dtype=torch.int32, device=device),
          torch.arange(1, rows + 1, device=device), torch.zeros(rows, 1, dtype=torch.int32, device=device),
          torch.zeros(rows, max(lengths), dtype=torch.int64, device=device), torch.tensor([rows], device=device))
    return m


def test_short_convolution_matches_canonical_packed_sequences():
    torch.manual_seed(7)
    m = _metadata([5, 2], [9, 0])
    x = (torch.randint(-16, 17, (7, 8)).float() / 8).bfloat16()
    previous = (torch.randint(-16, 17, (2, 3, 8)).float() / 8).bfloat16()
    previous[1].zero_()
    weight = torch.randint(-4, 5, (8, 1, 4)).float() / 4
    actual = short_convolution(x, weight, previous, m)
    expected = []
    for row, (begin, end) in enumerate(zip(m.query_start_loc[:-1], m.query_start_loc[1:])):
        source = x[begin:end].float()
        context = torch.cat((previous[row].float(), source)).T[None]
        convolved = F.conv1d(context, weight, groups=8).squeeze(0).T
        expected.append(((convolved + source) / 5 ** .5).bfloat16())
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)


def test_attention_residual_mixes_normalized_values():
    from transformers.models.inkling2.modeling_inkling2 import Inkling2BlockAttentionResidual
    from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
    torch.manual_seed(11)
    module = Inkling2BlockAttentionResidual(Inkling2TextConfig(hidden_size=64), num_queries=3)
    module.query.data.normal_()
    sources = tuple(torch.randn(7, 64).bfloat16() for _ in range(4))
    actual = torch.stack(residual_mix(sources, module.query * 64 ** -.5), dim=1)
    expected = torch.stack(module(torch.stack(sources[:-1], -2), sources[-1]), dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize('threshold', [8.0, 128 ** .5 * 2])
def test_l2_clip_preserves_canonical_round_before_length_scale(threshold):
    from transformers.models.inkling2.modeling_inkling2 import Inkling2ScaledL2Clip
    torch.manual_seed(17)
    x = torch.randn(31, 12, 128).bfloat16() * 4
    clip = Inkling2ScaledL2Clip(12, 128, threshold)
    clip.scale.data.uniform_(.5, 1.5)
    tau = torch.linspace(1, 1.8, 31)
    expected = (clip(x).float() * tau[:, None, None]).bfloat16()
    torch.testing.assert_close(l2_clip(x, threshold, scale=clip.scale, tau=tau), expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned SM120 GPU')
@pytest.mark.parametrize('layer', [0, 5])
@pytest.mark.parametrize('mode,tokens,live', [('prefill', 128, 17), ('decode', 1, 1), ('verify', 8, 8)])
@pytest.mark.parametrize('compiled', [False, True])
def test_checkpoint_attention_fp8_and_fp32_projection(configuration, layer, mode, tokens, live, compiled):
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.inkling2.attention import LayerCache, PreparedAttention
    from hyperflare_recipes.recipes.inkling2.pool import ContinuationPool
    from hyperflare_recipes.recipes.inkling2.weights import InklingWeightLoader

    snapshot = Path(os.environ.get('FLARE_INKLING2_SNAPSHOT', '/data/models/Inkling-2-Small-NVFP4'))
    device = torch.device('cuda', 0)
    config = ModelConfig.from_dict(configuration)
    shard = TensorParallelShard(0, 4)
    with InklingWeightLoader(inspect_indexed_checkpoint(snapshot), device, shard=shard, vision=False) as loader:
        weights = loader.load_layer(layer).attention
    text = config.text.prefix(6)
    pool = ContinuationPool.allocate(text, RankGeometry.of(text, 4), 3, 4, device)
    m = Metadata.allocate(tokens, 1, 4, device)
    with RecordedPreparationSession(device=device, checkpoint='inkling2-checkpoint-attention',
            workload_schema='inkling2-checkpoint-attention', numerical_policy='default', autotune_effort='off') as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operation = PreparedAttention(layer, weights, text, LayerCache.of(pool, layer), m,
            mode=mode, tokens=tokens, session=session, workspace=arena, name='attention', tuning_page_count=4)
        session.tune()
        streams = tuple(torch.randn(tokens, text.hidden, device=device).bfloat16() for _ in weights.projections)
        m.set(torch.tensor([0], device=device), torch.tensor([live], dtype=torch.int32, device=device),
            torch.tensor([1], device=device), torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
            torch.zeros(1, tokens, dtype=torch.int64, device=device), torch.tensor([1], device=device))
        def forward():
            return operation.attend(streams, m)
        if compiled:
            from ignition.preparation.compilation import compile_forward
            forward = compile_forward(forward)
        attended, partial, values = forward()
        n = live
        q = values['query'][:n].double()
        k = values['key'][:n].clamp(-448, 448).to(torch.float8_e4m3fn).double().repeat_interleave(6, 1)
        v = values['value'][:n].clamp(-448, 448).to(torch.float8_e4m3fn).double().repeat_interleave(6, 1)
        distances = torch.arange(n, device=device)[:, None] - torch.arange(n, device=device)[None]
        bias = (values['relative'][:n].float() @ weights.relative_projection.float()).bfloat16().double()
        scores = torch.einsum('thd,khd->thk', q, k) / 128 ** .5
        scores += bias.gather(2, distances.clamp_min(0)[:, None].expand(-1, 12, -1))
        scores.masked_fill_(distances[:, None] < 0, -float('inf'))
        probability = torch.cat((scores, values['sink'][:n].double().unsqueeze(-1)), -1).softmax(-1)[..., :-1]
        reference = torch.einsum('thk,khd->thd', probability, v)
        mass = torch.einsum('thk,khd->thd', probability, v.abs())
        gate = values['gate'][:n].double().sigmoid()
        expected = reference * gate
        bound = (2 ** -8 * (mass + reference.abs()) + 2 ** -11 * mass) * gate + 2 ** -8 * expected.abs()
        error = (attended[:n].view(n, 12, 128).double() - expected).abs()
        assert ((error - bound).max() <= 0).item()
        assert partial.dtype == torch.float32
        exact = attended[:n].double() @ weights.output.double().T
        accumulated = attended[:n].double().abs() @ weights.output.double().abs().T
        gamma = attended.shape[1] * 2 ** -24 / (1 - attended.shape[1] * 2 ** -24)
        assert (((partial[:n].double() - exact).abs() - gamma * accumulated).max() <= 0).item()


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned SM120 GPU')
def test_accepted_window_prefix_commits_replay_without_candidate_mutation():
    from hyperflare_recipes.recipes.inkling2.attention import retain_batch, stage_batch

    device = torch.device('cuda', 0)
    m = _metadata([8, 8, 8], [0, 1150, 3499], device)
    m.state_rows.copy_(torch.tensor([3, 1, 2], device=device))
    initial = torch.randn(5, 1152, 2, 128, device=device).to(torch.float8_e4m3fn)
    rings = initial.clone().view(torch.uint8).view(5, -1)
    values = torch.randn(24, 2, 128, dtype=torch.bfloat16, device=device)
    staged, lengths = stage_batch(rings, values, m, window=1024, pages=9)
    torch.testing.assert_close(rings, initial.view(torch.uint8).view(5, -1), rtol=0, atol=0)
    assert lengths.tolist() == [8, 1031, 1031]
    for lane, start in enumerate(m.starts.tolist()):
        first = max(start - 1023, 0)
        expected = torch.cat((initial[m.state_rows[lane], first % 1152:].bfloat16(), initial[m.state_rows[lane], :first % 1152].bfloat16()))
        previous = expected[:start - first]
        retained = torch.cat((previous, values[lane * 8:(lane + 1) * 8])).to(torch.float8_e4m3fn)
        actual = staged.view(3, 1152, 2, 128)[lane]
        torch.testing.assert_close(actual[:retained.shape[0]].view(torch.uint8), retained.view(torch.uint8), rtol=0, atol=0)
        assert not actual[retained.shape[0]:].view(torch.uint8).any()
    counts = torch.zeros(3, dtype=torch.int32, device=device)
    retain_batch(rings, values, counts, m)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        retain_batch(rings, values, counts, m)
    for accepted in range(9):
        rings.copy_(initial.view(torch.uint8).view(5, -1))
        selected = [accepted, (accepted + 1) % 9, 8 - accepted]
        counts.copy_(torch.tensor(selected, dtype=torch.int32, device=device))
        before = torch.cuda.memory_stats()['allocation.all.allocated']
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
        expected = initial.clone()
        for lane, count in enumerate(selected):
            for index in range(count):
                position = (int(m.starts[lane]) + index) % 1152
                expected[m.state_rows[lane], position] = values[lane * 8 + index].to(torch.float8_e4m3fn)
        torch.testing.assert_close(rings, expected.view(torch.uint8).view(5, -1), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned SM120 GPU')
@pytest.mark.parametrize('tokens,streams,sources,add',
    [(1, streams, sources, add) for streams in (1, 3) for sources in range(1, 10) for add in (False, True)]
    + [(64, 1, 5, True), (64, 3, 9, False), (4096, 1, 9, False), (4096, 3, 9, True)])
def test_prepared_attention_residual_reference_and_capture(tokens, streams, sources, add):
    from cuda.bindings import driver
    from hyperflare_recipes.recipes.inkling2.arithmetic import rms_norm
    from hyperflare_recipes.recipes.inkling2.residual import PreparedResidualMix
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.manual_seed(19)
    device = torch.device('cuda', 0)
    values = tuple(torch.randn(tokens, 4608, device=device).bfloat16() for _ in range(sources))
    queries = torch.randn(streams, 4608, device=device) * 4608 ** -.5
    norms = tuple(torch.randn(4608, device=device).bfloat16() for _ in range(streams))
    addend = torch.randn_like(values[-1]) if add else None
    with RecordedPreparationSession(device=device, checkpoint='inkling2-residual-fusion',
            workload_schema='inkling2-residual', numerical_policy='default', autotune_effort='off') as session:
        operation = PreparedResidualMix(values[0], queries, norms, sources=sources, eps=1e-6,
                                        session=session, name='residual', addend=add)
        session.tune()
        reference = residual_mix(values, queries, addend=addend)
        mixture = operation.mix(values, addend=addend)
        for actual, expected in zip(mixture, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        expected = tuple(rms_norm(value, weight) for value, weight in zip(reference, norms, strict=True))
        actual = operation(values, addend=addend)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2 ** -7, atol=2e-5)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with session.capture(), torch.cuda.graph(graph):
            outputs = operation(values, addend=addend)
        status, _, count = driver.cuGraphGetNodes(graph.raw_cuda_graph())
        assert status == driver.CUresult.CUDA_SUCCESS
        assert count == 1 + streams
        for _ in range(3):
            before = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
            for got, want in zip(outputs, expected, strict=True):
                torch.testing.assert_close(got, want, rtol=2 ** -7, atol=2e-5)


class _ResidualForward:
    def __init__(self, operation, values, addend):
        self.operation, self.values, self.addend = operation, values, addend

    def __call__(self):
        return self.operation(self.values, addend=self.addend)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned SM120 GPU')
@pytest.mark.parametrize('tokens,streams,sources,add',
    [(1, 1, sources, False) for sources in range(2, 10)]
    + [(64, 1, 9, True), (8, 3, 2, False), (4096, 3, 9, False)])
@torch.no_grad()
def test_prepared_attention_residual_aot_restore_and_replay(tmp_path, tokens, streams, sources, add):
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.inkling2.residual import PreparedResidualMix

    torch.manual_seed(47)
    device = torch.device('cuda', 0)
    values = tuple(torch.randn(tokens, 4608, device=device).bfloat16() for _ in range(sources))
    queries = torch.randn(streams, 4608, device=device) * 4608 ** -.5
    norms = tuple(torch.randn(4608, device=device).bfloat16() for _ in range(streams))
    addend = torch.randn_like(values[-1]) if add else None
    with RecordedPreparationSession(device=device, checkpoint='inkling2-residual-aot',
            workload_schema='inkling2-residual-aot', numerical_policy='default') as session:
        operation = PreparedResidualMix(values[0], queries, norms, sources=sources, eps=1e-6,
                                        session=session, name='residual', addend=add)
        session.tune()
        forward = _ResidualForward(operation, values, addend)
        expected = forward()
        compiled = aot_forward(forward)
        actual = compiled(forward)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        path = tmp_path / 'residual.bin'
        compiled.save_compiled_function(str(path))
        with path.open('rb') as file:
            restored = torch.compiler.load_compiled_function(file)
        restored(forward)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            outputs = restored(forward)
        session.freeze()
        for _ in range(3):
            for value in values:
                value.normal_()
            if addend is not None:
                addend.normal_()
            expected = forward()
            before = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
            for got, want in zip(outputs, expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned SM120 GPU')
@pytest.mark.parametrize('tokens', [1, 8, 64, 4096])
@pytest.mark.parametrize('projection_width', [3280, 3792])
@torch.no_grad()
def test_shared_attention_gate_and_value_norm_compiled_replay(tokens, projection_width):
    from hyperflare_recipes.components.head_gate import PreparedHeadGate
    from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.inkling2.arithmetic import rms_norm

    torch.manual_seed(37)
    device = torch.device('cuda', 0)
    projected = torch.randn(tokens, projection_width, dtype=torch.bfloat16, device=device)
    gates = projected[:, 1536:3072]
    attended = torch.randn(tokens, 12, 128, dtype=torch.bfloat16, device=device)
    value = torch.randn(tokens * 2, 128, dtype=torch.bfloat16, device=device)
    norm_output = torch.empty_like(value)
    with RecordedPreparationSession(device=device, checkpoint='inkling2-shared-attention-pointwise',
            workload_schema='inkling2-shared-attention-pointwise', numerical_policy='default') as session:
        gating = PreparedHeadGate(attended.view(tokens, 1536, 1), gates,
            session=session, name='gate').functional(session=session)
        normalization = PreparedRMSNorm(value, torch.ones(128, dtype=torch.bfloat16, device=device),
            norm_output, eps=1e-6, session=session, name='value_norm').functional()
        session.tune()

        def forward():
            normalized = normalization(value)
            gated = gating(attended.view(tokens, 1536, 1), gates).view(tokens, 12, 128)
            return normalized, gated

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            normalized, gated = compiled()
        session.freeze()
        for scale in (1., 1e-10, 1e10):
            projected.normal_()
            gates[:, :7].copy_(torch.tensor([-float('inf'), -100., -1., 0., 1., 100., float('inf')],
                                           dtype=torch.bfloat16, device=device))
            original_projection = projected.clone()
            attended.normal_()
            attended.mul_(scale)
            if tokens > 1:
                attended[-1].zero_()
            else:
                attended[0, 0, 7:].zero_()
            seed = attended.clone()
            value.normal_()
            value.mul_(scale)
            value[0].zero_()
            if value.shape[0] > 2:
                value[1].fill_(1e-20)
                value[2].fill_(1e10)
            original_value = value.clone()
            expected_norm = rms_norm(value)
            exact_gate = gates.double().sigmoid().view(tokens, 12, 128)
            expected_gate = seed.double() * exact_gate
            unit = 2 ** -8
            gate_round_bound = unit * exact_gate + 2 ** -134
            bound = seed.double().abs() * (gate_round_bound + unit * (exact_gate + gate_round_bound))
            bound += 2 ** -20 * expected_gate.abs() + 2 ** -134
            before = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
            assert torch.equal(projected, original_projection)
            assert torch.equal(value, original_value)
            if tokens > 1:
                assert not gated[-1].any()
            else:
                assert not gated[0, 0, 7:].any()
            assert not normalized[0].any()
            assert torch.isfinite(normalized).all() and torch.isfinite(gated).all()
            assert (((gated.double() - expected_gate).abs() - bound).max() <= 0).item()
            torch.testing.assert_close(normalized, expected_norm, rtol=2 ** -7, atol=2 ** -126)
        graph.reset()
