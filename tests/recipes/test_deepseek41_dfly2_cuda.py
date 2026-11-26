# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and CUDA parity tests for native DeepSeek V4.1 Flash DFly2-B6 drafting."""

from dataclasses import replace
import os
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import triton

from ignition.parameters import SpeculationParameters
from ignition.planning import Gpu, plan as plan_preparation
from ignition.preparation.rebuild_manifest import declare_weights, weight_payload_declaration
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.vocabulary_sampling import PreparedVocabularySoftmax
from hyperflare_recipes.parameters import (
    DraftQuantizationParameters,
    ExpertScaleCompressionParameters,
)
from hyperflare_recipes.recipes.deepseek41.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.deepseek41.config import (
    DFly2Config, ModelConfig, QUALIFIED_ENCODER_SCHEDULE, TEXT_GEOMETRY,
)
from hyperflare_recipes.recipes.deepseek41.dfly2 import (
    PreparedDFly2Context,
    _corr_block_key0_kernel,
    _corr_combine_step_kernel,
    _gated_block_attn_add_kernel,
)
from hyperflare_recipes.recipes.deepseek41.pool import ContinuationPool, continuation_layout
from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE
from ignition.components.sampling import DeviceRequestState


DFLY2_SNAPSHOT = Path(
    os.environ.get(
        "DFLY2_SNAPSHOT",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--AngelSlim--DS-V4.1-Flash-DFly2-B6/"
            "snapshots/d08e55a6897b1974307ea9771038838de5986e66"
        ),
    )
)


def _sample_target_config():
    text = {
        **TEXT_GEOMETRY,
        **QUALIFIED_ENCODER_SCHEDULE,
        "compress_ratios": QUALIFIED_ENCODER_SCHEDULE["compress_ratios"] + [0, 0, 0],
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "compress_rope_theta": 160000.0,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 7,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [20, 28, 36],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    }
    quant = {
        "quant_method": "fp8",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "activation_scheme": "dynamic",
    }
    return ModelConfig(text, quant, None)


def _sample_dfly2_config():
    return DFly2Config(
        layers=3,
        block_size=6,
        noise_token=128799,
        target_kv_layers=(28, 32, 36),
        target_index_layers=(28, 32, 36),
        index_topk=512,
        experts=128,
        top_k=3,
        correction_intermediate=5120,
        mtp_fusion="slot0",
        max_block_slots=16,
    )


def test_dfly2_layout_and_planning():
    target = _sample_target_config()
    dfly_cfg = replace(target, draft_override=_sample_dfly2_config())
    plain_layout = continuation_layout(target)
    dspark_layout = continuation_layout(target, dspark=True)
    dfly_layout = continuation_layout(dfly_cfg, dfly2=True)

    assert plain_layout.row_bytes == 2_841_088
    assert dspark_layout.row_bytes == 3_043_840
    assert dfly_layout.row_bytes == 2_847_232
    assert dfly_layout.row_bytes < dspark_layout.row_bytes
    for i in range(3):
        region = dfly_layout.region(f"draft_index{i}")
        assert region.shape == (512,) and region.dtype == torch.int32

    gpus = tuple(Gpu(i, "RTX PRO 6000", f"GPU-{i}", 96 << 30, 96 << 30, (12, 0)) for i in range(4))
    spec = SpeculationParameters(
        draft_tokens=5,
        max_draft_tokens=7,
        token_capacity=256,
        packed_verification=True,
        draft_prefix_selection="gpu",
        graph_budget=80,
        method="dfly2",
        mtp_only=True,
    )
    params = replace(
        RECIPE.parameters,
        max_concurrency=64,
        prefill_chunk=8192,
        speculation=spec,
        expert_scale_compression=ExpertScaleCompressionParameters(encoding=None),
        draft_quantization=DraftQuantizationParameters(
            draft_head_quantization="nvfp4",
            draft_body_quantization="mxfp8",
            draft_head_choices=("nvfp4", "mxfp8"),
            draft_body_choices=("mxfp8",),
            markov_head_choices=("nvfp4", "fp8"),
        ),
        mla_compute_mode="bf16",
        vocabulary_parallel_sampling=True,
        row_shard_prefill=True,
        row_shard_attention=True,
    )
    plan = plan_preparation(
        RECIPE,
        RECIPE.checkpoints[0],
        gpus,
        selected=(0, 1, 2, 3),
        reserve_bytes=0,
        parameters=params,
    )
    assert plan.costs.state_row_bytes == 2_847_232
    assert plan.speculation.method == "dfly2"
    assert plan.parameters.draft_quantization.draft_body_quantization == "mxfp8"


def test_dfly2_preparation_config_validation():
    def _dfly2_params(*, draft_tokens=5, vps=True, markov=None, body="mxfp8", head="nvfp4"):
        return replace(
            RECIPE.parameters,
            max_concurrency=64,
            speculation=SpeculationParameters(
                draft_tokens=draft_tokens,
                max_draft_tokens=7,
                token_capacity=256,
                packed_verification=True,
                draft_prefix_selection="gpu",
                graph_budget=80,
                method="dfly2",
                mtp_only=True,
            ),
            vocabulary_parallel_sampling=vps,
            draft_quantization=DraftQuantizationParameters(
                draft_body_quantization=body,
                draft_head_quantization=head,
                markov_head_quantization=markov,
                draft_head_choices=("nvfp4", "mxfp8"),
                draft_body_choices=("mxfp8",),
                markov_head_choices=("nvfp4", "fp8"),
            ),
        )

    cfg = PreparationConfig(
        parameters=_dfly2_params(),
        devices=(0, 1, 2, 3),
        draft_checkpoint="/tmp/dfly2",
    )
    assert cfg.is_dfly2
    assert cfg.draft_tokens == 5
    assert cfg.draft_block_size == 6
    assert cfg.all_reduce_tokens >= 64 * 6

    with pytest.raises(ValueError, match="at most five draft tokens"):
        PreparationConfig(
            parameters=_dfly2_params(draft_tokens=6),
            devices=(0, 1, 2, 3),
            draft_checkpoint="/tmp/dfly2",
        )
    with pytest.raises(ValueError, match="vocabulary-parallel-sampling"):
        PreparationConfig(
            parameters=_dfly2_params(vps=False),
            devices=(0, 1, 2, 3),
            draft_checkpoint="/tmp/dfly2",
        )
    with pytest.raises(ValueError, match="Markov"):
        PreparationConfig(
            parameters=_dfly2_params(markov="fp8"),
            devices=(0, 1, 2, 3),
            draft_checkpoint="/tmp/dfly2",
        )


def _ref_rmsnorm(x, w, eps=1e-20):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).bfloat16()


def _ref_rope(x, angles):
    cos = angles[..., :32].float()
    sin = angles[..., 32:].float()
    tail = x[..., -64:].float().unflatten(-1, (32, 2))
    a, b = tail[..., 0], tail[..., 1]
    rot = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2).bfloat16()
    return torch.cat((x[..., :-64], rot), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 4, 64])
def test_dfly2_triton_kernels_parity(batch):
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(1234 + batch)
    inter = 1280
    heads = 16
    fused = (2 * inter + 1280 + 512 + heads + 127) // 128 * 128
    eps = 1e-20

    h_proj = torch.randn((batch, fused), device=device, dtype=torch.bfloat16, generator=gen) * 0.5
    e_proj = torch.randn((batch, fused), device=device, dtype=torch.bfloat16, generator=gen) * 0.5
    q_norm_w = torch.randn((1280,), device=device, dtype=torch.bfloat16, generator=gen) * 0.5 + 1.0
    kv_norm_w = torch.randn((512,), device=device, dtype=torch.bfloat16, generator=gen) * 0.5 + 1.0
    theta = torch.randn((batch, 32), device=device, dtype=torch.float32, generator=gen)
    angles = torch.cat((theta.cos(), theta.sin()), dim=-1).contiguous()

    # 1. _corr_block_key0_kernel
    kv_off = 2 * inter + 1280
    bk0_out = torch.empty((batch, 512), device=device, dtype=torch.bfloat16)
    _corr_block_key0_kernel[(batch,)](
        h_proj, e_proj, kv_norm_w, angles, bk0_out,
        h_proj.stride(0), e_proj.stride(0),
        KV_OFFSET=kv_off, EPS=eps, num_warps=4,
    )
    kv_sum_ref = h_proj[:, kv_off:kv_off + 512].float() + e_proj[:, kv_off:kv_off + 512].float()
    bk0_ref = _ref_rope(_ref_rmsnorm(kv_sum_ref, kv_norm_w, eps), angles)
    torch.testing.assert_close(bk0_out, bk0_ref, atol=2e-2, rtol=2e-2)

    # 2. _corr_combine_step_kernel
    swiglu_out = torch.empty((batch, inter), device=device, dtype=torch.bfloat16)
    qlow_out = torch.empty((batch, 1280), device=device, dtype=torch.bfloat16)
    bk_out = torch.empty((batch, 512), device=device, dtype=torch.bfloat16)
    gate_out = torch.empty((batch, heads), device=device, dtype=torch.float32)
    _corr_combine_step_kernel[(batch, 4)](
        h_proj, e_proj, q_norm_w, kv_norm_w, angles,
        swiglu_out, qlow_out, bk_out, gate_out,
        h_proj.stride(0), e_proj.stride(0),
        INTER=inter, INTER_PAD=triton.next_power_of_2(inter),
        HEADS=heads, HEADS_PAD=triton.next_power_of_2(heads),
        EPS=eps, num_warps=4,
    )
    g_ref = (h_proj[:, :inter] + e_proj[:, :inter])
    u_ref = (h_proj[:, inter:2 * inter] + e_proj[:, inter:2 * inter])
    swiglu_ref = (F.silu(g_ref) * u_ref).bfloat16()
    q_ref = _ref_rmsnorm(
        h_proj[:, 2 * inter:2 * inter + 1280].float() + e_proj[:, 2 * inter:2 * inter + 1280].float(),
        q_norm_w, eps,
    )
    gate_ref = h_proj[:, kv_off + 512:kv_off + 512 + heads].float() + e_proj[:, kv_off + 512:kv_off + 512 + heads].float()
    torch.testing.assert_close(swiglu_out, swiglu_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(qlow_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(bk_out, bk0_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gate_out, gate_ref, atol=1e-5, rtol=1e-5)

    # 3. _gated_block_attn_add_kernel across step 1..5 with both starts=0 and starts>0
    rel_bias = torch.randn((heads, 16), device=device, dtype=torch.float32, generator=gen) * 0.2
    block_sink = torch.randn((heads,), device=device, dtype=torch.float32, generator=gen) * 0.5
    gate_bias = torch.randn((heads, 16), device=device, dtype=torch.float32, generator=gen) * 0.2
    starts = torch.arange(batch, device=device, dtype=torch.int64)  # row 0 has start=0, others > 0
    for step in (1, 3, 5):
        q = torch.randn((batch, heads, 512), device=device, dtype=torch.bfloat16, generator=gen) * 0.3
        bkeys = torch.randn((batch, step + 1, 512), device=device, dtype=torch.bfloat16, generator=gen) * 0.3
        read_ctx = torch.randn((batch, heads, 512), device=device, dtype=torch.bfloat16, generator=gen) * 0.3
        out = torch.empty_like(read_ctx)
        _gated_block_attn_add_kernel[(batch, heads)](
            q, bkeys, read_ctx, rel_bias, block_sink, gate_out, gate_bias, starts, out,
            HEADS=heads, BIAS_STRIDE=rel_bias.stride(0), STEP=step, num_warps=4,
        )
        # PyTorch FP32 reference
        dots = torch.einsum("bhd,bkd->bhk", q.float(), bkeys.float()) * (512.0 ** -0.5)
        dist = torch.tensor([step - j for j in range(step + 1)], device=device, dtype=torch.long)
        logits = dots + rel_bias[:, dist].unsqueeze(0)
        logits = torch.cat((logits, block_sink.view(1, heads, 1).expand(batch, -1, 1)), dim=-1)
        probs = logits.softmax(dim=-1)[..., :-1]
        read_blk = torch.einsum("bhk,bkd->bhd", probs, bkeys.float())
        gate = torch.tanh(gate_out + gate_bias[:, step].unsqueeze(0)).unsqueeze(-1)
        ctx_ref = torch.where((starts > 0).view(batch, 1, 1), read_ctx.float(), 0.0)
        ref = (ctx_ref + read_blk * gate).bfloat16()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vocabulary_softmax_simpson_confidence_parity():
    device = torch.device("cuda:0")
    batch, steps, vocab = 4, 5, 1024
    requests = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch)
    requests.status.fill_(1)
    requests.generation.fill_(1)
    # parameters columns: [temperature, top_p, min_p, repetition_penalty, presence_penalty, frequency_penalty]
    requests.parameters[:, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
    requests.parameters[:, 1] = 1.0
    requests.parameters[:, 2] = 0.0
    requests.parameters[:, 3] = 1.0
    requests.parameters[:, 4] = 0.0
    requests.parameters[:, 5] = 0.0
    requests.top_k.fill_(-1)

    gen = torch.Generator(device=device).manual_seed(42)
    logits = (torch.randn((batch, steps, vocab), device=device, dtype=torch.bfloat16, generator=gen) * 2.5).contiguous()
    probs = torch.zeros((batch, steps, vocab), device=device, dtype=torch.float32)
    valid = torch.zeros((batch, steps), device=device, dtype=torch.bool)
    conf = torch.full((batch, steps), -1.0, device=device, dtype=torch.float32)
    drafts = torch.zeros((batch, steps), device=device, dtype=torch.int64)
    tile_masses = torch.zeros((batch, steps, vocab // 256), device=device, dtype=torch.float32)
    row_slots = torch.arange(batch, device=device, dtype=torch.int32)
    row_gens = torch.ones(batch, device=device, dtype=torch.int64)
    live_rows = torch.tensor([3], device=device, dtype=torch.int32)  # row 3 inactive
    pending = torch.zeros(batch, device=device, dtype=torch.int64)
    seeds = torch.arange(batch, device=device, dtype=torch.int64)
    offsets = torch.zeros(batch, device=device, dtype=torch.int64)
    limits = torch.tensor([6, 4, 6, 0], device=device, dtype=torch.int32)

    op = PreparedVocabularySoftmax(
        logits, probs, valid, requests, row_slots, row_gens, live_rows, drafts,
        start=0, runtime=None, session=None, coordinator=None, name="test.conf",
        pending=pending, seeds=seeds, offsets=offsets, tokens=drafts,
        tile_masses=tile_masses, token_limits=limits, confidence=conf,
    )
    op.prepare()
    op.functional(session=None)
    for step in range(steps):
        op.run_step(
            step, logits, drafts, pending=pending, seeds=seeds, offsets=offsets,
            probabilities=probs, valid=valid, tokens=drafts,
            tile_masses=tile_masses, token_limits=limits, confidence=conf,
        )

    # Row 0 (greedy, limit=6 -> all 5 drafts active): T=0 returns exact softmax(logits)^2 sum!
    ref_simpson = logits.float().softmax(dim=-1).square().sum(dim=-1)
    torch.testing.assert_close(conf[0, :], ref_simpson[0, :], atol=1e-4, rtol=1e-4)
    # Row 1 (greedy, limit=4 -> drafts 0..2 active, drafts 3..4 inactive -> 0.0)
    torch.testing.assert_close(conf[1, :3], ref_simpson[1, :3], atol=1e-4, rtol=1e-4)
    assert torch.equal(conf[1, 3:], torch.zeros(2, device=device))
    # Row 2 (T=1.0, limit=6 -> all 5 active): equals sum_v probs(v)^2!
    torch.testing.assert_close(conf[2, :], probs[2, :].square().sum(dim=-1), atol=1e-4, rtol=1e-4)
    # Row 3 (inactive): all zeros
    assert torch.equal(conf[3, :], torch.zeros(steps, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dfly2_context_compact_and_packed():
    device = torch.device("cuda:0")
    cfg = replace(_sample_target_config(), draft_override=_sample_dfly2_config())
    pool = ContinuationPool.allocate(cfg, slots=8, pages=16, device=device, dfly2=True)

    class _DummyWeights:
        config = cfg

    # 1. Compact prefill (128-row suffix out of 512 tokens)
    meta_pf = Metadata.allocate(512, 1, 16, device)
    meta_pf.state_rows.fill_(3)
    meta_pf.lengths.fill_(300)
    aux_pf = torch.arange(128 * 3 * 512, device=device, dtype=torch.int32).view(128, 3, 512)
    ctx_pf = PreparedDFly2Context(_DummyWeights(), aux_pf, meta_pf, pool,
                                  session=None, name="pf", workspace=None, compact_prefill=True)
    ctx_pf(aux_pf, torch.tensor([300], device=device, dtype=torch.int32), meta_pf)
    for o in range(3):
        assert torch.equal(pool.draft_index(o)[3], aux_pf[127, o, :])

    # Short prefill chunk (<128) inside compact prefill
    ctx_pf(aux_pf, torch.tensor([17], device=device, dtype=torch.int32), meta_pf)
    for o in range(3):
        assert torch.equal(pool.draft_index(o)[3], aux_pf[16, o, :])

    # 2. Packed multi-request verify with an inactive hole in row 1
    starts = torch.tensor([10, 20, 30, 40], device=device, dtype=torch.int64)
    lengths = torch.tensor([6, 0, 4, 5], device=device, dtype=torch.int32)
    state_rows = torch.tensor([2, 0, 5, 6], device=device, dtype=torch.int64)
    pages = torch.zeros((4, 16), device=device, dtype=torch.int32)
    toks = torch.zeros((4, 6), device=device, dtype=torch.int64)
    meta_v = Metadata.pack(starts, lengths, state_rows, pages, toks,
                           torch.tensor([4], device=device, dtype=torch.int32),
                           token_capacity=24, sample_count=24, max_query_tokens=6)
    aux_v = torch.arange(24 * 3 * 512, device=device, dtype=torch.int32).view(24, 3, 512)
    ctx_v = PreparedDFly2Context(_DummyWeights(), aux_v, meta_v, pool,
                                 session=None, name="v", workspace=None, compact_prefill=False)
    counts = torch.tensor([3, 0, 4, 2], device=device, dtype=torch.int32)
    # Row 0: qsl=0, count=3 -> idx 2; Row 1: inactive; Row 2: qsl=6, count=4 -> idx 9; Row 3: qsl=10, count=2 -> idx 11
    ctx_v(aux_v, counts, meta_v)
    for o in range(3):
        assert torch.equal(pool.draft_index(o)[2], aux_v[2, o, :])
        assert torch.equal(pool.draft_index(o)[5], aux_v[9, o, :])
        assert torch.equal(pool.draft_index(o)[6], aux_v[11, o, :])


@pytest.mark.skipif(not torch.cuda.is_available() or not DFLY2_SNAPSHOT.is_dir(), reason="CUDA and local DFly2 checkpoint required")
def test_dfly2_weights_load_and_declare_weights():
    from hyperflare_recipes.recipes.deepseek41.dfly2_weights import DFly2Weights

    device = torch.device("cuda:0")
    target_cfg = _sample_target_config()
    dfly_cfg = DFly2Config.load(DFLY2_SNAPSHOT, target_cfg)
    model_cfg = replace(target_cfg, draft_override=dfly_cfg)
    shard = TensorParallelShard(0, 4)
    _, local_vocab = shard.padded_partition(129280, alignment=64)
    dummy_head = torch.randn((local_vocab, 5120), device=device, dtype=torch.bfloat16) * 0.02

    w = DFly2Weights.load(
        DFLY2_SNAPSHOT,
        model_cfg,
        device,
        shard=shard,
        target_head=dummy_head,
        draft_body_quantization="mxfp8",
        draft_head_quantization="nvfp4",
    )
    assert len(w.layers) == 3
    assert w.confidence_head is None
    assert w.draft_head is not None
    assert w.allocated_bytes > 0

    decl = declare_weights({"draft": w}, components=("draft",))
    payload = weight_payload_declaration(decl)
    assert len(payload["views"]) > 30


@pytest.mark.skipif(not torch.cuda.is_available() or not DFLY2_SNAPSHOT.is_dir(), reason="CUDA and local DFly2 checkpoint required")
def test_dfly2_draft_and_correction_cuda_graph_and_holed_batch_parity():
    from types import SimpleNamespace
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
    from hyperflare_recipes.recipes.deepseek41.decoder import DraftScratch
    from hyperflare_recipes.recipes.deepseek41.dfly2 import PreparedDFly2Draft
    from hyperflare_recipes.recipes.deepseek41.dfly2_weights import DFly2Weights

    device = torch.device("cuda:0")
    target_cfg = _sample_target_config()
    dfly_cfg = DFly2Config.load(DFLY2_SNAPSHOT, target_cfg)
    model_cfg = replace(target_cfg, draft_override=dfly_cfg)
    shard = TensorParallelShard(0, 4)
    _, local_vocab = shard.padded_partition(129280, alignment=64)

    gen = torch.Generator(device=device).manual_seed(777)
    embedding = torch.randn((129280, 5120), device=device, dtype=torch.bfloat16, generator=gen) * 0.02
    head = torch.randn((local_vocab, 5120), device=device, dtype=torch.bfloat16, generator=gen) * 0.02
    dfly_w = DFly2Weights.load(
        DFLY2_SNAPSHOT, model_cfg, device, shard=shard, target_head=head,
        draft_body_quantization="mxfp8", draft_head_quantization="nvfp4",
    )
    weights = SimpleNamespace(
        config=model_cfg, shard=shard, embedding=embedding, head=head, draft=dfly_w,
    )
    pool = ContinuationPool.allocate(model_cfg, slots=8, pages=16, device=device, dfly2=True)
    # Populate realistic SWA windows (layers 28, 32, 36), layer20 compressed latent pages, last_hidden, and draft_index{0,1,2}
    for tgt_l in (28, 32, 36):
        hist = torch.randn((8 * 128, 512), device=device, dtype=torch.bfloat16, generator=gen) * 0.1
        packed = pack_deepseek_v41_cache_reference(hist, page_size=128, cache_kind="swa")
        pool.window(tgt_l, draft=False).copy_(packed.reshape_as(pool.window(tgt_l, draft=False)))
    lat = torch.randn((16 * 128, 512), device=device, dtype=torch.bfloat16, generator=gen) * 0.1
    packed_lat = pack_deepseek_v41_cache_reference(lat, page_size=128, cache_kind="indexed")
    pool.pages["layer20.latent_page"].copy_(packed_lat.reshape_as(pool.pages["layer20.latent_page"]))
    pool.last_hidden.copy_(torch.randn((8, 5120), device=device, dtype=torch.bfloat16, generator=gen) * 0.1)
    for o in range(3):
        idx = torch.arange(512, device=device, dtype=torch.int32).unsqueeze(0).expand(8, -1)
        pool.draft_index(o).copy_(idx)

    batch, block = 4, 6
    meta_tpl = Metadata.allocate(batch * block, batch, 16, device, samples=batch * block, max_query_tokens=block)
    scratch = DraftScratch.allocate(batch * block, batch * block, 4, device, experts=128, top_k=3)
    arena = ScratchArena(device, overlap=False)
    with RecordedPreparationSession(
        compile_workers=1, autotune_effort="off", device=device, checkpoint="test-dfly2",
        workload_schema="test", numerical_policy="default",
    ) as session:
        session.register_scratch(arena)
        draft = PreparedDFly2Draft(
            weights, scratch, meta_tpl, pool,
            runtime=None, vocabulary_runtime=None, coordinator=None,
            session=session, name="test.dfly2", workspace=arena,
            mla_compute_mode="bf16", vocabulary_parallel=True,
        )
        session.tune(show_progress=False)
        arena.release_preparation()

    # Compare full active batch [0, 1, 2, 3] vs holed batch where row 1 is INACTIVE (length=0)!
    starts = torch.tensor([200, 0, 150, 300], device=device, dtype=torch.int64)
    states = torch.tensor([2, 3, 5, 6], device=device, dtype=torch.int64)
    pages = torch.arange(16, device=device, dtype=torch.int32).unsqueeze(0).expand(batch, -1).contiguous()
    draft_inputs = torch.full((batch, block), 128799, device=device, dtype=torch.int64)
    draft_inputs[:, 0] = torch.tensor([101, 202, 303, 404], device=device, dtype=torch.int64)

    # 1. Verify bit-exact parity of MTPFusion + saved indices + layer 0 attention between full and holed batches!
    def _layer0_attn(lengths):
        meta = Metadata.pack(
            starts, lengths, states, pages, draft_inputs,
            torch.tensor([batch], device=device, dtype=torch.int32),
            token_capacity=batch * block, sample_count=batch * block, max_query_tokens=block,
        )
        emb = draft.embedding(token_ids=meta.token_ids, num_rows=meta.live_counts[:1])
        slot0_pos = meta.query_start_loc[:-1].long().clamp(0, batch * block - 1)
        slot0 = emb[slot0_pos]
        last_hidden = torch.empty((batch, 5120), dtype=torch.bfloat16, device=device)
        from ignition.components.state import gather_rows
        from hyperflare_recipes.recipes.deepseek41.attention import normalize
        gather_rows(pool.last_hidden, meta.state_rows, last_hidden)
        en = torch.empty_like(slot0)
        hn = torch.empty_like(last_hidden)
        normalize(slot0, draft.mtp_enorm_w, en, draft.eps)
        normalize(last_hidden, draft.mtp_hnorm_w, hn, draft.eps)
        fused = draft.mtp_eh_proj(torch.cat((en, hn), dim=-1))
        delta = draft.mtp_gate.to(fused.dtype) * fused
        is_s0 = (meta.positions >= 0) & (meta.positions == meta.starts[meta.requests]) & (meta.starts[meta.requests] > 0)
        emb = torch.where(is_s0[:, None], emb + delta[meta.requests], emb)
        saved0 = torch.empty((batch, 512), dtype=torch.int32, device=device)
        gather_rows(pool.draft_index(0), meta.state_rows, saved0)
        ilen = torch.where(meta.positions >= 0, meta.starts[meta.requests].clamp(max=512), 0).to(torch.int32)
        inc = (emb[:, None, :].expand(-1, 4, -1).contiguous(),)
        pm = torch.zeros((batch * block, 4), dtype=torch.float32, device=device)
        pm[:, 0].fill_(1.0)
        indices = draft.window_metadata(meta)
        attn_mix = torch.empty_like(pm)
        _, _, _, normalized = draft.layers[0].attention_hc(*inc, pre_mix=pm, pre_out=attn_mix)
        attended, _, _ = draft.layers[0].attention(
            normalized, meta, (saved0[meta.requests], ilen, None), indices=indices,
        )
        return attended[meta.sample_indices].view(batch, block, -1)

    att_full = _layer0_attn(torch.tensor([6, 6, 6, 6], device=device, dtype=torch.int32))
    att_hole = _layer0_attn(torch.tensor([6, 0, 6, 6], device=device, dtype=torch.int32))
    for r in (0, 2, 3):
        torch.testing.assert_close(att_hole[r], att_full[r], atol=0.0, rtol=0.0)

    # 2. Verify full draft + recurrent correction parity & CUDAGraph capture/replay
    prev_seq = torch.tensor(
        [[101, 11, 12, 13, 14], [202, 21, 22, 23, 24], [303, 31, 32, 33, 34], [404, 41, 42, 43, 44]],
        device=device, dtype=torch.int64,
    )

    def _run(lengths):
        meta = Metadata.pack(
            starts, lengths, states, pages, draft_inputs,
            torch.tensor([batch], device=device, dtype=torch.int32),
            token_capacity=batch * block, sample_count=batch * block, max_query_tokens=block,
        )
        _, _, feats = draft(meta)
        hidden = feats["hidden"].view(batch, block, -1)
        ctx = feats["recurrent_context"]
        st = draft.recurrent.begin(ctx)
        outs = []
        for s_idx in range(block - 1):
            st, logits = draft.recurrent.step(st, prev_seq[:, s_idx], hidden[:, s_idx + 1].contiguous(), ctx, None)
            outs.append(logits)
        return hidden, torch.stack(outs, dim=1)

    h_full, l_full = _run(torch.tensor([6, 6, 6, 6], device=device, dtype=torch.int32))
    h_hole, l_hole = _run(torch.tensor([6, 0, 6, 6], device=device, dtype=torch.int32))

    assert torch.isfinite(h_full).all() and torch.isfinite(l_full).all()
    assert torch.isfinite(h_hole).all() and torch.isfinite(l_hole).all()
    for r in (0, 2, 3):
        cos = F.cosine_similarity(h_hole[r].float().reshape(-1), h_full[r].float().reshape(-1), dim=0)
        assert float(cos) > 0.95, f"row {r} hidden cosine {float(cos):.6f}"
        cos_l = F.cosine_similarity(l_hole[r].float().reshape(-1), l_full[r].float().reshape(-1), dim=0)
        assert float(cos_l) > 0.95, f"row {r} logits cosine {float(cos_l):.6f}"

    # 3. Verify bit-exact CUDA Graph capture & replay of PreparedDFly2Correction (5 sequential steps)
    meta_g = Metadata.pack(
        starts, torch.tensor([6, 0, 6, 6], device=device, dtype=torch.int32), states, pages, draft_inputs,
        torch.tensor([batch], device=device, dtype=torch.int32),
        token_capacity=batch * block, sample_count=batch * block, max_query_tokens=block,
    )
    _, _, fixed_feats = draft(meta_g)
    fixed_hidden = fixed_feats["hidden"].view(batch, block, -1).clone()
    fixed_ctx = tuple(t.clone() for t in fixed_feats["recurrent_context"])
    corr_eager = torch.empty_like(l_hole)
    st = draft.recurrent.begin(fixed_ctx)
    for s_idx in range(block - 1):
        st, logits = draft.recurrent.step(st, prev_seq[:, s_idx], fixed_hidden[:, s_idx + 1].contiguous(), fixed_ctx, None)
        corr_eager[:, s_idx].copy_(logits)

    corr_graph_out = torch.empty_like(l_hole)

    def _corr_graph_body():
        st_g = draft.recurrent.begin(fixed_ctx)
        for s_idx in range(block - 1):
            st_g, logits_g = draft.recurrent.step(
                st_g, prev_seq[:, s_idx], fixed_hidden[:, s_idx + 1].contiguous(), fixed_ctx, None,
            )
            corr_graph_out[:, s_idx].copy_(logits_g)

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        _corr_graph_body()
    stream.synchronize()
    g_corr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_corr, stream=stream):
        _corr_graph_body()
    corr_graph_out.fill_(float("nan"))
    g_corr.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(corr_graph_out, corr_eager, atol=0.0, rtol=0.0)

    # 4. Verify full end-to-end PreparedDFly2Draft + PreparedDFly2Correction CUDA Graph capture & replay
    out_buf = torch.empty_like(l_hole)

    def _graph_body():
        _, _, feats = draft(meta_g)
        hidden = feats["hidden"].view(batch, block, -1)
        ctx = feats["recurrent_context"]
        st_g = draft.recurrent.begin(ctx)
        for s_idx in range(block - 1):
            st_g, logits = draft.recurrent.step(st_g, prev_seq[:, s_idx], hidden[:, s_idx + 1].contiguous(), ctx, None)
            out_buf[:, s_idx].copy_(logits)

    with torch.cuda.stream(stream):
        _graph_body()
    stream.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        _graph_body()
    out_buf.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize(device)
    assert torch.isfinite(out_buf).all()
    for r in (0, 2, 3):
        cos_g = F.cosine_similarity(out_buf[r].float().reshape(-1), l_hole[r].float().reshape(-1), dim=0)
        assert float(cos_g) > 0.95, f"row {r} graph logits cosine {float(cos_g):.6f}"

