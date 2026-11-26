# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native HyperFlare DFly2-B6 drafter sharing target KV windows and sparse indices.

Key speed optimizations:
- Zero draft KV cache and zero prefill context GEMMs: `PreparedDFly2Context` saves
  only the 3x512 int32 target indexer selections (6 KiB/slot) for layers 28, 32, 36.
- Backbone layers reuse `PreparedLayer` (mHC, MXFP4 MoE, shared-expert overlap,
  compressed MLA over `pool.window(28/32/36)` + `layer20.latent_page`).
- Correction pre-splits `[h_norm; e_norm]` projections into two fused stacked matrices:
  `h_proj` runs ONCE batched across all `B * 6` block slots before the sequential loop,
  cutting weight bandwidth in the serial loop by 2x and GEMM launches by 5x.
- Slot 0 and slot 1 share the anchor's `e_proj` output (zero extra GEMM for slot 0).
- Fused Triton kernels combine SwiGLU, q_norm, kv_norm+RoPE, and gated causal
  intra-block attention + context sum; SwiGLU down_proj and attentive wo_b share
  one single fused-addend TP AllReduce per slot.
"""

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.attention_output import PreparedGroupedOutputProjection
from hyperflare_recipes.components.compressed_mla import PreparedCompressedMLA
from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.linear import prepare_projection
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from ignition.components.collectives import PreparedAllReduce
from ignition.components.state import gather_rows, gather_rows_three, save_hidden
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer

from .attention import AttentionIndices, PreparedBF16GroupedOutputProjection, Rotary, normalize
from .decoder import PreparedLayer


@triton.jit
def _corr_block_key0_kernel(
    H_PROJ,
    E_PROJ,
    KV_NORM_W,
    ANGLES,
    OUT_BK,
    H_ROW_STRIDE,
    E_ROW_STRIDE,
    KV_OFFSET: tl.constexpr,
    EPS: tl.constexpr,
):
    """Compute slot-0 block key: apply_rope(RMSNorm(h_kv[0] + e_kv[anchor], kv_norm), angles[0])."""
    b = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, 512)
    h_base = H_PROJ + b * H_ROW_STRIDE + KV_OFFSET
    e_base = E_PROJ + b * E_ROW_STRIDE + KV_OFFSET
    x = tl.load(h_base + c).to(tl.float32) + tl.load(e_base + c).to(tl.float32)
    w = tl.load(KV_NORM_W + c).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 512.0 + EPS)
    normed = (x * inv * w).to(tl.bfloat16)
    tl.store(OUT_BK + b * 512 + c, normed, mask=c < 448)
    pair = tl.arange(0, 32)
    xa = (tl.load(h_base + 448 + 2 * pair).to(tl.float32) + tl.load(e_base + 448 + 2 * pair).to(tl.float32))
    xb = (tl.load(h_base + 448 + 2 * pair + 1).to(tl.float32) + tl.load(e_base + 448 + 2 * pair + 1).to(tl.float32))
    wa = tl.load(KV_NORM_W + 448 + 2 * pair).to(tl.float32)
    wb = tl.load(KV_NORM_W + 448 + 2 * pair + 1).to(tl.float32)
    a = (xa * inv * wa).to(tl.bfloat16).to(tl.float32)
    bv = (xb * inv * wb).to(tl.bfloat16).to(tl.float32)
    cos = tl.load(ANGLES + b * 64 + pair)
    sin = tl.load(ANGLES + b * 64 + 32 + pair)
    tl.store(OUT_BK + b * 512 + 448 + 2 * pair, (a * cos - bv * sin).to(tl.bfloat16))
    tl.store(OUT_BK + b * 512 + 448 + 2 * pair + 1, (a * sin + bv * cos).to(tl.bfloat16))


@triton.jit
def _corr_combine_step_kernel(
    H_PROJ,
    E_PROJ,
    Q_NORM_W,
    KV_NORM_W,
    ANGLES,
    OUT_SWIGLU,
    OUT_Q_LOW,
    OUT_BK,
    OUT_GATE_RAW,
    H_ROW_STRIDE,
    E_ROW_STRIDE,
    INTER: tl.constexpr,
    INTER_PAD: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_PAD: tl.constexpr,
    EPS: tl.constexpr,
):
    """Fuse h_proj + e_proj into SwiGLU(gate, up), RMSNorm(q_in), RoPE(RMSNorm(kv_in)), and block_gate_raw."""
    b = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    h_base = H_PROJ + b * H_ROW_STRIDE
    e_base = E_PROJ + b * E_ROW_STRIDE
    if task == 0:
        c = tl.arange(0, INTER_PAD)
        m = c < INTER
        g = (tl.load(h_base + c, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
             + tl.load(e_base + c, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32))
        u = (tl.load(h_base + INTER + c, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
             + tl.load(e_base + INTER + c, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32))
        # Match PyTorch BF16 linear sum -> silu(g_bf16) * u_bf16
        g_bf = g.to(tl.bfloat16).to(tl.float32)
        u_bf = u.to(tl.bfloat16).to(tl.float32)
        silu_g = (g_bf * tl.sigmoid(g_bf)).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT_SWIGLU + b * INTER + c, (silu_g * u_bf).to(tl.bfloat16), mask=m)
    elif task == 1:
        qc = tl.arange(0, 2048)
        qm = qc < 1280
        off = 2 * INTER
        qx = (tl.load(h_base + off + qc, mask=qm, other=0.0).to(tl.float32)
              + tl.load(e_base + off + qc, mask=qm, other=0.0).to(tl.float32))
        qw = tl.load(Q_NORM_W + qc, mask=qm, other=0.0).to(tl.float32)
        qinv = tl.rsqrt(tl.sum(qx * qx, 0) / 1280.0 + EPS)
        tl.store(OUT_Q_LOW + b * 1280 + qc, (qx * qinv * qw).to(tl.bfloat16), mask=qm)
    elif task == 2:
        kc = tl.arange(0, 512)
        off = 2 * INTER + 1280
        kx = (tl.load(h_base + off + kc).to(tl.float32)
              + tl.load(e_base + off + kc).to(tl.float32))
        kw = tl.load(KV_NORM_W + kc).to(tl.float32)
        kinv = tl.rsqrt(tl.sum(kx * kx, 0) / 512.0 + EPS)
        tl.store(OUT_BK + b * 512 + kc, (kx * kinv * kw).to(tl.bfloat16), mask=kc < 448)
        pair = tl.arange(0, 32)
        xa = (tl.load(h_base + off + 448 + 2 * pair).to(tl.float32)
              + tl.load(e_base + off + 448 + 2 * pair).to(tl.float32))
        xb = (tl.load(h_base + off + 448 + 2 * pair + 1).to(tl.float32)
              + tl.load(e_base + off + 448 + 2 * pair + 1).to(tl.float32))
        wa = tl.load(KV_NORM_W + 448 + 2 * pair).to(tl.float32)
        wb = tl.load(KV_NORM_W + 448 + 2 * pair + 1).to(tl.float32)
        a = (xa * kinv * wa).to(tl.bfloat16).to(tl.float32)
        bv = (xb * kinv * wb).to(tl.bfloat16).to(tl.float32)
        cos = tl.load(ANGLES + b * 64 + pair)
        sin = tl.load(ANGLES + b * 64 + 32 + pair)
        tl.store(OUT_BK + b * 512 + 448 + 2 * pair, (a * cos - bv * sin).to(tl.bfloat16))
        tl.store(OUT_BK + b * 512 + 448 + 2 * pair + 1, (a * sin + bv * cos).to(tl.bfloat16))
    else:
        hc = tl.arange(0, HEADS_PAD)
        hm = hc < HEADS
        off = 2 * INTER + 1280 + 512
        bg = (tl.load(h_base + off + hc, mask=hm, other=0.0).to(tl.float32)
              + tl.load(e_base + off + hc, mask=hm, other=0.0).to(tl.float32))
        tl.store(OUT_GATE_RAW + b * HEADS + hc, bg, mask=hm)


@triton.jit
def _gated_block_attn_add_kernel(
    Q,
    BLOCK_KEYS,
    READ_CTX,
    REL_BIAS,
    BLOCK_SINK,
    GATE_RAW,
    GATE_BIAS,
    STARTS,
    OUT,
    HEADS: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    STEP: tl.constexpr,
):
    """Fused FP32 gated causal intra-block attention over slots 0..STEP plus context read."""
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 512)
    q = tl.load(Q + (b * HEADS + h) * 512 + d).to(tl.float32)
    scale = 0.04419417382415922  # 512 ** -0.5

    # Online log-sum-exp over keys j = 0..STEP plus block_sink
    sink = tl.load(BLOCK_SINK + h).to(tl.float32)
    running_max = sink
    running_sum = tl.full((), 1.0, tl.float32)  # exp(sink - sink)
    acc = tl.zeros((512,), dtype=tl.float32)

    for j in tl.static_range(STEP + 1):
        key = tl.load(BLOCK_KEYS + (b * (STEP + 1) + j) * 512 + d).to(tl.float32)
        dot = tl.sum(q * key, 0) * scale
        bias = tl.load(REL_BIAS + h * BIAS_STRIDE + (STEP - j)).to(tl.float32)
        logit = dot + bias
        new_max = tl.maximum(running_max, logit)
        alpha = tl.exp(running_max - new_max)
        w = tl.exp(logit - new_max)
        running_sum = running_sum * alpha + w
        acc = acc * alpha + w * key
        running_max = new_max

    read_blk = acc / running_sum
    gate_in = tl.load(GATE_RAW + b * HEADS + h).to(tl.float32) + tl.load(GATE_BIAS + h * BIAS_STRIDE + STEP).to(tl.float32)
    # tanh(x) = 2 * sigmoid(2x) - 1
    gate = 2.0 * tl.sigmoid(2.0 * gate_in) - 1.0
    has_ctx = tl.load(STARTS + b) > 0
    ctx = tl.where(has_ctx, tl.load(READ_CTX + (b * HEADS + h) * 512 + d).to(tl.float32), 0.0)
    total = (ctx + read_blk * gate).to(tl.bfloat16)
    tl.store(OUT + (b * HEADS + h) * 512 + d, total)


class PreparedDFly2Context:
    """Zero-GEMM DFly2 context updater saving target layer 28/32/36 top-512 indices."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, compact_prefill=False):
        del session, name, workspace
        self.pool = pool
        self.layers = len(weights.config.draft.target_index_layers)
        self.compact = compact_prefill and metadata.starts.numel() == 1 and metadata.positions.numel() > 128
        if self.compact and auxiliary.shape[0] != 128:
            raise ValueError("compact prefill DFly2 index capture must contain the 128-row suffix")

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        if self.compact:
            kept = counts[0].long().clamp(max=auxiliary.shape[0])
            last = (kept - 1).clamp_min(0).view(1)
        else:
            last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, auxiliary.shape[0] - 1)
        active = counts > 0
        for ordinal in range(self.layers):
            save_hidden(auxiliary[last, ordinal, :], self.pool.draft_index(ordinal), m.state_rows, active)


class PreparedDFly2Correction:
    """Sequential HiddenStatesCorrection + DSV41AttentiveCorrection implementing PreparedParallelDraftRound.recurrent."""

    def __init__(
        self,
        weights,
        batch,
        blocks,
        pool,
        rotary,
        *,
        runtime,
        coordinator,
        session,
        name,
        workspace,
        mla_compute_mode="bf16",
    ):
        c = weights.draft.correction
        text = weights.config.text
        device = weights.embedding.device
        self.batch = batch
        self.eps = text["rms_norm_eps"]
        self.local_corr_inter = c["local_corr_inter"]
        self.inter_pad = triton.next_power_of_2(self.local_corr_inter)
        self.local_heads = c["local_heads"]
        self.heads_pad = triton.next_power_of_2(self.local_heads)
        self.fused_rows = c["fused_rows"]
        self.hidden_size = text["hidden_size"]
        self.kv_offset = 2 * self.local_corr_inter + text["q_lora_rank"]

        self.embed_norm_w = c["embed_norm"]
        self.q_norm_w = c["q_norm"]
        self.kv_norm_w = c["kv_norm"]
        self.block_sink = c["block_sink"]
        self.block_rel_bias = c["block_rel_bias"]
        self.block_gate_bias = c["block_gate_bias"]
        for t in (self.embed_norm_w, self.q_norm_w, self.kv_norm_w,
                  self.block_sink, self.block_rel_bias, self.block_gate_bias):
            record_numeric_tensor(t)

        # Single-step [B, ...] template tensors for preparation
        tok_tpl = torch.zeros((batch,), dtype=torch.int64, device=device)
        emb_tpl = torch.zeros((batch, self.hidden_size), dtype=torch.bfloat16, device=device)
        enorm_tpl = torch.zeros_like(emb_tpl)
        eproj_tpl = torch.zeros((batch, self.fused_rows), dtype=torch.bfloat16, device=device)
        swiglu_tpl = torch.zeros((batch, self.local_corr_inter), dtype=torch.bfloat16, device=device)
        qlow_tpl = torch.zeros((batch, text["q_lora_rank"]), dtype=torch.bfloat16, device=device)
        q_tpl = torch.zeros((batch, self.local_heads, text["head_dim"]), dtype=torch.bfloat16, device=device)
        out_tpl = torch.zeros_like(q_tpl)
        partial_tpl = torch.zeros((batch, self.hidden_size), dtype=torch.bfloat16, device=device)
        reduced_tpl = torch.zeros_like(partial_tpl)
        widx_tpl = torch.full((batch, 128), -1, dtype=torch.int32, device=device)
        wlen_tpl = torch.zeros((batch,), dtype=torch.int32, device=device)
        iidx_tpl = torch.full((batch, weights.config.draft.index_topk), -1, dtype=torch.int32, device=device)
        ilen_tpl = torch.zeros((batch,), dtype=torch.int32, device=device)
        pt_tpl = torch.zeros((batch, blocks), dtype=torch.int32, device=device)

        self.token_offsets = persistent_copy(torch.arange(batch, dtype=torch.int64, device=device))
        record_numeric_tensor(self.token_offsets)
        angles_tpl = torch.zeros((batch, 64), dtype=torch.float32, device=device)

        self.embed = PreparedEmbedding(
            weights.embedding, tok_tpl, emb_tpl, session=session, name=name + ".embed"
        ).functional(session=session, dynamic_inputs=True)
        self.embed_norm = PreparedRMSNorm(
            emb_tpl, self.embed_norm_w, enorm_tpl, eps=self.eps, session=session, name=name + ".embed_norm"
        ).functional()
        self.e_proj = prepare_projection(
            enorm_tpl, c["e_proj"], eproj_tpl, session=session, name=name + ".e_proj", workspace=workspace
        ).functional(session=session)
        self.down_proj = prepare_projection(
            swiglu_tpl, c["down_proj"], partial_tpl, session=session, name=name + ".down_proj", workspace=workspace
        ).functional(session=session)
        self.wq_b = prepare_projection(
            qlow_tpl, c["wq_b"], q_tpl.view(batch, -1), session=session, name=name + ".wq_b", workspace=workspace
        ).functional(session=session)

        tgt_layer = weights.config.draft.target_kv_layers[-1]
        source_layer = weights.config.source(tgt_layer).layer
        self.corr_attn = PreparedCompressedMLA(
            q_tpl,
            pool.window(tgt_layer, draft=False),
            widx_tpl,
            wlen_tpl,
            c["attn_sink"],
            out_tpl,
            indexed=pool.pages[f"layer{source_layer}.latent_page"],
            indexed_indices=iidx_tpl,
            indexed_lengths=ilen_tpl,
            page_table=pt_tpl,
            window_page_size=128,
            indexed_page_size=128,
            mode="decode",
            decode_row_capacity=batch,
            session=session,
            name=name + ".attn",
            workspace=workspace,
            compute_mode=mla_compute_mode,
        ).functional(session=session)

        # Shared split rotary angles helper for q RoPE (and BF16 inv_rope fallback)
        from hyperflare_recipes.components.rotary import PreparedRotaryAngles
        self.rotary_helper = PreparedRotaryAngles(self.token_offsets, rotary.frequencies[True], split=True).functional()

        if isinstance(c["output"], dict):
            self.output_proj = PreparedBF16GroupedOutputProjection(
                out_tpl, c["output"], partial_tpl, session=session, name=name + ".output", angles=self.rotary_helper
            ).functional(session=session)
        else:
            self.output_proj = PreparedGroupedOutputProjection(
                out_tpl, c["output"], partial_tpl, session=session, name=name + ".output", workspace=workspace,
                inv_rope=True, heads_per_group=self.local_heads // c["output"].groups,
                nope_dim=448, rope_dim=64, positions=self.token_offsets, cos_sin_cache=angles_tpl,
            ).functional(session=session)

        self.reduce = PreparedAllReduce(
            partial_tpl, reduced_tpl, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".reduce", has_addend=True,
        ).functional(session=session)

        head = weights.head if weights.draft.draft_head is None else weights.draft.draft_head
        local_vocab = head.shape[0] if isinstance(head, torch.Tensor) else head.out_features
        local_logits_tpl = torch.zeros((batch, local_vocab), dtype=torch.bfloat16, device=device)
        self.head = PreparedVocabularyProjection(
            reduced_tpl, head, local_logits_tpl, local_logits_tpl,
            runtime=None, coordinator=coordinator, session=session,
            name=name + ".head", workspace=workspace, gather=False,
        ).functional(session=session)

    def begin(self, context):
        del context
        return (0, ())

    def step(self, state, previous, hidden, context, base):
        del base
        step_idx, prev_keys = state
        k = step_idx + 1  # block slot 1..5
        (h_proj_all, angles_all, corr_w_idx, corr_w_len, corr_i_idx, corr_i_len, page_table, starts) = context
        b = self.batch
        dev = hidden.device

        e_norm = self.embed_norm(self.embed(token_ids=previous.contiguous()))
        e_proj = self.e_proj(e_norm)

        # Slot 0 and slot 1 share anchor's e_proj — compute block_key[0] with zero extra GEMM!
        if step_idx == 0:
            bk0 = torch.empty((b, 512), dtype=torch.bfloat16, device=dev)
            h0 = h_proj_all[:, 0, :]
            a0 = angles_all[:, 0, :].contiguous()
            torch.library.wrap_triton(_corr_block_key0_kernel)[(b,)](
                h0, e_proj, self.kv_norm_w, a0, bk0,
                h0.stride(0), e_proj.stride(0),
                KV_OFFSET=self.kv_offset, EPS=self.eps, num_warps=4,
            )
            prev_keys = (bk0,)

        hk = h_proj_all[:, k, :]
        ak = angles_all[:, k, :].contiguous()
        swiglu_inter = torch.empty((b, self.local_corr_inter), dtype=torch.bfloat16, device=dev)
        q_low = torch.empty((b, 1280), dtype=torch.bfloat16, device=dev)
        bk_k = torch.empty((b, 512), dtype=torch.bfloat16, device=dev)
        gate_raw = torch.empty((b, self.local_heads), dtype=torch.float32, device=dev)
        torch.library.wrap_triton(_corr_combine_step_kernel)[(b, 4)](
            hk, e_proj, self.q_norm_w, self.kv_norm_w, ak,
            swiglu_inter, q_low, bk_k, gate_raw,
            hk.stride(0), e_proj.stride(0),
            INTER=self.local_corr_inter, INTER_PAD=self.inter_pad,
            HEADS=self.local_heads, HEADS_PAD=self.heads_pad,
            EPS=self.eps, num_warps=4,
        )

        delta_mlp_partial = self.down_proj(swiglu_inter)

        q_k = self.wq_b(q_low).view(b, self.local_heads, 512)
        self.rotary_helper.apply(q_k, ak)
        read_ctx = self.corr_attn(
            q_k, corr_w_idx, corr_w_len,
            indexed_indices=corr_i_idx, indexed_lengths=corr_i_len, page_table=page_table,
        )

        all_keys = (*prev_keys, bk_k)
        keys_stack = torch.stack(all_keys, dim=1).contiguous()  # [B, k+1, 512]
        read_total = torch.empty_like(read_ctx)
        torch.library.wrap_triton(_gated_block_attn_add_kernel)[(b, self.local_heads)](
            q_k, keys_stack, read_ctx,
            self.block_rel_bias, self.block_sink, gate_raw, self.block_gate_bias, starts, read_total,
            HEADS=self.local_heads, BIAS_STRIDE=self.block_rel_bias.stride(0), STEP=k,
            num_warps=4,
        )

        delta_attn_partial = self.output_proj(read_total, positions=self.token_offsets, cos_sin_cache=ak)
        delta = (
            self.reduce(delta_mlp_partial, addend=delta_attn_partial)
            if getattr(self.reduce, "fused_addend", False)
            else self.reduce(delta_mlp_partial + delta_attn_partial)
        )
        h_corr = hidden + delta
        logits = self.head(h_corr)
        return (step_idx + 1, all_keys), logits


class PreparedDFly2Draft:
    """Block-parallel DFly2-B6 backbone + recurrent attentive correction."""

    recurrent_only = True

    def __init__(
        self,
        weights,
        scratch,
        metadata,
        pool,
        *,
        runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name,
        workspace,
        mla_compute_mode="bf16",
        vocabulary_parallel=False,
        examples=None,
    ):
        del vocabulary_runtime
        if not vocabulary_parallel:
            raise ValueError("DeepSeek V4.1 DFly2 draft requires vocabulary_parallel_sampling")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        draft_cfg = weights.config.draft
        device = scratch.embedding.device
        tokens = metadata.positions.numel()
        batch = metadata.starts.numel()
        block = draft_cfg.block_size
        if tokens != batch * block:
            raise ValueError(f"DFly2 draft requires tokens == batch * {block}, got {tokens} vs {batch} * {block}")
        self.batch, self.block = batch, block
        self.eps = weights.config.text["rms_norm_eps"]
        self.hidden_size = weights.config.text["hidden_size"]
        self.target_kv_layers = draft_cfg.target_kv_layers

        self.embedding = PreparedEmbedding(
            weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding",
        ).functional(session=session, dynamic_inputs=True)

        # MTPFusion on slot 0 (anchor slot)
        mf = weights.draft.mtp_fusion
        self.mtp_enorm_w = mf["enorm"]
        self.mtp_hnorm_w = mf["hnorm"]
        self.mtp_gate = mf["gate"]
        self.final_norm_w = weights.draft.norm
        self.corr_hnorm_w = weights.draft.correction["hidden_norm"]
        for t in (self.mtp_enorm_w, self.mtp_hnorm_w, self.mtp_gate, self.final_norm_w, self.corr_hnorm_w):
            record_numeric_tensor(t)
        mf_in_tpl = torch.zeros((batch, 2 * self.hidden_size), dtype=torch.bfloat16, device=device)
        mf_out_tpl = torch.zeros((batch, self.hidden_size), dtype=torch.bfloat16, device=device)
        self.mtp_eh_proj = prepare_projection(
            mf_in_tpl, mf["eh_proj"], mf_out_tpl,
            session=session, name=name + ".mtp_fusion", workspace=workspace,
        ).functional(session=session)

        rotary = Rotary(weights.config, device)
        page_size = 128 + metadata.max_query_tokens
        self.window_metadata = AttentionIndices(
            metadata, mode="draft", tokens=tokens, page_size=page_size, width=page_size,
        )
        self.layers = tuple(
            PreparedLayer(
                i, layer, None, scratch, metadata, pool, rotary,
                shard=weights.shard, mode="draft", runtime=runtime, coordinator=coordinator,
                session=session, name=f"{name}.layer{i}", workspace=workspace,
                carry=i > 0, defer_post=i + 1 < len(weights.draft.layers),
                mla_compute_mode=mla_compute_mode, window_metadata=self.window_metadata,
                examples=examples, dfly_target_layer=self.target_kv_layers[i],
            )
            for i, layer in enumerate(weights.draft.layers)
        )
        self.window_metadata.functional()

        # Pre-loop batched correction h_proj across all [B * 6] slots
        c = weights.draft.correction
        h_norm_tpl = torch.zeros((tokens, self.hidden_size), dtype=torch.bfloat16, device=device)
        h_proj_tpl = torch.zeros((tokens, c["fused_rows"]), dtype=torch.bfloat16, device=device)
        self.corr_h_norm = PreparedRMSNorm(
            scratch.hidden, self.corr_hnorm_w, h_norm_tpl, eps=self.eps,
            session=session, name=name + ".corr.hidden_norm",
        ).functional()
        self.corr_h_proj = prepare_projection(
            h_norm_tpl, c["h_proj"], h_proj_tpl,
            session=session, name=name + ".corr.h_proj", workspace=workspace,
        ).functional(session=session)

        self.recurrent = PreparedDFly2Correction(
            weights, batch, metadata.page_table.shape[1], pool, rotary,
            runtime=runtime, coordinator=coordinator, session=session,
            name=name + ".corr", workspace=workspace, mla_compute_mode=mla_compute_mode,
        )

        self.head_hidden_shape = (tokens, self.hidden_size)
        self.logit_shape, self.logit_dtype = tuple(scratch.logits.shape), scratch.logits.dtype
        self.local_logit_shape = (batch * (block - 1), scratch.local_logits.shape[-1])
        self.local_logit_dtype = torch.bfloat16
        self.window_offsets = persistent_copy(torch.arange(128, dtype=torch.int64, device=device))
        record_numeric_tensor(self.window_offsets)
        register_finalizer(session, self.release_outputs)

    def release_outputs(self):
        self.scratch = self.metadata = None

    def __call__(self, metadata):
        b, block = self.batch, self.block
        dev = metadata.token_ids.device
        embedding = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])

        # Slot-0 MTPFusion when starts > 0, safe under compacted Metadata.pack with inactive lanes
        slot0_pos = metadata.query_start_loc[:-1].long().clamp(0, b * block - 1)
        slot0 = embedding[slot0_pos]
        last_hidden = torch.empty((b, self.hidden_size), dtype=torch.bfloat16, device=dev)
        gather_rows(self.pool.last_hidden, metadata.state_rows, last_hidden)
        en = torch.empty_like(slot0)
        hn = torch.empty_like(last_hidden)
        normalize(slot0, self.mtp_enorm_w, en, self.eps)
        normalize(last_hidden, self.mtp_hnorm_w, hn, self.eps)
        fused = self.mtp_eh_proj(torch.cat((en, hn), dim=-1))
        delta_slot0 = self.mtp_gate.to(fused.dtype) * fused
        is_slot0 = (
            (metadata.positions >= 0)
            & (metadata.positions == metadata.starts[metadata.requests])
            & (metadata.starts[metadata.requests] > 0)
        )
        embedding = torch.where(is_slot0[:, None], embedding + delta_slot0[metadata.requests], embedding)

        # Gather saved target top-512 indices for layers 28, 32, 36 in one fused kernel
        saved_per_layer = tuple(
            torch.empty((b, 512), dtype=torch.int32, device=dev) for _ in range(len(self.target_kv_layers))
        )
        if len(saved_per_layer) == 3:
            gather_rows_three(
                (self.pool.draft_index(0), self.pool.draft_index(1), self.pool.draft_index(2)),
                metadata.state_rows,
                saved_per_layer,
            )
        else:
            for ordinal, buf in enumerate(saved_per_layer):
                gather_rows(self.pool.draft_index(ordinal), metadata.state_rows, buf)

        live = metadata.positions >= 0
        idx_lengths_tok = torch.where(live, metadata.starts[metadata.requests].clamp(max=512), 0).to(torch.int32)

        incoming = (embedding[:, None, :].expand(-1, 4, -1).contiguous(),)
        pre_mix = torch.zeros((b * block, 4), dtype=torch.float32, device=dev)
        pre_mix[:, 0].fill_(1.0)
        indices = self.window_metadata(metadata)
        for ordinal, layer in enumerate(self.layers):
            layer_sel = (saved_per_layer[ordinal][metadata.requests], idx_lengths_tok, None)
            incoming, pre_mix, _, _ = layer(incoming, pre_mix, metadata, indices, layer_sel)

        residual, = incoming
        samples = metadata.sample_indices
        collapsed = (residual[samples].float() * pre_mix[samples, :, None]).sum(1).bfloat16()
        h_backbone = torch.empty_like(collapsed)
        normalize(collapsed, self.final_norm_w, h_backbone, self.eps)

        # Precompute batched h_norm + h_proj for all 6 slots once before the sequential correction loop
        h_proj_all = self.corr_h_proj(self.corr_h_norm(h_backbone)).view(b, block, -1)
        angles_all = indices.angles[(True, 0)][samples].view(b, block, 64)

        # Context-only SWA window indices into pool.window(36) for AttentiveCorrection
        active = metadata.lengths > 0
        ctx_swa_len = torch.where(active, metadata.starts.clamp(max=128), 0).to(torch.int32)
        left = (metadata.starts - 128).clamp_min(0)
        ring_pos = (left[:, None] + self.window_offsets[None, :]) % 128
        raw_widx = (metadata.state_rows[:, None] * 128 + ring_pos).to(torch.int32)
        corr_w_idx = torch.where(self.window_offsets[None, :] < ctx_swa_len[:, None], raw_widx, -1)
        corr_i_len = torch.where(active, metadata.starts.clamp(max=512), 0).to(torch.int32)
        starts_masked = torch.where(active, metadata.starts, 0)

        recurrent_context = (
            h_proj_all,
            angles_all,
            corr_w_idx,
            ctx_swa_len,
            saved_per_layer[-1],
            corr_i_len,
            metadata.page_table,
            starts_masked,
        )
        features = {"hidden": h_backbone, "recurrent_context": recurrent_context}
        return None, None, features
