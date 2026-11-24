# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek text stack with lagged hyperconnections."""

from dataclasses import dataclass, fields, is_dataclass, replace

import torch

from ignition.components.collectives import ConcurrentAllGather, ConcurrentAllReduce, PreparedAllReduce
from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.expert_overlap import ConcurrentSharedExpert
from hyperflare_recipes.components.ring_commit import PreparedRingCommit
from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.mhc import MHCWeights, MHCScratch, PreparedMHC, PreparedMHCPost
from hyperflare_recipes.components.mhc_overlap import SplitLaggedMHC
from hyperflare_recipes.components.mlp import MLPScratch, PreparedMLP
from hyperflare_recipes.components.moe import RouterScratch, PreparedRouter, PreparedRoutedExperts
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from hyperflare_recipes.components.parallel import TensorParallelShard
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer

from .attention import AttentionIndices, AttentionScratch, PreparedAttention, PreparedGlobalKV, Rotary, normalize
from .decoder_prefill import PreparedDecoderSuffix
from .engram import EngramScratch, PreparedEngram, PreparedTableRequests


@dataclass(frozen=True)
class Scratch:
    """BF16 decoder activations and FP32 target vocabulary logits."""

    logits_dtype = torch.float32

    embedding: torch.Tensor
    residual: torch.Tensor
    engram_residual: torch.Tensor
    engram: EngramScratch
    pre_mix: torch.Tensor
    attn_mix: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    routed: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            if is_dataclass(value):
                return type(value)(*(select(getattr(value, field.name), count) for field in fields(value)))
            raise TypeError("decoder scratch must contain tensors and tensor dataclasses")
        return type(self)(*(select(getattr(self, field.name), samples if field.name in
            ("hidden", "local_logits", "logits") else tokens) for field in fields(self)))

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, experts=384, top_k=6,
                 candidate_tokens=None):
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        residual, engram_residual = zeros(tokens, 4, 5120), zeros(tokens, 4, 5120)
        pre_mix, attn_mix = zeros(tokens, 4, dtype=torch.float32), zeros(tokens, 4, dtype=torch.float32)
        attention_hc = MHCScratch.allocate(tokens, 5120, device)
        ffn_hc = MHCScratch.allocate(tokens, 5120, device)
        attention = AttentionScratch.allocate(tokens, world, device, candidate_tokens=candidate_tokens)
        attention_output, mlp_output = zeros(tokens, 5120), zeros(tokens, 5120)
        _, vocabulary = TensorParallelShard(0, world).padded_partition(129280, alignment=64)
        hidden = zeros(samples, 5120)
        local_logits = zeros(samples, vocabulary, dtype=cls.logits_dtype)
        logits = zeros(samples, 129280, dtype=cls.logits_dtype)
        embedding = zeros(tokens, 5120)
        engram = EngramScratch.allocate(tokens, device, world=world)
        shared = MLPScratch.allocate(tokens, 5120, 2304 // world, device)
        router_logits = zeros(tokens, experts, dtype=torch.float32)
        router = RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k)
        routed, partial = zeros(tokens, 5120), zeros(tokens, 5120)
        return cls(embedding, residual, engram_residual, engram, pre_mix, attn_mix, attention_hc, ffn_hc,
            attention, attention_output, shared, router_logits, router, routed, partial, mlp_output,
            hidden, local_logits, logits)



class DraftScratch(Scratch):
    """Retain BF16 proposal logits independently of FP32 target logits."""

    logits_dtype = torch.bfloat16


def _is_dma_runtime(runtime):
    if runtime is None:
        return False
    from b12x.comm import pcie

    return isinstance(runtime, pcie.DmaAllReduce)


def _runtime_accepts(runtime, tensor):
    if runtime is None:
        return True
    if _is_dma_runtime(runtime):
        return runtime.should_allreduce(tensor)
    if getattr(runtime, "algorithm", None) == "oneshot":
        return runtime.for_stream().should_allreduce(tensor)
    accepts = getattr(runtime, "should_allreduce", None)
    return True if accepts is None else accepts(tensor)


def _mhc_weights(weights, kind, norm):
    residual = weights.residual
    return MHCWeights(residual[f"hc_{kind}_fn"], residual[f"hc_{kind}_scale"],
                      residual[f"hc_{kind}_base"], residual[norm + ".weight"])


class PreparedDecoderBoundary:
    """Produce complete global KV, then optionally gather the decoder's bounded replay rows."""

    def __init__(self, index, weights, source, target, packing, pool, rotary, *, carry,
                 session, name, workspace, window_metadata, source_indices):
        self.packing = packing
        incoming = ((source.mlp_output, source.ffn_hc.residual, source.ffn_hc.post, source.ffn_hc.comb)
                    if carry else (source.residual,))
        text = pool.config.text
        self.mhc = PreparedMHC(incoming, _mhc_weights(weights, "attn", "attn_norm"), source.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace,
            pre_mix=source.pre_mix, pre_out=source.attn_mix, rms_eps=text["rms_norm_eps"],
            norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"], sinkhorn_iters=text["hc_sinkhorn_iters"]).functional()
        self.global_kv = PreparedGlobalKV(index, weights.attention, source.attention_hc.normalized, source.attention,
            source_indices.metadata, pool, rotary, session=session, name=name + ".global_kv",
            indices=source_indices)
        self.indexer = None
        if packing is None:
            from hyperflare_recipes.components.mxfp4_indexer import PreparedMXFP4Indexer

            query_rows = min(source.attention.index_query.shape[0], 128)
            geometry = source_indices.compressed(self.global_kv.ratio)
            candidates = {}
            if (index == text.get("candidate_source_layer_id")
                    and source.attention.candidates.shape[0] >= query_rows):
                candidates = dict(
                    candidates_out=source.attention.candidates[:query_rows],
                    counts_out=source.attention.candidate_counts[:query_rows],
                )
            self.indexer = PreparedMXFP4Indexer(
                source.attention.index_query[:query_rows],
                source.attention.index_weights[:query_rows],
                pool.pages[f"layer{index}.index_page"],
                source_indices.metadata.token_page_table[:query_rows],
                geometry.lengths[:query_rows],
                geometry.active_width,
                source.attention.selected[:query_rows],
                page_size=128 // self.global_kv.ratio,
                session=session,
                name=name + ".index0",
                workspace=workspace,
                mode="prefill",
                **candidates,
            ).functional(session=session)

    def __call__(self, incoming, pre_mix, metadata, indices, indexer=None):
        suffix = None if self.packing is None else self.packing.begin(metadata)
        attn_mix = torch.empty_like(pre_mix)
        residual, post, comb, normalized = self.mhc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
        self.global_kv(normalized, self.indexer if indexer is None else indexer, metadata,
            indices.compressed(self.global_kv.ratio),
            angles=indices.angles[(True, self.global_kv.ratio)])
        if suffix is None:
            return None, None, None
        return suffix, tuple(suffix.gather(value) for value in (residual, post, comb, normalized)), suffix.gather(attn_mix)


class PreparedLayer:
    def __init__(self, index, weights, tables, scratch, metadata, pool, rotary, *, shard, mode,
                 runtime, coordinator, session, name, workspace, carry, defer_post,
                 boundary=None, replay_start=None, window_staging=None, mla_compute_mode="auto",
                 window_metadata=None, examples=None, pipelined_ffn=False, pipelined_attn_hc=False,
                 defer_mlp_b=False, defer_engram_reduce=False, pipelined_attn=None):
        s, text = scratch, pool.config.text
        self.scratch, self.carry = s, carry
        self.metadata = metadata
        self.image_mask = (torch.zeros(metadata.token_ids.shape, dtype=torch.bool, device=metadata.token_ids.device)
                           if mode == "prefill" and weights.experts.vision_correction is not None else None)
        self.boundary = boundary
        self.pipelined_ffn = bool(pipelined_ffn)
        self.pipelined_attn_hc = bool(pipelined_attn_hc)
        self.pipelined_attn = pipelined_attn
        self.defer_mlp_b = bool(defer_mlp_b)
        tokens = s.attention.output.shape[0]
        h = tokens // 2
        self.half_tokens = h
        if self.pipelined_attn_hc and (not self.pipelined_ffn or not carry or boundary is not None):
            raise ValueError("pipelined attention mHC requires a pipelined carry layer without a boundary")
        if self.pipelined_attn is not None and (not self.pipelined_ffn or boundary is not None):
            raise ValueError("pipelined attention requires a pipelined FFN layer without a boundary")
        if self.defer_mlp_b and (not self.pipelined_ffn or not defer_post):
            raise ValueError("deferred mlp_B overlap requires pipelined FFN with deferred post")
        self.engram = None
        if weights.engram is not None:
            self.engram = PreparedEngram(index, weights.engram, tables, s.residual, s.engram_residual,
                metadata, pool,
                scratch=s.engram, session=session, name=name + ".engram", workspace=workspace,
                runtime=runtime, coordinator=coordinator, examples=examples,
                defer_reduce_functional=defer_engram_reduce)
        source = s.engram_residual if self.engram is not None else s.residual
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        from b12x.norm.mhc._split_lagged import supports_split_lagged

        self.split_mhc = (
            mode in ("decode", "verify", "draft")
            and not self.pipelined_ffn
            and not self.pipelined_attn_hc
            and self.pipelined_attn is None
            and boundary is None
            and runtime is not None
            and shard.world_size > 1
            and isinstance(workspace, ScratchArena)
            and workspace.overlap
            and supports_split_lagged(
                source.device, tokens, source.shape[-1], weights.residual["attn_norm.weight"].dtype)
            and supports_split_lagged(
                source.device, tokens, source.shape[-1], weights.residual["ffn_norm.weight"].dtype)
        )
        if self.pipelined_attn_hc or self.pipelined_attn == "full":
            attn_incoming = ((s.mlp_output[:h], s.ffn_hc.residual[:h], s.ffn_hc.post[:h], s.ffn_hc.comb[:h])
                             if carry else (source[:h],))
            self.attention_hc = PreparedMHC(
                attn_incoming, _mhc_weights(weights, "attn", "attn_norm"), s.attention_hc.rows(h),
                session=session, name=name + ".attn_hc", workspace=workspace,
                pre_mix=s.pre_mix[:h], pre_out=s.attn_mix[:h], **options)
        elif self.split_mhc:
            incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if carry else (source,))
            self.attention_hc = SplitLaggedMHC(
                incoming, _mhc_weights(weights, "attn", "attn_norm"),
                session=session, workspace=workspace, **options)
        else:
            incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if carry else (source,))
            self.attention_hc = None if boundary is not None else PreparedMHC(
                incoming, _mhc_weights(weights, "attn", "attn_norm"), s.attention_hc,
                session=session, name=name + ".attn_hc", workspace=workspace, pre_mix=s.pre_mix, pre_out=s.attn_mix, **options)
        self.attention = PreparedAttention(index, weights.attention, s.attention_hc.normalized, s.attention,
            metadata, pool, rotary, mode=mode, world=shard.world_size,
            session=session, name=name + ".attention", workspace=workspace, replay_start=replay_start,
            window_staging=window_staging, mla_compute_mode=mla_compute_mode,
            window_metadata=window_metadata, examples=examples, pipelined_mode=self.pipelined_attn)
        if self.pipelined_ffn:
            self.attention_reduce = None
            self.att_reduce_0 = PreparedAllReduce(s.attention.output[:h], s.attention_output[:h], runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".attention_reduce.0")
            self.att_reduce_1 = PreparedAllReduce(s.attention.output[h:], s.attention_output[h:], runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".attention_reduce.1")
            ffn_hc_scratch = s.ffn_hc.rows(h)
            self.ffn_hc = PreparedMHC((s.attention_output[:h], s.attention_hc.residual[:h],
                s.attention_hc.post[:h], s.attention_hc.comb[:h]),
                _mhc_weights(weights, "ffn", "ffn_norm"), ffn_hc_scratch, session=session,
                name=name + ".ffn_hc", workspace=workspace,
                pre_mix=s.attn_mix[:h], pre_out=s.pre_mix[:h], **options)
            self.router_projection = UnquantizedProjection(ffn_hc_scratch.normalized, weights.experts.router,
                s.router_logits[:h], session=session, name=name + ".router")
            router_scratch = s.router.rows(h)
            self.router = PreparedRouter(s.router_logits[:h], router_scratch, session=session, name=name + ".topk",
                score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
                routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
                image_correction_bias=weights.experts.vision_correction if self.image_mask is not None else None,
                image_mask=None if self.image_mask is None else self.image_mask[:h])
            self.experts = PreparedRoutedExperts(ffn_hc_scratch.normalized, weights.experts.experts,
                router_scratch.ids, router_scratch.weights, s.routed[:h], session=session, name=name + ".experts",
                workspace=workspace, router_plan=self.router.plan)
            shared_scratch = s.shared.rows(h)
            self.shared = PreparedMLP(ffn_hc_scratch.normalized, weights.experts.shared, shared_scratch.gate_up,
                shared_scratch.intermediate, shared_scratch.output, session=session, name=name + ".shared",
                workspace=workspace, swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")
            self.concurrent_shared = None
            self.mlp_reduce = None
            self.mlp_reduce_0 = PreparedAllReduce(s.mlp_partial[:h], s.mlp_output[:h], runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".mlp_reduce.0")
            self.mlp_reduce_1 = PreparedAllReduce(s.mlp_partial[h:], s.mlp_output[h:], runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".mlp_reduce.1")
            self.post = None if defer_post else PreparedMHCPost(s.mlp_output[:h], ffn_hc_scratch.residual,
                ffn_hc_scratch.post, ffn_hc_scratch.comb, s.residual[:h], session=session, name=name + ".post")
            ffn_plans = tuple(p for p in (
                getattr(self.ffn_hc, "plan", None),
                self.router_projection.plan,
                self.router.plan,
                self.experts.plan,
                getattr(self.experts, "scale_plan", None),
                self.shared.up_projection.plan,
                getattr(self.shared.up_projection, "quantizer_plan", None),
                self.shared.activation_plan,
                self.shared.down_projection.plan,
                getattr(self.shared.down_projection, "quantizer_plan", None),
            ) if p is not None)
            if self.pipelined_attn is not None:
                attn_parallel = list(self.attention.plans())
                if getattr(self.attention_hc, "plan", None) is not None:
                    attn_parallel.append(self.attention_hc.plan)
                self.cr_att_0 = ConcurrentAllReduce(self.att_reduce_0, workspace=workspace, session=session,
                    parallel_plans=attn_parallel)
            else:
                self.cr_att_0 = None
            self.cr_att_1 = ConcurrentAllReduce(self.att_reduce_1, workspace=workspace, session=session,
                parallel_plans=ffn_plans)
            self.cr_mlp_0 = ConcurrentAllReduce(self.mlp_reduce_0, workspace=workspace, session=session,
                parallel_plans=ffn_plans)
            self.cr_mlp_1 = ConcurrentAllReduce(self.mlp_reduce_1, workspace=workspace, session=session,
                parallel_plans=(self.post.plan,) if self.post is not None else ffn_plans)
        else:
            self.att_reduce_0 = self.att_reduce_1 = self.mlp_reduce_0 = self.mlp_reduce_1 = None
            self.cr_att_0 = self.cr_att_1 = self.cr_mlp_0 = self.cr_mlp_1 = None
            self.attention_reduce = PreparedAllReduce(s.attention.output, s.attention_output, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".attention_reduce")
            self.router_projection = UnquantizedProjection(s.ffn_hc.normalized, weights.experts.router, s.router_logits,
                session=session, name=name + ".router")
            self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
                score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
                routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
                image_correction_bias=weights.experts.vision_correction if self.image_mask is not None else None,
                image_mask=self.image_mask)
            self.experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.experts, s.router.ids,
                s.router.weights, s.routed, session=session, name=name + ".experts", workspace=workspace,
                router_plan=self.router.plan)
            if self.split_mhc:
                self.ffn_hc = SplitLaggedMHC(
                    (s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
                    _mhc_weights(weights, "ffn", "ffn_norm"),
                    session=session, workspace=workspace, experts=self.experts, **options)
            else:
                self.ffn_hc = PreparedMHC((s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
                    _mhc_weights(weights, "ffn", "ffn_norm"), s.ffn_hc, session=session, name=name + ".ffn_hc", workspace=workspace,
                    pre_mix=s.attn_mix, pre_out=s.pre_mix, **options)
            self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up,
                s.shared.intermediate, s.shared.output, session=session, name=name + ".shared", workspace=workspace,
                swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")
            self.concurrent_shared = None
            if mode in ("decode", "verify", "draft") and s.ffn_hc.normalized.shape[0] <= 128 and workspace.overlap:
                self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                    parallel_plans=(self.router_projection.plan, self.router.plan, self.experts.plan),
                    session=session, workspace=workspace,
                    defer_launch=s.ffn_hc.normalized.shape[0] > 4)
            self.mlp_reduce = PreparedAllReduce(s.mlp_partial, s.mlp_output, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".mlp_reduce", has_addend=True)
            self.post = None if defer_post else PreparedMHCPost(s.mlp_output, s.ffn_hc.residual,
                s.ffn_hc.post, s.ffn_hc.comb, s.residual, session=session, name=name + ".post")

        if self.attention_hc is not None:
            self.attention_hc.functional()
        if self.pipelined_ffn:
            if self.cr_att_0 is not None:
                self.cr_att_0.functional(session=session)
            else:
                self.att_reduce_0.functional(session=session)
            self.cr_att_1.functional(session=session)
            self.cr_mlp_0.functional(session=session)
            self.cr_mlp_1.functional(session=session)
        else:
            self.attention_reduce.functional(session=session)
            self.mlp_reduce.functional(session=session)
        self.ffn_hc.functional()
        self.router_projection.functional(session=session)
        self.router.functional(session=session)
        self.experts.functional(session=session)
        if self.concurrent_shared is None:
            self.shared.functional(session=session)
        else:
            self.concurrent_shared.functional(session=session)
        if self.post is not None:
            self.post.functional()
        self.has_image_mask = self.image_mask is not None
        register_finalizer(session, self.release_examples)

    def release_examples(self):
        self.scratch = self.metadata = self.image_mask = None
        if self.concurrent_shared is not None:
            self.shared = None

    def __call__(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None,
                 image_mask=None, engram_projected=None, engram_gathered=None, engram_overlap=None,
                 engram_project=None):
        incoming, next_mix, records, selection, _ = self._run(incoming, pre_mix, metadata, indices,
            selection, boundary=boundary, window_cache=window_cache, image_mask=image_mask,
            engram_projected=engram_projected, engram_gathered=engram_gathered, engram_overlap=engram_overlap,
            engram_project=engram_project)
        return incoming, next_mix, records, selection

    def _run(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None,
             image_mask=None, engram_projected=None, engram_gathered=None, engram_overlap=None,
             engram_project=None):
        if self.engram is not None:
            if engram_gathered is not None:
                incoming = (self.engram.mix(incoming[0], self.engram.project_gathered(engram_gathered), metadata),)
            elif engram_projected is not None:
                incoming = (self.engram.mix(incoming[0], engram_projected, metadata),)
            else:
                incoming = (self.engram(incoming[0], metadata),)
        h = getattr(self, "half_tokens", 0)
        pipelined_attn = getattr(self, "pipelined_attn", None)
        pipelined_attn_hc = getattr(self, "pipelined_attn_hc", False)
        pipelined_ffn = getattr(self, "pipelined_ffn", False)
        if pipelined_attn == "full":
            if pipelined_attn_hc:
                mlp_a, prev_res_a, prev_post_a, prev_comb_a, p_mlp_b, prev_res_b, prev_post_b, prev_comb_b = incoming
                pre_mix_a, pre_mix_b = pre_mix
                attn_mix_a = torch.empty_like(pre_mix_a)
                res_a, post_a, comb_a, norm_a = self.attention_hc(
                    mlp_a, prev_res_a, prev_post_a, prev_comb_a, pre_mix=pre_mix_a, pre_out=attn_mix_a)
                wc = self.attention.init_window_cache(metadata, norm_a.device) if window_cache is None else window_cache
                out_a = self.attention.run_half(norm_a, metadata, selection, indices=indices, window_cache=wc, half_idx=0)
                mlp_b = self.cr_att_0.finish(p_mlp_b, out_a)
                p_att_a = self.cr_att_0.start(out_a, dependencies=(mlp_b,))
                attn_mix_b = torch.empty_like(pre_mix_b)
                res_b, post_b, comb_b, norm_b = self.attention_hc(
                    mlp_b, prev_res_b, prev_post_b, prev_comb_b, pre_mix=pre_mix_b, pre_out=attn_mix_b)
                out_b = self.attention.run_half(norm_b, metadata, selection, indices=indices, window_cache=wc, half_idx=1)
            else:
                src = incoming[0]
                attn_mix_a = torch.empty_like(pre_mix[:h])
                res_a, post_a, comb_a, norm_a = self.attention_hc(src[:h], pre_mix=pre_mix[:h], pre_out=attn_mix_a)
                wc = self.attention.init_window_cache(metadata, norm_a.device) if window_cache is None else window_cache
                out_a = self.attention.run_half(norm_a, metadata, selection, indices=indices, window_cache=wc, half_idx=0)
                p_att_a = self.cr_att_0.start(out_a, dependencies=(out_a,))
                attn_mix_b = torch.empty_like(pre_mix[h:])
                res_b, post_b, comb_b, norm_b = self.attention_hc(src[h:], pre_mix=pre_mix[h:], pre_out=attn_mix_b)
                out_b = self.attention.run_half(norm_b, metadata, selection, indices=indices, window_cache=wc, half_idx=1)
            records = (wc, None)
            if self.attention.mode == "prefill":
                self.attention.commit(metadata.lengths, records, metadata)
            att_a = self.cr_att_0.finish(p_att_a, out_b)
            attention_normalized = norm_a
            observations = {}
        elif pipelined_attn == "mla_wo":
            if pipelined_attn_hc:
                mlp_a, prev_res_a, prev_post_a, prev_comb_a, p_mlp_b, prev_res_b, prev_post_b, prev_comb_b = incoming
                pre_mix_a, pre_mix_b = pre_mix
                attn_mix_a = torch.empty_like(pre_mix_a)
                res_a, post_a, comb_a, norm_a = self.attention_hc(
                    mlp_a, prev_res_a, prev_post_a, prev_comb_a, pre_mix=pre_mix_a, pre_out=attn_mix_a)
                mlp_b = self.cr_att_0.finish(p_mlp_b, norm_a)
                attn_mix_b = torch.empty_like(pre_mix_b)
                res_b, post_b, comb_b, norm_b = self.attention_hc(
                    mlp_b, prev_res_b, prev_post_b, prev_comb_b, pre_mix=pre_mix_b, pre_out=attn_mix_b)
                normalized = torch.cat((norm_a, norm_b), dim=0)
            else:
                attn_mix = torch.empty_like(pre_mix)
                residual, post, comb, normalized = self.attention_hc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
                res_a, res_b = residual[:h], residual[h:]
                post_a, post_b = post[:h], post[h:]
                comb_a, comb_b = comb[:h], comb[h:]
                attn_mix_a, attn_mix_b = attn_mix[:h], attn_mix[h:]
            attention_normalized = normalized
            pending_engram = None
            if engram_overlap is not None:
                prepare_engram, cr_engram, bounds, engram_state, use_gather = engram_overlap
                if prepare_engram is not None:
                    if use_gather:
                        engram_state["partial"] = prepare_engram.lookup_local(normalized, metadata)
                    else:
                        engram_state["partial"] = prepare_engram.project(normalized, metadata).view(-1, 5120)
                begin, end = bounds
                pending_engram = cr_engram.start(
                    engram_state["partial"][begin:end].clone(), dependencies=(normalized,))
            prefix_state = self.attention.run_prefix(normalized, metadata, selection,
                indices=indices, window_cache=window_cache)
            _, wc, selection, _, _, kv_records = prefix_state
            out_a = self.attention.run_mla_wo_half(prefix_state, metadata, indices=indices, half_idx=0)
            att0_deps = ()
            if pending_engram is not None:
                engram_chunk = cr_engram.finish(pending_engram, out_a)
                engram_state["chunks"].append(engram_chunk)
                att0_deps = (engram_chunk,)
            p_att_a = self.cr_att_0.start(out_a, dependencies=att0_deps)
            out_b = self.attention.run_mla_wo_half(prefix_state, metadata, indices=indices, half_idx=1)
            records = (wc, kv_records)
            if self.attention.mode == "prefill":
                self.attention.commit(metadata.lengths, records, metadata)
            att_a = self.cr_att_0.finish(p_att_a, out_b)
            observations = {}
        else:
            if pipelined_attn_hc:
                mlp_a, prev_res_a, prev_post_a, prev_comb_a, p_mlp_b, prev_res_b, prev_post_b, prev_comb_b = incoming
                pre_mix_a, pre_mix_b = pre_mix
                attn_mix_a = torch.empty_like(pre_mix_a)
                res_a, post_a, comb_a, norm_a = self.attention_hc(
                    mlp_a, prev_res_a, prev_post_a, prev_comb_a, pre_mix=pre_mix_a, pre_out=attn_mix_a)
                mlp_b = self.cr_att_1.finish(p_mlp_b, norm_a)
                attn_mix_b = torch.empty_like(pre_mix_b)
                res_b, post_b, comb_b, norm_b = self.attention_hc(
                    mlp_b, prev_res_b, prev_post_b, prev_comb_b, pre_mix=pre_mix_b, pre_out=attn_mix_b)
                normalized = torch.cat((norm_a, norm_b), dim=0)
                residual = post = comb = attn_mix = None
            elif boundary is None:
                if getattr(self, "split_mhc", False):
                    residual, normalized, _, _ = self.attention_hc.post_y_norm(*incoming, pre_mix=pre_mix)
                    post = comb = attn_mix = None
                else:
                    attn_mix = torch.empty_like(pre_mix)
                    residual, post, comb, normalized = self.attention_hc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
            else:
                (residual, post, comb, normalized), attn_mix = boundary
            attention_normalized = normalized
            pending_engram = None
            if engram_overlap is not None:
                prepare_engram, cr_engram, bounds, engram_state, use_gather = engram_overlap
                if prepare_engram is not None:
                    if use_gather:
                        engram_state["partial"] = prepare_engram.lookup_local(normalized, metadata)
                    else:
                        engram_state["partial"] = prepare_engram.project(normalized, metadata).view(-1, 5120)
                begin, end = bounds
                pending_engram = cr_engram.start(
                    engram_state["partial"][begin:end].clone(), dependencies=(normalized,))
            attended, records, selection, observations = self.attention._run(normalized, metadata, selection,
                indices=indices, window_cache=window_cache)
            att_reduce_deps = ()
            if pending_engram is not None:
                engram_chunk = cr_engram.finish(pending_engram, attended)
                engram_state["chunks"].append(engram_chunk)
                att_reduce_deps = (engram_chunk,)
        if self.has_image_mask:
            if image_mask is None:
                image_mask = metadata.token_ids >= 129280
        else:
            image_mask = None
        if pipelined_ffn:
            if self.pipelined_attn is None:
                if not self.pipelined_attn_hc:
                    res_a, res_b = residual[:h], residual[h:]
                    post_a, post_b = post[:h], post[h:]
                    comb_a, comb_b = comb[:h], comb[h:]
                    attn_mix_a, attn_mix_b = attn_mix[:h], attn_mix[h:]
                att_a = self.att_reduce_0(attended[:h], dependencies=att_reduce_deps)
                out_b = attended[h:]
            p_att_b = self.cr_att_1.start(out_b, dependencies=(att_a,))
            next_mix_a = torch.empty_like(attn_mix_a)
            res_a, post_a, comb_a, norm_a = self.ffn_hc(
                att_a, res_a, post_a, comb_a, pre_mix=attn_mix_a, pre_out=next_mix_a)
            mask_a = None if image_mask is None else image_mask[:h]
            logits_a = self.router_projection(norm_a)
            route_a = self.router(logits_a, image_mask=mask_a)
            routed_a = self.experts(norm_a, route_a.ids, route_a.weights, router_logits=logits_a,
                selected_logits=route_a.selected_logits, image_mask=mask_a)
            shared_a = self.shared(norm_a)
            partial_a = routed_a + shared_a
            att_b = self.cr_att_1.finish(p_att_b, partial_a)

            p_mlp_a = self.cr_mlp_0.start(partial_a, dependencies=(att_b,))
            next_mix_b = torch.empty_like(attn_mix_b)
            res_b, post_b, comb_b, norm_b = self.ffn_hc(
                att_b, res_b, post_b, comb_b, pre_mix=attn_mix_b, pre_out=next_mix_b)
            mask_b = None if image_mask is None else image_mask[h:]
            logits_b = self.router_projection(norm_b)
            route_b = self.router(logits_b, image_mask=mask_b)
            routed_b = self.experts(norm_b, route_b.ids, route_b.weights, router_logits=logits_b,
                selected_logits=route_b.selected_logits, image_mask=mask_b)
            shared_b = self.shared(norm_b)
            partial_b = routed_b + shared_b
            mlp_a = self.cr_mlp_0.finish(p_mlp_a, partial_b)

            p_mlp_b = self.cr_mlp_1.start(partial_b, dependencies=(mlp_a,))
            if self.defer_mlp_b:
                incoming = (mlp_a, res_a, post_a, comb_a, p_mlp_b, res_b, post_b, comb_b)
                next_mix = (next_mix_a, next_mix_b)
            elif self.post is not None:
                out_a = self.post(mlp_a, res_a, post_a, comb_a)
                finish_dep = out_a
                if engram_project is not None:
                    consumer_engram, engram_state = engram_project
                    projected = consumer_engram.project_gathered(engram_state["chunks"])
                    engram_state["projected"] = projected
                    finish_dep = projected
                next_mix = torch.cat((next_mix_a, next_mix_b), dim=0)
                mlp_b = self.cr_mlp_1.finish(p_mlp_b, finish_dep)
                out_b = self.post(mlp_b, res_b, post_b, comb_b)
                incoming = (torch.cat((out_a, out_b), dim=0),)
            else:
                res_full = torch.cat((res_a, res_b), dim=0)
                post_full = torch.cat((post_a, post_b), dim=0)
                comb_full = torch.cat((comb_a, comb_b), dim=0)
                next_mix = torch.cat((next_mix_a, next_mix_b), dim=0)
                mlp_b = self.cr_mlp_1.finish(p_mlp_b, res_full)
                mlp_full = torch.cat((mlp_a, mlp_b), dim=0)
                incoming = (mlp_full, res_full, post_full, comb_full)
            observations = {"attention." + key: value for key, value in observations.items()}
            observations.update({"attention_hc.normalized": attention_normalized,
                                 "router.ids": torch.cat((route_a.ids, route_b.ids), dim=0),
                                 "output": incoming[0]})
            return incoming, next_mix, records, selection, observations

        if getattr(self, "split_mhc", False):
            pending_attn_hc = self.attention_hc.start_sinkhorn(residual, attended)
            att_deps = (*att_reduce_deps, pending_attn_hc.pre_out)
            attention_output = self.attention_reduce(attended, dependencies=att_deps)
            post, comb, attn_mix = self.attention_hc.finish_sinkhorn(pending_attn_hc, attention_output)
            residual, normalized, prequant_v, prequant_s = self.ffn_hc.post_y_norm(
                attention_output, residual, post, comb, pre_mix=attn_mix)
            pending = None if self.concurrent_shared is None else self.concurrent_shared.start(normalized)
            router_logits = self.router_projection(normalized)
            route = self.router(router_logits, image_mask=image_mask)
            routed = self.experts(normalized, route.ids, route.weights, router_logits=router_logits,
                                  selected_logits=route.selected_logits, image_mask=image_mask,
                                  prequant_a_values=prequant_v, prequant_a_scales=prequant_s)
            shared = (self.shared(normalized) if self.concurrent_shared is None
                      else self.concurrent_shared.finish(pending, routed))
            pending_ffn_hc = self.ffn_hc.start_sinkhorn(residual, shared, routed)
            output = (self.mlp_reduce(routed, addend=shared, dependencies=(pending_ffn_hc.pre_out,))
                      if getattr(self.mlp_reduce, "fused_addend", False)
                      else self.mlp_reduce(routed + shared, dependencies=(pending_ffn_hc.pre_out,)))
            post, comb, next_mix = self.ffn_hc.finish_sinkhorn(pending_ffn_hc, output)
            incoming = ((self.post(output, residual, post, comb),) if self.post is not None
                        else (output, residual, post, comb))
            observations = {"attention." + key: value for key, value in observations.items()}
            observations.update({"attention_hc.normalized": attention_normalized,
                                 "router.ids": route.ids,
                                 "output": incoming[0]})
            return incoming, next_mix, records, selection, observations

        attention_output = (self.attention_reduce(attended, dependencies=att_reduce_deps)
                            if att_reduce_deps else self.attention_reduce(attended))
        next_mix = torch.empty_like(attn_mix)
        residual, post, comb, normalized = self.ffn_hc(attention_output, residual, post, comb,
            pre_mix=attn_mix, pre_out=next_mix)
        pending = None if self.concurrent_shared is None else self.concurrent_shared.start(normalized)
        router_logits = self.router_projection(normalized)
        route = self.router(router_logits, image_mask=image_mask)
        routed = self.experts(normalized, route.ids, route.weights, router_logits=router_logits,
                              selected_logits=route.selected_logits, image_mask=image_mask)
        shared = (self.shared(normalized) if self.concurrent_shared is None
                  else self.concurrent_shared.finish(pending, routed))
        output = (self.mlp_reduce(routed, addend=shared)
                  if getattr(self.mlp_reduce, "fused_addend", False) else self.mlp_reduce(routed + shared))
        incoming = ((self.post(output, residual, post, comb),) if self.post is not None
                    else (output, residual, post, comb))
        observations = {"attention." + key: value for key, value in observations.items()}
        observations.update({"attention_hc.normalized": attention_normalized, "router.ids": route.ids,
                             "output": incoming[0]})
        return incoming, next_mix, records, selection, observations


class PreparedDecoder:
    def __init__(self, weights, tables, scratch, metadata, pool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace,
                 vision=None, mla_compute_mode="auto", scoring=None, vocabulary_parallel=False, examples=None,
                 suffix_runtime=None, encoder_only=False):
        if scratch.local_logits.dtype != torch.float32 or scratch.logits.dtype != torch.float32:
            raise ValueError("DeepSeek target vocabulary logits require FP32 storage")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.encoder_only = bool(encoder_only)
        self.vision = vision if mode == "prefill" else None
        self.table_requests = PreparedTableRequests(tables, metadata, pool, session=session, name=name + ".table_requests")
        decoder_start = weights.config.decoder_start
        if self.encoder_only:
            if mode != "prefill" or decoder_start is None or scoring is not None:
                raise ValueError("encoder-only prefill requires a non-scored prefill with a decoder boundary")
            if (any(layer.engram is not None for layer in weights.layers[decoder_start:])
                    or any(source.layer > decoder_start for source in weights.config.sources)):
                raise ValueError("encoder-only prefill requires Engram and compressed KV sources before its boundary")
        self.packing = None
        if (not self.encoder_only and mode == "prefill" and metadata.positions.numel() > 128
                and decoder_start is not None and scoring is None):
            if (any(layer.engram is not None for layer in weights.layers[decoder_start:])
                    or any(source.layer > decoder_start for source in weights.config.sources)):
                raise ValueError("decoder suffix replay requires Engram and compressed KV sources before its boundary")
            self.packing = PreparedDecoderSuffix(metadata)
        self.output_metadata = metadata if self.packing is None else self.packing.metadata
        self.output_scratch = scratch if self.packing is None else Scratch.allocate(
            128, scratch.hidden.shape[0], weights.shard.world_size, scratch.embedding.device)
        if suffix_runtime is not None and not _runtime_accepts(suffix_runtime, self.output_scratch.attention.output):
            suffix_runtime = None
        self.aux_layers = () if (self.encoder_only or weights.draft is None) else weights.config.draft.target_layers
        if self.packing is not None and any(layer <= decoder_start for layer in self.aux_layers):
            raise ValueError("bounded decoder feedback requires draft target layers after its global-KV boundary")
        self.auxiliary_shape = (0, 0) if self.encoder_only else (self.output_metadata.token_ids.numel(), len(self.aux_layers) * 5120)
        self.auxiliary = None if self.encoder_only else torch.zeros(self.auxiliary_shape, dtype=torch.bfloat16, device=metadata.positions.device)
        self.safe_tokens = torch.zeros_like(metadata.token_ids) if self.vision is not None else metadata.token_ids
        self.embedding = PreparedEmbedding(weights.embedding, self.safe_tokens, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.window_commit = None
        if mode == "verify":
            self.window_commit = PreparedRingCommit(pool.reserved_backing,
                [pool.layout.region(f"layer{index}.window").offset for index in range(len(weights.layers))],
                metadata.state_rows, metadata.starts,
                token_capacity=metadata.max_query_tokens,
                window_tokens=128, record_bytes=528)
        ratios = weights.config.text["compress_ratios"][:len(weights.layers)]
        prefix_ratios = ratios if (self.packing is None and not self.encoder_only) else ratios[:decoder_start + 1]
        self.window_metadata = AttentionIndices(metadata, mode=mode,
            tokens=metadata.positions.numel(),
            page_size=128 if mode == "decode" else 128 + metadata.max_query_tokens, width=128,
            ratios=set(prefix_ratios) - {0})
        self.suffix_window_metadata = None
        if self.packing is not None:
            self.suffix_window_metadata = AttentionIndices(self.output_metadata, mode=mode,
                tokens=self.output_metadata.positions.numel(),
                page_size=128 + self.output_metadata.max_query_tokens, width=128,
                replay_start=self.packing.replay_start, ratios=set(ratios[decoder_start:]) - {0})

        def _layer_runtime(index):
            if self.packing is not None and index >= decoder_start and suffix_runtime is not None:
                return suffix_runtime
            return runtime

        def _layer_tokens(index):
            if self.packing is not None and index >= decoder_start:
                return self.output_scratch.attention.output.shape[0]
            return scratch.attention.output.shape[0]

        can_overlap = isinstance(workspace, ScratchArena) and workspace.overlap and mode == "prefill"
        layer_pipelined = [
            can_overlap and _is_dma_runtime(_layer_runtime(i)) and _layer_tokens(i) >= 1024 and _layer_tokens(i) % 2 == 0
            for i in range(len(weights.layers))
        ]
        self.layers, carry, prev_defer_mlp_b = [], False, False
        self.engram_overlaps = {}
        self.engram_consumers = {}
        self.engram_projects = {}
        active_layers = weights.layers[:decoder_start] if self.encoder_only else weights.layers
        for index, layer in enumerate(active_layers):
            boundary = replay_start = None
            layer_scratch, layer_metadata = scratch, metadata
            layer_runtime = _layer_runtime(index)
            if self.packing is not None and index >= decoder_start:
                layer_scratch, layer_metadata = self.output_scratch, self.output_metadata
                replay_start = self.packing.replay_start
                if index == decoder_start:
                    boundary = PreparedDecoderBoundary(index, layer, scratch, layer_scratch, self.packing,
                        pool, self.rotary, carry=carry, session=session, name=f"{name}.layer{index}", workspace=workspace,
                        window_metadata=self.suffix_window_metadata,
                        source_indices=self.window_metadata)
                    layer = replace(layer, attention={key: value for key, value in layer.attention.items()
                                                      if key not in ("compressor", "index_k", "index_norm")})
            defer = (index + 1 < len(weights.layers) and weights.layers[index + 1].engram is None
                     and index + 1 not in self.aux_layers)
            pipelined_ffn = layer_pipelined[index]
            pipelined_attn = None
            if pipelined_ffn and boundary is None:
                if "compressor" not in layer.attention and "index_q" not in layer.attention:
                    pipelined_attn = "full"
                else:
                    pipelined_attn = "mla_wo"
            pipelined_attn_hc = prev_defer_mlp_b
            next_has_boundary = (self.packing is not None or self.encoder_only) and index + 1 == decoder_start
            defer_mlp_b = (pipelined_ffn and defer and index + 1 < len(active_layers)
                           and layer_pipelined[index + 1] and not next_has_boundary)
            overlap_engram = (can_overlap and _is_dma_runtime(layer_runtime)
                              and layer.engram is not None and weights.shard.world_size > 1
                              and index >= 8 and ((self.packing is None and not self.encoder_only) or index < decoder_start))
            prepared_layer = PreparedLayer(index, layer, tables, layer_scratch, layer_metadata, pool, self.rotary,
                shard=weights.shard, mode=mode, runtime=layer_runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
                carry=carry, defer_post=defer, boundary=boundary, replay_start=replay_start,
                window_staging=None if self.window_commit is None else self.window_commit.staging[index],
                mla_compute_mode=mla_compute_mode,
                window_metadata=(self.window_metadata if layer_metadata is metadata else self.suffix_window_metadata),
                examples=examples, pipelined_ffn=pipelined_ffn, pipelined_attn_hc=pipelined_attn_hc,
                defer_mlp_b=defer_mlp_b, defer_engram_reduce=overlap_engram, pipelined_attn=pipelined_attn)
            if pipelined_attn_hc:
                prev_plan = self.layers[index - 1].cr_mlp_1.reduce.plan
                if getattr(prepared_layer.attention_hc, "plan", None) is not None:
                    workspace.register_parallel(prev_plan, prepared_layer.attention_hc.plan)
                if pipelined_attn == "full":
                    for plan in prepared_layer.attention.plans():
                        workspace.register_parallel(prev_plan, plan)
            if overlap_engram and prepared_layer.engram is not None:
                engram = prepared_layer.engram
                if engram.use_all_gather:
                    parts = engram.gather_parts
                    bounds = engram.gather_bounds
                    # Place single-chunk DMA AllGather on a preceding mla_wo layer (layer 8 for layer 14)
                    # where prefill_collective is idle during QKV + compressor + indexer + MLA(A).
                    mla_wo_targets = [t for t in range(2, index)
                                      if self.layers[t].pipelined_attn == "mla_wo" and t not in self.engram_overlaps]
                    non_pipe_targets = [t for t in range(2, index)
                                        if self.layers[t].pipelined_attn is None and t not in self.engram_overlaps]
                    candidates = mla_wo_targets if len(mla_wo_targets) >= len(parts) else non_pipe_targets
                    if len(candidates) >= len(parts):
                        chosen = candidates[-len(parts):]
                        for ordinal, (target, part, bound) in enumerate(zip(chosen, parts, bounds, strict=True)):
                            cr_part = ConcurrentAllGather(part, workspace=workspace, session=session,
                                parallel_plans=self.layers[target].attention.plans()).functional(session=session)
                            self.engram_overlaps[target] = (index, engram if ordinal == 0 else None, cr_part, bound, True)
                        self.engram_consumers[index] = (len(parts), True)
                        prev_layer = self.layers[index - 1]
                        if prev_layer.pipelined_ffn and prev_layer.post is not None:
                            self.engram_projects[index - 1] = (index, engram)
                            workspace.register_parallel(prev_layer.cr_mlp_1.reduce.plan, engram.projection.plan)
                            if getattr(engram.projection, "quantizer_plan", None) is not None:
                                workspace.register_parallel(prev_layer.cr_mlp_1.reduce.plan,
                                                            engram.projection.quantizer_plan)
                    else:
                        for part in parts:
                            part.functional(session=session)
                elif engram.reduce is not None:
                    parts = engram.reduce.parts
                    bounds = engram.reduce.bounds
                    start_layer = index - 1 - len(parts)
                    if (start_layer >= 2
                            and all(self.layers[target].pipelined_attn is None and target not in self.engram_overlaps
                                    for target in range(start_layer, index - 1))):
                        for ordinal, (part, bound) in enumerate(zip(parts, bounds, strict=True)):
                            target = start_layer + ordinal
                            cr_part = ConcurrentAllReduce(part, workspace=workspace, session=session,
                                parallel_plans=self.layers[target].attention.plans()).functional(session=session)
                            self.engram_overlaps[target] = (index, engram if ordinal == 0 else None, cr_part, bound, False)
                        engram.reduce.functional(session=session)
                        self.engram_consumers[index] = (len(parts), False)
                    else:
                        engram.reduce.functional(session=session)
            self.layers.append(prepared_layer)
            carry = defer
            prev_defer_mlp_b = defer_mlp_b
        self.encoder_boundary = None
        if self.encoder_only:
            self.encoder_boundary = PreparedDecoderBoundary(
                decoder_start, weights.layers[decoder_start], scratch, None, None,
                pool, self.rotary, carry=carry, session=session,
                name=f"{name}.layer{decoder_start}", workspace=workspace,
                window_metadata=None, source_indices=self.window_metadata,
            )
            self.scoring = self.head = self.head_hidden = self.local_logits = None
            self.compact_head = False
        else:
            self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
                vocabulary=129280, session=session, coordinator=coordinator, name=name + ".prompt",
                logits_dtype=torch.float32)
            self._prepare_head(vocabulary_parallel=vocabulary_parallel, vocabulary_runtime=vocabulary_runtime,
                coordinator=coordinator, session=session, name=name, workspace=workspace)
        self.window_metadata.functional()
        if self.suffix_window_metadata is not None:
            self.suffix_window_metadata.functional()
            self.packing.functional()
        self.staging_shape = None if self.window_commit is None else tuple(self.window_commit.staging.shape)
        if self.window_commit is not None:
            self.window_commit.functional()
        register_finalizer(session, self.release_outputs)

    def _prepare_head(self, *, vocabulary_parallel, vocabulary_runtime, coordinator, session, name, workspace=None):
        scratch, weights = self.scratch, self.weights
        self.compact_head = scratch.hidden.shape[0] > self.output_metadata.token_ids.numel()
        head_rows = self.output_metadata.token_ids.numel() if self.compact_head else scratch.hidden.shape[0]
        self.head_hidden = scratch.hidden[:head_rows]
        head_output = scratch.logits
        if vocabulary_parallel and self.compact_head:
            # Declare expanded local logits independently of the physical projection rows.
            head_output = scratch.logits.view(-1)[:scratch.local_logits.numel()].view_as(scratch.local_logits)
        self.head = PreparedVocabularyProjection(self.head_hidden, weights.head, scratch.local_logits[:head_rows],
            head_output, runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            row_indices=self.output_metadata.sample_indices if self.compact_head else None, gather=not vocabulary_parallel,
            workspace=workspace)

        self.local_logits = self.head.output if vocabulary_parallel else scratch.local_logits
        self.head.functional(session=session, dynamic_indices=True)

    def release_outputs(self):
        self.scratch = self.metadata = self.output_scratch = self.output_metadata = None
        self.auxiliary = self.safe_tokens = self.head_hidden = self.local_logits = None

    def __call__(self, metadata):
        source_metadata = metadata
        raw_history = self.table_requests(metadata)
        tokens = (metadata.token_ids.clamp(0, self.weights.embedding.shape[0] - 1)
                  if self.vision is not None else metadata.token_ids)
        embedding = self.embedding(token_ids=tokens, num_rows=metadata.live_counts[:1])
        if self.vision is not None:
            self.vision.inject(metadata, embedding)
        residual = embedding[:, None, :].expand(-1, 4, -1).contiguous()
        pre_mix = torch.zeros((embedding.shape[0], 4), dtype=torch.float32, device=embedding.device)
        pre_mix[:, 0].fill_(1)
        incoming = (residual,)
        indices = self.window_metadata(metadata)
        selection = (torch.full((embedding.shape[0], 512), -1, dtype=torch.int32, device=embedding.device), None, None)
        staging = (None if self.staging_shape is None else torch.empty(
            self.staging_shape, dtype=torch.uint8, device=embedding.device))
        auxiliary = None if self.encoder_only else torch.empty(self.auxiliary_shape, dtype=torch.bfloat16, device=embedding.device)
        suffix, records = None, []
        has_image_mask = any(layer.has_image_mask for layer in self.layers)
        image_mask = metadata.token_ids >= 129280 if has_image_mask else None
        engram_states = {consumer: {"partial": None, "chunks": []} for consumer in self.engram_consumers}
        for index, layer in enumerate(self.layers):
            boundary = None
            if layer.boundary is not None:
                suffix, state, attn_mix = layer.boundary(incoming, pre_mix, metadata, indices,
                                                        layer.attention.selectors[0])
                metadata = suffix.metadata
                indices = self.suffix_window_metadata(metadata, replay_start=suffix.replay_start)
                boundary = state, attn_mix
                selection = (torch.full((metadata.positions.numel(), 512), -1,
                                       dtype=torch.int32, device=embedding.device), None, None)
                image_mask = metadata.token_ids >= 129280 if has_image_mask else None
            if index in self.aux_layers:
                feedback = incoming[0].float().mean(1)
                if suffix is not None:
                    feedback = torch.where(suffix.live[:, None], feedback, 0)
                column = self.aux_layers.index(index)
                auxiliary[:, column * 5120:(column + 1) * 5120].copy_(feedback)
            engram_projected = None
            engram_gathered = None
            if index in engram_states:
                if "projected" in engram_states[index]:
                    engram_projected = engram_states[index]["projected"]
                elif self.engram_consumers[index][1]:
                    engram_gathered = engram_states[index]["chunks"]
                else:
                    engram_projected = torch.cat(engram_states[index]["chunks"], dim=0).view(incoming[0].shape[0], 25600)
            engram_overlap = None
            if index in self.engram_overlaps:
                consumer, prepare_engram, cr_part, bound, use_gather = self.engram_overlaps[index]
                engram_overlap = (prepare_engram, cr_part, bound, engram_states[consumer], use_gather)
            engram_project = None
            if index in self.engram_projects:
                consumer, consumer_engram = self.engram_projects[index]
                engram_project = (consumer_engram, engram_states[consumer])
            incoming, pre_mix, record, selection = layer(incoming, pre_mix, metadata, indices, selection,
                boundary=boundary, window_cache=None if staging is None else staging[index],
                image_mask=image_mask, engram_projected=engram_projected, engram_gathered=engram_gathered,
                engram_overlap=engram_overlap, engram_project=engram_project)
            records.append(record)
        if self.encoder_only:
            self.encoder_boundary(incoming, pre_mix, metadata, indices)
            self.table_requests.commit(source_metadata.lengths, raw_history, source_metadata)
            return None, None, None, (staging, tuple(records), raw_history)
        residual, = incoming
        samples = metadata.sample_indices
        head_residual = residual if self.compact_head else residual[samples]
        head_mix = pre_mix if self.compact_head else pre_mix[samples]
        collapsed = (head_residual.float() * head_mix[:, :, None]).sum(1).bfloat16()
        if self.scoring is not None:
            collapsed = (residual.float() * pre_mix[:, :, None]).sum(1).bfloat16()
            normalized = torch.empty_like(collapsed)
            normalize(collapsed, self.weights.norm, normalized, self.weights.config.text["rms_norm_eps"])
            self.scoring(normalized)
            hidden = normalized[samples]
        else:
            hidden = torch.empty_like(collapsed)
            normalize(collapsed, self.weights.norm, hidden, self.weights.config.text["rms_norm_eps"])
        logits = self.head(hidden, row_indices=samples if self.compact_head else None)
        if self.compact_head:
            hidden = hidden[samples]
        if self.mode != "verify":
            self.table_requests.commit(source_metadata.lengths, raw_history, source_metadata)
        return hidden, logits, auxiliary, (staging, tuple(records), raw_history)

    def commit(self, counts, records, metadata):
        if self.window_commit is None:
            raise ValueError("accepted decoder commits require verification staging")
        staging, layers, raw_history = records
        self.window_commit(counts, staging=staging, rows=metadata.state_rows, starts=metadata.starts)
        for layer, record in zip(self.layers, layers, strict=True):
            layer.attention.commit_pending(counts, record, metadata)
        self.table_requests.commit(counts, raw_history, metadata)
