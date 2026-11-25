# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek text stack with lagged hyperconnections."""

from dataclasses import dataclass, fields, is_dataclass, replace

import torch
from b12x.preparation import PreparedCall

from ignition.components.collectives import (
    ConcurrentAllGather, ConcurrentAllReduce, ConcurrentGatherShards, ConcurrentReduceScatter,
    PreparedAllReduce, PreparedChunkedGatherShards, PreparedGatherShards, PreparedReduceScatter,
    row_shard_surfaces,
)
from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.expert_overlap import ConcurrentSharedExpert
from hyperflare_recipes.components.ring_commit import PreparedRingCommit
from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.mhc import MHCWeights, MHCScratch, PreparedMHC, PreparedMHCPost
from hyperflare_recipes.components.mhc_overlap import SplitLaggedMHC
from hyperflare_recipes.components.mlp import MLPScratch, PreparedMLP
from hyperflare_recipes.components.moe import RouteRecord, RouterScratch, PreparedRouter, PreparedRoutedExperts
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from hyperflare_recipes.components.parallel import TensorParallelShard
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import declare, register_finalizer

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


def _mxfp8_scale_mma(scale_rows_2d: torch.Tensor, hidden: int) -> torch.Tensor:
    """Swizzle contiguous [rows, hidden // 32] UE8M0 scale rows into b12x MMA tile layout."""
    m_tiles, k_tiles = scale_rows_2d.shape[0] // 128, hidden // 128
    return (
        scale_rows_2d.view(m_tiles, 4, 32, k_tiles, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(1, m_tiles, k_tiles, 32, 4, 4)
    )


class PreparedDecoderBoundary:
    """Produce complete global KV, then optionally gather the decoder's bounded replay rows."""

    def __init__(self, index, weights, source, target, packing, pool, rotary, *, carry,
                 session, name, workspace, window_metadata, source_indices,
                 row_shard_dma=False, row_shard_roce=False, shard=None, runtime=None,
                 coordinator=None, examples=None):
        self.packing = packing
        self.row_shard_dma = bool(row_shard_dma)
        self.row_shard_roce = bool(row_shard_roce)
        if self.row_shard_dma and self.row_shard_roce:
            raise ValueError("decoder boundary cannot combine DMA and RoCE row-sharding")
        text = pool.config.text
        global_kv_source = source.attention_hc.normalized
        global_kv_row_shard = False
        self.norm_gather = self.res_gather = self.state_gather = None
        if self.row_shard_dma:
            tokens = source.attention.output.shape[0]
            h = tokens // 2
            s = h // shard.world_size
            self.shard_tokens = s
            incoming = ((source.mlp_output[:s], source.ffn_hc.residual[:s], source.ffn_hc.post[:s], source.ffn_hc.comb[:s])
                        if carry else (source.residual[:s],))
            self.mhc = PreparedMHC(incoming, _mhc_weights(weights, "attn", "attn_norm"), source.attention_hc.rows(s),
                session=session, name=name + ".attn_hc", workspace=workspace,
                pre_mix=source.pre_mix[:s], pre_out=source.attn_mix[:s], rms_eps=text["rms_norm_eps"],
                norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"], sinkhorn_iters=text["hc_sinkhorn_iters"]).functional()
            self.norm_gather_0 = PreparedGatherShards(
                source.attention_hc.normalized[:s], source.attention_hc.normalized[:h],
                runtime=runtime, session=session, coordinator=coordinator,
                name=name + ".boundary_norm_gather.0").functional(session=session)
            self.norm_gather_1 = PreparedGatherShards(
                source.attention_hc.normalized[:s], source.attention_hc.normalized[h:],
                runtime=runtime, session=session, coordinator=coordinator,
                name=name + ".boundary_norm_gather.1").functional(session=session)
            if packing is not None:
                self.res_gather_0 = PreparedChunkedGatherShards(
                    source.attention_hc.residual[:s], source.attention_hc.residual[:h],
                    runtime=runtime, session=session, coordinator=coordinator,
                    name=name + ".boundary_res_gather.0", examples=examples).functional(session=session)
                self.res_gather_1 = PreparedChunkedGatherShards(
                    source.attention_hc.residual[:s], source.attention_hc.residual[h:],
                    runtime=runtime, session=session, coordinator=coordinator,
                    name=name + ".boundary_res_gather.1", examples=examples).functional(session=session)
                dev = source.residual.device
                state_in = declaration_tensor(examples, "row_shard.boundary_state_in", (s, 48), torch.bfloat16, dev)
                state_out = declaration_tensor(examples, "row_shard.boundary_state_out", (h, 48), torch.bfloat16, dev)
                self.state_gather_0 = PreparedGatherShards(
                    state_in, state_out, runtime=runtime, session=session, coordinator=coordinator,
                    name=name + ".boundary_state_gather.0").functional(session=session)
                self.state_gather_1 = PreparedGatherShards(
                    state_in, state_out, runtime=runtime, session=session, coordinator=coordinator,
                    name=name + ".boundary_state_gather.1").functional(session=session)
            else:
                self.res_gather_0 = self.res_gather_1 = self.state_gather_0 = self.state_gather_1 = None
        elif self.row_shard_roce:
            tokens = source.attention.output.shape[0]
            s = tokens // shard.world_size
            self.shard_tokens = s
            incoming = ((source.mlp_output[:s], source.ffn_hc.residual[:s], source.ffn_hc.post[:s], source.ffn_hc.comb[:s])
                        if carry else (source.residual[:s],))
            self.mhc = PreparedMHC(incoming, _mhc_weights(weights, "attn", "attn_norm"), source.attention_hc.rows(s),
                session=session, name=name + ".attn_hc", workspace=workspace,
                pre_mix=source.pre_mix[:s], pre_out=source.attn_mix[:s], rms_eps=text["rms_norm_eps"],
                norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"], sinkhorn_iters=text["hc_sinkhorn_iters"]).functional()
            self.norm_gather_0 = self.norm_gather_1 = None
            self.res_gather_0 = self.res_gather_1 = self.state_gather_0 = self.state_gather_1 = None
            if packing is None:
                # Encoder-only boundary only needs the compact compressed KV projection.
                global_kv_source = source.attention_hc.normalized[:s]
                global_kv_row_shard = True
            else:
                self.norm_gather = PreparedGatherShards(
                    source.attention_hc.normalized[:s], source.attention_hc.normalized,
                    runtime=runtime, session=session, coordinator=coordinator,
                    name=name + ".boundary_norm_gather").functional(session=session)
                self.res_gather = PreparedChunkedGatherShards(
                    source.attention_hc.residual[:s], source.attention_hc.residual,
                    runtime=runtime, session=session, coordinator=coordinator,
                    name=name + ".boundary_res_gather", examples=examples).functional(session=session)
                dev = source.residual.device
                state_in = declaration_tensor(examples, "row_shard_roce.boundary_state_in", (s, 48), torch.bfloat16, dev)
                state_out = declaration_tensor(examples, "row_shard_roce.boundary_state_out", (tokens, 48), torch.bfloat16, dev)
                self.state_gather = PreparedGatherShards(
                    state_in, state_out, runtime=runtime, session=session, coordinator=coordinator,
                    name=name + ".boundary_state_gather").functional(session=session)
        else:
            incoming = ((source.mlp_output, source.ffn_hc.residual, source.ffn_hc.post, source.ffn_hc.comb)
                        if carry else (source.residual,))
            self.mhc = PreparedMHC(incoming, _mhc_weights(weights, "attn", "attn_norm"), source.attention_hc,
                session=session, name=name + ".attn_hc", workspace=workspace,
                pre_mix=source.pre_mix, pre_out=source.attn_mix, rms_eps=text["rms_norm_eps"],
                norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"], sinkhorn_iters=text["hc_sinkhorn_iters"]).functional()
            self.norm_gather_0 = self.norm_gather_1 = None
            self.res_gather_0 = self.res_gather_1 = self.state_gather_0 = self.state_gather_1 = None
        self.global_kv = PreparedGlobalKV(index, weights.attention, global_kv_source, source.attention,
            source_indices.metadata, pool, rotary, session=session, name=name + ".global_kv",
            indices=source_indices, row_shard=global_kv_row_shard, runtime=runtime, coordinator=coordinator)
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
        if self.row_shard_dma:
            pre_mix_a, pre_mix_b = pre_mix
            if len(incoming) == 8:
                inc_a = incoming[:4]
                inc_b = incoming[4:]
            else:
                inc_a = (incoming[0],)
                inc_b = (incoming[1],)
            attn_mix_a = torch.empty_like(pre_mix_a)
            r_a, p_a, c_a, norm_a_sh = self.mhc(*inc_a, pre_mix=pre_mix_a, pre_out=attn_mix_a)
            attn_mix_b = torch.empty_like(pre_mix_b)
            r_b, p_b, c_b, norm_b_sh = self.mhc(*inc_b, pre_mix=pre_mix_b, pre_out=attn_mix_b)
            norm_a = self.norm_gather_0(norm_a_sh)
            norm_b = self.norm_gather_1(norm_b_sh, dependencies=(norm_a,))
            normalized = torch.cat((norm_a, norm_b), dim=0)
            self.global_kv(normalized, self.indexer if indexer is None else indexer, metadata,
                indices.compressed(self.global_kv.ratio),
                angles=indices.angles[(True, self.global_kv.ratio)])
            if suffix is None:
                return None, None, None
            res_a = self.res_gather_0(r_a, dependencies=(normalized,))
            res_b = self.res_gather_1(r_b, dependencies=(res_a,))
            residual = torch.cat((res_a, res_b), dim=0)
            s = self.shard_tokens
            st_a = torch.cat((p_a, c_a.view(s, 16), attn_mix_a), dim=1).view(torch.bfloat16)
            st_b = torch.cat((p_b, c_b.view(s, 16), attn_mix_b), dim=1).view(torch.bfloat16)
            g_a = self.state_gather_0(st_a).view(torch.float32)
            g_b = self.state_gather_1(st_b, dependencies=(g_a,)).view(torch.float32)
            g = torch.cat((g_a, g_b), dim=0)
            post = g[:, :4].contiguous()
            comb = g[:, 4:20].contiguous().view(g.shape[0], 4, 4)
            attn_mix = g[:, 20:24].contiguous()
            return suffix, tuple(suffix.gather(value) for value in (residual, post, comb, normalized)), suffix.gather(attn_mix)
        if self.row_shard_roce:
            attn_mix_sh = torch.empty_like(pre_mix)
            r_sh, p_sh, c_sh, norm_sh = self.mhc(*incoming, pre_mix=pre_mix, pre_out=attn_mix_sh)
            if suffix is None:
                self.global_kv(norm_sh, self.indexer if indexer is None else indexer, metadata,
                    indices.compressed(self.global_kv.ratio),
                    angles=indices.angles[(True, self.global_kv.ratio)])
                return None, None, None
            normalized = self.norm_gather(norm_sh)
            self.global_kv(normalized, self.indexer if indexer is None else indexer, metadata,
                indices.compressed(self.global_kv.ratio),
                angles=indices.angles[(True, self.global_kv.ratio)])
            residual = self.res_gather(r_sh)
            s = self.shard_tokens
            st_sh = torch.cat((p_sh, c_sh.view(s, 16), attn_mix_sh), dim=1).view(torch.bfloat16)
            g = self.state_gather(st_sh).view(torch.float32)
            post = g[:, :4].contiguous()
            comb = g[:, 4:20].contiguous().view(g.shape[0], 4, 4)
            attn_mix = g[:, 20:24].contiguous()
            return suffix, tuple(suffix.gather(value) for value in (residual, post, comb, normalized)), suffix.gather(attn_mix)
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
                 defer_mlp_b=False, defer_engram_reduce=False, pipelined_attn=None,
                 row_shard_dma=False, row_shard_roce=False, row_shard_attn=False,
                 shard_rows=None, shard_attention_scratch=None, window_target=None):
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
        self.row_shard_dma = bool(row_shard_dma)
        self.row_shard_roce = bool(row_shard_roce)
        self.row_shard_attn = bool(row_shard_attn) and self.row_shard_roce
        if self.row_shard_dma and self.row_shard_roce:
            raise ValueError("a layer cannot combine DMA and RoCE row-sharded prefill")
        tokens = s.attention.output.shape[0]
        h = tokens // 2
        self.half_tokens = h
        if self.row_shard_dma:
            self.shard_tokens = h // shard.world_size
        elif self.row_shard_roce:
            self.shard_tokens = tokens // shard.world_size
        else:
            self.shard_tokens = 0
        self.shard_rows = self.shard_offset = None
        if self.row_shard_dma:
            if shard_rows is None:
                dev = s.residual.device
                s0 = shard.rank * self.shard_tokens
                self.shard_rows_a = persistent_copy(torch.arange(s0, s0 + self.shard_tokens, dtype=torch.int64, device=dev))
                self.shard_rows_b = persistent_copy(torch.arange(h + s0, h + s0 + self.shard_tokens, dtype=torch.int64, device=dev))
                self.shard_offset_a = persistent_copy(torch.tensor([s0], dtype=torch.int32, device=dev))
                self.shard_offset_b = persistent_copy(torch.tensor([h + s0], dtype=torch.int32, device=dev))
            else:
                self.shard_rows_a, self.shard_rows_b, self.shard_offset_a, self.shard_offset_b = shard_rows
        elif self.row_shard_roce:
            self.shard_rows_a = self.shard_rows_b = self.shard_offset_a = self.shard_offset_b = None
            if shard_rows is None:
                dev = s.residual.device
                s0 = shard.rank * self.shard_tokens
                self.shard_rows = persistent_copy(torch.arange(s0, s0 + self.shard_tokens, dtype=torch.int64, device=dev))
                self.shard_offset = persistent_copy(torch.tensor([s0], dtype=torch.int32, device=dev))
            else:
                self.shard_rows, self.shard_offset = shard_rows
        else:
            self.shard_rows_a = self.shard_rows_b = self.shard_offset_a = self.shard_offset_b = None
        self.attn_quantizer_plan = None
        self.attn_local_shapes = None
        self.attn_vals_bf16_cols = 0
        self.ffn_quantizer_plan = None
        self.ffn_local_shapes = None
        self.ffn_meta_scale_cols = 0
        self.attn_gather = None
        self.attn_gather_0 = self.attn_gather_1 = None
        self.cr_attn_gather_0 = self.cr_attn_gather_1 = None
        self.ffn_vals_gather = self.ffn_meta_gather = None
        self.ffn_vals_0 = self.ffn_vals_1 = self.ffn_meta_0 = self.ffn_meta_1 = None
        self.cr_ffn_vals_0 = self.cr_ffn_vals_1 = self.cr_ffn_meta_0 = self.cr_ffn_meta_1 = None
        if self.pipelined_attn_hc and (not self.pipelined_ffn or not carry or boundary is not None):
            raise ValueError("pipelined attention mHC requires a pipelined carry layer without a boundary")
        if self.pipelined_attn is not None and (not self.pipelined_ffn or boundary is not None):
            raise ValueError("pipelined attention requires a pipelined FFN layer without a boundary")
        if self.defer_mlp_b and (not self.pipelined_ffn or not defer_post):
            raise ValueError("deferred mlp_B overlap requires pipelined FFN with deferred post")
        if self.row_shard_dma and (not self.pipelined_ffn or boundary is not None
                or self.pipelined_attn not in ("full", "mla_wo") or h % shard.world_size != 0):
            raise ValueError("DMA row-sharded prefill requires pipelined attention/FFN with divisible half-chunks")
        if self.row_shard_roce and (self.pipelined_ffn or boundary is not None
                or tokens % shard.world_size != 0):
            raise ValueError("RoCE row-sharded prefill requires serial execution without a boundary and divisible chunks")
        self.engram = None
        if weights.engram is not None:
            if self.row_shard_dma:
                engram_row_shard = (shard.rank, h, self.shard_tokens)
            elif self.row_shard_roce:
                engram_row_shard = (shard.rank, self.shard_tokens)
            else:
                engram_row_shard = None
            self.engram = PreparedEngram(index, weights.engram, tables, s.residual, s.engram_residual,
                metadata, pool,
                scratch=s.engram, session=session, name=name + ".engram", workspace=workspace,
                runtime=runtime, coordinator=coordinator, examples=examples,
                defer_reduce_functional=defer_engram_reduce, row_shard=engram_row_shard)
            if (self.row_shard_dma or self.row_shard_roce) and self.engram.row_shard is None:
                raise ValueError("row-sharded prefill requires full-projection Engram gather")
        source = s.engram_residual if self.engram is not None else s.residual
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        from b12x.norm.mhc._split_lagged import supports_split_lagged

        self.split_mhc = (
            mode in ("decode", "verify", "draft")
            and not self.pipelined_ffn
            and not self.pipelined_attn_hc
            and self.pipelined_attn is None
            and not self.row_shard_roce
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
        if self.row_shard_dma or self.row_shard_roce:
            st = self.shard_tokens
            attn_incoming = ((s.mlp_output[:st], s.ffn_hc.residual[:st], s.ffn_hc.post[:st], s.ffn_hc.comb[:st])
                             if carry else (source[:st],))
            self.attention_hc = PreparedMHC(
                attn_incoming, _mhc_weights(weights, "attn", "attn_norm"), s.attention_hc.rows(st),
                session=session, name=name + ".attn_hc", workspace=workspace,
                pre_mix=s.pre_mix[:st], pre_out=s.attn_mix[:st], **options)
        elif self.pipelined_attn_hc or self.pipelined_attn == "full":
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
        attn_qkv_encoded = None
        attn_gather_in = None
        attn_gather_out = None
        if self.row_shard_dma and self.pipelined_attn == "full":
            st = self.shard_tokens
            dev = s.attention_hc.normalized.device
            hidden = s.attention_hc.normalized.shape[1]
            if dev.type == "cuda" and st % 128 == 0 and h % 128 == 0 and h > 28 and hidden % 128 == 0:
                from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
                from b12x.quantization import mxfp8
                from hyperflare_recipes.components import sm103_projection

                qkv_weight = weights.attention["qkv_a"]
                if (not sm103_projection.is_sm103(dev)
                        and isinstance(qkv_weight, MXFP8LinearWeight)
                        and qkv_weight.padded_in_features == hidden):
                    local_shapes = mxfp8.dense_rows_storage_shapes(st, hidden)
                    full_shapes = mxfp8.dense_rows_storage_shapes(h, hidden)
                    local_vals = declaration_tensor(examples, "row_shard.attn_local_vals", local_shapes[0], torch.float8_e4m3fn, dev)
                    local_srows = declaration_tensor(examples, "row_shard.attn_local_srows", local_shapes[1], torch.uint8, dev)
                    local_smma = declaration_tensor(examples, "row_shard.attn_local_smma", local_shapes[2], torch.uint8, dev)
                    full_vals = declaration_tensor(examples, "row_shard.attn_full_vals", full_shapes[0], torch.float8_e4m3fn, dev)
                    full_srows = declaration_tensor(examples, "row_shard.attn_full_srows", full_shapes[1], torch.uint8, dev)
                    full_smma = declaration_tensor(examples, "row_shard.attn_full_smma", full_shapes[2], torch.uint8, dev)
                    vals_bf16_cols = hidden // 2
                    scale_bf16_cols = local_srows.view(st, -1).view(torch.bfloat16).shape[1]
                    local_packed = declaration_tensor(examples, "row_shard.attn_local_packed",
                                                      (st, vals_bf16_cols + scale_bf16_cols), torch.bfloat16, dev)
                    full_packed = declaration_tensor(examples, "row_shard.attn_full_packed",
                                                     (h, vals_bf16_cols + scale_bf16_cols), torch.bfloat16, dev)
                    if PreparedGatherShards.fits(runtime, local_packed, full_packed):
                        local_encoded = mxfp8.dense_rows_from_bases(local_vals, local_srows, local_smma)
                        full_encoded = mxfp8.dense_rows_from_bases(full_vals, full_srows, full_smma)
                        self.attn_local_shapes = local_shapes
                        self.attn_vals_bf16_cols = vals_bf16_cols
                        # Match PreparedAttention's qkv_a min_amax=1e-4 so shard quantization is bit-exact.
                        src_attn_norm = s.attention_hc.normalized[:st]
                        self.attn_quantizer_plan = mxfp8.plan(mxfp8.query_from_call(
                            src_attn_norm, local_encoded.values, local_encoded.scale_rows,
                            local_encoded.scale_mma, min_amax=1e-4))
                        declare(session, (self.attn_quantizer_plan.request(
                            name=name + ".attn_quant",
                            prepare_call=lambda state, src=src_attn_norm, enc=local_encoded: PreparedCall(
                                run=lambda: state.run(src, enc.values, enc.scale_rows, enc.scale_mma),
                                owners=(src, enc))),))
                        attn_qkv_encoded = full_encoded
                        attn_gather_in = local_packed
                        attn_gather_out = full_packed
        if self.row_shard_attn:
            st = self.shard_tokens
            attn_source = s.attention_hc.normalized[:st]
            attn_scratch = (shard_attention_scratch if shard_attention_scratch is not None
                            else AttentionScratch.allocate(tokens, shard.world_size, s.residual.device,
                                                           shard_rows=st, examples=examples))
            attn_window_target = (window_target if window_target is not None
                                  else declaration_tensor(examples, "row_shard_roce.window_full",
                                                          (tokens, 512), torch.bfloat16, s.residual.device))
            self.attention = PreparedAttention(index, weights.attention, attn_source, attn_scratch,
                metadata, pool, rotary, mode=mode, world=shard.world_size,
                session=session, name=name + ".attention", workspace=workspace, replay_start=replay_start,
                window_staging=window_staging, mla_compute_mode=mla_compute_mode,
                window_metadata=window_metadata, examples=examples,
                shard_rows=self.shard_rows, window_target=attn_window_target,
                runtime=runtime, coordinator=coordinator)
        else:
            self.attention = PreparedAttention(index, weights.attention, s.attention_hc.normalized, s.attention,
                metadata, pool, rotary, mode=mode, world=shard.world_size,
                session=session, name=name + ".attention", workspace=workspace, replay_start=replay_start,
                window_staging=window_staging, mla_compute_mode=mla_compute_mode,
                window_metadata=window_metadata, examples=examples, pipelined_mode=self.pipelined_attn,
                qkv_encoded=attn_qkv_encoded)
        if self.row_shard_dma:
            st = self.shard_tokens
            self.attention_reduce = None
            self.attn_gather_0 = PreparedGatherShards(
                s.attention_hc.normalized[:st] if attn_gather_in is None else attn_gather_in,
                s.attention_hc.normalized[:h] if attn_gather_out is None else attn_gather_out,
                runtime=runtime, coordinator=coordinator, session=session, name=name + ".attn_gather.0")
            self.attn_gather_1 = PreparedGatherShards(
                s.attention_hc.normalized[:st] if attn_gather_in is None else attn_gather_in,
                s.attention_hc.normalized[h:] if attn_gather_out is None else attn_gather_out,
                runtime=runtime, coordinator=coordinator, session=session, name=name + ".attn_gather.1")
            self.att_reduce_0 = PreparedReduceScatter(
                s.attention.output[:h], s.attention_output[:st],
                runtime=runtime, coordinator=coordinator, session=session, name=name + ".attention_rs.0")
            self.att_reduce_1 = PreparedReduceScatter(
                s.attention.output[h:], s.attention_output[:st],
                runtime=runtime, coordinator=coordinator, session=session, name=name + ".attention_rs.1")
            ffn_hc_scratch = s.ffn_hc.rows(st)
            self.ffn_hc = PreparedMHC((s.attention_output[:st], s.attention_hc.residual[:st],
                s.attention_hc.post[:st], s.attention_hc.comb[:st]),
                _mhc_weights(weights, "ffn", "ffn_norm"), ffn_hc_scratch, session=session,
                name=name + ".ffn_hc", workspace=workspace,
                pre_mix=s.attn_mix[:st], pre_out=s.pre_mix[:st], **options)
            self.router_projection = UnquantizedProjection(ffn_hc_scratch.normalized, weights.experts.router,
                s.router_logits[:st], session=session, name=name + ".router")
            router_scratch = s.router.rows(st)
            self.router = PreparedRouter(s.router_logits[:st], router_scratch, session=session, name=name + ".topk",
                score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
                routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
                image_correction_bias=weights.experts.vision_correction if self.image_mask is not None else None,
                image_mask=None if self.image_mask is None else self.image_mask[:st],
                num_valid_tokens=metadata.live_counts[:1])
            dev = ffn_hc_scratch.normalized.device
            top_k = router_scratch.ids.shape[1]
            route_bf16_cols = 4 * top_k
            use_mxfp8_gather = False
            ffn_experts_source = s.ffn_hc.normalized[:h]
            ffn_experts_scales = None
            ffn_shared_encoded = None
            if (dev.type == "cuda" and st % 128 == 0 and h % 128 == 0
                    and ffn_hc_scratch.normalized.shape[1] % 128 == 0):
                from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
                from hyperflare_recipes.components import sm103_projection

                experts_plan = getattr(weights.experts.experts, "plan", None)
                quant_modes = getattr(getattr(experts_plan, "_impl", None), "quant_modes", ())
                if (not sm103_projection.is_sm103(dev)
                        and isinstance(weights.experts.shared.gate_up, MXFP8LinearWeight)
                        and "w4a8_mx" in quant_modes):
                    from b12x.quantization import mxfp8

                    local_shapes = mxfp8.dense_rows_storage_shapes(st, ffn_hc_scratch.normalized.shape[1])
                    full_shapes = mxfp8.dense_rows_storage_shapes(h, s.ffn_hc.normalized.shape[1])
                    local_vals = declaration_tensor(examples, "row_shard.ffn_local_vals", local_shapes[0], torch.float8_e4m3fn, dev)
                    local_srows = declaration_tensor(examples, "row_shard.ffn_local_srows", local_shapes[1], torch.uint8, dev)
                    local_smma = declaration_tensor(examples, "row_shard.ffn_local_smma", local_shapes[2], torch.uint8, dev)
                    full_vals = declaration_tensor(examples, "row_shard.ffn_full_vals", full_shapes[0], torch.float8_e4m3fn, dev)
                    full_srows = declaration_tensor(examples, "row_shard.ffn_full_srows", full_shapes[1], torch.uint8, dev)
                    full_smma = declaration_tensor(examples, "row_shard.ffn_full_smma", full_shapes[2], torch.uint8, dev)
                    scale_bf16_cols = local_srows.view(st, -1).view(torch.bfloat16).shape[1]
                    local_meta = declaration_tensor(examples, "row_shard.ffn_local_meta",
                                                    (st, scale_bf16_cols + route_bf16_cols), torch.bfloat16, dev)
                    full_meta = declaration_tensor(examples, "row_shard.ffn_full_meta",
                                                   (h, scale_bf16_cols + route_bf16_cols), torch.bfloat16, dev)
                    if (PreparedGatherShards.fits(runtime, local_vals.view(torch.bfloat16), full_vals.view(torch.bfloat16))
                            and PreparedGatherShards.fits(runtime, local_meta, full_meta)):
                        use_mxfp8_gather = True
                        local_encoded = mxfp8.dense_rows_from_bases(local_vals, local_srows, local_smma)
                        full_encoded = mxfp8.dense_rows_from_bases(full_vals, full_srows, full_smma)
                        self.ffn_local_shapes = local_shapes
                        self.ffn_meta_scale_cols = scale_bf16_cols
                        self.ffn_quantizer_plan = mxfp8.plan(mxfp8.query_from_call(
                            ffn_hc_scratch.normalized, local_encoded.values, local_encoded.scale_rows,
                            local_encoded.scale_mma, min_amax=0.0))
                        src_norm = ffn_hc_scratch.normalized
                        declare(session, (self.ffn_quantizer_plan.request(
                            name=name + ".ffn_quant",
                            prepare_call=lambda state, src=src_norm, enc=local_encoded: PreparedCall(
                                run=lambda: state.run(src, enc.values, enc.scale_rows, enc.scale_mma),
                                owners=(src, enc))),))
                        self.ffn_vals_0 = PreparedGatherShards(
                            local_vals.view(torch.bfloat16), full_vals.view(torch.bfloat16),
                            runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_vals.0")
                        self.ffn_vals_1 = PreparedGatherShards(
                            local_vals.view(torch.bfloat16), full_vals.view(torch.bfloat16),
                            runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_vals.1")
                        self.ffn_meta_0 = PreparedGatherShards(
                            local_meta, full_meta,
                            runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_meta.0")
                        self.ffn_meta_1 = PreparedGatherShards(
                            local_meta, full_meta,
                            runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_meta.1")
                        ffn_experts_source = full_vals
                        ffn_experts_scales = full_srows
                        ffn_shared_encoded = full_encoded
            if not use_mxfp8_gather:
                local_meta = declaration_tensor(examples, "row_shard.ffn_route_local_meta",
                                                (st, route_bf16_cols), torch.bfloat16, dev)
                full_meta = declaration_tensor(examples, "row_shard.ffn_route_full_meta",
                                               (h, route_bf16_cols), torch.bfloat16, dev)
                self.ffn_vals_0 = PreparedGatherShards(
                    ffn_hc_scratch.normalized, s.ffn_hc.normalized[:h],
                    runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_vals.0")
                self.ffn_vals_1 = PreparedGatherShards(
                    ffn_hc_scratch.normalized, s.ffn_hc.normalized[h:],
                    runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_vals.1")
                self.ffn_meta_0 = PreparedGatherShards(
                    local_meta, full_meta,
                    runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_meta.0")
                self.ffn_meta_1 = PreparedGatherShards(
                    local_meta, full_meta,
                    runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_meta.1")
            self.experts = PreparedRoutedExperts(ffn_experts_source, weights.experts.experts,
                s.router.ids[:h], s.router.weights[:h], s.routed[:h], session=session, name=name + ".experts",
                workspace=workspace, router_plan=self.router.plan, a_scales=ffn_experts_scales)
            shared_scratch = s.shared.rows(h)
            self.shared = PreparedMLP(s.ffn_hc.normalized[:h], weights.experts.shared, shared_scratch.gate_up,
                shared_scratch.intermediate, shared_scratch.output, session=session, name=name + ".shared",
                workspace=workspace, swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto",
                encoded=ffn_shared_encoded)
            self.concurrent_shared = None
            self.mlp_reduce = None
            self.mlp_reduce_0 = PreparedReduceScatter(s.mlp_partial[:h], s.mlp_output[:st], runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".mlp_rs.0")
            self.mlp_reduce_1 = PreparedReduceScatter(s.mlp_partial[h:], s.mlp_output[:st], runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".mlp_rs.1")
            self.post = None if defer_post else PreparedMHCPost(s.mlp_output[:st], ffn_hc_scratch.residual,
                ffn_hc_scratch.post, ffn_hc_scratch.comb, s.residual[:st], session=session, name=name + ".post")
            shard_prep_plans = tuple(p for p in (
                getattr(self.ffn_hc, "plan", None),
                self.router_projection.plan,
                self.router.plan,
                self.ffn_quantizer_plan,
            ) if p is not None)
            ffn_compute_plans = tuple(p for p in (
                self.experts.plan,
                getattr(self.experts, "scale_plan", None),
                self.shared.up_projection.plan,
                getattr(self.shared.up_projection, "quantizer_plan", None),
                self.shared.activation_plan,
                self.shared.down_projection.plan,
                getattr(self.shared.down_projection, "quantizer_plan", None),
            ) if p is not None)
            attn_parallel = list(self.attention.plans())
            if getattr(self.attention_hc, "plan", None) is not None:
                attn_parallel.append(self.attention_hc.plan)
            if self.attn_quantizer_plan is not None:
                attn_parallel.append(self.attn_quantizer_plan)
            self.cr_attn_gather_0 = ConcurrentGatherShards(self.attn_gather_0, workspace=workspace, session=session,
                parallel_plans=tuple(p for p in (self.attention_hc.plan, self.attn_quantizer_plan) if p is not None))
            if self.pipelined_attn == "full":
                self.cr_attn_gather_1 = ConcurrentGatherShards(self.attn_gather_1, workspace=workspace, session=session,
                    parallel_plans=attn_parallel)
            self.cr_att_0 = ConcurrentReduceScatter(self.att_reduce_0, workspace=workspace, session=session,
                parallel_plans=attn_parallel)
            self.cr_att_1 = ConcurrentReduceScatter(self.att_reduce_1, workspace=workspace, session=session,
                parallel_plans=shard_prep_plans)
            self.cr_ffn_vals_0 = ConcurrentGatherShards(self.ffn_vals_0, workspace=workspace, session=session,
                parallel_plans=shard_prep_plans)
            self.cr_ffn_meta_0 = ConcurrentGatherShards(self.ffn_meta_0, workspace=workspace, session=session,
                parallel_plans=shard_prep_plans)
            self.cr_ffn_vals_1 = ConcurrentGatherShards(self.ffn_vals_1, workspace=workspace, session=session,
                parallel_plans=ffn_compute_plans)
            self.cr_ffn_meta_1 = ConcurrentGatherShards(self.ffn_meta_1, workspace=workspace, session=session,
                parallel_plans=ffn_compute_plans)
            self.cr_mlp_0 = ConcurrentReduceScatter(self.mlp_reduce_0, workspace=workspace, session=session,
                parallel_plans=ffn_compute_plans)
            self.cr_mlp_1 = ConcurrentReduceScatter(self.mlp_reduce_1, workspace=workspace, session=session,
                parallel_plans=(self.post.plan,) if self.post is not None else ffn_compute_plans)
        elif self.row_shard_roce:
            st = self.shard_tokens
            self.att_reduce_0 = self.att_reduce_1 = self.mlp_reduce_0 = self.mlp_reduce_1 = None
            self.cr_att_0 = self.cr_att_1 = self.cr_mlp_0 = self.cr_mlp_1 = None
            if self.row_shard_attn:
                self.attn_gather = None
                self.attention_reduce = None
            else:
                self.attn_gather = PreparedGatherShards(
                    s.attention_hc.normalized[:st], s.attention_hc.normalized,
                    runtime=runtime, coordinator=coordinator, session=session, name=name + ".attn_gather")
                self.attention_reduce = PreparedReduceScatter(
                    s.attention.output, s.attention_output[:st],
                    runtime=runtime, coordinator=coordinator, session=session, name=name + ".attention_rs")
            ffn_hc_scratch = s.ffn_hc.rows(st)
            self.ffn_hc = PreparedMHC((s.attention_output[:st], s.attention_hc.residual[:st],
                s.attention_hc.post[:st], s.attention_hc.comb[:st]),
                _mhc_weights(weights, "ffn", "ffn_norm"), ffn_hc_scratch, session=session,
                name=name + ".ffn_hc", workspace=workspace,
                pre_mix=s.attn_mix[:st], pre_out=s.pre_mix[:st], **options)
            self.router_projection = UnquantizedProjection(ffn_hc_scratch.normalized, weights.experts.router,
                s.router_logits[:st], session=session, name=name + ".router")
            router_scratch = s.router.rows(st)
            self.router = PreparedRouter(s.router_logits[:st], router_scratch, session=session, name=name + ".topk",
                score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
                routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
                image_correction_bias=weights.experts.vision_correction if self.image_mask is not None else None,
                image_mask=None if self.image_mask is None else self.image_mask[:st],
                num_valid_tokens=metadata.live_counts[:1])
            dev = ffn_hc_scratch.normalized.device
            top_k = router_scratch.ids.shape[1]
            route_bf16_cols = 4 * top_k
            use_mxfp8_gather = False
            ffn_experts_source = s.ffn_hc.normalized
            ffn_experts_scales = None
            ffn_shared_encoded = None
            if (dev.type == "cuda" and st % 128 == 0 and tokens % 128 == 0
                    and ffn_hc_scratch.normalized.shape[1] % 128 == 0):
                from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
                from hyperflare_recipes.components import sm103_projection

                experts_plan = getattr(weights.experts.experts, "plan", None)
                quant_modes = getattr(getattr(experts_plan, "_impl", None), "quant_modes", ())
                if (not sm103_projection.is_sm103(dev)
                        and isinstance(weights.experts.shared.gate_up, MXFP8LinearWeight)
                        and "w4a8_mx" in quant_modes):
                    from b12x.quantization import mxfp8

                    local_shapes = mxfp8.dense_rows_storage_shapes(st, ffn_hc_scratch.normalized.shape[1])
                    full_shapes = mxfp8.dense_rows_storage_shapes(tokens, s.ffn_hc.normalized.shape[1])
                    local_vals = declaration_tensor(examples, "row_shard_roce.ffn_local_vals", local_shapes[0], torch.float8_e4m3fn, dev)
                    local_srows = declaration_tensor(examples, "row_shard_roce.ffn_local_srows", local_shapes[1], torch.uint8, dev)
                    local_smma = declaration_tensor(examples, "row_shard_roce.ffn_local_smma", local_shapes[2], torch.uint8, dev)
                    full_vals = declaration_tensor(examples, "row_shard_roce.ffn_full_vals", full_shapes[0], torch.float8_e4m3fn, dev)
                    full_srows = declaration_tensor(examples, "row_shard_roce.ffn_full_srows", full_shapes[1], torch.uint8, dev)
                    full_smma = declaration_tensor(examples, "row_shard_roce.ffn_full_smma", full_shapes[2], torch.uint8, dev)
                    scale_bf16_cols = local_srows.view(st, -1).view(torch.bfloat16).shape[1]
                    local_meta = declaration_tensor(examples, "row_shard_roce.ffn_local_meta",
                                                    (st, scale_bf16_cols + route_bf16_cols), torch.bfloat16, dev)
                    full_meta = declaration_tensor(examples, "row_shard_roce.ffn_full_meta",
                                                   (tokens, scale_bf16_cols + route_bf16_cols), torch.bfloat16, dev)
                    if (PreparedGatherShards.fits(runtime, local_vals.view(torch.bfloat16), full_vals.view(torch.bfloat16))
                            and PreparedGatherShards.fits(runtime, local_meta, full_meta)):
                        use_mxfp8_gather = True
                        local_encoded = mxfp8.dense_rows_from_bases(local_vals, local_srows, local_smma)
                        full_encoded = mxfp8.dense_rows_from_bases(full_vals, full_srows, full_smma)
                        self.ffn_local_shapes = local_shapes
                        self.ffn_meta_scale_cols = scale_bf16_cols
                        self.ffn_quantizer_plan = mxfp8.plan(mxfp8.query_from_call(
                            ffn_hc_scratch.normalized, local_encoded.values, local_encoded.scale_rows,
                            local_encoded.scale_mma, min_amax=0.0))
                        src_norm = ffn_hc_scratch.normalized
                        declare(session, (self.ffn_quantizer_plan.request(
                            name=name + ".ffn_quant",
                            prepare_call=lambda state, src=src_norm, enc=local_encoded: PreparedCall(
                                run=lambda: state.run(src, enc.values, enc.scale_rows, enc.scale_mma),
                                owners=(src, enc))),))
                        self.ffn_vals_gather = PreparedGatherShards(
                            local_vals.view(torch.bfloat16), full_vals.view(torch.bfloat16),
                            runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_vals")
                        self.ffn_meta_gather = PreparedGatherShards(
                            local_meta, full_meta,
                            runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_meta")
                        ffn_experts_source = full_vals
                        ffn_experts_scales = full_srows
                        ffn_shared_encoded = full_encoded
            if not use_mxfp8_gather:
                local_meta = declaration_tensor(examples, "row_shard_roce.ffn_route_local_meta",
                                                (st, route_bf16_cols), torch.bfloat16, dev)
                full_meta = declaration_tensor(examples, "row_shard_roce.ffn_route_full_meta",
                                               (tokens, route_bf16_cols), torch.bfloat16, dev)
                self.ffn_vals_gather = PreparedGatherShards(
                    ffn_hc_scratch.normalized, s.ffn_hc.normalized,
                    runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_vals")
                self.ffn_meta_gather = PreparedGatherShards(
                    local_meta, full_meta,
                    runtime=runtime, coordinator=coordinator, session=session, name=name + ".ffn_meta")
            self.experts = PreparedRoutedExperts(ffn_experts_source, weights.experts.experts,
                s.router.ids, s.router.weights, s.routed, session=session, name=name + ".experts",
                workspace=workspace, router_plan=self.router.plan, a_scales=ffn_experts_scales)
            self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up,
                s.shared.intermediate, s.shared.output, session=session, name=name + ".shared",
                workspace=workspace, swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto",
                encoded=ffn_shared_encoded)
            self.concurrent_shared = None
            self.mlp_reduce = PreparedReduceScatter(s.mlp_partial, s.mlp_output[:st], runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".mlp_rs")
            self.post = None if defer_post else PreparedMHCPost(s.mlp_output[:st], ffn_hc_scratch.residual,
                ffn_hc_scratch.post, ffn_hc_scratch.comb, s.residual[:st], session=session, name=name + ".post")
        elif self.pipelined_ffn:
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
        if self.row_shard_dma:
            self.cr_attn_gather_0.functional(session=session)
            if self.cr_attn_gather_1 is not None:
                self.cr_attn_gather_1.functional(session=session)
            else:
                self.attn_gather_1.functional(session=session)
            self.cr_att_0.functional(session=session)
            self.cr_att_1.functional(session=session)
            self.cr_ffn_vals_0.functional(session=session)
            self.cr_ffn_meta_0.functional(session=session)
            self.cr_ffn_vals_1.functional(session=session)
            self.cr_ffn_meta_1.functional(session=session)
            self.cr_mlp_0.functional(session=session)
            self.cr_mlp_1.functional(session=session)
        elif self.row_shard_roce:
            if self.attn_gather is not None:
                self.attn_gather.functional(session=session)
            if self.attention_reduce is not None:
                self.attention_reduce.functional(session=session)
            self.ffn_vals_gather.functional(session=session)
            self.ffn_meta_gather.functional(session=session)
            self.mlp_reduce.functional(session=session)
        elif self.pipelined_ffn:
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

    def _prepare_attn_shard(self, normalized):
        if self.attn_quantizer_plan is None:
            return normalized
        from b12x.quantization import mxfp8

        local_vals = torch.empty(self.attn_local_shapes[0], dtype=torch.float8_e4m3fn, device=normalized.device)
        local_srows = torch.empty(self.attn_local_shapes[1], dtype=torch.uint8, device=normalized.device)
        local_smma = torch.empty(self.attn_local_shapes[2], dtype=torch.uint8, device=normalized.device)
        mxfp8.quantize_dense_rows(normalized, local_vals, local_srows, local_smma, plan=self.attn_quantizer_plan)
        return torch.cat(
            (local_vals.view(torch.bfloat16), local_srows.view(normalized.shape[0], -1).view(torch.bfloat16)),
            dim=1,
        )

    def _unpack_attn_gathered(self, packed_full):
        if self.attn_quantizer_plan is None:
            return packed_full, None
        full_rows = packed_full.shape[0]
        full_vals = packed_full[:, :self.attn_vals_bf16_cols].contiguous().view(torch.float8_e4m3fn)
        hidden = full_vals.shape[1]
        full_srows_2d = packed_full[:, self.attn_vals_bf16_cols:].contiguous().view(torch.uint8)
        full_srows = full_srows_2d.view(1, full_rows, hidden // 32)
        full_smma = _mxfp8_scale_mma(full_srows_2d, hidden)
        return full_vals, (full_vals, full_srows, full_smma)

    def _prepare_ffn_shard(self, normalized, mask, num_valid_tokens):
        logits = self.router_projection(normalized)
        route = self.router(logits, image_mask=mask, num_valid_tokens=num_valid_tokens)
        packed_routes = RouteRecord.empty(route.ids.shape[0], route.ids.shape[1], route.ids.device)
        packed_routes.write(route.ids, route.weights)
        if self.ffn_quantizer_plan is not None:
            from b12x.quantization import mxfp8

            local_vals = torch.empty(self.ffn_local_shapes[0], dtype=torch.float8_e4m3fn, device=normalized.device)
            local_srows = torch.empty(self.ffn_local_shapes[1], dtype=torch.uint8, device=normalized.device)
            local_smma = torch.empty(self.ffn_local_shapes[2], dtype=torch.uint8, device=normalized.device)
            mxfp8.quantize_dense_rows(normalized, local_vals, local_srows, local_smma, plan=self.ffn_quantizer_plan)
            local_meta = torch.cat(
                (local_srows.view(normalized.shape[0], -1).view(torch.bfloat16), packed_routes.packed),
                dim=1,
            )
            return local_vals.view(torch.bfloat16), local_meta
        return normalized, packed_routes.packed

    def _run_ffn_gathered(self, vals_full_bf16, meta_full):
        full_rows = vals_full_bf16.shape[0]
        if self.ffn_quantizer_plan is not None:
            full_vals = vals_full_bf16.view(torch.float8_e4m3fn)
            hidden = full_vals.shape[1]
            full_srows_2d = meta_full[:, :self.ffn_meta_scale_cols].contiguous().view(torch.uint8)
            full_srows = full_srows_2d.view(1, full_rows, hidden // 32)
            full_smma = _mxfp8_scale_mma(full_srows_2d, hidden)
            ffn_encoded_bases = (full_vals, full_srows, full_smma)
            route_words = meta_full[:, self.ffn_meta_scale_cols:].contiguous().view(torch.int32)
            top_k = route_words.shape[1] // 2
            route_ids = torch.empty((full_rows, top_k), dtype=torch.int32, device=meta_full.device)
            route_weights = torch.empty((full_rows, top_k), dtype=torch.float32, device=meta_full.device)
            RouteRecord(route_words).read(route_ids, route_weights)
            routed = self.experts(full_vals, route_ids, route_weights,
                                  a_scales=full_srows, router_logits=None,
                                  selected_logits=None, image_mask=None)
            shared = self.shared(full_vals, encoded_bases=ffn_encoded_bases)
            return routed + shared, route_ids
        route_words = meta_full.contiguous().view(torch.int32)
        top_k = route_words.shape[1] // 2
        route_ids = torch.empty((full_rows, top_k), dtype=torch.int32, device=meta_full.device)
        route_weights = torch.empty((full_rows, top_k), dtype=torch.float32, device=meta_full.device)
        RouteRecord(route_words).read(route_ids, route_weights)
        routed = self.experts(vals_full_bf16, route_ids, route_weights,
                              router_logits=None, selected_logits=None, image_mask=None)
        shared = self.shared(vals_full_bf16)
        return routed + shared, route_ids

    def _run_row_shard_dma(self, incoming, pre_mix, metadata, indices, selection, *,
                           window_cache=None, image_mask=None, engram_projected=None,
                           engram_gathered=None, engram_overlap=None, engram_project=None):
        s_tok = self.shard_tokens
        if self.engram is not None:
            res_cat = torch.cat(incoming, dim=0)
            if engram_gathered is not None:
                mixed = self.engram.mix(res_cat, self.engram.project_gathered(engram_gathered), metadata)
            elif engram_projected is not None:
                mixed = self.engram.mix(res_cat, engram_projected, metadata)
            else:
                mixed = self.engram(res_cat, metadata)
            incoming = (mixed[:s_tok], mixed[s_tok:])
        pre_mix_a, pre_mix_b = pre_mix
        if self.pipelined_attn_hc:
            mlp_a, prev_res_a, prev_post_a, prev_comb_a, p_mlp_b, prev_res_b, prev_post_b, prev_comb_b = incoming
            attn_mix_a = torch.empty_like(pre_mix_a)
            res_a, post_a, comb_a, norm_a_sh = self.attention_hc(
                mlp_a, prev_res_a, prev_post_a, prev_comb_a, pre_mix=pre_mix_a, pre_out=attn_mix_a)
            attn_a_sh = self._prepare_attn_shard(norm_a_sh)
            mlp_b = self.cr_att_0.finish(p_mlp_b, attn_a_sh)
            p_ag_a = self.cr_attn_gather_0.start(attn_a_sh, dependencies=(mlp_b,))
            attn_mix_b = torch.empty_like(pre_mix_b)
            res_b, post_b, comb_b, norm_b_sh = self.attention_hc(
                mlp_b, prev_res_b, prev_post_b, prev_comb_b, pre_mix=pre_mix_b, pre_out=attn_mix_b)
            attn_b_sh = self._prepare_attn_shard(norm_b_sh)
        else:
            src_a, src_b = incoming
            attn_mix_a = torch.empty_like(pre_mix_a)
            res_a, post_a, comb_a, norm_a_sh = self.attention_hc(src_a, pre_mix=pre_mix_a, pre_out=attn_mix_a)
            attn_a_sh = self._prepare_attn_shard(norm_a_sh)
            p_ag_a = self.cr_attn_gather_0.start(attn_a_sh, dependencies=(attn_a_sh,))
            attn_mix_b = torch.empty_like(pre_mix_b)
            res_b, post_b, comb_b, norm_b_sh = self.attention_hc(src_b, pre_mix=pre_mix_b, pre_out=attn_mix_b)
            attn_b_sh = self._prepare_attn_shard(norm_b_sh)
        norm_a = self.cr_attn_gather_0.finish(p_ag_a, attn_b_sh)
        if self.pipelined_attn == "full":
            p_ag_b = self.cr_attn_gather_1.start(attn_b_sh, dependencies=(norm_a,))
            wc = self.attention.init_window_cache(metadata, norm_a.device) if window_cache is None else window_cache
            src_a_att, enc_a_att = self._unpack_attn_gathered(norm_a)
            out_a = self.attention.run_half(
                src_a_att, metadata, selection, indices=indices, window_cache=wc, half_idx=0,
                qkv_encoded_bases=enc_a_att)
            norm_b = self.cr_attn_gather_1.finish(p_ag_b, out_a)
            p_att_a = self.cr_att_0.start(out_a, dependencies=(norm_b,))
            src_b_att, enc_b_att = self._unpack_attn_gathered(norm_b)
            out_b = self.attention.run_half(
                src_b_att, metadata, selection, indices=indices, window_cache=wc, half_idx=1,
                qkv_encoded_bases=enc_b_att)
            records = (wc, None)
            if self.attention.mode == "prefill":
                self.attention.commit(metadata.lengths, records, metadata)
            att_a = self.cr_att_0.finish(p_att_a, out_b)
            attention_normalized = src_a_att
        else:
            norm_b = self.attn_gather_1(attn_b_sh, dependencies=(norm_a,))
            normalized = torch.cat((norm_a, norm_b), dim=0)
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

        if self.has_image_mask:
            if image_mask is None:
                image_mask = metadata.token_ids >= 129280
            mask_a = torch.index_select(image_mask, 0, self.shard_rows_a)
            mask_b = torch.index_select(image_mask, 0, self.shard_rows_b)
        else:
            mask_a = mask_b = None
        valid_a = (metadata.live_counts[:1] - self.shard_offset_a).clamp(0, s_tok).to(torch.int32)
        valid_b = (metadata.live_counts[:1] - self.shard_offset_b).clamp(0, s_tok).to(torch.int32)

        p_att_b = self.cr_att_1.start(out_b, dependencies=(att_a,))
        next_mix_a = torch.empty_like(attn_mix_a)
        res_a, post_a, comb_a, ffn_norm_a_sh = self.ffn_hc(
            att_a, res_a, post_a, comb_a, pre_mix=attn_mix_a, pre_out=next_mix_a)
        vals_a_sh, meta_a_sh = self._prepare_ffn_shard(ffn_norm_a_sh, mask_a, valid_a)
        att_b = self.cr_att_1.finish(p_att_b, meta_a_sh)

        p_vals_a = self.cr_ffn_vals_0.start(vals_a_sh, dependencies=(att_b,))
        p_meta_a = self.cr_ffn_meta_0.start(meta_a_sh, dependencies=(p_vals_a.output,))
        next_mix_b = torch.empty_like(attn_mix_b)
        res_b, post_b, comb_b, ffn_norm_b_sh = self.ffn_hc(
            att_b, res_b, post_b, comb_b, pre_mix=attn_mix_b, pre_out=next_mix_b)
        vals_b_sh, meta_b_sh = self._prepare_ffn_shard(ffn_norm_b_sh, mask_b, valid_b)
        vals_a_full = self.cr_ffn_vals_0.finish(p_vals_a, meta_b_sh)
        meta_a_full = self.cr_ffn_meta_0.finish(p_meta_a, vals_a_full)

        p_vals_b = self.cr_ffn_vals_1.start(vals_b_sh, dependencies=(meta_a_full,))
        p_meta_b = self.cr_ffn_meta_1.start(meta_b_sh, dependencies=(p_vals_b.output,))
        partial_a, route_ids_a = self._run_ffn_gathered(vals_a_full, meta_a_full)
        vals_b_full = self.cr_ffn_vals_1.finish(p_vals_b, partial_a)
        meta_b_full = self.cr_ffn_meta_1.finish(p_meta_b, vals_b_full)

        p_mlp_a = self.cr_mlp_0.start(partial_a, dependencies=(meta_b_full,))
        partial_b, route_ids_b = self._run_ffn_gathered(vals_b_full, meta_b_full)
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
            mlp_b = self.cr_mlp_1.finish(p_mlp_b, finish_dep)
            out_b = self.post(mlp_b, res_b, post_b, comb_b)
            incoming = (out_a, out_b)
            next_mix = (next_mix_a, next_mix_b)
        else:
            mlp_b = self.cr_mlp_1.finish(p_mlp_b, res_a)
            incoming = (mlp_a, res_a, post_a, comb_a, mlp_b, res_b, post_b, comb_b)
            next_mix = (next_mix_a, next_mix_b)
        observations = {
            "attention_hc.normalized": attention_normalized,
            "router.ids": torch.cat((route_ids_a, route_ids_b), dim=0),
            "output": incoming[0],
        }
        return incoming, next_mix, records, selection, observations

    def _run_row_shard_roce(self, incoming, pre_mix, metadata, indices, selection, *,
                            window_cache=None, image_mask=None):
        s_tok = self.shard_tokens
        if self.engram is not None:
            incoming = (self.engram(incoming[0], metadata),)
        attn_mix = torch.empty_like(pre_mix)
        residual, post, comb, norm_sh = self.attention_hc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
        if self.row_shard_attn:
            attention_normalized = norm_sh
            attention_output, records, selection, attn_obs = self.attention._run(
                norm_sh, metadata, selection, indices=indices, window_cache=window_cache)
        else:
            norm_full = self.attn_gather(norm_sh)
            attention_normalized = norm_full
            attended, records, selection, attn_obs = self.attention._run(
                norm_full, metadata, selection, indices=indices, window_cache=window_cache)
            attention_output = self.attention_reduce(attended)
        next_mix = torch.empty_like(attn_mix)
        residual, post, comb, ffn_norm_sh = self.ffn_hc(
            attention_output, residual, post, comb, pre_mix=attn_mix, pre_out=next_mix)
        if self.has_image_mask:
            if image_mask is None:
                image_mask = metadata.token_ids >= 129280
            mask_sh = torch.index_select(image_mask, 0, self.shard_rows)
        else:
            mask_sh = None
        valid_sh = (metadata.live_counts[:1] - self.shard_offset).clamp(0, s_tok).to(torch.int32)
        vals_sh, meta_sh = self._prepare_ffn_shard(ffn_norm_sh, mask_sh, valid_sh)
        vals_full = self.ffn_vals_gather(vals_sh)
        meta_full = self.ffn_meta_gather(meta_sh)
        partial, route_ids = self._run_ffn_gathered(vals_full, meta_full)
        mlp_sh = self.mlp_reduce(partial)
        incoming = ((self.post(mlp_sh, residual, post, comb),) if self.post is not None
                    else (mlp_sh, residual, post, comb))
        observations = {"attention." + key: value for key, value in attn_obs.items()}
        observations.update({
            "attention_hc.normalized": attention_normalized,
            "router.ids": route_ids,
            "output": incoming[0],
        })
        return incoming, next_mix, records, selection, observations

    def _run(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None,
             image_mask=None, engram_projected=None, engram_gathered=None, engram_overlap=None,
             engram_project=None):
        if getattr(self, "row_shard_dma", False):
            return self._run_row_shard_dma(incoming, pre_mix, metadata, indices, selection,
                window_cache=window_cache, image_mask=image_mask, engram_projected=engram_projected,
                engram_gathered=engram_gathered, engram_overlap=engram_overlap, engram_project=engram_project)
        if getattr(self, "row_shard_roce", False):
            return self._run_row_shard_roce(incoming, pre_mix, metadata, indices, selection,
                window_cache=window_cache, image_mask=image_mask)
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
                 suffix_runtime=None, encoder_only=False, row_shard_prefill=False):
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
        encoder_tokens = scratch.attention.output.shape[0]
        world = weights.shard.world_size
        self.row_shard_dma = (
            bool(row_shard_prefill)
            and mode == "prefill"
            and scoring is None
            and can_overlap
            and _is_dma_runtime(runtime)
            and row_shard_surfaces(runtime)
            and world > 1
            and decoder_start is not None
            and encoder_tokens >= 1024
            and encoder_tokens % (2 * world) == 0
            and (encoder_tokens // (2 * world)) % 128 == 0
            and PreparedReduceScatter.fits(
                runtime, scratch.attention.output[:encoder_tokens // 2],
                scratch.attention_output[:encoder_tokens // (2 * world)])
            and PreparedGatherShards.fits(
                runtime, scratch.attention_hc.normalized[:encoder_tokens // (2 * world)],
                scratch.attention_hc.normalized[:encoder_tokens // 2])
        )
        roce_shard_tokens = encoder_tokens // world if world > 1 else 0
        self.row_shard_roce = (
            bool(row_shard_prefill)
            and mode == "prefill"
            and scoring is None
            and not self.row_shard_dma
            and runtime is not None
            and not _is_dma_runtime(runtime)
            and row_shard_surfaces(runtime)
            and world > 1
            and decoder_start is not None
            and encoder_tokens >= 512
            and encoder_tokens % world == 0
            and roce_shard_tokens % 128 == 0
            and all(
                layer.engram is None or "projection_full" in layer.engram
                for layer in weights.layers[:decoder_start]
            )
            and PreparedReduceScatter.fits(
                runtime, scratch.attention.output,
                scratch.attention_output[:roce_shard_tokens])
            and PreparedGatherShards.fits(
                runtime, scratch.attention_hc.normalized[:roce_shard_tokens],
                scratch.attention_hc.normalized)
            and PreparedChunkedGatherShards.fits(
                runtime, scratch.residual[:roce_shard_tokens],
                scratch.residual)
        )
        self.row_shard_roce_attn = False
        roce_attn_scratch = None
        roce_window_target = None
        if self.row_shard_roce:
            dev = scratch.residual.device
            roce_window_local = declaration_tensor(
                examples, "row_shard_roce.window_local", (roce_shard_tokens, 512), torch.bfloat16, dev)
            roce_window_target = declaration_tensor(
                examples, "row_shard_roce.window_full", (encoder_tokens, 512), torch.bfloat16, dev)
            if (all({"q_b_full", "output_full", "sink_full"} <= set(layer.attention)
                    for layer in weights.layers[:decoder_start])
                    and PreparedGatherShards.fits(runtime, roce_window_local, roce_window_target)):
                self.row_shard_roce_attn = True
                roce_attn_scratch = AttentionScratch.allocate(
                    encoder_tokens, world, dev, shard_rows=roce_shard_tokens, examples=examples)
        self.half_tokens = encoder_tokens // 2
        if self.row_shard_dma:
            self.shard_tokens = self.half_tokens // world
        elif self.row_shard_roce:
            self.shard_tokens = roce_shard_tokens
        else:
            self.shard_tokens = 0
        self.shard_rows = self.shard_offset = None
        if self.row_shard_dma:
            r = getattr(weights.shard, "rank", 0)
            dev = scratch.residual.device
            s0 = r * self.shard_tokens
            self.shard_rows_a = persistent_copy(torch.arange(s0, s0 + self.shard_tokens, dtype=torch.int64, device=dev))
            self.shard_rows_b = persistent_copy(torch.arange(self.half_tokens + s0, self.half_tokens + s0 + self.shard_tokens, dtype=torch.int64, device=dev))
            self.shard_offset_a = persistent_copy(torch.tensor([s0], dtype=torch.int32, device=dev))
            self.shard_offset_b = persistent_copy(torch.tensor([self.half_tokens + s0], dtype=torch.int32, device=dev))
            shared_shard_rows = (self.shard_rows_a, self.shard_rows_b, self.shard_offset_a, self.shard_offset_b)
        elif self.row_shard_roce:
            r = getattr(weights.shard, "rank", 0)
            dev = scratch.residual.device
            s0 = r * self.shard_tokens
            self.shard_rows_a = self.shard_rows_b = self.shard_offset_a = self.shard_offset_b = None
            self.shard_rows = persistent_copy(torch.arange(s0, s0 + self.shard_tokens, dtype=torch.int64, device=dev))
            self.shard_offset = persistent_copy(torch.tensor([s0], dtype=torch.int32, device=dev))
            shared_shard_rows = (self.shard_rows, self.shard_offset)
        else:
            self.shard_rows_a = self.shard_rows_b = self.shard_offset_a = self.shard_offset_b = None
            shared_shard_rows = None
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
                        source_indices=self.window_metadata,
                        row_shard_dma=self.row_shard_dma, row_shard_roce=self.row_shard_roce,
                        shard=weights.shard, runtime=runtime, coordinator=coordinator, examples=examples)
                    layer = replace(layer, attention={key: value for key, value in layer.attention.items()
                                                      if key not in ("compressor", "index_k", "index_norm")})
            defer = (index + 1 < len(weights.layers) and weights.layers[index + 1].engram is None
                     and index + 1 not in self.aux_layers)
            pipelined_ffn = layer_pipelined[index]
            layer_row_shard_dma = self.row_shard_dma and index < decoder_start and pipelined_ffn
            layer_row_shard_roce = self.row_shard_roce and index < decoder_start
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
                defer_mlp_b=defer_mlp_b, defer_engram_reduce=overlap_engram, pipelined_attn=pipelined_attn,
                row_shard_dma=layer_row_shard_dma, row_shard_roce=layer_row_shard_roce,
                row_shard_attn=layer_row_shard_roce and self.row_shard_roce_attn,
                shard_rows=shared_shard_rows if (layer_row_shard_dma or layer_row_shard_roce) else None,
                shard_attention_scratch=roce_attn_scratch if (layer_row_shard_roce and self.row_shard_roce_attn) else None,
                window_target=roce_window_target if (layer_row_shard_roce and self.row_shard_roce_attn) else None)
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
                row_shard_dma=self.row_shard_dma, row_shard_roce=self.row_shard_roce,
                shard=weights.shard, runtime=runtime, coordinator=coordinator, examples=examples,
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
        if getattr(self, "row_shard_dma", False):
            s_tok = self.shard_tokens
            res_a = torch.index_select(embedding, 0, self.shard_rows_a)[:, None, :].expand(-1, 4, -1).contiguous()
            res_b = torch.index_select(embedding, 0, self.shard_rows_b)[:, None, :].expand(-1, 4, -1).contiguous()
            incoming = (res_a, res_b)
            pre_mix_a = torch.zeros((s_tok, 4), dtype=torch.float32, device=embedding.device)
            pre_mix_a[:, 0].fill_(1)
            pre_mix_b = torch.zeros((s_tok, 4), dtype=torch.float32, device=embedding.device)
            pre_mix_b[:, 0].fill_(1)
            pre_mix = (pre_mix_a, pre_mix_b)
        elif getattr(self, "row_shard_roce", False):
            s_tok = self.shard_tokens
            res_sh = torch.index_select(embedding, 0, self.shard_rows)[:, None, :].expand(-1, 4, -1).contiguous()
            incoming = (res_sh,)
            pre_mix = torch.zeros((s_tok, 4), dtype=torch.float32, device=embedding.device)
            pre_mix[:, 0].fill_(1)
        else:
            residual = embedding[:, None, :].expand(-1, 4, -1).contiguous()
            pre_mix = torch.zeros((embedding.shape[0], 4), dtype=torch.float32, device=embedding.device)
            pre_mix[:, 0].fill_(1)
            incoming = (residual,)
        indices = self.window_metadata(metadata)
        selection_rows = self.shard_tokens if getattr(self, "row_shard_roce_attn", False) else embedding.shape[0]
        selection = (torch.full((selection_rows, 512), -1, dtype=torch.int32, device=embedding.device), None, None)
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
