# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Split-attention prefill token halves for DeepSeek V4.1 PCIe tensor-parallel graphs.

Each full-row decoder layer before the global-KV boundary splits an 8192-row or larger
prefill chunk into two equal micro-batches for attention while keeping routed and shared
experts over the complete chunk. Attention for half 0 commits its sliding window,
compressor state, and compressed pages before half 1 reads them, matching two sequential
half-sized chunks. Hidden-state reductions run on two non-blocking side streams over two
dedicated half-sized DMA channels, overlapping attention compute while joining both side
streams before whole-chunk MoE so persistent cooperative expert kernels never contend with
spin-waiting DMA flag kernels.
"""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce, WeightedGroupSource
from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.mhc import MHCScratch, PreparedMHC, PreparedMHCPost
from hyperflare_recipes.components.mlp import PreparedMLP
from hyperflare_recipes.components.moe import PreparedRouter, PreparedRoutedExperts
from hyperflare_recipes.components.paged_batch import Metadata
from .attention import AttentionIndices, AttentionScratch, PreparedAttention
from .engram import PreparedEngram

# Split only 8192-token or larger prefill chunks into 128-aligned halves of at least 4096 tokens.
MIN_HALF_ROWS = 4096


def halves_size(chunk_rows):
    """Rows per half for a prefill chunk, or None when the chunk should not split."""
    half = chunk_rows // 2
    if chunk_rows % 2 or half < MIN_HALF_ROWS or half % 128:
        return None
    return half


@dataclass(frozen=True)
class PendingHalvesCarry:
    """Deferred MLP side-stream reductions and FFN residual state carried into the next halves layer."""

    pending_mlp: tuple
    ffn_state: tuple
    mlp_reduce: tuple

    def materialize(self):
        """Join both halves and concatenate into the whole-chunk 4-tensor carry tuple."""
        mlp_0 = self.mlp_reduce[0].join_side(self.pending_mlp[0])
        mlp_1 = self.mlp_reduce[1].join_side(self.pending_mlp[1])
        res_0, post_0, comb_0 = self.ffn_state[0]
        res_1, post_1, comb_1 = self.ffn_state[1]
        return (
            torch.cat((mlp_0, mlp_1), dim=0),
            torch.cat((res_0, res_1), dim=0),
            torch.cat((post_0, post_1), dim=0),
            torch.cat((comb_0, comb_1), dim=0),
        )


class ChunkHalves:
    """Per-half declaration metadata and functional SSA half-chunk geometry builders."""

    def __init__(self, metadata, size, *, ratios=()):
        if metadata.starts.numel() != 1 or metadata.positions.numel() != 2 * size:
            raise ValueError("token halves require one prefill request filling exactly two halves of rows")
        device = metadata.positions.device
        self.size = size
        blocks = metadata.page_table.shape[1]
        self.metadata = tuple(Metadata.allocate(size, 1, blocks, device, samples=1,
                                                max_query_tokens=size) for _ in range(2))
        self.window = tuple(AttentionIndices(half, mode="prefill", tokens=size, page_size=128 + size,
                                             width=128, ratios=ratios) for half in self.metadata)

    def functional(self):
        for window in self.window:
            window.functional()
        self.metadata = None
        return self

    def begin(self, source):
        size = self.size
        device = source.positions.device
        count = source.lengths[0].long()
        pages = source.page_table
        zeros_1 = torch.zeros(1, dtype=torch.int32, device=device)
        zeros_req = torch.zeros(size, dtype=torch.int64, device=device)
        metas, geoms = [], []
        for index in range(2):
            first = index * size
            kept = (count - first).clamp(0, size)
            rows = slice(first, first + size)
            meta = Metadata(
                token_ids=source.token_ids[rows],
                sample_indices=(kept - 1).clamp(min=0).view(1),
                query_start_loc=torch.cat((zeros_1, kept.view(1).int())),
                state_rows=torch.where(kept > 0, source.state_rows, 0),
                starts=source.starts + first,
                lengths=kept.view(1).int(),
                requests=zeros_req,
                positions=source.positions[rows],
                page_table=pages,
                token_page_table=pages.expand(size, -1),
                live_counts=torch.cat((kept.view(1).int(), source.live_counts[1:])),
                max_query_tokens=size,
            )
            metas.append(meta)
            geoms.append(self.window[index](meta))
        return tuple(metas), tuple(geoms)


_SUFFIX_ONLY_FIELDS = ("candidates", "candidate_counts")


def _attention_rows(scratch, first, size):
    def rows(field, value):
        if field in _SUFFIX_ONLY_FIELDS:
            return value
        if value.shape[0] < first + size:
            raise ValueError(f"attention scratch {field} rows do not cover both token halves")
        return value[first:first + size]
    return AttentionScratch(*(rows(field, getattr(scratch, field)) for field in AttentionScratch.__dataclass_fields__))


@dataclass(frozen=True)
class HalfScratch:
    residual: torch.Tensor
    engram_residual: torch.Tensor
    pre_mix: torch.Tensor
    attn_mix: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor

    @classmethod
    def of(cls, scratch, first, size):
        rows = slice(first, first + size)
        return cls(scratch.residual[rows], scratch.engram_residual[rows], scratch.pre_mix[rows],
                   scratch.attn_mix[rows], scratch.attention_hc.rows(size, start=first),
                   scratch.ffn_hc.rows(size, start=first), _attention_rows(scratch.attention, first, size),
                   scratch.attention_output[rows], scratch.mlp_partial[rows], scratch.mlp_output[rows])


class PreparedHalvesLayer:
    """Full-row prefill layer with split attention halves and whole-chunk MoE."""

    def __init__(self, index, weights, tables, scratch, metadata, halves, pool, rotary, *, shard, runtime,
                 coordinator, session, name, workspace, carry, defer_post, half_runtimes, sides,
                 mla_compute_mode="auto", examples=None):
        from .decoder import _mhc_weights

        s, text = scratch, pool.config.text
        if index >= text["candidate_source_layer_id"]:
            raise ValueError("token halves apply only to full-row layers before the candidate source layer")
        self.index, self.carry, self.sides = index, carry, sides
        self.size = halves.size
        self.has_image_mask = weights.experts.vision_correction is not None
        self.boundary = None
        self.engram = None
        if weights.engram is not None:
            self.engram = PreparedEngram(index, weights.engram, tables, s.residual, s.engram_residual,
                metadata, pool, scratch=s.engram, session=session, name=name + ".engram", workspace=workspace,
                runtime=runtime, coordinator=coordinator, examples=examples)
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        parts = tuple(HalfScratch.of(s, half * halves.size, halves.size) for half in range(2))
        self.attention_hc, self.attention, self.attention_reduce = [], [], []
        self.ffn_hc, self.mlp_reduce, self.post = [], [], []
        for half, part in enumerate(parts):
            source = part.engram_residual if self.engram is not None else part.residual
            incoming = ((part.mlp_output, part.ffn_hc.residual, part.ffn_hc.post, part.ffn_hc.comb)
                        if carry else (source,))
            self.attention_hc.append(PreparedMHC(incoming, _mhc_weights(weights, "attn", "attn_norm"),
                part.attention_hc, session=session, name=f"{name}.attn_hc.half{half}", workspace=workspace,
                pre_mix=part.pre_mix, pre_out=part.attn_mix, **options).functional())
            self.attention.append(PreparedAttention(index, weights.attention, part.attention_hc.normalized,
                part.attention, halves.metadata[half], pool, rotary, mode="prefill", world=shard.world_size,
                session=session, name=f"{name}.attention.half{half}", workspace=workspace,
                mla_compute_mode=mla_compute_mode, window_metadata=halves.window[half], examples=examples))
            self.attention_reduce.append(PreparedAllReduce(part.attention.output, part.attention_output,
                runtime=half_runtimes[half], coordinator=coordinator, session=session,
                name=f"{name}.attention_reduce.half{half}").functional(session=session))
            self.ffn_hc.append(PreparedMHC((part.attention_output, part.attention_hc.residual,
                part.attention_hc.post, part.attention_hc.comb), _mhc_weights(weights, "ffn", "ffn_norm"),
                part.ffn_hc, session=session, name=f"{name}.ffn_hc.half{half}", workspace=workspace,
                pre_mix=part.attn_mix, pre_out=part.pre_mix, **options).functional())
            self.mlp_reduce.append(PreparedAllReduce(part.mlp_partial, part.mlp_output,
                runtime=half_runtimes[half], coordinator=coordinator, session=session,
                name=f"{name}.mlp_reduce.half{half}").functional(session=session))
            self.post.append(None if defer_post else PreparedMHCPost(part.mlp_output, part.ffn_hc.residual,
                part.ffn_hc.post, part.ffn_hc.comb, part.residual, session=session,
                name=f"{name}.post.half{half}").functional())

        image_mask = (torch.zeros(metadata.token_ids.shape, dtype=torch.bool, device=metadata.token_ids.device)
                      if self.has_image_mask else None)
        self.router_projection = UnquantizedProjection(s.ffn_hc.normalized, weights.experts.router, s.router_logits,
            session=session, name=name + ".router").functional(session=session)
        self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
            image_correction_bias=weights.experts.vision_correction if self.has_image_mask else None,
            image_mask=image_mask).functional(session=session)
        self.experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.experts, s.router.ids,
            s.router.weights, s.routed, session=session, name=name + ".experts", workspace=workspace,
            router_plan=self.router.plan).functional(session=session)
        self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up,
            s.shared.intermediate, s.shared.output, session=session, name=name + ".shared", workspace=workspace,
            swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto").functional(session=session)

    def __call__(self, incoming, pre_mix, metadata, indices, selection, *, halves_state=None, boundary=None, window_cache=None):
        h = self.size
        side_a, side_b = self.sides
        half_metas, half_geoms = halves_state

        if self.engram is not None:
            if isinstance(incoming, PendingHalvesCarry):
                raise RuntimeError("Engram layer cannot consume deferred halves carry")
            incoming = (self.engram(incoming[0], metadata),)

        selected, candidates, counts = selection
        sel_0 = (selected[:h], None if candidates is None else candidates[:h], None if counts is None else counts[:h])
        sel_1 = (selected[h:], None if candidates is None else candidates[h:], None if counts is None else counts[h:])
        mix_0, mix_1 = pre_mix[:h], pre_mix[h:]

        # Half 0: join previous layer's half-0 MLP reduce if carried, run attention(0) overlapping prev mlp(1).
        if isinstance(incoming, PendingHalvesCarry):
            mlp_out_0 = incoming.mlp_reduce[0].join_side(incoming.pending_mlp[0])
            in_0 = (mlp_out_0, *incoming.ffn_state[0])
        else:
            in_0 = tuple(v[:h] for v in incoming)
        attn_mix_0 = torch.zeros_like(mix_0)
        a_res_0, a_post_0, a_comb_0, a_norm_0 = self.attention_hc[0](*in_0, pre_mix=mix_0, pre_out=attn_mix_0)
        att_0, _, sel_0, _ = self.attention[0]._run(a_norm_0, half_metas[0], sel_0, indices=half_geoms[0], window_cache=None)
        pend_att_0 = self.attention_reduce[0].start_side(att_0, side_a)

        # Half 1: join previous layer's half-1 MLP reduce after attention(0), run attention(1) overlapping att_reduce(0).
        if isinstance(incoming, PendingHalvesCarry):
            mlp_out_1 = incoming.mlp_reduce[1].join_side(incoming.pending_mlp[1], dependencies=(att_0, pend_att_0.output))
            in_1 = (mlp_out_1, *incoming.ffn_state[1])
        else:
            in_1 = tuple(v[h:] for v in incoming)
        attn_mix_1 = torch.zeros_like(mix_1)
        a_res_1, a_post_1, a_comb_1, a_norm_1 = self.attention_hc[1](*in_1, pre_mix=mix_1, pre_out=attn_mix_1)
        att_1, _, sel_1, _ = self.attention[1]._run(a_norm_1, half_metas[1], sel_1, indices=half_geoms[1], window_cache=None)
        pend_att_1 = self.attention_reduce[1].start_side(att_1, side_b, after=side_a, dependencies=(pend_att_0.output,))

        # Propagate newly selected compressed-KV indices from indexer layers to downstream attention layers.
        if self.attention[0].selectors:
            next_selection = (torch.cat((sel_0[0], sel_1[0]), dim=0), None, None)
        else:
            next_selection = selection

        # Join both attention reductions before whole-chunk MoE so cooperative expert kernels never overlap DMA waits.
        att_out_0 = self.attention_reduce[0].join_side(pend_att_0, dependencies=(att_1, pend_att_1.output))
        next_mix_0 = torch.zeros_like(attn_mix_0)
        f_res_0, f_post_0, f_comb_0, f_norm_0 = self.ffn_hc[0](att_out_0, a_res_0, a_post_0, a_comb_0,
            pre_mix=attn_mix_0, pre_out=next_mix_0)

        att_out_1 = self.attention_reduce[1].join_side(pend_att_1, dependencies=(f_norm_0,))
        next_mix_1 = torch.zeros_like(attn_mix_1)
        f_res_1, f_post_1, f_comb_1, f_norm_1 = self.ffn_hc[1](att_out_1, a_res_1, a_post_1, a_comb_1,
            pre_mix=attn_mix_1, pre_out=next_mix_1)

        # Whole-chunk MoE + shared expert so expert weights are read once.
        whole = torch.cat((f_norm_0, f_norm_1), dim=0)
        image_mask = metadata.token_ids >= 129280 if self.has_image_mask else None
        router_logits = self.router_projection(whole)
        route = self.router(router_logits, image_mask=image_mask)
        routed = self.experts(whole, route.ids, route.weights, router_logits=router_logits,
                              selected_logits=route.selected_logits, image_mask=image_mask)
        shared = self.shared(whole)
        mlp_partial = (routed.values if isinstance(routed, WeightedGroupSource) else routed) + shared

        pend_mlp_0 = self.mlp_reduce[0].start_side(mlp_partial[:h].contiguous(), side_a)
        pend_mlp_1 = self.mlp_reduce[1].start_side(mlp_partial[h:].contiguous(), side_b, after=side_a,
                                                   dependencies=(pend_mlp_0.output,))
        next_mix = torch.cat((next_mix_0, next_mix_1), dim=0)

        if self.post[0] is not None:
            mlp_out_0 = self.mlp_reduce[0].join_side(pend_mlp_0)
            res_0 = self.post[0](mlp_out_0, f_res_0, f_post_0, f_comb_0)
            mlp_out_1 = self.mlp_reduce[1].join_side(pend_mlp_1, dependencies=(res_0,))
            res_1 = self.post[1](mlp_out_1, f_res_1, f_post_1, f_comb_1)
            next_incoming = (torch.cat((res_0, res_1), dim=0),)
        else:
            next_incoming = PendingHalvesCarry(
                pending_mlp=(pend_mlp_0, pend_mlp_1),
                ffn_state=((f_res_0, f_post_0, f_comb_0), (f_res_1, f_post_1, f_comb_1)),
                mlp_reduce=(self.mlp_reduce[0], self.mlp_reduce[1]),
            )
        return next_incoming, next_mix, None, next_selection
