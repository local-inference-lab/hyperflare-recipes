# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""ERNIE 5.5 Flash Preview mHC decoder stack with Latent Multi-Router SiTU MoE and 2-microbatch DMA prefill."""

from dataclasses import dataclass, fields, is_dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as tldevice

from hyperflare_recipes.components.dspark import PreparedFeedbackCapture
from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.expert_overlap import ConcurrentSharedExpert
from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.markov_sampling import apply_multimax_inplace
from hyperflare_recipes.components.mhc import (
    MHCScratch,
    MHCWeights,
    PreparedMHC,
    PreparedMHCHead,
    PreparedMHCPost,
)
from hyperflare_recipes.components.mlp import MLPScratch, PreparedMLP
from hyperflare_recipes.components.moe import PreparedRoutedExperts, RouterScratch
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.rotary import CompressedRotary as Rotary
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from ignition.components.collectives import ConcurrentAllReduce, PreparedAllReduce
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer

from .attention import (
    AttentionIndices,
    AttentionScratch,
    PreparedDSAAttention,
    PreparedHCAAttention,
)
from .contract import VOCABULARY


# ---------------------------------------------------------------------------
# Fused 1-Warp ERNIE 5.5 Split-Feature Sigmoid Multi-Router Kernel
# ---------------------------------------------------------------------------

@triton.jit
def _sigmoid_rn(x):
    return tldevice.div_rn(1.0, 1.0 + tldevice.exp(-x))


@triton.jit
def _ernie55_split_sigmoid_routing_kernel(
    logits_ptr,
    bias_ptr,
    scale_ptr,
    positions_ptr,
    weights_ptr,
    ids_ptr,
    stride,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    RENORM: tl.constexpr,
    MASK_PAD: tl.constexpr,
):
    """Fused split_sigmoid_sum(logits[:, :E], logits[:, E:]) + bias top-k + renorm + learnable scale."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, E)
    kofs = tl.arange(0, TOPK_PAD)
    l0 = tl.load(logits_ptr + row * stride + offs).to(tl.float32)
    l1 = tl.load(logits_ptr + row * stride + E + offs).to(tl.float32)
    scores = _sigmoid_rn(l0) + _sigmoid_rn(l1)
    sel = scores + tl.load(bias_ptr + offs)
    ids = tl.full((TOPK_PAD,), E, dtype=tl.int32)
    vals = tl.zeros((TOPK_PAD,), dtype=tl.float32)
    weights = tl.zeros((TOPK_PAD,), dtype=tl.float32)
    best = tl.zeros((), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(sel, 0)
        idx = tl.minimum(tl.min(tl.where(sel == best, offs, E), 0), E - 1)
        hit = offs == idx
        ids = tl.where(kofs == k, idx, ids)
        vals = tl.where(kofs == k, best, vals)
        weights = tl.where(kofs == k, tl.sum(tl.where(hit, scores, 0.0), 0), weights)
        sel = tl.where(hit, float("-inf"), sel)
    kmask = kofs < TOPK
    above = kmask & (vals > best)
    key = tl.where(above, ids, ids + E)
    slot = tl.sum((key[None, :] < key[:, None]).to(tl.int32), 1)
    move = slot[None, :] == kofs[:, None]
    ids = tl.sum(tl.where(move, ids[None, :], 0), 1)
    weights = tl.sum(tl.where(move, weights[None, :], 0.0), 1)
    if RENORM:
        total = tl.maximum(tl.sum(weights, 0), 1e-20)
        weights = tldevice.div_rn(weights, total)
    weights = weights * tl.load(scale_ptr + ids, mask=kmask & (ids >= 0) & (ids < E), other=0.0)
    if MASK_PAD:
        is_pad = tl.load(positions_ptr + row) < 0
        weights = tl.where(is_pad, 0.0, weights)
        ids = tl.where(is_pad, -1, ids)
    tl.store(weights_ptr + row * TOPK + kofs, weights, mask=kmask)
    tl.store(ids_ptr + row * TOPK + kofs, ids, mask=kmask)


def ernie55_route(logits, bias, scale, positions, *, top_k=10, renormalize=True, mask_pad=True):
    tokens, double_e = logits.shape
    num_experts = double_e // 2
    weights = torch.empty((tokens, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((tokens, top_k), dtype=torch.int32, device=logits.device)
    torch.library.wrap_triton(_ernie55_split_sigmoid_routing_kernel)[(tokens,)](
        logits,
        bias,
        scale,
        positions,
        weights,
        ids,
        logits.stride(0),
        E=num_experts,
        TOPK=top_k,
        TOPK_PAD=triton.next_power_of_2(top_k),
        RENORM=renormalize,
        MASK_PAD=mask_pad,
        enable_reflect_ftz=False,
        num_warps=2,
    )
    return ids, weights


# ---------------------------------------------------------------------------
# Decoder Scratch
# ---------------------------------------------------------------------------

def _is_dma_runtime(runtime):
    if runtime is None:
        return False
    from b12x.comm import pcie

    return isinstance(runtime, pcie.DmaAllReduce)


def _mhc_weights(weights, kind, norm):
    r = weights.residual
    return MHCWeights(r[f"hc_{kind}_fn"], r[f"hc_{kind}_scale"], r[f"hc_{kind}_base"], r[norm + ".weight"])


@dataclass(frozen=True)
class Scratch:
    """BF16 decoder activations and FP32 target vocabulary logits."""

    logits_dtype = torch.float32

    embedding: torch.Tensor
    residual: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    dense_mlp: MLPScratch
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    latent_in: torch.Tensor
    routed_partial: torch.Tensor
    routed_reduced: torch.Tensor
    latent_normed: torch.Tensor
    routed_up: torch.Tensor
    mlp_output: torch.Tensor
    head_residual: torch.Tensor
    collapsed: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[: min(count, value.shape[0])]
            if is_dataclass(value):
                return type(value)(*(select(getattr(value, f.name), count) for f in fields(value)))
            raise TypeError("decoder scratch must contain tensors and tensor dataclasses")

        sampled = ("head_residual", "collapsed", "hidden", "local_logits", "logits")
        return type(self)(
            *(select(getattr(self, f.name), samples if f.name in sampled else tokens) for f in fields(self))
        )

    @classmethod
    def allocate(
        cls,
        tokens,
        samples,
        world,
        device,
        *,
        blocks=8192,
        experts=512,
        top_k=10,
        include_dsa=True,
        include_dense=True,
        vha_dtype=torch.bfloat16,
    ):
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        _, local_vocab = TensorParallelShard(0, world).padded_partition(VOCABULARY, alignment=64)
        dense_tokens = tokens if include_dense else 1
        return cls(
            embedding=zeros(tokens, 4096),
            residual=zeros(tokens, 4, 4096),
            attention_hc=MHCScratch.allocate(tokens, 4096, device),
            ffn_hc=MHCScratch.allocate(tokens, 4096, device),
            attention=AttentionScratch.allocate(
                tokens, world, device, blocks=blocks, include_dsa=include_dsa, vha_dtype=vha_dtype
            ),
            attention_output=zeros(tokens, 4096),
            dense_mlp=MLPScratch.allocate(dense_tokens, 4096, 14336 // world, device),
            shared=MLPScratch.allocate(tokens, 4096, 2048, device),
            router_logits=zeros(tokens, 2 * experts, dtype=torch.float32),
            router=RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k),
            latent_in=zeros(tokens, 2048),
            routed_partial=zeros(tokens, 2048),
            routed_reduced=zeros(tokens, 2048),
            latent_normed=zeros(tokens, 2048),
            routed_up=zeros(tokens, 4096),
            mlp_output=zeros(tokens, 4096),
            head_residual=zeros(samples, 4, 4096),
            collapsed=zeros(samples, 4096),
            hidden=zeros(samples, 4096),
            local_logits=zeros(samples, local_vocab, dtype=cls.logits_dtype),
            logits=zeros(samples, VOCABULARY, dtype=cls.logits_dtype),
        )


class DraftScratch(Scratch):
    """BF16 draft logits for DSpark proposals (multimax runs after Markov bias)."""

    logits_dtype = torch.bfloat16

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, blocks=8192, experts=512, top_k=10, **kwargs):
        return super().allocate(
            tokens,
            samples,
            world,
            device,
            blocks=blocks,
            experts=experts,
            top_k=top_k,
            include_dsa=False,
            include_dense=False,
            vha_dtype=torch.bfloat16,
        )


# ---------------------------------------------------------------------------
# Prepared Decoder Layer (HCA/DSA + Dense SiTU or Latent Multi-Router SiTU MoE)
# ---------------------------------------------------------------------------

class PreparedLayer:
    def __init__(
        self,
        index,
        weights,
        scratch: Scratch,
        metadata,
        pool,
        rotary,
        *,
        shard,
        mode,
        runtime,
        coordinator,
        session,
        name,
        workspace,
        carry,
        defer_post,
        indices,
        max_cache_pages=None,
        mla_compute_mode="auto",
        examples=None,
        pipelined=False,
        prev_deferred_reduce=False,
        defer_reduce_b=False,
        return_pre_residual=False,
    ):
        self.index, self.weights, self.carry = index, weights, carry
        self.mode = mode
        self.is_dsa = weights.is_dsa
        self.is_moe = weights.experts is not None
        self.pipelined = bool(pipelined)
        self.full_half_hca = self.pipelined and not self.is_dsa
        self.prev_deferred_reduce = bool(prev_deferred_reduce)
        self.defer_post = bool(defer_post)
        self.defer_reduce_b = bool(defer_reduce_b)
        self.return_pre_residual = bool(return_pre_residual)
        s, text = scratch, pool.config.text
        tokens = s.attention.output.shape[0]
        h = tokens // 2 if self.pipelined else tokens
        self.half_tokens = h
        self.top_k = text["num_experts_per_tok"]
        self.renormalize = text["norm_topk_prob"]
        self.mask_pad = mode in ("prefill", "verify")

        mhc_opts = dict(
            rms_eps=text["rms_norm_eps"],
            norm_eps=text["rms_norm_eps"],
            hc_eps=text["hc_eps"],
            sinkhorn_iters=text["hc_sinkhorn_iters"],
            rms_eps_outside_sqrt=True,
        )
        # Attention mHC (half-sized when prev_deferred_reduce or full_half_hca!)
        if self.prev_deferred_reduce or self.full_half_hca:
            attn_in = (
                (s.mlp_output[:h], s.ffn_hc.residual[:h], s.ffn_hc.post[:h], s.ffn_hc.comb[:h])
                if (self.prev_deferred_reduce or carry)
                else (s.residual[:h],)
            )
            self.attention_hc = PreparedMHC(
                attn_in,
                _mhc_weights(weights, "attn", "attn_norm"),
                s.attention_hc.rows(h),
                session=session,
                name=name + ".attn_hc",
                workspace=workspace,
                **mhc_opts,
            ).functional()
        else:
            attn_in = (s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if carry else (s.residual,)
            self.attention_hc = PreparedMHC(
                attn_in,
                _mhc_weights(weights, "attn", "attn_norm"),
                s.attention_hc,
                session=session,
                name=name + ".attn_hc",
                workspace=workspace,
                max_rows_per_call=2048 if mode == "prefill" and carry and tokens > 2048 else None,
                **mhc_opts,
            ).functional()

        # Attention (HCA or DSA)
        if self.is_dsa:
            self.attention = PreparedDSAAttention(
                index,
                weights.attention,
                s.attention_hc.normalized,
                s.attention,
                metadata,
                pool,
                rotary,
                mode=mode,
                world=shard.world_size,
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".attention",
                workspace=workspace,
                indices=indices,
                max_cache_pages=max_cache_pages,
                examples=examples,
                pipelined=self.pipelined,
            )
        else:
            self.attention = PreparedHCAAttention(
                index,
                weights.attention,
                s.attention_hc.normalized,
                s.attention,
                metadata,
                pool,
                rotary,
                mode=mode,
                world=shard.world_size,
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".attention",
                workspace=workspace,
                indices=indices,
                mla_compute_mode=mla_compute_mode,
                examples=examples,
                pipelined=self.pipelined,
            )

        # Attention AllReduce
        if self.pipelined:
            self.attention_reduce = None
            self.att_reduce_0 = PreparedAllReduce(
                s.attention.output[:h],
                s.attention_output[:h],
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".attention_reduce.0",
            )
            self.att_reduce_1 = PreparedAllReduce(
                s.attention.output[h:],
                s.attention_output[h:],
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".attention_reduce.1",
            )
        else:
            self.att_reduce_0 = self.att_reduce_1 = None
            self.attention_reduce = PreparedAllReduce(
                s.attention.output,
                s.attention_output,
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".attention_reduce",
            ).functional(session=session)

        # FFN mHC
        ffn_hc_s = s.ffn_hc.rows(h) if self.pipelined else s.ffn_hc
        self.ffn_hc = PreparedMHC(
            (
                s.attention_output[:h],
                s.attention_hc.residual[:h],
                s.attention_hc.post[:h],
                s.attention_hc.comb[:h],
            ),
            _mhc_weights(weights, "ffn", "ffn_norm"),
            ffn_hc_s,
            session=session,
            name=name + ".ffn_hc",
            workspace=workspace,
            max_rows_per_call=2048 if mode == "prefill" and not self.pipelined and h > 2048 else None,
            **mhc_opts,
        ).functional()

        # FFN: Dense SiTU MLP (layer 0) OR Latent Multi-Router SiTU MoE (layers 1..42 & draft)
        self.dense_mlp = self.mlp_reduce = self.mlp_reduce_0 = self.mlp_reduce_1 = None
        self.fc1_latent = self.router_proj = self.experts = self.latent_reduce = None
        self.latent_reduce_0 = self.latent_reduce_1 = self.latent_norm = self.fc2_latent = None
        self.shared = self.concurrent_shared = None
        self.cr_att_0 = self.cr_att_1 = self.cr_ffn_0 = self.cr_ffn_1 = None

        if not self.is_moe:
            dense_s = s.dense_mlp.rows(h) if self.pipelined else s.dense_mlp
            self.dense_mlp = PreparedMLP(
                ffn_hc_s.normalized,
                weights.mlp,
                dense_s.gate_up,
                dense_s.intermediate,
                dense_s.output,
                session=session,
                name=name + ".mlp",
                workspace=workspace,
                activation="situ",
                activation_mode="auto",
            ).functional(session=session)
            att1_plans = ffn0_plans = (
                self.ffn_hc.plan,
                self.dense_mlp.up_projection.plan,
                self.dense_mlp.activation_plan,
                self.dense_mlp.down_projection.plan,
            )
            ffn1_plans = (self.ffn_hc.plan,)
            if self.pipelined:
                self.mlp_reduce_0 = PreparedAllReduce(
                    dense_s.output,
                    s.mlp_output[:h],
                    runtime=runtime,
                    coordinator=coordinator,
                    session=session,
                    name=name + ".mlp_reduce.0",
                )
                self.mlp_reduce_1 = PreparedAllReduce(
                    s.dense_mlp.output[h:],
                    s.mlp_output[h:],
                    runtime=runtime,
                    coordinator=coordinator,
                    session=session,
                    name=name + ".mlp_reduce.1",
                )
            else:
                self.mlp_reduce = PreparedAllReduce(
                    s.dense_mlp.output,
                    s.mlp_output,
                    runtime=runtime,
                    coordinator=coordinator,
                    session=session,
                    name=name + ".mlp_reduce",
                ).functional(session=session)
        else:
            e = weights.experts
            self.router_correction = e.correction
            self.router_scale = e.scale
            record_numeric_tensor(self.router_correction)
            record_numeric_tensor(self.router_scale)
            self.fc1_latent = QuantizedProjection(
                ffn_hc_s.normalized,
                e.fc1_latent,
                s.latent_in[:h],
                session=session,
                name=name + ".fc1_latent",
                workspace=workspace,
                activation_mode="auto",
                examples=examples,
                example_role="moe.fc1_latent",
            )
            self.router_proj = UnquantizedProjection(
                ffn_hc_s.normalized,
                e.router,
                s.router_logits[:h],
                session=session,
                name=name + ".router",
            )
            # Single-Shot Full-Chunk MoE in pipelined prefill: run experts ONCE at full tokens
            # so all 76 GB of MoE weights are read once per chunk instead of twice!
            self.experts = PreparedRoutedExperts(
                s.latent_in,
                e.experts,
                s.router.ids,
                s.router.weights,
                s.routed_partial,
                session=session,
                name=name + ".experts",
                workspace=workspace,
            )
            self.latent_norm = PreparedRMSNorm(
                s.routed_reduced[:h],
                e.latent_norm,
                s.latent_normed[:h],
                eps=text["rms_norm_eps"],
                session=session,
                name=name + ".latent_norm",
            )
            self.fc2_latent = QuantizedProjection(
                s.latent_normed[:h],
                e.fc2_latent,
                s.routed_up[:h],
                session=session,
                name=name + ".fc2_latent",
                workspace=workspace,
                activation_mode="auto",
                examples=examples,
                example_role="moe.fc2_latent",
            )
            shared_s = s.shared.rows(h) if self.pipelined else s.shared
            self.shared = PreparedMLP(
                ffn_hc_s.normalized,
                e.shared,
                shared_s.gate_up,
                shared_s.intermediate,
                shared_s.output,
                session=session,
                name=name + ".shared",
                workspace=workspace,
                activation="situ",
                activation_mode="auto",
            )
            if (
                not self.pipelined
                and mode in ("decode", "verify", "draft")
                and h <= 256
                and isinstance(workspace, ScratchArena)
                and workspace.overlap
            ):
                parallel = [
                    p
                    for p in (
                        self.router_proj.plan,
                        self.experts.plan,
                        getattr(self.experts, "scale_plan", None),
                        self.latent_norm.plan,
                        self.fc2_latent.plan,
                        getattr(self.fc2_latent, "quantizer_plan", None),
                    )
                    if p is not None
                ]
                self.concurrent_shared = ConcurrentSharedExpert(
                    self.shared,
                    parallel_plans=parallel,
                    session=session,
                    workspace=workspace,
                    defer_launch=h > 4,
                )
            att1_plans = tuple(
                p
                for p in (
                    getattr(self.ffn_hc, "plan", None),
                    self.fc1_latent.plan,
                    getattr(self.fc1_latent, "quantizer_plan", None),
                    self.router_proj.plan,
                    self.shared.up_projection.plan,
                    getattr(self.shared.up_projection, "quantizer_plan", None),
                    self.shared.activation_plan,
                    self.shared.down_projection.plan,
                    getattr(self.shared.down_projection, "quantizer_plan", None),
                )
                if p is not None
            )
            ffn0_plans = tuple(
                p
                for p in (
                    self.shared.up_projection.plan,
                    getattr(self.shared.up_projection, "quantizer_plan", None),
                    self.shared.activation_plan,
                    self.shared.down_projection.plan,
                    getattr(self.shared.down_projection, "quantizer_plan", None),
                )
                if p is not None
            )
            ffn1_plans = tuple(
                p
                for p in (
                    self.latent_norm.plan,
                    self.fc2_latent.plan,
                    getattr(self.fc2_latent, "quantizer_plan", None),
                )
                if p is not None
            )
            if self.pipelined:
                self.latent_reduce_0 = PreparedAllReduce(
                    s.routed_partial[:h],
                    s.routed_reduced[:h],
                    runtime=runtime,
                    coordinator=coordinator,
                    session=session,
                    name=name + ".latent_reduce.0",
                )
                self.latent_reduce_1 = PreparedAllReduce(
                    s.routed_partial[h:],
                    s.routed_reduced[h:],
                    runtime=runtime,
                    coordinator=coordinator,
                    session=session,
                    name=name + ".latent_reduce.1",
                )
            else:
                self.latent_reduce = PreparedAllReduce(
                    s.routed_partial,
                    s.routed_reduced,
                    runtime=runtime,
                    coordinator=coordinator,
                    session=session,
                    name=name + ".latent_reduce",
                ).functional(session=session)
            self.fc1_latent.functional(session=session)
            self.router_proj.functional(session=session)
            self.experts.functional(session=session)
            self.latent_norm.functional()
            self.fc2_latent.functional(session=session)
            if self.concurrent_shared is not None:
                self.concurrent_shared.functional(session=session)
            else:
                self.shared.functional(session=session)

        # Post mHC (when not deferred to next layer's attention_hc)
        self.post = (
            None
            if defer_post
            else PreparedMHCPost(
                s.mlp_output[:h],
                ffn_hc_s.residual,
                ffn_hc_s.post,
                ffn_hc_s.comb,
                s.residual[:h],
                session=session,
                name=name + ".post",
            ).functional()
        )

        # Wire 2-microbatch DMA overlap wrappers
        if self.pipelined:
            att_parallel = self.attention.plans()
            if not self.att_reduce_0.single:
                self.cr_att_0 = ConcurrentAllReduce(
                    self.att_reduce_0, workspace=workspace, session=session, parallel_plans=att_parallel
                ).functional(session=session)
                self.cr_att_1 = ConcurrentAllReduce(
                    self.att_reduce_1, workspace=workspace, session=session, parallel_plans=att1_plans
                ).functional(session=session)
            else:
                self.att_reduce_0.functional(session=session)
                self.att_reduce_1.functional(session=session)
            ffn_r0 = self.latent_reduce_0 if self.is_moe else self.mlp_reduce_0
            ffn_r1 = self.latent_reduce_1 if self.is_moe else self.mlp_reduce_1
            if not ffn_r0.single:
                self.cr_ffn_0 = ConcurrentAllReduce(
                    ffn_r0, workspace=workspace, session=session, parallel_plans=ffn0_plans
                ).functional(session=session)
                end_parallel = (*ffn1_plans, self.post.plan) if self.post is not None else ffn1_plans
                self.cr_ffn_1 = ConcurrentAllReduce(
                    ffn_r1, workspace=workspace, session=session, parallel_plans=end_parallel
                ).functional(session=session)
            else:
                ffn_r0.functional(session=session)
                ffn_r1.functional(session=session)

        register_finalizer(session, self._release_examples)

    def _release_examples(self):
        if self.concurrent_shared is not None:
            self.shared = None

    def _run_ffn_single(self, normalized, positions):
        if not self.is_moe:
            return self.mlp_reduce(self.dense_mlp(normalized))
        latent = self.fc1_latent(normalized)
        # Start shared expert AFTER fc1_latent so fc1_latent runs at uncontended GDDR7 BW,
        # while shared_up/down overlaps with router + routing + csf_expand + experts + AR!
        pending = None if self.concurrent_shared is None else self.concurrent_shared.start(normalized)
        logits = self.router_proj(normalized)
        ids, weights = ernie55_route(
            logits,
            self.router_correction,
            self.router_scale,
            positions,
            top_k=self.top_k,
            renormalize=self.renormalize,
            mask_pad=self.mask_pad,
        )
        routed = self.experts(latent, ids, weights)
        reduced = self.latent_reduce(routed)
        routed_up = self.fc2_latent(self.latent_norm(reduced))
        shared = self.shared(normalized) if pending is None else self.concurrent_shared.finish(pending, routed_up)
        return routed_up + shared

    def __call__(self, incoming, metadata, indices, *, prev_cr=None):
        if not self.pipelined:
            residual, post, comb, norm_in = self.attention_hc(*incoming)
            pre_res = residual if self.return_pre_residual else None
            attended, records = self.attention(norm_in, metadata, indices)
            attended = self.attention_reduce(attended)
            residual, post, comb, ffn_in = self.ffn_hc(attended, residual, post, comb)
            mlp_out = self._run_ffn_single(ffn_in, metadata.positions)
            if self.post is None:
                return (mlp_out, residual, post, comb), records, pre_res
            return (self.post(mlp_out, residual, post, comb),), records, pre_res

        # 2-microbatch DMA/SM overlapped prefill!
        h = self.half_tokens
        sl_a, sl_b = slice(0, h), slice(h, 2 * h)
        if self.full_half_hca:
            if self.prev_deferred_reduce:
                mlp_a, prev_res_a, prev_post_a, prev_comb_a, p_ffn_b, prev_b_state = incoming
                res_a, post_a, comb_a, norm_a = self.attention_hc(mlp_a, prev_res_a, prev_post_a, prev_comb_a)

                def _prepare_b(ffn_b_red):
                    prev_res_b, prev_post_b, prev_comb_b, prev_is_moe, prev_layer, shared_b_prev = prev_b_state
                    mlp_b = (
                        prev_layer.fc2_latent(prev_layer.latent_norm(ffn_b_red)) + shared_b_prev
                        if prev_is_moe
                        else ffn_b_red
                    )
                    res_b_loc, post_b_loc, comb_b_loc, norm_b_loc = self.attention_hc(
                        mlp_b, prev_res_b, prev_post_b, prev_comb_b
                    )
                    return norm_b_loc, (res_b_loc, post_b_loc, comb_b_loc)

                att_a, out_b, records, (res_b, post_b, comb_b) = self.attention.run_pipelined_full(
                    norm_a,
                    _prepare_b,
                    metadata,
                    indices,
                    cr_att_0=self.cr_att_0,
                    att_reduce_0=self.att_reduce_0,
                    prev_pending=p_ffn_b,
                    prev_finish_cr=prev_cr,
                )
            else:
                in_a = tuple(t[sl_a] for t in incoming)
                in_b = tuple(t[sl_b] for t in incoming)
                res_a, post_a, comb_a, norm_a = self.attention_hc(*in_a)

                def _prepare_b(_unused):
                    res_b_loc, post_b_loc, comb_b_loc, norm_b_loc = self.attention_hc(*in_b)
                    return norm_b_loc, (res_b_loc, post_b_loc, comb_b_loc)

                att_a, out_b, records, (res_b, post_b, comb_b) = self.attention.run_pipelined_full(
                    norm_a,
                    _prepare_b,
                    metadata,
                    indices,
                    cr_att_0=self.cr_att_0,
                    att_reduce_0=self.att_reduce_0,
                )
        else:
            prev_pending = None
            if self.prev_deferred_reduce:
                mlp_a, prev_res_a, prev_post_a, prev_comb_a, p_ffn_b, prev_b_state = incoming
                res_a, post_a, comb_a, norm_a = self.attention_hc(mlp_a, prev_res_a, prev_post_a, prev_comb_a)
                ffn_b_red = prev_cr.finish(p_ffn_b, norm_a)
                prev_res_b, prev_post_b, prev_comb_b, prev_is_moe, prev_layer, shared_b_prev = prev_b_state
                if prev_is_moe:
                    mlp_b = prev_layer.fc2_latent(prev_layer.latent_norm(ffn_b_red)) + shared_b_prev
                else:
                    mlp_b = ffn_b_red
                res_b, post_b, comb_b, norm_b = self.attention_hc(mlp_b, prev_res_b, prev_post_b, prev_comb_b)
                norm_in = torch.cat((norm_a, norm_b), dim=0)
            else:
                residual, post, comb, norm_in = self.attention_hc(*incoming)
                res_a, res_b = residual[sl_a], residual[sl_b]
                post_a, post_b = post[sl_a], post[sl_b]
                comb_a, comb_b = comb[sl_a], comb[sl_b]

            att_a, out_b, records, _ = self.attention.run_pipelined(
                norm_in,
                metadata,
                indices,
                cr_att_0=self.cr_att_0,
                att_reduce_0=self.att_reduce_0,
                prev_pending=prev_pending,
                prev_finish_cr=prev_cr,
            )
        pre_res = torch.cat((res_a, res_b), dim=0) if self.return_pre_residual else None
        # Overlap Attn AR(B) with Half A FFN front-end (+ shared_a for MoE)!
        p_att_b = self.cr_att_1.start(out_b, dependencies=(att_a,)) if self.cr_att_1 is not None else None
        res_a, post_a, comb_a, ffn_a = self.ffn_hc(att_a, res_a, post_a, comb_a)
        if self.is_moe:
            lat_a = self.fc1_latent(ffn_a)
            ids_a, w_a = ernie55_route(
                self.router_proj(ffn_a),
                self.router_correction,
                self.router_scale,
                metadata.positions[sl_a],
                top_k=self.top_k,
                renormalize=self.renormalize,
                mask_pad=self.mask_pad,
            )
            shared_a = self.shared(ffn_a)
            att_b = (
                self.cr_att_1.finish(p_att_b, shared_a)
                if p_att_b is not None
                else self.att_reduce_1(out_b)
            )
            res_b, post_b, comb_b, ffn_b = self.ffn_hc(att_b, res_b, post_b, comb_b)
            lat_b = self.fc1_latent(ffn_b)
            ids_b, w_b = ernie55_route(
                self.router_proj(ffn_b),
                self.router_correction,
                self.router_scale,
                metadata.positions[sl_b],
                top_k=self.top_k,
                renormalize=self.renormalize,
                mask_pad=self.mask_pad,
            )
            # Single-Shot Full-Chunk MoE: read all 76 GB of MoE weights ONCE at full T!
            part = self.experts(
                torch.cat((lat_a, lat_b), dim=0),
                torch.cat((ids_a, ids_b), dim=0),
                torch.cat((w_a, w_b), dim=0),
            )
            part_a, part_b = part[sl_a], part[sl_b]
            # Overlap FFN AR(A) with Half B shared expert!
            p_ffn_a = self.cr_ffn_0.start(part_a) if self.cr_ffn_0 is not None else None
            shared_b = self.shared(ffn_b)
            red_a = (
                self.cr_ffn_0.finish(p_ffn_a, shared_b)
                if p_ffn_a is not None
                else self.latent_reduce_0(part_a)
            )
        else:
            part_a = self.dense_mlp(ffn_a)
            shared_a = shared_b = None
            att_b = (
                self.cr_att_1.finish(p_att_b, part_a)
                if p_att_b is not None
                else self.att_reduce_1(out_b)
            )
            p_ffn_a = self.cr_ffn_0.start(part_a, dependencies=(att_b,)) if self.cr_ffn_0 is not None else None
            res_b, post_b, comb_b, ffn_b = self.ffn_hc(att_b, res_b, post_b, comb_b)
            part_b = self.dense_mlp(ffn_b)
            red_a = (
                self.cr_ffn_0.finish(p_ffn_a, part_b)
                if p_ffn_a is not None
                else self.mlp_reduce_0(part_a)
            )

        ffn_r1 = self.latent_reduce_1 if self.is_moe else self.mlp_reduce_1
        # Overlap FFN AR(B) with Half A latent_norm+fc2_latent (and next layer's Half A HCA)!
        p_ffn_b = self.cr_ffn_1.start(part_b, dependencies=(red_a,)) if self.cr_ffn_1 is not None else None
        mlp_a = (self.fc2_latent(self.latent_norm(red_a)) + shared_a) if self.is_moe else red_a

        if self.defer_reduce_b and p_ffn_b is not None:
            prev_b_state = (res_b, post_b, comb_b, self.is_moe, self, shared_b)
            return (mlp_a, res_a, post_a, comb_a, p_ffn_b, prev_b_state), records, pre_res

        if self.post is not None:
            out_res_a = self.post(mlp_a, res_a, post_a, comb_a)
            red_b = self.cr_ffn_1.finish(p_ffn_b, out_res_a) if p_ffn_b is not None else ffn_r1(part_b)
            mlp_b = (self.fc2_latent(self.latent_norm(red_b)) + shared_b) if self.is_moe else red_b
            out_res_b = self.post(mlp_b, res_b, post_b, comb_b)
            return (torch.cat((out_res_a, out_res_b), dim=0),), records, pre_res

        red_b = self.cr_ffn_1.finish(p_ffn_b, mlp_a) if p_ffn_b is not None else ffn_r1(part_b)
        mlp_b = (self.fc2_latent(self.latent_norm(red_b)) + shared_b) if self.is_moe else red_b
        return (
            torch.cat((mlp_a, mlp_b), dim=0),
            torch.cat((res_a, res_b), dim=0),
            torch.cat((post_a, post_b), dim=0),
            torch.cat((comb_a, comb_b), dim=0),
        ), records, pre_res


# ---------------------------------------------------------------------------
# Prepared Full Target Decoder (43 layers + mHC head + norm + lm_head + MultiMax)
# ---------------------------------------------------------------------------

class PreparedDecoder:
    def __init__(
        self,
        weights,
        tables,
        scratch: Scratch,
        metadata,
        pool,
        *,
        mode,
        runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name,
        workspace,
        mla_compute_mode="auto",
        scoring=None,
        vocabulary_parallel=False,
        max_cache_pages=None,
        examples=None,
    ):
        del tables
        if scratch.local_logits.dtype != torch.float32 or scratch.logits.dtype != torch.float32:
            raise ValueError("ERNIE 5.5 target vocabulary logits require FP32 storage")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        self.feedback = (
            PreparedFeedbackCapture(metadata, len(self.aux_layers) * 4096, compact_prefill=mode == "prefill")
            if self.aux_layers
            else None
        )
        feedback_rows = (
            min(128, metadata.token_ids.numel()) if mode == "prefill" else metadata.token_ids.numel()
        )
        self.auxiliary = (
            None
            if self.feedback is None
            else torch.empty(
                (feedback_rows, len(self.aux_layers) * 4096),
                dtype=torch.bfloat16,
                device=scratch.embedding.device,
            )
        )
        self.embedding = PreparedEmbedding(
            weights.embedding,
            metadata.token_ids,
            scratch.embedding,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        ).functional(session=session, dynamic_inputs=True)
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        tokens = metadata.positions.numel()
        self.indices = AttentionIndices(metadata, mode=mode, tokens=tokens, include_dsa=True)

        can_pipe = (
            isinstance(workspace, ScratchArena)
            and workspace.overlap
            and mode == "prefill"
            and _is_dma_runtime(runtime)
            and tokens >= 1024
            and tokens % 2 == 0
        )

        self.layers = []
        carry = False
        prev_defer_b = False
        n_layers = len(weights.layers)
        for index, layer in enumerate(weights.layers):
            # Keep fused post_pre carry & defer_reduce_b through aux_layers (40, 41):
            # layers 41 & 42 return their incoming materialized 4-stream post-mHC residual
            # directly from attention_hc(carry=True), avoiding slow MHCPostPrePartialKernel!
            defer_post = index + 1 < n_layers
            defer_reduce_b = can_pipe and defer_post and (index + 1 < n_layers)
            return_pre_residual = (index - 1) in self.aux_layers
            prepared_layer = PreparedLayer(
                index,
                layer,
                scratch,
                metadata,
                pool,
                self.rotary,
                shard=weights.shard,
                mode=mode,
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=f"{name}.layer{index}",
                workspace=workspace,
                carry=carry,
                defer_post=defer_post,
                indices=self.indices,
                max_cache_pages=max_cache_pages,
                mla_compute_mode=mla_compute_mode,
                examples=examples,
                pipelined=can_pipe,
                prev_deferred_reduce=prev_defer_b,
                defer_reduce_b=defer_reduce_b,
                return_pre_residual=return_pre_residual,
            )
            if prev_defer_b and self.layers[-1].cr_ffn_1 is not None:
                prev_layer = self.layers[-1]
                prev_plan = prev_layer.cr_ffn_1.reduce.plan
                if getattr(prepared_layer.attention_hc, "plan", None) is not None:
                    workspace.register_parallel(prev_plan, prepared_layer.attention_hc.plan)
                if prepared_layer.full_half_hca:
                    for plan in prepared_layer.attention.plans():
                        workspace.register_parallel(prev_plan, plan)
            if prepared_layer.full_half_hca and getattr(prepared_layer.attention, "cr_vha_0", None) is not None:
                vha0_plan = prepared_layer.attention.cr_vha_0.reduce.plan
                if getattr(prepared_layer.attention_hc, "plan", None) is not None:
                    workspace.register_parallel(vha0_plan, prepared_layer.attention_hc.plan)
                if prev_defer_b and self.layers[-1].is_moe:
                    prev_layer = self.layers[-1]
                    for plan in (
                        prev_layer.latent_norm.plan,
                        prev_layer.fc2_latent.plan,
                        getattr(prev_layer.fc2_latent, "quantizer_plan", None),
                    ):
                        if plan is not None:
                            workspace.register_parallel(vha0_plan, plan)
            self.layers.append(prepared_layer)
            carry = defer_post
            prev_defer_b = defer_reduce_b

        text = weights.config.text
        self.compact_head = scratch.hidden.shape[0] > metadata.token_ids.numel()
        head_rows = metadata.token_ids.numel() if self.compact_head else scratch.hidden.shape[0]
        self.collapse = PreparedMHCHead(
            scratch.head_residual[:head_rows],
            weights.head_mix,
            scratch.collapsed[:head_rows],
            rms_eps=text["rms_norm_eps"],
            hc_eps=text["hc_eps"],
            session=session,
            name=name + ".collapse",
        ).functional()
        self.norm = PreparedRMSNorm(
            scratch.collapsed[:head_rows],
            weights.norm,
            scratch.hidden[:head_rows],
            eps=text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        ).functional()
        self.scoring = (
            None
            if scoring is None
            else scoring.prepare(
                scratch.embedding,
                weights.head,
                vocabulary=VOCABULARY,
                session=session,
                coordinator=coordinator,
                name=name + ".prompt",
                logits_dtype=torch.float32,
                multimax=weights.multimax,
            )
        )
        self.score_collapse = (
            None
            if self.scoring is None
            else PreparedMHCHead(
                scratch.residual,
                weights.head_mix,
                scratch.embedding,
                rms_eps=text["rms_norm_eps"],
                hc_eps=text["hc_eps"],
                session=session,
                name=name + ".prompt_collapse",
            ).functional()
        )
        self.score_norm = (
            None
            if self.scoring is None
            else PreparedRMSNorm(
                scratch.embedding,
                weights.norm,
                scratch.embedding,
                eps=text["rms_norm_eps"],
                session=session,
                name=name + ".prompt_norm",
            ).functional()
        )
        head_output = scratch.logits
        if vocabulary_parallel and self.compact_head:
            head_output = scratch.logits.view(-1)[: scratch.local_logits.numel()].view_as(
                scratch.local_logits
            )
        self.head = PreparedVocabularyProjection(
            scratch.hidden[:head_rows],
            weights.head,
            scratch.local_logits[:head_rows],
            head_output,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
            row_indices=metadata.sample_indices if self.compact_head else None,
            gather=not vocabulary_parallel,
            workspace=workspace,
        )
        self.local_logits = self.head.output if vocabulary_parallel else scratch.local_logits
        self.head.functional(session=session, dynamic_indices=True)
        self.indices.functional()
        register_finalizer(session, self.release_outputs)

    def release_outputs(self):
        self.scratch = self.metadata = self.auxiliary = self.local_logits = None

    def __call__(self, metadata):
        embedding = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        residual = embedding[:, None, :].expand(-1, 4, -1).contiguous()
        incoming = (residual,)
        indices = self.indices(metadata)
        feedback = None if self.feedback is None else self.feedback.begin(metadata)
        records = []
        prev_cr = None
        for index, layer in enumerate(self.layers):
            incoming, record, pre_res = layer(incoming, metadata, indices, prev_cr=prev_cr)
            records.append(record)
            prev_cr = layer.cr_ffn_1 if layer.defer_reduce_b else None
            if pre_res is not None:
                col = self.aux_layers.index(index - 1) * 4096
                feedback = self.feedback(feedback, pre_res, col)
            if index in self.aux_layers and not layer.defer_post:
                # Final layer materialized its 4-stream post-mHC residual in incoming[0].
                col = self.aux_layers.index(index) * 4096
                feedback = self.feedback(feedback, incoming[0], col)
        (residual,) = incoming
        samples = metadata.sample_indices
        if self.scoring is not None:
            all_hidden = self.score_norm(self.score_collapse(residual))
            self.scoring(all_hidden)
            hidden = all_hidden[samples]
            logits = self.head(hidden)
        else:
            head_res = residual if self.compact_head else residual[samples]
            hidden = self.norm(self.collapse(head_res))
            logits = self.head(hidden, row_indices=samples if self.compact_head else None)
            if self.compact_head:
                hidden = hidden[samples]
        apply_multimax_inplace(logits, self.weights.multimax)
        auxiliary = None if feedback is None else feedback[0]
        return hidden, logits, auxiliary, tuple(records)

    def commit(self, counts, records, metadata):
        for layer, record in zip(self.layers, records, strict=True):
            layer.attention.commit(counts, record, metadata)
