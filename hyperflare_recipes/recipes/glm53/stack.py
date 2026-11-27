# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete GLM text forwards with invocation-local tensors and sharded vocabulary weights."""

from dataclasses import dataclass, replace

import torch
from b12x.comm import pcie
from flashinfer.triton.norm import rms_norm

from ignition.components.collectives import (
    PreparedAllReduce,
    PreparedGatherShards,
    PreparedReduceScatter,
    _roce,
    _row_shard_capacity_bytes,
    row_shard_surfaces,
)
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.scratch import ScratchArena
from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.mhc import MHCScratch
from hyperflare_recipes.components.mlp import MLPScratch
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.recurrent import DeltaMetadata
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from hyperflare_recipes.components.weight_gather import (
    AttentionWeightGatherer, full_kda_template, full_mla_template,
)

from .decoder import LayerScratch, PreparedGlmLayer
from .kda import (
    KDAScratch, ROW_SHARD_SCAN_SHAPE, row_shard_halo_shape, row_shard_sinit_shape,
)
from .mla import MLAScratch
from .moe import MoeScratch
from .state import DecoderMetadata, DecoderState
from .weights import ModelWeights, MoEWeights


DMA_PREFILL_TOKENS = 1024
PIPELINED_PREFILL_TOKENS = 2048
ROW_SHARD_ATTENTION_MIN_TOKENS = 4096
ROW_SHARD_ATTENTION_MIN_TOKENS_PCIE = 12288
ROW_SHARD_ATTENTION_MIN_TOKENS_ROCE_BF16 = 8192
QUARTER_MLP_PREFILL_TOKENS = 10240



@dataclass(frozen=True)
class RowShardExchangeScratch:
    """Shared declaration examples reused across row-owned KDA/MLA layers and prefill capacities."""

    sinit_sh: torch.Tensor
    sinit_full: torch.Tensor
    halo_sh: torch.Tensor
    halo_full: torch.Tensor
    scan_sh: torch.Tensor
    scan_full: torch.Tensor
    cache_sh: torch.Tensor
    cache_full: torch.Tensor
    kda_scratch: KDAScratch
    kda_seg_qkv: torch.Tensor
    kda_seg_beta: torch.Tensor
    kda_segment_pool: torch.Tensor
    kda_local_conv: torch.Tensor
    kda_rank_index: torch.Tensor
    kda_segment_eye: torch.Tensor
    attn_norm: torch.Tensor
    mla_scratch: MLAScratch | None = None

    @classmethod
    def allocate(cls, max_prefill_tokens, shard: TensorParallelShard, device, *,
                 blocks=0, context=None, shared_slab=None, single_segment=False):
        half = max(max_prefill_tokens // 2, 1)
        rows = max(half // shard.world_size, 1)
        seg_tokens = rows * 2
        mla_rows = seg_tokens if single_segment else rows
        cache_full_rows = max_prefill_tokens if single_segment else half
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        mla_scratch = None
        if blocks > 0 and context is not None and context.records is not None:
            mla_scratch = MLAScratch.allocate_row_shard(
                max_prefill_tokens, mla_rows, blocks, device,
                context_ranks=context.shards.world, remote=context.remote,
                gather_context=True, shared_slab=shared_slab)
        w = shard.world_size
        sinit_shape = row_shard_sinit_shape(w)
        halo_shape = row_shard_halo_shape(w)
        return cls(
            sinit_sh=zeros(*sinit_shape),
            sinit_full=zeros(sinit_shape[0] * w, sinit_shape[1]),
            halo_sh=zeros(*halo_shape),
            halo_full=zeros(halo_shape[0] * w, halo_shape[1]),
            scan_sh=zeros(*ROW_SHARD_SCAN_SHAPE),
            scan_full=zeros(ROW_SHARD_SCAN_SHAPE[0] * w, ROW_SHARD_SCAN_SHAPE[1]),
            cache_sh=zeros(mla_rows, 768),
            cache_full=zeros(cache_full_rows, 768),
            kda_scratch=KDAScratch.allocate(seg_tokens, 64, device),
            kda_seg_qkv=zeros(seg_tokens, 24768),
            kda_seg_beta=zeros(seg_tokens, 64),
            kda_segment_pool=zeros(6, 64, 128, 128, dtype=torch.float32),
            kda_local_conv=zeros(1, 24576, 3),
            kda_rank_index=persistent_copy(torch.tensor([shard.rank], dtype=torch.int64, device=device)),
            kda_segment_eye=persistent_copy(torch.eye(128, dtype=torch.float32, device=device).unsqueeze(0)),
            attn_norm=zeros(rows, 4096),
            mla_scratch=mla_scratch,
        )


@dataclass(frozen=True)
class DecoderScratch:
    embedding: torch.Tensor
    residual: torch.Tensor
    selected_residual: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    attention_hc: MHCScratch
    mlp_hc: MHCScratch
    kda: KDAScratch
    mla: MLAScratch
    dense: MLPScratch
    moe: MoeScratch
    attention_output: torch.Tensor
    mlp_output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, mode, device, *, shard: TensorParallelShard, sample_rows, context_ranks=1,
                 remote=False, gather_context=False):
        """Allocate declaration examples for each operator's layouts and tuning."""
        if type(sample_rows) is not int or not 0 < sample_rows <= tokens:
            raise ValueError("GLM sampling capacity must be positive and fit the token capacity")
        def buffer(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        layers = (KDAScratch.allocate(tokens, 64 // shard.world_size, device),
            MLAScratch.allocate(tokens, 64 // shard.world_size, blocks, mode, device, context_ranks=context_ranks,
                                remote=remote, gather_context=gather_context),
            MLPScratch.allocate(tokens, 4096, 12288 // shard.world_size, device),
            MoeScratch.allocate(tokens, shard.world_size, device))
        return cls(buffer(tokens, 4096), buffer(tokens, 4, 4096), buffer(sample_rows, 4, 4096),
            buffer(sample_rows, 4096), buffer(sample_rows, 154880 // shard.world_size), buffer(sample_rows, 154880),
            MHCScratch.allocate(tokens, 4096, device), MHCScratch.allocate(tokens, 4096, device),
            *layers, buffer(tokens, 4096), buffer(tokens, 4096))

    def rows(self, tokens, mode, *, sample_rows):
        """Select declaration examples with the graph variant's shapes and strides."""
        if type(tokens) is not int or not 0 < tokens <= self.embedding.shape[0]:
            raise ValueError("GLM decoder rows exceed scratch capacity")
        if type(sample_rows) is not int or not 0 < sample_rows <= min(tokens, self.hidden.shape[0]):
            raise ValueError("GLM sample rows exceed scratch capacity")
        return DecoderScratch(self.embedding[:tokens], self.residual[:tokens],
            self.selected_residual[:sample_rows], self.hidden[:sample_rows],
            self.local_logits[:sample_rows], self.logits[:sample_rows],
            self.attention_hc.rows(tokens), self.mlp_hc.rows(tokens), self.kda.rows(tokens),
            self.mla.rows(tokens, mode), self.dense.rows(tokens), self.moe.rows(tokens),
            self.attention_output[:tokens], self.mlp_output[:tokens])

    def layer(self, kind, *, sparse):
        if kind not in ("linear_attention", "indexed_attention"):
            raise ValueError(f"unsupported canonical GLM attention type: {kind}")
        return LayerScratch(self.attention_hc, self.mlp_hc,
            self.kda if kind == "linear_attention" else self.mla, self.attention_output,
            self.moe if sparse else self.dense, self.mlp_output)


def _prefill_shard_rows(enabled, mode, shard, runtime, scratch):
    tokens = scratch.embedding.shape[0]
    # Shard rows stay 128-aligned for the mHC and RMSNorm row tiles.
    if (not enabled or mode != "prefill" or shard.world_size not in (2, 4) or tokens < DMA_PREFILL_TOKENS
            or tokens % (128 * shard.world_size)
            or not (isinstance(runtime, pcie.DmaAllReduce) or _roce(runtime))
            or not row_shard_surfaces(runtime)):
        return None
    rows = tokens // shard.world_size
    if (not PreparedReduceScatter.fits(runtime, scratch.kda.output, scratch.attention_output[:rows])
            or not PreparedGatherShards.fits(runtime, scratch.attention_hc.normalized[:rows], scratch.embedding)):
        return None
    # All ranks bind the same compiled forward with their own row indices.
    first, last = shard.partition(tokens)
    return persistent_copy(torch.arange(first, last, device=scratch.embedding.device))


def _prefill_pipelined(enabled, mode, shard, runtime, workspace, scratch, *, quarter_mlp_eligible=False):
    tokens = scratch.embedding.shape[0]
    if (not enabled or mode != "prefill" or shard.world_size not in (2, 4) or tokens < PIPELINED_PREFILL_TOKENS
            or tokens % (2 * 128 * shard.world_size) or not isinstance(runtime, pcie.DmaAllReduce)
            or not row_shard_surfaces(runtime)
            or not isinstance(workspace, ScratchArena) or not workspace.overlap):
        return False
    half = tokens // 2
    rows = half // shard.world_size
    if (PreparedReduceScatter.fits(runtime, scratch.kda.output[:half], scratch.attention_output[:rows])
            and PreparedGatherShards.fits(runtime, scratch.attention_hc.normalized[:rows], scratch.embedding[:half])):
        return True
    if not quarter_mlp_eligible:
        return False
    quarter = tokens // 4
    q_rows = quarter // shard.world_size
    cap = _row_shard_capacity_bytes(runtime)
    scan_bytes = shard.world_size * ROW_SHARD_SCAN_SHAPE[0] * ROW_SHARD_SCAN_SHAPE[1] * 2
    cache_bytes = half * 768 * 2
    return (
        PreparedReduceScatter.fits(runtime, scratch.kda.output[:quarter], scratch.attention_output[:q_rows])
        and PreparedGatherShards.fits(runtime, scratch.attention_hc.normalized[:q_rows], scratch.embedding[:quarter])
        and scan_bytes <= cap
        and cache_bytes <= cap
    )


def _split_recurrent_metadata(metadata: DeltaMetadata, half: int) -> tuple[DeltaMetadata, DeltaMetadata]:
    """Split single-sequence prefill DeltaMetadata across microbatches [0, half) and [half, 2*half)."""
    live = metadata.num_tokens
    active = metadata.num_seqs > 0
    # Build [0, len] via pointwise scaling instead of torch.cat((zero, len)) so Inductor's
    # ConcatKernel cannot alias len_a/len_b inside query_start_loc and prematurely reuse its pool slot.
    qsl_scale = torch.arange(2, dtype=torch.int32, device=live.device)

    len_a = live.clamp(0, half).to(torch.int32)
    has_a = active & (len_a > 0)
    ckpt_a = (metadata.checkpoint_offsets > 0) & (metadata.checkpoint_offsets <= half)
    meta_a = DeltaMetadata(
        query_start_loc=qsl_scale * len_a,
        convolution_slots=torch.where(has_a, metadata.convolution_slots, -1),
        fresh=metadata.fresh & has_a,
        accepted_tokens=metadata.accepted_tokens,
        state_indices=metadata.state_indices,
        initial_state_indices=metadata.initial_state_indices,
        final_state_indices=metadata.final_state_indices,
        checkpoint_state_indices=torch.where(ckpt_a, metadata.checkpoint_state_indices, 0),
        checkpoint_offsets=torch.where(ckpt_a, metadata.checkpoint_offsets, 0),
        num_seqs=has_a.to(torch.int32),
        num_tokens=len_a,
    )

    len_b = (live - half).clamp(0, half).to(torch.int32)
    has_b = active & (len_b > 0)
    ckpt_b = metadata.checkpoint_offsets > half
    meta_b = DeltaMetadata(
        query_start_loc=qsl_scale * len_b,
        convolution_slots=torch.where(has_b, metadata.convolution_slots, -1),
        fresh=torch.zeros_like(metadata.fresh),
        accepted_tokens=metadata.accepted_tokens,
        state_indices=metadata.state_indices,
        initial_state_indices=torch.where(has_b, metadata.final_state_indices, 0),
        final_state_indices=torch.where(has_b, metadata.final_state_indices, 0),
        checkpoint_state_indices=torch.where(ckpt_b, metadata.checkpoint_state_indices, 0),
        checkpoint_offsets=torch.where(ckpt_b, metadata.checkpoint_offsets - half, 0),
        num_seqs=has_b.to(torch.int32),
        num_tokens=len_b,
    )
    return meta_a, meta_b


def _weight_format_key(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", value.dtype, tuple(value.shape))
    if hasattr(value, "parts") and hasattr(value, "widths"):
        return ("grouped", tuple(_weight_format_key(p) for p in value.parts), tuple(value.widths))
    if hasattr(value, "weight") and hasattr(value.weight, "values"):
        return (type(value).__name__, value.weight.values.dtype, tuple(value.weight.values.shape))
    if hasattr(value, "values") and hasattr(value, "scales"):
        return (type(value).__name__, value.values.dtype, tuple(value.values.shape),
                value.scales.dtype, tuple(value.scales.shape))
    return (type(value).__name__,)


def _attention_format_key(attn: dict[str, object], keys: tuple[str, ...]):
    return tuple((k, _weight_format_key(attn[k])) for k in keys if k in attn)


class PreparedDecoder:
    """Prepare all 45 layers into a serial forward owned by one TP rank.

    Layers carry MLP output and mixing coefficients into the following fused
    mHC post/pre operation. Inductor manages invocation-local intermediates.
    Each layer's cache and recurrent state have separate owners. Vocabulary
    projection shards consume the selected normalized rows on every rank.
    This object executes in Python only during preparation and CUDA graph capture.
    """

    def __init__(self, weights: ModelWeights, scratch: DecoderScratch, state: DecoderState,
                 metadata: DecoderMetadata, *, mode, runtime, vocabulary_runtime, coordinator, session, name,
                 workspace, progress=None, feedback=None, max_cache_pages=None, context=None, kda_records=None,
                 scoring=None, vision=None, row_shard_prefill=False, row_shard_attention=False, row_shard_exchange=None,
                 boundary_runtime=None, sharded_feedback=False):
        language = weights.tree.model.language_model
        if len(weights.layers) != 45 or len(language.layers) != 45 or len(state.layers) != 45:
            raise ValueError("GLM text execution requires all 45 decoder layers")
        if metadata.token_ids.shape != scratch.embedding.shape[:1]:
            raise ValueError("GLM token metadata and scratch capacities differ")
        if metadata.sample_indices.shape != scratch.hidden.shape[:1]:
            raise ValueError("GLM sample indices and hidden-output capacities differ")
        self.weights, self.scratch, self.state, self.metadata = weights, scratch, state, metadata
        self.tokens, self.blocks, self.sample_rows = (scratch.embedding.shape[0],
            scratch.mla.index.page_table.shape[1], scratch.hidden.shape[0])
        self.mode = mode
        self.device, self.context, self.kda_records = metadata.token_ids.device, context, kda_records
        self.full_feedback = feedback is not None
        self.vision = vision if mode == "prefill" else None
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=154880, session=session, coordinator=coordinator, name=name + ".prompt")
        self.feedback = feedback
        self.embedding = PreparedEmbedding(language.embed_tokens.weight, metadata.token_ids, scratch.embedding,
            num_rows=metadata.recurrent.num_tokens, session=session, name=f"{name}.embedding")

        has_attn_weights = (
            weights.layers[0].attention is not None
            and isinstance(weights.layers[0].attention, dict)
            and "input_proj" in weights.layers[0].attention
        )
        if _roce(runtime):
            # BF16 attention weights are 2x larger per layer than quantized FP8 weights, so on RoCE
            # enable row-sharded attention at 8K for BF16 checkpoints and 4K for quantized attention
            # (also avoiding non-row-shard tail-capacity Graph capture pool spikes).
            has_bf16_attn = has_attn_weights and any(
                isinstance(lw.attention.get("input_proj"), torch.Tensor)
                for m, lw in zip(language.layers, weights.layers, strict=True)
                if m.block_type == "linear_attention" and isinstance(lw.attention, dict)
            )
            attn_min_tokens = (
                ROW_SHARD_ATTENTION_MIN_TOKENS_ROCE_BF16
                if has_bf16_attn
                else ROW_SHARD_ATTENTION_MIN_TOKENS
            )
        else:
            attn_min_tokens = ROW_SHARD_ATTENTION_MIN_TOKENS_PCIE
        attn_eligible = (
            row_shard_attention
            and has_attn_weights
            and self.tokens >= attn_min_tokens
        )
        quarter_mlp_eligible = (
            attn_eligible
            and context is not None and context.records is not None
            and any(m.block_type == "indexed_attention" for m in language.layers)
            and any(m.block_type == "linear_attention" for m in language.layers)
            and self.tokens >= QUARTER_MLP_PREFILL_TOKENS
            and self.tokens % (4 * 128 * weights.shard.world_size) == 0
            and scoring is None and self.sample_rows == 1
            and boundary_runtime is not None
            and (not self.full_feedback or sharded_feedback)
        )
        self.pipelined = _prefill_pipelined(
            row_shard_prefill and scoring is None, mode, weights.shard, runtime, workspace, scratch,
            quarter_mlp_eligible=quarter_mlp_eligible)
        self.half_tokens = self.tokens // 2 if self.pipelined else 0
        self.shard_rows = self.shard_rows_a = self.shard_rows_b = None
        self.final_gather = self.final_gather_a = self.final_gather_b = None
        self.final_boundary_reduce = None
        self.sharded_final_norm = False
        self.rank_index = None
        gathered_input = None
        full_attention_input = None
        mla_row_shard_scratch = None
        recurrent_meta = metadata.recurrent
        use_row_shard_mla = False
        self.contiguous_shards = False
        self.mlp_quarters = False
        gathered_mlp = None
        final_norm_source = None
        self.weight_gather = None
        if self.pipelined:
            half = self.half_tokens
            rows = half // weights.shard.world_size
            gathered_input = scratch.embedding[:half]
            full_attention_input = scratch.embedding
            use_row_shard_mla = (
                attn_eligible
                and context is not None and context.records is not None
                and any(m.block_type == "indexed_attention" for m in language.layers)
            )
            exchange = row_shard_exchange
            if exchange is None and attn_eligible:
                exchange = RowShardExchangeScratch.allocate(self.tokens, weights.shard, self.device)
            use_row_shard_kda = (
                attn_eligible and exchange is not None
                and any(m.block_type == "linear_attention" for m in language.layers)
                and PreparedGatherShards.fits(runtime, exchange.scan_sh, exchange.scan_full)
            )
            self.contiguous_shards = bool(use_row_shard_kda)
            self.mlp_quarters = (
                use_row_shard_mla and use_row_shard_kda
                and self.tokens >= QUARTER_MLP_PREFILL_TOKENS
                and self.tokens % (4 * 128 * weights.shard.world_size) == 0
            )
            if self.contiguous_shards:
                base = weights.shard.rank * (rows * 2)
                self.shard_rows_a = persistent_copy(torch.arange(base, base + rows, device=self.device))
                self.shard_rows_b = persistent_copy(torch.arange(base + rows, base + rows * 2, device=self.device))
            else:
                first, last = weights.shard.partition(half)
                self.shard_rows_a = persistent_copy(torch.arange(first, last, device=self.device))
                self.shard_rows_b = persistent_copy(torch.arange(half + first, half + last, device=self.device))
            if use_row_shard_mla:
                if exchange is not None and exchange.mla_scratch is not None:
                    mla_row_shard_scratch = exchange.mla_scratch.rows_row_shard(self.tokens, rows)
                else:
                    mla_row_shard_scratch = MLAScratch.allocate_row_shard(
                        self.tokens, rows, self.blocks, self.device,
                        context_ranks=context.shards.world, remote=context.remote, gather_context=True,
                        shared_slab=scratch.mla.slab)
            kda_rank_index = kda_segment_eye = kda_segment_pool = kda_local_conv = None
            if use_row_shard_kda:
                kda_row_shard_scratch = exchange.kda_scratch.rows(rows * 2)
                kda_rank_index = exchange.kda_rank_index
                kda_segment_eye = exchange.kda_segment_eye
                kda_segment_pool = exchange.kda_segment_pool
                kda_local_conv = exchange.kda_local_conv
            final_norm_source = scratch.mlp_hc.normalized[:rows]
            mlp_tokens = half // 2 if self.mlp_quarters else half
            hc_rows = rows // 2 if self.mlp_quarters else rows
            gathered_mlp = scratch.embedding[:mlp_tokens] if self.mlp_quarters else None
            half_kda = scratch.kda.rows(half)
            mlp_dense = scratch.dense.rows(mlp_tokens)
            mlp_moe = scratch.moe.rows(mlp_tokens)
            scratch = replace(scratch,
                residual=scratch.residual[:hc_rows],
                attention_hc=scratch.attention_hc.rows(hc_rows),
                mlp_hc=scratch.mlp_hc.rows(hc_rows),
                kda=kda_row_shard_scratch if use_row_shard_kda else half_kda,
                mla=mla_row_shard_scratch if use_row_shard_mla else scratch.mla,
                dense=mlp_dense,
                moe=mlp_moe,
                attention_output=scratch.attention_output[:hc_rows],
                mlp_output=scratch.mlp_output[:hc_rows])
            recurrent_meta = DeltaMetadata.allocate(1, self.device)
        else:
            use_row_shard_mla = use_row_shard_kda = False
            exchange = kda_rank_index = kda_segment_eye = kda_segment_pool = kda_local_conv = None
            self.shard_rows = _prefill_shard_rows(
                row_shard_prefill and scoring is None, mode, weights.shard, runtime, scratch)
            if self.shard_rows is not None:
                rows = self.shard_rows.numel()
                # After embedding selection, all serial gathers can reuse its full-row declaration buffer.
                gathered_input = scratch.embedding
                use_row_shard_mla = (
                    attn_eligible
                    and context is not None and context.records is not None
                    and any(m.block_type == "indexed_attention" for m in language.layers)
                )
                exchange = row_shard_exchange
                if exchange is None and attn_eligible:
                    exchange = RowShardExchangeScratch.allocate(
                        self.tokens, weights.shard, self.device, single_segment=True)
                use_row_shard_kda = (
                    attn_eligible and exchange is not None
                    and any(m.block_type == "linear_attention" for m in language.layers)
                    and PreparedGatherShards.fits(runtime, exchange.scan_sh, exchange.scan_full)
                )
                self.contiguous_shards = bool(use_row_shard_kda)
                if use_row_shard_mla:
                    if exchange is not None and exchange.mla_scratch is not None:
                        mla_row_shard_scratch = exchange.mla_scratch.rows_row_shard(self.tokens, rows)
                    else:
                        mla_row_shard_scratch = MLAScratch.allocate_row_shard(
                            self.tokens, rows, self.blocks, self.device,
                            context_ranks=context.shards.world, remote=context.remote, gather_context=True,
                            shared_slab=scratch.mla.slab)
                if use_row_shard_kda:
                    kda_row_shard_scratch = exchange.kda_scratch.rows(rows)
                    kda_rank_index = exchange.kda_rank_index
                    kda_segment_eye = exchange.kda_segment_eye
                    kda_segment_pool = exchange.kda_segment_pool
                    kda_local_conv = exchange.kda_local_conv
                scratch = replace(scratch, residual=scratch.residual[:rows],
                    attention_hc=scratch.attention_hc.rows(rows), mlp_hc=scratch.mlp_hc.rows(rows),
                    kda=kda_row_shard_scratch if use_row_shard_kda else scratch.kda,
                    mla=mla_row_shard_scratch if use_row_shard_mla else scratch.mla,
                    attention_output=scratch.attention_output[:rows], mlp_output=scratch.mlp_output[:rows])

        kda_templates: dict[object, dict[str, object]] = {}
        mla_templates: dict[object, dict[str, object]] = {}
        if use_row_shard_kda or use_row_shard_mla:
            # Share one transient full-head weight template per distinct layer weight format for
            # tuning only; projections drop their references at finalization so zero replicated
            # weights persist, while mixed-precision checkpoints still match per-layer formats.
            self.weight_gather = AttentionWeightGatherer(
                [lw.attention for lw in weights.layers],
                [m.block_type for m in language.layers],
                world_size=weights.shard.world_size,
                runtime=runtime, coordinator=coordinator, session=session,
                name=name, workspace=workspace, pipelined=self.pipelined, device=self.device,
            )
        self.layers = []
        for index, (module, layer_weights, layer_state) in enumerate(zip(
                language.layers, weights.layers, state.layers, strict=True)):
            indexed = module.block_type == "indexed_attention"
            layer_rs_attn = (use_row_shard_mla and indexed) or (use_row_shard_kda and not indexed)
            layer_template = None
            if layer_rs_attn:
                if indexed:
                    fmt = _attention_format_key(layer_weights.attention, ("q_b_proj", "kv_b_proj", "o_proj"))
                    layer_template = mla_templates.get(fmt)
                    if layer_template is None:
                        layer_template = full_mla_template(
                            layer_weights.attention, world_size=weights.shard.world_size, device=self.device)
                        mla_templates[fmt] = layer_template
                else:
                    fmt = _attention_format_key(layer_weights.attention, ("input_proj", "o_proj"))
                    layer_template = kda_templates.get(fmt)
                    if layer_template is None:
                        layer_template = full_kda_template(
                            layer_weights.attention, world_size=weights.shard.world_size, device=self.device)
                        kda_templates[fmt] = layer_template
            layer = PreparedGlmLayer(module, layer_weights.attention, layer_weights.mlp, layer_weights.residual,
                scratch.residual if index == 0 else scratch.mlp_output, scratch.residual,
                scratch.layer(module.block_type, sparse=isinstance(layer_weights.mlp, MoEWeights)), layer_state,
                metadata.indexed if indexed else recurrent_meta, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
                carry=None if index == 0 else scratch.mlp_hc, defer_post=index < len(language.layers) - 1,
                gathered_input=gathered_input,
                max_cache_pages=max_cache_pages, context=context,
                kda_records=None if kda_records is None or indexed else kda_records.layer(index),
                pipelined=self.pipelined,
                row_shard_attention=layer_rs_attn,
                replicated_attention=layer_template,
                runtime_weights=layer_rs_attn,
                full_attention_input=full_attention_input if (self.pipelined and indexed and not use_row_shard_mla) else None,
                rank_index=kda_rank_index if (use_row_shard_kda and not indexed) else None,
                segment_eye=kda_segment_eye if (use_row_shard_kda and not indexed) else None,
                kda_segment_pool=kda_segment_pool if (use_row_shard_kda and not indexed) else None,
                kda_local_conv=kda_local_conv if (use_row_shard_kda and not indexed) else None,
                row_shard_buffers=exchange if layer_rs_attn else None,
                contiguous_shards=self.contiguous_shards,
                mlp_quarters=self.mlp_quarters,
                gathered_mlp=gathered_mlp)
            self.layers.append(layer)
            if progress is not None:
                progress(f"declared {mode} layer {index + 1}/45")
        if self.weight_gather is not None and (self.pipelined or self.weight_gather.async_roce):
            from .decoder import _collect_plans

            for layer in self.layers:
                self.weight_gather.register_parallel(
                    workspace, _collect_plans(layer.attention, layer.mlp_hc, layer.mlp))

        if ((self.pipelined or self.shard_rows is not None) and self.contiguous_shards and self.scoring is None
                and self.sample_rows == 1 and boundary_runtime is not None
                and kda_rank_index is not None and (not self.full_feedback or sharded_feedback)):
            self.sharded_final_norm = True
            self.rank_index = kda_rank_index
            boundary_buf = torch.zeros((weights.shard.world_size + 1, 4096), dtype=torch.bfloat16, device=self.device)
            self.final_boundary_reduce = PreparedAllReduce(
                boundary_buf, torch.zeros_like(boundary_buf), runtime=boundary_runtime,
                coordinator=coordinator, session=session, name=f"{name}.final_boundary")
        elif self.pipelined:
            self.final_gather_a = PreparedGatherShards(
                final_norm_source, gathered_input, runtime=runtime, coordinator=coordinator,
                session=session, name=f"{name}.final_gather.0")
            self.final_gather_b = PreparedGatherShards(
                final_norm_source, gathered_input, runtime=runtime, coordinator=coordinator,
                session=session, name=f"{name}.final_gather.1")
        elif gathered_input is not None:
            self.final_gather = PreparedGatherShards(
                scratch.mlp_hc.normalized, gathered_input, runtime=runtime, coordinator=coordinator,
                session=session, name=f"{name}.final_gather")

        self.lm_head = PreparedVocabularyProjection(scratch.hidden, weights.head,
            scratch.local_logits, scratch.logits, runtime=vocabulary_runtime, coordinator=coordinator,
            session=session, name=f"{name}.lm_head", workspace=workspace)

    def functional(self, *, session):
        self.embedding.functional(session=session, dynamic_inputs=True)
        if self.weight_gather is not None:
            self.weight_gather.functional(session=session)
        for layer in self.layers:
            layer.release_outputs(session=session)
        if self.final_gather is not None:
            self.final_gather.functional(session=session)
        if self.final_gather_a is not None:
            self.final_gather_a.functional(session=session)
        if self.final_gather_b is not None:
            self.final_gather_b.functional(session=session)
        if self.final_boundary_reduce is not None:
            self.final_boundary_reduce.functional(session=session)
        self.lm_head.functional(session=session)
        self.scratch = self.metadata = self.feedback = self.kda_records = None
        return self

    def rebind_pages(self):
        """Rebind every sparse-attention layer to the continuation pool's current page pools."""
        for layer in self.layers:
            if hasattr(layer.attention, "rebind_pages"):
                layer.attention.rebind_pages()

    def __call__(self, metadata, *, records=None):
        language = self.weights.tree.model.language_model
        embedded = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.recurrent.num_tokens)
        if self.vision is not None:
            self.vision.inject(metadata, embedded)
        if self.pipelined:
            return self._call_pipelined(language, embedded, metadata, records=records)
        pos_sh = None
        if self.shard_rows is not None:
            embedded = torch.index_select(embedded, 0, self.shard_rows)
            if metadata.indexed is not None:
                pos_sh = torch.index_select(metadata.indexed.positions, 0, self.shard_rows)
        value, carry = embedded[:, None, :].expand(-1, 4, -1).contiguous(), None
        use_channel_order = self.weight_gather is not None
        channel_token = torch.empty(1, dtype=torch.int32, device=self.device) if use_channel_order else None
        if channel_token is not None:
            from .decoder import _order_barrier

            _order_barrier(channel_token, [embedded])
        use_async_roce = self.weight_gather is not None and self.weight_gather.async_roce
        pending_roce_w = None
        cur_roce_w = (
            self.weight_gather.gather_sync(0, channel_token=channel_token)
            if use_async_roce and (getattr(self.layers[0], "row_shard_kda", False) or getattr(self.layers[0], "row_shard_mla", False))
            else None
        )
        n_layers = len(self.layers)
        for index, layer in enumerate(self.layers):
            layer_records = None if records is None or index not in records.layers else records.layer(index)
            indexed = layer.module.block_type == "indexed_attention"
            attention_metadata = metadata.indexed if indexed else metadata.recurrent
            layer_needs_w = getattr(layer, "row_shard_kda", False) or getattr(layer, "row_shard_mla", False)
            if use_async_roce:
                if pending_roce_w is not None:
                    cur_roce_w = self.weight_gather.finish_input(pending_roce_w, dependencies=(value,))
                    pending_roce_w = None
                attn_w = cur_roce_w if layer_needs_w else None
                next_idx = (
                    index + 1
                    if (index + 1 < n_layers
                        and (getattr(self.layers[index + 1], "row_shard_kda", False) or getattr(self.layers[index + 1], "row_shard_mla", False)))
                    else None
                )
                res = layer(value, attention_metadata, carry, records=layer_records,
                            pos_sh=pos_sh if indexed else None, attn_weights=attn_w,
                            weight_gather=self.weight_gather, layer_index=index,
                            next_layer_index=next_idx, channel_token=channel_token)
                if len(res) == 3:
                    value, carry, pending_roce_w = res
                else:
                    value, carry = res
            else:
                attn_w = (self.weight_gather.gather_sync(index, channel_token=channel_token)
                          if self.weight_gather is not None and layer_needs_w else None)
                value, carry = layer(value, attention_metadata, carry, records=layer_records,
                                     pos_sh=pos_sh if indexed else None, attn_weights=attn_w,
                                     channel_token=channel_token)
        if channel_token is not None:
            from .decoder import _order_barrier

            _order_barrier(value, [channel_token])
        if self.sharded_final_norm:
            return self._finish_sharded_final_norm(language, value, metadata, records)
        auxiliary = None
        if self.final_gather is None and not self.full_feedback and self.scoring is None:
            selected = value[metadata.sample_indices]
            collapsed = language.hc_head(selected[:, None]).squeeze(1)
            hidden = torch.empty_like(collapsed)
            rms_norm(collapsed, language.norm.weight, hidden, language.norm.variance_epsilon,
                round_before_weight=True)
        else:
            collapsed = language.hc_head(value[:, None]).squeeze(1)
            normalized = torch.empty_like(collapsed)
            rms_norm(collapsed, language.norm.weight, normalized, language.norm.variance_epsilon,
                round_before_weight=True)
            if self.final_gather is not None:
                normalized = self.final_gather(normalized)
            auxiliary = normalized if self.full_feedback or self.scoring is not None else None
            hidden = normalized[metadata.sample_indices]
        if self.scoring is not None:
            self.scoring(auxiliary)
        logits = self.lm_head(hidden)
        if self.full_feedback and self.scoring is None:
            from .decoder import _order_barrier

            _order_barrier(auxiliary, [logits])
        return hidden, logits, auxiliary, records

    def _call_pipelined(self, language, embedded, metadata, *, records=None):
        emb_a = torch.index_select(embedded, 0, self.shard_rows_a)
        emb_b = torch.index_select(embedded, 0, self.shard_rows_b)
        if self.mlp_quarters:
            q_shard = emb_a.shape[0] // 2
            emb_0, emb_1 = emb_a.split(q_shard, dim=0)
            emb_2, emb_3 = emb_b.split(q_shard, dim=0)
            incoming = tuple(e[:, None, :].expand(-1, 4, -1).contiguous() for e in (emb_0, emb_1, emb_2, emb_3))
        else:
            incoming = (
                emb_a[:, None, :].expand(-1, 4, -1).contiguous(),
                emb_b[:, None, :].expand(-1, 4, -1).contiguous(),
            )
        meta_a, meta_b = _split_recurrent_metadata(metadata.recurrent, self.half_tokens)
        pos_a = pos_b = None
        if metadata.indexed is not None:
            pos_a = torch.index_select(metadata.indexed.positions, 0, self.shard_rows_a)
            pos_b = torch.index_select(metadata.indexed.positions, 0, self.shard_rows_b)
        from .decoder import _order_barrier

        channel_token = torch.empty(1, dtype=torch.int32, device=self.device)
        _order_barrier(channel_token, [embedded])
        pending_mlp_b = None
        pending_weights = None
        cur_weights = (
            self.weight_gather.gather_sync(0, channel_token=channel_token)
            if self.weight_gather is not None and (getattr(self.layers[0], "row_shard_kda", False) or getattr(self.layers[0], "row_shard_mla", False))
            else None
        )
        n_layers = len(self.layers)
        for index, layer in enumerate(self.layers):
            layer_records = None if records is None or index not in records.layers else records.layer(index)
            indexed = layer.module.block_type == "indexed_attention"
            cur_pending_o = None
            if pending_weights is not None:
                cur_weights = self.weight_gather.finish_input(pending_weights, dependencies=(incoming[0],))
                cur_pending_o = pending_weights
                pending_weights = None
            next_idx = (
                index + 1
                if (self.weight_gather is not None and index + 1 < n_layers
                    and (getattr(self.layers[index + 1], "row_shard_kda", False) or getattr(self.layers[index + 1], "row_shard_mla", False)))
                else None
            )
            result = layer.forward_pipelined(
                incoming,
                metadata.indexed if indexed else metadata.recurrent,
                meta_a=None if indexed else meta_a,
                meta_b=None if indexed else meta_b,
                pos_a=pos_a if indexed else None,
                pos_b=pos_b if indexed else None,
                pending_mlp_b=pending_mlp_b,
                records=layer_records,
                attn_weights=cur_weights if (getattr(layer, "row_shard_kda", False) or getattr(layer, "row_shard_mla", False)) else None,
                weight_gather=self.weight_gather,
                next_layer_index=next_idx,
                pending_cur_o=cur_pending_o,
                channel_token=channel_token,
            )
            if len(result) == 3:
                incoming, pending_mlp_b, pending_weights = result
            else:
                incoming, pending_mlp_b = result
        from .decoder import _order_barrier

        _order_barrier(incoming[0], [channel_token])
        if self.sharded_final_norm:
            return self._finish_sharded_final_norm(language, torch.cat(incoming, dim=0), metadata, records)
        if self.mlp_quarters:
            out_0, out_1, out_2, out_3 = incoming
            out_a = torch.cat((out_0, out_1), dim=0)
            out_b = torch.cat((out_2, out_3), dim=0)
        else:
            out_a, out_b = incoming
        collapsed_a = language.hc_head(out_a[:, None]).squeeze(1)
        norm_a_sh = torch.empty_like(collapsed_a)
        rms_norm(collapsed_a, language.norm.weight, norm_a_sh, language.norm.variance_epsilon,
            round_before_weight=True)
        collapsed_b = language.hc_head(out_b[:, None]).squeeze(1)
        norm_b_sh = torch.empty_like(collapsed_b)
        rms_norm(collapsed_b, language.norm.weight, norm_b_sh, language.norm.variance_epsilon,
            round_before_weight=True)
        norm_a = self.final_gather_a(norm_a_sh)
        norm_b = self.final_gather_b(norm_b_sh, dependencies=(norm_a,))
        if self.contiguous_shards:
            world = self.weights.shard.world_size
            rows = self.half_tokens // world
            normalized = torch.stack(
                (norm_a.view(world, rows, -1), norm_b.view(world, rows, -1)), dim=1
            ).reshape(self.tokens, -1)
        else:
            normalized = torch.cat((norm_a, norm_b), dim=0)
        auxiliary = normalized if self.full_feedback or self.scoring is not None else None
        hidden = normalized[metadata.sample_indices]
        if self.scoring is not None:
            self.scoring(auxiliary)
        logits = self.lm_head(hidden)
        if self.full_feedback and self.scoring is None:
            _order_barrier(auxiliary, [logits])
        return hidden, logits, auxiliary, records

    def _finish_sharded_final_norm(self, language, out_sh, metadata, records):
        collapsed = language.hc_head(out_sh[:, None]).squeeze(1)
        norm_sh = torch.empty_like(collapsed)
        rms_norm(collapsed, language.norm.weight, norm_sh, language.norm.variance_epsilon,
            round_before_weight=True)
        shard_rows = norm_sh.shape[0]
        idx = metadata.sample_indices[:1]
        base = self.rank_index * shard_rows
        in_shard = (idx >= base) & (idx < base + shard_rows)
        local_idx = (idx - base).clamp(0, shard_rows - 1)
        selected = torch.index_select(norm_sh, 0, local_idx)
        local_sample = torch.where(in_shard.view(1, 1), selected, torch.zeros_like(selected))
        world = self.weights.shard.world_size
        boundary = torch.zeros((world + 1, 4096), dtype=norm_sh.dtype, device=norm_sh.device)
        boundary.index_copy_(0, self.rank_index, norm_sh[-1:])
        boundary[world:world + 1].copy_(local_sample)
        reduced = self.final_boundary_reduce(boundary)
        rank_tails = reduced[:world]
        hidden = reduced[world:world + 1]
        logits = self.lm_head(hidden)
        if self.full_feedback:
            from .decoder import _order_barrier

            _order_barrier(norm_sh, [logits])
        auxiliary = (norm_sh, rank_tails) if self.full_feedback else None
        return hidden, logits, auxiliary, records
