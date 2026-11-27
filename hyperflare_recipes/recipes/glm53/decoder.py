# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM decoder-layer composition with b12x operators and tensor-parallel reductions."""

from dataclasses import dataclass
import itertools
import weakref

import torch

from ignition.components.collectives import (
    PreparedAllReduce,
    PreparedGatherShards,
    PreparedReduceScatter,
)
from hyperflare_recipes.components.mhc import MHCScratch, MHCWeights, PreparedMHC, PreparedMHCPost
from hyperflare_recipes.components.mlp import MLPScratch, PreparedMLP

from .kda import (
    KDAScratch, PreparedGlmKDA,
    ROW_SHARD_SCAN_SHAPE, row_shard_halo_shape, row_shard_sinit_shape,
)
from .mla import MLAScratch, PreparedGlmMLA
from .moe import MoeScratch, PreparedGlmMoe
from .weights import MoEWeights


_DMA_OPS = weakref.WeakValueDictionary()
_DMA_OP_HANDLES = itertools.count()
_DMA_EVENTS = weakref.WeakValueDictionary()
_DMA_EVENT_HANDLES = itertools.count()


class _DmaOpBinding:
    """Registry holder whose `.handle` attribute is rebound per rank by AOT."""

    def __init__(self, op):
        self.op = op
        self.handle = next(_DMA_OP_HANDLES)
        _DMA_OPS[self.handle] = op


class _StreamEventSlot:
    """Lazily created persistent CUDA event whose `.handle` is rebound per rank by AOT."""

    def __init__(self):
        self.event = None
        self.handle = next(_DMA_EVENT_HANDLES)
        _DMA_EVENTS[self.handle] = self

    def get(self):
        if self.event is None:
            self.event = torch.cuda.Event()
        return self.event


@torch.library.custom_op("flare::order_barrier", mutates_args=("anchor",))
def _order_barrier(anchor: torch.Tensor, dependencies: list[torch.Tensor]) -> None:
    pass


@_order_barrier.register_fake
def _fake_order_barrier(anchor, dependencies):
    pass


@torch.library.custom_op("flare::stream_event_wait", mutates_args=("output",))
def _stream_event_wait(output: torch.Tensor, dependencies: list[torch.Tensor], ev_handle: int) -> None:
    torch.cuda.current_stream(output.device).wait_event(_DMA_EVENTS[ev_handle].get())


@_stream_event_wait.register_fake
def _fake_stream_event_wait(output, dependencies, ev_handle):
    pass


@torch.library.custom_op("flare::dma_gather_enqueue", mutates_args=("output", "channel_token"))
def _dma_gather_enqueue(source: torch.Tensor, output: torch.Tensor, channel_token: torch.Tensor,
                        dependencies: list[torch.Tensor], op_handle: int, stream: int,
                        handoff_ev: int, done_ev: int) -> None:
    main = torch.cuda.current_stream(source.device)
    aux = torch.cuda.ExternalStream(stream, device=source.device)
    if handoff_ev >= 0:
        ev = _DMA_EVENTS[handoff_ev].get()
        ev.record(main)
        aux.wait_event(ev)
    op = _DMA_OPS[op_handle]
    with torch.cuda.stream(aux):
        if op.single:
            output.copy_(source)
        elif op.pcie:
            op.runtime.all_gather(source, out=output.view(op.runtime.world_size, *source.shape), plan=op.plan)
        else:
            op.runtime.gather_shards(source, out=output, plan=op.plan)
        _DMA_EVENTS[done_ev].get().record(aux)


@_dma_gather_enqueue.register_fake
def _fake_dma_gather_enqueue(source, output, channel_token, dependencies, op_handle, stream, handoff_ev, done_ev):
    pass


@torch.library.custom_op("flare::dma_reduce_enqueue", mutates_args=("output", "channel_token"))
def _dma_reduce_enqueue(source: torch.Tensor, output: torch.Tensor, channel_token: torch.Tensor,
                        dependencies: list[torch.Tensor], op_handle: int, stream: int,
                        handoff_ev: int, done_ev: int) -> None:
    main = torch.cuda.current_stream(source.device)
    aux = torch.cuda.ExternalStream(stream, device=source.device)
    if handoff_ev >= 0:
        ev = _DMA_EVENTS[handoff_ev].get()
        ev.record(main)
        aux.wait_event(ev)
    op = _DMA_OPS[op_handle]
    with torch.cuda.stream(aux):
        if op.single:
            output.copy_(source)
        else:
            op.runtime.reduce_scatter(source, out=output, plan=op.plan)
        _DMA_EVENTS[done_ev].get().record(aux)


@_dma_reduce_enqueue.register_fake
def _fake_dma_reduce_enqueue(source, output, channel_token, dependencies, op_handle, stream, handoff_ev, done_ev):
    pass


@torch.library.custom_op("flare::ordered_gather_shards", mutates_args=("output", "channel_token"))
def _ordered_gather_shards(source: torch.Tensor, output: torch.Tensor, channel_token: torch.Tensor,
                           dependencies: list[torch.Tensor], op_handle: int) -> None:
    op = _DMA_OPS[op_handle]
    if op.single:
        output.copy_(source)
    elif op.pcie:
        op.runtime.all_gather(source, out=output.view(op.runtime.world_size, *source.shape), plan=op.plan)
    else:
        op.runtime.gather_shards(source, out=output, plan=op.plan)


@_ordered_gather_shards.register_fake
def _fake_ordered_gather_shards(source, output, channel_token, dependencies, op_handle):
    pass


@torch.library.custom_op("flare::ordered_reduce_scatter", mutates_args=("output", "channel_token"))
def _ordered_reduce_scatter(source: torch.Tensor, output: torch.Tensor, channel_token: torch.Tensor,
                            dependencies: list[torch.Tensor], op_handle: int) -> None:
    op = _DMA_OPS[op_handle]
    if op.single:
        output.copy_(source)
    else:
        op.runtime.reduce_scatter(source, out=output, plan=op.plan)


@_ordered_reduce_scatter.register_fake
def _fake_ordered_reduce_scatter(source, output, channel_token, dependencies, op_handle):
    pass


class _AsyncDmaOp:
    """Ordered asynchronous collective on ``workspace.auxiliary_stream``.

    Mutates a shared ``channel_token`` to enforce identical collective order across TP ranks,
    performs an atomic main->aux handoff inside enqueue, and waits on its own per-op completion
    event rather than ``wait_stream(aux)`` (eliminating false stalls behind queued weight gathers).
    """

    def __init__(self, op, *, kind: str, stream_handle: int, handoff_ev: _StreamEventSlot):
        self.op = op
        self.kind = kind
        self.binding = _DmaOpBinding(op)
        self.stream_handle = stream_handle
        self.handoff_ev = handoff_ev
        self.done_ev = _StreamEventSlot()

    def functional(self, *, session):
        self.op.functional(session=session)
        return self

    def start(self, source: torch.Tensor, channel_token: torch.Tensor, *, dependencies=()):
        if getattr(self.op, "output", None) is not None:
            output = self.op.output
        else:
            shape, stride, dtype = self.op.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        deps = [source, *dependencies]
        if self.kind == "gather":
            _dma_gather_enqueue(source, output, channel_token, deps, self.binding.handle,
                                self.stream_handle, self.handoff_ev.handle, self.done_ev.handle)
        else:
            _dma_reduce_enqueue(source, output, channel_token, deps, self.binding.handle,
                                self.stream_handle, self.handoff_ev.handle, self.done_ev.handle)
        return (source, output)

    def finish(self, pending, dependency):
        source, output = pending
        _stream_event_wait(output, [source, dependency], self.done_ev.handle)
        return output


@dataclass(frozen=True)
class LayerScratch:
    attention_hc: MHCScratch
    mlp_hc: MHCScratch
    attention: KDAScratch | MLAScratch
    attention_output: torch.Tensor
    mlp: MLPScratch | MoeScratch
    mlp_output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world_size, device, *, sparse, block_type, mode, blocks=0):
        if world_size not in (1, 2, 4):
            raise ValueError("GLM layers require one, two or four ranks")
        def buffer(width):
            return torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
        if block_type == "linear_attention":
            attention = KDAScratch.allocate(tokens, 64 // world_size, device)
        elif block_type == "indexed_attention":
            attention = MLAScratch.allocate(tokens, 64 // world_size, blocks, mode, device)
        else:
            raise ValueError("GLM layers require linear_attention or indexed_attention")
        return cls(MHCScratch.allocate(tokens, 4096, device), MHCScratch.allocate(tokens, 4096, device),
                   attention, buffer(4096),
                   MoeScratch.allocate(tokens, world_size, device) if sparse else
                   MLPScratch.allocate(tokens, 4096, 12288 // world_size, device), buffer(4096))


def _install_mhc(module, norm, weights, prefix, norm_name):
    for attribute in ("fn", "scale", "base"):
        key = f"{prefix}_{attribute}"
        weights[key] = weights[key].float()
        setattr(module, attribute, torch.nn.Parameter(weights[key], requires_grad=False))
    norm.weight = torch.nn.Parameter(weights[f"{norm_name}.weight"], requires_grad=False)
    return MHCWeights(module.fn, module.scale, module.base, norm.weight)


def _projection_plans(projection):
    if projection is None:
        return ()
    plans = []
    if hasattr(projection, "plans"):
        plans.extend(projection.plans)
    elif getattr(projection, "plan", None) is not None:
        plans.append(projection.plan)
    qplan = getattr(projection, "quantizer_plan", None)
    if qplan is not None:
        plans.append(qplan)
    return tuple(plans)


def _collect_plans(*operators):
    plans = []
    for op in operators:
        if op is None:
            continue
        if isinstance(op, (PreparedMHC, PreparedMHCPost)):
            plans.append(op.plan)
        elif isinstance(op, PreparedMLP):
            plans.extend(_projection_plans(op.up_projection))
            if getattr(op, "activation_plan", None) is not None:
                plans.append(op.activation_plan)
            plans.extend(_projection_plans(op.down_projection))
        elif isinstance(op, PreparedGlmMoe):
            plans.extend(_projection_plans(op.router_projection))
            plans.append(op.routing.plan)
            plans.append(op.experts.plan)
            if getattr(op.experts, "scale_plan", None) is not None:
                plans.append(op.experts.scale_plan)
            plans.extend(_collect_plans(op.shared))
        elif isinstance(op, PreparedGlmKDA):
            for proj in op.projections.values():
                plans.extend(_projection_plans(proj))
            plans.append(op.conv_plan)
            if op.recurrent_plan is not None:
                plans.append(op.recurrent_plan)
        elif isinstance(op, PreparedGlmMLA):
            for proj in op.projections.values():
                plans.extend(_projection_plans(proj))
            for attr in ("query_absorption", "value_expansion"):
                bmm = getattr(op, attr, None)
                if bmm is not None and getattr(bmm, "plan", None) is not None:
                    plans.append(bmm.plan)
            if hasattr(op, "attention") and getattr(op.attention, "plan", None) is not None:
                plans.append(op.attention.plan)
            if getattr(op.writer, "plan", None) is not None:
                plans.append(op.writer.plan)
            idx = op.indexer
            plans.extend(_projection_plans(idx.query_projection))
            plans.extend(_projection_plans(idx.gate_projection))
            plans.extend((idx.query_plan, idx.pool_plan))
            for sel in idx.selectors:
                if getattr(sel, "plan", None) is not None:
                    plans.append(sel.plan)
    return tuple(p for p in plans if p is not None)


class PreparedGlmLayer:
    """One GLM decoder layer with fused mHC pre/post mixing and TP reductions."""

    def __init__(self, module, attention_weights, mlp_weights, residual_weights, source, output,
                 scratch: LayerScratch, state, metadata, *, mode, runtime, coordinator, session, name, workspace,
                 carry: MHCScratch | None = None, defer_post=False, gathered_input=None, max_cache_pages=None,
                 context=None, kda_records=None, pipelined=False, row_shard_attention=False,
                 replicated_attention=None, runtime_weights=False, full_attention_input=None,
                 rank_index=None, segment_eye=None, kda_segment_pool=None, kda_local_conv=None,
                 row_shard_buffers=None, contiguous_shards=False, mlp_quarters=False, gathered_mlp=None):
        if module.block_type not in ("linear_attention", "indexed_attention"):
            raise ValueError("prepared GLM layer requires canonical KDA or sparse MLA attention")
        self.module, self.scratch, self.output = module, scratch, output
        self.has_carry, self.defer_post = carry is not None, defer_post
        self.pipelined = pipelined
        self.row_shard_mla = False
        self.row_shard_kda = False
        self.contiguous_shards = bool(contiguous_shards)
        self.mlp_quarters = bool(mlp_quarters)
        attention_mhc = _install_mhc(module.attn_hc, module.input_layernorm, residual_weights,
                                     "hc_attn", "input_layernorm")
        mlp_mhc = _install_mhc(module.ffn_hc, module.post_attention_layernorm, residual_weights,
                               "hc_ffn", "post_attention_layernorm")
        options = {"rms_eps": module.attn_hc.input_norm.eps,
                   "hc_eps": module.attn_hc.hc_eps, "sinkhorn_iters": module.attn_hc.hc_sinkhorn_iters,
                   "norm_eps": module.input_layernorm.variance_epsilon}
        attention_hc_inputs = (source,) if carry is None else (source, carry.residual, carry.post, carry.comb)
        self.attention_hc = PreparedMHC(attention_hc_inputs, attention_mhc, scratch.attention_hc,
            session=session, name=f"{name}.attn_hc", workspace=workspace, **options)
        collective = dict(runtime=runtime, session=session, coordinator=coordinator)

        if not pipelined:
            self._init_serial(module, attention_weights, mlp_weights, scratch, state, metadata,
                              mode=mode, collective=collective, session=session, name=name, workspace=workspace,
                              mlp_mhc=mlp_mhc, options=options, output=output, defer_post=defer_post,
                              gathered_input=gathered_input, max_cache_pages=max_cache_pages,
                              context=context, kda_records=kda_records,
                              row_shard_attention=row_shard_attention,
                              replicated_attention=replicated_attention,
                              runtime_weights=runtime_weights,
                              rank_index=rank_index, segment_eye=segment_eye,
                              kda_segment_pool=kda_segment_pool, kda_local_conv=kda_local_conv,
                              row_shard_buffers=row_shard_buffers)
            return

        if gathered_input is None:
            raise ValueError("pipelined GLM layer requires a half-chunk gathered_input buffer")
        self.half_tokens = gathered_input.shape[0]
        self.q_shard_tokens = scratch.attention_hc.normalized.shape[0]
        self.shard_tokens = self.q_shard_tokens * 2 if self.mlp_quarters else self.q_shard_tokens
        attn_norm_source = (
            row_shard_buffers.attn_norm[:self.shard_tokens]
            if self.mlp_quarters and row_shard_buffers is not None
            else (torch.zeros((self.shard_tokens, 4096), dtype=torch.bfloat16,
                              device=scratch.attention_hc.normalized.device)
                  if self.mlp_quarters else scratch.attention_hc.normalized)
        )
        self.row_shard_mla = (
            module.block_type == "indexed_attention"
            and row_shard_attention
            and replicated_attention is not None
            and context is not None
            and context.records is not None
        )
        self.row_shard_kda = (
            module.block_type == "linear_attention"
            and row_shard_attention
            and replicated_attention is not None
            and rank_index is not None
            and segment_eye is not None
        )
        self.attention_gather_a = self.attention_gather_b = None
        self.attention_reduce_a = self.attention_reduce_b = None
        self.cache_gather_a = self.cache_gather_b = None
        self.sinit_gather = self.halo_gather = self.scan_gather = None
        self.cr_attention_gather_a = self.cr_attention_gather_b = None
        self.cr_attention_reduce_a = self.cr_attention_reduce_b = None
        self.cr_cache_gather_a = self.cr_cache_gather_b = None
        self.cr_sinit_gather = self.cr_halo_gather = self.cr_scan_gather = None
        self.mlp_gather_q = self.mlp_reduce_q = None
        self._dma_events = ()

        if self.row_shard_kda:
            device = scratch.attention_hc.normalized.device
            if row_shard_buffers is not None:
                sinit_sh, sinit_full = row_shard_buffers.sinit_sh, row_shard_buffers.sinit_full
                halo_sh, halo_full = row_shard_buffers.halo_sh, row_shard_buffers.halo_full
                scan_sh, scan_full = row_shard_buffers.scan_sh, row_shard_buffers.scan_full
                seg_qkv = row_shard_buffers.kda_seg_qkv[:self.shard_tokens * 2]
                seg_beta = row_shard_buffers.kda_seg_beta[:self.shard_tokens * 2]
            else:
                w = collective["runtime"].world_size
                sinit_shape, halo_shape = row_shard_sinit_shape(w), row_shard_halo_shape(w)
                sinit_sh = torch.zeros(sinit_shape, dtype=torch.bfloat16, device=device)
                sinit_full = torch.zeros((sinit_shape[0] * w, sinit_shape[1]),
                                         dtype=torch.bfloat16, device=device)
                halo_sh = torch.zeros(halo_shape, dtype=torch.bfloat16, device=device)
                halo_full = torch.zeros((halo_shape[0] * w, halo_shape[1]),
                                        dtype=torch.bfloat16, device=device)
                scan_sh = torch.zeros(ROW_SHARD_SCAN_SHAPE, dtype=torch.bfloat16, device=device)
                scan_full = torch.zeros((ROW_SHARD_SCAN_SHAPE[0] * w, ROW_SHARD_SCAN_SHAPE[1]),
                                        dtype=torch.bfloat16, device=device)
                seg_qkv = torch.zeros((self.shard_tokens * 2, 24768), dtype=torch.bfloat16, device=device)
                seg_beta = torch.zeros((self.shard_tokens * 2, 64), dtype=torch.bfloat16, device=device)
            self.sinit_gather = PreparedGatherShards(sinit_sh, sinit_full,
                name=f"{name}.kda_sinit", **collective)
            self.halo_gather = PreparedGatherShards(halo_sh, halo_full,
                name=f"{name}.kda_halo", **collective)
            self.scan_gather = PreparedGatherShards(scan_sh, scan_full,
                name=f"{name}.kda_scan", **collective)
            self.attention = PreparedGlmKDA(module.self_attn, {**attention_weights, **replicated_attention}, attn_norm_source,
                scratch.attention, state, metadata, mode=mode, session=session, name=f"{name}.kda",
                workspace=workspace, checkpoint_export=False, records=None, overlap_gate=False,
                row_shard=True, rank_index=rank_index, segment_eye=segment_eye,
                half_tokens=self.half_tokens, segment_pool=kda_segment_pool, local_conv=kda_local_conv,
                seg_qkv=seg_qkv, seg_beta=seg_beta,
                output_tokens=self.q_shard_tokens if self.mlp_quarters else None,
                runtime_weights=runtime_weights)
        elif module.block_type == "linear_attention":
            self.attention_gather_a = PreparedGatherShards(scratch.attention_hc.normalized, gathered_input,
                name=f"{name}.attention_gather.0", **collective)
            self.attention_gather_b = PreparedGatherShards(scratch.attention_hc.normalized, gathered_input,
                name=f"{name}.attention_gather.1", **collective)
            self.attention = PreparedGlmKDA(module.self_attn, attention_weights, gathered_input,
                scratch.attention, state, metadata, mode=mode, session=session, name=f"{name}.kda",
                workspace=workspace, checkpoint_export=mode == "prefill", records=kda_records,
                overlap_gate=False)
            self.attention_reduce_a = PreparedReduceScatter(scratch.attention.output, scratch.attention_output,
                name=f"{name}.attention_scatter.0", **collective)
            self.attention_reduce_b = PreparedReduceScatter(scratch.attention.output, scratch.attention_output,
                name=f"{name}.attention_scatter.1", **collective)
        elif self.row_shard_mla:
            device = scratch.attention_hc.normalized.device
            if row_shard_buffers is not None:
                cache_sh = row_shard_buffers.cache_sh[:self.shard_tokens]
                cache_full = row_shard_buffers.cache_full[:self.half_tokens]
            else:
                cache_sh = torch.zeros((self.shard_tokens, 768), dtype=torch.bfloat16, device=device)
                cache_full = torch.zeros((self.half_tokens, 768), dtype=torch.bfloat16, device=device)
            self.cache_gather_a = PreparedGatherShards(cache_sh, cache_full,
                name=f"{name}.cache_gather.0", **collective)
            self.cache_gather_b = PreparedGatherShards(cache_sh, cache_full,
                name=f"{name}.cache_gather.1", **collective)
            self.attention = PreparedGlmMLA(module.self_attn, attention_weights, attn_norm_source,
                scratch.attention, state, metadata, mode=mode, session=session, name=f"{name}.mla",
                workspace=workspace, max_cache_pages=max_cache_pages, context=context,
                row_shard_rows=self.shard_tokens, replicated_weights=replicated_attention,
                runtime_weights=runtime_weights)
        else:
            if full_attention_input is None:
                raise ValueError("head-parallel pipelined MLA requires full_attention_input")
            self.attention_gather_a = PreparedGatherShards(scratch.attention_hc.normalized, gathered_input,
                name=f"{name}.attention_gather.0", **collective)
            self.attention_gather_b = PreparedGatherShards(scratch.attention_hc.normalized, gathered_input,
                name=f"{name}.attention_gather.1", **collective)
            self.attention = PreparedGlmMLA(module.self_attn, attention_weights, full_attention_input,
                scratch.attention, state, metadata, mode=mode, session=session, name=f"{name}.mla",
                workspace=workspace, max_cache_pages=max_cache_pages, context=context)
            self.attention_reduce_a = PreparedReduceScatter(scratch.attention.output[:self.half_tokens],
                scratch.attention_output, name=f"{name}.attention_scatter.0", **collective)
            self.attention_reduce_b = PreparedReduceScatter(scratch.attention.output[self.half_tokens:],
                scratch.attention_output, name=f"{name}.attention_scatter.1", **collective)

        mlp_buf = gathered_mlp if self.mlp_quarters else gathered_input
        if self.mlp_quarters and mlp_buf is None:
            raise ValueError("quarter-microbatch MLP requires a quarter-chunk gathered_mlp buffer")
        self.mlp_hc = PreparedMHC((scratch.attention_output, scratch.attention_hc.residual,
                                  scratch.attention_hc.post, scratch.attention_hc.comb),
            mlp_mhc, scratch.mlp_hc, session=session, name=f"{name}.ffn_hc", workspace=workspace, **options)
        if self.mlp_quarters:
            self.mlp_gather_q = tuple(
                PreparedGatherShards(scratch.mlp_hc.normalized, mlp_buf,
                                     name=f"{name}.mlp_gather.{q}", **collective)
                for q in range(4)
            )
            self.mlp_gather_a = self.mlp_gather_q[0]
            self.mlp_gather_b = None
        else:
            self.mlp_gather_a = PreparedGatherShards(scratch.mlp_hc.normalized, mlp_buf,
                name=f"{name}.mlp_gather.0", **collective)
            self.mlp_gather_b = PreparedGatherShards(scratch.mlp_hc.normalized, mlp_buf,
                name=f"{name}.mlp_gather.1", **collective)
        if isinstance(mlp_weights, MoEWeights):
            self.mlp = PreparedGlmMoe(module.mlp, mlp_buf, mlp_weights, scratch.mlp,
                                      session=session, name=f"{name}.moe", workspace=workspace,
                                      prefill=mode == "prefill")
        else:
            self.mlp = PreparedMLP(mlp_buf, mlp_weights, scratch.mlp.gate_up,
                scratch.mlp.intermediate, scratch.mlp.output, session=session, name=f"{name}.mlp",
                workspace=workspace, swiglu_limit=module.mlp.swiglu_limit)
        if self.mlp_quarters:
            self.mlp_reduce_q = tuple(
                PreparedReduceScatter(scratch.mlp.output, scratch.mlp_output,
                                      name=f"{name}.mlp_scatter.{q}", **collective)
                for q in range(4)
            )
            self.mlp_reduce_a = self.mlp_reduce_b = None
        else:
            self.mlp_reduce_a = PreparedReduceScatter(scratch.mlp.output, scratch.mlp_output,
                name=f"{name}.mlp_scatter.0", **collective)
            self.mlp_reduce_b = PreparedReduceScatter(scratch.mlp.output, scratch.mlp_output,
                name=f"{name}.mlp_scatter.1", **collective)
        self.post = None if defer_post else PreparedMHCPost(scratch.mlp_output, scratch.mlp_hc.residual,
            scratch.mlp_hc.post, scratch.mlp_hc.comb, output, session=session, name=f"{name}.post")

        attn_hc_plans = _collect_plans(self.attention_hc)
        mlp_hc_plans = _collect_plans(self.mlp_hc)
        attn_plans = _collect_plans(self.attention)
        mlp_plans = _collect_plans(self.mlp)
        post_plans = _collect_plans(self.post) if self.post is not None else mlp_plans
        self.stream_handle = workspace.auxiliary_stream.cuda_stream
        self._ev_handoff = _StreamEventSlot()
        self._cache_gather_b_binding = None
        self._attention_gather_b_binding = None

        def _make_async_dma(op, kind, parallel_plans):
            if not op.single:
                for plan in parallel_plans:
                    workspace.register_parallel(op.plan, plan)
            return _AsyncDmaOp(op, kind=kind, stream_handle=self.stream_handle, handoff_ev=self._ev_handoff)

        if self.row_shard_kda:
            self.cr_sinit_gather = _make_async_dma(self.sinit_gather, "gather", attn_hc_plans + attn_plans)
            self.cr_halo_gather = _make_async_dma(self.halo_gather, "gather", attn_plans)
            self.cr_scan_gather = _make_async_dma(self.scan_gather, "gather", attn_plans)
            self.cr_mlp_gather_a = _make_async_dma(self.mlp_gather_a, "gather", attn_plans + mlp_hc_plans)
        elif self.row_shard_mla:
            self.cr_cache_gather_a = _make_async_dma(self.cache_gather_a, "gather", attn_hc_plans + attn_plans)
            if self.mlp_quarters:
                self.cr_cache_gather_b = _make_async_dma(self.cache_gather_b, "gather", attn_plans)
            else:
                self._cache_gather_b_binding = _DmaOpBinding(self.cache_gather_b)
            self.cr_mlp_gather_a = _make_async_dma(self.mlp_gather_a, "gather", attn_plans + mlp_hc_plans)
        else:
            self.cr_attention_gather_a = _make_async_dma(self.attention_gather_a, "gather", attn_hc_plans)
            if module.block_type == "linear_attention":
                self.cr_attention_gather_b = _make_async_dma(self.attention_gather_b, "gather", attn_plans)
            else:
                self._attention_gather_b_binding = _DmaOpBinding(self.attention_gather_b)
            self.cr_attention_reduce_a = _make_async_dma(self.attention_reduce_a, "reduce", attn_plans)
            self.cr_attention_reduce_b = _make_async_dma(self.attention_reduce_b, "reduce", mlp_hc_plans)
            self.cr_mlp_gather_a = _make_async_dma(self.mlp_gather_a, "gather", mlp_hc_plans)
        if self.mlp_quarters:
            self.cr_mlp_gather_b = self.cr_mlp_reduce_a = self.cr_mlp_reduce_b = None
            self._gather_q = tuple(_DmaOpBinding(op) for op in self.mlp_gather_q)
            self._reduce_q = tuple(_DmaOpBinding(op) for op in self.mlp_reduce_q)
            for op in self.mlp_gather_q:
                if not op.single:
                    for plan in (*attn_plans, *mlp_hc_plans, *mlp_plans):
                        workspace.register_parallel(op.plan, plan)
            for op in self.mlp_reduce_q:
                if not op.single:
                    for plan in (*mlp_plans, *post_plans, *attn_hc_plans, *attn_plans):
                        workspace.register_parallel(op.plan, plan)
            self._dma_events = tuple(_StreamEventSlot() for _ in range(8))
            (self._ev_g0, self._ev_g1, self._ev_g2, self._ev_g3,
             self._ev_r0, self._ev_r1, self._ev_r2, self._ev_r3) = self._dma_events
        else:
            self.cr_mlp_gather_b = _make_async_dma(self.mlp_gather_b, "gather", mlp_plans)
            self.cr_mlp_reduce_a = _make_async_dma(self.mlp_reduce_a, "reduce", mlp_plans)
            self.cr_mlp_reduce_b = _make_async_dma(self.mlp_reduce_b, "reduce", post_plans)

    def _init_serial(self, module, attention_weights, mlp_weights, scratch, state, metadata, *,
                     mode, collective, session, name, workspace, mlp_mhc, options, output, defer_post,
                     gathered_input, max_cache_pages, context, kda_records,
                     row_shard_attention=False, replicated_attention=None, runtime_weights=False, rank_index=None,
                     segment_eye=None, kda_segment_pool=None, kda_local_conv=None, row_shard_buffers=None):
        reduce, suffix = (PreparedAllReduce, "reduce") if gathered_input is None else (PreparedReduceScatter, "scatter")
        self.attention_gather = self.mlp_gather = None
        self.sinit_gather = self.halo_gather = self.scan_gather = self.cache_gather = None
        self.attention_reduce = None
        attention_input = scratch.attention_hc.normalized
        rows = attention_input.shape[0]
        tokens = gathered_input.shape[0] if gathered_input is not None else rows
        self.row_shard_mla = (
            gathered_input is not None
            and module.block_type == "indexed_attention"
            and row_shard_attention
            and replicated_attention is not None
            and context is not None
            and context.records is not None
        )
        self.row_shard_kda = (
            gathered_input is not None
            and module.block_type == "linear_attention"
            and row_shard_attention
            and replicated_attention is not None
            and rank_index is not None
            and segment_eye is not None
            and kda_segment_pool is not None
            and kda_local_conv is not None
        )
        if self.row_shard_kda:
            device = attention_input.device
            if row_shard_buffers is not None:
                sinit_sh, sinit_full = row_shard_buffers.sinit_sh, row_shard_buffers.sinit_full
                halo_sh, halo_full = row_shard_buffers.halo_sh, row_shard_buffers.halo_full
                scan_sh, scan_full = row_shard_buffers.scan_sh, row_shard_buffers.scan_full
                seg_qkv = row_shard_buffers.kda_seg_qkv[:rows]
                seg_beta = row_shard_buffers.kda_seg_beta[:rows]
            else:
                w = collective["runtime"].world_size
                sinit_shape, halo_shape = row_shard_sinit_shape(w), row_shard_halo_shape(w)
                sinit_sh = torch.zeros(sinit_shape, dtype=torch.bfloat16, device=device)
                sinit_full = torch.zeros((sinit_shape[0] * w, sinit_shape[1]),
                                         dtype=torch.bfloat16, device=device)
                halo_sh = torch.zeros(halo_shape, dtype=torch.bfloat16, device=device)
                halo_full = torch.zeros((halo_shape[0] * w, halo_shape[1]),
                                        dtype=torch.bfloat16, device=device)
                scan_sh = torch.zeros(ROW_SHARD_SCAN_SHAPE, dtype=torch.bfloat16, device=device)
                scan_full = torch.zeros((ROW_SHARD_SCAN_SHAPE[0] * w, ROW_SHARD_SCAN_SHAPE[1]),
                                        dtype=torch.bfloat16, device=device)
                seg_qkv = torch.zeros((rows, 24768), dtype=torch.bfloat16, device=device)
                seg_beta = torch.zeros((rows, 64), dtype=torch.bfloat16, device=device)
            self.sinit_gather = PreparedGatherShards(sinit_sh, sinit_full,
                name=f"{name}.kda_sinit", **collective)
            self.halo_gather = PreparedGatherShards(halo_sh, halo_full,
                name=f"{name}.kda_halo", **collective)
            self.scan_gather = PreparedGatherShards(scan_sh, scan_full,
                name=f"{name}.kda_scan", **collective)
            self.attention = PreparedGlmKDA(module.self_attn, {**attention_weights, **replicated_attention},
                attention_input, scratch.attention, state, metadata, mode=mode, session=session,
                name=f"{name}.kda", workspace=workspace, checkpoint_export=False, records=None,
                overlap_gate=False, row_shard=True, single_segment=True, rank_index=rank_index,
                segment_eye=segment_eye, half_tokens=tokens, segment_pool=kda_segment_pool,
                local_conv=kda_local_conv, seg_qkv=seg_qkv, seg_beta=seg_beta,
                runtime_weights=runtime_weights)
        elif self.row_shard_mla:
            device = attention_input.device
            if row_shard_buffers is not None:
                cache_sh = row_shard_buffers.cache_sh[:rows]
                cache_full = row_shard_buffers.cache_full[:tokens]
            else:
                cache_sh = torch.zeros((rows, 768), dtype=torch.bfloat16, device=device)
                cache_full = torch.zeros((tokens, 768), dtype=torch.bfloat16, device=device)
            self.cache_gather = PreparedGatherShards(cache_sh, cache_full,
                name=f"{name}.cache_gather", **collective)
            self.attention = PreparedGlmMLA(module.self_attn, attention_weights, attention_input,
                scratch.attention, state, metadata, mode=mode, session=session, name=f"{name}.mla",
                workspace=workspace, max_cache_pages=max_cache_pages, context=context,
                row_shard_rows=rows, replicated_weights=replicated_attention,
                runtime_weights=runtime_weights)
        else:
            if gathered_input is not None:
                self.attention_gather = PreparedGatherShards(attention_input, gathered_input,
                    name=f"{name}.attention_gather", **collective)
                attention_input = gathered_input
            attention_args = (module.self_attn, attention_weights, attention_input,
                              scratch.attention, state, metadata)
            if module.block_type == "linear_attention":
                self.attention = PreparedGlmKDA(*attention_args, mode=mode, session=session, name=f"{name}.kda",
                                                workspace=workspace, checkpoint_export=mode == "prefill",
                                                records=kda_records)
            else:
                self.attention = PreparedGlmMLA(*attention_args, mode=mode, session=session, name=f"{name}.mla",
                                                workspace=workspace, max_cache_pages=max_cache_pages,
                                                context=context)
            self.attention_reduce = reduce(scratch.attention.output, scratch.attention_output,
                name=f"{name}.attention_{suffix}", **collective)
        self.mlp_hc = PreparedMHC((scratch.attention_output, scratch.attention_hc.residual,
                                  scratch.attention_hc.post, scratch.attention_hc.comb),
            mlp_mhc, scratch.mlp_hc, session=session, name=f"{name}.ffn_hc", workspace=workspace, **options)
        mlp_input = scratch.mlp_hc.normalized
        if gathered_input is not None:
            self.mlp_gather = PreparedGatherShards(mlp_input, gathered_input,
                name=f"{name}.mlp_gather", **collective)
            mlp_input = gathered_input
        if isinstance(mlp_weights, MoEWeights):
            self.mlp = PreparedGlmMoe(module.mlp, mlp_input, mlp_weights, scratch.mlp,
                                      session=session, name=f"{name}.moe", workspace=workspace,
                                      prefill=mode == "prefill")
        else:
            self.mlp = PreparedMLP(mlp_input, mlp_weights, scratch.mlp.gate_up,
                scratch.mlp.intermediate, scratch.mlp.output, session=session, name=f"{name}.mlp", workspace=workspace,
                swiglu_limit=module.mlp.swiglu_limit)
        self.mlp_reduce = reduce(scratch.mlp.output, scratch.mlp_output,
            name=f"{name}.mlp_{suffix}", **collective)
        self.post = None if defer_post else PreparedMHCPost(scratch.mlp_output, scratch.mlp_hc.residual,
            scratch.mlp_hc.post, scratch.mlp_hc.comb, output, session=session, name=f"{name}.post")
        self._sinit_binding = _DmaOpBinding(self.sinit_gather) if self.sinit_gather is not None else None
        self._halo_binding = _DmaOpBinding(self.halo_gather) if self.halo_gather is not None else None
        self._scan_binding = _DmaOpBinding(self.scan_gather) if self.scan_gather is not None else None
        self._cache_binding = _DmaOpBinding(self.cache_gather) if self.cache_gather is not None else None
        self._attn_gather_binding = _DmaOpBinding(self.attention_gather) if self.attention_gather is not None else None
        self._attn_reduce_binding = (
            _DmaOpBinding(self.attention_reduce) if isinstance(self.attention_reduce, PreparedReduceScatter) else None
        )
        self._mlp_gather_binding = _DmaOpBinding(self.mlp_gather) if self.mlp_gather is not None else None
        self._mlp_reduce_binding = (
            _DmaOpBinding(self.mlp_reduce) if isinstance(self.mlp_reduce, PreparedReduceScatter) else None
        )

    @staticmethod
    def _op_layout(op):
        if getattr(op, "output", None) is not None:
            return tuple(op.output.shape), tuple(op.output.stride()), op.output.dtype
        return op.output_layout

    @staticmethod
    def _sync_gather(binding, op, source, channel_token, dependencies=()):
        if channel_token is None or binding is None:
            return op(source, dependencies=dependencies) if dependencies else op(source)
        if getattr(op, "output", None) is not None:
            output = op.output
        else:
            shape, stride, dtype = op.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        _ordered_gather_shards(source, output, channel_token, list(dependencies), binding.handle)
        return output

    @staticmethod
    def _sync_reduce(binding, op, source, channel_token, dependencies=()):
        if channel_token is None or binding is None:
            return op(source)
        if getattr(op, "output", None) is not None:
            output = op.output
        else:
            shape, stride, dtype = op.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        _ordered_reduce_scatter(source, output, channel_token, list(dependencies), binding.handle)
        return output

    def release_outputs(self, *, session):
        self.attention_hc.functional()
        if self.pipelined:
            for attr in ("cr_attention_gather_a", "cr_attention_gather_b", "cr_attention_reduce_a",
                         "cr_attention_reduce_b", "cr_cache_gather_a", "cr_cache_gather_b",
                         "cr_sinit_gather", "cr_halo_gather", "cr_scan_gather",
                         "cr_mlp_gather_a", "cr_mlp_gather_b", "cr_mlp_reduce_a", "cr_mlp_reduce_b"):
                op = getattr(self, attr, None)
                if op is not None:
                    op.functional(session=session)
            if self.mlp_gather_q is not None:
                for op in self.mlp_gather_q[1:]:
                    op.functional(session=session)
            if self.mlp_reduce_q is not None:
                for op in self.mlp_reduce_q:
                    op.functional(session=session)
            if self.cache_gather_b is not None and self.cr_cache_gather_b is None:
                self.cache_gather_b.functional(session=session)
            if self.attention_gather_b is not None and self.cr_attention_gather_b is None:
                self.attention_gather_b.functional(session=session)
            self.attention.functional(session=session)
            self.mlp_hc.functional()
            self.mlp.functional(session=session)
            if self.post is not None:
                self.post.functional()
            self.scratch = self.output = None
            return self
        for attr in ("attention_gather", "sinit_gather", "halo_gather", "scan_gather", "cache_gather"):
            op = getattr(self, attr, None)
            if op is not None:
                op.functional(session=session)
        self.attention.functional(session=session)
        if self.attention_reduce is not None:
            self.attention_reduce.functional(session=session)
        self.mlp_hc.functional()
        if self.mlp_gather is not None:
            self.mlp_gather.functional(session=session)
        self.mlp.functional(session=session)
        self.mlp_reduce.functional(session=session)
        if self.post is not None:
            self.post.functional()
        self.scratch = self.output = None
        return self

    def __call__(self, source, metadata, carry=None, *, records=None, pos_sh=None, attn_weights=None,
                 weight_gather=None, layer_index=None, next_layer_index=None, channel_token=None):
        if self.has_carry:
            if carry is None:
                raise ValueError("GLM post/pre layer requires runtime residual and mixing carry")
            inputs = (source, *carry)
        else:
            inputs = (source,)
        use_async_roce = weight_gather is not None and weight_gather.async_roce
        attention_residual, attention_post, attention_comb, attention_input = self.attention_hc(*inputs)
        if self.row_shard_kda:
            sinit_sh = self.attention.pack_initial_state(metadata)
            s_init_full = self._sync_gather(self._sinit_binding, self.sinit_gather, sinit_sh, channel_token)
            qkv, raw_beta, forget_in = self.attention.project_input_shard(attention_input, weights=attn_weights)
            halo_sh = self.attention.pack_conv_halo(qkv, metadata)
            halo_full = self._sync_gather(self._halo_binding, self.halo_gather, halo_sh, channel_token)
            raw_g = self.attention.project_forget_shard(forget_in, weights=attn_weights)
            summary, scan_sh = self.attention.summarize_segment(
                qkv, None, raw_g, None, raw_beta, None, halo_full, metadata, weights=attn_weights)
            scan_full = self._sync_gather(self._scan_binding, self.scan_gather, scan_sh, channel_token)
            p_cur_o = (
                weight_gather.start_o_only(layer_index, [scan_full], channel_token=channel_token)
                if (use_async_roce and layer_index is not None
                    and attn_weights is not None and attn_weights.o_proj is None)
                else None
            )
            if p_cur_o is not None and channel_token is not None:
                _order_barrier(scan_full, [channel_token])
                _order_barrier(attention_input, [channel_token])
            z_sh = self.attention.project_gate_shard(attention_input, weights=attn_weights)
            rec_sh, live_sh, s4, conv_all = self.attention.finalize_segment(
                summary, s_init_full, scan_full, metadata, weights=attn_weights)
            if p_cur_o is not None:
                attn_weights = weight_gather.attach_o(attn_weights, p_cur_o, dependencies=(rec_sh, z_sh))
            attention = self.attention.output_shard(rec_sh, z_sh, live_sh, weights=attn_weights)
            self.attention.commit_state(s4, conv_all, metadata)
        elif self.row_shard_mla:
            query_res, idx_query, weight_in, cache_sh = self.attention.project_shard(attention_input)
            cache_full = self._sync_gather(self._cache_binding, self.cache_gather, cache_sh, channel_token)
            global_index_cache, slab = self.attention.write_and_gather_context(cache_full, metadata)
            if channel_token is not None:
                _order_barrier(channel_token, [global_index_cache, slab])
            p_cur_o = (
                weight_gather.start_o_only(layer_index, [global_index_cache, slab], channel_token=channel_token)
                if (use_async_roce and layer_index is not None
                    and attn_weights is not None and attn_weights.o_proj is None)
                else None
            )
            if p_cur_o is not None and channel_token is not None:
                _order_barrier(idx_query, [channel_token])
            physical, selected_lengths = self.attention.select_shard(
                idx_query, weight_in, pos_sh, global_index_cache)
            attach_o = (
                (lambda dep: weight_gather.attach_o(attn_weights, p_cur_o, dependencies=(physical, dep)))
                if p_cur_o is not None else None
            )
            attention = self.attention.attend_selected_shard(
                query_res, physical, selected_lengths, pos_sh, slab, weights=attn_weights, attach_o=attach_o)
        else:
            if self.attention_gather is not None:
                attention_input = self._sync_gather(
                    self._attn_gather_binding, self.attention_gather, attention_input, channel_token)
            if self.module.block_type == "indexed_attention":
                attended = self.attention(attention_input, metadata)
            else:
                attended = self.attention(attention_input, metadata, records=records)
            attention = self._sync_reduce(self._attn_reduce_binding, self.attention_reduce, attended, channel_token)
        mlp_residual, mlp_post, mlp_comb, mlp_input = self.mlp_hc(
            attention, attention_residual, attention_post, attention_comb)
        if self.mlp_gather is not None:
            mlp_input = self._sync_gather(self._mlp_gather_binding, self.mlp_gather, mlp_input, channel_token)
        pending_next_w = (
            weight_gather.start_input_all(next_layer_index, [mlp_input], channel_token=channel_token)
            if use_async_roce and next_layer_index is not None else None
        )
        if pending_next_w is not None and channel_token is not None:
            _order_barrier(mlp_input, [channel_token])
        mlp_out = self.mlp(mlp_input)
        mlp = self._sync_reduce(self._mlp_reduce_binding, self.mlp_reduce, mlp_out, channel_token)
        out = mlp if self.post is None else self.post(mlp, mlp_residual, mlp_post, mlp_comb)
        carry_out = (mlp_residual, mlp_post, mlp_comb) if self.post is None else None
        if pending_next_w is not None:
            return out, carry_out, pending_next_w
        return out, carry_out

    def forward_pipelined(self, incoming, metadata, *, meta_a=None, meta_b=None,
                          pos_a=None, pos_b=None, pending_mlp_b=None, records=None,
                          attn_weights=None, weight_gather=None, next_layer_index=None,
                          pending_cur_o=None, channel_token=None):
        """Execute one layer over two row-sharded microbatches with a single ordered DMA channel."""
        if channel_token is None:
            channel_token = torch.empty(1, dtype=torch.int32, device=incoming[0].device)
        if self.row_shard_kda:
            return self._forward_pipelined_row_shard_kda(
                incoming, metadata, pending_mlp_b=pending_mlp_b,
                attn_weights=attn_weights, weight_gather=weight_gather, next_layer_index=next_layer_index,
                pending_cur_o=pending_cur_o, channel_token=channel_token)
        if self.row_shard_mla:
            return self._forward_pipelined_row_shard_mla(
                incoming, metadata, pos_a=pos_a, pos_b=pos_b, pending_mlp_b=pending_mlp_b,
                attn_weights=attn_weights, weight_gather=weight_gather, next_layer_index=next_layer_index,
                pending_cur_o=pending_cur_o, channel_token=channel_token)
        if self.has_carry:
            mlp_a, prev_res_a, prev_post_a, prev_comb_a, prev_res_b, prev_post_b, prev_comb_b = incoming
            res_a, post_a, comb_a, norm_a_sh = self.attention_hc(mlp_a, prev_res_a, prev_post_a, prev_comb_a)
            prev_cr, p_prev_b = pending_mlp_b
            mlp_b = prev_cr.finish(p_prev_b, norm_a_sh)
            p_ag_a = self.cr_attention_gather_a.start(norm_a_sh, channel_token, dependencies=(mlp_b,))
            res_b, post_b, comb_b, norm_b_sh = self.attention_hc(mlp_b, prev_res_b, prev_post_b, prev_comb_b)
        else:
            src_a, src_b = incoming
            res_a, post_a, comb_a, norm_a_sh = self.attention_hc(src_a)
            p_ag_a = self.cr_attention_gather_a.start(norm_a_sh, channel_token, dependencies=(norm_a_sh,))
            res_b, post_b, comb_b, norm_b_sh = self.attention_hc(src_b)
        norm_a = self.cr_attention_gather_a.finish(p_ag_a, norm_b_sh)

        if self.module.block_type == "linear_attention":
            p_ag_b = self.cr_attention_gather_b.start(norm_b_sh, channel_token, dependencies=(norm_a,))
            out_a = self.attention(norm_a, meta_a, records=records)
            norm_b = self.cr_attention_gather_b.finish(p_ag_b, out_a)
            p_rs_a = self.cr_attention_reduce_a.start(out_a, channel_token, dependencies=(norm_b,))
            out_b = self.attention(norm_b, meta_b, records=records)
            att_a = self.cr_attention_reduce_a.finish(p_rs_a, out_b)
        else:
            norm_b = self._sync_gather(
                self._attention_gather_b_binding, self.attention_gather_b, norm_b_sh, channel_token,
                dependencies=(norm_a,))
            if self.contiguous_shards:
                w = self.half_tokens // self.shard_tokens
                full_norm = torch.stack(
                    (norm_a.view(w, self.shard_tokens, -1), norm_b.view(w, self.shard_tokens, -1)), dim=1
                ).reshape(self.half_tokens * 2, -1)
            else:
                full_norm = torch.cat((norm_a, norm_b), dim=0)
            full_out = self.attention(full_norm, metadata)
            if self.contiguous_shards:
                w = self.half_tokens // self.shard_tokens
                paired = full_out.view(w, 2, self.shard_tokens, -1)
                out_a = paired[:, 0].reshape(self.half_tokens, -1).contiguous()
                out_b = paired[:, 1].reshape(self.half_tokens, -1).contiguous()
            else:
                out_a, out_b = full_out[:self.half_tokens].contiguous(), full_out[self.half_tokens:].contiguous()
            p_rs_a = self.cr_attention_reduce_a.start(out_a, channel_token, dependencies=(full_out,))
            att_a = self.cr_attention_reduce_a.finish(p_rs_a, out_b)

        p_rs_b = self.cr_attention_reduce_b.start(out_b, channel_token, dependencies=(att_a,))
        mres_a, mpost_a, mcomb_a, ffn_a_sh = self.mlp_hc(att_a, res_a, post_a, comb_a)
        att_b = self.cr_attention_reduce_b.finish(p_rs_b, ffn_a_sh)

        p_ffn_a = self.cr_mlp_gather_a.start(ffn_a_sh, channel_token, dependencies=(att_b,))
        mres_b, mpost_b, mcomb_b, ffn_b_sh = self.mlp_hc(att_b, res_b, post_b, comb_b)
        ffn_a = self.cr_mlp_gather_a.finish(p_ffn_a, ffn_b_sh)

        out, pend_b, _ = self._finish_pipelined_ffn(
            ffn_a, ffn_b_sh, (mres_a, mpost_a, mcomb_a), (mres_b, mpost_b, mcomb_b),
            channel_token=channel_token)
        return out, pend_b


    def _forward_pipelined_row_shard_kda(self, incoming, metadata, *, pending_mlp_b,
                                         attn_weights=None, weight_gather=None, next_layer_index=None,
                                         pending_cur_o=None, channel_token=None):
        if self.mlp_quarters:
            return self._forward_pipelined_row_shard_kda_quarters(
                incoming, metadata, pending_mlp_b=pending_mlp_b,
                attn_weights=attn_weights, weight_gather=weight_gather, next_layer_index=next_layer_index,
                pending_cur_o=pending_cur_o, channel_token=channel_token)
        zero_copy = self.attention.supports_zero_copy_segment
        half = self.shard_tokens
        seg_proj = seg_raw_g = None
        if self.has_carry:
            mlp_a, prev_res_a, prev_post_a, prev_comb_a, prev_res_b, prev_post_b, prev_comb_b = incoming
            if zero_copy:
                seg_proj = torch.empty((half * 2, 24768), dtype=torch.bfloat16, device=mlp_a.device)
                seg_raw_g = torch.empty((half * 2, 8192), dtype=torch.bfloat16, device=mlp_a.device)
            res_a, post_a, comb_a, norm_a_sh = self.attention_hc(mlp_a, prev_res_a, prev_post_a, prev_comb_a)
            qkv_a, beta_a, fin_a = self.attention.project_input_shard(
                norm_a_sh, out=seg_proj[:half] if zero_copy else None, weights=attn_weights)
            raw_g_a = self.attention.project_forget_shard(
                fin_a, out=seg_raw_g[:half] if zero_copy else None, weights=attn_weights)
            prev_cr, p_prev_b = pending_mlp_b
            mlp_b = prev_cr.finish(p_prev_b, raw_g_a)
            sinit_sh = self.attention.pack_initial_state(metadata)
            p_sinit = self.cr_sinit_gather.start(sinit_sh, channel_token, dependencies=(mlp_b,))
            res_b, post_b, comb_b, norm_b_sh = self.attention_hc(mlp_b, prev_res_b, prev_post_b, prev_comb_b)
        else:
            src_a, src_b = incoming
            if zero_copy:
                seg_proj = torch.empty((half * 2, 24768), dtype=torch.bfloat16, device=src_a.device)
                seg_raw_g = torch.empty((half * 2, 8192), dtype=torch.bfloat16, device=src_a.device)
            res_a, post_a, comb_a, norm_a_sh = self.attention_hc(src_a)
            qkv_a, beta_a, fin_a = self.attention.project_input_shard(
                norm_a_sh, out=seg_proj[:half] if zero_copy else None, weights=attn_weights)
            raw_g_a = self.attention.project_forget_shard(
                fin_a, out=seg_raw_g[:half] if zero_copy else None, weights=attn_weights)
            sinit_sh = self.attention.pack_initial_state(metadata)
            p_sinit = self.cr_sinit_gather.start(sinit_sh, channel_token, dependencies=(raw_g_a,))
            res_b, post_b, comb_b, norm_b_sh = self.attention_hc(src_b)

        p_w1a = (
            weight_gather.start_phase1a(next_layer_index, [sinit_sh, norm_b_sh], channel_token=channel_token)
            if weight_gather is not None and next_layer_index is not None else None
        )
        if p_w1a is not None:
            _order_barrier(norm_b_sh, [channel_token])
        qkv_b, beta_b, fin_b = self.attention.project_input_shard(
            norm_b_sh, out=seg_proj[half:] if zero_copy else None, weights=attn_weights)
        sinit_full = self.cr_sinit_gather.finish(p_sinit, qkv_b)

        halo_sh = self.attention.pack_conv_halo(qkv_b, metadata)
        halo_deps = (sinit_full, *p_w1a.buffers) if p_w1a is not None else (sinit_full,)
        p_halo = self.cr_halo_gather.start(halo_sh, channel_token, dependencies=halo_deps)
        raw_g_b = self.attention.project_forget_shard(
            fin_b, out=seg_raw_g[half:] if zero_copy else None, weights=attn_weights)
        halo_full = self.cr_halo_gather.finish(p_halo, raw_g_b)

        p_w1b = (
            weight_gather.start_phase1b(p_w1a, [halo_full], channel_token=channel_token)
            if p_w1a is not None else None
        )
        if p_w1b is not None:
            _order_barrier(halo_full, [channel_token])
        summary, scan_sh = self.attention.summarize_segment(
            qkv_a, qkv_b, raw_g_a, raw_g_b, beta_a, beta_b, halo_full, metadata,
            seg_projected=seg_proj, seg_raw_g=seg_raw_g, weights=attn_weights)

        scan_deps = (halo_full, *p_w1b.buffers) if p_w1b is not None else (halo_full,)
        p_scan = self.cr_scan_gather.start(scan_sh, channel_token, dependencies=scan_deps)
        z_a = self.attention.project_gate_shard(norm_a_sh, weights=attn_weights)
        z_b = self.attention.project_gate_shard(norm_b_sh, weights=attn_weights)
        scan_full = self.cr_scan_gather.finish(p_scan, z_b)

        p_w2a = (
            weight_gather.start_phase2a(p_w1b, [scan_full], channel_token=channel_token)
            if p_w1b is not None else None
        )
        if p_w2a is not None:
            _order_barrier(scan_full, [channel_token])
        rec_a, rec_b, live_a, live_b, s4, conv_all = self.attention.finalize_segment(
            summary, sinit_full, scan_full, metadata, weights=attn_weights)
        if weight_gather is not None and pending_cur_o is not None:
            attn_weights = weight_gather.attach_o(attn_weights, pending_cur_o, dependencies=(rec_a,))
        att_a = self.attention.output_shard(rec_a, z_a, live_a, weights=attn_weights)
        mres_a, mpost_a, mcomb_a, ffn_a_sh = self.mlp_hc(att_a, res_a, post_a, comb_a)

        ffn_deps = (scan_full, *p_w2a.buffers, *p_w2a.o_buffers) if p_w2a is not None else (scan_full,)
        p_ffn_a = self.cr_mlp_gather_a.start(ffn_a_sh, channel_token, dependencies=ffn_deps)
        att_b = self.attention.output_shard(rec_b, z_b, live_b, weights=attn_weights)
        self.attention.commit_state(s4, conv_all, metadata)
        mres_b, mpost_b, mcomb_b, ffn_b_sh = self.mlp_hc(att_b, res_b, post_b, comb_b)
        ffn_a = self.cr_mlp_gather_a.finish(p_ffn_a, ffn_b_sh)

        out, pend_b, pending_next_w = self._finish_pipelined_ffn(
            ffn_a, ffn_b_sh, (mres_a, mpost_a, mcomb_a), (mres_b, mpost_b, mcomb_b),
            weight_gather=weight_gather, p_w2a=p_w2a, channel_token=channel_token)
        return (out, pend_b, pending_next_w) if weight_gather is not None else (out, pend_b)

    def _forward_pipelined_row_shard_kda_quarters(self, incoming, metadata, *, pending_mlp_b,
                                                  attn_weights=None, weight_gather=None, next_layer_index=None,
                                                  pending_cur_o=None, channel_token=None):
        zero_copy = self.attention.supports_zero_copy_segment
        half = self.shard_tokens
        seg_proj = seg_raw_g = None
        if self.has_carry:
            mlp_0, mlp_1, mlp_2, c0, c1, c2, c3 = incoming
            device = mlp_0.device
            if zero_copy:
                seg_proj = torch.empty((half * 2, 24768), dtype=torch.bfloat16, device=device)
                seg_raw_g = torch.empty((half * 2, 8192), dtype=torch.bfloat16, device=device)
            r0, p0, k0, n0 = self.attention_hc(mlp_0, *c0)
            r1, p1, k1, n1 = self.attention_hc(mlp_1, *c1)
            norm_a_sh = torch.cat((n0, n1), dim=0)
            qkv_a, beta_a, fin_a = self.attention.project_input_shard(
                norm_a_sh, out=seg_proj[:half] if zero_copy else None, weights=attn_weights)
            raw_g_a = self.attention.project_forget_shard(
                fin_a, out=seg_raw_g[:half] if zero_copy else None, weights=attn_weights)
            r2, p2, k2, n2 = self.attention_hc(mlp_2, *c2)
            prev_ev_r3, mlp_3, prev_mlp_out_3 = pending_mlp_b
            _stream_event_wait(mlp_3, [prev_mlp_out_3, raw_g_a, n2], prev_ev_r3.handle)
            sinit_sh = self.attention.pack_initial_state(metadata)
            p_sinit = self.cr_sinit_gather.start(sinit_sh, channel_token, dependencies=(mlp_3,))
            r3, p3, k3, n3 = self.attention_hc(mlp_3, *c3)
            norm_b_sh = torch.cat((n2, n3), dim=0)
        else:
            src_0, src_1, src_2, src_3 = incoming
            device = src_0.device
            if zero_copy:
                seg_proj = torch.empty((half * 2, 24768), dtype=torch.bfloat16, device=device)
                seg_raw_g = torch.empty((half * 2, 8192), dtype=torch.bfloat16, device=device)
            r0, p0, k0, n0 = self.attention_hc(src_0)
            r1, p1, k1, n1 = self.attention_hc(src_1)
            norm_a_sh = torch.cat((n0, n1), dim=0)
            qkv_a, beta_a, fin_a = self.attention.project_input_shard(
                norm_a_sh, out=seg_proj[:half] if zero_copy else None, weights=attn_weights)
            raw_g_a = self.attention.project_forget_shard(
                fin_a, out=seg_raw_g[:half] if zero_copy else None, weights=attn_weights)
            sinit_sh = self.attention.pack_initial_state(metadata)
            p_sinit = self.cr_sinit_gather.start(sinit_sh, channel_token, dependencies=(raw_g_a,))
            r2, p2, k2, n2 = self.attention_hc(src_2)
            r3, p3, k3, n3 = self.attention_hc(src_3)
            norm_b_sh = torch.cat((n2, n3), dim=0)

        # Phase 1a (~64 MB, ~1.6 ms): overlap behind qkv_b (2.4 ms); finish(p_sinit) waits only on p_sinit.done_ev.
        p_w1a = (
            weight_gather.start_phase1a(next_layer_index, [sinit_sh, norm_b_sh], channel_token=channel_token)
            if weight_gather is not None and next_layer_index is not None else None
        )
        if p_w1a is not None:
            _order_barrier(norm_b_sh, [channel_token])
        qkv_b, beta_b, fin_b = self.attention.project_input_shard(
            norm_b_sh, out=seg_proj[half:] if zero_copy else None, weights=attn_weights)
        sinit_full = self.cr_sinit_gather.finish(p_sinit, qkv_b)

        halo_sh = self.attention.pack_conv_halo(qkv_b, metadata)
        halo_deps = (sinit_full, *p_w1a.buffers) if p_w1a is not None else (sinit_full,)
        p_halo = self.cr_halo_gather.start(halo_sh, channel_token, dependencies=halo_deps)
        raw_g_b = self.attention.project_forget_shard(
            fin_b, out=seg_raw_g[half:] if zero_copy else None, weights=attn_weights)
        halo_full = self.cr_halo_gather.finish(p_halo, raw_g_b)

        # Phase 1b (~80 MB, ~2.0 ms): overlap behind summarize_segment (~2.5 ms).
        p_w1b = (
            weight_gather.start_phase1b(p_w1a, [halo_full], channel_token=channel_token)
            if p_w1a is not None else None
        )
        if p_w1b is not None:
            _order_barrier(halo_full, [channel_token])
        summary, scan_sh = self.attention.summarize_segment(
            qkv_a, qkv_b, raw_g_a, raw_g_b, beta_a, beta_b, halo_full, metadata,
            seg_projected=seg_proj, seg_raw_g=seg_raw_g, weights=attn_weights)

        scan_deps = (halo_full, *p_w1b.buffers) if p_w1b is not None else (halo_full,)
        p_scan = self.cr_scan_gather.start(scan_sh, channel_token, dependencies=scan_deps)
        z_a = self.attention.project_gate_shard(norm_a_sh, weights=attn_weights)
        z_b = self.attention.project_gate_shard(norm_b_sh, weights=attn_weights)
        scan_full = self.cr_scan_gather.finish(p_scan, z_b)

        # Phase 2a (~64 MB, ~1.6 ms): overlap behind finalize_segment (~2.6 ms).
        p_w2a = (
            weight_gather.start_phase2a(p_w1b, [scan_full], channel_token=channel_token)
            if p_w1b is not None else None
        )
        if p_w2a is not None:
            _order_barrier(scan_full, [channel_token])
        rec_a, rec_b, live_a, live_b, s4, conv_all = self.attention.finalize_segment(
            summary, sinit_full, scan_full, metadata, weights=attn_weights)
        if weight_gather is not None and pending_cur_o is not None:
            attn_weights = weight_gather.attach_o(attn_weights, pending_cur_o, dependencies=(rec_a,))
        q = self.q_shard_tokens
        rec_0, rec_1 = rec_a.split(q, dim=0)
        rec_2, rec_3 = rec_b.split(q, dim=0)
        z_0, z_1 = z_a.split(q, dim=0)
        z_2, z_3 = z_b.split(q, dim=0)
        live_0 = live_a.clamp(0, q).to(torch.int32)
        live_1 = (live_a - q).clamp(0, q).to(torch.int32)
        live_2 = live_b.clamp(0, q).to(torch.int32)
        live_3 = (live_b - q).clamp(0, q).to(torch.int32)

        aux = self.stream_handle
        handoff = self._ev_handoff.handle
        g_shape, g_stride, g_dtype = self._op_layout(self.mlp_gather_q[0])
        ffn_0 = torch.empty_strided(g_shape, g_stride, dtype=g_dtype, device=device)
        ffn_1 = torch.empty_strided(g_shape, g_stride, dtype=g_dtype, device=device)
        ffn_2 = torch.empty_strided(g_shape, g_stride, dtype=g_dtype, device=device)
        ffn_3 = torch.empty_strided(g_shape, g_stride, dtype=g_dtype, device=device)

        # Enqueue each quarter gather immediately as its shard becomes ready so g0..g3 run back-to-back during att_1..3.
        att_0 = self.attention.output_shard(rec_0.contiguous(), z_0.contiguous(), live_0, weights=attn_weights)
        mr0, mp0, mk0, ffn_0_sh = self.mlp_hc(att_0, r0, p0, k0)
        ffn0_deps = ([scan_full, *p_w2a.buffers, *p_w2a.o_buffers, ffn_0_sh]
                     if p_w2a is not None else [scan_full, ffn_0_sh])
        _dma_gather_enqueue(ffn_0_sh, ffn_0, channel_token, ffn0_deps,
                            self._gather_q[0].handle, aux, handoff, self._ev_g0.handle)

        att_1 = self.attention.output_shard(rec_1.contiguous(), z_1.contiguous(), live_1, weights=attn_weights)
        mr1, mp1, mk1, ffn_1_sh = self.mlp_hc(att_1, r1, p1, k1)
        _dma_gather_enqueue(ffn_1_sh, ffn_1, channel_token, [ffn_0, ffn_1_sh],
                            self._gather_q[1].handle, aux, handoff, self._ev_g1.handle)

        att_2 = self.attention.output_shard(rec_2.contiguous(), z_2.contiguous(), live_2, weights=attn_weights)
        mr2, mp2, mk2, ffn_2_sh = self.mlp_hc(att_2, r2, p2, k2)
        _dma_gather_enqueue(ffn_2_sh, ffn_2, channel_token, [ffn_1, ffn_2_sh],
                            self._gather_q[2].handle, aux, handoff, self._ev_g2.handle)

        att_3 = self.attention.output_shard(rec_3.contiguous(), z_3.contiguous(), live_3, weights=attn_weights)
        self.attention.commit_state(s4, conv_all, metadata)
        mr3, mp3, mk3, ffn_3_sh = self.mlp_hc(att_3, r3, p3, k3)
        _dma_gather_enqueue(ffn_3_sh, ffn_3, channel_token, [ffn_2, ffn_3_sh],
                            self._gather_q[3].handle, aux, handoff, self._ev_g3.handle)

        # Phase 2b (~64 MB o_proj): enqueued after mlp_rs_3, hidden behind next layer's qkv_a (~2.4 ms).
        out, pend_b, pending_next_w = self._finish_pipelined_ffn_quarters(
            (ffn_0, ffn_1, ffn_2, ffn_3), (ffn_0_sh, ffn_1_sh, ffn_2_sh, ffn_3_sh),
            ((mr0, mp0, mk0), (mr1, mp1, mk1), (mr2, mp2, mk2), (mr3, mp3, mk3)),
            weight_gather=weight_gather, p_w2a=p_w2a, channel_token=channel_token,
        )
        return (out, pend_b, pending_next_w) if weight_gather is not None else (out, pend_b)

    def _forward_pipelined_row_shard_mla(self, incoming, metadata, *, pos_a, pos_b, pending_mlp_b,
                                         attn_weights=None, weight_gather=None, next_layer_index=None,
                                         pending_cur_o=None, channel_token=None):
        if self.mlp_quarters:
            return self._forward_pipelined_row_shard_mla_quarters(
                incoming, metadata, pos_a=pos_a, pos_b=pos_b, pending_mlp_b=pending_mlp_b,
                attn_weights=attn_weights, weight_gather=weight_gather, next_layer_index=next_layer_index,
                pending_cur_o=pending_cur_o, channel_token=channel_token)
        if self.has_carry:
            mlp_a, prev_res_a, prev_post_a, prev_comb_a, prev_res_b, prev_post_b, prev_comb_b = incoming
            res_a, post_a, comb_a, norm_a_sh = self.attention_hc(mlp_a, prev_res_a, prev_post_a, prev_comb_a)
            qres_a, idxq_a, win_a, cache_a_sh = self.attention.project_shard(norm_a_sh)
            absorbed_a = self.attention.prepare_query_shard(qres_a, weights=attn_weights)
            prev_cr, p_prev_b = pending_mlp_b
            mlp_b = prev_cr.finish(p_prev_b, absorbed_a)
            p_cache_a = self.cr_cache_gather_a.start(cache_a_sh, channel_token, dependencies=(mlp_b,))
            res_b, post_b, comb_b, norm_b_sh = self.attention_hc(mlp_b, prev_res_b, prev_post_b, prev_comb_b)
        else:
            src_a, src_b = incoming
            res_a, post_a, comb_a, norm_a_sh = self.attention_hc(src_a)
            qres_a, idxq_a, win_a, cache_a_sh = self.attention.project_shard(norm_a_sh)
            absorbed_a = self.attention.prepare_query_shard(qres_a, weights=attn_weights)
            p_cache_a = self.cr_cache_gather_a.start(cache_a_sh, channel_token, dependencies=(absorbed_a,))
            res_b, post_b, comb_b, norm_b_sh = self.attention_hc(src_b)
        qres_b, idxq_b, win_b, cache_b_sh = self.attention.project_shard(norm_b_sh)
        cache_a_full = self.cr_cache_gather_a.finish(p_cache_a, cache_b_sh)
        cache_b_full = self._sync_gather(
            self._cache_gather_b_binding, self.cache_gather_b, cache_b_sh, channel_token,
            dependencies=(cache_a_full,))

        if self.contiguous_shards:
            w = self.attention.context_ranks
            cache_all = torch.stack(
                (cache_a_full.view(w, self.shard_tokens, 768),
                 cache_b_full.view(w, self.shard_tokens, 768)), dim=1
            ).reshape(self.half_tokens * 2, 768)
        else:
            cache_all = torch.cat((cache_a_full, cache_b_full), dim=0)

        # Overlap QueryChain(B) behind cache writes + C4 pool gather, then overlap C4 selection(A+B)
        # on the auxiliary stream behind main-stream VarlenRing latent slab gather.
        p_qb = self.attention.start_query_shard(qres_b, dependencies=(cache_b_full,), weights=attn_weights)
        if p_qb is not None:
            _order_barrier(cache_all, [p_qb.absorbed])
        global_index_cache, end = self.attention.write_and_gather_pools(cache_all, metadata)
        absorbed_b = self.attention.finish_query_shard(p_qb, qres_b, global_index_cache, weights=attn_weights)

        p_sel = self.attention.start_select_shards(
            idxq_a, win_a, pos_a, idxq_b, win_b, pos_b, global_index_cache, dependencies=(absorbed_b,))
        if p_sel is not None:
            _order_barrier(end, [absorbed_b, p_sel.outputs[3]])
        else:
            _order_barrier(end, [global_index_cache])
        slab = self.attention.gather_slab(metadata, end)
        (phys_a, sel_a), (phys_b, sel_b) = self.attention.finish_select_shards(
            p_sel, idxq_a, win_a, pos_a, idxq_b, win_b, pos_b, global_index_cache, slab)

        if weight_gather is not None and pending_cur_o is not None:
            attn_weights = weight_gather.attach_o(attn_weights, pending_cur_o, dependencies=(phys_b, slab))
        # Overlap next layer's input weight prefetch (Phases 1a-2a, ~5.2 ms) behind sparse MLA att_a (~7.5 ms).
        p_w2a = (
            weight_gather.start_input_all(next_layer_index, [phys_b, slab], channel_token=channel_token)
            if weight_gather is not None and next_layer_index is not None else None
        )
        if p_w2a is not None:
            _order_barrier(slab, [channel_token])
        att_a = self.attention.attend_prepared_shard(absorbed_a, phys_a, sel_a, pos_a, slab, weights=attn_weights)
        mres_a, mpost_a, mcomb_a, ffn_a_sh = self.mlp_hc(att_a, res_a, post_a, comb_a)
        ffn_deps = (ffn_a_sh, absorbed_b, *p_w2a.buffers, *p_w2a.o_buffers) if p_w2a is not None else (ffn_a_sh, absorbed_b)
        p_ffn_a = self.cr_mlp_gather_a.start(ffn_a_sh, channel_token, dependencies=ffn_deps)
        att_b = self.attention.attend_prepared_shard(absorbed_b, phys_b, sel_b, pos_b, slab, weights=attn_weights)
        mres_b, mpost_b, mcomb_b, ffn_b_sh = self.mlp_hc(att_b, res_b, post_b, comb_b)
        ffn_a = self.cr_mlp_gather_a.finish(p_ffn_a, ffn_b_sh)

        out, pend_b, pending_next_w = self._finish_pipelined_ffn(
            ffn_a, ffn_b_sh, (mres_a, mpost_a, mcomb_a), (mres_b, mpost_b, mcomb_b),
            weight_gather=weight_gather, p_w2a=p_w2a, channel_token=channel_token)
        return (out, pend_b, pending_next_w) if weight_gather is not None else (out, pend_b)

    def _forward_pipelined_row_shard_mla_quarters(self, incoming, metadata, *, pos_a, pos_b, pending_mlp_b,
                                                  attn_weights=None, weight_gather=None, next_layer_index=None,
                                                  pending_cur_o=None, channel_token=None):
        if self.has_carry:
            mlp_0, mlp_1, mlp_2, c0, c1, c2, c3 = incoming
            r0, p0, k0, n0 = self.attention_hc(mlp_0, *c0)
            r1, p1, k1, n1 = self.attention_hc(mlp_1, *c1)
            norm_a_sh = torch.cat((n0, n1), dim=0)
            qres_a, idxq_a, win_a, cache_a_sh = self.attention.project_shard(norm_a_sh)
            r2, p2, k2, n2 = self.attention_hc(mlp_2, *c2)
            prev_ev_r3, mlp_3, prev_mlp_out_3 = pending_mlp_b
            _stream_event_wait(mlp_3, [prev_mlp_out_3, cache_a_sh, n2], prev_ev_r3.handle)
            p_cache_a = self.cr_cache_gather_a.start(cache_a_sh, channel_token, dependencies=(mlp_3,))
            r3, p3, k3, n3 = self.attention_hc(mlp_3, *c3)
            norm_b_sh = torch.cat((n2, n3), dim=0)
        else:
            src_0, src_1, src_2, src_3 = incoming
            r0, p0, k0, n0 = self.attention_hc(src_0)
            r1, p1, k1, n1 = self.attention_hc(src_1)
            norm_a_sh = torch.cat((n0, n1), dim=0)
            qres_a, idxq_a, win_a, cache_a_sh = self.attention.project_shard(norm_a_sh)
            p_cache_a = self.cr_cache_gather_a.start(cache_a_sh, channel_token, dependencies=(cache_a_sh,))
            r2, p2, k2, n2 = self.attention_hc(src_2)
            r3, p3, k3, n3 = self.attention_hc(src_3)
            norm_b_sh = torch.cat((n2, n3), dim=0)
        qres_b, idxq_b, win_b, cache_b_sh = self.attention.project_shard(norm_b_sh)
        cache_a_full = self.cr_cache_gather_a.finish(p_cache_a, cache_b_sh)

        # Overlap cache_gather_b behind q_b_proj(A) + query_absorption(A).
        p_cache_b = self.cr_cache_gather_b.start(cache_b_sh, channel_token, dependencies=(cache_a_full,))
        absorbed_a = self.attention.prepare_query_shard(qres_a, weights=attn_weights)
        cache_b_full = self.cr_cache_gather_b.finish(p_cache_b, absorbed_a)

        if self.contiguous_shards:
            w = self.attention.context_ranks
            cache_all = torch.stack(
                (cache_a_full.view(w, self.shard_tokens, 768),
                 cache_b_full.view(w, self.shard_tokens, 768)), dim=1
            ).reshape(self.half_tokens * 2, 768)
        else:
            cache_all = torch.cat((cache_a_full, cache_b_full), dim=0)

        p_qb = self.attention.start_query_shard(qres_b, dependencies=(cache_b_full,), weights=attn_weights)
        if p_qb is not None:
            _order_barrier(cache_all, [p_qb.absorbed])
        global_index_cache, end = self.attention.write_and_gather_pools(cache_all, metadata)
        absorbed_b = self.attention.finish_query_shard(p_qb, qres_b, global_index_cache, weights=attn_weights)

        p_sel = self.attention.start_select_shards(
            idxq_a, win_a, pos_a, idxq_b, win_b, pos_b, global_index_cache, dependencies=(absorbed_b,))
        if p_sel is not None:
            _order_barrier(end, [absorbed_b, p_sel.outputs[3]])
        else:
            _order_barrier(end, [global_index_cache])
        slab = self.attention.gather_slab(metadata, end)
        (phys_a, sel_a), (phys_b, sel_b) = self.attention.finish_select_shards(
            p_sel, idxq_a, win_a, pos_a, idxq_b, win_b, pos_b, global_index_cache, slab)

        q = self.q_shard_tokens
        aux = self.stream_handle
        handoff = self._ev_handoff.handle
        device = absorbed_a.device
        g_shape, g_stride, g_dtype = self._op_layout(self.mlp_gather_q[0])
        ffn_0 = torch.empty_strided(g_shape, g_stride, dtype=g_dtype, device=device)
        ffn_1 = torch.empty_strided(g_shape, g_stride, dtype=g_dtype, device=device)
        ffn_2 = torch.empty_strided(g_shape, g_stride, dtype=g_dtype, device=device)
        ffn_3 = torch.empty_strided(g_shape, g_stride, dtype=g_dtype, device=device)

        if weight_gather is not None and pending_cur_o is not None:
            attn_weights = weight_gather.attach_o(attn_weights, pending_cur_o, dependencies=(phys_b, slab))
        # Overlap next layer's input weight prefetch (Phases 1a-2a, ~5.2 ms) inside att_a (~7.5 ms idle DMA),
        # then overlap g0..g1 inside att_b (~7.5 ms), and Phase 2b o_proj after mlp_rs_3 behind next qkv_a.
        p_w2a = (
            weight_gather.start_input_all(next_layer_index, [phys_b, slab], channel_token=channel_token)
            if weight_gather is not None and next_layer_index is not None else None
        )
        if p_w2a is not None:
            _order_barrier(slab, [channel_token])
        att_a = self.attention.attend_prepared_shard(absorbed_a, phys_a, sel_a, pos_a, slab, weights=attn_weights)
        att_0, att_1 = att_a.split(q, dim=0)
        mr0, mp0, mk0, ffn_0_sh = self.mlp_hc(att_0.contiguous(), r0, p0, k0)
        ffn0_deps = ([phys_b, absorbed_b, *p_w2a.buffers, *p_w2a.o_buffers, ffn_0_sh]
                     if p_w2a is not None else [phys_b, absorbed_b, ffn_0_sh])
        _dma_gather_enqueue(ffn_0_sh, ffn_0, channel_token, ffn0_deps,
                            self._gather_q[0].handle, aux, handoff, self._ev_g0.handle)

        mr1, mp1, mk1, ffn_1_sh = self.mlp_hc(att_1.contiguous(), r1, p1, k1)
        _dma_gather_enqueue(ffn_1_sh, ffn_1, channel_token, [ffn_0, ffn_1_sh],
                            self._gather_q[1].handle, aux, handoff, self._ev_g1.handle)

        att_b = self.attention.attend_prepared_shard(absorbed_b, phys_b, sel_b, pos_b, slab, weights=attn_weights)
        att_2, att_3 = att_b.split(q, dim=0)
        mr2, mp2, mk2, ffn_2_sh = self.mlp_hc(att_2.contiguous(), r2, p2, k2)
        _dma_gather_enqueue(ffn_2_sh, ffn_2, channel_token, [ffn_1, ffn_2_sh],
                            self._gather_q[2].handle, aux, handoff, self._ev_g2.handle)

        mr3, mp3, mk3, ffn_3_sh = self.mlp_hc(att_3.contiguous(), r3, p3, k3)
        _dma_gather_enqueue(ffn_3_sh, ffn_3, channel_token, [ffn_2, ffn_3_sh],
                            self._gather_q[3].handle, aux, handoff, self._ev_g3.handle)

        out, pend_b, pending_next_w = self._finish_pipelined_ffn_quarters(
            (ffn_0, ffn_1, ffn_2, ffn_3), (ffn_0_sh, ffn_1_sh, ffn_2_sh, ffn_3_sh),
            ((mr0, mp0, mk0), (mr1, mp1, mk1), (mr2, mp2, mk2), (mr3, mp3, mk3)),
            weight_gather=weight_gather, p_w2a=p_w2a, channel_token=channel_token,
        )
        return (out, pend_b, pending_next_w) if weight_gather is not None else (out, pend_b)

    def _finish_pipelined_ffn_quarters(self, ffns, ffn_shs, carries, *,
                                       weight_gather=None, p_w2a=None, channel_token=None):
        ffn_0, ffn_1, ffn_2, ffn_3 = ffns
        ffn_0_sh, ffn_1_sh, ffn_2_sh, ffn_3_sh = ffn_shs
        c0, c1, c2, c3 = carries
        device = ffn_0.device
        aux = self.stream_handle
        handoff = self._ev_handoff.handle
        r_shape, r_stride, r_dtype = self._op_layout(self.mlp_reduce_q[0])

        mlp_0 = torch.empty_strided(r_shape, r_stride, dtype=r_dtype, device=device)
        mlp_1 = torch.empty_strided(r_shape, r_stride, dtype=r_dtype, device=device)
        mlp_2 = torch.empty_strided(r_shape, r_stride, dtype=r_dtype, device=device)
        mlp_3 = torch.empty_strided(r_shape, r_stride, dtype=r_dtype, device=device)

        _stream_event_wait(ffn_0, [ffn_0_sh, ffn_1, ffn_2, ffn_3], self._ev_g0.handle)
        mlp_out_0 = self.mlp(ffn_0)
        _dma_reduce_enqueue(mlp_out_0, mlp_0, channel_token, [ffn_3, mlp_out_0],
                            self._reduce_q[0].handle, aux, handoff, self._ev_r0.handle)

        _stream_event_wait(ffn_1, [mlp_0, mlp_out_0, ffn_1_sh], self._ev_g1.handle)
        mlp_out_1 = self.mlp(ffn_1)
        _dma_reduce_enqueue(mlp_out_1, mlp_1, channel_token, [mlp_0, mlp_out_1],
                            self._reduce_q[1].handle, aux, handoff, self._ev_r1.handle)

        # Wait for reduce_0 (already completed during mlp_out_1) so mlp_out_0 is freed before mlp_out_2.
        _stream_event_wait(mlp_0, [mlp_1, mlp_out_0], self._ev_r0.handle)
        _stream_event_wait(ffn_2, [mlp_0, ffn_2_sh], self._ev_g2.handle)
        mlp_out_2 = self.mlp(ffn_2)
        _dma_reduce_enqueue(mlp_out_2, mlp_2, channel_token, [mlp_1, mlp_out_2],
                            self._reduce_q[2].handle, aux, handoff, self._ev_r2.handle)

        # Wait for reduce_1 (already completed during mlp_out_2) so mlp_out_1 is freed before mlp_out_3.
        _stream_event_wait(mlp_1, [mlp_2, mlp_out_1], self._ev_r1.handle)
        _stream_event_wait(ffn_3, [mlp_1, ffn_3_sh], self._ev_g3.handle)
        mlp_out_3 = self.mlp(ffn_3)
        _dma_reduce_enqueue(mlp_out_3, mlp_3, channel_token, [mlp_2, mlp_out_3],
                            self._reduce_q[3].handle, aux, handoff, self._ev_r3.handle)

        pending_next_w = (
            weight_gather.start_phase2b(p_w2a, [mlp_3], channel_token=channel_token)
            if weight_gather is not None and p_w2a is not None else None
        )
        if pending_next_w is not None:
            _order_barrier(mlp_0, [channel_token])

        _stream_event_wait(mlp_2, [mlp_3, mlp_out_2], self._ev_r2.handle)
        if self.post is None:
            return (mlp_0, mlp_1, mlp_2, c0, c1, c2, c3), (self._ev_r3, mlp_3, mlp_out_3), pending_next_w
        out_0 = self.post(mlp_0, *c0)
        out_1 = self.post(mlp_1, *c1)
        out_2 = self.post(mlp_2, *c2)
        _stream_event_wait(mlp_3, [mlp_out_3, out_2], self._ev_r3.handle)
        out_3 = self.post(mlp_3, *c3)
        return (out_0, out_1, out_2, out_3), None, pending_next_w

    def _finish_pipelined_ffn(self, ffn_a, ffn_b_sh, carry_a, carry_b, *,
                              weight_gather=None, p_w2a=None, channel_token=None):
        mres_a, mpost_a, mcomb_a = carry_a
        mres_b, mpost_b, mcomb_b = carry_b
        p_ffn_b = self.cr_mlp_gather_b.start(ffn_b_sh, channel_token, dependencies=(ffn_a,))
        mlp_out_a = self.mlp(ffn_a)
        ffn_b = self.cr_mlp_gather_b.finish(p_ffn_b, mlp_out_a)

        p_mlp_a = self.cr_mlp_reduce_a.start(mlp_out_a, channel_token, dependencies=(ffn_b,))
        mlp_out_b = self.mlp(ffn_b)
        mlp_a = self.cr_mlp_reduce_a.finish(p_mlp_a, mlp_out_b)

        p_mlp_b = self.cr_mlp_reduce_b.start(mlp_out_b, channel_token, dependencies=(mlp_a,))
        pending_next_w = (
            weight_gather.start_phase2b(p_w2a, [mlp_out_b, mlp_a], channel_token=channel_token)
            if weight_gather is not None and p_w2a is not None else None
        )
        if pending_next_w is not None:
            _order_barrier(mlp_a, [channel_token])
        if self.post is None:
            return (mlp_a, mres_a, mpost_a, mcomb_a, mres_b, mpost_b, mcomb_b), (self.cr_mlp_reduce_b, p_mlp_b), pending_next_w
        out_a = self.post(mlp_a, mres_a, mpost_a, mcomb_a)
        mlp_b = self.cr_mlp_reduce_b.finish(p_mlp_b, out_a)
        out_b = self.post(mlp_b, mres_b, mpost_b, mcomb_b)
        return (out_a, out_b), None, pending_next_w
