# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""On-the-fly ZeRO-3 weight all-gather for row-sharded prefill without weight replication.

Keeps all 2D attention weights sharded 1/W across tensor-parallel ranks (identical to decode
storage) and gathers each layer's weights on demand into transient scratch reused across layers.
Supports BF16, MXFP8, Block-FP8, and mixed grouped linear weights across TP2 and TP4, over
4-phase zero-stall PCIe DMA prefetch and coalesced async RoCE RDMA overlap on a dedicated
non-pooled auxiliary stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import weakref

import torch

from ignition.components.collectives import PreparedGatherShards, _roce, _row_shard_capacity_bytes
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer

from .block_fp8 import BlockFP8Weight
from .weights import GroupedLinearWeights


_WEIGHT_OPS = weakref.WeakValueDictionary()
_WEIGHT_OP_HANDLES = itertools.count()
_WEIGHT_EVENTS = weakref.WeakValueDictionary()
_WEIGHT_EVENT_HANDLES = itertools.count()


class _WeightGatherOp:
    """Registry wrapper around a PreparedGatherShards plan for AOT handle rebinding."""

    def __init__(self, gather: PreparedGatherShards, src_shape: tuple[int, ...], dst_shape: tuple[int, ...]):
        self.gather = gather
        self.src_shape = src_shape
        self.dst_shape = dst_shape
        self.handle = next(_WEIGHT_OP_HANDLES)
        _WEIGHT_OPS[self.handle] = self


class _WeightEventSlot:
    """Lazily created persistent CUDA event whose `.handle` is rebound per rank by AOT."""

    def __init__(self):
        self.event = None
        self.handle = next(_WEIGHT_EVENT_HANDLES)
        _WEIGHT_EVENTS[self.handle] = self

    def get(self):
        if self.event is None:
            self.event = torch.cuda.Event()
        return self.event


@torch.library.custom_op("flare::weight_event_wait", mutates_args=("outputs",))
def _weight_event_wait(outputs: list[torch.Tensor], dependencies: list[torch.Tensor], ev_handle: int) -> None:
    """Wait on the main stream for an auxiliary weight-gather completion event.

    Mutates every tensor in ``outputs`` so Inductor cannot hoist any consumer of any
    waited weight buffer above the wait.
    """
    torch.cuda.current_stream(outputs[0].device).wait_event(_WEIGHT_EVENTS[ev_handle].get())


@_weight_event_wait.register_fake
def _fake_weight_event_wait(outputs, dependencies, ev_handle):
    pass


def _exec_gather(op: _WeightGatherOp, source: torch.Tensor, dst: torch.Tensor, col_row_bytes: int) -> None:
    src_bf16 = source.view(torch.bfloat16).reshape(op.src_shape)
    dst_bf16 = dst.view(torch.bfloat16).reshape(op.dst_shape)
    g = op.gather
    if g.single:
        dst_bf16.copy_(src_bf16)
    elif g.pcie:
        g.runtime.all_gather(src_bf16, out=dst_bf16.view(g.runtime.world_size, *src_bf16.shape), plan=g.plan)
    elif col_row_bytes:
        g.runtime.gather_shards(src_bf16, out=dst_bf16, plan=g.plan, col_row_bytes=col_row_bytes)
    else:
        g.runtime.gather_shards(src_bf16, out=dst_bf16, plan=g.plan)


@torch.library.custom_op("flare::weight_gather_slice", mutates_args=("output", "channel_token"))
def _weight_gather_slice(
    source: torch.Tensor,
    output: torch.Tensor,
    channel_token: torch.Tensor,
    row_start: int,
    row_end: int,
    op_handle: int,
    col_row_bytes: int,
) -> None:
    """Run a synchronous weight shard gather on the current stream with total channel ordering."""
    _exec_gather(_WEIGHT_OPS[op_handle], source, output[row_start:row_end], col_row_bytes)


@_weight_gather_slice.register_fake
def _fake_weight_gather_slice(source, output, channel_token, row_start, row_end, op_handle, col_row_bytes):
    pass


@torch.library.custom_op("flare::weight_dma_gather_slice", mutates_args=("output", "channel_token"))
def _weight_dma_gather_slice(
    source: torch.Tensor,
    output: torch.Tensor,
    channel_token: torch.Tensor,
    dependencies: list[torch.Tensor],
    row_start: int,
    row_end: int,
    op_handle: int,
    stream_handle: int,
    handoff_ev: int,
    done_ev: int,
    col_row_bytes: int,
) -> None:
    """Run an asynchronous weight shard gather on the auxiliary stream.

    When ``handoff_ev >= 0``, atomically records a main->aux event handoff inside this op so
    Inductor cannot insert main-stream work (or alias a freed pool buffer) between the handoff
    and the auxiliary write. ``channel_token`` enforces identical collective order across ranks.
    """
    main = torch.cuda.current_stream(source.device)
    aux = torch.cuda.ExternalStream(stream_handle, device=source.device)
    if handoff_ev >= 0:
        ev = _WEIGHT_EVENTS[handoff_ev].get()
        ev.record(main)
        aux.wait_event(ev)
    op = _WEIGHT_OPS[op_handle]
    dst = output[row_start:row_end]
    with torch.cuda.stream(aux):
        _exec_gather(op, source, dst, col_row_bytes)
        if done_ev >= 0:
            _WEIGHT_EVENTS[done_ev].get().record(aux)


@_weight_dma_gather_slice.register_fake
def _fake_weight_dma_gather_slice(
    source, output, channel_token, dependencies, row_start, row_end, op_handle, stream_handle,
    handoff_ev, done_ev, col_row_bytes,
):
    pass


def _is_mxfp8_weight(weight) -> bool:
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    return isinstance(weight, MXFP8LinearWeight)


def _mxfp8_scale_phys(weight) -> torch.Tensor:
    """Return the zero-copy contiguous [1, m_tiles, k_tiles, 32, 4, 4] byte view of scale_mma."""
    return weight.weight.scale_mma.permute(5, 2, 4, 0, 1, 3).view(torch.uint8)


def _mxfp8_from_parts(values: torch.Tensor, scale_phys_u8: torch.Tensor, *,
                      in_features: int, padded_in_features: int, out_features: int,
                      scale_rows: torch.Tensor | None = None):
    """Build an MXFP8LinearWeight from runtime tensors without data_ptr() checks."""
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    scale_mma = scale_phys_u8.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)
    return MXFP8LinearWeight(
        weight=MXFP8Rows(values=values, scale_rows=scale_rows, scale_mma=scale_mma),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


def _bf16_pack_shape(tensor: torch.Tensor) -> tuple[int, int]:
    """View any contiguous tensor with 16-byte aligned rows as 2D BF16 [rows, cols_bf16]."""
    if tensor.ndim < 1:
        raise ValueError("weight gather requires at least 1D tensors")
    rows = int(tensor.shape[0])
    row_bytes = (tensor.numel() // max(rows, 1)) * tensor.element_size()
    if row_bytes % 16 != 0:
        raise ValueError(f"weight gather row width must be 16-byte aligned, got {row_bytes} bytes")
    return (rows, row_bytes // 2)


def full_kda_template(sharded_attn: dict[str, object], *, world_size: int, device: torch.device) -> dict[str, object]:
    """Allocate a transient full 64-head KDA weight template for preparation and tuning only.

    Excludes already-replicated per-layer weights (``g_a_proj`` and ``o_norm.weight``) so merging
    ``{**layer_weights, **template}`` preserves every layer's own weights.
    """
    del world_size
    ch = 8192
    tpl: dict[str, object] = {
        "f_b_proj": torch.zeros((ch, 128), dtype=torch.bfloat16, device=device),
        "g_b_proj": torch.zeros((ch, 128), dtype=torch.bfloat16, device=device),
        "A_log": torch.zeros((64,), dtype=torch.float32, device=device),
        "dt_bias": torch.zeros((ch,), dtype=torch.float32, device=device),
        "conv1d.weight": torch.zeros((3 * ch, 4), dtype=torch.bfloat16, device=device),
    }
    inp = sharded_attn["input_proj"]
    if isinstance(inp, torch.Tensor):
        tpl["input_proj"] = torch.zeros((3 * ch + 64 + 128, 4096), dtype=inp.dtype, device=device)
    elif isinstance(inp, GroupedLinearWeights):
        p0, p1 = inp.parts
        if not _is_mxfp8_weight(p0) or not isinstance(p1, torch.Tensor):
            raise ValueError("KDA grouped input_proj template expects (MXFP8LinearWeight, BF16 Tensor)")
        vals = torch.zeros((3 * ch, p0.padded_in_features), dtype=torch.float8_e4m3fn, device=device)
        m_tiles = (3 * ch + 127) // 128
        k_tiles = p0.padded_in_features // 128
        sc_phys = torch.full((1, m_tiles, k_tiles, 32, 4, 4), 127, dtype=torch.uint8, device=device)
        sc_rows = torch.full((1, 3 * ch, p0.padded_in_features // 32), 127, dtype=torch.uint8, device=device)
        full_p0 = _mxfp8_from_parts(
            vals, sc_phys, in_features=p0.in_features, padded_in_features=p0.padded_in_features,
            out_features=3 * ch, scale_rows=sc_rows.view(torch.float8_e8m0fnu),
        )
        full_p1 = torch.zeros((64 + 128, 4096), dtype=p1.dtype, device=device)
        tpl["input_proj"] = GroupedLinearWeights(parts=(full_p0, full_p1), widths=(3 * ch, 192))
    else:
        raise ValueError(f"unsupported KDA input_proj weight type: {type(inp).__name__}")

    o = sharded_attn["o_proj"]
    if isinstance(o, torch.Tensor):
        tpl["o_proj"] = torch.zeros((4096, ch), dtype=o.dtype, device=device)
    elif _is_mxfp8_weight(o):
        vals = torch.zeros((4096, ch), dtype=torch.float8_e4m3fn, device=device)
        sc_phys = torch.full((1, 4096 // 128, ch // 128, 32, 4, 4), 127, dtype=torch.uint8, device=device)
        sc_rows = torch.full((1, 4096, ch // 32), 127, dtype=torch.uint8, device=device)
        tpl["o_proj"] = _mxfp8_from_parts(
            vals, sc_phys, in_features=ch, padded_in_features=ch, out_features=4096,
            scale_rows=sc_rows.view(torch.float8_e8m0fnu),
        )
    elif isinstance(o, BlockFP8Weight):
        vals = torch.zeros((4096, ch), dtype=torch.float8_e4m3fn, device=device)
        scales = torch.ones((4096 // 128, ch // 128), dtype=torch.float32, device=device)
        tpl["o_proj"] = BlockFP8Weight(values=vals, scales=scales)
    else:
        raise ValueError(f"unsupported KDA o_proj weight type: {type(o).__name__}")
    return tpl


def full_mla_template(sharded_attn: dict[str, object], *, world_size: int, device: torch.device) -> dict[str, object]:
    """Allocate a transient full 64-head MLA weight template for preparation and tuning only."""
    del world_size
    tpl: dict[str, object] = {}
    for key, (rows, cols) in (("q_b_proj", (16384, 1536)), ("kv_b_proj", (32768, 512)), ("o_proj", (4096, 16384))):
        w = sharded_attn[key]
        if isinstance(w, torch.Tensor):
            tpl[key] = torch.zeros((rows, cols), dtype=w.dtype, device=device)
        elif _is_mxfp8_weight(w):
            vals = torch.zeros((rows, cols), dtype=torch.float8_e4m3fn, device=device)
            sc_phys = torch.full((1, (rows + 127) // 128, cols // 128, 32, 4, 4), 127, dtype=torch.uint8, device=device)
            sc_rows = torch.full((1, rows, cols // 32), 127, dtype=torch.uint8, device=device)
            tpl[key] = _mxfp8_from_parts(
                vals, sc_phys, in_features=cols, padded_in_features=cols, out_features=rows,
                scale_rows=sc_rows.view(torch.float8_e8m0fnu),
            )
        elif isinstance(w, BlockFP8Weight):
            vals = torch.zeros((rows, cols), dtype=torch.float8_e4m3fn, device=device)
            scales = torch.ones((rows // 128, cols // 128), dtype=torch.float32, device=device)
            tpl[key] = BlockFP8Weight(values=vals, scales=scales)
        else:
            raise ValueError(f"unsupported MLA {key} weight type: {type(w).__name__}")
    return tpl


def _supports_roce_col_gather(runtime) -> bool:
    if runtime is None or not _roce(runtime):
        return False
    import inspect

    try:
        return "col_row_bytes" in inspect.signature(runtime.gather_shards).parameters
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class GatheredKdaWeights:
    """Transient full 64-head KDA weights reconstructed for a single layer."""

    input_proj: object
    f_b_proj: torch.Tensor
    g_b_proj: torch.Tensor
    o_proj: object | None
    conv1d: torch.Tensor
    dt_bias: torch.Tensor
    A_log: torch.Tensor


@dataclass(frozen=True)
class GatheredMlaWeights:
    """Transient full 64-head MLA weights reconstructed for a single layer."""

    q_b_proj: object
    absorb: object
    expand: object
    o_proj: object | None


@dataclass(frozen=True)
class PendingWeightGather:
    """In-flight async weight all-gather on the auxiliary stream."""

    layer_index: int
    kind: str
    buffers: tuple[torch.Tensor, ...]
    done_ev: _WeightEventSlot
    extra: tuple[torch.Tensor, ...] = ()
    o_buffers: tuple[torch.Tensor, ...] = ()
    o_done_ev: _WeightEventSlot | None = None


class AttentionWeightGatherer:
    """Shape-deduplicated ZeRO-3 weight all-gather manager for GLM-5.3 row-sharded prefill.

    All 45 decoder layers share PreparedGatherShards collective plans keyed by tensor shape
    (or a single 16-byte shape-agnostic plan on RoCE). Supports:
    - 4-phase zero-stall PCIe DMA prefetch (input weights ready before layer start, o_proj ready
      before output_shard/attend_prepared_shard).
    - Coalesced async RoCE RDMA overlap on ``workspace.auxiliary_stream`` behind MoE/attention
      compute, with direct 2D column-gather for ``o_proj`` and coalesced ``q/k/v`` gathers.
    - Total collective ordering via a shared ``channel_token`` mutated by every collective op.
    - Atomic main->aux event handoff inside ``flare::weight_dma_gather_slice`` for every
      newly-allocated buffer, and multi-buffer ``outputs`` mutation on ``flare::weight_event_wait``.
    """

    def __init__(self, layer_weights: list[dict[str, object]], layer_types: list[str], *,
                 world_size: int, runtime, coordinator, session, name: str,
                 workspace: ScratchArena | None = None, pipelined: bool = False,
                 device: torch.device, channel_token: torch.Tensor | None = None):
        if world_size not in (2, 4):
            raise ValueError(f"attention weight gather requires TP2 or TP4, got {world_size}")
        self.layer_weights = tuple(layer_weights)
        self.layer_types = tuple(layer_types)
        self.world = int(world_size)
        self.runtime = runtime
        self.device = device
        self.pipelined = bool(pipelined)
        self.is_roce = runtime is not None and _roce(runtime)
        self.use_roce_2d = _supports_roce_col_gather(runtime)
        self.max_gather_bytes = _row_shard_capacity_bytes(runtime) if runtime is not None else (1 << 30)
        self.async_roce = (
            (not self.pipelined)
            and self.is_roce
            and isinstance(workspace, ScratchArena)
            and workspace.overlap
        )
        # Always reuse workspace.auxiliary_stream (non-pooled high-priority stream) for overlap.
        if (self.pipelined or self.async_roce) and isinstance(workspace, ScratchArena) and workspace.overlap:
            self.stream_handle = workspace.auxiliary_stream.cuda_stream
        else:
            self.stream_handle = 0

        # Shared 4-byte token mutated by every collective to enforce identical FX order across ranks.
        self.channel_token = (
            channel_token if channel_token is not None
            else torch.zeros(1, dtype=torch.int32, device=device)
        )

        # On PCIe DMA, share one transient dummy pool across preparation declarations (released at
        # finalization). On RoCE, a single 16-byte PreparedGatherShards plan serves every shape.
        self._dummy_pool: torch.Tensor | None = (
            None if (runtime is None or self.is_roce)
            else torch.zeros(96 << 20, dtype=torch.bfloat16, device=device)
        )
        self._roce_gather: PreparedGatherShards | None = None
        if self.is_roce:
            src_tiny = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
            dst_tiny = torch.zeros((self.world, 8), dtype=torch.bfloat16, device=device)
            self._roce_gather = PreparedGatherShards(
                src_tiny, dst_tiny, runtime=runtime, coordinator=coordinator,
                session=session, name=f"{name}.wgather.roce",
            )

        self._ops: dict[tuple[int, int], _WeightGatherOp] = {}
        self._gathers: list[PreparedGatherShards] = (
            [self._roce_gather] if self._roce_gather is not None else []
        )

        n_layers = len(layer_weights)
        need_events = self.pipelined or self.async_roce
        self._ev_handoff = _WeightEventSlot() if need_events else None
        self._ev_inp_done = tuple(_WeightEventSlot() for _ in range(n_layers)) if need_events else ()
        self._ev_o_done = tuple(_WeightEventSlot() for _ in range(n_layers)) if need_events else ()

        # Precompute KDA small-pack layout: rows of width 4096 BF16 (8192 B, 16-byte aligned).
        ch = 8192 // world_size
        lh = 64 // world_size
        self._kda_ch = ch
        self._kda_lh = lh
        self._kda_r_b = lh
        self._kda_r_fb = ch * 128 // 4096
        self._kda_r_gb = ch * 128 // 4096
        self._kda_r_conv = 3 * ch * 4 // 4096
        self._kda_r_dt = ch * 2 // 4096
        self._kda_r_alog = 1
        self._kda_small_rows = (
            self._kda_r_b + self._kda_r_fb + self._kda_r_gb
            + self._kda_r_conv + self._kda_r_dt + self._kda_r_alog
        )

        # Register gather ops for EVERY layer (supports mixed-precision checkpoints cleanly).
        self._layer_ops: list[dict[str, object]] = []
        for ltype, attn in zip(self.layer_types, self.layer_weights, strict=True):
            if ltype == "linear_attention":
                self._layer_ops.append(self._declare_kda_shapes(
                    attn, runtime=runtime, coordinator=coordinator, session=session, name=name))
            elif ltype == "indexed_attention":
                self._layer_ops.append(self._declare_mla_shapes(
                    attn, runtime=runtime, coordinator=coordinator, session=session, name=name))
            else:
                self._layer_ops.append({})

        if session is not None:
            register_finalizer(session, self._release_dummy_pool)
            if (self.pipelined or self.async_roce) and isinstance(workspace, ScratchArena):
                register_finalizer(session, self._validate_zero_scratch)

    def _release_dummy_pool(self):
        self._dummy_pool = None

    def _validate_zero_scratch(self):
        for g in self._gathers:
            if not g.single and g.plan.memory_requirements().scratch_nbytes:
                raise ValueError("async attention weight gather requires zero-scratch collective plans")

    def register_parallel(self, workspace: ScratchArena, plans):
        if (not self.pipelined and not self.async_roce) or not isinstance(workspace, ScratchArena) or not workspace.overlap:
            return
        for g in self._gathers:
            if not g.single:
                for p in plans:
                    workspace.register_parallel(g.plan, p)

    def functional(self, *, session):
        for g in self._gathers:
            g.functional(session=session)
        return self

    def _get_or_create_op(self, src_shape: tuple[int, int], *, runtime, coordinator, session, name) -> _WeightGatherOp:
        if src_shape in self._ops:
            return self._ops[src_shape]
        dst_shape = (src_shape[0] * self.world, src_shape[1])
        if self._roce_gather is not None:
            op = _WeightGatherOp(self._roce_gather, src_shape, dst_shape)
            self._ops[src_shape] = op
            return op
        src_n = src_shape[0] * src_shape[1]
        dst_n = dst_shape[0] * dst_shape[1]
        src_aligned = (src_n + 7) & ~7
        if self._dummy_pool is not None and src_aligned + dst_n <= self._dummy_pool.numel():
            src_ex = self._dummy_pool[:src_n].view(src_shape)
            dst_ex = self._dummy_pool[src_aligned:src_aligned + dst_n].view(dst_shape)
        else:
            src_ex = torch.zeros(src_shape, dtype=torch.bfloat16, device=self.device)
            dst_ex = torch.zeros(dst_shape, dtype=torch.bfloat16, device=self.device)
        idx = len(self._ops)
        gather = PreparedGatherShards(
            src_ex, dst_ex, runtime=runtime, coordinator=coordinator,
            session=session, name=f"{name}.wgather.{idx}",
        )
        op = _WeightGatherOp(gather, src_shape, dst_shape)
        self._ops[src_shape] = op
        self._gathers.append(gather)
        return op

    def _declare_kda_shapes(self, attn, *, runtime, coordinator, session, name) -> dict[str, object]:
        ch, w = self._kda_ch, self.world
        spec: dict[str, object] = {}
        inp = attn["input_proj"]
        if isinstance(inp, torch.Tensor):
            spec["qkv"] = self._get_or_create_op((ch, 4096), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["qkv_sc"] = None
            row_bytes = 4096 * 2
            if self.use_roce_2d and 3 * ch * row_bytes * w <= self.max_gather_bytes:
                spec["qkv_coalesce"] = 3
                spec["qkv_bulk"] = self._get_or_create_op((3 * ch, 4096), runtime=runtime, coordinator=coordinator, session=session, name=name)
            elif self.use_roce_2d and 2 * ch * row_bytes * w <= self.max_gather_bytes:
                spec["qkv_coalesce"] = 2
                spec["qkv_bulk"] = self._get_or_create_op((2 * ch, 4096), runtime=runtime, coordinator=coordinator, session=session, name=name)
            else:
                spec["qkv_coalesce"] = 1
                spec["qkv_bulk"] = None
            spec["qkv_sc_bulk"] = None
        elif isinstance(inp, GroupedLinearWeights):
            p0 = inp.parts[0]
            cols_bf16 = p0.padded_in_features // 2
            spec["qkv"] = self._get_or_create_op((ch, cols_bf16), runtime=runtime, coordinator=coordinator, session=session, name=name)
            mt = ch // 128
            kt = p0.padded_in_features // 128
            spec["qkv_sc"] = self._get_or_create_op((mt, kt * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
            row_bytes = p0.padded_in_features
            if self.use_roce_2d and 3 * ch * row_bytes * w <= self.max_gather_bytes:
                spec["qkv_coalesce"] = 3
                spec["qkv_bulk"] = self._get_or_create_op((3 * ch, cols_bf16), runtime=runtime, coordinator=coordinator, session=session, name=name)
            elif self.use_roce_2d and 2 * ch * row_bytes * w <= self.max_gather_bytes:
                spec["qkv_coalesce"] = 2
                spec["qkv_bulk"] = self._get_or_create_op((2 * ch, cols_bf16), runtime=runtime, coordinator=coordinator, session=session, name=name)
            else:
                spec["qkv_coalesce"] = 1
                spec["qkv_bulk"] = None
            spec["qkv_sc_bulk"] = (
                self._get_or_create_op((3 * mt, kt * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
                if self.use_roce_2d and 3 * mt * kt * 512 * w <= self.max_gather_bytes else None
            )
        else:
            raise ValueError(f"unsupported KDA input_proj type: {type(inp).__name__}")

        spec["small"] = self._get_or_create_op((self._kda_small_rows, 4096), runtime=runtime, coordinator=coordinator, session=session, name=name)

        o = attn["o_proj"]
        if isinstance(o, torch.Tensor):
            spec["o"] = self._get_or_create_op((4096, ch), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["o_sc"] = None
        elif _is_mxfp8_weight(o):
            spec["o"] = self._get_or_create_op((4096, ch // 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["o_sc"] = self._get_or_create_op((32, (ch // 128) * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
        elif isinstance(o, BlockFP8Weight):
            spec["o"] = self._get_or_create_op((4096, ch // 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["o_sc"] = self._get_or_create_op((32, (ch // 128) * 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported KDA o_proj type: {type(o).__name__}")
        return spec

    def _declare_mla_shapes(self, attn, *, runtime, coordinator, session, name) -> dict[str, object]:
        w = self.world
        spec: dict[str, object] = {}
        qb = attn["q_b_proj"]
        if isinstance(qb, torch.Tensor):
            spec["qb"] = self._get_or_create_op((16384 // w, 1536), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["qb_sc"] = None
        elif isinstance(qb, BlockFP8Weight):
            spec["qb"] = self._get_or_create_op((16384 // w, 768), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["qb_sc"] = self._get_or_create_op((128 // w, 24), runtime=runtime, coordinator=coordinator, session=session, name=name)
        elif _is_mxfp8_weight(qb):
            spec["qb"] = self._get_or_create_op((16384 // w, 768), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["qb_sc"] = self._get_or_create_op((128 // w, 12 * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported MLA q_b_proj type: {type(qb).__name__}")

        kvb = attn["kv_b_proj"]
        if isinstance(kvb, torch.Tensor):
            spec["kvb"] = self._get_or_create_op((32768 // w, 512), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["kvb_sc"] = None
        elif _is_mxfp8_weight(kvb):
            spec["kvb"] = self._get_or_create_op((32768 // w, 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["kvb_sc"] = self._get_or_create_op((32768 // w, 8), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported MLA kv_b_proj type: {type(kvb).__name__}")

        o = attn["o_proj"]
        if isinstance(o, torch.Tensor):
            full_bytes = 4096 * 16384 * 2
            split = full_bytes > self.max_gather_bytes
            spec["mla_o_split"] = split
            o_rows = 2048 if split else 4096
            spec["o"] = self._get_or_create_op((o_rows, 16384 // w), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["o_sc"] = None
        elif isinstance(o, BlockFP8Weight):
            spec["mla_o_split"] = False
            spec["o"] = self._get_or_create_op((4096, 8192 // w), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["o_sc"] = self._get_or_create_op((32, (128 // w) * 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
        elif _is_mxfp8_weight(o):
            spec["mla_o_split"] = False
            spec["o"] = self._get_or_create_op((4096, 8192 // w), runtime=runtime, coordinator=coordinator, session=session, name=name)
            spec["o_sc"] = self._get_or_create_op((32, (128 // w) * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported MLA o_proj type: {type(o).__name__}")
        return spec

    def _pack_kda_small(self, attn: dict[str, object]) -> torch.Tensor:
        """Pack this rank's tiny KDA weights (b, f_b, g_b, conv1d, dt_bias, A_log) into [R, 4096] BF16."""
        ch, lh = self._kda_ch, self._kda_lh
        inp = attn["input_proj"]
        b_sh = inp[3 * ch:3 * ch + lh] if isinstance(inp, torch.Tensor) else inp.parts[1][:lh]
        fb_sh = attn["f_b_proj"].reshape(self._kda_r_fb, 4096)
        gb_sh = attn["g_b_proj"].reshape(self._kda_r_gb, 4096)
        conv_sh = attn["conv1d.weight"].reshape(self._kda_r_conv, 4096)
        dt_sh = attn["dt_bias"].contiguous().view(torch.bfloat16).reshape(self._kda_r_dt, 4096)
        alog_row = torch.zeros((1, 4096), dtype=torch.bfloat16, device=b_sh.device)
        alog_row[0, :lh * 2].copy_(attn["A_log"].contiguous().view(torch.bfloat16))
        return torch.cat((b_sh, fb_sh, gb_sh, conv_sh, dt_sh, alog_row), dim=0).contiguous()

    def _unpack_kda_o(self, layer_index: int, o_bufs: tuple[torch.Tensor, ...]):
        attn = self.layer_weights[layer_index]
        w, ch = self.world, self._kda_ch
        o_sh = attn["o_proj"]
        if self.use_roce_2d:
            if isinstance(o_sh, torch.Tensor):
                return o_bufs[0]
            o_vals, o_sc = o_bufs
            if _is_mxfp8_weight(o_sh):
                return _mxfp8_from_parts(
                    o_vals, o_sc.unsqueeze(0), in_features=8192, padded_in_features=8192, out_features=4096)
            return BlockFP8Weight(values=o_vals, scales=o_sc)
        if isinstance(o_sh, torch.Tensor):
            (o_staged,) = o_bufs
            return o_staged.view(w, 4096, ch).permute(1, 0, 2).reshape(4096, 8192).contiguous()
        o_staged, o_sc_staged = o_bufs
        if _is_mxfp8_weight(o_sh):
            o_vals = o_staged.view(w, 4096, ch).permute(1, 0, 2).reshape(4096, 8192).contiguous()
            kt = ch // 128
            o_sc = o_sc_staged.view(w, 32, kt, 32, 4, 4).permute(1, 0, 2, 3, 4, 5).reshape(
                1, 32, 64, 32, 4, 4).contiguous()
            return _mxfp8_from_parts(o_vals, o_sc, in_features=8192, padded_in_features=8192, out_features=4096)
        o_vals = o_staged.view(w, 4096, ch).permute(1, 0, 2).reshape(4096, 8192).contiguous()
        o_sc = o_sc_staged.view(w, 32, ch // 128).permute(1, 0, 2).reshape(32, 64).contiguous()
        return BlockFP8Weight(values=o_vals, scales=o_sc)

    def _unpack_kda_input(self, layer_index: int, bufs: tuple[torch.Tensor, ...]) -> GatheredKdaWeights:
        attn = self.layer_weights[layer_index]
        w, ch, lh = self.world, self._kda_ch, self._kda_lh
        r0 = self._kda_r_b
        r1 = r0 + self._kda_r_fb
        r2 = r1 + self._kda_r_gb
        r3 = r2 + self._kda_r_conv
        r4 = r3 + self._kda_r_dt

        inp = attn["input_proj"]
        if isinstance(inp, torch.Tensor):
            full_inp, small_full = bufs[:2]
            sf = small_full.view(w, self._kda_small_rows, 4096)
            b_full = sf[:, :r0].reshape(64, 4096)
            f_a = inp[3 * ch + lh:]
            full_inp[24576:24640].copy_(b_full)
            full_inp[24640:24768].copy_(f_a)
            input_proj = full_inp
        else:
            p0, p1 = inp.parts
            qkv_vals, qkv_sc, small_full = bufs[:3]
            sf = small_full.view(w, self._kda_small_rows, 4096)
            b_full = sf[:, :r0].reshape(64, 4096)
            f_a = p1[lh:]
            part1_full = torch.cat((b_full, f_a), dim=0).contiguous()
            part0_full = _mxfp8_from_parts(
                qkv_vals, qkv_sc.unsqueeze(0),
                in_features=p0.in_features, padded_in_features=p0.padded_in_features, out_features=24576,
            )
            input_proj = GroupedLinearWeights(parts=(part0_full, part1_full), widths=(24576, 192))

        fb_full = sf[:, r0:r1].reshape(8192, 128).contiguous()
        gb_full = sf[:, r1:r2].reshape(8192, 128).contiguous()
        conv_full = sf[:, r2:r3].reshape(w, 3, ch, 4).permute(1, 0, 2, 3).reshape(24576, 4).contiguous()
        dt_full = sf[:, r3:r4].reshape(8192 * 2).contiguous().view(torch.float32)
        alog_full = sf[:, r4, :lh * 2].contiguous().view(torch.float32).reshape(64)
        return GatheredKdaWeights(
            input_proj=input_proj, f_b_proj=fb_full, g_b_proj=gb_full, o_proj=None,
            conv1d=conv_full, dt_bias=dt_full, A_log=alog_full,
        )

    def _unpack_mla_o(self, layer_index: int, o_bufs: tuple[torch.Tensor, ...]):
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        w = self.world
        o_sh = attn["o_proj"]
        if self.use_roce_2d:
            if isinstance(o_sh, torch.Tensor):
                return o_bufs[0]
            o_v, o_s = o_bufs
            if isinstance(o_sh, BlockFP8Weight):
                return BlockFP8Weight(values=o_v, scales=o_s)
            return _mxfp8_from_parts(
                o_v, o_s.unsqueeze(0), in_features=16384, padded_in_features=16384, out_features=4096)
        if isinstance(o_sh, torch.Tensor):
            if spec["mla_o_split"]:
                h0, h1 = o_bufs
                o_top = h0.view(w, 2048, 16384 // w).permute(1, 0, 2).reshape(2048, 16384)
                o_bot = h1.view(w, 2048, 16384 // w).permute(1, 0, 2).reshape(2048, 16384)
                return torch.cat((o_top, o_bot), dim=0).contiguous()
            (o_st,) = o_bufs
            return o_st.view(w, 4096, 16384 // w).permute(1, 0, 2).reshape(4096, 16384).contiguous()
        o_v, o_s = o_bufs
        if isinstance(o_sh, BlockFP8Weight):
            o_vals = o_v.view(w, 4096, 16384 // w).permute(1, 0, 2).reshape(4096, 16384).contiguous()
            o_sc = o_s.view(w, 32, 128 // w).permute(1, 0, 2).reshape(32, 128).contiguous()
            return BlockFP8Weight(values=o_vals, scales=o_sc)
        o_vals = o_v.view(w, 4096, 16384 // w).permute(1, 0, 2).reshape(4096, 16384).contiguous()
        kt = 128 // w
        o_sc = o_s.view(w, 32, kt, 32, 4, 4).permute(1, 0, 2, 3, 4, 5).reshape(
            1, 32, 128, 32, 4, 4).contiguous()
        return _mxfp8_from_parts(o_vals, o_sc, in_features=16384, padded_in_features=16384, out_features=4096)

    def _unpack_mla_input(self, layer_index: int, bufs: tuple[torch.Tensor, ...]) -> GatheredMlaWeights:
        attn = self.layer_weights[layer_index]
        qb_sh, kvb_sh = attn["q_b_proj"], attn["kv_b_proj"]
        idx = 0
        if isinstance(qb_sh, torch.Tensor):
            qb_full = bufs[idx]; idx += 1
        elif isinstance(qb_sh, BlockFP8Weight):
            qb_full = BlockFP8Weight(values=bufs[idx], scales=bufs[idx + 1]); idx += 2
        else:
            qb_full = _mxfp8_from_parts(
                bufs[idx], bufs[idx + 1].unsqueeze(0), in_features=1536, padded_in_features=1536, out_features=16384)
            idx += 2

        if isinstance(kvb_sh, torch.Tensor):
            kvb_3d = bufs[idx].view(64, 512, 512)
            absorb_w, expand_w = kvb_3d[:, :256], kvb_3d[:, 256:]
        else:
            vals = bufs[idx].view(64, 512, 512)
            sc = bufs[idx + 1].view(torch.float8_e8m0fnu).view(64, 512, 16)
            absorb_w = (vals[:, :256], sc[:, :256])
            expand_w = (vals[:, 256:], sc[:, 256:])

        return GatheredMlaWeights(q_b_proj=qb_full, absorb=absorb_w, expand=expand_w, o_proj=None)

    def _gather_kda_o_sync(self, layer_index: int, tok: torch.Tensor) -> tuple[torch.Tensor, ...]:
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        dev, w, ch = self.device, self.world, self._kda_ch
        o_sh = attn["o_proj"]
        if self.use_roce_2d:
            if isinstance(o_sh, torch.Tensor):
                o_full = torch.empty((4096, 8192), dtype=torch.bfloat16, device=dev)
                _weight_gather_slice(o_sh, o_full, tok, 0, 4096, spec["o"].handle, ch * 2)
                return (o_full,)
            o_full = torch.empty((4096, 8192), dtype=torch.float8_e4m3fn, device=dev)
            if _is_mxfp8_weight(o_sh):
                o_sc = torch.empty((32, 64, 32, 4, 4), dtype=torch.uint8, device=dev)
                _weight_gather_slice(o_sh.weight.values, o_full, tok, 0, 4096, spec["o"].handle, ch)
                _weight_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_sc, tok, 0, 32, spec["o_sc"].handle, (ch // 128) * 512)
                return (o_full, o_sc)
            o_sc = torch.empty((32, 64), dtype=torch.float32, device=dev)
            _weight_gather_slice(o_sh.values, o_full, tok, 0, 4096, spec["o"].handle, ch)
            _weight_gather_slice(o_sh.scales, o_sc, tok, 0, 32, spec["o_sc"].handle, (ch // 128) * 4)
            return (o_full, o_sc)
        if isinstance(o_sh, torch.Tensor):
            o_staged = torch.empty((w * 4096, ch), dtype=torch.bfloat16, device=dev)
            _weight_gather_slice(o_sh, o_staged, tok, 0, o_staged.shape[0], spec["o"].handle, 0)
            return (o_staged,)
        o_staged = torch.empty((w * 4096, ch), dtype=torch.float8_e4m3fn, device=dev)
        if _is_mxfp8_weight(o_sh):
            o_sc_staged = torch.empty((w * 32, ch // 128, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_gather_slice(o_sh.weight.values, o_staged, tok, 0, o_staged.shape[0], spec["o"].handle, 0)
            _weight_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_sc_staged, tok, 0, o_sc_staged.shape[0], spec["o_sc"].handle, 0)
            return (o_staged, o_sc_staged)
        o_sc_staged = torch.empty((w * 32, ch // 128), dtype=torch.float32, device=dev)
        _weight_gather_slice(o_sh.values, o_staged, tok, 0, o_staged.shape[0], spec["o"].handle, 0)
        _weight_gather_slice(o_sh.scales, o_sc_staged, tok, 0, o_sc_staged.shape[0], spec["o_sc"].handle, 0)
        return (o_staged, o_sc_staged)

    def _gather_mla_o_sync(self, layer_index: int, tok: torch.Tensor) -> tuple[torch.Tensor, ...]:
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        dev, w = self.device, self.world
        o_sh = attn["o_proj"]
        if self.use_roce_2d:
            if isinstance(o_sh, torch.Tensor):
                o_full = torch.empty((4096, 16384), dtype=torch.bfloat16, device=dev)
                col_b = (16384 // w) * 2
                if spec["mla_o_split"]:
                    _weight_gather_slice(o_sh[:2048], o_full, tok, 0, 2048, spec["o"].handle, col_b)
                    _weight_gather_slice(o_sh[2048:], o_full, tok, 2048, 4096, spec["o"].handle, col_b)
                else:
                    _weight_gather_slice(o_sh, o_full, tok, 0, 4096, spec["o"].handle, col_b)
                return (o_full,)
            o_v = torch.empty((4096, 16384), dtype=torch.float8_e4m3fn, device=dev)
            if isinstance(o_sh, BlockFP8Weight):
                o_s = torch.empty((32, 128), dtype=torch.float32, device=dev)
                _weight_gather_slice(o_sh.values, o_v, tok, 0, 4096, spec["o"].handle, 16384 // w)
                _weight_gather_slice(o_sh.scales, o_s, tok, 0, 32, spec["o_sc"].handle, (128 // w) * 4)
                return (o_v, o_s)
            o_s = torch.empty((32, 128, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_gather_slice(o_sh.weight.values, o_v, tok, 0, 4096, spec["o"].handle, 16384 // w)
            _weight_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_s, tok, 0, 32, spec["o_sc"].handle, (128 // w) * 512)
            return (o_v, o_s)
        if isinstance(o_sh, torch.Tensor):
            if spec["mla_o_split"]:
                h0 = torch.empty((w * 2048, 16384 // w), dtype=torch.bfloat16, device=dev)
                h1 = torch.empty((w * 2048, 16384 // w), dtype=torch.bfloat16, device=dev)
                _weight_gather_slice(o_sh[:2048], h0, tok, 0, h0.shape[0], spec["o"].handle, 0)
                _weight_gather_slice(o_sh[2048:], h1, tok, 0, h1.shape[0], spec["o"].handle, 0)
                return (h0, h1)
            o_st = torch.empty((w * 4096, 16384 // w), dtype=torch.bfloat16, device=dev)
            _weight_gather_slice(o_sh, o_st, tok, 0, o_st.shape[0], spec["o"].handle, 0)
            return (o_st,)
        o_v = torch.empty((w * 4096, 16384 // w), dtype=torch.float8_e4m3fn, device=dev)
        if isinstance(o_sh, BlockFP8Weight):
            o_s = torch.empty((w * 32, 128 // w), dtype=torch.float32, device=dev)
            _weight_gather_slice(o_sh.values, o_v, tok, 0, o_v.shape[0], spec["o"].handle, 0)
            _weight_gather_slice(o_sh.scales, o_s, tok, 0, o_s.shape[0], spec["o_sc"].handle, 0)
            return (o_v, o_s)
        o_s = torch.empty((w * 32, 128 // w, 32, 4, 4), dtype=torch.uint8, device=dev)
        _weight_gather_slice(o_sh.weight.values, o_v, tok, 0, o_v.shape[0], spec["o"].handle, 0)
        _weight_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_s, tok, 0, o_s.shape[0], spec["o_sc"].handle, 0)
        return (o_v, o_s)

    def gather_sync(self, layer_index: int, *, channel_token: torch.Tensor | None = None) -> GatheredKdaWeights | GatheredMlaWeights:
        """Gather all weights for ``layer_index`` synchronously on the main stream."""
        ltype = self.layer_types[layer_index]
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        tok = self.channel_token if channel_token is None else channel_token
        dev, w = self.device, self.world

        if ltype == "linear_attention":
            ch = self._kda_ch
            inp = attn["input_proj"]
            small_sh = self._pack_kda_small(attn)
            small_full = torch.empty((w * self._kda_small_rows, 4096), dtype=torch.bfloat16, device=dev)
            if isinstance(inp, torch.Tensor):
                full_inp = torch.empty((24768, 4096), dtype=torch.bfloat16, device=dev)
                coal = spec["qkv_coalesce"]
                if coal == 3:
                    _weight_gather_slice(inp[0:3 * ch], full_inp, tok, 0, 24576, spec["qkv_bulk"].handle, ch * 4096 * 2)
                elif coal == 2:
                    _weight_gather_slice(inp[0:2 * ch], full_inp, tok, 0, 16384, spec["qkv_bulk"].handle, ch * 4096 * 2)
                    _weight_gather_slice(inp[2 * ch:3 * ch], full_inp, tok, 16384, 24576, spec["qkv"].handle, 0)
                else:
                    _weight_gather_slice(inp[0:ch], full_inp, tok, 0, 8192, spec["qkv"].handle, 0)
                    _weight_gather_slice(inp[ch:2 * ch], full_inp, tok, 8192, 16384, spec["qkv"].handle, 0)
                    _weight_gather_slice(inp[2 * ch:3 * ch], full_inp, tok, 16384, 24576, spec["qkv"].handle, 0)
                _weight_gather_slice(small_sh, small_full, tok, 0, small_full.shape[0], spec["small"].handle, 0)
                inp_w = self._unpack_kda_input(layer_index, (full_inp, small_full))
            else:
                p0 = inp.parts[0]
                p0_vals = p0.weight.values
                p0_sc = _mxfp8_scale_phys(p0)[0]
                mt = ch // 128
                kt = p0.padded_in_features // 128
                qkv_vals = torch.empty((24576, p0.padded_in_features), dtype=torch.float8_e4m3fn, device=dev)
                qkv_sc = torch.empty((192, kt, 32, 4, 4), dtype=torch.uint8, device=dev)
                coal = spec["qkv_coalesce"]
                if coal == 3:
                    _weight_gather_slice(p0_vals[0:3 * ch], qkv_vals, tok, 0, 24576, spec["qkv_bulk"].handle, ch * p0.padded_in_features)
                elif coal == 2:
                    _weight_gather_slice(p0_vals[0:2 * ch], qkv_vals, tok, 0, 16384, spec["qkv_bulk"].handle, ch * p0.padded_in_features)
                    _weight_gather_slice(p0_vals[2 * ch:3 * ch], qkv_vals, tok, 16384, 24576, spec["qkv"].handle, 0)
                else:
                    _weight_gather_slice(p0_vals[0:ch], qkv_vals, tok, 0, 8192, spec["qkv"].handle, 0)
                    _weight_gather_slice(p0_vals[ch:2 * ch], qkv_vals, tok, 8192, 16384, spec["qkv"].handle, 0)
                    _weight_gather_slice(p0_vals[2 * ch:3 * ch], qkv_vals, tok, 16384, 24576, spec["qkv"].handle, 0)
                if spec["qkv_sc_bulk"] is not None:
                    _weight_gather_slice(p0_sc[0:3 * mt], qkv_sc, tok, 0, 192, spec["qkv_sc_bulk"].handle, mt * kt * 512)
                else:
                    _weight_gather_slice(p0_sc[0:mt], qkv_sc, tok, 0, 64, spec["qkv_sc"].handle, 0)
                    _weight_gather_slice(p0_sc[mt:2 * mt], qkv_sc, tok, 64, 128, spec["qkv_sc"].handle, 0)
                    _weight_gather_slice(p0_sc[2 * mt:3 * mt], qkv_sc, tok, 128, 192, spec["qkv_sc"].handle, 0)
                _weight_gather_slice(small_sh, small_full, tok, 0, small_full.shape[0], spec["small"].handle, 0)
                inp_w = self._unpack_kda_input(layer_index, (qkv_vals, qkv_sc, small_full))
            o_full = self._unpack_kda_o(layer_index, self._gather_kda_o_sync(layer_index, tok))
            return GatheredKdaWeights(
                input_proj=inp_w.input_proj, f_b_proj=inp_w.f_b_proj, g_b_proj=inp_w.g_b_proj,
                o_proj=o_full, conv1d=inp_w.conv1d, dt_bias=inp_w.dt_bias, A_log=inp_w.A_log,
            )

        # MLA layer
        qb_sh, kvb_sh = attn["q_b_proj"], attn["kv_b_proj"]
        bufs: list[torch.Tensor] = []
        if isinstance(qb_sh, torch.Tensor):
            qb_full = torch.empty((16384, 1536), dtype=torch.bfloat16, device=dev)
            _weight_gather_slice(qb_sh, qb_full, tok, 0, 16384, spec["qb"].handle, 0)
            bufs.append(qb_full)
        elif isinstance(qb_sh, BlockFP8Weight):
            qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
            qb_s = torch.empty((128, 12), dtype=torch.float32, device=dev)
            _weight_gather_slice(qb_sh.values, qb_v, tok, 0, 16384, spec["qb"].handle, 0)
            _weight_gather_slice(qb_sh.scales, qb_s, tok, 0, 128, spec["qb_sc"].handle, 0)
            bufs.extend((qb_v, qb_s))
        else:
            qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
            qb_s = torch.empty((128, 12, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_gather_slice(qb_sh.weight.values, qb_v, tok, 0, 16384, spec["qb"].handle, 0)
            _weight_gather_slice(_mxfp8_scale_phys(qb_sh)[0], qb_s, tok, 0, 128, spec["qb_sc"].handle, 0)
            bufs.extend((qb_v, qb_s))

        if isinstance(kvb_sh, torch.Tensor):
            kvb_full = torch.empty((32768, 512), dtype=torch.bfloat16, device=dev)
            _weight_gather_slice(kvb_sh, kvb_full, tok, 0, 32768, spec["kvb"].handle, 0)
            bufs.append(kvb_full)
        else:
            kvb_v = torch.empty((32768, 512), dtype=torch.float8_e4m3fn, device=dev)
            kvb_s = torch.empty((32768, 16), dtype=torch.uint8, device=dev)
            _weight_gather_slice(kvb_sh.weight.values, kvb_v, tok, 0, 32768, spec["kvb"].handle, 0)
            _weight_gather_slice(kvb_sh.weight.scale_rows[0].view(torch.uint8), kvb_s, tok, 0, 32768, spec["kvb_sc"].handle, 0)
            bufs.extend((kvb_v, kvb_s))

        inp_w = self._unpack_mla_input(layer_index, tuple(bufs))
        o_full = self._unpack_mla_o(layer_index, self._gather_mla_o_sync(layer_index, tok))
        return GatheredMlaWeights(q_b_proj=inp_w.q_b_proj, absorb=inp_w.absorb, expand=inp_w.expand, o_proj=o_full)

    # ------------------------------------------------------------------
    # 4-phase zero-stall async prefetch (PCIe DMA + coalesced async RoCE)
    # ------------------------------------------------------------------

    def start_phase1a(self, layer_index: int, dependencies: list[torch.Tensor], *,
                      channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Phase 1a: KDA q_proj (+ q_sc) | MLA q_b_proj."""
        ltype = self.layer_types[layer_index]
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        tok = self.channel_token if channel_token is None else channel_token
        dev, aux, handoff = self.device, self.stream_handle, self._ev_handoff.handle
        ev_inp = self._ev_inp_done[layer_index]
        deps = list(dependencies)

        if ltype == "linear_attention":
            ch = self._kda_ch
            inp = attn["input_proj"]
            if isinstance(inp, torch.Tensor):
                full_inp = torch.empty((24768, 4096), dtype=torch.bfloat16, device=dev)
                _weight_dma_gather_slice(inp[0:ch], full_inp, tok, deps, 0, 8192, spec["qkv"].handle, aux, handoff, -1, 0)
                return PendingWeightGather(layer_index, ltype, (full_inp,), ev_inp)
            p0 = inp.parts[0]
            p0_vals = p0.weight.values
            p0_sc = _mxfp8_scale_phys(p0)[0]
            mt = ch // 128
            kt = p0.padded_in_features // 128
            qkv_vals = torch.empty((24576, p0.padded_in_features), dtype=torch.float8_e4m3fn, device=dev)
            qkv_sc = torch.empty((192, kt, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_dma_gather_slice(p0_vals[0:ch], qkv_vals, tok, deps, 0, 8192, spec["qkv"].handle, aux, handoff, -1, 0)
            _weight_dma_gather_slice(p0_sc[0:mt], qkv_sc, tok, [qkv_vals], 0, 64, spec["qkv_sc"].handle, aux, handoff, -1, 0)
            return PendingWeightGather(layer_index, ltype, (qkv_vals, qkv_sc), ev_inp)

        qb_sh = attn["q_b_proj"]
        if isinstance(qb_sh, torch.Tensor):
            qb_full = torch.empty((16384, 1536), dtype=torch.bfloat16, device=dev)
            _weight_dma_gather_slice(qb_sh, qb_full, tok, deps, 0, 16384, spec["qb"].handle, aux, handoff, -1, 0)
            return PendingWeightGather(layer_index, ltype, (qb_full,), ev_inp)
        if isinstance(qb_sh, BlockFP8Weight):
            qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
            qb_s = torch.empty((128, 12), dtype=torch.float32, device=dev)
            _weight_dma_gather_slice(qb_sh.values, qb_v, tok, deps, 0, 16384, spec["qb"].handle, aux, handoff, -1, 0)
            _weight_dma_gather_slice(qb_sh.scales, qb_s, tok, [qb_v], 0, 128, spec["qb_sc"].handle, aux, handoff, -1, 0)
            return PendingWeightGather(layer_index, ltype, (qb_v, qb_s), ev_inp)
        qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
        qb_s = torch.empty((128, 12, 32, 4, 4), dtype=torch.uint8, device=dev)
        _weight_dma_gather_slice(qb_sh.weight.values, qb_v, tok, deps, 0, 16384, spec["qb"].handle, aux, handoff, -1, 0)
        _weight_dma_gather_slice(_mxfp8_scale_phys(qb_sh)[0], qb_s, tok, [qb_v], 0, 128, spec["qb_sc"].handle, aux, handoff, -1, 0)
        return PendingWeightGather(layer_index, ltype, (qb_v, qb_s), ev_inp)

    def start_phase1b(self, pending1a: PendingWeightGather, dependencies: list[torch.Tensor], *,
                      channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Phase 1b: KDA k_proj + small | MLA kv_b_proj (signals ev_inp_done for MLA)."""
        layer_index = pending1a.layer_index
        ltype = pending1a.kind
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        tok = self.channel_token if channel_token is None else channel_token
        dev, w, aux, handoff = self.device, self.world, self.stream_handle, self._ev_handoff.handle
        ev_inp = self._ev_inp_done[layer_index]
        deps = [*dependencies, *pending1a.buffers]

        if ltype == "linear_attention":
            ch = self._kda_ch
            inp = attn["input_proj"]
            small_sh = self._pack_kda_small(attn)
            small_full = torch.empty((w * self._kda_small_rows, 4096), dtype=torch.bfloat16, device=dev)
            if isinstance(inp, torch.Tensor):
                (full_inp,) = pending1a.buffers
                _weight_dma_gather_slice(inp[ch:2 * ch], full_inp, tok, deps, 8192, 16384, spec["qkv"].handle, aux, -1, -1, 0)
                _weight_dma_gather_slice(small_sh, small_full, tok, [full_inp, small_sh], 0, small_full.shape[0],
                                         spec["small"].handle, aux, handoff, -1, 0)
                return PendingWeightGather(layer_index, ltype, (full_inp, small_full), ev_inp, extra=(small_sh,))
            p0 = inp.parts[0]
            p0_vals = p0.weight.values
            p0_sc = _mxfp8_scale_phys(p0)[0]
            mt = ch // 128
            qkv_vals, qkv_sc = pending1a.buffers
            _weight_dma_gather_slice(p0_vals[ch:2 * ch], qkv_vals, tok, deps, 8192, 16384, spec["qkv"].handle, aux, -1, -1, 0)
            _weight_dma_gather_slice(p0_sc[mt:2 * mt], qkv_sc, tok, [qkv_vals], 64, 128, spec["qkv_sc"].handle, aux, -1, -1, 0)
            _weight_dma_gather_slice(small_sh, small_full, tok, [qkv_sc, small_sh], 0, small_full.shape[0],
                                     spec["small"].handle, aux, handoff, -1, 0)
            return PendingWeightGather(layer_index, ltype, (qkv_vals, qkv_sc, small_full), ev_inp, extra=(small_sh,))

        kvb_sh = attn["kv_b_proj"]
        bufs = list(pending1a.buffers)
        if isinstance(kvb_sh, torch.Tensor):
            kvb_full = torch.empty((32768, 512), dtype=torch.bfloat16, device=dev)
            _weight_dma_gather_slice(kvb_sh, kvb_full, tok, deps, 0, 32768, spec["kvb"].handle, aux, handoff, ev_inp.handle, 0)
            bufs.append(kvb_full)
        else:
            kvb_v = torch.empty((32768, 512), dtype=torch.float8_e4m3fn, device=dev)
            kvb_s = torch.empty((32768, 16), dtype=torch.uint8, device=dev)
            _weight_dma_gather_slice(kvb_sh.weight.values, kvb_v, tok, deps, 0, 32768, spec["kvb"].handle, aux, handoff, -1, 0)
            _weight_dma_gather_slice(kvb_sh.weight.scale_rows[0].view(torch.uint8), kvb_s, tok, [kvb_v], 0, 32768,
                                     spec["kvb_sc"].handle, aux, handoff, ev_inp.handle, 0)
            bufs.extend((kvb_v, kvb_s))
        return PendingWeightGather(layer_index, ltype, tuple(bufs), ev_inp)

    def start_phase2a(self, pending1b: PendingWeightGather, dependencies: list[torch.Tensor], *,
                      channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Phase 2a: KDA v_proj (signals ev_inp_done) | MLA split o_proj h0."""
        layer_index = pending1b.layer_index
        ltype = pending1b.kind
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        tok = self.channel_token if channel_token is None else channel_token
        dev, w, aux, handoff = self.device, self.world, self.stream_handle, self._ev_handoff.handle
        ev_inp = self._ev_inp_done[layer_index]
        deps = [*dependencies, *pending1b.buffers]

        if ltype == "linear_attention":
            ch = self._kda_ch
            inp = attn["input_proj"]
            if isinstance(inp, torch.Tensor):
                full_inp, small_full = pending1b.buffers
                _weight_dma_gather_slice(inp[2 * ch:3 * ch], full_inp, tok, deps, 16384, 24576,
                                         spec["qkv"].handle, aux, -1, ev_inp.handle, 0)
                return PendingWeightGather(layer_index, ltype, (full_inp, small_full), ev_inp, extra=pending1b.extra)
            p0 = inp.parts[0]
            p0_vals = p0.weight.values
            p0_sc = _mxfp8_scale_phys(p0)[0]
            mt = ch // 128
            qkv_vals, qkv_sc, small_full = pending1b.buffers
            _weight_dma_gather_slice(p0_vals[2 * ch:3 * ch], qkv_vals, tok, deps, 16384, 24576,
                                     spec["qkv"].handle, aux, -1, -1, 0)
            _weight_dma_gather_slice(p0_sc[2 * mt:3 * mt], qkv_sc, tok, [qkv_vals], 128, 192,
                                     spec["qkv_sc"].handle, aux, -1, ev_inp.handle, 0)
            return PendingWeightGather(layer_index, ltype, (qkv_vals, qkv_sc, small_full), ev_inp, extra=pending1b.extra)

        o_sh = attn["o_proj"]
        if isinstance(o_sh, torch.Tensor) and spec["mla_o_split"] and not self.use_roce_2d:
            h0 = torch.empty((w * 2048, 16384 // w), dtype=torch.bfloat16, device=dev)
            _weight_dma_gather_slice(o_sh[:2048], h0, tok, deps, 0, h0.shape[0], spec["o"].handle, aux, handoff, -1, 0)
            return PendingWeightGather(
                layer_index, ltype, pending1b.buffers, ev_inp,
                extra=pending1b.extra, o_buffers=(h0,),
            )
        return pending1b

    def start_phase2b(self, pending2a: PendingWeightGather, dependencies: list[torch.Tensor], *,
                      channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Phase 2b: KDA o_proj | MLA o_proj (or h1 if split), signaling ev_o_done."""
        layer_index = pending2a.layer_index
        ltype = pending2a.kind
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        tok = self.channel_token if channel_token is None else channel_token
        dev, w, aux, handoff = self.device, self.world, self.stream_handle, self._ev_handoff.handle
        ev_o = self._ev_o_done[layer_index]
        deps = [*dependencies, *pending2a.buffers, *pending2a.o_buffers]

        if ltype == "linear_attention":
            ch = self._kda_ch
            o_sh = attn["o_proj"]
            if self.use_roce_2d:
                if isinstance(o_sh, torch.Tensor):
                    o_full = torch.empty((4096, 8192), dtype=torch.bfloat16, device=dev)
                    _weight_dma_gather_slice(o_sh, o_full, tok, deps, 0, 4096, spec["o"].handle, aux, handoff, ev_o.handle, ch * 2)
                    o_bufs: tuple[torch.Tensor, ...] = (o_full,)
                elif _is_mxfp8_weight(o_sh):
                    o_full = torch.empty((4096, 8192), dtype=torch.float8_e4m3fn, device=dev)
                    o_sc = torch.empty((32, 64, 32, 4, 4), dtype=torch.uint8, device=dev)
                    _weight_dma_gather_slice(o_sh.weight.values, o_full, tok, deps, 0, 4096, spec["o"].handle, aux, handoff, -1, ch)
                    _weight_dma_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_sc, tok, [o_full], 0, 32, spec["o_sc"].handle, aux, handoff, ev_o.handle, (ch // 128) * 512)
                    o_bufs = (o_full, o_sc)
                else:
                    o_full = torch.empty((4096, 8192), dtype=torch.float8_e4m3fn, device=dev)
                    o_sc = torch.empty((32, 64), dtype=torch.float32, device=dev)
                    _weight_dma_gather_slice(o_sh.values, o_full, tok, deps, 0, 4096, spec["o"].handle, aux, handoff, -1, ch)
                    _weight_dma_gather_slice(o_sh.scales, o_sc, tok, [o_full], 0, 32, spec["o_sc"].handle, aux, handoff, ev_o.handle, (ch // 128) * 4)
                    o_bufs = (o_full, o_sc)
            elif isinstance(o_sh, torch.Tensor):
                o_staged = torch.empty((w * 4096, ch), dtype=torch.bfloat16, device=dev)
                _weight_dma_gather_slice(o_sh, o_staged, tok, deps, 0, o_staged.shape[0], spec["o"].handle, aux, handoff, ev_o.handle, 0)
                o_bufs = (o_staged,)
            elif _is_mxfp8_weight(o_sh):
                o_staged = torch.empty((w * 4096, ch), dtype=torch.float8_e4m3fn, device=dev)
                o_sc_staged = torch.empty((w * 32, ch // 128, 32, 4, 4), dtype=torch.uint8, device=dev)
                _weight_dma_gather_slice(o_sh.weight.values, o_staged, tok, deps, 0, o_staged.shape[0], spec["o"].handle, aux, handoff, -1, 0)
                _weight_dma_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_sc_staged, tok, [o_staged], 0, o_sc_staged.shape[0],
                                         spec["o_sc"].handle, aux, handoff, ev_o.handle, 0)
                o_bufs = (o_staged, o_sc_staged)
            else:
                o_staged = torch.empty((w * 4096, ch), dtype=torch.float8_e4m3fn, device=dev)
                o_sc_staged = torch.empty((w * 32, ch // 128), dtype=torch.float32, device=dev)
                _weight_dma_gather_slice(o_sh.values, o_staged, tok, deps, 0, o_staged.shape[0], spec["o"].handle, aux, handoff, -1, 0)
                _weight_dma_gather_slice(o_sh.scales, o_sc_staged, tok, [o_staged], 0, o_sc_staged.shape[0],
                                         spec["o_sc"].handle, aux, handoff, ev_o.handle, 0)
                o_bufs = (o_staged, o_sc_staged)
            return PendingWeightGather(
                layer_index, ltype, pending2a.buffers, pending2a.done_ev,
                extra=pending2a.extra, o_buffers=o_bufs, o_done_ev=ev_o,
            )

        o_sh = attn["o_proj"]
        if self.use_roce_2d:
            if isinstance(o_sh, torch.Tensor):
                o_full = torch.empty((4096, 16384), dtype=torch.bfloat16, device=dev)
                col_b = (16384 // w) * 2
                if spec["mla_o_split"]:
                    _weight_dma_gather_slice(o_sh[:2048], o_full, tok, deps, 0, 2048, spec["o"].handle, aux, handoff, -1, col_b)
                    _weight_dma_gather_slice(o_sh[2048:], o_full, tok, [o_full], 2048, 4096, spec["o"].handle, aux, -1, ev_o.handle, col_b)
                else:
                    _weight_dma_gather_slice(o_sh, o_full, tok, deps, 0, 4096, spec["o"].handle, aux, handoff, ev_o.handle, col_b)
                o_bufs = (o_full,)
            elif isinstance(o_sh, BlockFP8Weight):
                o_v = torch.empty((4096, 16384), dtype=torch.float8_e4m3fn, device=dev)
                o_s = torch.empty((32, 128), dtype=torch.float32, device=dev)
                _weight_dma_gather_slice(o_sh.values, o_v, tok, deps, 0, 4096, spec["o"].handle, aux, handoff, -1, 16384 // w)
                _weight_dma_gather_slice(o_sh.scales, o_s, tok, [o_v], 0, 32, spec["o_sc"].handle, aux, handoff, ev_o.handle, (128 // w) * 4)
                o_bufs = (o_v, o_s)
            else:
                o_v = torch.empty((4096, 16384), dtype=torch.float8_e4m3fn, device=dev)
                o_s = torch.empty((32, 128, 32, 4, 4), dtype=torch.uint8, device=dev)
                _weight_dma_gather_slice(o_sh.weight.values, o_v, tok, deps, 0, 4096, spec["o"].handle, aux, handoff, -1, 16384 // w)
                _weight_dma_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_s, tok, [o_v], 0, 32, spec["o_sc"].handle, aux, handoff, ev_o.handle, (128 // w) * 512)
                o_bufs = (o_v, o_s)
        elif isinstance(o_sh, torch.Tensor):
            if spec["mla_o_split"]:
                (h0,) = pending2a.o_buffers
                h1 = torch.empty((w * 2048, 16384 // w), dtype=torch.bfloat16, device=dev)
                _weight_dma_gather_slice(o_sh[2048:], h1, tok, deps, 0, h1.shape[0], spec["o"].handle, aux, handoff, ev_o.handle, 0)
                o_bufs = (h0, h1)
            else:
                o_st = torch.empty((w * 4096, 16384 // w), dtype=torch.bfloat16, device=dev)
                _weight_dma_gather_slice(o_sh, o_st, tok, deps, 0, o_st.shape[0], spec["o"].handle, aux, handoff, ev_o.handle, 0)
                o_bufs = (o_st,)
        elif isinstance(o_sh, BlockFP8Weight):
            o_v = torch.empty((w * 4096, 16384 // w), dtype=torch.float8_e4m3fn, device=dev)
            o_s = torch.empty((w * 32, 128 // w), dtype=torch.float32, device=dev)
            _weight_dma_gather_slice(o_sh.values, o_v, tok, deps, 0, o_v.shape[0], spec["o"].handle, aux, handoff, -1, 0)
            _weight_dma_gather_slice(o_sh.scales, o_s, tok, [o_v], 0, o_s.shape[0], spec["o_sc"].handle, aux, handoff, ev_o.handle, 0)
            o_bufs = (o_v, o_s)
        else:
            o_v = torch.empty((w * 4096, 16384 // w), dtype=torch.float8_e4m3fn, device=dev)
            o_s = torch.empty((w * 32, 128 // w, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_dma_gather_slice(o_sh.weight.values, o_v, tok, deps, 0, o_v.shape[0], spec["o"].handle, aux, handoff, -1, 0)
            _weight_dma_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_s, tok, [o_v], 0, o_s.shape[0], spec["o_sc"].handle, aux, handoff, ev_o.handle, 0)
            o_bufs = (o_v, o_s)
        return PendingWeightGather(
            layer_index, ltype, pending2a.buffers, pending2a.done_ev,
            extra=pending2a.extra, o_buffers=o_bufs, o_done_ev=ev_o,
        )

    def start_input_all(self, layer_index: int, dependencies: list[torch.Tensor], *,
                        channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Enqueue all input weights (Phases 1a-2a) back-to-back on the auxiliary stream.

        On RoCE with 2D column-gather support, coalesces KDA ``q/k/v`` and ``q_sc/k_sc/v_sc``
        into the fewest possible large gathers to prevent persistent-MoE SM starvation.
        """
        ltype = self.layer_types[layer_index]
        if self.use_roce_2d and ltype == "linear_attention":
            attn = self.layer_weights[layer_index]
            spec = self._layer_ops[layer_index]
            tok = self.channel_token if channel_token is None else channel_token
            dev, w, ch = self.device, self.world, self._kda_ch
            aux, handoff = self.stream_handle, self._ev_handoff.handle
            ev_inp = self._ev_inp_done[layer_index]
            deps = list(dependencies)
            inp = attn["input_proj"]
            small_sh = self._pack_kda_small(attn)
            small_full = torch.empty((w * self._kda_small_rows, 4096), dtype=torch.bfloat16, device=dev)
            if isinstance(inp, torch.Tensor):
                full_inp = torch.empty((24768, 4096), dtype=torch.bfloat16, device=dev)
                coal = spec["qkv_coalesce"]
                if coal == 3:
                    _weight_dma_gather_slice(inp[0:3 * ch], full_inp, tok, deps, 0, 24576,
                                             spec["qkv_bulk"].handle, aux, handoff, -1, ch * 4096 * 2)
                elif coal == 2:
                    _weight_dma_gather_slice(inp[0:2 * ch], full_inp, tok, deps, 0, 16384,
                                             spec["qkv_bulk"].handle, aux, handoff, -1, ch * 4096 * 2)
                    _weight_dma_gather_slice(inp[2 * ch:3 * ch], full_inp, tok, [full_inp], 16384, 24576,
                                             spec["qkv"].handle, aux, -1, -1, 0)
                else:
                    _weight_dma_gather_slice(inp[0:ch], full_inp, tok, deps, 0, 8192, spec["qkv"].handle, aux, handoff, -1, 0)
                    _weight_dma_gather_slice(inp[ch:2 * ch], full_inp, tok, [full_inp], 8192, 16384, spec["qkv"].handle, aux, -1, -1, 0)
                    _weight_dma_gather_slice(inp[2 * ch:3 * ch], full_inp, tok, [full_inp], 16384, 24576, spec["qkv"].handle, aux, -1, -1, 0)
                _weight_dma_gather_slice(small_sh, small_full, tok, [full_inp, small_sh], 0, small_full.shape[0],
                                         spec["small"].handle, aux, handoff, ev_inp.handle, 0)
                return PendingWeightGather(layer_index, ltype, (full_inp, small_full), ev_inp, extra=(small_sh,))
            p0 = inp.parts[0]
            p0_vals = p0.weight.values
            p0_sc = _mxfp8_scale_phys(p0)[0]
            mt = ch // 128
            kt = p0.padded_in_features // 128
            qkv_vals = torch.empty((24576, p0.padded_in_features), dtype=torch.float8_e4m3fn, device=dev)
            qkv_sc = torch.empty((192, kt, 32, 4, 4), dtype=torch.uint8, device=dev)
            coal = spec["qkv_coalesce"]
            if coal == 3:
                _weight_dma_gather_slice(p0_vals[0:3 * ch], qkv_vals, tok, deps, 0, 24576,
                                         spec["qkv_bulk"].handle, aux, handoff, -1, ch * p0.padded_in_features)
            elif coal == 2:
                _weight_dma_gather_slice(p0_vals[0:2 * ch], qkv_vals, tok, deps, 0, 16384,
                                         spec["qkv_bulk"].handle, aux, handoff, -1, ch * p0.padded_in_features)
                _weight_dma_gather_slice(p0_vals[2 * ch:3 * ch], qkv_vals, tok, [qkv_vals], 16384, 24576,
                                         spec["qkv"].handle, aux, -1, -1, 0)
            else:
                _weight_dma_gather_slice(p0_vals[0:ch], qkv_vals, tok, deps, 0, 8192, spec["qkv"].handle, aux, handoff, -1, 0)
                _weight_dma_gather_slice(p0_vals[ch:2 * ch], qkv_vals, tok, [qkv_vals], 8192, 16384, spec["qkv"].handle, aux, -1, -1, 0)
                _weight_dma_gather_slice(p0_vals[2 * ch:3 * ch], qkv_vals, tok, [qkv_vals], 16384, 24576, spec["qkv"].handle, aux, -1, -1, 0)
            if spec["qkv_sc_bulk"] is not None:
                _weight_dma_gather_slice(p0_sc[0:3 * mt], qkv_sc, tok, [qkv_vals], 0, 192,
                                         spec["qkv_sc_bulk"].handle, aux, handoff, -1, mt * kt * 512)
            else:
                _weight_dma_gather_slice(p0_sc[0:mt], qkv_sc, tok, [qkv_vals], 0, 64, spec["qkv_sc"].handle, aux, handoff, -1, 0)
                _weight_dma_gather_slice(p0_sc[mt:2 * mt], qkv_sc, tok, [qkv_sc], 64, 128, spec["qkv_sc"].handle, aux, -1, -1, 0)
                _weight_dma_gather_slice(p0_sc[2 * mt:3 * mt], qkv_sc, tok, [qkv_sc], 128, 192, spec["qkv_sc"].handle, aux, -1, -1, 0)
            _weight_dma_gather_slice(small_sh, small_full, tok, [qkv_sc, small_sh], 0, small_full.shape[0],
                                     spec["small"].handle, aux, handoff, ev_inp.handle, 0)
            return PendingWeightGather(layer_index, ltype, (qkv_vals, qkv_sc, small_full), ev_inp, extra=(small_sh,))

        p1a = self.start_phase1a(layer_index, dependencies, channel_token=channel_token)
        p1b = self.start_phase1b(p1a, [], channel_token=channel_token)
        return self.start_phase2a(p1b, [], channel_token=channel_token)

    def start_o_only(self, layer_index: int, dependencies: list[torch.Tensor], *,
                     channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Enqueue only ``o_proj`` for ``layer_index`` on the auxiliary stream (used by async RoCE)."""
        ltype = self.layer_types[layer_index]
        dummy = PendingWeightGather(layer_index, ltype, (), self._ev_inp_done[layer_index])
        if (not self.use_roce_2d and ltype == "indexed_attention"
                and self._layer_ops[layer_index].get("mla_o_split")):
            dummy = self.start_phase2a(dummy, dependencies, channel_token=channel_token)
            return self.start_phase2b(dummy, [], channel_token=channel_token)
        return self.start_phase2b(dummy, dependencies, channel_token=channel_token)

    def start_full(self, layer_index: int, dependencies: list[torch.Tensor], *,
                   channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Enqueue both input weights and ``o_proj`` back-to-back on the auxiliary stream."""
        p2a = self.start_input_all(layer_index, dependencies, channel_token=channel_token)
        return self.start_phase2b(p2a, [], channel_token=channel_token)

    def finish_input(self, pending: PendingWeightGather, dependencies: list[torch.Tensor] = ()) -> GatheredKdaWeights | GatheredMlaWeights:
        """Wait for input weights (Phases 1a-2a) on main stream while o_proj stays in flight."""
        _weight_event_wait(list(pending.buffers), [*dependencies, *pending.extra], pending.done_ev.handle)
        if pending.kind == "linear_attention":
            return self._unpack_kda_input(pending.layer_index, pending.buffers)
        return self._unpack_mla_input(pending.layer_index, pending.buffers)

    def attach_o(
        self,
        weights: GatheredKdaWeights | GatheredMlaWeights,
        pending: PendingWeightGather,
        dependencies: list[torch.Tensor] = (),
    ) -> GatheredKdaWeights | GatheredMlaWeights:
        """Wait for Phase 2b ``o_proj`` on main stream and attach it to ``weights``."""
        if pending.o_done_ev is None or not pending.o_buffers:
            raise ValueError("attach_o requires a pending weight gather with in-flight o_proj buffers")
        _weight_event_wait(list(pending.o_buffers), list(dependencies), pending.o_done_ev.handle)
        if pending.kind == "linear_attention":
            o_full = self._unpack_kda_o(pending.layer_index, pending.o_buffers)
            return GatheredKdaWeights(
                input_proj=weights.input_proj, f_b_proj=weights.f_b_proj, g_b_proj=weights.g_b_proj,
                o_proj=o_full, conv1d=weights.conv1d, dt_bias=weights.dt_bias, A_log=weights.A_log,
            )
        o_full = self._unpack_mla_o(pending.layer_index, pending.o_buffers)
        return GatheredMlaWeights(
            q_b_proj=weights.q_b_proj, absorb=weights.absorb, expand=weights.expand, o_proj=o_full,
        )

    def finish(self, pending: PendingWeightGather, dependencies: list[torch.Tensor] = ()) -> GatheredKdaWeights | GatheredMlaWeights:
        """Wait for both input weights and ``o_proj`` to complete on main stream and unpack."""
        inp_w = self.finish_input(pending, dependencies=dependencies)
        if pending.o_done_ev is not None and pending.o_buffers:
            return self.attach_o(inp_w, pending, dependencies=dependencies)
        return inp_w
