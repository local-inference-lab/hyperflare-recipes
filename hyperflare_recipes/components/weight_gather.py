# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""On-the-fly ZeRO-3 weight all-gather for row-sharded prefill without weight replication.

Keeps all 2D attention weights sharded 1/W across tensor-parallel ranks (identical to decode
storage) and gathers each layer\x27s weights on demand into transient scratch reused across layers.
Supports BF16, MXFP8, Block-FP8, and mixed grouped linear weights across TP2 and TP4, over
both serial RoCE/DMA gathers and layer-pipelined PCIe DMA prefetch on an auxiliary stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import weakref

import torch

from ignition.components.collectives import PreparedGatherShards, _row_shard_capacity_bytes
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer

from .block_fp8 import BlockFP8Weight
from .weights import GroupedLinearWeights


_WEIGHT_OPS = weakref.WeakValueDictionary()
_WEIGHT_OP_HANDLES = itertools.count()
_WEIGHT_EVENTS = weakref.WeakValueDictionary()
_WEIGHT_EVENT_HANDLES = itertools.count()


class _WeightGatherOp:
    """Registry wrapper around a shape-keyed PreparedGatherShards for AOT handle rebinding."""

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


@torch.library.custom_op("flare::weight_event_record", mutates_args=("anchor",))
def _weight_event_record(anchor: torch.Tensor, dependencies: list[torch.Tensor], ev_handle: int) -> None:
    _WEIGHT_EVENTS[ev_handle].get().record(torch.cuda.current_stream(anchor.device))


@_weight_event_record.register_fake
def _fake_weight_event_record(anchor, dependencies, ev_handle):
    pass


@torch.library.custom_op("flare::weight_event_wait", mutates_args=("output",))
def _weight_event_wait(output: torch.Tensor, dependencies: list[torch.Tensor], ev_handle: int) -> None:
    torch.cuda.current_stream(output.device).wait_event(_WEIGHT_EVENTS[ev_handle].get())


@_weight_event_wait.register_fake
def _fake_weight_event_wait(output, dependencies, ev_handle):
    pass


@torch.library.custom_op("flare::weight_gather_slice", mutates_args=("output",))
def _weight_gather_slice(
    source: torch.Tensor,
    output: torch.Tensor,
    row_start: int,
    row_end: int,
    op_handle: int,
) -> None:
    op = _WEIGHT_OPS[op_handle]
    dst = output[row_start:row_end]
    src_bf16 = source.view(torch.bfloat16).reshape(op.src_shape)
    dst_bf16 = dst.view(torch.bfloat16).reshape(op.dst_shape)
    g = op.gather
    if g.single:
        dst_bf16.copy_(src_bf16)
    elif g.dma:
        g.runtime.all_gather(src_bf16, out=dst_bf16.view(g.runtime.world_size, *src_bf16.shape), plan=g.plan)
    else:
        g.runtime.gather_shards(src_bf16, out=dst_bf16, plan=g.plan)


@_weight_gather_slice.register_fake
def _fake_weight_gather_slice(source, output, row_start, row_end, op_handle):
    pass


@torch.library.custom_op("flare::weight_dma_gather_slice", mutates_args=("output",))
def _weight_dma_gather_slice(
    source: torch.Tensor,
    output: torch.Tensor,
    dependencies: list[torch.Tensor],
    row_start: int,
    row_end: int,
    op_handle: int,
    stream_handle: int,
    wait_ev: int,
    done_ev: int,
) -> None:
    aux = torch.cuda.ExternalStream(stream_handle, device=source.device)
    if wait_ev >= 0:
        aux.wait_event(_WEIGHT_EVENTS[wait_ev].get())
    op = _WEIGHT_OPS[op_handle]
    dst = output[row_start:row_end]
    src_bf16 = source.view(torch.bfloat16).reshape(op.src_shape)
    dst_bf16 = dst.view(torch.bfloat16).reshape(op.dst_shape)
    g = op.gather
    with torch.cuda.stream(aux):
        g.runtime.all_gather(src_bf16, out=dst_bf16.view(g.runtime.world_size, *src_bf16.shape), plan=g.plan)
        if done_ev >= 0:
            _WEIGHT_EVENTS[done_ev].get().record(aux)


@_weight_dma_gather_slice.register_fake
def _fake_weight_dma_gather_slice(
    source, output, dependencies, row_start, row_end, op_handle, stream_handle, wait_ev, done_ev
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
    """Allocate a transient full 64-head KDA weight template for preparation and tuning only."""
    w = world_size
    ch = 8192
    tpl: dict[str, object] = {
        "g_a_proj": sharded_attn["g_a_proj"],
        "o_norm.weight": sharded_attn["o_norm.weight"],
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
        tpl["o_proj"] = BlockFP8Weight(
            values=torch.zeros((4096, ch), dtype=torch.float8_e4m3fn, device=device),
            scales=torch.ones((4096 // 128, ch // 128), dtype=torch.float32, device=device),
        )
    else:
        raise ValueError(f"unsupported KDA o_proj weight type: {type(o).__name__}")
    del w
    return tpl


def full_mla_template(sharded_attn: dict[str, object], *, world_size: int, device: torch.device) -> dict[str, object]:
    """Allocate a transient full 64-head MLA weight template for preparation and tuning only."""
    del world_size
    tpl: dict[str, object] = {}
    qb = sharded_attn["q_b_proj"]
    if isinstance(qb, torch.Tensor):
        tpl["q_b_proj"] = torch.zeros((16384, 1536), dtype=qb.dtype, device=device)
    elif isinstance(qb, BlockFP8Weight):
        tpl["q_b_proj"] = BlockFP8Weight(
            values=torch.zeros((16384, 1536), dtype=torch.float8_e4m3fn, device=device),
            scales=torch.ones((16384 // 128, 1536 // 128), dtype=torch.float32, device=device),
        )
    elif _is_mxfp8_weight(qb):
        vals = torch.zeros((16384, 1536), dtype=torch.float8_e4m3fn, device=device)
        sc_phys = torch.full((1, 16384 // 128, 1536 // 128, 32, 4, 4), 127, dtype=torch.uint8, device=device)
        sc_rows = torch.full((1, 16384, 1536 // 32), 127, dtype=torch.uint8, device=device)
        tpl["q_b_proj"] = _mxfp8_from_parts(
            vals, sc_phys, in_features=1536, padded_in_features=1536, out_features=16384,
            scale_rows=sc_rows.view(torch.float8_e8m0fnu),
        )
    else:
        raise ValueError(f"unsupported MLA q_b_proj weight type: {type(qb).__name__}")

    kvb = sharded_attn["kv_b_proj"]
    if isinstance(kvb, torch.Tensor):
        tpl["kv_b_proj"] = torch.zeros((32768, 512), dtype=kvb.dtype, device=device)
    elif _is_mxfp8_weight(kvb):
        vals = torch.zeros((32768, 512), dtype=torch.float8_e4m3fn, device=device)
        sc_phys = torch.full((1, 32768 // 128, 512 // 128, 32, 4, 4), 127, dtype=torch.uint8, device=device)
        sc_rows = torch.full((1, 32768, 512 // 32), 127, dtype=torch.uint8, device=device)
        tpl["kv_b_proj"] = _mxfp8_from_parts(
            vals, sc_phys, in_features=512, padded_in_features=512, out_features=32768,
            scale_rows=sc_rows.view(torch.float8_e8m0fnu),
        )
    else:
        raise ValueError(f"unsupported MLA kv_b_proj weight type: {type(kvb).__name__}")

    o = sharded_attn["o_proj"]
    if isinstance(o, torch.Tensor):
        tpl["o_proj"] = torch.zeros((4096, 16384), dtype=o.dtype, device=device)
    elif isinstance(o, BlockFP8Weight):
        tpl["o_proj"] = BlockFP8Weight(
            values=torch.zeros((4096, 16384), dtype=torch.float8_e4m3fn, device=device),
            scales=torch.ones((4096 // 128, 16384 // 128), dtype=torch.float32, device=device),
        )
    elif _is_mxfp8_weight(o):
        vals = torch.zeros((4096, 16384), dtype=torch.float8_e4m3fn, device=device)
        sc_phys = torch.full((1, 4096 // 128, 16384 // 128, 32, 4, 4), 127, dtype=torch.uint8, device=device)
        sc_rows = torch.full((1, 4096, 16384 // 32), 127, dtype=torch.uint8, device=device)
        tpl["o_proj"] = _mxfp8_from_parts(
            vals, sc_phys, in_features=16384, padded_in_features=16384, out_features=4096,
            scale_rows=sc_rows.view(torch.float8_e8m0fnu),
        )
    else:
        raise ValueError(f"unsupported MLA o_proj weight type: {type(o).__name__}")
    return tpl


@dataclass(frozen=True)
class GatheredKdaWeights:
    input_proj: object
    f_b_proj: torch.Tensor
    g_b_proj: torch.Tensor
    o_proj: object
    conv1d: torch.Tensor
    dt_bias: torch.Tensor
    A_log: torch.Tensor


@dataclass(frozen=True)
class GatheredMlaWeights:
    q_b_proj: object
    absorb: object
    expand: object
    o_proj: object


@dataclass(frozen=True)
class PendingWeightGather:
    """In-flight async weight gather buffers on the auxiliary DMA stream."""

    layer_index: int
    kind: str
    buffers: tuple[torch.Tensor, ...]
    done_ev: _WeightEventSlot
    extra: tuple[object, ...] = ()


class AttentionWeightGatherer:
    """Shared shape-keyed ZeRO-3 weight all-gather pool for all GLM-5.3 attention layers.

    Declares at most one PreparedGatherShards per distinct transfer shape across all 45 layers
    and reuses the existing prefill collective runtime without adding persistent weights or IPC slabs.
    """

    def __init__(
        self,
        layer_weights: list[dict[str, object]],
        layer_types: list[str],
        *,
        world_size: int,
        runtime,
        coordinator,
        session,
        name: str,
        workspace: ScratchArena | None = None,
        pipelined: bool = False,
        device: torch.device,
    ):
        if world_size not in (2, 4):
            raise ValueError(f"attention weight gather requires TP2 or TP4, got {world_size}")
        self.layer_weights = tuple(layer_weights)
        self.layer_types = tuple(layer_types)
        self.world = world_size
        self.runtime = runtime
        self.device = device
        self.pipelined = bool(pipelined)
        self.max_gather_bytes = _row_shard_capacity_bytes(runtime) if runtime is not None else (1 << 30)
        self.stream_handle = (
            workspace.auxiliary_stream.cuda_stream
            if self.pipelined and isinstance(workspace, ScratchArena) and workspace.overlap
            else 0
        )

        # Deduplicate PreparedGatherShards by (rows, cols_bf16) so 45 layers share ~6-8 collective plans.
        self._ops: dict[tuple[int, int], _WeightGatherOp] = {}
        self._gathers: list[PreparedGatherShards] = []
        self._ev_small_slots = tuple(_WeightEventSlot() for _ in range(len(layer_weights))) if self.pipelined else ()
        self._ev_done_slots = tuple(_WeightEventSlot() for _ in range(len(layer_weights))) if self.pipelined else ()
        self._ev_p1_slots = tuple(_WeightEventSlot() for _ in range(len(layer_weights))) if self.pipelined else ()

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

        # Register all distinct gather shapes needed across all target layers.
        self._mla_o_split = False
        seen_kda = False
        seen_mla = False
        for ltype, attn in zip(self.layer_types, self.layer_weights, strict=True):
            if ltype == "linear_attention" and not seen_kda:
                seen_kda = True
                self._declare_kda_shapes(attn, runtime=runtime, coordinator=coordinator, session=session, name=name)
            elif ltype == "indexed_attention" and not seen_mla:
                seen_mla = True
                self._declare_mla_shapes(attn, runtime=runtime, coordinator=coordinator, session=session, name=name)

        if self.pipelined and isinstance(workspace, ScratchArena):
            register_finalizer(session, self._validate_zero_scratch)

    def _validate_zero_scratch(self):
        for g in self._gathers:
            if not g.single and g.plan.memory_requirements().scratch_nbytes:
                raise ValueError("pipelined attention weight gather requires zero-scratch DMA plans")

    def register_parallel(self, workspace: ScratchArena, plans):
        if not self.pipelined or not isinstance(workspace, ScratchArena) or not workspace.overlap:
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

    def _declare_kda_shapes(self, attn, *, runtime, coordinator, session, name):
        ch = self._kda_ch
        inp = attn["input_proj"]
        if isinstance(inp, torch.Tensor):
            self._op_kda_qkv = self._get_or_create_op(
                (ch, 4096), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_kda_qkv_sc = None
        elif isinstance(inp, GroupedLinearWeights):
            p0 = inp.parts[0]
            self._op_kda_qkv = self._get_or_create_op(
                (ch, p0.padded_in_features // 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
            mt = ch // 128
            kt = p0.padded_in_features // 128
            self._op_kda_qkv_sc = self._get_or_create_op(
                (mt, kt * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported KDA input_proj type: {type(inp).__name__}")

        self._op_kda_small = self._get_or_create_op(
            (self._kda_small_rows, 4096), runtime=runtime, coordinator=coordinator, session=session, name=name)

        o = attn["o_proj"]
        if isinstance(o, torch.Tensor):
            self._op_kda_o = self._get_or_create_op(
                (4096, ch), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_kda_o_sc = None
        elif _is_mxfp8_weight(o):
            self._op_kda_o = self._get_or_create_op(
                (4096, ch // 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
            kt = ch // 128
            self._op_kda_o_sc = self._get_or_create_op(
                (32, kt * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
        elif isinstance(o, BlockFP8Weight):
            self._op_kda_o = self._get_or_create_op(
                (4096, ch // 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_kda_o_sc = self._get_or_create_op(
                (32, (ch // 128) * 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported KDA o_proj type: {type(o).__name__}")

    def _declare_mla_shapes(self, attn, *, runtime, coordinator, session, name):
        w = self.world
        qb = attn["q_b_proj"]
        if isinstance(qb, torch.Tensor):
            self._op_mla_qb = self._get_or_create_op(
                (16384 // w, 1536), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_mla_qb_sc = None
        elif isinstance(qb, BlockFP8Weight):
            self._op_mla_qb = self._get_or_create_op(
                (16384 // w, 768), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_mla_qb_sc = self._get_or_create_op(
                (128 // w, 24), runtime=runtime, coordinator=coordinator, session=session, name=name)
        elif _is_mxfp8_weight(qb):
            self._op_mla_qb = self._get_or_create_op(
                (16384 // w, 768), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_mla_qb_sc = self._get_or_create_op(
                (128 // w, 12 * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported MLA q_b_proj type: {type(qb).__name__}")

        kvb = attn["kv_b_proj"]
        if isinstance(kvb, torch.Tensor):
            self._op_mla_kvb = self._get_or_create_op(
                (32768 // w, 512), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_mla_kvb_sc = None
        elif _is_mxfp8_weight(kvb):
            self._op_mla_kvb = self._get_or_create_op(
                (32768 // w, 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_mla_kvb_sc = self._get_or_create_op(
                (32768 // w, 8), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported MLA kv_b_proj type: {type(kvb).__name__}")

        o = attn["o_proj"]
        if isinstance(o, torch.Tensor):
            full_bytes = 4096 * 16384 * 2
            self._mla_o_split = full_bytes > self.max_gather_bytes
            o_rows = 2048 if self._mla_o_split else 4096
            self._op_mla_o = self._get_or_create_op(
                (o_rows, 16384 // w), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_mla_o_sc = None
        elif isinstance(o, BlockFP8Weight):
            self._mla_o_split = False
            self._op_mla_o = self._get_or_create_op(
                (4096, 8192 // w), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_mla_o_sc = self._get_or_create_op(
                (32, (128 // w) * 2), runtime=runtime, coordinator=coordinator, session=session, name=name)
        elif _is_mxfp8_weight(o):
            self._mla_o_split = False
            self._op_mla_o = self._get_or_create_op(
                (4096, 8192 // w), runtime=runtime, coordinator=coordinator, session=session, name=name)
            self._op_mla_o_sc = self._get_or_create_op(
                (32, (128 // w) * 256), runtime=runtime, coordinator=coordinator, session=session, name=name)
        else:
            raise ValueError(f"unsupported MLA o_proj type: {type(o).__name__}")

    def _pack_kda_small(self, attn: dict[str, object]) -> torch.Tensor:
        """Pack this rank\x27s tiny KDA weights (b, f_b, g_b, conv1d, dt_bias, A_log) into [R, 4096] BF16."""
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

    def _unpack_kda(self, layer_index: int, bufs: tuple[torch.Tensor, ...]) -> GatheredKdaWeights:
        attn = self.layer_weights[layer_index]
        w, ch, lh = self.world, self._kda_ch, self._kda_lh
        r0 = self._kda_r_b
        r1 = r0 + self._kda_r_fb
        r2 = r1 + self._kda_r_gb
        r3 = r2 + self._kda_r_conv
        r4 = r3 + self._kda_r_dt

        inp = attn["input_proj"]
        o_sh = attn["o_proj"]
        if isinstance(inp, torch.Tensor):
            full_inp, small_full, o_staged = bufs
            sf = small_full.view(w, self._kda_small_rows, 4096)
            b_full = sf[:, :r0].reshape(64, 4096)
            f_a = inp[3 * ch + lh:]
            full_inp[24576:24640].copy_(b_full)
            full_inp[24640:24768].copy_(f_a)
            input_proj = full_inp
        else:
            p0, p1 = inp.parts
            qkv_vals, qkv_sc, small_full, o_staged, o_sc_staged = bufs
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

        if isinstance(o_sh, torch.Tensor):
            o_full = o_staged.view(w, 4096, ch).permute(1, 0, 2).reshape(4096, 8192).contiguous()
        elif _is_mxfp8_weight(o_sh):
            o_vals = o_staged.view(w, 4096, ch).permute(1, 0, 2).reshape(4096, 8192).contiguous()
            kt = ch // 128
            o_sc = o_sc_staged.view(w, 32, kt, 32, 4, 4).permute(1, 0, 2, 3, 4, 5).reshape(
                1, 32, 64, 32, 4, 4).contiguous()
            o_full = _mxfp8_from_parts(o_vals, o_sc, in_features=8192, padded_in_features=8192, out_features=4096)
        else:
            o_vals = o_staged.view(w, 4096, ch).permute(1, 0, 2).reshape(4096, 8192).contiguous()
            o_sc = o_sc_staged.view(w, 32, ch // 128).permute(1, 0, 2).reshape(32, 64).contiguous()
            o_full = BlockFP8Weight(values=o_vals, scales=o_sc)

        return GatheredKdaWeights(
            input_proj=input_proj, f_b_proj=fb_full, g_b_proj=gb_full,
            o_proj=o_full, conv1d=conv_full, dt_bias=dt_full, A_log=alog_full,
        )

    def _unpack_mla(self, layer_index: int, bufs: tuple[torch.Tensor, ...]) -> GatheredMlaWeights:
        attn = self.layer_weights[layer_index]
        w = self.world
        qb_sh, kvb_sh, o_sh = attn["q_b_proj"], attn["kv_b_proj"], attn["o_proj"]
        idx = 0
        if isinstance(qb_sh, torch.Tensor):
            qb_full = bufs[idx]
            idx += 1
        elif isinstance(qb_sh, BlockFP8Weight):
            qb_full = BlockFP8Weight(values=bufs[idx], scales=bufs[idx + 1])
            idx += 2
        else:
            qb_full = _mxfp8_from_parts(
                bufs[idx], bufs[idx + 1].unsqueeze(0),
                in_features=1536, padded_in_features=1536, out_features=16384,
            )
            idx += 2

        if isinstance(kvb_sh, torch.Tensor):
            kvb_v = bufs[idx].view(64, 512, 512)
            absorb_w, expand_w = kvb_v[:, :256], kvb_v[:, 256:]
            idx += 1
        else:
            kv_vals = bufs[idx].view(64, 512, 512)
            kv_sc = bufs[idx + 1].view(torch.float8_e8m0fnu).view(64, 512, 16)
            absorb_w = (kv_vals[:, :256], kv_sc[:, :256])
            expand_w = (kv_vals[:, 256:], kv_sc[:, 256:])
            idx += 2

        if isinstance(o_sh, torch.Tensor):
            if self._mla_o_split:
                h0 = bufs[idx].view(w, 2048, 16384 // w).permute(1, 0, 2).reshape(2048, 16384)
                h1 = bufs[idx + 1].view(w, 2048, 16384 // w).permute(1, 0, 2).reshape(2048, 16384)
                o_full = torch.cat((h0, h1), dim=0).contiguous()
            else:
                o_full = bufs[idx].view(w, 4096, 16384 // w).permute(1, 0, 2).reshape(4096, 16384).contiguous()
        elif isinstance(o_sh, BlockFP8Weight):
            o_vals = bufs[idx].view(w, 4096, 16384 // w).permute(1, 0, 2).reshape(4096, 16384).contiguous()
            o_sc = bufs[idx + 1].view(w, 32, 128 // w).permute(1, 0, 2).reshape(32, 128).contiguous()
            o_full = BlockFP8Weight(values=o_vals, scales=o_sc)
        else:
            o_vals = bufs[idx].view(w, 4096, 16384 // w).permute(1, 0, 2).reshape(4096, 16384).contiguous()
            kt = 128 // w
            o_sc = bufs[idx + 1].view(w, 32, kt, 32, 4, 4).permute(1, 0, 2, 3, 4, 5).reshape(
                1, 32, 128, 32, 4, 4).contiguous()
            o_full = _mxfp8_from_parts(o_vals, o_sc, in_features=16384, padded_in_features=16384, out_features=4096)

        return GatheredMlaWeights(q_b_proj=qb_full, absorb=absorb_w, expand=expand_w, o_proj=o_full)

    def gather_sync(self, layer_index: int) -> GatheredKdaWeights | GatheredMlaWeights:
        """Synchronously gather one layer\x27s full attention weights on the current stream."""
        ltype = self.layer_types[layer_index]
        attn = self.layer_weights[layer_index]
        dev = self.device
        w = self.world

        if ltype == "linear_attention":
            ch = self._kda_ch
            inp = attn["input_proj"]
            o_sh = attn["o_proj"]
            small_sh = self._pack_kda_small(attn)
            small_full = torch.empty((w * self._kda_small_rows, 4096), dtype=torch.bfloat16, device=dev)
            if isinstance(inp, torch.Tensor):
                full_inp = torch.empty((24768, 4096), dtype=torch.bfloat16, device=dev)
                _weight_gather_slice(inp[0:ch], full_inp, 0, 8192, self._op_kda_qkv.handle)
                _weight_gather_slice(inp[ch:2 * ch], full_inp, 8192, 16384, self._op_kda_qkv.handle)
                _weight_gather_slice(inp[2 * ch:3 * ch], full_inp, 16384, 24576, self._op_kda_qkv.handle)
                _weight_gather_slice(small_sh, small_full, 0, small_full.shape[0], self._op_kda_small.handle)
                o_staged = torch.empty((w * 4096, ch), dtype=torch.bfloat16, device=dev)
                _weight_gather_slice(o_sh, o_staged, 0, o_staged.shape[0], self._op_kda_o.handle)
                return self._unpack_kda(layer_index, (full_inp, small_full, o_staged))
            p0 = inp.parts[0]
            p0_vals = p0.weight.values
            p0_sc = _mxfp8_scale_phys(p0)[0]
            mt = ch // 128
            kt = p0.padded_in_features // 128
            qkv_vals = torch.empty((24576, p0.padded_in_features), dtype=torch.float8_e4m3fn, device=dev)
            qkv_sc = torch.empty((192, kt, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_gather_slice(p0_vals[0:ch], qkv_vals, 0, 8192, self._op_kda_qkv.handle)
            _weight_gather_slice(p0_vals[ch:2 * ch], qkv_vals, 8192, 16384, self._op_kda_qkv.handle)
            _weight_gather_slice(p0_vals[2 * ch:3 * ch], qkv_vals, 16384, 24576, self._op_kda_qkv.handle)
            _weight_gather_slice(p0_sc[0:mt], qkv_sc, 0, 64, self._op_kda_qkv_sc.handle)
            _weight_gather_slice(p0_sc[mt:2 * mt], qkv_sc, 64, 128, self._op_kda_qkv_sc.handle)
            _weight_gather_slice(p0_sc[2 * mt:3 * mt], qkv_sc, 128, 192, self._op_kda_qkv_sc.handle)
            _weight_gather_slice(small_sh, small_full, 0, small_full.shape[0], self._op_kda_small.handle)
            if _is_mxfp8_weight(o_sh):
                o_staged = torch.empty((w * 4096, ch), dtype=torch.float8_e4m3fn, device=dev)
                o_sc_staged = torch.empty((w * 32, ch // 128, 32, 4, 4), dtype=torch.uint8, device=dev)
                _weight_gather_slice(o_sh.weight.values, o_staged, 0, o_staged.shape[0], self._op_kda_o.handle)
                _weight_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_sc_staged, 0, o_sc_staged.shape[0], self._op_kda_o_sc.handle)
            else:
                o_staged = torch.empty((w * 4096, ch), dtype=torch.float8_e4m3fn, device=dev)
                o_sc_staged = torch.empty((w * 32, ch // 128), dtype=torch.float32, device=dev)
                _weight_gather_slice(o_sh.values, o_staged, 0, o_staged.shape[0], self._op_kda_o.handle)
                _weight_gather_slice(o_sh.scales, o_sc_staged, 0, o_sc_staged.shape[0], self._op_kda_o_sc.handle)
            return self._unpack_kda(layer_index, (qkv_vals, qkv_sc, small_full, o_staged, o_sc_staged))

        # MLA layer
        qb_sh, kvb_sh, o_sh = attn["q_b_proj"], attn["kv_b_proj"], attn["o_proj"]
        bufs: list[torch.Tensor] = []
        if isinstance(qb_sh, torch.Tensor):
            qb_full = torch.empty((16384, 1536), dtype=torch.bfloat16, device=dev)
            _weight_gather_slice(qb_sh, qb_full, 0, 16384, self._op_mla_qb.handle)
            bufs.append(qb_full)
        elif isinstance(qb_sh, BlockFP8Weight):
            qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
            qb_s = torch.empty((128, 12), dtype=torch.float32, device=dev)
            _weight_gather_slice(qb_sh.values, qb_v, 0, 16384, self._op_mla_qb.handle)
            _weight_gather_slice(qb_sh.scales, qb_s, 0, 128, self._op_mla_qb_sc.handle)
            bufs.extend((qb_v, qb_s))
        else:
            qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
            qb_s = torch.empty((128, 12, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_gather_slice(qb_sh.weight.values, qb_v, 0, 16384, self._op_mla_qb.handle)
            _weight_gather_slice(_mxfp8_scale_phys(qb_sh)[0], qb_s, 0, 128, self._op_mla_qb_sc.handle)
            bufs.extend((qb_v, qb_s))

        if isinstance(kvb_sh, torch.Tensor):
            kvb_full = torch.empty((32768, 512), dtype=torch.bfloat16, device=dev)
            _weight_gather_slice(kvb_sh, kvb_full, 0, 32768, self._op_mla_kvb.handle)
            bufs.append(kvb_full)
        else:
            kvb_v = torch.empty((32768, 512), dtype=torch.float8_e4m3fn, device=dev)
            kvb_s = torch.empty((32768, 16), dtype=torch.uint8, device=dev)
            _weight_gather_slice(kvb_sh.weight.values, kvb_v, 0, 32768, self._op_mla_kvb.handle)
            _weight_gather_slice(kvb_sh.weight.scale_rows[0].view(torch.uint8), kvb_s, 0, 32768, self._op_mla_kvb_sc.handle)
            bufs.extend((kvb_v, kvb_s))

        if isinstance(o_sh, torch.Tensor):
            if self._mla_o_split:
                h0 = torch.empty((w * 2048, 16384 // w), dtype=torch.bfloat16, device=dev)
                h1 = torch.empty((w * 2048, 16384 // w), dtype=torch.bfloat16, device=dev)
                _weight_gather_slice(o_sh[:2048], h0, 0, h0.shape[0], self._op_mla_o.handle)
                _weight_gather_slice(o_sh[2048:], h1, 0, h1.shape[0], self._op_mla_o.handle)
                bufs.extend((h0, h1))
            else:
                o_st = torch.empty((w * 4096, 16384 // w), dtype=torch.bfloat16, device=dev)
                _weight_gather_slice(o_sh, o_st, 0, o_st.shape[0], self._op_mla_o.handle)
                bufs.append(o_st)
        elif isinstance(o_sh, BlockFP8Weight):
            o_v = torch.empty((w * 4096, 16384 // w), dtype=torch.float8_e4m3fn, device=dev)
            o_s = torch.empty((w * 32, 128 // w), dtype=torch.float32, device=dev)
            _weight_gather_slice(o_sh.values, o_v, 0, o_v.shape[0], self._op_mla_o.handle)
            _weight_gather_slice(o_sh.scales, o_s, 0, o_s.shape[0], self._op_mla_o_sc.handle)
            bufs.extend((o_v, o_s))
        else:
            o_v = torch.empty((w * 4096, 16384 // w), dtype=torch.float8_e4m3fn, device=dev)
            o_s = torch.empty((w * 32, 128 // w, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_gather_slice(o_sh.weight.values, o_v, 0, o_v.shape[0], self._op_mla_o.handle)
            _weight_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_s, 0, o_s.shape[0], self._op_mla_o_sc.handle)
            bufs.extend((o_v, o_s))

        return self._unpack_mla(layer_index, tuple(bufs))

    def start_phase1(self, layer_index: int, dependencies: list[torch.Tensor]) -> PendingWeightGather:
        """Enqueue Phase 1 of layer_index\x27s async weight prefetch on the auxiliary DMA stream.

        For KDA: q_proj + k_proj + small (+ q/k scales if MXFP8).
        For MLA: q_b_proj + kv_b_proj.
        """
        ltype = self.layer_types[layer_index]
        attn = self.layer_weights[layer_index]
        dev = self.device
        w = self.world
        aux = self.stream_handle
        ev_p1 = self._ev_p1_slots[layer_index]
        deps = list(dependencies)

        if ltype == "linear_attention":
            ch = self._kda_ch
            inp = attn["input_proj"]
            small_sh = self._pack_kda_small(attn)
            ev_small = self._ev_small_slots[layer_index]
            _weight_event_record(small_sh, deps, ev_small.handle)
            small_full = torch.empty((w * self._kda_small_rows, 4096), dtype=torch.bfloat16, device=dev)
            if isinstance(inp, torch.Tensor):
                full_inp = torch.empty((24768, 4096), dtype=torch.bfloat16, device=dev)
                _weight_dma_gather_slice(inp[0:ch], full_inp, deps, 0, 8192,
                                         self._op_kda_qkv.handle, aux, -1, -1)
                _weight_dma_gather_slice(inp[ch:2 * ch], full_inp, [full_inp], 8192, 16384,
                                         self._op_kda_qkv.handle, aux, -1, -1)
                _weight_dma_gather_slice(small_sh, small_full, [full_inp, small_sh], 0, small_full.shape[0],
                                         self._op_kda_small.handle, aux, ev_small.handle, ev_p1.handle)
                return PendingWeightGather(layer_index, ltype, (full_inp, small_full), ev_p1, extra=(small_sh,))
            p0 = inp.parts[0]
            p0_vals = p0.weight.values
            p0_sc = _mxfp8_scale_phys(p0)[0]
            mt = ch // 128
            kt = p0.padded_in_features // 128
            qkv_vals = torch.empty((24576, p0.padded_in_features), dtype=torch.float8_e4m3fn, device=dev)
            qkv_sc = torch.empty((192, kt, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_dma_gather_slice(p0_vals[0:ch], qkv_vals, deps, 0, 8192,
                                     self._op_kda_qkv.handle, aux, -1, -1)
            _weight_dma_gather_slice(p0_vals[ch:2 * ch], qkv_vals, [qkv_vals], 8192, 16384,
                                     self._op_kda_qkv.handle, aux, -1, -1)
            _weight_dma_gather_slice(p0_sc[0:mt], qkv_sc, [qkv_vals], 0, 64,
                                     self._op_kda_qkv_sc.handle, aux, -1, -1)
            _weight_dma_gather_slice(p0_sc[mt:2 * mt], qkv_sc, [qkv_sc], 64, 128,
                                     self._op_kda_qkv_sc.handle, aux, -1, -1)
            _weight_dma_gather_slice(small_sh, small_full, [qkv_sc, small_sh], 0, small_full.shape[0],
                                     self._op_kda_small.handle, aux, ev_small.handle, ev_p1.handle)
            return PendingWeightGather(layer_index, ltype, (qkv_vals, qkv_sc, small_full), ev_p1, extra=(small_sh,))

        # MLA Phase 1: q_b_proj + kv_b_proj
        qb_sh, kvb_sh = attn["q_b_proj"], attn["kv_b_proj"]
        bufs: list[torch.Tensor] = []
        cur_deps = deps
        if isinstance(qb_sh, torch.Tensor):
            qb_full = torch.empty((16384, 1536), dtype=torch.bfloat16, device=dev)
            _weight_dma_gather_slice(qb_sh, qb_full, cur_deps, 0, 16384, self._op_mla_qb.handle, aux, -1, -1)
            bufs.append(qb_full)
            cur_deps = [qb_full]
        elif isinstance(qb_sh, BlockFP8Weight):
            qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
            qb_s = torch.empty((128, 12), dtype=torch.float32, device=dev)
            _weight_dma_gather_slice(qb_sh.values, qb_v, cur_deps, 0, 16384, self._op_mla_qb.handle, aux, -1, -1)
            _weight_dma_gather_slice(qb_sh.scales, qb_s, [qb_v], 0, 128, self._op_mla_qb_sc.handle, aux, -1, -1)
            bufs.extend((qb_v, qb_s))
            cur_deps = [qb_s]
        else:
            qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
            qb_s = torch.empty((128, 12, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_dma_gather_slice(qb_sh.weight.values, qb_v, cur_deps, 0, 16384, self._op_mla_qb.handle, aux, -1, -1)
            _weight_dma_gather_slice(_mxfp8_scale_phys(qb_sh)[0], qb_s, [qb_v], 0, 128, self._op_mla_qb_sc.handle, aux, -1, -1)
            bufs.extend((qb_v, qb_s))
            cur_deps = [qb_s]

        if isinstance(kvb_sh, torch.Tensor):
            kvb_full = torch.empty((32768, 512), dtype=torch.bfloat16, device=dev)
            _weight_dma_gather_slice(kvb_sh, kvb_full, cur_deps, 0, 32768,
                                     self._op_mla_kvb.handle, aux, -1, ev_p1.handle)
            bufs.append(kvb_full)
        else:
            kvb_v = torch.empty((32768, 512), dtype=torch.float8_e4m3fn, device=dev)
            kvb_s = torch.empty((32768, 16), dtype=torch.uint8, device=dev)
            _weight_dma_gather_slice(kvb_sh.weight.values, kvb_v, cur_deps, 0, 32768,
                                     self._op_mla_kvb.handle, aux, -1, -1)
            _weight_dma_gather_slice(kvb_sh.weight.scale_rows[0].view(torch.uint8), kvb_s, [kvb_v], 0, 32768,
                                     self._op_mla_kvb_sc.handle, aux, -1, ev_p1.handle)
            bufs.extend((kvb_v, kvb_s))
        return PendingWeightGather(layer_index, ltype, tuple(bufs), ev_p1)

    def start_phase2(self, pending1: PendingWeightGather, dependencies: list[torch.Tensor]) -> PendingWeightGather:
        """Enqueue Phase 2 of layer_index\x27s async weight prefetch on the auxiliary DMA stream.

        For KDA: v_proj + o_proj (+ v/o scales if MXFP8).
        For MLA: o_proj (+ o scales if FP8/MXFP8).
        """
        layer_index = pending1.layer_index
        ltype = pending1.kind
        attn = self.layer_weights[layer_index]
        dev = self.device
        w = self.world
        aux = self.stream_handle
        ev_done = self._ev_done_slots[layer_index]
        deps = [*dependencies, *pending1.buffers, *pending1.extra]

        if ltype == "linear_attention":
            ch = self._kda_ch
            inp = attn["input_proj"]
            o_sh = attn["o_proj"]
            if isinstance(inp, torch.Tensor):
                full_inp, small_full = pending1.buffers
                o_staged = torch.empty((w * 4096, ch), dtype=torch.bfloat16, device=dev)
                _weight_dma_gather_slice(inp[2 * ch:3 * ch], full_inp, deps, 16384, 24576,
                                         self._op_kda_qkv.handle, aux, -1, -1)
                _weight_dma_gather_slice(o_sh, o_staged, [full_inp], 0, o_staged.shape[0],
                                         self._op_kda_o.handle, aux, -1, ev_done.handle)
                return PendingWeightGather(layer_index, ltype, (full_inp, small_full, o_staged), ev_done, extra=pending1.extra)
            p0 = inp.parts[0]
            p0_vals = p0.weight.values
            p0_sc = _mxfp8_scale_phys(p0)[0]
            mt = ch // 128
            qkv_vals, qkv_sc, small_full = pending1.buffers
            _weight_dma_gather_slice(p0_vals[2 * ch:3 * ch], qkv_vals, deps, 16384, 24576,
                                     self._op_kda_qkv.handle, aux, -1, -1)
            _weight_dma_gather_slice(p0_sc[2 * mt:3 * mt], qkv_sc, [qkv_vals], 128, 192,
                                     self._op_kda_qkv_sc.handle, aux, -1, -1)
            if _is_mxfp8_weight(o_sh):
                o_staged = torch.empty((w * 4096, ch), dtype=torch.float8_e4m3fn, device=dev)
                o_sc_staged = torch.empty((w * 32, ch // 128, 32, 4, 4), dtype=torch.uint8, device=dev)
                _weight_dma_gather_slice(o_sh.weight.values, o_staged, [qkv_sc], 0, o_staged.shape[0],
                                         self._op_kda_o.handle, aux, -1, -1)
                _weight_dma_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_sc_staged, [o_staged], 0, o_sc_staged.shape[0],
                                         self._op_kda_o_sc.handle, aux, -1, ev_done.handle)
            else:
                o_staged = torch.empty((w * 4096, ch), dtype=torch.float8_e4m3fn, device=dev)
                o_sc_staged = torch.empty((w * 32, ch // 128), dtype=torch.float32, device=dev)
                _weight_dma_gather_slice(o_sh.values, o_staged, [qkv_sc], 0, o_staged.shape[0],
                                         self._op_kda_o.handle, aux, -1, -1)
                _weight_dma_gather_slice(o_sh.scales, o_sc_staged, [o_staged], 0, o_sc_staged.shape[0],
                                         self._op_kda_o_sc.handle, aux, -1, ev_done.handle)
            return PendingWeightGather(
                layer_index, ltype, (qkv_vals, qkv_sc, small_full, o_staged, o_sc_staged),
                ev_done, extra=pending1.extra,
            )

        # MLA Phase 2: o_proj
        o_sh = attn["o_proj"]
        bufs = list(pending1.buffers)
        if isinstance(o_sh, torch.Tensor):
            if self._mla_o_split:
                h0 = torch.empty((w * 2048, 16384 // w), dtype=torch.bfloat16, device=dev)
                h1 = torch.empty((w * 2048, 16384 // w), dtype=torch.bfloat16, device=dev)
                _weight_dma_gather_slice(o_sh[:2048], h0, deps, 0, h0.shape[0], self._op_mla_o.handle, aux, -1, -1)
                _weight_dma_gather_slice(o_sh[2048:], h1, [h0], 0, h1.shape[0],
                                         self._op_mla_o.handle, aux, -1, ev_done.handle)
                bufs.extend((h0, h1))
            else:
                o_st = torch.empty((w * 4096, 16384 // w), dtype=torch.bfloat16, device=dev)
                _weight_dma_gather_slice(o_sh, o_st, deps, 0, o_st.shape[0],
                                         self._op_mla_o.handle, aux, -1, ev_done.handle)
                bufs.append(o_st)
        elif isinstance(o_sh, BlockFP8Weight):
            o_v = torch.empty((w * 4096, 16384 // w), dtype=torch.float8_e4m3fn, device=dev)
            o_s = torch.empty((w * 32, 128 // w), dtype=torch.float32, device=dev)
            _weight_dma_gather_slice(o_sh.values, o_v, deps, 0, o_v.shape[0], self._op_mla_o.handle, aux, -1, -1)
            _weight_dma_gather_slice(o_sh.scales, o_s, [o_v], 0, o_s.shape[0],
                                     self._op_mla_o_sc.handle, aux, -1, ev_done.handle)
            bufs.extend((o_v, o_s))
        else:
            o_v = torch.empty((w * 4096, 16384 // w), dtype=torch.float8_e4m3fn, device=dev)
            o_s = torch.empty((w * 32, 128 // w, 32, 4, 4), dtype=torch.uint8, device=dev)
            _weight_dma_gather_slice(o_sh.weight.values, o_v, deps, 0, o_v.shape[0], self._op_mla_o.handle, aux, -1, -1)
            _weight_dma_gather_slice(_mxfp8_scale_phys(o_sh)[0], o_s, [o_v], 0, o_s.shape[0],
                                     self._op_mla_o_sc.handle, aux, -1, ev_done.handle)
            bufs.extend((o_v, o_s))
        return PendingWeightGather(layer_index, ltype, tuple(bufs), ev_done)

    def start_full(self, layer_index: int, dependencies: list[torch.Tensor]) -> PendingWeightGather:
        """Enqueue both Phase 1 and Phase 2 back-to-back on the auxiliary DMA stream (used during MLA)."""
        p1 = self.start_phase1(layer_index, dependencies)
        return self.start_phase2(p1, [])

    def finish(self, pending: PendingWeightGather, dependencies: list[torch.Tensor] = ()) -> GatheredKdaWeights | GatheredMlaWeights:
        """Wait for the async weight gather to complete on main stream and unpack full weights."""
        anchor = pending.buffers[-1]
        deps = [*dependencies, *pending.buffers[:-1], *pending.extra]
        _weight_event_wait(anchor, deps, pending.done_ev.handle)
        if pending.kind == "linear_attention":
            return self._unpack_kda(pending.layer_index, pending.buffers)
        return self._unpack_mla(pending.layer_index, pending.buffers)
