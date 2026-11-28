# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""On-the-fly ZeRO-3 attention weight all-gather for GLM-5.3 row-sharded prefill.

Keeps all 2D attention weights sharded 1/W across tensor-parallel ranks (identical to decode
storage) and gathers each layer's weights on demand into transient scratch reused across layers.
Supports BF16, MXFP8, Block-FP8, and mixed grouped linear weights across TP2 and TP4, over
4-phase zero-stall PCIe DMA prefetch and coalesced async RoCE RDMA overlap on a dedicated
non-pooled auxiliary stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyperflare_recipes.components.block_fp8 import BlockFP8Weight
from hyperflare_recipes.components.collectives import (
    GatherSliceOp,
    PreparedGatherShards,
    StreamEventSlot,
    _roce,
    _row_shard_capacity_bytes,
    dma_gather_enqueue,
    ordered_gather_shards,
    stream_event_wait,
)
from hyperflare_recipes.components.weights import GroupedLinearWeights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer


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


def _template_linear(weight, rows: int, cols: int, device: torch.device, *,
                     in_features: int | None = None, padded_in_features: int | None = None):
    """Allocate a transient full-shape weight matching ``weight``'s quantization format."""
    in_f = cols if in_features is None else in_features
    pad_in = cols if padded_in_features is None else padded_in_features
    if isinstance(weight, torch.Tensor):
        return torch.zeros((rows, cols), dtype=weight.dtype, device=device)
    if _is_mxfp8_weight(weight):
        vals = torch.zeros((rows, pad_in), dtype=torch.float8_e4m3fn, device=device)
        sc_phys = torch.full((1, (rows + 127) // 128, pad_in // 128, 32, 4, 4), 127, dtype=torch.uint8, device=device)
        sc_rows = torch.full((1, rows, pad_in // 32), 127, dtype=torch.uint8, device=device)
        return _mxfp8_from_parts(
            vals, sc_phys, in_features=in_f, padded_in_features=pad_in,
            out_features=rows, scale_rows=sc_rows.view(torch.float8_e8m0fnu),
        )
    if isinstance(weight, BlockFP8Weight):
        vals = torch.zeros((rows, cols), dtype=torch.float8_e4m3fn, device=device)
        scales = torch.ones((rows // 128, cols // 128), dtype=torch.float32, device=device)
        return BlockFP8Weight(values=vals, scales=scales)
    raise ValueError(f"unsupported attention weight type: {type(weight).__name__}")


def full_kda_template(sharded_attn: dict[str, object], *, world_size: int, device: torch.device) -> dict[str, object]:
    """Allocate a transient full 64-head KDA weight template for preparation and tuning only."""
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
        full_p0 = _template_linear(
            p0, 3 * ch, p0.padded_in_features, device,
            in_features=p0.in_features, padded_in_features=p0.padded_in_features,
        )
        full_p1 = torch.zeros((64 + 128, 4096), dtype=p1.dtype, device=device)
        tpl["input_proj"] = GroupedLinearWeights(parts=(full_p0, full_p1), widths=(3 * ch, 192))
    else:
        raise ValueError(f"unsupported KDA input_proj weight type: {type(inp).__name__}")
    tpl["o_proj"] = _template_linear(sharded_attn["o_proj"], 4096, ch, device)
    return tpl


def full_mla_template(sharded_attn: dict[str, object], *, world_size: int, device: torch.device) -> dict[str, object]:
    """Allocate a transient full 64-head MLA weight template for preparation and tuning only."""
    del world_size
    return {
        key: _template_linear(sharded_attn[key], rows, cols, device)
        for key, (rows, cols) in (("q_b_proj", (16384, 1536)), ("kv_b_proj", (32768, 512)), ("o_proj", (4096, 16384)))
    }


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
    done_ev: StreamEventSlot
    extra: tuple[torch.Tensor, ...] = ()
    o_buffers: tuple[torch.Tensor, ...] = ()
    o_done_ev: StreamEventSlot | None = None


class AttentionWeightGatherer:
    """Shape-deduplicated ZeRO-3 weight all-gather manager for GLM-5.3 row-sharded prefill."""

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
        self.use_roce_2d = self.is_roce
        self.max_gather_bytes = _row_shard_capacity_bytes(runtime) if runtime is not None else (1 << 30)
        self.async_roce = (
            (not self.pipelined)
            and self.is_roce
            and isinstance(workspace, ScratchArena)
            and workspace.overlap
        )
        if (self.pipelined or self.async_roce) and isinstance(workspace, ScratchArena) and workspace.overlap:
            self.stream_handle = workspace.auxiliary_stream.cuda_stream
        else:
            self.stream_handle = 0

        self.channel_token = (
            channel_token if channel_token is not None
            else torch.zeros(1, dtype=torch.int32, device=device)
        )

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

        self._ops: dict[tuple[int, int], GatherSliceOp] = {}
        self._gathers: list[PreparedGatherShards] = (
            [self._roce_gather] if self._roce_gather is not None else []
        )

        n_layers = len(layer_weights)
        need_events = self.pipelined or self.async_roce
        self._ev_handoff = StreamEventSlot() if need_events else None
        self._ev_inp_done = tuple(StreamEventSlot() for _ in range(n_layers)) if need_events else ()
        self._ev_o_done = tuple(StreamEventSlot() for _ in range(n_layers)) if need_events else ()

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

    def _get_or_create_op(self, src_shape: tuple[int, int], *, runtime, coordinator, session, name) -> GatherSliceOp:
        if src_shape in self._ops:
            return self._ops[src_shape]
        dst_shape = (src_shape[0] * self.world, src_shape[1])
        if self._roce_gather is not None:
            op = GatherSliceOp(self._roce_gather, src_shape, dst_shape)
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
        op = GatherSliceOp(gather, src_shape, dst_shape)
        self._ops[src_shape] = op
        self._gathers.append(gather)
        return op

    def _declare_kda_shapes(self, attn, *, runtime, coordinator, session, name) -> dict[str, object]:
        ch, w = self._kda_ch, self.world
        kw = dict(runtime=runtime, coordinator=coordinator, session=session, name=name)
        spec: dict[str, object] = {}
        inp = attn["input_proj"]
        if isinstance(inp, torch.Tensor):
            cols_bf16, row_bytes = 4096, 4096 * 2
            spec["qkv_sc"] = spec["qkv_sc_bulk"] = None
        elif isinstance(inp, GroupedLinearWeights):
            p0 = inp.parts[0]
            cols_bf16, row_bytes = p0.padded_in_features // 2, p0.padded_in_features
            mt, kt = ch // 128, p0.padded_in_features // 128
            spec["qkv_sc"] = self._get_or_create_op((mt, kt * 256), **kw)
            spec["qkv_sc_bulk"] = (
                self._get_or_create_op((3 * mt, kt * 256), **kw)
                if self.use_roce_2d and 3 * mt * kt * 512 * w <= self.max_gather_bytes else None
            )
        else:
            raise ValueError(f"unsupported KDA input_proj type: {type(inp).__name__}")

        spec["qkv"] = self._get_or_create_op((ch, cols_bf16), **kw)
        if self.use_roce_2d and 3 * ch * row_bytes * w <= self.max_gather_bytes:
            spec["qkv_coalesce"], spec["qkv_bulk"] = 3, self._get_or_create_op((3 * ch, cols_bf16), **kw)
        elif self.use_roce_2d and 2 * ch * row_bytes * w <= self.max_gather_bytes:
            spec["qkv_coalesce"], spec["qkv_bulk"] = 2, self._get_or_create_op((2 * ch, cols_bf16), **kw)
        else:
            spec["qkv_coalesce"], spec["qkv_bulk"] = 1, None
        spec["small"] = self._get_or_create_op((self._kda_small_rows, 4096), **kw)

        o = attn["o_proj"]
        if isinstance(o, torch.Tensor):
            spec["o"], spec["o_sc"] = self._get_or_create_op((4096, ch), **kw), None
        elif _is_mxfp8_weight(o):
            spec["o"] = self._get_or_create_op((4096, ch // 2), **kw)
            spec["o_sc"] = self._get_or_create_op((32, (ch // 128) * 256), **kw)
        elif isinstance(o, BlockFP8Weight):
            spec["o"] = self._get_or_create_op((4096, ch // 2), **kw)
            spec["o_sc"] = self._get_or_create_op((32, (ch // 128) * 2), **kw)
        else:
            raise ValueError(f"unsupported KDA o_proj type: {type(o).__name__}")
        return spec

    def _declare_mla_shapes(self, attn, *, runtime, coordinator, session, name) -> dict[str, object]:
        w = self.world
        kw = dict(runtime=runtime, coordinator=coordinator, session=session, name=name)
        spec: dict[str, object] = {}
        qb = attn["q_b_proj"]
        if isinstance(qb, torch.Tensor):
            spec["qb"], spec["qb_sc"] = self._get_or_create_op((16384 // w, 1536), **kw), None
        elif isinstance(qb, BlockFP8Weight):
            spec["qb"] = self._get_or_create_op((16384 // w, 768), **kw)
            spec["qb_sc"] = self._get_or_create_op((128 // w, 24), **kw)
        elif _is_mxfp8_weight(qb):
            spec["qb"] = self._get_or_create_op((16384 // w, 768), **kw)
            spec["qb_sc"] = self._get_or_create_op((128 // w, 12 * 256), **kw)
        else:
            raise ValueError(f"unsupported MLA q_b_proj type: {type(qb).__name__}")

        kvb = attn["kv_b_proj"]
        if isinstance(kvb, torch.Tensor):
            spec["kvb"], spec["kvb_sc"] = self._get_or_create_op((32768 // w, 512), **kw), None
        elif _is_mxfp8_weight(kvb):
            spec["kvb"] = self._get_or_create_op((32768 // w, 256), **kw)
            spec["kvb_sc"] = self._get_or_create_op((32768 // w, 8), **kw)
        else:
            raise ValueError(f"unsupported MLA kv_b_proj type: {type(kvb).__name__}")

        o = attn["o_proj"]
        if isinstance(o, torch.Tensor):
            split = 4096 * 16384 * 2 > self.max_gather_bytes
            spec["mla_o_split"] = split
            spec["o"], spec["o_sc"] = self._get_or_create_op((2048 if split else 4096, 16384 // w), **kw), None
        elif isinstance(o, BlockFP8Weight):
            spec["mla_o_split"] = False
            spec["o"] = self._get_or_create_op((4096, 8192 // w), **kw)
            spec["o_sc"] = self._get_or_create_op((32, (128 // w) * 2), **kw)
        elif _is_mxfp8_weight(o):
            spec["mla_o_split"] = False
            spec["o"] = self._get_or_create_op((4096, 8192 // w), **kw)
            spec["o_sc"] = self._get_or_create_op((32, (128 // w) * 256), **kw)
        else:
            raise ValueError(f"unsupported MLA o_proj type: {type(o).__name__}")
        return spec

    def _emit(self, src: torch.Tensor, dst: torch.Tensor, r0: int, r1: int, op: GatherSliceOp,
              col_b: int, tok: torch.Tensor, *, async_mode: bool, deps: list[torch.Tensor] = (),
              handoff: int = -1, done_ev: int = -1) -> None:
        if async_mode:
            dma_gather_enqueue(src, dst, tok, list(deps), r0, r1, op.handle,
                               self.stream_handle, handoff, done_ev, col_b)
        else:
            ordered_gather_shards(src, dst, tok, [], r0, r1, op.handle, col_b)

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

    def _unpack_col_sharded_o(self, o_sh, o_bufs: tuple[torch.Tensor, ...], *,
                              in_features: int, split_o: bool = False):
        w = self.world
        col_sh = in_features // w
        kt_full, kt_sh = in_features // 128, col_sh // 128
        if self.use_roce_2d:
            if isinstance(o_sh, torch.Tensor):
                return o_bufs[0]
            o_v, o_s = o_bufs
            if _is_mxfp8_weight(o_sh):
                return _mxfp8_from_parts(o_v, o_s.unsqueeze(0), in_features=in_features,
                                         padded_in_features=in_features, out_features=4096)
            return BlockFP8Weight(values=o_v, scales=o_s)
        if isinstance(o_sh, torch.Tensor):
            if split_o:
                h0, h1 = o_bufs
                o_top = h0.view(w, 2048, col_sh).permute(1, 0, 2).reshape(2048, in_features)
                o_bot = h1.view(w, 2048, col_sh).permute(1, 0, 2).reshape(2048, in_features)
                return torch.cat((o_top, o_bot), dim=0).contiguous()
            (o_st,) = o_bufs
            return o_st.view(w, 4096, col_sh).permute(1, 0, 2).reshape(4096, in_features).contiguous()
        o_v, o_s = o_bufs
        o_vals = o_v.view(w, 4096, col_sh).permute(1, 0, 2).reshape(4096, in_features).contiguous()
        if _is_mxfp8_weight(o_sh):
            o_sc = o_s.view(w, 32, kt_sh, 32, 4, 4).permute(1, 0, 2, 3, 4, 5).reshape(
                1, 32, kt_full, 32, 4, 4).contiguous()
            return _mxfp8_from_parts(o_vals, o_sc, in_features=in_features,
                                     padded_in_features=in_features, out_features=4096)
        o_sc = o_s.view(w, 32, kt_sh).permute(1, 0, 2).reshape(32, kt_full).contiguous()
        return BlockFP8Weight(values=o_vals, scales=o_sc)

    def _unpack_kda_o(self, layer_index: int, o_bufs: tuple[torch.Tensor, ...]):
        return self._unpack_col_sharded_o(self.layer_weights[layer_index]["o_proj"], o_bufs, in_features=8192)

    def _unpack_mla_o(self, layer_index: int, o_bufs: tuple[torch.Tensor, ...]):
        return self._unpack_col_sharded_o(
            self.layer_weights[layer_index]["o_proj"], o_bufs, in_features=16384,
            split_o=bool(self._layer_ops[layer_index].get("mla_o_split")),
        )

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
            full_inp[24576:24640].copy_(sf[:, :r0].reshape(64, 4096))
            full_inp[24640:24768].copy_(inp[3 * ch + lh:])
            input_proj = full_inp
        else:
            p0, p1 = inp.parts
            qkv_vals, qkv_sc, small_full = bufs[:3]
            sf = small_full.view(w, self._kda_small_rows, 4096)
            part1_full = torch.cat((sf[:, :r0].reshape(64, 4096), p1[lh:]), dim=0).contiguous()
            part0_full = _mxfp8_from_parts(
                qkv_vals, qkv_sc.unsqueeze(0),
                in_features=p0.in_features, padded_in_features=p0.padded_in_features, out_features=24576,
            )
            input_proj = GroupedLinearWeights(parts=(part0_full, part1_full), widths=(24576, 192))

        return GatheredKdaWeights(
            input_proj=input_proj,
            f_b_proj=sf[:, r0:r1].reshape(8192, 128).contiguous(),
            g_b_proj=sf[:, r1:r2].reshape(8192, 128).contiguous(),
            o_proj=None,
            conv1d=sf[:, r2:r3].reshape(w, 3, ch, 4).permute(1, 0, 2, 3).reshape(24576, 4).contiguous(),
            dt_bias=sf[:, r3:r4].reshape(8192 * 2).contiguous().view(torch.float32),
            A_log=sf[:, r4, :lh * 2].contiguous().view(torch.float32).reshape(64),
        )

    def _unpack_mla_input(self, layer_index: int, bufs: tuple[torch.Tensor, ...]) -> GatheredMlaWeights:
        attn = self.layer_weights[layer_index]
        qb_sh, kvb_sh = attn["q_b_proj"], attn["kv_b_proj"]
        if isinstance(qb_sh, torch.Tensor):
            qb_full, idx = bufs[0], 1
        elif isinstance(qb_sh, BlockFP8Weight):
            qb_full, idx = BlockFP8Weight(values=bufs[0], scales=bufs[1]), 2
        else:
            qb_full, idx = _mxfp8_from_parts(
                bufs[0], bufs[1].unsqueeze(0), in_features=1536, padded_in_features=1536, out_features=16384), 2

        if isinstance(kvb_sh, torch.Tensor):
            kvb_3d = bufs[idx].view(64, 512, 512)
            absorb_w, expand_w = kvb_3d[:, :256], kvb_3d[:, 256:]
        else:
            vals = bufs[idx].view(64, 512, 512)
            sc = bufs[idx + 1].view(torch.float8_e8m0fnu).view(64, 512, 16)
            absorb_w, expand_w = (vals[:, :256], sc[:, :256]), (vals[:, 256:], sc[:, 256:])
        return GatheredMlaWeights(q_b_proj=qb_full, absorb=absorb_w, expand=expand_w, o_proj=None)

    # ------------------------------------------------------------------
    # Unified gather emitters (shared by sync and async paths)
    # ------------------------------------------------------------------

    def _gather_col_sharded_o(self, layer_index: int, *, in_features: int, tok: torch.Tensor,
                              async_mode: bool, deps: list[torch.Tensor] = (),
                              done_ev: int = -1, pre_h0: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        dev, w = self.device, self.world
        o_sh = attn["o_proj"]
        col_sh = in_features // w
        kt_full, kt_sh = in_features // 128, col_sh // 128
        handoff = self._ev_handoff.handle if async_mode else -1
        split_o = bool(spec.get("mla_o_split"))

        if self.use_roce_2d:
            if isinstance(o_sh, torch.Tensor):
                o_full = torch.empty((4096, in_features), dtype=torch.bfloat16, device=dev)
                col_b = col_sh * 2
                if split_o:
                    self._emit(o_sh[:2048], o_full, 0, 2048, spec["o"], col_b, tok,
                               async_mode=async_mode, deps=deps, handoff=handoff, done_ev=-1)
                    self._emit(o_sh[2048:], o_full, 2048, 4096, spec["o"], col_b, tok,
                               async_mode=async_mode, deps=[o_full], handoff=-1, done_ev=done_ev)
                else:
                    self._emit(o_sh, o_full, 0, 4096, spec["o"], col_b, tok,
                               async_mode=async_mode, deps=deps, handoff=handoff, done_ev=done_ev)
                return (o_full,)
            o_v = torch.empty((4096, in_features), dtype=torch.float8_e4m3fn, device=dev)
            if _is_mxfp8_weight(o_sh):
                o_s = torch.empty((32, kt_full, 32, 4, 4), dtype=torch.uint8, device=dev)
                src_v, src_s, sc_col_b = o_sh.weight.values, _mxfp8_scale_phys(o_sh)[0], kt_sh * 512
            else:
                o_s = torch.empty((32, kt_full), dtype=torch.float32, device=dev)
                src_v, src_s, sc_col_b = o_sh.values, o_sh.scales, kt_sh * 4
            self._emit(src_v, o_v, 0, 4096, spec["o"], col_sh, tok,
                       async_mode=async_mode, deps=deps, handoff=handoff, done_ev=-1)
            self._emit(src_s, o_s, 0, 32, spec["o_sc"], sc_col_b, tok,
                       async_mode=async_mode, deps=[o_v], handoff=handoff, done_ev=done_ev)
            return (o_v, o_s)

        if isinstance(o_sh, torch.Tensor):
            if split_o:
                if pre_h0 is None:
                    h0 = torch.empty((w * 2048, col_sh), dtype=torch.bfloat16, device=dev)
                    self._emit(o_sh[:2048], h0, 0, h0.shape[0], spec["o"], 0, tok,
                               async_mode=async_mode, deps=deps, handoff=handoff, done_ev=-1)
                    h1_deps = [h0]
                else:
                    h0, h1_deps = pre_h0, deps
                h1 = torch.empty((w * 2048, col_sh), dtype=torch.bfloat16, device=dev)
                self._emit(o_sh[2048:], h1, 0, h1.shape[0], spec["o"], 0, tok,
                           async_mode=async_mode, deps=h1_deps, handoff=handoff, done_ev=done_ev)
                return (h0, h1)
            o_st = torch.empty((w * 4096, col_sh), dtype=torch.bfloat16, device=dev)
            self._emit(o_sh, o_st, 0, o_st.shape[0], spec["o"], 0, tok,
                       async_mode=async_mode, deps=deps, handoff=handoff, done_ev=done_ev)
            return (o_st,)

        o_v = torch.empty((w * 4096, col_sh), dtype=torch.float8_e4m3fn, device=dev)
        if _is_mxfp8_weight(o_sh):
            o_s = torch.empty((w * 32, kt_sh, 32, 4, 4), dtype=torch.uint8, device=dev)
            src_v, src_s = o_sh.weight.values, _mxfp8_scale_phys(o_sh)[0]
        else:
            o_s = torch.empty((w * 32, kt_sh), dtype=torch.float32, device=dev)
            src_v, src_s = o_sh.values, o_sh.scales
        self._emit(src_v, o_v, 0, o_v.shape[0], spec["o"], 0, tok,
                   async_mode=async_mode, deps=deps, handoff=handoff, done_ev=-1)
        self._emit(src_s, o_s, 0, o_s.shape[0], spec["o_sc"], 0, tok,
                   async_mode=async_mode, deps=[o_v], handoff=handoff, done_ev=done_ev)
        return (o_v, o_s)

    def _gather_kda_input_coalesced(self, layer_index: int, tok: torch.Tensor, *,
                                    async_mode: bool, deps: list[torch.Tensor] = (),
                                    done_ev: int = -1) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        attn = self.layer_weights[layer_index]
        spec = self._layer_ops[layer_index]
        dev, w, ch = self.device, self.world, self._kda_ch
        handoff = self._ev_handoff.handle if async_mode else -1
        inp = attn["input_proj"]
        small_sh = self._pack_kda_small(attn)
        small_full = torch.empty((w * self._kda_small_rows, 4096), dtype=torch.bfloat16, device=dev)
        coal = spec["qkv_coalesce"]

        if isinstance(inp, torch.Tensor):
            full_inp = torch.empty((24768, 4096), dtype=torch.bfloat16, device=dev)
            if coal == 3:
                self._emit(inp[0:3 * ch], full_inp, 0, 24576, spec["qkv_bulk"], ch * 4096 * 2, tok,
                           async_mode=async_mode, deps=deps, handoff=handoff)
            elif coal == 2:
                self._emit(inp[0:2 * ch], full_inp, 0, 16384, spec["qkv_bulk"], ch * 4096 * 2, tok,
                           async_mode=async_mode, deps=deps, handoff=handoff)
                self._emit(inp[2 * ch:3 * ch], full_inp, 16384, 24576, spec["qkv"], 0, tok,
                           async_mode=async_mode, deps=[full_inp])
            else:
                self._emit(inp[0:ch], full_inp, 0, 8192, spec["qkv"], 0, tok,
                           async_mode=async_mode, deps=deps, handoff=handoff)
                self._emit(inp[ch:2 * ch], full_inp, 8192, 16384, spec["qkv"], 0, tok,
                           async_mode=async_mode, deps=[full_inp])
                self._emit(inp[2 * ch:3 * ch], full_inp, 16384, 24576, spec["qkv"], 0, tok,
                           async_mode=async_mode, deps=[full_inp])
            self._emit(small_sh, small_full, 0, small_full.shape[0], spec["small"], 0, tok,
                       async_mode=async_mode, deps=[full_inp, small_sh], handoff=handoff, done_ev=done_ev)
            return (full_inp, small_full), small_sh

        p0 = inp.parts[0]
        p0_vals, p0_sc = p0.weight.values, _mxfp8_scale_phys(p0)[0]
        mt, kt = ch // 128, p0.padded_in_features // 128
        qkv_vals = torch.empty((24576, p0.padded_in_features), dtype=torch.float8_e4m3fn, device=dev)
        qkv_sc = torch.empty((192, kt, 32, 4, 4), dtype=torch.uint8, device=dev)
        if coal == 3:
            self._emit(p0_vals[0:3 * ch], qkv_vals, 0, 24576, spec["qkv_bulk"], ch * p0.padded_in_features, tok,
                       async_mode=async_mode, deps=deps, handoff=handoff)
        elif coal == 2:
            self._emit(p0_vals[0:2 * ch], qkv_vals, 0, 16384, spec["qkv_bulk"], ch * p0.padded_in_features, tok,
                       async_mode=async_mode, deps=deps, handoff=handoff)
            self._emit(p0_vals[2 * ch:3 * ch], qkv_vals, 16384, 24576, spec["qkv"], 0, tok,
                       async_mode=async_mode, deps=[qkv_vals])
        else:
            self._emit(p0_vals[0:ch], qkv_vals, 0, 8192, spec["qkv"], 0, tok,
                       async_mode=async_mode, deps=deps, handoff=handoff)
            self._emit(p0_vals[ch:2 * ch], qkv_vals, 8192, 16384, spec["qkv"], 0, tok,
                       async_mode=async_mode, deps=[qkv_vals])
            self._emit(p0_vals[2 * ch:3 * ch], qkv_vals, 16384, 24576, spec["qkv"], 0, tok,
                       async_mode=async_mode, deps=[qkv_vals])
        if spec["qkv_sc_bulk"] is not None:
            self._emit(p0_sc[0:3 * mt], qkv_sc, 0, 192, spec["qkv_sc_bulk"], mt * kt * 512, tok,
                       async_mode=async_mode, deps=[qkv_vals], handoff=handoff)
        else:
            self._emit(p0_sc[0:mt], qkv_sc, 0, 64, spec["qkv_sc"], 0, tok,
                       async_mode=async_mode, deps=[qkv_vals], handoff=handoff)
            self._emit(p0_sc[mt:2 * mt], qkv_sc, 64, 128, spec["qkv_sc"], 0, tok,
                       async_mode=async_mode, deps=[qkv_sc])
            self._emit(p0_sc[2 * mt:3 * mt], qkv_sc, 128, 192, spec["qkv_sc"], 0, tok,
                       async_mode=async_mode, deps=[qkv_sc])
        self._emit(small_sh, small_full, 0, small_full.shape[0], spec["small"], 0, tok,
                   async_mode=async_mode, deps=[qkv_sc, small_sh], handoff=handoff, done_ev=done_ev)
        return (qkv_vals, qkv_sc, small_full), small_sh

    def _gather_mla_qb(self, layer_index: int, tok: torch.Tensor, *,
                       async_mode: bool, deps: list[torch.Tensor] = ()) -> tuple[torch.Tensor, ...]:
        attn, spec = self.layer_weights[layer_index], self._layer_ops[layer_index]
        dev = self.device
        handoff = self._ev_handoff.handle if async_mode else -1
        qb_sh = attn["q_b_proj"]
        if isinstance(qb_sh, torch.Tensor):
            qb_full = torch.empty((16384, 1536), dtype=torch.bfloat16, device=dev)
            self._emit(qb_sh, qb_full, 0, 16384, spec["qb"], 0, tok,
                       async_mode=async_mode, deps=deps, handoff=handoff)
            return (qb_full,)
        qb_v = torch.empty((16384, 1536), dtype=torch.float8_e4m3fn, device=dev)
        if isinstance(qb_sh, BlockFP8Weight):
            qb_s = torch.empty((128, 12), dtype=torch.float32, device=dev)
            src_v, src_s = qb_sh.values, qb_sh.scales
        else:
            qb_s = torch.empty((128, 12, 32, 4, 4), dtype=torch.uint8, device=dev)
            src_v, src_s = qb_sh.weight.values, _mxfp8_scale_phys(qb_sh)[0]
        self._emit(src_v, qb_v, 0, 16384, spec["qb"], 0, tok,
                   async_mode=async_mode, deps=deps, handoff=handoff)
        self._emit(src_s, qb_s, 0, 128, spec["qb_sc"], 0, tok,
                   async_mode=async_mode, deps=[qb_v], handoff=handoff)
        return (qb_v, qb_s)

    def _gather_mla_kvb(self, layer_index: int, tok: torch.Tensor, *,
                        async_mode: bool, deps: list[torch.Tensor] = (),
                        done_ev: int = -1) -> tuple[torch.Tensor, ...]:
        attn, spec = self.layer_weights[layer_index], self._layer_ops[layer_index]
        dev = self.device
        handoff = self._ev_handoff.handle if async_mode else -1
        kvb_sh = attn["kv_b_proj"]
        if isinstance(kvb_sh, torch.Tensor):
            kvb_full = torch.empty((32768, 512), dtype=torch.bfloat16, device=dev)
            self._emit(kvb_sh, kvb_full, 0, 32768, spec["kvb"], 0, tok,
                       async_mode=async_mode, deps=deps, handoff=handoff, done_ev=done_ev)
            return (kvb_full,)
        kvb_v = torch.empty((32768, 512), dtype=torch.float8_e4m3fn, device=dev)
        kvb_s = torch.empty((32768, 16), dtype=torch.uint8, device=dev)
        self._emit(kvb_sh.weight.values, kvb_v, 0, 32768, spec["kvb"], 0, tok,
                   async_mode=async_mode, deps=deps, handoff=handoff)
        self._emit(kvb_sh.weight.scale_rows[0].view(torch.uint8), kvb_s, 0, 32768, spec["kvb_sc"], 0, tok,
                   async_mode=async_mode, deps=[kvb_v], handoff=handoff, done_ev=done_ev)
        return (kvb_v, kvb_s)

    def gather_sync(self, layer_index: int, *, channel_token: torch.Tensor | None = None) -> GatheredKdaWeights | GatheredMlaWeights:
        """Gather all weights for ``layer_index`` synchronously on the main stream."""
        ltype = self.layer_types[layer_index]
        tok = self.channel_token if channel_token is None else channel_token
        if ltype == "linear_attention":
            bufs, _ = self._gather_kda_input_coalesced(layer_index, tok, async_mode=False)
            inp_w = self._unpack_kda_input(layer_index, bufs)
            o_bufs = self._gather_col_sharded_o(layer_index, in_features=8192, tok=tok, async_mode=False)
            o_full = self._unpack_kda_o(layer_index, o_bufs)
            return GatheredKdaWeights(
                input_proj=inp_w.input_proj, f_b_proj=inp_w.f_b_proj, g_b_proj=inp_w.g_b_proj,
                o_proj=o_full, conv1d=inp_w.conv1d, dt_bias=inp_w.dt_bias, A_log=inp_w.A_log,
            )
        qb_bufs = self._gather_mla_qb(layer_index, tok, async_mode=False)
        kvb_bufs = self._gather_mla_kvb(layer_index, tok, async_mode=False)
        inp_w = self._unpack_mla_input(layer_index, (*qb_bufs, *kvb_bufs))
        o_bufs = self._gather_col_sharded_o(layer_index, in_features=16384, tok=tok, async_mode=False)
        o_full = self._unpack_mla_o(layer_index, o_bufs)
        return GatheredMlaWeights(q_b_proj=inp_w.q_b_proj, absorb=inp_w.absorb, expand=inp_w.expand, o_proj=o_full)

    # ------------------------------------------------------------------
    # 4-phase zero-stall async prefetch (PCIe DMA + coalesced async RoCE)
    # ------------------------------------------------------------------

    def start_phase1a(self, layer_index: int, dependencies: list[torch.Tensor], *,
                      channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Phase 1a: KDA q_proj (+ q_sc) | MLA q_b_proj."""
        ltype = self.layer_types[layer_index]
        tok = self.channel_token if channel_token is None else channel_token
        ev_inp = self._ev_inp_done[layer_index]
        deps = list(dependencies)
        if ltype == "linear_attention":
            attn, spec = self.layer_weights[layer_index], self._layer_ops[layer_index]
            ch, dev, handoff = self._kda_ch, self.device, self._ev_handoff.handle
            inp = attn["input_proj"]
            if isinstance(inp, torch.Tensor):
                full_inp = torch.empty((24768, 4096), dtype=torch.bfloat16, device=dev)
                self._emit(inp[0:ch], full_inp, 0, 8192, spec["qkv"], 0, tok,
                           async_mode=True, deps=deps, handoff=handoff)
                return PendingWeightGather(layer_index, ltype, (full_inp,), ev_inp)
            p0 = inp.parts[0]
            mt, kt = ch // 128, p0.padded_in_features // 128
            qkv_vals = torch.empty((24576, p0.padded_in_features), dtype=torch.float8_e4m3fn, device=dev)
            qkv_sc = torch.empty((192, kt, 32, 4, 4), dtype=torch.uint8, device=dev)
            self._emit(p0.weight.values[0:ch], qkv_vals, 0, 8192, spec["qkv"], 0, tok,
                       async_mode=True, deps=deps, handoff=handoff)
            self._emit(_mxfp8_scale_phys(p0)[0][0:mt], qkv_sc, 0, 64, spec["qkv_sc"], 0, tok,
                       async_mode=True, deps=[qkv_vals], handoff=handoff)
            return PendingWeightGather(layer_index, ltype, (qkv_vals, qkv_sc), ev_inp)
        bufs = self._gather_mla_qb(layer_index, tok, async_mode=True, deps=deps)
        return PendingWeightGather(layer_index, ltype, bufs, ev_inp)

    def start_phase1b(self, pending1a: PendingWeightGather, dependencies: list[torch.Tensor], *,
                      channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Phase 1b: KDA k_proj + small | MLA kv_b_proj (signals ev_inp_done for MLA)."""
        layer_index, ltype = pending1a.layer_index, pending1a.kind
        tok = self.channel_token if channel_token is None else channel_token
        ev_inp = self._ev_inp_done[layer_index]
        deps = [*dependencies, *pending1a.buffers]
        if ltype == "linear_attention":
            attn, spec = self.layer_weights[layer_index], self._layer_ops[layer_index]
            ch, dev, w, handoff = self._kda_ch, self.device, self.world, self._ev_handoff.handle
            inp = attn["input_proj"]
            small_sh = self._pack_kda_small(attn)
            small_full = torch.empty((w * self._kda_small_rows, 4096), dtype=torch.bfloat16, device=dev)
            if isinstance(inp, torch.Tensor):
                (full_inp,) = pending1a.buffers
                self._emit(inp[ch:2 * ch], full_inp, 8192, 16384, spec["qkv"], 0, tok, async_mode=True, deps=deps)
                self._emit(small_sh, small_full, 0, small_full.shape[0], spec["small"], 0, tok,
                           async_mode=True, deps=[full_inp, small_sh], handoff=handoff)
                return PendingWeightGather(layer_index, ltype, (full_inp, small_full), ev_inp, extra=(small_sh,))
            p0 = inp.parts[0]
            mt = ch // 128
            qkv_vals, qkv_sc = pending1a.buffers
            self._emit(p0.weight.values[ch:2 * ch], qkv_vals, 8192, 16384, spec["qkv"], 0, tok, async_mode=True, deps=deps)
            self._emit(_mxfp8_scale_phys(p0)[0][mt:2 * mt], qkv_sc, 64, 128, spec["qkv_sc"], 0, tok, async_mode=True, deps=[qkv_vals])
            self._emit(small_sh, small_full, 0, small_full.shape[0], spec["small"], 0, tok,
                       async_mode=True, deps=[qkv_sc, small_sh], handoff=handoff)
            return PendingWeightGather(layer_index, ltype, (qkv_vals, qkv_sc, small_full), ev_inp, extra=(small_sh,))
        kvb_bufs = self._gather_mla_kvb(layer_index, tok, async_mode=True, deps=deps, done_ev=ev_inp.handle)
        return PendingWeightGather(layer_index, ltype, (*pending1a.buffers, *kvb_bufs), ev_inp)

    def start_phase2a(self, pending1b: PendingWeightGather, dependencies: list[torch.Tensor], *,
                      channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Phase 2a: KDA v_proj (signals ev_inp_done) | MLA split o_proj h0."""
        layer_index, ltype = pending1b.layer_index, pending1b.kind
        attn, spec = self.layer_weights[layer_index], self._layer_ops[layer_index]
        tok = self.channel_token if channel_token is None else channel_token
        ev_inp = self._ev_inp_done[layer_index]
        deps = [*dependencies, *pending1b.buffers]
        if ltype == "linear_attention":
            ch = self._kda_ch
            inp = attn["input_proj"]
            if isinstance(inp, torch.Tensor):
                full_inp, small_full = pending1b.buffers
                self._emit(inp[2 * ch:3 * ch], full_inp, 16384, 24576, spec["qkv"], 0, tok,
                           async_mode=True, deps=deps, done_ev=ev_inp.handle)
                return PendingWeightGather(layer_index, ltype, (full_inp, small_full), ev_inp, extra=pending1b.extra)
            p0 = inp.parts[0]
            mt = ch // 128
            qkv_vals, qkv_sc, small_full = pending1b.buffers
            self._emit(p0.weight.values[2 * ch:3 * ch], qkv_vals, 16384, 24576, spec["qkv"], 0, tok, async_mode=True, deps=deps)
            self._emit(_mxfp8_scale_phys(p0)[0][2 * mt:3 * mt], qkv_sc, 128, 192, spec["qkv_sc"], 0, tok,
                       async_mode=True, deps=[qkv_vals], done_ev=ev_inp.handle)
            return PendingWeightGather(layer_index, ltype, (qkv_vals, qkv_sc, small_full), ev_inp, extra=pending1b.extra)

        o_sh = attn["o_proj"]
        if isinstance(o_sh, torch.Tensor) and spec["mla_o_split"] and not self.use_roce_2d:
            h0 = torch.empty((self.world * 2048, 16384 // self.world), dtype=torch.bfloat16, device=self.device)
            self._emit(o_sh[:2048], h0, 0, h0.shape[0], spec["o"], 0, tok,
                       async_mode=True, deps=deps, handoff=self._ev_handoff.handle)
            return PendingWeightGather(layer_index, ltype, pending1b.buffers, ev_inp, extra=pending1b.extra, o_buffers=(h0,))
        return pending1b

    def start_phase2b(self, pending2a: PendingWeightGather, dependencies: list[torch.Tensor], *,
                      channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Phase 2b: KDA o_proj | MLA o_proj (or h1 if split), signaling ev_o_done."""
        layer_index, ltype = pending2a.layer_index, pending2a.kind
        tok = self.channel_token if channel_token is None else channel_token
        ev_o = self._ev_o_done[layer_index]
        deps = [*dependencies, *pending2a.buffers, *pending2a.o_buffers]
        in_features = 8192 if ltype == "linear_attention" else 16384
        pre_h0 = pending2a.o_buffers[0] if (ltype == "indexed_attention" and pending2a.o_buffers) else None
        o_bufs = self._gather_col_sharded_o(
            layer_index, in_features=in_features, tok=tok, async_mode=True,
            deps=deps, done_ev=ev_o.handle, pre_h0=pre_h0,
        )
        return PendingWeightGather(
            layer_index, ltype, pending2a.buffers, pending2a.done_ev,
            extra=pending2a.extra, o_buffers=o_bufs, o_done_ev=ev_o,
        )

    def start_input_all(self, layer_index: int, dependencies: list[torch.Tensor], *,
                        channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Enqueue all input weights (Phases 1a-2a) back-to-back on the auxiliary stream."""
        ltype = self.layer_types[layer_index]
        if self.use_roce_2d and ltype == "linear_attention":
            tok = self.channel_token if channel_token is None else channel_token
            ev_inp = self._ev_inp_done[layer_index]
            bufs, small_sh = self._gather_kda_input_coalesced(
                layer_index, tok, async_mode=True, deps=list(dependencies), done_ev=ev_inp.handle)
            return PendingWeightGather(layer_index, ltype, bufs, ev_inp, extra=(small_sh,))
        p1a = self.start_phase1a(layer_index, dependencies, channel_token=channel_token)
        p1b = self.start_phase1b(p1a, [], channel_token=channel_token)
        return self.start_phase2a(p1b, [], channel_token=channel_token)

    def start_o_only(self, layer_index: int, dependencies: list[torch.Tensor], *,
                     channel_token: torch.Tensor | None = None) -> PendingWeightGather:
        """Enqueue only ``o_proj`` for ``layer_index`` on the auxiliary stream."""
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
        stream_event_wait(list(pending.buffers), [*dependencies, *pending.extra], pending.done_ev.handle)
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
        stream_event_wait(list(pending.o_buffers), list(dependencies), pending.o_done_ev.handle)
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
