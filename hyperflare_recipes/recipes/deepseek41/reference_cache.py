# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete continuation storage exposing DeepInfer's attention cache views."""

from dataclasses import dataclass
from math import lcm

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout, gather_rows, scatter_rows
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor


@triton.jit
def _snapshot_ring_slots(
    Backing, StateRows, Starts, Lengths,
    RingBases, RingLens, RingWords, SnapOffsets,
    Output,
    BACKING_STRIDE: tl.constexpr,
    SNAP_STRIDE_B: tl.constexpr,
    SNAP_STRIDE_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    r = tl.program_id(1)
    k = tl.program_id(2)
    length = tl.load(Lengths + b).to(tl.int64)
    if k >= length:
        return
    state_row = tl.load(StateRows + b).to(tl.int64)
    start = tl.load(Starts + b).to(tl.int64)
    r_base = tl.load(RingBases + r).to(tl.int64)
    r_len = tl.load(RingLens + r).to(tl.int64)
    r_words = tl.load(RingWords + r).to(tl.int32)
    s_off = tl.load(SnapOffsets + r).to(tl.int64)
    slot = (start + k) % r_len
    src_base = state_row * BACKING_STRIDE + r_base + slot * r_words
    dst_base = b.to(tl.int64) * SNAP_STRIDE_B + k.to(tl.int64) * SNAP_STRIDE_K + s_off
    offs = tl.arange(0, BLOCK)
    for col in range(0, r_words, BLOCK):
        idx = col + offs
        mask = idx < r_words
        val = tl.load(Backing + src_base + idx, mask=mask, other=0)
        tl.store(Output + dst_base + idx, val, mask=mask)


@triton.jit
def _restore_ring_slots(
    Backing, StateRows, Starts, Lengths, Counts,
    RingBases, RingLens, RingWords, SnapOffsets,
    Snapshot,
    BACKING_STRIDE: tl.constexpr,
    SNAP_STRIDE_B: tl.constexpr,
    SNAP_STRIDE_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    r = tl.program_id(1)
    k = tl.program_id(2)
    length = tl.load(Lengths + b).to(tl.int64)
    accepted = tl.load(Counts + b).to(tl.int64)
    if (length <= 0) | (k < accepted) | (k >= length):
        return
    state_row = tl.load(StateRows + b).to(tl.int64)
    start = tl.load(Starts + b).to(tl.int64)
    r_base = tl.load(RingBases + r).to(tl.int64)
    r_len = tl.load(RingLens + r).to(tl.int64)
    r_words = tl.load(RingWords + r).to(tl.int32)
    s_off = tl.load(SnapOffsets + r).to(tl.int64)
    slot = (start + k) % r_len
    dst_base = state_row * BACKING_STRIDE + r_base + slot * r_words
    src_base = b.to(tl.int64) * SNAP_STRIDE_B + k.to(tl.int64) * SNAP_STRIDE_K + s_off
    offs = tl.arange(0, BLOCK)
    for col in range(0, r_words, BLOCK):
        idx = col + offs
        mask = idx < r_words
        val = tl.load(Snapshot + src_base + idx, mask=mask, other=0)
        tl.store(Backing + dst_base + idx, val, mask=mask)


@triton.jit
def _gather_pages_4(
    V0, R0, O0, W0: tl.constexpr,
    V1, R1, O1, W1: tl.constexpr,
    V2, R2, O2, W2: tl.constexpr,
    V3, R3, O3, W3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    r0 = tl.load(R0 + t).to(tl.int64)
    for col in range(0, W0, BLOCK):
        c = col + offs
        m = c < W0
        tl.store(O0 + t * W0 + c, tl.load(V0 + r0 * W0 + c, mask=m, other=0), mask=m)
    r1 = tl.load(R1 + t).to(tl.int64)
    for col in range(0, W1, BLOCK):
        c = col + offs
        m = c < W1
        tl.store(O1 + t * W1 + c, tl.load(V1 + r1 * W1 + c, mask=m, other=0), mask=m)
    r2 = tl.load(R2 + t).to(tl.int64)
    for col in range(0, W2, BLOCK):
        c = col + offs
        m = c < W2
        tl.store(O2 + t * W2 + c, tl.load(V2 + r2 * W2 + c, mask=m, other=0), mask=m)
    r3 = tl.load(R3 + t).to(tl.int64)
    for col in range(0, W3, BLOCK):
        c = col + offs
        m = c < W3
        tl.store(O3 + t * W3 + c, tl.load(V3 + r3 * W3 + c, mask=m, other=0), mask=m)


@triton.jit
def _scatter_pages_4(
    S0, R0, V0, A0, W0: tl.constexpr, N0: tl.constexpr,
    S1, R1, V1, A1, W1: tl.constexpr, N1: tl.constexpr,
    S2, R2, V2, A2, W2: tl.constexpr, N2: tl.constexpr,
    S3, R3, V3, A3, W3: tl.constexpr, N3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    r0 = tl.load(R0 + t).to(tl.int64)
    if (tl.load(A0 + t) != 0) & (r0 >= 0) & (r0 < N0):
        for col in range(0, W0, BLOCK):
            c = col + offs
            m = c < W0
            tl.store(V0 + r0 * W0 + c, tl.load(S0 + t * W0 + c, mask=m, other=0), mask=m)
    r1 = tl.load(R1 + t).to(tl.int64)
    if (tl.load(A1 + t) != 0) & (r1 >= 0) & (r1 < N1):
        for col in range(0, W1, BLOCK):
            c = col + offs
            m = c < W1
            tl.store(V1 + r1 * W1 + c, tl.load(S1 + t * W1 + c, mask=m, other=0), mask=m)
    r2 = tl.load(R2 + t).to(tl.int64)
    if (tl.load(A2 + t) != 0) & (r2 >= 0) & (r2 < N2):
        for col in range(0, W2, BLOCK):
            c = col + offs
            m = c < W2
            tl.store(V2 + r2 * W2 + c, tl.load(S2 + t * W2 + c, mask=m, other=0), mask=m)
    r3 = tl.load(R3 + t).to(tl.int64)
    if (tl.load(A3 + t) != 0) & (r3 >= 0) & (r3 < N3):
        for col in range(0, W3, BLOCK):
            c = col + offs
            m = c < W3
            tl.store(V3 + r3 * W3 + c, tl.load(S3 + t * W3 + c, mask=m, other=0), mask=m)


@dataclass(frozen=True)
class ReferenceLayerCache:
    window: str | None
    latent: str | None
    latent_state: str | None
    index: str | None
    index_state: str | None
    ratio: int
    latent_bytes: int
    index_bytes: int


@dataclass(frozen=True)
class ReferenceCacheLayout:
    continuation: StateLayout
    pages: tuple
    layers: tuple[ReferenceLayerCache, ...]
    window_tokens: int
    verification_width: int
    page_tokens: int

    @classmethod
    def from_config(cls, cfg, *, verification_width, page_tokens=128, dspark=True):
        from deepinfer.model_executor.layer_utils import iter_runtime_attn_modules
        from deepinfer.model_executor.layers.tilelang_ops.cache_common import V41_FORMAT, V41_FP4_FORMAT
        from deepinfer.model_executor.models.v4_modules.cross_layer import build_shared_layer_topology

        if (type(verification_width) is not int or verification_width < 1
                or type(page_tokens) is not int or page_tokens < 1
                or type(dspark) is not bool):
            raise ValueError("reference cache requires positive page and verification capacities")
        modules = tuple(iter_runtime_attn_modules(cfg.attn_module_list, cfg.num_hidden_layers))
        if not dspark:
            modules = modules[:cfg.num_hidden_layers]
        topology = build_shared_layer_topology(cfg, len(modules))
        dimension = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        window = cfg.window_size[0]
        if window <= 0 or dimension != V41_FORMAT.num_elements:
            raise ValueError("reference cache requires the checkpoint's declared sliding-window MLA geometry")
        window_tokens = window + verification_width - 1
        latent_bytes = (V41_FP4_FORMAT if cfg.use_fp4_sparse_kv else V41_FORMAT).bytes_per_token
        specs, pages, layers = [], [], []
        for layer, _, module in modules:
            c = module.cfg
            ratio = c.get("kv_compress_ratio", 0)
            if ratio not in (0, 1, 2) or (ratio and page_tokens % ratio):
                raise ValueError("reference cache page must contain whole C1/C2 compression blocks")
            names = dict(window=None, latent=None, latent_state=None, index=None, index_state=None)
            if c.get("sink_mode", "") == "swa":
                names["window"] = f"layer{layer}.window"
                specs.append((names["window"], (window_tokens, 1, V41_FORMAT.bytes_per_token), torch.uint8))
            elif not ratio:
                raise ValueError("reference cache requires sliding-window or compressed attention")
            index_dim = c.attn_index.index_head_dim if c.get("index_topk", 0) else 0
            if index_dim and not ratio:
                raise ValueError("reference index cache binding requires compressed attention geometry")
            if index_dim % 32:
                raise ValueError("reference FP4 index cache requires complete 32-element scale blocks")
            index_bytes = index_dim // 2 + index_dim // 32
            for kind, source, record_bytes, state_dim, needs_state in (
                ("latent", layer if topology is None else topology.layer_kv_map[layer], latent_bytes,
                 dimension, ratio > 1),
                ("index", layer if topology is None else topology.layer_index_k_map[layer], index_bytes,
                 index_dim, ratio > 1 and not c.get("attn_index", {}).get("use_kproj", False)),
            ):
                if not ratio or (kind == "index" and not index_dim):
                    continue
                if source != layer:
                    if not 0 <= source < len(layers) or layers[source].ratio != ratio:
                        raise ValueError("reference shared cache source must precede its compatible consumer")
                    names[kind] = getattr(layers[source], kind)
                    if names[kind] is None or getattr(layers[source], kind + "_bytes") != record_bytes:
                        raise ValueError("reference shared cache has incompatible record geometry")
                else:
                    names[kind] = f"layer{layer}.{kind}_page"
                    pages.append((names[kind], (page_tokens // ratio * record_bytes,)))
                    if needs_state:
                        names[kind + "_state"] = f"layer{layer}.{kind}_history"
                        specs.append((names[kind + "_state"],
                                      (ratio + verification_width - 1, 2, state_dim), torch.float32))
            layers.append(ReferenceLayerCache(**names, ratio=ratio,
                latent_bytes=latent_bytes, index_bytes=index_bytes))
        # Partial pages and every history ring participate in continuation copies.
        specs.extend((name, shape, torch.uint8) for name, shape in pages)
        specs.extend((("last_hidden", (cfg.hidden_size,), torch.bfloat16),
                      ("consumed_tokens", (1,), torch.int64), ("engram_history", (3,), torch.int64)))
        layout = StateLayout.from_specs(specs)
        # FlashMLA flattens cache blocks into whole encoded-token rows for TMA.
        stride = lcm(256, V41_FORMAT.bytes_per_token)
        layout = StateLayout(layout.regions, (layout.row_bytes + stride - 1) // stride * stride)
        return cls(layout, tuple(pages), tuple(layers),
                   window_tokens, verification_width, page_tokens)


@dataclass(frozen=True)
class ReferenceCacheStorage(PagedStateStorage):
    geometry: ReferenceCacheLayout

    @classmethod
    def allocate(cls, geometry, slots, pages, device, **kwargs):
        storage = PagedStateStorage.allocate(geometry.continuation, geometry.pages, slots, pages, device, **kwargs)
        return cls(storage.backing, storage.layout, storage.memory_capacity,
                   storage.pages, storage.reserved_backing, geometry)

    def cache_views(self):
        """Return reference operands aliasing complete native-managed storage."""
        from deepinfer.worker.cache_engine import KVCache

        page_views = {}
        layers = []
        for layer in self.geometry.layers:
            fields = {}
            for kind, field in (("latent", "ca"), ("index", "ica")):
                name = getattr(layer, kind)
                if name is not None and name not in page_views:
                    page_views[name] = self.pages[name].view(self.pages[name].shape[0],
                        self.geometry.page_tokens // layer.ratio, 1, getattr(layer, kind + "_bytes"))
                fields[field + "_cache"] = None if name is None else page_views[name]
                state = getattr(layer, kind + "_state")
                fields[field + "_state"] = None if state is None else self.reserved_region(state)
            fields["swa_cache"] = None if layer.window is None else self.reserved_region(layer.window)
            layers.append(KVCache(**fields))
        return tuple(layers)


class ReferenceCacheTransaction:
    """Retain cache writes only for the accepted verification input prefix.

    Snapshot continuation rings before draft execution can overwrite them and
    retain only the compressed page rows that verification may write. Temporary
    snapshots belong to complete-forward storage planning. Native execution owns
    the backing storage and never performs this model computation.
    """

    def __init__(self, pool):
        from deepinfer.model_executor.layers.tilelang_ops.cache_common import (
            get_kv_cache_format, split_kv_cache,
        )

        self.pool = pool
        rings, pages = {}, {}
        for layer in pool.geometry.layers:
            for name in (layer.window, layer.latent_state, layer.index_state):
                if name is not None:
                    rings[name] = pool.reserved_region(name)
            for name, size in ((layer.latent, layer.latent_bytes), (layer.index, layer.index_bytes)):
                if name is not None and name not in pages:
                    backing = pool.pages[name]
                    cache = backing.view(backing.shape[0], pool.geometry.page_tokens // layer.ratio, 1, size)
                    layout = get_kv_cache_format(cache)
                    sections = (cache,) if layout.token_interleaved else tuple(
                        part for part in split_kv_cache(cache, layout)[1] if part is not None)
                    parts = []
                    for section in sections:
                        part = section.view(torch.uint8)
                        width = part.shape[-1]
                        offset = part.storage_offset() - backing.storage_offset()
                        if backing.shape[1] % width or offset % width:
                            raise ValueError("reference cache section must divide its physical page stride")
                        # Index values and scales occupy separate page sections.
                        # Address each section in its own contiguous byte rows.
                        parts.append((backing.view(-1, width), layer.ratio,
                                      backing.shape[1] // width, offset // width))
                    pages[name] = parts
        self.rings = tuple(rings.values())
        self.pages = tuple(part for parts in pages.values() for part in parts)
        self.max_slots = pool.geometry.verification_width
        device = pool.reserved_backing.device
        ring_names = tuple(rings)
        if ring_names:
            rings_end = max(pool.layout.region(name).offset + pool.layout.region(name).nbytes for name in ring_names)
            if rings_end % 8 or pool.layout.row_bytes % 8:
                raise ValueError("reference continuation ring layout must be 8-byte aligned")
            bases, lens, slot_words_list, snap_offsets = [], [], [], []
            total_slot_words = 0
            for name in ring_names:
                reg = pool.layout.region(name)
                ring_len = reg.shape[0]
                if ring_len < self.max_slots:
                    raise ValueError(f"reference ring {name} length {ring_len} smaller than verification width {self.max_slots}")
                if reg.offset % 8 or reg.nbytes % (ring_len * 8):
                    raise ValueError(f"reference ring {name} slot bytes must be 8-byte aligned")
                slot_words = reg.nbytes // (ring_len * 8)
                begin_word = reg.offset // 8
                bases.append(begin_word)
                lens.append(ring_len)
                slot_words_list.append(slot_words)
                snap_offsets.append(total_slot_words)
                total_slot_words += slot_words
            self.ring_backing = pool.reserved_backing[:, :rings_end].view(torch.int64)
            self._ring_specs = tuple(zip(bases, lens, slot_words_list, snap_offsets, strict=True))
            self.ring_bases = persistent_copy(torch.tensor(bases, dtype=torch.int32, device=device))
            self.ring_lens = persistent_copy(torch.tensor(lens, dtype=torch.int32, device=device))
            self.ring_words = persistent_copy(torch.tensor(slot_words_list, dtype=torch.int32, device=device))
            self.ring_snap_offsets = persistent_copy(torch.tensor(snap_offsets, dtype=torch.int32, device=device))
            self.total_slot_words = total_slot_words
            if device.type == "cuda":
                for tensor in (self.ring_bases, self.ring_lens, self.ring_words, self.ring_snap_offsets):
                    record_numeric_tensor(tensor)
        else:
            self.ring_backing = None
            self._ring_specs = ()
            self.ring_bases = self.ring_lens = self.ring_words = self.ring_snap_offsets = None
            self.total_slot_words = 0

    def begin(self, metadata):
        """Snapshot validated request rows before either draft or target writes."""
        m = metadata
        if self.ring_backing is not None:
            if self.ring_backing.is_cuda:
                batch = m.state_rows.numel()
                max_slots = min(self.max_slots, m.positions.numel())
                rings = torch.empty((batch, max_slots, self.total_slot_words),
                                    dtype=torch.int64, device=self.ring_backing.device)
                torch.library.wrap_triton(_snapshot_ring_slots)[(batch, len(self.rings), max_slots)](
                    self.ring_backing, m.state_rows, m.starts, m.lengths,
                    self.ring_bases, self.ring_lens, self.ring_words, self.ring_snap_offsets,
                    rings, self.ring_backing.stride(0), rings.stride(0), rings.stride(1), 128,
                )
            else:
                rings = gather_rows(self.ring_backing, m.state_rows, torch.empty(
                    (m.state_rows.numel(), self.ring_backing.shape[1]),
                    dtype=torch.int64, device=self.ring_backing.device))
        else:
            rings = ()
        pages = []
        address_cache = {}
        position = m.positions.clamp_min(0)
        physical = m.token_page_table.gather(1,
            (position // self.pool.geometry.page_tokens)[:, None]).flatten().long()
        resolved = []
        for value, ratio, stride, offset in self.pages:
            key = (ratio, stride, offset)
            if key not in address_cache:
                per_page = self.pool.geometry.page_tokens // ratio
                rows = physical.clamp_min(0) * stride + offset + position // ratio % per_page
                written = (m.positions >= 0) & ((position + 1) % ratio == 0)
                address_cache[key] = (rows, written)
            else:
                rows, written = address_cache[key]
            resolved.append((value, rows, written))
        if resolved and resolved[0][0].is_cuda:
            tokens = m.positions.numel()
            idx = 0
            while idx + 4 <= len(resolved):
                group = resolved[idx:idx + 4]
                outs = [torch.empty((tokens, v.shape[1]), dtype=torch.uint8, device=v.device) for v, _, _ in group]
                torch.library.wrap_triton(_gather_pages_4)[(tokens,)](
                    group[0][0], group[0][1], outs[0], group[0][0].shape[1],
                    group[1][0], group[1][1], outs[1], group[1][0].shape[1],
                    group[2][0], group[2][1], outs[2], group[2][0].shape[1],
                    group[3][0], group[3][1], outs[3], group[3][0].shape[1],
                    512,
                )
                for (_, rows, written), out in zip(group, outs, strict=True):
                    pages.append((rows, written, out))
                idx += 4
            for value, rows, written in resolved[idx:]:
                pages.append((rows, written, value[rows].clone()))
        else:
            for value, rows, written in resolved:
                pages.append((rows, written, value[rows].clone()))
        return rings, tuple(pages)

    def commit(self, metadata, counts, snapshot):
        """Restore rejected writes; inactive requests leave every backing byte intact."""
        m = metadata
        if counts.shape != m.lengths.shape or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("reference cache commit requires one accepted input count per request")
        rings, pages = snapshot
        if self.ring_backing is not None:
            if self.ring_backing.is_cuda:
                batch, max_slots = rings.shape[0], rings.shape[1]
                torch.library.wrap_triton(_restore_ring_slots)[(batch, len(self.rings), max_slots)](
                    self.ring_backing, m.state_rows, m.starts, m.lengths, counts,
                    self.ring_bases, self.ring_lens, self.ring_words, self.ring_snap_offsets,
                    rings, self.ring_backing.stride(0), rings.stride(0), rings.stride(1), 128,
                )
            else:
                for b in range(m.state_rows.numel()):
                    length = int(m.lengths[b].item())
                    accepted = int(counts[b].item())
                    if length <= 0 or accepted >= length:
                        continue
                    row = int(m.state_rows[b].item())
                    start = int(m.starts[b].item())
                    for base_word, ring_len, slot_words, _ in self._ring_specs:
                        for k in range(accepted, length):
                            slot = (start + k) % ring_len
                            lo = base_word + slot * slot_words
                            hi = lo + slot_words
                            self.ring_backing[row, lo:hi].copy_(rings[b, lo:hi])
        rejected = m.positions - m.starts[m.requests] >= counts[m.requests]
        if self.pages and self.pages[0][0].is_cuda:
            tokens = m.positions.numel()
            active_cache = {}
            pairs = []
            for (value, _, _, _), (rows, written, before) in zip(self.pages, pages, strict=True):
                key = id(written)
                if key not in active_cache:
                    active_cache[key] = written & rejected
                pairs.append((before, rows, value, active_cache[key]))
            idx = 0
            while idx + 4 <= len(pairs):
                g = pairs[idx:idx + 4]
                torch.library.wrap_triton(_scatter_pages_4)[(tokens,)](
                    g[0][0], g[0][1], g[0][2], g[0][3], g[0][2].shape[1], g[0][2].shape[0],
                    g[1][0], g[1][1], g[1][2], g[1][3], g[1][2].shape[1], g[1][2].shape[0],
                    g[2][0], g[2][1], g[2][2], g[2][3], g[2][2].shape[1], g[2][2].shape[0],
                    g[3][0], g[3][1], g[3][2], g[3][3], g[3][2].shape[1], g[3][2].shape[0],
                    512,
                )
                idx += 4
            for before, rows, value, active in pairs[idx:]:
                scatter_rows(before, rows, value, active)
        else:
            for (value, _, _, _), (rows, written, before) in zip(self.pages, pages, strict=True):
                scatter_rows(before, rows, value, written & rejected)
