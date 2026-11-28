# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Paged global-layer histories and continuation rows of one Inkling 2 rank.

Global layers keep K/V in 128-token pages stored as FP8 (`fp8`, default) or
block-16 NVFP4 (`nvfp4_4o6k`: 4-or-6 MAE block-scaled K and max/6 block-scaled
V, cutting global KV page bytes by 43.75%). Each continuation row holds, for
every local layer, FP8 K and V rings of `window + PAGE_TOKENS` positions;
position `p` occupies slot `p mod ring`. A ring of one extra page lets decode
read the window through a rotated page table: the pages from the one holding
position `max(p - window + 1, 0)` through the one holding `p` list the window's
keys in position order, which the relative-distance bias requires, and none of
them has been overwritten. Each row also holds the raw `W - 1` most recent
inputs of every layer's K/V short convolutions, the reduced BF16 raw outputs
before the attention-output and expert-output convolutions, partial pages of
the global layers for snapshots, the last normalized hidden state from which
a resumed request samples, and the consumed-token count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from hyperflare_recipes.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

PAGE_TOKENS = 128
KV_DTYPE = torch.float8_e4m3fn
KV_FORMATS = ("fp8", "nvfp4_4o6k")
NVFP4_GROUP = 16


@dataclass(frozen=True)
class RankGeometry:
    """Heads, KV heads and expert slice one rank holds."""

    heads: int
    kv_heads: int
    head_dim: int
    hidden: int
    window: int
    convolution_width: int
    kv_format: str = "fp8"

    def __post_init__(self):
        if self.kv_format not in KV_FORMATS:
            raise ValueError(f"unsupported Inkling KV cache format: {self.kv_format!r}")

    @classmethod
    def of(cls, text, world_size, *, kv_format: str = "fp8"):
        if text.heads % world_size or text.kv_heads % world_size:
            raise ValueError("query and KV heads must divide evenly among ranks")
        return cls(text.heads // world_size, text.kv_heads // world_size, text.head_dim, text.hidden, text.window,
                   text.convolution_width, kv_format=kv_format)

    @property
    def ring(self):
        return self.window + PAGE_TOKENS

    @property
    def page_bytes(self):
        return PAGE_TOKENS * self.kv_heads * self.head_dim * KV_DTYPE.itemsize

    @property
    def global_data_bytes(self):
        return self.page_bytes // 2 if self.kv_format == "nvfp4_4o6k" else self.page_bytes

    @property
    def global_scale_bytes(self):
        return self.page_bytes // NVFP4_GROUP if self.kv_format == "nvfp4_4o6k" else 0


def page_specs(text, geometry: RankGeometry, draft=None):
    for layer in text.global_layers:
        for kind in ("key", "value"):
            yield f"layer{layer}.{kind}", (geometry.global_data_bytes,)
            if geometry.global_scale_bytes:
                yield f"layer{layer}.{kind}_scales", (geometry.global_scale_bytes,)
    if draft is not None:
        for layer in range(draft.layers):
            if layer not in draft.local_layers:
                for kind in ("key", "value"):
                    yield f"draft{layer}.{kind}", (geometry.page_bytes * 2,)


def continuation_layout(text, geometry: RankGeometry, draft=None):
    if geometry.ring % PAGE_TOKENS:
        raise ValueError("sliding-window rings must hold whole pages")
    regions = []
    for layer in range(text.layers):
        if text.local(layer):
            for kind in ("key", "value"):
                regions.append((f"layer{layer}.{kind}", (geometry.ring, geometry.kv_heads, geometry.head_dim), KV_DTYPE))
    history = geometry.convolution_width - 1
    for layer in range(text.layers):
        for kind, channels in (("key_convolution", geometry.kv_heads * geometry.head_dim),
                               ("value_convolution", geometry.kv_heads * geometry.head_dim),
                               ("attention_convolution", geometry.hidden), ("mlp_convolution", geometry.hidden)):
            regions.append((f"layer{layer}.{kind}", (history, channels), torch.bfloat16))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(text, geometry, draft))
    if draft is not None:
        for layer in sorted(draft.local_layers):
            for kind in ("key", "value"):
                regions.append((f"draft{layer}.{kind}",
                                (draft.window, geometry.kv_heads, geometry.head_dim), torch.bfloat16))
        regions.append(("draft.rnn_context", (geometry.hidden,), torch.bfloat16))
    regions.extend((("last_hidden", (geometry.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    layout = StateLayout.from_specs(regions)
    # Ring pages of adjacent rows share one page stride when rows hold whole pages.
    alignment = math.lcm(256, geometry.page_bytes)
    return StateLayout(layout.regions, (layout.row_bytes + alignment - 1) // alignment * alignment)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    text: object
    geometry: RankGeometry
    draft: object | None = None

    @classmethod
    def allocate(cls, text, geometry, slots, pages, device, *, storage=None, maximum_pages=None, maximum_rows=None,
                 backed_slots=None, draft=None):
        layout = continuation_layout(text, geometry, draft)
        backing = PagedStateStorage.allocate(layout, page_specs(text, geometry, draft), slots, pages, device,
            storage=storage, maximum_pages=maximum_pages, maximum_rows=maximum_rows, backed_slots=backed_slots)
        return cls(backing.backing, backing.layout, backing.memory_capacity, backing.pages,
                   backing.reserved_backing, text, geometry, draft)

    def key(self, layer, *, draft=False):
        return self._history_cache(layer, "key", draft=draft)

    def value(self, layer, *, draft=False):
        return self._history_cache(layer, "value", draft=draft)

    def _history_cache(self, layer, kind, *, draft):
        if not draft:
            return self.cache(layer, kind)
        if layer in self.draft.local_layers:
            return self.reserved_region(f"draft{layer}.{kind}")
        g = self.geometry
        return self.pages[f"draft{layer}.{kind}"].view(torch.bfloat16).view(
            -1, PAGE_TOKENS, g.kv_heads, g.head_dim)

    def cache(self, layer, kind):
        """Paged K or V of a global layer (`fp8` or packed `uint8` for `nvfp4_4o6k`)."""
        if self.text.local(layer):
            raise ValueError(f"layer {layer} keeps its keys in continuation rings")
        g = self.geometry
        if g.kv_format == "nvfp4_4o6k":
            return self.pages[f"layer{layer}.{kind}"].view(-1, PAGE_TOKENS, g.kv_heads, g.head_dim // 2)
        return self.pages[f"layer{layer}.{kind}"].view(KV_DTYPE).view(-1, PAGE_TOKENS, g.kv_heads, g.head_dim)

    def cache_scales(self, layer, kind):
        """Block-16 E4M3 scale bytes of a global NVFP4 layer, or None for FP8."""
        if self.text.local(layer):
            raise ValueError(f"layer {layer} keeps its keys in continuation rings")
        g = self.geometry
        if g.kv_format != "nvfp4_4o6k":
            return None
        return self.pages[f"layer{layer}.{kind}_scales"].view(-1, PAGE_TOKENS, g.kv_heads, g.head_dim // NVFP4_GROUP)

    def history(self, layer, kind):
        """`[rows, W - 1, channels]` raw inputs of a short convolution, oldest first."""
        return self.reserved_region(f"layer{layer}.{kind}")

    @property
    def ring_row_pages(self):
        """Pages between the same ring slot of adjacent continuation rows."""
        if self.layout.row_bytes % self.geometry.page_bytes:
            raise ValueError("continuation rows must hold whole KV pages")
        return self.layout.row_bytes // self.geometry.page_bytes

    def ring_rows(self, layer, kind):
        """A local layer's K or V ring in every reserved row, as bytes `[rows, ring * kv_heads * head_dim]`."""
        if not self.text.local(layer):
            raise ValueError(f"layer {layer} keeps its keys in paged regions")
        region = self.layout.region(f"layer{layer}.{kind}")
        return self.reserved_backing[:, region.offset:region.offset + region.nbytes]

    def ring_pages(self, layer, kind):
        """A local layer's K or V rings of every reserved row as pages.

        Ring page `j` of state row `r` is page `r * ring_row_pages + j`; the K and V
        views of a layer share that page numbering.
        """
        if not self.text.local(layer):
            raise ValueError(f"layer {layer} keeps its keys in paged regions")
        g = self.geometry
        region = self.layout.region(f"layer{layer}.{kind}")
        pages = (self.reserved_backing.shape[0] - 1) * self.ring_row_pages + g.ring // PAGE_TOKENS
        return self.reserved_backing.view(KV_DTYPE).as_strided(
            (pages, PAGE_TOKENS, g.kv_heads, g.head_dim),
            (PAGE_TOKENS * g.kv_heads * g.head_dim, g.kv_heads * g.head_dim, g.head_dim, 1),
            storage_offset=region.offset // KV_DTYPE.itemsize)
