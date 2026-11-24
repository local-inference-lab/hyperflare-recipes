# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hybrid HCA C128 + DSA 656B/FP8-index paged histories and continuation rows."""

from dataclasses import dataclass

import torch

from hyperflare_recipes.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

# 128 tokens per logical model page = 1 HCA C128 record = 2 DSA 64-token sub-pages.
PAGE_TOKENS = 128
DSA_SUBPAGE_TOKENS = 64
DSA_SUBPAGES_PER_PAGE = PAGE_TOKENS // DSA_SUBPAGE_TOKENS
DSA_KV_RECORD_BYTES = 656
INDEX_RECORD_BYTES = 132


def latent_page_bytes(records):
    # V4/ERNIE HCA places FP8/BF16 payloads before a page-local UE8M0 scale footer.
    return ((records * 584 + 575) // 576) * 576


def page_specs(config):
    for layer in range(config.layers):
        if config.is_dsa_layer(layer):
            yield f"layer{layer}.dsa_kv_page", (PAGE_TOKENS * DSA_KV_RECORD_BYTES,)
            yield f"layer{layer}.dsa_index_page", (PAGE_TOKENS * INDEX_RECORD_BYTES,)
        else:
            yield f"layer{layer}.latent_page", (latent_page_bytes(PAGE_TOKENS // 128),)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, *, dspark=True):
    window_bytes = latent_page_bytes(config.text["sliding_window"])
    regions = [
        (f"layer{layer}.window", (window_bytes,), torch.uint8)
        for layer in config.hca_layers
    ]
    for compressor in config.compressors:
        regions.extend(
            (f"{compressor.prefix}.{kind}", compressor.state_shape, torch.float32)
            for kind in ("values", "scores")
        )
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    if dspark:
        regions.extend(
            (f"draft{i}.window", (window_bytes,), torch.uint8)
            for i in range(config.draft.layers)
        )
    regions.extend(
        (
            ("last_hidden", (config.text["hidden_size"],), torch.bfloat16),
            ("consumed_tokens", (1,), torch.int64),
        )
    )
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object

    @classmethod
    def allocate(
        cls,
        config,
        slots,
        pages,
        device,
        *,
        storage=None,
        maximum_pages=None,
        maximum_rows=None,
        backed_slots=None,
        dspark=True,
    ):
        backing = PagedStateStorage.allocate(
            continuation_layout(config, dspark=dspark),
            page_specs(config),
            slots,
            pages,
            device,
            storage=storage,
            maximum_pages=maximum_pages,
            maximum_rows=maximum_rows,
            backed_slots=backed_slots,
        )
        return cls(
            backing.backing,
            backing.layout,
            backing.memory_capacity,
            backing.pages,
            backing.reserved_backing,
            config,
        )

    def window(self, layer, *, draft=False):
        return self.reserved_region(f"{'draft' if draft else 'layer'}{layer}.window")

    def dsa_kv_cache(self, layer):
        # View [pages, 128 * 656] as [pages * 2, 64, 656] for 64-token b12x sparse_mla.
        raw = self.pages[f"layer{layer}.dsa_kv_page"]
        return raw.view(raw.shape[0] * DSA_SUBPAGES_PER_PAGE, DSA_SUBPAGE_TOKENS, DSA_KV_RECORD_BYTES)

    def dsa_index_cache(self, layer):
        # View [pages, 128 * 132] as [pages * 2, 64 * 132] for 64-token b12x dsa_indexer.
        raw = self.pages[f"layer{layer}.dsa_index_page"]
        return raw.view(raw.shape[0] * DSA_SUBPAGES_PER_PAGE, DSA_SUBPAGE_TOKENS * INDEX_RECORD_BYTES)
