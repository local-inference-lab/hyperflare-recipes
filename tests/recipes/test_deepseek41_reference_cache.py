# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete native continuation rows and reference cache aliases."""

import os
from pathlib import Path

import pytest
import torch

from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint
from hyperflare_recipes.recipes.deepseek41.reference_cache import ReferenceCacheLayout, ReferenceCacheStorage


@pytest.fixture(scope="module")
def reference_config():
    pytest.importorskip("deepinfer")
    path = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not path:
        pytest.skip("requires the DeepSeek V4.1 CSF checkpoint")
    return ReferenceCheckpoint(Path(path)).model_config().gpt


@pytest.mark.parametrize("width", [1, 6, 8])
def test_complete_reference_continuation(reference_config, width):
    geometry = ReferenceCacheLayout.from_config(reference_config, verification_width=width)
    pool = ReferenceCacheStorage.allocate(geometry, 3, 2, "cpu")
    views = pool.cache_views()
    assert len(views) == 43
    assert geometry.window_tokens == 128 + width - 1
    assert len(pool.pages) == 8
    for layer in (2, 8, 14):
        assert views[layer].ca_state.shape == (3, width + 1, 2, 512)
        assert views[layer].ica_state is None
        assert views[layer].ca_cache.shape == (2, 64, 1, 288)
        assert views[layer].ica_cache.shape == (2, 64, 1, 68)
    assert views[20].ca_cache.shape == (2, 128, 1, 288)
    assert views[20].ca_state is None
    for owner, last in ((2, 7), (8, 13), (14, 19), (20, 39)):
        for layer in range(owner + 1, last + 1):
            assert views[layer].ca_cache is views[owner].ca_cache
            assert views[layer].ica_cache is views[owner].ica_cache
            assert views[layer].ca_state is None
    for layer in (0, 1, 40, 41, 42):
        assert views[layer].ca_cache is None and views[layer].ica_cache is None
    for index, region in enumerate(pool.layout.regions):
        pool.region(region.name)[1].fill_(index + 1)
    expected = pool.backing[1].clone()
    pool.backing[2].copy_(pool.backing[1])
    pool.backing[1].zero_()
    pool.backing[1].copy_(pool.backing[2])
    assert torch.equal(pool.backing[1], expected)
    assert not pool.backing[0].count_nonzero()
    for layer, view in enumerate(views):
        assert view.swa_cache.shape == (3, 128 + width - 1, 1, 528)
        assert view.swa_cache.data_ptr() == pool.region(f"layer{layer}.window").data_ptr()
        assert view.swa_cache.stride(0) % 528 == 0
        assert view.swa_cache.stride(0) == pool.layout.row_bytes
        if view.ca_state is not None:
            assert view.ca_state.data_ptr() == pool.region(f"layer{layer}.latent_history").data_ptr()
            assert view.ca_state.stride(0) * 4 == pool.layout.row_bytes
    for name, page in pool.partial_pages():
        assert pool.region(name).shape[1:] == page.shape[1:]
    pool.clear_for_preparation()
    assert not pool.backing.count_nonzero()
    assert all(not page.count_nonzero() for page in pool.pages.values())


@pytest.mark.parametrize("width,page", [(0, 128), (True, 128), (8, 127)])
def test_reference_cache_rejects_invalid_capacities(reference_config, width, page):
    with pytest.raises(ValueError):
        ReferenceCacheLayout.from_config(reference_config, verification_width=width, page_tokens=page)
