# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded checkpoint packing fills physical tier rows without persistent temporaries."""

import os
import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
from hyperflare_recipes.components.megamoe_tiers import MegaMoeHostTier, MegaMoeLayerLayout
from hyperflare_recipes.components.megamoe_weights import (
    allocate_mxfp4_weights, load_mxfp4_planes, load_mxfp4_weights_into, pack_mxfp4_planes,
)
from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from tests.components.test_megamoe_weights_cuda import source

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def test_bounded_csf_loading_preserves_rows_and_retained_memory(source, megamoe_core_storage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    root, _ = source
    manifest = inspect_indexed_checkpoint(root)
    prefixes = ("experts.1", "experts.0")
    storage = megamoe_core_storage
    hbm = allocate_mxfp4_weights(2, 128, 256, device="cuda:0")
    tier = MegaMoeHostTier(MegaMoeGeometry(4, 8, 1, 128, 256, 0), rows=2,
                          storage=storage, name="loaded")
    retained = storage._persistent_arena.used
    try:
        with DirectWeightLoader(manifest, "cuda:0") as loader:
            expected = pack_mxfp4_planes(load_mxfp4_planes(loader, prefixes,
                hidden_size=128, intermediate_size=256), temporary=True)
            for batch in (1, 3):
                for target in (tier.cpu_weights, hbm):
                    for value in (*target.gate_up, *target.down):
                        value.zero_()
                    actual = load_mxfp4_weights_into(loader, prefixes, target,
                        hidden_size=128, intermediate_size=256, batch_experts=batch)
                    torch.cuda.synchronize()
                    assert actual is target
                    assert storage._persistent_arena.used == retained
                    for a, b in zip((*target.gate_up, *target.down), (*expected.gate_up, *expected.down)):
                        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
            with pytest.raises(ValueError, match="positive batch size"):
                load_mxfp4_weights_into(loader, prefixes, hbm, hidden_size=128,
                                        intermediate_size=256, batch_experts=0)
            with pytest.raises(ValueError, match="unique expert prefixes"):
                load_mxfp4_weights_into(loader, (prefixes[0], prefixes[0]), hbm,
                                        hidden_size=128, intermediate_size=256)
            with pytest.raises(ValueError, match="destination geometry"):
                load_mxfp4_weights_into(loader, prefixes, hbm, hidden_size=128,
                                        intermediate_size=256, intermediate_range=(0, 128))
    finally:
        torch.cuda.synchronize()
        for region in tier.regions:
            region.close()


def test_layer_layout_loads_canonical_copies_and_rejects_invalid_placements(source, tmp_path, megamoe_core_storage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    root, planes = source
    # Execution workspace requires a 256-wide hidden dimension. The loader-only
    # fixture uses 128, so form a native-scale source with doubled hidden width.
    tensors = {}
    for name, (weight, scale) in planes.items():
        repeat = (2, 1) if name.endswith("w2") else (1, 2)
        tensors[name + ".weight"] = weight.repeat(*repeat)
        tensors[name + ".scale"] = scale.repeat(*repeat)
    save_file(tensors, str(root / "weights.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    workspace = MegaMoeWorkspace(MegaMoeGeometry(4, 8, 1, 256, 256, 0),
                                 device="cuda:0", kernel_cache=tmp_path)
    layout = MegaMoeLayerLayout(workspace, (0, 1), (1, 0), megamoe_core_storage)
    with DirectWeightLoader(inspect_indexed_checkpoint(root), "cuda:0") as loader:
        prefixes = ("experts.0", "experts.1")
        retained = megamoe_core_storage._persistent_arena.used
        for invalid in (replace(layout, host_experts=(0, 0)),
                        replace(layout, hbm_experts=(1, 2)),
                        replace(layout, host_experts=(0,)),
                        replace(layout, hbm_experts=())):
            with pytest.raises(ValueError, match="cover canonical experts"):
                invalid.load(loader, prefixes, shared_prefix=None, block_size=(32, 32), name="invalid")
            assert megamoe_core_storage._persistent_arena.used == retained
        weights = layout.load(loader, prefixes, shared_prefix=None, block_size=(32, 32), name="copies")
        try:
            assert weights.shared is None
            assert weights.expert_map.tolist() == [3, 2]
            for hbm, host in zip((*weights.routed.gate_up, *weights.routed.down), weights.host_tier.host_views):
                torch.testing.assert_close(hbm.cpu(), host.flip(0), rtol=0, atol=0)
        finally:
            torch.cuda.synchronize()
            for region in weights.host_tier.regions:
                region.close()
