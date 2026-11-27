# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contract checks for MegaMoE placement, weight layouts, staging and validation."""

from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.delivered_fp8_projection import DeliveredFP8Projection
from hyperflare_recipes.components.expert_placement import ExpertPlacement, PrefillExpertStaging
from hyperflare_recipes.components.megamoe import (
    CONTRACT as MEGAMOE_CONTRACT,
    MegaMoeGeometry,
    MegaMoeQuery,
    _stage_constants,
    _validate_inputs,
    _validate_weights,
)
from hyperflare_recipes.components.megamoe_kernel.inputs import prepare_megamoe_inputs
from hyperflare_recipes.components.megamoe_staging import (
    CONTRACT as STAGE_CONTRACT,
    StageQuery,
)
from hyperflare_recipes.components.megamoe_tiers import MegaMoeHostTier, MegaMoeLayerLayout
from hyperflare_recipes.components.megamoe_weights import (
    MegaMoeWeights,
    allocate_mxfp4_weights,
    mxfp4_weight_layouts,
)


def test_prefill_expert_staging_validation_and_serialization():
    default = PrefillExpertStaging()
    assert default.to_dict() == {"min_rows": 1024, "min_cold_experts": 8}
    custom = PrefillExpertStaging(min_rows=256, min_cold_experts=4)
    assert custom.to_dict() == {"min_rows": 256, "min_cold_experts": 4}
    for kwargs in (
        {"min_rows": 0},
        {"min_rows": -1},
        {"min_cold_experts": 0},
        {"min_rows": True},
        {"min_cold_experts": 2.0},
    ):
        with pytest.raises(ValueError, match="positive integers"):
            PrefillExpertStaging(**kwargs)


def test_expert_placement_coverage_copies_and_rejections():
    disjoint = ExpertPlacement(experts=4, host=(0, 1), hbm=(2, 3))
    assert disjoint.physical_rows == 4
    assert disjoint.to_dict() == {"experts": 4, "host": [0, 1], "hbm": [2, 3]}

    cached = ExpertPlacement(experts=4, host=(0, 1), hbm=(2, 3, 0))
    assert cached.physical_rows == 5
    assert cached.to_dict() == {"experts": 4, "host": [0, 1], "hbm": [2, 3, 0]}

    hbm_only = ExpertPlacement(experts=2, host=(), hbm=(1, 0))
    assert hbm_only.physical_rows == 2

    for args in (
        (0, (), ()),
        (True, (), (0,)),
        (4, [0, 1], (2, 3)),
        (4, (0, 1), [2, 3]),
        (4, (0, 1, 2, 3), ()),
        (4, (0, 0), (1, 2, 3)),
        (4, (0, 1), (2, 3, 3)),
        (4, (0, 1), (2,)),
        (4, (0, 1), (2, 4)),
        (4, (0, 1), (2, True)),
    ):
        with pytest.raises(ValueError, match="complete canonical coverage"):
            ExpertPlacement(*args)


def test_megamoe_geometry_and_query_encoding():
    geometry = MegaMoeGeometry(experts=8, capacity=256, top_k=6, hidden=5120, intermediate=2304, shared_experts=1)
    assert geometry.shared_experts == 1
    assert MegaMoeGeometry(experts=4, capacity=16, top_k=4, hidden=128, intermediate=128, shared_experts=0).top_k == 4

    for args in (
        (0, 16, 1, 128, 128, 0),
        (4, 0, 1, 128, 128, 0),
        (4, 16, 5, 128, 128, 0),
        (4, 16, 2, 192, 128, 0),
        (4, 16, 2, 128, 64, 0),
        (4, 16, 2, 128, 128, 2),
        (4, 16, True, 128, 128, 0),
    ):
        with pytest.raises(ValueError, match="128-aligned widths"):
            MegaMoeGeometry(*args)

    query = MegaMoeQuery(
        rows=64,
        experts=8,
        capacity=256,
        top_k=6,
        hidden=5120,
        intermediate=2304,
        shared_experts=1,
        x_sf_stride=[1, 256],
        shared_sf_stride=[1, 256],
        shared_block_m=64,
        clamp=10.0,
        host_rows=4,
        host_lane_clusters=4,
    )
    assert query.x_sf_stride == (1, 256)
    assert query.shared_sf_stride == (1, 256)
    encoded = MEGAMOE_CONTRACT.encode_query(query)
    assert MegaMoeQuery(**dict(encoded)) == query
    constants = _stage_constants(query)
    assert constants[-6:] == (8, 128, 32, 8, 64, True)


def test_stage_query_validation_and_encoding():
    query = StageQuery(rows=4, words=[1024, 64, 512, 32], min_cold=2)
    assert query.words == (1024, 64, 512, 32)
    encoded = STAGE_CONTRACT.encode_query(query)
    assert StageQuery(**dict(encoded)) == query

    for args in (
        (0, (1, 1, 1, 1), 1),
        (True, (1, 1, 1, 1), 1),
        (4, (1, 1, 1), 1),
        (4, (1, 0, 1, 1), 1),
        (4, (1, True, 1, 1), 1),
        (4, (1, 1, 1, 1), 0),
        (4, (1, 1, 1, 1), 5),
        (4, (1, 1, 1, 1), True),
    ):
        with pytest.raises(ValueError, match="four positive row extents"):
            StageQuery(*args)


def test_mxfp4_weight_layouts_and_weight_validation():
    layouts = mxfp4_weight_layouts(count=4, hidden=256, intermediate=128)
    assert layouts == (
        ((4, 256, 128), torch.int8, (256 * 128, 128, 1)),
        ((4, 256, 2), torch.int32, (256 * 2, 1, 256)),
        ((4, 256, 64), torch.int8, (256 * 64, 64, 1)),
        ((4, 256, 1), torch.int32, (256 * 1, 1, 256)),
    )
    for args in ((0, 128, 128), (2, 192, 128), (2, 128, 64), (True, 128, 128)):
        with pytest.raises(ValueError, match="positive rows and 128-aligned widths"):
            mxfp4_weight_layouts(*args)

    geometry = MegaMoeGeometry(experts=4, capacity=16, top_k=2, hidden=256, intermediate=128, shared_experts=1)
    routed = allocate_mxfp4_weights(4, 256, 128, device="cpu")
    _validate_weights(routed, geometry, shared=False)

    # Row-major scales must be rejected because DeepGEMM expects MN-major UTCCP strides.
    bad_scales = MegaMoeWeights(
        (routed.gate_up[0], torch.empty_like(routed.gate_up[1], memory_format=torch.contiguous_format)),
        routed.down,
    )
    with pytest.raises(ValueError, match="MN-major UTCCP scales"):
        _validate_weights(bad_scales, geometry, shared=False)

    shared = MegaMoeWeights(
        (torch.empty((256, 256), dtype=torch.float8_e4m3fn),
         torch.empty_strided((256, 2), (1, 256), dtype=torch.int32)),
        (torch.empty((256, 128), dtype=torch.float8_e4m3fn),
         torch.empty_strided((256, 1), (1, 256), dtype=torch.int32)),
    )
    _validate_weights(shared, geometry, shared=True)

    source = torch.zeros((8, 256), dtype=torch.bfloat16)
    ids = torch.zeros((8, 2), dtype=torch.int32)
    probabilities = torch.full((8, 2), 0.5, dtype=torch.float32)
    padding = torch.zeros(8, dtype=torch.bool)
    _validate_inputs(source, ids, probabilities, padding, geometry, torch.device("cpu"), 8)
    with pytest.raises(ValueError, match="prepared contiguous"):
        _validate_inputs(source.float(), ids, probabilities, padding, geometry, torch.device("cpu"), 8)
    with pytest.raises(ValueError, match="prepared contiguous"):
        _validate_inputs(source, ids.long(), probabilities, padding, geometry, torch.device("cpu"), 8)


def test_host_tier_copy_weights_and_layer_layout_validation():
    tier = object.__new__(MegaMoeHostTier)
    tier.rows = 4
    layouts = mxfp4_weight_layouts(4, 256, 128)
    tier.host_views = [torch.zeros(shape, dtype=dtype).as_strided(shape, stride)
                       for shape, dtype, stride in layouts]

    batch = allocate_mxfp4_weights(2, 256, 128, device="cpu")
    for idx, tensor in enumerate((*batch.gate_up, *batch.down), start=1):
        tensor.fill_(idx)
    tier.copy_weights(batch, start=1)
    for idx, dst in enumerate(tier.host_views, start=1):
        assert torch.all(dst[0] == 0)
        assert torch.all(dst[1:3] == idx)
        assert torch.all(dst[3] == 0)

    with pytest.raises(ValueError, match="exceeds the allocated expert rows"):
        tier.copy_weights(batch, start=3)
    with pytest.raises(ValueError, match="exceeds the allocated expert rows"):
        tier.copy_weights(batch, start=-1)
    with pytest.raises(ValueError, match="exceeds the allocated expert rows"):
        tier.copy_weights(batch, start=True)

    bad_stride = MegaMoeWeights(
        (batch.gate_up[0], torch.empty_like(batch.gate_up[1], memory_format=torch.contiguous_format)),
        batch.down,
    )
    with pytest.raises(ValueError, match="prepared physical layout"):
        tier.copy_weights(bad_stride, start=0)

    workspace = SimpleNamespace(
        geometry=MegaMoeGeometry(experts=4, capacity=16, top_k=2, hidden=256, intermediate=128, shared_experts=0),
        buffer=torch.zeros(1, dtype=torch.int8),
    )
    loader = SimpleNamespace(device=torch.device("cpu"))
    prefixes = ("experts.0", "experts.1")
    for host, hbm, shared_prefix, prefix_override in (
        ((0, 0), (1, 0), None, prefixes),
        ((0, 1), (1, 2), None, prefixes),
        ((0, 1), (), None, prefixes),
        ((0, 1), (1, 0), "shared", prefixes),
        ((0, 1), (1, 0), None, ("experts.0", "experts.0")),
    ):
        layout = MegaMoeLayerLayout(workspace, host, hbm, storage=None)
        with pytest.raises(ValueError, match="cover canonical experts"):
            layout.load(loader, prefix_override, shared_prefix=shared_prefix, block_size=(32, 32), name="invalid")


def test_prepare_megamoe_inputs_cpu_validation():
    x = torch.zeros((4, 128), dtype=torch.bfloat16)
    ids = torch.zeros((4, 2), dtype=torch.int32)
    weights = torch.ones((4, 2), dtype=torch.float32)
    x_fp8 = torch.empty((4, 128), dtype=torch.float8_e4m3fn)
    x_sf = torch.empty((4, 1), dtype=torch.int32)
    ids_out = torch.empty((4, 2), dtype=torch.int64)
    weights_out = torch.empty((4, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="BF16 activation matrix"):
        prepare_megamoe_inputs(x.float(), weights, ids, x_fp8, x_sf, ids_out, weights_out)
    with pytest.raises(ValueError, match="multiple of 128"):
        prepare_megamoe_inputs(torch.zeros((4, 64), dtype=torch.bfloat16), weights, ids, x_fp8, x_sf, ids_out, weights_out)
    with pytest.raises(ValueError, match="one nonempty top-k row"):
        prepare_megamoe_inputs(x, weights, ids[:3], x_fp8, x_sf, ids_out, weights_out)
    with pytest.raises(ValueError, match="same shape"):
        prepare_megamoe_inputs(x, weights[:, :1], ids, x_fp8, x_sf, ids_out, weights_out)
    with pytest.raises(ValueError, match="int32/int64 IDs and FP32 weights"):
        prepare_megamoe_inputs(x, weights.bfloat16(), ids, x_fp8, x_sf, ids_out, weights_out)
    with pytest.raises(ValueError, match="x_sf requires"):
        prepare_megamoe_inputs(x, weights, ids, x_fp8, x_sf.float(), ids_out, weights_out)
    with pytest.raises(ValueError, match="boolean vector"):
        prepare_megamoe_inputs(x, weights, ids, x_fp8, x_sf, ids_out, weights_out,
                               is_padding=torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="packed int32 storage"):
        prepare_megamoe_inputs(x, weights, ids, x_fp8, x_sf, ids_out, weights_out,
                               shared_x_sf=torch.empty((128, 1), dtype=torch.float32), shared_block_m=64)
    with pytest.raises(ValueError, match="share one CUDA device"):
        prepare_megamoe_inputs(x, weights, ids, x_fp8, x_sf, ids_out, weights_out)


def test_delivered_fp8_projection_rejects_invalid_weights():
    ready = SimpleNamespace(cuda_event=0)
    timings = SimpleNamespace(timestamps=torch.zeros((1, 3), dtype=torch.int64))
    weight = torch.zeros((64, 128), dtype=torch.float8_e4m3fn)
    scales = torch.zeros((64, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="reference-packed FP8 weights"):
        DeliveredFP8Projection((weight, scales), ready=ready, timings=timings, interval=0)
