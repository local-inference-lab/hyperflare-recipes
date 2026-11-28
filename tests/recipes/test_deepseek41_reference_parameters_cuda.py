# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete text, draft and vision weight loading without expert/table allocation."""

import os
from bisect import bisect_right
from pathlib import Path

import pytest
import torch

from ignition.preparation.tensor_metadata import NumericStorage, tensor_ranges
from hyperflare_recipes.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint
from hyperflare_recipes.recipes.deepseek41.reference_parameters import ReferenceParameters

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@torch.no_grad()
def test_complete_ordinary_reference_parameters():
    pytest.importorskip("deepinfer")
    from deepinfer.hugging_face import model_params
    from deepinfer.hugging_face.names import scale_name
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    path = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not path:
        pytest.skip("requires the DeepSeek V4.1 CSF checkpoint")
    if PG.world is None:
        PG.initialize_local()
    checkpoint = ReferenceCheckpoint(Path(path))
    prepared = ReferenceParameters(checkpoint)
    assert all(value.is_meta for value in model_params(prepared.model).values())
    with NumericStorage() as numeric:
        result = prepared.load_ordinary("cuda:0")
    numeric_ranges = numeric.ranges()
    numeric_starts = [start for start, _ in numeric_ranges]
    assert prepared.loaded
    assert result["expert_params"] == result["experts"] == result["engram_tables"] == 0
    params = model_params(prepared.model)
    for name, sources in prepared.coverage.tensors.items():
        expected = checkpoint.read_concat(sources)
        actual = params[name].cpu()
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name
        if params[name].dtype == torch.float8_e4m3fn:
            scale = checkpoint.read_concat(tuple(scale_name(source) for source in sources)).float()
            assert torch.equal(params[name].sf.cpu().view(torch.int32), scale.view(torch.int32)), name
            assert len(params[name].ue8m0_sf) == 2
            assert all(value.is_cuda for value in params[name].ue8m0_sf)
            for packed in params[name].ue8m0_sf:
                for start, end in tensor_ranges(packed):
                    index = bisect_right(numeric_starts, start) - 1
                    assert index >= 0 and end <= numeric_ranges[index][1], name
    for name, source in prepared.coverage.aliases.items():
        assert params[name].data_ptr() == params[source].data_ptr()
    assert len(prepared.coverage.expert_tensors) == 86
    assert len(prepared.coverage.engram_tables) == 2
    for name in set(prepared.coverage.expert_tensors) | set(prepared.coverage.engram_tables):
        assert params[name].is_meta
    vision = prepared.model.vision_pipeline.aligner
    assert all(value.is_cuda for value in vision.parameters())
    assert vision.visual_encoder.rope_table.is_cuda
    assert vision.visual_encoder.rotary_pos_emb.inv_freq.is_cuda
    with pytest.raises(RuntimeError, match="already been loaded"):
        prepared.load_ordinary("cuda:0")
    assert not torch.distributed.is_initialized()
