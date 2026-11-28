# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check compiler metadata against native sparse-indexer allocations."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("rows,blocks,block_size", [(1, 4, 8), (8, 128, 16), (129, 4096, 8)])
def test_sparse_schedule_tensor_contract(paged, rows, blocks, block_size):
    pytest.importorskip("deepinfer")
    import deep_gemm
    from deepinfer.model_executor.layers import index_ops

    candidates = torch.full((rows, blocks), -1, dtype=torch.int32, device="cuda")
    candidates[:, 0] = 0
    lengths = torch.full((rows,), 7, dtype=torch.int32, device="cuda")
    sms = deep_gemm.get_num_sms()
    if paged:
        tables = torch.ones((rows, 1), dtype=torch.int32, device="cuda")
        indices = torch.zeros(rows, dtype=torch.int32, device="cuda")
        op = index_ops.get_paged_sparse_mqa_logits_metadata
        args = (lengths, tables, indices, 64, candidates, torch.int8, block_size, sms)
    else:
        starts = torch.zeros_like(lengths)
        op = index_ops.get_sparse_mqa_logits_metadata
        args = (starts, lengths, 64, candidates, torch.int8, block_size, sms, True)
    torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))
    torch.cuda.synchronize()


def test_mqa_rejects_unaligned_weight_pointer():
    pytest.importorskip("deepinfer")
    from deepinfer.model_executor.layers import index_ops

    q = torch.zeros((1, 32, 64), dtype=torch.int8, device="cuda")
    qs = torch.zeros((1, 32), dtype=torch.int32, device="cuda")
    k = torch.zeros((64, 64), dtype=torch.int8, device="cuda")
    ks = torch.zeros(64, dtype=torch.int32, device="cuda")
    lengths = torch.ones(1, dtype=torch.int32, device="cuda")
    weights = torch.zeros(33, dtype=torch.bfloat16, device="cuda")[1:].view(1, 32)
    assert weights.stride(0) * weights.element_size() % 16 == 0
    with pytest.raises(AssertionError, match="unaligned MQA weights"):
        index_ops.fp8_fp4_mqa_logits((q, qs), (k, ks), weights,
                                    lengths - 1, lengths, max_seqlen_k=64)
