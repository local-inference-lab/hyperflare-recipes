# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepGEMM projection of native-delivered FP8 rows after their readiness event."""

from typing import NamedTuple

import torch
from torch import nn

from ignition.components.readiness import wait_for_readiness
from ignition.preparation.tensor_metadata import record_numeric_tensor


class DeliveredFP8Rows(NamedTuple):
    values: torch.Tensor
    scale_bytes: torch.Tensor
    valid: torch.Tensor
    hashes: torch.Tensor
    mailbox: torch.Tensor


@torch.library.custom_op("flare::delivered_fp8_projection", mutates_args=("output", "timestamps"))
def _project(values: torch.Tensor, scale_bytes: torch.Tensor, valid: torch.Tensor,
             hashes: torch.Tensor, mailbox: torch.Tensor, weight: torch.Tensor,
             weight_scales: torch.Tensor, output: torch.Tensor, timestamps: torch.Tensor,
             frame: torch.Tensor, interval: int, event: int) -> None:
    import deep_gemm

    # Hashes and the published mailbox are explicit reads: native delivery must
    # observe those requests before the captured consumer waits for their data.
    wait_for_readiness(values, event, timestamps, frame, interval, frame, frame, 0)
    rows = values.shape[0]
    packed = scale_bytes.view(torch.int32)
    storage = torch.empty((packed.shape[1], (rows + 3) // 4 * 4), dtype=torch.int32, device=values.device)
    scales = storage[:, :rows].mT
    scales.copy_(packed)
    deep_gemm.fp8_fp4_gemm_nt((values, scales), (weight, weight_scales), output, recipe=(1, 1, 32))
    # Native delivery may upload fewer rows than the graph's physical capacity.
    # Those trailing bytes can belong to an earlier invocation.
    output.masked_fill_(~valid[:, None], 0)


@_project.register_fake
def _project_fake(values, scale_bytes, valid, hashes, mailbox, weight, weight_scales,
                  output, timestamps, frame, interval, event):
    pass


class DeliveredFP8Projection(nn.Module):
    """Consume FP8/E8M0 row planes without a BF16 gather or activation recast.

    The native table owner retains row buffers and the external readiness event
    until graph drain. Scale packing is lossless and uses graph-local storage.
    ``weight`` is the reference projection's packed FP8 weight/scale pair.
    This component targets single-device preparation; it performs no collective.
    """

    def __init__(self, weight, *, ready, timings, interval):
        super().__init__()
        values, scales = weight
        if (values.ndim != 2 or values.dtype != torch.float8_e4m3fn or not values.is_cuda
                or not values.is_contiguous() or values.shape[1] % 128
                or scales.dtype != torch.int32 or scales.device != values.device
                or scales.shape != (values.shape[0], values.shape[1] // 128)):
            raise ValueError("delivered projection requires reference-packed FP8 weights and int32 scales")
        if type(interval) is not int or not 0 <= interval < timings.timestamps.shape[1] // 3:
            raise ValueError("delivered projection interval is outside readiness storage")
        self.register_buffer("weight", values)
        self.register_buffer("weight_scales", scales)
        self.ready, self.timings, self.interval = ready, timings, interval
        self.event = ready.cuda_event
        record_numeric_tensor(values)
        record_numeric_tensor(scales)

    def forward(self, inputs: DeliveredFP8Rows):
        values, scales, valid, hashes, mailbox = inputs
        rows, width = values.shape
        if (values.dtype != torch.float8_e4m3fn or width != self.weight.shape[1]
                or not values.is_contiguous() or scales.shape != (rows, width // 32)
                or scales.dtype != torch.uint8 or not scales.is_contiguous()
                or valid.shape != (rows,) or valid.dtype != torch.bool
                or hashes.ndim != 2 or hashes.shape[0] != rows or hashes.dtype != torch.int64
                or mailbox.ndim != 1 or mailbox.numel() < hashes.numel() + 1
                or mailbox.dtype != torch.int64
                or any(t.device != self.weight.device for t in (values, scales, valid, hashes, mailbox))):
            raise ValueError("delivered projection inputs must match FP8 rows, scales and request geometry")
        output = torch.empty((rows, self.weight.shape[0]), dtype=torch.bfloat16, device=values.device)
        _project(values, scales, valid, hashes, mailbox, self.weight, self.weight_scales, output,
                 self.timings.timestamps, self.timings.frame, self.interval, self.event)
        return output
