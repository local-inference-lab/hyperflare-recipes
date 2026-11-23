# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared SM103 fused decode MLA with explicit inter-launch synchronization state."""

from dataclasses import dataclass
import math
from pathlib import Path
import struct

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer

from .mega_attention_kernel import prepare
from .native_operator import native_contract, native_plan

MEGA_ATTENTION_MAX_ROWS = 8


class MegaAttentionWorkspace:
    """Epoch-tagged state shared by serialized attention calls on one device.

    The parallel kernel uses tags to distinguish launches without clearing its
    partial results. Preserve the tags and epoch together across graph variants;
    this storage cannot alias a different operator's temporary workspace.
    """

    def __init__(self, capacity: int, *, device, compute_sm_count: int = 0):
        if type(capacity) is not int or not 1 <= capacity <= 76:
            raise ValueError("fused decode attention requires a capacity from 1 through 76")
        if type(compute_sm_count) is not int or compute_sm_count < 0:
            raise ValueError("fused decode attention compute_sm_count must be a non-negative integer")
        device = torch.device(device)
        with torch.cuda.device(device):
            if torch.cuda.get_device_capability() != (10, 3):
                raise ValueError("fused decode attention requires SM103")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("attention state must be allocated before capture")
            # Only the parallel mode (up to 16 rows) uses tagged partials.
            words = min(capacity, 16) * 10 * (128 + 128 + 2048 + 1)
            self.tensors = [
                persistent_zeros((2 * words,), device=device, dtype=torch.float32),
                persistent_zeros((1,), device=device, dtype=torch.float32),
                persistent_zeros((1,), device=device, dtype=torch.int32),
                persistent_zeros((2,), device=device, dtype=torch.int32),
            ]
        self.capacity = capacity
        self.compute_sm_count = compute_sm_count
        self.device = self.tensors[0].device


def _spec(tensor):
    return None if tensor is None else (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def _validate_operands(query, positions, cos_sin, sink, window, indices, lengths,
                       compressed, compressed_indices, compressed_lengths, workspace,
                       *, token_interleaved: bool = False):
    rows = query.shape[0]
    operands = (query, positions, cos_sin, sink, window, indices, lengths,
                compressed, compressed_indices, compressed_lengths)
    if any(value.device != workspace.device for value in operands if value is not None):
        raise ValueError("attention operands and synchronization state must share one CUDA device")
    if (query.shape != (rows, 64, 512) or query.dtype != torch.bfloat16
            or query.stride(1) != 512 or query.stride(2) != 1
            or query.stride(0) < 64 * 512 or query.stride(0) % 8 != 0
            or not 1 <= rows <= workspace.capacity):
        raise ValueError("attention requires 16-byte-aligned BF16 queries with 64 heads and 512 contiguous dimensions")
    if (positions.shape != (rows,) or positions.dtype != torch.int32 or not positions.is_contiguous()
            or cos_sin.ndim != 2 or cos_sin.shape[0] < 1 or cos_sin.shape[1] != 64
            or cos_sin.dtype != torch.float32 or not cos_sin.is_contiguous()
            or sink.shape != (64,) or sink.dtype != torch.float32 or not sink.is_contiguous()):
        raise ValueError("attention requires int32 positions, contiguous FP32 rotary tables and 64 sinks")
    if (compressed is None) != (compressed_indices is None) or (compressed is None) != (compressed_lengths is None):
        raise ValueError("compressed cache, indices and lengths must be supplied together")
    page_align = 16 if token_interleaved else 512
    for cache, ids, lens, width in ((window, indices, lengths, 528),
                                   (compressed, compressed_indices, compressed_lengths, 288)):
        if cache is None:
            continue
        if (cache.ndim != 3 or min(cache.shape) <= 0 or cache.shape[2] != width
                or cache.dtype != torch.uint8 or cache.stride()[1:] != (width, 1)
                or cache.stride(0) < cache.shape[1] * width or cache.stride(0) % page_align):
            raise ValueError(
                f"attention cache requires {page_align}-byte aligned page stride and {width}-byte token records")
        if (ids.ndim != 2 or ids.shape[0] != rows or ids.shape[1] < 1 or ids.dtype != torch.int32
                or not ids.is_contiguous() or lens.shape != (rows,) or lens.dtype != torch.int32
                or not lens.is_contiguous()):
            raise ValueError("attention indices and lengths must be contiguous int32 with the query row count")
    slots = indices.shape[1] + (compressed_indices.shape[1] if compressed is not None else 0)
    if slots > 640:
        raise ValueError("fused decode attention supports at most ten 64-key blocks")
    if compressed is None:
        if rows > 24:
            raise ValueError("window-only reference attention supports at most 24 rows")
        mode = -2
    else:
        mode = -1 if rows <= 16 else -3
        if mode == -3 and indices.shape[1] > 128:
            raise ValueError("cluster attention requires a window of at most 128 slots")
    return operands, mode


@dataclass(frozen=True)
class MegaAttentionQuery:
    rows: int
    mode: int
    nsub: int
    scale_log2: float
    token_interleaved: bool = False
    max_ctas: int = 0


CONTRACT = native_contract(component_id="flare.mega_attention.decode",
                           query_type=MegaAttentionQuery, tactics=lambda query: (None,),
                           semantic_version=2)


class _State:
    def __init__(self, query, kernel_cache):
        self.query = query
        self.library = prepare(kernel_cache, num_sms=query.max_ctas)

    def run(self, operands, output, scales, state):
        query, positions, cos_sin, sink, window, indices, lengths, compressed, ci, cl = operands
        q = self.query
        self.library.w1_attn_decode(query, positions, cos_sin, sink, q.scale_log2,
            window, indices, lengths, compressed, ci, cl, output, scales, *state,
            None, q.mode, q.nsub, True, q.max_ctas, None, q.token_interleaved)


@torch.library.custom_op("flare::mega_attention", mutates_args=("output", "scales", "state"))
def _run(query: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
         sink: torch.Tensor, window: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor,
         compressed: torch.Tensor | None, compressed_indices: torch.Tensor | None,
         compressed_lengths: torch.Tensor | None, output: torch.Tensor, scales: torch.Tensor,
         state: list[torch.Tensor], handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, query.device).run(
        (query, positions, cos_sin, sink, window, indices, lengths,
         compressed, compressed_indices, compressed_lengths), output, scales, state)


@_run.register_fake
def _run_fake(query, positions, cos_sin, sink, window, indices, lengths, compressed,
              compressed_indices, compressed_lengths, output, scales, state, handle):
    pass


def _outputs(rows, device):
    output = torch.empty((rows, 8, 4096), device=device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((8, 32, (rows + 3) // 4 * 4), device=device, dtype=torch.int32)
    return output, scales.permute(2, 0, 1)[:rows]


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class PreparedMegaAttention:
    """Declare the reference decode modes; cache contents use the FlashMLA page format.

    Query heads must already have the reference's 16-element permutation. The
    caller supplies in-range rotary positions and cache indices, with lengths
    bounded by each index width. This component does not write or repack caches.
    """

    def __init__(self, query, positions, cos_sin, sink, window, indices, lengths,
                 compressed=None, compressed_indices=None, compressed_lengths=None, *,
                 workspace: MegaAttentionWorkspace, scale: float, kernel_cache: Path, session, name,
                 token_interleaved: bool = False, compute_sm_count: int = 0):
        operands, mode = _validate_operands(query, positions, cos_sin, sink, window, indices,
            lengths, compressed, compressed_indices, compressed_lengths, workspace,
            token_interleaved=token_interleaved)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("attention scale must be finite and positive")
        if type(compute_sm_count) is not int or compute_sm_count < 0:
            raise ValueError("attention compute_sm_count must be a non-negative integer")
        rows = query.shape[0]
        device_sms = torch.cuda.get_device_properties(workspace.device).multi_processor_count
        effective_sms = compute_sm_count if compute_sm_count > 0 else workspace.compute_sm_count
        sms = min(effective_sms, device_sms) if effective_sms > 0 else device_sms
        nsub = (1 if 8 * rows <= sms else (2 if 4 * rows <= sms else 4)) if mode == -2 else 0
        q = MegaAttentionQuery(rows, mode, nsub, _f32(_f32(scale) * _f32(math.log2(math.e))),
                               token_interleaved=token_interleaved,
                               max_ctas=sms if effective_sms > 0 else 0)
        self.specs = tuple(_spec(value) for value in operands)
        self.workspace = workspace
        self.plan = native_plan(contract=CONTRACT, query=q, device=workspace.device,
            scratch_bytes=lambda tactic: 0, materialize=lambda tactic: _State(q, kernel_cache))
        examples = list(operands)

        def prepare_call(state):
            output, scales = _outputs(rows, workspace.device)
            return PreparedCall(run=lambda: state.run(examples, output, scales, workspace.tensors),
                                output=output, owners=(workspace,))

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)

    def __call__(self, query, positions, cos_sin, sink, window, indices, lengths,
                 compressed=None, compressed_indices=None, compressed_lengths=None):
        operands = (query, positions, cos_sin, sink, window, indices, lengths,
                    compressed, compressed_indices, compressed_lengths)
        if tuple(_spec(value) for value in operands) != self.specs:
            raise ValueError("attention operands must match their prepared geometry, dtype, strides and device")
        output, scales = _outputs(self.plan.query.rows, query.device)
        _run(*operands, output, scales, self.workspace.tensors, self.plan.handle)
        return output, scales


class MegaAttentionDecoder:
    """Multi-row SM103 fused decode MLA bound to token-interleaved reference KV caches."""

    def __init__(self, rows_list, *, scale: float, workspace: MegaAttentionWorkspace,
                 kernel_cache: Path, session, name: str, compressed: bool = True,
                 swa_width: int = 128, comp_width: int = 512, token_interleaved: bool = True,
                 compute_sm_count: int = 0):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("attention scale must be finite and positive")
        if type(compute_sm_count) is not int or compute_sm_count < 0:
            raise ValueError("attention compute_sm_count must be a non-negative integer")
        self.workspace = workspace
        self.compressed = compressed
        self.swa_width = swa_width
        self.comp_width = comp_width
        self.token_interleaved = token_interleaved
        device = workspace.device
        scale_log2 = _f32(_f32(scale) * _f32(math.log2(math.e)))
        device_sms = torch.cuda.get_device_properties(device).multi_processor_count
        effective_sms = compute_sm_count if compute_sm_count > 0 else workspace.compute_sm_count
        sms = min(effective_sms, device_sms) if effective_sms > 0 else device_sms
        max_ctas = sms if effective_sms > 0 else 0
        self.compute_sm_count = max_ctas
        mode = -1 if compressed else -2
        max_rows = min(workspace.capacity, MEGA_ATTENTION_MAX_ROWS if compressed else 16)
        self._plans = {}
        self._full_swa_lens = {}
        self._full_comp_lens = {}
        requests = []
        for rows in sorted({r for r in rows_list if 1 <= r <= max_rows}):
            nsub = (1 if 8 * rows <= sms else (2 if 4 * rows <= sms else 4)) if mode == -2 else 0
            q = MegaAttentionQuery(rows, mode, nsub, scale_log2, token_interleaved=token_interleaved,
                                   max_ctas=max_ctas)
            op_plan = native_plan(
                contract=CONTRACT,
                query=q,
                device=device,
                scratch_bytes=lambda tactic: 0,
                materialize=lambda tactic, q=q: _State(q, kernel_cache),
            )
            self._plans[rows] = op_plan
            swa_lens = persistent_copy(torch.full((rows,), swa_width, dtype=torch.int32, device=device))
            record_numeric_tensor(swa_lens)
            self._full_swa_lens[rows] = swa_lens
            if compressed:
                comp_lens = persistent_copy(torch.full((rows,), comp_width, dtype=torch.int32, device=device))
                record_numeric_tensor(comp_lens)
                self._full_comp_lens[rows] = comp_lens

            def make_prepare(rows=rows, compressed=compressed):
                def prepare_call(state):
                    query = torch.zeros((rows, 64, 512), dtype=torch.bfloat16, device=device)
                    positions = torch.zeros((rows,), dtype=torch.int32, device=device)
                    cos_sin = torch.zeros((1, 64), dtype=torch.float32, device=device)
                    sink = torch.zeros((64,), dtype=torch.float32, device=device)
                    window = torch.zeros((1, swa_width, 528), dtype=torch.uint8, device=device)
                    indices = torch.full((rows, swa_width), -1, dtype=torch.int32, device=device)
                    lengths = torch.zeros((rows,), dtype=torch.int32, device=device)
                    comp = torch.zeros((1, 64, 288), dtype=torch.uint8, device=device) if compressed else None
                    ci = torch.full((rows, comp_width), -1, dtype=torch.int32, device=device) if compressed else None
                    cl = torch.zeros((rows,), dtype=torch.int32, device=device) if compressed else None
                    output, scales = _outputs(rows, device)
                    operands = (query, positions, cos_sin, sink, window, indices, lengths, comp, ci, cl)
                    return PreparedCall(
                        run=lambda: state.run(operands, output, scales, workspace.tensors),
                        output=output,
                        owners=(workspace,),
                    )
                return prepare_call

            requests.append(op_plan.request(name=f"{name}.m{rows}", prepare_call=make_prepare()))
        if requests:
            declare(session, tuple(requests))

    def supports(self, rows: int) -> bool:
        return rows in self._plans

    def __call__(self, query, positions, cos_sin, sink, window, indices,
                 compressed=None, compressed_indices=None, compressed_lengths=None):
        rows = query.shape[0]
        op_plan = self._plans[rows]
        swa_lens = self._full_swa_lens[rows]
        comp_lens = compressed_lengths if compressed_lengths is not None else self._full_comp_lens.get(rows)
        output, scales = _outputs(rows, query.device)
        _run(query, positions, cos_sin, sink, window, indices, swa_lens,
             compressed, compressed_indices, comp_lens, output, scales,
             self.workspace.tensors, op_plan.handle)
        return output, scales
