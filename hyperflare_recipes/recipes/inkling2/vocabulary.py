# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary-parallel Inkling 2 embedding rows and softcapped logits of one rank.

Each rank holds `width` consecutive rows of the padded 201024-row embedding and
unembedding tables, starting at `rank * width`; rows past the table are zero.
The embedding contributes a token's row on the rank that holds it and zero
elsewhere, so the cross-rank sum is the row exactly. Logits project the final
normalized hidden state in FP32, apply `softcap * tanh(x / softcap)` and are
`-inf` at IDs from the valid vocabulary size on.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.linear import prepare_projection
from ignition.preparation.triton import launch_triton, prepare_triton


@triton.jit
def _softcap_mask_inplace_kernel(
    Buf,
    SOFTCAP: tl.constexpr,
    COLS: tl.constexpr,
    VALID_START: tl.constexpr,
    VALID_END: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """In-place FP32 softcap `c * tanh(x / c)` + valid-vocabulary `-inf` mask."""
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1).to(tl.int64)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    in_bounds = cols < COLS
    offset = row * COLS + cols
    x = tl.load(Buf + offset, mask=in_bounds, other=0.0).to(tl.float32)
    capped = SOFTCAP * tl.extra.cuda.libdevice.tanh(x / SOFTCAP)
    valid = (VALID_START + cols) < VALID_END
    out = tl.where(valid, capped, float("-inf"))
    tl.store(Buf + offset, out, mask=in_bounds)


@triton.jit
def _softcap_mask_kernel(
    Src, Dst,
    SOFTCAP: tl.constexpr,
    COLS: tl.constexpr,
    VALID_START: tl.constexpr,
    VALID_END: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused FP32 softcap `c * tanh(x / c)` + valid-vocabulary `-inf` mask (supports BF16 or FP32 input)."""
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1).to(tl.int64)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    in_bounds = cols < COLS
    offset = row * COLS + cols
    x = tl.load(Src + offset, mask=in_bounds, other=0.0).to(tl.float32)
    capped = SOFTCAP * tl.extra.cuda.libdevice.tanh(x / SOFTCAP)
    valid = (VALID_START + cols) < VALID_END
    out = tl.where(valid, capped, float("-inf"))
    tl.store(Dst + offset, out, mask=in_bounds)


def head_rows(weight) -> int:
    return weight.shape[0] if isinstance(weight, torch.Tensor) else weight.out_features


def embedding_rows(table, ids, first):
    """This rank's embedding rows of `ids`, zero for IDs another rank holds and for padded rows (`ids < 0`).

    `first` is the token ID of the rank's first table row, `rank * width`; compiled
    forwards pass it as a device tensor so that every rank binds the same code.
    """
    width = table.shape[0]
    local = ids - first
    inside = (local >= 0) & (local < width)
    return torch.where(inside[:, None], table[local.clamp(0, width - 1)], 0)


class PreparedLogits:
    """FP32 logits of this rank's vocabulary columns with fused softcap and valid-vocabulary mask."""

    def __init__(self, weight, text, rank, *, tokens, session, name, workspace=None, examples=None):
        from ignition.preparation.examples import declaration_tensor

        examples = {} if examples is None else examples
        is_tensor = isinstance(weight, torch.Tensor)
        cols = head_rows(weight)
        device = weight.device if is_tensor else weight.weight.values.device
        source = declaration_tensor(examples, "inkling.logits.source", (tokens, text.hidden), torch.bfloat16, device)
        proj_dtype = torch.float32 if is_tensor else torch.bfloat16
        proj_role = "inkling.logits.local" if is_tensor else "inkling.logits.local_bf16"
        proj_out = declaration_tensor(examples, proj_role, (tokens, cols), proj_dtype, device)
        activation_mode = "a16" if (not is_tensor and tokens <= 64) else "auto"
        self.projection = prepare_projection(source, weight, proj_out, session=session,
                                             name=f"{name}.projection", workspace=workspace,
                                             activation_mode=activation_mode).functional(session=session)
        self.in_place = is_tensor
        self.cols = cols
        self.softcap = float(text.softcap)
        self._block_n = 1024
        self._grid = (tokens, triton.cdiv(cols, self._block_n), 1)
        self._opts = dict(
            SOFTCAP=self.softcap,
            COLS=cols,
            VALID_START=rank * cols,
            VALID_END=text.valid_vocabulary,
            BLOCK_N=self._block_n,
            num_warps=4,
        )
        if is_tensor:
            self._compiled = prepare_triton(_softcap_mask_inplace_kernel, (proj_out,), self._grid, **self._opts).compiled
        else:
            dst_ex = declaration_tensor(examples, "inkling.logits.local", (tokens, cols), torch.float32, device)
            self._compiled = prepare_triton(_softcap_mask_kernel, (proj_out, dst_ex), self._grid, **self._opts).compiled

    def __call__(self, hidden):
        logits = self.projection(hidden)
        grid = (logits.shape[0], triton.cdiv(self.cols, self._block_n), 1)
        if self.in_place:
            launch_triton(_softcap_mask_inplace_kernel, self._compiled, (logits,), grid, **self._opts)
            return logits
        out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
        launch_triton(_softcap_mask_kernel, self._compiled, (logits, out), grid, **self._opts)
        return out
