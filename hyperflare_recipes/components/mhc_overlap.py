# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 split-lagged mHC overlapped with sublayer PCIe AllReduce (+ fused MXFP8 quant)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyperflare_recipes.components.mhc import MHCWeights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer


@torch.library.custom_op("flare::mhc_cluster_post_y_norm", mutates_args=("out", "y"))
def _mhc_cluster_post_y_norm(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    prev_post: torch.Tensor | None,
    prev_comb: torch.Tensor | None,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    y: torch.Tensor,
    pre_only: bool,
    norm_eps: float,
) -> None:
    from b12x.norm.mhc._split_lagged import compile_cluster_post_y_norm

    runner = compile_cluster_post_y_norm(
        int(residual.device.index or 0),
        int(residual.shape[-1]),
        pre_only,
        norm_eps,
        norm_weight.dtype,
        emit_mxfp8=False,
    )
    runner(
        x,
        residual,
        prev_post,
        prev_comb,
        pre_mix,
        norm_weight,
        out,
        y,
        stream=torch.cuda.current_stream(residual.device),
    )


@_mhc_cluster_post_y_norm.register_fake
def _fake_mhc_cluster_post_y_norm(
    x, residual, prev_post, prev_comb, pre_mix, norm_weight, out, y, pre_only, norm_eps
):
    pass


@torch.library.custom_op(
    "flare::mhc_cluster_post_y_norm_mxfp8",
    mutates_args=("out", "y", "a_values", "a_scales"),
)
def _mhc_cluster_post_y_norm_mxfp8(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    prev_post: torch.Tensor | None,
    prev_comb: torch.Tensor | None,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    y: torch.Tensor,
    a_values: torch.Tensor,
    a_scales: torch.Tensor,
    pre_only: bool,
    norm_eps: float,
) -> None:
    from b12x.norm.mhc._split_lagged import compile_cluster_post_y_norm

    runner = compile_cluster_post_y_norm(
        int(residual.device.index or 0),
        int(residual.shape[-1]),
        pre_only,
        norm_eps,
        norm_weight.dtype,
        emit_mxfp8=True,
    )
    runner(
        x,
        residual,
        prev_post,
        prev_comb,
        pre_mix,
        norm_weight,
        out,
        y,
        a_values=a_values,
        a_scales=a_scales,
        stream=torch.cuda.current_stream(residual.device),
    )


@_mhc_cluster_post_y_norm_mxfp8.register_fake
def _fake_mhc_cluster_post_y_norm_mxfp8(
    x,
    residual,
    prev_post,
    prev_comb,
    pre_mix,
    norm_weight,
    out,
    y,
    a_values,
    a_scales,
    pre_only,
    norm_eps,
):
    pass


@torch.library.custom_op(
    "flare::start_mhc_fn_sinkhorn",
    mutates_args=("partials", "pre_out", "post_out", "comb_out"),
)
def _start_mhc_fn_sinkhorn(
    residual_out: torch.Tensor,
    trigger: torch.Tensor,
    trigger2: torch.Tensor | None,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    partials: torch.Tensor,
    pre_out: torch.Tensor,
    post_out: torch.Tensor,
    comb_out: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    stream: int,
) -> None:
    from b12x.norm.mhc._split_lagged import compile_readonly_fn_sinkhorn

    main = torch.cuda.current_stream(residual_out.device)
    auxiliary = torch.cuda.ExternalStream(stream, device=residual_out.device)
    auxiliary.wait_stream(main)
    runner = compile_readonly_fn_sinkhorn(
        int(residual_out.device.index or 0),
        int(residual_out.shape[-1]),
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )
    with torch.cuda.stream(auxiliary):
        runner(
            residual_out,
            fn,
            scale,
            bias,
            partials,
            pre_out,
            post_out,
            comb_out,
            stream=auxiliary,
        )


@_start_mhc_fn_sinkhorn.register_fake
def _fake_start_mhc_fn_sinkhorn(
    residual_out,
    trigger,
    trigger2,
    fn,
    scale,
    bias,
    partials,
    pre_out,
    post_out,
    comb_out,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    stream,
):
    pass


@torch.library.custom_op(
    "flare::join_mhc_fn_sinkhorn",
    mutates_args=("partials", "pre_out", "post_out", "comb_out"),
)
def _join_mhc_fn_sinkhorn(
    residual_out: torch.Tensor,
    dependency: torch.Tensor,
    partials: torch.Tensor,
    pre_out: torch.Tensor,
    post_out: torch.Tensor,
    comb_out: torch.Tensor,
    stream: int,
) -> None:
    torch.cuda.current_stream(residual_out.device).wait_stream(
        torch.cuda.ExternalStream(stream, device=residual_out.device)
    )


@_join_mhc_fn_sinkhorn.register_fake
def _fake_join_mhc_fn_sinkhorn(
    residual_out, dependency, partials, pre_out, post_out, comb_out, stream
):
    pass


_SPLIT_MHC_RUNNERS: dict[tuple, object] = {}


@dataclass(frozen=True)
class PendingMHCSinkhorn:
    residual_out: torch.Tensor
    partials: torch.Tensor
    pre_out: torch.Tensor
    post_out: torch.Tensor
    comb_out: torch.Tensor


class SplitLaggedMHC:
    """Split lagged mHC for SM120 decode, verify, and draft (M <= 8).

    Stage 1 (``post_y_norm``) runs ``ClusterMhcPostYNorm`` on the main stream to
    produce ``(residual_out, normalized)`` and optionally pre-quantized MXFP8
    ``(a_values, a_scales)`` for compact W4A8 micro routed experts.

    Stage 2 (``start_sinkhorn`` / ``finish_sinkhorn``) runs ``ReadOnlyFnPartialM8``
    and ``_SplitFinalize`` on ``workspace.auxiliary_stream`` concurrently with the
    sublayer's tensor-parallel AllReduce.
    """

    def __init__(
        self,
        inputs,
        weights: MHCWeights,
        *,
        session,
        workspace: ScratchArena,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_eps: float,
        experts=None,
    ):
        from b12x.norm.mhc._split_lagged import (
            split_lagged_partials_shape,
            supports_split_lagged,
        )

        self.pre_only = len(inputs) == 1
        if len(inputs) not in (1, 4):
            raise ValueError("SplitLaggedMHC requires 1 (pre) or 4 (post_pre) inputs")
        residual = inputs[0 if self.pre_only else 1]
        if (
            residual.ndim != 3
            or residual.shape[1] != 4
            or residual.dtype != torch.bfloat16
        ):
            raise ValueError(
                "SplitLaggedMHC residual requires BF16 [tokens, 4, hidden]"
            )
        if not isinstance(workspace, ScratchArena) or not workspace.overlap:
            raise ValueError("SplitLaggedMHC requires an overlapping ScratchArena")
        tokens, _, hidden = residual.shape
        if not supports_split_lagged(
            residual.device, tokens, hidden, weights.norm.dtype
        ):
            raise ValueError("SplitLaggedMHC geometry or device is not supported")
        if (
            weights.fn.shape != (24, 4 * hidden)
            or weights.fn.dtype != torch.float32
            or weights.scale.shape != (3,)
            or weights.scale.dtype != torch.float32
            or weights.base.shape != (24,)
            or weights.base.dtype != torch.float32
            or weights.norm.shape != (hidden,)
            or weights.norm.dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError("SplitLaggedMHC weights must match mHC contract")
        for tensor in (weights.fn, weights.scale, weights.base, weights.norm):
            record_numeric_tensor(tensor)

        self.weights = weights
        self.device = residual.device
        self.tokens = int(tokens)
        self.hidden = int(hidden)
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.norm_eps = float(norm_eps)
        self.partials_shape = split_lagged_partials_shape(self.tokens, self.hidden)
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        self.experts = experts
        self.emit_mxfp8 = False
        register_finalizer(session, self._finalize)

    def functional(self):
        return self

    def _finalize(self):
        from b12x._lib.compile_plan import load_programs
        from b12x.moe.fused_moe._binding_ops import supports_prequant_mxfp8
        from b12x.norm.mhc._split_lagged import (
            compile_cluster_post_y_norm,
            compile_readonly_fn_sinkhorn,
        )

        if self.experts is not None and not getattr(
            self.experts, "integrated_routing", False
        ):
            self.emit_mxfp8 = supports_prequant_mxfp8(self.experts.plan, self.tokens)
        self.experts = None
        ordinal = int(self.device.index or 0)
        post_y = compile_cluster_post_y_norm(
            ordinal,
            self.hidden,
            self.pre_only,
            self.norm_eps,
            self.weights.norm.dtype,
            emit_mxfp8=self.emit_mxfp8,
        )
        load_programs(post_y)
        _SPLIT_MHC_RUNNERS[
            (
                "post_y",
                ordinal,
                self.hidden,
                self.pre_only,
                self.norm_eps,
                self.weights.norm.dtype,
                self.emit_mxfp8,
            )
        ] = post_y
        fn_sink = compile_readonly_fn_sinkhorn(
            ordinal,
            self.hidden,
            self.rms_eps,
            self.hc_eps,
            self.sinkhorn_iters,
        )
        load_programs(fn_sink)
        _SPLIT_MHC_RUNNERS[
            (
                "fn_sink",
                ordinal,
                self.hidden,
                self.rms_eps,
                self.hc_eps,
                self.sinkhorn_iters,
            )
        ] = fn_sink

    def post_y_norm(self, *inputs, pre_mix: torch.Tensor):
        if self.pre_only:
            (residual,) = inputs
            x = prev_post = prev_comb = None
        else:
            x, residual, prev_post, prev_comb = inputs
        tokens, _, hidden = residual.shape
        device = residual.device
        out = torch.empty((tokens, 4, hidden), dtype=residual.dtype, device=device)
        y = torch.empty((tokens, hidden), dtype=residual.dtype, device=device)
        if self.emit_mxfp8:
            a_values = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
            a_scales = torch.empty(
                (tokens, hidden // 32), dtype=torch.uint8, device=device
            )
            _mhc_cluster_post_y_norm_mxfp8(
                x,
                residual,
                prev_post,
                prev_comb,
                pre_mix,
                self.weights.norm,
                out,
                y,
                a_values,
                a_scales,
                self.pre_only,
                self.norm_eps,
            )
            return out, y, a_values, a_scales
        _mhc_cluster_post_y_norm(
            x,
            residual,
            prev_post,
            prev_comb,
            pre_mix,
            self.weights.norm,
            out,
            y,
            self.pre_only,
            self.norm_eps,
        )
        return out, y, None, None

    def start_sinkhorn(
        self,
        residual_out: torch.Tensor,
        trigger: torch.Tensor,
        trigger2: torch.Tensor | None = None,
    ) -> PendingMHCSinkhorn:
        tokens = residual_out.shape[0]
        device = residual_out.device
        partials = torch.empty(self.partials_shape, dtype=torch.float32, device=device)
        pre_out = torch.empty((tokens, 4), dtype=torch.float32, device=device)
        post_out = torch.empty((tokens, 4), dtype=torch.float32, device=device)
        comb_out = torch.empty((tokens, 4, 4), dtype=torch.float32, device=device)
        _start_mhc_fn_sinkhorn(
            residual_out,
            trigger,
            trigger2,
            self.weights.fn,
            self.weights.scale,
            self.weights.base,
            partials,
            pre_out,
            post_out,
            comb_out,
            self.rms_eps,
            self.hc_eps,
            self.sinkhorn_iters,
            self.stream_handle,
        )
        return PendingMHCSinkhorn(residual_out, partials, pre_out, post_out, comb_out)

    def finish_sinkhorn(
        self, pending: PendingMHCSinkhorn, dependency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _join_mhc_fn_sinkhorn(
            pending.residual_out,
            dependency,
            pending.partials,
            pending.pre_out,
            pending.post_out,
            pending.comb_out,
            self.stream_handle,
        )
        return pending.post_out, pending.comb_out, pending.pre_out
