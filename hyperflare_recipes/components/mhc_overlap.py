# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 split-lagged mHC overlapped with sublayer PCIe AllReduce (+ fused MXFP8 quant)."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import itertools
import weakref

import torch

from hyperflare_recipes.components.mhc import MHCWeights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer


_FUSED_MHC_AR_OWNERS: weakref.WeakValueDictionary[int, "PreparedFusedMhcAllReducePostYNorm"] = (
    weakref.WeakValueDictionary()
)
_FUSED_MHC_AR_HANDLES = itertools.count()


def _is_roce_or_dma_runtime(runtime) -> bool:
    if runtime is None:
        return True
    try:
        from b12x.comm import roce

        if isinstance(runtime, roce.AllReduce):
            return True
    except ImportError:
        pass
    try:
        from b12x.comm import pcie

        if isinstance(runtime, pcie.DmaAllReduce):
            return True
    except ImportError:
        pass
    return False


def _resolve_oneshot_channel(runtime):
    """Return the single-channel PCIe oneshot channel backing ``runtime``, or ``None``."""
    if _is_roce_or_dma_runtime(runtime):
        return None
    algorithm = getattr(runtime, "algorithm", None)
    if algorithm is not None:
        if algorithm != "oneshot" or not getattr(runtime, "single_channel", False):
            return None
        for_stream = getattr(runtime, "for_stream", None)
        if for_stream is None:
            return None
        try:
            return for_stream()
        except Exception:
            return None
    if hasattr(runtime, "_ext") and hasattr(runtime, "_ptr"):
        return runtime
    return None


def _channel_sentinel_capacity_bytes(channel) -> int | None:
    """Return the Lamport-sentinel payload capacity in bytes, or ``None`` if unavailable."""
    ext = getattr(channel, "_ext", None)
    ptr = getattr(channel, "_ptr", None)
    if ext is None or ptr is None:
        return None
    get_state = getattr(ext, "_state", None)
    if get_state is None:
        return None
    try:
        state = get_state(ptr)
    except Exception:
        return None
    if (
        getattr(state, "eager_tables", None) is None
        or getattr(state, "eager_buffer_bytes", None) is None
        or int(getattr(state, "plain_payload_ready_region_packs", 0)) <= 0
    ):
        return None
    cap = int(state.eager_buffer_bytes)
    channel_max = getattr(channel, "max_size", None)
    if channel_max is not None:
        cap = min(cap, int(channel_max))
    return cap if cap > 0 else None


def supports_fused_mhc_allreduce(
    runtime,
    tokens: int,
    hidden: int,
    norm_dtype: torch.dtype,
    *,
    device: torch.device | None = None,
) -> bool:
    """Return True when b12x fused Lamport-sentinel AllReduce + ClusterMhcPostYNorm fits."""
    try:
        from b12x.norm.mhc import _split_lagged
    except ImportError:
        return False

    supports_fn = getattr(_split_lagged, "supports_cluster_allreduce_post_y_norm", None)
    compile_fn = getattr(_split_lagged, "compile_cluster_allreduce_post_y_norm", None)
    params_fn = getattr(_split_lagged, "oneshot_channel_sentinel_params", None)
    if supports_fn is None or compile_fn is None or params_fn is None:
        return False

    channel = _resolve_oneshot_channel(runtime)
    if channel is None:
        return False

    tokens = int(tokens)
    hidden = int(hidden)
    max_tokens = int(getattr(_split_lagged, "CLUSTER_AR_MHC_MAX_TOKENS", 32))
    if not (1 <= tokens <= max_tokens) or hidden <= 0 or hidden % 8 != 0:
        return False

    world_size = getattr(runtime, "world_size", getattr(channel, "world_size", 0))
    if not isinstance(world_size, int) or world_size < 2:
        return False

    payload_bytes = tokens * hidden * 2
    runtime_max = getattr(runtime, "max_size", None)
    if runtime_max is not None and payload_bytes > int(runtime_max):
        return False

    sentinel_cap = _channel_sentinel_capacity_bytes(channel)
    if sentinel_cap is None or payload_bytes > sentinel_cap:
        return False

    target_device = (
        device
        if device is not None
        else getattr(channel, "device", getattr(runtime, "device", None))
    )
    if target_device is None:
        return False

    return bool(supports_fn(target_device, tokens, hidden, norm_dtype, world_size))


def pack_mxfp8_scale_mma(scale_rows: torch.Tensor, hidden: int) -> torch.Tensor:
    """Swizzle row-major ``[tokens, hidden // 32]`` UE8M0 scales into b12x MMA tile layout.

    When ``tokens`` is not a multiple of 128, pads trailing rows with neutral UE8M0
    unit scales (127) before swizzling into ``[1, m_tiles, hidden // 128, 32, 4, 4]``.
    """
    hidden = int(hidden)
    if scale_rows.ndim != 2 or scale_rows.shape[1] != hidden // 32 or hidden % 128 != 0:
        raise ValueError("pack_mxfp8_scale_mma requires [tokens, hidden // 32] with hidden % 128 == 0")
    rows = scale_rows.shape[0]
    padded_rows = ((rows + 127) // 128) * 128
    if padded_rows != rows:
        padded = torch.full(
            (padded_rows, scale_rows.shape[1]),
            127,
            dtype=scale_rows.dtype,
            device=scale_rows.device,
        )
        padded[:rows].copy_(scale_rows)
        scale_rows = padded
    m_tiles, k_tiles = padded_rows // 128, hidden // 128
    return (
        scale_rows.view(m_tiles, 4, 32, k_tiles, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(1, m_tiles, k_tiles, 32, 4, 4)
    )


def _cluster_size_for_tokens(tokens: int) -> int:
    from b12x.norm.mhc import _split_lagged

    selector = getattr(_split_lagged, "split_lagged_cluster_size", None)
    return 8 if selector is None else int(selector(tokens))


def _fn_sinkhorn_bucket(tokens: int) -> int | None:
    from b12x.norm.mhc import _split_lagged

    limit = int(getattr(_split_lagged, "SPLIT_LAGGED_MAX_TOKENS", 8))
    if limit <= 8:
        return None
    return 8 if int(tokens) <= 8 else limit


def _compile_fn_sinkhorn(
    ordinal: int,
    hidden: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    tokens: int,
):
    from b12x.norm.mhc._split_lagged import compile_readonly_fn_sinkhorn

    bucket = _fn_sinkhorn_bucket(tokens)
    if bucket is None:
        return compile_readonly_fn_sinkhorn(
            ordinal, hidden, rms_eps, hc_eps, sinkhorn_iters
        ), None
    return (
        compile_readonly_fn_sinkhorn(
            ordinal, hidden, rms_eps, hc_eps, sinkhorn_iters, max_tokens=bucket
        ),
        bucket,
    )


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
        cluster_size=_cluster_size_for_tokens(int(residual.shape[0])),
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
        cluster_size=_cluster_size_for_tokens(int(residual.shape[0])),
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
    main = torch.cuda.current_stream(residual_out.device)
    auxiliary = torch.cuda.ExternalStream(stream, device=residual_out.device)
    auxiliary.wait_stream(main)
    runner, _ = _compile_fn_sinkhorn(
        int(residual_out.device.index or 0),
        int(residual_out.shape[-1]),
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        int(residual_out.shape[0]),
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


@torch.library.custom_op(
    "flare::cluster_mhc_allreduce_post_y_norm",
    mutates_args=("out", "y", "a_values", "a_scales"),
)
def _cluster_mhc_allreduce_post_y_norm(
    x: torch.Tensor,
    input_addend: torch.Tensor | None,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    y: torch.Tensor,
    a_values: torch.Tensor | None,
    a_scales: torch.Tensor | None,
    dependencies: list[torch.Tensor],
    handle: int,
) -> None:
    owner = _FUSED_MHC_AR_OWNERS[handle]
    owner._launch(
        x,
        input_addend,
        residual,
        prev_post,
        prev_comb,
        pre_mix,
        norm_weight,
        out,
        y,
        a_values,
        a_scales,
    )


@_cluster_mhc_allreduce_post_y_norm.register_fake
def _fake_cluster_mhc_allreduce_post_y_norm(
    x,
    input_addend,
    residual,
    prev_post,
    prev_comb,
    pre_mix,
    norm_weight,
    out,
    y,
    a_values,
    a_scales,
    dependencies,
    handle,
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
    """Split lagged mHC for SM120 decode, verify, and draft (up to b12x's supported M, M <= 128).

    Stage 1 (``post_y_norm``) runs ``ClusterMhcPostYNorm`` on the main stream to
    produce ``(residual_out, normalized)`` and optionally pre-quantized MXFP8
    ``(a_values, a_scales)`` for compact W4A8 micro routed experts.

    Stage 2 (``start_sinkhorn`` / ``finish_sinkhorn``) runs ``ReadOnlyFnPartialM8``
    (M <= 8) or ``ReadOnlyFnPartial`` (9 <= M <= 128) and ``_SplitFinalize`` on
    ``workspace.auxiliary_stream`` concurrently with the sublayer's tensor-parallel
    AllReduce.
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
        compile_post_y: bool = True,
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
        self.compile_post_y = bool(compile_post_y)
        register_finalizer(session, self._finalize)

    def functional(self):
        return self

    def _finalize(self):
        from b12x._lib.compile_plan import load_programs
        from b12x.moe.fused_moe._binding_ops import supports_prequant_mxfp8
        from b12x.norm.mhc._split_lagged import compile_cluster_post_y_norm

        if self.experts is not None and not getattr(
            self.experts, "integrated_routing", False
        ):
            self.emit_mxfp8 = supports_prequant_mxfp8(self.experts.plan, self.tokens)
        self.experts = None
        ordinal = int(self.device.index or 0)
        if self.compile_post_y:
            cluster_size = _cluster_size_for_tokens(self.tokens)
            post_y = compile_cluster_post_y_norm(
                ordinal,
                self.hidden,
                self.pre_only,
                self.norm_eps,
                self.weights.norm.dtype,
                cluster_size=cluster_size,
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
                    cluster_size,
                    self.emit_mxfp8,
                )
            ] = post_y
        fn_sink, bucket = _compile_fn_sinkhorn(
            ordinal,
            self.hidden,
            self.rms_eps,
            self.hc_eps,
            self.sinkhorn_iters,
            self.tokens,
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
                bucket,
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


def _check_fused_mhc_operand(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtypes: tuple[torch.dtype, ...],
    device: torch.device,
    align: int,
) -> None:
    if tensor.device != device:
        raise ValueError(
            f"PreparedFusedMhcAllReducePostYNorm {name} must be on {device}, got {tensor.device}"
        )
    if tensor.dtype not in dtypes:
        raise ValueError(
            f"PreparedFusedMhcAllReducePostYNorm {name} requires dtype in {dtypes}, got {tensor.dtype}"
        )
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"PreparedFusedMhcAllReducePostYNorm {name} requires shape {shape}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"PreparedFusedMhcAllReducePostYNorm {name} must be row-major contiguous"
        )
    if tensor.numel() > 0 and tensor.data_ptr() % align != 0:
        raise ValueError(
            f"PreparedFusedMhcAllReducePostYNorm {name} must be {align}-byte aligned"
        )


class PreparedFusedMhcAllReducePostYNorm:
    """Fused Lamport-sentinel PCIe AllReduce + ``ClusterMhcPostYNorm`` for lagged mHC.

    Replaces a standalone single-channel PCIe ``OneshotAllReduce`` (with optional
    BF16 input addend) followed by ``ClusterMhcPostYNorm`` (with optional bit-exact
    MXFP8 quantization and SM120 PDL dependent launch release) in a single kernel
    launch without writing or reading the intermediate ``[tokens, hidden]`` BF16
    reduced activation in HBM.

    Pre-compiles and loads all Lamport double-buffer phase variants
    ``(device_slot_selection=False, slot_bias=0)`` (pre-capture warmup) and
    ``(device_slot_selection=True, slot_bias in {0, 1})`` (CUDA graph capture)
    during preparation finalization so graph capture performs zero compilation
    and tracks the live channel epoch identically to ``PCIeOneshotAllReduce``.
    """

    @staticmethod
    def fits(
        runtime,
        tokens: int,
        hidden: int,
        norm_dtype: torch.dtype,
        *,
        device: torch.device | None = None,
    ) -> bool:
        return supports_fused_mhc_allreduce(
            runtime, tokens, hidden, norm_dtype, device=device
        )

    def __init__(
        self,
        inputs,
        weights: MHCWeights,
        *,
        runtime,
        session,
        norm_eps: float,
        has_addend: bool = False,
        emit_mxfp8: bool = False,
        release_dependents: bool = True,
        experts=None,
    ):
        if len(inputs) != 4:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm requires 4 post_pre inputs "
                "(x, residual, prev_post, prev_comb)"
            )
        x, residual, prev_post, prev_comb = inputs
        if (
            residual.ndim != 3
            or residual.shape[1] != 4
            or residual.dtype != torch.bfloat16
        ):
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm residual requires BF16 [tokens, 4, hidden]"
            )
        tokens, _, hidden = residual.shape
        if x.shape != (tokens, hidden) or x.dtype != torch.bfloat16:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm branch input requires BF16 [tokens, hidden]"
            )
        if prev_post.shape != (tokens, 4) or prev_post.dtype != torch.float32:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm prev_post requires FP32 [tokens, 4]"
            )
        if prev_comb.shape != (tokens, 4, 4) or prev_comb.dtype != torch.float32:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm prev_comb requires FP32 [tokens, 4, 4]"
            )
        if weights.norm.shape != (hidden,) or weights.norm.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm norm weight requires BF16 or FP32 [hidden]"
            )
        if not self.fits(
            runtime, tokens, hidden, weights.norm.dtype, device=residual.device
        ):
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm runtime, geometry, or b12x build is not supported"
            )
        record_numeric_tensor(weights.norm)

        self.runtime = runtime
        self.weights = weights
        self.device = residual.device
        self.tokens = int(tokens)
        self.hidden = int(hidden)
        self.world_size = int(runtime.world_size)
        self.rank = int(runtime.rank)
        self.norm_eps = float(norm_eps)
        self.has_addend = bool(has_addend)
        self.emit_mxfp8 = bool(emit_mxfp8)
        self.release_dependents = bool(release_dependents)
        self.experts = experts
        self._runners: dict[tuple[bool, int], object] = {}
        self.handle = next(_FUSED_MHC_AR_HANDLES)
        _FUSED_MHC_AR_OWNERS[self.handle] = self
        register_finalizer(session, self._finalize)

    def _channel(self):
        channel = _resolve_oneshot_channel(self.runtime)
        if channel is None:
            raise RuntimeError(
                "PreparedFusedMhcAllReducePostYNorm requires a live single-channel PCIe oneshot channel"
            )
        return channel

    def functional(self, *, session=None):
        return self

    def _transport(self) -> str:
        from b12x.norm.mhc import _split_lagged

        selector = getattr(_split_lagged, "cluster_allreduce_transport", None)
        return "auto" if selector is None else str(selector(self.tokens, self.world_size))

    def _compile_variant(self, device_slot_selection: bool, slot_bias: int):
        from b12x.norm.mhc._split_lagged import compile_cluster_allreduce_post_y_norm

        ordinal = int(self.device.index or 0)
        cluster_size = _cluster_size_for_tokens(self.tokens)
        bias = int(slot_bias) & 1 if device_slot_selection else 0
        extra_kwargs: dict[str, object] = {}
        try:
            params = inspect.signature(compile_cluster_allreduce_post_y_norm).parameters
        except (TypeError, ValueError):
            params = ()
        if "transport" in params:
            extra_kwargs["transport"] = self._transport()
        return compile_cluster_allreduce_post_y_norm(
            ordinal,
            self.hidden,
            self.world_size,
            self.rank,
            self.norm_eps,
            self.weights.norm.dtype,
            cluster_size=cluster_size,
            emit_mxfp8=self.emit_mxfp8,
            has_addend=self.has_addend,
            device_slot_selection=device_slot_selection,
            slot_bias=bias,
            release_dependents=self.release_dependents,
            **extra_kwargs,
        )

    def _finalize(self):
        from b12x._lib.compile_plan import load_programs
        from b12x.moe.fused_moe._binding_ops import supports_prequant_mxfp8

        if self.experts is not None and not getattr(
            self.experts, "integrated_routing", False
        ):
            self.emit_mxfp8 = bool(
                self.emit_mxfp8
                or supports_prequant_mxfp8(self.experts.plan, self.tokens)
            )
        self.experts = None
        ordinal = int(self.device.index or 0)
        cluster_size = _cluster_size_for_tokens(self.tokens)
        transport = self._transport()
        for dev_sel, bias in ((False, 0), (True, 0), (True, 1)):
            runner = self._compile_variant(dev_sel, bias)
            load_programs(runner)
            self._runners[(dev_sel, bias)] = runner
            _SPLIT_MHC_RUNNERS[
                (
                    "fused_ar_post_y",
                    ordinal,
                    self.hidden,
                    self.world_size,
                    self.rank,
                    self.norm_eps,
                    self.weights.norm.dtype,
                    cluster_size,
                    self.emit_mxfp8,
                    self.has_addend,
                    dev_sel,
                    bias,
                    self.release_dependents,
                    transport,
                )
            ] = runner

    def _validate_runtime_launch(
        self,
        channel,
        x: torch.Tensor,
        input_addend: torch.Tensor | None,
        residual: torch.Tensor,
        prev_post: torch.Tensor,
        prev_comb: torch.Tensor,
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        out: torch.Tensor,
        y: torch.Tensor,
        a_values: torch.Tensor | None,
        a_scales: torch.Tensor | None,
    ) -> None:
        from b12x.norm.mhc import _split_lagged

        if residual.ndim != 3 or residual.shape[1] != 4 or residual.shape[2] != self.hidden:
            raise ValueError(
                f"PreparedFusedMhcAllReducePostYNorm residual requires shape [tokens, 4, {self.hidden}], "
                f"got {tuple(residual.shape)}"
            )
        tokens = int(residual.shape[0])
        max_tokens = int(getattr(_split_lagged, "CLUSTER_AR_MHC_MAX_TOKENS", 32))
        if tokens <= 0 or tokens > self.tokens or tokens > max_tokens:
            raise ValueError(
                f"PreparedFusedMhcAllReducePostYNorm requires 1 <= tokens <= "
                f"{min(self.tokens, max_tokens)}, got {tokens}"
            )
        hidden = self.hidden
        device = self.device
        _check_fused_mhc_operand(
            "x", x, shape=(tokens, hidden), dtypes=(torch.bfloat16,), device=device, align=16
        )
        if self.has_addend:
            if input_addend is None:
                raise ValueError(
                    "PreparedFusedMhcAllReducePostYNorm with has_addend=True requires input_addend"
                )
            _check_fused_mhc_operand(
                "input_addend",
                input_addend,
                shape=(tokens, hidden),
                dtypes=(torch.bfloat16,),
                device=device,
                align=16,
            )
        elif input_addend is not None:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm with has_addend=False does not accept input_addend"
            )
        _check_fused_mhc_operand(
            "residual",
            residual,
            shape=(tokens, 4, hidden),
            dtypes=(torch.bfloat16,),
            device=device,
            align=16,
        )
        _check_fused_mhc_operand(
            "prev_post",
            prev_post,
            shape=(tokens, 4),
            dtypes=(torch.float32,),
            device=device,
            align=4,
        )
        _check_fused_mhc_operand(
            "prev_comb",
            prev_comb,
            shape=(tokens, 4, 4),
            dtypes=(torch.float32,),
            device=device,
            align=4,
        )
        _check_fused_mhc_operand(
            "pre_mix",
            pre_mix,
            shape=(tokens, 4),
            dtypes=(torch.float32,),
            device=device,
            align=4,
        )
        _check_fused_mhc_operand(
            "norm_weight",
            norm_weight,
            shape=(hidden,),
            dtypes=(self.weights.norm.dtype,),
            device=device,
            align=4,
        )
        _check_fused_mhc_operand(
            "out",
            out,
            shape=(tokens, 4, hidden),
            dtypes=(torch.bfloat16,),
            device=device,
            align=16,
        )
        _check_fused_mhc_operand(
            "y",
            y,
            shape=(tokens, hidden),
            dtypes=(torch.bfloat16,),
            device=device,
            align=16,
        )
        if self.emit_mxfp8:
            if a_values is None or a_scales is None:
                raise ValueError(
                    "PreparedFusedMhcAllReducePostYNorm with emit_mxfp8=True requires a_values and a_scales"
                )
            _check_fused_mhc_operand(
                "a_values",
                a_values,
                shape=(tokens, hidden),
                dtypes=(torch.uint8, torch.float8_e4m3fn),
                device=device,
                align=16,
            )
            _check_fused_mhc_operand(
                "a_scales",
                a_scales,
                shape=(tokens, hidden // 32),
                dtypes=(torch.uint8, torch.float8_e8m0fnu),
                device=device,
                align=16,
            )
        elif a_values is not None or a_scales is not None:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm with emit_mxfp8=False does not accept a_values or a_scales"
            )

        payload_bytes = tokens * hidden * 2
        runtime_max = getattr(self.runtime, "max_size", None)
        if runtime_max is not None and payload_bytes > int(runtime_max):
            raise ValueError(
                f"PreparedFusedMhcAllReducePostYNorm payload ({payload_bytes} bytes) "
                f"exceeds runtime max_size ({int(runtime_max)} bytes)"
            )
        sentinel_cap = _channel_sentinel_capacity_bytes(channel)
        if sentinel_cap is None:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm requires live Lamport sentinel storage"
            )
        if payload_bytes > sentinel_cap:
            raise ValueError(
                f"PreparedFusedMhcAllReducePostYNorm payload ({payload_bytes} bytes) "
                f"exceeds live sentinel shard capacity ({sentinel_cap} bytes)"
            )

    def _launch(
        self,
        x: torch.Tensor,
        input_addend: torch.Tensor | None,
        residual: torch.Tensor,
        prev_post: torch.Tensor,
        prev_comb: torch.Tensor,
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        out: torch.Tensor,
        y: torch.Tensor,
        a_values: torch.Tensor | None,
        a_scales: torch.Tensor | None,
    ) -> None:
        from b12x.norm.mhc._split_lagged import oneshot_channel_sentinel_params

        channel = self._channel()
        self._validate_runtime_launch(
            channel,
            x,
            input_addend,
            residual,
            prev_post,
            prev_comb,
            pre_mix,
            norm_weight,
            out,
            y,
            a_values,
            a_scales,
        )
        (
            peer_table_address,
            signal_table_address,
            shard_packs,
            payload_ready_region_packs,
            device_slot_selection,
            slot_bias,
        ) = oneshot_channel_sentinel_params(channel)
        key = (bool(device_slot_selection), int(slot_bias) & 1 if device_slot_selection else 0)
        runner = self._runners.get(key)
        if runner is None:
            runner = self._compile_variant(key[0], key[1])
            self._runners[key] = runner
        stream = (
            torch.cuda.current_stream(residual.device)
            if residual.device.type == "cuda"
            else None
        )
        runner(
            peer_table_address,
            signal_table_address,
            x,
            residual,
            prev_post,
            prev_comb,
            pre_mix,
            norm_weight,
            out,
            y,
            shard_packs=shard_packs,
            payload_ready_region_packs=payload_ready_region_packs,
            input_addend=input_addend,
            a_values=a_values,
            a_scales=a_scales,
            stream=stream,
        )

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        prev_post: torch.Tensor,
        prev_comb: torch.Tensor,
        *,
        pre_mix: torch.Tensor,
        input_addend: torch.Tensor | None = None,
        dependencies=(),
    ):
        if self.has_addend and input_addend is None:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm with has_addend=True requires input_addend"
            )
        if not self.has_addend and input_addend is not None:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm with has_addend=False does not accept input_addend"
            )
        if residual.ndim != 3 or residual.shape[1] != 4:
            raise ValueError(
                "PreparedFusedMhcAllReducePostYNorm residual requires [tokens, 4, hidden]"
            )
        tokens, _, hidden = residual.shape
        device = residual.device
        out = torch.empty((tokens, 4, hidden), dtype=residual.dtype, device=device)
        y = torch.empty((tokens, hidden), dtype=residual.dtype, device=device)
        if self.emit_mxfp8:
            a_values = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
            a_scales = torch.empty(
                (tokens, hidden // 32), dtype=torch.uint8, device=device
            )
        else:
            a_values = None
            a_scales = None
        _cluster_mhc_allreduce_post_y_norm(
            x,
            input_addend,
            residual,
            prev_post,
            prev_comb,
            pre_mix,
            self.weights.norm,
            out,
            y,
            a_values,
            a_scales,
            list(dependencies),
            self.handle,
        )
        return out, y, a_values, a_scales


def link_fused_mhc_layers(layers) -> None:
    """Defer FFN AllReduce into the following layer's fused AllReduce+mHC when both fit."""
    for prev_layer, next_layer in zip(layers, layers[1:]):
        if (
            getattr(prev_layer, "split_mhc", False)
            and prev_layer.post is None
            and getattr(next_layer, "split_mhc", False)
            and next_layer.carry
            and next_layer.engram is None
            and getattr(next_layer, "attn_fused_mhc", None) is not None
            and getattr(prev_layer, "fused_mhc_runtime", None)
            is next_layer.attn_fused_mhc.runtime
        ):
            prev_layer.defer_mlp_reduce = True
