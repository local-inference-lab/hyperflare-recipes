# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed projections on SM103 through FlashInfer's native SM100-family GEMMs.

b12x prepares SM120/SM121 kernels. On SM103 (GB300/B300) these plans keep
b12x's preparation contract: sessions declare, tune, size scratch, and warm them
like b12x plans, while FlashInfer kernels execute the arithmetic.

MXFP8 weights keep b12x's packed storage. Their UE8M0 scales already use the
F8_128x4 layout that FlashInfer's MXFP8 GEMMs read, and b12x's Triton quantizer
packs activations into caller workspace in that same layout. Tuning chooses the
FlashInfer GEMM backend and its kernel tactic for each shape. Shapes through
eight rows also admit FlashInfer's low-latency block-scaled GEMM with the same
packed operands and scale layout.

NVFP4 weights without an activation scale run with BF16 activations through
FlashInfer's BF16-by-FP4 GEMM, which reads b12x's packed values and F8_128x4
scales in place. NVFP4 weights with padded K, and MXFP8 weights narrower than
the 128 output rows FlashInfer's MXFP8 GEMMs require, instead expand into
caller workspace on each call: E2M1 times its E4M3 block
scale and E4M3 times its UE8M0 scale are exact in BF16, and folding in an NVFP4
global scale rounds once. A BF16 GEMM then runs over the logical K, so padded K
needs no activation copy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache

import torch
from torch import nn
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec
from b12x.preparation import MemoryRequirements, PersistentMemory, Plan, PreparedCall
from b12x.preparation.tuning import BackendConfig, Knob, TuningContract
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare

from .native_operator import native_contract, native_plan

CAPABILITY = (10, 3)
COMPONENT = "flare.sm103.packed_projection"
_QUANTIZED_BACKENDS = ("cute-dsl", "cutlass")
_LOW_LATENCY_BACKEND = "cutedsl_low_latency"
_BF16_BACKENDS = ("cute-dsl", "tinygemm", "cutlass")
_FP4_BACKENDS = ("cute-dsl",)
# FlashInfer's MXFP8 GEMMs need at least one 128-row output tile.
_MIN_QUANTIZED_N = 128
# From this row capacity FlashInfer's CUDA activation quantizer is about twice as fast as b12x's
# Triton quantizer (6.9 against 12.8 us at 4096x2560); below it b12x's launch is shorter.
_CUDA_QUANTIZER_ROWS = 256
_EXPAND_BLOCK = 1024


def is_sm103(device) -> bool:
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == CAPABILITY


@dataclass(frozen=True, kw_only=True)
class PackedProjectionQuery:
    recipe: str
    rows: int
    in_features: int
    padded_in_features: int
    out_features: int
    global_scale_kind: str
    workspace_form: str

    @property
    def mode(self) -> str:
        """quantized MXFP8 GEMM, BF16-by-FP4 GEMM, or BF16 GEMM over an expanded weight."""
        if self.recipe == "nvfp4":
            return "fp4" if self.in_features == self.padded_in_features else "expanded"
        return "quantized" if self.out_features >= _MIN_QUANTIZED_N else "expanded"

    @property
    def expanded(self) -> bool:
        return self.mode == "expanded"

    @property
    def resident(self) -> bool:
        """Narrow MXFP8 weights expand once into preparation-owned BF16 storage."""
        return self.expanded and self.recipe == "mxfp8"


def _mxfp8_runner(backend, device):
    from flashinfer.gemm import gemm_base

    major, minor = torch.cuda.get_device_capability(device)
    if backend == "cutlass":
        return gemm_base.get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
    if backend == _LOW_LATENCY_BACKEND:
        return gemm_base._cutedsl_low_latency_blockscaled_gemm_runner(major * 10 + minor, True)
    return gemm_base._cute_dsl_gemm_mxfp8_runner(major, minor, True, torch.bfloat16)


def _mxfp8_backends(query):
    # CUTLASS requires 32-aligned output features; the low-latency kernel bounds M at eight.
    backends = tuple(backend for backend in _QUANTIZED_BACKENDS
                     if backend != "cutlass" or query.out_features % 32 == 0)
    if query.rows <= 8 and query.padded_in_features % 128 == 0:
        backends += (_LOW_LATENCY_BACKEND,)
    return backends


def _mxfp8_tactics(query, device):
    """FlashInfer's valid MXFP8 kernel tactics for one shape, encoded as config strings."""
    if device is None:
        return tuple(f"{backend}:-1" for backend in _mxfp8_backends(query))
    # The identity describes the architecture; enumerate on the process's current device of that architecture.
    target = torch.device("cuda", torch.cuda.current_device())
    rows, k, n = query.rows, query.padded_in_features, query.out_features
    # Tactic enumeration reads shapes and the device only; operands need no storage.
    a = torch.empty((1,), dtype=torch.float8_e4m3fn, device=target).expand(rows, k)
    b = torch.empty((1,), dtype=torch.float8_e4m3fn, device=target).expand(k, n)
    sf = torch.empty((1,), dtype=torch.uint8, device=target)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="meta")
    inputs = [a, b, sf, sf, torch.bfloat16, out, sf]
    candidates = []
    for backend in _mxfp8_backends(query):
        candidates.append(f"{backend}:-1")
        for tactic in _mxfp8_runner(backend, target).get_valid_tactics(inputs, None):
            if backend != "cute-dsl" or _curated_cute_tactic(tactic, rows):
                candidates.append(f"{backend}:{tactic!r}")
    return tuple(dict.fromkeys(candidates))


def _curated_cute_tactic(tactic, rows):
    """Bound CuTe DSL compilations: each tactic compiles separately, unlike CUTLASS's prebuilt set.

    Decode rows keep swapped small-N tiles; prefill rows keep 128- and 256-wide tiles. Both keep
    one- and two-CTA clusters and every split-K variant, which covered the fastest measured tactics.
    """
    (tile_m, tile_n), cluster, swapped, _, split = tactic
    if split > 1:
        return True
    if cluster not in ((1, 1), (2, 1)):
        return False
    if rows <= 64:
        return swapped and tile_m == 128 and tile_n <= 64
    return not swapped and tile_m in (128, 256) and tile_n in (128, 256)


def _decode_tactic(choice):
    import ast

    backend, _, tactic = choice.partition(":")
    return backend, ast.literal_eval(tactic) if tactic else -1


def _backends(query, device=None):
    if query.mode == "quantized":
        return _mxfp8_tactics(query, device)
    return {"expanded": _BF16_BACKENDS, "fp4": _FP4_BACKENDS}[query.mode]


def _validate_query(query, device):
    del device
    if not isinstance(query, PackedProjectionQuery):
        raise TypeError("query must be PackedProjectionQuery")
    if query.recipe not in ("mxfp8", "nvfp4"):
        raise ValueError("SM103 packed projections support MXFP8 and BF16-activation NVFP4")
    if min(query.rows, query.in_features, query.out_features) <= 0:
        raise ValueError("packed projection dimensions must be positive")
    if query.in_features > query.padded_in_features or query.in_features % 8 or query.out_features % 8:
        raise ValueError("packed projections require K8 and N8 with logical K within stored K")
    if query.recipe == "mxfp8" and (query.padded_in_features % 128 or query.global_scale_kind != "none"):
        raise ValueError("MXFP8 projections require stored K128 and no global scale")
    if query.recipe == "nvfp4" and (query.padded_in_features % 16
                                    or query.global_scale_kind not in ("multiplier", "reciprocal")):
        raise ValueError("NVFP4 projections require stored K16 and a weight global scale")
    if query.workspace_form not in ("owned", "provided"):
        raise ValueError("workspace form must be owned or provided")


def _validate_config(query, config, device):
    if not isinstance(config, BackendConfig):
        raise ValueError(f"{COMPONENT} requires a backend configuration")
    if query.mode == "quantized":
        backend, _ = _decode_tactic(config.backend)
        if backend not in _mxfp8_backends(query):
            raise ValueError(f"unsupported {COMPONENT} MXFP8 backend {config.backend!r}")
    elif config.backend not in _backends(query, device):
        raise ValueError(f"unsupported {COMPONENT} backend {getattr(config, 'backend', None)!r}")


TUNING = TuningContract(
    component_id=COMPONENT,
    query_schema_version=1,
    config_schema_version=2,
    candidate_contract_version=2,
    query_fields=frozenset(field.name for field in fields(PackedProjectionQuery)),
    config_fields=frozenset({"backend"}),
    encode_query=lambda query: {field.name: getattr(query, field.name)
                                for field in fields(PackedProjectionQuery)},
    encode_config=BackendConfig.to_dict,
    decode_config=BackendConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=lambda query, device: BackendConfig(backend=_backends(query)[0]),
    knobs=(Knob(name="backend", values=None),),
    parameters=lambda query, device: {"backend": _backends(query, device)},
)


def _offsets(query):
    """Quantized values, swizzled scales, and the quantizer's alpha slot."""
    if query.mode == "fp4":
        # A reciprocal global scale becomes the GEMM's alpha.
        return None, 0, 256
    if query.resident:
        return None, None, 256
    if query.expanded:
        return None, None, triton.cdiv(query.out_features * query.in_features * 2, 256) * 256
    values = query.rows * query.padded_in_features
    scales = triton.cdiv(query.rows, 128) * triton.cdiv(query.padded_in_features // 32, 4) * 512
    scale_start = triton.cdiv(values, 256) * 256
    alpha_start = triton.cdiv(scale_start + scales, 256) * 256
    return scale_start, alpha_start, alpha_start + 256


def workspace_bytes(query) -> int:
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

    return _offsets(query)[2] + (0 if query.mode == "fp4" else DEFAULT_WORKSPACE_SIZE)


@triton.jit
def _expand_nvfp4(W, S, G, OUT, K: tl.constexpr, STORED_K: tl.constexpr, GROUPS: tl.constexpr,
                  RECIPROCAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = k < K
    byte = tl.load(W + row * (STORED_K // 2) + k // 2, mask, 0)
    code = tl.where(k % 2 == 0, byte & 15, byte >> 4)
    magnitude = code & 7
    value = tl.where(magnitude < 4, magnitude.to(tl.float32) * 0.5,
                     tl.where(magnitude == 4, 2.0,
                              tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0))))
    value = tl.where(code >= 8, -value, value)
    group = k // 16
    offset = ((row // 128 * ((GROUPS + 3) // 4) + group // 4) * 512
              + row % 32 * 16 + row // 32 % 4 * 4 + group % 4)
    scale = tl.load(S + offset, mask, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    global_scale = tl.load(G)
    if RECIPROCAL:
        global_scale = tl.div_rn(1.0, global_scale)
    tl.store(OUT + row * K + k, (value * scale * global_scale).to(tl.bfloat16), mask)


@triton.jit
def _expand_mxfp8(W, S, OUT, K: tl.constexpr, STORED_K: tl.constexpr, GROUPS: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = k < K
    value = tl.load(W + row * STORED_K + k, mask, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    group = k // 32
    offset = ((row // 128 * ((GROUPS + 3) // 4) + group // 4) * 512
              + row % 32 * 16 + row // 32 % 4 * 4 + group % 4)
    exponent = tl.load(S + offset, mask, 127).to(tl.int32)
    tl.store(OUT + row * K + k, (value * tl.exp2((exponent - 127).to(tl.float32))).to(tl.bfloat16), mask)


# In-process preparation resolves Triton programs it recognizes as native; flare's own kernels register.
_NATIVE_JITS.update((_expand_nvfp4, _expand_mxfp8))


def _expand_grid(query):
    return (query.out_features, triton.cdiv(query.in_features, _EXPAND_BLOCK), 1)


def _quantize_grid(query, rows):
    return (rows, triton.cdiv(query.padded_in_features // 32, 16), 1)


def _expand_constants(query):
    if query.recipe == "nvfp4":
        return (query.in_features, query.padded_in_features, query.padded_in_features // 16,
                query.global_scale_kind == "reciprocal", _EXPAND_BLOCK)
    return query.in_features, query.padded_in_features, query.padded_in_features // 32, _EXPAND_BLOCK


def _quantize_constants(query):
    return query.in_features, query.padded_in_features, False, False, 32, 16


def _cuda_quantizer(query):
    return query.rows >= _CUDA_QUANTIZER_ROWS and query.in_features == query.padded_in_features


@program_cache(scope="preparation")
def compile_programs(query_payload, ordinal):
    """Compile the Triton packing programs; FlashInfer builds its GEMMs on first preparation use."""
    query = PackedProjectionQuery(**dict(query_payload))
    with torch.cuda.device(ordinal):
        if query.expanded:
            names = (("K", "STORED_K", "GROUPS", "RECIPROCAL", "BLOCK") if query.recipe == "nvfp4"
                     else ("K", "STORED_K", "GROUPS", "BLOCK"))
            kernel, types = ((_expand_nvfp4, (torch.uint8, torch.uint8, torch.float32, torch.bfloat16))
                             if query.recipe == "nvfp4" else
                             (_expand_mxfp8, (torch.uint8, torch.uint8, torch.bfloat16)))
            return {"expand": kernel.warmup(*types, **dict(zip(names, _expand_constants(query))),
                                            grid=_expand_grid(query), num_warps=4)}
        from b12x.gemm.blockscaled import _quantize

        names = ("INPUT_K", "K", "FP4", "RECIPROCAL", "GROUP", "CHUNKS")
        return {"quantize": _quantize._quantize.warmup(
            torch.bfloat16, torch.float8_e4m3fn, torch.uint8, torch.float32, torch.float32, torch.float32,
            query.rows, **dict(zip(names, _quantize_constants(query))), grid=_quantize_grid(query, query.rows),
            num_warps=4, num_stages=1, enable_fp_fusion=False)}


@dataclass(frozen=True)
class PreparedPackedProjection:
    query: PackedProjectionQuery
    backend: str
    device: torch.device
    programs: dict
    required_workspace: int
    workspace: torch.Tensor | None
    runner: object = None
    tactic: object = None
    expanded: torch.Tensor | None = None
    weight: torch.Tensor | None = None

    @property
    def owned_nbytes(self) -> int:
        return 0 if self.workspace is None else self.workspace.numel()

    def run(self, source, values, scales, global_scale, *, activation_scale=None,
            out=None, workspace=None, stream=None):
        from b12x.gemm.blockscaled._a16 import _check_tensor, _overlap, _stream_context, scale_storage

        q = self.query
        nvfp4 = q.recipe == "nvfp4"
        if activation_scale is not None:
            raise ValueError("SM103 packed projections run without an activation global scale")
        if source.device != self.device or source.dtype != torch.bfloat16 or source.shape[-1] != q.in_features:
            raise ValueError("packed source differs from the prepared dtype, device, or K")
        _check_tensor("source", source, self.device, torch.bfloat16)
        rows = source.numel() // q.in_features
        if rows > q.rows:
            raise ValueError("packed source exceeds the prepared row capacity")
        if out is None or out.dtype != torch.bfloat16 or tuple(out.shape) != (*source.shape[:-1], q.out_features):
            raise ValueError("packed projections write a caller-owned BF16 output")
        _check_tensor("out", out, self.device, torch.bfloat16)
        stored_k = q.padded_in_features // (2 if nvfp4 else 1)
        _check_tensor("weight", values, self.device, torch.uint8 if nvfp4 else torch.float8_e4m3fn)
        if tuple(values.shape) != (q.out_features, stored_k):
            raise ValueError("packed weight geometry differs from preparation")
        scale_bytes = scale_storage(scales, q.out_features, q.padded_in_features, 16 if nvfp4 else 32)
        if nvfp4 != (global_scale is not None):
            raise ValueError("only NVFP4 weights carry a global scale")
        if q.workspace_form == "provided":
            if workspace is None:
                raise ValueError("packed projection requires its caller-provided workspace")
        elif workspace is not None:
            raise ValueError("owned-workspace projection does not accept a caller workspace")
        else:
            workspace = self.workspace
        _check_tensor("workspace", workspace, self.device, torch.uint8)
        if workspace.numel() < self.required_workspace:
            raise ValueError("workspace is smaller than the prepared requirement")
        reads = (source, values, scale_bytes) + ((global_scale,) if nvfp4 else ())
        if any(_overlap(out, tensor) or _overlap(workspace, tensor) for tensor in reads) or _overlap(workspace, out):
            raise ValueError("output and workspace must not overlap packed operands")
        if rows == 0:
            return out
        source_2d, out_2d = source.view(rows, q.in_features), out.view(rows, q.out_features)
        with torch.cuda.device(self.device), _stream_context(stream, self.device):
            if q.mode == "fp4":
                from flashinfer.gemm import mm_bf16_fp4, prepare_bf16_fp4_weights

                alpha = global_scale.view(1)
                if q.global_scale_kind == "reciprocal":
                    alpha = torch.reciprocal(alpha, out=workspace[:4].view(torch.float32))
                # SM100-family preparation is a strided view of the packed values and scales.
                weight, weight_scales, alpha = prepare_bf16_fp4_weights(
                    values, scale_bytes, alpha, backend=self.backend)
                mm_bf16_fp4(source_2d, weight, weight_scales, alpha, backend=self.backend, out=out_2d)
                return out
            if q.expanded:
                from flashinfer.gemm import mm_bf16

                if q.resident:
                    if values.data_ptr() != self.weight.data_ptr():
                        raise ValueError("resident expansion belongs to a different weight")
                    expanded = self.expanded
                else:
                    expanded = workspace[: q.out_features * q.in_features * 2].view(torch.bfloat16)
                    expanded = expanded.view(q.out_features, q.in_features)
                    operands = ((values, scale_bytes, global_scale, expanded) if nvfp4
                                else (values.view(torch.uint8), scale_bytes, expanded))
                    self.programs["expand"][_expand_grid(q)](*operands, *_expand_constants(q))
                mm_bf16(source_2d, expanded.T, out=out_2d, backend=self.backend,
                        workspace_buffer=workspace[_offsets(q)[2]:])
                return out
            scale_start, alpha_start, _ = _offsets(q)
            packed_values = workspace[: rows * stored_k].view(torch.float8_e4m3fn)
            scale_size = triton.cdiv(rows, 128) * triton.cdiv(q.padded_in_features // 32, 4) * 512
            packed_scales = workspace[scale_start: scale_start + scale_size]
            alpha = workspace[alpha_start: alpha_start + 4].view(torch.float32)
            if _cuda_quantizer(q):
                # Writes values and F8_128x4 scales for the live rows; padded rows are never stored.
                from flashinfer import SfLayout

                self.programs["quantize"].mxfp8_quantize(
                    source_2d, packed_values.view(rows, stored_k), packed_scales,
                    SfLayout.layout_128x4.value, 32, False)
            else:
                self.programs["quantize"][_quantize_grid(q, rows)](
                    source, packed_values, packed_scales, alpha, alpha, alpha, rows, *_quantize_constants(q))
            workspace_buffer = workspace[_offsets(q)[2]:]
            self.runner(inputs=[packed_values.view(rows, stored_k), values.T, packed_scales, scale_bytes,
                                torch.bfloat16, out_2d, workspace_buffer], tactic=self.tactic)
            return out


def query_for(source, weight, output, *, workspace_form) -> PackedProjectionQuery:
    from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight, _weight_parts

    values, _, _, fp4 = _weight_parts(weight)
    if isinstance(weight, NVFP4LinearWeight):
        recipe, scale_kind, padded = "nvfp4", weight.global_scale_kind, values.shape[1] * 2
    elif values.dtype == torch.float8_e4m3fn and not fp4:
        recipe, scale_kind, padded = "mxfp8", "none", weight.padded_in_features
    else:
        raise ValueError("SM103 packed projections support MXFP8 and NVFP4 weights")
    return PackedProjectionQuery(
        recipe=recipe, rows=source.numel() // source.shape[-1], in_features=source.shape[-1],
        padded_in_features=padded, out_features=output.shape[-1], global_scale_kind=scale_kind,
        workspace_form=workspace_form)


def _triton_programs(query) -> bool:
    """Whether the call launches Triton packing programs; FlashInfer's own kernels build on first use."""
    return query.expanded or query.mode == "quantized" and not _cuda_quantizer(query)


def plan(query: PackedProjectionQuery, *, weight=None) -> Plan:
    """Declare one projection; resident expansions bind their weight's values and scales."""
    if query.resident and weight is None:
        raise ValueError("narrow MXFP8 projections expand their own weight and require it")

    def memory(config, device):
        from b12x.preparation import current_plan, current_prepared_state

        needed = workspace_bytes(query)
        existing = current_prepared_state()
        persistent = []
        if query.resident:
            nbytes = query.out_features * query.in_features * 2
            held = 0 if existing is None or existing.expanded is None else nbytes
            persistent.append(PersistentMemory((COMPONENT, "expanded", current_plan()), nbytes, held))
        if query.workspace_form == "provided":
            return MemoryRequirements((scratch_buffer_spec(
                "sm103_projection.scratch", nbytes=needed, device=torch.device("cuda", device.ordinal)),),
                tuple(persistent))
        resident = 0 if existing is None else min(needed, existing.owned_nbytes)
        persistent.append(PersistentMemory((COMPONENT, current_plan()), needed, resident))
        return MemoryRequirements((), tuple(persistent))

    def materialize(selection, device):
        from b12x.preparation import current_prepared_state

        resolved = torch.device("cuda", device.ordinal)
        needed = workspace_bytes(query)
        workspace = None
        if query.workspace_form == "owned":
            existing = current_prepared_state()
            previous = None if existing is None else existing.workspace
            workspace = (previous if previous is not None and previous.numel() >= needed
                         else torch.empty(needed, dtype=torch.uint8, device=resolved))
        programs = {}
        if _triton_programs(query):
            programs = compile_programs(TUNING.encode_query(query), device.ordinal)
        elif query.mode == "quantized":
            from flashinfer.quantization.fp8_quantization import gen_mxfp8_quantization_sm100_module

            programs = {"quantize": gen_mxfp8_quantization_sm100_module().build_and_load()}
        runner = tactic = None
        if query.mode == "quantized":
            backend, tactic = _decode_tactic(selection.config.backend)
            runner = _mxfp8_runner(backend, resolved)
        expanded = values = None
        if query.resident:
            from b12x.gemm.blockscaled._a16 import scale_storage
            from b12x.preparation import current_prepared_state

            values, scales = weight
            existing = current_prepared_state()
            expanded = existing.expanded if existing is not None and existing.expanded is not None else None
            if expanded is None:
                expanded = torch.empty((query.out_features, query.in_features), dtype=torch.bfloat16,
                                       device=resolved)
                with torch.cuda.device(resolved):
                    programs["expand"][_expand_grid(query)](
                        values.view(torch.uint8), scale_storage(scales, query.out_features,
                                                                query.padded_in_features, 32),
                        expanded, *_expand_constants(query))
        return PreparedPackedProjection(query, selection.config.backend, resolved, programs, needed, workspace,
                                        runner, tactic, expanded, values)

    def jobs(config, device):
        if not _triton_programs(query):
            return ()
        return (CompileJob.create("hyperflare_recipes.components.sm103_projection:compile_programs",
                                  TUNING.encode_query(query), device.ordinal),)

    return Plan(contract=TUNING, query=query, shared=not query.resident, _compile_jobs=jobs,
                _memory_requirements=memory, _materialize=materialize)


def owns(plan_handle: int) -> bool:
    return plan_from_handle(plan_handle).component_id == COMPONENT


def run(source, values, scales, global_scale, out, workspace, plan_handle):
    state = require_prepared(plan_from_handle(plan_handle), COMPONENT, source.device)
    state.run(source, values, scales, global_scale, out=out, workspace=workspace)


@torch.library.custom_op("flare::sm103_packed_out", mutates_args=("out", "workspace"))
def _packed_out(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                global_scale: torch.Tensor | None, out: torch.Tensor, workspace: torch.Tensor,
                plan_handle: int) -> None:
    run(source, values, scales, global_scale, out, workspace, plan_handle)


@_packed_out.register_fake
def _fake_packed_out(source, values, scales, global_scale, out, workspace, plan_handle):
    pass


def packed_out_workspace(source, values, scales, global_scale, activation_scale, out, workspace,
                         global_scale_kind, stream, plan_handle, required_mode):
    """Run a prepared packed projection with b12x's caller-buffer signature on any architecture."""
    if owns(plan_handle):
        if activation_scale is not None or stream is not None:
            raise ValueError("SM103 packed projections run on the current stream without an activation scale")
        run(source, values, scales, global_scale, out, workspace, plan_handle)
        return
    from b12x.gemm.blockscaled._ops import _out_workspace

    _out_workspace(source, values, scales, global_scale, activation_scale, out, workspace,
                   global_scale_kind, stream, plan_handle, required_mode)


# BF16 projections on SM103 run FlashInfer's GEMMs for prefill rows and wide outputs: on GB300 b12x's GEMV
# reaches about 1.7 TB/s (734 us for a 154880x4096 vocabulary at one row, against 227 us) and runs prefill rows
# as GEMVs (173 us for GLM's 288-expert router at 4096 rows, against 8 us). Narrow decode projections keep b12x's
# single-launch GEMV: cuBLASLt often splits K into a GEMM and a reduction, which in a dependent decode chain cost
# Qwen 0.37 ms per round although each call alone is as fast.
GEMV_ROWS = 32
GEMV_OUT_FEATURES = 4096


def dense_serves(rows: int, out_features: int) -> bool:
    return rows > GEMV_ROWS or out_features >= GEMV_OUT_FEATURES


DENSE_COMPONENT = "flare.sm103.dense_projection"
_DENSE_BACKENDS = ("cublaslt", "cutlass", "cudnn")


@dataclass(frozen=True)
class DenseProjectionQuery:
    rows: int
    in_features: int
    out_features: int
    out_dtype: str


class PreparedDenseProjection:
    """One FlashInfer BF16 GEMM backend, writing BF16 or FP32 into caller-owned output."""

    def __init__(self, backend):
        self.backend = backend

    def run(self, source, weight, out, bias=None, workspace=None):
        from flashinfer.gemm import mm_bf16

        if bias is not None:
            raise ValueError("SM103 dense projections take no bias")
        rows = source.numel() // source.shape[-1]
        if rows:
            mm_bf16(source.view(rows, -1), weight.T, out=out.view(rows, -1), out_dtype=out.dtype,
                    backend=self.backend, workspace_buffer=workspace)
        return out


def _dense_backends(query):
    # CUTLASS's and cuDNN's TMA GEMMs need 16-byte rows; cuBLASLt serves any shape (a one-wide shared-expert gate).
    output_bytes = 4 if query.out_dtype == "float32" else 2
    aligned = query.out_features * output_bytes % 16 == 0 and query.in_features % 8 == 0
    return _DENSE_BACKENDS if aligned else _DENSE_BACKENDS[:1]


DENSE_TUNING = native_contract(component_id=DENSE_COMPONENT, query_type=DenseProjectionQuery,
                               tactics=_dense_backends)


def dense_plan(query: DenseProjectionQuery, device) -> Plan:
    return native_plan(contract=DENSE_TUNING, query=query, device=device, scratch_bytes=lambda tactic: 0,
                       materialize=PreparedDenseProjection)


def bf16_out(source, weight, out, plan_handle):
    """Run a prepared BF16 projection by its plan handle: FlashInfer's GEMM on SM103, b12x's GEMV elsewhere."""
    plan = plan_from_handle(plan_handle)
    if plan.component_id == DENSE_COMPONENT:
        dense_out(source, weight, out, plan_handle)
        return
    torch.ops.b12x.bf16_gemv_small_n_out(source, weight, out, plan_handle, None)


def dense_out(source, weight, out, plan_handle):
    """Run a selected SM103 dense plan with compiler-visible workspace."""
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

    workspace = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=source.device)
    _dense_out_workspace(source, weight, out, workspace, plan_handle)


@torch.library.custom_op("flare::sm103_dense_out", mutates_args=("out",))
def _dense_out(source: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, plan_handle: int) -> None:
    require_prepared(plan_from_handle(plan_handle), DENSE_COMPONENT, source.device).run(source, weight, out=out)


@_dense_out.register_fake
def _fake_dense_out(source, weight, out, plan_handle):
    pass


@torch.library.custom_op("flare::sm103_dense_out_workspace", mutates_args=("out", "workspace"))
def _dense_out_workspace(source: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
                          workspace: torch.Tensor, plan_handle: int) -> None:
    require_prepared(plan_from_handle(plan_handle), DENSE_COMPONENT, source.device).run(
        source, weight, out=out, workspace=workspace)


@_dense_out_workspace.register_fake
def _fake_dense_out_workspace(source, weight, out, workspace, plan_handle):
    pass


LOW_LATENCY_FP8_COMPONENT = "flare.sm103.low_latency_fp8"
LOW_LATENCY_FP8_MAX_ROWS = 64
_LOW_LATENCY_GEMM_MAX_ROWS = 8


@triton.jit
def _deepgemm_sf_to_f8_128x4(
    Src, Dst, src_stride_col,
    M: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = tile_k < K_TILES
    if M <= 8:
        for r in tl.static_range(M):
            val = tl.load(Src + r + tile_k * src_stride_col, mask=mask_k, other=0)
            tl.store(Dst + tile_k * 128 + r * 4, val, mask=mask_k)
    else:
        r = tl.arange(0, BLOCK_M)
        mask = (r[:, None] < M) & mask_k[None, :]
        val = tl.load(Src + r[:, None] + tile_k[None, :] * src_stride_col, mask=mask, other=0)
        dst_off = tile_k[None, :] * 128 + (r[:, None] % 32) * 4 + (r[:, None] // 32)
        tl.store(Dst + dst_off, val, mask=mask)


@triton.jit
def _quantize_bf16_to_f8_128x4(
    X, YFp8, YSf, x_stride_m,
    H: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    kt = tl.program_id(1).to(tl.int64)
    g = tl.arange(0, 4)
    e = tl.arange(0, 32)
    col = kt * 128 + g[:, None] * 32 + e[None, :]
    xf = tl.load(X + m * x_stride_m + col).to(tl.float32)
    amax = tl.max(tl.abs(xf), axis=1)
    raw_sf = tl.maximum(amax * (1.0 / 448.0), 1e-30)
    bits = raw_sf.to(tl.int32, bitcast=True)
    exp_u8 = ((bits + 0x7FFFFF) >> 23) & 0xFF
    scale_f = (exp_u8 << 23).to(tl.float32, bitcast=True)
    q = (xf / scale_f[:, None]).to(tl.float8e4nv)
    tl.store(YFp8 + m * H + col, q)
    shifts = tl.arange(0, 4) * 8
    packed = tl.sum(exp_u8 << shifts, axis=0)
    dst_off = kt * 128 + (m % 32) * 4 + (m // 32)
    tl.store(YSf + dst_off, packed)


_NATIVE_JITS.update((_deepgemm_sf_to_f8_128x4, _quantize_bf16_to_f8_128x4))


@torch.no_grad()
def deepgemm_weight_sf_to_f8_128x4(w_sf_int32: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Convert DeepGEMM packed int32 UE8M0 weight scales to FlashInfer F8_128x4 uint8 layout."""
    if w_sf_int32.dtype != torch.int32 or in_features % 128 or out_features % 128:
        raise ValueError("low-latency FP8 weight scales require int32 packed UE8M0 and 128-aligned N, K")
    raw = w_sf_int32.contiguous().view(torch.uint8).reshape(out_features, in_features // 32)
    k_blk = in_features // 32
    swizzled = raw.view(out_features // 128, 4, 32, k_blk // 4, 4).permute(0, 3, 2, 1, 4).contiguous()
    return swizzled.view(-1)


@dataclass(frozen=True)
class LowLatencyFP8Query:
    rows: int
    in_features: int
    out_features: int
    pre_swizzled_input: bool = False
    bf16_input: bool = False

    def __post_init__(self):
        if (type(self.rows) is not int or not 1 <= self.rows <= LOW_LATENCY_FP8_MAX_ROWS
                or type(self.in_features) is not int or self.in_features <= 0 or self.in_features % 128
                or type(self.out_features) is not int or self.out_features <= 0 or self.out_features % 128
                or type(self.pre_swizzled_input) is not bool or type(self.bf16_input) is not bool):
            raise ValueError(
                f"SM103 low-latency FP8 projection requires 1..{LOW_LATENCY_FP8_MAX_ROWS} rows and 128-aligned N, K")


@lru_cache(maxsize=None)
def _low_latency_fp8_tactics(query: LowLatencyFP8Query) -> tuple:
    target = torch.device("cuda", torch.cuda.current_device())
    rows, k, n = query.rows, query.in_features, query.out_features
    a = torch.empty((1,), dtype=torch.float8_e4m3fn, device=target).expand(rows, k)
    b = torch.empty((1,), dtype=torch.float8_e4m3fn, device=target).expand(k, n)
    sf = torch.empty((1,), dtype=torch.uint8, device=target)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="meta")
    candidates = []
    if rows <= _LOW_LATENCY_GEMM_MAX_ROWS:
        valid = tuple(_mxfp8_runner(_LOW_LATENCY_BACKEND, target).get_valid_tactics(
            [a, b, sf, sf, torch.bfloat16, out, None], None))
        if n >= 16384:
            preferred = ((128, 5, 1, 1), (128, 6, 1, 1), (128, 4, 1, 1))
            curated = [t for t in valid if t[2] == 1 and t[3] == 1 and t[0] in (128, 256) and t[1] in (3, 4, 5, 6)]
        elif n >= 4096:
            preferred = ((128, 6, 1, 2), (128, 8, 1, 2), (256, 5, 1, 2))
            curated = [t for t in valid if t[2] == 1 and t[3] in (1, 2) and t[0] in (128, 256) and t[1] in (5, 6, 8, 9)]
        else:
            preferred = ((128, 6, 1, 8), (640, 2, 1, 8), (128, 7, 1, 8), (128, 5, 1, 4))
            curated = [t for t in valid if t[2] == 1 and t[3] in (4, 8) and t[0] in (128, 256, 640) and t[1] in (2, 4, 5, 6, 7)]
        ordered = [t for t in preferred if t in valid] + [t for t in curated if t not in preferred]
        for t in (ordered or valid):
            candidates.append((_LOW_LATENCY_BACKEND, t))
    inps = [a, b, sf, sf, torch.bfloat16, out, sf]
    cute_valid = tuple(_mxfp8_runner("cute-dsl", target).get_valid_tactics(inps, None))
    cute_curated = [
        t for t in cute_valid
        if t[4] == 1 and t[2] and t[0][0] == 128 and t[0][1] in (8, 16, 32, 64) and t[1] in ((1, 1), (2, 1))
        and not t[3]
    ]
    for t in cute_curated:
        candidates.append(("cute-dsl", t))
    if rows > _LOW_LATENCY_GEMM_MAX_ROWS and n % 32 == 0:
        cutlass_valid = tuple(_mxfp8_runner("cutlass", target).get_valid_tactics(inps, None))
        for t in cutlass_valid[:10]:
            candidates.append(("cutlass", t))
    candidates.append(("deep_gemm", None))
    return tuple(dict.fromkeys(candidates))


LOW_LATENCY_FP8_CONTRACT = native_contract(
    component_id=LOW_LATENCY_FP8_COMPONENT,
    query_type=LowLatencyFP8Query,
    tactics=_low_latency_fp8_tactics,
    semantic_version=3,
)


@program_cache(scope="preparation")
def compile_low_latency_fp8_swizzle(payload, ordinal):
    query = LowLatencyFP8Query(**payload["query"])
    k_tiles = query.in_features // 128
    block_m = triton.next_power_of_2(query.rows)
    block_k = min(64, triton.next_power_of_2(k_tiles))
    grid = (triton.cdiv(k_tiles, block_k), 1, 1)
    with torch.cuda.device(ordinal):
        return (_deepgemm_sf_to_f8_128x4.warmup(
            torch.int32, torch.int32, 16,
            query.rows, k_tiles, block_m, block_k,
            grid=grid, num_warps=4,
        ),)


@program_cache(scope="preparation")
def compile_low_latency_fp8_bf16_quant(payload, ordinal):
    query = LowLatencyFP8Query(**payload["query"])
    k_tiles = query.in_features // 128
    with torch.cuda.device(ordinal):
        return (_quantize_bf16_to_f8_128x4.warmup(
            torch.bfloat16, torch.float8_e4m3fn, torch.int32, query.in_features,
            H=query.in_features,
            grid=(query.rows, k_tiles, 1), num_warps=4,
        ),)


class PreparedLowLatencyFP8:
    def __init__(self, query: LowLatencyFP8Query, tactic, device):
        self.query = query
        self.backend, self.tactic = tactic
        self.device = torch.device(device)
        self.k_tiles = query.in_features // 128
        self.block_m = triton.next_power_of_2(query.rows)
        self.block_k = min(64, triton.next_power_of_2(self.k_tiles))
        self.grid = (triton.cdiv(self.k_tiles, self.block_k), 1, 1)
        self.needs_workspace = self.backend in ("cute-dsl", "cutlass")
        if self.backend == "deep_gemm":
            self.swizzle = None
            self.bf16_quant = None
            self.runner = None
        else:
            encoded = dict(query=LOW_LATENCY_FP8_CONTRACT.encode_query(query))
            self.swizzle, = load_programs(compile_low_latency_fp8_swizzle(encoded, self.device.index))
            self.bf16_quant = (load_programs(compile_low_latency_fp8_bf16_quant(encoded, self.device.index))[0]
                               if query.bf16_input else None)
            self.runner = _mxfp8_runner(self.backend, self.device)

    def run(self, x_fp8, x_sf, weight, weight_sf_f8, weight_sf_dg, out, sf_scratch, workspace):
        if self.backend == "deep_gemm":
            import deep_gemm

            deep_gemm.fp8_gemm_nt((x_fp8, x_sf), (weight, weight_sf_dg), out, recipe=(1, 1, 32))
            return
        if x_sf.dtype == torch.uint8 and x_sf.ndim == 1:
            sf_f8 = x_sf
        else:
            self.swizzle[self.grid](
                x_sf, sf_scratch, x_sf.stride(1),
                self.query.rows, self.k_tiles, self.block_m, self.block_k,
            )
            sf_f8 = sf_scratch.view(torch.uint8)
        ws = workspace if self.needs_workspace else None
        self.runner(
            inputs=[x_fp8, weight.T, sf_f8, weight_sf_f8, torch.bfloat16, out, ws],
            tactic=self.tactic,
        )

    def run_bf16(self, x_bf16, weight, weight_sf_f8, x_fp8, sf_scratch, out, workspace):
        self.bf16_quant[(self.query.rows, self.k_tiles, 1)](
            x_bf16, x_fp8, sf_scratch, x_bf16.stride(0), self.query.in_features,
        )
        ws = workspace if self.needs_workspace else None
        self.runner(
            inputs=[x_fp8, weight.T, sf_scratch.view(torch.uint8), weight_sf_f8, torch.bfloat16, out, ws],
            tactic=self.tactic,
        )


def low_latency_fp8_plan(query: LowLatencyFP8Query, device) -> Plan:
    resolved = torch.device(device)

    def jobs(tactic, detected):
        if tactic[0] == "deep_gemm":
            return ()
        encoded = dict(query=LOW_LATENCY_FP8_CONTRACT.encode_query(query))
        out = [CompileJob.create(
            "hyperflare_recipes.components.sm103_projection:compile_low_latency_fp8_swizzle",
            encoded, detected.ordinal)]
        if query.bf16_input:
            out.append(CompileJob.create(
                "hyperflare_recipes.components.sm103_projection:compile_low_latency_fp8_bf16_quant",
                encoded, detected.ordinal))
        return tuple(out)

    return native_plan(
        contract=LOW_LATENCY_FP8_CONTRACT,
        query=query,
        device=resolved,
        scratch_bytes=lambda tactic: 0,
        compile_jobs=jobs,
        materialize=lambda tactic: PreparedLowLatencyFP8(query, tactic, resolved),
    )


@torch.library.custom_op("flare::sm103_low_latency_fp8", mutates_args=("out", "sf_scratch", "workspace"))
def _low_latency_fp8(
    x_fp8: torch.Tensor, x_sf: torch.Tensor, weight: torch.Tensor,
    weight_sf_f8: torch.Tensor, weight_sf_dg: torch.Tensor,
    out: torch.Tensor, sf_scratch: torch.Tensor | None,
    workspace: torch.Tensor | None, plan_handle: int,
) -> None:
    require_prepared(plan_from_handle(plan_handle), LOW_LATENCY_FP8_COMPONENT, x_fp8.device).run(
        x_fp8, x_sf, weight, weight_sf_f8, weight_sf_dg, out, sf_scratch, workspace)


@_low_latency_fp8.register_fake
def _fake_low_latency_fp8(x_fp8, x_sf, weight, weight_sf_f8, weight_sf_dg, out, sf_scratch, workspace, plan_handle):
    pass


@torch.library.custom_op("flare::sm103_low_latency_fp8_bf16", mutates_args=("x_fp8", "sf_scratch", "out", "workspace"))
def _low_latency_fp8_bf16(
    x_bf16: torch.Tensor, weight: torch.Tensor, weight_sf_f8: torch.Tensor,
    x_fp8: torch.Tensor, sf_scratch: torch.Tensor, out: torch.Tensor,
    workspace: torch.Tensor | None, plan_handle: int,
) -> None:
    require_prepared(plan_from_handle(plan_handle), LOW_LATENCY_FP8_COMPONENT, x_bf16.device).run_bf16(
        x_bf16, weight, weight_sf_f8, x_fp8, sf_scratch, out, workspace)


@_low_latency_fp8_bf16.register_fake
def _fake_low_latency_fp8_bf16(x_bf16, weight, weight_sf_f8, x_fp8, sf_scratch, out, workspace, plan_handle):
    pass


class LowLatencyFP8Linear(nn.Module):
    """Wrap a DeepInfer FP8 ReplicatedLinear to run tuned SM103 FlashInfer GEMMs for M <= 64."""

    def __init__(self, base, rows_list, *, session, name: str,
                 pre_swizzled_input: bool = False, bf16_input: bool = False):
        super().__init__()
        self.base = base
        self.input_size = base.input_size
        self.output_size = base.output_size
        self.use_fp32_output = base.use_fp32_output
        self.pre_swizzled_input = pre_swizzled_input
        self.bf16_input = bf16_input
        device = base.weight.device
        sf_f8 = persistent_copy(deepgemm_weight_sf_to_f8_128x4(
            base.weight.sf, self.output_size, self.input_size))
        self.register_buffer("weight_sf_f8", sf_f8)
        record_numeric_tensor(self.weight_sf_f8)
        self._plans = {}
        requests = []
        for rows in sorted({r for r in rows_list if 1 <= r <= LOW_LATENCY_FP8_MAX_ROWS}):
            query = LowLatencyFP8Query(
                rows, self.input_size, self.output_size,
                pre_swizzled_input=pre_swizzled_input, bf16_input=bf16_input,
            )
            op_plan = low_latency_fp8_plan(query, device)
            self._plans[rows] = op_plan

            def make_prepare(rows=rows, k=self.input_size, n=self.output_size):
                def prepare(state):
                    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

                    out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
                    workspace = (torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
                                 if state.needs_workspace else None)
                    if bf16_input:
                        x_bf16 = torch.zeros((rows, k), dtype=torch.bfloat16, device=device)
                        if state.backend == "deep_gemm":
                            from deepinfer.model_executor.layers.linear import cast_fp8

                            def run_dg():
                                xf, xs = cast_fp8(x_bf16)
                                state.run(xf, xs, base.weight, self.weight_sf_f8, base.weight.sf,
                                          out, None, workspace)

                            return PreparedCall(
                                run=run_dg,
                                produce=x_bf16.zero_,
                                output=out,
                                owners=(base.weight, base.weight.sf, self.weight_sf_f8),
                            )
                        x_fp8 = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device)
                        sf_scratch = torch.empty((128 * (k // 128),), dtype=torch.int32, device=device)
                        return PreparedCall(
                            run=lambda: state.run_bf16(x_bf16, base.weight, self.weight_sf_f8,
                                                       x_fp8, sf_scratch, out, workspace),
                            produce=x_bf16.zero_,
                            output=out,
                            owners=(base.weight, base.weight.sf, self.weight_sf_f8),
                        )

                    x_fp8 = torch.zeros((rows, k), dtype=torch.float8_e4m3fn, device=device)
                    if pre_swizzled_input and state.backend != "deep_gemm":
                        x_sf = torch.full((128 * (k // 128) * 4,), 0x7F, dtype=torch.uint8, device=device)
                        sf_scratch = None

                        def produce():
                            x_fp8.zero_()
                            x_sf.fill_(0x7F)
                    else:
                        aligned_m = ((rows * 4 + 15) // 16) * 16 // 4
                        x_sf = torch.empty_strided((rows, k // 128), (1, aligned_m), dtype=torch.int32, device=device)
                        x_sf.fill_(0x7F7F7F7F)
                        sf_scratch = (None if state.backend == "deep_gemm"
                                      else torch.empty((128 * (k // 128),), dtype=torch.int32, device=device))

                        def produce():
                            x_fp8.zero_()
                            x_sf.fill_(0x7F7F7F7F)

                    return PreparedCall(
                        run=lambda: state.run(x_fp8, x_sf, base.weight, self.weight_sf_f8, base.weight.sf,
                                              out, sf_scratch, workspace),
                        produce=produce,
                        output=out,
                        owners=(base.weight, base.weight.sf, self.weight_sf_f8),
                    )
                return prepare

            prep = make_prepare()
            requests.append(op_plan.request(name=f"{name}.m{rows}", prepare_call=prep, benchmark_call=prep))
        if requests:
            declare(session, tuple(requests))

    def prefers_f8_128x4_scales(self, rows: int) -> bool:
        """Whether this projection executes a FlashInfer MXFP8 kernel for ``rows``."""
        op_plan = self._plans.get(rows)
        if op_plan is None:
            return False
        state = require_prepared(op_plan, LOW_LATENCY_FP8_COMPONENT, self.base.weight.device)
        return state.backend != "deep_gemm"

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x, out_acc=None):
        if out_acc is None:
            if isinstance(x, torch.Tensor):
                rows = x.shape[0] if x.ndim == 2 else 0
            else:
                rows = x[0].shape[0] if x[0].ndim == 2 else 0
            op_plan = self._plans.get(rows)
            if op_plan is not None:
                state = require_prepared(op_plan, LOW_LATENCY_FP8_COMPONENT, self.base.weight.device)
                if isinstance(x, torch.Tensor):
                    if (state.backend != "deep_gemm" and self.bf16_input
                            and x.dtype == torch.bfloat16 and x.stride(1) == 1):
                        x_fp8 = torch.empty((rows, self.input_size), dtype=torch.float8_e4m3fn, device=x.device)
                        sf_scratch = torch.empty((128 * (self.input_size // 128),), dtype=torch.int32, device=x.device)
                        out = torch.empty((rows, self.output_size), dtype=torch.bfloat16, device=x.device)
                        workspace = None
                        if state.needs_workspace:
                            from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE
                            workspace = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=x.device)
                        _low_latency_fp8_bf16(x, self.base.weight, self.weight_sf_f8,
                                              x_fp8, sf_scratch, out, workspace, op_plan.handle)
                        return out
                    from deepinfer.model_executor.layers.linear import cast_fp8
                    x_fp8, x_sf = cast_fp8(x)
                else:
                    x_fp8, x_sf = x
                pre_swizzled = x_sf.dtype == torch.uint8 and x_sf.ndim == 1
                if state.backend == "deep_gemm":
                    if not pre_swizzled:
                        return self.base((x_fp8, x_sf), out_acc=None)
                elif x_fp8.is_contiguous() and (pre_swizzled or (x_sf.dtype == torch.int32 and x_sf.stride(0) == 1)):
                    out = torch.empty((rows, self.output_size), dtype=torch.bfloat16, device=x_fp8.device)
                    sf_scratch = None if pre_swizzled else torch.empty(
                        (128 * (self.input_size // 128),), dtype=torch.int32, device=x_fp8.device)
                    workspace = None
                    if state.needs_workspace:
                        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE
                        workspace = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=x_fp8.device)
                    _low_latency_fp8(x_fp8, x_sf, self.base.weight, self.weight_sf_f8, self.base.weight.sf,
                                     out, sf_scratch, workspace, op_plan.handle)
                    return out
        return self.base(x, out_acc=out_acc)
