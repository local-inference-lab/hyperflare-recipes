# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared high-occupancy SM103 BF16 MoE router for low-to-medium decode batches."""

from dataclasses import dataclass
from pathlib import Path

import torch

from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.persistent import persistent_empty
from ignition.preparation.tuning import declare

from .native_operator import native_contract, native_plan

SM103_ROUTER_MAX_ROWS = 64
SM103_ROUTER_COMPONENT = "flare.sm103.bf16_router"
SUPPORTED_ROUTER_GEOMETRIES = frozenset(((384, 5120, 6), (128, 5120, 3)))


def supports_sm103_router(experts: int, hidden: int, top_k: int) -> bool:
    """Return True when the SM103 PDL router covers the requested geometry."""
    return (experts, hidden, top_k) in SUPPORTED_ROUTER_GEOMETRIES


class SM103RouterWorkspace:
    """Persistent Split-K partial-logit buffer shared by serialized SM103 routers."""

    def __init__(self, *, max_rows: int = SM103_ROUTER_MAX_ROWS, max_experts: int = 384,
                 splits: int = 10, device):
        if type(max_rows) is not int or not 1 <= max_rows <= SM103_ROUTER_MAX_ROWS:
            raise ValueError("SM103 router workspace requires 1..64 rows")
        if type(max_experts) is not int or max_experts <= 0 or type(splits) is not int or splits <= 0:
            raise ValueError("SM103 router workspace requires positive expert and split counts")
        device = torch.device(device)
        with torch.cuda.device(device):
            if torch.cuda.get_device_capability() != (10, 3):
                raise ValueError("SM103 router workspace requires an SM103 device")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("SM103 router workspace must be allocated before capture")
            self.part_buf = persistent_empty((splits, max_rows, max_experts), torch.float32, device)
        self.max_rows = max_rows
        self.max_experts = max_experts
        self.splits = splits
        self.device = self.part_buf.device


@dataclass(frozen=True)
class SM103RouterQuery:
    rows: int
    experts: int
    hidden: int
    top_k: int

    def __post_init__(self):
        if (type(self.rows) is not int or not 1 <= self.rows <= SM103_ROUTER_MAX_ROWS
                or not supports_sm103_router(self.experts, self.hidden, self.top_k)):
            raise ValueError("unsupported SM103 BF16 router query geometry")


CONTRACT = native_contract(
    component_id=SM103_ROUTER_COMPONENT,
    query_type=SM103RouterQuery,
    tactics=lambda query: (None,),
)


class _PreparedSM103Router:
    def __init__(self, query: SM103RouterQuery, kernel_cache: Path):
        from .sm103_router_kernel import prepare

        self.query = query
        self.library = prepare(kernel_cache)

    def run(self, x, weight, bias, image_bias, image_mask, valid, placement,
            part_buf, ids, probs, routes, scaling: float):
        self.library.sm103_router_run(
            x, weight, bias, image_bias, image_mask, valid, placement,
            part_buf, ids, probs, routes, scaling,
        )


@torch.library.custom_op("flare::sm103_bf16_router", mutates_args=("part_buf", "ids", "probabilities"))
def _sm103_router_op(
    source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    image_bias: torch.Tensor | None, valid: torch.Tensor,
    image_mask: torch.Tensor | None, placement: torch.Tensor,
    part_buf: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
    scaling: float, handle: int,
) -> None:
    require_prepared(plan_from_handle(handle), SM103_ROUTER_COMPONENT, source.device).run(
        source, weight, bias, image_bias if image_mask is not None else None,
        image_mask, valid, placement, part_buf, ids, probabilities, None, scaling,
    )


@_sm103_router_op.register_fake
def _sm103_router_op_fake(
    source, weight, bias, image_bias, valid, image_mask, placement,
    part_buf, ids, probabilities, scaling, handle,
):
    pass


@torch.library.custom_op(
    "flare::sm103_bf16_router_traced",
    mutates_args=("part_buf", "ids", "probabilities", "routes"),
)
def _sm103_router_traced_op(
    source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    image_bias: torch.Tensor | None, valid: torch.Tensor,
    image_mask: torch.Tensor | None, placement: torch.Tensor,
    part_buf: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
    routes: torch.Tensor, scaling: float, handle: int,
) -> None:
    require_prepared(plan_from_handle(handle), SM103_ROUTER_COMPONENT, source.device).run(
        source, weight, bias, image_bias if image_mask is not None else None,
        image_mask, valid, placement, part_buf, ids, probabilities, routes, scaling,
    )


@_sm103_router_traced_op.register_fake
def _sm103_router_traced_op_fake(
    source, weight, bias, image_bias, valid, image_mask, placement,
    part_buf, ids, probabilities, routes, scaling, handle,
):
    pass


def declare_sm103_router_plans(rows_list, *, experts: int, hidden: int, top_k: int,
                               workspace: SM103RouterWorkspace, kernel_cache: Path,
                               device, session, name: str):
    """Declare SM103 BF16 router plans for ``rows_list`` and return (plans, row_to_handle)."""
    resolved = torch.device(device)
    cache_path = Path(kernel_cache).resolve()
    plans = []
    handles = {}
    requests = []
    for rows in sorted({int(r) for r in rows_list if 1 <= int(r) <= SM103_ROUTER_MAX_ROWS}):
        query = SM103RouterQuery(rows, experts, hidden, top_k)
        plan = native_plan(
            contract=CONTRACT,
            query=query,
            device=resolved,
            scratch_bytes=lambda tactic: 0,
            materialize=lambda tactic, q=query: _PreparedSM103Router(q, cache_path),
        )
        plans.append(plan)
        handles[rows] = plan.handle

        def make_prepare(rows=rows):
            def prepare_call(state):
                x = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=resolved)
                w = torch.zeros((experts, hidden), dtype=torch.bfloat16, device=resolved)
                b = torch.zeros((experts,), dtype=torch.float32, device=resolved)
                v = torch.ones((rows,), dtype=torch.bool, device=resolved)
                p = torch.arange(experts, dtype=torch.int32, device=resolved).view(experts, 1)
                ids = torch.empty((rows, top_k), dtype=torch.int64, device=resolved)
                probs = torch.empty((rows, top_k), dtype=torch.float32, device=resolved)
                return PreparedCall(
                    run=lambda: state.run(x, w, b, None, None, v, p, workspace.part_buf, ids, probs, None, 1.0),
                    output=ids,
                    owners=(workspace,),
                )
            return prepare_call

        requests.append(plan.request(name=f"{name}.m{rows}", prepare_call=make_prepare()))
    if requests:
        declare(session, tuple(requests))
    return tuple(plans), handles


def run_sm103_router(source, weight, bias, image_bias, valid, image_mask, placement,
                     part_buf, ids, probabilities, scaling: float, handle: int, *, routes=None):
    """Execute the prepared SM103 BF16 router, optionally recording canonical routes."""
    if routes is not None:
        _sm103_router_traced_op(
            source, weight, bias, image_bias, valid, image_mask, placement,
            part_buf, ids, probabilities, routes, scaling, handle,
        )
    else:
        _sm103_router_op(
            source, weight, bias, image_bias, valid, image_mask, placement,
            part_buf, ids, probabilities, scaling, handle,
        )
