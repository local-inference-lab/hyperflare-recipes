# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared graph inputs, hybrid HCA/DSA continuation storage, and ERNIE frontend syntax."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p


VOCABULARY = 201216
INPUT_VOCABULARY = 201216
FRONTEND = c.frontend(
    tools=p.ERNIE_XML,
    reasoning_kind=p.ERNIE_THINK,
    reasoning_open="<ernie:think>",
    reasoning_close="</ernie:think>",
    assistant_marker="<|ernie:im_start|>assistant\n",
    message_end_markers=["<|ernie:im_end|>", "<|ERNIE_MASK|>"],
    prefix_boundary=c.prefix_boundary(
        ["system"],
        required_start="<|ERNIE_MASK|><|ernie:im_start|>system\n",
        required_marker="<|ernie:im_end|>",
    ),
)


def prepare(
    signatures,
    buffers,
    pool,
    *,
    page_tokens,
    draft_lag,
    rows=1,
    embedding_tables=(),
    regions=(),
):
    regions = list(regions)
    for name, tensor in pool.pages.items():
        layer = int(name.split(".")[0].removeprefix("layer"))
        tokens_per_record = 1 if pool.config.is_dsa_layer(layer) else 128
        regions.append(
            c.history_region(
                buffers,
                "cache." + name,
                tensor,
                p.PAGED_LATENT,
                tokens_per_record=tokens_per_record,
            )
        )
    inputs = {
        kind: c.text_inputs(kind)
        for kind in {s.kind for s in signatures.values()}
        if kind != "vision"
    }
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = (
        None
        if draft_lag is None
        else dict(
            state=c.state_view(
                "state_pool",
                pool.layout.row_bytes,
                offset=consumed.offset,
                nbytes=consumed.nbytes,
            ),
            lag_tokens=draft_lag,
        )
    )
    pending = c.state_view("request_pending_token", 8)
    graphs = c.graph_contracts(
        signatures,
        inputs,
        continuation=continuation,
        pending=pending,
        cursor=cursor,
        rows=rows,
        embedding_tables=embedding_tables,
    )
    return c.model_contract(
        vocabulary=VOCABULARY,
        input_vocabulary=INPUT_VOCABULARY,
        page_tokens=page_tokens,
        page_tables=[
            dict(buffer="page_tables", page_offset=0),
            *(
                [dict(buffer="draft_page_tables", page_offset=1)]
                if "draft_page_tables" in buffers
                else []
            ),
        ],
        regions=[
            dict(
                kind=p.CONTINUATION_STATE,
                storage=continuation,
                tokens_per_record=1,
                window_tokens=0,
            ),
            *regions,
        ],
        graphs=graphs,
        vision=None,
        frontend=FRONTEND,
    )
