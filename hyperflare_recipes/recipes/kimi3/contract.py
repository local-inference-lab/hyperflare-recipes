# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""XTML frontend, KDA continuations, and paged latent attention bindings."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .config import PAGE_TOKENS
from .dcp import RECORD_BYTES
from .vision import PATCH_CAPACITIES, FEATURES

VISION = dict(features_per_request=FEATURES, max_images=4, patches_per_feature=4, patch_capacities=list(PATCH_CAPACITIES))


IMAGES = dict(image_tokens=dict(feature_token=163605, span=None),
    image_preparation=p.MOONVIT_IMAGE_PREPARATION, image_layout=p.MOONVIT_IMAGE_LAYOUT)


def frontend(vision=True):
    """XTML chat and reasoning markers; with ``vision``, MoonViT image preparation and placement."""
    result = c.frontend(tools=p.XTML, **(IMAGES if vision else {}),
        assistant_marker='<|open|>message role="assistant"<|sep|>',
        message_end_markers=["<|close|>message<|sep|>", "<|end_of_msg|>"],
        prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
            required_start='<|open|>message role="system"', required_marker="<|end_of_msg|>"))
    result["reasoning"] = dict(kind=p.XTML_THINK, open="<|open|>think<|sep|>", close="<|close|>think<|sep|>",
                               wrapping_newlines=0, terminal_markers=[])
    return result


FRONTEND = frontend()


def checkpoint_encoding(name, dtype, shape):
    """Lossless KV codec encoding for one contiguous checkpoint field (0 stores it unencoded)."""
    if name.endswith(".recurrent") and str(dtype) == "torch.float32" and tuple(shape[-2:]) == (128, 128):
        return 1
    if str(dtype) == "torch.bfloat16":
        if name.endswith(".convolution") and shape[-1] in (10, 16):
            return 2 * shape[-1]
        return 2
    if str(dtype) == "torch.float8_e4m3fn":
        if (name.startswith(("draft", "cache.draft")) and name.endswith(".latent")
                and len(shape) == 2 and shape[0] % PAGE_TOKENS == 0 and shape[1] == RECORD_BYTES):
            return 48
        return 0
    if str(dtype) == "torch.uint8":
        if (name.startswith(("layer", "cache.layer")) and name.endswith(".latent")
                and tuple(shape) == (PAGE_TOKENS * RECORD_BYTES,)):
            return 48
        if (name.startswith(("layer", "cache.layer")) and name.endswith(".indexer")
                and tuple(shape) == (PAGE_TOKENS * 132,)):
            return 41
        if (name.startswith(("draft", "cache.draft")) and name.endswith(".latent")
                and tuple(shape) == (PAGE_TOKENS * RECORD_BYTES * 2,)):
            return 2
    return 0


def prepare(signatures, buffers, pool, *, image_slots=4, vocabulary, page_tokens=PAGE_TOKENS, draft_lag=None,
            vision=True):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    # Speculative continuations draft from every accepted position: the draft context trails consumption by draft_lag.
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_LATENT)
               for name, tensor in pool.pages.items()]
    return c.model_contract(vocabulary=vocabulary, page_tokens=page_tokens,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        codec_layout=c.codec_layout(pool, regions, buffers, field_encoding=checkpoint_encoding),
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending, cursor=cursor),
        vision={**VISION, "feature_slots": image_slots} if vision else None,
        frontend=FRONTEND if vision else frontend(vision=False))
