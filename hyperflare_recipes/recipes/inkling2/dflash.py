# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pinned IncoAI DFlash2 drafter and speculative preparation for Inkling 2 Mini."""

from __future__ import annotations

from hyperflare_recipes.components.dflash2_geometry import DFlash2Config
from hyperflare_recipes.components.dflash2_preparation import DFlash2Preparation

REPOSITORY = "incoai/Inkling-2-Mini-DFlash2"
REVISION = "4f9088ea66c21f2b99ed657d1941c306b03e84d2"
WEIGHT_BYTES = 989_966_848

EXPECTED = dict(
    hidden_size=2304,
    intermediate_size=6912,
    num_hidden_layers=5,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=201024,
    num_target_layers=48,
    rms_norm_eps=1e-6,
    sliding_window=2048,
    rope_parameters=dict(rope_type="default", rope_theta=10000.0),
    dflash_config=dict(
        block_size=8,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=16,
        mask_token_id=200056,
        final_logit_softcapping=30.0,
        selector_softcap_mode="after_correction",
        target_layer_ids=[5, 23, 41, 48],
    ),
)


def load_config(snapshot, target):
    return DFlash2Config.load(
        snapshot,
        target_hidden=target.hidden,
        target_vocabulary=target.vocabulary,
        target_layers=len(target.visits),
        expected=EXPECTED,
        allow_final_norm_tap=True,
    )


class SpeculativePreparation(DFlash2Preparation):
    def __init__(
        self,
        weights,
        pool,
        requests,
        inputs,
        scratch,
        *,
        decoder,
        completion,
        frame,
        sequence,
        session,
        workspace,
        collectives,
        verification_width=8,
        page_tokens=128,
    ):
        draft_config = weights.config.draft
        vision_state = collectives.get("vision_state")
        examples = collectives.get("examples")

        def target_decoder(w, tables, s, metadata, p, **kwargs):
            return decoder(
                w,
                tables,
                s,
                metadata,
                p,
                vision_state=vision_state,
                examples=examples,
                **kwargs,
            )

        draft_head = weights.draft_head if weights.draft_head is not None else weights.head
        super().__init__(
            weights,
            pool,
            requests,
            inputs,
            scratch,
            decoder=target_decoder,
            embedding=weights.embedding,
            embedding_norm_weight=weights.embedding_norm,
            head=draft_head,
            page_tokens=page_tokens,
            completion=completion,
            frame=frame,
            sequence=sequence,
            session=session,
            workspace=workspace,
            verification_width=verification_width,
            runtime=collectives["runtime"],
            vocabulary_runtime=collectives["vocabulary_runtime"],
            coordinator=collectives["coordinator"],
            block_size=draft_config.block_size,
            examples=examples,
            block_proposal_broadcast=True,
            shortlist_only=True,
            output_multiplier=1.0,
            final_logit_softcapping=draft_config.final_logit_softcapping,
            selector_softcap_mode=draft_config.selector_softcap_mode,
            valid_vocabulary=weights.config.text.valid_vocabulary,
        )
