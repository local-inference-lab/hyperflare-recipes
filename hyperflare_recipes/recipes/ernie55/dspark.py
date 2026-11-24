# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""ERNIE 5.5 DSpark 3-layer SWA-only draft decoder (multimax runs after Markov bias)."""

import torch

from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.mhc import PreparedMHCHead
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.rotary import CompressedRotary as Rotary
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.tuning import register_finalizer

from .attention import AttentionIndices
from .decoder import DraftScratch, PreparedLayer


class PreparedDraft:
    def __init__(
        self,
        weights,
        scratch: DraftScratch,
        metadata,
        pool,
        *,
        runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name,
        workspace,
        mla_compute_mode="auto",
        vocabulary_parallel=False,
        examples=None,
    ):
        if scratch.local_logits.dtype != torch.bfloat16 or scratch.logits.dtype != torch.bfloat16:
            raise ValueError("ERNIE 5.5 DSpark proposal vocabulary logits require BF16 storage")
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        self.embedding = PreparedEmbedding(
            weights.embedding,
            metadata.token_ids,
            scratch.embedding,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        ).functional(session=session, dynamic_inputs=True)
        rotary = Rotary(weights.config, scratch.embedding.device)
        self.indices = AttentionIndices(
            metadata, mode="draft", tokens=metadata.positions.numel(), include_dsa=False
        )
        self.layers = tuple(
            PreparedLayer(
                i,
                layer,
                scratch,
                metadata,
                pool,
                rotary,
                shard=weights.shard,
                mode="draft",
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=f"{name}.layer{i}",
                workspace=workspace,
                carry=i > 0,
                defer_post=i + 1 < len(weights.draft.layers),
                indices=self.indices,
                mla_compute_mode=mla_compute_mode,
                examples=examples,
            )
            for i, layer in enumerate(weights.draft.layers)
        )
        text = weights.config.text
        self.head_hidden = scratch.collapsed
        self.head_hidden_shape = tuple(scratch.collapsed.shape)
        self.logit_shape, self.logit_dtype = tuple(scratch.logits.shape), scratch.logits.dtype
        self.local_logit_shape, self.local_logit_dtype = (
            tuple(scratch.local_logits.shape),
            scratch.local_logits.dtype,
        )
        self.collapse = PreparedMHCHead(
            scratch.head_residual,
            weights.draft.head_mix,
            self.head_hidden,
            rms_eps=text["rms_norm_eps"],
            hc_eps=text["hc_eps"],
            session=session,
            name=name + ".collapse",
        ).functional()
        self.norm = PreparedRMSNorm(
            self.head_hidden,
            weights.draft.norm,
            scratch.hidden,
            eps=text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        ).functional()
        head = weights.head if weights.draft.draft_head is None else weights.draft.draft_head
        self.head = PreparedVocabularyProjection(
            scratch.hidden,
            head,
            scratch.local_logits,
            scratch.logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
            workspace=workspace,
            gather=not vocabulary_parallel,
        ).functional(session=session)
        self.indices.functional()
        register_finalizer(session, self.release_outputs)

    def release_outputs(self):
        self.scratch = self.metadata = self.head_hidden = None

    def __call__(self, metadata):
        embedding = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        incoming = (embedding[:, None, :].expand(-1, 4, -1).contiguous(),)
        indices = self.indices(metadata)
        for layer in self.layers:
            incoming, _, _ = layer(incoming, metadata, indices)
        (residual,) = incoming
        head_hidden = self.collapse(residual[metadata.sample_indices])
        hidden = self.norm(head_hidden)
        # Raw logits (NO multimax yet): Markov bias is added first, then multimax in proposal sampling.
        logits = self.head(hidden)
        features = {"head_hidden": head_hidden.view(metadata.starts.numel(), -1, head_hidden.shape[-1])}
        return logits, None, features
