# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose the reference transformer with shared Flare request completion."""

import torch

from ignition.components.sampling import PreparedSampling
from ignition.components.state import save_hidden
from hyperflare_recipes.recipes.deepseek41.reference_metadata import ReferenceDecodeMetadata, ReferencePrefillMetadata


class ReferenceGeneration:
    """Complete ordinary generation over validated, native-owned request storage.

    The reference transformer updates target and draft attention history. Shared
    batch and sampling operations own token consumption, penalties, RNG state and
    completion records. This composition admits one GPU and ordinary prefill or
    one-token decode; speculative acceptance requires a separate complete graph.
    """

    def __init__(self, model, batch, history, completion, frame, sequence, *, session, vision=None,
                 speculation_enabled=True):
        if batch.shard.world_size != 1 or batch.verification_width != 1:
            raise ValueError("reference generation requires one GPU and ordinary token consumption")
        if batch.requests.vocab_size != model.cfg.vocab_size:
            raise ValueError("reference generation vocabulary differs from request storage")
        if type(speculation_enabled) is not bool:
            raise TypeError("speculation_enabled must be a boolean")
        self.model, self.batch, self.history = model, batch, history
        self.speculation_enabled = speculation_enabled
        self.vision = vision if batch.mode == "prefill" else None
        if self.vision is not None and (
                vision.requests is not batch.requests or vision.inputs is not batch.inputs
                or vision.token_base != model.cfg.vocab_size
                or batch.input_vocabulary != vision.token_base + 4
                or batch.validate_prefill != vision.validate_prefill):
            raise ValueError("reference image generation requires validated shared image storage")
        self.pool = batch.pool
        self.caches = self.pool.cache_views()
        self.metadata = (
            ReferencePrefillMetadata(model.cfg, self.pool.geometry,
                yoco_keep_tokens=getattr(model, "yoco_keep_tokens", None))
            if batch.mode == "prefill"
            else ReferenceDecodeMetadata(model.cfg, self.pool.geometry))
        example = torch.empty((batch.rows, model.cfg.vocab_size), dtype=torch.float32,
                              device=batch.requests.generation.device)
        b, rows = batch.inputs, batch.rows
        self.sampling = PreparedSampling(example, batch.requests, b.row_slot_ids[:rows],
            b.row_generations[:rows], b.live_rows, completion, frame, sequence,
            sample_enabled=b.sample_enabled[:rows],
            draft_context_lag=0 if self.speculation_enabled else None)
        self.sampling.prepare()
        self.sampling.functional(session=session)
        batch.functional()

    def __call__(self):
        self._run()

    def observe(self):
        """Expose forward values to diagnostic comparisons before they expire."""
        return self._run(observe=True)

    def _run(self, *, observe=False):
        from deepinfer.model_executor.layers.sampler_utils import get_logits

        validation, present = self.batch.validate()
        packed = self.batch.begin(validation, present=present)
        images = ((packed.token_ids >= self.vision.token_base) & (packed.positions >= 0)
                  if self.vision is not None else None)
        prepared = self.history(packed, images_seq_mask=images)
        metadata = self.metadata(packed, ngram_token_ids=prepared.ngrams, images_seq_mask=images)
        embedding = None
        if self.vision is not None:
            embedding = self.model.model.embed_tokens(metadata.input_ids)
            self.vision.inject(packed, embedding)
        target = self.model(self.caches, metadata, hidden_states=embedding,
                            speculation_enabled=self.speculation_enabled).hidden_states
        if metadata.yoco_compaction_enabled:
            kept = packed.lengths[0].clamp(0, target.shape[0])
            clip_prefix = (packed.lengths[0] - kept).clamp_min(0)
            sample_indices = (packed.sample_indices - clip_prefix).clamp(0, target.shape[0] - 1)
        else:
            sample_indices = packed.sample_indices
        hidden = target[sample_indices]
        logits = get_logits(hidden, self.model.lm_head.weight, self.model.cfg.vocab_size)
        live = packed.lengths > 0
        save_hidden(hidden, self.pool.reserved_region("last_hidden"), packed.state_rows, live)
        self.history.commit(packed, packed.lengths, prepared)
        self.batch.finish(packed)
        save_hidden((packed.starts + packed.lengths)[:, None], self.pool.reserved_region("consumed_tokens"),
                    packed.state_rows, live)
        self.sampling.run(logits)
        if observe:
            return hidden, logits
