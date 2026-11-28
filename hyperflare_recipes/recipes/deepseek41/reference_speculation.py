# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture reference draft computation with shared stochastic verification."""

from dataclasses import replace

import torch

from ignition.components.flashinfer_sampling import sample_probabilities
from hyperflare_recipes.components.causal_prefix import PreparedCausalPrefixAllocation
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit, PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from ignition.components.state import save_hidden
from .reference_cache import ReferenceCacheTransaction
from .reference_metadata import ReferenceDecodeMetadata


class ReferenceSpeculation:
    """One complete proposal, target verification and accepted-state transaction.

    Proposal and verification RNG storage is shared across prepared widths.
    Original DFlash attention, projections and Markov logits produce the draft
    scores. Shared FlashInfer operations own sampling and chain acceptance;
    recipe computation commits model cache and Engram history by accepted count.
    Width-one continuations execute target verification without draft proposals.
    """

    def __init__(self, model, batch, history, completion, frame, sequence, *,
                 rng, proposal_generation, proposal_offset, session,
                 prefix_limits=None, confidence_output=None, target_token_capacity=None,
                 transaction=None, workspace=None):
        from hyperflare_recipes.components.markov_sampling import PreparedMarkovProposals
        from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection

        width, rows = batch.verification_width, batch.rows
        if (batch.shard.world_size != 1 or batch.mode != "decode" or width < 1
                or (width > 1 and (model.draft is None or width > model.draft.max_MTP_value + 1))
                or width > batch.pool.geometry.verification_width):
            raise ValueError("reference speculation requires one GPU and a prepared draft/verification width")
        state = batch.requests
        for value in (proposal_generation, proposal_offset):
            if (value.shape != state.generation.shape or value.dtype != torch.int64
                    or value.device != state.generation.device):
                raise ValueError("proposal RNG storage must match physical request slots")
        self.model, self.batch, self.history = model, batch, history
        self.width, self.rows = width, rows
        self.pool, self.caches = batch.pool, batch.pool.cache_views()
        self.metadata = ReferenceDecodeMetadata(model.cfg, self.pool.geometry)
        if width > 1:
            if transaction is not None and transaction.pool is not self.pool:
                raise ValueError("reference cache transaction must bind the batch continuation pool")
            self.transaction = ReferenceCacheTransaction(self.pool) if transaction is None else transaction
        else:
            self.transaction = None
        self.rng, self.proposal_generation, self.proposal_offset = rng, proposal_generation, proposal_offset
        b, device, vocab = batch.inputs, state.generation.device, state.vocab_size
        self.prefix_limits, self.confidence_output, self.frame = prefix_limits, confidence_output, frame
        self.target_tokens = rows * width if target_token_capacity is None else target_token_capacity
        if (type(self.target_tokens) is not int or not rows <= self.target_tokens <= rows * width
                or batch.target_token_capacity != rows * width):
            raise ValueError("reference proposals require full prefix admission and a bounded target capacity")
        if prefix_limits is None:
            if confidence_output is not None or self.target_tokens != rows * width:
                raise ValueError("packed reference verification requires prefix limits and confidence storage")
            self.allocator = None
        else:
            if (width < 2 or prefix_limits.shape != (state.capacity,) or prefix_limits.dtype != torch.int32
                    or prefix_limits.device != device or not prefix_limits.is_contiguous()
                    or confidence_output is None or confidence_output.ndim != 3
                    or confidence_output.shape[:2] != (2, state.capacity)
                    or confidence_output.shape[2] < width - 1
                    or confidence_output.dtype != torch.float32 or confidence_output.device != device
                    or not confidence_output.is_contiguous()):
                raise ValueError("reference prefix limits and two confidence frames must match request storage")
            begin, query = prefix_limits.data_ptr(), b.query_lengths.data_ptr()
            if begin < query + b.query_lengths.nbytes and query < begin + prefix_limits.nbytes:
                raise ValueError("host prefix limits must not overlap device-owned query lengths")
            confidence = torch.empty((rows, width - 1), dtype=torch.float32, device=device)
            slots = torch.empty(rows, dtype=torch.int64, device=device)
            submitted, valid, admitted = (torch.empty(rows, dtype=torch.bool, device=device) for _ in range(3))
            self.allocator = PreparedCausalPrefixAllocation(confidence, slots, submitted, valid,
                prefix_limits[:rows], b.query_lengths[:rows], admitted,
                token_capacity=self.target_tokens, request_capacity=state.capacity).functional()
        args = state, b.row_slot_ids[:rows], b.row_generations[:rows], b.live_rows
        logits = torch.zeros((rows, width, vocab), device=device)
        drafts = torch.zeros((rows, width - 1), device=device, dtype=torch.int64)
        candidates = torch.zeros((rows, width), device=device, dtype=torch.int64)
        depth = width - 1
        draft_head = getattr(model, "draft_head", None)
        markov_head = getattr(model, "markov_head", None)
        if depth and draft_head is not None:
            draft_source = torch.empty((rows * depth, model.cfg.hidden_size), dtype=torch.bfloat16, device=device)
            draft_out = torch.empty((rows * depth, vocab), dtype=torch.bfloat16, device=device)
            self.draft_head_proj = PreparedVocabularyProjection(
                draft_source, draft_head, draft_out, draft_out, runtime=None, session=session,
                coordinator=None, name=f"reference.draft_head.{rows}x{depth}.t{self.target_tokens}",
                workspace=workspace, gather=False).functional(session=session)
        else:
            self.draft_head_proj = None
        use_fused_markov = (depth > 0 and model.draft is not None and model.draft.enable_markov
                            and model.draft.markov_rank % 64 == 0)
        if depth and markov_head is not None and not use_fused_markov:
            raise ValueError("FP8 Markov head requires an enabled 64-aligned Markov head")
        probabilities = torch.zeros((rows, depth, vocab), device=device)
        if use_fused_markov:
            base_dtype = torch.bfloat16 if draft_head is not None else torch.float32
            base_example = torch.zeros((rows, depth, vocab), dtype=base_dtype, device=device)
            pending_example = torch.zeros((rows,), dtype=torch.int64, device=device)
            valid_example = torch.zeros((rows, depth), dtype=torch.bool, device=device)
            seed_example = torch.zeros((rows,), dtype=torch.int64, device=device)
            offset_example = torch.zeros((rows,), dtype=torch.int64, device=device)
            resolved_markov = model.draft.markov_w2.weight if markov_head is None else markov_head
            self.markov_proposals = PreparedMarkovProposals(
                base_example, model.draft.markov_w1.weight, resolved_markov, pending_example,
                drafts, probabilities, valid_example, state, b.row_slot_ids[:rows],
                b.row_generations[:rows], b.live_rows, seed_example, offset_example, workspace=None)
            self.markov_proposals.prepare()
            self.markov_proposals.functional(session=session)
            self.proposals = ()
        else:
            self.markov_proposals = None
            self.proposals = tuple(PreparedSpeculativeDistributions(logits[:, :1].contiguous(),
                *args, drafts, prefix_offset=index) for index in range(depth))
        self.target = PreparedSpeculativeDistributions(logits, *args, drafts)
        for operation in (*self.proposals, self.target):
            operation.prepare()
        self.verification = PreparedStochasticVerification(probabilities, drafts,
            self.target.probabilities, *args, rng=rng, target_valid=self.target.valid,
            token_limits=b.query_lengths[:rows])
        self.verification.prepare()
        self.commit = PreparedSpeculativeCommit(state, self.verification.output, candidates,
            *args[1:], completion, frame, sequence, logits=logits, rng=rng,
            draft_context_lag=0, token_limits=b.query_lengths[:rows])
        self.commit.prepare()
        for operation in (*self.proposals, self.target, self.verification, self.commit):
            operation.functional(session=session)
        batch.functional()

    @staticmethod
    def _scratch(operation, device):
        return {name: torch.empty(shape, dtype=dtype, device=device)
                for name, (shape, dtype) in operation.layout.items()}

    def __call__(self):
        self._run()

    def observe(self):
        """Return transient proposal and target values for qualification."""
        return self._run(observe=True)

    def _run(self, *, observe=False):
        from deepinfer.model_executor.layers.sampler_utils import get_logits

        b, state, width = self.batch.inputs, self.batch.requests, self.width
        depth = width - 1
        if self.prefix_limits is not None:
            b.query_lengths[:self.rows].copy_(self.prefix_limits[:self.rows])
        validation, present = self.batch.validate()
        first = self.batch.begin(validation, present=present)
        active = first.lengths > 0
        slots = b.row_slot_ids[:self.rows].long()
        safe = slots.clamp(0, state.capacity - 1)
        limits = torch.where(active, b.query_lengths[:self.rows], 0)
        pending = first.token_ids[first.sample_indices]
        candidates = torch.zeros((self.rows, width), dtype=torch.int64, device=pending.device)
        candidates[:, 0].copy_(pending)
        drafts = torch.zeros((self.rows, depth), dtype=torch.int64, device=pending.device)
        offset = torch.where(self.proposal_generation[safe] == state.generation[safe], self.proposal_offset[safe], 0)
        if depth:
            offered = Metadata.pack(first.starts, limits, first.state_rows, first.page_table,
                candidates, b.live_rows, token_capacity=self.rows * width,
                sample_count=self.rows * width, max_query_tokens=width)
            saved = self.transaction.begin(offered)
            previous = replace(first, positions=torch.where(first.positions >= 0, first.positions - 1, -1))
            reference = self.metadata(previous)
            first_rows = torch.where(active & (first.starts > 0),
                first.query_start_loc[:-1], -1).int()
            draft_metadata = reference.build_dflash_input(first_rows, torch.zeros_like(first_rows),
                depth, self.model.draft.window_size, swa_block_size=self.pool.geometry.window_tokens)
            draft_hidden = self.model.draft.forward_drafts(
                torch.where(active, pending, 0), draft_metadata, self.caches, block_size=depth)
            normed_draft = self.model.draft.post_norm(draft_hidden)
            if self.draft_head_proj is not None:
                base_logits = self.draft_head_proj(normed_draft).view(self.rows, depth, -1)
            else:
                base_logits = get_logits(normed_draft, self.model.lm_head.weight,
                                         self.model.cfg.vocab_size).view(self.rows, depth, -1)
            probabilities = torch.empty((self.rows, depth, state.vocab_size), dtype=torch.float32, device=pending.device)
            valid = torch.empty((self.rows, depth), dtype=torch.bool, device=pending.device)
            confidences = []
            seed = state.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
            prior = torch.where(active, pending, 0)
            if self.markov_proposals is not None:
                self.markov_proposals.run(base_logits, prior, drafts, probabilities, valid, seed, offset)
                valid = valid & active[:, None] & (first.starts > 0)[:, None]
                if self.allocator is not None:
                    priors = torch.cat((prior[:, None], drafts[:, :-1]), dim=1) if depth > 1 else prior[:, None]
                    markov_embs = self.model.draft.markov_w1(priors.reshape(-1))
                    temps = self.model.draft.confidence_head.temperature[:depth]
                    if len(set(temps)) == 1:
                        confidence = self.model.draft.confidence_head(
                            draft_hidden.contiguous(), markov_embs, step_i=0).view(self.rows, depth)
                    else:
                        markov_3d = markov_embs.view(self.rows, depth, -1)
                        draft_hidden_3d = draft_hidden.view(self.rows, depth, -1)
                        for index in range(depth):
                            confidences.append(self.model.draft.confidence_head(
                                draft_hidden_3d[:, index].contiguous(),
                                markov_3d[:, index].contiguous(), step_i=index))
                        confidence = torch.stack(confidences, 1)
            else:
                sampled = torch.empty(self.rows, dtype=torch.int32, device=pending.device)
                row_valid = torch.empty(self.rows, dtype=torch.bool, device=pending.device)
                for index, sampler in enumerate(self.proposals):
                    scores = base_logits[:, index].float().clone()
                    markov = None
                    if self.model.draft.enable_markov:
                        markov = self.model.draft.markov_w1(prior)
                        scores = self.model.draft.markov_w2(markov, out_acc=scores)
                    if self.allocator is not None:
                        confidences.append(self.model.draft.confidence_head(
                            draft_hidden.view(self.rows, depth, -1)[:, index],
                            markov, step_i=index))
                    scratch = self._scratch(sampler, scores.device)
                    sampler.run(scratch, scores[:, None].contiguous(), candidates=drafts)
                    for row in range(self.rows):
                        token, ok = sample_probabilities(scratch["probabilities"][row],
                            seed[row:row + 1], offset[row:row + 1] + 4 * index, return_valid=True)
                        sampled[row:row + 1].copy_(token)
                        row_valid[row:row + 1].copy_(ok)
                    prior = sampled.long()
                    drafts[:, index].copy_(prior)
                    probabilities[:, index].copy_(scratch["probabilities"][:, 0])
                    valid[:, index].copy_(scratch["valid"][:, 0] & row_valid & active & (first.starts > 0))
            candidates[:, 1:].copy_(drafts)
            if self.allocator is not None:
                finished = ((slots >= 0) & (slots < state.capacity)
                    & (b.row_generations[:self.rows] == state.generation[safe]) & (state.status[safe] == 2))
                submitted = (torch.arange(self.rows, device=pending.device) < b.live_rows) & ~finished
                if self.markov_proposals is None:
                    confidence = torch.stack(confidences, 1)
                source_valid = active & valid.all(1)
                confidence = confidence.masked_fill(~submitted[:, None], 0.)
                self.confidence_output.index_copy_(0, self.frame.long(), torch.nn.functional.pad(confidence,
                    (0, self.confidence_output.shape[2] - depth, 0, state.capacity - self.rows)).unsqueeze(0))
                admitted = self.allocator(confidence, slots, submitted, source_valid,
                    self.prefix_limits[:self.rows], b.query_lengths[:self.rows])
                new_lengths = torch.where(admitted, b.query_lengths[:self.rows], 0)
                b.query_lengths[:self.rows].copy_(new_lengths)
                # Prefix limits, pages, state rows, duplicates, and pending tokens were already validated
                # above; the only newly rejected rows are active rows that prefix allocation failed to admit.
                rejected_row = active & (~admitted | (new_lengths <= 0) | (new_lengths > width))
                slot_ids = torch.arange(state.capacity, device=pending.device)
                bad_slots = present & ((slot_ids[:, None] == slots[None, :]) & rejected_row[None, :]).any(1)
                state.status.copy_(torch.where(bad_slots, 2, state.status))
                state.error_code.copy_(torch.where(bad_slots, 32, state.error_code))
                limits = torch.where(admitted & active, new_lengths, 0)
        else:
            offered = saved = None
            base_logits = torch.zeros((self.rows, 0, state.vocab_size), dtype=torch.float32, device=pending.device)
            probabilities = torch.zeros((self.rows, 0, state.vocab_size), dtype=torch.float32, device=pending.device)
            valid = torch.ones((self.rows, 0), dtype=torch.bool, device=pending.device)
        packed = Metadata.pack(first.starts, limits, first.state_rows, first.page_table,
            candidates, b.live_rows, token_capacity=self.target_tokens,
            sample_count=self.rows * width, max_query_tokens=width)
        prepared = self.history(packed)
        target_hidden = self.model(self.caches, self.metadata(packed, ngram_token_ids=prepared.ngrams)).hidden_states
        sampled_hidden = target_hidden[packed.sample_indices].view(self.rows, width, -1)
        if self.target_tokens < self.rows * width:
            target_logits = get_logits(target_hidden, self.model.lm_head.weight, self.model.cfg.vocab_size)
            logits = target_logits[packed.sample_indices].view(self.rows, width, -1)
        else:
            logits = get_logits(sampled_hidden.flatten(0, 1), self.model.lm_head.weight, self.model.cfg.vocab_size).view(self.rows, width, -1)
        target = self._scratch(self.target, logits.device)
        self.target.run(target, logits, candidates=drafts)
        acceptance = self.verification.run(self._scratch(self.verification, logits.device),
            probabilities, target["probabilities"], target["valid"],
            drafts=drafts, draft_valid=valid, disjoint=True)
        committed = self.commit(logits, acceptance=acceptance, inputs=candidates)
        counts = committed.emitted_count
        if self.transaction is not None:
            self.transaction.commit(offered, counts, saved)
        self.history.commit(packed, counts, prepared)
        hidden = sampled_hidden[torch.arange(self.rows, device=logits.device),
                                                       (counts - 1).long().clamp_min(0)]
        save_hidden(hidden, self.pool.reserved_region("last_hidden"), packed.state_rows, counts > 0)
        save_hidden((packed.starts + counts)[:, None], self.pool.reserved_region("consumed_tokens"),
                    packed.state_rows, counts > 0)
        if depth:
            physical = torch.arange(state.capacity, device=logits.device)
            selected = active[None, :] & (physical[:, None] == slots[None, :])
            next_offset = torch.where(selected, offset[None, :] + 4 * depth, 0).sum(1)
            self.proposal_offset.copy_(torch.where(selected.any(1), next_offset, self.proposal_offset))
            self.proposal_generation.copy_(torch.where(selected.any(1), state.generation, self.proposal_generation))
        if observe:
            return candidates, logits, committed, base_logits, probabilities, target["probabilities"]
