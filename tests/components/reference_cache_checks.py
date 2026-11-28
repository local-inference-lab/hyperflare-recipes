# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare speculative cache transactions with accepted-prefix model execution."""

from dataclasses import replace

import torch

from hyperflare_recipes.components.paged_batch import Metadata
from ignition.components.state import save_hidden


@torch.no_grad()
def check_reference_cache_prefixes(operation, candidates):
    """Use the native-delivered rows for these candidates while its worker is idle.

    The caller restores the request state preceding the native round. Every
    candidate's Engram rows must still occupy the delivery buffers. Prefix runs
    retain the full physical row capacity, masking the rejected suffix, so this
    checks cache transactions without changing the arithmetic shape.
    """
    pool, history, batch = operation.pool, operation.history, operation.batch
    tensors = (pool.backing, *pool.pages.values())
    initial = tuple(value.clone() for value in tensors)

    def restore(values):
        for value, saved in zip(tensors, values, strict=True):
            value.copy_(saved)

    validation, present = batch.validate()
    first = batch.begin(validation, present=present)
    assert bool((first.lengths == 1).all())

    def pack(counts):
        return Metadata.pack(first.starts, counts, first.state_rows, first.page_table,
            candidates, batch.inputs.live_rows, token_capacity=candidates.numel(),
            sample_count=candidates.numel(), max_query_tokens=operation.width)

    def target(metadata):
        prepared = history(metadata)
        hidden = operation.model(operation.caches,
            operation.metadata(metadata, ngram_token_ids=prepared.ngrams)).hidden_states
        return prepared, hidden

    def finish(metadata, counts, prepared, hidden):
        history.commit(metadata, counts, prepared)
        indices = (metadata.query_start_loc[:-1] + counts - 1).long().clamp_min(0)
        save_hidden(hidden[indices], pool.last_hidden, metadata.state_rows, counts > 0)
        save_hidden((metadata.starts + counts)[:, None], pool.region("consumed_tokens"),
                    metadata.state_rows, counts > 0)

    try:
        full = pack(torch.full_like(first.lengths, operation.width))
        snapshot = operation.transaction.begin(full)
        previous = replace(first, positions=first.positions - 1)
        draft = operation.metadata(previous).build_dflash_input(
            first.query_start_loc[:-1], torch.zeros_like(first.lengths),
            operation.model.draft.dflash_block_size, operation.model.draft.window_size,
            swa_block_size=pool.geometry.window_tokens)
        operation.model.draft.forward_drafts(first.token_ids, draft, operation.caches)
        prepared, hidden = target(full)
        speculative = tuple(value.clone() for value in tensors)
        for count in range(operation.width + 1):
            counts = torch.full_like(first.lengths, count)
            restore(speculative)
            operation.transaction.commit(full, counts, snapshot)
            finish(full, counts, prepared, hidden)
            actual = tuple(value.clone() for value in tensors)

            restore(initial)
            prefix = pack(counts)
            prefix_history, prefix_hidden = target(prefix)
            finish(prefix, counts, prefix_history, prefix_hidden)
            for index, (value, saved) in enumerate(zip(tensors, actual, strict=True)):
                torch.testing.assert_close(value, saved, rtol=0, atol=0,
                    msg=lambda message: f"accepted inputs={count}, cache plane={index}: {message}")
    finally:
        restore(initial)
        torch.cuda.synchronize()
