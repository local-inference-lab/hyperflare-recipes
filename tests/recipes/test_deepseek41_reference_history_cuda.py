# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured request-local Engram history against the reference input builder."""

import os

import pytest
import torch

from hyperflare_recipes.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from hyperflare_recipes.recipes.deepseek41.reference_engram import ReferenceEngramHistory
from tests.recipes.test_deepseek41_reference_cache import reference_config

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def native_ngrams(histories, inputs, lookup, pad):
    from deepinfer._C.engine import PrepareArgs, SequenceData, SequenceMetadata, prepare_prompt_v4
    from deepinfer.sampling_params import SamplingParams

    sequences = []
    for index, (history, tokens) in enumerate(zip(histories, inputs, strict=True)):
        full = history + tokens
        mapped = [lookup[token] if 0 <= token < len(lookup) else -1 for token in full]
        raw, compressed = SequenceData(full), SequenceData(mapped)
        pages = list(range((len(full) + 127) // 128))
        sequences.append(SequenceMetadata(str(index), index, True, raw, compressed,
            None, SamplingParams(), pages, len(tokens), len(full), 0, [], [], state_id=index))
    args = PrepareArgs()
    args.seq_metadata_list = sequences
    args.is_prompt = True
    args.block_size = 128
    args.window_size = 128
    args.swa_block_size = 135
    args.max_ngram_size = 4
    args.engram_compressed_pad_token = pad
    args.image_sentinel = -1
    return prepare_prompt_v4(args)["ngram_token_ids"]


@pytest.mark.parametrize("batch,width", [(4, 1), (4, 6), (4, 8), (1, 129), (1, 8192)])
@torch.no_grad()
def test_reference_history_capture_acceptance_and_restore(reference_config, batch, width):
    capacity = batch * width + 3
    backing = torch.full((8, 32), 73, dtype=torch.int64, device="cuda")
    storage = backing[:, 7:10]
    binding = ReferenceEngramHistory.from_config(storage, reference_config)
    lookup = binding.token_map.cpu().tolist()
    source = Metadata.allocate(capacity, batch, 256, "cuda", max_query_tokens=width)
    starts = torch.zeros(batch, dtype=torch.int64, device="cuda")
    lengths = torch.full((batch,), width, dtype=torch.int32, device="cuda")
    states = torch.tensor([4, 1, 6, 2][:batch], device="cuda")
    pages = torch.zeros(batch, 256, dtype=torch.int32, device="cuda")
    ids = torch.zeros(batch, width, dtype=torch.int64, device="cuda")
    live = torch.tensor([batch], dtype=torch.int32, device="cuda")
    images = torch.zeros(capacity, dtype=torch.bool, device="cuda")
    counts = lengths.clone()
    output = torch.empty(capacity, 4, dtype=torch.int32, device="cuda")

    def run():
        prepared = binding(source, images_seq_mask=images)
        output.copy_(prepared.ngrams)
        binding.commit(source, counts, prepared)

    source.set(starts, lengths, states, pages, ids, live)
    compiled = compile_forward(run)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    backing.fill_(73)
    histories = [[] for _ in range(batch)]
    snapshot = backing.clone()
    try:
        for step in range(6):
            if step == 4:
                backing.copy_(snapshot)
                histories = [value.copy() for value in snapshot_histories]
            request_lengths = [width if step % 2 == 0 else max(1, width - index)
                               for index in range(batch)]
            if batch > 1:
                request_lengths[(step + 1) % batch] = 0
            if step == 5:
                request_lengths = [0] * batch
            offered = []
            masks = []
            for index, length in enumerate(request_lengths):
                tokens = [(1000 + step * 173 + index * 311 + token) % len(lookup)
                          for token in range(width)]
                ids[index].copy_(torch.tensor(tokens, device="cuda"))
                tokens = tokens[:length]
                mask = [False] * length
                if length and step in (0, 2, 4):
                    mask[-1] = True
                    tokens[-1] = -1
                if length > 3 and step == 1:
                    mask[1] = True
                    tokens[1] = -1
                offered.append(tokens)
                masks.extend(mask)
            lengths.copy_(torch.tensor(request_lengths, dtype=torch.int32, device="cuda"))
            starts.copy_(torch.tensor([len(value) for value in histories], device="cuda"))
            source.set(starts, lengths, states, pages, ids, live)
            images.zero_()
            images[:len(masks)].copy_(torch.tensor(masks, dtype=torch.bool, device="cuda"))
            # Full, partial and zero acceptance use the same captured graph.
            accepted = [length if step in (0, 4) else (length // 2 if step in (1, 3) else 0)
                        for length in request_lengths]
            counts.copy_(torch.tensor(accepted, dtype=torch.int32, device="cuda"))
            expected_ngrams = native_ngrams(histories, offered, lookup, binding.pad)
            expected_state = backing.cpu().clone()
            for index, (tokens, count) in enumerate(zip(offered, accepted, strict=True)):
                slot = int(states[index])
                if count:
                    tail = expected_state[slot, 7:10].tolist() + tokens[:count]
                    expected_state[slot, 7:10] = torch.tensor(tail[-3:])
                histories[index].extend(tokens[:count])
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
            assert torch.equal(output[:len(masks)].cpu(), expected_ngrams)
            assert (output[len(masks):] == binding.pad).all()
            assert torch.equal(backing.cpu(), expected_state)
            if step == 1:
                snapshot.copy_(backing)
                snapshot_histories = [value.copy() for value in histories]
    finally:
        torch.cuda.synchronize()
        graph.reset()
