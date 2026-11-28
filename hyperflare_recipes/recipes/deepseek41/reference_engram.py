# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind reference Engram consumers to Helios's single-device row delivery."""

from dataclasses import dataclass
from functools import partial

import torch

from hyperflare_recipes.components.delivered_fp8_projection import DeliveredFP8Projection, DeliveredFP8Rows
from ignition.components.ngram_delivery import publish_requests
from ignition.components.readiness import ReadinessTimings
from ignition.components.state import save_hidden
from ignition.preparation.runtime_core import FileSource
from ignition.preparation.tensor_metadata import record_numeric_tensor


def _rows(inputs):
    return inputs


@dataclass(frozen=True)
class ReferenceEngramTokens:
    ngrams: torch.Tensor
    prior_history: torch.Tensor
    raw_tokens: torch.Tensor


class ReferenceEngramHistory:
    """Prepare reference n-grams and commit only consumed request tokens.

    History rows belong to native continuation storage. Admission validates
    packed inputs and unique active state rows; sampling supplies counts within
    each request's input length. Each invocation snapshots history before commit,
    so rejected speculative tokens never become persistent context.
    """

    def __init__(self, storage, token_map, *, pad_token_id, max_ngram_size):
        values = tuple(token_map)
        if (storage.ndim != 2 or storage.dtype != torch.int64 or storage.shape[0] < 2
                or type(max_ngram_size) is not int or max_ngram_size < 2
                or storage.shape[1] != max_ngram_size - 1
                or type(pad_token_id) is not int or not 0 <= pad_token_id < len(values)
                or any(type(value) is not int or not 0 <= value <= torch.iinfo(torch.int32).max
                       for value in values)):
            raise ValueError("reference Engram history requires matching continuation and vocabulary geometry")
        self.storage = storage
        self.width = max_ngram_size - 1
        self.pad = values[pad_token_id]
        self.token_map = torch.tensor(values, dtype=torch.int32, device=storage.device)
        record_numeric_tensor(self.token_map)

    @classmethod
    def from_config(cls, storage, config):
        from deepinfer.model_executor.models.v4_modules.engram_metadata import CompressedVocabulary

        cfg = config.engram
        vocabulary = CompressedVocabulary(cfg.tokenizer_name_or_path)
        if len(vocabulary.old2new) != cfg.tokenizer_vocab_size:
            raise ValueError("reference Engram tokenizer differs from the checkpoint vocabulary")
        values = vocabulary.old2new if cfg.enable_tokenizer_compression else range(cfg.tokenizer_vocab_size)
        pad = vocabulary.tokenizer.token_to_id("<｜▁pad▁｜>")
        return cls(storage, values, pad_token_id=pad, max_ngram_size=cfg.engram_max_ngram_size)

    def __call__(self, metadata, *, images_seq_mask=None):
        m = metadata
        if m.token_ids.device != self.storage.device:
            raise ValueError("reference Engram inputs and history must share a device")
        if images_seq_mask is not None and (
                images_seq_mask.shape != m.token_ids.shape or images_seq_mask.dtype != torch.bool
                or images_seq_mask.device != m.token_ids.device):
            raise ValueError("reference Engram image mask must match the packed tokens")
        raw = m.token_ids if images_seq_mask is None else torch.where(images_seq_mask, -1, m.token_ids)
        prior = self.storage[m.state_rows]
        row = torch.arange(raw.numel(), device=raw.device)[:, None]
        lookback = torch.arange(self.width + 1, device=raw.device)[None, :]
        relative = row - m.query_start_loc[m.requests, None] - lookback
        indices = (row - lookback).clamp(0, raw.numel() - 1)
        history = prior[m.requests].gather(1, (relative + self.width).clamp(0, self.width - 1))
        tokens = torch.where(relative >= 0, raw[indices], history)
        exists = (m.positions[:, None] >= lookback) & (m.positions[:, None] >= 0)
        text = (tokens >= 0) & (tokens < self.token_map.numel())
        image_boundary = ((~text) & exists).int().cumsum(1) > 0
        mapped = self.token_map[tokens.clamp(0, self.token_map.numel() - 1)]
        ngrams = torch.where(exists & ~image_boundary, mapped, self.pad)
        return ReferenceEngramTokens(ngrams, prior, raw)

    def commit(self, metadata, counts, prepared):
        """Append accepted input prefixes; zero-count and inactive rows retain bytes."""
        m = metadata
        if (counts.shape != m.lengths.shape or counts.dtype not in (torch.int32, torch.int64)
                or counts.device != m.token_ids.device):
            raise ValueError("reference Engram commit requires one integer count per request")
        columns = torch.arange(self.width, device=m.token_ids.device)
        relative = counts[:, None].long() + columns - self.width
        indices = (m.query_start_loc[:-1, None] + relative).clamp(0, m.token_ids.numel() - 1)
        history = torch.where(relative >= 0, prepared.raw_tokens[indices],
            prepared.prior_history.gather(1, (relative + self.width).clamp(0, self.width - 1)))
        save_hidden(history, self.storage, m.state_rows, counts > 0)


class ReferenceEngramDelivery:
    """Own compact FP8 planes, mapped request mailboxes and readiness events.

    The caller retains this owner through native drain. ``fetch`` executes only
    during preparation and capture; native graph replay publishes the requests.
    Table files are bound by the native resident-table owner or core exporter.
    This binding has no DeepEP allocation or distributed initialization.
    """

    def __init__(self, sources, *, vocabulary, heads, width, capacity, sequence, frame, host_allocator,
                 token_map=None, pad_token_id=2, compressed_vocabulary=None, table_hashes=None):
        if (not sources or len(sources) > 64
                or vocabulary <= 0 or heads <= 0 or width <= 0 or width % 128 or capacity <= 0
                or sequence.shape != (1,) or sequence.dtype != torch.int64 or not sequence.is_cuda
                or sequence.device != frame.device):
            raise ValueError("reference Engram delivery requires table geometry and a CUDA submission sequence")
        for weights, scales in sources.values():
            if weights.nbytes <= 0 or weights.nbytes % width or scales.nbytes * 32 != weights.nbytes:
                raise ValueError("reference Engram planes must contain matching FP8 and E8M0 rows")
        self.sources = dict(sources)
        self.vocabulary, self.heads, self.width, self.capacity = vocabulary, heads, width, capacity
        self.token_map = None if token_map is None else tuple(int(v) for v in token_map)
        self.pad_token_id = int(pad_token_id)
        self.compressed_vocabulary = (None if compressed_vocabulary is None else int(compressed_vocabulary))
        self.table_hashes = None if table_hashes is None else dict(table_hashes)
        self.sequence = sequence
        self.timings = ReadinessTimings(frame, len(sources))
        self.planes, self.mailboxes, self.events = {}, {}, {}
        for layer in sources:
            device = sequence.device
            weights = torch.zeros((capacity * heads, width), dtype=torch.uint8, device=device)
            scales = torch.zeros((capacity * heads, width // 32), dtype=torch.uint8, device=device)
            hashes = torch.full((capacity, heads), -1, dtype=torch.int64, device=device)
            self.planes[layer] = weights, scales, hashes
            mailbox = host_allocator(f"reference_engram_{layer}", (capacity * heads + 1,), torch.int64)
            mailbox.host_view.zero_()
            self.mailboxes[layer] = mailbox
            event = torch.cuda.Event(external=True)
            event.record(torch.cuda.current_stream(device))
            self.events[layer] = event

    @classmethod
    def export_hash_metadata(cls, config):
        """Extract canonical tokenizer compression and XOR-modulo hash tables on CPU."""
        from deepinfer.model_executor.models.v4_modules.engram_metadata import (
            CompressedVocabulary, NgramHashMapping,
        )

        cfg = config.engram
        vocab = CompressedVocabulary(cfg.tokenizer_name_or_path)
        if len(vocab.old2new) != cfg.tokenizer_vocab_size:
            raise ValueError("reference Engram tokenizer differs from the checkpoint vocabulary")
        token_map = vocab.old2new if cfg.enable_tokenizer_compression else range(cfg.tokenizer_vocab_size)
        compressed_vocab = len(vocab) if cfg.enable_tokenizer_compression else int(cfg.tokenizer_vocab_size)
        pad_id = vocab.tokenizer.token_to_id("<｜▁pad▁｜>")
        mapping = NgramHashMapping(cfg, device="cpu")
        orders = [order for order, count in zip(range(2, int(cfg.engram_max_ngram_size) + 1),
                                                cfg.n_head_per_ngram, strict=True)
                  for _ in range(int(count))]
        table_hashes = {}
        for ordinal, layer in enumerate(mapping.layer_ids):
            table_hashes[int(layer)] = dict(
                method="XOR_MODULO",
                orders=orders,
                multipliers=mapping.ngram_multipliers[ordinal].tolist(),
                primes=mapping.ngram_vocab_sizes[ordinal].flatten().tolist(),
                offsets=mapping.ngram_offsets[ordinal].flatten().tolist(),
            )
        return dict(token_map=tuple(token_map), pad_token_id=int(pad_id),
                    compressed_vocabulary=int(compressed_vocab), table_hashes=table_hashes)

    @classmethod
    def from_checkpoint(cls, checkpoint, config, **kwargs):
        cfg = config.engram
        heads = sum(cfg.n_head_per_ngram)
        widths = {dim // count for dim, count in zip(cfg.n_embed_per_ngram, cfg.n_head_per_ngram, strict=True)}
        if len(widths) != 1:
            raise ValueError("reference Engram delivery requires equal head widths")
        width = widths.pop()
        sources = {}
        for layer in cfg.engram_layer_ids:
            pair = []
            rows = None
            for suffix, columns, dtype in (("weight", width, "F8_E4M3"), ("scale", width // 32, "F8_E8M0")):
                tensor = checkpoint.source.tensors[f"layers.{layer}.engram.embed.{suffix}"]
                if (len(tensor.shape) != 2 or tensor.shape[1] != columns or tensor.dtype != dtype
                        or (rows is not None and tensor.shape[0] != rows)):
                    raise ValueError("reference Engram checkpoint planes differ from configured geometry")
                rows = tensor.shape[0]
                pair.append(FileSource(checkpoint.source.snapshot / tensor.shard, tensor.offset, rows * columns))
            sources[layer] = tuple(pair)
        metadata = cls.export_hash_metadata(config)
        return cls(sources, vocabulary=cfg.tokenizer_vocab_size, heads=heads, width=width, **metadata, **kwargs)

    def bind(self, layers):
        layers = tuple(layers)
        if tuple(layer.layer_id for layer in layers) != tuple(self.sources):
            raise ValueError("reference Engram consumers must match native table order")
        replacements = []
        for interval, layer in enumerate(layers):
            projection = layer.fused_kv_projection
            if projection.weight.shape[1] != self.heads * self.width:
                raise ValueError("reference Engram projection does not match delivered head geometry")
            replacements.append(DeliveredFP8Projection((projection.weight, projection.weight.sf),
                ready=self.events[layer.layer_id], timings=self.timings, interval=interval))
        for layer, projection in zip(layers, replacements, strict=True):
            layer.fused_kv_projection = projection

    def fetch(self, hashes):
        if (hashes.ndim != 3 or hashes.shape[0] != len(self.sources) or hashes.shape[2] != self.heads
                or not 0 < hashes.shape[1] <= self.capacity or hashes.dtype != torch.int32
                or hashes.device != self.sequence.device):
            raise ValueError("reference Engram hashes exceed prepared native delivery geometry")
        tokens = hashes.shape[1]
        hooks = {}
        for index, (layer, (weights, scales, requests)) in enumerate(self.planes.items()):
            requests[:tokens].copy_(hashes[index])
            mailbox = self.mailboxes[layer].device_view
            publish_requests(requests[:tokens], mailbox, self.sequence)
            inputs = DeliveredFP8Rows(weights[:tokens * self.heads].view(torch.float8_e4m3fn).view(tokens, -1),
                scales[:tokens * self.heads].view(tokens, -1), (hashes[index] >= 0).any(dim=1),
                requests[:tokens], mailbox)
            hooks[layer] = partial(_rows, inputs)
        return hooks

    def buffers(self):
        return {**{f"engram_{layer}_{name}": tensor for layer, planes in self.planes.items()
                   for name, tensor in zip(("weights", "scales", "hashes"), planes, strict=True)},
                **{f"engram_{layer}_requests": box.device_view for layer, box in self.mailboxes.items()},
                **self.timings.buffers()}

    def export_events(self):
        return {f"engram_{layer}_ready": event.cuda_event for layer, event in self.events.items()}

    def resident_catalog(self):
        return dict(vocabulary=self.vocabulary, tables=[dict(
            planes=[dict(segments=[dict(path=str(source.path), offset=source.offset, nbytes=source.nbytes)],
                         row_bytes=width) for source, width in zip(pair, (self.width, self.width // 32), strict=True)],
            heads=self.heads, weights=f"engram_{layer}_weights", scales=f"engram_{layer}_scales",
            hashes=f"engram_{layer}_hashes", event=f"engram_{layer}_ready", request=f"engram_{layer}_requests")
            for layer, pair in self.sources.items()])

    def export_metadata(self):
        """Describe table geometry, hash functions and checkpoint spans for native export."""
        if (self.token_map is None or self.compressed_vocabulary is None or self.table_hashes is None
                or len(self.token_map) != self.vocabulary
                or set(self.table_hashes) != set(self.sources)):
            raise ValueError("reference Engram export requires complete tokenizer and hash metadata")
        tables = []
        for layer, (weight_source, scale_source) in self.sources.items():
            rows = weight_source.nbytes // self.width
            hash_meta = self.table_hashes[layer]
            if (len(hash_meta["orders"]) != self.heads or len(hash_meta["primes"]) != self.heads
                    or len(hash_meta["offsets"]) != self.heads
                    or hash_meta["offsets"][-1] + hash_meta["primes"][-1] != rows):
                raise ValueError(f"reference Engram hash metadata differs from table {layer} geometry")
            tables.append(dict(
                layer=layer,
                rows=rows,
                embedding_width=self.width,
                scale_block=32,
                encoding="FP8_E4M3_E8M0",
                weight_buffer=f"engram_{layer}_weights",
                scale_buffer=f"engram_{layer}_scales",
                hash_buffer=f"engram_{layer}_hashes",
                ready_event=f"engram_{layer}_ready",
                request_buffer=f"engram_{layer}_requests",
                weights=[dict(path=str(weight_source.path), offset=weight_source.offset, nbytes=weight_source.nbytes)],
                scales=[dict(path=str(scale_source.path), offset=scale_source.offset, nbytes=scale_source.nbytes)],
                hash=dict(
                    method=hash_meta["method"],
                    orders=list(hash_meta["orders"]),
                    multipliers=list(hash_meta["multipliers"]),
                    primes=list(hash_meta["primes"]),
                    offsets=list(hash_meta["offsets"]),
                ),
            ))
        return dict(token_map=list(self.token_map), pad_token_id=self.pad_token_id,
                    compressed_vocabulary=self.compressed_vocabulary, tables=tables)
