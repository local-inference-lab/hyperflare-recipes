# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured attention metadata derived from Flare's validated packed requests.

Cache indexing follows DeepInfer's prepare_tensor_input_v4.cpp (Apache-2.0),
with explicit inactive write destinations for Flare's padded graph capacities.
"""

import torch


class ReferenceDecodeMetadata:
    """Bind request positions and page tables to the reference attention inputs.

    Admission and packed-input validation must precede this computation. It
    preserves the reference's C1/C2 positions and physical cache addressing,
    while preventing padded tokens from reading or writing request state.
    Engram history and image routing masks are explicit caller-owned inputs.
    This binding does not prepare prefill metadata or commit request history.
    """

    def __init__(self, config, geometry):
        from deepinfer.model_executor.model_metadata_utils import derive_compress_ratios

        self.dense_ratios, self.sparse_ratios = derive_compress_ratios(config)
        self.ratios = sorted(set(self.dense_ratios) | set(self.sparse_ratios))
        self.geometry = geometry
        self.window = config.window_size[0]
        if (geometry.window_tokens != self.window + geometry.verification_width - 1
                or any(ratio not in (1, 2) or geometry.page_tokens % ratio for ratio in self.ratios)):
            raise ValueError("reference decode metadata requires matching window and C1/C2 cache geometry")

    def __call__(self, source, *, ngram_token_ids=None, images_seq_mask=None):
        from deepinfer.model_executor.input_metadata import InputMetadataV4VL

        g = self.geometry
        tokens = source.positions.numel()
        if source.max_query_tokens > g.verification_width:
            raise ValueError("decode query capacity exceeds the prepared verification width")
        if (source.token_page_table.ndim != 2 or source.token_page_table.shape[0] != tokens
                or tokens * source.token_page_table.shape[1] * g.page_tokens > torch.iinfo(torch.int32).max):
            raise ValueError("reference decode page geometry exceeds int32 indexing")
        if ngram_token_ids is not None and (
                ngram_token_ids.ndim != 2 or ngram_token_ids.shape[0] != tokens
                or ngram_token_ids.dtype != torch.int32 or ngram_token_ids.device != source.positions.device):
            raise ValueError("reference Engram history must contain int32 rows for every packed token")
        if images_seq_mask is not None and (
                images_seq_mask.shape != source.positions.shape or images_seq_mask.dtype != torch.bool
                or images_seq_mask.device != source.positions.device):
            raise ValueError("reference image routing mask must match the packed tokens")
        valid = source.positions >= 0
        position = torch.where(valid, source.positions, 0).int()
        context = torch.where(valid, position + 1, 0)
        states = source.state_rows[source.requests].int()
        window_length = context.clamp(max=self.window)
        window_start = states * g.window_tokens + (position - window_length + 1) % g.window_tokens
        swa_index = torch.stack((torch.where(valid, window_start, 0), window_length), dim=1)
        swa_store = torch.where(valid, states * g.window_tokens + position % g.window_tokens, -1)
        pages = source.token_page_table.int().contiguous()
        page_column = (position // g.page_tokens).long()
        physical_page = pages.gather(1, page_column[:, None]).flatten()
        positions_c, stores, dense, klen = {}, {}, {}, {}
        for ratio in self.ratios:
            key = str(ratio)
            emit = valid & ((position + 1) % ratio == 0)
            compressed = position // ratio
            block = g.page_tokens // ratio
            positions_c[key] = torch.where(emit, compressed * ratio, 0)
            stores[key] = torch.where(emit, physical_page * block + compressed % block, -1)
            if ratio in self.sparse_ratios:
                klen[key] = context // ratio
            if ratio in self.dense_ratios:
                offset = torch.arange(tokens, device=position.device, dtype=torch.int32) * pages.shape[1] * block
                dense[key] = torch.stack((offset, context // ratio), dim=1)
        result = InputMetadataV4VL(is_prompt=False,
            input_ids=torch.where(valid, source.token_ids, 0).int(), positions=position,
            valid_token_mask=valid, verification_width=g.verification_width,
            ngram_token_ids=ngram_token_ids, images_seq_mask=images_seq_mask,
            state_block_idx=torch.where(valid, states, -1), context_lens=context,
            # The reference groups padded queries under this out-of-range ID.
            query_indices=torch.where(valid, source.requests, 998244353).int(),
            block_tables=pages, slot_mapping_store_swa=swa_store,
            kv_swa_index_offset_len=swa_index, positions_c=positions_c,
            slot_mapping_store_ca=stores, kv_ca_index_offset_len=dense, klen=klen)
        result.valid_token_count = valid.sum(dtype=torch.int32).reshape(1)
        return result


class ReferencePrefillMetadata:
    """Bind one prefill request to fixed-capacity reference cache operations.

    Flare's prefill graphs admit one request. Cache gathers cover the prepared
    page-table capacity; unused rows are zero-filled and excluded by attention
    lengths. Compression retains the actual chunk boundaries, including a pair
    that begins in the preceding chunk. All computation is captured in Ignition.
    """

    def __init__(self, config, geometry, *, yoco_keep_tokens=None):
        from deepinfer.model_executor.model_metadata_utils import derive_compress_ratios

        self.dense_ratios, self.sparse_ratios = derive_compress_ratios(config)
        self.ratios = sorted(set(self.dense_ratios) | set(self.sparse_ratios))
        self.geometry = geometry
        self.window = config.window_size[0]
        if (geometry.window_tokens != self.window + geometry.verification_width - 1
                or any(ratio not in (1, 2) or geometry.page_tokens % ratio for ratio in self.ratios)):
            raise ValueError("reference prefill metadata requires matching window and C1/C2 cache geometry")
        if yoco_keep_tokens is not None and (
                type(yoco_keep_tokens) is not int or yoco_keep_tokens < geometry.window_tokens):
            raise ValueError("reference YOCO compaction must cover the sliding-window ring capacity")
        self.yoco_keep_tokens = yoco_keep_tokens

    def __call__(self, source, *, ngram_token_ids=None, images_seq_mask=None):
        from deepinfer.model_executor.input_metadata import InputMetadataV4VL

        g = self.geometry
        tokens = source.positions.numel()
        if source.lengths.numel() != 1 or source.page_table.shape[0] != 1:
            raise ValueError("reference prefill requires one admitted request")
        blocks = source.page_table.shape[1]
        if blocks < 1 or blocks * g.page_tokens > torch.iinfo(torch.int32).max:
            raise ValueError("reference prefill page geometry exceeds int32 indexing")
        if ngram_token_ids is not None and (
                ngram_token_ids.ndim != 2 or ngram_token_ids.shape[0] != tokens
                or ngram_token_ids.dtype != torch.int32 or ngram_token_ids.device != source.positions.device):
            raise ValueError("reference Engram history must contain int32 rows for every packed token")
        if images_seq_mask is not None and (
                images_seq_mask.shape != source.positions.shape or images_seq_mask.dtype != torch.bool
                or images_seq_mask.device != source.positions.device):
            raise ValueError("reference image routing mask must match the packed tokens")
        valid = source.positions >= 0
        length = source.lengths[0].int()
        start = torch.where(length > 0, source.starts[0], 0).int()
        end = start + length
        state = source.state_rows[0].int()
        position = torch.where(valid, source.positions, 0).int()
        rows = torch.arange(tokens, device=position.device, dtype=torch.int32)
        zero = torch.zeros((), device=position.device, dtype=torch.int32)
        cu_q = torch.stack((zero, length))
        cu_k = torch.stack((zero, end))
        swa_store = torch.where(valid & (rows >= length - g.window_tokens),
                                state * g.window_tokens + position % g.window_tokens, -1)

        history = start.clamp(max=self.window)
        padding = self.window - history
        ring_begin = (start - history) % g.window_tokens
        first = torch.minimum(history, g.window_tokens - ring_begin)
        load_swa = torch.stack((
            torch.stack((zero - 1, zero, padding, zero)),
            torch.stack((state, ring_begin, ring_begin + first, padding)),
            torch.stack((state, zero, history - first, padding + first))))
        lower = (position - self.window + 1).clamp_min(0)
        history_begin = torch.minimum(lower, start) - start + self.window
        chunk_begin = (lower - start).clamp_min(0)
        pivots = torch.stack((history_begin, self.window - history_begin,
                              self.window + chunk_begin, rows - chunk_begin + 1), dim=1)
        pivots = torch.where(valid[:, None], pivots, 0)

        positions_c, stores, cu_c, mapping = {}, {}, {}, {}
        loads, totals, dense, ks, ke = {}, {}, {}, {}, {}
        page_rows = torch.arange(blocks, device=position.device, dtype=torch.int32)
        for ratio in self.ratios:
            key = str(ratio)
            block = g.page_tokens // ratio
            count = end // ratio - start // ratio
            compressed_rows = torch.arange((tokens + ratio - 1) // ratio,
                                           device=position.device, dtype=torch.int32)
            emitted = compressed_rows < count
            compressed = torch.where(emitted, start // ratio + compressed_rows, 0)
            pages = source.page_table[0].gather(0, (compressed // block).long())
            positions_c[key] = torch.where(emitted, compressed * ratio, 0)
            stores[key] = torch.where(emitted, pages * block + compressed % block, -1)
            cu_c[key] = torch.stack((zero, count.clamp_min(1)))
            # Tail outputs still belong to this request. The reference compressor
            # zeroes entries beyond its actual count without reading padded input.
            mapping[key] = torch.zeros_like(compressed_rows)
            available = (end // ratio - page_rows * block).clamp(0, block)
            offset = page_rows * block
            read = torch.stack((torch.where(available > 0, source.page_table[0], -1),
                                torch.zeros_like(offset), available, offset), dim=1)
            clear = torch.stack((torch.full_like(offset, -1), available,
                                 torch.full_like(offset, block), offset + available), dim=1)
            loads[key] = torch.stack((read, clear), dim=1).reshape(-1, 4)
            totals[key] = blocks * block
            visible = torch.where(valid, (position + 1) // ratio, 0)
            if ratio in self.dense_ratios:
                dense[key] = torch.stack((torch.zeros_like(visible), visible), dim=1)
            if ratio in self.sparse_ratios:
                ks[key] = torch.zeros_like(visible)
                ke[key] = visible
        yoco_kwargs = {}
        if self.yoco_keep_tokens is not None and tokens > self.yoco_keep_tokens:
            keep = self.yoco_keep_tokens
            kept = length.clamp(0, keep)
            clip_prefix = (length - kept).clamp_min(0)
            rows_c = torch.arange(keep, device=position.device, dtype=torch.int32)
            valid_c = rows_c < kept
            j_c = clip_prefix + rows_c
            query_indices_c = torch.where(valid_c, j_c, -1)
            cu_q_c = torch.stack((zero, kept))
            pos_c = torch.where(valid_c, start + j_c, 0)
            swa_store_c = torch.where(valid_c & (length - j_c <= g.window_tokens),
                                      state * g.window_tokens + pos_c % g.window_tokens, -1)
            rect_lo = end - keep
            abs_lo = torch.maximum(pos_c - self.window + 1, rect_lo).clamp_min(0).minimum(pos_c + 1)
            sec1_lo_abs = torch.minimum(abs_lo, start)
            start_slot = sec1_lo_abs - (start - self.window)
            k_lo = (abs_lo - start).clamp_min(0)
            pivots_c = torch.stack((start_slot, self.window - start_slot,
                                    self.window + (k_lo - clip_prefix), j_c - k_lo + 1), dim=1)
            pivots_c = torch.where(valid_c[:, None], pivots_c, 0)
            dense_c, ks_c, ke_c = {}, {}, {}
            for ratio in self.ratios:
                key = str(ratio)
                visible_c = torch.where(valid_c, (pos_c + 1) // ratio, 0)
                if ratio in self.dense_ratios:
                    dense_c[key] = torch.stack((torch.zeros_like(visible_c), visible_c), dim=1)
                if ratio in self.sparse_ratios:
                    ks_c[key] = torch.zeros_like(visible_c)
                    ke_c[key] = visible_c
            yoco_kwargs = dict(
                yoco_consumer_query_indices=query_indices_c,
                yoco_consumer_cu_seqlens_q=cu_q_c,
                yoco_consumer_total_q=keep,
                yoco_consumer_positions=pos_c,
                yoco_consumer_swa_index_pivots=pivots_c,
                yoco_consumer_ks=ks_c,
                yoco_consumer_ke=ke_c,
                yoco_consumer_kv_ca_index_offset_len=dense_c,
                yoco_consumer_slot_mapping_store_swa=swa_store_c,
            )
        result = InputMetadataV4VL(is_prompt=True,
            input_ids=torch.where(valid, source.token_ids, 0).int(), positions=position,
            valid_token_mask=valid, ngram_token_ids=ngram_token_ids, images_seq_mask=images_seq_mask,
            total_q=tokens, num_reqs=1, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            state_block_idx=torch.where(length > 0, state, -1).reshape(1),
            slot_mapping_store_swa=swa_store, kv_swa_full_index_pivots=pivots,
            load_segments_swa=load_swa, total_k_swa=self.window,
            positions_c=positions_c, slot_mapping_store_ca=stores,
            cu_compressed_seqlens_q=cu_c, compressed_batch_mapping=mapping,
            load_segments_ca=loads, total_k_compressed=totals,
            kv_ca_index_offset_len=dense, ks=ks, ke=ke, **yoco_kwargs)
        result.valid_token_count = valid.sum(dtype=torch.int32).reshape(1)
        return result
