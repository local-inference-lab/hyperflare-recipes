# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM C4 index projections, sequence pooling and sparse physical selection."""

from dataclasses import dataclass, replace

import torch
from b12x.attention import dsa_indexer, sparse_mla
from b12x.attention.dsa_indexer import glm_pool
from b12x.preparation import PreparedCall

from hyperflare_recipes.components import sm103
from hyperflare_recipes.components.indexer import PreparedSparseIndexer
from hyperflare_recipes.components.indexer_sm103 import Sm103SparseIndexer
from hyperflare_recipes.components.linear import prepare_projection, UnquantizedProjection
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tuning import declare


_SELECT_ROWS = 512


def gather_candidates(gathers, slices, candidates, shape):
    """Gather every rank's packed FP32 (score, pool) candidates slice by slice into FP32 `shape` rows.

    Each gather writes plain row slices of one BF16 buffer, viewed as FP32 pairs only after the last one.
    Gathers must not write through dtype-viewed row slices of a graph-local FP32 buffer: Inductor can size
    the reinplaced base of the custom op's output to a single slice."""
    rows = candidates.shape[0]
    packed = candidates.view(torch.bfloat16).view(rows, -1)
    gathered = torch.empty((rows, shape[1] * shape[2] * 2), dtype=torch.bfloat16, device=candidates.device)
    for gather, (start, end) in zip(gathers, slices, strict=True):
        gather(packed[start:end], output=gathered[start:end])
    return gathered.view(torch.float32).view(shape)


@dataclass(frozen=True)
class IndexState:
    cache: torch.Tensor
    tail: torch.Tensor

    @classmethod
    def allocate(cls, pages, slots, device, *, ring=4):
        if pages <= 0 or slots <= 0 or ring < 4 or ring % 4:
            raise ValueError("GLM index state requires positive pages/slots and a ring divisible by four")
        return cls(torch.zeros((pages, 64, 132), dtype=torch.uint8, device=device),
                   torch.zeros((slots, 2, ring, 128), dtype=torch.bfloat16, device=device))


@dataclass(frozen=True)
class IndexMetadata:
    positions: torch.Tensor
    token_slots: torch.Tensor
    requests: torch.Tensor
    state_slots: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor

    @classmethod
    def allocate(cls, tokens, sequences, blocks, device):
        if min(tokens, sequences, blocks) <= 0:
            raise ValueError("GLM index metadata requires positive capacities")
        return cls(torch.full((tokens,), -1, dtype=torch.int64, device=device),
            torch.full((tokens,), -1, dtype=torch.int64, device=device),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.full((sequences,), -1, dtype=torch.int64, device=device),
            torch.zeros(sequences + 1, dtype=torch.int32, device=device),
            torch.full((sequences, blocks), -1, dtype=torch.int32, device=device))


@dataclass(frozen=True)
class IndexScratch:
    query: torch.Tensor
    query_fp8: torch.Tensor
    query_scales: torch.Tensor
    normalized_key: torch.Tensor
    gate: torch.Tensor
    weights: torch.Tensor
    pool_lengths: torch.Tensor
    pools: torch.Tensor
    physical: torch.Tensor
    selected_lengths: torch.Tensor
    page_table: torch.Tensor
    # Decode context parallel selection: local scores, packed and gathered candidates, global pools.
    scores: torch.Tensor
    candidates: torch.Tensor
    gathered: torch.Tensor
    global_pools: torch.Tensor
    # Row-owned prefill selection: gathered global C4 index pages and variable-length gather staging.
    global_cache: torch.Tensor | None = None
    pool_records: torch.Tensor | None = None
    pool_counts: torch.Tensor | None = None

    @classmethod
    def allocate_row_shard(cls, total_tokens, query_rows, blocks, device, *, context_ranks):
        if total_tokens <= 0 or query_rows <= 0 or blocks <= 0 or context_ranks <= 1:
            raise ValueError("row-sharded GLM index scratch requires positive capacities and DCP ranks > 1")
        def tensor(rows, width, dtype=torch.bfloat16):
            return torch.zeros((rows, width), dtype=dtype, device=device)
        global_pages = blocks * context_ranks
        return cls(
            tensor(query_rows, 4096), tensor(query_rows, 4096, torch.float8_e4m3fn),
            tensor(query_rows, 32, torch.float32),
            tensor(total_tokens, 128), tensor(total_tokens, 128), tensor(query_rows, 32, torch.float32),
            torch.zeros(query_rows, dtype=torch.int32, device=device),
            tensor(query_rows, 512, torch.int32),
            tensor(query_rows, 2051, torch.int32), torch.zeros(query_rows, dtype=torch.int32, device=device),
            torch.zeros((1, global_pages), dtype=torch.int32, device=device),
            *cls._context_selection(0, context_ranks, device),
            global_cache=torch.zeros((global_pages, 64, 132), dtype=torch.uint8, device=device),
            pool_records=torch.zeros((context_ranks, blocks * 64 * 132), dtype=torch.uint8, device=device),
            pool_counts=torch.zeros(context_ranks, dtype=torch.int64, device=device),
        )

    @classmethod
    def allocate(cls, tokens, blocks, mode, device, *, context_ranks=1):
        if tokens <= 0 or blocks <= 0 or mode not in ("decode", "prefill"):
            raise ValueError("GLM index scratch requires positive capacities and decode/prefill mode")
        def tensor(width, dtype=torch.bfloat16):
            return torch.zeros((tokens, width), dtype=dtype, device=device)
        return cls(tensor(4096), tensor(4096, torch.float8_e4m3fn), tensor(32, torch.float32),
            tensor(128), tensor(128), tensor(32, torch.float32),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            tensor(512, torch.int32),
            tensor(2051, torch.int32), torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.zeros((tokens if mode == "decode" else 1, blocks), dtype=torch.int32, device=device),
            *cls._context_selection(tokens if context_ranks > 1 else 0, context_ranks, device))

    @staticmethod
    def _context_selection(tokens, ranks, device):
        return (torch.zeros((tokens, 512), dtype=torch.float32, device=device),
                torch.zeros((tokens, 512, 2), dtype=torch.float32, device=device),
                torch.zeros((tokens, ranks * 512, 2), dtype=torch.float32, device=device),
                torch.full((tokens, 512), -1, dtype=torch.int32, device=device))

    def rows_row_shard(self, total_tokens, query_rows):
        if (type(total_tokens) is not int or type(query_rows) is not int
                or not 0 < total_tokens <= self.normalized_key.shape[0]
                or not 0 < query_rows <= self.query.shape[0]):
            raise ValueError("row-sharded GLM index rows exceed scratch capacity")
        return IndexScratch(
            self.query[:query_rows], self.query_fp8[:query_rows], self.query_scales[:query_rows],
            self.normalized_key[:total_tokens], self.gate[:total_tokens], self.weights[:query_rows],
            self.pool_lengths[:query_rows], self.pools[:query_rows], self.physical[:query_rows],
            self.selected_lengths[:query_rows], self.page_table,
            self.scores, self.candidates, self.gathered, self.global_pools,
            global_cache=self.global_cache, pool_records=self.pool_records, pool_counts=self.pool_counts,
        )

    def rows(self, tokens, mode):
        if type(tokens) is not int or not 0 < tokens <= self.query.shape[0] or mode not in ("decode", "prefill"):
            raise ValueError("GLM index rows require valid capacity and decode/prefill mode")
        sequences = tokens if mode == "decode" else 1
        if sequences > self.page_table.shape[0]:
            raise ValueError("GLM index request rows exceed page-table capacity")
        return IndexScratch(self.query[:tokens], self.query_fp8[:tokens], self.query_scales[:tokens],
            self.normalized_key[:tokens], self.gate[:tokens], self.weights[:tokens], self.pool_lengths[:tokens],
            self.pools[:tokens], self.physical[:tokens], self.selected_lengths[:tokens],
            self.page_table[:sequences], self.scores[:tokens], self.candidates[:tokens], self.gathered[:tokens],
            self.global_pools[:tokens])


class PreparedGlmIndexer:
    """Replicated C4 selection for GLM sparse MLA with 256-token model pages.

    Position -1 masks an inactive query; negative state slots mask inactive
    requests. Token slots and page tables address the same physical model pages.
    Prefill admits one request, while decode supports packed per-request rows.
    Tails and cache outlive graph variants; intermediates can be shared by serial
    layers. Returned physical indices address a latent cache of 64-record pages.
    """

    def __init__(self, module, weights, source, query_residual, key_input, weight_input, scratch: IndexScratch,
                 state: IndexState, metadata: IndexMetadata, *, mode, session, name, workspace,
                 max_cache_pages=None, context=None, row_shard_queries=None):
        rows = source.shape[0]
        self.row_shard = row_shard_queries is not None
        if self.row_shard:
            if mode != "prefill" or context is None or context.records is None:
                raise ValueError("row-sharded GLM indexer requires DCP prefill with a variable-length gather runtime")
            if rows != row_shard_queries:
                raise ValueError("row-sharded GLM indexer source rows must equal row_shard_queries")
        if (source.shape != (rows, 4096) or query_residual.shape != (rows, 1536)
                or source.dtype != torch.bfloat16 or query_residual.dtype != source.dtype
                or source.device != query_residual.device):
            raise ValueError("GLM index inputs require BF16 hidden[rows,4096] and query[rows,1536]")
        if (module.n_heads, module.head_dim, module.index_kpool, module.index_topk) != (32, 128, 4, 2048):
            raise ValueError("GLM indexing requires 32 heads, 128 channels, C4 pools and 2048 history tokens")
        if mode not in ("decode", "prefill") or (mode == "prefill" and metadata.state_slots.numel() != 1):
            raise ValueError("GLM index prefill requires one request")
        expected_table_cols = (metadata.block_table.shape[1] * context.shards.world
                               if self.row_shard else metadata.block_table.shape[1])
        if (scratch.query.shape != (rows, 4096)
                or (not self.row_shard and metadata.positions.shape != (rows,))
                or expected_table_cols != scratch.page_table.shape[1]
                or scratch.page_table.shape[0] != (1 if mode == "prefill" else rows)):
            raise ValueError("GLM index scratch and metadata capacities differ")
        self.module, self.scratch, self.state, self.cache, self.metadata, self.mode = (
            module, scratch, state, state.cache, metadata, mode)
        self.blocks = metadata.block_table.shape[1]
        self.context = context
        if context is not None and not self.row_shard and scratch.gathered.shape != (rows, context.shards.world * 512, 2):
            raise ValueError("GLM context-parallel selection requires candidate scratch for every row")
        # The index key and head weights arrive as columns of the layer's fused input projection.
        if key_input.shape != (rows, 128) or weight_input.shape != (rows, 32):
            raise ValueError("GLM index key and weight inputs require [rows, 128] and [rows, 32] projections")
        self.source, self.query_residual = source, query_residual
        self.key_input, self.weight_input = key_input, weight_input
        module.k_norm.weight = torch.nn.Parameter(weights["indexer.k_norm.weight"], requires_grad=False)
        module.k_norm.bias = torch.nn.Parameter(weights["indexer.k_norm.bias"], requires_grad=False)
        self.ape = weights["indexer.index_kpool_compress_ape"]
        self.query_projection = prepare_projection(query_residual, weights["indexer.wq_b"], scratch.query,
            session=session, name=f"{name}.query", workspace=workspace, activation_mode="auto")
        self.gate_projection = UnquantizedProjection(source, weights["indexer.index_kpool_compress_gate"],
            scratch.gate[:rows], session=session, name=f"{name}.gate")
        self.query_arguments = (scratch.query.view(-1, 128), scratch.query_fp8.view(-1, 128),
                                scratch.query_scales.view(-1), scratch.weights.view(-1))
        self.query_plan = glm_pool.plan_queries(*self.query_arguments)
        self.pool_arguments = (state.cache, state.tail, metadata.state_slots, metadata.query_start_loc,
            scratch.normalized_key, scratch.gate, self.ape, metadata.token_slots, metadata.positions)
        self.pool_plan = glm_pool.plan_pools(*self.pool_arguments, mode=mode)
        query_arguments, pool_arguments = self.query_arguments, self.pool_arguments
        declare(session, (
            self.query_plan.request(name=f"{name}.rotate", prepare_call=lambda state:
                PreparedCall(run=lambda: state.quantize(*query_arguments))),
            self.pool_plan.request(name=f"{name}.pool", prepare_call=lambda state:
                PreparedCall(run=lambda: state.update(*pool_arguments))),
        ))
        self.pool_gather = None
        self.global_page_table = None
        if self.row_shard:
            from .dcp import PreparedPoolGather

            self.pool_gather = PreparedPoolGather(state, scratch.pool_records, scratch.pool_counts,
                scratch.global_cache, context, session=session, name=name)
            global_pages = self.blocks * context.shards.world
            self.global_pages = global_pages
            self.global_page_table = persistent_copy(
                torch.arange(global_pages, dtype=torch.int32, device=source.device).unsqueeze(0))
            page_table = self.global_page_table.expand(rows, -1)
            cache = scratch.global_cache.view(global_pages, -1)
        else:
            page_table = scratch.page_table.expand(rows, -1) if mode == "prefill" else scratch.page_table
            cache = state.cache.view(state.cache.shape[0], -1)
        # Prefill selection scratch grows with query rows (about 140 KB each), so long chunks select in slices.
        step = _SELECT_ROWS if mode == "prefill" else rows
        # SM103 selects through FlashInfer's paged FP8 logits and top-k; b12x's indexer serves SM120/SM121.
        selector = Sm103SparseIndexer if sm103.is_sm103(source.device) else PreparedSparseIndexer
        self.selectors = tuple(selector(scratch.query_fp8[start:end].view(end - start, 32, 128),
            scratch.weights[start:end], cache, page_table[start:end], scratch.pool_lengths[start:end],
            None, scratch.pools[start:end],
            caps=None if selector is Sm103SparseIndexer else dsa_indexer.Caps(
                device=source.device, num_q_heads=32, max_q_rows=end - start,
                max_page_table_width=page_table.shape[1], topk=512, mode=mode),
            session=session,
            name=f"{name}.select" if end - start == rows else f"{name}.select{start}", workspace=workspace,
            scores=None if (context is None or self.row_shard) else scratch.scores[start:end])
            for start in range(0, rows, step) for end in (min(start + step, rows),))
        self.scratch_plans = (self.query_projection.plan,
            *(selector.plan for selector in self.selectors if isinstance(selector, PreparedSparseIndexer)))
        self.gathers = ()
        if context is not None and not self.row_shard:
            from ignition.components.collectives import PreparedAllGather

            columns, world = 1024, context.shards.world
            self.gathers = tuple(PreparedAllGather(scratch.candidates[start:end].view(torch.bfloat16).view(-1, 2 * columns),
                scratch.gathered[start:end].view(torch.bfloat16).view(-1, 2 * columns * world),
                runtime=context.candidates, session=session, coordinator=context.coordinator,
                name=f"{name}.candidates{start}") for start, end in context.slices(rows))
        # SM103 prefill and row-sharded DCP prefill attend over a global slab, so they select global positions.
        self.logical = self.row_shard or (context is None and mode == "prefill" and sm103.is_sm103(source.device))
        self.expand_arguments = (scratch.pools, metadata.positions, metadata.requests,
                                 metadata.block_table, scratch.physical, scratch.selected_lengths)
        if self.logical:
            return
        self.expand_options = {"pool_size": 4, "block_size": 256, "block_stride_rows": 256,
                               "num_cache_blocks": state.cache.shape[0]}
        self.expand_plan = sparse_mla.plan_pooled_selection(device=source.device, max_rows=rows,
            page_size=256, max_page_table_width=metadata.block_table.shape[1],
            num_cache_blocks=max(state.cache.shape[0], max_cache_pages or 0))
        expand_arguments, expand_options = self.expand_arguments, self.expand_options
        declare(session, (self.expand_plan.request(name=f"{name}.expand", prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*expand_arguments, **expand_options))),))

    def rebind_pages(self):
        """Address the state's current index cache; plans bound its page count from above."""
        cache = self.state.cache
        self.cache = cache
        if not getattr(self, "row_shard", False):
            for selector in self.selectors:
                selector.rebind_cache(cache.view(cache.shape[0], -1))
        if not self.logical:
            self.expand_options["num_cache_blocks"] = cache.shape[0]

    def functional(self, *, session):
        self.query_projection.functional(session=session)
        self.gate_projection.functional(session=session)
        for selector in self.selectors:
            selector.functional(session=session)
        for gather in self.gathers:
            gather.functional(session=session)
        if self.pool_gather is not None:
            self.pool_gather.functional(session=session)
        self.scratch = self.metadata = self.query_arguments = self.pool_arguments = None
        self.expand_arguments = None
        self.source = self.query_residual = self.key_input = self.weight_input = None
        return self

    def __call__(self, source, query_residual, key_input, weight_input, metadata, scratch=None):
        result, _ = self.forward(source, query_residual, key_input, weight_input, metadata, scratch)
        return result

    def forward(self, source, query_residual, key_input, weight_input, metadata, scratch=None):
        rows = source.shape[0]
        s = IndexScratch.allocate(rows, self.blocks, self.mode, source.device,
            context_ranks=1 if self.context is None else self.context.shards.world) if scratch is None else scratch
        query = self.query_projection(query_residual)
        s.normalized_key.copy_(self.module.k_norm(key_input))
        gate = self.gate_projection(source)
        s = replace(s, query=query, gate=gate)
        s.weights.copy_(weight_input)
        glm_pool.quantize_queries(query.view(-1, 128), s.query_fp8.view(-1, 128),
            s.query_scales.view(-1), s.weights.view(-1), plan=self.query_plan)
        glm_pool.update_pools(self.cache, self.state.tail, metadata.state_slots, metadata.query_start_loc,
            s.normalized_key, gate, self.ape, metadata.token_slots, metadata.positions, plan=self.pool_plan)
        pools = (metadata.positions + 1) // 4
        s.pool_lengths.copy_(pools if self.context is None else self.context.shards.local_pools(pools))
        if self.mode == "prefill":
            s.page_table.copy_(metadata.block_table)
        else:
            s.page_table.copy_(metadata.block_table[metadata.requests.long()])
        page_table = s.page_table.expand(rows, -1) if self.mode == "prefill" else s.page_table
        for selector, (start, end) in zip(self.selectors,
                ((start, min(rows, start + (_SELECT_ROWS if self.mode == "prefill" else rows)))
                 for start in range(0, rows, _SELECT_ROWS if self.mode == "prefill" else rows)), strict=True):
            result = selector(s.query_fp8[start:end].view(end - start, 32, 128), s.weights[start:end], page_table[start:end],
                s.pool_lengths[start:end], None, cache=self.cache.view(self.cache.shape[0], -1))
            if isinstance(result, tuple):
                selected, scores = result
                s.scores[start:end].copy_(scores)
            else:
                selected = result
            s.pools[start:end].copy_(selected)
        if self.logical:
            from .dcp import expand_tokens

            expand_tokens(s.pools, metadata.positions, s.physical, s.selected_lengths)
            return (s.physical, s.selected_lengths, page_table), s
        if self.context is None:
            sparse_mla.expand_pooled_topk_to_physical_slots(s.pools, metadata.positions, metadata.requests,
                metadata.block_table, s.physical, s.selected_lengths,
                **self.expand_options, plan=self.expand_plan)
            return (s.physical, s.selected_lengths, page_table), s
        from .dcp import expand_tokens, merge_candidates, merge_expand_local, pack_candidates

        # Every rank merges identical global candidates, then keeps the records it owns.
        pack_candidates(s.pools, s.scores, s.candidates, self.context.shards)
        s = replace(s, gathered=gather_candidates(self.gathers, self.context.slices(rows), s.candidates,
                                                    s.gathered.shape))
        if self.mode == "prefill":
            # Prefill attends over gathered context blocks addressed by global position.
            merge_candidates(s.gathered, s.global_pools)
            expand_tokens(s.global_pools, metadata.positions, s.physical, s.selected_lengths)
        else:
            merge_expand_local(s.gathered, s.global_pools, metadata.positions, metadata.requests,
                               metadata.block_table, s.physical,
                               s.selected_lengths, self.context.shards)
        return (s.physical, s.selected_lengths, page_table), s


    def project_shard(self, source_sh, query_res_sh, key_in_sh):
        """Project this rank's owned rows for C4 index queries, normalized keys and pooling gates."""
        idx_query = self.query_projection(query_res_sh)
        norm_key = self.module.k_norm(key_in_sh)
        gate = self.gate_projection(source_sh)
        return idx_query, norm_key, gate

    def update_cache(self, norm_key_all, gate_all, metadata):
        """Update this rank's striped DCP C4 index cache and replicated tail from gathered chunk rows."""
        glm_pool.update_pools(self.cache, self.state.tail, metadata.state_slots, metadata.query_start_loc,
            norm_key_all, gate_all, self.ape, metadata.token_slots, metadata.positions, plan=self.pool_plan)

    def gather_pools(self, table, end):
        """Gather every rank's visible C4 pools into contiguous global 64-pool index pages."""
        return self.pool_gather(table, end)

    def select_shard(self, idx_query_sh, weight_in_sh, positions_sh, global_cache):
        """Score this rank's owned queries against gathered global C4 pools and expand to global positions."""
        from .dcp import expand_tokens

        rows = idx_query_sh.shape[0]
        device = idx_query_sh.device
        query_fp8 = torch.empty((rows, 4096), dtype=torch.float8_e4m3fn, device=device)
        query_scales = torch.empty((rows, 32), dtype=torch.float32, device=device)
        weights = weight_in_sh.to(torch.float32).contiguous()
        glm_pool.quantize_queries(idx_query_sh.view(-1, 128), query_fp8.view(-1, 128),
            query_scales.view(-1), weights.view(-1), plan=self.query_plan)
        pool_lengths = ((positions_sh + 1) // 4).clamp_min(0).to(torch.int32)
        page_table = self.global_page_table.expand(rows, -1)
        flat_cache = global_cache.view(self.global_pages, -1)
        pools = torch.empty((rows, 512), dtype=torch.int32, device=device)
        for selector, start in zip(self.selectors, range(0, rows, _SELECT_ROWS), strict=True):
            end = min(rows, start + _SELECT_ROWS)
            selected = selector(query_fp8[start:end].view(end - start, 32, 128), weights[start:end],
                page_table[start:end], pool_lengths[start:end], None, cache=flat_cache)
            pools[start:end].copy_(selected)
        physical = torch.empty((rows, 2051), dtype=torch.int32, device=device)
        selected_lengths = torch.empty(rows, dtype=torch.int32, device=device)
        expand_tokens(pools, positions_sh, physical, selected_lengths)
        return physical, selected_lengths

    def select_shard_into(self, idx_query_sh, weights_sh, pool_lengths_sh, positions_sh, global_cache,
                          query_fp8, query_scales, pools, physical, selected_lengths, scratches):
        """Run C4 selection on the active stream into caller-owned buffers (zero internal allocations)."""
        from .dcp import expand_tokens

        rows = idx_query_sh.shape[0]
        glm_pool.quantize_queries(idx_query_sh.view(-1, 128), query_fp8.view(-1, 128),
            query_scales.view(-1), weights_sh.view(-1), plan=self.query_plan)
        page_table = self.global_page_table.expand(rows, -1)
        flat_cache = global_cache.view(self.global_pages, -1)
        for selector, scratch, start in zip(self.selectors, scratches, range(0, rows, _SELECT_ROWS), strict=True):
            end = min(rows, start + _SELECT_ROWS)
            binding = dsa_indexer.bind(
                selector.plan, scratch=scratch,
                q_fp8=query_fp8[start:end].view(end - start, 32, 128),
                query_weights=weights_sh[start:end], index_k_cache=flat_cache,
                page_table=page_table[start:end], cache_lengths=pool_lengths_sh[start:end],
                active_width=None, output_indices=pools[start:end], output_scores=None,
            )
            dsa_indexer.run(binding)
        expand_tokens(pools, positions_sh, physical, selected_lengths)
