# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel target execution with complete accepted-state commits.

Ranks on one host reduce FP32 partial sums. Ranks on separate hosts reduce through RoCEnante,
which carries BF16: each rank rounds its partial sum to BF16 before the reduction.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from hyperflare_recipes.components.causal_convolution import PreparedCausalConvolution
from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllReduce, _roce
from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.rms_norm import PreparedTorchRMSNorm
from hyperflare_recipes.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.preparation.examples import declaration_tensor

from .attention import LayerCache, PreparedAttention
from .residual import PreparedResidualMix
from .vocabulary import PreparedLogits, embedding_rows


@dataclass(frozen=True)
class Scratch:
    logits: torch.Tensor
    local_logits: torch.Tensor
    tokens: int

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device):
        first, last = shard.padded_partition(config.text.vocabulary, alignment=64)
        return cls(torch.zeros((samples, config.text.valid_vocabulary), dtype=torch.float32, device=device),
                   torch.zeros((samples, last - first), dtype=torch.float32, device=device), tokens)

    def rows(self, tokens, samples):
        return type(self)(self.logits[:samples], self.local_logits[:samples], tokens)


def reduction_dtype(runtime):
    """Element type of cross-rank hidden reductions: BF16 over RoCEnante, FP32 on one host."""
    return torch.bfloat16 if runtime is not None and _roce(runtime) else torch.float32


@dataclass(frozen=True)
class ResidualState:
    blocks: tuple[torch.Tensor, ...]
    partial: torch.Tensor


class PreparedLayer(torch.nn.Module):
    def __init__(self, layer, weights, text, pool, metadata, reduction, reduction_dtype, *, mode, tokens,
                 session, workspace, name, examples=None, cache_layer=None):
        super().__init__()
        self.layer, self.weights, self.text, self.mode = layer, weights, text, mode
        self.reduction, self.reduction_dtype = reduction, reduction_dtype
        cache_layer = layer if cache_layer is None else cache_layer
        self.attention = PreparedAttention(layer, weights.attention, text, LayerCache.of(pool, cache_layer), metadata,
            mode=mode, tokens=tokens, session=session, workspace=workspace, name=name + ".attention", examples=examples,
            tuning_page_count=((int(pool.memory_capacity[0].item()) - 1) * pool.ring_row_pages + pool.geometry.ring // 128
                               if text.local(layer) else int(pool.memory_capacity[1].item())))
        device = weights.router.device
        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "inkling.mlp." + role, shape, dtype, device)
        source = example("source", (tokens, text.hidden))
        routes = self._build_router(source, weights, text, metadata, tokens=tokens, session=session,
            name=name, example=example)
        self.experts = self._build_experts(source, weights, routes, tokens=tokens, mode=mode,
            session=session, workspace=workspace, name=name, example=example)
        self.attention_convolution = PreparedCausalConvolution(weights.attention_convolution,
            pool.history(cache_layer, "attention_convolution"))
        self.mlp_convolution = PreparedCausalConvolution(weights.mlp_convolution, pool.history(cache_layer, "mlp_convolution"))
        blocks = (layer + text.residual_block - 1) // text.residual_block
        self.input_mix = (None if weights.attention_query is None else PreparedResidualMix(source,
            weights.attention_query, weights.input_norms, sources=blocks + 1, eps=text.eps,
            session=session, name=name + ".attention_input", examples=examples))
        self.input_norms = (() if self.input_mix is not None else tuple(PreparedTorchRMSNorm(source, norm,
            example(f"input_norm{index}", source.shape), eps=text.eps, session=session, name=f"{name}.input_norm{index}").functional(session=session)
            for index, norm in enumerate(weights.input_norms)))
        self.mlp_mix = PreparedResidualMix(source, weights.mlp_query, (weights.mlp_norm,),
            sources=blocks + (2 if layer % text.residual_block == 0 else 1), eps=text.eps,
            session=session, name=name + ".mlp_input", addend=layer % text.residual_block != 0, examples=examples)

    def _build_router(self, source, weights, text, metadata, *, tokens, session, name, example):
        device = weights.router.device
        logits = example("router_logits", (tokens, text.experts), torch.float32)
        self.router_dtype = weights.router.dtype
        self.router_projection = UnquantizedProjection(example("router_source", source.shape, self.router_dtype),
            weights.router, logits, session=session, name=name + ".router").functional(session=session)
        routes = RouterScratch.allocate(tokens, device, experts=text.experts, top_k=text.experts_per_token)
        self.router = PreparedRouter(logits, routes, session=session, name=name + ".routing", score_func="sigmoid",
            renormalize=True, routed_scaling_factor=text.route_scale, correction_bias=weights.correction,
            num_valid_tokens=metadata.live_counts[:1]).functional(session=session)
        return routes

    def _build_experts(self, source, weights, routes, *, tokens, mode, session, workspace, name, example):
        del tokens
        return PreparedRoutedExperts(source, weights.experts, routes.ids, routes.weights,
            example("partial", source.shape), session=session, name=name + ".experts", workspace=workspace,
            activation_mode="a4" if mode == "prefill" else "a16").functional(session=session)

    def forward(self, state, metadata):
        w, t, m = self.weights, self.text, metadata
        streams = (tuple(norm(state.partial) for norm in self.input_norms) if self.input_mix is None
                   else self.input_mix((*state.blocks, state.partial)))
        _, attention_partial, attention_values = self.attention.attend(streams, m)
        attention_raw = self.reduction(attention_partial.to(self.reduction_dtype)).to(torch.bfloat16)
        attention, attention_previous = self.attention_convolution(attention_raw, m)
        if self.layer % t.residual_block == 0:
            mlp_input = self.mlp_mix((*state.blocks, state.partial, attention))[0]
        else:
            mlp_input = self.mlp_mix((*state.blocks, state.partial), addend=attention)[0]
        routes = self.router(self.router_projection(mlp_input.float()), num_valid_tokens=m.live_counts[:1])
        expert_partial = self.experts(mlp_input, routes.ids, routes.weights)
        mlp_raw = self.reduction(expert_partial.to(self.reduction_dtype)).to(torch.bfloat16)
        mlp, mlp_previous = self.mlp_convolution(mlp_raw, m)
        if self.mode != "verify":
            self.attention_convolution.commit(attention_previous, attention_raw, m)
            self.mlp_convolution.commit(mlp_previous, mlp_raw, m)
        records = (attention_values, attention_previous, attention_raw, mlp_previous, mlp_raw)
        if self.layer % t.residual_block == 0:
            state = ResidualState((*state.blocks, state.partial), (attention.float() + mlp.float()).to(torch.bfloat16))
        else:
            state = ResidualState(state.blocks, (state.partial.float() + attention.float() + mlp.float()).to(torch.bfloat16))
        return state, records

    def commit(self, records, counts, metadata):
        values, attention_previous, attention_raw, mlp_previous, mlp_raw = records
        self.attention.commit(values, counts, metadata)
        self.attention_convolution.commit(attention_previous, attention_raw, metadata, counts=counts)
        self.mlp_convolution.commit(mlp_previous, mlp_raw, metadata, counts=counts)


class PreparedDecoder:
    def __init__(self, weights, tables, scratch, metadata, pool, *, mode, session, name, workspace,
                 runtime, vocabulary_runtime, coordinator, examples=None, vision_state=None,
                 vocabulary_parallel=False, **kwargs):
        del tables, kwargs
        self.weights, self.pool, self.mode, self.scratch = weights, pool, mode, scratch
        self.vision_state, self.vocabulary_parallel = vision_state, vocabulary_parallel
        text, tokens, samples = weights.config.text, scratch.tokens, scratch.logits.shape[0]
        device = weights.embedding.device
        examples = {} if examples is None else examples
        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "inkling.decoder." + role, shape, dtype, device)
        # Traced forwards read this attribute, never the runtime, which compile workers see as a placeholder.
        self.reduction_dtype = reduction_dtype(runtime)
        partial = example("partial", (tokens, text.hidden), self.reduction_dtype)
        self.reduce = PreparedChunkedAllReduce(partial, example("reduced", partial.shape, self.reduction_dtype),
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".reduction").functional(session=session)
        self.layers = [PreparedLayer(index, layer, text, pool, metadata, self.reduce, self.reduction_dtype, mode=mode, tokens=tokens,
            session=session, workspace=workspace, name=f"{name}.layer{index}", examples=examples)
            for index, layer in enumerate(weights.layers)]
        self.logits = PreparedLogits(weights.head, text, weights.rank, tokens=samples,
                                     session=session, name=name + ".logits", examples=examples)
        local = example("local_logits", (samples, weights.head.shape[0]), torch.float32)
        gathered = example("gathered_logits", (samples, weights.head.shape[0] * weights.world), torch.float32)
        self.gather = PreparedAllGather(local.view(torch.bfloat16), gathered.view(torch.bfloat16),
            runtime=vocabulary_runtime, coordinator=coordinator, session=session,
            name=name + ".logit_gather").functional(session=session)
        self.head = SimpleNamespace(gather=SimpleNamespace(runtime=vocabulary_runtime))
        self.first_token = torch.full((1,), weights.rank * weights.embedding.shape[0], dtype=torch.int64, device=device)
        self.auxiliary = (example("auxiliary", (tokens, len(weights.config.draft.taps), text.hidden))
                          if weights.draft is not None else None)
        self.local_logits = scratch.local_logits
        embedded = example("embedded", (tokens, text.hidden))
        self.embedding_norm = PreparedTorchRMSNorm(embedded, weights.embedding_norm, example("normalized_embedding", embedded.shape),
            eps=text.eps, session=session, name=name + ".embedding_norm").functional(session=session)
        self.output_mix = PreparedResidualMix(embedded, weights.output_query, (weights.final_norm,),
            sources=(len(weights.layers) + text.residual_block - 1) // text.residual_block + 1,
            eps=text.eps, session=session, name=name + ".output_mix", examples=examples)

    def release_outputs(self):
        self.scratch = self.local_logits = self.auxiliary = None

    def __call__(self, metadata):
        w, t, m = self.weights, self.weights.config.text, metadata
        embedded = embedding_rows(w.embedding, m.token_ids, self.first_token)
        hidden = self.reduce(embedded.to(self.reduction_dtype)).to(torch.bfloat16)
        hidden = self.embedding_norm(hidden)
        if self.vision_state is not None and self.mode == "prefill":
            self.vision_state.inject(m, hidden)
        hidden = torch.where((m.positions >= 0)[:, None], hidden, 0)
        state, records, taps = ResidualState((), hidden), [], []
        for index, layer in enumerate(self.layers):
            if w.draft is not None and index in w.config.draft.taps:
                taps.append(state.partial)
            state, record = layer(state, m)
            if self.mode == "verify":
                records.append(record)
            del record
        normalized = self.output_mix((*state.blocks, state.partial))[0]
        if w.draft is not None:
            taps.append(normalized)
        sampled_hidden = normalized[m.sample_indices]
        local = self.logits(sampled_hidden)
        logits = local if self.vocabulary_parallel else self.gather(local.view(torch.bfloat16)).view(torch.float32)[:, :t.valid_vocabulary].contiguous()
        auxiliary = torch.stack(taps, dim=1) if taps else None
        return sampled_hidden, logits, auxiliary, tuple(records)

    def commit(self, counts, records, metadata):
        for layer, record in zip(self.layers, records, strict=True):
            layer.commit(record, counts, metadata)
