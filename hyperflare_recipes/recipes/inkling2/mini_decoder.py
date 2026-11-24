# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mini layer visits and cross-pass message execution with independent cache histories."""

from types import SimpleNamespace

import torch

from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllReduce
from hyperflare_recipes.components.inkling2_router_ops import (
    PreparedInkling2SplitKRouter,
    splitk_router_eligible,
)
from hyperflare_recipes.components.moe import RouterScratch
from hyperflare_recipes.components.rms_norm import PreparedTorchRMSNorm
from hyperflare_recipes.components.situ_experts import PreparedSiTUExperts
from ignition.preparation.examples import declaration_tensor

from .decoder import PreparedDecoder, PreparedLayer, ResidualState, reduction_dtype
from .loop import PreparedCrossPassRouter, PreparedMessageGate, PreparedSharedQKVNormCat
from .residual import PreparedResidualMix
from .vocabulary import PreparedLogits, embedding_rows


class PreparedLoopLayer(PreparedLayer):
    def __init__(self, layer, weights, text, pool, metadata, reduction, reduction_dtype, *, mode, tokens,
                 session, workspace, name, examples=None, cache_layer=None, messages=False,
                 core_offset=0, num_cross_pass_streams=0):
        super().__init__(layer, weights, text, pool, metadata, reduction, reduction_dtype,
            mode=mode, tokens=tokens, session=session, workspace=workspace, name=name,
            examples=examples, cache_layer=cache_layer)
        self.gates = None
        self.shared_qkv_cat = None
        if messages:
            norms = weights.input_norms if len(weights.input_norms) == 3 else weights.input_norms * 3
            base = 4 * core_offset
            ctrl = weights.controllers
            self.gates = tuple(PreparedMessageGate(gate_w, norm_w, stream_idx=base + i,
                num_streams=num_cross_pass_streams, tokens=tokens, hidden=text.hidden, eps=text.eps,
                examples=examples) for i, (gate_w, norm_w) in enumerate(
                zip((ctrl.q, ctrl.k, ctrl.v, ctrl.mlp), (*norms, weights.mlp_norm), strict=True)))
            if len(weights.input_norms) == 1:
                self.shared_qkv_cat = PreparedSharedQKVNormCat(weights.input_norms[0],
                    base_stream=base, num_streams=num_cross_pass_streams, tokens=tokens,
                    hidden=text.hidden, eps=text.eps, device=weights.router.device, examples=examples)

    def _build_router(self, source, weights, text, metadata, *, tokens, session, name, example):
        if splitk_router_eligible(source, weights.router, weights.correction, None, text.experts_per_token):
            device = weights.router.device
            logits = example("router_logits", (tokens, text.experts), torch.float32)
            routes = RouterScratch.allocate(tokens, device, experts=text.experts, top_k=text.experts_per_token)
            self.router_dtype = weights.router.dtype
            self.router_projection = None
            self.router = None
            self.splitk_router = PreparedInkling2SplitKRouter(
                source, weights.router, weights.correction, logits, routes.ids, routes.weights,
                session=session, name=name + ".splitk_router", route_scale=text.route_scale,
                num_valid_tokens=metadata.live_counts[:1],
            ).functional(session=session)
            return routes
        self.splitk_router = None
        return super()._build_router(source, weights, text, metadata, tokens=tokens, session=session,
                                     name=name, example=example)

    def _build_experts(self, source, weights, routes, *, tokens, mode, session, workspace, name, example):
        del source, routes, mode, session, workspace, name, example
        return PreparedSiTUExperts(weights.experts, tokens=tokens, top_k=self.text.experts_per_token)

    def forward(self, state, metadata, cross_pass=None):
        t, m = self.text, metadata
        if cross_pass is None:
            streams = (tuple(norm(state.partial) for norm in self.input_norms) if self.input_mix is None
                       else self.input_mix((*state.blocks, state.partial)))
        else:
            n0, n1, probs = cross_pass
            dests = self.input_mix.mix((*state.blocks, state.partial))
            if self.shared_qkv_cat is not None:
                h = dests[0]
                cat_q, cat_k, cat_v = self.shared_qkv_cat(h, n0, n1, probs)
                if self.shared_qkv_cat.use_fused_mix:
                    streams = self.shared_qkv_cat.mix_qkv(
                        h, n0, n1, probs,
                        self.gates[0].compute_fg(cat_q),
                        self.gates[1].compute_fg(cat_k),
                        self.gates[2].compute_fg(cat_v),
                    )
                else:
                    streams = (
                        self.gates[0].apply_with_cat(h, cat_q, n0, n1, probs),
                        self.gates[1].apply_with_cat(h, cat_k, n0, n1, probs),
                        self.gates[2].apply_with_cat(h, cat_v, n0, n1, probs),
                    )
            else:
                streams = tuple(gate(dest, n0, n1, probs)
                                for gate, dest in zip(self.gates[:3], dests, strict=True))
        _, attention_partial, attention_values = self.attention.attend(streams, m)
        attention_raw = self.reduction(attention_partial.to(self.reduction_dtype)).to(torch.bfloat16)
        attention, attention_previous = self.attention_convolution(attention_raw, m)
        sources = (*state.blocks, state.partial, attention) if self.layer % t.residual_block == 0 else (*state.blocks, state.partial)
        options = {} if self.layer % t.residual_block == 0 else dict(addend=attention)
        if cross_pass is None:
            mlp_input = self.mlp_mix(sources, **options)[0]
        else:
            n0, n1, probs = cross_pass
            mlp_input = self.gates[3](self.mlp_mix.mix(sources, **options)[0], n0, n1, probs)
        if self.splitk_router is not None:
            routes = self.splitk_router(mlp_input, num_valid_tokens=m.live_counts[:1])
        else:
            routes = self.router(self.router_projection(mlp_input.to(self.router_dtype)), num_valid_tokens=m.live_counts[:1])
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


class PreparedLoopDecoder(PreparedDecoder):
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
        self.visits = text.visits
        core_start, core_end = text.loop.core_start, text.loop.core_end
        bank_sources = (core_end - core_start) // text.residual_block
        if bank_sources != 2:
            raise ValueError(f"PreparedCrossPassRouter requires 2 core residual blocks, got {bank_sources}")
        if weights.cross_pass_queries_t is not None:
            self.cross_pass_router = PreparedCrossPassRouter(
                queries_all_t=weights.cross_pass_queries_t, tokens=tokens, hidden=text.hidden,
                eps=text.eps, examples=examples)
        else:
            queries_all = torch.cat([weights.layers[i].message_queries for i in range(core_start, core_end)], dim=0)
            self.cross_pass_router = PreparedCrossPassRouter(
                queries_all, tokens=tokens, hidden=text.hidden, eps=text.eps, examples=examples)
        num_streams = self.cross_pass_router.num_streams
        self.layers = [PreparedLoopLayer(index, weights.layers[index], text, pool, metadata, self.reduce,
            self.reduction_dtype, mode=mode, tokens=tokens, session=session, workspace=workspace,
            name=f"{name}.visit{visit}", examples=examples, cache_layer=visit, messages=pass_index > 0,
            core_offset=index - core_start if pass_index > 0 else 0,
            num_cross_pass_streams=num_streams if pass_index > 0 else 0)
            for visit, (index, pass_index) in enumerate(self.visits)]
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

    def __call__(self, metadata):
        w, t, m = self.weights, self.weights.config.text, metadata
        embedded = embedding_rows(w.embedding, m.token_ids, self.first_token)
        hidden = self.reduce(embedded.to(self.reduction_dtype)).to(torch.bfloat16)
        hidden = self.embedding_norm(hidden)
        if self.vision_state is not None and self.mode == "prefill":
            self.vision_state.inject(m, hidden)
        hidden = torch.where((m.positions >= 0)[:, None], hidden, 0)
        state, records, taps = ResidualState((), hidden), [], []
        prelude = cross_pass = None
        for visit, (layer, (index, pass_index)) in enumerate(zip(self.layers, self.visits, strict=True)):
            if t.loop is not None and index == t.loop.core_start:
                if pass_index == 0:
                    prelude = state
                else:
                    s0, s1 = (*state.blocks[len(prelude.blocks) + 1:], state.partial)
                    cross_pass = self.cross_pass_router(s0, s1)
                    state = prelude
            if w.draft is not None and visit in w.config.draft.taps:
                taps.append(state.partial)
            state, record = layer(state, m, cross_pass if pass_index > 0 else None)
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
