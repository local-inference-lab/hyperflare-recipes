# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared FP32 attention-residual mixtures with fused downstream RMSNorm."""

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.inkling2_vec_residual import (
    PreparedVecResidualMix,
    supports_vec_residual_mix,
)
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.triton import launch_triton, prepare_triton


@triton.jit
def _mixture(Source0, Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8,
             Queries, Output0, Output1, Output2, Addend, STRIDES: tl.constexpr, HIDDEN: tl.constexpr,
             SOURCES: tl.constexpr, STREAMS: tl.constexpr, ADD: tl.constexpr, ADD_STRIDE: tl.constexpr,
             EPS: tl.constexpr, BLOCK: tl.constexpr):
    Sources = (Source0, Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8)
    Outputs = (Output0, Output1, Output2)
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    valid = columns < HIDDEN
    for stream in tl.static_range(STREAMS):
        query = tl.load(Queries + stream * HIDDEN + columns, valid, other=0.)
        maximum = tl.full((), -float('inf'), tl.float32)
        denominator = tl.full((), 0., tl.float32)
        mixture = tl.full((BLOCK,), 0., tl.float32)
        for source in tl.static_range(SOURCES):
            value = tl.load(Sources[source] + row * STRIDES[source] + columns, valid, other=0.).to(tl.float32)
            if ADD and source == SOURCES - 1:
                value += tl.load(Addend + row * ADD_STRIDE + columns, valid, other=0.).to(tl.float32)
            inverse = tl.rsqrt(tl.sum(value * value, 0) / HIDDEN + EPS)
            score = tl.sum(value * query, 0) * inverse
            peak = tl.maximum(maximum, score)
            rescale = tl.exp(maximum - peak)
            probability = tl.exp(score - peak)
            denominator = denominator * rescale + probability
            mixture = mixture * rescale + value * (inverse * probability)
            maximum = peak
        tl.store(Outputs[stream] + row * HIDDEN + columns, mixture / denominator, valid)


@triton.jit
def _mixture_norm(Source0, Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8,
                  Queries, Norm0, Norm1, Norm2, Output0, Output1, Output2, Addend,
                  STRIDES: tl.constexpr, HIDDEN: tl.constexpr, SOURCES: tl.constexpr,
                  STREAMS: tl.constexpr, ADD: tl.constexpr, ADD_STRIDE: tl.constexpr,
                  EPS: tl.constexpr, BLOCK: tl.constexpr):
    Sources = (Source0, Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8)
    Norms = (Norm0, Norm1, Norm2)
    Outputs = (Output0, Output1, Output2)
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    valid = columns < HIDDEN
    for stream in tl.static_range(STREAMS):
        query = tl.load(Queries + stream * HIDDEN + columns, valid, other=0.)
        maximum = tl.full((), -float('inf'), tl.float32)
        denominator = tl.full((), 0., tl.float32)
        mixture = tl.full((BLOCK,), 0., tl.float32)
        for source in tl.static_range(SOURCES):
            value = tl.load(Sources[source] + row * STRIDES[source] + columns, valid, other=0.).to(tl.float32)
            if ADD and source == SOURCES - 1:
                value += tl.load(Addend + row * ADD_STRIDE + columns, valid, other=0.).to(tl.float32)
            inverse = tl.rsqrt(tl.sum(value * value, 0) / HIDDEN + EPS)
            score = tl.sum(value * query, 0) * inverse
            peak = tl.maximum(maximum, score)
            rescale = tl.exp(maximum - peak)
            probability = tl.exp(score - peak)
            denominator = denominator * rescale + probability
            mixture = mixture * rescale + value * (inverse * probability)
            maximum = peak
        mixed = mixture / denominator
        out_inv = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
        gamma = tl.load(Norms[stream] + columns, valid, other=0.).to(tl.float32)
        tl.store(Outputs[stream] + row * HIDDEN + columns, (mixed * out_inv * gamma).to(tl.bfloat16), valid)


class PreparedResidualMix:
    def __init__(self, source, queries, norms, *, sources, eps, session, name, addend=False, examples=None):
        if not 1 <= sources <= 9 or not 1 <= len(norms) <= 3:
            raise ValueError('attention-residual mixtures require one to nine sources and one to three output streams')
        if queries.dtype != torch.float32 or queries.shape != (len(norms), source.shape[1]):
            raise ValueError('attention-residual queries must contain one FP32 row per output stream')
        self.queries, self.norms, self.sources = queries, tuple(norms), sources
        self.eps, self.streams, self.addend = eps, len(norms), addend
        self.tokens, self.hidden = source.shape
        self.options = dict(HIDDEN=self.hidden, SOURCES=sources, STREAMS=self.streams, ADD=addend,
            ADD_STRIDE=source.stride(0), EPS=eps, BLOCK=triton.next_power_of_2(self.hidden),
            STRIDES=(source.stride(0),) * sources, num_warps=8, enable_fp_fusion=False)
        declarations = {} if examples is None else examples
        mixtures = tuple(declaration_tensor(declarations, f"inkling.residual.mixture{index}",
            tuple(source.shape), torch.float32, source.device) for index in range(self.streams))
        outputs = tuple(declaration_tensor(declarations, f"inkling.residual.output{index}",
            tuple(source.shape), torch.bfloat16, source.device) for index in range(self.streams))
        arguments = ((source,) * 9 + (queries,) + mixtures
                     + (mixtures[0],) * (3 - self.streams) + (source,))
        self.compiled = prepare_triton(_mixture, arguments, (self.tokens, 1, 1), **self.options).compiled
        self._vec_mix = None
        self.fused_compiled = None
        if (
            supports_vec_residual_mix(self.tokens, self.hidden, self.sources, source.device)
            and source.dtype == torch.bfloat16
            and source.is_contiguous()
            and queries.is_contiguous()
            and queries.data_ptr() % 16 == 0
        ):
            self._vec_mix = PreparedVecResidualMix(
                source, queries, self.norms, sources=sources, eps=eps,
                addend=addend, session=session, name=name,
            )
            return
        norm_args = self.norms + (self.norms[0],) * (3 - self.streams)
        out_args = outputs + (outputs[0],) * (3 - self.streams)
        fused_args = (source,) * 9 + (queries,) + norm_args + out_args + (source,)
        self.fused_compiled = prepare_triton(
            _mixture_norm, fused_args, (self.tokens, 1, 1), **self.options
        ).compiled

    def mix(self, sources, *, addend=None):
        if len(sources) != self.sources:
            raise ValueError('attention-residual source count differs from the prepared layer')
        if (addend is not None) != self.addend:
            raise ValueError('attention-residual addend differs from the prepared layer')
        output = tuple(torch.empty((self.tokens, self.hidden), dtype=torch.float32,
            device=sources[0].device) for _ in range(self.streams))
        arguments = (sources + (sources[0],) * (9 - self.sources) + (self.queries,) + output
                     + (output[0],) * (3 - self.streams) + (sources[0] if addend is None else addend,))
        launch_triton(_mixture, self.compiled,
            arguments, (self.tokens, 1, 1), **self.options)
        return output

    def __call__(self, sources, *, addend=None):
        if len(sources) != self.sources:
            raise ValueError('attention-residual source count differs from the prepared layer')
        if (addend is not None) != self.addend:
            raise ValueError('attention-residual addend differs from the prepared layer')
        if self._vec_mix is not None:
            return self._vec_mix(sources, addend=addend)
        outputs = tuple(torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16,
            device=sources[0].device) for _ in range(self.streams))
        norm_args = self.norms + (self.norms[0],) * (3 - self.streams)
        out_args = outputs + (outputs[0],) * (3 - self.streams)
        arguments = (sources + (sources[0],) * (9 - self.sources) + (self.queries,)
                     + norm_args + out_args + (sources[0] if addend is None else addend,))
        launch_triton(_mixture_norm, self.fused_compiled, arguments, (self.tokens, 1, 1), **self.options)
        return outputs
