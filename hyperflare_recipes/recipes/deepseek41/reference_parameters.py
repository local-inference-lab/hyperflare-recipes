# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the complete reference parameter tree with explicit storage owners."""

from dataclasses import replace

import torch

from ignition.preparation.tensor_metadata import record_numeric_tensor


class ReferenceParameters:
    """Load text, draft and vision operands through the reference name/layout mapping.

    Routed experts and Engram tables remain meta declarations. Their execution
    bindings must supply residency-owned weights and native table delivery before
    a reference forward can run. This object does not construct a serving graph.
    """

    def __init__(self, checkpoint):
        from deepinfer.hugging_face import cover_tied_parameters, model_params, validate_hf_shapes
        from deepinfer.hugging_face.weights import build_coverage
        from deepinfer.model_executor.models.v4_modules.builder import V41Builder
        from deepinfer.model_executor.parallel_utils.parallel_state import PG, LocalProcessGroup

        if torch.distributed.is_initialized() or not isinstance(PG.world, LocalProcessGroup):
            raise ValueError("reference parameter preparation requires explicit single-device process geometry")
        self.checkpoint = checkpoint
        self.config = checkpoint.model_config()
        with torch.device("meta"):
            self.model = V41Builder(self.config).build_model(vision_device="meta")
        params = model_params(self.model)
        logical_map = {name: checkpoint.source.tensors[
            name + ".mxfp4_csf_fixed" if name in checkpoint.compressed_scales else name].shard
            for name in checkpoint.tensors}
        coverage = build_coverage(logical_map, gpt_cfg=self.config.gpt)
        self.coverage = cover_tied_parameters(coverage, set(params), n_layer=self.config.gpt.num_hidden_layers)
        if missing := set(params) - self.coverage.model_names():
            raise ValueError(f"reference checkpoint lacks model parameters: {sorted(missing)}")
        validate_hf_shapes(params, self.coverage, hf_dir=checkpoint.source.snapshot)
        if not all(value.is_meta for value in params.values()):
            raise ValueError("reference construction must leave learned parameters unallocated")
        self.loaded = False
        self.assembled = False

    @torch.no_grad()
    def load_ordinary(self, device):
        """Materialize complete ordinary operands without allocating expert/table copies."""
        from deepinfer.hugging_face import model_params
        from deepinfer.hugging_face.weights import load_covered_params
        from deepinfer.model_executor.models.v4_modules.containers import materialize_hf_parameter

        if self.loaded:
            raise RuntimeError("reference ordinary parameters have already been loaded")
        params = model_params(self.model)
        ordinary = set(self.coverage.tensors) | set(self.coverage.aliases)
        replacements = {}
        for name, value in params.items():
            if name not in ordinary:
                continue
            if name in self.coverage.aliases:
                continue
            replacements[name] = materialize_hf_parameter(value, device=device)
        for name, source in self.coverage.aliases.items():
            original = params[name]
            owner = replacements[source]
            alias = torch.nn.Parameter(owner.data, requires_grad=False)
            for key, value in original.__dict__.items():
                if key != "sf":
                    setattr(alias, key, value)
            if hasattr(owner, "sf"):
                alias.sf = owner.sf
            replacements[name] = alias
        for name, value in replacements.items():
            parent, leaf = name.rsplit(".", 1)
            module = self.model.get_submodule(parent)
            if leaf in module._buffers:
                module._buffers[leaf] = value
            else:
                setattr(module, leaf, value)
        coverage = replace(self.coverage, expert_tensors={}, engram_tables={})
        result = load_covered_params(self.model, coverage, hf_dir=self.checkpoint.source.snapshot)
        params = model_params(self.model)
        for name in ordinary:
            tensor = params[name]
            if tensor.is_meta:
                raise RuntimeError(f"reference ordinary parameter was not materialized: {name}")
            record_numeric_tensor(tensor)
            scales = (() if not hasattr(tensor, "sf") else (tensor.sf,))
            scales += getattr(tensor, "ue8m0_sf", ())
            for scale in scales:
                if scale.is_meta:
                    raise RuntimeError(f"reference scale was not materialized: {name}")
                record_numeric_tensor(scale)
        for name in set(self.coverage.expert_tensors) | set(self.coverage.engram_tables):
            if not params[name].is_meta:
                raise RuntimeError(f"reference loader allocated residency-owned storage: {name}")
        for name, source in self.coverage.aliases.items():
            if params[name].data_ptr() != params[source].data_ptr():
                raise RuntimeError(f"reference tied parameter lost its storage owner: {name}")
        self.model.vision_pipeline.aligner.visual_encoder.materialize_rotary(device)
        self.loaded = True
        return result
