# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct loading of Inkling 2 vocabulary, whole-head attention and expert intermediate shards."""

from dataclasses import dataclass, replace
from functools import cached_property

import torch

from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.persistent import persistent_copy, persistent_zeros

from .attention import AttentionWeights, attention_weights
from .config import ModelConfig, TEXT_PREFIX
from ignition.preparation.rebuild_manifest import weight_container


def scaled_query(query, hidden):
    return persistent_copy(query.float().reshape(-1, hidden) * hidden ** -0.5)


@weight_container(('attention', 'experts', 'router', 'correction', 'attention_convolution', 'mlp_convolution', 'input_norms', 'mlp_norm', 'attention_query', 'mlp_query', 'controllers', 'message_queries'))
@dataclass(frozen=True)
class LayerWeights:
    attention: AttentionWeights
    experts: object
    router: torch.Tensor
    correction: torch.Tensor
    attention_convolution: torch.Tensor
    mlp_convolution: torch.Tensor
    input_norms: tuple[torch.Tensor, ...]
    mlp_norm: torch.Tensor
    attention_query: torch.Tensor | None
    mlp_query: torch.Tensor
    controllers: object | None = None
    message_queries: torch.Tensor | None = None


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    layers: tuple[LayerWeights, ...]
    embedding: torch.Tensor
    embedding_norm: torch.Tensor
    final_norm: torch.Tensor
    output_query: torch.Tensor
    unembedding: object
    allocated_bytes: int
    draft: object | None = None
    vision: object | None = None
    cross_pass_queries_t: torch.Tensor | None = None
    draft_head: object | None = None

    @property
    def rank(self):
        return self.shard.rank

    @property
    def world(self):
        return self.shard.world_size

    @property
    def head(self):
        return self.unembedding

    @property
    def embed_norm(self):
        return self.embedding_norm

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, drafting=True, vision=True, layers=None,
             draft_head_quantization=None, markov_head_quantization=None,
             speculative_method="dspark", draft_snapshot=None, online_mxfp8=False):
        before = torch.cuda.memory_allocated(device)
        loader_type = InklingWeightLoader
        if "model.llm.embed.weight" in manifest.tensors:
            from .mini_weights import MiniWeightLoader
            loader_type = MiniWeightLoader
        with loader_type(manifest, device, shard=shard, progress=progress, vision=vision) as loader:
            c = loader.config
            count = c.text.layers if layers is None else layers
            if type(count) is not int or not 1 <= count <= c.text.layers:
                raise ValueError("decoder layer count must fit the checkpoint")
            first, last = shard.padded_partition(c.text.vocabulary, alignment=64)
            def vocabulary_rows(name):
                value = persistent_zeros((last - first, c.text.hidden), dtype=torch.bfloat16, device=device)
                stop = min(last, c.text.vocabulary)
                loader._copy(value[:stop - first], loader._sources[name][first:stop])
                loader._session.flush()
                return value
            embedding = vocabulary_rows(TEXT_PREFIX + ".embed_tokens.weight")
            head = vocabulary_rows("lm_head.weight")
            loaded = tuple(loader.load_layer(index) for index in range(count))
            norm = loader.load_tensor(TEXT_PREFIX + ".norm.weight")
            embed_norm = loader.load_tensor(TEXT_PREFIX + ".embed_norm.weight")
            query = scaled_query(loader.load_tensor(TEXT_PREFIX + ".final_input.query"), c.text.hidden)
            tower = None
            if vision:
                import json
                from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
                from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
                from .vision import VisionWeights
                raw = json.loads((manifest.snapshot / "config.json").read_text())
                with torch.device("meta"):
                    module = Inkling2VisionModel(Inkling2VisionConfig.from_dict(
                        raw["vision_config"], **(dict(text_hidden_size=c.text.hidden) if c.text.loop is not None else {})))
                tower = VisionWeights.load(module, loader)
        draft = None
        draft_head = None
        if drafting:
            if speculative_method == "dflash2":
                from hyperflare_recipes.components.dflash2_weights import DraftWeights as DFlashWeights
                from hyperflare_recipes.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
                from .dflash import load_config as load_dflash_config

                dflash_config = load_dflash_config(draft_snapshot, c.text)
                c = replace(c, draft=dflash_config)
                draft = DFlashWeights.load(draft_snapshot, dflash_config, device, shard,
                                           progress=progress, online_mxfp8=online_mxfp8)
                if draft_head_quantization is not None:
                    draft_head = {"nvfp4": nvfp4_draft_head, "mxfp8": mxfp8_draft_head}[draft_head_quantization](head)
            else:
                from .dspark_weights import DraftWeights
                draft = DraftWeights.load(manifest.snapshot, c, device, shard, target_head=head, progress=progress,
                                          draft_head_quantization=draft_head_quantization,
                                          markov_head_quantization=markov_head_quantization)
        cross_pass_queries_t = None
        if c.text.loop is not None and count >= c.text.loop.core_end:
            core = range(c.text.loop.core_start, c.text.loop.core_end)
            cross_pass_queries_t = persistent_copy(
                torch.cat([loaded[i].message_queries for i in core], dim=0).t().contiguous()
            )
        return cls(c, shard, loaded, embedding, embed_norm, norm, query, head,
                   torch.cuda.memory_allocated(device) - before, draft, tower, cross_pass_queries_t,
                   draft_head)


class InklingWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, vision=True):
        super().__init__(manifest, device, progress=progress)
        self.config = ModelConfig.load(manifest.snapshot)
        self.config.validate_shard(shard)
        self.shard, self.vision = shard, vision

    @cached_property
    def quantization(self):
        return {name.removesuffix(".weight"): {"quant_algo": "NVFP4", "group_size": 16}
                for name in self.manifest.tensors
                if name.endswith(".weight") and name.removesuffix(".weight") + ".weight_scale_2" in self.manifest.tensors}

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if self.vision or not name.startswith("model.vision_tower.")}

    def load_layer(self, index):
        from .experts import load_experts

        c, s = self.config.text, self.shard
        prefix = f"{TEXT_PREFIX}.layers.{index}"
        a = prefix + ".self_attn"
        q0, q1 = s.partition(c.heads)
        k0, k1 = s.partition(c.kv_heads)
        query_rows = (q0 * c.head_dim, q1 * c.head_dim)
        key_rows = (k0 * c.head_dim, k1 * c.head_dim)
        placed = {}
        for role, name, extent in (("query", "q_proj", query_rows), ("output_gate", "gate_proj", query_rows),
                                   ("key", "k_proj", key_rows), ("value", "v_proj", key_rows),
                                   ("relative_query", "r_proj", (q0 * c.relative_width, q1 * c.relative_width)),
                                   ("sink", "sink_proj", (q0, q1))):
            placed[role] = self.load_linear(a + "." + name, rows=extent, temporary=True)
        placed["output"] = self.load_linear(a + ".o_proj", columns=query_rows)
        for role, name in (("query_clip_scale", "q_l2_clip.scale"),
                           ("relative_clip_scale", "rel_l2_clip.scale")):
            placed[role] = self.load_tensor(a + "." + name, extent=slice(q0, q1))
        placed["relative_projection"] = self.load_tensor(a + ".rel_logits_proj.proj")
        if c.local(index):
            for kind in ("key", "value"):
                n = "k" if kind == "key" else "v"
                placed[kind + "_convolution"] = self.load_tensor(
                    a + f".{n}_sconv.conv1d.weight", extent=slice(*key_rows))
            norms = (self.load_tensor(prefix + ".input_layernorm.weight"),)
        else:
            for kind in ("key", "value"):
                n = "k" if kind == "key" else "v"
                placed[kind + "_convolution_static"] = self.load_tensor(
                    a + f".{n}_sconv.static_weight", extent=(slice(None), slice(*key_rows)))
                placed[kind + "_convolution_dynamic"] = self.load_tensor(
                    a + f".{n}_sconv.weight_proj",
                    extent=slice(k0 * c.convolution_width, k1 * c.convolution_width), temporary=True)
            norms = tuple(self.load_tensor(prefix + f".{n}_input_layernorm.weight") for n in ("q", "k", "v"))
        attention = attention_weights(placed, local=c.local(index), heads=q1 - q0, kv_heads=k1 - k0,
                                      dim=c.head_dim, width=c.convolution_width,
                                      separate_inputs=c.loop is not None and c.loop.core_start <= index < c.loop.core_end)
        attention_query = None if index == 0 else scaled_query(
            self.load_tensor(prefix + ".attention_input.query"), c.hidden)
        return LayerWeights(attention, load_experts(self, index, shard=s),
            self.load_tensor(prefix + ".mlp.gate.weight"),
            self.load_tensor(prefix + ".mlp.gate.e_score_correction_bias"),
            self.load_tensor(prefix + ".post_attention_convolution.conv1d.weight"),
            self.load_tensor(prefix + ".post_mlp_convolution.conv1d.weight"), norms,
            self.load_tensor(prefix + ".post_attention_layernorm.weight"), attention_query,
            scaled_query(self.load_tensor(prefix + ".mlp_input.query"), c.hidden))
