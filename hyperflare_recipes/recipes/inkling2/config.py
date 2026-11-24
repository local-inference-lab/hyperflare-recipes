# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validated Inkling 2 text, vision, draft, and loop geometry."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

ARCHITECTURE = "Inkling2ForConditionalGeneration"
TEXT_PREFIX = "model.language_model"
VISION_PREFIX = "model.vision_tower"
DRAFT_FILE = "mtp.safetensors"
WORLD_SIZE = 4
VOCABULARY_ALIGNMENT = 64
# Activation blocks of both expert formats (NVFP4 16, MXFP8 32) must stay inside one rank's slice.
EXPERT_ALIGNMENT = 32

_TOP_LEVEL = {"architectures", "model_type", "eos_token_id", "text_config", "vision_config", "mtp_config",
              "transformers_version", "torch_dtype", "quantization_config"}
_TEXT = {
    "model_max_length", "torch_dtype", "hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
    "num_key_value_heads", "head_dim", "d_rel", "rel_extent", "q_bias", "o_bias", "log_scaling_n_floor",
    "log_scaling_alpha", "rms_norm_eps", "pre_norm_pre_affine", "final_norm_pre_affine",
    "use_value_norm_no_gamma", "qk_l2_clip_threshold", "rel_l2_clip_threshold", "local_layer_ids",
    "sconv_kernel_size", "unpadded_vocab_size", "final_logit_softcapping", "swa_head_dim",
    "swa_num_attention_heads", "swa_num_key_value_heads", "sliding_window_size", "n_layers_per_attn_res_block",
    "attn_res_normalize_values", "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
    "intermediate_size", "mlp_gate_softcap", "mlp_up_softcap", "route_scale", "use_gate_bias", "loop_config",
}
_VISION = {"temporal_patch_size", "native_patch_size", "adapter_merge_size", "projection_hidden_dim",
           "projector_bias", "hidden_size", "num_hidden_layers", "num_attention_heads", "mlp_ratio",
           "posemb_grid_size", "layer_norm_eps"}
_DRAFT = {"hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
          "intermediate_size", "max_position_embeddings", "torch_dtype", "q_bias", "o_bias", "rms_norm_eps",
          "rope_theta", "local_layer_ids", "sliding_window_size", "block_size", "markov_rank",
          "markov_head_type", "target_layer_ids"}



def _integer(section, name, *, minimum=1):
    value = section.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _real(section, name, *, positive=True):
    value = section.get(name)
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite number")
    return float(value)


def _flag(section, name, required):
    if section.get(name) is not required:
        raise ValueError(f"Inkling 2 preparation requires {name}={required}")


def _layer_ids(section, name, layers):
    ids = section.get(name)
    if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < layers for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError(f"{name} must list unique layer indices below {layers}")
    return frozenset(ids)


def _known(section, keys, label):
    if not isinstance(section, dict):
        raise ValueError(f"{label} must be an object")
    unknown = section.keys() - keys
    if unknown:
        raise ValueError(f"{label} declares unsupported fields: {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class LoopConfig:
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int

    @classmethod
    def from_dict(cls, value, layers, block):
        _known(value, {"core_start", "core_end", "num_passes", "gate_rank", "is_varloop"}, "loop_config")
        config = cls(*(_integer(value, key) for key in
                       ("core_start", "core_end", "num_passes", "gate_rank")))
        # is_varloop records training-time variable-loop metadata; preparation always runs fixed num_passes.
        if type(value.get("is_varloop", False)) is not bool:
            raise ValueError("is_varloop must be a boolean")
        if not 0 < config.core_start < config.core_end <= layers:
            raise ValueError("loop core must follow a prelude and fit the decoder")
        if config.core_start % block or config.core_end % block:
            raise ValueError("loop core endpoints must align with attention-residual blocks")
        return config


@dataclass(frozen=True)
class TextConfig:
    """Decoder geometry. Local layers attend a sliding window; the others attend the full context."""

    hidden: int
    layers: int
    vocabulary: int
    valid_vocabulary: int
    heads: int
    kv_heads: int
    head_dim: int
    relative_width: int
    relative_extent: int
    window: int
    local_layers: frozenset[int]
    eps: float
    qk_clip: float
    relative_clip: float
    log_scaling_floor: int
    log_scaling_alpha: float
    convolution_width: int
    softcap: float
    residual_block: int
    experts: int
    experts_per_token: int
    intermediate: int
    gate_softcap: float
    up_softcap: float
    route_scale: float
    max_positions: int
    eos_token: int

    @property
    def loop(self):
        return None

    @classmethod
    def from_dict(cls, c, eos_token, *, looped=False):
        _known(c, _TEXT if looped else _TEXT - {"loop_config"}, "text_config")
        c = dict(c)
        if looped:
            c.setdefault("pre_norm_pre_affine", False)
            c.setdefault("final_norm_pre_affine", False)
        missing = _TEXT - {"loop_config"} - c.keys()
        if missing:
            raise ValueError(f"text_config is missing {', '.join(sorted(missing))}")
        if c["torch_dtype"] != "bfloat16":
            raise ValueError("Inkling 2 preparation requires a bfloat16 text model")
        for name, value in (("q_bias", False), ("o_bias", False), ("pre_norm_pre_affine", False),
                            ("final_norm_pre_affine", False), ("use_value_norm_no_gamma", True),
                            ("attn_res_normalize_values", True), ("use_gate_bias", True)):
            _flag(c, name, value)
        if c["n_shared_experts"] != 0:
            raise ValueError("Inkling 2 preparation implements routed experts without shared experts")
        layers = _integer(c, "num_hidden_layers")
        loop = (dict(loop=LoopConfig.from_dict(c.get("loop_config"), layers, c["n_layers_per_attn_res_block"]))
                if looped else {})
        text_type = MiniTextConfig if looped else cls
        config = text_type(
            hidden=_integer(c, "hidden_size"), layers=layers, vocabulary=_integer(c, "vocab_size"),
            valid_vocabulary=_integer(c, "unpadded_vocab_size"), heads=_integer(c, "num_attention_heads"),
            kv_heads=_integer(c, "num_key_value_heads"), head_dim=_integer(c, "head_dim"),
            relative_width=_integer(c, "d_rel"), relative_extent=_integer(c, "rel_extent"),
            window=_integer(c, "sliding_window_size"), local_layers=_layer_ids(c, "local_layer_ids", layers),
            eps=_real(c, "rms_norm_eps"), qk_clip=_real(c, "qk_l2_clip_threshold"),
            relative_clip=_real(c, "rel_l2_clip_threshold"), log_scaling_floor=_integer(c, "log_scaling_n_floor"),
            log_scaling_alpha=_real(c, "log_scaling_alpha", positive=False),
            convolution_width=_integer(c, "sconv_kernel_size", minimum=2),
            softcap=_real(c, "final_logit_softcapping"), residual_block=_integer(c, "n_layers_per_attn_res_block"),
            experts=_integer(c, "n_routed_experts"), experts_per_token=_integer(c, "num_experts_per_tok"),
            intermediate=_integer(c, "intermediate_size"), gate_softcap=_real(c, "mlp_gate_softcap"),
            up_softcap=_real(c, "mlp_up_softcap"), route_scale=_real(c, "route_scale"),
            max_positions=_integer(c, "model_max_length"), eos_token=eos_token, **loop)
        if (c["swa_head_dim"], c["swa_num_attention_heads"], c["swa_num_key_value_heads"]) != (
                config.head_dim, config.heads, config.kv_heads):
            raise ValueError("Inkling 2 preparation requires sliding-window layers with the global head geometry")
        geometry = (config.hidden, config.intermediate, config.experts)
        if (geometry != ((2304, 1152, 128) if looped else (4608, 2304, 384))
                or (config.gate_softcap, config.up_softcap) != (4.0, 25.0)):
            raise ValueError("Inkling 2 experts require Small or Mini geometry and SiTU caps 4/25")
        if config.heads % config.kv_heads:
            raise ValueError("query heads must divide evenly among KV heads")
        if not config.valid_vocabulary <= config.vocabulary or not 0 <= eos_token < config.valid_vocabulary:
            raise ValueError("the valid vocabulary and EOS token must lie inside the embedding table")
        if config.residual_block > layers:
            raise ValueError("the attention-residual block size must not exceed the layer count")
        # Routing is implemented and tested for the checkpoint's eight experts per token.
        if config.experts_per_token != 8 or config.experts < config.experts_per_token:
            raise ValueError("Inkling 2 preparation implements top-8 routing over at least eight experts")
        for name, width, block in (("hidden_size", config.hidden, EXPERT_ALIGNMENT),
                                   ("intermediate_size", config.intermediate, EXPERT_ALIGNMENT)):
            if width % block:
                raise ValueError(f"{name} must be a multiple of {block} for block-scaled experts")
        return config

    def __getitem__(self, name):
        aliases = {"hidden_size": "hidden", "rms_norm_eps": "eps", "vocab_size": "vocabulary",
                   "num_attention_heads": "heads", "num_key_value_heads": "kv_heads"}
        return getattr(self, aliases.get(name, name))

    def local(self, layer: int) -> bool:
        if not 0 <= layer < self.layers:
            raise ValueError(f"decoder layer {layer} is outside 0..{self.layers - 1}")
        return layer in self.local_layers

    def extent(self, layer: int) -> int:
        """Relative-bias distances of a layer: the window for local layers, `rel_extent` otherwise."""
        return self.window if self.local(layer) else self.relative_extent

    @property
    def global_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.layers) if i not in self.local_layers)

    @property
    def visits(self) -> tuple[tuple[int, int], ...]:
        if self.loop is None:
            return tuple((i, 0) for i in range(self.layers))
        c = self.loop
        return (tuple((i, 0) for i in range(c.core_start))
                + tuple((i, p) for p in range(c.num_passes) for i in range(c.core_start, c.core_end))
                + tuple((i, 0) for i in range(c.core_end, self.layers)))

    @property
    def cache_config(self) -> "TextConfig":
        """Give each executed layer visit independent KV and convolution storage."""
        if self.loop is None:
            return self
        return replace(self, layers=len(self.visits), loop=None,
                       local_layers=frozenset(v for v, (i, _) in enumerate(self.visits) if self.local(i)))

    def prefix(self, layers: int) -> "TextConfig":
        """The geometry of the first `layers` decoder layers, which development preparation runs alone."""
        if type(layers) is not int or not 1 <= layers <= self.layers:
            raise ValueError(f"a decoder-layer prefix holds 1 through {self.layers} layers")
        if self.loop is not None and layers != self.layers:
            raise ValueError("a looped decoder requires all checkpoint layers")
        return replace(self, layers=layers, local_layers=frozenset(i for i in self.local_layers if i < layers))


@dataclass(frozen=True)
class MiniTextConfig(TextConfig):
    """Loop metadata belongs only to Mini's preparation identity."""

    loop: LoopConfig | None = None


@dataclass(frozen=True)
class VisionConfig:
    """Image tower geometry: native patches, merged `merge x merge` and projected to the text width."""

    hidden: int
    layers: int
    heads: int
    mlp_hidden: int
    patch: int
    temporal_patch: int
    merge: int
    projection_hidden: int
    position_grid: int
    eps: float

    @classmethod
    def from_dict(cls, c):
        _known(c, _VISION, "vision_config")
        missing = _VISION - c.keys()
        if missing:
            raise ValueError(f"vision_config is missing {', '.join(sorted(missing))}")
        _flag(c, "projector_bias", False)
        hidden, ratio = _integer(c, "hidden_size"), _real(c, "mlp_ratio")
        if not float(hidden * ratio).is_integer():
            raise ValueError("vision MLP width must be an integer multiple of the hidden size")
        config = cls(hidden=hidden, layers=_integer(c, "num_hidden_layers"), heads=_integer(c, "num_attention_heads"),
            mlp_hidden=int(hidden * ratio), patch=_integer(c, "native_patch_size"),
            temporal_patch=_integer(c, "temporal_patch_size"), merge=_integer(c, "adapter_merge_size"),
            projection_hidden=_integer(c, "projection_hidden_dim"), position_grid=_integer(c, "posemb_grid_size"),
            eps=_real(c, "layer_norm_eps"))
        if config.hidden % config.heads:
            raise ValueError("vision heads must divide the vision hidden size")
        return config


@dataclass(frozen=True)
class DraftConfig:
    """Block drafter: a non-causal stack over target taps with a recurrent Markov head."""

    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    eps: float
    rope_theta: float
    local_layers: frozenset[int]
    window: int
    block: int
    markov_rank: int
    taps: tuple[int, ...]
    max_positions: int

    @property
    def theta(self):
        return self.rope_theta

    @property
    def auxiliary_layers(self):
        return self.taps

    @property
    def block_size(self):
        return self.block

    @property
    def noise_token(self):
        return 0

    @property
    def text(self):
        return {"hidden_size": self.hidden, "rms_norm_eps": self.eps,
                "intermediate_size": self.intermediate, "num_attention_heads": self.heads,
                "num_key_value_heads": self.kv_heads, "head_dim": self.head_dim}

    @classmethod
    def from_dict(cls, c, text: TextConfig):
        _known(c, _DRAFT, "mtp_config")
        missing = _DRAFT - c.keys()
        if missing:
            raise ValueError(f"mtp_config is missing {', '.join(sorted(missing))}")
        if c["torch_dtype"] != "bfloat16":
            raise ValueError("the Inkling 2 drafter must be bfloat16")
        _flag(c, "q_bias", False)
        _flag(c, "o_bias", False)
        if c["markov_head_type"] != "rnn_hidden":
            raise ValueError("Inkling 2 preparation implements the rnn_hidden Markov head")
        layers = _integer(c, "num_hidden_layers")
        taps = c["target_layer_ids"]
        # The last tap is the sentinel for the target's final normalized hidden state.
        if (not isinstance(taps, list) or len(taps) < 2 or any(type(i) is not int for i in taps)
                or taps != sorted(set(taps)) or taps[-1] != len(text.visits) or not 0 <= taps[0]):
            raise ValueError("target_layer_ids must be sorted unique target layers ending with the layer count")
        config = cls(hidden=_integer(c, "hidden_size"), layers=layers, heads=_integer(c, "num_attention_heads"),
            kv_heads=_integer(c, "num_key_value_heads"), head_dim=_integer(c, "head_dim"),
            intermediate=_integer(c, "intermediate_size"), eps=_real(c, "rms_norm_eps"),
            rope_theta=_real(c, "rope_theta"), local_layers=_layer_ids(c, "local_layer_ids", layers),
            window=_integer(c, "sliding_window_size"), block=_integer(c, "block_size", minimum=2),
            markov_rank=_integer(c, "markov_rank"), taps=tuple(taps),
            max_positions=_integer(c, "max_position_embeddings"))
        if config.hidden != text.hidden or config.heads % config.kv_heads:
            raise ValueError("the drafter must share the target width and group its query heads evenly")
        return config


@dataclass(frozen=True)
class ModelConfig:
    text: TextConfig
    vision: VisionConfig
    draft: DraftConfig
    quantization: dict

    @property
    def hidden(self):
        return self.text.hidden

    @property
    def vocabulary(self):
        return self.text.vocabulary

    @classmethod
    def load(cls, snapshot):
        config = json.loads((Path(snapshot) / "config.json").read_text())
        precision = Path(snapshot) / "hf_quant_config.json"
        if config.get("model_type") == "inkling2_loop" and precision.is_file():
            # The public config's NVFP4 summary does not describe Mini's MXFP8 layers.
            config["quantization_config"] = json.loads(precision.read_text())
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config):
        mini = config.get("model_type") == "inkling2_loop"
        _known(config, _TOP_LEVEL | ({"dtype"} if mini else set()), "config.json")
        expected_architecture = "Inkling2LoopForConditionalGeneration" if mini else ARCHITECTURE
        if (config.get("model_type") not in ("inkling2", "inkling2_loop")
                or config.get("architectures") != [expected_architecture]):
            raise ValueError(f"Inkling 2 preparation requires {expected_architecture}")
        eos = config.get("eos_token_id")
        if type(eos) is not int:
            raise ValueError("eos_token_id must be one token ID")
        text_config, vision_config = config.get("text_config"), config.get("vision_config")
        if mini:
            if config.get("dtype", "bfloat16") != "bfloat16":
                raise ValueError("Inkling 2 Mini requires bfloat16 dtype")
            text_config, vision_config = dict(text_config), dict(vision_config)
            if text_config.pop("model_type", "inkling2_loop_text") != "inkling2_loop_text":
                raise ValueError("Inkling 2 Mini requires inkling2_loop_text")
            if vision_config.pop("model_type", "inkling2_vision") != "inkling2_vision":
                raise ValueError("Inkling 2 Mini requires inkling2_vision")
            if vision_config.pop("text_hidden_size", 2304) != 2304:
                raise ValueError("Inkling 2 Mini vision requires text_hidden_size=2304")
        text = TextConfig.from_dict(text_config, eos, looped=mini)
        quantization = config.get("quantization_config", {})
        if mini:
            if not quantization.get("model", {}).get("parameter_groups"):
                raise ValueError("Inkling 2 Mini requires hf_quant_config.json parameter_groups")
        elif quantization.get("quant_algo") != "NVFP4":
            raise ValueError("Inkling 2 preparation requires the checkpoint's NVFP4 weights")
        return cls(text, VisionConfig.from_dict(vision_config),
                   DraftConfig.from_dict(config.get("mtp_config"), text), quantization)

    def validate_shard(self, shard):
        required = 1 if self.text.loop is not None else WORLD_SIZE
        if shard.world_size != required:
            raise ValueError(f"this Inkling 2 checkpoint requires tensor parallelism of {required}")
        for width in (self.text.heads, self.text.kv_heads, self.draft.heads, self.draft.kv_heads):
            shard.partition(width)
        shard.partition(self.text.intermediate, alignment=EXPERT_ALIGNMENT)
        shard.partition(self.draft.intermediate, alignment=EXPERT_ALIGNMENT)
        shard.padded_partition(self.text.vocabulary, alignment=VOCABULARY_ALIGNMENT)
