# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native FP8, MXFP4-CSF SiTU experts, FlashIndexer, and bundled DSpark transfers for ERNIE 5.5."""

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from hyperflare_recipes.components.block_fp8_weights import BlockFP8WeightLoader
from hyperflare_recipes.components.markov_sampling import fp8_markov_head
from hyperflare_recipes.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
from hyperflare_recipes.components.weights import MLPWeights, _aligned_range
from ignition.preparation.persistent import (
    allocated_tensor_bytes,
    persistent_cat,
    persistent_copy,
    persistent_empty,
    persistent_zeros,
)
from ignition.preparation.rebuild_manifest import weight_container

from .config import ModelConfig
from .contract import VOCABULARY


@weight_container(
    ("router", "correction", "scale", "fc1_latent", "latent_norm", "fc2_latent", "experts", "shared")
)
@dataclass(frozen=True)
class LatentExpertWeights:
    router: torch.Tensor
    correction: torch.Tensor
    scale: torch.Tensor
    fc1_latent: object
    latent_norm: torch.Tensor
    fc2_latent: object
    experts: object
    shared: MLPWeights


@weight_container(("attention", "mlp", "experts", "residual"))
@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    mlp: MLPWeights | None
    experts: LatentExpertWeights | None
    residual: dict
    is_dsa: bool = False


@weight_container(
    (
        "layers",
        "main_projection",
        "context_kv",
        "main_norm",
        "head_mix",
        "norm",
        "markov_embedding",
        "markov_head",
        "confidence_head",
        "draft_head",
    )
)
@dataclass(frozen=True)
class DSparkWeights:
    layers: tuple[LayerWeights, ...]
    main_projection: object
    context_kv: object
    main_norm: torch.Tensor
    head_mix: dict
    norm: torch.Tensor
    markov_embedding: torch.Tensor
    markov_head: object
    confidence_head: torch.Tensor | None = None
    draft_head: object | None = None

    @classmethod
    def load(
        cls,
        loader,
        *,
        confidence=False,
        vocabulary_parallel_sampling=False,
        markov_head_quantization=None,
    ):
        config = loader.config.draft
        markov_name = "model.mtp.markov_head.markov_w2.weight"
        temporary = markov_head_quantization in ("nvfp4", "fp8")
        # Stored as [rank, vocab]; embedding lookup requires [vocab, rank].
        w1 = loader.load_tensor("model.mtp.markov_head.markov_w1.weight", temporary=True)
        markov_embedding = persistent_copy(w1.t().contiguous())
        del w1
        markov_head = (
            loader.load_padded_tensor(
                markov_name,
                rows=loader.shard.padded_partition(VOCABULARY, alignment=64),
                temporary=temporary,
            )
            if vocabulary_parallel_sampling
            else loader.load_tensor(markov_name, temporary=temporary)
        )
        confidence_weight = None
        if confidence and "model.mtp.confidence_head.projection.weight" in loader._sources:
            confidence_weight = loader.load_tensor("model.mtp.confidence_head.projection.weight")
        return cls(
            layers=tuple(loader.load_layer(i, draft=True) for i in range(config.layers)),
            main_projection=loader.load_linear("model.mtp.feature_projection"),
            context_kv=loader.load_linear_group(
                tuple(f"model.mtp.{i}.self_attn.linear_kv_proj" for i in range(config.layers))
            ),
            main_norm=loader.load_tensor("model.mtp.feature_norm.weight"),
            head_mix=loader.load_head_mix("model.mtp.mhc_contract."),
            norm=loader.load_tensor("model.mtp.norm.weight"),
            markov_embedding=markov_embedding,
            markov_head=markov_head,
            confidence_head=confidence_weight,
        )


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    head_mix: dict
    multimax: tuple[tuple[float, ...], tuple[float, ...]]
    layers: tuple[LayerWeights, ...]
    draft: DSparkWeights | None
    allocated_bytes: int

    @classmethod
    def load(
        cls,
        manifest,
        device,
        *,
        shard,
        progress=None,
        dspark=True,
        confidence=False,
        expert_scale_compression=None,
        expert_activation_mode="a8",
        draft_head_quantization=None,
        markov_head_quantization=None,
        vocabulary_parallel_sampling=False,
    ):
        config = ModelConfig.load(manifest.metadata_path)
        before = allocated_tensor_bytes(device)
        with ErnieWeightLoader(
            manifest,
            device,
            shard=shard,
            progress=progress,
            dspark=dspark,
            expert_scale_compression=expert_scale_compression,
            expert_activation_mode=expert_activation_mode,
        ) as loader:
            embedding, norm = loader.load_tensors(("model.embed_tokens.weight", "model.norm.weight"))
            head = loader.load_padded_tensor(
                "lm_head.weight", rows=shard.padded_partition(VOCABULARY, alignment=64)
            )
            head_mix = loader.load_head_mix("model.mhc_contract.")
            ranges = loader.load_tensor("model.lm_head.multimax_ranges", dtype=torch.float32, temporary=True)
            ts = loader.load_tensor("model.lm_head.multimax_ts", dtype=torch.float32, temporary=True)
            multimax = (
                tuple(float(v) for v in ranges.cpu().tolist()),
                tuple(float(v) for v in ts.cpu().tolist()),
            )
            del ranges, ts
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = (
                DSparkWeights.load(
                    loader,
                    confidence=confidence,
                    vocabulary_parallel_sampling=vocabulary_parallel_sampling,
                    markov_head_quantization=markov_head_quantization,
                )
                if dspark
                else None
            )
            if draft is not None:
                if draft_head_quantization == "nvfp4":
                    draft = replace(draft, draft_head=nvfp4_draft_head(head))
                elif draft_head_quantization == "mxfp8":
                    draft = replace(draft, draft_head=mxfp8_draft_head(head))
                if markov_head_quantization == "nvfp4":
                    draft = replace(draft, markov_head=nvfp4_draft_head(draft.markov_head))
                elif markov_head_quantization == "fp8":
                    draft = replace(draft, markov_head=fp8_markov_head(draft.markov_head))
        return cls(
            config=config,
            shard=shard,
            embedding=embedding,
            norm=norm,
            head=head,
            head_mix=head_mix,
            multimax=multimax,
            layers=layers,
            draft=draft,
            allocated_bytes=allocated_tensor_bytes(device) - before,
        )


def _fp32_scales_to_ue8m0(scales_f32: torch.Tensor, prefix: str) -> torch.Tensor:
    bits = scales_f32.float().view(torch.int32)
    if bool((bits & ~0x7F800000).any()):
        raise ValueError(f"{prefix}: expected positive power-of-two FP32 block scales")
    return (bits >> 23).to(torch.uint8)


class ErnieWeightLoader(BlockFP8WeightLoader):
    def __init__(
        self,
        manifest,
        device,
        *,
        shard,
        progress=None,
        prefixes=None,
        dspark=True,
        expert_scale_compression=None,
        expert_activation_mode="a8",
    ):
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("ERNIE 5.5 expert scale compression must be None or 'csf'")
        if expert_activation_mode not in ("a8", "a16"):
            raise ValueError("ERNIE 5.5 expert activation mode must be 'a8' or 'a16'")
        self.dspark = dspark
        self.expert_scale_compression = expert_scale_compression
        self.expert_activation_mode = expert_activation_mode
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config = ModelConfig.load(manifest.metadata_path)
        self.shard = shard
        self._expert_scale_scratch = {}
        self._flashindexer = self._load_flashindexer_weights(manifest.snapshot)

    def _load_flashindexer_weights(self, snapshot: Path) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        path = Path(snapshot) / "flashindexer" / "flashindexer.safetensors"
        if not path.is_file():
            return {}
        from safetensors.torch import safe_open

        gates = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for layer in self.config.dsa_layers:
                g_key = f"model.layers.{layer}.self_attn.core_attention.indexer.linear_gate.weight"
                b_key = f"model.layers.{layer}.self_attn.core_attention.indexer.flashindexer_bias"
                if g_key not in keys or b_key not in keys:
                    raise ValueError(f"flashindexer.safetensors missing DSA layer {layer} gate parameters")
                gate_w = persistent_copy(f.get_tensor(g_key).to(device=self.device, dtype=torch.bfloat16))
                bias_f = persistent_copy(f.get_tensor(b_key).to(device=self.device, dtype=torch.float32))
                gates[layer] = (gate_w, bias_f)
        return gates

    def _selected_names(self):
        return {
            name
            for name in super()._selected_names()
            if (
                name.startswith("model.layers.")
                or (self.dspark and name.startswith("model.mtp."))
                or name.startswith("model.mhc_contract.")
                or name.startswith("model.lm_head.")
                or name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
            )
        }

    def _pack_blocks(self, parts, *, dim, temporary=False):
        from b12x.gemm.blockscaled import pack_weight

        self._session.flush()
        columns = sum(part[0].shape[1] for part in parts) if dim == 1 else parts[0][0].shape[1]
        concatenate = torch.cat if temporary or columns % 128 else persistent_cat
        values = concatenate([p[0] for p in parts], dim=dim) if len(parts) > 1 else parts[0][0]
        expanded = []
        for val, sc in parts:
            sc_u8 = _fp32_scales_to_ue8m0(sc, "block_fp8") if sc.dtype == torch.float32 else sc
            s = sc_u8.repeat_interleave(self.block_size, dim=0)[: val.shape[0]]
            s = s.repeat_interleave(self.block_size // 32, dim=1)[:, : val.shape[1] // 32]
            expanded.append(s)
        scales = concatenate(expanded, dim=dim) if len(expanded) > 1 else expanded[0]
        options = {} if temporary else {"allocate": persistent_empty}
        return pack_weight(values, scales, recipe="mxfp8", **options)

    def _dense_blocks(self, prefix, *, rows=None, columns=None, pad=False, temporary=True):
        self._require_open()
        if f"{prefix}.weight" in self._sources:
            weight = self._sources[f"{prefix}.weight"]
            scale = self._sources[f"{prefix}.weight_scale_inv"]
        else:
            # HCA linear_o_group_proj omits the `.weight` leaf suffix.
            weight = self._sources[prefix]
            scale = self._sources[f"{prefix}_scale_inv"]
        block = self.block_size
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise ValueError(f"{prefix}: expected native FP8 dense weights")
        n, k = weight.shape
        expected_scale = ((n + block - 1) // block, (k + block - 1) // block)
        if scale.dtype not in (torch.float32, torch.float8_e8m0fnu) or scale.shape != expected_scale:
            raise ValueError(
                f"{prefix}: expected FP32/UE8M0 scales {expected_scale} for {block}-by-{block} blocks, "
                f"got {scale.dtype} {tuple(scale.shape)}"
            )
        row_align = 32 if rows is None and n % block else block
        col_align = 32 if columns is None and k % block else block
        r0, r1 = _aligned_range(rows, max(n, rows[1]) if pad and rows else n, row_align)
        c0, c1 = _aligned_range(columns, max(k, columns[1]) if pad and columns else k, col_align)
        if r0 % block or c0 % block:
            raise ValueError(f"{prefix}: slice start ({r0}, {c0}) must align to block {block}")
        allocate = torch.empty if temporary or (c1 - c0) % 128 else persistent_empty
        values = allocate((r1 - r0, c1 - c0), dtype=weight.dtype, device=self.device)
        sr_out = -(-(r1 - r0) // block)
        sc_out = -(-(c1 - c0) // block)
        scale_dtype = torch.float32 if scale.dtype == torch.float32 else torch.uint8
        scales = torch.empty((sr_out, sc_out), dtype=scale_dtype, device=self.device)
        nr, nc = max(0, min(r1, n) - r0), max(0, min(c1, k) - c0)
        if nr != r1 - r0 or nc != c1 - c0:
            values.zero_()
            scales.fill_(1.0 if scale_dtype == torch.float32 else 127)
        if nr and nc:
            sr, sc = -(-nr // block), -(-nc // block)
            self._copy(values[:nr, :nc], weight[r0 : r0 + nr, c0 : c0 + nc])
            src_scale = scale if scale_dtype == torch.float32 else scale.view(torch.uint8)
            self._copy(
                scales[:sr, :sc],
                src_scale[r0 // block : r0 // block + sr, c0 // block : c0 // block + sc],
            )
        return values, scales

    def load_attention_output(self, prefix, groups, *, pad=False):
        from b12x.gemm import wo_projection

        a, sa = self._dense_blocks(prefix + ".linear_o_group_proj", rows=groups, pad=pad, temporary=False)
        b, sb = self._dense_blocks(prefix + ".o_proj", columns=groups, pad=pad, temporary=False)
        self._session.flush()
        sa_u8 = _fp32_scales_to_ue8m0(sa, prefix + ".linear_o_group_proj") if sa.dtype == torch.float32 else sa
        sb_u8 = _fp32_scales_to_ue8m0(sb, prefix + ".o_proj") if sb.dtype == torch.float32 else sb
        c = self.config.text
        return wo_projection.pack_weights(
            a,
            sa_u8.view(torch.float8_e8m0fnu),
            b,
            sb_u8.view(torch.float8_e8m0fnu),
            groups=(groups[1] - groups[0]) // c["o_lora_rank"],
            group_width=c["num_attention_heads"] * c["head_dim"] // c["o_groups"],
            rank=c["o_lora_rank"],
            hidden=c["hidden_size"],
            block_size=(self.block_size, self.block_size),
            allocate=persistent_empty,
        )

    def load_head_mix(self, prefix):
        fn = self.load_tensor(prefix + "hc_head_fn", dtype=torch.float32, temporary=True)
        return {
            "fn": persistent_copy(fn.t().contiguous()),
            "base": self.load_tensor(prefix + "hc_head_base", dtype=torch.float32),
            "scale": self.load_tensor(prefix + "hc_head_scale", dtype=torch.float32),
        }

    def load_mhc(self, prefix):
        result = {
            "attn_norm.weight": self.load_tensor(prefix + ".input_layernorm.weight"),
            "ffn_norm.weight": self.load_tensor(prefix + ".post_attention_layernorm.weight"),
        }
        for kind, ckpt in (("attn", "self_attention_hyper_connection"), ("ffn", "mlp_hyper_connection")):
            p = f"{prefix}.{ckpt}"
            fn = self.load_tensor(f"{p}.mapping_proj.weight", dtype=torch.float32, temporary=True)
            result[f"hc_{kind}_fn"] = persistent_copy(fn.t().contiguous())
            result[f"hc_{kind}_base"] = self.load_tensor(f"{p}.bias", dtype=torch.float32)
            alphas = self.load_tensors(
                (f"{p}.alpha_pre", f"{p}.alpha_post", f"{p}.alpha_res"), temporary=True
            )
            result[f"hc_{kind}_scale"] = persistent_cat([a.float().reshape(1) for a in alphas])
        return result

    def _load_dequantized_kv_b(self, prefix, rows, heads):
        # kv_b_proj is [64 * (192 + 256), 512] FP8 with [224, 4] block-128 scales.
        # On TP2: rows = 32 * 448 = 14336 (divisible by 128), cols = 512 (divisible by 128).
        val, sc = self._dense_blocks(prefix, rows=rows, temporary=True)
        self._session.flush()
        s = sc.float().repeat_interleave(self.block_size, dim=0)[: val.shape[0]]
        s = s.repeat_interleave(self.block_size, dim=1)[:, : val.shape[1]]
        deq = (val.float() * s).to(torch.bfloat16).view(heads, 192 + 256, 512)
        w_uk = persistent_copy(deq[:, :192, :].contiguous())
        w_uv = persistent_copy(deq[:, 192:, :].contiguous())
        return w_uk, w_uv

    def load_attention(self, layer, *, draft=False):
        prefix = f"model.mtp.{layer}" if draft else f"model.layers.{layer}"
        attn = prefix + ".self_attn"
        shard = self.shard
        is_dsa = (not draft) and self.config.is_dsa_layer(layer)
        h0, h1 = shard.partition(64)
        local_heads = h1 - h0
        vha_u = self.load_tensor(f"{attn}.vha_postmix_U", extent=slice(h0, h1))
        vha_v = self.load_tensor(f"{attn}.vha_postmix_V", extent=slice(h0, h1))
        if not is_dsa:
            heads = shard.partition(64 * 512, alignment=512)
            groups = shard.partition(8 * 1024, alignment=1024)
            result = {
                "qkv_a": self.load_linear_group((f"{attn}.linear_q_down_proj", f"{attn}.linear_kv_proj")),
                "q_b": self.load_linear(f"{attn}.linear_q_up_proj", rows=heads),
                "o_gate": self.load_linear(f"{attn}.gate_proj", rows=groups),
                "output": self.load_attention_output(attn, groups),
                "q_norm": self.load_tensor(f"{attn}.q_layernorm.weight"),
                "kv_norm": self.load_tensor(f"{attn}.kv_layernorm.weight"),
                "sink": self.load_tensor(f"{attn}.core_attention.attn_sink", dtype=torch.float32, extent=slice(h0, h1)),
                "vha_u": vha_u,
                "vha_v": vha_v,
            }
            if not draft:
                comp = f"{attn}.core_attention.compressor"
                wkv = self.load_tensor(f"{comp}.linear_wkv.weight", temporary=True)
                wgate = self.load_tensor(f"{comp}.linear_wgate.weight", temporary=True)
                result["compressor"] = {
                    "norm": self.load_tensor(f"{comp}.norm.weight"),
                    "ape": self.load_tensor(f"{comp}.ape", dtype=torch.float32),
                }
                result["compression"] = persistent_cat((wkv, wgate))
            return result, False

        # DSA layer (6 layers: 7, 14, 21, 28, 35, 42)
        q_rows = (h0 * 256, h1 * 256)
        kv_b_rows = (h0 * 448, h1 * 448)
        w_uk, w_uv = self._load_dequantized_kv_b(f"{attn}.kv_b_proj", kv_b_rows, local_heads)
        idx = f"{attn}.core_attention.indexer"
        wk = self.load_tensor(f"{idx}.wk.weight", temporary=True)
        wproj = self.load_tensor(f"{idx}.weights_proj.weight", temporary=True)
        result = {
            "qkv_a": self.load_linear_group((f"{attn}.q_a_proj", f"{attn}.kv_a_proj_with_mqa")),
            "q_b": self.load_linear(f"{attn}.q_b_proj", rows=q_rows),
            "o_gate": self.load_linear(f"{attn}.gate_proj", rows=q_rows),
            "o_proj": self.load_linear(f"{attn}.o_proj", columns=q_rows),
            "q_norm": self.load_tensor(f"{attn}.q_a_layernorm.weight"),
            "kv_norm": self.load_tensor(f"{attn}.kv_a_layernorm.weight"),
            "w_uk": w_uk,
            "w_uv": w_uv,
            "sink": self.load_tensor(f"{attn}.core_attention.softmax_offset", dtype=torch.float32, extent=slice(h0, h1)),
            "vha_u": vha_u,
            "vha_v": vha_v,
            "index_q": self.load_linear(f"{idx}.wq_b"),
            "index_kw": persistent_cat((wk, wproj)),
            "index_k_norm_w": self.load_tensor(f"{idx}.k_norm.weight"),
            "index_k_norm_b": self.load_tensor(f"{idx}.k_norm.bias"),
        }
        if layer in self._flashindexer:
            result["flash_gate"], result["flash_bias"] = self._flashindexer[layer]
        return result, True

    def load_experts(self, prefixes, interval):
        text = self.config.text
        hidden = text["moe_latent_size"]
        inter = text["moe_intermediate_size"]
        if self.expert_activation_mode == "a16":
            first, last = _aligned_range(interval, inter, 32)
            local, count = last - first, len(prefixes)
            key = ("a16", count, hidden, local)
            if self.expert_scale_compression == "csf" and key not in self._expert_scale_scratch:
                self._expert_scale_scratch[key] = (
                    torch.empty((count, (2 * local + 255) // 256 * 256, (hidden // 32 + 7) // 8 * 8),
                                dtype=torch.uint8, device=self.device),
                    torch.empty((count, (hidden + 255) // 256 * 256, (local // 32 + 7) // 8 * 8),
                                dtype=torch.uint8, device=self.device),
                )
            return self.load_fp4_experts(
                prefixes,
                hidden_size=hidden,
                intermediate_size=inter,
                format="mxfp4",
                activation_mode="a16",
                intermediate_range=interval,
                projection_names=("w1", "w3", "w2"),
                scale_name="weight_scale_inv",
                activation="situ",
                w13_layout="w31",
                scale_compression=self.expert_scale_compression,
                scale_scratch=self._expert_scale_scratch.get(key),
            )
        return self.load_mxfp4_experts(
            prefixes,
            hidden_size=hidden,
            intermediate_size=inter,
            intermediate_range=interval,
            scale_name="weight_scale_inv",
            projection_names=("w3", "w1", "w2"),
            activation="situ",
            activation_mode="a8",
            scale_compression=self.expert_scale_compression,
            scale_scratch=self._expert_scale_scratch,
        )

    def load_layer(self, layer, *, draft=False):
        prefix = f"model.mtp.{layer}" if draft else f"model.layers.{layer}"
        text, shard = self.config.text, self.shard
        attention, is_dsa = self.load_attention(layer, draft=draft)
        residual = self.load_mhc(prefix)
        if not draft and layer < text["first_k_dense_replace"]:
            interval = shard.partition(text["intermediate_size"], alignment=128)
            mlp = MLPWeights(
                self.load_linear_group(
                    (f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"),
                    row_ranges=(interval, interval),
                ),
                self.load_linear(f"{prefix}.mlp.down_proj", columns=interval),
            )
            return LayerWeights(attention=attention, mlp=mlp, experts=None, residual=residual, is_dsa=is_dsa)
        moe = prefix + ".block_sparse_moe"
        interval = shard.partition(text["moe_intermediate_size"], alignment=128)
        experts = self.load_experts(
            [f"{moe}.experts.{i}" for i in range(text["n_routed_experts"])],
            interval,
        )
        # Replicate the tiny 2048-wide shared expert on every TP rank to avoid a second [T, 4096] AllReduce.
        shared = MLPWeights(
            self.load_linear_group((f"{moe}.shared_experts.w1", f"{moe}.shared_experts.w3")),
            self.load_linear(f"{moe}.shared_experts.w2"),
        )
        g0 = self.load_tensor(f"{moe}.gate.weight", temporary=True)
        g1 = self.load_tensor(f"{moe}.gate.weight_1", temporary=True)
        router = persistent_cat((g0, g1))
        del g0, g1
        routed = LatentExpertWeights(
            router=router,
            correction=self.load_tensor(f"{moe}.e_score_correction_bias", dtype=torch.float32),
            scale=self.load_tensor(f"{moe}.gate.routed_scaling_factor_param", dtype=torch.float32),
            fc1_latent=self.load_linear(f"{moe}.fc1_latent_proj"),
            latent_norm=self.load_tensor(f"{moe}.latent_norm.weight"),
            fc2_latent=self.load_linear(f"{moe}.fc2_latent_proj"),
            experts=experts,
            shared=shared,
        )
        return LayerWeights(attention=attention, mlp=None, experts=routed, residual=residual, is_dsa=is_dsa)
