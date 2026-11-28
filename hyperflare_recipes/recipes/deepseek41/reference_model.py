# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose reference target, draft and vision computation for Ignition capture."""

from dataclasses import dataclass

import torch
from torch import nn

from hyperflare_recipes.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
from hyperflare_recipes.components.megamoe_tiers import MegaMoeLayerLayout
from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.tensor_metadata import record_numeric_tensor
from hyperflare_recipes.recipes.deepseek41.config import ModelConfig
from hyperflare_recipes.recipes.deepseek41.reference_experts import ReferenceExperts


def _record_model_numeric_tensors(root):
    """Register learned, scale and derived numeric tensors with the core exporter."""
    def record(value):
        if isinstance(value, torch.Tensor) and value.is_cuda and value.layout == torch.strided:
            record_numeric_tensor(value)
            sf = getattr(value, "sf", None)
            if isinstance(sf, torch.Tensor):
                record(sf)
            for extra in getattr(value, "ue8m0_sf", ()):
                record(extra)

    for module in root.modules():
        for tensor in (*module._parameters.values(), *module._buffers.values()):
            if tensor is not None:
                record(tensor)
        for key, value in vars(module).items():
            if key in ("_parameters", "_buffers", "_modules", "operations", "delivery", "engram_runtime"):
                continue
            if isinstance(value, torch.Tensor):
                record(value)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        record(item)


@dataclass(frozen=True)
class ReferenceDeclaredWeights:
    """Semantic weight container pairing per-layer MegaMoE storage with reference modules."""

    layers: tuple
    draft_experts: tuple
    draft_heads: tuple
    model: object
    scales: tuple


def declare_reference_weights(model):
    """Build the prepared-weight declaration matching target expert row bindings."""
    target_count = len(model.model.layers)
    draft_count = 0 if model.draft is None else len(model.draft.layers)
    scales, seen_scales = [], set()
    for parameter in model.parameters():
        for extra in (getattr(parameter, "sf", None), *getattr(parameter, "ue8m0_sf", ())):
            if isinstance(extra, torch.Tensor) and extra.data_ptr() not in seen_scales:
                seen_scales.add(extra.data_ptr())
                scales.append(extra)
    derived_heads = tuple(h for h in (getattr(model, "draft_head", None), getattr(model, "markov_head", None))
                          if h is not None)
    roots = ReferenceDeclaredWeights(
        layers=tuple(model.experts.weights[f"layers.{i}.ffn"] for i in range(target_count)),
        draft_experts=tuple(model.experts.weights[f"mtp.{i}.ffn"] for i in range(draft_count)),
        draft_heads=derived_heads,
        model=model,
        scales=tuple(scales),
    )
    components = ["model", "scales"]
    if draft_count:
        components.insert(0, "draft_experts")
    if derived_heads:
        components.insert(0, "draft_heads")
    aliases = ()
    if draft_count:
        aliases = (
            ("weights.model.draft.embed_tokens.weight", "weights.model.model.embed_tokens.weight"),
            ("weights.model.draft.lm_head.weight", "weights.model.lm_head.weight"),
        )
    return declare_weights(roots, components=tuple(components), layers=("layers",), aliases=aliases)


@dataclass(frozen=True)
class ReferenceExpertPreparation:
    modules: dict
    weights: dict
    mover: object | None = None

    @classmethod
    def load(cls, checkpoint, placements, *, target_rows, draft_rows=(), storage,
             kernel_cache, session, host_lane_clusters=4, dspark=True,
             dynamic_mover=None, compute_sm_count=0):
        """Load all target/draft experts into explicit tiers using bounded batches.

        The caller owns the native storage context and retains this preparation
        until graph execution drains. Scale expansion preserves checkpoint bits;
        the prepared MegaMoE planes use native scales in both physical tiers.
        """
        from hyperflare_recipes.components.megamoe_tiers import MegaMoeDynamicHostTier, MegaMoeLayerWeights
        from hyperflare_recipes.components.megamoe_weights import (
            allocate_mxfp4_weights, load_mxfp4_weights_into,
            load_shared_fp8_planes, pack_shared_fp8_planes,
        )
        from hyperflare_recipes.components.sm103_expert_mover import SM103ExpertMoverRuntime
        from ignition.preparation.persistent import persistent_copy

        model = ModelConfig.load(checkpoint.source.metadata_path)
        placements = dict(placements)
        required = [f"layers.{i}.ffn" for i in range(model.layers)]
        if dspark:
            required.extend(f"mtp.{i}.ffn" for i in range(model.draft.layers))
        if set(placements) != set(required):
            raise ValueError("reference preparation requires placements for every enabled target and draft layer")
        if not target_rows or (dspark and not draft_rows):
            raise ValueError("reference preparation requires row capacities for every enabled expert group")
        device = torch.device("cuda", storage.device)
        workspaces, modules, owners = {}, {}, {}
        dynamic_layers = []
        for name in required:
            draft = name.startswith("mtp.")
            rows = tuple(draft_rows if draft else target_rows)
            placement = placements[name]
            experts = model.draft.experts if draft else model.text["n_routed_experts"]
            if placement.experts != experts:
                raise ValueError(f"reference placement differs from checkpoint expert count: {name}")
            use_dynamic = dynamic_mover is not None and not draft and bool(placement.host)
            physical_rows = experts if use_dynamic else placement.physical_rows
            geometry = MegaMoeGeometry(physical_rows, max(rows),
                model.draft.top_k if draft else model.text["num_experts_per_tok"],
                model.text["hidden_size"], model.text["moe_intermediate_size"], model.text["n_shared_experts"])
            if geometry not in workspaces:
                workspaces[geometry] = MegaMoeWorkspace(geometry, device=device, kernel_cache=kernel_cache)
            ws = workspaces[geometry]
            with DirectWeightLoader(checkpoint.source, device, prefixes=(name + ".",)) as loader:
                if use_dynamic:
                    from hyperflare_recipes.components.megamoe_weights import MegaMoeWeights

                    host_experts = placement.host
                    host_set = set(host_experts)
                    pinned_experts = tuple(e for e in placement.hbm if e not in host_set)
                    overlapped_hbm = tuple(e for e in placement.hbm if e in host_set)
                    internal_order = host_experts + pinned_experts
                    canonical_to_internal = [0] * experts
                    for internal_idx, canonical in enumerate(internal_order):
                        canonical_to_internal[canonical] = internal_idx
                    initial_hbm_canonical = pinned_experts + overlapped_hbm
                    initial_hbm_internal = tuple(canonical_to_internal[e] for e in initial_hbm_canonical)
                    hbm_slots = len(placement.hbm) + dynamic_mover.spare_slots
                    tier = MegaMoeDynamicHostTier(
                        ws.geometry,
                        hbm_slots=hbm_slots,
                        initial_hbm_experts=initial_hbm_internal,
                        storage=storage,
                        name=name,
                        host_experts=len(host_experts),
                        host_lane_clusters=host_lane_clusters,
                    )
                    load_mxfp4_weights_into(
                        loader,
                        [f"{name}.experts.{e}" for e in host_experts],
                        tier.cpu_weights,
                        hidden_size=ws.geometry.hidden, intermediate_size=ws.geometry.intermediate,
                        projection_names=("w1", "w3", "w2"), scale_name="scale",
                    )
                    routed = allocate_mxfp4_weights(hbm_slots, ws.geometry.hidden, ws.geometry.intermediate, device=device)
                    n_pinned = len(pinned_experts)
                    if n_pinned > 0:
                        pinned_view = MegaMoeWeights(
                            (routed.gate_up[0][:n_pinned], routed.gate_up[1][:n_pinned]),
                            (routed.down[0][:n_pinned], routed.down[1][:n_pinned]),
                        )
                        load_mxfp4_weights_into(
                            loader,
                            [f"{name}.experts.{e}" for e in pinned_experts],
                            pinned_view,
                            hidden_size=ws.geometry.hidden, intermediate_size=ws.geometry.intermediate,
                            projection_names=("w1", "w3", "w2"), scale_name="scale",
                        )
                    for dst, src_cpu in zip((*routed.gate_up, *routed.down), tier.host_views):
                        for offset, canonical in enumerate(overlapped_hbm):
                            dst[n_pinned + offset].copy_(src_cpu[canonical_to_internal[canonical]].to(device))
                        dst[len(placement.hbm):].zero_()
                    shared = pack_shared_fp8_planes(load_shared_fp8_planes(
                        loader, name + ".shared_experts",
                        hidden_size=ws.geometry.hidden, intermediate_size=ws.geometry.intermediate,
                        block_size=tuple(model.quantization["weight_block_size"]),
                        projection_names=("w1", "w3", "w2"), scale_name="scale",
                    ))
                    expert_map = persistent_copy(torch.tensor(canonical_to_internal, dtype=torch.int32, device=device))
                    record_numeric_tensor(expert_map)
                    weights = MegaMoeLayerWeights(
                        routed, shared, ws, tier, expert_map,
                        host_experts, initial_hbm_canonical,
                    )
                    dynamic_layers.append((tier, routed))
                else:
                    layout = MegaMoeLayerLayout(ws, placement.host, placement.hbm,
                                                storage, host_lane_clusters)
                    weights = layout.load(loader, (f"{name}.experts.{i}" for i in range(experts)),
                        shared_prefix=name + ".shared_experts", block_size=tuple(model.quantization["weight_block_size"]),
                        projection_names=("w1", "w3", "w2"), name=name)
                weight = loader.load_tensor(name + ".gate.weight")
                bias = loader.load_tensor(name + ".gate.bias")
                image_bias = loader.load_tensor(name + ".gate.bias_vl") if model.vision is not None else None
            owners[name] = weights
            modules[name] = ReferenceExperts.prepare(weights, rows, weight=weight, bias=bias, image_bias=image_bias,
                scaling=model.text["routed_scaling_factor"], scoring_func=model.text["scoring_func"],
                clamp=model.text["swiglu_limit"], session=session, name=name,
                compute_sm_count=compute_sm_count)
            if modules[name].placement.data_ptr() != weights.expert_map.data_ptr():
                raise RuntimeError(f"reference router placement lost its tier mapping alias: {name}")
        mover = None
        if dynamic_layers:
            mover = SM103ExpertMoverRuntime(
                dynamic_layers,
                storage=storage,
                kernel_cache=kernel_cache,
                worker_sm_count=dynamic_mover.worker_sm_count,
                queue_capacity=dynamic_mover.queue_capacity,
            )
            mover.attach_predictors(dynamic_mover.predictor_config())
        return cls(modules, owners, mover=mover)

    def close_host_regions(self):
        """Release mapped host weights after every reader and graph has drained."""
        if self.mover is not None:
            self.mover.close()
        for weights in self.weights.values():
            if weights.host_tier is not None:
                for region in weights.host_tier.regions:
                    region.close()


REFERENCE_YOCO_MIN_KEEP_TOKENS = 144


def reference_yoco_keep_tokens(n_prefill_layer, num_layers, window_size, verification_width, prefill_tokens):
    """Return the 16-aligned YOCO consumer suffix width when CED truncation applies."""
    if not 0 < n_prefill_layer < num_layers:
        return None
    keep = max(REFERENCE_YOCO_MIN_KEEP_TOKENS, (window_size + verification_width - 1 + 15) // 16 * 16)
    return keep if prefill_tokens > keep else None


class ReferenceModel(nn.Module):
    """Own the original computation with Flare expert and Engram storage bindings.

    This is a preparation/capture component. Sampling, accepted-state commit,
    vision input packing and native graph publication belong to the enclosing
    recipe. Construction consumes reference parameter storage; it cannot be
    retried with the same parameter tree after a partial failure.
    """

    def __init__(self, parameters, experts, delivery, *, execution, context,
                 request_capacity, prefill_tokens, verification_tokens, yoco_keep_tokens=None):
        from deepinfer import AttrDict
        from deepinfer.engine.engine_type import EngineType
        from deepinfer.hugging_face import build_mm_processor, load_processor_config
        from deepinfer.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from deepinfer.model_executor.models.deepseek import DeepseekModel
        from deepinfer.model_executor.models.deepseek_vl import DeepseekVLForCausalLM
        from deepinfer.model_executor.models.mtp.dflash import DFlash
        from deepinfer.model_executor.models.v4_modules.cross_layer import build_shared_layer_topology
        from deepinfer.model_executor.models.vision_encoder.vl_aligner import VisionLanguageInstructTuner
        from deepinfer.timer import cpu_timer, cuda_graph_timer, cuda_timer

        super().__init__()
        cpu_timer.active = cuda_graph_timer.active = cuda_timer.active = False
        if not parameters.loaded or parameters.assembled:
            raise ValueError("reference model requires loaded, unconsumed parameters")
        if (any(type(value) is not int or value < 1 for value in
                (context, request_capacity, prefill_tokens, verification_tokens))
                or prefill_tokens > context or execution.draft_tokens < 0):
            raise ValueError("reference assembly requires positive capacities within the context limit")
        if yoco_keep_tokens is None:
            if execution.yoco_prefill_truncation != "off":
                raise ValueError("reference YOCO compaction requires an explicit suffix token capacity")
        elif (type(yoco_keep_tokens) is not int or execution.yoco_prefill_truncation != "compact"
                or yoco_keep_tokens < parameters.config.gpt.window_size[0] + execution.draft_tokens
                or yoco_keep_tokens >= prefill_tokens):
            raise ValueError("reference YOCO suffix capacity must cover the sliding-window ring within prefill chunks")
        self.yoco_keep_tokens = yoco_keep_tokens
        self.expert_stagers = {}
        self.stage_min_rows = 1024
        self.draft_head = None
        self.markov_head = None
        cfg = AttrDict(parameters.config.gpt, runtime_max_model_len=context,
            max_num_seqs=request_capacity, max_num_batched_tokens=max(prefill_tokens, verification_tokens),
            max_num_decoding_tokens=verification_tokens, max_num_prefilling_tokens=prefill_tokens,
            engine_type=EngineType.NORMAL)
        target_keys = [f"layers.{i}.ffn" for i in range(cfg.num_hidden_layers)]
        draft_keys = ([f"mtp.{i}.ffn" for i in range(cfg.num_nextn_predict_layers)]
                      if execution.draft_tokens else [])
        if set(experts.modules) != set(target_keys + draft_keys):
            raise ValueError("reference model requires every prepared target and draft expert module")
        cfg.vision = parameters.config.vl_aligner.vision.params
        cfg.aligner = parameters.config.vl_aligner.aligner.params
        processor = build_mm_processor(load_processor_config(parameters.checkpoint.source.metadata_path))
        DeepseekVLForCausalLM.derive_image_transform_cfg(processor.image_processor, cfg)
        self.cfg, self.experts, self.delivery = cfg, experts, delivery
        parameters.assembled = True
        sources = parameters.model.module.module
        total_layers = cfg.num_hidden_layers + (cfg.num_nextn_predict_layers if execution.draft_tokens else 0)
        topology = build_shared_layer_topology(cfg, total_layers)
        self.model = DeepseekModel(sources[:cfg.num_hidden_layers + 1], sources[cfg.num_hidden_layers + 1],
            cfg, execution.draft_tokens, topology,
            prepared_mlps=[experts.modules[key] for key in target_keys], execution=execution)
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size,
            weight=sources[cfg.num_hidden_layers + 1].shared_head.head.weight)
        self.draft = (DFlash(parameters.model, cfg, self.model.embed_tokens, self.lm_head,
            prepared_mlps=[experts.modules[key] for key in draft_keys], execution=execution)
            if execution.draft_tokens else None)
        layers = [self.model.layers[index].engram for index in cfg.engram.engram_layer_ids]
        delivery.bind(layers)
        self.model.engram_runtime = delivery
        with torch.device(self.lm_head.weight.device):
            self.vision_language_tuner = VisionLanguageInstructTuner(cfg, sources[0],
                vision_source=parameters.model.vision_pipeline.aligner)
        _record_model_numeric_tensors(self)

    def _special_image_token_mask(self, input_ids):
        from deepinfer.model_executor.models.deepseek_vl import DeepseekVLForCausalLM

        return DeepseekVLForCausalLM._special_image_token_mask(self, input_ids)

    def bind_draft_heads(self, *, draft_head=None, markov_head=None):
        """Attach quantized draft vocabulary or Markov heads derived from checkpoint weights."""
        if (draft_head is not None or markov_head is not None) and self.draft is None:
            raise ValueError("derived draft heads require an enabled draft model")
        if markov_head is not None and not self.draft.enable_markov:
            raise ValueError("derived Markov head requires an enabled Markov head")
        self.draft_head = draft_head
        self.markov_head = markov_head
        for obj in (draft_head, markov_head):
            if obj is None:
                continue
            for value in vars(obj).values():
                if isinstance(value, torch.Tensor):
                    record_numeric_tensor(value)

    def bind_prefill_staging(self, expert_stagers, *, min_rows=1024):
        """Attach prepared double-buffered cold-expert stagers for prompt execution."""
        if type(min_rows) is not int or min_rows < 1:
            raise ValueError("prefill staging requires a positive row threshold")
        self.expert_stagers = dict(expert_stagers)
        self.stage_min_rows = min_rows

    def bind_low_latency_fp8_projections(self, *, target_rows, draft_rows=(), session):
        """Wrap eligible FP8 attention and draft projections with tuned SM103 GEMMs for M <= 64."""
        from deepinfer.model_executor.layers.linear import ReplicatedLinear
        from hyperflare_recipes.components.sm103_projection import (
            LOW_LATENCY_FP8_MAX_ROWS, LowLatencyFP8Linear, is_sm103,
        )

        if not is_sm103(self.lm_head.weight.device):
            return
        small_target = tuple(sorted({r for r in target_rows if 1 <= r <= LOW_LATENCY_FP8_MAX_ROWS}))
        small_draft = tuple(sorted({r for r in draft_rows if 1 <= r <= LOW_LATENCY_FP8_MAX_ROWS}))

        def wrap_linear(owner, attr, rows, full_name, *, pre_swizzled_input=False, bf16_input=False):
            mod = getattr(owner, attr, None)
            if (isinstance(mod, ReplicatedLinear)
                    and mod.weight.dtype == torch.float8_e4m3fn
                    and mod.bias is None and not mod.use_fp32_output
                    and mod.input_size % 128 == 0 and mod.output_size % 128 == 0):
                setattr(owner, attr, LowLatencyFP8Linear(
                    mod, rows, session=session, name=full_name,
                    pre_swizzled_input=pre_swizzled_input, bf16_input=bf16_input,
                ))

        def wrap_attention(attn, rows, prefix):
            if not rows:
                return
            for attr, pre_swz, bf16_in in (
                ("q_a_kv_swa_proj", True, False),
                ("q_a_kv_proj", True, False),
                ("kv_swa_proj", True, True),
                ("q_b_proj", True, False),
                ("out_proj", False, False),
            ):
                wrap_linear(attn, attr, rows, f"{prefix}.{attr}",
                            pre_swizzled_input=pre_swz, bf16_input=bf16_in)

        for index, layer in enumerate(self.model.layers):
            wrap_attention(layer.self_attn, small_target, f"layers.{index}.attn")
            if layer.engram is not None and small_target:
                wrap_linear(layer.engram, "fused_kv_projection", small_target,
                            f"layers.{index}.engram.fused_kv_projection", pre_swizzled_input=False)
        if self.draft is not None:
            if small_target:
                wrap_linear(self.draft, "projection", small_target, "mtp.projection",
                            pre_swizzled_input=True, bf16_input=True)
                wrap_linear(self.draft, "main_kv_proj", small_target, "mtp.main_kv_proj",
                            pre_swizzled_input=True, bf16_input=False)
            if small_draft:
                for index, layer in enumerate(self.draft.layers):
                    wrap_attention(layer.self_attn, small_draft, f"mtp.{index}.attn")

    def bind_sm103_decode_optimizations(self, *, target_rows, draft_rows=(), kernel_cache,
                                        session, stream_overlap: bool = True, compute_sm_count: int = 0):
        """Bind SM103 direct F8_128x4 norms, fused decode attention, and overlapped mHC seams.

        Returns non-picklable lifetime owners (auxiliary CUDA streams and shared workspaces)
        for the caller to retain in ``PreparedModel.resources``.
        """
        from deepinfer.model_executor.layers.layernorm import RMSNorm
        from hyperflare_recipes.components.mega_attention import (
            MEGA_ATTENTION_MAX_ROWS, MegaAttentionDecoder, MegaAttentionWorkspace,
        )
        from hyperflare_recipes.components.sm103_mhc_overlap import (
            MHC_SEAM_MAX_ROWS, FastRMSNormF8, bind_indexer_kv_overlap,
            bind_mhc_overlap_layer, bind_swa_kv_overlap,
        )
        from hyperflare_recipes.components.sm103_projection import is_sm103
        from hyperflare_recipes.components.sm103_router import SM103RouterWorkspace, supports_sm103_router

        device = self.lm_head.weight.device
        if not is_sm103(device):
            return ()
        resources = []
        small_target = tuple(sorted({r for r in target_rows if 1 <= r <= MHC_SEAM_MAX_ROWS}))
        small_draft = tuple(sorted({r for r in draft_rows if 1 <= r <= MHC_SEAM_MAX_ROWS}))
        attn_rows = tuple(r for r in small_target if r <= MEGA_ATTENTION_MAX_ROWS)
        yoco_boundary = self.model.yoco_boundary_layer_idx

        def wrap_rmsnorm(owner, norm_attr, consumer, rows, full_name, *, return_bf16=False):
            norm = getattr(owner, norm_attr, None)
            if (rows and isinstance(norm, RMSNorm) and norm.use_fp8
                    and norm.weight.dtype == torch.bfloat16 and norm.weight.numel() % 128 == 0
                    and consumer is not None and hasattr(consumer, "prefers_f8_128x4_scales")):
                setattr(owner, norm_attr, FastRMSNormF8(
                    norm, consumer, rows, session=session, name=full_name, return_bf16=return_bf16))

        for index, layer in enumerate(self.model.layers):
            wrap_rmsnorm(layer.self_attn, "q_a_norm", getattr(layer.self_attn, "q_b_proj", None),
                         small_target, f"layers.{index}.attn.q_a_norm")
            if index == 0 or layer.engram is not None or not stream_overlap:
                attn_consumer = (getattr(layer.self_attn, "q_a_kv_swa_proj", None)
                                 or getattr(layer.self_attn, "q_a_kv_proj", None))
                wrap_rmsnorm(layer, "attn_layernorm", attn_consumer, small_target,
                             f"layers.{index}.attn_layernorm", return_bf16=True)
        if self.draft is not None:
            wrap_rmsnorm(self.draft, "projection_norm", getattr(self.draft, "main_kv_proj", None),
                         small_target, "mtp.projection_norm")
            for index, layer in enumerate(self.draft.layers):
                wrap_rmsnorm(layer.self_attn, "q_a_norm", getattr(layer.self_attn, "q_b_proj", None),
                             small_draft, f"mtp.{index}.attn.q_a_norm")
                if index == 0 or not stream_overlap:
                    attn_consumer = (getattr(layer.self_attn, "q_a_kv_swa_proj", None)
                                     or getattr(layer.self_attn, "q_a_kv_proj", None))
                    wrap_rmsnorm(layer, "attn_layernorm", attn_consumer, small_draft,
                                 f"mtp.{index}.attn_layernorm", return_bf16=True)

        if attn_rows and kernel_cache is not None:
            workspace = MegaAttentionWorkspace(
                max(attn_rows), device=device, compute_sm_count=compute_sm_count)
            resources.append(workspace)
            for index, layer in enumerate(self.model.layers):
                attn = layer.self_attn
                if not (getattr(attn, "use_swa", False) and getattr(attn, "has_packed_kv", False)
                        and getattr(attn, "is_sparse", False)
                        and getattr(attn, "use_fused_norm_rope_attn_rope_cast", False)
                        and not getattr(attn, "use_q_b_norm", True)
                        and attn.num_heads == 64 and attn.qk_head_dim == 512
                        and attn.qk_rope_head_dim == 64 and attn.n_wv_group == 8
                        and not attn.rotary_emb.is_neox_style
                        and attn.rotary_emb.cos_sin_cache.dtype == torch.float32
                        and attn.attn_sink_params is not None
                        and attn.attn_sink_params.dtype == torch.float32):
                    continue
                decoder = MegaAttentionDecoder(
                    attn_rows, scale=attn.softmax_scale, workspace=workspace,
                    kernel_cache=kernel_cache, session=session,
                    name=f"layers.{index}.mega_attn", compressed=True,
                    swa_width=attn.window_size, comp_width=attn.layer_cfg.get("index_topk", 512),
                    token_interleaved=True, compute_sm_count=compute_sm_count,
                )
                resources.append(decoder)
                attn._decode_attention = _BoundMegaDecodeAttention(attn, decoder)

        if kernel_cache is not None and (small_target or small_draft):
            router_layers = []
            for index, layer in enumerate(self.model.layers):
                mlp = layer.mlp
                if hasattr(mlp, "bind_sm103_router") and small_target:
                    first_op = next(iter(mlp.operations.values()))
                    g = first_op.workspace.geometry
                    if supports_sm103_router(int(mlp.weight.shape[0]), int(g.hidden), int(g.top_k)):
                        router_layers.append((mlp, small_target, f"layers.{index}.router"))
            if self.draft is not None and small_draft:
                for index, layer in enumerate(self.draft.layers):
                    mlp = layer.mlp
                    if hasattr(mlp, "bind_sm103_router"):
                        first_op = next(iter(mlp.operations.values()))
                        g = first_op.workspace.geometry
                        if supports_sm103_router(int(mlp.weight.shape[0]), int(g.hidden), int(g.top_k)):
                            router_layers.append((mlp, small_draft, f"mtp.{index}.router"))
            if router_layers:
                max_router_rows = max(max(rows) for _, rows, _ in router_layers)
                max_router_experts = max(int(mlp.weight.shape[0]) for mlp, _, _ in router_layers)
                router_ws = SM103RouterWorkspace(
                    max_rows=max_router_rows, max_experts=max_router_experts, device=device,
                )
                resources.append(router_ws)
                for mlp, rows, rname in router_layers:
                    plans = mlp.bind_sm103_router(
                        rows, workspace=router_ws, kernel_cache=kernel_cache,
                        session=session, name=rname,
                    )
                    resources.extend(plans)

        if stream_overlap and (small_target or small_draft):
            aux_stream = torch.cuda.Stream(device=device)
            aux_kv_stream = torch.cuda.Stream(device=device)
            self._sm103_aux_stream = aux_stream
            self._sm103_aux_kv_stream = aux_kv_stream
            resources.extend((aux_stream, aux_kv_stream))
            num_layers = len(self.model.layers)
            for index, layer in enumerate(self.model.layers):
                allow_attn = index > 0 and layer.engram is None
                join_after = (index == num_layers - 1 or (yoco_boundary is not None and index == yoco_boundary - 1))
                bind_mhc_overlap_layer(
                    layer, small_target, aux_stream=aux_stream, session=session,
                    name=f"layers.{index}", allow_attn_overlap=allow_attn,
                    allow_ffn_overlap=True, join_after_forward=join_after, is_draft=False,
                )
                bind_swa_kv_overlap(layer.self_attn, small_target, aux_stream=aux_stream, is_draft=False)
                bind_indexer_kv_overlap(layer.self_attn, small_target, aux_kv_stream=aux_kv_stream)
            if self.draft is not None and small_draft:
                draft_count = len(self.draft.layers)
                for index, layer in enumerate(self.draft.layers):
                    bind_mhc_overlap_layer(
                        layer, small_draft, aux_stream=aux_stream, session=session,
                        name=f"mtp.{index}", allow_attn_overlap=(index > 0),
                        allow_ffn_overlap=True, join_after_forward=(index == draft_count - 1),
                        is_draft=True,
                    )
                    bind_swa_kv_overlap(layer.self_attn, small_draft, aux_stream=aux_stream, is_draft=True)
        return tuple(resources)

    def forward(self, caches, metadata, hidden_states=None, *, speculation_enabled=True):
        from deepinfer.model_executor.models.deepseek_vl import DeepseekVLForCausalLM

        if hidden_states is None and metadata.is_prompt:
            hidden_states = DeepseekVLForCausalLM.get_input_embeddings(self, metadata)
        dflash = self.draft if speculation_enabled else None
        return self.model(caches, metadata, hidden_states=hidden_states, dflash=dflash,
                          expert_stagers=self.expert_stagers,
                          stage_min_rows=self.stage_min_rows)


class _BoundMegaDecodeAttention:
    """Picklable decode-attention dispatcher for token-interleaved SM103 MegaAttention."""

    def __init__(self, attn, decoder):
        self.attn = attn
        self.decoder = decoder

    def __call__(self, q_b, kv_ca_index, kv_swa_index, kv_cache, input_metadata):
        attn = self.attn
        decoder = self.decoder
        rows = q_b.shape[0]
        if (decoder.supports(rows) and q_b.ndim == 3
                and q_b.stride(1) == 512 and q_b.stride(2) == 1
                and q_b.stride(0) >= 64 * 512 and q_b.stride(0) % 8 == 0
                and kv_swa_index is not None and kv_ca_index is not None
                and kv_cache.swa_cache is not None and kv_cache.ca_cache is not None
                and kv_swa_index.shape[1] == decoder.swa_width
                and kv_ca_index.shape[1] == decoder.comp_width
                and kv_cache.swa_cache.shape[-1] == 528 and kv_cache.ca_cache.shape[-1] == 288):
            cache_lens = getattr(input_metadata, "_mega_attn_comp_lens", None)
            if cache_lens is None:
                cache_lens = {}
                input_metadata._mega_attn_comp_lens = cache_lens
            comp_lens = cache_lens.get(attn.ratio_key)
            if comp_lens is None:
                klen = input_metadata.klen[attn.ratio_key]
                comp_lens = torch.where(klen > 0, decoder.comp_width, 0).to(torch.int32)
                cache_lens[attn.ratio_key] = comp_lens
            return decoder(
                q_b, input_metadata.positions, attn.rotary_emb.cos_sin_cache,
                attn.attn_sink_params, kv_cache.swa_cache.squeeze(2), kv_swa_index,
                kv_cache.ca_cache.squeeze(2), kv_ca_index, comp_lens,
            )
        return type(attn)._decode_attention(attn, q_b, kv_ca_index, kv_swa_index, kv_cache, input_metadata)
