# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualified checkpoint, serving range, and memory costs of the GLM 5.3 recipe."""

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from hyperflare_recipes.components.trellismx_release import TRELLISMX_OVERLAY
from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation

# Each sparse-attention layer stores four latent blocks and one index block per
# 256-token page on every rank. The target has eleven such layers; learned drafts
# add the draft layer's.
_PAGE_LAYER_BYTES = 4 * 64 * 528 + 64 * 132
_TARGET_PAGE_LAYERS = 11
# Rough estimate of what a rank holds beyond stored weights, state rows, and
# pages: operator scratch, graphs, collectives, and the CUDA context. It serves
# every rank count and speculation setting.
_GRAPH_BYTES = 6656 << 20


def _costs(state_row_bytes, graph_bytes, *, ranks, speculating):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    # Decode context parallelism shards each page's latent records over the ranks.
    return RankCosts(state_row_bytes, layers * _PAGE_LAYER_BYTES, graph_bytes,
                     page_tokens=256 * ranks)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank
    from .frontend import chat_template_asset

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity, "speculative_fixed_width": speculation.fixed_width}
    config = PreparationConfig(mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        image_concurrency=plan.image_concurrency, language_model_only=plan.language_model_only,
        max_active=plan.concurrency, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain, context_parallel=plan.context_parallel, stream_overlap=plan.stream_overlap,
        context=plan.recipe.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, kernel_cache=kernel_cache,
        # SM103 runs routed experts through trtllm-gen with NVFP4 activations at every size.
        expert_activation="a4" if any(gpu.capability == (10, 3) for gpu in plan.gpus) else "a16",
        # Expanded scales take 2 x 288 x 4096 x 2048 / 16 bytes per routed layer over the ranks: about
        # 4.9 GB per rank at TP4, which GB10 holds beside its weights, but 9.7 GB at TP2, which it does not.
        expand_expert_scales=plan.world_size >= 4,
        mtp=speculation is not None, **drafting)
    # A refusal or a failed template download is reported here, before any rank starts, so that its guidance
    # reaches the command line at every rank count; a rank that fails during preparation prints a traceback.
    from ignition.preparation.checkpoint import checkpoint_assets_root

    assets_root = checkpoint_assets_root(Path(snapshot).resolve(), plan.checkpoint.overlay)
    if (speculation is not None and plan.checkpoint.overlay is None
            and any(gpu.capability == (10, 3) for gpu in plan.gpus)):
        from ignition.preparation.quantization import quantized_layers

        from .weights import require_draft_experts_supported

        require_draft_experts_supported(
            quantized_layers(json.loads((assets_root / "config.json").read_text())), sm103=True)
    config = replace(config, chat_template=chat_template_asset(assets_root, cache_dir=plan.hub_cache))
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="glm53", served_model_name="GLM-5.3-Flash", prepare=prepare, usage=usage,
    # Bounds span HLE, BabyVision, and coding presets: https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
        ranges={"temperature": SamplingRange(.95, 1.), "top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(
        alias="glm-5.3-flash", repository="nvidia/GLM-5.3-Flash-FP4-Local",
        revision="76348ddb2be2e35e39ecfc00e75dc07703f4f3d5", architecture="Glm5NextForConditionalGeneration",
        stored_bytes=177_421_592_008,
        # Language model, draft layer, output head, and replicated vision tower.
        device_bytes=177_421_592_008,
        # Decoder layer 45, the MTP module.
        draft_bytes=4_301_658_112), Checkpoint(
        alias="glm-5.3-flash-nvfp4", repository="local-inference-lab/GLM-5.3-Flash-NVFP4",
        revision="378ca54585c46542bad1f3cb3ed0d73ae51cdb62", architecture="Glm5NextForConditionalGeneration",
        stored_bytes=198_042_331_512, device_bytes=198_042_331_512,
        draft_bytes=7_843_920_512,
        # The draft layer proposes with an NVFP4 copy of the BF16 vocabulary head, made at load: for the
        # 154,880 x 4096 head, 4-bit values, one E4M3 scale per 16 values with each TP4 shard's rows padded
        # to 128 for the swizzle, and an FP32 gain per shard, summed over four shards. Weights estimates add
        # it without speculation too, which overstates them by 89 MB per rank at TP4 with --no-mtp.
        expansion_bytes=356_909_072),
        # The TrellisMX release overlays its routed experts on the pinned carrier: the
        # carrier's 19_339_670_136 payload bytes plus 165_721_576_928 bytes of sidecars,
        # one per layer and rank of a TP4 partition, so only TP4 can load it. Tensors stay
        # as stored, but preparation keeps more than the stored bytes: each sidecar's b12x
        # bundle holds 1_021_002_880 bytes (its payload, one repacked copy of both UE8M0
        # scale planes, and the intermediate rotations; measured on the CPU-prepared
        # bundle). The attached P8 runtime's repacked planes replace the bundle's
        # byte-identical copies, and the runtime adds 905_348 bytes of its own, almost all
        # its packed boundary vectors (measured on a CPU-attached runtime). The 172 sidecars
        # thus occupy 1_021_908_228 bytes each, 58 MB beyond their stored size.
        Checkpoint(
            alias="glm-5.3-flash-trellismx", repository="brandonmusic/glm-5.3-flash-tr3-fp8",
            revision="80cbe535eabbeddfbd567e376a09ac559bcbac75", architecture="Glm5NextForConditionalGeneration",
            stored_bytes=185_061_247_064, device_bytes=185_061_247_064, expansion_bytes=10_046_638_288,
            # Layer 45's four sidecars, as prepared, plus its 25 non-expert tensors.
            draft_bytes=4_457_303_696, tensor_parallel=(4,), overlay=TRELLISMX_OVERLAY),),
    context=1048576, page_tokens=256, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5, token_capacity=128), max_draft_tokens=7, verification_rows=0,
    # Shared experts, KDA gate projections, and decode query chains run on an auxiliary stream.
    stream_overlap=True, max_images=4, image_bytes=32 << 20, language_model_only=True,
    # b12x retains activation scales for A4 prefill launches in W4A16 expert paths when set.
    b12x_environment=MappingProxyType({"B12X_W4A16_A4_PREFILL": "1"}),
    ranks=MappingProxyType({
        # DGX Station can hold GLM on one GPU; TP1 must remain a supported rank count.
        # One rank holds every head: twice TP2's recurrent rows less the replicated part.
        1: _costs(161_267_200, _GRAPH_BYTES, ranks=1, speculating=True),
        2: _costs(81_608_192, _GRAPH_BYTES, ranks=2, speculating=True),
        4: _costs(41_778_688, _GRAPH_BYTES, ranks=4, speculating=True),
    }),
    plain_ranks=MappingProxyType({
        1: _costs(149_230_592, _GRAPH_BYTES, ranks=1, speculating=False),
        2: _costs(75_420_672, _GRAPH_BYTES, ranks=2, speculating=False),
        4: _costs(38_515_712, _GRAPH_BYTES, ranks=4, speculating=False),
    }),
)
