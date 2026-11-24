# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""ERNIE 5.5 Flash Preview recipe definition and tensor-parallel memory geometry."""

from types import MappingProxyType

from ignition.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    SamplingRange,
    Speculation,
)

from .config import MODEL_ID, REVISION


def prepare(
    snapshot,
    plan,
    *,
    compile_forwards,
    kernel_cache,
    progress,
    retain_graphs=False,
    core_storage=None,
    ranks=None,
):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = (
        {}
        if speculation is None
        else {
            "speculative_widths": speculation.widths,
            "speculative_token_capacity": plan.speculative_token_capacity,
            "speculative_graph_budget": speculation.graph_budget,
            "speculative_fixed_width": speculation.fixed_width,
            "packed_verification": speculation.packed_verification,
            "draft_prefix_selection": speculation.draft_prefix_selection,
            "draft_head_quantization": speculation.draft_head_quantization,
            "markov_head_quantization": speculation.markov_head_quantization,
            "vocabulary_parallel_sampling": speculation.vocabulary_parallel_sampling,
        }
    )
    config = PreparationConfig(
        mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        collective_tuning=plan.collective_tuning,
        collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers,
        autotune_effort=plan.autotune_effort,
        tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        roce_ring=plan.roce_ring,
        roce_chain=plan.roce_chain,
        compile_decoder=compile_forwards,
        retain_graphs=retain_graphs,
        max_active=plan.concurrency,
        context=plan.recipe.context,
        prefill_chunk_size=plan.prefill_chunk,
        minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes,
        kernel_cache=kernel_cache,
        dspark=speculation is not None,
        stream_overlap=plan.stream_overlap,
        expert_scale_compression=plan.expert_scale_compression,
        mla_compute_mode=plan.mla_compute_mode,
        **drafting,
    )
    return prepare_group(
        snapshot,
        config,
        prepare_rank,
        progress=progress,
        core_storage=core_storage,
        ranks=ranks,
        checkpoint=plan.checkpoint,
    )


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


# Exact continuation row & page geometry from pool.py:
#   page_bytes = 37 * 1152 + 6 * (128 * 656 + 128 * 132) = 42,624 + 605,184 = 647,808 B / 128-tok page
#   window_bytes = latent_page_bytes(128) = 74,880 B
#   compressor state = 37 * 2 * (128 * 512 * 4) = 19,398,656 B
#   row_bytes (dspark=True)  ~= 37*74,880 + 19,398,656 + 647,808 + 3*74,880 + 8,200 ~= 23,050,000 B
#   row_bytes (dspark=False) ~= 37*74,880 + 19,398,656 + 647,808 + 8,200 ~= 22,825,000 B
RECIPE = Recipe(
    name="ernie55",
    served_model_name="ERNIE-5.5-Flash-Preview",
    prepare=prepare,
    usage=usage,
    mla_compute_modes=("auto",),
    mla_compute_mode="auto",
    expert_scale_compression="csf",
    draft_head_quantizations=("nvfp4", "mxfp8"),
    markov_head_quantizations=("nvfp4", "fp8"),
    sampling_defaults=SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        ranges={"repetition_penalty": SamplingRange(1.0, 1.2)},
    ),
    checkpoints=(
        Checkpoint(
            alias="ernie-5.5-flash",
            repository=MODEL_ID,
            revision=REVISION,
            architecture="Ernie5_5_MoeForCausalLM",
            stored_bytes=165_882_400_000,
            device_bytes=165_700_000_000,
            draft_bytes=11_650_000_000,
            scale_compressed_device_bytes={"csf": 155_700_000_000},
        ),
    ),
    context=1048576,
    page_tokens=128,
    concurrency=32,
    max_concurrency=64,
    prefill_chunk=4096,
    speculation=Speculation(
        draft_tokens=7,
        token_capacity=128,
        packed_verification=True,
        draft_prefix_selection="gpu",
        graph_budget=64,
    ),
    max_draft_tokens=7,
    stream_overlap=True,
    vocabulary_parallel_world_sizes=(2, 4),
    ranks=MappingProxyType({world: RankCosts(23_059_968, 647_808, 6 << 30) for world in (2, 4)}),
    plain_ranks=MappingProxyType({world: RankCosts(22_834_944, 647_808, 6 << 30) for world in (2, 4)}),
)
