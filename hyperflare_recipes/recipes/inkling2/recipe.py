# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 checkpoint declarations and preparation entry points."""

from types import MappingProxyType
from dataclasses import replace

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation

from .frontend import SAMPLING_DEFAULTS


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare Mini on one GPU or Small on four GPUs.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    """
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    if plan.prompt_logprobs:
        raise ValueError("Inkling 2 preparation does not capture prompt scoring graphs")
    speculation = plan.speculation
    drafting = {} if speculation is None else dict(speculative_widths=speculation.widths,
        speculative_fixed_width=speculation.fixed_width, speculative_token_capacity=plan.speculative_token_capacity,
        speculative_graph_budget=speculation.graph_budget or 64,
        draft_head_quantization=speculation.draft_head_quantization,
        markov_head_quantization=speculation.markov_head_quantization)
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs, compile_workers=plan.compile_workers,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain,
        context=plan.recipe.context, prefill_chunk_size=plan.prefill_chunk,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache,
        drafting=speculation is not None, vision=plan.recipe.max_images > 0 and not plan.language_model_only,
        mtp_only=speculation is not None and speculation.mtp_only,
        benchmark_selection=plan.benchmark_selection, **drafting)
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(name="inkling2", served_model_name="Inkling-2-Small-NVFP4", prepare=prepare, usage=usage,
    sampling_defaults=SAMPLING_DEFAULTS,
    checkpoints=(Checkpoint(alias="inkling-2-small-nvfp4", repository="/data/models/Inkling-2-Small-NVFP4",
        revision="local", architecture="Inkling2ForConditionalGeneration", stored_bytes=344_366_116_864,
        device_bytes=344_366_116_864, expansion_bytes=5_673_717_760, draft_bytes=2_684_203_392),),
    context=1048576, page_tokens=128, concurrency=8, max_concurrency=32, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64),
    max_draft_tokens=7, max_images=4, image_bytes=32 << 20, verification_rows=0, language_model_only=True,
    ranks=MappingProxyType({4: RankCosts(31_293_440, 655_360, 5 << 30)}),
    plain_ranks=MappingProxyType({4: RankCosts(26_935_296, 524_288, 5 << 30)}))


# Mini benchmark defaults: temperature 1, top_p 1, no top-k
# filter, and neutral penalties. The TML encoder supplies reasoning effort 0.9.
MINI_SAMPLING_DEFAULTS = SamplingDefaults(temperature=1., top_p=1., top_k=-1)

MINI_RECIPE = replace(RECIPE, name="inkling2_mini", served_model_name="inkling-2-mini",
    sampling_defaults=MINI_SAMPLING_DEFAULTS, max_concurrency=8, prefill_chunk=8192,
    speculation=Speculation(draft_tokens=7, token_capacity=64, packed_verification=True, graph_budget=64),
    draft_head_quantizations=("nvfp4", "mxfp8"), markov_head_quantizations=("nvfp4",),
    checkpoints=(Checkpoint(alias="inkling-2-mini", repository="thinkingmachines/Inkling-2-Mini",
        revision="ed40e7da625f594fc9c4990f5f66be24873c22d7", architecture="Inkling2LoopForConditionalGeneration",
        stored_bytes=19_525_870_752, device_bytes=20_263_810_112, draft_bytes=737_939_360),
        Checkpoint(alias="tml-d24-varloop-research", repository="thinkingmachines/TML-d24-varloop-research",
        revision="657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2", architecture="Inkling2LoopForConditionalGeneration",
        served_model_name="TML-d24-varloop-research",
        stored_bytes=19_525_870_752, device_bytes=20_263_810_112, draft_bytes=737_939_360),),
    ranks=MappingProxyType({1: RankCosts(107_347_968, 2_621_440, 3 << 30)}),
    plain_ranks=MappingProxyType({1: RankCosts(98_435_072, 2_097_152, 3 << 30)}))
