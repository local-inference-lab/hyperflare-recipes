# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare ERNIE 5.5 Flash Preview weights, hybrid HCA/DSA pools, and CUDA graph catalogs."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch

from hyperflare_recipes.components.dspark import (
    DSparkForwards,
    PreparedContext,
    PreparedMarkov,
    SpeculativePreparation,
)
from hyperflare_recipes.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from hyperflare_recipes.components.paged_generation import PreparedGeneration
from hyperflare_recipes.components.paged_maintenance import PreparedMaintenance
from hyperflare_recipes.components.paged_resume import PreparedResume
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.prompt_logprobs import (
    PreparedScoredPrefill,
    PromptScoring,
    prefill_graph_count,
    prefill_modes,
)
from ignition.components.capacity import (
    batch_capacities,
    prefill_capacities,
    request_capacity,
    serving_modes,
    speculative_token_capacity,
)
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.bootstrap import BasePreparationConfig
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
    create_result_broadcast_runtime,
    create_roce_runtime,
)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.host_memory import release_unused, virtual_device_memory
from ignition.preparation.model import PreparedModel
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.progress import phase

from .contract import VOCABULARY, prepare as prepare_contract
from .decoder import DraftScratch, PreparedDecoder, Scratch
from .dspark import PreparedDraft
from .frontend import chat_template_asset
from .pool import ContinuationPool, PAGE_TOKENS, continuation_layout, page_bytes
from .recipe import RECIPE
from .weights import ModelWeights

_CONTEXT = partial(PreparedContext, cache_format="deepseek_v4", compact_prefill=True)


def _prepare_draft(*args, draft_vocabulary_runtime, vocabulary_runtime=None, **kwargs):
    runtime = vocabulary_runtime if draft_vocabulary_runtime is None else draft_vocabulary_runtime
    return PreparedDraft(*args, vocabulary_runtime=runtime, **kwargs)


def _forward_factories(mla_compute_mode, *, draft_vocabulary_runtime=None, max_cache_pages=None, examples=None):
    return (
        partial(
            PreparedDecoder,
            mla_compute_mode=mla_compute_mode,
            max_cache_pages=max_cache_pages,
            examples=examples,
        ),
        partial(
            _prepare_draft,
            draft_vocabulary_runtime=draft_vocabulary_runtime,
            mla_compute_mode=mla_compute_mode,
            examples=examples,
        ),
    )


@dataclass(frozen=True)
class PreparationConfig(BasePreparationConfig):
    prompt_logprobs: bool = False
    devices: tuple[int, ...] = (0, 1)
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    roce_chain: bool = False
    context: int = 1048576
    max_active: int = 32
    minimum_pages: int = 8
    kv_pages: int | None = None
    dspark: bool = True
    draft_head_quantization: str | None = None
    markov_head_quantization: str | None = None
    vocabulary_parallel_sampling: bool = False
    speculative_token_capacity: int | None = 128
    speculative_graph_budget: int = 64
    stream_overlap: bool = True
    expert_scale_compression: str | None = "csf"
    expert_activation_mode: str = "a8"
    mla_compute_mode: str = "auto"
    packed_verification: bool | None = None
    draft_prefix_selection: str | None = None
    speculation_reference_graphs: bool = False

    def __post_init__(self):
        if self.mtp_only and (
            not self.dspark
            or max(self.speculative_widths, default=1) <= 1
            or self.speculative_fixed_width == 1
        ):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.speculative_widths and max(self.speculative_widths) > 8:
            raise ValueError("ERNIE 5.5 DSpark draft model supports at most seven draft tokens")
        from hyperflare_recipes.components.compressed_mla import validate_mla_compute_mode
        from ignition.declaration import Speculation

        validate_mla_compute_mode(self.mla_compute_mode)
        Speculation(
            self.draft_tokens,
            draft_head_quantization=self.draft_head_quantization,
            markov_head_quantization=self.markov_head_quantization,
            vocabulary_parallel_sampling=self.vocabulary_parallel_sampling,
        )
        self.speculative_shapes
        self.packed_speculative_shapes
        drafting = self.dspark and self.draft_tokens > 0
        if self.packed_verification is None:
            object.__setattr__(self, "packed_verification", drafting)
        elif type(self.packed_verification) is not bool:
            raise TypeError("packed_verification must be a boolean")
        if self.draft_prefix_selection is None:
            object.__setattr__(self, "draft_prefix_selection", "gpu" if drafting else "cpu")
        if self.draft_prefix_selection not in ("cpu", "gpu"):
            raise ValueError("draft prefix selection must be cpu or gpu")
        if drafting:
            if not self.packed_verification or self.draft_prefix_selection != "gpu":
                raise ValueError("ERNIE 5.5 DSpark requires packed GPU draft prefix selection")
            if self.speculative_fixed_width is not None or self.speculation_reference_graphs:
                raise ValueError("ERNIE 5.5 DSpark requires adaptive packed GPU rounds")
        elif self.packed_verification or self.draft_prefix_selection != "cpu" or self.speculation_reference_graphs:
            raise ValueError("packed GPU speculation requires positive DSpark draft depth")
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("ERNIE 5.5 expert scale compression must be None or 'csf'")
        if self.expert_activation_mode not in ("a8", "a16"):
            raise ValueError("ERNIE 5.5 expert activation mode must be 'a8' or 'a16'")
        if type(self.stream_overlap) is not bool:
            raise TypeError("stream_overlap must be a boolean")
        placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
        if (
            not isinstance(self.devices, tuple)
            or len(self.devices) not in (2, 4)
            or len(set(placements)) != len(self.devices)
            or self.hosts is not None
            and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
            or any(type(d) is not int or d < 0 for d in self.devices)
        ):
            raise ValueError("ERNIE 5.5 Flash preparation requires two or four distinct GPUs")
        from ignition.planning import host_shape_error

        error = host_shape_error(self.roce_ring, self.roce_chain, None if self.hosts is None else len(self.hosts))
        if error:
            raise ValueError(error)
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("ERNIE 5.5 context must be 128-aligned in [128, 1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64, context]")
        request_capacity(self.max_active)
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the prepared widths")
        if self.minimum_pages <= 0 or self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("page and snapshot budgets must be positive and headroom nonnegative")
        if self.kv_pages is not None and (type(self.kv_pages) is not int or self.kv_pages < self.minimum_pages):
            raise ValueError("kv_pages must cover the minimum page capacity")

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def prefill_capacities(self):
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return (
            tuple(
                (batch, width)
                for batch, width in speculative_shapes(
                    self.max_active, self.speculative_widths, self.speculative_token_capacity
                )
                if width == 1
            )
            if self.dspark
            else ()
        )

    @property
    def packed_speculative_shapes(self):
        from ignition.components.capacity import packed_speculative_shapes

        return (
            packed_speculative_shapes(
                self.max_active,
                max(self.speculative_widths),
                self.speculative_token_capacity,
                self.speculative_graph_budget,
            )
            if self.draft_tokens
            else ()
        )

    @property
    def decode_tokens(self):
        return max(self.request_capacity, max((tokens for _, tokens in self.packed_speculative_shapes), default=0))

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.dspark else 0

    @property
    def all_reduce_tokens(self):
        return max(
            self.decode_tokens,
            self.request_capacity * self.draft_tokens,
            *(capacity for capacity in self.prefill_capacities if capacity < 512),
        )

    @property
    def completion_token_capacity(self):
        return max(self.speculative_widths) if self.dspark else 1


def _prepare_speculation_catalog(config, speculation, *, add, buffers, forward_inputs, host_allocate, device):
    if speculation is None:
        return None
    from hyperflare_recipes.components.parallel_draft import PreparedParallelDraftRound

    confidence = None
    if config.draft_tokens:
        from hyperflare_recipes.components.packed_speculation import PreparedPackedSpeculativeRound
        from ignition.components.capacity import proposal_shapes

        slots = config.request_capacity
        confidence = host_allocate("proposal_confidence", (2, slots, config.draft_tokens), torch.float32)
        buffers["proposal_confidence"] = confidence.device_view
        prefix_limits = torch.zeros(slots, dtype=torch.int32, device=device)
        buffers["speculative_prefix_limits"] = prefix_limits
        packed_inputs = tuple(field for field in forward_inputs if field != "query_lengths")
        packed_inputs += ("speculative_prefix_limits",)
        packed_batches = {batch for batch, _ in config.packed_speculative_shapes}
        proposals = {}
        for batch, depth in sorted(proposal_shapes(config.max_active, config.draft_tokens), reverse=True):
            if batch not in packed_batches:
                continue
            name = f"proposal{batch}x{depth + 1}"
            source = PreparedParallelDraftRound(speculation, batch, depth + 1, name, proposal_only=True)
            source.prepare()
            proposals[batch] = source
    for batch, _ in reversed(config.speculative_shapes):
        name = f"speculative{batch}x1"
        forward = PreparedParallelDraftRound(speculation, batch, 1, name)
        forward.prepare()
        add(name, forward, "speculative", batch, 1, forward_inputs, enabled=True)
    for batch, packed_tokens in sorted(
        config.packed_speculative_shapes, key=lambda shape: (shape[1], shape[0]), reverse=True
    ):
        logical_width = max(config.speculative_widths)
        target_name = f"verification{batch}x{logical_width}-tokens{packed_tokens}"
        target = PreparedParallelDraftRound(
            speculation,
            batch,
            logical_width,
            target_name,
            generate_proposals=False,
            target_token_capacity=packed_tokens,
        )
        target.prepare()
        name = f"packed-speculative{batch}x{logical_width}-drafts{config.draft_tokens}-tokens{packed_tokens}"
        forward = PreparedPackedSpeculativeRound(
            proposals[batch], target, prefix_limits, confidence.device_view, name=name
        )
        add(
            name,
            forward,
            "speculative",
            batch,
            logical_width,
            packed_inputs,
            enabled=True,
            packed_tokens=packed_tokens,
            draft_tokens=config.draft_tokens,
        )
    return confidence


def _collective_runtimes(config, group, device, *, coordinator, core_storage, verification_samples):
    tokens = max(config.prefill_chunk_size, config.decode_tokens, config.request_capacity * config.draft_tokens)
    sample_rows = max(verification_samples, config.request_capacity * config.draft_tokens)
    layouts = ()
    if config.dspark:
        from hyperflare_recipes.components.draft_sampling import PreparedDraftBlockBroadcast
        from hyperflare_recipes.components.speculation import SpeculativeAcceptance

        batch = max(config.batch_capacities)
        layouts = (SpeculativeAcceptance.broadcast_layout(batch, max(config.speculative_widths)),)
        if config.draft_tokens:
            layouts += (PreparedDraftBlockBroadcast.broadcast_layout(batch, config.draft_tokens),)
    _, local_vocab = TensorParallelShard(0, len(config.devices)).padded_partition(VOCABULARY, alignment=64)
    if coordinator.spans_hosts:
        from ignition.components.collectives import packed_broadcast_layout

        result_bytes = max((packed_broadcast_layout(layout.values())[1] for layout in layouts), default=0)
        gather_bytes = max(sample_rows * local_vocab * 4, (result_bytes + 15) // 16 * 16)
        runtime = create_roce_runtime(
            group,
            device,
            max_tokens=2 * config.all_reduce_tokens,
            pipe_tokens=2 * tokens,
            hidden_size=4096,
            gather_bytes=gather_bytes,
            ring=config.roce_ring,
            ring_open=config.roce_chain,
            core_storage=core_storage,
        )
        return runtime, runtime, runtime, runtime if layouts else None, runtime if config.dspark else None
    ipc = core_storage.peer_storage
    # Size collective channels to cover HCA BF16 VHA [T, 16, 512] (= 2x BF16 [T, 4096]).
    collective = create_all_reduce_runtime(
        group,
        device,
        collective_tuning=config.collective_tuning,
        max_tokens=2 * config.all_reduce_tokens,
        hidden_size=4096,
        ipc=ipc,
    )
    # 2-microbatch DMA prefill reduces half-chunk VHA [T/2, 16, 512] BF16 (= T * 4096 BF16 equiv)
    # and full non-pipelined 512-token VHA [512, 16, 512] BF16 (= 1024 * 4096 BF16 equiv).
    dma_equiv_tokens = max(config.prefill_chunk_size, 1024)
    prefill = create_dma_all_reduce_runtime(
        group,
        device,
        max_tokens=dma_equiv_tokens,
        hidden_size=4096,
        ipc=ipc,
    )
    vocabulary = create_all_gather_runtime(
        group,
        device,
        max_rows=sample_rows,
        columns=local_vocab * len(config.devices) * 2,
        ipc=ipc,
    )
    draft_vocabulary = (
        create_all_gather_runtime(
            group,
            device,
            max_rows=sample_rows,
            columns=local_vocab * len(config.devices),
            ipc=ipc,
        )
        if config.dspark
        else None
    )
    result = create_result_broadcast_runtime(group, device, layouts=layouts, ipc=ipc) if layouts else None
    return collective, prefill, vocabulary, result, draft_vocabulary


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage=None):
    if core_storage is None:
        raise ValueError("ERNIE 5.5 preparation requires core storage")
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    metadata_path = manifest.metadata_path
    coordinator = CollectivePreparation(group)
    with core_storage.transient():
        weights = ModelWeights.load(
            manifest,
            device,
            shard=shard,
            progress=progress,
            dspark=config.dspark,
            confidence=config.draft_tokens > 0,
            expert_scale_compression=config.expert_scale_compression,
            expert_activation_mode=config.expert_activation_mode,
            draft_head_quantization=config.draft_head_quantization,
            markov_head_quantization=config.markov_head_quantization,
            vocabulary_parallel_sampling=config.vocabulary_parallel_sampling,
        )
        release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    if progress is not None:
        progress(f"Resident model tensors: {weights.allocated_bytes / (1 << 30):.3f} GiB per rank")
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    with phase(progress, "Allocate model state and collectives"):
        layout = continuation_layout(weights.config, dspark=config.dspark)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete ERNIE 5.5 continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        execution_rows = slots * (max(config.speculative_widths) if config.dspark else 1)
        state_rows = snapshots + execution_rows
        pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        if config.kv_pages is not None:
            pages = config.kv_pages
        bytes_per_page = page_bytes(weights.config)
        total_memory = virtual_device_memory(device, group, tuning_replica=config.tuning_replica)
        maximum_pages = total_memory // bytes_per_page
        pool = ContinuationPool.allocate(
            weights.config,
            state_rows + 1,
            pages,
            device,
            storage=core_storage,
            maximum_pages=maximum_pages,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + 2 * slots),
            dspark=config.dspark,
        )
        requests = DeviceRequestState.allocate(device, vocab_size=VOCABULARY, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=VOCABULARY, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        completion = CompletionOutputs.allocate(
            (2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity), device=device
        )
        verification_samples = (
            slots * max(config.speculative_widths) if config.packed_verification else config.decode_tokens
        )
        (
            collective,
            prefill_collective,
            vocabulary_collective,
            result_collective,
            draft_vocabulary_collective,
        ) = _collective_runtimes(
            config,
            group,
            device,
            coordinator=coordinator,
            core_storage=core_storage,
            verification_samples=verification_samples,
        )
        sampling_collective = logprob_collective = None
        if config.vocabulary_parallel_sampling and shard.world_size > 1:
            sampling_collective = (
                collective
                if coordinator.spans_hosts
                else create_all_gather_runtime(
                    group,
                    device,
                    max_rows=verification_samples,
                    columns=64 * shard.world_size,
                    ipc=core_storage.peer_storage,
                )
            )
            logprob_collective = (
                collective
                if coordinator.spans_hosts
                else create_all_gather_runtime(
                    group,
                    device,
                    max_rows=verification_samples,
                    columns=512 * shard.world_size,
                    ipc=core_storage.peer_storage,
                )
            )
        identity = hashlib.sha256(
            json.dumps(
                {
                    "config": json.loads((metadata_path / "config.json").read_text()),
                    "headers": [s.header_sha256 for s in manifest.shards],
                    "world_size": shard.world_size,
                    "contract": "ernie55.graph-inputs.v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers,
            collective_tuning=config.collective_tuning,
            collective_ladder=config.collective_ladder,
            autotune_effort=config.autotune_effort,
            tuning_replica=config.tuning_replica,
            native_dense_activations=config.native_dense_activations,
            lock_gpu_clocks=config.lock_gpu_clocks,
            device=device,
            checkpoint=identity,
            workload_schema="ernie55-text",
            numerical_policy="default",
            coordinator=coordinator,
            headroom_bytes=config.headroom_bytes,
            benchmark_storage=core_storage,
        )
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        examples = {}
        prefill_storage = Scratch.allocate(
            config.prefill_chunk_size, 1, shard.world_size, device, blocks=blocks
        )
        decoder_storage = Scratch.allocate(
            config.decode_tokens, verification_samples, shard.world_size, device, blocks=blocks
        )
        decoder_factory, draft_factory = _forward_factories(
            config.mla_compute_mode,
            draft_vocabulary_runtime=draft_vocabulary_collective,
            max_cache_pages=maximum_pages,
            examples=examples,
        )
        buffers = dict(
            inputs.buffers(),
            **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing,
            memory_capacity=pool.memory_capacity,
            sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame,
            submission_sequence=sequence,
            **completion.buffers(),
        )
        forwards, signatures, metadata, scratch = {}, {}, {}, {}
        speculation = None
        if config.dspark:
            speculation = SpeculativePreparation(
                weights,
                None,
                pool,
                requests,
                inputs,
                decoder_storage,
                draft_tokens=config.draft_tokens,
                completion=completion,
                frame=frame,
                sequence=sequence,
                session=session,
                workspace=arena,
                runtime=collective,
                vocabulary_runtime=vocabulary_collective,
                markov_runtime=None,
                coordinator=coordinator,
                result_runtime=result_collective,
                block_proposal_broadcast=True,
                vocabulary_parallel_sampling=config.vocabulary_parallel_sampling,
                sampling_runtime=sampling_collective,
                logprob_runtime=logprob_collective,
                forwards=DSparkForwards(decoder_factory, DraftScratch, _CONTEXT, draft_factory, PreparedMarkov),
                page_tokens=PAGE_TOKENS,
            )

    scoring = PromptScoring.allocate(
        config,
        device,
        buffers,
        frame=frame,
        group=group,
        ipc=core_storage.peer_storage,
        runtime=collective,
        coordinator=coordinator,
    )
    del manifest
    modes = serving_modes(config.dspark, config.mtp_only)
    count = (
        len(modes) * (prefill_graph_count(config.prefill_capacities, scoring) + 4 * len(config.batch_capacities))
        + len(config.speculative_shapes)
        + len(config.packed_speculative_shapes)
        + (len(config.batch_capacities) if False in modes else 0)
    )
    catalog_bytes = {}
    catalog_previous = torch.cuda.memory_allocated(device)
    with phase(progress, "Construct graph catalog", total=count) as stage:

        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0, draft_tokens=None):
            nonlocal catalog_previous
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(
                kind,
                rows,
                tokens,
                tuple(fields),
                speculation_enabled=enabled,
                packed_token_capacity=packed_tokens,
                draft_tokens=draft_tokens,
                confidence_width=draft_tokens if "speculative_prefix_limits" in fields else 0,
            )
            allocated = torch.cuda.memory_allocated(device)
            catalog_bytes[kind] = catalog_bytes.get(kind, 0) + allocated - catalog_previous
            catalog_previous = allocated
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, capacities in prefill_modes(
                config.prefill_capacities, () if enabled else config.batch_capacities, scoring
            ):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    storage = prefill_storage if mode == "prefill" else decoder_storage
                    scratch[name] = storage.rows(capacity, rows)
                    decoder = decoder_factory(
                        mode_weights,
                        None,
                        scratch[name],
                        metadata[name],
                        pool,
                        mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        runtime=prefill_collective if mode == "prefill" and capacity >= 512 else collective,
                        vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator,
                        session=session,
                        name=name,
                        workspace=arena,
                    )
                    batch = PreparedBatch(
                        inputs,
                        requests,
                        metadata[name],
                        mode=mode,
                        shard=shard,
                        pool=pool,
                        page_tokens=PAGE_TOKENS,
                    )
                    forward = PreparedGeneration(
                        decoder,
                        batch,
                        completion,
                        frame,
                        sequence,
                        runtime=collective,
                        coordinator=coordinator,
                        session=session,
                        name=name + ".generation",
                        workspace=arena,
                        context_factory=_CONTEXT,
                    )
                    add(
                        name,
                        forward,
                        kind,
                        rows,
                        capacity if mode == "prefill" else 1,
                        forward_inputs,
                        enabled=enabled,
                    )
        confidence = _prepare_speculation_catalog(
            config,
            speculation,
            add=add,
            buffers=buffers,
            forward_inputs=forward_inputs,
            host_allocate=core_storage.host_allocate,
            device=device,
        )
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(
                    mode_weights,
                    requests,
                    pool,
                    inputs,
                    completion,
                    frame,
                    sequence,
                    rows=capacity,
                    runtime=collective,
                    vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator,
                    session=session,
                    name=name,
                    logits_dtype=torch.float32,
                    workspace=arena,
                )
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for m_kind in ("init", "restore", "snapshot"):
                    name = f"{m_kind}{capacity}" + suffix
                    forward = PreparedMaintenance(
                        m_kind,
                        requests,
                        pool,
                        commands,
                        inputs,
                        completion,
                        frame,
                        sequence,
                        runtime=collective,
                        coordinator=coordinator,
                        session=session,
                        name=name,
                        capacity=capacity,
                        admission_buffers=admission,
                        draft_state=enabled,
                        page_tokens=PAGE_TOKENS,
                    )
                    fields = (
                        row_inputs
                        if m_kind == "snapshot"
                        else ("request_state_rows", *admission.__dataclass_fields__)
                    )
                    add(
                        name,
                        forward,
                        m_kind,
                        capacity,
                        1,
                        (*fields, "page_tables", *commands.buffers()),
                        enabled=enabled,
                    )
    transient.close()
    release_unused()
    if progress is not None:
        progress(
            f"before tuning: {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
            f"{torch.cuda.memory_reserved(device) >> 20} MiB reserved, "
            f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB free"
        )
    with phase(progress, "Prepare and tune kernels") as stage, core_storage.transient():
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id
    if speculation is not None:
        speculation.release_examples()
    metadata.clear()
    scratch.clear()
    examples.clear()
    prefill_storage = decoder_storage = storage = decoder = batch = None
    arena.release_preparation()
    release_unused()

    def materialize():
        pool.clear_for_preparation()

    graphs, compiled = capture_forwards(
        forwards,
        signatures,
        session=session,
        device=device,
        stream=stream,
        pool=graph_pool,
        collective=collective,
        coordinator=coordinator,
        compile_decoder=config.compile_decoder,
        progress=progress,
        cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs,
        materialize=materialize,
        warmup_storage=core_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative"),
    )
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    free, total = torch.cuda.mem_get_info(device)
    assets = {
        "tokenizer": {"path": str(metadata_path / "tokenizer.json")},
        "chat_template": chat_template_asset(),
    }
    buffers["request_pending_token"] = requests.pending_token
    frontend = prepare_frontend(metadata_path, eos_token_ids=[2, 8])
    server_config = {
        "identity": identity,
        "model_id": snapshot.name,
        "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages,
        "state_rows": state_rows,
        "snapshots": snapshots,
        "page_tokens": PAGE_TOKENS,
        "frontend": frontend,
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
        "assets": assets,
        "model_contract": prepare_contract(
            signatures,
            buffers,
            pool,
            page_tokens=PAGE_TOKENS,
            draft_lag=0 if config.dspark else None,
            rows=None,
        ),
    }
    memory = {
        "rank": rank,
        "world_size": shard.world_size,
        "device": config.devices[rank],
        "catalog_operand_bytes": catalog_bytes,
        "weights_bytes": weights.allocated_bytes,
        "mapped_host_bytes": 0,
        "offloaded_bytes": 0,
        "expert_scale_compression": config.expert_scale_compression,
        "expert_activation_mode": config.expert_activation_mode,
        "mla_compute_mode": config.mla_compute_mode,
        "draft_head_quantization": config.draft_head_quantization,
        "markov_head_quantization": config.markov_head_quantization,
        "vocabulary_parallel_sampling": config.vocabulary_parallel_sampling,
        "target_logits_dtype": "float32",
        "draft_logits_dtype": "bfloat16" if config.dspark else None,
        "continuation_bytes": pool.backing.nbytes,
        "state_rows": state_rows,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "pages": pages,
        "page_bytes": bytes_per_page,
        "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * bytes_per_page,
        "completion_output_bytes": completion.nbytes,
        "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs),
        "prefill_capacities": config.prefill_capacities,
        "free_device_bytes": free,
        "total_device_bytes": total,
        "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning,
        "compiled_decoder": config.compile_decoder,
        "draft_prefix_selection": config.draft_prefix_selection,
    }
    if scoring is not None:
        memory.update(scoring.memory_report())
    if config.dspark:
        server_config.update(
            speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, max(config.speculative_widths), config.speculative_token_capacity
            ),
            speculative_fixed_width=config.speculative_fixed_width,
        )
        memory.update(
            speculative_shapes=config.speculative_shapes,
            packed_speculative_shapes=config.packed_speculative_shapes,
            speculative_graph_budget=config.speculative_graph_budget,
            packed_speculative_graphs=len(config.packed_speculative_shapes),
        )
    disposable = []
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
    model = PreparedModel(
        rebuild_weights=declare_weights(
            weights, components=("embedding", "norm", "head", "head_mix", "draft")
        ),
        device=config.devices[rank],
        graphs=graphs,
        buffers=buffers,
        signatures=signatures,
        max_active=config.max_active,
        session=session,
        server_config=server_config,
        memory_report=memory,
        export_zero_ranges=(
            *(() if scoring is None else scoring.zero_ranges()),
            *storage_ranges(disposable),
            *completion.zero_ranges(),
        ),
        resources=(
            scoring,
            weights,
            pool,
            requests,
            inputs,
            admission,
            commands,
            forwards,
            compiled,
            stream,
            graph_pool,
            session,
            collective,
            prefill_collective,
            vocabulary_collective,
            draft_vocabulary_collective,
            result_collective,
            sampling_collective,
            logprob_collective,
            coordinator,
            speculation,
            confidence,
        ),
        close_resources=tuple(
            dict.fromkeys(
                c
                for c in (
                    scoring,
                    sampling_collective,
                    logprob_collective,
                    result_collective,
                    prefill_collective,
                    vocabulary_collective,
                    draft_vocabulary_collective,
                    collective,
                )
                if c is not None
            )
        ),
    )
    if confidence is not None:
        model.proposal_confidence = confidence.host_view
    return model
