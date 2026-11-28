# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared b12x GQA over explicit cache pages, metadata and shared scratch."""

import torch
import triton
from b12x.attention import paged, paged_decode, paged_kv_write, paged_nvfp4
from b12x.preparation import PreparedCall, plan_from_handle

from ignition.preparation.persistent import persistent_copy, persistent_empty
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer


_BOUND_OPERAND = object()


def _scratch(specs, workspace):
    specs = tuple(specs)
    if not specs:
        return []
    values = planned_scratch(specs, workspace)
    return list(values) if isinstance(values, tuple) else [values]


@torch.library.custom_op("flare::paged_attention", mutates_args=("output", "scratch", "live_metadata"))
def _attend(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pages: torch.Tensor,
            lengths: torch.Tensor, cu: torch.Tensor, sink: torch.Tensor | None,
            output: torch.Tensor, scratch: list[torch.Tensor], k_descale: torch.Tensor | None,
            v_descale: torch.Tensor | None, selected_blocks: torch.Tensor | None,
            handle: int, decode: bool, window: int,
            schedule_storage: torch.Tensor | None, schedule_slot: int, output_gate: torch.Tensor | None,
            metadata_seeds: list[torch.Tensor], live_metadata: list[torch.Tensor],
            chunk_pages_lut: torch.Tensor | None, relative_bias: torch.Tensor | None,
            relative_bias_factors: torch.Tensor | None, relative_bias_projection: torch.Tensor | None,
            tail_key: torch.Tensor | None, tail_value: torch.Tensor | None) -> None:
    plan = plan_from_handle(handle)
    if plan.component_id == "flare.paged_tma":
        from b12x.preparation.types import require_prepared

        require_prepared(plan, plan.component_id).run(q, key, value, pages, lengths, cu, output, scratch[0], output_gate)
        return
    arguments = dict(q=q, k_cache=key, v_cache=value, page_table=pages, cache_seqlens=lengths,
                     cu_seqlens_q=cu, attention_sink_bias=sink, output=output, scratch=scratch,
                     k_descale=k_descale, v_descale=v_descale, relative_attention_bias=relative_bias)
    if decode:
        arguments.update(tail_k_cache=tail_key, tail_v_cache=tail_value)
    else:
        arguments.update(relative_bias_factors=relative_bias_factors, relative_bias_projection=relative_bias_projection)
    if metadata_seeds:
        arguments["replay_metadata"] = paged.ReplayMetadata(
            tuple(live_metadata if live_metadata else metadata_seeds), chunk_pages_lut)
    if selected_blocks is not None:
        arguments["q2k_indices"] = selected_blocks
    if schedule_slot >= 0:
        from b12x.attention.paged.schedule import PrefillSchedule

        arguments["external_schedule"] = PrefillSchedule.from_plan(plan).bind(schedule_storage[schedule_slot])
    if decode:
        paged_decode.run(paged_decode.bind(plan, **arguments))
    else:
        paged.run(binding=paged.bind(plan, **arguments, window_left=window), plan=plan)


@_attend.register_fake
def _fake_attend(q, key, value, pages, lengths, cu, sink, output, scratch, k_descale, v_descale, selected_blocks, handle, decode, window, schedule_storage, schedule_slot, output_gate, metadata_seeds, live_metadata, chunk_pages_lut, relative_bias, relative_bias_factors, relative_bias_projection, tail_key, tail_value):
    pass


@torch.library.custom_op("flare::paged_nvfp4_attention", mutates_args=("output", "scratch"))
def _attend_nvfp4(q: torch.Tensor, k_cache: torch.Tensor, k_scales: torch.Tensor,
                  v_cache: torch.Tensor, v_scales: torch.Tensor, pages: torch.Tensor,
                  lengths: torch.Tensor, cu: torch.Tensor, sink: torch.Tensor | None,
                  output: torch.Tensor, scratch: list[torch.Tensor], handle: int,
                  relative_bias: torch.Tensor | None, relative_bias_factors: torch.Tensor | None,
                  relative_bias_projection: torch.Tensor | None) -> None:
    plan = plan_from_handle(handle)
    paged_nvfp4.run(paged_nvfp4.bind(
        plan, scratch=scratch, q=q, k_cache=k_cache, k_scales=k_scales,
        v_cache=v_cache, v_scales=v_scales, output=output, page_table=pages,
        cache_seqlens=lengths, cu_seqlens_q=cu, attention_sink_bias=sink,
        relative_attention_bias=relative_bias, relative_bias_factors=relative_bias_factors,
        relative_bias_projection=relative_bias_projection))


@_attend_nvfp4.register_fake
def _fake_attend_nvfp4(q, k_cache, k_scales, v_cache, v_scales, pages, lengths, cu, sink, output, scratch, handle, relative_bias, relative_bias_factors, relative_bias_projection):
    pass


@torch.library.custom_op("flare::paged_nvfp4_write_kv", mutates_args=("k_cache", "k_scales", "v_cache", "v_scales"))
def _write_nvfp4(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor,
                 k_scales: torch.Tensor, v_cache: torch.Tensor, v_scales: torch.Tensor,
                 slots: torch.Tensor, handle: int) -> None:
    paged_nvfp4.write_kv(plan_from_handle(handle), key=key, value=value, k_cache=k_cache,
                         k_scales=k_scales, v_cache=v_cache, v_scales=v_scales, slot_mapping=slots)


@_write_nvfp4.register_fake
def _fake_write_nvfp4(key, value, k_cache, k_scales, v_cache, v_scales, slots, handle):
    pass


class PreparedPagedAttention:
    def __init__(self, q, key, value, pages, lengths, cu, output, *, sink=None, mode,
                 window=-1, width=1, causal=True, k_descale=None, v_descale=None,
                 selected_blocks=None, session, workspace, name, schedules=None, tuning_page_count=None, output_gate=None,
                 caller_owned_placeholders=False, relative_bias=None, relative_bias_factors=None,
                 relative_bias_projection=None, tail_key=None, tail_value=None,
                 kv_format="fp8", k_scales=None, v_scales=None):
        if (tail_key is None) != (tail_value is None):
            raise ValueError("private tail key and value must be supplied together")
        tail_pages = 0
        if tail_key is not None:
            if mode not in ("decode", "verify") or tail_key.shape[0] % pages.shape[0]:
                raise ValueError("private tail pages require decode/verify with equal pages per request")
            tail_pages = tail_key.shape[0] // pages.shape[0]
            if tail_pages < 1:
                raise ValueError("private tail pages must have positive capacity")
        if (relative_bias_factors is None) != (relative_bias_projection is None):
            raise ValueError("relative bias factors and projection must be supplied together")
        if relative_bias is not None and relative_bias_factors is not None:
            raise ValueError("dense and factored relative bias cannot be combined")
        if relative_bias_factors is not None and mode in ("decode", "verify"):
            raise ValueError("decode and verify require dense relative bias")
        if tuning_page_count is None:
            tuning_page_count = min(key.shape[0], value.shape[0])
        if type(tuning_page_count) is not int or not 0 < tuning_page_count <= min(key.shape[0], value.shape[0]):
            raise ValueError("attention tuning requires a positive backed page count within both cache views")
        self.tuning_page_count = tuning_page_count
        self.schedules = schedules
        from .paged_tma import PreparedTma, supports

        self.tma = None
        self.output_gate = None
        self.nvfp4 = kv_format != "fp8"
        if self.nvfp4:
            if kv_format != "nvfp4_4o6k" or k_scales is None or v_scales is None:
                raise ValueError("NVFP4 paged attention requires kv_format='nvfp4_4o6k' with k_scales and v_scales")
            if selected_blocks is not None or tail_key is not None or not causal or output_gate is not None:
                raise ValueError("NVFP4 paged attention requires causal GQA without selected blocks, tail pages, or output gate")
            nvfp4_mode = "prefill" if mode == "extend" else mode
            self.schedule_storage, self.schedule_slot = None, -1
            self.selected_blocks = None
            self.decode = nvfp4_mode in ("decode", "verify")
            self.compact_decode = None
            self.window, self.workspace = window, workspace
            self.arguments = dict(q=q, k_cache=key, k_scales=k_scales, v_cache=value, v_scales=v_scales,
                page_table=pages, cache_seqlens=lengths, cu_seqlens_q=cu, output=output, attention_sink_bias=sink,
                relative_attention_bias=relative_bias, relative_bias_factors=relative_bias_factors,
                relative_bias_projection=relative_bias_projection)
            self.plan = paged_nvfp4.plan(paged_nvfp4.Caps(
                device=q.device, mode=nvfp4_mode, kv_format=kv_format,
                num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2], head_dim_vo=q.shape[2],
                page_size=key.shape[1], max_batch=pages.shape[0],
                max_q_per_req=q.shape[0] if nvfp4_mode == "prefill" else width,
                window_left=window, causal=True, has_sinks=sink is not None,
                sink_layout="row" if sink is not None and sink.ndim == 2 else "head",
                has_relative_bias=relative_bias is not None,
                has_factored_relative_bias=relative_bias_factors is not None,
                caller_owned_placeholders=caller_owned_placeholders))
        elif (mode in ("prefill", "extend") and causal and window == -1 and sink is None
                and selected_blocks is None and k_descale is None and v_descale is None
                and relative_bias is None and relative_bias_factors is None
                and supports(q, key, value, pages, cu, output)):
            self.tma = PreparedTma(q, key, value, pages, lengths, cu, output,
                session=session, workspace=workspace, name=name, output_gate=output_gate)
            self.output_gate = output_gate
            self.plan = self.tma.plan
            return
        if not self.nvfp4:
            self.schedule_storage, self.schedule_slot = None, -1
            if key.dtype == torch.float8_e4m3fn:
                if k_descale is None:
                    k_descale = persistent_empty((1,), torch.float32, key.device).fill_(1)
                if v_descale is None:
                    v_descale = persistent_empty((1,), torch.float32, value.device).fill_(1)
            self.selected_blocks = selected_blocks
            if selected_blocks is not None:
                if (selected_blocks.shape != (key.shape[2], q.shape[0], 16)
                        or selected_blocks.dtype != torch.int32 or not selected_blocks.is_contiguous()):
                    raise ValueError("selected blocks must be contiguous int32 [kv_heads, query_capacity, 16]")
                if not causal or window != -1 or sink is not None or relative_bias is not None or relative_bias_factors is not None:
                    raise ValueError("selected-block attention requires causal full-context GQA without sinks")
                if mode in ("decode", "verify"):
                    mode = "decode" if width == 1 else "verify"
                else:
                    mode = "extend"
            self.decode = selected_blocks is None and mode in ("decode", "verify") and paged_decode.supports(
                num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
                head_dim_vo=value.shape[3], page_size=key.shape[1], kv_dtype=key.dtype,
                max_q_per_req=width, window_left=window, causal=causal, has_sinks=sink is not None,
                has_relative_bias=relative_bias is not None, sink_layout="row" if sink is not None and sink.ndim == 2 else "head",
                tail_pages_per_request=tail_pages)
            if tail_key is not None and not self.decode:
                raise ValueError("private tail pages require a supported paged-decode geometry")
            if not causal and not self.decode:
                raise ValueError("noncausal paged attention requires a supported paged-decode geometry")
            self.compact_decode = None
            if not self.decode and mode == "decode":
                # The GQA decode schedule has one physical query row per request.
                # Recurrent batches pack active queries, including holes between live requests.
                self.compact_decode = (q, output, cu, selected_blocks)
                q, output = torch.zeros_like(q), torch.zeros_like(output)
                cu = torch.arange(pages.shape[0] + 1, dtype=torch.int32, device=q.device)
                if selected_blocks is not None:
                    selected_blocks = torch.full_like(selected_blocks, -1)
                    self.selected_blocks = selected_blocks
            self.window, self.workspace = window, workspace
            self.arguments = dict(q=q, k_cache=key, v_cache=value, page_table=pages,
                cache_seqlens=lengths, cu_seqlens_q=cu, output=output, attention_sink_bias=sink,
                k_descale=k_descale, v_descale=v_descale, relative_attention_bias=relative_bias)
            if self.decode:
                self.arguments.update(tail_k_cache=tail_key, tail_v_cache=tail_value)
            else:
                self.arguments.update(relative_bias_factors=relative_bias_factors, relative_bias_projection=relative_bias_projection)
            if selected_blocks is not None:
                self.arguments["q2k_indices"] = selected_blocks
            if self.decode:
                self.plan = paged_decode.plan(paged_decode.Caps(device=q.device, num_q_heads=q.shape[1],
                    num_kv_heads=key.shape[2], head_dim_qk=q.shape[2], head_dim_vo=value.shape[3],
                    page_size=key.shape[1], max_batch=pages.shape[0], max_q_per_req=width,
                    window_left=window, has_sinks=sink is not None, causal=causal, kv_dtype=key.dtype,
                    caller_owned_placeholders=caller_owned_placeholders, has_relative_bias=relative_bias is not None,
                    sink_layout="row" if sink is not None and sink.ndim == 2 else "head",
                    tail_pages_per_request=tail_pages))
            else:
                geometry = dict(device=q.device, q_dtype=q.dtype, kv_dtype=key.dtype,
                    num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
                    head_dim_vo=value.shape[3], page_size=key.shape[1], batch=pages.shape[0],
                    max_cache_page_count=pages.shape[1], window_left=window)
                if mode == "decode":
                    capacity = paged.decode_graph_capacity(**geometry)
                elif mode == "verify":
                    capacity = paged.verify_graph_capacity(**geometry, query_len=width)
                else:
                    capacity = paged.extend_graph_capacity(**geometry, total_q_capacity=q.shape[0])
                # Verify replay reserves a schedule for W queries per request;
                # live cumulative lengths still index the P-row query/output tensors.
                schedule_rows = pages.shape[0] * width if mode == "verify" else q.shape[0]
                caps = paged.Caps(device=q.device, mode=mode if mode in ("decode", "verify") else "extend",
                    dtype=q.dtype, kv_dtype=key.dtype,
                    num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
                    head_dim_vo=value.shape[3], page_size=key.shape[1], max_total_q=schedule_rows,
                    max_batch=pages.shape[0], max_page_table_width=pages.shape[1], num_cache_pages=key.shape[0],
                    max_work_items=max(capacity.max_work_items, schedule_rows * 32) if selected_blocks is not None and mode in ("decode", "verify") else capacity.max_work_items,
                    max_partial_rows=max(getattr(capacity, "max_partial_rows", 0), schedule_rows * 32) if selected_blocks is not None and mode in ("decode", "verify") else getattr(capacity, "max_partial_rows", 0),
                    use_cuda_graph=True,
                    copy_runtime_metadata=not all(t.dtype == torch.int32 and t.is_contiguous()
                                                  for t in (pages, lengths, cu)),
                    msa_block_sparse=selected_blocks is not None,
                    msa_union_tile=selected_blocks is not None and mode == "extend")
                invocation = dict(paged.invocation_from_tensors(caps, **self.arguments), window_left=window)
                if mode == "verify":
                    invocation["query_len"] = width
                if selected_blocks is not None and mode in ("decode", "verify"):
                    invocation["force_split_kv"] = True
                self.plan = paged.plan(caps, invocation=invocation)
                if schedules is not None and mode in ("verify", "extend"):
                    schedules.register(self, caps)

        def prepared_call(state, arguments):
            from b12x.attention.paged_decode._preparation import scratch_specs

            if self.nvfp4:
                specs = state.scratch_specs()
            elif self.decode:
                specs = scratch_specs(state.query, state.config, q.device)
            else:
                specs = state.scratch_plan.scratch_specs()
            scratch = _scratch(specs, workspace)
            kwargs = {"plan": self.plan} if (self.decode or self.nvfp4) else {"window_left": window}
            binding = state.bind(scratch=scratch, **arguments, **kwargs)
            return PreparedCall(run=lambda: state.run(binding),
                                produce=lambda: state.bind(scratch=scratch, **arguments, **kwargs),
                                owners=prime_owners(workspace, scratch, binding))

        def benchmark(state):
            def inputs():
                arguments = dict(self.arguments)
                for field in ("q", "page_table", "cache_seqlens", "cu_seqlens_q", "output"):
                    value = arguments[field]
                    arguments[field] = torch.empty_strided(value.shape, value.stride(),
                                                         dtype=value.dtype, device=value.device)
                arguments["q"].normal_(std=.2)
                extent = min(max(4096, q.shape[0]), pages.shape[1] * key.shape[1])
                pages_per_row = triton.cdiv(extent, key.shape[1])
                page_ids = (torch.arange(pages.shape[0], device=pages.device, dtype=torch.int64)[:, None]
                            * pages_per_row + torch.arange(pages.shape[1], device=pages.device, dtype=torch.int64))
                arguments["page_table"].copy_((page_ids % self.tuning_page_count).to(pages.dtype))
                arguments["cache_seqlens"].fill_(extent)
                arguments["cu_seqlens_q"].copy_((torch.arange(cu.numel(), device=cu.device, dtype=torch.int64)
                                                   * q.shape[0] // pages.shape[0]).to(cu.dtype))
                if selected_blocks is not None:
                    blocks = torch.arange(16, dtype=torch.int32, device=q.device)
                    blocks = torch.where(blocks < triton.cdiv(extent, 128), blocks, -1)
                    arguments["q2k_indices"] = blocks.expand_as(selected_blocks).contiguous()
                return arguments, arguments["q"].clone()

            operands, seed = benchmark_resource(session, ("paged_attention", name), inputs)
            call = prepared_call(state, operands)
            def produce():
                operands["q"].copy_(seed)
                # Candidate bindings share scratch, including their scheduling metadata.
                call.produce()

            return PreparedCall(run=call.run, produce=produce,
                                owners=(*call.owners, operands, seed), output=operands["output"])

        def prime(state):
            call = benchmark(state)
            return PreparedCall(run=call.run, produce=call.produce,
                                owners=prime_owners(workspace))

        declare(session, (self.plan.request(name=name,
            prepare_call=prime if selected_blocks is not None else lambda state: prepared_call(state, self.arguments),
            benchmark_call=benchmark),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = _scratch(self.plan.scratch_specs(), self.workspace)
        if not self.decode and not self.nvfp4:
            paged.bind(self.plan, scratch=self.scratch, **self.arguments, window_left=self.window)

    def functional(self, *, session, k_cache=_BOUND_OPERAND, v_cache=_BOUND_OPERAND,
                   k_scales=_BOUND_OPERAND, v_scales=_BOUND_OPERAND,
                   cache_seqlens=_BOUND_OPERAND, page_table=_BOUND_OPERAND, cu_seqlens_q=_BOUND_OPERAND,
                   selected_blocks=_BOUND_OPERAND, schedule_storage=_BOUND_OPERAND, sink=_BOUND_OPERAND,
                   relative_bias=_BOUND_OPERAND, relative_bias_factors=_BOUND_OPERAND, relative_bias_projection=_BOUND_OPERAND,
                   tail_key=_BOUND_OPERAND, tail_value=_BOUND_OPERAND):
        """Return attention with local output and selected-plan scratch.

        Omitted operands retain their prepared bindings. Bind an operand to None
        to require it at invocation without retaining its preparation tensor.
        """
        operation = _FunctionalPagedAttention(self)
        for name, value in (("k_cache", k_cache), ("v_cache", v_cache), ("k_scales", k_scales), ("v_scales", v_scales),
                            ("cache_seqlens", cache_seqlens),
                            ("page_table", page_table), ("cu_seqlens_q", cu_seqlens_q),
                            ("attention_sink_bias", sink), ("relative_attention_bias", relative_bias),
                            ("relative_bias_factors", relative_bias_factors), ("relative_bias_projection", relative_bias_projection),
                            ("tail_k_cache", tail_key), ("tail_v_cache", tail_value)):
            if value is not _BOUND_OPERAND:
                operation.arguments[name] = value
        if cu_seqlens_q is None and operation.compact_decode is not None:
            operation.compact_decode = (None, operation.compact_decode[1])
        if selected_blocks is not _BOUND_OPERAND:
            operation.selected_blocks = selected_blocks
            if operation.compact_decode is not None:
                operation.compact_decode = (operation.compact_decode[0], selected_blocks)
        if schedule_storage is not _BOUND_OPERAND:
            operation.schedule_storage = schedule_storage
        if self.schedules is not None:
            if (schedule_storage is _BOUND_OPERAND
                    and any(component is self for component in self.schedules.components)
                    and operation.arguments["cache_seqlens"] is not self.schedules.lengths):
                raise ValueError("shared attention schedules require their bound cache lengths")
            self.schedules.components[:] = [operation if component is self else component
                                           for component in self.schedules.components]
        register_finalizer(session, operation.bind)
        if not operation.tma and not operation.decode and not operation.nvfp4:
            self.workspace.register_preparation_release(operation.release_metadata)
        return operation

    def __call__(self):
        if self.tma is not None:
            q, key, value, pages, lengths, cu, output = self.tma.arguments
            _attend(q, key, value, pages, lengths, cu, None, output, [self.tma.scratch],
                None, None, None, self.plan.handle, False, -1, None, -1, self.output_gate, [], [], None, None, None, None, None, None)
            return output
        a = self.arguments
        if self.nvfp4:
            _attend_nvfp4(a["q"], a["k_cache"], a["k_scales"], a["v_cache"], a["v_scales"],
                          a["page_table"], a["cache_seqlens"], a["cu_seqlens_q"], a["attention_sink_bias"],
                          a["output"], self.scratch, self.plan.handle, a["relative_attention_bias"],
                          a.get("relative_bias_factors"), a.get("relative_bias_projection"))
            return a["output"]
        if self.compact_decode is not None:
            source, destination, cu, selected = self.compact_decode
            indices = cu[:-1].long().clamp(0, source.shape[0] - 1)
            active = cu[1:] > cu[:-1]
            a["q"].copy_(torch.where(active[:, None, None], source[indices], 0))
            if selected is not None:
                self.selected_blocks.copy_(torch.where(active[None, :, None], selected[:, indices], -1))
        _attend(a["q"], a["k_cache"], a["v_cache"], a["page_table"], a["cache_seqlens"],
                a["cu_seqlens_q"], a["attention_sink_bias"], a["output"], self.scratch,
                a["k_descale"], a["v_descale"], self.selected_blocks,
                self.plan.handle, self.decode, self.window, self.schedule_storage, self.schedule_slot, None, [], [], None, a["relative_attention_bias"],
                a.get("relative_bias_factors"), a.get("relative_bias_projection"),
                a.get("tail_k_cache"), a.get("tail_v_cache"))
        if self.compact_decode is not None:
            positions = torch.arange(destination.shape[0], device=destination.device)
            requests = (positions[:, None] >= cu[None, 1:]).sum(1).clamp_max(cu.numel() - 2)
            destination.copy_(torch.where((positions < cu[-1])[:, None, None], a["output"][requests], 0))
            return destination
        return a["output"]


class _FunctionalPagedAttention:
    def __init__(self, prepared):
        self.plan = prepared.plan
        self.tma = prepared.tma is not None
        self.nvfp4 = prepared.nvfp4
        if self.tma:
            q, key, value, pages, lengths, cu, output = prepared.tma.arguments
            kernel_output = output
            self.arguments = dict(k_cache=key, v_cache=value, page_table=pages,
                cache_seqlens=lengths, cu_seqlens_q=cu, attention_sink_bias=None,
                k_descale=None, v_descale=None)
            self.decode, self.window = False, -1
            self.selected_blocks, self.compact_decode = None, None
            self.schedule_storage, self.schedule_slot = None, -1
        else:
            q, output = prepared.arguments["q"], prepared.arguments["output"]
            kernel_output = output
            self.arguments = {name: value for name, value in prepared.arguments.items()
                              if name not in ("q", "output", "q2k_indices")}
            self.decode, self.window = prepared.decode, prepared.window
            self.selected_blocks = prepared.selected_blocks
            self.compact_decode = None
            if prepared.compact_decode is not None:
                q, output, cu, selected = prepared.compact_decode
                self.compact_decode = cu, selected
                self.selected_blocks = None
            self.schedule_storage, self.schedule_slot = prepared.schedule_storage, prepared.schedule_slot
        self.optional_layouts = {name: (tuple(value.shape), tuple(value.stride()), value.dtype)
                             for name in ("attention_sink_bias", "relative_attention_bias", "relative_bias_factors", "relative_bias_projection", "tail_k_cache", "tail_v_cache")
                             if (value := self.arguments.get(name)) is not None}
        selected = None if self.tma else prepared.selected_blocks
        self.selected_shape = None if selected is None else tuple(selected.shape)
        self.query_shape, self.query_dtype = tuple(q.shape), q.dtype
        self.output_shape, self.output_dtype = tuple(output.shape), output.dtype
        self.kernel_output_shape = tuple(kernel_output.shape)
        self.gated = prepared.output_gate is not None
        self.scratch_layout = None
        self.metadata_seeds, self.chunk_pages_lut = (), None
        self.live_metadata = False

    def bind(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        if not self.tma and not self.decode and not self.nvfp4:
            metadata = paged.replay_metadata(self.plan)
            self.metadata_seeds = metadata.tensors
            self.chunk_pages_lut = metadata.chunk_pages_lut
            self.live_metadata = metadata.live

    def release_metadata(self):
        self.metadata_seeds = tuple(persistent_copy(value) for value in self.metadata_seeds)
        if self.chunk_pages_lut is not None:
            self.chunk_pages_lut = persistent_copy(self.chunk_pages_lut)
        paged.release_replay_metadata(self.plan)

    def __call__(self, q, *, output_gate=None, k_cache=None, v_cache=None, k_scales=None, v_scales=None,
                 cache_seqlens=None, page_table=None, cu_seqlens_q=None, selected_blocks=None, schedule_storage=None,
                 sink=None, relative_bias=None, relative_bias_factors=None, relative_bias_projection=None,
                 tail_key=None, tail_value=None):
        if self.scratch_layout is None:
            raise RuntimeError("functional paged attention requires finalized preparation")
        if tuple(q.shape) != self.query_shape or q.dtype != self.query_dtype:
            raise ValueError("functional paged attention query differs from prepared geometry")
        if (output_gate is not None) != self.gated:
            raise ValueError("functional paged attention gate differs from prepared geometry")
        a = self.arguments
        if (self.schedule_slot >= 0 and schedule_storage is None and cache_seqlens is not None
                and cache_seqlens is not a["cache_seqlens"]):
            raise ValueError("shared attention schedules require their bound cache lengths")
        optional_operands = {}
        for name, value in (("attention_sink_bias", sink), ("relative_attention_bias", relative_bias),
                            ("relative_bias_factors", relative_bias_factors), ("relative_bias_projection", relative_bias_projection),
                            ("tail_k_cache", tail_key), ("tail_v_cache", tail_value)):
            operand = a.get(name) if value is None else value
            layout = self.optional_layouts.get(name)
            if ((operand is None) != (layout is None) or operand is not None
                    and (tuple(operand.shape), tuple(operand.stride()), operand.dtype) != layout):
                raise ValueError(f"functional paged attention requires {name} with its prepared geometry")
            optional_operands[name] = operand
        key = a["k_cache"] if k_cache is None else k_cache
        value = a["v_cache"] if v_cache is None else v_cache
        lengths = a["cache_seqlens"] if cache_seqlens is None else cache_seqlens
        pages = a["page_table"] if page_table is None else page_table
        cu = a["cu_seqlens_q"] if cu_seqlens_q is None else cu_seqlens_q
        if key is None or value is None or lengths is None or pages is None or cu is None:
            raise ValueError("functional paged attention requires cache tensors and metadata at invocation")
        output = (torch.empty(self.kernel_output_shape, dtype=self.output_dtype, device=q.device) if self.tma
                  else torch.zeros(self.kernel_output_shape, dtype=self.output_dtype, device=q.device))
        scratch = [torch.empty(shape, dtype=dtype, device=q.device) for shape, dtype in self.scratch_layout]
        selected = self.selected_blocks if selected_blocks is None else selected_blocks
        if self.compact_decode is not None:
            selected = self.compact_decode[1] if selected_blocks is None else selected_blocks
        if ((selected is None) != (self.selected_shape is None)
                or selected is not None and (tuple(selected.shape) != self.selected_shape
                    or selected.dtype != torch.int32 or not selected.is_contiguous())):
            raise ValueError("functional selected blocks differ from prepared geometry")
        schedules = self.schedule_storage if schedule_storage is None else schedule_storage
        if self.schedule_slot >= 0 and schedules is None:
            raise ValueError("functional paged attention requires shared scheduling storage")
        if self.compact_decode is not None:
            bound_cu, bound_selected = self.compact_decode
            selected = bound_selected if selected_blocks is None else selected_blocks
            cu = cu if bound_cu is None else bound_cu
            indices = cu[:-1].long().clamp(0, q.shape[0] - 1)
            active = cu[1:] > cu[:-1]
            q = torch.where(active[:, None, None], q[indices], 0)
            if selected is not None:
                selected = torch.where(active[None, :, None], selected[:, indices], -1)
        if self.nvfp4:
            ks = a.get("k_scales") if k_scales is None else k_scales
            vs = a.get("v_scales") if v_scales is None else v_scales
            if ks is None or vs is None:
                raise ValueError("functional NVFP4 paged attention requires k_scales and v_scales at invocation")
            _attend_nvfp4(q, key, ks, value, vs, pages, lengths, cu, optional_operands["attention_sink_bias"],
                          output, scratch, self.plan.handle, optional_operands["relative_attention_bias"],
                          optional_operands["relative_bias_factors"], optional_operands["relative_bias_projection"])
            return output
        live_metadata = [value.clone() for value in self.metadata_seeds] if self.live_metadata else []
        _attend(q, key, value, pages, lengths,
            cu, optional_operands["attention_sink_bias"], output, scratch,
            a["k_descale"], a["v_descale"], selected, self.plan.handle, self.decode,
            self.window, schedules,
            self.schedule_slot, output_gate, list(self.metadata_seeds), live_metadata, self.chunk_pages_lut, optional_operands["relative_attention_bias"],
            optional_operands["relative_bias_factors"], optional_operands["relative_bias_projection"],
            optional_operands["tail_k_cache"], optional_operands["tail_v_cache"])
        if self.compact_decode is not None:
            positions = torch.arange(self.output_shape[0], device=output.device)
            requests = (positions[:, None] >= cu[None, 1:]).sum(1).clamp_max(cu.numel() - 2)
            return torch.where((positions < cu[-1])[:, None, None], output[requests], 0)
        return output


class PreparedPagedKVWriter:
    def __init__(self, key, value, key_cache, value_cache, slots, *, session, name, k_scale=None, v_scale=None,
                 caller_owned_placeholders=False, kv_format="fp8", k_scales=None, v_scales=None,
                 attention_plan=None):
        self.nvfp4 = kv_format != "fp8"
        if self.nvfp4:
            if kv_format != "nvfp4_4o6k" or k_scales is None or v_scales is None:
                raise ValueError("NVFP4 KV writer requires kv_format='nvfp4_4o6k' with k_scales and v_scales")
            self.arguments = dict(key=key, value=value, k_cache=key_cache, k_scales=k_scales,
                                  v_cache=value_cache, v_scales=v_scales, slot_mapping=slots)
            if attention_plan is not None:
                self.plan = attention_plan
            else:
                self.plan = paged_nvfp4.plan(paged_nvfp4.Caps(
                    device=key.device, mode="decode", kv_format=kv_format,
                    num_q_heads=key.shape[1], num_kv_heads=key.shape[1],
                    head_dim_qk=key.shape[2], head_dim_vo=value.shape[2],
                    page_size=key_cache.shape[1], max_batch=1, max_q_per_req=1,
                    caller_owned_placeholders=caller_owned_placeholders))
                declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
                    run=lambda: state.write_kv(**self.arguments))),))
            return
        if key_cache.dtype == torch.float8_e4m3fn:
            if k_scale is None:
                k_scale = persistent_empty((1,), torch.float32, key.device).fill_(1)
            if v_scale is None:
                v_scale = persistent_empty((1,), torch.float32, value.device).fill_(1)
        self.arguments = dict(key=key, value=value, k_cache=key_cache, v_cache=value_cache, slot_mapping=slots,
                              k_scale=k_scale, v_scale=v_scale)
        if caller_owned_placeholders:
            self.arguments["dummy_scale"] = torch.empty((1,), dtype=torch.float32, device=key.device)
        self.plan = paged_kv_write.plan(paged_kv_write.Caps(device=key.device, num_kv_heads=key.shape[1],
            head_dim_k=key.shape[2], head_dim_v=value.shape[2], page_size=key_cache.shape[1], kv_dtype=key_cache.dtype,
            caller_owned_placeholders=caller_owned_placeholders))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(**self.arguments))),))

    def __call__(self):
        if self.nvfp4:
            a = self.arguments
            _write_nvfp4(a["key"], a["value"], a["k_cache"], a["k_scales"],
                         a["v_cache"], a["v_scales"], a["slot_mapping"], self.plan.handle)
            return
        paged_kv_write.run(self.plan, **self.arguments)

    def functional(self, *, k_cache=_BOUND_OPERAND, v_cache=_BOUND_OPERAND,
                   k_scales=_BOUND_OPERAND, v_scales=_BOUND_OPERAND):
        """Accept projected rows, slots, and explicitly unbound caches at invocation."""
        operation = _FunctionalPagedKVWriter(self)
        for name, value in (("k_cache", k_cache), ("v_cache", v_cache), ("k_scales", k_scales), ("v_scales", v_scales)):
            if value is not _BOUND_OPERAND:
                operation.arguments[name] = value
        return operation


class _FunctionalPagedKVWriter:
    def __init__(self, prepared):
        self.plan = prepared.plan
        self.nvfp4 = prepared.nvfp4
        self.arguments = {name: value for name, value in prepared.arguments.items()
                          if name not in ("key", "value", "slot_mapping", "dummy_scale")}
        self.caller_owned_placeholders = "dummy_scale" in prepared.arguments

    def __call__(self, key, value, slots, *, k_cache=None, v_cache=None, k_scales=None, v_scales=None):
        arguments = dict(self.arguments)
        if k_cache is not None:
            arguments["k_cache"] = k_cache
        if v_cache is not None:
            arguments["v_cache"] = v_cache
        if self.nvfp4:
            if k_scales is not None:
                arguments["k_scales"] = k_scales
            if v_scales is not None:
                arguments["v_scales"] = v_scales
            if (arguments.get("k_cache") is None or arguments.get("v_cache") is None
                    or arguments.get("k_scales") is None or arguments.get("v_scales") is None):
                raise ValueError("functional NVFP4 cache writing requires cache and scale tensors at invocation")
            _write_nvfp4(key, value, arguments["k_cache"], arguments["k_scales"],
                         arguments["v_cache"], arguments["v_scales"], slots, self.plan.handle)
            return
        if arguments["k_cache"] is None or arguments["v_cache"] is None:
            raise ValueError("functional cache writing requires cache tensors at invocation")
        dummy_scale = (torch.empty((1,), dtype=torch.float32, device=key.device)
                       if self.caller_owned_placeholders else None)
        paged_kv_write.run(self.plan, key=key, value=value,
            slot_mapping=slots, dummy_scale=dummy_scale, **arguments)


class PagedAttentionMetadata:
    """Cache-write slots and sequence extents shared by layers with one page geometry."""

    def __init__(self, metadata, *, page_tokens, buffers):
        self.metadata, self.page_tokens = metadata, page_tokens
        self.slots = buffers.zeros(metadata.positions.shape, torch.int64).fill_(-1)
        self.lengths = buffers.zeros(metadata.lengths.shape, torch.int32)

    def __call__(self):
        m = self.metadata
        columns = (m.positions.clamp_min(0) // self.page_tokens).clamp(max=m.token_page_table.shape[1] - 1)
        rows = torch.arange(m.positions.numel(), device=m.positions.device)
        pages = m.token_page_table[rows, columns]
        self.slots.copy_(torch.where((m.positions >= 0) & (pages >= 0),
            pages.long() * self.page_tokens + m.positions.remainder(self.page_tokens), -1))
        self.lengths.copy_(torch.where(m.lengths > 0, (m.starts + m.lengths).int(), 0))
