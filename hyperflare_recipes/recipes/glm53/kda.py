# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared GLM KDA projections, convolution, recurrence and gated output norm."""

from dataclasses import dataclass, fields

import torch
from b12x.sequence import gdn_decode, kda_prefill, short_conv
from b12x.sequence.gdn_decode._kernels import _gdn_decode_op as kda_decode_out
from b12x.sequence.kda_prefill._ops import _run as kda_prefill_out

from hyperflare_recipes.components import sm103
from hyperflare_recipes.components.kda_speculation import KDARecords, _record, register_recording
from hyperflare_recipes.components.gated_norm import PreparedGatedRMSNorm
from hyperflare_recipes.components.linear import prepare_projection
from hyperflare_recipes.components.projection_overlap import ConcurrentProjectionChain
from b12x.preparation import PreparedCall
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare

from hyperflare_recipes.components.recurrent import (
    DeltaMetadata,
    DeltaState,
    prepare_sequence_binding,
    run_short_conv,
)

# Row-owned contiguous single-segment SP-KDA exchange shapes (BF16 contiguous, 16-byte row aligned).
# Sinit packs this rank's 16-head FP32 initial recurrent state as BF16 [64, 8192] = 1.0 MiB/rank
# (4.0 MiB gathered, overlapped behind att_hc_b + input_proj_b).
# Halo packs 16-head rank conv state [3, 6144] + last-3 pre-conv QKV rows of Half B [3, 24576]
# = 184 KiB/rank (737 KiB gathered, overlapped behind f_b_proj).
# Scan packs per-row FP8 A_r [64, 8192] + FP32 A_sc [64, 256] + per-row INT16 B_r [64, 16384]
# + FP32 B_sc [64, 256] + post-conv state [64, 1152] = 3.20 MiB/rank (12.81 MiB gathered).
ROW_SHARD_SINIT_SHAPE = (64, 8192)
ROW_SHARD_HALO_SHAPE = (3, 30720)
ROW_SHARD_SCAN_SHAPE = (64, 26240)


@dataclass(frozen=True)
class KDAScratch:
    projected_input: torch.Tensor
    projected_qkv: torch.Tensor
    convolved_qkv: torch.Tensor
    forget_input: torch.Tensor
    gate_input: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    z: torch.Tensor
    recurrence: torch.Tensor
    gated: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, heads, device):
        if type(tokens) is not int or tokens <= 0 or heads not in (16, 32, 64):
            raise ValueError("GLM KDA scratch requires positive capacity and TP1/TP2/TP4 head counts")
        width = heads * 128
        def buffer(channels):
            return torch.zeros((tokens, channels), dtype=torch.bfloat16, device=device)
        projected = buffer(3 * width + heads + 128)
        qkv, beta, forget = projected.split((3 * width, heads, 128), dim=-1)
        if tokens > 1:
            forget = buffer(128)
        return cls(projected, qkv, buffer(3 * width), forget, buffer(128), buffer(width), beta,
                   buffer(width), buffer(width), buffer(width), buffer(4096))

    def rows(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("KDA rows exceed scratch capacity")
        views = {field.name: getattr(self, field.name)[:tokens] for field in fields(self)}
        if tokens == 1:
            views["forget_input"] = views["projected_input"][:, -128:]
        return KDAScratch(**views)


class PreparedGlmKDA:
    """One rank's KDA branch before the row-parallel output reduction.

    Decode and prefill use b12x recurrence kernels. Prefill output normalization
    fuses the canonical sigmoid-gated FP32 RMSNorm and zeroes inactive rows. Device
    metadata selects live rows and persistent state slots; zero live counts are
    required while preparing and tuning kernels.
    """

    def __init__(self, module, weights, source, scratch: KDAScratch, state: DeltaState,
                 metadata: DeltaMetadata, *, mode, session, name, workspace, checkpoint_export=False,
                 records=None, overlap_gate=True, row_shard=False, rank_index=None, segment_eye=None,
                 half_tokens=None, segment_pool=None, local_conv=None, seg_qkv=None, seg_beta=None,
                 output_tokens=None):
        if (source.ndim != 2 or source.shape[1] != 4096 or source.dtype != torch.bfloat16
                or source.device.type != "cuda"):
            raise ValueError("GLM KDA requires BF16 [token_capacity, 4096] CUDA input")
        if mode not in ("decode", "prefill"):
            raise ValueError("KDA mode must be decode or prefill")
        if (module.num_heads != 64 or module.head_dim != 128 or module.conv_kernel_size != 4
                or module.o_norm.activation != "sigmoid" or module.forget_gate.safe_gate_lower_bound != -5.0):
            raise ValueError("canonical KDA geometry differs from GLM 5.3 Flash")
        self.row_shard = bool(row_shard)
        self.tokens, self.device = source.shape[0], source.device
        self.output_tokens = self.tokens if output_tokens is None else int(output_tokens)
        self.sequences = metadata.query_start_loc.numel() - 1
        self.columns = metadata.state_indices.shape[1]
        self.drafts = state.convolution.shape[-1] - 3
        self.local_heads = state.recurrent.shape[1]
        self.local_width = self.local_heads * 128
        if (self.local_heads not in (16, 32, 64) or state.recurrent.shape[2:] != (128, 128)
                or state.convolution.shape[:2] != (state.recurrent.shape[0], 3 * self.local_width)):
            raise ValueError("GLM KDA state must match one TP1/TP2/TP4 head partition")
        if self.row_shard:
            if mode != "prefill" or self.local_heads != 16 or records is not None:
                raise ValueError("row-owned KDA requires TP4 prefill without verification records")
            if rank_index is None or segment_eye is None or half_tokens != self.tokens * 4:
                raise ValueError("row-owned KDA requires rank_index, segment_eye, and TP4 half_tokens")
            if segment_pool is None or local_conv is None or seg_qkv is None or seg_beta is None:
                raise ValueError("row-owned KDA requires declaration segment buffers")
            self.heads = 64
            self.width = 8192
            self.seg_tokens = self.tokens * 2
            self.half_tokens = half_tokens
            self.rank_index = rank_index
            self.segment_eye = segment_eye
            overlap_gate = False
        else:
            self.heads = self.local_heads
            self.width = self.local_width
            self.seg_tokens = self.tokens
            self.half_tokens = self.rank_index = self.segment_eye = None
        if metadata.num_tokens.item() or metadata.num_seqs.item():
            raise ValueError("KDA preparation requires zero live counts")
        if mode == "decode" and (self.tokens > self.sequences * self.columns or self.columns > self.drafts + 1):
            raise ValueError("KDA decode capacity exceeds metadata or convolution history")
        if records is not None and mode != "decode":
            raise ValueError("KDA verification records belong to decode rounds")
        self.module, self.weights, self.scratch = module, weights, scratch
        self.arena = workspace
        self.recording = records is not None
        self.records = self.verification = None
        self.state, self.metadata, self.mode = state, metadata, mode
        module.o_norm.weight = torch.nn.Parameter(weights["o_norm.weight"], requires_grad=False)
        proj_scratch = scratch.rows(self.tokens) if self.row_shard else scratch
        out_scratch = scratch.rows(self.output_tokens) if self.row_shard else scratch
        self.projections = {}
        for key, input, output in (
            ("input_proj", source, proj_scratch.projected_input),
            ("f_b_proj", proj_scratch.forget_input, proj_scratch.raw_g),
            ("g_a_proj", source, proj_scratch.gate_input),
            ("g_b_proj", proj_scratch.gate_input, proj_scratch.z),
            ("o_proj", out_scratch.gated, out_scratch.output),
        ):
            self.projections[key] = prepare_projection(input, weights[key], output,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto",
                # Convolution consumes the projected rows after this staging has been copied out.
                staging=proj_scratch.convolved_qkv if key == "input_proj" else None)
        # The output-gate chain runs beside the input projections, or after them on the main stream.
        self.gate_overlap = None
        if workspace.overlap and overlap_gate:
            self.gate_overlap = ConcurrentProjectionChain(
                (self.projections["g_a_proj"], self.projections["g_b_proj"]),
                parallel_plans=(*self.projections["input_proj"].plans, self.projections["f_b_proj"].plan)
                    if hasattr(self.projections["input_proj"], "plans") else
                    (self.projections["input_proj"].plan, self.projections["f_b_proj"].plan),
                session=session, workspace=workspace)
        conv_state_buf = local_conv if self.row_shard else state.convolution
        conv_drafts = 0 if self.row_shard else self.drafts
        conv_x = seg_qkv[:, :24576] if self.row_shard else scratch.projected_qkv
        conv_args = {
            "x": conv_x, "conv_weight": weights["conv1d.weight"],
            "query_start_loc": metadata.query_start_loc, "state_slot_ids": metadata.convolution_slots,
            "state_is_fresh": metadata.fresh, "num_accepted_tokens": metadata.accepted_tokens,
            "num_seqs": metadata.num_seqs, "num_tokens": metadata.num_tokens,
            "conv_state": conv_state_buf, "out": scratch.convolved_qkv,
        }
        conv_caps = short_conv.Caps(device=self.device, mode=mode, channels=3 * self.width,
            kernel_size=4, max_tokens=self.seg_tokens, max_seqs=self.sequences,
            max_state_slots=conv_state_buf.shape[0], max_speculative_tokens=conv_drafts)
        self.conv_plan = short_conv.plan(conv_caps, invocation=short_conv.invocation_from_tensors(**conv_args))
        self.conv_binding = prepare_sequence_binding(self.conv_plan, conv_args,
            session=session, name=f"{name}.short_conv", public_bind=short_conv.bind,
            mode=mode, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "conv_binding", binding))
        common = {
            "raw_g": scratch.raw_g.view(self.seg_tokens, self.heads, 128),
            "raw_beta": seg_beta if self.row_shard else scratch.raw_beta,
            "A_log": weights["A_log"], "dt_bias": weights["dt_bias"].view(self.heads, 128),
            "recurrent_state": state.recurrent, "num_seqs": metadata.num_seqs, "num_tokens": metadata.num_tokens,
            # Decode writes normalized output straight into the output projection's input.
            "output": (scratch.gated if mode == "decode" else scratch.recurrence).view(self.seg_tokens, self.heads, 128),
        }
        self.options = {"lower_bound": -5.0}
        if mode == "decode" and records is not None and sm103.is_sm103(self.device):
            # On SM103 Triton records verification (b12x's CuTe recording runs 1.6x slower there); the gated
            # output norm follows as a separate launch.
            from .kda_recovery import PreparedKDAVerification

            self.verification = PreparedKDAVerification(mixed_qkv=scratch.convolved_qkv, raw_g=common["raw_g"],
                raw_beta=scratch.raw_beta, A_log=weights["A_log"], dt_bias=common["dt_bias"],
                state=state.recurrent, metadata=metadata, output=scratch.recurrence, records=records,
                heads=self.heads)
            self.output_norm = PreparedGatedRMSNorm(scratch.recurrence.view(self.tokens, self.heads, 128),
                scratch.z.view(self.tokens, self.heads, 128), module.o_norm.weight,
                scratch.gated.view(self.tokens, self.heads, 128), metadata.num_tokens,
                eps=module.layer_norm_epsilon)
            self.recurrent_plan = None
            return
        if mode == "decode":
            self.options["eps"] = module.layer_norm_epsilon
            # Speculative rounds record each verified token instead of persisting its FP32 state.
            caps = gdn_decode.Caps(device=self.device, max_tokens=self.tokens, max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0], key_heads=self.heads, value_heads=self.heads,
                state_index_columns=self.columns, gate_activation="sigmoid", null_state_index=0,
                recover_speculative_state=records is not None)
            args = dict(common, mixed_qkv=scratch.convolved_qkv, z=scratch.z.view(self.tokens, self.heads, 128),
                norm_weight=module.o_norm.weight, query_start_loc=metadata.query_start_loc,
                num_accepted_tokens=metadata.accepted_tokens, state_indices=metadata.state_indices)
            if records is not None:
                args.update(correction_cache=records[0], kg_cache=records[1])
                self.records = records
                self.handle = register_recording(self)
            self.recurrent_plan = gdn_decode.plan(caps, invocation=gdn_decode.invocation_from_tensors(caps, **args))
            self.record_scratch_bytes = sum(spec.nbytes for spec in self.recurrent_plan.scratch_specs())
            public_bind, bind_method = gdn_decode.bind_kda, "bind_kda"
        elif self.row_shard:
            caps = kda_prefill.Caps(device=self.device, max_tokens=self.seg_tokens, max_seqs=1,
                max_state_slots=6, heads=64, null_state_index=0, checkpoint_export=False)
            q, k, v = (t.view(self.seg_tokens, 64, 128) for t in scratch.convolved_qkv.split(8192, dim=-1))
            seg_inputs = {
                "q": q, "k": k, "v": v, "raw_g": scratch.raw_g.view(self.seg_tokens, 64, 128),
                "raw_beta": seg_beta, "A_log": weights["A_log"],
                "dt_bias": weights["dt_bias"].view(64, 128), "cu_seqlens": metadata.query_start_loc,
                "checkpoint_state_indices": metadata.checkpoint_state_indices,
                "checkpoint_offsets": metadata.checkpoint_offsets,
                "num_seqs": metadata.num_seqs, "num_tokens": metadata.num_tokens,
            }
            self.recurrent_plan = kda_prefill.segment_plan(
                caps,
                invocation=kda_prefill.invocation_from_tensors(
                    A_log=weights["A_log"],
                    dt_bias=weights["dt_bias"].view(64, 128),
                    initial_state_indices=metadata.initial_state_indices,
                ),
            )
            seg_out = scratch.recurrence.view(self.seg_tokens, 64, 128)
            plan_ref = self.recurrent_plan

            def prepare_seg(prepared):
                sc = planned_scratch(prepared.layout.scratch_specs(), workspace)
                def run():
                    prepared.summarize(plan=plan_ref, inputs=seg_inputs, scratch=sc, pool=segment_pool,
                                       output=seg_out, lower_bound=-5.0, scale=None, eps=1e-6)
                    prepared.finalize(plan=plan_ref, inputs=seg_inputs, scratch=sc, pool=segment_pool,
                                      output=seg_out)
                return PreparedCall(run=run, owners=prime_owners(workspace, sc, segment_pool, seg_out))

            def benchmark_seg(prepared):
                sc = planned_scratch(prepared.layout.scratch_specs(), workspace)
                dev = self.device
                bench_qsl = torch.tensor([0, self.seg_tokens], dtype=torch.int32, device=dev)
                bench_ns = torch.tensor([1], dtype=torch.int32, device=dev)
                bench_nt = torch.tensor([self.seg_tokens], dtype=torch.int32, device=dev)
                bench_pool = torch.zeros((6, 64, 128, 128), dtype=torch.float32, device=dev)
                bench_pool[1].copy_(self.segment_eye)
                bench_out = torch.zeros((self.seg_tokens, 64, 128), dtype=torch.bfloat16, device=dev)
                bench_inputs = dict(seg_inputs, cu_seqlens=bench_qsl, num_seqs=bench_ns, num_tokens=bench_nt)
                eye_ref = self.segment_eye
                def reset():
                    bench_pool[0].zero_()
                    bench_pool[1].copy_(eye_ref)
                    bench_pool[4].zero_()
                    bench_out.zero_()
                def run():
                    prepared.summarize(plan=plan_ref, inputs=bench_inputs, scratch=sc, pool=bench_pool,
                                       output=bench_out, lower_bound=-5.0, scale=None, eps=1e-6)
                    prepared.finalize(plan=plan_ref, inputs=bench_inputs, scratch=sc, pool=bench_pool,
                                      output=bench_out)
                return PreparedCall(run=run, output=bench_out, produce=reset,
                    benchmark_producers=(reset,), reset=reset, restore=reset,
                    owners=prime_owners(workspace, sc, bench_pool, bench_out, bench_qsl, bench_ns, bench_nt))

            declare(session, (self.recurrent_plan.request(
                name=f"{name}.segment", prepare_call=prepare_seg, benchmark_call=benchmark_seg),))
            self.recurrence = None
            self.output_norm = PreparedGatedRMSNorm(out_scratch.recurrence.view(self.output_tokens, 64, 128),
                out_scratch.z.view(self.output_tokens, 64, 128), module.o_norm.weight,
                out_scratch.gated.view(self.output_tokens, 64, 128), metadata.num_tokens,
                eps=module.o_norm.variance_epsilon)
            return
        else:
            caps = kda_prefill.Caps(device=self.device, max_tokens=self.tokens, max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0], heads=self.heads, null_state_index=0,
                checkpoint_export=checkpoint_export)
            q, k, v = (t.view(self.tokens, self.heads, 128) for t in scratch.convolved_qkv.split(self.width, dim=-1))
            args = dict(common, q=q, k=k, v=v, cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices, final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices, checkpoint_offsets=metadata.checkpoint_offsets)
            self.recurrent_plan = kda_prefill.plan(caps, invocation=kda_prefill.invocation_from_tensors(**args))
            public_bind, bind_method = kda_prefill.bind, "bind"
        self.recurrence = prepare_sequence_binding(self.recurrent_plan, args,
            session=session, name=f"{name}.{mode}", public_bind=public_bind, bind_method=bind_method,
            mode=mode, workspace=workspace, run_options=self.options,
            set_binding=lambda binding: setattr(self, "recurrence", binding))
        if mode == "prefill":
            # The canonical sigmoid-gated FP32 RMSNorm, fused, writing the output projection's input.
            self.output_norm = PreparedGatedRMSNorm(scratch.recurrence.view(self.tokens, self.heads, 128),
                scratch.z.view(self.tokens, self.heads, 128), module.o_norm.weight,
                scratch.gated.view(self.tokens, self.heads, 128), metadata.num_tokens,
                eps=module.o_norm.variance_epsilon)

    def functional(self, *, session):
        if self.gate_overlap is not None:
            self.gate_overlap.functional(session=session)
        for projection in self.projections.values():
            projection.functional(session=session)
        if self.verification is not None:
            self.verification.functional()
            self.output_norm.functional(session=session, dynamic_live=True)
        elif self.mode == "prefill":
            self.output_norm.functional(session=session, dynamic_live=True)
        self.scratch = self.metadata = self.records = None
        self.arena.register_preparation_release(self._release_bindings)
        return self

    def _release_bindings(self):
        self.conv_binding = self.recurrence = self.arena = None

    def __call__(self, source, metadata, *, records=None):
        if self.recording and records is None:
            raise ValueError("recording KDA execution requires invocation-local records")
        tokens = source.shape[0]
        scratch = KDAScratch.allocate(tokens, self.heads, source.device)
        pending = None if self.gate_overlap is None else self.gate_overlap.start(source)
        projected = self.projections["input_proj"](source)
        projected_qkv = projected[:, :3 * self.width]
        forget_input = projected[:, -128:]
        if tokens > 1:
            scratch.forget_input.copy_(forget_input)
            forget_input = scratch.forget_input
        raw_g = self.projections["f_b_proj"](forget_input)
        if self.gate_overlap is None:
            gate_input = self.projections["g_a_proj"](source)
            z = self.projections["g_b_proj"](gate_input)
        else:
            z = self.gate_overlap.finish(pending, raw_g)
        conv_args = dict(x=projected_qkv, conv_weight=self.weights["conv1d.weight"],
            query_start_loc=metadata.query_start_loc, state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh, num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            conv_state=self.state.convolution, out=scratch.convolved_qkv)
        conv_scratch = torch.empty(self.conv_plan.scratch_specs()[0].nbytes,
                                   dtype=torch.uint8, device=source.device)
        run_short_conv(self.conv_plan, scratch=conv_scratch, **conv_args)
        raw_g = raw_g.view(tokens, self.heads, 128)
        z = z.view(tokens, self.heads, 128)
        raw_beta = projected[:, 3 * self.width:3 * self.width + self.heads]
        gated = scratch.gated.view(tokens, self.heads, 128)
        if self.verification is not None:
            recurrence = scratch.recurrence.view(tokens, self.heads, 128)
            self.verification(mixed_qkv=scratch.convolved_qkv, raw_g=raw_g, raw_beta=raw_beta,
                A_log=self.weights["A_log"], dt_bias=self.weights["dt_bias"].view(self.heads, 128),
                state=self.state.recurrent, metadata=metadata, output=recurrence, records=records)
            gated = self.output_norm(recurrence, z, live=metadata.num_tokens)
        elif self.mode == "decode" and self.recording:
            record_scratch = torch.empty((self.record_scratch_bytes,), dtype=torch.uint8, device=self.device)
            _record([scratch.convolved_qkv, raw_g, raw_beta, z, self.weights["A_log"],
                     self.weights["dt_bias"].view(self.heads, 128), self.module.o_norm.weight,
                     metadata.query_start_loc, metadata.accepted_tokens, metadata.state_indices,
                     metadata.num_seqs, metadata.num_tokens], self.state.recurrent, gated,
                    *records, record_scratch, self.handle)
        elif self.mode == "decode":
            kda_decode_out(scratch.convolved_qkv, raw_g, raw_beta, z, self.weights["A_log"],
                self.weights["dt_bias"].view(self.heads, 128), self.module.o_norm.weight,
                self.state.recurrent, metadata.query_start_loc, metadata.accepted_tokens,
                metadata.state_indices, metadata.num_seqs, metadata.num_tokens, gated,
                self.module.layer_norm_epsilon, 128 ** -.5, -5.0, self.recurrent_plan.handle)
        else:
            recurrence_scratch = torch.empty(self.recurrent_plan.scratch_specs()[0].nbytes,
                                             dtype=torch.uint8, device=source.device)
            q, k, v = (value.view(tokens, self.heads, 128)
                       for value in scratch.convolved_qkv.split(self.width, dim=-1))
            recurrence = scratch.recurrence.view(tokens, self.heads, 128)
            kda_prefill_out([q, k, v, raw_g, raw_beta, self.weights["A_log"],
                self.weights["dt_bias"].view(self.heads, 128), metadata.query_start_loc,
                metadata.initial_state_indices, metadata.final_state_indices,
                metadata.checkpoint_state_indices, metadata.checkpoint_offsets, metadata.num_seqs,
                metadata.num_tokens], recurrence_scratch, self.state.recurrent, recurrence,
                self.recurrent_plan.handle, -5.0, None, 1e-6, None, None)
            gated = self.output_norm(recurrence, z, live=metadata.num_tokens)
        return self.projections["o_proj"](gated.reshape(tokens, -1))


    def pack_initial_state(self, metadata):
        """Pack this rank's 16 decode heads of FP32 initial recurrent state as BF16 [64, 8192]."""
        init_slot = metadata.initial_state_indices[:1].clamp_min(0).to(torch.int64)
        s_init_16 = torch.index_select(self.state.recurrent, 0, init_slot).squeeze(0)
        return s_init_16.contiguous().view(torch.bfloat16).reshape(ROW_SHARD_SINIT_SHAPE)

    @property
    def supports_zero_copy_segment(self):
        from hyperflare_recipes.components.linear import UnquantizedProjection
        return (isinstance(self.projections["input_proj"], UnquantizedProjection)
                and isinstance(self.projections["f_b_proj"], UnquantizedProjection))

    def project_input_shard(self, source, *, out=None):
        """Run 64-head input projection on one half-segment of owned rows."""
        proj = self.projections["input_proj"]
        projected = proj(source, output=out) if out is not None else proj(source)
        qkv = projected[:, :24576]
        raw_beta = projected[:, 24576:24640]
        forget_in = projected[:, -128:].contiguous()
        return qkv, raw_beta, forget_in

    def pack_conv_halo(self, qkv_b, metadata):
        """Pack this rank's 16-head conv history [3, 6144] + Half B last-3 QKV rows [3, 24576]."""
        conv_slot = metadata.convolution_slots[:1].clamp_min(0).to(torch.int64)
        rank_conv = torch.index_select(self.state.convolution, 0, conv_slot).squeeze(0)[:, :3].transpose(0, 1)
        return torch.cat((rank_conv, qkv_b[-3:, :]), dim=-1).contiguous()

    def project_forget_shard(self, forget_in, *, out=None):
        """Run 64-head forget projection on one half-segment (overlaps halo gather)."""
        proj = self.projections["f_b_proj"]
        return proj(forget_in, output=out) if out is not None else proj(forget_in)

    def project_gate_shard(self, source):
        """Run 64-head output gate chain on one half-segment (overlaps state scan gather)."""
        gate_in = self.projections["g_a_proj"](source)
        return self.projections["g_b_proj"](gate_in).view(self.tokens, 64, 128)

    def summarize_segment(self, qkv_a, qkv_b, raw_g_a, raw_g_b, raw_beta_a, raw_beta_b, halo_full, metadata,
                          *, seg_projected=None, seg_raw_g=None):
        """Run one contiguous 2*tokens short_conv + segment_summarize and pack compressed scan payload."""
        if seg_projected is not None and seg_raw_g is not None:
            qkv = seg_projected[:, :24576]
            raw_g = seg_raw_g
            raw_beta = seg_projected[:, 24576:24640].contiguous()
        else:
            qkv = torch.cat((qkv_a, qkv_b), dim=0)
            raw_g = torch.cat((raw_g_a, raw_g_b), dim=0)
            raw_beta = torch.cat((raw_beta_a, raw_beta_b), dim=0)
        device = qkv.device

        prev_rank = (self.rank_index - 1).clamp_min(0)
        is_rank0 = self.rank_index == 0
        hf = halo_full.view(4, 3, 30720)
        c_init = hf[:, :, :6144].view(4, 3, 3, 2048).permute(2, 0, 3, 1).reshape(24576, 3)
        prev_tail = torch.index_select(hf[:, :, 6144:], 0, prev_rank).squeeze(0).transpose(0, 1)
        local_conv = torch.where(is_rank0.view(1, 1), c_init, prev_tail).unsqueeze(0).contiguous()
        conv_fresh = metadata.fresh[:1] & is_rank0

        offset = (self.rank_index * self.seg_tokens).to(torch.int32)
        live_r = (metadata.num_tokens - offset).clamp(0, self.seg_tokens).to(torch.int32)
        has_r = (metadata.num_seqs > 0) & (live_r > 0)
        zero_i32 = metadata.query_start_loc[:1]
        qsl_r = torch.cat((zero_i32, live_r))
        slot0 = torch.where(has_r, torch.zeros_like(metadata.convolution_slots[:1]),
                            torch.full_like(metadata.convolution_slots[:1], -1))
        num_seqs_conv = has_r.to(torch.int32)

        convolved = torch.empty((self.seg_tokens, 24576), dtype=torch.bfloat16, device=device)
        conv_scratch = torch.empty(self.conv_plan.scratch_specs()[0].nbytes, dtype=torch.uint8, device=device)
        run_short_conv(
            self.conv_plan, scratch=conv_scratch, x=qkv, conv_weight=self.weights["conv1d.weight"],
            query_start_loc=qsl_r, state_slot_ids=slot0, state_is_fresh=conv_fresh,
            num_accepted_tokens=metadata.accepted_tokens[:1], num_seqs=num_seqs_conv, num_tokens=live_r,
            conv_state=local_conv, out=convolved,
        )

        q, k, v = (val.view(self.seg_tokens, 64, 128) for val in convolved.split(8192, dim=-1))
        raw_g_h = raw_g.view(self.seg_tokens, 64, 128)
        seg_scratch = torch.empty(self.recurrent_plan.scratch_specs()[0].nbytes, dtype=torch.uint8, device=device)
        pool = torch.empty((6, 64, 128, 128), dtype=torch.float32, device=device)
        pool[0].zero_()
        pool[1].copy_(self.segment_eye)
        rec_out = torch.empty((self.seg_tokens, 64, 128), dtype=torch.bfloat16, device=device)
        ns_one = torch.ones_like(live_r)
        zero_ckpt_idx = torch.zeros_like(metadata.checkpoint_state_indices[:1])
        zero_ckpt_off = torch.zeros_like(metadata.checkpoint_offsets[:1])
        kda_prefill.segment_summarize(
            self.recurrent_plan, q=q, k=k, v=v, raw_g=raw_g_h, raw_beta=raw_beta,
            A_log=self.weights["A_log"], dt_bias=self.weights["dt_bias"].view(64, 128),
            cu_seqlens=qsl_r, checkpoint_state_indices=zero_ckpt_idx,
            checkpoint_offsets=zero_ckpt_off, num_seqs=ns_one, num_tokens=live_r,
            scratch=seg_scratch, pool=pool, output=rec_out, lower_bound=-5.0, eps=1e-6,
        )

        # Per-row FP8 A_r + per-row INT16 B_r + post-conv state = 3.20 MiB/rank (12.81 MiB gathered).
        A_raw = pool[2]
        a_sc = A_raw.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 448.0
        a_q = (A_raw / a_sc).to(torch.float8_e4m3fn).view(torch.uint8).reshape(64, 16384).view(torch.bfloat16)
        a_sc_p = a_sc.squeeze(-1).contiguous().view(torch.bfloat16)

        B_raw = pool[3]
        b_sc = B_raw.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 32767.0
        b_q = torch.round(B_raw / b_sc).clamp(-32767.0, 32767.0).to(torch.int16).view(torch.bfloat16).reshape(64, 16384)
        b_sc_p = b_sc.squeeze(-1).contiguous().view(torch.bfloat16)

        conv_packed = local_conv[0].reshape(64, 1152)
        scan_sh = torch.cat((a_q, a_sc_p, b_q, b_sc_p, conv_packed), dim=-1).contiguous()
        summary = (q, k, v, raw_g_h, raw_beta, qsl_r, ns_one, live_r,
                   zero_ckpt_idx, zero_ckpt_off, seg_scratch, pool, rec_out)
        return summary, scan_sh

    def finalize_segment(self, summary, s_init_full, scan_full, metadata):
        """Run 4-rank FP32 affine prefix scan and single contiguous segment_finalize."""
        (q, k, v, raw_g_h, raw_beta, qsl_r, ns_one, live_r,
         zero_ckpt_idx, zero_ckpt_off, seg_scratch, pool, rec_out) = summary
        s0_raw = s_init_full.contiguous().view(torch.float32).reshape(64, 128, 128)
        s0 = torch.where(metadata.fresh[:1].view(1, 1, 1), torch.zeros_like(s0_raw), s0_raw)

        sf = scan_full.view(4, 64, ROW_SHARD_SCAN_SHAPE[1])
        A_q = sf[:, :, 0:8192].contiguous().view(torch.float8_e4m3fn).reshape(4, 64, 128, 128).float()
        A_sc = sf[:, :, 8192:8448].contiguous().view(torch.float32).reshape(4, 64, 128, 1)
        A_all = A_q * A_sc
        B_q = sf[:, :, 8448:24832].contiguous().view(torch.int16).reshape(4, 64, 128, 128).float()
        B_sc = sf[:, :, 24832:25088].contiguous().view(torch.float32).reshape(4, 64, 128, 1)
        B_all = B_q * B_sc
        conv_all = sf[:, :, 25088:26240].reshape(4, 24576, 3)

        s1 = torch.bmm(s0, A_all[0]) + B_all[0]
        s2 = torch.bmm(s1, A_all[1]) + B_all[1]
        s3 = torch.bmm(s2, A_all[2]) + B_all[2]
        s4 = torch.bmm(s3, A_all[3]) + B_all[3]
        s_in = torch.index_select(torch.stack((s0, s1, s2, s3), dim=0), 0, self.rank_index).squeeze(0)
        pool[4].copy_(s_in)

        kda_prefill.segment_finalize(
            self.recurrent_plan, q=q, k=k, v=v, raw_g=raw_g_h, raw_beta=raw_beta,
            A_log=self.weights["A_log"], dt_bias=self.weights["dt_bias"].view(64, 128),
            cu_seqlens=qsl_r, checkpoint_state_indices=zero_ckpt_idx,
            checkpoint_offsets=zero_ckpt_off, num_seqs=ns_one, num_tokens=live_r,
            scratch=seg_scratch, pool=pool, output=rec_out,
        )
        rec_a, rec_b = rec_out.split(self.tokens, dim=0)
        live_a = live_r.clamp(0, self.tokens).to(torch.int32)
        live_b = (live_r - self.tokens).clamp(0, self.tokens).to(torch.int32)
        return rec_a.contiguous(), rec_b.contiguous(), live_a, live_b, s4, conv_all

    def output_shard(self, rec_sh, z_sh, live_sh):
        """Run fused gated RMSNorm and 64-head output projection on one shard segment."""
        gated = self.output_norm(rec_sh, z_sh, live=live_sh)
        return self.projections["o_proj"](gated.reshape(self.output_tokens, 8192))

    def commit_state(self, s4, conv_all, metadata):
        """Write this rank's 16 decode heads of final recurrent and convolution state back to DeltaState."""
        active = (metadata.num_seqs > 0) & (metadata.num_tokens > 0)
        last_rank = torch.div((metadata.num_tokens - 1).clamp_min(0), self.seg_tokens,
                              rounding_mode="floor").clamp(0, 3).to(torch.int64)
        conv_final = torch.index_select(conv_all, 0, last_rank).squeeze(0)
        s_final_16 = torch.index_select(s4.view(4, 16, 128, 128), 0, self.rank_index).squeeze(0)
        conv_final_16 = torch.index_select(
            conv_final.view(3, 4, 2048, 3), 1, self.rank_index).squeeze(1).reshape(6144, 3)

        zero_i64 = torch.zeros_like(metadata.final_state_indices[:1])
        final_slot = torch.where(active, metadata.final_state_indices[:1].clamp_min(0), zero_i64).to(torch.int64)
        old_rec = torch.index_select(self.state.recurrent, 0, final_slot).squeeze(0)
        self.state.recurrent.index_copy_(
            0, final_slot, torch.where(active.view(1, 1, 1), s_final_16, old_rec).unsqueeze(0))

        conv_slot = torch.where(active, metadata.convolution_slots[:1].clamp_min(0), zero_i64).to(torch.int64)
        old_conv_full = torch.index_select(self.state.convolution, 0, conv_slot).squeeze(0)
        new_conv_3 = torch.where(active.view(1, 1), conv_final_16, old_conv_full[:, :3])
        if old_conv_full.shape[-1] > 3:
            tail = torch.where(active.view(1, 1), torch.zeros_like(old_conv_full[:, 3:]), old_conv_full[:, 3:])
            new_conv_full = torch.cat((new_conv_3, tail), dim=-1)
        else:
            new_conv_full = new_conv_3
        self.state.convolution.index_copy_(0, conv_slot, new_conv_full.unsqueeze(0))

    def run_recorded(self, reads, recurrent_state, output, corrections, keys_gates, scratch):
        (mixed_qkv, raw_g, raw_beta, z, A_log, dt_bias, norm_weight, query_start_loc,
         num_accepted_tokens, state_indices, num_seqs, num_tokens) = reads
        binding = gdn_decode.bind_kda(self.recurrent_plan, scratch=scratch,
            mixed_qkv=mixed_qkv, raw_g=raw_g, raw_beta=raw_beta, z=z, A_log=A_log,
            dt_bias=dt_bias, norm_weight=norm_weight, recurrent_state=recurrent_state,
            query_start_loc=query_start_loc, num_accepted_tokens=num_accepted_tokens,
            state_indices=state_indices, num_seqs=num_seqs, num_tokens=num_tokens,
            output=output, correction_cache=corrections, kg_cache=keys_gates)
        gdn_decode.run_kda(binding, lower_bound=-5.0, eps=self.module.layer_norm_epsilon)
