// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0

// SM103 GPU-side persistent expert mover (1D TMA cp.async.bulk over NVLink-C2C
// HOST_NUMA VMM), lock-free MPSC prefetch ring buffer, and Dekker/Epoch-RCU
// layer-enter snapshot kernel.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace flare::sm103_mover {

static constexpr uint32_t kTmaChunkBytes = 32768u; // 32 KiB x 6 stages = 192 KiB dynamic SMEM
static constexpr uint32_t kTmaStages = 6u;
static constexpr uint32_t kMaxPredictExperts = 1024u;
static constexpr uint32_t kNumStageSlots = 4u;
static constexpr uint32_t kStageSlotMask = kNumStageSlots - 1u;
static constexpr uint32_t kPredictRingSize = 64u;
static constexpr uint32_t kPredictRingMask = kPredictRingSize - 1u;
static constexpr uint32_t kMaxPredictRoutes = 768u; // covers up to 128 rows x 6 topk

struct alignas(64) LayerMoverDesc {
    uint64_t host_planes[4];     // HOST_NUMA base pointers: gate_up, gate_up_sf, down, down_sf
    uint64_t hbm_planes[4];      // HBM pool base pointers: gate_up, gate_up_sf, down, down_sf
    uint64_t plane_row_bytes[4]; // 16-byte-aligned byte stride per expert row
    uint64_t live_slot_map;      // int32_t[num_experts] in HBM (-1 = cold, >= 0 = HBM slot)
    uint64_t slot_owner;         // int32_t[num_hbm_slots] in HBM (-1 = spare/free)
    uint64_t layer_epoch;        // uint32_t* in HBM (even = idle, odd = active)
    uint32_t num_experts;
    uint32_t num_hbm_slots;
    uint32_t victim_cursor;      // clock-hand cursor in [0, num_hbm_slots)
    uint32_t num_host_experts;   // movable experts in [0, num_host_experts); [num_host_experts, num_experts) pinned HBM-only
    uint64_t predictor_state;    // PredictorLayerState* in HBM (0 if predictor not attached)
};

static_assert(sizeof(LayerMoverDesc) == 192, "LayerMoverDesc layout mismatch");

struct alignas(16) MoverRequest {
    uint32_t layer_idx;
    int32_t  expert_in;
    int32_t  victim_expert; // -1 = choose resident victim via clock hand
    uint32_t seq;           // MPSC publication tag: queue index + 1
};

static_assert(sizeof(MoverRequest) == 16, "MoverRequest layout mismatch");

struct alignas(64) StageRequest {
    uint64_t dst_planes[4]; // HBM staging destination planes
    uint64_t staged_mask;   // int32_t[num_host_experts] written by worker
    uint32_t layer_idx;     // layer index in LayerMoverDesc[]
    uint32_t min_cold;      // minimum cold expert count to copy
    uint32_t req_seq;       // monotonic request sequence (published with release)
    uint32_t done_seq;      // worker release-stores req_seq when copy completes
    uint32_t staged_count;  // number of cold experts copied into dst_planes
    uint32_t _pad;
};

static_assert(sizeof(StageRequest) == 64, "StageRequest layout mismatch");

struct alignas(64) PredictEntry {
    uint32_t seq;                       // MPSC publication tag: queue index + 1
    uint32_t layer_idx;
    uint32_t num_routes;
    uint32_t _pad;
    int16_t  routes[kMaxPredictRoutes]; // routed expert IDs (-1 = padded)
};

static_assert(sizeof(PredictEntry) == 1600, "PredictEntry layout mismatch");

struct alignas(64) MoverQueueHeader {
    uint32_t head;
    uint32_t tail;
    uint32_t mask;
    uint32_t barrier_count;
    uint32_t barrier_sense;
    uint32_t active_stop;
    uint32_t active_layer;
    int32_t  active_expert_in;
    int32_t  active_spare_slot;
    int32_t  active_victim_expert;
    int32_t  active_victim_slot;
    uint32_t _pad0;
    uint64_t predict_hits;
    uint64_t predict_misses;
    uint64_t predict_enqueues;
    uint64_t _pad1;
    uint32_t worker_running;     // 1 while persistent worker is active on green context
    uint32_t stage_req_seq;      // monotonic prefill-stage request sequence
    uint32_t stage_head_seq;     // next prefill-stage sequence for worker to consume
    uint32_t stage_in_flight;    // active prefill-stage count (pauses decode swaps while > 0)
    uint32_t active_cmd;         // 0 = stop, 1 = decode swap, 2 = prefill stage
    uint32_t active_stage_slot;  // active slot index in stage_slots[0..3]
    uint32_t active_cold_count;  // active cold row count in stage_cold_rows[]
    uint32_t _pad2[5];
    StageRequest stage_slots[kNumStageSlots];
    int32_t stage_cold_rows[kMaxPredictExperts];
    uint32_t predict_head;       // consumer head for worker-offloaded predictor ring
    uint32_t predict_tail;       // producer tail for worker-offloaded predictor ring
    uint32_t _pad3[14];
    PredictEntry predict_ring[kPredictRingSize];
};

static_assert(sizeof(MoverQueueHeader) == 106944, "MoverQueueHeader layout mismatch");

struct alignas(64) PredictorLayerState {
    uint64_t scores;            // uint32_t[num_host_experts] fixed-point decayed cold-miss score (1 cold miss = 256)
    uint64_t last_seen;         // uint32_t[num_host_experts] monotonic step clock of last HBM hit or physical install
    uint64_t born;              // uint32_t[num_host_experts] bit 31 = was_resident, bits 0..30 = install step clock
    uint64_t cooldown;          // uint32_t[num_host_experts] monotonic step clock of last enqueue
    uint32_t step_clock;        // initialized >= max(1, cooldown_steps, min_age_steps, interval_steps); incremented on active steps
    uint32_t last_enqueue_step; // monotonic step clock of the most recent enqueue wave for this layer
    uint32_t layer_idx;
    uint32_t decay_shift;       // e.g. 6 -> score -= score >> 6 (64-step exponential decay window)
    uint32_t promote_threshold; // fixed-point cold-miss score threshold (256 = 1.0 miss)
    uint32_t min_age_steps;     // minimum resident age in active steps before a newly installed expert is evictable
    uint32_t interval_steps;    // minimum active steps between enqueue waves for this layer
    uint32_t cooldown_steps;    // minimum active steps before re-enqueuing the same cold expert
    uint32_t max_enqueues;      // cap on swap requests enqueued per layer invocation (<= 4)
    uint32_t max_predict_rows;  // skip scoring/enqueueing when num_rows > max_predict_rows
    uint32_t num_host_experts;  // movable host-backed experts [0, num_host_experts); >= num_host_experts are pinned HBM-only
    uint32_t _pad[13];
};

static_assert(sizeof(PredictorLayerState) == 128, "PredictorLayerState layout mismatch");

__device__ __forceinline__ void worker_grid_barrier(MoverQueueHeader* hdr, uint32_t& local_sense) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t target_sense = local_sense ^ 1u;
        uint32_t arrived = 0;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(arrived) : "l"(&hdr->barrier_count) : "memory");
        if (arrived + 1u == gridDim.x) {
            asm volatile("st.relaxed.gpu.global.u32 [%0], 0;"
                         :: "l"(&hdr->barrier_count) : "memory");
            asm volatile("st.release.gpu.global.u32 [%0], %1;"
                         :: "l"(&hdr->barrier_sense), "r"(target_sense) : "memory");
        } else {
            uint32_t cur = 0;
            do {
                asm volatile("nanosleep.u32 32;");
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(cur) : "l"(&hdr->barrier_sense) : "memory");
            } while (cur != target_sense);
        }
        local_sense = target_sense;
    }
    __syncthreads();
}

__device__ __forceinline__ void store_metric_sys(uint64_t* metrics, uint32_t metrics_count, uint32_t idx, uint64_t value) {
    if (metrics != nullptr && idx < metrics_count) {
        asm volatile("st.release.sys.global.u64 [%0], %1;"
                     :: "l"(metrics + idx), "l"(value) : "memory");
    }
}

__device__ __forceinline__ void store_metric_sys_relaxed(uint64_t* metrics, uint32_t metrics_count, uint32_t idx, uint64_t value) {
    if (metrics != nullptr && idx < metrics_count) {
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
                     :: "l"(metrics + idx), "l"(value) : "memory");
    }
}

__global__ void sm103_layer_enter_kernel(
    const int32_t* __restrict__ live_slot_map,
    int32_t* __restrict__ active_slot_map,
    uint32_t* __restrict__ layer_epoch,
    uint32_t num_experts
) {
    if (blockIdx.x != 0) return;
    if (threadIdx.x == 0) {
        uint32_t old_ep;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(old_ep) : "l"(layer_epoch) : "memory");
        asm volatile("fence.sc.gpu;" ::: "memory");
    }
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
        int32_t slot = -1;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                     : "=r"(slot) : "l"(live_slot_map + e) : "memory");
        active_slot_map[e] = slot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
}

__global__ void sm103_stage_request_kernel(
    MoverQueueHeader* __restrict__ hdr,
    uint32_t layer_idx,
    uint64_t dst_plane0,
    uint64_t dst_plane1,
    uint64_t dst_plane2,
    uint64_t dst_plane3,
    int32_t* __restrict__ staged_mask,
    uint32_t* __restrict__ req_seq_out,
    uint32_t num_host_experts,
    uint32_t min_cold
) {
    if (blockIdx.x != 0) return;
    __shared__ uint32_t s_running;
    if (threadIdx.x == 0) {
        uint32_t running = 0u;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                     : "=r"(running) : "l"(&hdr->worker_running) : "memory");
        s_running = running;
        if (running == 0u) {
            *req_seq_out = 0u;
        } else {
            uint32_t prev = 0u;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                         : "=r"(prev) : "l"(&hdr->stage_req_seq) : "memory");
            const uint32_t seq = prev + 1u;
            const uint32_t s = (seq - 1u) & kStageSlotMask;
            StageRequest* slot = &hdr->stage_slots[s];
            // Wait if slot has an uncompleted older request (cannot happen under depth-2 staging with 4 slots).
            while (true) {
                uint32_t cur_req = 0u, cur_done = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(cur_req) : "l"(&slot->req_seq) : "memory");
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(cur_done) : "l"(&slot->done_seq) : "memory");
                if (cur_done == cur_req) break;
                uint32_t still_up = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(still_up) : "l"(&hdr->worker_running) : "memory");
                if (still_up == 0u) {
                    s_running = 0u;
                    *req_seq_out = 0u;
                    break;
                }
                asm volatile("nanosleep.u32 32;");
            }
            if (s_running != 0u) {
                uint32_t dummy = 0u;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                             : "=r"(dummy) : "l"(&hdr->stage_in_flight) : "memory");
                slot->dst_planes[0] = dst_plane0;
                slot->dst_planes[1] = dst_plane1;
                slot->dst_planes[2] = dst_plane2;
                slot->dst_planes[3] = dst_plane3;
                slot->staged_mask = reinterpret_cast<uint64_t>(staged_mask);
                slot->layer_idx = layer_idx;
                slot->min_cold = min_cold;
                slot->staged_count = 0u;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                             :: "l"(&slot->req_seq), "r"(seq) : "memory");
                *req_seq_out = seq;
            }
        }
    }
    __syncthreads();
    if (s_running == 0u) {
        for (uint32_t e = threadIdx.x; e < num_host_experts; e += blockDim.x) {
            staged_mask[e] = 0;
        }
    }
}

__global__ void sm103_layer_enter_staged_kernel(
    const int32_t* __restrict__ live_slot_map,
    const int32_t* __restrict__ base_control,
    int32_t* __restrict__ staged_control,
    uint32_t* __restrict__ layer_epoch,
    const int32_t* __restrict__ staged_mask,
    uint64_t staged_plane0,
    uint64_t staged_plane1,
    uint64_t staged_plane2,
    uint64_t staged_plane3,
    uint32_t num_experts,
    uint32_t num_host_experts,
    uint32_t min_cold,
    uint32_t has_host_lane_counters,
    MoverQueueHeader* __restrict__ queue_hdr,
    const uint32_t* __restrict__ req_seq_ptr
) {
    if (blockIdx.x != 0) return;

    __shared__ uint32_t s_staged_count;
    __shared__ uint32_t s_missed;
    __shared__ uint32_t s_waited_worker;

    if (threadIdx.x == 0) {
        s_staged_count = 0u;
        s_missed = 0u;
        s_waited_worker = 0u;
        if (queue_hdr != nullptr && req_seq_ptr != nullptr) {
            const uint32_t seq = *req_seq_ptr;
            if (seq != 0u) {
                s_waited_worker = 1u;
                StageRequest* slot = &queue_hdr->stage_slots[(seq - 1u) & kStageSlotMask];
                while (true) {
                    uint32_t done = 0u;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                 : "=r"(done) : "l"(&slot->done_seq) : "memory");
                    if (done == seq) break;
                    uint32_t running = 0u;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                 : "=r"(running) : "l"(&queue_hdr->worker_running) : "memory");
                    if (running == 0u) {
                        s_missed = 1u;
                        break;
                    }
                    asm volatile("nanosleep.u32 32;");
                }
            }
        }
        // Dekker half 1: mark layer ACTIVE (even -> odd) and fence SC before reading live_slot_map.
        uint32_t old_ep;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(old_ep) : "l"(layer_epoch) : "memory");
        asm volatile("fence.sc.gpu;" ::: "memory");
    }
    if (threadIdx.x < 16u) {
        staged_control[threadIdx.x] = base_control[threadIdx.x];
    }
    __syncthreads();

    const uint32_t host_limit = num_host_experts <= num_experts ? num_host_experts : num_experts;
    int32_t* active_slot_map = staged_control + 16;
    uint32_t local_staged = 0u;
    uint32_t local_missed = 0u;
    for (uint32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
        int32_t slot = -1;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                     : "=r"(slot) : "l"(live_slot_map + e) : "memory");
        active_slot_map[e] = slot;
        if (e < host_limit) {
            const int32_t present = staged_mask[e];
            if (present != 0) {
                local_staged += 1u;
            } else if (slot < 0) {
                // Rare race: background mover evicted expert e between stager.start() and Dekker enter.
                // Fall back to mapped-host pointers for this layer invocation so output stays bit-exact.
                local_missed = 1u;
            }
        }
    }
    if (local_staged > 0u) {
        atomicAdd(&s_staged_count, local_staged);
    }
    if (local_missed != 0u) {
        atomicOr(&s_missed, 1u);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (s_waited_worker != 0u) {
            uint32_t prev_in_flight = 0u;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], -1;"
                         : "=r"(prev_in_flight) : "l"(&queue_hdr->stage_in_flight) : "memory");
        }
        if (has_host_lane_counters != 0u) {
            staged_control[16u + num_experts] = 0;
            staged_control[16u + num_experts + 1u] = 0;
        }
        if (s_missed == 0u && s_staged_count >= min_cold) {
            uint64_t* hdr64 = reinterpret_cast<uint64_t*>(staged_control);
            hdr64[1] = staged_plane0;
            hdr64[2] = staged_plane1;
            hdr64[3] = staged_plane2;
            hdr64[4] = staged_plane3;
            // Staged rows live in HBM; disable the C2C host lane so all clusters execute HBM tasks.
            staged_control[10] = 0;
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
}

// Lock-free multi-producer slot reservation via CAS on hdr->tail.
// Guarantees concurrent producers can never overflow the bounded ring capacity.
__device__ __forceinline__ bool mpsc_try_reserve(
    MoverQueueHeader* hdr,
    uint32_t count,
    uint32_t& base_idx
) {
    const uint32_t capacity = hdr->mask + 1u;
    uint32_t tail = 0u;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(tail) : "l"(&hdr->tail) : "memory");
    while (true) {
        uint32_t head = 0u;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(head) : "l"(&hdr->head) : "memory");
        const uint32_t used = tail - head;
        if (used >= capacity || count > capacity - used) {
            return false;
        }
        uint32_t observed = 0u;
        asm volatile("atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;"
                     : "=r"(observed)
                     : "l"(&hdr->tail), "r"(tail), "r"(tail + count)
                     : "memory");
        if (observed == tail) {
            base_idx = tail;
            return true;
        }
        tail = observed;
    }
}

__global__ void sm103_layer_enter_predict_kernel(
    const int32_t* __restrict__ live_slot_map,
    int32_t* __restrict__ active_slot_map,
    uint32_t* __restrict__ layer_epoch,
    const int64_t* __restrict__ topk_ids,
    uint32_t num_rows,
    uint32_t num_topk,
    uint32_t num_experts,
    PredictorLayerState* __restrict__ pred,
    MoverQueueHeader* __restrict__ queue_hdr,
    MoverRequest* __restrict__ queue_ring,
    uint64_t* __restrict__ metrics,
    uint32_t metrics_count
) {
    if (blockIdx.x != 0) return;

    // 1. Dekker half 1: mark layer ACTIVE (even -> odd) and fence SC before reading live_slot_map.
    if (threadIdx.x == 0) {
        uint32_t old_ep;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(old_ep) : "l"(layer_epoch) : "memory");
        asm volatile("fence.sc.gpu;" ::: "memory");
    }
    __syncthreads();

    const bool run_predict = (pred != nullptr && topk_ids != nullptr &&
                              num_experts <= kMaxPredictExperts &&
                              num_rows <= pred->max_predict_rows);

    __shared__ int32_t  s_slot[kMaxPredictExperts];
    __shared__ uint32_t s_hits[kMaxPredictExperts];
    __shared__ uint32_t s_active_routes;
    __shared__ uint32_t s_hbm_hits;
    __shared__ uint32_t s_host_misses;

    if (run_predict && threadIdx.x == 0) {
        s_active_routes = 0u;
        s_hbm_hits = 0u;
        s_host_misses = 0u;
    }

    // 2. Snapshot live_slot_map[0..E-1] -> active_slot_map[0..E-1] (independent of router output).
    // Use 128-bit vectorized loads when 16-byte aligned and num_experts % 4 == 0, followed by a
    // single GPU-scope acquire + async-proxy fence instead of per-element acquire loads.
    const bool vec4_ok =
        ((num_experts & 3u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(live_slot_map) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(active_slot_map) & 15u) == 0u);
    if (vec4_ok) {
        const uint32_t num_vec4 = num_experts >> 2;
        const int4* live4 = reinterpret_cast<const int4*>(live_slot_map);
        int4* active4 = reinterpret_cast<int4*>(active_slot_map);
        for (uint32_t v = threadIdx.x; v < num_vec4; v += blockDim.x) {
            int4 vals;
            asm volatile("ld.relaxed.gpu.global.v4.s32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(vals.x), "=r"(vals.y), "=r"(vals.z), "=r"(vals.w)
                         : "l"(live4 + v) : "memory");
            active4[v] = vals;
            if (run_predict) {
                const uint32_t base = v << 2;
                s_slot[base + 0u] = vals.x;
                s_slot[base + 1u] = vals.y;
                s_slot[base + 2u] = vals.z;
                s_slot[base + 3u] = vals.w;
                s_hits[base + 0u] = 0u;
                s_hits[base + 1u] = 0u;
                s_hits[base + 2u] = 0u;
                s_hits[base + 3u] = 0u;
            }
        }
    } else {
        for (uint32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
            int32_t slot = -1;
            asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];"
                         : "=r"(slot) : "l"(live_slot_map + e) : "memory");
            active_slot_map[e] = slot;
            if (run_predict) {
                s_slot[e] = slot;
                s_hits[e] = 0u;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        // Release MegaMoE to launch CTAs and early shared-expert loads while scoring completes.
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    // Wait for predecessor (router / input stager) before reading topk_ids or completing.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    __syncthreads();

    // 3. Tally active routed token-expert pairs for decode/verification batches (rows <= max_predict_rows).
    // When the 8-SM persistent worker is live, offload scoring/LRU/victim-selection off the compute stream
    // by copying only the compact int16 route list into hdr->predict_ring (~0.2 us on compute stream).
    if (run_predict) {
        const uint32_t total_routes = num_rows * num_topk;
        __shared__ uint32_t s_offload_mode; // 0 = inline fallback, 1 = offloaded, 2 = ring full drop
        __shared__ uint32_t s_offload_idx;
        if (threadIdx.x == 0) {
            s_offload_mode = 0u;
            s_offload_idx = 0u;
            if (queue_hdr != nullptr && total_routes <= kMaxPredictRoutes) {
                uint32_t running = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(running) : "l"(&queue_hdr->worker_running) : "memory");
                if (running != 0u) {
                    uint32_t ptail = 0u;
                    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
                                 : "=r"(ptail) : "l"(&queue_hdr->predict_tail) : "memory");
                    while (true) {
                        uint32_t phead = 0u;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                     : "=r"(phead) : "l"(&queue_hdr->predict_head) : "memory");
                        if (ptail - phead >= kPredictRingSize) {
                            s_offload_mode = 2u;
                            break;
                        }
                        uint32_t observed = 0u;
                        asm volatile("atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;"
                                     : "=r"(observed)
                                     : "l"(&queue_hdr->predict_tail), "r"(ptail), "r"(ptail + 1u)
                                     : "memory");
                        if (observed == ptail) {
                            s_offload_idx = ptail;
                            s_offload_mode = 1u;
                            break;
                        }
                        ptail = observed;
                    }
                }
            }
        }
        __syncthreads();
        if (s_offload_mode == 1u) {
            PredictEntry* entry = &queue_hdr->predict_ring[s_offload_idx & kPredictRingMask];
            for (uint32_t i = threadIdx.x; i < total_routes; i += blockDim.x) {
                const int64_t e = topk_ids[i];
                entry->routes[i] = (e >= 0 && static_cast<uint64_t>(e) < static_cast<uint64_t>(num_experts))
                    ? static_cast<int16_t>(e)
                    : static_cast<int16_t>(-1);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                entry->layer_idx = pred->layer_idx;
                entry->num_routes = total_routes;
                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                             :: "l"(&entry->seq), "r"(s_offload_idx + 1u) : "memory");
            }
            return;
        }
        if (s_offload_mode == 2u) {
            return;
        }
        const uint32_t lane_id = threadIdx.x & 31u;
        uint32_t local_hbm_hits = 0u;
        uint32_t local_host_misses = 0u;
        for (uint32_t base = 0u; base < total_routes; base += blockDim.x) {
            const uint32_t i = base + threadIdx.x;
            bool is_hit = false;
            bool is_miss = false;
            if (i < total_routes) {
                const int64_t e = topk_ids[i];
                if (e >= 0 && static_cast<uint64_t>(e) < static_cast<uint64_t>(num_experts)) {
                    const uint32_t ue = static_cast<uint32_t>(e);
                    atomicAdd(&s_hits[ue], 1u);
                    if (s_slot[ue] >= 0) {
                        is_hit = true;
                    } else {
                        is_miss = true;
                    }
                }
            }
            const uint32_t warp_hits = __popc(__ballot_sync(0xFFFFFFFFu, is_hit));
            const uint32_t warp_misses = __popc(__ballot_sync(0xFFFFFFFFu, is_miss));
            if (lane_id == 0u) {
                local_hbm_hits += warp_hits;
                local_host_misses += warp_misses;
            }
        }
        if (lane_id == 0u && (local_hbm_hits + local_host_misses) > 0u) {
            atomicAdd(&s_hbm_hits, local_hbm_hits);
            atomicAdd(&s_host_misses, local_host_misses);
            atomicAdd(&s_active_routes, local_hbm_hits + local_host_misses);
        }
        __syncthreads();

        // 4. Update decayed cold-miss scores, birth stamps, and HBM recency ONLY when at least one active route occurred.
        if (s_active_routes > 0u) {
            const uint32_t step_now = pred->step_clock + 1u;
            uint32_t* scores = reinterpret_cast<uint32_t*>(pred->scores);
            uint32_t* last_seen = reinterpret_cast<uint32_t*>(pred->last_seen);
            uint32_t* born = reinterpret_cast<uint32_t*>(pred->born);
            uint32_t* cooldown = reinterpret_cast<uint32_t*>(pred->cooldown);

            // Parallel decayed-miss LRU state update and candidate key evaluation across all 256 threads.
            // Cold key (maximize): ((uint64_t)sc << 32) | (uint64_t)(e + 1), 0 = ineligible.
            // Victim key (minimize): ((uint64_t)last_seen[e] << 32) | (uint64_t)e, ~0ull = ineligible.
            uint64_t my_cold_key = 0ull;
            uint64_t my_vic_key = ~0ull;
            const bool eval_swap = (s_host_misses > 0u && pred->max_enqueues > 0u &&
                                    (step_now - pred->last_enqueue_step) >= pred->interval_steps);
            const uint32_t decay_shift = pred->decay_shift;
            const uint32_t promote_threshold = pred->promote_threshold;
            const uint32_t min_age_steps = pred->min_age_steps;
            const uint32_t cooldown_steps = pred->cooldown_steps;
            const uint32_t host_limit =
                (pred->num_host_experts > 0u && pred->num_host_experts <= num_experts)
                    ? pred->num_host_experts
                    : num_experts;

            for (uint32_t e = threadIdx.x; e < host_limit; e += blockDim.x) {
                const int32_t slot = s_slot[e];
                const uint32_t hits = s_hits[e];
                uint32_t b = born[e];
                uint32_t sc = scores[e];

                if (slot >= 0) {
                    uint32_t ls = last_seen[e];
                    if ((b & 0x80000000u) == 0u) {
                        // Physical installation into HBM just became visible: stamp birth step and initial recency.
                        b = 0x80000000u | (step_now & 0x7FFFFFFFu);
                        born[e] = b;
                        ls = step_now;
                        last_seen[e] = ls;
                    } else if (hits > 0u) {
                        ls = step_now;
                        last_seen[e] = ls;
                    }
                    if (sc != 0u) {
                        scores[e] = 0u;
                    }
                    if (eval_swap && hits == 0u) {
                        const uint32_t age = (step_now - (b & 0x7FFFFFFFu)) & 0x7FFFFFFFu;
                        if (age >= min_age_steps) {
                            const uint64_t vk = (static_cast<uint64_t>(ls) << 32) | static_cast<uint64_t>(e);
                            if (vk < my_vic_key) my_vic_key = vk;
                        }
                    }
                } else {
                    if ((b & 0x80000000u) != 0u) {
                        // Physical eviction from HBM just became visible: clear resident flag.
                        born[e] = 0u;
                    }
                    if (decay_shift > 0u && decay_shift < 31u) {
                        sc = sc - (sc >> decay_shift);
                    }
                    if (hits > 0u) {
                        const uint32_t capped = hits < 16u ? hits : 16u;
                        const uint32_t boost = capped << 8;
                        sc = (sc + boost < 0x7FFFFFFFu) ? (sc + boost) : 0x7FFFFFFFu;
                    }
                    scores[e] = sc;
                    if (eval_swap && hits > 0u && sc >= promote_threshold &&
                        (step_now - cooldown[e]) >= cooldown_steps) {
                        const uint64_t ck = (static_cast<uint64_t>(sc) << 32) | static_cast<uint64_t>(e + 1u);
                        if (ck > my_cold_key) my_cold_key = ck;
                    }
                }
            }

            // 8-warp parallel reduction: each of the 8 warps produces its top cold candidate & best victim in 5 shuffles.
            __shared__ uint64_t s_warp_cold[8];
            __shared__ uint64_t s_warp_vic[8];
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5;
            if (eval_swap) {
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const uint64_t other_c = __shfl_down_sync(0xFFFFFFFFu, my_cold_key, offset);
                    if (other_c > my_cold_key) my_cold_key = other_c;
                    const uint64_t other_v = __shfl_down_sync(0xFFFFFFFFu, my_vic_key, offset);
                    if (other_v < my_vic_key) my_vic_key = other_v;
                }
                if (lane == 0u && warp < 8u) {
                    s_warp_cold[warp] = my_cold_key;
                    s_warp_vic[warp] = my_vic_key;
                }
            }
            __syncthreads();

            // 5. Thread 0 inspects the 8 warp winners and updates HBM counters without host-bus traffic.
            if (threadIdx.x == 0) {
                pred->step_clock = step_now;
                if (queue_hdr != nullptr) {
                    const uint64_t tot_hits = queue_hdr->predict_hits + static_cast<uint64_t>(s_hbm_hits);
                    const uint64_t tot_misses = queue_hdr->predict_misses + static_cast<uint64_t>(s_host_misses);
                    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                 :: "l"(&queue_hdr->predict_hits), "l"(tot_hits) : "memory");
                    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                 :: "l"(&queue_hdr->predict_misses), "l"(tot_misses) : "memory");
                }

                if (eval_swap && queue_hdr != nullptr && queue_ring != nullptr) {
                    const uint32_t k_max = pred->max_enqueues < 4u ? pred->max_enqueues : 4u;
                    uint64_t top_cold[4] = {0ull, 0ull, 0ull, 0ull};
                    uint64_t top_vic[4] = {~0ull, ~0ull, ~0ull, ~0ull};
                    uint32_t n_cold = 0u;
                    uint32_t n_vic = 0u;

                    #pragma unroll
                    for (uint32_t w = 0u; w < 8u; ++w) {
                        const uint64_t ck = s_warp_cold[w];
                        if (ck != 0ull) {
                            uint32_t pos = n_cold;
                            if (pos < k_max) {
                                n_cold += 1u;
                            } else if (ck <= top_cold[k_max - 1u]) {
                                pos = 4u;
                            } else {
                                pos = k_max - 1u;
                            }
                            if (pos < 4u) {
                                while (pos > 0u && ck > top_cold[pos - 1u]) {
                                    top_cold[pos] = top_cold[pos - 1u];
                                    pos -= 1u;
                                }
                                top_cold[pos] = ck;
                            }
                        }
                        const uint64_t vk = s_warp_vic[w];
                        if (vk != ~0ull) {
                            uint32_t pos = n_vic;
                            if (pos < k_max) {
                                n_vic += 1u;
                            } else if (vk >= top_vic[k_max - 1u]) {
                                pos = 4u;
                            } else {
                                pos = k_max - 1u;
                            }
                            if (pos < 4u) {
                                while (pos > 0u && vk < top_vic[pos - 1u]) {
                                    top_vic[pos] = top_vic[pos - 1u];
                                    pos -= 1u;
                                }
                                top_vic[pos] = vk;
                            }
                        }
                    }

                    const uint32_t pairs = n_cold < n_vic ? n_cold : n_vic;
                    int32_t enq_cold[4];
                    int32_t enq_vic[4];
                    uint32_t n_valid = pairs;
                    for (uint32_t p = 0u; p < pairs; ++p) {
                        enq_cold[p] = static_cast<int32_t>((top_cold[p] & 0xFFFFFFFFull) - 1ull);
                        enq_vic[p] = static_cast<int32_t>(top_vic[p] & 0xFFFFFFFFull);
                    }

                    uint32_t base_idx = 0u;
                    while (n_valid > 0u && !mpsc_try_reserve(queue_hdr, n_valid, base_idx)) {
                        n_valid -= 1u;
                    }
                    if (n_valid > 0u) {
                        const uint32_t mask = queue_hdr->mask;
                        for (uint32_t p = 0u; p < n_valid; ++p) {
                            MoverRequest* rslot = queue_ring + ((base_idx + p) & mask);
                            rslot->layer_idx = pred->layer_idx;
                            rslot->expert_in = enq_cold[p];
                            rslot->victim_expert = enq_vic[p];
                            cooldown[enq_cold[p]] = step_now;
                        }
                        pred->last_enqueue_step = step_now;
                        asm volatile("fence.acq_rel.gpu;" ::: "memory");
                        for (uint32_t p = 0u; p < n_valid; ++p) {
                            MoverRequest* rslot = queue_ring + ((base_idx + p) & mask);
                            asm volatile("st.relaxed.gpu.global.u32 [%0], %1;"
                                         :: "l"(&rslot->seq), "r"(base_idx + p + 1u) : "memory");
                        }
                        const uint64_t tot_enq = queue_hdr->predict_enqueues + static_cast<uint64_t>(n_valid);
                        asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                     :: "l"(&queue_hdr->predict_enqueues), "l"(tot_enq) : "memory");
                    }
                }
            }
        }
    }

    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
    }
}

__global__ void sm103_enqueue_swap_kernel(
    MoverQueueHeader* __restrict__ hdr,
    MoverRequest* __restrict__ ring,
    uint32_t layer_idx,
    int32_t expert_in,
    int32_t victim_expert
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t idx = 0u;
    if (!mpsc_try_reserve(hdr, 1u, idx)) {
        // Queue full: drop best-effort prefetch request rather than stalling compute.
        return;
    }
    MoverRequest* slot = ring + (idx & hdr->mask);
    slot->layer_idx = layer_idx;
    slot->expert_in = expert_in;
    slot->victim_expert = victim_expert;
    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                 :: "l"(&slot->seq), "r"(idx + 1u) : "memory");
}

__global__ void sm103_enqueue_swap_batch_kernel(
    MoverQueueHeader* __restrict__ hdr,
    MoverRequest* __restrict__ ring,
    const uint32_t* __restrict__ layers,
    const int32_t* __restrict__ experts_in,
    const int32_t* __restrict__ victims,
    uint32_t count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = 0u;
        if (!mpsc_try_reserve(hdr, 1u, idx)) {
            break;
        }
        MoverRequest* slot = ring + (idx & hdr->mask);
        slot->layer_idx = layers[i];
        slot->expert_in = experts_in[i];
        slot->victim_expert = victims[i];
        asm volatile("st.release.gpu.global.u32 [%0], %1;"
                     :: "l"(&slot->seq), "r"(idx + 1u) : "memory");
    }
}

__global__ void sm103_persistent_mover_kernel(
    LayerMoverDesc* __restrict__ layers,
    uint32_t num_layers,
    MoverQueueHeader* __restrict__ hdr,
    MoverRequest* __restrict__ ring,
    volatile uint32_t* __restrict__ stop_flag,
    uint64_t* __restrict__ metrics,
    uint32_t metrics_count
) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t mbar[kTmaStages];
    __shared__ uint16_t s_pred_hits[kMaxPredictExperts];
    uint8_t* buf[kTmaStages];
    #pragma unroll
    for (uint32_t s = 0u; s < kTmaStages; ++s) {
        buf[s] = smem + s * kTmaChunkBytes;
    }

    if (threadIdx.x == 0) {
        #pragma unroll
        for (uint32_t s = 0u; s < kTmaStages; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[s]))));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    uint32_t phase[kTmaStages];
    #pragma unroll
    for (uint32_t s = 0u; s < kTmaStages; ++s) {
        phase[s] = 0u;
    }
    uint32_t local_sense = 0u;
    uint64_t swaps_completed = 0ull;
    uint64_t bytes_copied = 0ull;
    uint64_t rcu_waits = 0ull;
    uint64_t skipped_resident = 0ull;
    uint64_t mirrored_hits = 0ull;
    uint64_t mirrored_misses = 0ull;
    uint64_t mirrored_enqueues = 0ull;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("st.release.gpu.global.u32 [%0], 1;"
                     :: "l"(&hdr->worker_running) : "memory");
    }

    auto mirror_predictor_metrics = [&]() {
        if (metrics == nullptr || metrics_count < 8u) return;
        uint64_t h = 0ull, m = 0ull, e = 0ull;
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(h) : "l"(&hdr->predict_hits) : "memory");
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(m) : "l"(&hdr->predict_misses) : "memory");
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(e) : "l"(&hdr->predict_enqueues) : "memory");
        if (h != mirrored_hits) {
            mirrored_hits = h;
            store_metric_sys(metrics, metrics_count, 5u, h);
        }
        if (m != mirrored_misses) {
            mirrored_misses = m;
            store_metric_sys(metrics, metrics_count, 6u, m);
        }
        if (e != mirrored_enqueues) {
            mirrored_enqueues = e;
            store_metric_sys(metrics, metrics_count, 7u, e);
        }
    };

    while (true) {
        // Step 1: CTA 0 polls stop_flag, high-priority prefill stage slots, and lock-free MPSC swap ring.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            uint32_t should_stop = 0u;
            uint32_t chosen_cmd = 0u;
            uint32_t chosen_layer = 0u;
            int32_t chosen_expert_in = -1;
            int32_t chosen_spare_slot = -1;
            int32_t chosen_victim_expert = -1;
            int32_t chosen_victim_slot = -1;
            uint32_t chosen_stage_slot = 0u;
            uint32_t chosen_cold_count = 0u;

            while (true) {
                mirror_predictor_metrics();
                uint32_t stop = 0u;
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                             : "=r"(stop) : "l"(stop_flag) : "memory");
                if (stop != 0u) {
                    mirror_predictor_metrics();
                    should_stop = 1u;
                    break;
                }

                // Priority 1: prefill cold-expert staging (zero compute SM contention, pauses decode swaps).
                const uint32_t stage_next = hdr->stage_head_seq + 1u;
                const uint32_t s_idx = (stage_next - 1u) & kStageSlotMask;
                StageRequest* sslot = &hdr->stage_slots[s_idx];
                uint32_t s_req = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(s_req) : "l"(&sslot->req_seq) : "memory");
                if (s_req == stage_next) {
                    const uint32_t s_layer = sslot->layer_idx;
                    uint32_t cold_count = 0u;
                    if (s_layer < num_layers) {
                        const LayerMoverDesc& Ls = layers[s_layer];
                        const uint32_t host_limit =
                            (Ls.num_host_experts > 0u && Ls.num_host_experts <= Ls.num_experts)
                                ? Ls.num_host_experts
                                : Ls.num_experts;
                        const int32_t* live_map = reinterpret_cast<const int32_t*>(Ls.live_slot_map);
                        int32_t* mask_out = reinterpret_cast<int32_t*>(sslot->staged_mask);
                        for (uint32_t e = 0u; e < host_limit && e < kMaxPredictExperts; ++e) {
                            int32_t slot_val = -1;
                            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                                         : "=r"(slot_val) : "l"(live_map + e) : "memory");
                            if (slot_val < 0) {
                                mask_out[e] = 1;
                                hdr->stage_cold_rows[cold_count++] = static_cast<int32_t>(e);
                            } else {
                                mask_out[e] = 0;
                            }
                        }
                    }
                    sslot->staged_count = cold_count;
                    if (cold_count < sslot->min_cold || s_layer >= num_layers) {
                        asm volatile("fence.acq_rel.gpu;" ::: "memory");
                        asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                     :: "l"(&sslot->done_seq), "r"(s_req) : "memory");
                        hdr->stage_head_seq = stage_next;
                        continue;
                    }
                    chosen_cmd = 2u;
                    chosen_layer = s_layer;
                    chosen_stage_slot = s_idx;
                    chosen_cold_count = cold_count;
                    break;
                }

                // Priority 2: worker-offloaded decode predictor scoring & swap enqueue (off compute stream).
                uint32_t ptail = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(ptail) : "l"(&hdr->predict_tail) : "memory");
                const uint32_t phead = hdr->predict_head;
                if (ptail != phead) {
                    PredictEntry* pe = &hdr->predict_ring[phead & kPredictRingMask];
                    uint32_t pseq = 0u;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                 : "=r"(pseq) : "l"(&pe->seq) : "memory");
                    if (pseq == phead + 1u) {
                        const uint32_t p_layer = pe->layer_idx;
                        const uint32_t p_routes = pe->num_routes <= kMaxPredictRoutes ? pe->num_routes : kMaxPredictRoutes;
                        if (p_layer < num_layers && layers[p_layer].predictor_state != 0ull) {
                            const LayerMoverDesc& Lp = layers[p_layer];
                            PredictorLayerState* pred = reinterpret_cast<PredictorLayerState*>(Lp.predictor_state);
                            const uint32_t num_exp = Lp.num_experts <= kMaxPredictExperts ? Lp.num_experts : kMaxPredictExperts;
                            const int32_t* live_map = reinterpret_cast<const int32_t*>(Lp.live_slot_map);
                            for (uint32_t e = 0u; e < num_exp; ++e) {
                                s_pred_hits[e] = 0u;
                            }
                            uint32_t p_hbm_hits = 0u;
                            uint32_t p_host_misses = 0u;
                            for (uint32_t r = 0u; r < p_routes; ++r) {
                                const int32_t e = static_cast<int32_t>(pe->routes[r]);
                                if (e >= 0 && static_cast<uint32_t>(e) < num_exp) {
                                    s_pred_hits[e] += 1u;
                                    if (live_map[e] >= 0) {
                                        p_hbm_hits += 1u;
                                    } else {
                                        p_host_misses += 1u;
                                    }
                                }
                            }
                            if ((p_hbm_hits + p_host_misses) > 0u) {
                                const uint32_t step_now = pred->step_clock + 1u;
                                pred->step_clock = step_now;
                                uint32_t* scores = reinterpret_cast<uint32_t*>(pred->scores);
                                uint32_t* last_seen = reinterpret_cast<uint32_t*>(pred->last_seen);
                                uint32_t* born = reinterpret_cast<uint32_t*>(pred->born);
                                uint32_t* cooldown = reinterpret_cast<uint32_t*>(pred->cooldown);

                                const bool eval_swap = (p_host_misses > 0u && pred->max_enqueues > 0u &&
                                                        (step_now - pred->last_enqueue_step) >= pred->interval_steps);
                                const uint32_t decay_shift = pred->decay_shift;
                                const uint32_t promote_threshold = pred->promote_threshold;
                                const uint32_t min_age_steps = pred->min_age_steps;
                                const uint32_t cooldown_steps = pred->cooldown_steps;
                                const uint32_t host_limit =
                                    (pred->num_host_experts > 0u && pred->num_host_experts <= num_exp)
                                        ? pred->num_host_experts
                                        : num_exp;
                                const uint32_t k_max = pred->max_enqueues < 4u ? pred->max_enqueues : 4u;
                                uint64_t top_cold[4] = {0ull, 0ull, 0ull, 0ull};
                                uint64_t top_vic[4] = {~0ull, ~0ull, ~0ull, ~0ull};
                                uint32_t n_cold = 0u;
                                uint32_t n_vic = 0u;

                                for (uint32_t e = 0u; e < host_limit; ++e) {
                                    const int32_t slot_val = live_map[e];
                                    const uint32_t hits = static_cast<uint32_t>(s_pred_hits[e]);
                                    uint32_t b = born[e];
                                    uint32_t sc = scores[e];
                                    if (slot_val >= 0) {
                                        uint32_t ls = last_seen[e];
                                        if ((b & 0x80000000u) == 0u) {
                                            b = 0x80000000u | (step_now & 0x7FFFFFFFu);
                                            born[e] = b;
                                            ls = step_now;
                                            last_seen[e] = ls;
                                        } else if (hits > 0u) {
                                            ls = step_now;
                                            last_seen[e] = ls;
                                        }
                                        if (sc != 0u) scores[e] = 0u;
                                        if (eval_swap && hits == 0u) {
                                            const uint32_t age = (step_now - (b & 0x7FFFFFFFu)) & 0x7FFFFFFFu;
                                            if (age >= min_age_steps) {
                                                const uint64_t vk = (static_cast<uint64_t>(ls) << 32) | static_cast<uint64_t>(e);
                                                uint32_t pos = n_vic;
                                                if (pos < k_max) {
                                                    n_vic += 1u;
                                                } else if (vk >= top_vic[k_max - 1u]) {
                                                    pos = 4u;
                                                } else {
                                                    pos = k_max - 1u;
                                                }
                                                if (pos < 4u) {
                                                    while (pos > 0u && vk < top_vic[pos - 1u]) {
                                                        top_vic[pos] = top_vic[pos - 1u];
                                                        pos -= 1u;
                                                    }
                                                    top_vic[pos] = vk;
                                                }
                                            }
                                        }
                                    } else {
                                        if ((b & 0x80000000u) != 0u) born[e] = 0u;
                                        if (decay_shift > 0u && decay_shift < 31u) sc = sc - (sc >> decay_shift);
                                        if (hits > 0u) {
                                            const uint32_t capped = hits < 16u ? hits : 16u;
                                            const uint32_t boost = capped << 8;
                                            sc = (sc + boost < 0x7FFFFFFFu) ? (sc + boost) : 0x7FFFFFFFu;
                                        }
                                        scores[e] = sc;
                                        if (eval_swap && hits > 0u && sc >= promote_threshold &&
                                            (step_now - cooldown[e]) >= cooldown_steps) {
                                            const uint64_t ck = (static_cast<uint64_t>(sc) << 32) | static_cast<uint64_t>(e + 1u);
                                            uint32_t pos = n_cold;
                                            if (pos < k_max) {
                                                n_cold += 1u;
                                            } else if (ck <= top_cold[k_max - 1u]) {
                                                pos = 4u;
                                            } else {
                                                pos = k_max - 1u;
                                            }
                                            if (pos < 4u) {
                                                while (pos > 0u && ck > top_cold[pos - 1u]) {
                                                    top_cold[pos] = top_cold[pos - 1u];
                                                    pos -= 1u;
                                                }
                                                top_cold[pos] = ck;
                                            }
                                        }
                                    }
                                }

                                const uint64_t tot_hits = hdr->predict_hits + static_cast<uint64_t>(p_hbm_hits);
                                const uint64_t tot_misses = hdr->predict_misses + static_cast<uint64_t>(p_host_misses);
                                asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                             :: "l"(&hdr->predict_hits), "l"(tot_hits) : "memory");
                                asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                             :: "l"(&hdr->predict_misses), "l"(tot_misses) : "memory");

                                const uint32_t pairs = n_cold < n_vic ? n_cold : n_vic;
                                if (eval_swap && pairs > 0u) {
                                    uint32_t n_valid = pairs;
                                    uint32_t base_idx = 0u;
                                    while (n_valid > 0u && !mpsc_try_reserve(hdr, n_valid, base_idx)) {
                                        n_valid -= 1u;
                                    }
                                    if (n_valid > 0u) {
                                        const uint32_t mask = hdr->mask;
                                        for (uint32_t p = 0u; p < n_valid; ++p) {
                                            const int32_t cold_e = static_cast<int32_t>((top_cold[p] & 0xFFFFFFFFull) - 1ull);
                                            const int32_t vic_e = static_cast<int32_t>(top_vic[p] & 0xFFFFFFFFull);
                                            MoverRequest* rslot = ring + ((base_idx + p) & mask);
                                            rslot->layer_idx = p_layer;
                                            rslot->expert_in = cold_e;
                                            rslot->victim_expert = vic_e;
                                            cooldown[cold_e] = step_now;
                                        }
                                        pred->last_enqueue_step = step_now;
                                        asm volatile("fence.acq_rel.gpu;" ::: "memory");
                                        for (uint32_t p = 0u; p < n_valid; ++p) {
                                            MoverRequest* rslot = ring + ((base_idx + p) & mask);
                                            asm volatile("st.relaxed.gpu.global.u32 [%0], %1;"
                                                         :: "l"(&rslot->seq), "r"(base_idx + p + 1u) : "memory");
                                        }
                                        const uint64_t tot_enq = hdr->predict_enqueues + static_cast<uint64_t>(n_valid);
                                        asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                                     :: "l"(&hdr->predict_enqueues), "l"(tot_enq) : "memory");
                                    }
                                }
                            }
                        }
                        asm volatile("st.release.gpu.global.u32 [%0], %1;"
                                     :: "l"(&hdr->predict_head), "r"(phead + 1u) : "memory");
                        continue;
                    }
                }

                // Priority 3: background decode expert swaps (paused while any prefill stage is in flight).
                uint32_t in_flight = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(in_flight) : "l"(&hdr->stage_in_flight) : "memory");
                if (in_flight != 0u) {
                    asm volatile("nanosleep.u32 32;");
                    continue;
                }

                uint32_t tail = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(tail) : "l"(&hdr->tail) : "memory");
                const uint32_t head = hdr->head;
                if (tail == head) {
                    asm volatile("nanosleep.u32 64;");
                    continue;
                }

                MoverRequest* slot = ring + (head & hdr->mask);
                uint32_t seq = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(seq) : "l"(&slot->seq) : "memory");
                if (seq != head + 1u) {
                    // Producer reserved tail and is still writing payload fields.
                    asm volatile("nanosleep.u32 32;");
                    continue;
                }

                const uint32_t layer_idx = slot->layer_idx;
                const int32_t expert_in = slot->expert_in;
                const int32_t req_victim = slot->victim_expert;
                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                             :: "l"(&hdr->head), "r"(head + 1u) : "memory");

                if (layer_idx >= num_layers) {
                    continue;
                }
                LayerMoverDesc& L = layers[layer_idx];
                const uint32_t host_limit =
                    (L.num_host_experts > 0u && L.num_host_experts <= L.num_experts)
                        ? L.num_host_experts
                        : L.num_experts;
                if (expert_in < 0 || static_cast<uint32_t>(expert_in) >= host_limit || L.num_hbm_slots < 2u) {
                    continue;
                }
                int32_t* live_map = reinterpret_cast<int32_t*>(L.live_slot_map);
                int32_t* slot_owner = reinterpret_cast<int32_t*>(L.slot_owner);

                if (live_map[expert_in] >= 0) {
                    skipped_resident += 1ull;
                    store_metric_sys(metrics, metrics_count, 3u, skipped_resident);
                    continue;
                }

                // Locate an unmapped spare HBM slot.
                int32_t spare_slot = -1;
                for (uint32_t s = 0; s < L.num_hbm_slots; ++s) {
                    if (slot_owner[s] < 0) {
                        spare_slot = static_cast<int32_t>(s);
                        break;
                    }
                }
                if (spare_slot < 0) {
                    continue;
                }

                // Choose resident movable victim (< host_limit): explicit request if valid, else clock hand.
                int32_t victim_expert = -1;
                int32_t victim_slot = -1;
                if (req_victim >= 0 && static_cast<uint32_t>(req_victim) < host_limit) {
                    int32_t s = live_map[req_victim];
                    if (s >= 0 && s != spare_slot) {
                        victim_expert = req_victim;
                        victim_slot = s;
                    }
                }
                if (victim_slot < 0) {
                    uint32_t cursor = L.victim_cursor % L.num_hbm_slots;
                    for (uint32_t step = 0; step < L.num_hbm_slots; ++step) {
                        const uint32_t cand = (cursor + step) % L.num_hbm_slots;
                        const int32_t owner = slot_owner[cand];
                        if (owner >= 0 && static_cast<uint32_t>(owner) < host_limit && static_cast<int32_t>(cand) != spare_slot) {
                            victim_expert = owner;
                            victim_slot = static_cast<int32_t>(cand);
                            L.victim_cursor = (cand + 1u) % L.num_hbm_slots;
                            break;
                        }
                    }
                }
                if (victim_slot < 0) {
                    continue;
                }

                chosen_cmd = 1u;
                chosen_layer = layer_idx;
                chosen_expert_in = expert_in;
                chosen_spare_slot = spare_slot;
                chosen_victim_expert = victim_expert;
                chosen_victim_slot = victim_slot;
                break;
            }

            hdr->active_stop = should_stop;
            hdr->active_cmd = chosen_cmd;
            hdr->active_layer = chosen_layer;
            hdr->active_expert_in = chosen_expert_in;
            hdr->active_spare_slot = chosen_spare_slot;
            hdr->active_victim_expert = chosen_victim_expert;
            hdr->active_victim_slot = chosen_victim_slot;
            hdr->active_stage_slot = chosen_stage_slot;
            hdr->active_cold_count = chosen_cold_count;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }

        // Step 2: Inter-CTA Barrier #1 broadcasts command (or stop) to all worker CTAs.
        worker_grid_barrier(hdr, local_sense);
        uint32_t stop_cmd = 0u;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(stop_cmd) : "l"(&hdr->active_stop) : "memory");
        if (stop_cmd != 0u) {
            break;
        }

        uint32_t active_cmd = 0u;
        uint32_t layer_idx = 0u;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(active_cmd) : "l"(&hdr->active_cmd) : "memory");
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(layer_idx) : "l"(&hdr->active_layer) : "memory");
        const LayerMoverDesc& L = layers[layer_idx];

        // Step 3: Parallel 8-SM 6-stage pipelined 1D TMA copy across all 4 MXFP4 planes.
        // Keeps 4-5 NVLink-C2C 32 KiB loads in flight per SM while overlapping HBM stores (~353 GB/s).
        if (threadIdx.x == 0) {
            uint32_t plane_chunks[4];
            uint32_t plane_prefix[5];
            plane_prefix[0] = 0u;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint64_t rb = L.plane_row_bytes[p];
                plane_chunks[p] = rb == 0ull ? 0u : static_cast<uint32_t>((rb + kTmaChunkBytes - 1ull) / kTmaChunkBytes);
                plane_prefix[p + 1] = plane_prefix[p] + plane_chunks[p];
            }
            const uint32_t chunks_per_expert = plane_prefix[4];
            uint32_t num_rows_copy = 1u;
            int32_t swap_src_row = 0;
            int32_t swap_dst_row = 0;
            uint64_t dst_bases[4];
            if (active_cmd == 2u) {
                uint32_t s_idx = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(s_idx) : "l"(&hdr->active_stage_slot) : "memory");
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(num_rows_copy) : "l"(&hdr->active_cold_count) : "memory");
                const StageRequest& sslot = hdr->stage_slots[s_idx];
                #pragma unroll
                for (int p = 0; p < 4; ++p) dst_bases[p] = sslot.dst_planes[p];
            } else {
                asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(swap_src_row) : "l"(&hdr->active_expert_in) : "memory");
                asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(swap_dst_row) : "l"(&hdr->active_spare_slot) : "memory");
                #pragma unroll
                for (int p = 0; p < 4; ++p) dst_bases[p] = L.hbm_planes[p];
            }
            const uint32_t total_chunks = num_rows_copy * chunks_per_expert;

            uint8_t* pending_dst[kTmaStages];
            uint32_t pending_bytes[kTmaStages];
            uint32_t iter = 0u;
            uint32_t st = 0u;
            uint32_t store_st = 0u;

            auto issue_store = [&](uint32_t stage) {
                const uint32_t b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[stage]));
                uint32_t done = 0u;
                do {
                    asm volatile(
                        "{ .reg .pred p; "
                        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
                        "selp.b32 %0, 1, 0, p; }"
                        : "=r"(done) : "r"(b_addr), "r"(phase[stage]) : "memory"
                    );
                } while (!done);
                phase[stage] ^= 1u;
                asm volatile("fence.proxy.async.shared::cta;");
                const uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(buf[stage]));
                asm volatile(
                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                    "cp.async.bulk.commit_group;\n"
                    :: "l"(pending_dst[stage]), "r"(s_addr), "r"(pending_bytes[stage]) : "memory"
                );
            };

            for (uint32_t gch = blockIdx.x; gch < total_chunks; gch += gridDim.x) {
                uint32_t expert_ch = gch;
                int32_t src_row = swap_src_row;
                int32_t dst_row = swap_dst_row;
                if (active_cmd == 2u) {
                    const uint32_t r_idx = gch / chunks_per_expert;
                    expert_ch = gch - r_idx * chunks_per_expert;
                    const int32_t row = hdr->stage_cold_rows[r_idx];
                    src_row = row;
                    dst_row = row;
                }
                int p = 0;
                if (expert_ch >= plane_prefix[2]) {
                    p = (expert_ch >= plane_prefix[3]) ? 3 : 2;
                } else {
                    p = (expert_ch >= plane_prefix[1]) ? 1 : 0;
                }
                const uint32_t local_ch = expert_ch - plane_prefix[p];
                const uint64_t row_bytes = L.plane_row_bytes[p];
                const uint64_t off = static_cast<uint64_t>(local_ch) * kTmaChunkBytes;
                const uint64_t rem = row_bytes - off;
                const uint32_t chunk_bytes = static_cast<uint32_t>(rem < kTmaChunkBytes ? rem : kTmaChunkBytes);
                const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(L.host_planes[p]) +
                                         static_cast<uint64_t>(src_row) * row_bytes + off;
                uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_bases[p]) +
                                   static_cast<uint64_t>(dst_row) * row_bytes + off;

                if (iter >= kTmaStages) {
                    // At most 1 committed bulk_group (iter - 1) may still read SMEM; iter - 6 (same stage st) is done.
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                const uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(buf[st]));
                const uint32_t b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[st]));
                pending_dst[st] = dst_ptr;
                pending_bytes[st] = chunk_bytes;
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%2], [%3], %1, [%0];\n"
                    :: "r"(b_addr), "r"(chunk_bytes), "r"(s_addr), "l"(src_ptr) : "memory"
                );
                if (iter >= kTmaStages - 2u) {
                    issue_store(store_st);
                    store_st = (store_st + 1u == kTmaStages) ? 0u : (store_st + 1u);
                }
                st = (st + 1u == kTmaStages) ? 0u : (st + 1u);
                iter += 1u;
            }
            // Drain remaining in-flight loads and emit their HBM stores.
            const uint32_t drain_count = iter < (kTmaStages - 2u) ? iter : (kTmaStages - 2u);
            for (uint32_t d = 0u; d < drain_count; ++d) {
                issue_store(store_st);
                store_st = (store_st + 1u == kTmaStages) ? 0u : (store_st + 1u);
            }
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }

        // Step 4: Inter-CTA Barrier #2 ensures all 4 planes are committed in HBM.
        worker_grid_barrier(hdr, local_sense);

        // Step 5: CTA 0 publishes completion (prefill stage done_seq or decode swap + Dekker RCU).
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (active_cmd == 2u) {
                const uint32_t s_idx = hdr->active_stage_slot;
                StageRequest* sslot = &hdr->stage_slots[s_idx];
                const uint32_t done_val = sslot->req_seq;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                             :: "l"(&sslot->done_seq), "r"(done_val) : "memory");
                hdr->stage_head_seq = done_val;
            } else {
                const int32_t expert_in = hdr->active_expert_in;
                const int32_t spare_slot = hdr->active_spare_slot;
                const int32_t victim_expert = hdr->active_victim_expert;
                const int32_t victim_slot = hdr->active_victim_slot;
                int32_t* live_map = reinterpret_cast<int32_t*>(L.live_slot_map);
                int32_t* slot_owner = reinterpret_cast<int32_t*>(L.slot_owner);
                uint32_t* epoch_ptr = reinterpret_cast<uint32_t*>(L.layer_epoch);

                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                asm volatile("st.release.gpu.global.s32 [%0], %1;"
                             :: "l"(live_map + victim_expert), "r"(-1) : "memory");
                asm volatile("st.release.gpu.global.s32 [%0], %1;"
                             :: "l"(live_map + expert_in), "r"(spare_slot) : "memory");
                // Dekker half 2: order unmap of victim_expert before reading layer_epoch.
                asm volatile("fence.sc.gpu;" ::: "memory");

                uint32_t ep = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(ep) : "l"(epoch_ptr) : "memory");
                if (ep & 1u) {
                    rcu_waits += 1ull;
                    uint32_t cur = ep;
                    while (cur == ep) {
                        uint32_t stop = 0u;
                        asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                                     : "=r"(stop) : "l"(stop_flag) : "memory");
                        if (stop != 0u) break;
                        asm volatile("nanosleep.u32 32;");
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                     : "=r"(cur) : "l"(epoch_ptr) : "memory");
                    }
                }

                // Victim slot is now guaranteed unused by any active MegaMoE invocation.
                slot_owner[spare_slot] = expert_in;
                slot_owner[victim_slot] = -1;

                swaps_completed += 1ull;
                bytes_copied += L.plane_row_bytes[0] + L.plane_row_bytes[1] + L.plane_row_bytes[2] + L.plane_row_bytes[3];
                store_metric_sys(metrics, metrics_count, 0u, swaps_completed);
                store_metric_sys(metrics, metrics_count, 1u, bytes_copied);
                store_metric_sys(metrics, metrics_count, 2u, rcu_waits);
                store_metric_sys(metrics, metrics_count, 4u,
                                 (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_in));
            }
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("st.release.gpu.global.u32 [%0], 0;"
                     :: "l"(&hdr->worker_running) : "memory");
    }
}

void launch_layer_enter(
    torch::Tensor live_slot_map,
    torch::Tensor control,
    torch::Tensor layer_epoch,
    int num_experts
) {
    TORCH_CHECK(live_slot_map.is_cuda() && live_slot_map.scalar_type() == torch::kInt && live_slot_map.is_contiguous());
    TORCH_CHECK(control.is_cuda() && control.scalar_type() == torch::kInt && control.is_contiguous());
    TORCH_CHECK(layer_epoch.is_cuda() && layer_epoch.is_contiguous() && layer_epoch.numel() >= 1);
    TORCH_CHECK(num_experts > 0 && live_slot_map.numel() >= num_experts && control.numel() >= 16 + num_experts);
    const c10::cuda::CUDAGuard device_guard(live_slot_map.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_layer_enter_kernel<<<1, 256, 0, stream>>>(
        live_slot_map.data_ptr<int32_t>(),
        control.data_ptr<int32_t>() + 16,
        reinterpret_cast<uint32_t*>(layer_epoch.data_ptr()),
        static_cast<uint32_t>(num_experts)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_layer_enter_predict(
    torch::Tensor live_slot_map,
    torch::Tensor control,
    torch::Tensor layer_epoch,
    torch::Tensor topk_ids,
    int num_rows,
    int num_topk,
    int num_experts,
    torch::Tensor predictor_state,
    torch::Tensor queue_header,
    torch::Tensor queue_ring,
    uint64_t metrics_ptr,
    int metrics_count
) {
    TORCH_CHECK(live_slot_map.is_cuda() && live_slot_map.scalar_type() == torch::kInt && live_slot_map.is_contiguous());
    TORCH_CHECK(control.is_cuda() && control.scalar_type() == torch::kInt && control.is_contiguous());
    TORCH_CHECK(layer_epoch.is_cuda() && layer_epoch.is_contiguous() && layer_epoch.numel() >= 1);
    TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kLong && topk_ids.is_contiguous());
    TORCH_CHECK(predictor_state.is_cuda() && predictor_state.is_contiguous());
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_ring.is_cuda() && queue_ring.is_contiguous());
    TORCH_CHECK(num_rows > 0 && num_topk > 0 && num_experts > 0);
    TORCH_CHECK(live_slot_map.numel() >= num_experts && control.numel() >= 16 + num_experts);
    TORCH_CHECK(predictor_state.numel() * predictor_state.element_size() >= static_cast<int64_t>(sizeof(PredictorLayerState)));
    const c10::cuda::CUDAGuard device_guard(live_slot_map.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(1u, 1u, 1u);
    config.blockDim = dim3(256u, 1u, 1u);
    config.dynamicSmemBytes = 0u;
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attr;
    config.numAttrs = 1;
    cudaError_t err = cudaLaunchKernelEx(
        &config,
        sm103_layer_enter_predict_kernel,
        live_slot_map.data_ptr<int32_t>(),
        control.data_ptr<int32_t>() + 16,
        reinterpret_cast<uint32_t*>(layer_epoch.data_ptr()),
        topk_ids.data_ptr<int64_t>(),
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(num_topk),
        static_cast<uint32_t>(num_experts),
        reinterpret_cast<PredictorLayerState*>(predictor_state.data_ptr()),
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<MoverRequest*>(queue_ring.data_ptr()),
        reinterpret_cast<uint64_t*>(metrics_ptr),
        static_cast<uint32_t>(metrics_count)
    );
    TORCH_CHECK(err == cudaSuccess, "launch_layer_enter_predict failed: ", cudaGetErrorString(err));
}

void launch_stage_request(
    torch::Tensor queue_header,
    int layer_idx,
    torch::Tensor plane0,
    torch::Tensor plane1,
    torch::Tensor plane2,
    torch::Tensor plane3,
    torch::Tensor staged_mask,
    torch::Tensor req_seq_out,
    int num_host_experts,
    int min_cold
) {
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_header.numel() * queue_header.element_size() >= static_cast<int64_t>(sizeof(MoverQueueHeader)));
    TORCH_CHECK(plane0.is_cuda() && plane1.is_cuda() && plane2.is_cuda() && plane3.is_cuda());
    TORCH_CHECK(staged_mask.is_cuda() && staged_mask.scalar_type() == torch::kInt && staged_mask.is_contiguous());
    TORCH_CHECK(req_seq_out.is_cuda() && req_seq_out.scalar_type() == torch::kInt && req_seq_out.is_contiguous() && req_seq_out.numel() >= 1);
    TORCH_CHECK(layer_idx >= 0 && num_host_experts > 0 && staged_mask.numel() >= num_host_experts && min_cold >= 1);
    const c10::cuda::CUDAGuard device_guard(queue_header.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_stage_request_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        static_cast<uint32_t>(layer_idx),
        reinterpret_cast<uint64_t>(plane0.data_ptr()),
        reinterpret_cast<uint64_t>(plane1.data_ptr()),
        reinterpret_cast<uint64_t>(plane2.data_ptr()),
        reinterpret_cast<uint64_t>(plane3.data_ptr()),
        staged_mask.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(req_seq_out.data_ptr()),
        static_cast<uint32_t>(num_host_experts),
        static_cast<uint32_t>(min_cold)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void launch_layer_enter_staged_impl(
    torch::Tensor live_slot_map,
    torch::Tensor base_control,
    torch::Tensor staged_control,
    torch::Tensor layer_epoch,
    torch::Tensor staged_mask,
    torch::Tensor plane0,
    torch::Tensor plane1,
    torch::Tensor plane2,
    torch::Tensor plane3,
    int num_experts,
    int num_host_experts,
    int min_cold,
    MoverQueueHeader* queue_hdr,
    const uint32_t* req_seq_ptr
) {
    TORCH_CHECK(live_slot_map.is_cuda() && live_slot_map.scalar_type() == torch::kInt && live_slot_map.is_contiguous());
    TORCH_CHECK(base_control.is_cuda() && base_control.scalar_type() == torch::kInt && base_control.is_contiguous());
    TORCH_CHECK(staged_control.is_cuda() && staged_control.scalar_type() == torch::kInt && staged_control.is_contiguous());
    TORCH_CHECK(layer_epoch.is_cuda() && layer_epoch.is_contiguous() && layer_epoch.numel() >= 1);
    TORCH_CHECK(staged_mask.is_cuda() && staged_mask.scalar_type() == torch::kInt && staged_mask.is_contiguous());
    TORCH_CHECK(plane0.is_cuda() && plane1.is_cuda() && plane2.is_cuda() && plane3.is_cuda());
    TORCH_CHECK(num_experts > 0 && num_host_experts > 0 && num_host_experts <= num_experts && min_cold >= 1);
    TORCH_CHECK(live_slot_map.numel() >= num_experts && base_control.numel() >= 16 + num_experts &&
                staged_control.numel() >= 16 + num_experts && staged_mask.numel() >= num_host_experts);
    const uint32_t has_host_lane_counters = staged_control.numel() >= 16 + num_experts + 2 ? 1u : 0u;
    const c10::cuda::CUDAGuard device_guard(live_slot_map.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_layer_enter_staged_kernel<<<1, 256, 0, stream>>>(
        live_slot_map.data_ptr<int32_t>(),
        base_control.data_ptr<int32_t>(),
        staged_control.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(layer_epoch.data_ptr()),
        staged_mask.data_ptr<int32_t>(),
        reinterpret_cast<uint64_t>(plane0.data_ptr()),
        reinterpret_cast<uint64_t>(plane1.data_ptr()),
        reinterpret_cast<uint64_t>(plane2.data_ptr()),
        reinterpret_cast<uint64_t>(plane3.data_ptr()),
        static_cast<uint32_t>(num_experts),
        static_cast<uint32_t>(num_host_experts),
        static_cast<uint32_t>(min_cold),
        has_host_lane_counters,
        queue_hdr,
        req_seq_ptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_layer_enter_staged(
    torch::Tensor live_slot_map,
    torch::Tensor base_control,
    torch::Tensor staged_control,
    torch::Tensor layer_epoch,
    torch::Tensor staged_mask,
    torch::Tensor plane0,
    torch::Tensor plane1,
    torch::Tensor plane2,
    torch::Tensor plane3,
    int num_experts,
    int num_host_experts,
    int min_cold
) {
    launch_layer_enter_staged_impl(
        live_slot_map, base_control, staged_control, layer_epoch, staged_mask,
        plane0, plane1, plane2, plane3, num_experts, num_host_experts, min_cold,
        nullptr, nullptr
    );
}

void launch_layer_enter_staged_worker(
    torch::Tensor live_slot_map,
    torch::Tensor base_control,
    torch::Tensor staged_control,
    torch::Tensor layer_epoch,
    torch::Tensor staged_mask,
    torch::Tensor plane0,
    torch::Tensor plane1,
    torch::Tensor plane2,
    torch::Tensor plane3,
    int num_experts,
    int num_host_experts,
    int min_cold,
    torch::Tensor queue_header,
    torch::Tensor req_seq
) {
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_header.numel() * queue_header.element_size() >= static_cast<int64_t>(sizeof(MoverQueueHeader)));
    TORCH_CHECK(req_seq.is_cuda() && req_seq.scalar_type() == torch::kInt && req_seq.is_contiguous() && req_seq.numel() >= 1);
    launch_layer_enter_staged_impl(
        live_slot_map, base_control, staged_control, layer_epoch, staged_mask,
        plane0, plane1, plane2, plane3, num_experts, num_host_experts, min_cold,
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<const uint32_t*>(req_seq.data_ptr())
    );
}

void launch_enqueue_swap(
    torch::Tensor queue_header,
    torch::Tensor queue_ring,
    int layer_idx,
    int expert_in,
    int victim_expert
) {
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_ring.is_cuda() && queue_ring.is_contiguous());
    const c10::cuda::CUDAGuard device_guard(queue_header.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_enqueue_swap_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<MoverRequest*>(queue_ring.data_ptr()),
        static_cast<uint32_t>(layer_idx),
        static_cast<int32_t>(expert_in),
        static_cast<int32_t>(victim_expert)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_enqueue_swap_batch(
    torch::Tensor queue_header,
    torch::Tensor queue_ring,
    torch::Tensor layers,
    torch::Tensor experts_in,
    torch::Tensor victims
) {
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_ring.is_cuda() && queue_ring.is_contiguous());
    TORCH_CHECK(layers.is_cuda() && layers.scalar_type() == torch::kInt && layers.is_contiguous());
    TORCH_CHECK(experts_in.is_cuda() && experts_in.scalar_type() == torch::kInt && experts_in.is_contiguous());
    TORCH_CHECK(victims.is_cuda() && victims.scalar_type() == torch::kInt && victims.is_contiguous());
    TORCH_CHECK(layers.numel() == experts_in.numel() && layers.numel() == victims.numel());
    const c10::cuda::CUDAGuard device_guard(queue_header.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_enqueue_swap_batch_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<MoverRequest*>(queue_ring.data_ptr()),
        reinterpret_cast<const uint32_t*>(layers.data_ptr()),
        experts_in.data_ptr<int32_t>(),
        victims.data_ptr<int32_t>(),
        static_cast<uint32_t>(layers.numel())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void configure_persistent_mover() {
    const size_t smem_bytes = kTmaStages * kTmaChunkBytes;
    cudaError_t err = cudaFuncSetAttribute(
        sm103_persistent_mover_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)
    );
    TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));
}

void launch_persistent_mover(
    torch::Tensor layer_descs,
    int num_layers,
    torch::Tensor queue_header,
    torch::Tensor queue_ring,
    uint64_t stop_flag_ptr,
    uint64_t metrics_ptr,
    int metrics_count,
    int worker_sm_count,
    uint64_t stream_handle
) {
    TORCH_CHECK(layer_descs.is_cuda() && layer_descs.is_contiguous());
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_ring.is_cuda() && queue_ring.is_contiguous());
    TORCH_CHECK(num_layers > 0 && worker_sm_count > 0 && stop_flag_ptr != 0);
    TORCH_CHECK(layer_descs.numel() * layer_descs.element_size() >= static_cast<int64_t>(num_layers) * static_cast<int64_t>(sizeof(LayerMoverDesc)));
    TORCH_CHECK(queue_header.numel() * queue_header.element_size() >= static_cast<int64_t>(sizeof(MoverQueueHeader)));
    const c10::cuda::CUDAGuard device_guard(layer_descs.device());
    cudaStream_t stream = stream_handle != 0
        ? reinterpret_cast<cudaStream_t>(stream_handle)
        : c10::cuda::getCurrentCUDAStream().stream();
    const size_t smem_bytes = kTmaStages * kTmaChunkBytes;
    sm103_persistent_mover_kernel<<<static_cast<uint32_t>(worker_sm_count), 32, smem_bytes, stream>>>(
        reinterpret_cast<LayerMoverDesc*>(layer_descs.data_ptr()),
        static_cast<uint32_t>(num_layers),
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<MoverRequest*>(queue_ring.data_ptr()),
        reinterpret_cast<volatile uint32_t*>(stop_flag_ptr),
        reinterpret_cast<uint64_t*>(metrics_ptr),
        static_cast<uint32_t>(metrics_count)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int layer_desc_bytes() { return static_cast<int>(sizeof(LayerMoverDesc)); }
int queue_header_bytes() { return static_cast<int>(sizeof(MoverQueueHeader)); }
int queue_request_bytes() { return static_cast<int>(sizeof(MoverRequest)); }
int predictor_state_bytes() { return static_cast<int>(sizeof(PredictorLayerState)); }

void store_stop_flag_release(uint64_t stop_flag_ptr, uint32_t value) {
    TORCH_CHECK(stop_flag_ptr != 0);
    __atomic_store_n(reinterpret_cast<uint32_t*>(stop_flag_ptr), value, __ATOMIC_RELEASE);
}

} // namespace flare::sm103_mover

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("configure_persistent_mover", &flare::sm103_mover::configure_persistent_mover);
    m.def("launch_layer_enter", &flare::sm103_mover::launch_layer_enter);
    m.def("launch_layer_enter_predict", &flare::sm103_mover::launch_layer_enter_predict);
    m.def("launch_stage_request", &flare::sm103_mover::launch_stage_request);
    m.def("launch_layer_enter_staged", &flare::sm103_mover::launch_layer_enter_staged);
    m.def("launch_layer_enter_staged_worker", &flare::sm103_mover::launch_layer_enter_staged_worker);
    m.def("launch_enqueue_swap", &flare::sm103_mover::launch_enqueue_swap);
    m.def("launch_enqueue_swap_batch", &flare::sm103_mover::launch_enqueue_swap_batch);
    m.def("launch_persistent_mover", &flare::sm103_mover::launch_persistent_mover);
    m.def("layer_desc_bytes", &flare::sm103_mover::layer_desc_bytes);
    m.def("queue_header_bytes", &flare::sm103_mover::queue_header_bytes);
    m.def("queue_request_bytes", &flare::sm103_mover::queue_request_bytes);
    m.def("predictor_state_bytes", &flare::sm103_mover::predictor_state_bytes);
    m.def("store_stop_flag_release", &flare::sm103_mover::store_stop_flag_release);
}
