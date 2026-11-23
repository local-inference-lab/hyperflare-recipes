// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0

// High-occupancy SM103 BF16 MoE router using Split-K tensor-core GEMM and
// parallel warp top-k selection connected via programmatic dependent launch.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float sqrt_softplus(float v) {
    float sp = v > 20.0f ? v : log1pf(expf(v));
    return sqrtf(sp);
}

__device__ __forceinline__ float redux_max_f32(float v) {
    uint32_t u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0u;
    uint32_t ord = u ^ ((u & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    uint32_t best_ord = __reduce_max_sync(0xffffffff, ord);
    uint32_t best_u = best_ord ^ ((best_ord & 0x80000000u) ? 0x80000000u : 0xffffffffu);
    return __uint_as_float(best_u);
}

__device__ __forceinline__ void compare_swap_best_first(
    float& lhs_s, int& lhs_i, float& rhs_s, int& rhs_i) {
    if (rhs_s > lhs_s || (rhs_s == lhs_s && rhs_i < lhs_i)) {
        float ts = lhs_s; int ti = lhs_i;
        lhs_s = rhs_s; lhs_i = rhs_i;
        rhs_s = ts; rhs_i = ti;
    }
}

__device__ __forceinline__ uint32_t sort4_best_first(
    float& s0, float& s1, float& s2, float& s3) {
    int i0 = 0, i1 = 1, i2 = 2, i3 = 3;
    compare_swap_best_first(s0, i0, s1, i1);
    compare_swap_best_first(s2, i2, s3, i3);
    compare_swap_best_first(s0, i0, s2, i2);
    compare_swap_best_first(s1, i1, s3, i3);
    compare_swap_best_first(s1, i1, s2, i2);
    return static_cast<uint32_t>(i0 | (i1 << 2) | (i2 << 4) | (i3 << 6));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(dst)), "l"(src) : "memory");
}

template <int E, int K, int N_TILE, int K_SPLIT, int M_MAX_TILE>
constexpr size_t gemm_smem_bytes() {
    constexpr int K_STRIDE = 136;
    size_t pipe = (4 * N_TILE * K_STRIDE + 4 * M_MAX_TILE * K_STRIDE) * sizeof(__nv_bfloat16);
    size_t red = 8 * M_MAX_TILE * N_TILE * sizeof(float);
    return pipe > red ? pipe : red;
}

template <int E, int K, int N_TILE, int K_SPLIT, int M_MAX_TILE>
__global__ void __launch_bounds__(256, 2)
sm103_router_gemm_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    float* __restrict__ part_buf,
    int M
) {
    constexpr int N_TILES = E / N_TILE;
    constexpr int N_SUB = N_TILE / 16;
    constexpr int K_PER_SPLIT = K / K_SPLIT;
    constexpr int NUM_STAGES = K_PER_SPLIT / 128;
    static_assert(NUM_STAGES == 4, "router GEMM expects K_PER_SPLIT == 512");
    constexpr int M_GROUPS = M_MAX_TILE / 8;
    constexpr int K_STRIDE = 136;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int bid = blockIdx.x;
    const int n_tile = bid % N_TILES;
    const int sk = bid / N_TILES;
    const int e_base = n_tile * N_TILE;
    const int k_base = sk * K_PER_SPLIT;

    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* s_x = s_w + 4 * N_TILE * K_STRIDE;

    auto issue_stage = [&](int st) {
        const int k_off = k_base + st * 128;
        #pragma unroll
        for (int idx = tid; idx < N_TILE * 16; idx += 256) {
            int r = idx >> 4;
            int c8 = idx & 15;
            cp_async16(s_w + (st * N_TILE + r) * K_STRIDE + c8 * 8,
                       W + static_cast<size_t>(e_base + r) * K + k_off + c8 * 8);
        }
        #pragma unroll
        for (int idx = tid; idx < M_MAX_TILE * 16; idx += 256) {
            int r = idx >> 4;
            int c8 = idx & 15;
            if (r < M) {
                cp_async16(s_x + (st * M_MAX_TILE + r) * K_STRIDE + c8 * 8,
                           X + static_cast<size_t>(r) * K + k_off + c8 * 8);
            } else {
                *reinterpret_cast<uint4*>(s_x + (st * M_MAX_TILE + r) * K_STRIDE + c8 * 8) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    issue_stage(0);
    issue_stage(1);
    issue_stage(2);
    issue_stage(3);

    float acc[N_SUB][M_GROUPS][4];
    #pragma unroll
    for (int ns = 0; ns < N_SUB; ++ns) {
        #pragma unroll
        for (int mg = 0; mg < M_GROUPS; ++mg) {
            acc[ns][mg][0] = 0.0f;
            acc[ns][mg][1] = 0.0f;
            acc[ns][mg][2] = 0.0f;
            acc[ns][mg][3] = 0.0f;
        }
    }

    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;
    const int warp_k = warp * 16;

    auto consume_stage = [&](int st) {
        const __nv_bfloat16* x_base_s = s_x + st * M_MAX_TILE * K_STRIDE + warp_k;
        uint32_t b_frag[M_GROUPS][2];
        #pragma unroll
        for (int mg = 0; mg < M_GROUPS; ++mg) {
            const __nv_bfloat16* xm = x_base_s + (mg * 8 + group_id) * K_STRIDE;
            b_frag[mg][0] = *reinterpret_cast<const uint32_t*>(xm + tid_in_grp * 2);
            b_frag[mg][1] = *reinterpret_cast<const uint32_t*>(xm + tid_in_grp * 2 + 8);
        }
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            const __nv_bfloat16* w_base_s = s_w + (st * N_TILE + ns * 16) * K_STRIDE + warp_k;
            uint32_t a[4];
            a[0] = *reinterpret_cast<const uint32_t*>(w_base_s + group_id * K_STRIDE + tid_in_grp * 2);
            a[1] = *reinterpret_cast<const uint32_t*>(w_base_s + (group_id + 8) * K_STRIDE + tid_in_grp * 2);
            a[2] = *reinterpret_cast<const uint32_t*>(w_base_s + group_id * K_STRIDE + tid_in_grp * 2 + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(w_base_s + (group_id + 8) * K_STRIDE + tid_in_grp * 2 + 8);
            #pragma unroll
            for (int mg = 0; mg < M_GROUPS; ++mg) {
                mma_m16n8k16_bf16(acc[ns][mg], a, b_frag[mg]);
            }
        }
    };

    asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    __syncthreads();
    consume_stage(0);
    asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    __syncthreads();
    consume_stage(1);
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    __syncthreads();
    consume_stage(2);
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    consume_stage(3);

    __syncthreads();
    float* s_red = reinterpret_cast<float*>(smem_raw);
    #pragma unroll
    for (int ns = 0; ns < N_SUB; ++ns) {
        #pragma unroll
        for (int mg = 0; mg < M_GROUPS; ++mg) {
            int m0 = mg * 8 + tid_in_grp * 2;
            int e0 = ns * 16 + group_id;
            s_red[(warp * M_MAX_TILE + m0) * N_TILE + e0] = acc[ns][mg][0];
            s_red[(warp * M_MAX_TILE + m0 + 1) * N_TILE + e0] = acc[ns][mg][1];
            s_red[(warp * M_MAX_TILE + m0) * N_TILE + e0 + 8] = acc[ns][mg][2];
            s_red[(warp * M_MAX_TILE + m0 + 1) * N_TILE + e0 + 8] = acc[ns][mg][3];
        }
    }
    __syncthreads();

    int total_out = M * N_TILE;
    for (int idx = tid; idx < total_out; idx += 256) {
        int m = idx / N_TILE;
        int el = idx % N_TILE;
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            sum += s_red[(w * M_MAX_TILE + m) * N_TILE + el];
        }
        part_buf[(static_cast<size_t>(sk) * M + m) * E + e_base + el] = sum;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

template <int E, int TOPK, int K_SPLIT>
__global__ void __launch_bounds__((E / 128) * 32, 4)
sm103_router_topk_kernel(
    const float* __restrict__ part_buf,
    const float* __restrict__ bias,
    const float* __restrict__ image_bias,
    const bool* __restrict__ image_mask,
    const bool* __restrict__ valid_mask,
    const int32_t* __restrict__ placement,
    int64_t* __restrict__ out_ids,
    float* __restrict__ out_probs,
    int64_t* __restrict__ out_routes,
    int M,
    float scaling
) {
    constexpr int WAVES = E / 128;
    const int m = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    __shared__ float s_u[E];
    __shared__ float s_cand_s[32];
    __shared__ int s_cand_e[32];

    bool is_valid = valid_mask[m];
    const float* b_ptr = (image_mask && image_bias && image_mask[m]) ? image_bias : bias;
    int base = warp * 128 + lane * 4;
    float4 b_reg = *reinterpret_cast<const float4*>(b_ptr + base);

    asm volatile("griddepcontrol.wait;" ::: "memory");
    // Release dependent kernel (e.g. layer_enter_predict Dekker snapshot) to run its
    // input-independent prologue concurrently with top-k reduction and normalization.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    if (!is_valid) {
        if (warp == 0 && lane < TOPK) {
            out_ids[m * TOPK + lane] = -1;
            out_probs[m * TOPK + lane] = 0.0f;
            if (out_routes) out_routes[m * TOPK + lane] = -1;
        }
        return;
    }

    float4 acc4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int s = 0; s < K_SPLIT; ++s) {
        float4 pv = *reinterpret_cast<const float4*>(part_buf + (static_cast<size_t>(s) * M + m) * E + base);
        acc4.x += pv.x;
        acc4.y += pv.y;
        acc4.z += pv.z;
        acc4.w += pv.w;
    }
    float u0 = sqrt_softplus(acc4.x);
    float u1 = sqrt_softplus(acc4.y);
    float u2 = sqrt_softplus(acc4.z);
    float u3 = sqrt_softplus(acc4.w);
    *reinterpret_cast<float4*>(s_u + base) = make_float4(u0, u1, u2, u3);

    float b0 = u0 + b_reg.x;
    float b1 = u1 + b_reg.y;
    float b2 = u2 + b_reg.z;
    float b3 = u3 + b_reg.w;
    uint32_t perm = sort4_best_first(b0, b1, b2, b3);

    uint32_t cursor = 0;
    float local_top_s = -INFINITY;
    int local_top_e = -1;
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float cs = cursor == 4 ? -INFINITY :
                   (cursor & 2 ? (cursor & 1 ? b3 : b2) : (cursor & 1 ? b1 : b0));
        int ce = base + static_cast<int>((perm >> (cursor * 2)) & 3u);
        float win_s = redux_max_f32(cs);
        uint32_t tied = (cs == win_s) ? static_cast<uint32_t>(ce) : 0xffffffffu;
        int win_e = static_cast<int>(__reduce_min_sync(0xffffffff, tied));
        if (ce == win_e) ++cursor;
        if (lane == k) {
            local_top_s = win_s;
            local_top_e = win_e;
        }
    }

    int sel_expert = local_top_e;
    if constexpr (WAVES > 1) {
        if (lane < TOPK) {
            s_cand_s[warp * TOPK + lane] = local_top_s;
            s_cand_e[warp * TOPK + lane] = local_top_e;
        }
        __syncthreads();
        if (warp != 0) return;
        constexpr int TOTAL_CAND = WAVES * TOPK;
        float my_s = (lane < TOTAL_CAND) ? s_cand_s[lane] : -INFINITY;
        int my_e = (lane < TOTAL_CAND) ? s_cand_e[lane] : -1;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            float win_s = redux_max_f32(my_s);
            uint32_t tied = (my_s == win_s && my_e >= 0) ? static_cast<uint32_t>(my_e) : 0xffffffffu;
            int win_e = static_cast<int>(__reduce_min_sync(0xffffffff, tied));
            if (my_e == win_e) {
                my_s = -INFINITY;
                my_e = -1;
            }
            if (lane == k) sel_expert = win_e;
        }
    } else {
        __syncwarp();
    }

    float sel_u = (lane < TOPK) ? s_u[sel_expert] : 0.0f;
    float sum_u = sel_u;
    #pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
        sum_u += __shfl_xor_sync(0xffffffff, sum_u, off);
    }
    if (lane < TOPK) {
        out_ids[m * TOPK + lane] = static_cast<int64_t>(placement[sel_expert]);
        out_probs[m * TOPK + lane] = sel_u / (sum_u + 1e-20f) * scaling;
        if (out_routes) out_routes[m * TOPK + lane] = static_cast<int64_t>(sel_expert);
    }
}

template <int E, int K, int N_TILE, int K_SPLIT, int M_MAX_TILE>
void configure_gemm() {
    constexpr size_t smem = gemm_smem_bytes<E, K, N_TILE, K_SPLIT, M_MAX_TILE>();
    if constexpr (smem > 48 * 1024) {
        cudaError_t err = cudaFuncSetAttribute(
            sm103_router_gemm_kernel<E, K, N_TILE, K_SPLIT, M_MAX_TILE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        TORCH_CHECK(err == cudaSuccess, "sm103 router smem opt-in failed: ", cudaGetErrorString(err));
    }
}

template <int E, int K, int TOPK, int N_TILE, int K_SPLIT, int M_MAX_TILE>
void launch_router(
    const __nv_bfloat16* X, const __nv_bfloat16* W, const float* bias,
    const float* ib, const bool* im, const bool* valid, const int32_t* placement,
    float* part_buf, int64_t* ids, float* probs, int64_t* rt,
    int M, float scaling, cudaStream_t stream
) {
    constexpr size_t smem = gemm_smem_bytes<E, K, N_TILE, K_SPLIT, M_MAX_TILE>();
    int grid1 = (E / N_TILE) * K_SPLIT;

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg1 = {};
    cfg1.gridDim = dim3(grid1);
    cfg1.blockDim = dim3(256);
    cfg1.dynamicSmemBytes = smem;
    cfg1.stream = stream;
    cfg1.attrs = attr;
    cfg1.numAttrs = 1;
    cudaError_t e1 = cudaLaunchKernelEx(
        &cfg1, sm103_router_gemm_kernel<E, K, N_TILE, K_SPLIT, M_MAX_TILE>, X, W, part_buf, M);
    TORCH_CHECK(e1 == cudaSuccess, "sm103 router GEMM launch failed: ", cudaGetErrorString(e1));

    cudaLaunchConfig_t cfg2 = {};
    cfg2.gridDim = dim3(M);
    cfg2.blockDim = dim3((E / 128) * 32);
    cfg2.dynamicSmemBytes = 0;
    cfg2.stream = stream;
    cfg2.attrs = attr;
    cfg2.numAttrs = 1;
    cudaError_t e2 = cudaLaunchKernelEx(
        &cfg2, sm103_router_topk_kernel<E, TOPK, K_SPLIT>,
        part_buf, bias, ib, im, valid, placement, ids, probs, rt, M, scaling);
    TORCH_CHECK(e2 == cudaSuccess, "sm103 router top-k launch failed: ", cudaGetErrorString(e2));
}

void sm103_router_prepare() {
    configure_gemm<384, 5120, 32, 10, 8>();
    configure_gemm<384, 5120, 32, 10, 16>();
    configure_gemm<384, 5120, 32, 10, 32>();
    configure_gemm<384, 5120, 32, 10, 64>();
    configure_gemm<128, 5120, 16, 10, 8>();
    configure_gemm<128, 5120, 16, 10, 16>();
    configure_gemm<128, 5120, 16, 10, 32>();
    configure_gemm<128, 5120, 16, 10, 64>();
}

void sm103_router_run(
    torch::Tensor x, torch::Tensor w, torch::Tensor bias,
    c10::optional<torch::Tensor> image_bias, c10::optional<torch::Tensor> image_mask,
    torch::Tensor valid, torch::Tensor placement,
    torch::Tensor part_buf,
    torch::Tensor ids, torch::Tensor probs, c10::optional<torch::Tensor> routes,
    double scaling
) {
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int E = static_cast<int>(w.size(0));
    const int TOPK = static_cast<int>(ids.size(1));
    TORCH_CHECK(M >= 1 && M <= 64 && K == 5120, "sm103 router requires 1..64 rows and K=5120");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.is_contiguous() && w.size(1) == K);
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && bias.is_contiguous() && bias.numel() == E);
    TORCH_CHECK(valid.scalar_type() == torch::kBool && valid.is_contiguous() && valid.numel() == M);
    TORCH_CHECK(placement.scalar_type() == torch::kInt32 && placement.is_contiguous() && placement.numel() == E);
    TORCH_CHECK(part_buf.scalar_type() == torch::kFloat32 && part_buf.is_contiguous() && part_buf.numel() >= 10LL * M * E);
    TORCH_CHECK(ids.scalar_type() == torch::kInt64 && ids.is_contiguous() && ids.size(0) == M);
    TORCH_CHECK(probs.scalar_type() == torch::kFloat32 && probs.is_contiguous() && probs.size(0) == M && probs.size(1) == TOPK);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    const float* bp = bias.data_ptr<float>();
    const float* ib = image_bias.has_value() ? image_bias->data_ptr<float>() : nullptr;
    const bool* im = image_mask.has_value() ? image_mask->data_ptr<bool>() : nullptr;
    const bool* vp = valid.data_ptr<bool>();
    const int32_t* pp = placement.data_ptr<int32_t>();
    float* pb = part_buf.data_ptr<float>();
    int64_t* ip = ids.data_ptr<int64_t>();
    float* prp = probs.data_ptr<float>();
    int64_t* rt = routes.has_value() ? routes->data_ptr<int64_t>() : nullptr;
    const float sc = static_cast<float>(scaling);

    if (E == 384 && TOPK == 6) {
        if (M <= 8) {
            launch_router<384, 5120, 6, 32, 10, 8>(xp, wp, bp, ib, im, vp, pp, pb, ip, prp, rt, M, sc, stream);
        } else if (M <= 16) {
            launch_router<384, 5120, 6, 32, 10, 16>(xp, wp, bp, ib, im, vp, pp, pb, ip, prp, rt, M, sc, stream);
        } else if (M <= 32) {
            launch_router<384, 5120, 6, 32, 10, 32>(xp, wp, bp, ib, im, vp, pp, pb, ip, prp, rt, M, sc, stream);
        } else {
            launch_router<384, 5120, 6, 32, 10, 64>(xp, wp, bp, ib, im, vp, pp, pb, ip, prp, rt, M, sc, stream);
        }
    } else if (E == 128 && TOPK == 3) {
        if (M <= 8) {
            launch_router<128, 5120, 3, 16, 10, 8>(xp, wp, bp, ib, im, vp, pp, pb, ip, prp, rt, M, sc, stream);
        } else if (M <= 16) {
            launch_router<128, 5120, 3, 16, 10, 16>(xp, wp, bp, ib, im, vp, pp, pb, ip, prp, rt, M, sc, stream);
        } else if (M <= 32) {
            launch_router<128, 5120, 3, 16, 10, 32>(xp, wp, bp, ib, im, vp, pp, pb, ip, prp, rt, M, sc, stream);
        } else {
            launch_router<128, 5120, 3, 16, 10, 64>(xp, wp, bp, ib, im, vp, pp, pb, ip, prp, rt, M, sc, stream);
        }
    } else {
        TORCH_CHECK(false, "unsupported SM103 router geometry (E=", E, ", top_k=", TOPK, ")");
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sm103_router_prepare", &sm103_router_prepare);
    m.def("sm103_router_run", &sm103_router_run);
}
