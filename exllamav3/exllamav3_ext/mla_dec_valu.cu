#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "mla_dec_valu.cuh"

// MLA decode absorb / unfold for R <= 8 rows on VALU, bit-identical to the Triton WMMA kernels
// (_mla_absorb_kernel / _mla_unfold_kernel in attention_fn/mla_triton.py).
//
// Numerics: v_wmma_f32_16x16x16_f16 equals a chain of 8 v_dot2_f32_f16 over k ascending from the running fp32 accumulator
// (exl3_moe_valu.cuh, proven on device). Each output element here is one such chain over the whole K, started from 0.0f, then one
// round-to-nearest-even half conversion, exactly tl.dot(acc over K) + acc.to(fp16). A row's bits do not depend on the other rows
// or on R. Verified bitwise against the Triton kernels, R = 1..8, tools/glm/ext_test.py (CPU bounds: tools/glm/cpu_bounds.py).
//
// Why VALU: the Triton kernels read W_UK / W_UV at 140-145 GB/s in isolation (and about half of that in the model). One lane owns a
// whole serial chain, so the weight stream is the only cost: absorb stages 32 W rows (one head) coalesced through LDS, 8 heads
// adjacent in the grid so concurrently running waves share the same 4 KB row pages; unfold reads W rows coalesced (a lane owns a
// v pair) with the rows 2k, 2k+1 packed into half2 for the dot2 chain.

typedef _Float16 mdv_h2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float mdv_dot2(uint32_t a, uint32_t b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(mdv_h2, a), __builtin_bit_cast(mdv_h2, b), c, false);
}
__device__ __forceinline__ uint16_t mdv_h(float f) { return __half_as_ushort(__float2half_rn(f)); }

// absorb: coalesced. One wave (32 lanes) = 32 consecutive n of one head. The wave loads the 32 row segments (2 * DN bytes each)
// coalesced into LDS (pitch DN/8 + 1 uint4, conflict-free for lane-per-row b128 reads), then lane l runs the serial chain of row n0 + l.
// grid (H, DC / 32) with HF (head fastest), 32 threads.
#define MDV2_ROWS 32
template <int R, int DN, int UN, bool HF>
__global__ __launch_bounds__(MDV2_ROWS)
void mla_absorb2_valu_kernel
(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
    const int qs, const int qk, const int heads, const int dc
)
{
    constexpr int CH = DN / 8;                      // uint4 per row segment (32 for DN 256)
    static_assert(CH == 32, "one lane per uint4 of a row segment");
    __shared__ __align__(16) uint4 qsh[R][CH];
    __shared__ __align__(16) uint4 wsh[MDV2_ROWS][CH + 1];
    const int h = HF ? blockIdx.x : blockIdx.y;
    const int l = threadIdx.x;
    const int n0 = (HF ? blockIdx.y : blockIdx.x) * MDV2_ROWS;
    #pragma unroll
    for (int r = 0; r < R; ++r) qsh[r][l] = *((const uint4*) (q + (size_t) r * qs + (size_t) h * qk) + l);
    #pragma unroll 1
    for (int i0 = 0; i0 < MDV2_ROWS; i0 += UN)
    {
        uint4 v[UN];
        #pragma unroll
        for (int j = 0; j < UN; ++j)
            v[j] = *((const uint4*) (w + (size_t) (n0 + i0 + j) * heads * DN + (size_t) h * DN) + l);
        #pragma unroll
        for (int j = 0; j < UN; ++j) wsh[i0 + j][l] = v[j];
    }
    __syncthreads();
    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    #pragma unroll 4
    for (int c = 0; c < CH; ++c)
    {
        const uint4 wv = wsh[l][c];
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            const uint4 a = qsh[r][c];
            float s = acc[r];
            s = mdv_dot2(wv.x, a.x, s);
            s = mdv_dot2(wv.y, a.y, s);
            s = mdv_dot2(wv.z, a.z, s);
            s = mdv_dot2(wv.w, a.w, s);
            acc[r] = s;
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) out[((size_t) h * R + r) * dc + n0 + l] = mdv_h(acc[r]);
}

// out[r, h, v] = sum_k o[h, r, k] * w[k, h * DV + v]; one lane = a v pair, serial chain over k (pairs of rows k, k+1).
// grid (H), DV / 2 threads. DC % (2 * UN) == 0.
template <int R, int DV, int UN>
__global__ __launch_bounds__(DV / 2)
void mla_unfold_valu_kernel
(
    const uint16_t* __restrict__ o, const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
    const int heads, const int dc
)
{
    extern __shared__ __align__(16) uint32_t osh[];     // [R][dc / 2]
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int dc2 = dc / 2;
    for (int i = t; i < R * dc2; i += DV / 2) osh[i] = ((const uint32_t*) (o + (size_t) h * R * dc))[i];
    __syncthreads();
    const uint32_t* wp = (const uint32_t*) (w + (size_t) h * DV) + t;
    const size_t rs = (size_t) heads * DV / 2;            // row stride in uint32
    float a0[R], a1[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; }
    #pragma unroll 1
    for (int k2 = 0; k2 < dc2; k2 += UN)
    {
        uint32_t wa[UN], wb[UN];
        #pragma unroll
        for (int j = 0; j < UN; ++j)
        {
            wa[j] = wp[(size_t) (2 * (k2 + j)) * rs];
            wb[j] = wp[(size_t) (2 * (k2 + j) + 1) * rs];
        }
        #pragma unroll
        for (int j = 0; j < UN; ++j)
        {
            const uint32_t p0 = (wa[j] & 0xffffu) | (wb[j] << 16);
            const uint32_t p1 = (wa[j] >> 16) | (wb[j] & 0xffff0000u);
            #pragma unroll
            for (int r = 0; r < R; ++r)
            {
                const uint32_t ov = osh[r * dc2 + k2 + j];
                a0[r] = mdv_dot2(p0, ov, a0[r]);
                a1[r] = mdv_dot2(p1, ov, a1[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r)
        ((uint32_t*) out)[(((size_t) r * heads + h) * DV) / 2 + t] = (uint32_t) mdv_h(a0[r]) | ((uint32_t) mdv_h(a1[r]) << 16);
}


#define MDV_ABS_UN 4
#define MDV_UNF_UN 4

void mla_absorb_dec
(
    const at::Tensor& q,         // (R, H, qk_dim) fp16, head stride qk_dim, row stride any multiple of 8
    const at::Tensor& w_uk_flat, // (D_c, H * 256) fp16 contiguous
    at::Tensor& out              // (H, R, D_c) fp16 contiguous
)
{
    const at::Device device = q.device();
    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(q.dim() == 3 && q.scalar_type() == at::kHalf && w_uk_flat.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
                "mla_absorb_dec: fp16 q (R, H, qk_dim), w, out");
    const int R = q.size(0), H = q.size(1), qk = q.size(2);
    const int dc = w_uk_flat.size(0);
    TORCH_CHECK(R >= 1 && R <= 8, "mla_absorb_dec: 1..8 rows");
    TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == qk && qk % 8 == 0 && qk >= 256 && q.stride(0) % 8 == 0, "mla_absorb_dec: q layout");
    TORCH_CHECK(((uintptr_t) q.data_ptr()) % 16 == 0 && ((uintptr_t) w_uk_flat.data_ptr()) % 16 == 0, "mla_absorb_dec: 16-byte alignment");
    TORCH_CHECK(w_uk_flat.is_contiguous() && w_uk_flat.dim() == 2 && w_uk_flat.size(1) == (int64_t) H * 256 && dc % 32 == 0, "mla_absorb_dec: w shape");
    TORCH_CHECK(out.is_contiguous() && out.numel() == (int64_t) H * R * dc, "mla_absorb_dec: out shape");
    const uint16_t* qp = (const uint16_t*) q.data_ptr();
    const uint16_t* wp = (const uint16_t*) w_uk_flat.data_ptr();
    uint16_t* op = (uint16_t*) out.data_ptr();
    const int qs = q.stride(0);
    const dim3 grid(H, dc / MDV2_ROWS);
    #define MDV_A(RR) case RR: mla_absorb2_valu_kernel<RR, 256, MDV_ABS_UN, true><<<grid, MDV2_ROWS, 0, stream>>>(qp, wp, op, qs, qk, H, dc); break;
    switch (R) { MDV_A(1) MDV_A(2) MDV_A(3) MDV_A(4) MDV_A(5) MDV_A(6) MDV_A(7) MDV_A(8) }
    #undef MDV_A
    cuda_check(cudaPeekAtLastError());
}

void mla_unfold_dec
(
    const at::Tensor& o_lat,     // (H, R, D_c) fp16 contiguous
    const at::Tensor& w_uv_flat, // (D_c, H * 256) fp16 contiguous
    at::Tensor& out              // (R, H, 256) fp16 contiguous
)
{
    const at::Device device = o_lat.device();
    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(o_lat.dim() == 3 && o_lat.scalar_type() == at::kHalf && w_uv_flat.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
                "mla_unfold_dec: fp16 o_lat (H, R, D_c), w, out");
    const int H = o_lat.size(0), R = o_lat.size(1), dc = o_lat.size(2);
    TORCH_CHECK(R >= 1 && R <= 8, "mla_unfold_dec: 1..8 rows");
    TORCH_CHECK(o_lat.is_contiguous() && w_uv_flat.is_contiguous() && out.is_contiguous(), "mla_unfold_dec: contiguous only");
    TORCH_CHECK(w_uv_flat.dim() == 2 && w_uv_flat.size(0) == dc && w_uv_flat.size(1) == (int64_t) H * 256 && dc % (2 * MDV_UNF_UN) == 0,
                "mla_unfold_dec: w shape");
    TORCH_CHECK(out.numel() == (int64_t) R * H * 256, "mla_unfold_dec: out shape");
    TORCH_CHECK(((uintptr_t) o_lat.data_ptr()) % 4 == 0 && ((uintptr_t) w_uv_flat.data_ptr()) % 4 == 0 && ((uintptr_t) out.data_ptr()) % 4 == 0,
                "mla_unfold_dec: 4-byte alignment");
    const uint16_t* op_ = (const uint16_t*) o_lat.data_ptr();
    const uint16_t* wp = (const uint16_t*) w_uv_flat.data_ptr();
    uint16_t* outp = (uint16_t*) out.data_ptr();
    #define MDV_U(RR) case RR: mla_unfold_valu_kernel<RR, 256, MDV_UNF_UN><<<H, 128, RR * dc * 2, stream>>>(op_, wp, outp, H, dc); break;
    switch (R) { MDV_U(1) MDV_U(2) MDV_U(3) MDV_U(4) MDV_U(5) MDV_U(6) MDV_U(7) MDV_U(8) }
    #undef MDV_U
    cuda_check(cudaPeekAtLastError());
}
#endif
