// Grouped decode MoE with q8_1 activations for K-quant/IQ4_XS experts
// same (slot, tile) grid as fq_moe_q8.cu (one 256-thread
// block per (slot s, 8-row tile t), gate+up + down + fold launches) but the
// KQ/IQ4_XS integer-dp4a dots (fq_dot_*_q8 in fq_gemv.cuh) and a SwiGLU
// epilogue (MiniCPM MoE is SwiGLU, unlike Bolt's gateless latent MoE).
// h [1, K] is quantized once; the [S, I] act scratch is quantized per slot
// row for the down dots. FMT selects the dot (0 = Q4_K, 1 = Q5_K,
// 2 = Q6_K, 3 = IQ4_XS, 4 = IQ3_XXS, 5 = IQ2_S). Portable (no PTX/shuffles).
#include <cuda_fp16.h>
#include <map>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_kq_item.cuh"
#include <cstdlib>
#include "fq_api.h"

namespace {

// Up+gate kernel: block (s, t) computes up to 8 gate rows and 8 up rows of
// expert sel[s] against the pre-quantized h (aqb, one row), SwiGLU epilogue
// (silu(gate) * up, torch half path) into act[s]. Lane mapping: group grp
// owns output n; lane l takes sub-chunks round-robin (the dot's (lane, 32)
// split), 32 partials reduce in shared.
template <int FMT>
__global__ void fq_moe_kq_upgate_kernel(const uint8_t* __restrict__ aqb,
                                        const int64_t* __restrict__ sel,
                                        const uint8_t* const* __restrict__ gate_bases,
                                        const uint8_t* const* __restrict__ up_bases,
                                        int64_t gate_stride,
                                        int64_t up_stride,
                                        int64_t qb_stride,
                                        __half* __restrict__ act,
                                        int S, int I, int K, int E, int ntiles_u, int topk,
                                        const uint8_t* __restrict__ sh_gate = nullptr,
                                        const uint8_t* __restrict__ sh_up = nullptr,
                                        int64_t sh_stride = 0)
{
    int b = blockIdx.x;
    int s = b / ntiles_u;
    int t = b % ntiles_u;
    int grp = threadIdx.x / FQ_GEMV_LANES;
    int lane = threadIdx.x % FQ_GEMV_LANES;
    const bool shs = s >= S;   // shared-expert slot of row s - S (sh_gate set): Q8_0 gate / up
    int64_t e = shs ? 0 : sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    int n = t * FQ_GEMV_OUT_PER_BLOCK + grp;
    const uint8_t* aq = aqb + (int64_t)(shs ? s - S : s / topk) * qb_stride;   // assignment s = row * topk + j reads h row s / topk

    extern __shared__ float red[];

    float g = 0.0f, u = 0.0f;
    if (n < I)
    {
        const uint8_t* gr = shs ? sh_gate + (int64_t)n * sh_stride : gate_bases[e] + (int64_t)n * gate_stride;
        const uint8_t* ur = shs ? sh_up + (int64_t)n * sh_stride : up_bases[e] + (int64_t)n * up_stride;
        if (shs)
        {
            g = fq_dot_q8_0_q8(gr, aq, K, lane, FQ_GEMV_LANES);
            u = fq_dot_q8_0_q8(ur, aq, K, lane, FQ_GEMV_LANES);
        }
        else if constexpr (FMT == 0)
        {
            g = fq_dot_q4_k_q8(gr, aq, K, lane, FQ_GEMV_LANES);
            u = fq_dot_q4_k_q8(ur, aq, K, lane, FQ_GEMV_LANES);
        }
        else if constexpr (FMT == 1)
        {
            g = fq_dot_q5_k_q8(gr, aq, K, lane, FQ_GEMV_LANES);
            u = fq_dot_q5_k_q8(ur, aq, K, lane, FQ_GEMV_LANES);
        }
        else if constexpr (FMT == 2)
        {
            g = fq_dot_q6_k_q8(gr, aq, K, lane, FQ_GEMV_LANES);
            u = fq_dot_q6_k_q8(ur, aq, K, lane, FQ_GEMV_LANES);
        }
        else if constexpr (FMT == 3)
        {
            g = fq_dot_iq4_xs_q8(gr, aq, K, lane, FQ_GEMV_LANES);
            u = fq_dot_iq4_xs_q8(ur, aq, K, lane, FQ_GEMV_LANES);
        }
        else if constexpr (FMT == 4)
        {
            g = fq_dot_iq3_xxs_q8(gr, aq, K, lane, FQ_GEMV_LANES);
            u = fq_dot_iq3_xxs_q8(ur, aq, K, lane, FQ_GEMV_LANES);
        }
        else
        {
            g = fq_dot_iq2_s_q8(gr, aq, K, lane, FQ_GEMV_LANES);
            u = fq_dot_iq2_s_q8(ur, aq, K, lane, FQ_GEMV_LANES);
        }
    }
    // Two reductions in one shared row: red[2*tid] = g, red[2*tid+1] = u.
    red[2 * threadIdx.x] = g;
    red[2 * threadIdx.x + 1] = u;
    __syncthreads();
    if (n < I && lane == 0)
    {
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
        {
            sg = __fadd_rn(sg, red[(grp * 32 + l) * 2]);
            su = __fadd_rn(su, red[(grp * 32 + l) * 2 + 1]);
        }
        // Torch half path: g/u round to half (Linear out_dtype = half),
        // silu in float, product back to half.
        float gh = __half2float(__float2half_rn(sg));
        float uh = __half2float(__float2half_rn(su));
        float sig = 1.0f / (1.0f + expf(-gh));
        act[(int64_t)s * I + n] = __float2half_rn(sig * gh * uh);
    }
}

// Q4_K gate+up, v2 loads (bit-identical to fq_moe_kq_upgate_kernel<0>): 32 rows per block, 8 threads per row. Thread j of a
// row runs the items it = j, j + 8, ... (fq_kq_item_q4_k: 4 consecutive v1 sub-chunks from two 16-byte loads); sub-chunk
// c = 4 it + qo belongs to v1 lane c & 31 = 4 j + qo, so the four register partials are v1's lane partials 4j .. 4j+3 summed
// in the same increasing c order. Lane partials go to shared memory and one thread per row adds them lane by lane as v1 does.
constexpr int FQ_MOE_V2_ROWS = 32;
constexpr int FQ_MOE_V2_PS = 33;
__global__ void __launch_bounds__(256) fq_moe_kq_upgate_v2_kernel(const uint8_t* __restrict__ aqb,
                                        const int64_t* __restrict__ sel,
                                        const uint8_t* const* __restrict__ gate_bases,
                                        const uint8_t* const* __restrict__ up_bases,
                                        int64_t gate_stride,
                                        int64_t up_stride,
                                        int64_t qb_stride,
                                        __half* __restrict__ act,
                                        int S, int I, int K, int E, int ntiles_u, int topk,
                                        const uint8_t* __restrict__ sh_gate = nullptr,
                                        const uint8_t* __restrict__ sh_up = nullptr,
                                        int64_t sh_stride = 0)
{
    __shared__ float pg[FQ_MOE_V2_ROWS * FQ_MOE_V2_PS];
    __shared__ float pu[FQ_MOE_V2_ROWS * FQ_MOE_V2_PS];
    const int s = blockIdx.x / ntiles_u;
    const int t = blockIdx.x % ntiles_u;
    const int r = threadIdx.x >> 3, j = threadIdx.x & 7;
    const bool shs = s >= S;   // shared-expert slot of row s - S: Q8_0 gate / up, 8 lanes per row
    int64_t e = shs ? 0 : sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    const int n = t * FQ_MOE_V2_ROWS + r;
    const uint8_t* aq = aqb + (int64_t)(shs ? s - S : s / topk) * qb_stride;
    float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (n < I && shs)
    {
        ag[0] = fq_dot_q8_0_q8(sh_gate + (int64_t)n * sh_stride, aq, K, j, 8);
        au[0] = fq_dot_q8_0_q8(sh_up + (int64_t)n * sh_stride, aq, K, j, 8);
    }
    else if (n < I)
    {
        const uint8_t* gr = gate_bases[e] + (int64_t)n * gate_stride;
        const uint8_t* ur = up_bases[e] + (int64_t)n * up_stride;
        if ((((uintptr_t)gr | (uintptr_t)ur) & 15) == 0)
        {
            const int nitems = (K / 256) * 4;
            for (int it = j; it < nitems; it += 8)
            {
                float tg[4], tu[4];
                fq_kq_item_q4_k(gr, aq, it, tg);
                fq_kq_item_q4_k(ur, aq, it, tu);
                #pragma unroll
                for (int q = 0; q < 4; q++)
                {
                    ag[q] = __fadd_rn(ag[q], tg[q]);
                    au[q] = __fadd_rn(au[q], tu[q]);
                }
            }
        }
        else
        {
            #pragma unroll
            for (int q = 0; q < 4; q++)
            {
                ag[q] = fq_dot_q4_k_q8(gr, aq, K, 4 * j + q, FQ_GEMV_LANES);
                au[q] = fq_dot_q4_k_q8(ur, aq, K, 4 * j + q, FQ_GEMV_LANES);
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        pg[r * FQ_MOE_V2_PS + 4 * j + q] = ag[q];
        pu[r * FQ_MOE_V2_PS + 4 * j + q] = au[q];
    }
    __syncthreads();
    if (n < I && j == 0)
    {
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
        {
            sg = __fadd_rn(sg, pg[r * FQ_MOE_V2_PS + l]);
            su = __fadd_rn(su, pu[r * FQ_MOE_V2_PS + l]);
        }
        float gh = __half2float(__float2half_rn(sg));
        float uh = __half2float(__float2half_rn(su));
        float sig = 1.0f / (1.0f + expf(-gh));
        act[(int64_t)s * I + n] = __float2half_rn(sig * gh * uh);
    }
}

// Down, v2 items (bit-identical to fq_moe_kq_down_kernel<FMT>, FMT 0 Q4_K / 1 Q5_K / 2 Q6_K): 32 rows per block, 8 threads per
// row, thread j runs items j, j + 8, ...; the item's four values belong to v1 lanes 4 j + q (Q4_K / Q5_K) or j + 8 q (Q6_K),
// each added in increasing sub-chunk order as v1 does; then one thread per row adds the 32 lane partials lane by lane.
template <int FMT>
__global__ void __launch_bounds__(256) fq_moe_kq_down_v2_kernel(const uint8_t* __restrict__ aqb,
                                      int64_t qb_stride,
                                      const int64_t* __restrict__ sel,
                                      const uint8_t* __restrict__ const* down_bases,
                                      int64_t down_stride,
                                      float* __restrict__ partial,
                                      int S, int I, int K, int E, int ntiles_d,
                                      const uint8_t* __restrict__ sh_down = nullptr,
                                      int64_t sh_stride = 0)
{
    __shared__ float pp[FQ_MOE_V2_ROWS * FQ_MOE_V2_PS];
    const int s = blockIdx.x / ntiles_d;
    const int t = blockIdx.x % ntiles_d;
    const int r = threadIdx.x >> 3, j = threadIdx.x & 7;
    const bool shs = s >= S;   // shared-expert slot (act row s): Q8_0 down, 8 lanes per row
    int64_t e = shs ? 0 : sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    const int n = t * FQ_MOE_V2_ROWS + r;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (n < K && shs)
        acc[0] = fq_dot_q8_0_q8(sh_down + (int64_t)n * sh_stride, aqb + (int64_t)s * qb_stride, I, j, 8);
    else if (n < K)
    {
        const uint8_t* dr = down_bases[e] + (int64_t)n * down_stride;
        const uint8_t* aq = aqb + (int64_t)s * qb_stride;
        const uintptr_t amask = (FMT == 2) ? 1 : 15;
        if (((uintptr_t)dr & amask) == 0)
        {
            const int nitems = (I / 256) * ((FMT == 2) ? 8 : 4);
            for (int it = j; it < nitems; it += 8)
            {
                float tv[4];
                if constexpr (FMT == 0) fq_kq_item_q4_k(dr, aq, it, tv);
                else if constexpr (FMT == 1) fq_kq_item_q5_k(dr, aq, it, tv);
                else fq_kq_item_q6_k(dr, aq, it, tv);
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    acc[q] = __fadd_rn(acc[q], tv[q]);
            }
        }
        else
        {
            #pragma unroll
            for (int q = 0; q < 4; q++)
            {
                const int l = (FMT == 2) ? (j + 8 * q) : (4 * j + q);
                if constexpr (FMT == 0) acc[q] = fq_dot_q4_k_q8(dr, aq, I, l, FQ_GEMV_LANES);
                else if constexpr (FMT == 1) acc[q] = fq_dot_q5_k_q8(dr, aq, I, l, FQ_GEMV_LANES);
                else acc[q] = fq_dot_q6_k_q8(dr, aq, I, l, FQ_GEMV_LANES);
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < 4; q++)
        pp[r * FQ_MOE_V2_PS + ((FMT == 2) ? (j + 8 * q) : (4 * j + q))] = acc[q];
    __syncthreads();
    if (n < K && j == 0)
    {
        float sp = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
            sp = __fadd_rn(sp, pp[r * FQ_MOE_V2_PS + l]);
        partial[(int64_t)s * K + n] = sp;
    }
}

template <int FMT>
static void fq_moe_kq_down_v2_launch(const uint8_t* aq, int64_t qb_stride, const int64_t* sel, const uint8_t* const* bases,
                                     int64_t down_stride, float* partial, int S, int I, int K, int E, cudaStream_t stream,
                                     int Sg = 0, const uint8_t* sh_down = nullptr, int64_t sh_stride = 0)
{
    if constexpr (FMT <= 2)
    {
        const int nt2 = (K + FQ_MOE_V2_ROWS - 1) / FQ_MOE_V2_ROWS;
        fq_moe_kq_down_v2_kernel<FMT><<<(Sg ? Sg : S) * nt2, 256, 0, stream>>>(aq, qb_stride, sel, bases, down_stride, partial, S, I, K, E, nt2,
                                                                              sh_down, sh_stride);
    }
}

static bool fq_moe_v2_on()
{
    static const char* e = std::getenv("BOLT_FQ_MOE_V2");
    static const bool on = !(e && e[0] == '0');
    return on;
}

// Q5_1 x q8_1 dot (ggml vec_dot_q5_1_q8_1 arithmetic): per 32-block, 8 dp4a over the 5-bit values (0..31) and the int8 activations,
// then d_w * d_a * sumi + m_w * s_a (s_a = the q8_1 block's stored sum). Lanes take 32-blocks round-robin like fq_dot_q8_0_q8.
constexpr int FQ_Q5_1_BYTES = 24;
__device__ __forceinline__ float fq_dot_q5_1_q8(const uint8_t* __restrict__ wr, const uint8_t* __restrict__ aqb,
                                                int K, int tid, int nthreads)
{
    const int nchunks = K / 32;
    float acc = 0.0f;
    for (int c = tid; c < nchunks; c += nthreads)
    {
        const uint8_t* blk = wr + (size_t)c * FQ_Q5_1_BYTES;
        __half dwh, mwh, dah, sah;
        memcpy(&dwh, blk, 2);
        memcpy(&mwh, blk + 2, 2);
        const uint8_t* ab = aqb + (size_t)c * FQ_Q8_1_BYTES;
        memcpy(&dah, ab, 2);
        memcpy(&sah, ab + 2, 2);
        const uint32_t qh = (uint32_t)blk[4] | ((uint32_t)blk[5] << 8) | ((uint32_t)blk[6] << 16) | ((uint32_t)blk[7] << 24);
        const uint8_t* qs = blk + 8;
        const int32_t* aq32 = reinterpret_cast<const int32_t*>(ab + 4);
        int sumi = 0;
        #pragma unroll
        for (int w = 0; w < 4; w++)
        {
            int32_t lo = 0, hi = 0;
            #pragma unroll
            for (int b = 0; b < 4; b++)
            {
                const int j = 4 * w + b;
                lo |= (int32_t)((qs[j] & 15) | (((qh >> j) & 1) << 4)) << (8 * b);
                hi |= (int32_t)((qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) << (8 * b);
            }
            sumi = fq_dp4a(lo, aq32[w], sumi);
            sumi = fq_dp4a(hi, aq32[w + 4], sumi);
        }
        const float dw = __half2float(dwh), mw = __half2float(mwh), da = __half2float(dah), sa = __half2float(sah);
        acc = __fadd_rn(acc, __fadd_rn(__fmul_rn(__fmul_rn(dw, da), (float)sumi), __fmul_rn(mw, sa)));
    }
    return acc;
}

// Down kernel: block (s, t) dots down[sel[s]] rows against the
// pre-quantized act row s into partial[s]. Same lane split as above.
template <int FMT>
__global__ void fq_moe_kq_down_kernel(const uint8_t* __restrict__ aqb,
                                      int64_t qb_stride,
                                      const int64_t* __restrict__ sel,
                                      const uint8_t* __restrict__ const* down_bases,
                                      int64_t down_stride,
                                      float* __restrict__ partial,
                                      int S, int I, int K, int E, int ntiles_d,
                                      const uint8_t* __restrict__ sh_down = nullptr,
                                      int64_t sh_stride = 0)
{
    int b = blockIdx.x;
    int s = b / ntiles_d;
    int t = b % ntiles_d;
    int grp = threadIdx.x / FQ_GEMV_LANES;
    int lane = threadIdx.x % FQ_GEMV_LANES;
    const bool shs = s >= S;   // shared-expert slot (act row s): Q8_0 down
    int64_t e = shs ? 0 : sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    int n = t * FQ_GEMV_OUT_PER_BLOCK + grp;

    extern __shared__ float red[];

    float p = 0.0f;
    if (n < K)
    {
        const uint8_t* dr = shs ? sh_down + (int64_t)n * sh_stride : down_bases[e] + (int64_t)n * down_stride;
        const uint8_t* aq = aqb + (int64_t)s * qb_stride;
        if (shs)
            p = fq_dot_q8_0_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 0)
            p = fq_dot_q4_k_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 1)
            p = fq_dot_q5_k_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 2)
            p = fq_dot_q6_k_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 3)
            p = fq_dot_iq4_xs_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 4)
            p = fq_dot_iq3_xxs_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 6)
            p = fq_dot_q8_0_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 7)
            p = fq_dot_q5_1_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else
            p = fq_dot_iq2_s_q8(dr, aq, I, lane, FQ_GEMV_LANES);
    }
    red[threadIdx.x] = p;
    __syncthreads();
    if (n < K && lane == 0)
    {
        float sp = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
            sp = __fadd_rn(sp, red[grp * 32 + l]);
        partial[(int64_t)s * K + n] = sp;
    }
}

}  // namespace

// Weighted fold (same as fq_moe_q8.cu's): out[k] = sum_s w[s]*partial[s,k].
__global__ void fq_moe_kq_combine_kernel(const float* __restrict__ partial,
                                         const __half* __restrict__ wts,
                                         float* __restrict__ out, int S, int K,
                                         const float* __restrict__ sh_partial = nullptr,   // [R, K] shared-expert rows
                                         const __half* __restrict__ x = nullptr,           // [R, K] MoE input (shared gate)
                                         const float* __restrict__ wg = nullptr)           // [K] shared gate row or nullptr
{
    // shared expert: g = sigmoid(x_r . wg) (1 without a gate), the same for every block of row r
    __shared__ float gred[256];
    float g = 1.0f;
    if (sh_partial != nullptr && wg != nullptr)
    {
        const __half* xr = x + (int64_t)blockIdx.y * K;
        float d = 0.0f;
        for (int i = threadIdx.x; i < K; i += blockDim.x)
            d = fmaf(__half2float(xr[i]), wg[i], d);
        gred[threadIdx.x] = d;
        __syncthreads();
        for (int w = blockDim.x / 2; w > 0; w >>= 1)
        {
            if ((int)threadIdx.x < w) gred[threadIdx.x] += gred[threadIdx.x + w];
            __syncthreads();
        }
        g = 1.0f / (1.0f + expf(-gred[0]));
    }
    const float* shr = sh_partial != nullptr ? sh_partial + (int64_t)blockIdx.y * K : nullptr;
    // blockIdx.y = row r: its S = topk assignments are partial[r*S .. r*S+S), weights wts[r*S ..)
    partial += (int64_t)blockIdx.y * S * K;
    wts += (int64_t)blockIdx.y * S;
    out += (int64_t)blockIdx.y * K;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < K; k += gridDim.x * blockDim.x)
    {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 32; s++)
        {
            if (s < S)
                acc = __fadd_rn(acc, __fmul_rn(partial[(int64_t)s * K + k],
                                              __half2float(wts[s])));
        }
        if (shr != nullptr) acc = fmaf(g, shr[k], acc);
    out[k] = acc;
    }
}

// Shared expert fused into the grouped launches (all null: none). Q8_0 gate / up / down, intermediate I.
struct FqMoeSh
{
    const uint8_t* gate = nullptr;
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    int64_t ug_stride = 0;
    int64_t d_stride = 0;
    const float* wg = nullptr;
};

template <int FMT_UG, int FMT_D>
static void fq_moe_kq_grouped_launch(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                                     torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                                     torch::Tensor down_ptrs,
                                     int64_t gate_stride, int64_t up_stride,
                                     int64_t down_stride, int64_t block_bytes,  // block_bytes = down's format
                                     torch::Tensor partial, torch::Tensor out, FqMoeSh sh = FqMoeSh())
{
    int K = (int)h.size(1);
    int R = (int)h.size(0);
    int S = (int)sel.numel();   // R rows x topk picks, assignment s = row * topk + j
    int topk = S / R;
    TORCH_CHECK(S % R == 0 && R >= 1 && topk <= 32, "fq_moe_kq: sel must be R x topk with topk <= 32");
    TORCH_CHECK_DTYPE(h, kHalf);
    int E = (int)up_ptrs.numel();
    // I from the down row stride (down rows are I-wide, like Bolt's
    // derivation from down_stride).
    int I = (int)(down_stride / block_bytes) * ((FMT_D >= 6) ? 32 : FQ_Q4_K_GROUP);   // ids 6 / 7: 32-element blocks
    at::cuda::CUDAGuard guard(h.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int Sg = S + (sh.gate != nullptr ? R : 0);   // routed slots + one shared-expert slot per row
    TORCH_CHECK(partial.size(0) >= Sg, "fq_moe_kq: partial must hold the shared-expert rows too");
    int ntiles_u = (I + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    int ntiles_d = (K + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    torch::Tensor qb_h = fq_q8_scratch(h.get_device(), R, K);
    fq_quant_q8_1_rows(h, qb_h);
    int64_t qb_stride_h = (int64_t)(K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    static thread_local std::map<std::tuple<int, int, int>, torch::Tensor> act_cache;
    auto key = std::make_tuple((int)h.get_device(), Sg, I);
    auto it = act_cache.find(key);
    torch::Tensor act;
    if (it != act_cache.end())
        act = it->second;
    else
    {
        act = torch::empty({Sg, I}, torch::dtype(torch::kHalf).device(h.device()));
        act_cache[key] = act;
    }
    // Up+gate: 2 floats per thread in shared (g and u side by side).
    size_t shmem_u = (size_t)FQ_GEMV_THREADS * 2 * sizeof(float);
    if (FMT_UG == 0 && K % 256 == 0 && fq_moe_v2_on())
    {
        const int nt2 = (I + FQ_MOE_V2_ROWS - 1) / FQ_MOE_V2_ROWS;
        fq_moe_kq_upgate_v2_kernel<<<Sg * nt2, 256, 0, stream>>>(
            qb_h.data_ptr<uint8_t>(), sel.data_ptr<int64_t>(),
            reinterpret_cast<const uint8_t* const*>(gate_ptrs.data_ptr<int64_t>()),
            reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr<int64_t>()),
            gate_stride, up_stride, qb_stride_h,
            reinterpret_cast<__half*>(act.data_ptr<at::Half>()),
            S, I, K, E, nt2, topk, sh.gate, sh.up, sh.ug_stride);
    }
    else
    fq_moe_kq_upgate_kernel<FMT_UG><<<Sg * ntiles_u, FQ_GEMV_THREADS, shmem_u, stream>>>(
        qb_h.data_ptr<uint8_t>(), sel.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(gate_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr<int64_t>()),
        gate_stride, up_stride, qb_stride_h,
        reinterpret_cast<__half*>(act.data_ptr<at::Half>()),
        S, I, K, E, ntiles_u, topk, sh.gate, sh.up, sh.ug_stride);
    torch::Tensor qb_act = fq_q8_scratch(h.get_device(), Sg, I);
    fq_quant_q8_1_rows(act, qb_act);
    int64_t qb_stride_a = (int64_t)(I / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    size_t shmem_d = (size_t)FQ_GEMV_THREADS * sizeof(float);
    if (FMT_D <= 2 && I % 256 == 0 && fq_moe_v2_on())
        fq_moe_kq_down_v2_launch<FMT_D>(qb_act.data_ptr<uint8_t>(), qb_stride_a, sel.data_ptr<int64_t>(),
                                        reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr<int64_t>()),
                                        down_stride, partial.data_ptr<float>(), S, I, K, E, stream, Sg, sh.down, sh.d_stride);
    else
    fq_moe_kq_down_kernel<FMT_D><<<Sg * ntiles_d, FQ_GEMV_THREADS, shmem_d, stream>>>(
        qb_act.data_ptr<uint8_t>(), qb_stride_a, sel.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr<int64_t>()),
        down_stride, partial.data_ptr<float>(), S, I, K, E, ntiles_d, sh.down, sh.d_stride);
    int nblocks = (K + 255) / 256;
    fq_moe_kq_combine_kernel<<<dim3(nblocks, R), 256, 0, stream>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<const __half*>(wts.data_ptr<at::Half>()),
        out.data_ptr<float>(), topk, K,
        sh.gate != nullptr ? partial.data_ptr<float>() + (int64_t)S * K : nullptr,
        sh.gate != nullptr ? reinterpret_cast<const __half*>(h.data_ptr<at::Half>()) : nullptr, sh.wg);
}

void fq_moe_q4k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_launch<0, 0>(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                                gate_stride, up_stride, down_stride,
                                FQ_Q4_K_BYTES, partial, out);
}

void fq_moe_q5k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_launch<1, 1>(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                                gate_stride, up_stride, down_stride,
                                FQ_Q5_K_BYTES, partial, out);
}

void fq_moe_q6k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_launch<2, 2>(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                                gate_stride, up_stride, down_stride,
                                FQ_Q6_K_BYTES, partial, out);
}

void fq_moe_iq4xs_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                             torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                             torch::Tensor down_ptrs,
                             int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                             torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_launch<3, 3>(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                                gate_stride, up_stride, down_stride,
                                FQ_IQ4_XS_BYTES, partial, out);
}

static const int64_t kFmtBytes[8] = {FQ_Q4_K_BYTES, FQ_Q5_K_BYTES, FQ_Q6_K_BYTES, FQ_IQ4_XS_BYTES, FQ_IQ3_XXS_BYTES, FQ_IQ2_S_BYTES,
                                     FQ_Q8_0_BYTES, FQ_Q5_1_BYTES};

template <int FU>
static void fq_moe_kq_dispatch_d(int fd, torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                                 torch::Tensor gp, torch::Tensor up, torch::Tensor dp,
                                 int64_t gs, int64_t us, int64_t ds, torch::Tensor partial,
                                 torch::Tensor out, FqMoeSh sh = FqMoeSh())
{
    switch (fd)
    {
        case 0: fq_moe_kq_grouped_launch<FU, 0>(h, sel, wts, gp, up, dp, gs, us, ds, kFmtBytes[0], partial, out, sh); break;
        case 1: fq_moe_kq_grouped_launch<FU, 1>(h, sel, wts, gp, up, dp, gs, us, ds, kFmtBytes[1], partial, out, sh); break;
        case 2: fq_moe_kq_grouped_launch<FU, 2>(h, sel, wts, gp, up, dp, gs, us, ds, kFmtBytes[2], partial, out, sh); break;
        case 3: fq_moe_kq_grouped_launch<FU, 3>(h, sel, wts, gp, up, dp, gs, us, ds, kFmtBytes[3], partial, out, sh); break;
        case 4: fq_moe_kq_grouped_launch<FU, 4>(h, sel, wts, gp, up, dp, gs, us, ds, kFmtBytes[4], partial, out, sh); break;
        case 5: fq_moe_kq_grouped_launch<FU, 5>(h, sel, wts, gp, up, dp, gs, us, ds, kFmtBytes[5], partial, out, sh); break;
        case 6: fq_moe_kq_grouped_launch<FU, 6>(h, sel, wts, gp, up, dp, gs, us, ds, kFmtBytes[6], partial, out, sh); break;
        case 7: fq_moe_kq_grouped_launch<FU, 7>(h, sel, wts, gp, up, dp, gs, us, ds, kFmtBytes[7], partial, out, sh); break;
        default: TORCH_CHECK(false, "fq_moe_kq: bad down format ", fd);
    }
}

// Mixed per-projection formats (unsloth UD: gate/up Q4_K + down Q5_K). fmt ids: 0 q4_k, 1 q5_k, 2 q6_k, 3 iq4_xs, 4 iq3_xxs, 5 iq2_s.
static void fq_moe_kq_grouped_mixed_impl(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                                torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                                torch::Tensor down_ptrs,
                                int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                                int64_t fmt_ug, int64_t fmt_down,
                                torch::Tensor partial, torch::Tensor out, FqMoeSh sh)
{
    switch (fmt_ug)
    {
        case 0: fq_moe_kq_dispatch_d<0>((int)fmt_down, h, sel, wts, gate_ptrs, up_ptrs, down_ptrs, gate_stride, up_stride, down_stride, partial, out, sh); break;
        case 1: fq_moe_kq_dispatch_d<1>((int)fmt_down, h, sel, wts, gate_ptrs, up_ptrs, down_ptrs, gate_stride, up_stride, down_stride, partial, out, sh); break;
        case 2: fq_moe_kq_dispatch_d<2>((int)fmt_down, h, sel, wts, gate_ptrs, up_ptrs, down_ptrs, gate_stride, up_stride, down_stride, partial, out, sh); break;
        case 3: fq_moe_kq_dispatch_d<3>((int)fmt_down, h, sel, wts, gate_ptrs, up_ptrs, down_ptrs, gate_stride, up_stride, down_stride, partial, out, sh); break;
        case 4: fq_moe_kq_dispatch_d<4>((int)fmt_down, h, sel, wts, gate_ptrs, up_ptrs, down_ptrs, gate_stride, up_stride, down_stride, partial, out, sh); break;
        case 5: fq_moe_kq_dispatch_d<5>((int)fmt_down, h, sel, wts, gate_ptrs, up_ptrs, down_ptrs, gate_stride, up_stride, down_stride, partial, out, sh); break;
        default: TORCH_CHECK(false, "fq_moe_kq: bad gate/up format ", fmt_ug);
    }
}

void fq_moe_kq_grouped_mixed_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                                torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                                torch::Tensor down_ptrs,
                                int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                                int64_t fmt_ug, int64_t fmt_down,
                                torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_mixed_impl(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs, gate_stride, up_stride, down_stride,
                                 fmt_ug, fmt_down, partial, out, FqMoeSh());
}

// Same + the shared expert (Q8_0 gate / up [I, K] and down [K, I] packed rows, optional fp32 gate row [K]) in the same launches:
// partial needs S + R rows; out = routed + sigmoid(x . sh_wg) * shared (no gate: + shared).
void fq_moe_kq_grouped_mixed_sh_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                                   torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                                   torch::Tensor down_ptrs,
                                   int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                                   int64_t fmt_ug, int64_t fmt_down,
                                   torch::Tensor partial, torch::Tensor out,
                                   torch::Tensor sh_gate, torch::Tensor sh_up, torch::Tensor sh_down,
                                   torch::Tensor sh_wg)
{
    TORCH_CHECK(sh_gate.dtype() == torch::kUInt8 && sh_up.dtype() == torch::kUInt8 && sh_down.dtype() == torch::kUInt8,
                "fq_moe_kq: shared expert weights must be packed Q8_0 bytes");
    TORCH_CHECK(sh_gate.is_contiguous() && sh_up.is_contiguous() && sh_down.is_contiguous(), "fq_moe_kq: shared weights contiguous");
    FqMoeSh sh;
    sh.gate = sh_gate.data_ptr<uint8_t>();
    sh.up = sh_up.data_ptr<uint8_t>();
    sh.down = sh_down.data_ptr<uint8_t>();
    sh.ug_stride = (int64_t)(h.size(1) / 32) * FQ_Q8_0_BYTES;
    sh.d_stride = sh_down.numel() / h.size(1);
    if (sh_wg.numel() > 0)
    {
        TORCH_CHECK(sh_wg.dtype() == torch::kFloat && sh_wg.is_contiguous() && sh_wg.numel() == h.size(1), "fq_moe_kq: sh_wg [K] fp32");
        sh.wg = sh_wg.data_ptr<float>();
    }
    fq_moe_kq_grouped_mixed_impl(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs, gate_stride, up_stride, down_stride,
                                 fmt_ug, fmt_down, partial, out, sh);
}
