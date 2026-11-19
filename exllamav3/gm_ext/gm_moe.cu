// Generic GGUF MoE decode, one token (R = 1): Q4_K gate/up experts, Q4_K / Q6_K down experts, SwiGLU, no shared expert.
// Same arithmetic as exllamav3_ext/foreign_quant/fq_moe_kq.cu (fq_moe_kq_upgate_v2_kernel, fq_moe_kq_down_v2_kernel, fq_moe_kq_combine_kernel,
// fq_quant_q8_1_kernel): bit-identical results, fewer launches. What changes:
//   - the q8_1 quantization of the SwiGLU activation moves into the down kernel (each block quantizes the I values of its slot into shared memory);
//   - the weighted fold also adds the result into the residual stream (the `x += y` of the transformer block).
// 6 launches (quant, upgate, quant, down, combine, add) -> 4 (quant, upgate, down, combine + add).
#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <type_traits>

#define FQ_Q8_1_BYTES 36
#define FQ_Q4_K_BYTES 144
#define FQ_Q6_K_BYTES 210
#define FQ_Q3_K_BYTES 110
#define FQ_Q8_0_BYTES 34

__device__ __forceinline__ int fq_dp4a(int a, int b, int c)
{
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
      defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1200__) || defined(__gfx1201__))
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    return c + (int)a8[0] * (int)b8[0] + (int)a8[1] * (int)b8[1] + (int)a8[2] * (int)b8[2] + (int)a8[3] * (int)b8[3];
#endif
}

__device__ __forceinline__ uint32_t fq_vsub4(uint32_t a, uint32_t b)
{
    uint32_t v = 0;
    #pragma unroll
    for (int m = 0; m < 4; m++)
        v |= (((((a >> (8 * m)) & 0xFFu) - ((b >> (8 * m)) & 0xFFu)) & 0xFFu) << (8 * m));
    return v;
}

#include "gm_kq_item.cuh"

namespace {

constexpr int GM_ROWS = 32;        // output rows per block (8 threads per row)
constexpr int GM_PS = 33;
constexpr int GM_MAX_NB = 128;     // q8_1 blocks of the activation held in shared memory: I <= 4096

// One q8_1 block (32 values) by one thread, the arithmetic of fq_quant_q8_1_kernel: d = amax / 127, q = round-half-away(v / d), sum in index order.
__device__ __forceinline__ void gm_quant_block(const __half* __restrict__ x, uint8_t* __restrict__ dst)
{
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) { v[i] = __half2float(x[i]); amax = fmaxf(amax, fabsf(v[i])); }
    const float d = amax / 127.0f;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        int q = (amax == 0.0f) ? 0 : (int)roundf(v[i] / d);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        dst[4 + i] = (uint8_t)(int8_t)q;
        sum += v[i];
    }
    const __half dh = __float2half_rn(d);
    const __half sh = __float2half_rn(sum);
    uint16_t dhb, shb;
    memcpy(&dhb, &dh, 2);
    memcpy(&shb, &sh, 2);
    reinterpret_cast<uint16_t*>(dst)[0] = dhb;
    reinterpret_cast<uint16_t*>(dst)[1] = shb;
}

__global__ void gm_quant_rows_kernel(const __half* __restrict__ x, uint8_t* __restrict__ qb, int nb32)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb32) gm_quant_block(x + (int64_t)b * 32, qb + (int64_t)b * FQ_Q8_1_BYTES);
}

// fp32 activations (the Hy4 latent / shared paths): the arithmetic of fq_quant_q8_1_kernel<float> (same d, round-half-away, sum in index order).
__device__ __forceinline__ void gm_quant_block_f(const float* __restrict__ x, uint8_t* __restrict__ dst)
{
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) { v[i] = x[i]; amax = fmaxf(amax, fabsf(v[i])); }
    const float d = amax / 127.0f;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        int q = (amax == 0.0f) ? 0 : (int)roundf(v[i] / d);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        dst[4 + i] = (uint8_t)(int8_t)q;
        sum += v[i];
    }
    const __half dh = __float2half_rn(d);
    const __half sh = __float2half_rn(sum);
    uint16_t dhb, shb;
    memcpy(&dhb, &dh, 2);
    memcpy(&shb, &sh, 2);
    reinterpret_cast<uint16_t*>(dst)[0] = dhb;
    reinterpret_cast<uint16_t*>(dst)[1] = shb;
}

__global__ void gm_quant_rows_f_kernel(const float* __restrict__ x, uint8_t* __restrict__ qb, int nb32)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb32) gm_quant_block_f(x + (int64_t)b * 32, qb + (int64_t)b * FQ_Q8_1_BYTES);
}

// SwiGLU epilogue shared by the gate/up kernels: g / u rounded to half (Linear out_dtype), optional routed clamp (lim > 0: g <= lim, u in [-lim, lim]),
// silu in float, product back to half.
__device__ __forceinline__ __half gm_swiglu_act(float sg, float su, float lim)
{
    float gh = __half2float(__float2half_rn(sg));
    float uh = __half2float(__float2half_rn(su));
    if (lim > 0.0f) { gh = fminf(gh, lim); uh = fminf(fmaxf(uh, -lim), lim); }
    const float sig = 1.0f / (1.0f + expf(-gh));
    return __float2half_rn(sig * gh * uh);
}

// Expert-major groups (R >= 2): meta[u * GM_GRP] = number of slots of the u-th distinct expert, then those slot ids. Block group u loads the weight rows of its expert
// once per tile and runs every slot of the group on them (the weights stay in L1 / L2 for the repeated slots); without meta each slot is its own group.
constexpr int GM_GRP = 9;

__global__ void gm_meta_kernel(const int64_t* __restrict__ sel, int* __restrict__ meta, int SR, int E)
{
    __shared__ int ex[128];
    __shared__ int lead[128];
    const int s = threadIdx.x;
    if (s < SR)
    {
        int64_t e = sel[s];
        ex[s] = (int)((e < 0 || e >= E) ? 0 : e);
        meta[s * GM_GRP] = 0;
    }
    __syncthreads();
    int first = 0;
    if (s < SR)
    {
        first = 1;
        for (int p = 0; p < s; p++) if (ex[p] == ex[s]) first = 0;
        lead[s] = first;
    }
    __syncthreads();
    if (s < SR && first)
    {
        int rank = 0;
        for (int p = 0; p < s; p++) rank += lead[p];
        int cnt = 0;
        for (int p = s; p < SR; p++)
            if (ex[p] == ex[s] && cnt < GM_GRP - 1) meta[rank * GM_GRP + 1 + cnt++] = p;
        meta[rank * GM_GRP] = cnt;
    }
}

// Gate + up (Q4_K), one slot group per (blockIdx / ntiles) and 32 rows per block: fq_moe_kq_upgate_v2_kernel without the shared-expert slot and
// without the unaligned fallback (the host checks the alignment of every expert).
__global__ void __launch_bounds__(256) gm_upgate_q4k_kernel(const uint8_t* __restrict__ aqb, const int64_t* __restrict__ sel, const int* __restrict__ meta,
                                                            const uint8_t* const* __restrict__ gate_bases,
                                                            const uint8_t* const* __restrict__ up_bases,
                                                            int64_t gate_stride, int64_t up_stride,
                                                            __half* __restrict__ act, int I, int K, int E, int ntiles, int S_row, int qrow_bytes, float lim)
{
    __shared__ float pg[GM_ROWS * GM_PS];
    __shared__ float pu[GM_ROWS * GM_PS];
    const int u = blockIdx.x / ntiles;
    const int t = blockIdx.x % ntiles;
    const int ns = meta ? meta[u * GM_GRP] : 1;
    if (ns == 0) return;
    const int r = threadIdx.x >> 3, j = threadIdx.x & 7;
    const int n = t * GM_ROWS + r;
    int64_t e = sel[meta ? meta[u * GM_GRP + 1] : u];
    e = (e < 0 || e >= E) ? 0 : e;
    for (int m = 0; m < ns; m++)
    {
        const int s = meta ? meta[u * GM_GRP + 1 + m] : u;
        const uint8_t* aq = aqb + (int64_t)(s / S_row) * qrow_bytes;      // the q8_1 row of this slot's token (R rows, S_row slots each)
        float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (n < I)
        {
            const uint8_t* gr = gate_bases[e] + (int64_t)n * gate_stride;
            const uint8_t* ur = up_bases[e] + (int64_t)n * up_stride;
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
        #pragma unroll
        for (int q = 0; q < 4; q++)
        {
            pg[r * GM_PS + 4 * j + q] = ag[q];
            pu[r * GM_PS + 4 * j + q] = au[q];
        }
        __syncthreads();
        if (n < I && j == 0)
        {
            float sg = 0.0f, su = 0.0f;
            #pragma unroll
            for (int l = 0; l < 32; l++)
            {
                sg = __fadd_rn(sg, pg[r * GM_PS + l]);
                su = __fadd_rn(su, pu[r * GM_PS + l]);
            }
            act[(int64_t)s * I + n] = gm_swiglu_act(sg, su, lim);
        }
        __syncthreads();
    }
}

// Gate + up (Q3_K): the v1 lane layout of fq_moe_kq_upgate_kernel<8> (32 lane partials per row, summed in lane order), 8 rows per 256-thread block.
__global__ void __launch_bounds__(256) gm_upgate_q3k_kernel(const uint8_t* __restrict__ aqb, const int64_t* __restrict__ sel, const int* __restrict__ meta,
                                                            const uint8_t* const* __restrict__ gate_bases,
                                                            const uint8_t* const* __restrict__ up_bases,
                                                            int64_t gate_stride, int64_t up_stride,
                                                            __half* __restrict__ act, int I, int K, int E, int ntiles, int S_row, int qrow_bytes, float lim)
{
    __shared__ float red[256 * 2];
    const int u = blockIdx.x / ntiles;
    const int t = blockIdx.x % ntiles;
    const int ns = meta ? meta[u * GM_GRP] : 1;
    if (ns == 0) return;
    const int r = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = t * 8 + r;
    int64_t e = sel[meta ? meta[u * GM_GRP + 1] : u];
    e = (e < 0 || e >= E) ? 0 : e;
    for (int m = 0; m < ns; m++)
    {
        const int s = meta ? meta[u * GM_GRP + 1 + m] : u;
        const uint8_t* aq = aqb + (int64_t)(s / S_row) * qrow_bytes;
        float g = 0.0f, uu = 0.0f;
        if (n < I)
        {
            g = gm_dot_q3_k(gate_bases[e] + (int64_t)n * gate_stride, aq, K, lane, 32);
            uu = gm_dot_q3_k(up_bases[e] + (int64_t)n * up_stride, aq, K, lane, 32);
        }
        red[2 * threadIdx.x] = g;
        red[2 * threadIdx.x + 1] = uu;
        __syncthreads();
        if (n < I && lane == 0)
        {
            float sg = 0.0f, su = 0.0f;
            #pragma unroll
            for (int l = 0; l < 32; l++)
            {
                sg = __fadd_rn(sg, red[(r * 32 + l) * 2]);
                su = __fadd_rn(su, red[(r * 32 + l) * 2 + 1]);
            }
            act[(int64_t)s * I + n] = gm_swiglu_act(sg, su, lim);
        }
        __syncthreads();
    }
}

// Down (Q4_K = FMT 0, Q6_K = FMT 2): fq_moe_kq_down_v2_kernel with the q8_1 quantization of act[s] done in shared memory first.
template <int FMT>
__global__ void __launch_bounds__(256) gm_down_kernel(const __half* __restrict__ act, const int64_t* __restrict__ sel, const int* __restrict__ meta,
                                                      const uint8_t* const* __restrict__ down_bases, int64_t down_stride,
                                                      float* __restrict__ partial, int I, int K, int E, int ntiles)
{
    __shared__ float pp[GM_ROWS * GM_PS];
    __shared__ __align__(16) uint8_t qa[GM_MAX_NB * FQ_Q8_1_BYTES];
    const int u = blockIdx.x / ntiles;
    const int t = blockIdx.x % ntiles;
    const int ns = meta ? meta[u * GM_GRP] : 1;
    if (ns == 0) return;
    const int r = threadIdx.x >> 3, j = threadIdx.x & 7;
    const int nb = I / 32;
    int64_t e = sel[meta ? meta[u * GM_GRP + 1] : u];
    e = (e < 0 || e >= E) ? 0 : e;
    const int n = t * GM_ROWS + r;
    for (int m = 0; m < ns; m++)
    {
        const int s = meta ? meta[u * GM_GRP + 1 + m] : u;
        if ((int)threadIdx.x < nb) gm_quant_block(act + (int64_t)s * I + threadIdx.x * 32, qa + threadIdx.x * FQ_Q8_1_BYTES);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        __syncthreads();
        if (n < K)
        {
            const uint8_t* dr = down_bases[e] + (int64_t)n * down_stride;
            const int nitems = (I / 256) * ((FMT == 2) ? 8 : 4);
            for (int it = j; it < nitems; it += 8)
            {
                float tv[4];
                if constexpr (FMT == 0) fq_kq_item_q4_k(dr, qa, it, tv);
                else fq_kq_item_q6_k(dr, qa, it, tv);
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    acc[q] = __fadd_rn(acc[q], tv[q]);
            }
        }
        #pragma unroll
        for (int q = 0; q < 4; q++)
            pp[r * GM_PS + ((FMT == 2) ? (j + 8 * q) : (4 * j + q))] = acc[q];
        __syncthreads();
        if (n < K && j == 0)
        {
            float sp = 0.0f;
            #pragma unroll
            for (int l = 0; l < 32; l++)
                sp = __fadd_rn(sp, pp[r * GM_PS + l]);
            partial[(int64_t)s * K + n] = sp;
        }
        __syncthreads();
    }
}

// Weighted fold of the S slots (fq_moe_kq_combine_kernel), plus resid += out when resid != nullptr (torch's `x += y`: float add, rounded to the residual dtype).
template <typename RES_T>
__global__ void gm_combine_kernel(const float* __restrict__ partial, const __half* __restrict__ wts, float* __restrict__ out,
                                  RES_T* __restrict__ resid, int S, int K)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const int row = blockIdx.y;                      // token row: S slots of partial / wts, K outputs
    partial += (int64_t)row * S * K; wts += (int64_t)row * S; out += (int64_t)row * K;
    if (resid != nullptr) resid += (int64_t)row * K;
    float acc = 0.0f;
    for (int s = 0; s < S; s++)
        acc = __fadd_rn(acc, __fmul_rn(partial[(int64_t)s * K + k], __half2float(wts[s])));
    out[k] = acc;
    if (resid != nullptr)
    {
        if constexpr (std::is_same<RES_T, float>::value) resid[k] = resid[k] + acc;
        else resid[k] = __float2half_rn(__half2float(resid[k]) + acc);
    }
}

}  // namespace

// h [R, K] half, sel [R*S] int64, wts [R*S] half, expert pointer tables (int64 device addresses), strides in bytes,
// fmt_d 0 = Q4_K / 2 = Q6_K down, fmt_gu 0 = Q4_K / 8 = Q3_K gate/up, act [R*S, I] half scratch, qb [R*K/32*36] uint8 scratch, partial [R*S, K] float scratch,
// out [R, K] float (routed sum), resid [K] float / half or an empty tensor (no residual add), lim = routed swiglu clamp (0 = none: g <= lim, u in [-lim, lim]),
// meta = int32 scratch [>= R*S*9] switches on the expert-major order (R >= 2: every distinct expert's weights are read once per tile for all rows that picked it).
void gm_moe_kq_decode(const at::Tensor& h, const at::Tensor& sel, const at::Tensor& wts, const at::Tensor& gate_ptrs,
                      const at::Tensor& up_ptrs, const at::Tensor& down_ptrs, int64_t gate_stride, int64_t up_stride,
                      int64_t down_stride, int64_t fmt_d, at::Tensor& act, at::Tensor& qb, at::Tensor& partial,
                      at::Tensor& out, const at::Tensor& resid, int64_t fmt_gu, double lim, const at::Tensor& meta)
{
    TORCH_CHECK(h.is_cuda() && h.dtype() == at::kHalf && h.dim() == 2 && h.size(0) >= 1 && h.size(0) <= 8, "gm_moe_kq_decode: h must be [R, K] half on the device, R <= 8");
    TORCH_CHECK(fmt_d == 0 || fmt_d == 2, "gm_moe_kq_decode: down format must be Q4_K (0) or Q6_K (2)");
    TORCH_CHECK(fmt_gu == 0 || fmt_gu == 8, "gm_moe_kq_decode: gate/up format must be Q4_K (0) or Q3_K (8)");
    const at::cuda::OptionalCUDAGuard guard(h.device());
    const int K = (int)h.size(1);
    const int R = (int)h.size(0);
    TORCH_CHECK(sel.numel() % R == 0, "gm_moe_kq_decode: sel must hold R rows of top-k");
    const int S = (int)(sel.numel() / R);           // slots per row
    const int SR = R * S;                            // all slots
    const int E = (int)up_ptrs.numel();
    const int bb = fmt_d == 2 ? FQ_Q6_K_BYTES : FQ_Q4_K_BYTES;
    const int I = (int)(down_stride / bb) * 256;
    TORCH_CHECK(K % 256 == 0 && I % 256 == 0 && I / 32 <= GM_MAX_NB && S <= 32 && SR <= 128, "gm_moe_kq_decode: unsupported shape K=", K, " I=", I);
    const int64_t ug_bytes = fmt_gu == 8 ? FQ_Q3_K_BYTES : FQ_Q4_K_BYTES;
    TORCH_CHECK(gate_stride == (int64_t)(K / 256) * ug_bytes && up_stride == gate_stride, "gm_moe_kq_decode: gate/up stride does not match the gate/up format");
    TORCH_CHECK(act.numel() >= (int64_t)SR * I && qb.numel() >= (int64_t)R * (K / 32) * FQ_Q8_1_BYTES && partial.numel() >= (int64_t)SR * K && out.numel() >= (int64_t)R * K,
                "gm_moe_kq_decode: scratch too small");
    const bool grouped = R >= 2 && meta.numel() >= (int64_t)SR * GM_GRP;
    TORCH_CHECK(!grouped || meta.dtype() == at::kInt, "gm_moe_kq_decode: meta must be int32");
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uint8_t* qbp = reinterpret_cast<uint8_t*>(qb.data_ptr());
    const int nb32 = K / 32;
    gm_quant_rows_kernel<<<(R * nb32 + 63) / 64, 64, 0, stream>>>(reinterpret_cast<const __half*>(h.data_ptr()), qbp, R * nb32);
    const int64_t* selp = reinterpret_cast<const int64_t*>(sel.data_ptr());
    const int* metap = nullptr;
    if (grouped)
    {
        gm_meta_kernel<<<1, 128, 0, stream>>>(selp, reinterpret_cast<int*>(meta.data_ptr()), SR, E);
        metap = reinterpret_cast<const int*>(meta.data_ptr());
    }
    const int ntd = (K + GM_ROWS - 1) / GM_ROWS;
    __half* actp = reinterpret_cast<__half*>(act.data_ptr());
    const uint8_t* const* gp = reinterpret_cast<const uint8_t* const*>(gate_ptrs.data_ptr());
    const uint8_t* const* upp = reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr());
    if (fmt_gu == 0)
    {
        const int ntu = (I + GM_ROWS - 1) / GM_ROWS;
        gm_upgate_q4k_kernel<<<SR * ntu, 256, 0, stream>>>(qbp, selp, metap, gp, upp, gate_stride, up_stride, actp, I, K, E, ntu, S, nb32 * FQ_Q8_1_BYTES, (float)lim);
    }
    else
    {
        const int ntu = (I + 7) / 8;
        gm_upgate_q3k_kernel<<<SR * ntu, 256, 0, stream>>>(qbp, selp, metap, gp, upp, gate_stride, up_stride, actp, I, K, E, ntu, S, nb32 * FQ_Q8_1_BYTES, (float)lim);
    }
    float* partp = reinterpret_cast<float*>(partial.data_ptr());
    if (fmt_d == 0)
        gm_down_kernel<0><<<SR * ntd, 256, 0, stream>>>(actp, selp, metap, reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr()), down_stride, partp, I, K, E, ntd);
    else
        gm_down_kernel<2><<<SR * ntd, 256, 0, stream>>>(actp, selp, metap, reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr()), down_stride, partp, I, K, E, ntd);
    const __half* wp = reinterpret_cast<const __half*>(wts.data_ptr());
    float* outp = reinterpret_cast<float*>(out.data_ptr());
    const int cb = (K + 255) / 256;
    if (resid.numel() == 0)
        gm_combine_kernel<float><<<dim3(cb, R), 256, 0, stream>>>(partp, wp, outp, nullptr, S, K);
    else if (resid.dtype() == at::kFloat)
        gm_combine_kernel<float><<<dim3(cb, R), 256, 0, stream>>>(partp, wp, outp, reinterpret_cast<float*>(resid.data_ptr()), S, K);
    else
    {
        TORCH_CHECK(resid.dtype() == at::kHalf, "gm_moe_kq_decode: resid must be float or half");
        gm_combine_kernel<__half><<<dim3(cb, R), 256, 0, stream>>>(partp, wp, outp, reinterpret_cast<__half*>(resid.data_ptr()), S, K);
    }
    TORCH_CHECK(hipPeekAtLastError() == hipSuccess, "gm_moe_kq_decode: launch failed");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// q, k and v projections of one decode row in ONE launch (plus one q8_1 quantization): Q4_K / Q6_K rows, one row per 64-thread block,
// the arithmetic of fq_gemv_kq_rows_kernel<half, FMT, 1> (nt = number of v1 lane partials, as fq_kq_nt(K, 16)).
// The three matrices share the grid (blocks 0..N0-1 = matrix 0, ...), so the small k / v matrices run beside the large q one.
// ---------------------------------------------------------------------------------------------------------------------------------
struct Gm3
{
    const uint8_t* w[3];
    __half* out[3];
    int64_t stride[3];
    int N[3];
    int fmt[3];
    int R;
};

__global__ void __launch_bounds__(64) gm_gemv3_kernel(Gm3 a, const uint8_t* __restrict__ aqb, int K, int nt4, int nt6)
{
    __shared__ __align__(16) float part[260];
    const int row = blockIdx.y;
    aqb += (int64_t)row * (K / 32) * FQ_Q8_1_BYTES;
    int blk = blockIdx.x, mi = 0;
    if (blk >= a.N[0]) { blk -= a.N[0]; mi = 1; if (blk >= a.N[1]) { blk -= a.N[1]; mi = 2; } }
    const int j = threadIdx.x;
    const uint8_t* wr = a.w[mi] + (int64_t)blk * a.stride[mi];
    const int f = a.fmt[mi];
    const int nt = (f == 0) ? nt4 : nt6;      // fq_kq_nt(K, 16) for Q4_K, fq_kq_nt(K, 8) for Q6_K
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (f == 0)
    {
        const int nitems = (K / 256) * 4;
        for (int it = j; it < nitems; it += 64)
        {
            float t[4];
            fq_kq_item_q4_k(wr, aqb, it, t);
            #pragma unroll
            for (int q = 0; q < 4; q++) acc[q] = __fadd_rn(acc[q], t[q]);
        }
    }
    else
    {
        const int nitems = (K / 256) * 8;
        for (int it = j; it < nitems; it += 64)
        {
            float t[4];
            fq_kq_item_q6_k(wr, aqb, it, t);
            #pragma unroll
            for (int q = 0; q < 4; q++) acc[q] = __fadd_rn(acc[q], t[q]);
        }
    }
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        const int lane = (f == 0) ? (4 * j + q) : (32 * (j >> 3) + (j & 7) + 8 * q);
        if (lane < nt) part[lane] = acc[q];
    }
    __syncthreads();
    if (j == 0)
    {
        const float4* pv = reinterpret_cast<const float4*>(part);
        float s = 0.0f;
        #pragma unroll 8
        for (int l = 0; l < nt / 4; l++)
        {
            const float4 v = pv[l];
            s = __fadd_rn(s, v.x);
            s = __fadd_rn(s, v.y);
            s = __fadd_rn(s, v.z);
            s = __fadd_rn(s, v.w);
        }
        a.out[mi][(int64_t)row * a.N[mi] + blk] = __float2half_rn(s);
    }
}

// Q8_0 projections, 1..3 matrices of one launch: the arithmetic of fq_gemv_q8_q8_kernel (one block per output row, nt = fq_q8_nt(K) lane partials of
// fq_dot_q8_0_q8, summed in lane order). OUT = __half or float.
struct Gm3q
{
    const uint8_t* w[3];
    void* out[3];
    int N[3];
};

template <typename OUT>
__global__ void gm_gemv_q8_kernel(Gm3q a, const uint8_t* __restrict__ aqb, int K, int nt, int64_t row_bytes)
{
    extern __shared__ float red[];
    const int row = blockIdx.y;
    aqb += (int64_t)row * (K / 32) * FQ_Q8_1_BYTES;
    int blk = blockIdx.x, mi = 0;
    if (blk >= a.N[0]) { blk -= a.N[0]; mi = 1; if (blk >= a.N[1]) { blk -= a.N[1]; mi = 2; } }
    const int tid = threadIdx.x;
    red[tid] = gm_dot_q8_0(a.w[mi] + (int64_t)blk * row_bytes, aqb, K, tid, nt);
    __syncthreads();
    if (tid == 0)
    {
        float sum = 0.0f;
        for (int l = 0; l < nt; l++) sum = __fadd_rn(sum, red[l]);
        OUT* o = reinterpret_cast<OUT*>(a.out[mi]);
        if constexpr (std::is_same<OUT, __half>::value) o[(int64_t)row * a.N[mi] + blk] = __float2half_rn(sum);
        else o[(int64_t)row * a.N[mi] + blk] = sum;
    }
}

__device__ __forceinline__ float gm_to_f(float v) { return v; }
__device__ __forceinline__ float gm_to_f(__half v) { return __half2float(v); }

// SwiGLU of the shared expert fused into its Q8_0 down projection: act = half(silu(g) * u) (g, u half or float [R, I]), quantized to q8_1 in shared memory,
// then the Q8_0 dot of fq_gemv_q8_q8_kernel. accumulate: out += result (fp32 add), else out = result.
template <typename GT>
__global__ void gm_swiglu_down_q8_kernel(const GT* __restrict__ g, const GT* __restrict__ u, const uint8_t* __restrict__ wd, float* __restrict__ out,
                                         int I, int N, int nt, int64_t row_bytes, int accumulate)
{
    extern __shared__ float red[];
    __shared__ __align__(16) uint8_t qa[GM_MAX_NB * FQ_Q8_1_BYTES];
    const int row = blockIdx.y, n = blockIdx.x, tid = threadIdx.x;
    const int nb = I / 32;
    if (tid < nb)
    {
        __half a32[32];
        #pragma unroll
        for (int i = 0; i < 32; i++)
        {
            const float gv = gm_to_f(g[(int64_t)row * I + tid * 32 + i]);
            const float uv = gm_to_f(u[(int64_t)row * I + tid * 32 + i]);
            const float sig = 1.0f / (1.0f + expf(-gv));
            a32[i] = __float2half_rn(sig * gv * uv);
        }
        gm_quant_block(a32, qa + tid * FQ_Q8_1_BYTES);
    }
    __syncthreads();
    red[tid] = gm_dot_q8_0(wd + (int64_t)n * row_bytes, qa, I, tid, nt);
    __syncthreads();
    if (tid == 0)
    {
        float sum = 0.0f;
        for (int l = 0; l < nt; l++) sum = __fadd_rn(sum, red[l]);
        float* o = out + (int64_t)row * N + n;
        *o = accumulate ? __fadd_rn(*o, sum) : sum;
    }
}

// ws: 2 or 3 packed matrices [N_i, row bytes] uint8 (Q4_K = fmt 0 / Q6_K = fmt 2), out [1, sum N] half, x [1, K] half, qb scratch.
void gm_qkv_decode(const at::Tensor& x, const std::vector<at::Tensor>& ws, const std::vector<int64_t>& fmts, at::Tensor& out, at::Tensor& qb)
{
    TORCH_CHECK(x.is_cuda() && (x.dtype() == at::kHalf || x.dtype() == at::kFloat) && x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8, "gm_qkv_decode: x must be [R, K] half or float on the device, R <= 8");
    const int R = (int)x.size(0);
    TORCH_CHECK(ws.size() >= 1 && ws.size() <= 3 && ws.size() == fmts.size(), "gm_qkv_decode: 1..3 matrices");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int K = (int)x.size(1);
    const bool q8 = fmts[0] == 3;
    for (auto f : fmts) TORCH_CHECK((f == 3) == q8, "gm_qkv_decode: Q8_0 (3) matrices cannot mix with Q4_K / Q6_K in one launch");
    TORCH_CHECK(K % (q8 ? 32 : 256) == 0 && qb.numel() >= (int64_t)R * (K / 32) * FQ_Q8_1_BYTES, "gm_qkv_decode: K / scratch");
    TORCH_CHECK(!(x.dtype() == at::kFloat) || q8, "gm_qkv_decode: float x only with Q8_0");
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uint8_t* qbp = reinterpret_cast<uint8_t*>(qb.data_ptr());
    const int nb32 = K / 32;
    auto quant = [&]()
    {
        if (x.dtype() == at::kFloat) { gm_quant_rows_f_kernel<<<(R * nb32 + 63) / 64, 64, 0, stream>>>(x.data_ptr<float>(), qbp, R * nb32); }
        else { gm_quant_rows_kernel<<<(R * nb32 + 63) / 64, 64, 0, stream>>>(reinterpret_cast<const __half*>(x.data_ptr()), qbp, R * nb32); }
    };
    if (q8)
    {
        TORCH_CHECK(out.dtype() == at::kHalf || out.dtype() == at::kFloat, "gm_qkv_decode: out half or float for Q8_0");
        const int64_t rb = (int64_t)nb32 * FQ_Q8_0_BYTES;
        Gm3q a;
        memset(&a, 0, sizeof(a));
        int total = 0;
        const size_t esz = out.dtype() == at::kHalf ? 2 : 4;
        char* o = reinterpret_cast<char*>(out.data_ptr());
        for (size_t i = 0; i < ws.size(); i++)
        {
            TORCH_CHECK(ws[i].dim() == 2 && ws[i].size(1) == rb && ws[i].stride(1) == 1 && ws[i].stride(0) == rb, "gm_qkv_decode: packed Q8_0 matrix shape");
            TORCH_CHECK((reinterpret_cast<uintptr_t>(ws[i].data_ptr()) & 1) == 0, "gm_qkv_decode: unaligned weights");
            a.w[i] = reinterpret_cast<const uint8_t*>(ws[i].data_ptr());
            a.N[i] = (int)ws[i].size(0);
            a.out[i] = o + esz * (int64_t)R * total;
            total += a.N[i];
        }
        TORCH_CHECK(out.numel() >= (int64_t)R * total, "gm_qkv_decode: out too small");
        const int nt = nb32 >= 256 ? 256 : ((nb32 + 31) / 32) * 32;
        quant();
        if (out.dtype() == at::kHalf) { gm_gemv_q8_kernel<__half><<<dim3(total, R), nt, nt * sizeof(float), stream>>>(a, qbp, K, nt, rb); }
        else { gm_gemv_q8_kernel<float><<<dim3(total, R), nt, nt * sizeof(float), stream>>>(a, qbp, K, nt, rb); }
        TORCH_CHECK(hipPeekAtLastError() == hipSuccess, "gm_qkv_decode: launch failed");
        return;
    }
    Gm3 a;
    memset(&a, 0, sizeof(a));
    int total = 0;
    __half* o = reinterpret_cast<__half*>(out.data_ptr());
    for (size_t i = 0; i < ws.size(); i++)
    {
        TORCH_CHECK(fmts[i] == 0 || fmts[i] == 2, "gm_qkv_decode: Q4_K (0), Q6_K (2) or Q8_0 (3) only");
        const int64_t rb = (int64_t)(K / 256) * (fmts[i] == 0 ? FQ_Q4_K_BYTES : FQ_Q6_K_BYTES);
        TORCH_CHECK(ws[i].dim() == 2 && ws[i].size(1) == rb && ws[i].stride(1) == 1 && ws[i].stride(0) == rb, "gm_qkv_decode: packed matrix shape");
        const uintptr_t am = fmts[i] == 0 ? 15 : 1;
        TORCH_CHECK((reinterpret_cast<uintptr_t>(ws[i].data_ptr()) & am) == 0 && (rb & am) == 0, "gm_qkv_decode: unaligned weights");
        a.w[i] = reinterpret_cast<const uint8_t*>(ws[i].data_ptr());
        a.stride[i] = rb;
        a.N[i] = (int)ws[i].size(0);
        a.fmt[i] = (int)fmts[i];
        a.out[i] = o + (int64_t)R * total;            // q, k, v blocks one after the other, each [R, N_i] contiguous (R = 1: the old layout)
        total += a.N[i];
    }
    a.R = R;
    TORCH_CHECK(out.numel() >= (int64_t)R * total, "gm_qkv_decode: out too small");
    auto ntf = [](int nsub) { return nsub >= 256 ? 256 : ((nsub + 31) / 32) * 32; };
    const int nt4 = ntf(K / 16), nt6 = ntf(K / 8);
    quant();
    gm_gemv3_kernel<<<dim3(total, R), 64, 0, stream>>>(a, qbp, K, nt4, nt6);
    TORCH_CHECK(hipPeekAtLastError() == hipSuccess, "gm_qkv_decode: launch failed");
}

// Shared expert down projection with the SwiGLU fused in: g, u [R, I] (half or float, e.g. the two outputs of a Q8_0 gm_qkv_decode), wd packed Q8_0 [N, I/32*34],
// out float [R, N]; accumulate = out += result (add the shared expert onto the routed up-projection output of a previous gm_qkv_decode call).
void gm_q8_swiglu_down(const at::Tensor& g, const at::Tensor& u, const at::Tensor& wd, at::Tensor& out, bool accumulate)
{
    TORCH_CHECK(g.is_cuda() && g.dim() == 2 && g.sizes() == u.sizes() && g.dtype() == u.dtype() && (g.dtype() == at::kHalf || g.dtype() == at::kFloat) && g.is_contiguous() && u.is_contiguous(),
                "gm_q8_swiglu_down: g, u [R, I] contiguous half or float");
    const at::cuda::OptionalCUDAGuard guard(g.device());
    const int R = (int)g.size(0), I = (int)g.size(1), N = (int)wd.size(0);
    const int nb = I / 32;
    const int64_t rb = (int64_t)nb * FQ_Q8_0_BYTES;
    TORCH_CHECK(R <= 8 && I % 32 == 0 && nb <= GM_MAX_NB && wd.dim() == 2 && wd.size(1) == rb && wd.is_contiguous() && (reinterpret_cast<uintptr_t>(wd.data_ptr()) & 1) == 0,
                "gm_q8_swiglu_down: shape / alignment");
    TORCH_CHECK(out.dtype() == at::kFloat && out.numel() >= (int64_t)R * N, "gm_q8_swiglu_down: out float [R, N]");
    const int nt = nb >= 256 ? 256 : ((nb + 31) / 32) * 32;
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (g.dtype() == at::kHalf)
    {
        gm_swiglu_down_q8_kernel<__half><<<dim3(N, R), nt, nt * sizeof(float), stream>>>(reinterpret_cast<const __half*>(g.data_ptr()), reinterpret_cast<const __half*>(u.data_ptr()),
            reinterpret_cast<const uint8_t*>(wd.data_ptr()), out.data_ptr<float>(), I, N, nt, rb, accumulate ? 1 : 0);
    }
    else
    {
        gm_swiglu_down_q8_kernel<float><<<dim3(N, R), nt, nt * sizeof(float), stream>>>(g.data_ptr<float>(), u.data_ptr<float>(),
            reinterpret_cast<const uint8_t*>(wd.data_ptr()), out.data_ptr<float>(), I, N, nt, rb, accumulate ? 1 : 0);
    }
    TORCH_CHECK(hipPeekAtLastError() == hipSuccess, "gm_q8_swiglu_down: launch failed");
}

#endif // USE_ROCM
