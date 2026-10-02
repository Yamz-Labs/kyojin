// kdaC7: KDA inter-solve off-diagonal dots, one program per (head, 64-row chunk); head-fastest grid.
// Each program reads its 64 k/g/q rows once in full cache lines (wave = whole rows), writes the gated operands
// a = x * exp2(g_x - g_n) and b = k_j * exp2(g_n - g_j) per 32-wide K sub-chunk to LDS, and every output runs the
// same sequential fp32 FMA chain over k as the Triton FMA dot (bitwise equal to inter_off_kernel BK16 w2).
// The head-fastest grid matters: q/k/g rows are 16/32 KB apart, so a time-fastest grid lands every concurrent
// program on the same memory channels (kdaC7: 3.27 -> 0.88 ms at T2048).
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#include "kda_inter.cuh"

#if defined(USE_ROCM)

#define BT 64
#define KD 128
#define NTH 256
#define KC 32
#define LDP (KC + 4)
#define NSC (KD / KC)

__device__ __forceinline__ float kio_bf2f(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
__device__ __forceinline__ uint16_t kio_f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__global__ __launch_bounds__(NTH) void kda_inter_off_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const float* __restrict__ g,
    const uint16_t* __restrict__ beta, uint16_t* __restrict__ Aqk, float* __restrict__ Aoff, float scale, int T,
    int H, int HV)
{
    const int i_hv = blockIdx.x, i_t = blockIdx.y, i_b = blockIdx.z;
    const int i_h = i_hv / (HV / H);
    const long bos = (long)i_b * T;
    const int tc0 = i_t * BT;
    const long sk = (long)H * KD, sg = (long)HV * KD, sa = (long)HV * BT;
    q += (bos * H + i_h) * KD;
    k += (bos * H + i_h) * KD;
    g += (bos * HV + i_hv) * KD;
    Aqk += (bos * HV + i_hv) * BT;
    Aoff += (bos * HV + i_hv) * BT;
    beta += bos * HV + i_hv;

    __shared__ float sA[6][16][LDP];   // (I-1)*2 + {0: q, 1: k}
    __shared__ float sB[96][LDP];      // b rows: I=1 -> 0..15, I=2 -> 16..47, I=3 -> 48..95
    const int tid = threadIdx.x;
    const int r = tid >> 2, p = tid & 3;
    const int row = tc0 + r;
    const bool rv = row < T;

    // one wave instruction reads whole rows: lane covers k = 4*lane .. +3, wave w rows w, w+8, ..
    const int lane = tid & 31, wv = tid >> 5;
    float4 G4[8]; uint2 K4[8], Q4[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int rr = tc0 + wv + 8 * i;
        const int rc = rr < T ? rr : T - 1;
        G4[i] = *(const float4*)&g[rc * sg + 4 * lane];
        K4[i] = *(const uint2*)&k[rc * sk + 4 * lane];
        Q4[i] = *(const uint2*)&q[rc * sk + 4 * lane];
    }
    float4 GNf[3];
#pragma unroll
    for (int I = 1; I <= 3; ++I) {
        const int rn = tc0 + 16 * I < T ? tc0 + 16 * I : T - 1;
        GNf[I - 1] = *(const float4*)&g[rn * sg + 4 * lane];
    }
    const int x = tid >> 4, jj = tid & 15;
    // blocks: 0:(1,0) 1:(2,0) 2:(2,1) 3:(3,0) 4:(3,1) 5:(3,2)
    float aq[6], ak[6];
#pragma unroll
    for (int b = 0; b < 6; ++b) { aq[b] = 0.f; ak[b] = 0.f; }

#pragma unroll
    for (int c = 0; c < NSC; ++c) {
        if ((lane >> 3) == c) {
            const int ko = (lane & 7) * 4;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int rr = wv + 8 * i;
                const bool rvv = tc0 + rr < T;
                const float gq[4] = {G4[i].x, G4[i].y, G4[i].z, G4[i].w};
                const float kq[4] = {kio_bf2f(K4[i].x & 0xffff), kio_bf2f(K4[i].x >> 16), kio_bf2f(K4[i].y & 0xffff), kio_bf2f(K4[i].y >> 16)};
                const float qq[4] = {kio_bf2f(Q4[i].x & 0xffff), kio_bf2f(Q4[i].x >> 16), kio_bf2f(Q4[i].y & 0xffff), kio_bf2f(Q4[i].y >> 16)};
#pragma unroll
                for (int I = 1; I <= 3; ++I) {
                    if (tc0 + 16 * I < T && rr < 16 * I + 16) {
                        const float gn[4] = {GNf[I - 1].x, GNf[I - 1].y, GNf[I - 1].z, GNf[I - 1].w};
                        if (rr >= 16 * I) {
                            float4 va, vk;
                            float* pa = (float*)&va; float* pk = (float*)&vk;
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const float ex = rvv ? __builtin_amdgcn_exp2f(gq[e] - gn[e]) : 0.f;
                                pa[e] = qq[e] * ex; pk[e] = kq[e] * ex;
                            }
                            *(float4*)&sA[(I - 1) * 2][rr - 16 * I][ko] = va;
                            *(float4*)&sA[(I - 1) * 2 + 1][rr - 16 * I][ko] = vk;
                        } else {
                            const int ob = (I == 1) ? 0 : (I == 2 ? 16 : 48);
                            float4 vb; float* pb = (float*)&vb;
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                pb[e] = rvv ? kq[e] * __builtin_amdgcn_exp2f(gn[e] - gq[e]) : 0.f;
                            *(float4*)&sB[ob + rr][ko] = vb;
                        }
                    }
                }
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int kk = 0; kk < KC; kk += 4) {
            float4 a[6], bb[6];
#pragma unroll
            for (int s = 0; s < 6; ++s) a[s] = *(const float4*)&sA[s][x][kk];
            bb[0] = *(const float4*)&sB[jj][kk];
            bb[1] = *(const float4*)&sB[16 + jj][kk];
            bb[2] = *(const float4*)&sB[32 + jj][kk];
            bb[3] = *(const float4*)&sB[48 + jj][kk];
            bb[4] = *(const float4*)&sB[64 + jj][kk];
            bb[5] = *(const float4*)&sB[80 + jj][kk];
            const int ai[6] = {0, 1, 1, 2, 2, 2};
#pragma unroll
            for (int b = 0; b < 6; ++b) {
                const float4 A = a[ai[b] * 2], Kx = a[ai[b] * 2 + 1], B = bb[b];
                float* Q = &aq[b]; float* Kp = &ak[b];
                *Q = __builtin_fmaf(A.x, B.x, *Q); *Q = __builtin_fmaf(A.y, B.y, *Q);
                *Q = __builtin_fmaf(A.z, B.z, *Q); *Q = __builtin_fmaf(A.w, B.w, *Q);
                *Kp = __builtin_fmaf(Kx.x, B.x, *Kp); *Kp = __builtin_fmaf(Kx.y, B.y, *Kp);
                *Kp = __builtin_fmaf(Kx.z, B.z, *Kp); *Kp = __builtin_fmaf(Kx.w, B.w, *Kp);
            }
        }
        __syncthreads();
    }

    const int bI[6] = {1, 2, 2, 3, 3, 3}, bJ[6] = {0, 0, 1, 0, 1, 2};
#pragma unroll
    for (int b = 0; b < 6; ++b) {
        const int ro = tc0 + 16 * bI[b] + x;
        if (tc0 + 16 * bI[b] < T && ro < T) {
            Aqk[ro * sa + 16 * bJ[b] + jj] = kio_f2bf(aq[b] * scale);
            Aoff[ro * sa + 16 * bJ[b] + jj] = ak[b] * kio_bf2f(beta[(long)ro * HV]);
        }
    }
}

void kda_inter_off(const at::Tensor& q, const at::Tensor& k, const at::Tensor& g, const at::Tensor& beta,
                   at::Tensor& Aqk, at::Tensor& Aoff, double scale)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && g.is_contiguous() && beta.is_contiguous() &&
                Aqk.is_contiguous() && Aoff.is_contiguous(), "kda_inter_off: tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                beta.scalar_type() == at::kBFloat16 && Aqk.scalar_type() == at::kBFloat16 &&
                g.scalar_type() == at::kFloat && Aoff.scalar_type() == at::kFloat, "kda_inter_off: dtypes");
    const int B = k.size(0), T = k.size(1), H = k.size(2), HV = g.size(2);
    TORCH_CHECK(k.size(3) == KD && g.size(3) == KD && Aqk.size(3) == BT && Aoff.size(3) == BT, "kda_inter_off: shapes");
    if (T == 0) return;
    dim3 grid(HV, (T + BT - 1) / BT, B);
    kda_inter_off_kernel<<<grid, NTH, 0, stream>>>(
        (const uint16_t*) q.data_ptr(), (const uint16_t*) k.data_ptr(), (const float*) g.data_ptr(),
        (const uint16_t*) beta.data_ptr(), (uint16_t*) Aqk.data_ptr(), (float*) Aoff.data_ptr(), (float) scale, T, H, HV);
}

#undef BT
#undef KD
#undef NTH
#undef KC
#undef LDP
#undef NSC

#else

void kda_inter_off(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&,
                   double)
{
    TORCH_CHECK(false, "kda_inter_off: ROCm only");
}

#endif
