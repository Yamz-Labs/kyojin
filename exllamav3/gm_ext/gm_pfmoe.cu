// Generic GGUF MoE prefill experts, HIP / gfx1151 WMMA: fused dequant of the stored k-quant bytes + grouped fp16 GEMM over expert row blocks.
// Q4_K gate/up (+ SwiGLU) and Q4_K / Q6_K down, per-expert pointer tables (ForeignLinear.packed), no row floor: a block of 128 rows skips the
// 16-row groups beyond the expert's valid rows, so 30 rows per expert cost 2 of 8 WMMA groups. Weights are dequantised once per row block.
// Kernel structure and the Q4_K decode come from the Hy4 prefill agent's moe_hip.cu: fp16 operands,
// fp32 accumulate, fp16 dequant rounding, grid = (N tiles, row blocks) with the N tiles of one row block back to back.
//   block = 256 threads = 8 waves; tile 128 rows x 128 columns (each wave owns 16 columns of the tile for ALL 128 rows); K step 64.
//   A (activation rows gathered by sorted_pairs): global -> registers (one step ahead) -> LDS (XOR swizzle, double buffered).
//   B (weights): each lane loads the stored bytes of its column and dequantises its own 8 K values of every 16-wide k slice into registers;
//   v_permlanex16 supplies the other 8 (lanes 16-31 hold k 8-15 first, k 0-7 second; A is loaded in the same order).
#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cstdint>

typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 pf_h16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float pf_f8;

constexpr int PF_MT = 128, PF_KT = 64, PF_TH = 256;

__device__ __forceinline__ float pf_h2f(uint32_t b) { return (float)__builtin_bit_cast(_Float16, (uint16_t)b); }
__device__ __forceinline__ uint32_t pf_f2hb(float f) { return (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)f); }
__device__ __forceinline__ uint32_t pf_pk2(float a, float b) { return pf_f2hb(a) | (pf_f2hb(b) << 16); }
__device__ __forceinline__ uint32_t pf_perm16(uint32_t x) { return __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false); }
__device__ __forceinline__ uint32_t pf_funnel(uint32_t lo, uint32_t hi, uint32_t sh) { return __builtin_amdgcn_alignbit(hi, lo, sh); }

// 2-byte aligned loads of 4 / 8 bytes (Q6_K blocks are 210 bytes)
__device__ __forceinline__ uint32_t pf_ld4_a2(const uint8_t* p)
{
    uintptr_t a = (uintptr_t)p;
    const uint32_t* q = (const uint32_t*)(a & ~(uintptr_t)3);
    uint32_t x = q[0];
    if ((a & 2) == 0) return x;
    return pf_funnel(x, q[1], 16);
}
__device__ __forceinline__ uint2 pf_ld8_a2(const uint8_t* p)
{
    uintptr_t a = (uintptr_t)p;
    const uint32_t* q = (const uint32_t*)(a & ~(uintptr_t)3);
    uint2 r;
    if ((a & 2) == 0) { r = *(const uint2*)q; }
    else { uint32_t x = q[0], y = q[1], z = q[2]; r.x = pf_funnel(x, y, 16); r.y = pf_funnel(y, z, 16); }
    return r;
}

// stored-byte registers of one matrix for one K step (64 values). Q4_K: a (d, dmin, scales), q0 q1 (qs). Q6_K: q0..q3 (ql, one per k slice),
// m0 m1 (qh, slices 0/2 and 1/3), s0 (4 int8 scales), dd (d)
struct PfW { uint4 a; uint2 q0, q1, q2, q3, m0, m1; uint32_t s0, s1, dd; };   // FMT 3 (Q3_K): q0 q1 (2-bit qs), m0 m1 (hmask), s0 s1 (scale bytes), dd (d)
struct PfSc { float ds[4], dmn[4]; };

template <int FMT> __device__ __forceinline__ void pf_load(PfW& r, const uint8_t* wrow, int it, int h)
{
    const int kb = it >> 2, jq = it & 3;
    if (FMT == 0) {
        const uint8_t* blk = wrow + kb * 144;
        r.a = *(const uint4*)blk;
        r.q0 = *(const uint2*)(blk + 16 + 32 * jq + 8 * h);
        r.q1 = *(const uint2*)(blk + 16 + 32 * jq + 16 + 8 * h);
    } else if (FMT == 3) {       // Q3_K, 110-byte blocks: 32 hmask, 64 qs, 12 scale bytes, fp16 d
        const uint8_t* blk = wrow + kb * 110;
        const int half = jq >> 1;
        r.q0 = pf_ld8_a2(blk + 32 + half * 32 + 8 * h);
        r.q1 = pf_ld8_a2(blk + 32 + half * 32 + 16 + 8 * h);
        r.m0 = pf_ld8_a2(blk + 8 * h);
        r.m1 = pf_ld8_a2(blk + 16 + 8 * h);
        r.s0 = pf_ld4_a2(blk + 96 + 4 * (jq & 1));
        r.s1 = pf_ld4_a2(blk + 104);
        r.dd = pf_ld4_a2(blk + 108) & 0xffffu;
    } else {
        const uint8_t* blk = wrow + kb * 210;
        const int hf = jq >> 1, c = jq & 1;
        const uint8_t* ql = blk + hf * 64 + 8 * h;
        const uint8_t* qh = blk + 128 + hf * 32 + 8 * h;
        r.q0 = pf_ld8_a2(ql);
        r.q1 = pf_ld8_a2(ql + 16);
        r.q2 = pf_ld8_a2(ql + 32);
        r.q3 = pf_ld8_a2(ql + 48);
        r.m0 = pf_ld8_a2(qh);
        r.m1 = pf_ld8_a2(qh + 16);
        r.s0 = pf_ld4_a2(blk + 192 + hf * 8 + 4 * c);
        r.dd = (uint32_t)*(const uint16_t*)(blk + 208);
    }
}

template <int FMT> __device__ __forceinline__ void pf_scales(PfSc& c, const PfW& r, int it)
{
    const int jq = it & 3;
    if (FMT == 0) {
        const float d = pf_h2f(r.a.x & 0xffff), dm = pf_h2f(r.a.x >> 16);
        #pragma unroll
        for (int sbi = 0; sbi < 2; sbi++) {
            const int s = 2 * jq + sbi, jj = s & 3, sh = 8 * jj;
            uint32_t x0 = (r.a.y >> sh) & 255, x4 = (r.a.z >> sh) & 255, x8 = (r.a.w >> sh) & 255;
            uint32_t sc, mn;
            if (s >= 4) { sc = (x8 & 15) | ((x0 >> 6) << 4); mn = (x8 >> 4) | ((x4 >> 6) << 4); }
            else { sc = x0 & 63; mn = x4 & 63; }
            c.ds[sbi] = d * (float)sc;
            c.dmn[sbi] = dm * (float)mn;
        }
    } else if (FMT == 3) {       // slice ks -> scale idx 4 jq + ks: low nibble of byte 96 + (idx & 7), high 2 bits of byte 104 + (idx & 3); value = ((q * sc) * d)
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const uint32_t lo = ((r.s0 >> (8 * ks)) >> (4 * (jq >> 1))) & 15;
            const uint32_t hi = ((r.s1 >> (8 * ks)) >> (2 * jq)) & 3;
            c.ds[ks] = (float)((int)(lo | (hi << 4)) - 32);
            c.dmn[ks] = pf_h2f(r.dd);
        }
    } else {
        const float d = pf_h2f(r.dd);
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) c.ds[ks] = d * (float)(((int)(r.s0 << (24 - 8 * ks))) >> 24);
    }
}

// dequantised fp16 values of k slice ks (0..3 of the 64 step) for this lane's 8 k: 4 packed half2 dwords
template <int FMT> __device__ __forceinline__ void pf_dec(uint32_t o[4], const PfW& r, const PfSc& c, int ks, int jq)
{
    if (FMT == 0) {
        const int sbi = ks >> 1, t = ks & 1, sh0 = 4 * sbi;
        const uint2 q = t ? r.q1 : r.q0;
        const float ds = c.ds[sbi], dmn = c.dmn[sbi];
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            const uint32_t word = (p < 2) ? q.x : q.y;
            const int sh = 16 * (p & 1) + sh0;
            const float q0 = (float)((word >> sh) & 15), q1 = (float)((word >> (sh + 8)) & 15);
            o[p] = pf_pk2(fmaf(q0, ds, -dmn), fmaf(q1, ds, -dmn));
        }
    } else if (FMT == 3) {
        const int sbi = ks >> 1, t = ks & 1;
        const uint2 q = t ? r.q1 : r.q0, m = t ? r.m1 : r.m0;
        const int shq = 4 * (jq & 1) + 2 * sbi, shm = (2 * jq + sbi) & 7;
        const float sc = c.ds[ks], d = c.dmn[ks];
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            const uint32_t wq = (p < 2) ? q.x : q.y, wm = (p < 2) ? m.x : m.y;
            const int b0 = 16 * (p & 1), b1 = b0 + 8;
            const int v0 = (int)((wq >> (b0 + shq)) & 3) + 4 * (int)((wm >> (b0 + shm)) & 1) - 4;
            const int v1 = (int)((wq >> (b1 + shq)) & 3) + 4 * (int)((wm >> (b1 + shm)) & 1) - 4;
            o[p] = pf_pk2(((float)v0 * sc) * d, ((float)v1 * sc) * d);
        }
    } else {
        // value = d * sc * ((ql nibble c | 2 qh bits) - 32); chunk c = jq & 1 picks the ql nibble, qh bit pair 4 c (+2 for slices 2, 3)
        const int cc = jq & 1;
        const uint2 ql = ks == 0 ? r.q0 : ks == 1 ? r.q1 : ks == 2 ? r.q2 : r.q3;
        const uint2 qh = (ks & 1) ? r.m1 : r.m0;
        const int shl = 4 * cc, shh = 4 * cc + (ks >= 2 ? 2 : 0);
        const float ds = c.ds[ks];
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            const uint32_t wl = (p < 2) ? ql.x : ql.y, wh = (p < 2) ? qh.x : qh.y;
            const int b0 = 16 * (p & 1), b1 = b0 + 8;
            const int v0 = (int)(((wl >> (b0 + shl)) & 15) | (((wh >> (b0 + shh)) & 3) << 4)) - 32;
            const int v1 = (int)(((wl >> (b1 + shl)) & 15) | (((wh >> (b1 + shh)) & 3) << 4)) - 32;
            o[p] = pf_pk2((float)v0 * ds, (float)v1 * ds);
        }
    }
}

__device__ __forceinline__ pf_h16 pf_mk_frag(uint4 lo, uint4 hi)
{
    pf_h16 f; ((uint4*)&f)[0] = lo; ((uint4*)&f)[1] = hi; return f;
}
__device__ __forceinline__ pf_h16 pf_mk_bfrag(const uint32_t o[4])
{
    uint4 own = make_uint4(o[0], o[1], o[2], o[3]);
    uint4 oth = make_uint4(pf_perm16(o[0]), pf_perm16(o[1]), pf_perm16(o[2]), pf_perm16(o[3]));
    return pf_mk_frag(own, oth);
}

// GU = true: gate+up (two weight matrices, gathered rows of X, SwiGLU epilogue -> H); false: down (two column tiles per wave, rows of Hin,
// weighted fp32 scatter to OUT[pair]). Row groups of 16 beyond the block's valid rows are skipped (a wave-uniform branch).
template <int FMT, bool GU>
__device__ __forceinline__ void pf_body(const uint16_t* __restrict__ X, const uint8_t* const* __restrict__ W0, const uint8_t* const* __restrict__ W1,
                                        void* __restrict__ OUT, const float* __restrict__ wts, const int* __restrict__ sp, const int* __restrict__ be,
                                        int P, int topk, int E, float lim, int K, int N, int ROWB)
{
    constexpr int NM = 2;
    constexpr int CPB = GU ? 128 : 256;
    __shared__ uint4 sm[2][PF_MT * 8];
    const int bx = blockIdx.x, by = blockIdx.y;
    const int e = be[by];
    if (e >= E) return;
    const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, n = l & 15, h = l >> 4;
    const int ar = tid >> 1, ah = tid & 1;
    const int pair = sp[by * PF_MT + ar];
    const bool valid = pair < P;
    const int nv16 = (__syncthreads_count(valid) / 2 + 15) >> 4;     // valid rows are a prefix of the block
    const uint4* arow = (const uint4*)(X + (long)(GU ? (pair / topk) : (by * PF_MT + ar)) * K);
    const uint8_t* wr[NM];
    #pragma unroll
    for (int j = 0; j < NM; j++) {
        const long col = GU ? ((long)bx * CPB + w * 16 + n) : ((long)bx * CPB + w * 32 + 16 * j + n);
        wr[j] = ((GU && j == 1) ? W1[e] : W0[e]) + col * ROWB;
    }
    const int nit = K / PF_KT;

    pf_f8 acc[NM][8];
    #pragma unroll
    for (int j = 0; j < NM; j++)
        #pragma unroll
        for (int g = 0; g < 8; g++) acc[j][g] = (pf_f8){0, 0, 0, 0, 0, 0, 0, 0};

    uint4 ap[4];
    PfW wc[NM], wn[NM];
    #pragma unroll
    for (int i = 0; i < 4; i++) ap[i] = valid ? arow[4 * ah + i] : make_uint4(0, 0, 0, 0);
    #pragma unroll
    for (int j = 0; j < NM; j++) pf_load<FMT>(wc[j], wr[j], 0, h);

    for (int it = 0; it < nit; it++) {
        const int buf = it & 1;
        #pragma unroll
        for (int i = 0; i < 4; i++) sm[buf][ar * 8 + ((4 * ah + i) ^ (ar & 7))] = ap[i];
        __syncthreads();
        if (it + 1 < nit) {
            const int k8 = (it + 1) * (PF_KT / 8);
            #pragma unroll
            for (int i = 0; i < 4; i++) ap[i] = valid ? arow[k8 + 4 * ah + i] : make_uint4(0, 0, 0, 0);
            #pragma unroll
            for (int j = 0; j < NM; j++) pf_load<FMT>(wn[j], wr[j], it + 1, h);
        }
        PfSc c[NM];
        #pragma unroll
        for (int j = 0; j < NM; j++) pf_scales<FMT>(c[j], wc[j], it);
        const int jq = it & 3;
        const uint4* sA = sm[buf];
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            pf_h16 b[NM];
            #pragma unroll
            for (int j = 0; j < NM; j++) { uint32_t o[4]; pf_dec<FMT>(o, wc[j], c[j], ks, jq); b[j] = pf_mk_bfrag(o); }
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                if (g < nv16) {
                    const int r = g * 16 + n;
                    const uint4 lo = sA[r * 8 + ((2 * ks + h) ^ (r & 7))];
                    const uint4 hi = sA[r * 8 + ((2 * ks + (1 - h)) ^ (r & 7))];
                    const pf_h16 a = pf_mk_frag(lo, hi);
                    #pragma unroll
                    for (int j = 0; j < NM; j++) acc[j][g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[j], acc[j][g]);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < NM; j++) wc[j] = wn[j];
    }

    // epilogue: D lane l, reg i -> row 2 i + h, col n
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        if (g < nv16) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int rr = g * 16 + 2 * i + h;
                if (GU) {
                    const long col = (long)bx * CPB + w * 16 + n;
                    const float gg = fminf(acc[0][g][i], lim);
                    const float uu = fminf(fmaxf(acc[1][g][i], -lim), lim);
                    const float hv = gg / (1.0f + __expf(-gg)) * uu;
                    ((uint16_t*)OUT)[(long)(by * PF_MT + rr) * N + col] = (uint16_t)pf_f2hb(hv);
                } else {
                    const int pr = sp[by * PF_MT + rr];
                    if (pr < P) {
                        const float wv = wts[pr];
                        #pragma unroll
                        for (int j = 0; j < NM; j++) ((float*)OUT)[(long)pr * N + (long)bx * CPB + w * 32 + 16 * j + n] = acc[j][g][i] * wv;
                    }
                }
            }
        }
    }
}

template <int FMT>
__global__ __launch_bounds__(PF_TH) void gm_pf_gu_k(const uint16_t* X, const uint8_t* const* Wg, const uint8_t* const* Wu, uint16_t* H, const int* sp,
                                                    const int* be, int P, int topk, int E, float lim, int K, int N, int ROWB)
{ pf_body<FMT, true>(X, Wg, Wu, H, nullptr, sp, be, P, topk, E, lim, K, N, ROWB); }
template <int FMT>
__global__ __launch_bounds__(PF_TH) void gm_pf_down_k(const uint16_t* Hin, const uint8_t* const* Wd, float* Out, const float* wts, const int* sp,
                                                      const int* be, int P, int E, int K, int N, int ROWB)
{ pf_body<FMT, false>(Hin, Wd, nullptr, Out, wts, sp, be, P, 1, E, 0.f, K, N, ROWB); }

// x [R, K] fp16 -> H [maxb * 128, N] fp16 (SwiGLU applied, rows in the sorted block layout). sp / be = route_blocks tables (BM = 128).
void gm_pf_gate_up(const at::Tensor& x, const at::Tensor& gate_ptrs, const at::Tensor& up_ptrs, at::Tensor& H, const at::Tensor& sp, const at::Tensor& be,
                   int64_t P, int64_t topk, int64_t E, double lim, int64_t K, int64_t N, int64_t rowb, int64_t maxb, int64_t fmt)
{
    TORCH_CHECK(fmt == 0 || fmt == 3, "gm_pf_gate_up: Q4_K (0) or Q3_K (3)");
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kHalf && x.is_contiguous(), "gm_pf_gate_up: x must be contiguous half on the device");
    TORCH_CHECK(K % 256 == 0 && N % 128 == 0 && rowb == (K / 256) * (fmt == 0 ? 144 : 110), "gm_pf_gate_up: row bytes, K % 256, N % 128");
    TORCH_CHECK(H.numel() >= maxb * PF_MT * N && H.dtype() == at::kHalf, "gm_pf_gate_up: H too small");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto launch = fmt == 0 ? gm_pf_gu_k<0> : gm_pf_gu_k<3>;
    launch<<<dim3((unsigned)(N / 128), (unsigned)maxb), PF_TH, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(x.data_ptr()), reinterpret_cast<const uint8_t* const*>(gate_ptrs.data_ptr()),
        reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr()), reinterpret_cast<uint16_t*>(H.data_ptr()),
        reinterpret_cast<const int*>(sp.data_ptr()), reinterpret_cast<const int*>(be.data_ptr()), (int)P, (int)topk, (int)E, (float)lim, (int)K, (int)N, (int)rowb);
}

// H [maxb * 128, K] fp16 -> out [P, N] fp32 = routing weight * (H @ Wd^T), one row per (token, k) pair. fmt: 0 = Q4_K, 2 = Q6_K.
void gm_pf_down(const at::Tensor& H, const at::Tensor& down_ptrs, const at::Tensor& wts, const at::Tensor& sp, const at::Tensor& be, at::Tensor& out,
                int64_t P, int64_t E, int64_t K, int64_t N, int64_t rowb, int64_t fmt, int64_t maxb)
{
    TORCH_CHECK(fmt == 0 || fmt == 2, "gm_pf_down: Q4_K (0) or Q6_K (2)");
    TORCH_CHECK(K % 256 == 0 && N % 256 == 0 && rowb == (K / 256) * (fmt == 0 ? 144 : 210), "gm_pf_down: row bytes, K % 256, N % 256");
    TORCH_CHECK(out.numel() >= P * N && out.dtype() == at::kFloat && wts.dtype() == at::kFloat, "gm_pf_down: out fp32 [P, N], wts fp32");
    const at::cuda::OptionalCUDAGuard guard(H.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid((unsigned)(N / 256), (unsigned)maxb);
    auto a = std::make_tuple(reinterpret_cast<const uint16_t*>(H.data_ptr()), reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr()),
                             reinterpret_cast<float*>(out.data_ptr()), reinterpret_cast<const float*>(wts.data_ptr()),
                             reinterpret_cast<const int*>(sp.data_ptr()), reinterpret_cast<const int*>(be.data_ptr()));
    if (fmt == 0)
        gm_pf_down_k<0><<<grid, PF_TH, 0, stream>>>(std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a), std::get<4>(a), std::get<5>(a), (int)P, (int)E, (int)K, (int)N, (int)rowb);
    else
        gm_pf_down_k<2><<<grid, PF_TH, 0, stream>>>(std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a), std::get<4>(a), std::get<5>(a), (int)P, (int)E, (int)K, (int)N, (int)rowb);
}

#endif
