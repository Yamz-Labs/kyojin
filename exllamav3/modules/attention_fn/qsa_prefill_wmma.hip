// Qwen QSA sparse prefill attention, 4-row union variant for gfx11.5 WMMA f16 (wave32, 8 waves per block).
// Structure after Gufo's WmmaCausalAttentionKernel (https://github.com/gufo-org/gufo, MIT License, Copyright (c) 2026 Gufo contributors):
// one block serves several consecutive query rows of one kv head, walks the UNION of their selected key blocks in 16-key tiles (each K / V
// tile is gathered from global memory once for all rows), prefetches the next K tile / this V tile in registers behind the matrix work, and
// stages V transposed in LDS. Own code: the union is built in LDS (bitonic sort + dedupe) from the per-row index lists, rows keep their own
// selection through a per-(block, row) validity mask, softmax is online per (row, head) column in fp32.
// Block = 4 rows x 12 heads = 48 columns = 3 WMMA column tiles. Selection lists are 520 groups of 4 consecutive tokens (one 4-token block each).
// Fragment layout (gfx1151): A: lane l, elem r -> A[l & 15][r]; B: lane l, elem r -> B[r][l & 15]; C: lane l, reg r -> C[2r + (l >> 4)][l & 15]
// MEMORY SAFETY: global addresses are clamped (TOKC, PGC, row clamp); stores are guarded by the row bound.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 half_t;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

#define HD 256
#define GRP 12
#define NKV 2
#define PAGE 256
#define KPAD 2080
#define NGRP (KPAD / 4)        // 520 groups of 4 tokens per row
#define NR 4                   // rows per block
#define NCOL 48
#define LDQ 264                // q / K / out pitch (halves)
#define LDT 24                 // V^T and P pitch (halves): 16 keys + pad
#define SORTN 4096
#define MAXU 2096              // union entries (<= 2080) + pad

#define IX_Q(r_, h_, d_, p_) (((int64_t) (r_)) * (p_) + (h_) * HD + (d_))
#define IX_O(r_, h_, d_, p_) (((int64_t) (r_)) * (p_) + (h_) * HD + (d_))
#define IX_IDX(r_, n_) (((int64_t) (r_)) * KPAD + (n_))
#define PGC(pg_, NP_) min(max((pg_), 0), (NP_) - 1)
#define TOKC(t_, NT_) min(max((t_), 0), (NT_) - 1)
#define IX_KV(t_, g_, d_, p_) (((int64_t) (t_)) * (p_) + (g_) * HD + (d_))

__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
__device__ __forceinline__ v16h ld16h(const half_t* p) {
    v4u a = *(const v4u*) p, b = *(const v4u*) (p + 8);
    union { v4u u[2]; v16h h; } x; x.u[0] = a; x.u[1] = b; return x.h;
}
__device__ __forceinline__ __attribute__((convergent)) void sync_lds() {   // LDS-only exchange: do not drain global loads
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
}

extern "C" __global__ __launch_bounds__(256, 2) void qsa_pf_wmma(
    const half_t* __restrict__ q, const half_t* __restrict__ kc, const half_t* __restrict__ vc,
    const int* __restrict__ bt, const int* __restrict__ idx, half_t* __restrict__ o,
    int q_stride, int o_stride, int tok_stride, int ntok, int npages, int R, float scale)
{
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, n = lane & 15, hh = lane >> 4;
    const int g = blockIdx.x & 1, row0 = (blockIdx.x >> 1) * NR;

    __shared__ uint32_t sBlk[MAXU];
    __shared__ uint32_t sVm[MAXU];                       // bit 4 * row + (token & 3): token visible to that row
    __shared__ __attribute__((aligned(16))) char sBig[NCOL * LDQ * 2];   // sort buffer, then q, then output staging
    __shared__ __attribute__((aligned(16))) half_t sKV[256 * LDT];       // K tile [16][LDQ] (first 8448 B) or V^T [256][LDT]
    __shared__ __attribute__((aligned(16))) half_t sP[NCOL * LDT];
    __shared__ float sAlpha[NCOL];
    __shared__ float sL[NCOL];
    __shared__ __attribute__((aligned(16))) float sSp[3 * 32 * 8];
    __shared__ uint32_t sCnt[256];

    // ---- union of the 4 rows' selected blocks ----
    uint32_t* sSort = (uint32_t*) sBig;
    for (int e = tid; e < SORTN; e += 256) {
        uint32_t key = 0xFFFFFFFFu;
        if (e < NR * NGRP) {
            const int rho = e / NGRP, j = e - rho * NGRP;
            const int row = min(row0 + rho, R - 1);
            const int4 t = *(const int4*) (idx + IX_IDX(row, 4 * j));
            const uint32_t nib = (t.x >= 0 ? 1u : 0u) | (t.y >= 0 ? 2u : 0u) | (t.z >= 0 ? 4u : 0u) | (t.w >= 0 ? 8u : 0u);
            if (nib) {
                const int f = t.x >= 0 ? t.x : (t.y >= 0 ? t.y : (t.z >= 0 ? t.z : t.w));
                key = (((uint32_t) (f >> 2)) << 6) | ((uint32_t) rho << 4) | nib;
            }
        }
        sSort[e] = key;
    }
    for (int k = 2; k <= SORTN; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            sync_lds();
            for (int p = tid; p < SORTN / 2; p += 256) {
                const int i = ((p & ~(j - 1)) << 1) | (p & (j - 1)), i2 = i | j;
                const uint32_t a = sSort[i], b = sSort[i2];
                const bool up = (i & k) == 0;
                if ((a > b) == up) { sSort[i] = b; sSort[i2] = a; }
            }
        }
    sync_lds();
    {
        uint32_t cnt = 0;
        for (int e = tid * 16; e < tid * 16 + 16; ++e) {
            const uint32_t k = sSort[e];
            if (k != 0xFFFFFFFFu && (e == 0 || (sSort[e - 1] >> 6) != (k >> 6))) ++cnt;
        }
        sCnt[tid] = cnt;
        for (int off = 1; off < 256; off <<= 1) {
            sync_lds();
            const uint32_t v = tid >= off ? sCnt[tid - off] : 0u;
            sync_lds();
            sCnt[tid] += v;
        }
        sync_lds();
        uint32_t pos = sCnt[tid] - cnt;
        for (int e = tid * 16; e < tid * 16 + 16; ++e) {
            const uint32_t k = sSort[e];
            if (k != 0xFFFFFFFFu && (e == 0 || (sSort[e - 1] >> 6) != (k >> 6))) {
                uint32_t vm = 0;
                for (int u = e; u < e + 4 && u < SORTN && sSort[u] != 0xFFFFFFFFu && (sSort[u] >> 6) == (k >> 6); ++u)
                    vm |= (sSort[u] & 15u) << (4 * ((sSort[u] >> 4) & 3));
                sBlk[pos] = k >> 6; sVm[pos] = vm; ++pos;
            }
        }
    }
    sync_lds();
    const int nU = (int) sCnt[255];
    if (tid < 8) { sBlk[nU + tid] = 0; sVm[nU + tid] = 0; }
    sync_lds();
    const int ntile = (nU + 3) >> 2;

    // ---- q -> LDS ----
    half_t* sQ = (half_t*) sBig;
    for (int c = tid; c < NCOL * 32; c += 256) {
        const int col = c >> 5, ch = c & 31, rho = col / GRP, h = col - rho * GRP;
        const int row = min(row0 + rho, R - 1);
        *(v4u*) (sQ + col * LDQ + ch * 8) = *(const v4u*) (q + IX_Q(row, g * GRP + h, ch * 8, q_stride));
    }
    sync_lds();

    v8f acc[2][3];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 3; ++b) acc[a][b] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -__builtin_inff(), l = 0.0f;
    const float sl2 = scale * 1.44269504088896341f;
    const int kk = tid & 15, dch = tid >> 4;

    auto key_tok = [&](int i) -> int {
        const int blk = (int) sBlk[4 * i + (kk >> 2)];
        const int t0 = blk * 4;
        const int pg = bt[PGC(t0 / PAGE, npages)];
        return TOKC(pg * PAGE + (t0 % PAGE) + (kk & 3), ntok);
    };
    v4u kr0 = {0, 0, 0, 0}, kr1 = {0, 0, 0, 0}, vr0, vr1;
    if (ntile > 0) {
        const int tk = key_tok(0);
        kr0 = *(const v4u*) (kc + IX_KV(tk, g, dch * 16, tok_stride));
        kr1 = *(const v4u*) (kc + IX_KV(tk, g, dch * 16 + 8, tok_stride));
    }
    half_t* sK = sKV;

    for (int i = 0; i < ntile; ++i) {
        *(v4u*) (sK + kk * LDQ + dch * 16) = kr0;
        *(v4u*) (sK + kk * LDQ + dch * 16 + 8) = kr1;
        sync_lds();                                              // K tile visible
        {
            const int tk = key_tok(i);
            const uint32_t vm = sVm[4 * i + (kk >> 2)];
            const bool any = ((vm >> (kk & 3)) & 0x1111u) != 0u;
            vr0 = (v4u) {0, 0, 0, 0}; vr1 = (v4u) {0, 0, 0, 0};
            if (any) {
                vr0 = *(const v4u*) (vc + IX_KV(tk, g, dch * 16, tok_stride));
                vr1 = *(const v4u*) (vc + IX_KV(tk, g, dch * 16 + 8, tok_stride));
            }
        }
        v8f S = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
        if (w < 6) {                                             // scores: wave = (column tile, half of the head dim)
            const int ct = w % 3, dh = w / 3;
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int kq = dh * 8 + ks;
                S = wmma(ld16h(sK + n * LDQ + 16 * kq), ld16h(sQ + (16 * ct + n) * LDQ + 16 * kq), S);
            }
            if (dh) *(v8f*) (sSp + (ct * 32 + lane) * 8) = S;
        }
        sync_lds();                                              // partial scores of the second head-dim half
        if (w < 3) {
            S += *(const v8f*) (sSp + (w * 32 + lane) * 8);
            const int col = 16 * w + n, rho = col / GRP;
            float s[8], mx = -__builtin_inff();
            bool ok[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int kq = 2 * r + hh;
                const uint32_t vm = sVm[4 * i + (kq >> 2)];
                ok[r] = (vm >> (4 * rho + (kq & 3))) & 1u;
                s[r] = ok[r] ? S[r] * sl2 : -__builtin_inff();
                mx = fmaxf(mx, s[r]);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16));
            const float m_new = fmaxf(m, mx);
            const float m_exp = (m_new == -__builtin_inff()) ? 0.0f : m_new;
            const float alpha = (m == -__builtin_inff()) ? 0.0f : __builtin_amdgcn_exp2f(m - m_exp);
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float p = ok[r] ? __builtin_amdgcn_exp2f(s[r] - m_exp) : 0.0f;
                sum += p;
                sP[col * LDT + 2 * r + hh] = (half_t) p;
            }
            sum += __shfl_xor(sum, 16);
            l = l * alpha + sum;
            m = m_new;
            if (hh == 0) sAlpha[col] = alpha;
        }
        sync_lds();                                              // P, alpha; K tile no longer read
        {   // V^T into the same LDS (one key per lane group: conflict-free writes)
            half_t* sV = sKV;
            const half_t* a0 = (const half_t*) &vr0;
            const half_t* a1 = (const half_t*) &vr1;
#pragma unroll
            for (int j = 0; j < 8; ++j) { sV[(dch * 16 + j) * LDT + kk] = a0[j]; sV[(dch * 16 + 8 + j) * LDT + kk] = a1[j]; }
        }
        if (i + 1 < ntile) {                                     // next K tile, in flight behind the PV step
            const int tk = key_tok(i + 1);
            kr0 = *(const v4u*) (kc + IX_KV(tk, g, dch * 16, tok_stride));
            kr1 = *(const v4u*) (kc + IX_KV(tk, g, dch * 16 + 8, tok_stride));
        }
        sync_lds();                                              // V^T visible
        {
            float al[3];
            v16h Bp[3];
#pragma unroll
            for (int ct = 0; ct < 3; ++ct) { al[ct] = sAlpha[16 * ct + n]; Bp[ct] = ld16h(sP + (16 * ct + n) * LDT); }
#pragma unroll
            for (int dt = 0; dt < 2; ++dt) {
                const v16h A = ld16h(sKV + (16 * (2 * w + dt) + n) * LDT);
#pragma unroll
                for (int ct = 0; ct < 3; ++ct) {
                    v8f c = acc[dt][ct] * al[ct];
                    acc[dt][ct] = wmma(A, Bp[ct], c);
                }
            }
        }
        sync_lds();                                              // sKV / sP free
    }

    // ---- epilogue: acc / l, staged in LDS, coalesced stores ----
    if (w < 3 && hh == 0) sL[16 * w + n] = l;
    sync_lds();
    half_t* sO = (half_t*) sBig;
#pragma unroll
    for (int dt = 0; dt < 2; ++dt)
#pragma unroll
        for (int ct = 0; ct < 3; ++ct) {
            const float lv = sL[16 * ct + n];
#pragma unroll
            for (int r = 0; r < 8; ++r)
                sO[(16 * ct + n) * LDQ + 16 * (2 * w + dt) + 2 * r + hh] = (half_t) (lv > 0.0f ? acc[dt][ct][r] / lv : 0.0f);
        }
    sync_lds();
    for (int c = tid; c < NCOL * 32; c += 256) {
        const int col = c >> 5, ch = c & 31, rho = col / GRP, h = col - rho * GRP;
        const int row = row0 + rho;
        if (row < R) *(v4u*) (o + IX_O(row, g * GRP + h, ch * 8, o_stride)) = *(const v4u*) (sO + col * LDQ + ch * 8);
    }
}
