// Derived from hipfire (https://github.com/warpfront/hipfire, v0.4.2.1-13-gbfb90b9, kernels/src/gated_delta_chunk_wmma.gfx1151.hip),
// Copyright the hipfire authors, licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
// Changes for exllamav3 (EXL3_KDA_HF): Kimi Delta Attention (per-channel decay) instead of the scalar-gate GDN recurrence.
// The 16x16 intra-chunk blocks are computed in the kernel (bf16 WMMA on q, k scaled by 2^{+-(g - g_m)} around the chunk's middle
// row m; KHF_TP=1 instead reads them from fla's token_parallel kernel), the decay factors are per channel, raw o rows out.
// SPDX-License-Identifier: Apache-2.0
// hipfire's Apache-2.0 licence and NOTICE: LICENSE.hipfire-APACHE and NOTICE.hipfire next to this file.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// KDA prefill recurrence on gfx11 WMMA, 16-row chunks. Per value head (block) and chunk, G the 16-row-local cumulative log2
// decay (a vector per row, G = g - g[c0 - 1] from the caller's 64-row-local cumsum), GL = G of the chunk's last row:
//   L = -Akk (strictly lower, beta_i sum_c k_i k_j 2^{G_i - G_j}),  T = (I - L)^-1 (fp32 forward substitution)
//   Vnew = T (v beta - ((k beta) * 2^G) S)
//   O    = ((q scale) * 2^G) S + Aqk Vnew                  (Aqk = scale sum_c q_i k_j 2^{G_i - G_j}, j <= i)
//   S    = diag(2^GL) S + (k * 2^{GL - G})^T Vnew
// Every exponent is <= 0 (no overflow). Products are F16 WMMA with F32 accumulation (S, T, Vnew and the decayed operands round
// to F16): not bitwise equal to the 64-row chain, quality-gated (KLD).
// Wave w owns value columns 16w .. 16w + 15 of S as eight C-layout tiles in registers. Grid [value_heads], block 256, K = V = 128.
// Inputs (B = 1, contiguous): q, k [rows, H, 128] bf16; v [rows, HV, 128] bf16; g [rows, HV, 128] fp32 (64-row-local log2 cumsum);
// beta [rows, HV] bf16; akk [rows, HV, 16] fp32 (row i, column j - 16 (i / 16), valid for j < i); aqk [rows, HV, 64] bf16 (column
// j % 64, valid for j <= i in the same 16-row block). state [HV, 128, 128] fp32 in/out. output [rows, HV, 128] bf16.

typedef _Float16 __attribute__((ext_vector_type(16))) gc_h16;
typedef float __attribute__((ext_vector_type(8))) gc_f8;
typedef unsigned int __attribute__((ext_vector_type(8))) gc_u8;
typedef _Float16 __attribute__((ext_vector_type(2))) gc_h2;
// __syncthreads also waits for every outstanding global access (WGP-mode
// workgroup fence), which would drain the prefetch.
#define GC_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
// Wave-private LDS handoff: order this wave's stores before its loads.
#define GC_WAVE_SYNC() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront"); } while (0)
#define GC_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
typedef short __attribute__((ext_vector_type(16))) gc_s16;
#define GC_WMMA_BF(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c)
static __device__ __forceinline__ unsigned int gc_pack(float a, float b) {
    return __builtin_bit_cast(unsigned int, gc_h2{(_Float16)a, (_Float16)b});
}
// C-layout tile (lane: column l16, rows 2r + hi) -> the B operand of that
// matrix (lane: column l16, all 16 rows), by swapping half-waves.
static __device__ __forceinline__ gc_h16 gc_c_to_b(const gc_f8 c, int hi) {
    gc_u8 out;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const unsigned int own = gc_pack(c[2 * p], c[2 * p + 1]);
        const unsigned int oth = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xfedcba98u, false, false);
        const unsigned int a = hi ? oth : own, b = hi ? own : oth;
        out[2 * p] = __builtin_amdgcn_perm(b, a, 0x05040100u);
        out[2 * p + 1] = __builtin_amdgcn_perm(b, a, 0x07060302u);
    }
    return __builtin_bit_cast(gc_h16, out);
}
static __device__ __forceinline__ float bf_lo(unsigned int w) { return __uint_as_float(w << 16); }
static __device__ __forceinline__ float bf_hi(unsigned int w) { return __uint_as_float(w & 0xFFFF0000u); }

__launch_bounds__(256)
extern "C" __global__ void kda_chunk16_wmma(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k, const unsigned short* __restrict__ v,
    const float* __restrict__ g, const unsigned short* __restrict__ beta, const float* __restrict__ akk,
    const unsigned short* __restrict__ aqk, float* __restrict__ state, unsigned short* __restrict__ output,
    int rows, int key_heads, int value_heads, float query_scale,
    float* __restrict__ ck0, int ck_row0, float* __restrict__ ck1, int ck_row1,     // state entering row ck_row* (multiple of 16), or -1;
    float* __restrict__ ck2, int ck_row2, float* __restrict__ ck3, int ck_row3) {   // 4 slots: an 8192-row chunk carries 3 interior + the last page
    const int vh = blockIdx.x;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int kh = vh / (value_heads / key_heads);
    const size_t state_base = (size_t)vh * 128 * 128;
    __shared__ __align__(16) _Float16 q_s[16][136];    // q scale 2^G rows
    __shared__ __align__(16) _Float16 kb_s[16][136];   // k beta 2^G rows
    __shared__ __align__(16) _Float16 kdT[128][16];    // (k 2^{GL-G})^T
    __shared__ float vb_s[16][132];                    // v beta rows (F32)
    __shared__ float l_s[16][16];                      // Akk diagonal block (masked j < i)
    __shared__ __align__(16) _Float16 t_s[16][16];
    __shared__ float lf_s[16][17];                     // L (fp32) for the substitution
    __shared__ __align__(16) _Float16 p_s[16][16];     // Aqk diagonal block (masked j <= i)
    __shared__ float gl_s[128], egl_s[128];            // GL, 2^GL per key channel
    __shared__ __align__(16) unsigned short out_s[16][136];
#ifndef KHF_TP   // diagonal blocks in-kernel: bf16 WMMA on operands scaled around the chunk's middle row m,
                 // |g_i - g_m| <= 8 steps x 7.22 (gate lower bound -5 nat) < 58 -> 2^{+-58} fits bf16 (and fp32); rows >= nn are zero
    __shared__ __align__(16) unsigned short qg_s[16][136], kg_s[16][136], ki_s[16][136];   // q 2^{g-gm}, k 2^{g-gm}, k 2^{gm-g}
    __shared__ float gm_s[128], bt_s[16];
#endif
    const int dv0 = wave * 16;
    gc_f8 S[8];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r)
            S[t][r] = state[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16];
    const int si = tid & 15, sd = (tid >> 4) * 8;      // row / 8 channels of the per-row prep
    const int ai = tid >> 4, aj = tid & 15;            // element of the 16x16 blocks
    // Raw prefetch registers, widened where used so the loads stay in flight.
    uint4 pk, pq, pv; float4 pg0, pg1, pp0, pp1; unsigned short pb, paq; float pa;
    auto fetch = [&](int c0) {
        const int nn = min(16, rows - c0);
        pk = make_uint4(0, 0, 0, 0); pq = pk; pv = pk;
        pg0 = make_float4(0.f, 0.f, 0.f, 0.f); pg1 = pg0; pp0 = pg0; pp1 = pg0; pb = 0; paq = 0; pa = 0.f;
        if (si < nn) {
            const size_t bq = ((size_t)(c0 + si) * key_heads + kh) * 128 + sd;
            const size_t bv = ((size_t)(c0 + si) * value_heads + vh) * 128 + sd;
            pk = *(const uint4*)(k + bq);
            pq = *(const uint4*)(q + bq);
            pv = *(const uint4*)(v + bv);
            pg0 = *(const float4*)(g + bv); pg1 = *(const float4*)(g + bv + 4);
            if (c0 & 63) {
                const size_t bp = ((size_t)(c0 - 1) * value_heads + vh) * 128 + sd;
                pp0 = *(const float4*)(g + bp); pp1 = *(const float4*)(g + bp + 4);
            }
            pb = beta[(size_t)(c0 + si) * value_heads + vh];
        }
#ifdef KHF_TP
        if (ai < nn) {
            const size_t r = (size_t)(c0 + ai) * value_heads + vh;
            if (aj < ai) pa = akk[r * 16 + aj];
            if (aj <= ai) paq = aqk[r * 64 + (c0 & 63) + aj];
        }
#endif
    };
    fetch(0);
    for (int t0 = 0; t0 < rows; t0 += 16) {
        const int nn = min(16, rows - t0);
        float kf[8], gf[8], qf[8], gr[8];
        {
            const unsigned int kw[4] = {pk.x, pk.y, pk.z, pk.w}, qw[4] = {pq.x, pq.y, pq.z, pq.w}, vw[4] = {pv.x, pv.y, pv.z, pv.w};
            const float ga[8] = {pg0.x, pg0.y, pg0.z, pg0.w, pg1.x, pg1.y, pg1.z, pg1.w};
            const float gp[8] = {pp0.x, pp0.y, pp0.z, pp0.w, pp1.x, pp1.y, pp1.z, pp1.w};
            const float b = __uint_as_float((unsigned int)pb << 16);
            float vf[8], eg[8];
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                kf[2 * h] = bf_lo(kw[h]); kf[2 * h + 1] = bf_hi(kw[h]);
                qf[2 * h] = bf_lo(qw[h]); qf[2 * h + 1] = bf_hi(qw[h]);
                vf[2 * h] = bf_lo(vw[h]); vf[2 * h + 1] = bf_hi(vw[h]);
            }
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                gr[h] = ga[h];
                gf[h] = ga[h] - gp[h];
                eg[h] = exp2f(gf[h]);
                vb_s[si][sd + h] = vf[h] * b;
            }
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                *(unsigned int*)&q_s[si][sd + 2 * h] = gc_pack(qf[2 * h] * query_scale * eg[2 * h], qf[2 * h + 1] * query_scale * eg[2 * h + 1]);
                *(unsigned int*)&kb_s[si][sd + 2 * h] = gc_pack(kf[2 * h] * b * eg[2 * h], kf[2 * h + 1] * b * eg[2 * h + 1]);
            }
            if (si == nn - 1)
#pragma unroll
                for (int h = 0; h < 8; ++h) gl_s[sd + h] = gf[h];
#ifndef KHF_TP
            if (si == min(7, nn - 1))
#pragma unroll
                for (int h = 0; h < 8; ++h) gm_s[sd + h] = gr[h];
            if (sd == 0) bt_s[si] = b;
#else
            l_s[ai][aj] = pa;
            p_s[ai][aj] = (_Float16)__uint_as_float((unsigned int)paq << 16);
#endif
        }
        GC_BARRIER();
        {
#pragma unroll
            for (int h = 0; h < 8; ++h) kdT[sd + h][si] = (_Float16)(kf[h] * exp2f(gl_s[sd + h] - gf[h]));
            if (tid < 128) egl_s[tid] = exp2f(gl_s[tid]);
#ifndef KHF_TP
            unsigned int wq[4], wk[4], wi[4];
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                float a2[2], b2[2], c2[2];
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    // clamp: overflow guard only, never reached under GLM's gate bound (|e| < 58); a gate-less KDA model with
                    // per-step decay below ~-12 log2 would get inexact (not inf) diagonal terms
                    const float e = si < nn ? fminf(fmaxf(gr[2 * h + u] - gm_s[sd + 2 * h + u], -100.f), 100.f) : 0.f;
                    const float up = exp2f(e), dn = exp2f(-e);
                    a2[u] = qf[2 * h + u] * up; b2[u] = kf[2 * h + u] * up; c2[u] = kf[2 * h + u] * dn;
                }
                wq[h] = __builtin_bit_cast(unsigned short, (hip_bfloat16)a2[0]) | ((unsigned int)__builtin_bit_cast(unsigned short, (hip_bfloat16)a2[1]) << 16);
                wk[h] = __builtin_bit_cast(unsigned short, (hip_bfloat16)b2[0]) | ((unsigned int)__builtin_bit_cast(unsigned short, (hip_bfloat16)b2[1]) << 16);
                wi[h] = __builtin_bit_cast(unsigned short, (hip_bfloat16)c2[0]) | ((unsigned int)__builtin_bit_cast(unsigned short, (hip_bfloat16)c2[1]) << 16);
            }
            *(uint4*)&qg_s[si][sd] = make_uint4(wq[0], wq[1], wq[2], wq[3]);
            *(uint4*)&kg_s[si][sd] = make_uint4(wk[0], wk[1], wk[2], wk[3]);
            *(uint4*)&ki_s[si][sd] = make_uint4(wi[0], wi[1], wi[2], wi[3]);
#endif
        }
        if (t0 + 16 < rows) fetch(t0 + 16);
        GC_BARRIER();
        if (wave == 0) {
            // L strictly lower.
            gc_f8 P;
#ifndef KHF_TP
            gc_f8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s)
                c = GC_WMMA_BF(*(const gc_s16*)&kg_s[l16][16 * s], *(const gc_s16*)&ki_s[l16][16 * s], c);
#endif
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = 2 * r + hi, j = l16;
#ifndef KHF_TP
                P[r] = j < i ? -bt_s[i] * c[r] : 0.f;
#else
                P[r] = -l_s[i][j];
#endif
            }
            // T = (I - L)^-1 by forward substitution in fp32, rounded to f16 once [exllamav3 change: the product form
            // (I + L)(I + L^2)(I + L^4)(I + L^8) with f16 operands lost up to 10-15 % of T on real, correlated keys, |L^8| ~ 7e2]
#pragma unroll
            for (int r = 0; r < 8; ++r) lf_s[2 * r + hi][l16] = P[r];
            GC_WAVE_SYNC();
            if (hi == 0) {   // lane l16 = column c: T[i][c] = (i == c) + sum_{j < i} L[i][j] T[j][c], column c held in registers
                float tc[16];   // (same fmaf order as a rolled loop over an in-place LDS tile: bitwise the same T)
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float a = i == l16 ? 1.f : 0.f;
#pragma unroll
                    for (int j = 0; j < i; ++j) a = fmaf(lf_s[i][j], tc[j], a);
                    asm volatile("" : "+v"(a));   // a materialized in fp32: no fma -> f16 fusion (v_fma_mix) at the store
                    tc[i] = a;
                    t_s[i][l16] = (_Float16)a;
                }
            }
        }
#ifndef KHF_TP
        else if (wave == 1) {
            gc_f8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s)
                c = GC_WMMA_BF(*(const gc_s16*)&qg_s[l16][16 * s], *(const gc_s16*)&ki_s[l16][16 * s], c);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = 2 * r + hi, j = l16;
                p_s[i][j] = (_Float16)(j <= i ? query_scale * c[r] : 0.f);
            }
        }
#endif
        // K' = (k beta 2^G) S and O_inter = (q scale 2^G) S on this wave's slice (S as B from registers).
        gc_f8 Ks = {0, 0, 0, 0, 0, 0, 0, 0}, Oi = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const gc_h16 sb = gc_c_to_b(S[t], hi);
            Ks = GC_WMMA(*(const gc_h16*)&kb_s[l16][16 * t], sb, Ks);
            Oi = GC_WMMA(*(const gc_h16*)&q_s[l16][16 * t], sb, Oi);
        }
        gc_f8 Dm;
#pragma unroll
        for (int r = 0; r < 8; ++r) Dm[r] = vb_s[2 * r + hi][dv0 + l16] - Ks[r];
        GC_BARRIER();  // T ready
        gc_f8 Vn = {0, 0, 0, 0, 0, 0, 0, 0};
        Vn = GC_WMMA(*(const gc_h16*)&t_s[l16][0], gc_c_to_b(Dm, hi), Vn);
        const gc_h16 vb = gc_c_to_b(Vn, hi);
        gc_f8 O = Oi;
        O = GC_WMMA(*(const gc_h16*)&p_s[l16][0], vb, O);
#pragma unroll
        for (int r = 0; r < 8; ++r)
            out_s[2 * r + hi][dv0 + l16] = __builtin_bit_cast(unsigned short, (hip_bfloat16)O[r]);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int r = 0; r < 8; ++r) S[t][r] *= egl_s[16 * t + 2 * r + hi];
            S[t] = GC_WMMA(*(const gc_h16*)&kdT[16 * t + l16][0], vb, S[t]);
        }
        GC_BARRIER();  // chunk inputs consumed; out_s complete
        float* ck = t0 + 16 == ck_row0 ? ck0 : t0 + 16 == ck_row1 ? ck1 : t0 + 16 == ck_row2 ? ck2 : t0 + 16 == ck_row3 ? ck3 : nullptr;
        if (ck) {   // prefix-cache checkpoint: S after this chunk = state entering row t0 + 16
#pragma unroll
            for (int t = 0; t < 8; ++t)
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    ck[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16] = S[t][r];
        }
        {   // o rows (BF16 bits) -> output [rows, value_heads * 128]
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int e = tid + 256 * i, r = e >> 7, c = e & 127;
                if (r < nn) output[((size_t)(t0 + r) * value_heads + vh) * 128 + c] = out_s[r][c];
            }
            GC_BARRIER();   // out_s is rewritten by the next chunk
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r)
            state[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16] = S[t][r];
}
