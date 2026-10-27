// Adapted from hipfire (https://github.com/warpfront/hipfire, v0.4.2.1-13-gbfb90b9, kernels/src/gated_delta_chunk_wmma.gfx1151.hip),
// Copyright the hipfire authors, licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
// Changes for exllamav3 (EXL3_GDN_HF): KYO_RAW build = no gated RMSNorm epilogue, the chunk's BF16 o rows are stored as they are
// (our fused_gated_rms_norm runs after, unchanged); z / norm arguments are then unused (may be null); prefix-cache checkpoint
// side-write; T = (I - L)^-1 by fp32 forward substitution instead of the f16 product form (PPL, see the comment at the solve).
// SPDX-License-Identifier: Apache-2.0
// hipfire's Apache-2.0 licence and NOTICE: LICENSE.hipfire-APACHE and NOTICE.hipfire next to this file.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Gated DeltaNet prefill recurrence + gated RMSNorm on gfx11 WMMA, in the
// chunked (WY) form of torch_chunk_gated_delta_rule with 16-row chunks.  Per
// value head (block) and chunk, with G the chunk-local cumulative log decay:
//   L = strictly-lower(-beta_i (k_i . k_j) e^{G_i - G_j}),  T = (I - L)^-1
//   Vnew = T (v beta - (k beta e^G) S)          (S the state entering the chunk)
//   O    = e^G (q S) + (q k^T e^{G_i - G_j}, j <= i) Vnew
//   S    = e^{G_last} S + (k e^{G_last - G})^T Vnew
// T by fp32 forward substitution (exllamav3; hipfire: the product (I + L)(I + L^2)(I + L^4)(I + L^8)).  All
// products are F16 WMMA with F32 accumulation (q/k are BF16 values; S, T,
// Vnew and the decayed operands round to F16), so the route is KLD-gated;
// the gate is gated_delta_gate_bf16_f32_batched's expression.
// Wave w owns value columns 16w .. 16w + 15 of S as eight C-layout tiles in
// registers; a C tile becomes a B operand by swapping half-waves.  The next
// chunk's inputs are prefetched into registers during the current one; the
// block barriers wait on LDS only, so the prefetch stays in flight.
// `projection` is the convolution output as packed BF16 bits.  The output
// values are BF16-rounded; `output_bf16` stores them as BF16 bits, else F32.
// Grid [value_heads], block 256.  key_dim == value_dim == 128.

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
static __device__ __forceinline__ float gc_bf16r(float v) { return (float)(hip_bfloat16)v; }
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
__launch_bounds__(256)
extern "C" __global__ void gated_delta_chunk_gate_wmma(
    const unsigned short* __restrict__ projection, const unsigned short* __restrict__ qn, const unsigned short* __restrict__ kn,
    const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
    const float* __restrict__ z, const hip_bfloat16* __restrict__ norm, void* __restrict__ output,
    int rows, int qkv_width, int key_heads, int value_heads, float query_scale, int output_bf16
#ifdef KYO_RAW
    , float* __restrict__ ck0, int ck_row0, float* __restrict__ ck1, int ck_row1   // state entering row ck_row* (multiple of 16), or row -1
    , float* __restrict__ ck2, int ck_row2, float* __restrict__ ck3, int ck_row3
    , float* __restrict__ ck4, int ck_row4, float* __restrict__ ck5, int ck_row5
    , float* __restrict__ ck6, int ck_row6, float* __restrict__ ck7, int ck_row7   // 8 slots: dense prompt-tail checkpoints
#endif
#ifdef NPU_TRUNK_ARENA
    , int z_pitch, int z_col0
#endif
    ) {
    const int vh = blockIdx.x;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int kh = vh / (value_heads / key_heads);
    const int v_offset = 2 * key_heads * 128 + vh * 128;
    const int value_width = value_heads * 128;
    const size_t state_base = (size_t)vh * 128 * 128;
    __shared__ __align__(16) _Float16 k_s[16][136];    // K rows
    __shared__ __align__(16) _Float16 q_s[16][136];    // scaled Q rows
    __shared__ __align__(16) _Float16 kb_s[16][136];   // k beta e^G rows
    __shared__ __align__(16) _Float16 kdT[128][16];    // (k e^{GL-G})^T
    __shared__ float vb_s[16][132];                    // v beta rows (F32)
    __shared__ __align__(16) _Float16 t_s[16][16];
    __shared__ float lf_s[16][17];                     // L (fp32) for the substitution [exllamav3 change]
    __shared__ __align__(16) _Float16 p_s[16][16];
    __shared__ float g_s[16], bt_s[16], inv_s[16];
    __shared__ __align__(16) unsigned short out_s[16][136];
    const int dv0 = wave * 16;
    gc_f8 S[8];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r)
            S[t][r] = state[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16];
    const int si = tid & 15, sd = (tid >> 4) * 8;
    // Raw prefetch registers, widened where used so the loads stay in flight.
    uint4 pk, pq, pv; float pg = 0.f, pb = 0.f;
#ifdef KYO_RAW
    const float nw = 0.0f; (void) nw;
#else
    const float nw = (float)norm[tid & 127];
#endif
    auto fetch = [&](int c0) {
        const int nn = min(16, rows - c0);
        pk = make_uint4(0, 0, 0, 0); pq = pk; pv = pk;
        if (si < nn) {
            const size_t b = ((size_t)(c0 + si) * key_heads + kh) * 128 + sd;
            pk = *(const uint4*)(kn + b);
            pq = *(const uint4*)(qn + b);
            pv = *(const uint4*)(projection + (size_t)(c0 + si) * qkv_width + v_offset + sd);
        }
        if (tid < 16) {
            pg = tid < nn ? gate[(c0 + tid) * value_heads + vh] : 0.f;
            pb = tid < nn ? beta[(c0 + tid) * value_heads + vh] : 0.f;
        }
    };
    // Gated RMSNorm of a finished chunk's rows (out_s, all 128 columns).
    auto epilogue = [&](int c0, const float* zc) {
        const int nn = min(16, rows - c0);
        {
            const int r = 2 * wave + hi;
            const uint4 w = *(const uint4*)&out_s[r][l16 * 8];
            const unsigned int wv[4] = {w.x, w.y, w.z, w.w};
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float lo = __uint_as_float(wv[h] << 16), hv = __uint_as_float(wv[h] & 0xFFFF0000u);
                sum += lo * lo; sum += hv * hv;
            }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 16);
            if (l16 == 0) inv_s[r] = rsqrtf(sum / 128.0f + 1.0e-6f);
        }
        GC_BARRIER();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int e = tid + 256 * i, r = e >> 7, c = e & 127;
            if (r < nn) {
                const size_t at = (size_t)(c0 + r) * value_width + vh * 128 + c;
                const float v = __uint_as_float((unsigned int)out_s[r][c] << 16);
                const float normalized = gc_bf16r(v * inv_s[r] * nw);
                const float g = gc_bf16r(zc[i]);
                const float y = gc_bf16r(normalized / (1.0f + expf(-g)));
                if (output_bf16)
                    ((unsigned short*)output)[at] = (unsigned short)(__float_as_uint(y) >> 16);
                else
                    ((float*)output)[at] = y;
            }
        }
    };
    fetch(0);
    for (int t0 = 0; t0 < rows; t0 += 16) {
        float zc[8];
        {
            const int nn = min(16, rows - t0);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int e = tid + 256 * i, r = e >> 7;
#if defined(KYO_RAW)
                zc[i] = 0.0f; (void) r; (void) e;
#elif defined(NPU_TRUNK_ARENA)
                zc[i] = r < nn ? __uint_as_float((unsigned int)((const unsigned short*)z)[(size_t)(t0 + r) * z_pitch + z_col0 + vh * 128 + (e & 127)] << 16) : 0.0f;
#else
                zc[i] = r < nn ? z[(size_t)(t0 + r) * value_width + vh * 128 + (e & 127)] : 0.0f;
#endif
            }
        }
        const unsigned int kw[4] = {pk.x, pk.y, pk.z, pk.w}, qw[4] = {pq.x, pq.y, pq.z, pq.w};
        float kf[8];
        const float vf[8] = {__uint_as_float(pv.x << 16), __uint_as_float(pv.x & 0xFFFF0000u),
                             __uint_as_float(pv.y << 16), __uint_as_float(pv.y & 0xFFFF0000u),
                             __uint_as_float(pv.z << 16), __uint_as_float(pv.z & 0xFFFF0000u),
                             __uint_as_float(pv.w << 16), __uint_as_float(pv.w & 0xFFFF0000u)};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            kf[2 * h] = __uint_as_float(kw[h] << 16); kf[2 * h + 1] = __uint_as_float(kw[h] & 0xFFFF0000u);
            *(unsigned int*)&k_s[si][sd + 2 * h] = gc_pack(kf[2 * h], kf[2 * h + 1]);
            *(unsigned int*)&q_s[si][sd + 2 * h] = gc_pack(__uint_as_float(qw[h] << 16) * query_scale,
                                                          __uint_as_float(qw[h] & 0xFFFF0000u) * query_scale);
        }
        if (tid < 16) {
            float G = pg;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) {
                const float up = __shfl_up(G, o, 16);
                if (tid >= o) G += up;
            }
            g_s[tid] = G;
            bt_s[tid] = pb;
        }
        GC_BARRIER();
        {
            const float b = bt_s[si], eg = expf(g_s[si]), ed = expf(g_s[15] - g_s[si]);
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                vb_s[si][sd + h] = vf[h] * b;
                kdT[sd + h][si] = (_Float16)(kf[h] * ed);
            }
#pragma unroll
            for (int h = 0; h < 4; ++h)
                *(unsigned int*)&kb_s[si][sd + 2 * h] = gc_pack(kf[2 * h] * b * eg, kf[2 * h + 1] * b * eg);
        }
        if (t0 + 16 < rows) fetch(t0 + 16);
        GC_BARRIER();
        if (wave == 0) {
            gc_f8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s)
                c = GC_WMMA(*(const gc_h16*)&k_s[l16][16 * s], *(const gc_h16*)&k_s[l16][16 * s], c);
            // L strictly lower.
            // T = (I - L)^-1 by forward substitution in fp32, rounded to f16 once [exllamav3 change: the product form
            // (I + L)(I + L^2)(I + L^4)(I + L^8) with f16 operands lost up to 10-15 % of T on real, correlated keys, |L^8| ~ 7e2]
            gc_f8 P;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = 2 * r + hi, j = l16;
                P[r] = j < i ? -bt_s[i] * c[r] * expf(g_s[i] - g_s[j]) : 0.f;
            }
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
        } else if (wave == 1) {
            gc_f8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s)
                c = GC_WMMA(*(const gc_h16*)&q_s[l16][16 * s], *(const gc_h16*)&k_s[l16][16 * s], c);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = 2 * r + hi, j = l16;
                p_s[i][j] = (_Float16)(j <= i ? c[r] * expf(g_s[i] - g_s[j]) : 0.f);
            }
        }
        // K' = (k beta e^G) S and O_inter = Q S on this wave's slice (S as B from registers).
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
        GC_BARRIER();  // T, P ready
        gc_f8 Vn = {0, 0, 0, 0, 0, 0, 0, 0};
        Vn = GC_WMMA(*(const gc_h16*)&t_s[l16][0], gc_c_to_b(Dm, hi), Vn);
        const gc_h16 vb = gc_c_to_b(Vn, hi);
        gc_f8 O = {0, 0, 0, 0, 0, 0, 0, 0};
        O = GC_WMMA(*(const gc_h16*)&p_s[l16][0], vb, O);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int i = 2 * r + hi;
            out_s[i][dv0 + l16] = __builtin_bit_cast(unsigned short, (hip_bfloat16)(O[r] + expf(g_s[i]) * Oi[r]));
        }
        const float eGL = expf(g_s[15]);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int r = 0; r < 8; ++r) S[t][r] *= eGL;
            S[t] = GC_WMMA(*(const gc_h16*)&kdT[16 * t + l16][0], vb, S[t]);
        }
        GC_BARRIER();  // chunk inputs consumed; out_s complete
#ifdef KYO_RAW
        const int tn = t0 + 16;
        float* ck = tn == ck_row0 ? ck0 : tn == ck_row1 ? ck1 : tn == ck_row2 ? ck2 : tn == ck_row3 ? ck3 :
                    tn == ck_row4 ? ck4 : tn == ck_row5 ? ck5 : tn == ck_row6 ? ck6 : tn == ck_row7 ? ck7 : nullptr;
        if (ck != nullptr) {   // prefix-cache checkpoint: S after this chunk = state entering row t0 + 16
#pragma unroll
            for (int t = 0; t < 8; ++t)
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    ck[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16] = S[t][r];
        }
        {   // raw o rows (BF16 bits) -> output [rows, value_width]
            const int nn = min(16, rows - t0);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int e = tid + 256 * i, r = e >> 7, c = e & 127;
                if (r < nn) ((unsigned short*)output)[(size_t)(t0 + r) * value_width + vh * 128 + c] = out_s[r][c];
            }
            GC_BARRIER();   // out_s is rewritten by the next chunk
        }
#else
        epilogue(t0, zc);
#endif
    }
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r)
            state[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16] = S[t][r];
}
