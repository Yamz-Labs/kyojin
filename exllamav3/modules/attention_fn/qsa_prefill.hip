// Qwen QSA sparse prefill attention, hand-written for gfx11.5 WMMA f16 (wave32). Replaces
//   _qsa_sparse_split_kernel (splits == 1) + _paged_attn_decode_combine_kernel for fp16 paged caches.
// Same per-row arithmetic as the Triton pair (bit-identical per row): 32-key chunks in list order, scores = one fp32 WMMA chain
// over d = 0, 16, ... (A = K rows, B = q), scale, online softmax with Triton's exp lowering (log2e fma, +64 range trick, v_exp_f32,
// ldexp), l = fma(l, alpha, sum(p)) with the sum in Triton's register order, acc *= alpha, then two WMMA steps over the chunk's
// keys (A = V^T, B = p) accumulating onto acc, final acc / l with IEEE division, one rounding to fp16.
// One wave per (query row, kv head): QK and softmax are done once (Triton replicates them on 4 waves).
// Fragment layout (gfx1151): A: lane l, elem r -> A[l & 15][r]; B: lane l, elem r -> B[r][l & 15]; C: lane l, reg r -> C[2r + (l >> 4)][l & 15]
// MEMORY SAFETY: every global address comes from the IX_* macros with clamps (TOKC, PGC, row grid), every global store is guarded
// by the *_OK macros. check_qsa_hip_bounds.py parses these macros and replays all accesses (it must fail when a clamp or guard is mutated).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 half_t;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

#define HD 256
#define NH 24
#define NKV 2
#define GRP 12            // q heads per kv head
#define PAGE 256
#define KPAD 2080         // 4 * block_topk + 3 rounded up to 32
#define BN 32             // keys per chunk
#define NCH (KPAD / BN)
#define LDV 264           // V LDS pitch in halves (512 B + 16 B)
#define LDQ 264
#ifndef WSYNC
#define WSYNC 1
#endif
#ifndef QG
#define QG 1
#endif
#ifndef PF
#define PF 0
#endif
#ifndef ABL
#define ABL 0
#endif
#ifndef SUMT0
#define SUMT0 0           // 0: p sum chain runs tile 0 then tile 1; 1: tile 1 then tile 0
#endif

#define RCL(r_, R_) min((r_), (R_) - 1)
#define IX_Q(r_, h_, d_, p_) (((int64_t) (r_)) * (p_) + (h_) * HD + (d_))     // q rows at pitch p_ (halves)
#define IX_O(r_, h_, d_, p_) (((int64_t) (r_)) * (p_) + (h_) * HD + (d_))
#define IX_IDX(r_, n_) (((int64_t) (r_)) * KPAD + (n_))
#define IX_BT(r_, pg_, p_) (((int64_t) (r_)) * (p_) + (pg_))
#define PGC(pg_, NP_) min(max((pg_), 0), (NP_) - 1)
#define TOKC(t_, NT_) min(max((t_), 0), (NT_) - 1)
#define IX_KV(t_, g_, d_, p_) (((int64_t) (t_)) * (p_) + (g_) * HD + (d_))   // K / V tokens at pitch p_ (halves)
#define Q_OK(h_) ((h_) < GRP)
#define O_OK(h_) ((h_) < GRP)

__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }

// Triton tl.exp lowering on gfx11: t = x * log2e; adj = t < -126 ? 64 : 0; u = fma(x, log2e, adj); ldexp(exp2(u), t < -126 ? -64 : 0)
__device__ __forceinline__ float tl_exp(float x) {
    const float L2E = 1.44269502162933349609375f;   // 0x3fb8aa3b
    float t = x * L2E;
    bool lo = (-126.0f > t);
    float u = __builtin_fmaf(x, L2E, lo ? 64.0f : 0.0f);
    float e = __builtin_amdgcn_exp2f(u);
    return __builtin_amdgcn_ldexpf(e, lo ? -64 : 0);
}

__device__ __forceinline__ v16h ld16h(const half_t* p) {   // two 16 B loads (p is 16 B aligned)
    v4u a = *(const v4u*) p, b = *(const v4u*) (p + 8);
    union { v4u u[2]; v16h h; } x; x.u[0] = a; x.u[1] = b; return x.h;
}
__device__ __forceinline__ v16h ld16h_lds(const half_t* p) { return ld16h(p); }

#ifndef VB
#define VB 8
#endif
#if KS == 2
#define SVR 32            // K of the whole chunk (32 keys) in LDS, then both V halves in the same rows
#define V1R 16            // row offset of V half 1
#else
#define SVR 16
#define V1R 0
#endif
#ifndef KS
#define KS 0              // 2: both K halves staged at once (one q read) and both V halves staged together after QK; 1: K rows staged coalesced through LDS (half chunks, shared with the V tile); 0: K gathered from global per lane
#endif
// keys 16 hf_ .. 16 hf_ + 15 of src_ (K or V cache) -> LDS rows i at pitch LDV (invalid keys zero). A macro (a function spills 300 VGPRs). Locals carry the stg_ prefix:
// a macro argument that names an outer variable must never collide with them (the incident); check_expanded.py builds with -Werror=shadow -Werror=uninitialized.
#define STAGE(dst_, src_, hf_, mask_) do { \
        for (int stg_i0 = 0; stg_i0 < 16; stg_i0 += VB) { \
            v4u stg_vb[VB]; \
            _Pragma("unroll") for (int stg_j = 0; stg_j < VB; ++stg_j) { \
                const int stg_t = sTok[16 * (hf_) + stg_i0 + stg_j]; \
                stg_vb[stg_j] = (v4u) {0, 0, 0, 0}; \
                if (((mask_) >> (16 * (hf_) + stg_i0 + stg_j)) & 1u) stg_vb[stg_j] = *(const v4u*) ((src_) + IX_KV(stg_t, g, lane * 8, tok_stride)); \
            } \
            _Pragma("unroll") for (int stg_j = 0; stg_j < VB; ++stg_j) *(v4u*) ((dst_) + (stg_i0 + stg_j) * LDV + lane * 8) = stg_vb[stg_j]; \
        } } while (0)

#ifndef WPE
#define WPE 1
#endif
#if WSYNC
#define SYNC() asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory")   /* one wave per block: LDS ops complete in order */
#else
#define SYNC() __syncthreads()
#endif
extern "C" __global__ __launch_bounds__(32) __attribute__((amdgpu_waves_per_eu(WPE))) void qsa_pf(
    const half_t* __restrict__ q, const half_t* __restrict__ kc, const half_t* __restrict__ vc,
    const int* __restrict__ bt, const int* __restrict__ idx, half_t* __restrict__ o,
    int q_stride, int o_stride, int bt_stride, int tok_stride, int ntok, int npages, float scale)
{
    const int pid = blockIdx.x;
    const int row = pid / NKV, g = pid % NKV;        // grid = R * NKV exactly
    const int lane = threadIdx.x, n = lane & 15, hh = lane >> 4;
#if !QG
    __shared__ half_t sQ[16 * LDQ];
#endif
    __shared__ half_t sV[SVR * LDV];
    __shared__ int sTok[BN];

#if !QG
    for (int i = 0; i < 16; ++i) {                   // q rows of this kv head -> LDS (rows >= GRP zero)
        const int c = i * 32 + lane, h = c >> 5, ch = c & 31;
        v4u val = {0, 0, 0, 0};
        if (Q_OK(h)) val = *(const v4u*) (q + IX_Q(row, g * GRP + h, ch * 8, q_stride));
        *(v4u*) (sQ + h * LDQ + ch * 8) = val;
    }
    SYNC();
#endif

    v8f acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -__builtin_inff(), l = 0.0f;
    const bool head_ok = n < GRP;

    // software pipeline of the index chain: idx (c + 1) is requested at the top of chunk c, its block table entry before the PV step
    int id = idx[IX_IDX(row, lane)];
    int ph = bt[IX_BT(row, PGC(max(id, 0) / PAGE, npages), bt_stride)];
    for (int c = 0; c < NCH; ++c) {
        const int id_n = idx[IX_IDX(row, min(c + 1, NCH - 1) * BN + lane)];   // last chunk: re-reads itself (unused)
        const bool kv_valid = id >= 0;
        const unsigned vmask = (unsigned) __ballot(kv_valid);
        const int idc = max(id, 0);
        int ph_n;
        if (vmask == 0u) {                           // chunk without valid keys: exact no-op in the Triton recurrence
            ph_n = bt[IX_BT(row, PGC(max(id_n, 0) / PAGE, npages), bt_stride)];
            id = id_n; ph = ph_n;
            continue;
        }
        SYNC();
        const int tok = kv_valid ? TOKC(ph * PAGE + idc % PAGE, ntok) : 0;
        sTok[lane] = tok;
        SYNC();

#if !(ABL & 1) && !KS
        STAGE(sV, vc, 0, vmask);   // V half 0 -> LDS (invalid keys zero, like the masked Triton load)
#endif

        // scores
        v8f S[2] = {(v8f) {0, 0, 0, 0, 0, 0, 0, 0}, (v8f) {0, 0, 0, 0, 0, 0, 0, 0}};
#if KS == 2
        STAGE(sV, kc, 0, 0xffffffffu);
        STAGE(sV + 16 * LDV, kc, 1, 0xffffffffu);
        SYNC();
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            v16h B = (v16h) {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            if (Q_OK(n)) B = ld16h(q + IX_Q(row, g * GRP + n, 16 * ks, q_stride));
            S[0] = wmma(ld16h_lds(sV + n * LDV + 16 * ks), B, S[0]);
            S[1] = wmma(ld16h_lds(sV + (16 + n) * LDV + 16 * ks), B, S[1]);
        }
        SYNC();
#if !(ABL & 1)
        STAGE(sV, vc, 0, vmask);                // both V halves -> LDS while softmax runs
        STAGE(sV + 16 * LDV, vc, 1, vmask);
#endif
#elif KS
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            STAGE(sV, kc, kh, 0xffffffffu);   // all 16 keys of the half (masked later by ok[][])
            SYNC();
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                v16h B = (v16h) {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (Q_OK(n)) B = ld16h(q + IX_Q(row, g * GRP + n, 16 * ks, q_stride));
                S[kh] = wmma(ld16h_lds(sV + n * LDV + 16 * ks), B, S[kh]);
            }
            SYNC();
        }
#if !(ABL & 1)
        STAGE(sV, vc, 0, vmask);   // V half 0 -> LDS while softmax runs
#endif
#else
        const int tk0 = sTok[n], tk1 = sTok[16 + n];
        const half_t* kp0 = kc + IX_KV(tk0, g, 0, tok_stride);
        const half_t* kp1 = kc + IX_KV(tk1, g, 0, tok_stride);
#if !(ABL & 4)
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
#if QG
            v16h B = (v16h) {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            if (Q_OK(n)) B = ld16h(q + IX_Q(row, g * GRP + n, 16 * ks, q_stride));
#else
            v16h B = ld16h_lds(sQ + n * LDQ + 16 * ks);
#endif
            S[0] = wmma(ld16h(kp0 + 16 * ks), B, S[0]);
            S[1] = wmma(ld16h(kp1 + 16 * ks), B, S[1]);
        }
#else
        S[0][0] = (float) (tk0 & 255) * 0.01f; S[1][1] = (float) (tk1 & 255) * 0.01f;
#endif
#endif   // KS

        // softmax
        float s[2][8], mx = -__builtin_inff();
        bool ok[2][8];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int j = 16 * t + 2 * r + hh;
                ok[t][r] = head_ok && ((vmask >> j) & 1u);
                s[t][r] = ok[t][r] ? S[t][r] * scale : -__builtin_inff();
                mx = fmaxf(mx, s[t][r]);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16));
        const float m_new = fmaxf(m, mx);
        const float m_exp = (m_new == -__builtin_inff()) ? 0.0f : m_new;
        float p[2][8];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float e = tl_exp(s[t][r] - m_exp);
                p[t][r] = ok[t][r] ? e : 0.0f;
            }
        const float alpha = (m == -__builtin_inff()) ? 0.0f : tl_exp(m - m_exp);
        float sum;
#if SUMT0 == 0
        sum = p[0][0] + p[0][1];
#pragma unroll
        for (int r = 2; r < 8; ++r) sum = sum + p[0][r];
#pragma unroll
        for (int r = 0; r < 8; ++r) sum = sum + p[1][r];
#else
        sum = p[1][0] + p[1][1];
#pragma unroll
        for (int r = 2; r < 8; ++r) sum = sum + p[1][r];
#pragma unroll
        for (int r = 0; r < 8; ++r) sum = sum + p[0][r];
#endif
        sum = sum + __shfl_xor(sum, 16);
        l = __builtin_fmaf(l, alpha, sum);
        m = m_new;

        // p fragments (B operand: lane head n, elem r = key 16 ks + r); both half-waves hold the full 16 keys
        v16h pf[2];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const half_t own = (half_t) p[t][r];
                union { half_t h; uint16_t u; } cv; cv.h = own;
                const uint16_t other = (uint16_t) __shfl_xor((int) cv.u, 16);
                union { uint16_t u; half_t h; } oc; oc.u = other;
                pf[t][2 * r] = hh == 0 ? own : oc.h;
                pf[t][2 * r + 1] = hh == 0 ? oc.h : own;
            }

#if PF
        ph_n = bt[IX_BT(row, PGC(max(id_n, 0) / PAGE, npages), bt_stride)];
#endif
        SYNC();                             // V half 0 written by all lanes
#if (ABL & 2)
        acc[0][0] += pf[0][0] + pf[1][3]; acc[1][1] += alpha;
#else
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nt][i] = acc[nt][i] * alpha;
            v16h A;
#pragma unroll
            for (int r = 0; r < 16; ++r) A[r] = sV[r * LDV + 16 * nt + n];
            acc[nt] = wmma(A, pf[0], acc[nt]);
        }
#endif
#if KS != 2
        SYNC();
#if !(ABL & 1)
        STAGE(sV, vc, 1, vmask);
#endif
        SYNC();
#endif
#if !(ABL & 2)
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            v16h A;
#pragma unroll
            for (int r = 0; r < 16; ++r) A[r] = sV[(V1R + r) * LDV + 16 * nt + n];
            acc[nt] = wmma(A, pf[1], acc[nt]);
        }
#endif
        SYNC();                             // sV / sTok free for the next chunk
#if !PF
        ph_n = bt[IX_BT(row, PGC(max(id_n, 0) / PAGE, npages), bt_stride)];
#endif
        id = id_n; ph = ph_n;
    }

    // epilogue (num_splits == 1 combine, same expression order as _paged_attn_decode_combine_kernel)
    const float m_safe = (m == -__builtin_inff()) ? 0.0f : m;
    const float w = (m == -__builtin_inff()) ? 0.0f : tl_exp(m - m_safe);
    const float lw = l * w;
    const float l_sum = __builtin_fmaf(l, w, 0.0f);
    const float den = (lw != 0.0f) ? l_sum : 1.0f;
    half_t* sO = sV;                                 // (16 heads, HD) at pitch LDV
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float a = __builtin_fmaf(acc[nt][r], w, 0.0f);
            sO[n * LDV + 16 * nt + 2 * r + hh] = (half_t) (a / den);
        }
    SYNC();
    for (int i = 0; i < 12; ++i) {
        const int c = i * 32 + lane, h = c >> 5, ch = c & 31;
        if (O_OK(h)) *(v4u*) (o + IX_O(row, g * GRP + h, ch * 8, o_stride)) = *(const v4u*) (sO + h * LDV + ch * 8);
    }
}
