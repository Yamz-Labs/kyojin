// Quantized-KV flash-decoding split kernel for gfx11.5 (wave32, WMMA f16): QK^T / PV straight from the packed bit-plane words.
// HIP port of cu/qdec.cu with the same cache format and the same math (dequant = (raw - (2^(B-1) - 0.5)) * scale * 2^-(B-1) in fp16,
// fp32 accumulation, log2-domain online softmax), but its own split/combine contract (qdec_split writes partials, qdec_combine
// reduces them, rotates the result back with h32 and writes fp16).
//
// A CTA has NSW slab streams x DW waves. A stream works on 16-token slabs of one (batch, kv head) pair (streams interleave the slabs of
// the CTA's token range and are merged through LDS at the end); the DW waves of a stream split the head dim: wave dw owns d in
// [dw*HD/DW, (dw+1)*HD/DW) for K (partial S over its k-steps, summed through LDS) and for V/O. DW = 1 keeps everything in one wave.
// Operands come straight from global memory into registers (no LDS staging):
//   S (16 tokens x 16 q rows) = Kdeq (A: lane = token, 16 d per k-step) x q^T (B: lane = q row, from LDS)       -> C: lane = q row, 8 tokens/lane
//   O (16 q rows x 16 d)     += P (A: lane = q row, 16 tokens) x V (B: lane = d column, 16 tokens)
// The d columns of the PV product are permuted: column n of d-tile dt is d = d0 + DTW*n + dt, so lane n dequantizes a run of DTW
// consecutive values of every token (a few aligned words) and no transpose through LDS is needed. The permutation is undone when
// the partials are stored.
// Fragment layout (gfx1151): A: lane l, elem r -> A[l & 15][r]; B: lane l, elem r -> B[r][l & 15]; C: lane l, reg r -> C[2r + (l >> 4)][l & 15]
// MEMORY SAFETY: every global read is for a token clamped into [n_start, n_end) (n_end = cache_seqlens + QLEN, block table index below
// num_pages_per_seq), every store is guarded by the row < ROWS test; qdec_hip.py checks the shapes, test_qdec_hip.py runs with guard pages.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 half_t;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t u32;
typedef uint16_t u16;
typedef u32 v4u __attribute__((ext_vector_type(4)));
typedef u32 v8u __attribute__((ext_vector_type(8)));

#define G (HD / 32)
#define GROUP (NQH / NKV)
#define ROWS (QLEN * GROUP)
#define NT ((ROWS + 15) / 16)
#define RP (NT * 16)
#define GPT (NKV * HD / 32)
#define KCHW (G * KB)
#define VCHW (G * VB)
#define DT (HD / 16)
#define NSW (NW / DW)
#define KSW (DT / DW)                 // k-steps per wave
#define GWV (G / DW)                  // K groups per wave
#define KWW (GWV * KB)                // K words per token row for this wave
#define DTW (DT / DW)                 // d-tiles per wave
#define HDW (HD / DW)                 // d columns per wave
#define NTHREADS (NW * 32)
#define QSTR (HD + 8)
#define LOG2E 1.4426950408889634f
#define NEG_INF (-__builtin_huge_valf())
#ifndef HAS_SINKS
#define HAS_SINKS 0
#endif
#ifndef TQ
#define TQ 0                          // TurboQuant cache (cache/tq.py): codebook-LUT dequant, ONE fp16 scale per head vector, rotation done outside
#endif
#ifndef KEY_LO
#define KEY_LO 0                      // KVWIN: tokens below KEY_LO are not attended (the fp16 ring serves the sinks)
#endif

__device__ __forceinline__ u32 hsub2(u32 a, u32 b) { return __builtin_bit_cast(u32, __builtin_bit_cast(v2h, a) - __builtin_bit_cast(v2h, b)); }
__device__ __forceinline__ u32 hmul2(u32 a, u32 b) { return __builtin_bit_cast(u32, __builtin_bit_cast(v2h, a) * __builtin_bit_cast(v2h, b)); }
__device__ __forceinline__ u32 hfma2(u32 a, u32 b, u32 c)
{
    return __builtin_bit_cast(u32, __builtin_elementwise_fma(__builtin_bit_cast(v2h, a), __builtin_bit_cast(v2h, b), __builtin_bit_cast(v2h, c)));
}
__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
__device__ __forceinline__ u32 pack_h2(float lo, float hi)
{
    return __builtin_bit_cast(u32, __builtin_amdgcn_cvt_pkrtz(lo, hi));
}
__device__ __forceinline__ u32 perm(u32 hi, u32 lo, u32 sel) { return __builtin_amdgcn_perm(hi, lo, sel); }
__device__ __forceinline__ float h2f(u16 h) { return (float) __builtin_bit_cast(half_t, h); }

// grid step 2^-(B-1) as fp16 bits, duplicated in both halves
template<int B> __device__ __forceinline__ u32 inv2() { return (u32)((15 - (B - 1)) << 10) * 0x10001u; }
template<int B> __device__ __forceinline__ u32 c2() { return (u32)(0x6400 | (1 << (B - 1))) * 0x10001u; }
// half2(scale, scale) -> half2 of grid-step scales
template<int B> __device__ __forceinline__ u32 scale2(u16 a) { return hmul2((u32)a * 0x10001u, inv2<B>()); }

#define MAGIC 0x64006400u          // fp16 1024.0 in both halves: OR-ing a raw value below 1024 gives 1024 + raw

// Two adjacent raw values (positions p, p+1, p even; compile-time) of a packed plane array, as the fp16 pair (1024 + raw) in the two
// 16-bit halves. base[o8..] = 8-bit plane words, base[o4..] = 4-bit plane, base[o2..] / base[o1..] = 2-bit / 1-bit plane (present planes
// of B only); position p of a plane with W bits per value is in word p / (32 / W), bits (p % (32 / W)) * W. A K group has
// o8 = 0, o4 = 8 (if the 8-bit plane exists), ...; a V run has its own layout (load_run).
template<int B> __device__ __forceinline__ u32 raw_pair_m(const u32* base, int p, int o8, int o4, int o2, int o1)
{
    if (B == 8) {                  // one v_perm: [b0, 0x64, b1, 0x64]
        u32 w = base[o8 + (p >> 2)];
        return (p & 2) ? perm(0x64646464u, w, 0x04030402u) : perm(0x64646464u, w, 0x04010400u);
    }
    u32 raw = 0; bool first = true;
    if (B & 8) { u32 v = (base[o8 + (p >> 2)] >> ((p & 3) << 3)) & 0xFFFFu; raw = (v & 0xFFu) | ((v & 0xFF00u) << 8); first = false; }
    if (B & 4) { u32 v = (base[o4 + (p >> 3)] >> ((p & 7) << 2)) & 0xFFu; u32 x = (v & 0xFu) | ((v & 0xF0u) << 12); raw = first ? x : ((raw << 4) | x); first = false; }
    if (B & 2) { u32 v = (base[o2 + (p >> 4)] >> ((p & 15) << 1)) & 0xFu; u32 x = (v & 3u) | ((v & 12u) << 14); raw = first ? x : ((raw << 2) | x); first = false; }
    if (B & 1) { u32 v = (base[o1 + (p >> 5)] >> (p & 31)) & 3u; u32 x = (v & 1u) | ((v & 2u) << 15); raw = first ? x : ((raw << 1) | x); }
    return raw | MAGIC;
}
// fp16 pair (1024 + raw) + half2 grid-step scale -> half2 values (raw - (2^(B-1) - 0.5)) * scale, one rounding (s2h = s2 * 0.5)
template<int B> __device__ __forceinline__ u32 dequant2(u32 raw_m, u32 s2, u32 s2h)
{
    return hfma2(hsub2(raw_m, c2<B>()), s2, s2h);
}

// TurboQuant: two raw values (fp16 pair 1024 + raw) -> half2 (codebook[raw] * scale); the codebook sits in LDS as fp16 bits
#ifndef TQ_NOLUT
#define TQ_NOLUT 0                    // experiment: uniform grid instead of the codebook (timing only, wrong values)
#endif
template<int B> __device__ __forceinline__ u32 dequant2_tq(u32 raw_m, u32 s2, const u16* cb)
{
#if TQ_NOLUT
    return hmul2(hsub2(raw_m, c2<B>()), s2);
#else
    u32 lo = cb[raw_m & 0x3FFu], hi = cb[(raw_m >> 16) & 0x3FFu];
    return hmul2(lo | (hi << 16), s2);
#endif
}

// TQ fast path (TQF): instead of rebuilding fp16 (1024 + raw) pairs from the bit planes, build for every 8 values a pair of index words
// ze / zo whose byte m is the LDS byte address (2 * raw) of the value at position 2m / 2m + 1, then one bfe + one u16 table read per
// value. Same values, same rounding (codebook entry * fp16 scale) as the generic TQ path, so the output is bit-identical.
// (DTW = 4 runs: only bytes 0, 1 of ze / zo are used.) Inputs: w = 4-bit plane word (value p at bits 4p), y = low plane field(s) of the same 8 values: B = 6 -> 2-bit fields, value p at
// bits 2p + 16 * KH (KH = 1 for the upper half of a 32-bit plane word); B = 5 -> 1-bit fields, value p at bit p + 8 * KK.
// raw = nib << (B - 4) | field, so address = nib << (B - 3) | field << 1.
#ifndef TQF_OFF
#define TQF_OFF 0
#endif
#define TQF (TQ && !TQF_OFF && (KB == 5 || KB == 6) && (VB == 5 || VB == 6) && (DTW == 8 || DTW == 4))
template<int B, int K> __device__ __forceinline__ void zbuild(u32 w, u32 y, u32& ze, u32& zo)
{
    if (B == 6) {
        u32 u1 = perm(0u, y, (K & 1) ? 0x0C030C02u : 0x0C010C00u);               // bytes [y0, 0, y1, 0] of the 16-bit field word
        u32 v = ((u1 << 4) & 0x0F000F00u) | (u1 & 0x000F000Fu);                   // nibble m (fields 2m, 2m + 1) at byte m
        ze = (((v << 1) & 0x06060606u)) | ((w << 3) & 0x78787878u);
        zo = (((v >> 1) & 0x06060606u)) | ((w >> 1) & 0x78787878u);
    } else {
        u32 t = (y >> (8 * K)) & 0xFFu;
        ze = (((t & 0x55u) * 0x82082u) & 0x02020202u) | ((w << 2) & 0x3C3C3C3Cu);
        zo = (((t & 0xAAu) * 0x41041u) & 0x02020202u) | ((w >> 2) & 0x3C3C3C3Cu);
    }
}
__device__ __forceinline__ u32 zbyte(u32 z, int m) { return __builtin_amdgcn_ubfe(z, 8 * m, 8); }
// half2 {cb[a0], cb[a1]} from two byte addresses of the u16 codebook
__device__ __forceinline__ u32 lut2(const u16* cb, u32 a0, u32 a1)
{
    v2h r;
    r.x = *(const half_t*)((const char*) cb + a0);
    r.y = *(const half_t*)((const char*) cb + a1);
    return __builtin_bit_cast(u32, r);
}

// K group plane offsets (words)
#define KO4 ((KB & 8) ? 8 : 0)
#define KO2 (KO4 + ((KB & 4) ? 4 : 0))
#define KO1 (KO2 + ((KB & 2) ? 2 : 0))

// ---- V runs: lane n owns the DTW consecutive values p0 .. p0 + DTW - 1 of a group (p0 = (d0 + DTW * n) mod 32; DTW in 4, 8, 16)
// run layout (present planes only): [8-bit: ceil(DTW/4) words][4-bit: ceil(DTW/8) words, run shifted to bit 0][2-bit: 1 word, run in the
// low DTW*2 bits][1-bit: 1 word, run in the low DTW bits]
#define RUN_W8 ((VB & 8) ? (DTW + 3) / 4 : 0)
#define RUN_W4 ((VB & 4) ? (DTW + 7) / 8 : 0)
#define RUN_W2 ((VB & 2) ? 1 : 0)
#define RUN_W1 ((VB & 1) ? 1 : 0)
#define NRW (RUN_W8 + RUN_W4 + RUN_W2 + RUN_W1)

__device__ __forceinline__ void load_run(u32* run, const u32* gw, int p0)
{
    int pb = 0, o = 0;
    if (VB & 8) {
        #pragma unroll
        for (int k = 0; k < RUN_W8; k++) run[o + k] = gw[pb + (p0 >> 2) + k];
        pb += 8; o += RUN_W8;
    }
    if (VB & 4) {
        #pragma unroll
        for (int k = 0; k < RUN_W4; k++) run[o + k] = gw[pb + (p0 >> 3) + k] >> (DTW < 8 ? ((p0 & 7) << 2) : 0);
        pb += 4; o += RUN_W4;
    }
    if (VB & 2) { run[o] = gw[pb + (p0 >> 4)] >> ((p0 & 15) << 1); pb += 2; o += 1; }
    if (VB & 1) { run[o] = gw[pb] >> p0; }
}

extern "C" __global__ void __launch_bounds__(NTHREADS)
qdec_split(
    const u16* __restrict__ q, const u32* __restrict__ k_cache, const u32* __restrict__ v_cache,
    const int* __restrict__ block_table, const int* __restrict__ cache_seqlens,
    float* __restrict__ partial_o, float* __restrict__ partial_ml,
    const u16* __restrict__ k_scales, const u16* __restrict__ v_scales, const u16* __restrict__ h32,
    int split_len, int num_pages_per_seq, int num_splits,
    const u16* __restrict__ cbk, const u16* __restrict__ cbv)
{
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ u16 cbk_s[TQ ? (1 << KB) : 1];
    __shared__ u16 cbv_s[TQ ? (1 << VB) : 1];
    u16* Qs = (u16*) smem;
    unsigned char* scratch = smem + ((RP * QSTR * 2 + 15) / 16) * 16;

    const int tid = threadIdx.x, lane = tid & 31;
    const int wid = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int dw = wid % DW, sw = wid / DW;
    const int n = lane & 15, half = lane >> 4;
    const int pid = blockIdx.x, split = blockIdx.y;
    if (split >= num_splits) return;
    const int batch = pid / NKV, kvh = pid % NKV;

    const int total_k = cache_seqlens[batch] + QLEN;   // the cache already holds the new tokens
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k);
    if (n_end <= n_start) {                              // empty range (split grid sized from the block-table bound): m = -inf, l = 0
        for (int i = tid; i < RP; i += NTHREADS) {
            partial_ml[((size_t)(pid * num_splits + split) * RP + i) * 2] = NEG_INF;
            partial_ml[((size_t)(pid * num_splits + split) * RP + i) * 2 + 1] = 0.f;
        }
        return;
    }

    if (TQ) {
        for (int i = tid; i < (1 << KB); i += NTHREADS) cbk_s[i] = cbk[i];
        for (int i = tid; i < (1 << VB); i += NTHREADS) cbv_s[i] = cbv[i];
    }
    // ---- q rows -> rotated (h32) fp16 in smem; rows >= ROWS are zero
    if (TQ) {                                            // q arrives already rotated: plain copy
        for (int idx = tid; idx < RP * HD; idx += NTHREADS) {
            int r = idx / HD, d = idx % HD;
            u16 x = 0;
            if (r < ROWS) x = q[((size_t)((batch * QLEN + r / GROUP) * NQH + kvh * GROUP + r % GROUP)) * HD + d];
            Qs[r * QSTR + d] = x;
        }
        __syncthreads();
    } else {
        float* hs = (float*) scratch;                    // 32x32 fp32, the scratch buffer is idle until the loop
        for (int i = tid; i < 1024; i += NTHREADS) hs[i] = h2f(h32[i]);
        __syncthreads();
        for (int idx = tid; idx < RP * HD; idx += NTHREADS) {
            int r = idx / HD, d = idx % HD, gi = d >> 5, c = d & 31;
            float acc = 0.f;
            if (r < ROWS) {
                int row_q = r / GROUP, hl = r % GROUP;
                const u16* qp = q + ((size_t)((batch * QLEN + row_q) * NQH + kvh * GROUP + hl)) * HD + gi * 32;
                #pragma unroll
                for (int k = 0; k < 32; k++) acc += h2f(qp[k]) * hs[k * 32 + c];
            }
            Qs[r * QSTR + d] = __builtin_bit_cast(u16, (half_t) acc);
        }
        __syncthreads();
    }

    float m[NT], lsum[NT], qabs[NT];
    bool rvalid[NT];
    #pragma unroll
    for (int j = 0; j < NT; j++) {
        int r = 16 * j + n;
        m[j] = NEG_INF; lsum[j] = 0.f;
        rvalid[j] = r < ROWS;
        qabs[j] = (float)(total_k - QLEN + r / GROUP);
    }
    v8f O[NT][DTW];
    #pragma unroll
    for (int j = 0; j < NT; j++)
        #pragma unroll
        for (int dt = 0; dt < DTW; dt++) O[j][dt] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    const int nslabs = (n_end - n_start + 15) / 16;
    const int iters = (nslabs + NSW - 1) / NSW;          // uniform over the CTA (the DW waves of a stream exchange S every iteration)
    const float sl = SCALE * LOG2E;
    const float qmin = (float)(total_k - QLEN);

    float* Ssm = (float*) scratch;                       // DW > 1: [2][NSW*DW][NT][8][32] partial S, double buffered
    u32 kw[KWW];             // this lane's token row of K (kvh slice of this wave's groups)
    u16 ksc[GWV];            // its scales
    int row0 = 0, lim = 0;

    auto slab_rows = [&](int it) {
        int n0 = n_start + (sw + it * NSW) * 16;
        int page = min(n0 / PAGE, num_pages_per_seq - 1);   // never past the block table row (cache_seqlens is trusted otherwise)
        int phys = block_table[batch * num_pages_per_seq + page];
        row0 = phys * PAGE + min(n0 - page * PAGE, PAGE - 16);
        lim = n_end - 1 - n0;                             // last valid token of the slab (>= 0)
    };
    auto load_k = [&]() {
        size_t row = (size_t) row0 + min(n, lim);
        const u32* kp = k_cache + row * (GPT * KB) + kvh * KCHW + dw * KWW;
        if (KWW % 4 == 0 && KCHW % 4 == 0) {
            #pragma unroll
            for (int c = 0; c < KWW / 4; c++) {
                v4u v = ((const v4u*) kp)[c];
                kw[4 * c] = v.x; kw[4 * c + 1] = v.y; kw[4 * c + 2] = v.z; kw[4 * c + 3] = v.w;
            }
        } else {
            #pragma unroll
            for (int c = 0; c < KWW; c++) kw[c] = kp[c];
        }
        if (TQ) {
            u16 s1 = k_scales[row * NKV + kvh];
            #pragma unroll
            for (int c = 0; c < GWV; c++) ksc[c] = s1;
        } else {
            const u16* sp = k_scales + row * GPT + kvh * G + dw * GWV;
            #pragma unroll
            for (int c = 0; c < GWV; c++) ksc[c] = sp[c];
        }
    };

#if TQF
    // TQF: the V runs of a slab are fetched together with its K (one slab ahead), so the V latency overlaps the previous slab's compute
    const int dcol = dw * HDW + DTW * n;
    const int gi_v = dcol >> 5, p0 = dcol & 31;
    u32 vr[16][NRW > 0 ? NRW : 1];
    u32 vsc[16];
    auto load_v = [&]() {
        #pragma unroll
        for (int e = 0; e < 16; e++) {
            size_t row = (size_t) row0 + min(e, lim);
            load_run(vr[e], v_cache + row * (GPT * VB) + kvh * VCHW + gi_v * VB, p0);
            vsc[e] = (u32) v_scales[row * NKV + kvh] * 0x10001u;
        }
    };
    if (nslabs > sw) { slab_rows(0); load_k(); load_v(); }
#else
    if (nslabs > sw) { slab_rows(0); load_k(); }
#endif

    for (int it = 0; it < iters; it++) {
        const bool active = (sw + it * NSW) < nslabs;
        const int n0 = n_start + (sw + it * NSW) * 16;
        v8f S[NT];
        #pragma unroll
        for (int j = 0; j < NT; j++) S[j] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

        if (active) {
            // ---- S (partial over this wave's k-steps) = Kdeq x q^T
            #pragma unroll
            for (int gl = 0; gl < GWV; gl++) {
                u32 sc = TQ ? ((u32) ksc[gl] * 0x10001u) : scale2<KB>(ksc[gl]);
                u32 sch = hmul2(sc, 0x38003800u);        // 0x3800 = 0.5
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int ks = dw * KSW + gl * 2 + h;
                    v8u a;
#if TQF
                    {
                        const u32* g = kw + gl * KB;
                        u32 ze[2], zo[2];
                        if (h == 0) {
                            zbuild<KB, 0>(g[0], KB == 6 ? g[4] : g[4], ze[0], zo[0]);
                            zbuild<KB, 1>(g[1], KB == 6 ? g[4] : g[4], ze[1], zo[1]);
                        } else {
                            zbuild<KB, 2>(g[2], KB == 6 ? g[5] : g[4], ze[0], zo[0]);
                            zbuild<KB, 3>(g[3], KB == 6 ? g[5] : g[4], ze[1], zo[1]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; i++)
                            a[i] = hmul2(lut2(cbk_s, zbyte(ze[i >> 2], i & 3), zbyte(zo[i >> 2], i & 3)), sc);
                    }
#else
                    #pragma unroll
                    for (int i = 0; i < 8; i++)
                        a[i] = TQ ? dequant2_tq<KB>(raw_pair_m<KB>(kw + gl * KB, 16 * h + 2 * i, 0, KO4, KO2, KO1), sc, cbk_s)
                                  : dequant2<KB>(raw_pair_m<KB>(kw + gl * KB, 16 * h + 2 * i, 0, KO4, KO2, KO1), sc, sch);
#endif
                    v16h af = __builtin_bit_cast(v16h, a);
                    #pragma unroll
                    for (int j = 0; j < NT; j++) {
                        const v4u* qp = (const v4u*)(Qs + (16 * j + n) * QSTR + 16 * ks);
                        v8u b; v4u b0 = qp[0], b1 = qp[1];
                        b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
                        S[j] = wmma(af, __builtin_bit_cast(v16h, b), S[j]);
                    }
                }
            }
        }
        if (DW > 1) {
            // sum the partial S of the DW waves of this stream in a fixed order (every wave gets the same bits)
            float* sb = Ssm + (size_t)(it & 1) * NSW * DW * NT * 8 * 32;
            if (active) {
                #pragma unroll
                for (int j = 0; j < NT; j++)
                    #pragma unroll
                    for (int r = 0; r < 8; r++) sb[((((size_t) sw * DW + dw) * NT + j) * 8 + r) * 32 + lane] = S[j][r];
            }
            __syncthreads();
            if (active) {
                #pragma unroll
                for (int j = 0; j < NT; j++)
                    #pragma unroll
                    for (int r = 0; r < 8; r++) {
                        float t = sb[((((size_t) sw * DW) * NT + j) * 8 + r) * 32 + lane];
                        #pragma unroll
                        for (int w = 1; w < DW; w++) t += sb[((((size_t) sw * DW + w) * NT + j) * 8 + r) * 32 + lane];
                        S[j][r] = t;
                    }
            }
        }
        if (!active) continue;
#if TQF
        const bool more = it + 1 < iters && (sw + (it + 1) * NSW) < nslabs;
        if (more) { slab_rows(it + 1); load_k(); }       // kw is dead once S is formed
#endif

        // ---- online softmax (log2 domain); lane owns q row n of every tile and tokens 2r + half, r = 0..7
        const bool full = (n0 + 15 < n_end) && (n0 >= KEY_LO) && (!CAUSAL || (float)(n0 + 15) <= qmin);
        u32 pa[NT][8];
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            float s[8], p[8];
            float mx = NEG_INF;
            #pragma unroll
            for (int r = 0; r < 8; r++) s[r] = S[j][r] * sl;
            if (!full) {
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    int tk = n0 + 2 * r + half;
                    bool ok = rvalid[j] && tk < n_end && tk >= KEY_LO && (!CAUSAL || (float) tk <= qabs[j]);
                    if (!ok) s[r] = NEG_INF;
                }
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) mx = fmaxf(mx, s[r]);
            mx = fmaxf(mx, __shfl_xor(mx, 16));
            float mn = fmaxf(m[j], mx);
            float me = (mn == NEG_INF) ? 0.f : mn;
            float alpha = (m[j] == NEG_INF) ? 0.f : __builtin_amdgcn_exp2f(m[j] - me);
            float ps = 0.f;
            #pragma unroll
            for (int r = 0; r < 8; r++) { p[r] = __builtin_amdgcn_exp2f(s[r] - me); ps += p[r]; }
            lsum[j] = lsum[j] * alpha + ps;
            m[j] = mn;
            if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {   // skip the O rescale when no row's max moved
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    float ar = __shfl(alpha, 2 * r + half);          // O row 2r + half
                    #pragma unroll
                    for (int dt = 0; dt < DTW; dt++) O[j][dt][r] *= ar;
                }
            }
            // P as the A operand: element e = token e; lane half 0 owns even tokens, half 1 odd ones
            u32 own[4], oth[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) own[k] = pack_h2(p[2 * k], p[2 * k + 1]);
            #pragma unroll
            for (int k = 0; k < 4; k++) oth[k] = __shfl_xor(own[k], 16);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                u32 lo = half ? oth[k] : own[k];     // half-0 values (regs 2k, 2k+1)
                u32 hi = half ? own[k] : oth[k];     // half-1 values
                pa[j][2 * k] = perm(hi, lo, 0x05040100u);        // tokens 4k, 4k+1
                pa[j][2 * k + 1] = perm(hi, lo, 0x07060302u);    // tokens 4k+2, 4k+3
            }
        }

        // ---- O += P x V ; lane n owns the d run d0 + DTW*n .. + DTW - 1 of each token
#if TQF
        {
            u32 ze[16], zo[16], vsp[8];
            #pragma unroll
            for (int e = 0; e < 16; e++) zbuild<VB, 0>(vr[e][0], vr[e][1], ze[e], zo[e]);
            #pragma unroll
            for (int i = 0; i < 8; i++) vsp[i] = perm(vsc[2 * i + 1], vsc[2 * i], 0x05040100u);   // scales of tokens 2i, 2i + 1
            if (more) load_v();                              // vr is dead: the next slab's V runs fly during the lookups below
            #pragma unroll
            for (int u = 0; u < DTW / 2; u++) {
                v8u b0, b1;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    b0[i] = hmul2(lut2(cbv_s, zbyte(ze[2 * i], u), zbyte(ze[2 * i + 1], u)), vsp[i]);   // tokens 2i, 2i + 1 at d column 2u
                    b1[i] = hmul2(lut2(cbv_s, zbyte(zo[2 * i], u), zbyte(zo[2 * i + 1], u)), vsp[i]);   // ... at d column 2u + 1
                }
                #pragma unroll
                for (int j = 0; j < NT; j++) {
                    v8u pv;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) pv[i] = pa[j][i];
                    v16h af = __builtin_bit_cast(v16h, pv);
                    O[j][2 * u] = wmma(af, __builtin_bit_cast(v16h, b0), O[j][2 * u]);
                    O[j][2 * u + 1] = wmma(af, __builtin_bit_cast(v16h, b1), O[j][2 * u + 1]);
                }
            }
        }
#else
        const int cur_row0 = row0, cur_lim = lim;
        const int dcol = dw * HDW + DTW * n;
        const int gi_v = dcol >> 5, p0 = dcol & 31;
        u32 vr[16][NRW > 0 ? NRW : 1];
        u32 vsc[16];
        #pragma unroll
        for (int e = 0; e < 16; e++) {
            size_t row = (size_t) cur_row0 + min(e, cur_lim);
            load_run(vr[e], v_cache + row * (GPT * VB) + kvh * VCHW + gi_v * VB, p0);
            vsc[e] = TQ ? ((u32) v_scales[row * NKV + kvh] * 0x10001u) : scale2<VB>(v_scales[row * GPT + kvh * G + gi_v]);
        }
        // next slab's K (kw is dead now)
        if (it + 1 < iters && (sw + (it + 1) * NSW) < nslabs) { slab_rows(it + 1); load_k(); }
        #pragma unroll
        for (int u = 0; u < DTW / 2; u++) {
            u32 dq[16];
            #pragma unroll
            for (int e = 0; e < 16; e++) {
                u32 sh = hmul2(vsc[e], 0x38003800u);
                dq[e] = TQ ? dequant2_tq<VB>(raw_pair_m<VB>(vr[e], 2 * u, 0, RUN_W8, RUN_W8 + RUN_W4, RUN_W8 + RUN_W4 + RUN_W2), vsc[e], cbv_s)
                             : dequant2<VB>(raw_pair_m<VB>(vr[e], 2 * u, 0, RUN_W8, RUN_W8 + RUN_W4, RUN_W8 + RUN_W4 + RUN_W2), vsc[e], sh);
            }
            v8u b0, b1;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0[i] = perm(dq[2 * i + 1], dq[2 * i], 0x05040100u);   // tokens 2i, 2i+1 at d column 2u
                b1[i] = perm(dq[2 * i + 1], dq[2 * i], 0x07060302u);   // ... at d column 2u + 1
            }
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                v8u pv;
                #pragma unroll
                for (int i = 0; i < 8; i++) pv[i] = pa[j][i];
                v16h af = __builtin_bit_cast(v16h, pv);
                O[j][2 * u] = wmma(af, __builtin_bit_cast(v16h, b0), O[j][2 * u]);
                O[j][2 * u + 1] = wmma(af, __builtin_bit_cast(v16h, b1), O[j][2 * u + 1]);
            }
        }
#endif
    }

    // ---- row sums across the two token halves
    #pragma unroll
    for (int j = 0; j < NT; j++) lsum[j] += __shfl_xor(lsum[j], 16);

    // ---- merge the NSW streams through LDS: streams 1.. publish (m, l, O), stream 0 combines and stores
    constexpr int NO = NT * DTW * 8;
    float* Osm = (float*) scratch;                         // [NSW-1][DW][NO][32]
    float* Msm = Osm + (size_t)(NSW - 1) * DW * NO * 32;   // [NSW-1][NT][32]
    float* Lsm = Msm + (NSW - 1) * NT * 32;
    __syncthreads();
    if (sw > 0) {
        float* ob = Osm + ((size_t)(sw - 1) * DW + dw) * NO * 32;
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            #pragma unroll
            for (int dt = 0; dt < DTW; dt++)
                #pragma unroll
                for (int r = 0; r < 8; r++) ob[((j * DTW + dt) * 8 + r) * 32 + lane] = O[j][dt][r];
            if (dw == 0) {
                Msm[((sw - 1) * NT + j) * 32 + lane] = m[j];
                Lsm[((sw - 1) * NT + j) * 32 + lane] = lsum[j];
            }
        }
    }
    __syncthreads();
    if (sw > 0) return;

    const size_t pbase = (size_t)(pid * num_splits + split);
    float* po = partial_o + pbase * RP * HD;
    float* pml = partial_ml + pbase * RP * 2;
    #pragma unroll
    for (int j = 0; j < NT; j++) {
        float M = m[j];
        #pragma unroll
        for (int w = 0; w < NSW - 1; w++) M = fmaxf(M, Msm[(w * NT + j) * 32 + lane]);
        float Ms = (M == NEG_INF) ? 0.f : M;
        float f0 = (m[j] == NEG_INF) ? 0.f : __builtin_amdgcn_exp2f(m[j] - Ms);
        float L = lsum[j] * f0;
        float fo[8];
        #pragma unroll
        for (int r = 0; r < 8; r++) fo[r] = __shfl(f0, 2 * r + half);
        #pragma unroll
        for (int dt = 0; dt < DTW; dt++)
            #pragma unroll
            for (int r = 0; r < 8; r++) O[j][dt][r] *= fo[r];
        #pragma unroll
        for (int w = 0; w < NSW - 1; w++) {
            float mw = Msm[(w * NT + j) * 32 + lane];
            float fw = (mw == NEG_INF) ? 0.f : __builtin_amdgcn_exp2f(mw - Ms);
            L += Lsm[(w * NT + j) * 32 + lane] * fw;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                float fr = __shfl(fw, 2 * r + half);
                #pragma unroll
                for (int dt = 0; dt < DTW; dt++)
                    O[j][dt][r] += Osm[(((size_t) w * DW + dw) * NO + (j * DTW + dt) * 8 + r) * 32 + lane] * fr;
            }
        }
        if (dw == 0 && half == 0 && 16 * j + n < ROWS) {
            pml[(16 * j + n) * 2] = M;                     // log2 domain, see qdec_combine
            pml[(16 * j + n) * 2 + 1] = L;
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row = 16 * j + 2 * r + half;
            if (row < ROWS) {
                float* dst = po + (size_t) row * HD + dw * HDW + DTW * n;
                #pragma unroll
                for (int dt = 0; dt < DTW; dt++) dst[dt] = O[j][dt][r];
            }
        }
    }
}

// grid (batch * NKV, ROWS), HD threads: reduce the splits of one q row, normalize, rotate back with h32, write fp16 [batch, QLEN, NQH, HD]
extern "C" __global__ void __launch_bounds__(HD)
qdec_combine(
    const float* __restrict__ partial_o, const float* __restrict__ partial_ml, u16* __restrict__ out,
    const u16* __restrict__ h32, const float* __restrict__ sinks, int num_splits)
{
    __shared__ float hs[32 * 32];
    __shared__ float xs[HD];
    const int pid = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int batch = pid / NKV, kvh = pid % NKV;
    const int row_q = r / GROUP, hl = r % GROUP;
    if (!TQ) for (int i = d; i < 1024; i += HD) hs[i] = h2f(h32[i]);

    float M = NEG_INF;
    for (int s = 0; s < num_splits; s++) M = fmaxf(M, partial_ml[(((size_t) pid * num_splits + s) * RP + r) * 2]);
    float sink = 0.f;
    if (HAS_SINKS) { sink = sinks[kvh * GROUP + hl] * LOG2E; M = fmaxf(M, sink); }
    float Ms = (M == NEG_INF) ? 0.f : M;
    float acc = 0.f, L = 0.f;
    for (int s = 0; s < num_splits; s++) {
        size_t pb = ((size_t) pid * num_splits + s) * RP + r;
        float ms = partial_ml[pb * 2];
        if (ms == NEG_INF) continue;                     // empty split: its partial_o was never written
        float w = __builtin_amdgcn_exp2f(ms - Ms);
        acc += partial_o[pb * HD + d] * w;
        L += partial_ml[pb * 2 + 1] * w;
    }
    if (HAS_SINKS) L += __builtin_amdgcn_exp2f(sink - Ms);
    xs[d] = (float)(half_t)(acc / (L == 0.f ? 1.f : L));
    __syncthreads();
    const int g = d >> 5, c = d & 31;
    float y = 0.f;
    if (TQ) y = xs[d];                                   // the output is un-rotated outside
    else {
    #pragma unroll
    for (int k = 0; k < 32; k++) y += xs[g * 32 + k] * hs[k * 32 + c];
    }
    out[((size_t)((batch * QLEN + row_q) * NQH + kvh * GROUP + hl)) * HD + d] = __builtin_bit_cast(u16, (half_t) y);
}
