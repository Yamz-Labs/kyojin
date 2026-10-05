// Gated-residual mix chain (Qwen prefill), hand-written for gfx11.5 WMMA f16 (wave32). Replaces
//   dm = normed @ proj^T (fp16 out), t = silu(dm/H), post = 2 sigmoid(dm/H)     -> kernel gr_g1
//   g = t @ up^T (fp16), mixed = rne16(mean_h sigmoid(g) * normed)               -> kernel gr_g2 (g never written)
// Reduction order = one fp32 WMMA 16x16x16 chain per output element over k = 0, 16, 32, ... in order (no split-K),
// the order of the rocBLAS Tensile kernel the baseline uses (MT128x128x32 MI16x16x16, GSU1). Independent of R.
// Fragment layout (gfx1151): A: lane l, elem r -> A[l & 15][r]; B: lane l, elem r -> B[r][l & 15]; C: lane l, reg r -> C[2r + (l >> 4)][l & 15]
// MEMORY SAFETY: every global address comes from the IX_* macros with row clamps (RCL, NCL); every global store is guarded by
// the *_OK macros. check_gr_bounds.py parses these macros and replays all accesses over R = 1..4096 (it must fail when a clamp or guard is mutated).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 half_t;
typedef uint32_t v4u __attribute__((ext_vector_type(4)));   // native vector (the uint4 struct made the compiler park the load registers in scratch)
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define K1 10240      // H * D
#define DD 2560
#define HS 4
#define RANK 320
#define NTOT 324      // proj rows = RANK + HS
#ifndef BM1
#define BM1 32        // WG tile rows
#endif
#ifndef BN1
#define BN1 128       // WG tile columns; must divide 128 (the baseline's stagger is keyed on the 128-column Tensile tile index)
#endif
#ifndef KST
#define KST 64        // K per pipeline stage
#endif
#ifndef NBUF
#define NBUF 2        // LDS buffers: 2 = double buffer, 1 = single buffer with two barriers per stage
#endif
#define STAGH 128     // stagger step in K halves (Tensile StaggerUStride 256 B): column tile t starts its K loop at K = t * STAGH (wraps)
#define NS1 (K1 / KST)
#define WM (BM1 / 32)          // waves along m (32 rows each); 8 waves per WG
#define WN (8 / WM)            // waves along n
#define WNS (BN1 / WN)         // columns per wave
#define NNI (WNS / 16)         // n tiles per wave
#define CPR (KST / 8)          // 16 B chunks per tile row per stage
#define LAN (BM1 * CPR / 256)  // A chunks per thread per stage
#define LBN (BN1 * CPR / 256)  // B chunks per thread per stage
#define KSTEPS (KST / 16)
#define KCH(s_, t_) (((s_) + (t_) * (STAGH / KST)) % NS1)   // K stage processed at loop step s_ by column tile t_
#define LDK (KST + 8) // LDS row stride in halves (pad)
#define SA_ELEMS (BM1 * LDK)
#define SB_ELEMS (BN1 * LDK)
#ifndef PF
#define PF 2          // global-load prefetch depth in stages (2 needs NBUF 2)
#endif
#define BM2S 128      // gr_g2s tile rows; tile columns = 4 streams x 32 d
#define KS2 32
#define NS2 (RANK / KS2)
#define LDK2 (KS2 + 8)
#define SA2_ELEMS (BM2S * LDK2)
#define SB2_ELEMS (128 * LDK2)
#define LD2_R(idx_) ((idx_) >> 2)
#define LD2_C(idx_) (((idx_) & 3) * 8)
#define LAI2(row_, col_) ((row_) * LDK2 + (col_))
#define UROW(row_, db_) ((((row_) >> 5) * DD) + (db_) * 32 + ((row_) & 31))   // up row of B-tile row: stream (row / 32), d = db * 32 + row % 32
#ifndef G2ABL
#define G2ABL 0
#endif
#define BM2 64
#define BD2 32

#define RCL(r_, R_) min((r_), (R_) - 1)
#define NCL(n_) min((n_), NTOT - 1)
#define IX_X(r_, k_) (((int64_t) (r_)) * K1 + (k_))          // normed (R, K1) / proj (NTOT, K1)
#define IX_W(n_, k_) (((int64_t) (n_)) * K1 + (k_))
#define IX_T(r_, n_) (((int64_t) (r_)) * RANK + (n_))        // t (R, RANK) / up (K1, RANK)
#define IX_U(n_, k_) (((int64_t) (n_)) * RANK + (k_))
#define IX_PO(r_, j_) (((int64_t) (r_)) * HS + (j_))         // post (R, HS)
#define IX_XP(r_, k_, p_) (((int64_t) (r_)) * (p_) + (k_))     // normed rows at token pitch p_ (gr_g1 reads a padded buffer)
#define IX_NGP(r_, h_, d_, p_) ((((int64_t) (r_)) * (p_)) + (h_) * DD + (d_))   // normed (R, p_ >= HS*DD) at token pitch p_
#define IX_NG(r_, h_, d_) ((((int64_t) (r_)) * HS + (h_)) * DD + (d_))   // normed (R*HS, DD)
#define IX_MX(r_, d_) (((int64_t) (r_)) * DD + (d_))         // mixed (R, DD)
#define IX_MXP(r_, d_, p_) (((int64_t) (r_)) * (p_) + (d_))   // mixed at row pitch p_ (gr_g2s may write a padded buffer)
#define LD_R(idx_) ((idx_) / CPR)                             // loader: tile row of 16 B chunk idx
#define LD_C(idx_) (((idx_) % CPR) * 8)                       // loader: first half of the chunk in the KST-wide k stage
#define LAI(row_, col_) ((row_) * LDK + (col_))
#define T_OK(row_, n_, R_) (((row_) < (R_)) && ((n_) < RANK))
#define PO_OK(row_, n_, R_) (((row_) < (R_)) && ((n_) >= RANK) && ((n_) < NTOT))
#define MX_OK(row_, R_) ((row_) < (R_))

__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }

__device__ __forceinline__ v16h ld16(const half_t* base, int64_t off)   // 16 halves = 32 B, 32 B aligned
{
    const uint4* p = (const uint4*) (base + off);
    uint4 x = p[0], y = p[1];
    typedef uint32_t v8u __attribute__((ext_vector_type(8)));
    v8u r = {x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
    return __builtin_bit_cast(v16h, r);
}
__device__ __forceinline__ v16h ldl16(const half_t* base, int off)      // from LDS
{
    return ld16(base, off);
}

// ---- elementwise parts (same formulas as torch silu / sigmoid on fp16 / fp32 and the Triton gr_gate kernel)
__device__ __forceinline__ half_t silu_div(float dmf) { _Float16 d4 = (_Float16) (dmf / 4.0f); float x = (float) d4; if (dmf == 0.0f) return (_Float16) 0.0f; /* torch returns +0 for an exact -0 input */ return (_Float16) (x / (1.0f + expf(-x))); }
__device__ __forceinline__ float post_f(float dmf) { return 2.0f * (1.0f / (1.0f + expf(-(dmf / 4.0f)))); }
__device__ __forceinline__ half_t gate4(const float* g, const float* n)
{
    float acc = 0.0f;
    #pragma unroll
    for (int h = 0; h < HS; ++h) acc += (1.0f / (1.0f + expf(-g[h]))) * n[h];
    return (_Float16) (acc / 4.0f);
}

// ---- G1: dm = normed (R x K1) . proj^T (NTOT x K1). WG 8 waves = 4 (rows 32 each) x 2 (NNI n tiles each); tile 128 x BN1; grid (ceil(NTOT / BN1), ceil(R / 128)).
extern "C" __global__ __launch_bounds__(256) void gr_g1(const half_t* __restrict__ X, const half_t* __restrict__ W,
                                                         half_t* __restrict__ T, float* __restrict__ PO, int R, int XP)
{
    __shared__ __attribute__((aligned(16))) half_t sA[NBUF][SA_ELEMS];
    __shared__ __attribute__((aligned(16))) half_t sB[NBUF][SB_ELEMS];
    const int tid = threadIdx.x, wv = tid >> 5, lane = tid & 31, ll = lane & 15, half = lane >> 4;
    const int wm = wv % WM, wn = wv / WM;
    const int m0 = blockIdx.y * BM1, n0 = blockIdx.x * BN1, ctile = n0 / 128;

    v8f acc[2][NNI];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NNI; ++ni)
            acc[mi][ni] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};

    // load registers are named scalars (arrays of them were spilled to scratch by the compiler in the PF 2 loop)
    v4u ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3;
#if PF == 2
    v4u qa0, qa1, qa2, qa3, qb0, qb1, qb2, qb3;
#endif
    #define LCH(RA_, RB_, i_, k0_) \
        if (LAN > (i_)) { const int idx = tid + 256 * (i_); RA_##i_ = *(const v4u*) (X + IX_XP(RCL(m0 + LD_R(idx), R), (k0_) + LD_C(idx), XP)); } \
        if (LBN > (i_)) { const int idx = tid + 256 * (i_); RB_##i_ = *(const v4u*) (W + IX_W(NCL(n0 + LD_R(idx)), (k0_) + LD_C(idx))); }
    #define GLOADR(RA_, RB_, s_) do { const int kk_ = KCH((s_), ctile) * KST; \
        LCH(RA_, RB_, 0, kk_) LCH(RA_, RB_, 1, kk_) LCH(RA_, RB_, 2, kk_) LCH(RA_, RB_, 3, kk_) } while (0)
    #define SCH(RA_, RB_, i_, buf_) \
        if (LAN > (i_)) { const int idx = tid + 256 * (i_); *(v4u*) (&sA[buf_][LAI(LD_R(idx), LD_C(idx))]) = RA_##i_; } \
        if (LBN > (i_)) { const int idx = tid + 256 * (i_); *(v4u*) (&sB[buf_][LAI(LD_R(idx), LD_C(idx))]) = RB_##i_; }
    #define LSTORER(RA_, RB_, buf_) do { SCH(RA_, RB_, 0, buf_) SCH(RA_, RB_, 1, buf_) SCH(RA_, RB_, 2, buf_) SCH(RA_, RB_, 3, buf_) } while (0)
    #define COMPUTE(cur_) do { \
        _Pragma("unroll") for (int ks = 0; ks < KSTEPS; ++ks) { \
            v16h a[2]; \
            _Pragma("unroll") for (int mi = 0; mi < 2; ++mi) a[mi] = ldl16(sA[cur_], LAI(32 * wm + 16 * mi + ll, 16 * ks)); \
            _Pragma("unroll") for (int ni = 0; ni < NNI; ++ni) { \
                if (n0 + WNS * wn + 16 * ni < NTOT) {       /* wave-uniform: tile entirely past the last proj row */ \
                    v16h b = ldl16(sB[cur_], LAI(WNS * wn + 16 * ni + ll, 16 * ks)); \
                    _Pragma("unroll") for (int mi = 0; mi < 2; ++mi) acc[mi][ni] = wmma(a[mi], b, acc[mi][ni]); } } } } while (0)

#if PF == 2
    // two stages of global loads in flight; NS1 is even
    GLOADR(ra, rb, 0);
    GLOADR(qa, qb, 1);
    LSTORER(ra, rb, 0);
    __syncthreads();
    for (int s = 0; s < NS1; s += 2)
    {
        if (s + 2 < NS1) GLOADR(ra, rb, s + 2);
        COMPUTE(0);
        LSTORER(qa, qb, 1);
        __syncthreads();
        if (s + 3 < NS1) GLOADR(qa, qb, s + 3);
        COMPUTE(1);
        if (s + 2 < NS1) LSTORER(ra, rb, 0);
        __syncthreads();
    }
#else
    GLOADR(ra, rb, 0);
    LSTORER(ra, rb, 0);
    __syncthreads();
    for (int s = 0; s < NS1; ++s)
    {
        const int cur = NBUF == 2 ? (s & 1) : 0;
        if (s + 1 < NS1) GLOADR(ra, rb, s + 1);
        COMPUTE(cur);
#if NBUF == 2
        if (s + 1 < NS1) LSTORER(ra, rb, cur ^ 1);
        __syncthreads();
#else
        __syncthreads();
        if (s + 1 < NS1) LSTORER(ra, rb, 0);
        __syncthreads();
#endif
    }
#endif
    // epilogue: dm = fp16(acc); n < RANK -> t = silu(dm / H); RANK <= n < NTOT -> post = 2 sigmoid(dm / H)
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NNI; ++ni)
        {
            const int n = n0 + WNS * wn + 16 * ni + ll;
            if (n0 + WNS * wn + 16 * ni < NTOT)
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                {
                    const int row = m0 + 32 * wm + 16 * mi + 2 * r + half;
                    const float dmf = (float) (_Float16) acc[mi][ni][r];
                    if (T_OK(row, n, R)) T[IX_T(row, n)] = silu_div(dmf);
                    if (PO_OK(row, n, R)) PO[IX_PO(row, n - RANK)] = post_f(dmf);
                }
        }
}

// ---- G2: g = t (R x RANK) . up^T (K1 x RANK), gate in the epilogue. WG 8 waves = 4 (16 rows each) x 2 (16 d each); each wave owns
// the 4 stream tiles of its (rows, d) block, so the mean over streams stays in registers. grid (DD / 32, ceil(R / 64)).
extern "C" __global__ __launch_bounds__(256) void gr_g2(const half_t* __restrict__ T, const half_t* __restrict__ U,
                                                         const half_t* __restrict__ NG, half_t* __restrict__ MX, int R)
{
    const int tid = threadIdx.x, wv = tid >> 5, lane = tid & 31, ll = lane & 15, half = lane >> 4;
    const int r0 = blockIdx.y * BM2 + 16 * (wv & 3), d0 = blockIdx.x * BD2 + 16 * (wv >> 2);
    const int arow = RCL(r0 + ll, R);
    v8f acc[HS];
    #pragma unroll
    for (int h = 0; h < HS; ++h) acc[h] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll 4
    for (int ks = 0; ks < RANK / 16; ++ks)
    {
#if G2ABL == 3
        const v16h a = ld16(T, IX_T(arow, 0));          // timing ablation: A loaded once (subset of the accesses)
#else
        const v16h a = ld16(T, IX_T(arow, 16 * ks));
#endif
        #pragma unroll
        for (int h = 0; h < HS; ++h)
#if G2ABL == 2
            acc[h] = wmma(a, ld16(U, IX_U(h * DD + d0 + ll, 0)), acc[h]);   // timing ablation: B k fixed (subset of the accesses)
#else
            acc[h] = wmma(a, ld16(U, IX_U(h * DD + d0 + ll, 16 * ks)), acc[h]);
#endif
    }
    const int d = d0 + ll;
    #pragma unroll
    for (int r = 0; r < 8; ++r)
    {
        const int row = r0 + 2 * r + half;
        const int rowc = RCL(row, R);
        float g[HS], n[HS];
        #pragma unroll
        for (int h = 0; h < HS; ++h) { g[h] = (float) (_Float16) acc[h][r];
#if G2ABL == 1
          n[h] = 1.0f;                                  // timing ablation: no normed loads
#else
          n[h] = (float) NG[IX_NG(rowc, h, d)];
#endif
        }
        if (MX_OK(row, R)) MX[IX_MX(row, d)] = gate4(g, n);
    }
}


// ---- G2s: same as G2 but both operands staged through LDS (coalesced loads, two stages in flight). WG tile 128 rows x (4 streams x 32 d);
// 8 waves = 4 (32 rows each) x 2 (16 d each); a wave owns the 4 stream tiles of its (rows, d) block (gate in registers). grid (DD / 32, ceil(R / 128)).
extern "C" __global__ __launch_bounds__(256) void gr_g2s(const half_t* __restrict__ T, const half_t* __restrict__ U,
                                                          const half_t* __restrict__ NG, half_t* __restrict__ MX, int R, int MXP, int NGP)
{
    __shared__ __attribute__((aligned(16))) half_t sA[2][SA2_ELEMS];
    __shared__ __attribute__((aligned(16))) half_t sB[2][SB2_ELEMS];
    const int tid = threadIdx.x, wv = tid >> 5, lane = tid & 31, ll = lane & 15, half = lane >> 4;
    const int wm = wv & 3, wd = wv >> 2;
    const int dblk = blockIdx.x, m0 = blockIdx.y * BM2S;
    v8f acc[2][HS];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int h = 0; h < HS; ++h) acc[mi][h] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    v4u ra0, ra1, rb0, rb1, qa0, qa1, qb0, qb1;
    #define G2L(RA_, RB_, i_, k0_) { const int idx = tid + 256 * (i_); \
        RA_##i_ = *(const v4u*) (T + IX_T(RCL(m0 + LD2_R(idx), R), (k0_) + LD2_C(idx))); \
        RB_##i_ = *(const v4u*) (U + IX_U(UROW(LD2_R(idx), dblk), (k0_) + LD2_C(idx))); }
    #define G2LOAD(RA_, RB_, s_) do { const int kk_ = (s_) * KS2; G2L(RA_, RB_, 0, kk_) G2L(RA_, RB_, 1, kk_) } while (0)
    #define G2S(RA_, RB_, i_, buf_) { const int idx = tid + 256 * (i_); \
        *(v4u*) (&sA[buf_][LAI2(LD2_R(idx), LD2_C(idx))]) = RA_##i_; \
        *(v4u*) (&sB[buf_][LAI2(LD2_R(idx), LD2_C(idx))]) = RB_##i_; }
    #define G2STORE(RA_, RB_, buf_) do { G2S(RA_, RB_, 0, buf_) G2S(RA_, RB_, 1, buf_) } while (0)
    #define G2COMPUTE(cur_) do { \
        _Pragma("unroll") for (int ks = 0; ks < KS2 / 16; ++ks) { \
            v16h a[2]; \
            _Pragma("unroll") for (int mi = 0; mi < 2; ++mi) a[mi] = ldl16(sA[cur_], LAI2(32 * wm + 16 * mi + ll, 16 * ks)); \
            _Pragma("unroll") for (int h = 0; h < HS; ++h) { \
                v16h b = ldl16(sB[cur_], LAI2(32 * h + 16 * wd + ll, 16 * ks)); \
                _Pragma("unroll") for (int mi = 0; mi < 2; ++mi) acc[mi][h] = wmma(a[mi], b, acc[mi][h]); } } } while (0)
    G2LOAD(ra, rb, 0);
    G2LOAD(qa, qb, 1);
    G2STORE(ra, rb, 0);
    __syncthreads();
    for (int s = 0; s < NS2; s += 2)
    {
        if (s + 2 < NS2) G2LOAD(ra, rb, s + 2);
        G2COMPUTE(0);
        G2STORE(qa, qb, 1);
        __syncthreads();
        if (s + 3 < NS2) G2LOAD(qa, qb, s + 3);
        G2COMPUTE(1);
        if (s + 2 < NS2) G2STORE(ra, rb, 0);
        __syncthreads();
    }
    const int d = dblk * 32 + 16 * wd + ll;
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int r = 0; r < 8; ++r)
        {
            const int row = m0 + 32 * wm + 16 * mi + 2 * r + half;
            const int rowc = RCL(row, R);
            float g[HS], n[HS];
            #pragma unroll
            for (int h = 0; h < HS; ++h) { g[h] = (float) (_Float16) acc[mi][h][r]; n[h] = (float) NG[IX_NGP(rowc, h, d, NGP)]; }
            if (MX_OK(row, R)) MX[IX_MXP(row, d, MXP)] = gate4(g, n);
        }
}

// ---- tests of the elementwise parts (device functions shared with the kernels), n-guarded
extern "C" __global__ void gr_elem_test(const half_t* __restrict__ in, half_t* __restrict__ o_silu, float* __restrict__ o_post, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float f = (float) in[i]; o_silu[i] = silu_div(f); o_post[i] = post_f(f); }
}
extern "C" __global__ void gr_gate_test(const half_t* __restrict__ g, const half_t* __restrict__ nn, half_t* __restrict__ o, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float gg[HS], nv[HS];
        for (int h = 0; h < HS; ++h) { gg[h] = (float) g[(int64_t) i * HS + h]; nv[h] = (float) nn[(int64_t) i * HS + h]; }
        o[i] = gate4(gg, nv);
    }
}
