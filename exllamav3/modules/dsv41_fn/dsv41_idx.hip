// DS4.1 indexer scoring kernels for gfx1151 (wave32, WMMA f16 -> f32).
//   score[r, t] = sum_h w[r, h] * relu(q[r, h, :] . k[t, :])        (32 heads x 128 dims, fp16 q / k, fp32 accumulate)
// The reference computes the same sum with an fp32 einsum + relu + weighted head sum; the products are exact in fp32 (fp4-valued
// q and k), only the order of the 128-term sums differs, so scores agree to fp32 rounding.
// Keys with t >= clen(r) = (start + r + 1) / ratio score -inf (causal reach of row r).
//
// WMMA layout (gfx11): D[16x16] = A[16x16] * B[16x16]; lane l holds A row (l & 15) and B column (l & 15), 16 k values each (the two
// half-waves carry the same operand); the accumulator register i of lane l is D[row 2i + (l >> 4)][col l & 15].
// Here A = 16 heads x 16 dims of q, B = 16 dims x 16 keys of k^T, so one lane owns one key and the head sum is 8 registers
// plus one half-wave exchange.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef unsigned int u8v __attribute__((ext_vector_type(8)));
#define WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
#define QS 136                                   // halves per head row in LDS (272 B: conflict-free 16-byte reads)

static __device__ __forceinline__ h16 ld_frag(const _Float16* p)       // 16 contiguous halves (32 B, 16-byte aligned)
{
    uint4 a = ((const uint4*) p)[0], b = ((const uint4*) p)[1];
    u8v v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(h16, v);
}

// Packed fp4 keys: kq [rows, 64] bytes (dim 2j in the low nibble of byte j, e2m1: sign bit 3, magnitude code 0..7), ks [rows, 4]
// e8m0 power-of-two scales per 32 dims. The decode of a value is exact in fp16 (the pool holds fp4-quantized keys).
static __device__ __forceinline__ float e2m1_mag(unsigned m)
{
    return m < 4 ? 0.5f * (float) m : (float) (2 + (m & 1)) * (float) (1 << ((m >> 1) - 2));
}
static __device__ __forceinline__ h16 ld_kfrag(const _Float16* kp, const uint8_t* kq, const uint8_t* ks, long key, int s)
{
    if (!kq) return ld_frag(kp + key * 128 + 16 * s);
    uint2 w = *(const uint2*) (kq + key * 64 + 8 * s);
    float sc = __uint_as_float(((unsigned) ks[key * 4 + (s >> 1)]) << 23);
    h16 v;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        unsigned b = (j < 4 ? (w.x >> (8 * j)) : (w.y >> (8 * (j - 4)))) & 0xff;
        unsigned lo = b & 15, hi = b >> 4;
        float a = e2m1_mag(lo & 7) * sc, c = e2m1_mag(hi & 7) * sc;
        v[2 * j] = (_Float16) ((lo & 8) ? -a : a);
        v[2 * j + 1] = (_Float16) ((hi & 8) ? -c : c);
    }
    return v;
}

// Dense scoring of rows [0, R) against keys [0, ldo). q [R, 32, 128], w [R, 32], kp [rows, 128], out [R, ldo].
// grid (ceil(ldo / 128), ceil(R / rows_per_blk)), block 128 (4 waves, 32 keys each, key fragments live in registers while the
// block walks its rows; q rows are staged in LDS, double buffered).
extern "C" __global__ __launch_bounds__(128)
void ds41_idx_score_dense(const _Float16* __restrict__ q, const float* __restrict__ w, const _Float16* __restrict__ kp,
                          const uint8_t* __restrict__ kq, const uint8_t* __restrict__ ks,
                          float* __restrict__ out, int R, int nt, int ldo, int start, int ratio, int rows_per_blk)
{
    __shared__ _Float16 qs[2][32 * QS];
    __shared__ float ws[2][32];
    const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int key0 = blockIdx.x * 128 + wv * 32;
    const int r0 = blockIdx.y * rows_per_blk, r1 = min(R, r0 + rows_per_blk);
    const bool active = key0 < ldo;
    h16 kb[2][8];
    if (active) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int key = min(key0 + 16 * j + l16, nt - 1);
            #pragma unroll
            for (int s = 0; s < 8; ++s) kb[j][s] = ld_kfrag(kp, kq, ks, max(key, 0), s);
        }
    }
    uint4 pre[4]; float pw = 0.f;
    auto gload = [&](int r) {
        const uint4* src = (const uint4*) (q + (long) r * 4096);
        #pragma unroll
        for (int i = 0; i < 4; ++i) pre[i] = src[tid + 128 * i];
        if (tid < 32) pw = w[(long) r * 32 + tid];
    };
    auto sstore = [&](int buf) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + 128 * i;                       // 16-byte unit: head = idx / 16, chunk = idx % 16
            *(uint4*) (&qs[buf][(idx >> 4) * QS + (idx & 15) * 8]) = pre[i];
        }
        if (tid < 32) ws[buf][tid] = pw;
    };
    if (r0 < r1) { gload(r0); sstore(0); }
    __syncthreads();
    for (int r = r0; r < r1; ++r) {
        const int buf = (r - r0) & 1;
        const bool more = r + 1 < r1;
        if (more) gload(r + 1);
        const int clen = (start + r + 1) / ratio;
        if (active) {
            f8 acc[2][2];
            #pragma unroll
            for (int j = 0; j < 2; ++j) { acc[j][0] = f8{}; acc[j][1] = f8{}; }
            const bool run0 = key0 < clen, run1 = key0 + 16 < clen;
            if (run0) {
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    h16 a0 = ld_frag(&qs[buf][l16 * QS + 16 * s]);
                    h16 a1 = ld_frag(&qs[buf][(16 + l16) * QS + 16 * s]);
                    acc[0][0] = WMMA(a0, kb[0][s], acc[0][0]);
                    acc[0][1] = WMMA(a1, kb[0][s], acc[0][1]);
                    if (run1) {
                        acc[1][0] = WMMA(a0, kb[1][s], acc[1][0]);
                        acc[1][1] = WMMA(a1, kb[1][s], acc[1][1]);
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                float tot = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int h = 2 * i + hi;
                    tot += fmaxf(acc[j][0][i], 0.f) * ws[buf][h] + fmaxf(acc[j][1][i], 0.f) * ws[buf][16 + h];
                }
                tot += __shfl_xor(tot, 16, 32);
                int key = key0 + 16 * j + l16;
                if (hi == 0 && key < ldo) out[(long) r * ldo + key] = key < clen ? tot : -INFINITY;
            }
        }
        if (more) sstore(buf ^ 1);
        __syncthreads();
    }
}

// Scoring of the candidate slots only. blk [R, nsel] int32 (block ids of 8 keys, -1 = none), slot j of row r is key
// blk[r, j / 8] * 8 + j % 8; out [R, nsel * 8]. grid (ceil(nslots / (64 * tpw)), R), block 128 (each wave scores tpw tiles of
// 16 slots; the q row of the block sits in registers).
extern "C" __global__ __launch_bounds__(128)
void ds41_idx_score_cand(const _Float16* __restrict__ q, const float* __restrict__ w, const _Float16* __restrict__ kp,
                         const uint8_t* __restrict__ kq, const uint8_t* __restrict__ ks,
                         const int* __restrict__ blk, float* __restrict__ out, int R, int nsel, int start, int ratio, int tpw)
{
    const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int r = blockIdx.y, nslots = nsel * 8;
    const int clen = (start + r + 1) / ratio;
    const int tile0 = (blockIdx.x * 4 + wv) * tpw;
    if (tile0 * 16 >= nslots) return;
    h16 a0[8], a1[8];
    {
        const _Float16* p0 = q + ((long) r * 32 + l16) * 128;
        const _Float16* p1 = q + ((long) r * 32 + 16 + l16) * 128;
        #pragma unroll
        for (int s = 0; s < 8; ++s) { a0[s] = ld_frag(p0 + 16 * s); a1[s] = ld_frag(p1 + 16 * s); }
    }
    float wr[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { wr[i] = w[(long) r * 32 + 2 * i + hi]; wr[8 + i] = w[(long) r * 32 + 16 + 2 * i + hi]; }
    for (int t = 0; t < tpw; ++t) {
        const int slot = (tile0 + t) * 16 + l16;
        if ((tile0 + t) * 16 >= nslots) break;
        int b = slot >> 3;
        int id = b < nsel ? blk[(long) r * nsel + b] : -1;
        int key = id * 8 + (slot & 7);
        bool valid = id >= 0 && key < clen;
        h16 kb[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) kb[s] = ld_kfrag(kp, kq, ks, valid ? key : 0, s);
        f8 c0 = f8{}, c1 = f8{};
        #pragma unroll
        for (int s = 0; s < 8; ++s) { c0 = WMMA(a0[s], kb[s], c0); c1 = WMMA(a1[s], kb[s], c1); }
        float tot = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) tot += fmaxf(c0[i], 0.f) * wr[i] + fmaxf(c1[i], 0.f) * wr[8 + i];
        tot += __shfl_xor(tot, 16, 32);
        if (hi == 0 && slot < nslots) out[(long) r * nslots + slot] = valid ? tot : -INFINITY;
    }
}
