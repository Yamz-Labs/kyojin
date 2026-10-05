// Hand HIP kernels for the two torch fallbacks left in the Qwen prefill (ROCm, gfx1151, wave32):
//   * PLELayer.forward_streams_reference (everything after the key/value projections)
//   * deinterleave_qg
// Every kernel is bitwise identical to the torch chain it replaces. Rounding points and reduction orders
// (all found by probing torch on the target device, see VERDICT.md):
//   - grouped RMS mean(x^2): torch reduce kernel for a (rows>=16, 2560) fp32 tensor = one wave32 per row,
//     lane t sums vec4 groups (t + 32 j) into 4 accumulators (j = 0..19, in order), acc0+acc1, +acc2, +acc3,
//     then shfl_down tree with offsets 1,2,4,8,16, then times float(1/2560). x*x is rounded before the add (no fma).
//   - bmm (query . key): one sequential fp32 fma chain k = 0..2559, start 0.
//   - depthwise conv k=4, dilation 3: fp32 chain over taps 0..3 (products of halves are exact), rounded to half.
// Compiled with -ffp-contract=off; every fma is explicit.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef DD
#define DD 2560            // hidden size D (per stream)
#endif
#define HS 4               // hyper-connection streams
#define NV (DD / 4)        // vec4 groups per row (640)
#define NJ (NV / 32)       // iterations per lane (20)
#define ST 9               // conv state length (kernel 4 - 1) * dilation 3
#define DIL 3
#define TT 64              // tile edge of the transposing kernels
#define PITCH_V 66         // LDS pitch (halves) of the value tile in ple_norm_conv
#define PITCH_I 75         // LDS pitch (halves) of the conv input tile
#define PITCH_O 66         // LDS pitch (halves) of the conv output tile

// ---- index expressions: the bounds checker (check_ple_bounds.py) evaluates these macros over the real shapes ----
#define IX_K1_ROW(blk, wv)          ((blk) * 8 + (wv))
#define IX_K1_SRC(rl, t, j)         ((rl) * DD + ((t) + 32 * (j)) * 4)
#define IX_K2_ROW(blk, tid)         ((blk) * 64 + (tid))
#define IX_K2_SRC(i, kk)            ((i) * DD + (kk) * 8)
#define IX_K2_W(i, kk)              (((i) % HS) * DD + (kk) * 8)
#define IX_K3_ROW(blk, sl)          ((blk) * TT + (sl))
#define IX_K3_VAL(row, dc, dl)      ((row) * DD + (dc) * TT + (dl))
#define IX_K3_W(h, dc, dl)          ((h) * DD + (dc) * TT + (dl))
#define IX_K3_OUT(b, ch, s, S)      (((size_t)(b) * HS * DD + (ch)) * (size_t)(ST + (S)) + ST + (s))
#define IX_K3B_OUT(b, ch, j, S)     (((size_t)(b) * HS * DD + (ch)) * (size_t)(ST + (S)) + (j))
#define IX_K3B_SRC(b, ch, j)        (((size_t)(b) * HS * DD + (ch)) * ST + (j))
#define IX_K4_IN(b, c, j, S)        (((size_t)(b) * HS * DD + (c)) * (size_t)(ST + (S)) + (j))
#define IX_K4_ROW(b, s, S)          ((size_t)(b) * (S) + (s))
#define IX_K4_VAL(row, d)           ((row) * DD + (d))
#define IX_K4_OUT(row, c)           ((size_t)(row) * HS * DD + (c))
#define IX_DQ_SRC(i, hd8, d)        (((i) / (hd8)) * 2 * (hd8) + (d))

static __device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }
static __device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float(((uint32_t) u) << 16); }
// Materialize an fp32 value: without it the backend folds fptrunc(fmul/fma) into one v_fma_mix_f16 (single rounding, and
// -0 + +0 = +0), which breaks bitwise equality with torch's two separate roundings.
static __device__ __forceinline__ float rnd32(float v) { asm volatile("" : "+v"(v)); return v; }
static __device__ __forceinline__ float clamp_h(float v) {
    // torch.clamp(min=-65504, max=65504) on a float widened from fp16 (NaN passes through)
    if (v > 65504.0f) v = 65504.0f;
    if (v < -65504.0f) v = -65504.0f;
    return v;
}

// Torch's row reduction tree for mean(x^2): acc[4] per lane, combined sequentially, shfl_down 1..16, times 1/D
static __device__ __forceinline__ float tree_finish(float a0, float a1, float a2, float a3) {
    float v = a0;
    v = v + a1; v = v + a2; v = v + a3;
    for (int off = 1; off < 32; off <<= 1) v = v + __shfl_down(v, off, 32);
    return v;
}
static __device__ __forceinline__ float rs_from_sum(float tot, float eps) {
    const float mean = tot * (1.0f / (float) DD);
    // torch.rsqrt on this device is the correctly rounded 1/sqrt (hardware rsq differs in ~11 % of inputs)
    return (float) (1.0 / sqrt((double) (mean + eps)));
}

// ---- K1: 1/rms of every key row (half, clamped) and query row (float). One wave per row.
// rows [0, RH) = key rows (r, h), rows [RH, 2 RH) = query rows.
extern "C" __global__ __launch_bounds__(256)
void ple_rs(const uint16_t* __restrict__ key, const float* __restrict__ strm, float* __restrict__ rs, int RH, float eps)
{
    const int t = threadIdx.x & 31, wv = threadIdx.x >> 5;
    const int row = IX_K1_ROW(blockIdx.x, wv);
    if (row >= 2 * RH) return;
    const bool isq = row >= RH;
    const int rl = isq ? row - RH : row;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    if (!isq) {
        const uint2* p = (const uint2*) key;
        for (int j = 0; j < NJ; j++) {
            uint2 u = p[IX_K1_SRC(rl, t, j) / 4];
            float x0 = clamp_h(h2f(u.x & 0xffff)), x1 = clamp_h(h2f(u.x >> 16));
            float x2 = clamp_h(h2f(u.y & 0xffff)), x3 = clamp_h(h2f(u.y >> 16));
            float s0 = x0 * x0, s1 = x1 * x1, s2 = x2 * x2, s3 = x3 * x3;
            a0 = a0 + s0; a1 = a1 + s1; a2 = a2 + s2; a3 = a3 + s3;
        }
    } else {
        const float4* p = (const float4*) strm;
        for (int j = 0; j < NJ; j++) {
            float4 u = p[IX_K1_SRC(rl, t, j) / 4];
            float s0 = u.x * u.x, s1 = u.y * u.y, s2 = u.z * u.z, s3 = u.w * u.w;
            a0 = a0 + s0; a1 = a1 + s1; a2 = a2 + s2; a3 = a3 + s3;
        }
    }
    float v = tree_finish(a0, a1, a2, a3);
    if (t == 0) rs[row] = rs_from_sum(v, eps);
}

// ---- K2: gate. One lane per (row, stream): sequential fma chain over the normalized key . query, then
// s = sigmoid(sign(g) sqrt(max(|g|, 1e-6))), g = dot * gate_scale. Normalized values recomputed (same ops as torch).
extern "C" __global__ __launch_bounds__(64)
void ple_gate(const uint16_t* __restrict__ key, const float* __restrict__ strm, const float* __restrict__ rs,
              const uint16_t* __restrict__ wk, const uint16_t* __restrict__ wq, float* __restrict__ gs, int RH, float gate_scale)
{
    const int i = IX_K2_ROW(blockIdx.x, threadIdx.x);
    if (i >= RH) return;
    const float rk = rs[i], rq = rs[RH + i];
    float acc = 0.f;
    for (int kk = 0; kk < DD / 8; kk++) {
        const uint4 ku = *(const uint4*) (key + IX_K2_SRC(i, kk));
        const float4 q0 = *(const float4*) (strm + IX_K2_SRC(i, kk));
        const float4 q1 = *(const float4*) (strm + IX_K2_SRC(i, kk) + 4);
        const uint4 wku = *(const uint4*) (wk + IX_K2_W(i, kk));
        const uint4 wqu = *(const uint4*) (wq + IX_K2_W(i, kk));
        const uint32_t kv[4] = {ku.x, ku.y, ku.z, ku.w}, wkv[4] = {wku.x, wku.y, wku.z, wku.w}, wqv[4] = {wqu.x, wqu.y, wqu.z, wqu.w};
        const float qv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int u = 0; u < 8; u++) {
            const uint32_t kw = kv[u >> 1], kww = wkv[u >> 1], qww = wqv[u >> 1];
            const int sh = (u & 1) * 16;
            float kf = clamp_h(h2f((kw >> sh) & 0xffff));
            float kn = kf * rk;
            kn = kn * (bf2f((kww >> sh) & 0xffff) + 1.0f);
            float qn = qv[u] * rq;
            qn = qn * (bf2f((qww >> sh) & 0xffff) + 1.0f);
            acc = fmaf(qn, kn, acc);
        }
    }
    const float g = acc * gate_scale;
    const float a = fabsf(g);
    const float am = a < 1e-6f ? 1e-6f : a;
    const float sgn = (float) ((0.0f < g) - (g < 0.0f));
    const float ss = sgn * sqrtf(am);
    gs[i] = 1.0f / (1.0f + expf(-ss));
}

// ---- K3: gated = s * value (fp32), grouped RMS over D with the torch tree, normed = half(gated * rs * (w + 1)),
// written transposed into conv_stream[b, ch, ST + s]. One block = 64 consecutive rows x one stream h.
extern "C" __global__ __launch_bounds__(256)
void ple_norm_conv(const uint16_t* __restrict__ value, const float* __restrict__ gs, const uint16_t* __restrict__ wc,
                   uint16_t* __restrict__ cstream, int RT, int S, float eps)
{
    __shared__ float s_rs[TT];
    __shared__ uint16_t s_val[TT * PITCH_V];
    const int tid = threadIdx.x, t = tid & 31, wv = tid >> 5;
    const int h = blockIdx.y;
    // pass 1: per-row 1/rms of gated (wave wv owns rows wv*8 .. wv*8+7)
    for (int r8 = 0; r8 < 8; r8++) {
        const int sl = wv * 8 + r8;
        const int row = IX_K3_ROW(blockIdx.x, sl);
        float rsv = 0.f;
        if (row < RT) {
            const float s = gs[row * HS + h];
            const uint2* p = (const uint2*) value;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (int j = 0; j < NJ; j++) {
                uint2 u = p[IX_K1_SRC(row, t, j) / 4];
                float g0 = s * h2f(u.x & 0xffff), g1 = s * h2f(u.x >> 16), g2 = s * h2f(u.y & 0xffff), g3 = s * h2f(u.y >> 16);
                float s0 = g0 * g0, s1 = g1 * g1, s2 = g2 * g2, s3 = g3 * g3;
                a0 = a0 + s0; a1 = a1 + s1; a2 = a2 + s2; a3 = a3 + s3;
            }
            rsv = rs_from_sum(tree_finish(a0, a1, a2, a3), eps);
        }
        if (t == 0) s_rs[sl] = rsv;
    }
    __syncthreads();
    // pass 2: d chunks of 64
    const int sl = tid & 63, dd = tid >> 6;
    const int row = IX_K3_ROW(blockIdx.x, sl);
    const bool ok = row < RT;
    const int b = ok ? row / S : 0, s_in = ok ? row % S : 0;
    const float s_g = ok ? gs[row * HS + h] : 0.f;
    const float rsv = s_rs[sl];
    for (int dc = 0; dc < DD / TT; dc++) {
        // coalesced load of the value tile: row-major, each row 64 halves
        for (int e = tid; e < TT * TT; e += 256) {
            const int rl = e >> 6, dl = e & 63;
            const int rrow = IX_K3_ROW(blockIdx.x, rl);
            s_val[rl * PITCH_V + dl] = rrow < RT ? value[IX_K3_VAL(rrow, dc, dl)] : (uint16_t) 0;
        }
        __syncthreads();
        if (ok) {
#pragma unroll 4
            for (int i = 0; i < TT / 4; i++) {
                const int dl = dd + 4 * i;
                float g = s_g * h2f(s_val[sl * PITCH_V + dl]);
                float x = g * rsv;
                x = rnd32(x * (bf2f(wc[IX_K3_W(h, dc, dl)]) + 1.0f));
                const int ch = h * DD + dc * TT + dl;
                cstream[IX_K3_OUT(b, ch, s_in, S)] = __half_as_ushort(__float2half_rn(x));
            }
        }
        __syncthreads();
    }
}

// ---- K3b: conv state columns of conv_stream (zeros when there is no state)
extern "C" __global__ __launch_bounds__(256)
void ple_copy_state(const uint16_t* __restrict__ cstate, uint16_t* __restrict__ cstream, int B, int S, int has_state)
{
    const size_t n = (size_t) B * HS * DD * ST;
    const size_t i = (size_t) blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int j = (int) (i % ST);
    const size_t bc = i / ST;                 // b * HS*DD + ch
    const int b = (int) (bc / (HS * DD)), ch = (int) (bc % (HS * DD));
    cstream[IX_K3B_OUT(b, ch, j, S)] = has_state ? cstate[IX_K3B_SRC(b, ch, j)] : (uint16_t) 0;
}

// ---- K4: depthwise dilated conv + silu over the transposed stream, delta = gated + conv (fp32), written (row, c).
// One block = 64 channels (inside one stream) x 64 consecutive positions of one batch element.
extern "C" __global__ __launch_bounds__(256)
void ple_conv_out(const uint16_t* __restrict__ cstream, const uint16_t* __restrict__ convw, const uint16_t* __restrict__ value,
                  const float* __restrict__ gs, float* __restrict__ delta, int S)
{
    __shared__ uint16_t s_in[TT * PITCH_I];
    __shared__ uint16_t s_out[TT * PITCH_O];
    const int tid = threadIdx.x;
    const int ct = blockIdx.x, st = blockIdx.y, b = blockIdx.z;
    const int c0 = ct * TT, s0 = st * TT;
    const int h = c0 / DD, d0 = c0 % DD;
    // load the input window: channel cl, columns s0 .. s0 + 63 + ST (stream columns)
    for (int e = tid; e < TT * (TT + ST); e += 256) {
        const int cl = e / (TT + ST), j = e % (TT + ST);
        const int col = s0 + j;
        s_in[cl * PITCH_I + j] = col < ST + S ? cstream[IX_K4_IN(b, c0 + cl, col, S)] : (uint16_t) 0;
    }
    __syncthreads();
    // conv + silu: thread -> (channel cl, positions sl0, sl0+4, ...) with consecutive threads on consecutive positions
    {
        const int sl = tid & 63, cg = tid >> 6;
        for (int i = 0; i < TT / 4; i++) {
            const int cl = cg + 4 * i;
            const uint2 wu = *(const uint2*) (convw + (size_t) (c0 + cl) * 4);
            const float w0 = h2f(wu.x & 0xffff), w1 = h2f(wu.x >> 16), w2 = h2f(wu.y & 0xffff), w3 = h2f(wu.y >> 16);
            float acc = 0.f;
            acc = fmaf(w0, h2f(s_in[cl * PITCH_I + sl + 0 * DIL]), acc);
            acc = fmaf(w1, h2f(s_in[cl * PITCH_I + sl + 1 * DIL]), acc);
            acc = fmaf(w2, h2f(s_in[cl * PITCH_I + sl + 2 * DIL]), acc);
            acc = rnd32(fmaf(w3, h2f(s_in[cl * PITCH_I + sl + 3 * DIL]), acc));
            const float y = __half2float(__float2half_rn(acc));
            const float sv = rnd32(y / (1.0f + expf(-y)));
            s_out[sl * PITCH_O + cl] = __half_as_ushort(__float2half_rn(sv));
        }
    }
    __syncthreads();
    // write: thread -> channel cl, rows sl0 + 4 i
    {
        const int cl = tid & 63, sg = tid >> 6;
        for (int i = 0; i < TT / 4; i++) {
            const int sl = sg + 4 * i;
            const int s = s0 + sl;
            if (s >= S) continue;
            const size_t row = IX_K4_ROW(b, s, S);
            const float g = rnd32(gs[row * HS + h] * h2f(value[IX_K4_VAL(row, d0 + cl)]));
            delta[IX_K4_OUT(row, c0 + cl)] = g + h2f(s_out[sl * PITCH_O + cl]);
        }
    }
}

// ---- deinterleave_qg: [.., heads, (q: hd, g: hd)] -> contiguous q and g (pure 16-byte copy)
extern "C" __global__ __launch_bounds__(256)
void deinterleave_qg(const uint4* __restrict__ qg, uint4* __restrict__ q, uint4* __restrict__ g, int hd8, size_t n8)
{
    const size_t i = (size_t) blockIdx.x * 256 + threadIdx.x;
    if (i >= n8) return;
    const size_t d = i % hd8;
    const size_t src = IX_DQ_SRC(i, hd8, d);
    q[i] = qg[src];
    g[i] = qg[src + hd8];
}
