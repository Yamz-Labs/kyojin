// one-pass int8 GatedResidual mix for R = 2..4 rows (decode verify / MTP).
//
// gr_dots_q8_kernel / gr_finalize_q8_kernel use a (blocks, R) grid, so every int8 weight byte
// is fetched R times per call. These kernels give one block per fn row j (dots) and one block
// per column chunk (finalize) and loop over RT rows inside, so each weight word is loaded once
// and used for RT rows. Per (row, j) and per (row, h, column) the thread-to-element mapping,
// the fmaf order, the shuffle tree and the red[] order are the old ones, so outputs are
// bit-identical to the one-row-per-block kernels (checked by torch.equal in the parity test).
//
// Needs from the including file: NUM_THREADS (256), GR_THREADS_A (64), sigmoidf_, unpack_s8x4,
// half4, LOW_TO_FLOAT, HIGH_TO_FLOAT.
#pragma once

#define GRR_MAX_RT 4

// J fn rows per block (JT): the stream quads loaded for a column are reused by JT weight rows.
// Blocks 0 .. ceil(M / JT) - 1 do fn rows; the last block (index ceil(M / JT)) does the sum of squares
// (dots row M). Per (row, j) the c mapping, fmaf order and reduction tree are unchanged.
// Tuning knobs.
// Rows per block (JT) by group size RT = 1..4; JT * RT * H must stay <= GR_THREADS_A (64).
#ifndef GRR_J1
#define GRR_J1 2
#endif
#ifndef GRR_J2
#define GRR_J2 4
#endif
#ifndef GRR_J3
#define GRR_J3 4
#endif
#ifndef GRR_J4
#define GRR_J4 1
#endif
#ifndef GRR_R4_OLD_DOTS
#define GRR_R4_OLD_DOTS 1
#endif
#ifndef GRR_DOTS_UNROLL
#define GRR_DOTS_UNROLL 1
#endif
#ifndef GRR_FIN_UNROLL
#define GRR_FIN_UNROLL 1
#endif
template <int RT> struct GrrJ { static constexpr int v = RT == 1 ? GRR_J1 : RT == 2 ? GRR_J2 : RT == 3 ? GRR_J3 : GRR_J4; };

template <int H, int RT, int JT>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_q8r_kernel
(
    const float* __restrict__ streams,   // (RT, H, D)
    const int8_t* __restrict__ fn,       // (M, H * D) int8
    const float* __restrict__ fn_scale,  // (M)
    float* __restrict__ dots,            // (RT, M + 1, H)
    const int M,
    const int D,
    const int gs = 0
)
{
    static_assert(JT * RT * H <= GR_THREADS_A, "write-out phase uses one thread per (row, rowset, stream)");
    const int D4 = D / 4;
    const int ngrp = (M + JT - 1) / JT;
    const bool sq = blockIdx.x == ngrp;        // sum-of-squares block
    const int j0 = blockIdx.x * JT;

    __shared__ float red[JT][RT][H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a[JT][RT];
        #pragma unroll
        for (int u = 0; u < JT; ++u)
            #pragma unroll
            for (int q = 0; q < RT; ++q) a[u][q] = 0.0f;
        if (!sq)
        {
            const int2* f8[JT];
            #pragma unroll
            for (int u = 0; u < JT; ++u)
                f8[u] = (const int2*) (fn + ((size_t) min(j0 + u, M - 1) * H + h) * D);   // clamped: tail rows are discarded
            #pragma unroll GRR_DOTS_UNROLL
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float wv[JT][8];
                #pragma unroll
                for (int u = 0; u < JT; ++u)
                {
                    int2 pk = f8[u][c];
                    if (gs)
                    {
                        const float gsc = fn_scale[gs_fn_idx(min(j0 + u, M - 1), h, c, D)];
                        gs_unpack_s8x4((uint32_t) pk.x, gsc, wv[u][0], wv[u][1], wv[u][2], wv[u][3]);
                        gs_unpack_s8x4((uint32_t) pk.y, gsc, wv[u][4], wv[u][5], wv[u][6], wv[u][7]);
                    }
                    else
                    {
                        unpack_s8x4((uint32_t) pk.x, wv[u][0], wv[u][1], wv[u][2], wv[u][3]);
                        unpack_s8x4((uint32_t) pk.y, wv[u][4], wv[u][5], wv[u][6], wv[u][7]);
                    }
                }
                #pragma unroll
                for (int q = 0; q < RT; ++q)
                {
                    const float4* s4 = (const float4*) (streams + (size_t) q * H * D);
                    float4 s0 = s4[(size_t) h * D4 + 2 * c];
                    float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                    #pragma unroll
                    for (int u = 0; u < JT; ++u)
                    {
                        float x = a[u][q];
                        x = fmaf(s0.x, wv[u][0], x); x = fmaf(s0.y, wv[u][1], x);
                        x = fmaf(s0.z, wv[u][2], x); x = fmaf(s0.w, wv[u][3], x);
                        x = fmaf(s1.x, wv[u][4], x); x = fmaf(s1.y, wv[u][5], x);
                        x = fmaf(s1.z, wv[u][6], x); x = fmaf(s1.w, wv[u][7], x);
                        a[u][q] = x;
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int q = 0; q < RT; ++q)
            {
                const float4* s4 = (const float4*) (streams + (size_t) q * H * D);
                float x = 0.0f;
                for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
                {
                    float4 sv = s4[(size_t) h * D4 + c];
                    x = fmaf(sv.x, sv.x, fmaf(sv.y, sv.y, fmaf(sv.z, sv.z, fmaf(sv.w, sv.w, x))));
                }
                a[0][q] = x;
            }
        }
        #pragma unroll
        for (int u = 0; u < JT; ++u)
            #pragma unroll
            for (int q = 0; q < RT; ++q)
            {
                float x = a[u][q];
                for (int offset = 16; offset > 0; offset >>= 1)
                    x += __shfl_down_sync(0xffffffffu, x, offset);
                if (lane == 0) red[u][q][h][warp] = x;
            }
    }
    __syncthreads();
    if (threadIdx.x < JT * RT * H)
    {
        const int u = threadIdx.x / (RT * H);
        const int q = (threadIdx.x / H) % RT;
        const int h = threadIdx.x % H;
        const int j = sq ? M : j0 + u;
        if (sq ? u == 0 : j < M)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < GR_THREADS_A / 32; ++w)
                v += red[u][q][h][w];
            if (!sq && !gs) v *= fn_scale[j];
            dots[((size_t) q * (M + 1) + j) * H + h] = v;
        }
    }
}

template <int H, int RT, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_q8r_kernel
(
    const float* __restrict__ streams,   // (RT, H, D)
    const float* __restrict__ dots,      // (RT, M + 1, H)
    const int8_t* __restrict__ upt,      // (H, D / 4, LR, 4) int8
    const float* __restrict__ up_scale,  // (H * D)
    const half* __restrict__ w,          // (H * D)
    float* __restrict__ post,            // (RT, H) or nullptr
    void* __restrict__ mixed,            // (RT, D) half or float
    const int D,
    const int LR,
    const int chunk_cols,
    const float rms_eps,
    const int gs = 0
)
{
    const int M = LR + (post ? H : 0);

    __shared__ float rmr_s[RT][H];
    extern __shared__ float t_s[];       // (RT, LR)
    if (threadIdx.x < RT * H)
    {
        const int q = threadIdx.x / H;
        const int h = threadIdx.x % H;
        const float* dr = dots + (size_t) q * (M + 1) * H;
        rmr_s[q][h] = rsqrtf(dr[(size_t) M * H + h] / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    #pragma unroll
    for (int q = 0; q < RT; ++q)
    {
        const float* dr = dots + (size_t) q * (M + 1) * H;
        for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
        {
            float v = 0.0f;
            #pragma unroll
            for (int h = 0; h < H; ++h)
                v = fmaf(rmr_s[q][h], dr[(size_t) i * H + h], v);
            v *= inv_h;
            t_s[q * LR + i] = v * sigmoidf_(v);
        }
        if (post && blockIdx.x == 0 && threadIdx.x < H)
        {
            float v = 0.0f;
            #pragma unroll
            for (int h = 0; h < H; ++h)
                v = fmaf(rmr_s[q][h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
            post[(size_t) q * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
        }
    }
    __syncthreads();

    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[RT][H];
        #pragma unroll
        for (int q = 0; q < RT; ++q)
            #pragma unroll
            for (int h = 0; h < H; ++h) g[q][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 gsc[H];
        #pragma unroll GRR_FIN_UNROLL
        for (int i = lane; i < LR; i += 32)
        {
            float ti[RT];
            #pragma unroll
            for (int q = 0; q < RT; ++q) ti[q] = t_s[q * LR + i];
            if (gs && (((i - lane) & 32) == 0))     // the 32-rank step that opens a 64-rank group (warp-uniform, lane < 32)
            {
                #pragma unroll
                for (int h = 0; h < H; ++h) gsc[h] = *(const float4*) (up_scale + gs_up_idx(h, c, i, D, LR));
            }
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                uint32_t u = *(const uint32_t*) (upt + ((((size_t) h * D4 + c) * LR + i) * 4));
                float u0, u1, u2, u3;
                unpack_s8x4(u, u0, u1, u2, u3);
                if (gs) { u0 = gs_dq(u0, gsc[h].x); u1 = gs_dq(u1, gsc[h].y); u2 = gs_dq(u2, gsc[h].z); u3 = gs_dq(u3, gsc[h].w); }
                #pragma unroll
                for (int q = 0; q < RT; ++q)
                {
                    g[q][h].x = fmaf(ti[q], u0, g[q][h].x);
                    g[q][h].y = fmaf(ti[q], u1, g[q][h].y);
                    g[q][h].z = fmaf(ti[q], u2, g[q][h].z);
                    g[q][h].w = fmaf(ti[q], u3, g[q][h].w);
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < RT; ++q)
            #pragma unroll
            for (int h = 0; h < H; ++h)
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    g[q][h].x += __shfl_xor_sync(0xffffffffu, g[q][h].x, offset);
                    g[q][h].y += __shfl_xor_sync(0xffffffffu, g[q][h].y, offset);
                    g[q][h].z += __shfl_xor_sync(0xffffffffu, g[q][h].z, offset);
                    g[q][h].w += __shfl_xor_sync(0xffffffffu, g[q][h].w, offset);
                }
        if (lane != 0) continue;
        #pragma unroll
        for (int q = 0; q < RT; ++q)
        {
            const float4* s4 = (const float4*) (streams + (size_t) q * H * D);
            float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                const float4 sc = gs ? make_float4(1.0f, 1.0f, 1.0f, 1.0f) : *(const float4*) (up_scale + (size_t) h * D + 4 * c);
                float4 sv = s4[(size_t) h * D4 + c];
                half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
                float coef = rmr_s[q][h] * inv_h;
                o.x = fmaf(sigmoidf_(g[q][h].x * sc.x) * coef * LOW_TO_FLOAT(wq.x),  sv.x, o.x);
                o.y = fmaf(sigmoidf_(g[q][h].y * sc.y) * coef * HIGH_TO_FLOAT(wq.x), sv.y, o.y);
                o.z = fmaf(sigmoidf_(g[q][h].z * sc.z) * coef * LOW_TO_FLOAT(wq.y),  sv.z, o.z);
                o.w = fmaf(sigmoidf_(g[q][h].w * sc.w) * coef * HIGH_TO_FLOAT(wq.y), sv.w, o.w);
            }
            if (HALF_OUT)
            {
                half2* out2 = (half2*) ((half*) mixed + (size_t) q * D);
                out2[c * 2] = __floats2half2_rn(o.x, o.y);
                out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
            }
            else
                ((float4*) ((float*) mixed + (size_t) q * D))[c] = o;
        }
    }
}

// Host side launch geometry, shared by the extension wrapper and the standalone microbench.
struct GrrGeom
{
    int n_chunks;
    int chunk_cols;
    int smem;           // bytes of dynamic shared memory for finalize
};

static inline GrrGeom grr_geom(int D, int LR, int RT, int max_chunks)
{
    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, max_chunks));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    GrrGeom g;
    g.chunk_cols = chunk_cols;
    g.n_chunks = (D + chunk_cols - 1) / chunk_cols;
    g.smem = RT * LR * (int) sizeof(float);
    return g;
}

template <int RT>
static void grr_dots_group(const float* streams, const int8_t* fn, const float* fn_scale, float* dots, int M, int D, cudaStream_t stream, int gs)
{
    constexpr int JT = GrrJ<RT>::v;
    gr_dots_q8r_kernel<4, RT, JT><<<dim3((M + JT - 1) / JT + 1), GR_THREADS_A, 0, stream>>>(streams, fn, fn_scale, dots, M, D, gs);
}

// Launch one group of RT rows (pointers already offset to the group's first row).
template <int RT>
static void grr_launch_group
(
    const float* streams, const int8_t* fn, const float* fn_scale, const int8_t* upt,
    const float* up_scale, const half* w, float* post, void* mixed, float* dots,
    int M, int D, int LR, float rms_eps, bool half_out, int max_chunks, cudaStream_t stream, int gs
)
{
    // RT = 4: a (M + 1, 4) grid (4x the blocks of the one-pass dots) measured faster on the weights-in-MALL
    // regime than JT-row blocks (latency bound), so the group of 4 keeps the one-row-per-block dots kernel
    // and only the finalize is one-pass. Same numerics either way (bit-identical).
    if (RT == 4 && GRR_R4_OLD_DOTS)
        gr_dots_q8_kernel<4><<<dim3(M + 1, RT), GR_THREADS_A, 0, stream>>>(streams, fn, fn_scale, dots, M, D, gs);
    else
        grr_dots_group<RT>(streams, fn, fn_scale, dots, M, D, stream, gs);
    GrrGeom g = grr_geom(D, LR, RT, max_chunks);
    if (half_out)
        gr_finalize_q8r_kernel<4, RT, true><<<dim3(g.n_chunks), NUM_THREADS, g.smem, stream>>>
            (streams, dots, upt, up_scale, w, post, mixed, D, LR, g.chunk_cols, rms_eps, gs);
    else
        gr_finalize_q8r_kernel<4, RT, false><<<dim3(g.n_chunks), NUM_THREADS, g.smem, stream>>>
            (streams, dots, upt, up_scale, w, post, mixed, D, LR, g.chunk_cols, rms_eps, gs);
}

// All R rows: groups of GRR_MAX_RT rows, one weight pass per group.
static void grr_launch
(
    const float* streams, const int8_t* fn, const float* fn_scale, const int8_t* upt,
    const float* up_scale, const half* w, float* post, void* mixed, float* dots,
    int R, int M, int D, int LR, float rms_eps, bool half_out, int max_chunks, cudaStream_t stream, int gs = 0
)
{
    const size_t out_el = half_out ? sizeof(half) : sizeof(float);
    for (int r0 = 0; r0 < R; r0 += GRR_MAX_RT)
    {
        const int n = std::min(GRR_MAX_RT, R - r0);
        const float* s = streams + (size_t) r0 * 4 * D;
        float* d = dots + (size_t) r0 * (M + 1) * 4;
        float* p = post ? post + (size_t) r0 * 4 : nullptr;
        void* m = (char*) mixed + (size_t) r0 * D * out_el;
        #define GRR_CALL(N) grr_launch_group<N>(s, fn, fn_scale, upt, up_scale, w, p, m, d, M, D, LR, rms_eps, half_out, max_chunks, stream, gs)
        if (n == 1) GRR_CALL(1);
        else if (n == 2) GRR_CALL(2);
        else if (n == 3) GRR_CALL(3);
        else GRR_CALL(4);
        #undef GRR_CALL
    }
}
