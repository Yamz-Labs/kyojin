// Quantized-KV flash-decoding split kernel for MANY query rows (verify steps: q_len x GQA group up to ~56 rows per kv head).
// Same launch contract / partial layout as qdec.cu (the Triton combine kernel reduces the partials), but the CTA's NW warps
// cooperate on ONE 16-token slab instead of streaming independent slabs, splitting the head dimension between them
// (NW = hd/32: one 32-dim quantization group per warp), so the O accumulators (hd x rows fp32) fit the registers however
// many rows there are and no K/V dequantization is repeated:
//   * S^T = Kdeq x q^T : each warp dequantizes only ITS group of K and MMAs a partial S over it; the NW partial S tiles
//     (16 tokens x 8 rows per row tile, fp32) go through smem (barrier A);
//   * softmax: row tile j is owned by warp j % NW, which sums the partials of its tile, runs the online softmax for it and
//     publishes P (as the MMA B fragment) and the O rescale factor in smem (barrier B);
//   * O^T = Vdeq^T x P^T : each warp dequantizes only ITS slice of V and owns that slice of the accumulators of all rows,
//     so there is no cross-warp merge at the end.
// Per-warp cp.async ring of ST slabs of the warp's own slice of the packed K/V words.
#define G (HD / 32)
#define GW (G / NW)                     // 32-dim quantization groups per warp
#define GROUP (NQH / NKV)
#define ROWS (QLEN * GROUP)
#define NT ((ROWS + 7) / 8)
#define GPT (NKV * HD / 32)
#define KCHW (G * KB)
#define VCHW (G * VB)
#define KWW (GW * KB)                   // words per token of one warp's K slice
#define VWW (GW * VB)
#define KSW (KWW + ((KWW % 8 == 4) ? 0 : 4))
#define VSW (VWW + ((VWW % 8 == 4) ? 0 : 4))
#define KCP ((KWW % 4 == 0) ? 4 : 1)    // words per cp.async
#define VCP ((VWW % 4 == 0) ? 4 : 1)
#define KDATA_B (16 * KSW * 4)
#define VDATA_B (16 * VSW * 4)
#define SC_B (((16 * G * 2) + 15) / 16 * 16)
#define STAGE_B (KDATA_B + VDATA_B + 2 * SC_B)
#define NTHREADS (NW * 32)
#define LOG2E 1.4426950408889634f
#define NSLW (2 * GW)                   // 16-dim output slabs per warp
#define OWN ((NT + NW - 1) / NW)        // row tiles owned per warp
#define SX_B (NW * NT * 32 * 16)       // partial S exchange
#define PX_B (NT * 32 * 16)             // P fragments (2 words) + rescale factors (2 floats) per row tile and lane

// dynamic smem layout: [Sx][Px][ring: NW x ST x STAGE_B]   (q lives in registers as this warp's MMA B fragments)

__device__ __forceinline__ void cp_sc(void* dst, const void* src, bool pred)
{
    int sz = pred ? (G * 2) : 0;
    if (G * 2 == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(sz));
    else             asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(sz));
}
template<int CP> __device__ __forceinline__ void cpw(void* dst, const void* src, bool pred)
{
    if (CP == 4) cp16(dst, src, pred); else cp4(dst, src, pred);
}

#ifndef ABL
#define ABL 0
#endif
extern "C" __global__ void __launch_bounds__(NTHREADS)
qdec_split_rows(
    const u16* __restrict__ q, const u32* __restrict__ k_cache, const u32* __restrict__ v_cache,
    const int* __restrict__ block_table, const int* __restrict__ cache_seqlens, u16* out_unused,
    float* __restrict__ partial_o, float* __restrict__ partial_ml,
    const u16* __restrict__ k_scales, const u16* __restrict__ v_scales, const u16* __restrict__ h32,
    int split_len, int num_pages_per_seq, int num_splits, const float* sinks_unused,
    void* s0, void* s1)
{
    extern __shared__ __align__(16) unsigned char smem[];
    float4* Sx = (float4*)smem;
    uint2* Pp = (uint2*)(smem + SX_B);
    float2* Pa = (float2*)(smem + SX_B + NT * 32 * 8);
    unsigned char* ring = smem + SX_B + PX_B;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int pid = blockIdx.x, split = blockIdx.y;
    if (split >= num_splits) return;                  // uniform across the CTA
    const int batch = pid / NKV, kvh = pid % NKV;

    const int total_k = cache_seqlens[batch] + QLEN;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k);

    // ---- q rows -> rotated (h32) fp16, kept as this warp's B fragments (head-dim groups warp*GW ..)
    u32 qb[2 * GW][NT][2];
    {
        float* hs = (float*)(ring);                      // 32x32 fp32 scratch, ring is idle until the loop
        for (int i = tid; i < 1024; i += NTHREADS) hs[i] = h2f(h32[i]);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 2 * GW; kk++) {
            const int ks = warp * 2 * GW + kk, gi = ks >> 1, cb = 16 * (ks & 1) + 2 * t;
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                const int r = 8 * j + g;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                if (r < ROWS) {
                    const uint4* qv = (const uint4*)(q + ((size_t)((batch * QLEN + r % QLEN) * NQH + kvh * GROUP + r / QLEN)) * HD + gi * 32);
                    #pragma unroll 1
                    for (int v = 0; v < 4; v++) {
                        uint4 w4 = qv[v];
                        u32 w[4] = {w4.x, w4.y, w4.z, w4.w};
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            float x = h2f((u16)((w[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu));
                            const float* hr = hs + (v * 8 + k) * 32 + cb;
                            a0 += x * hr[0]; a1 += x * hr[1]; a2 += x * hr[8]; a3 += x * hr[9];
                        }
                    }
                }
                qb[kk][j][0] = (pack_h2(a0, 0.f) & 0xFFFFu) | (pack_h2(a1, 0.f) << 16);
                qb[kk][j][1] = (pack_h2(a2, 0.f) & 0xFFFFu) | (pack_h2(a3, 0.f) << 16);
            }
        }
        __syncthreads();
    }

    float m[OWN][2], l[OWN][2], qabs[OWN][2];
    bool rvalid[OWN][2];
    #pragma unroll
    for (int o = 0; o < OWN; o++)
        #pragma unroll
        for (int e = 0; e < 2; e++) {
            int r = 8 * (warp + o * NW) + 2 * t + e;
            m[o][e] = -INFINITY; l[o][e] = 0.f;
            rvalid[o][e] = r < ROWS;
            qabs[o][e] = (float)(total_k - QLEN + (r % QLEN));
        }
    float O[NSLW][NT][4];
    #pragma unroll
    for (int i = 0; i < NSLW; i++)
        #pragma unroll
        for (int j = 0; j < NT; j++) { O[i][j][0] = O[i][j][1] = O[i][j][2] = O[i][j][3] = 0.f; }

    unsigned char* wring = ring + (size_t)warp * ST * STAGE_B;
    const int nslabs = (n_end > n_start) ? (n_end - n_start + 15) / 16 : 0;

    auto fill = [&](int it) {
        unsigned char* st = wring + (it % ST) * STAGE_B;
        u32* ksm = (u32*)st; u32* vsm = (u32*)(st + KDATA_B);
        u16* ksc = (u16*)(st + KDATA_B + VDATA_B); u16* vsc = (u16*)(st + KDATA_B + VDATA_B + SC_B);
        if (it < nslabs) {
            int n0 = n_start + it * 16;
            int page = n0 / PAGE;
            long long phys = block_table[batch * num_pages_per_seq + page];
            long long row0 = phys * PAGE + (n0 - page * PAGE);
            #pragma unroll
            for (int c = lane; c < 16 * (KWW / KCP); c += 32) {
                int tk = c / (KWW / KCP), pt = c % (KWW / KCP);
                cpw<KCP>(ksm + tk * KSW + pt * KCP, k_cache + (row0 + tk) * (GPT * KB) + kvh * KCHW + warp * KWW + pt * KCP, n0 + tk < n_end);
            }
            #pragma unroll
            for (int c = lane; c < 16 * (VWW / VCP); c += 32) {
                int tk = c / (VWW / VCP), pt = c % (VWW / VCP);
                cpw<VCP>(vsm + tk * VSW + pt * VCP, v_cache + (row0 + tk) * (GPT * VB) + kvh * VCHW + warp * VWW + pt * VCP, n0 + tk < n_end);
            }
            if (lane < 16) cp_sc(ksc + lane * G, k_scales + (row0 + lane) * GPT + kvh * G, n0 + lane < n_end);
            else           cp_sc(vsc + (lane - 16) * G, v_scales + (row0 + lane - 16) * GPT + kvh * G, n0 + lane - 16 < n_end);
        }
        asm volatile("cp.async.commit_group;");
    };

    #pragma unroll
    for (int s = 0; s < ST - 1; s++) fill(s);

    const float sl = SCALE * LOG2E;
    for (int it = 0; it < nslabs; it++) {
        __syncwarp();
        fill(it + ST - 1);
        asm volatile("cp.async.wait_group %0;" :: "n"(ST - 1));
        __syncwarp();
        unsigned char* st = wring + (it % ST) * STAGE_B;
        const u32* ksm = (const u32*)st; const u32* vsm = (const u32*)(st + KDATA_B);
        const u16* ksc = (const u16*)(st + KDATA_B + VDATA_B); const u16* vsc = (const u16*)(st + KDATA_B + VDATA_B + SC_B);
        const int n0 = n_start + it * 16;

        // ---- partial S^T over this warp's head-dim groups
        float S[NT][4];
        #pragma unroll
        for (int j = 0; j < NT; j++) { S[j][0] = S[j][1] = S[j][2] = S[j][3] = 0.f; }
        const u32* kr0 = ksm + g * KSW;
        const u32* kr1 = ksm + (g + 8) * KSW;
        #pragma unroll
        for (int gi = 0; gi < GW; gi++) {
            const int gg = warp * GW + gi;
            u32 sc0 = scale2<KB>(ksc[g * G + gg], ksc[g * G + gg]);
            u32 sc1 = scale2<KB>(ksc[(g + 8) * G + gg], ksc[(g + 8) * G + gg]);
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int p0 = 16 * h + 2 * t, p1 = p0 + 8;
                u32 a0 = dequant2<KB>(raw_pair<KB>(kr0 + gi * KB, p0), sc0);
                u32 a1 = dequant2<KB>(raw_pair<KB>(kr1 + gi * KB, p0), sc1);
                u32 a2 = dequant2<KB>(raw_pair<KB>(kr0 + gi * KB, p1), sc0);
                u32 a3 = dequant2<KB>(raw_pair<KB>(kr1 + gi * KB, p1), sc1);
                #pragma unroll
                for (int j = 0; j < NT; j++) {
                    mma16816(S[j], a0, a1, a2, a3, qb[gi * 2 + h][j][0], qb[gi * 2 + h][j][1]);
                }
            }
        }

        // ---- partial S tiles -> smem (barrier A) ; the owner of a row tile sums them and does its online softmax
        float4* sxw = Sx;
        #pragma unroll
        for (int j = 0; j < NT; j++) sxw[(warp * NT + j) * 32 + lane] = make_float4(S[j][0], S[j][1], S[j][2], S[j][3]);
        if (ABL != 1) __syncthreads();
        #pragma unroll
        for (int o = 0; o < OWN; o++) {
            const int j = warp + o * NW;
            if (j < NT && ABL != 1) {
                float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
                #pragma unroll
                for (int w = 0; w < NW; w++) {
                    float4 v = sxw[(w * NT + j) * 32 + lane];
                    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
                }
                float sv[4] = {acc.x, acc.y, acc.z, acc.w};
                float s[4]; bool ok[4];
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    int tk = n0 + g + ((e >> 1) << 3);
                    int ri = e & 1;
                    ok[e] = rvalid[o][ri] && tk < n_end && (!CAUSAL || (float)tk <= qabs[o][ri]);
                    s[e] = ok[e] ? sv[e] * sl : -INFINITY;
                }
                float alpha[2];
                #pragma unroll
                for (int ri = 0; ri < 2; ri++) {
                    float mx = fmaxf(s[ri], s[2 + ri]);
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
                    float mn = fmaxf(m[o][ri], mx);
                    float me = (mn == -INFINITY) ? 0.f : mn;
                    alpha[ri] = (m[o][ri] == -INFINITY) ? 0.f : exp2f(m[o][ri] - me);
                    float p0 = ok[ri] ? exp2f(s[ri] - me) : 0.f;
                    float p1 = ok[2 + ri] ? exp2f(s[2 + ri] - me) : 0.f;
                    s[ri] = p0; s[2 + ri] = p1;
                    l[o][ri] = l[o][ri] * alpha[ri] + p0 + p1;
                    m[o][ri] = mn;
                }
                Pp[j * 32 + lane] = make_uint2(movtrans(pack_h2(s[0], s[1])), movtrans(pack_h2(s[2], s[3])));
                Pa[j * 32 + lane] = make_float2(alpha[0], alpha[1]);
            }
        }
        if (ABL != 1) __syncthreads();
        u32 plo[NT], phi[NT];
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            uint2 pp = Pp[j * 32 + lane]; float2 al = Pa[j * 32 + lane];
            plo[j] = pp.x; phi[j] = pp.y;
            #pragma unroll
            for (int i = 0; i < NSLW; i++) {
                O[i][j][0] *= al.x; O[i][j][1] *= al.y; O[i][j][2] *= al.x; O[i][j][3] *= al.y;
            }
        }

        // ---- O^T slice += Vdeq^T x P^T ; tokens 2t, 2t+1, 2t+8, 2t+9 of the slab
        const u32* vA = vsm + (2 * t) * VSW;
        const u32* vB = vsm + (2 * t + 1) * VSW;
        const u32* vC = vsm + (2 * t + 8) * VSW;
        const u32* vD = vsm + (2 * t + 9) * VSW;
        #pragma unroll
        for (int gi = 0; gi < GW; gi++) {
            const int gg = warp * GW + gi;
            u32 sAB = scale2<VB>(vsc[(2 * t) * G + gg], vsc[(2 * t + 1) * G + gg]);
            u32 sCD = scale2<VB>(vsc[(2 * t + 8) * G + gg], vsc[(2 * t + 9) * G + gg]);
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int p1 = 16 * h + g, p2 = p1 + 8, i = gi * 2 + h;
                u32 a0 = dequant2<VB>(raw_one<VB>(vA + gi * VB, p1) | (raw_one<VB>(vB + gi * VB, p1) << 16), sAB);
                u32 a1 = dequant2<VB>(raw_one<VB>(vA + gi * VB, p2) | (raw_one<VB>(vB + gi * VB, p2) << 16), sAB);
                u32 a2 = dequant2<VB>(raw_one<VB>(vC + gi * VB, p1) | (raw_one<VB>(vD + gi * VB, p1) << 16), sCD);
                u32 a3 = dequant2<VB>(raw_one<VB>(vC + gi * VB, p2) | (raw_one<VB>(vD + gi * VB, p2) << 16), sCD);
                #pragma unroll
                for (int j = 0; j < NT; j++) mma16816(O[i][j], a0, a1, a2, a3, plo[j], phi[j]);
            }
        }
    }

    // ---- row sums across the 8 token lanes (owned tiles)
    #pragma unroll
    for (int o = 0; o < OWN; o++)
        #pragma unroll
        for (int ri = 0; ri < 2; ri++) {
            float v = l[o][ri];
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            l[o][ri] = v;
        }
    asm volatile("cp.async.wait_all;");

    // ---- partials: every warp writes its own head-dim slice of all rows; m / l come from warp 0 (identical in all warps)
    const size_t pbase = (size_t)(pid * num_splits + split);
    float* po = partial_o + pbase * BLOCK_ROWS * HD_PAD;
    float* pml = partial_ml + pbase * BLOCK_ROWS * 2;
    #pragma unroll
    for (int j = 0; j < NT; j++) {
        #pragma unroll
        for (int i = 0; i < NSLW; i++) {
            const int d = 16 * (warp * NSLW + i) + g;
            float* ob = po + (size_t)(8 * j + 2 * t) * HD_PAD + d;
            ob[0] = O[i][j][0];
            ob[HD_PAD] = O[i][j][1];
            ob[8] = O[i][j][2];
            ob[HD_PAD + 8] = O[i][j][3];
        }
    }
    for (int idx = tid; idx < (BLOCK_ROWS - NT * 8) * HD_PAD; idx += NTHREADS) po[(size_t)NT * 8 * HD_PAD + idx] = 0.f;
    if (g == 0) {
        #pragma unroll
        for (int o = 0; o < OWN; o++) {
            const int j = warp + o * NW;
            if (j < NT) {
                #pragma unroll
                for (int ri = 0; ri < 2; ri++) {
                    int r = 8 * j + 2 * t + ri;
                    pml[r * 2] = (m[o][ri] == -INFINITY) ? m[o][ri] : m[o][ri] * 0.6931471805599453f;   // natural-log units for the combine kernel
                    pml[r * 2 + 1] = l[o][ri];
                }
            }
        }
    }
    if (warp == 0)
        for (int r = NT * 8 + lane; r < BLOCK_ROWS; r += 32) { pml[r * 2] = -INFINITY; pml[r * 2 + 1] = 0.f; }
}
