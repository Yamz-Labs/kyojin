// Quantized-KV flash-decoding split kernel (sm_80+): tensor-core QK^T / PV straight from packed bit-plane words.
// Same launch contract as triton_paged_cu._paged_attn_decode_split_kernel for QCK>0 and QCV>0 (partials feed the
// existing combine kernel). Compile-time config through -D macros, see qdec_cuda.py.
//
// Per warp: slabs of 16 tokens. S^T (16 tokens x 8 rows) = Kdeq (A, from smem words, dequantized into the mma
// fragment layout) x q^T (B). P^T is transposed in registers (movmatrix) into the B fragment of O^T = V^T x P^T,
// whose A fragment is dequantized from V words the same way. Warps are independent (no block barriers in the loop);
// data arrives through a per-warp cp.async ring of ST slabs.
#define G (HD / 32)
#define GROUP (NQH / NKV)
#define ROWS (QLEN * GROUP)
#define NT ((ROWS + 7) / 8)
#define GPT (NKV * HD / 32)
#define KCHW (G * KB)
#define VCHW (G * VB)
#define KSTRW (KCHW + ((KCHW % 8 == 4) ? 0 : 4))
#define VSTRW (VCHW + ((VCHW % 8 == 4) ? 0 : 4))
#define KDATA_B (16 * KSTRW * 4)
#define VDATA_B (16 * VSTRW * 4)
#define SC_B (((16 * G * 2) + 15) / 16 * 16)
#define STAGE_B (KDATA_B + VDATA_B + 2 * SC_B)
#define QSTR (HD + 8)
#define NTHREADS (NW * 32)
#define LOG2E 1.4426950408889634f
#define NSL (HD / 16)

// dynamic smem layout: [Qs fp16 NT*8 x QSTR][ring: NW x ST x STAGE_B]   (epilogue reuses the ring for the cross-warp merge)

__device__ __forceinline__ void cp_sc(void* dst, const void* src, bool pred)
{
    int sz = pred ? (G * 2) : 0;
    if (G * 2 == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(sz));
    else             asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(sz));
}
extern "C" __global__ void __launch_bounds__(NTHREADS)
qdec_split(
    const u16* __restrict__ q, const u32* __restrict__ k_cache, const u32* __restrict__ v_cache,
    const int* __restrict__ block_table, const int* __restrict__ cache_seqlens, u16* out_unused,
    float* __restrict__ partial_o, float* __restrict__ partial_ml,
    const u16* __restrict__ k_scales, const u16* __restrict__ v_scales, const u16* __restrict__ h32,
    int split_len, int num_pages_per_seq, int num_splits, const float* sinks_unused,
    void* s0, void* s1)
{
    extern __shared__ __align__(16) unsigned char smem[];
    u16* Qs = (u16*)smem;
    unsigned char* ring = smem + ((NT * 8 * QSTR * 2 + 15) / 16) * 16;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int pid = blockIdx.x, split = blockIdx.y;
    if (split >= num_splits) return;
    const int batch = pid / NKV, kvh = pid % NKV;

    const int total_k = cache_seqlens[batch] + QLEN;   // kv_append_len == q_len (cache already holds the new tokens)
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k);

    // ---- q rows -> rotated (h32) fp16 in smem
    {
        float* hs = (float*)(ring);                      // 32x32 fp32 scratch, ring is idle until the loop
        for (int i = tid; i < 1024; i += NTHREADS) hs[i] = h2f(h32[i]);
        __syncthreads();
        for (int idx = tid; idx < NT * 8 * HD; idx += NTHREADS) {
            int r = idx / HD, d = idx % HD, gi = d >> 5, c = d & 31;
            float acc = 0.f;
            if (r < ROWS) {
                int row_q = r % QLEN, hl = r / QLEN;
                const u16* qp = q + ((size_t)((batch * QLEN + row_q) * NQH + kvh * GROUP + hl)) * HD + gi * 32;
                #pragma unroll
                for (int k = 0; k < 32; k++) acc += h2f(qp[k]) * hs[k * 32 + c];
            }
            u32 hh = pack_h2(acc, 0.f) & 0xFFFFu;
            Qs[r * QSTR + d] = (u16)hh;
        }
        __syncthreads();
    }

    float m[NT][2], l[NT][2], qabs[NT][2];
    bool rvalid[NT][2];
    #pragma unroll
    for (int j = 0; j < NT; j++)
        #pragma unroll
        for (int e = 0; e < 2; e++) {
            int r = 8 * j + 2 * t + e;
            m[j][e] = -INFINITY; l[j][e] = 0.f;
            rvalid[j][e] = r < ROWS;
            qabs[j][e] = (float)(total_k - QLEN + (r % QLEN));
        }
    float O[NSL][NT][4];
    #pragma unroll
    for (int i = 0; i < NSL; i++)
        #pragma unroll
        for (int j = 0; j < NT; j++) { O[i][j][0] = O[i][j][1] = O[i][j][2] = O[i][j][3] = 0.f; }

    unsigned char* wring = ring + (size_t)warp * ST * STAGE_B;
    const int nslabs = (n_end > n_start) ? (n_end - n_start + 15) / 16 : 0;
    const int my_n = (nslabs > warp) ? (nslabs - warp + NW - 1) / NW : 0;

    auto fill = [&](int it) {
        unsigned char* st = wring + (it % ST) * STAGE_B;
        u32* ksm = (u32*)st; u32* vsm = (u32*)(st + KDATA_B);
        u16* ksc = (u16*)(st + KDATA_B + VDATA_B); u16* vsc = (u16*)(st + KDATA_B + VDATA_B + SC_B);
        if (it < my_n) {
            int n0 = n_start + (warp + it * NW) * 16;
            int page = n0 / PAGE;
            long long phys = block_table[batch * num_pages_per_seq + page];
            long long row0 = phys * PAGE + (n0 - page * PAGE);
            #pragma unroll
            for (int c = lane; c < 16 * (KCHW / 4); c += 32) {
                int tk = c / (KCHW / 4), pt = c % (KCHW / 4);
                cp16(ksm + tk * KSTRW + pt * 4, k_cache + (row0 + tk) * (GPT * KB) + kvh * KCHW + pt * 4, n0 + tk < n_end);
            }
            #pragma unroll
            for (int c = lane; c < 16 * (VCHW / 4); c += 32) {
                int tk = c / (VCHW / 4), pt = c % (VCHW / 4);
                cp16(vsm + tk * VSTRW + pt * 4, v_cache + (row0 + tk) * (GPT * VB) + kvh * VCHW + pt * 4, n0 + tk < n_end);
            }
            if (lane < 16) cp_sc(ksc + lane * G, k_scales + (row0 + lane) * GPT + kvh * G, n0 + lane < n_end);
            else           cp_sc(vsc + (lane - 16) * G, v_scales + (row0 + lane - 16) * GPT + kvh * G, n0 + lane - 16 < n_end);
        }
        asm volatile("cp.async.commit_group;");
    };

    #pragma unroll
    for (int s = 0; s < ST - 1; s++) fill(s);

    const float sl = SCALE * LOG2E;
    for (int it = 0; it < my_n; it++) {
        __syncwarp();
        fill(it + ST - 1);
        asm volatile("cp.async.wait_group %0;" :: "n"(ST - 1));
        __syncwarp();
        unsigned char* st = wring + (it % ST) * STAGE_B;
        const u32* ksm = (const u32*)st; const u32* vsm = (const u32*)(st + KDATA_B);
        const u16* ksc = (const u16*)(st + KDATA_B + VDATA_B); const u16* vsc = (const u16*)(st + KDATA_B + VDATA_B + SC_B);
        const int n0 = n_start + (warp + it * NW) * 16;

        // ---- S^T = Kdeq x q^T
        float S[NT][4];
        #pragma unroll
        for (int j = 0; j < NT; j++) { S[j][0] = S[j][1] = S[j][2] = S[j][3] = 0.f; }
        const u32* kr0 = ksm + g * KSTRW;
        const u32* kr1 = ksm + (g + 8) * KSTRW;
        #pragma unroll
        for (int gi = 0; gi < G; gi++) {
            u32 sc0 = scale2<KB>(ksc[g * G + gi], ksc[g * G + gi]);
            u32 sc1 = scale2<KB>(ksc[(g + 8) * G + gi], ksc[(g + 8) * G + gi]);
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int p0 = 16 * h + 2 * t, p1 = p0 + 8, ks = gi * 2 + h;
                u32 a0 = dequant2<KB>(raw_pair<KB>(kr0 + gi * KB, p0), sc0);
                u32 a1 = dequant2<KB>(raw_pair<KB>(kr1 + gi * KB, p0), sc1);
                u32 a2 = dequant2<KB>(raw_pair<KB>(kr0 + gi * KB, p1), sc0);
                u32 a3 = dequant2<KB>(raw_pair<KB>(kr1 + gi * KB, p1), sc1);
                #pragma unroll
                for (int j = 0; j < NT; j++) {
                    const u16* qp = Qs + (8 * j + g) * QSTR + 16 * ks + 2 * t;
                    u32 b0 = *(const u32*)qp, b1 = *(const u32*)(qp + 8);
                    mma16816(S[j], a0, a1, a2, a3, b0, b1);
                }
            }
        }

        // ---- online softmax (log2 domain), rows 2t / 2t+1 of every n-tile
        u32 plo[NT], phi[NT];
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            float s[4]; bool ok[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int tk = n0 + g + ((e >> 1) << 3);
                int ri = e & 1;
                ok[e] = rvalid[j][ri] && tk < n_end && (!CAUSAL || (float)tk <= qabs[j][ri]);
                s[e] = ok[e] ? S[j][e] * sl : -INFINITY;
            }
            float alpha[2];
            #pragma unroll
            for (int ri = 0; ri < 2; ri++) {
                float mx = fmaxf(s[ri], s[2 + ri]);
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
                float mn = fmaxf(m[j][ri], mx);
                float me = (mn == -INFINITY) ? 0.f : mn;
                alpha[ri] = (m[j][ri] == -INFINITY) ? 0.f : exp2f(m[j][ri] - me);
                float p0 = ok[ri] ? exp2f(s[ri] - me) : 0.f;
                float p1 = ok[2 + ri] ? exp2f(s[2 + ri] - me) : 0.f;
                s[ri] = p0; s[2 + ri] = p1;
                l[j][ri] = l[j][ri] * alpha[ri] + p0 + p1;
                m[j][ri] = mn;
            }
            #pragma unroll
            for (int i = 0; i < NSL; i++) {
                O[i][j][0] *= alpha[0]; O[i][j][1] *= alpha[1]; O[i][j][2] *= alpha[0]; O[i][j][3] *= alpha[1];
            }
            plo[j] = movtrans(pack_h2(s[0], s[1]));
            phi[j] = movtrans(pack_h2(s[2], s[3]));
        }

        // ---- O^T += Vdeq^T x P^T ; tokens 2t, 2t+1, 2t+8, 2t+9 of the slab
        const u32* vA = vsm + (2 * t) * VSTRW;
        const u32* vB = vsm + (2 * t + 1) * VSTRW;
        const u32* vC = vsm + (2 * t + 8) * VSTRW;
        const u32* vD = vsm + (2 * t + 9) * VSTRW;
        #pragma unroll
        for (int gi = 0; gi < G; gi++) {
            u32 sAB = scale2<VB>(vsc[(2 * t) * G + gi], vsc[(2 * t + 1) * G + gi]);
            u32 sCD = scale2<VB>(vsc[(2 * t + 8) * G + gi], vsc[(2 * t + 9) * G + gi]);
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

    // ---- row sums across the 8 token lanes
    #pragma unroll
    for (int j = 0; j < NT; j++)
        #pragma unroll
        for (int ri = 0; ri < 2; ri++) {
            float v = l[j][ri];
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            l[j][ri] = v;
        }

    // ---- merge warps through smem (ring is free: drain async copies first)
    asm volatile("cp.async.wait_all;");
    __syncthreads();
    float* Osm = (float*)ring;                          // [NW][NT*8][HD]
    float* Msm = Osm + NW * NT * 8 * HD;                // [NW][NT*8]
    float* Lsm = Msm + NW * NT * 8;
    #pragma unroll
    for (int j = 0; j < NT; j++) {
        #pragma unroll
        for (int i = 0; i < NSL; i++) {
            float* ob = Osm + ((size_t)warp * NT * 8 + 8 * j) * HD;
            ob[(2 * t) * HD + 16 * i + g] = O[i][j][0];
            ob[(2 * t + 1) * HD + 16 * i + g] = O[i][j][1];
            ob[(2 * t) * HD + 16 * i + g + 8] = O[i][j][2];
            ob[(2 * t + 1) * HD + 16 * i + g + 8] = O[i][j][3];
        }
        if (g == 0) {
            #pragma unroll
            for (int ri = 0; ri < 2; ri++) {
                Msm[warp * NT * 8 + 8 * j + 2 * t + ri] = m[j][ri];
                Lsm[warp * NT * 8 + 8 * j + 2 * t + ri] = l[j][ri];
            }
        }
    }
    __syncthreads();

    const size_t pbase = (size_t)(pid * num_splits + split);
    float* po = partial_o + pbase * BLOCK_ROWS * HD_PAD;
    float* pml = partial_ml + pbase * BLOCK_ROWS * 2;
    for (int idx = tid; idx < BLOCK_ROWS * HD_PAD; idx += NTHREADS) {
        int r = idx / HD_PAD, d = idx % HD_PAD;
        float v = 0.f;
        if (r < NT * 8 && d < HD) {
            float M = -INFINITY;
            #pragma unroll
            for (int w = 0; w < NW; w++) M = fmaxf(M, Msm[w * NT * 8 + r]);
            float Ms = (M == -INFINITY) ? 0.f : M;
            #pragma unroll
            for (int w = 0; w < NW; w++) {
                float mw = Msm[w * NT * 8 + r];
                float f = (mw == -INFINITY) ? 0.f : exp2f(mw - Ms);
                v += Osm[((size_t)w * NT * 8 + r) * HD + d] * f;
            }
        }
        po[idx] = v;
    }
    for (int r = tid; r < BLOCK_ROWS; r += NTHREADS) {
        float M = -INFINITY, L = 0.f;
        if (r < NT * 8) {
            #pragma unroll
            for (int w = 0; w < NW; w++) M = fmaxf(M, Msm[w * NT * 8 + r]);
            float Ms = (M == -INFINITY) ? 0.f : M;
            #pragma unroll
            for (int w = 0; w < NW; w++) {
                float mw = Msm[w * NT * 8 + r];
                L += (mw == -INFINITY) ? 0.f : Lsm[w * NT * 8 + r] * exp2f(mw - Ms);
            }
        }
        pml[r * 2] = (M == -INFINITY) ? M : M * 0.6931471805599453f;   // back to natural-log units for the combine kernel
        pml[r * 2 + 1] = L;
    }
}
