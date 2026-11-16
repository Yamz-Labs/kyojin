// gfx11 (RDNA3 / RDNA3.5) WMMA path of the Q4_K multi-row decode GEMV (x-rows 2..16). Include after fq_kq_item.cuh (fq_h2f_lo) and after
// FQ_Q4_K_BYTES / FQ_Q8_1_BYTES are defined, outside any namespace. HIP builds only.
//
// Every output row is bit-identical to the one-row v2 kernel. The one-row kernel's float result is a function of exact integer sums over 8
// activation values (one "chunk" of one 32-value sub-block) and of the float combine of these sums in a fixed order; WMMA iu8 gives the same
// exact integer sums, and the float combine is repeated expression for expression, in the same lane order and window order, so only the unit
// that produces the integers changes. (The compiler fuses the one-row kernel's __fadd_rn(x, __fmul_rn(y, z)) chains into FMAs, HIP defines both
// as plain x + y / x * y; the fused form is the reference result, so the FMAs are written explicitly below.)
//
// WMMA 16x16x16 iu8 on gfx11 (checked on gfx1151): the two half-waves are independent 8x16 products. Lanes 0-15 hold B columns 0..15 and A rows
// 0,2,..,14 (the odd A lanes are ignored); lanes 16-31 hold their own B columns and A rows 1,3,..,15 (the even A lanes are ignored). D register
// v of lane l = row 2v + (l >> 4), column l & 15. So one wave covers 32 weight rows (lane l = weight row l, B operand) against 4 x-rows (A
// operand, same data in both halves); a lane owns every partial of its own (weight row, x-row) pairs: there is no cross-lane reduction, the
// one-row kernel's lane-ordered sum runs in registers.
//
// A K=16 slot holds two chunks (8 values each): A row 2m+h selects half h (the other half is zero), so D[v = 2m+h] is the exact 8-value dot of
// x-row m with chunk 2p+h of WMMA p. Per position (block b, group g = 4 chunks x 2 sub-blocks): 4 WMMAs.
//
// Measured on gfx1151 (cold, see the patch notes): Q4_K 12288x4096 R=8 260 -> 200 us, R=16 499 -> 253 us; 4096x4096 R=8 96 -> 58 us. At R = 2, 3 and on
// thin matrices (N = 1024) the kernel is latency bound (one wave walks all K positions serially, a lane = a row, so too few waves) and loses to the
// generic kernel; the dispatcher keeps the generic kernel there.
// Q6_K has no WMMA path: its 210-byte blocks make the per-lane-row loads the bottleneck (see the patch notes).
#pragma once
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__))
#define FQ_KQ_WMMA_DEV 1
#endif

typedef int fq_v4i __attribute__((ext_vector_type(4)));
typedef int fq_v8i __attribute__((ext_vector_type(8)));

// Activation pack for the Q4_K WMMA kernel. One thread = one (pass, x-row-in-pass ml, position it).
//   afrag[((pass * nitems + it) * 8 + v) * 4 + ip] (uint4), v = 2 ml + h, ip = 2 i + p: the A operand of WMMA (i, p) for A row v
//        = {x, y, 0, 0} (h = 0) or {0, 0, x, y} (h = 1), x / y = q8_1 words 1 + qo / 5 + qo of sub-block 2g + i, qo = 2p + h.
//   atab[(pass * nitems + it) * 40 + ml * 8 + 4 i + qo] = exact sum of the 8 int8 values of (i, qo) as float,
//   atab[... + 32 + 2 ml + i] = d of sub-block 2g + i as float. Rows m >= M are all zero.
__global__ void fq_kq_wpack_q4_kernel(const uint8_t* __restrict__ aqb, uint4* __restrict__ afrag, float* __restrict__ atab,
                                      int M, int nitems, int64_t qb_stride, int64_t total)
{
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total)
        return;
    const int it = (int)(gid % nitems);
    const int64_t rest = gid / nitems;
    const int ml = (int)(rest & 3), pass = (int)(rest >> 2);
    const int m = pass * 4 + ml;
    const int b = it >> 2, g = it & 3;
    uint32_t xw[2][8];
    float d8[2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        #pragma unroll
        for (int w = 0; w < 8; w++)
            xw[i][w] = 0u;
        d8[i] = 0.0f;
    }
    if (m < M)
    {
        const uint8_t* aq = aqb + (int64_t)m * qb_stride;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            const uint32_t* a = reinterpret_cast<const uint32_t*>(aq + (size_t)(b * 8 + 2 * g + i) * FQ_Q8_1_BYTES);
            d8[i] = fq_h2f_lo(a[0]);
            #pragma unroll
            for (int w = 0; w < 8; w++)
                xw[i][w] = a[1 + w];
        }
    }
    const size_t pidx = (size_t)pass * nitems + it;
    float* tab = atab + pidx * 40;
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        #pragma unroll
        for (int qo = 0; qo < 4; qo++)
        {
            int s = 0;
            #pragma unroll
            for (int t = 0; t < 4; t++)
                s += (int)(int8_t)(xw[i][qo] >> (8 * t)) + (int)(int8_t)(xw[i][4 + qo] >> (8 * t));
            tab[ml * 8 + 4 * i + qo] = (float)s;
        }
        tab[32 + 2 * ml + i] = d8[i];
    }
    #pragma unroll
    for (int h = 0; h < 2; h++)
    {
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            #pragma unroll
            for (int p = 0; p < 2; p++)
            {
                const int qo = 2 * p + h;
                const uint32_t x = xw[i][qo], y = xw[i][4 + qo];
                afrag[(pidx * 8 + 2 * ml + h) * 4 + 2 * i + p] = h ? make_uint4(0u, 0u, x, y) : make_uint4(x, y, 0u, 0u);
            }
        }
    }
}

// Weights / activation fragments of one Q4_K position (block b = it >> 2, group g = it & 3) of one lane, loaded one position ahead.
struct FqKqWmmaLd4 { uint4 hdr, va, vb, a00, a01, a10, a11; };

__device__ __forceinline__ FqKqWmmaLd4 fq_kq_wmma_ld4(const uint8_t* __restrict__ wr, const uint4* __restrict__ afrag, int it, size_t pidx, int vA)
{
    FqKqWmmaLd4 l;
    const uint8_t* blk = wr + (size_t)(it >> 2) * FQ_Q4_K_BYTES;
    const int g = it & 3;
    l.hdr = *reinterpret_cast<const uint4*>(blk);
    l.va = *reinterpret_cast<const uint4*>(blk + 16 + 32 * g);
    l.vb = *reinterpret_cast<const uint4*>(blk + 32 + 32 * g);
    const uint4* af = afrag + (pidx * 8 + vA) * 4;
    l.a00 = af[0]; l.a01 = af[1]; l.a10 = af[2]; l.a11 = af[3];
    return l;
}

// Computes one Q4_K position of one lane for the 4 x-rows (MT of them) of a pass: 4 WMMAs, then the one-row kernel's float combine.
// acc[m][qo] = lane partial of one-row lane 4 jpos + qo (windows added in order: `first` starts it).
template <int MT>
__device__ __forceinline__ void fq_kq_wmma_pos4(const FqKqWmmaLd4& cur, const float* __restrict__ tab, int g, bool first, float (&acc)[MT][4])
{
    const int sh = 16 * (g & 1);
    const uint32_t hi = (g >= 2) ? 0xFFFFFFFFu : 0u;
    // scales / mins of sub-blocks 2g, 2g+1 (verbatim fq_kq_wld_q4_k)
    const float dm0 = fq_h2f_lo(cur.hdr.x), dm1 = fq_h2f_lo(cur.hdr.x >> 16);
    const uint32_t s0 = (cur.hdr.y >> sh) & 0xFFFFu, s2 = (cur.hdr.z >> sh) & 0xFFFFu, s4 = (cur.hdr.w >> sh) & 0xFFFFu;
    const uint16_t aux0 = (uint16_t)(((s0 & 0x3F3Fu) & ~hi) | ((((s4 >> 0) & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2)) & hi));
    const uint16_t aux1 = (uint16_t)(((s2 & 0x3F3Fu) & ~hi) | ((((s4 >> 4) & 0x0F0Fu) | ((s2 & 0xC0C0u) >> 2)) & hi));
    const float scf[2] = {(float)(int)(aux0 & 0xFFu), (float)(int)((aux0 >> 8) & 0xFFu)};
    const float mnf[2] = {(float)(int)(aux1 & 0xFFu), (float)(int)((aux1 >> 8) & 0xFFu)};
    const uint32_t w0[4] = {cur.va.x, cur.va.y, cur.va.z, cur.va.w};
    const uint32_t w1[4] = {cur.vb.x, cur.vb.y, cur.vb.z, cur.vb.w};
    const uint4 af4[2][2] = {{cur.a00, cur.a01}, {cur.a10, cur.a11}};
    fq_v8i c[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        #pragma unroll
        for (int p = 0; p < 2; p++)
        {
            const fq_v4i bf = {(int)((w0[2 * p] >> (4 * i)) & 0x0F0F0F0Fu), (int)((w1[2 * p] >> (4 * i)) & 0x0F0F0F0Fu),
                               (int)((w0[2 * p + 1] >> (4 * i)) & 0x0F0F0F0Fu), (int)((w1[2 * p + 1] >> (4 * i)) & 0x0F0F0F0Fu)};
            const fq_v4i af_ = {(int)af4[i][p].x, (int)af4[i][p].y, (int)af4[i][p].z, (int)af4[i][p].w};
            const fq_v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
            c[i][p] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af_, true, bf, z, false);
        }
    }
    #pragma unroll
    for (int m = 0; m < MT; m++)
    {
        const float d80 = tab[32 + 2 * m], d81 = tab[33 + 2 * m];
        #pragma unroll
        for (int qo = 0; qo < 4; qo++)
        {
            const int p = qo >> 1, v = 2 * m + (qo & 1);
            const float a0 = (float)c[0][p][v] * scf[0];
            const float a1 = (float)c[1][p][v] * scf[1];
            const float b0 = tab[m * 8 + qo] * mnf[0];
            const float b1 = tab[m * 8 + 4 + qo] * mnf[1];
            // the compiler fuses the one-row kernel's __fadd_rn(x, __fmul_rn(y, z)) chains into FMAs (HIP defines both as plain
            // x + y / x * y under -ffp-contract=fast); the fused form is the reference result, so it is written explicitly
            const float fd = __builtin_fmaf(d81, a1, d80 * a0);
            const float fm = __builtin_fmaf(d81, b1, d80 * b0);
            const float t = __builtin_fmaf(dm0, fd, -(dm1 * fm));
            acc[m][qo] = first ? t : (acc[m][qo] + t);
        }
    }
}

// One wave = 32 consecutive weight rows (lane = row) x the 4 x-rows of one pass. A workgroup of P * TPW waves: wave w = (tile
// blockIdx.x * TPW + w / P, pass w % P); the P waves of a tile read the same weights at the same time (L0 hits). MT = x-rows computed
// per pass (2 or 4). Positions it = jpos + 64 win (win ascending, as the one-row kernel's thread loop) are processed jpos by jpos. The loads run
// ND - 1 positions ahead of the compute (ring of ND register buffers, indices known at compile time).
template <typename OUT, int MT, int ND>
__global__ void __launch_bounds__(128) fq_gemv_kq_wmma_q4_kernel(const uint8_t* __restrict__ packed, const uint4* __restrict__ afrag,
                                                                 const float* __restrict__ atab, OUT* __restrict__ out,
                                                                 int M, int N, int64_t pstride, int nitems, int P, int TPW)
{
#if defined(FQ_KQ_WMMA_DEV)
#pragma clang fp contract(off)
    const int lane = threadIdx.x & 31;
    const int wid = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const int pass = wid % P;
    const int n_base = (blockIdx.x * TPW + wid / P) * 32;
    const uint8_t* wr = packed + (int64_t)min(n_base + lane, N - 1) * pstride;
    const int vA = (lane & 15) >> 1;
    float s[MT];
    float acc[MT][4];
    #pragma unroll
    for (int m = 0; m < MT; m++)
    {
        s[m] = 0.0f;
        #pragma unroll
        for (int qo = 0; qo < 4; qo++)
            acc[m][qo] = 0.0f;
    }
    const size_t pbase = (size_t)pass * nitems;
    int cj = 0, ci = 0;                       // compute cursor (jpos, it)
    int lj = 0, li = 0;                       // load cursor
    // The load cursor stops on the last position (the loads stay unconditional: the compiler can then count the loads in flight and wait
    // for the oldest buffer only).
#define FQ_KQ_ADV4(J, I)                                                                                                           \
    {                                                                                                                              \
        int ni = I + 64, nj = J;                                                                                                   \
        if (ni >= nitems) { nj = J + 1; ni = nj; }                                                                                 \
        if (nj < 64) { J = nj; I = ni; }                                                                                           \
    }
    FqKqWmmaLd4 buf[ND];
    #pragma unroll
    for (int j = 0; j < ND - 1; j++)
    {
        buf[j] = fq_kq_wmma_ld4(wr, afrag, li, pbase + li, vA);
        FQ_KQ_ADV4(lj, li)
    }
    bool fin = false;
    for (;;)
    {
        #pragma unroll
        for (int j = 0; j < ND; j++)
        {
            buf[(j + ND - 1) % ND] = fq_kq_wmma_ld4(wr, afrag, li, pbase + li, vA);
            FQ_KQ_ADV4(lj, li)
            __builtin_amdgcn_sched_barrier(0);        // keep the prefetch above the compute (the scheduler would sink the loads to their first use)
            fq_kq_wmma_pos4<MT>(buf[j], atab + (pbase + ci) * 40, ci & 3, ci == cj, acc);
            ci += 64;
            if (ci >= nitems)                   // all windows of this jpos done: one-row lanes 4 jpos .. 4 jpos + 3, in order
            {
                #pragma unroll
                for (int m = 0; m < MT; m++)
                {
                    #pragma unroll
                    for (int qo = 0; qo < 4; qo++)
                        s[m] = s[m] + acc[m][qo];
                }
                cj++; ci = cj;
                if (cj >= 64) { fin = true; break; }
            }
        }
        if (fin)
            break;
    }
#undef FQ_KQ_ADV4
    const int n = n_base + lane;
    #pragma unroll
    for (int m = 0; m < MT; m++)
    {
        if (pass * 4 + m < M && n < N)
        {
            if constexpr (std::is_same<OUT, __half>::value)
                out[(int64_t)(pass * 4 + m) * N + n] = __float2half_rn(s[m]);
            else
                out[(int64_t)(pass * 4 + m) * N + n] = s[m];
        }
    }
#endif
}

#endif  // USE_ROCM
