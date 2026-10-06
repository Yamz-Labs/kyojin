#pragma once
// core7: matvec phases (core5's gate/up and down) with compile-time ablation / decode variants. ABL bits: 1 = no trellis decode, 2 = every slice re-reads slice 0
// (no DRAM stream), 8 = hash multiply as two v_mul_u32_u24 + shift-add (bit-identical: all window states are 16 bit). ABL = 0 is core5's body.
// Generated from exl3_moe_fused.cuh (names mf_tile -> mf7_tile ...); ablated kernels compute garbage on purpose and are used for timing only.
namespace mf7_mv {
__device__ __forceinline__ uint32_t sad_u8_c7(uint32_t p, uint32_t addend)   // same instruction as exl3_sad_u8_ (V_SAD_U8(p, 0, addend))
{
    uint32_t u;
    asm("v_sad_u8 %0, %1, %2, %3" : "=v"(u) : "v"(p), "n"(0), "n"(addend));
    return u;
}
__device__ __forceinline__ uint32_t hash_m24(uint32_t x)            // == x * 0x83DCD12D (mod 2^32) for x < 2^24
{
    return __umul24(x, 0xDCD12Du) + (__umul24(x, 0x83u) << 24);
}
__device__ __forceinline__ half2 dec_pair_m24(uint32_t x0, uint32_t x1)
{
    x0 = hash_m24(x0); x1 = hash_m24(x1);
    const uint32_t sum1 = sad_u8_c7(x1, 0x6400u);
    const uint32_t sum0 = sad_u8_c7(x0, 0x6400u);
    const uint32_t packed = (sum1 << 16) + sum0;
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half_uint16 h0((uint16_t) packed);
    half_uint16 h1((uint16_t) (packed >> 16));
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}
__device__ __forceinline__ void decode8_m24(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    f0[0] = dec_pair_m24(w0, w1); f0[1] = dec_pair_m24(w2, w3); f1[0] = dec_pair_m24(w4, w5); f1[1] = dec_pair_m24(w6, w7);
}
__device__ __forceinline__ void dq8_regs_2bits_m24(uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2); BFE16_IMM(w5, b, 4); BFE16_IMM(w4, b, 6); BFE16_IMM(w3, b, 8); BFE16_IMM(w2, b, 10); BFE16_IMM(w1, b, 12); BFE16_IMM(w0, b, 14);
    decode8_m24(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}
__device__ __forceinline__ void dq8_regs_4bits_m24(uint32_t a, uint32_t b, FragB& f0, FragB& f1)
{
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4); BFE16_IMM(w5, b, 8); BFE16_IMM(w4, b, 12); BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4); BFE16_IMM(w0, s, 8);
    decode8_m24(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}
}  // namespace mf7_mv
namespace mf7 {
using namespace mf;
using mf7_mv::dq8_regs_2bits_m24; using mf7_mv::dq8_regs_4bits_m24;
template <int BITS, bool CF32, int RW, int KS, int ABL>
__device__ __forceinline__ void mf7_tile(const uint32_t* __restrict__ A32, const uint32_t* __restrict__ Bt, long ss, void* __restrict__ C,
                                        int size_n, int tile, float* red, int wave, int lane)
{
    static_assert(BITS >= 2 && BITS <= 6, "K1, K7.. need their own per-lane word set and decode body: not implemented");
    constexpr int NWD = BITS + 1, CH = (KS + 15) / 16, NRND = 4, K = KS * 16;
    constexpr int VU = CH > 5 ? 5 : CH;
    const int g8 = lane & 7, cq = lane >> 3;
    float rs0[RW], rs1[RW];
    #pragma unroll
    for (int r = 0; r < RW; ++r) { rs0[r] = 0.f; rs1[r] = 0.f; }
    uint32_t wa[VU][NWD], wb[VU][NWD];
    auto ld = [&](uint32_t (&w)[NWD], int s)
    {
        if constexpr ((ABL & 2) != 0) s = 0;   // ablation: every slice re-reads slice 0 (cache hits, same instruction count)
        w[0] = Bt[wrp_i(BITS, g8, s, ss)];
        const uint32_t* q = Bt + own_i(BITS, g8, s, ss);
        if constexpr (BITS == 2) { const uint2 v = *(const uint2*) q; w[1] = v.x; w[2] = v.y; }
        else if constexpr (BITS == 3) { w[1] = q[0]; w[2] = q[1]; w[3] = q[2]; }
        else if constexpr (BITS == 5) { w[1] = q[0]; w[2] = q[1]; w[3] = q[2]; w[4] = q[3]; w[5] = q[4]; }
        else if constexpr (BITS == 6) { const uint2 v0 = *(const uint2*) q, v1 = *(const uint2*) (q + 2), v2 = *(const uint2*) (q + 4); w[1] = v0.x; w[2] = v0.y; w[3] = v1.x; w[4] = v1.y; w[5] = v2.x; w[6] = v2.y; }
        else { const uint4 v = *(const uint4*) q; w[1] = v.x; w[2] = v.y; w[3] = v.z; w[4] = v.w; }
    };
    auto ldst = [&](uint32_t (&w)[VU][NWD], int rnd, int sg0)
    {
        const int cc = 4 * rnd + cq, s0n = cc * CH, m = myn(KS, cc);
        #pragma unroll
        for (int u = 0; u < VU; ++u) if (sg0 + u < m) ld(w[u], s0n + sg0 + u);
    };
    ldst(wa, 0, 0);
    #pragma unroll 1
    for (int round = 0; round < NRND; ++round)
    {
        const int c = 4 * round + cq;
        const int s0 = c * CH;
        const int myn_ = myn(KS, c);
        float acc0[RW], acc1[RW];
        #pragma unroll
        for (int r = 0; r < RW; ++r) { acc0[r] = 0.f; acc1[r] = 0.f; }
        #pragma unroll 1
        for (int sb = 0; sb < CH; sb += VU)
        {
            int nr = round, ns = sb + VU;
            if (ns >= CH) { nr = round + 1; ns = 0; }
            if (nr < NRND) ldst(wb, nr, ns);
            #pragma unroll
            for (int u = 0; u < VU; ++u)
            {
                if (sb + u < myn_)
                {
                    FragB f0[4], f1[4];
                    #pragma unroll
                    for (int t = 0; t < 4; ++t)
                    {
                        if constexpr ((ABL & 1) != 0)
                        {   // ablation: no trellis decode, the words themselves stand in for the decoded halves
                            f0[t][0] = __builtin_bit_cast(half2, wa[u][t & 1]); f0[t][1] = __builtin_bit_cast(half2, wa[u][1 + (t & 1)] ^ (uint32_t) t);
                            f1[t][0] = __builtin_bit_cast(half2, wa[u][(t + 1) & 1] + 0x1234u); f1[t][1] = __builtin_bit_cast(half2, wa[u][1 + ((t + 1) & 1)]);
                        }
                        else if constexpr ((ABL & 8) != 0 && BITS == 2) dq8_regs_2bits_m24(wa[u][t >> 1], wa[u][1 + (t >> 1)], t << 3, f0[t], f1[t]);
                        else if constexpr ((ABL & 8) != 0 && BITS == 4) dq8_regs_4bits_m24(wa[u][t], wa[u][t + 1], f0[t], f1[t]);
                        else if constexpr (BITS == 2) exl3_gemv_ns::dq8_regs_2bits<2>(wa[u][t >> 1], wa[u][1 + (t >> 1)], t << 3, f0[t], f1[t]);
                        else if constexpr (BITS == 3) exl3_gemv_ns::dq8_regs_3bits<2>(wa[u][t == 0 ? 0 : (t == 1 ? 1 : 2)], wa[u][t == 0 ? 1 : (t == 1 ? 2 : 3)], t == 0 ? 8 : (t == 1 ? 16 : (t == 2 ? 24 : 0)), f0[t], f1[t]);
                        else if constexpr (BITS == 5) mf_dq8_k5(wa[u], t, f0[t], f1[t]);
                        else if constexpr (BITS == 6) mf_dq8_k6(wa[u], t, f0[t], f1[t]);
                        else exl3_gemv_ns::dq8_regs_4bits<2>(wa[u][t], wa[u][t + 1], f0[t], f1[t]);
                    }
                    #pragma unroll
                    for (int r = 0; r < RW; ++r)
                    {
                        const long ai = a_idx(r, K, s0 + sb + u);
                        const uint4 alo = *(const uint4*) (A32 + ai);
                        const uint4 ahi = *(const uint4*) (A32 + ai + 4);
                        const uint32_t aq[8] = {alo.x, alo.y, alo.z, alo.w, ahi.x, ahi.y, ahi.z, ahi.w};
                        #pragma unroll
                        for (int q = 0; q < 8; ++q)
                        {
                            const int t = q & 3, h = q >> 2;
                            acc0[r] = mv_dot2(__builtin_bit_cast(uint32_t, f0[t][h]), aq[q], acc0[r]);
                            acc1[r] = mv_dot2(__builtin_bit_cast(uint32_t, f1[t][h]), aq[q], acc1[r]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int u = 0; u < VU; ++u)
                #pragma unroll
                for (int w = 0; w < NWD; ++w) wa[u][w] = wb[u][w];
        }
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            red[red_idx(wave, cq, g8, r)] = acc0[r];
            red[red_idx(wave, cq, g8 + 8, r)] = acc1[r];
        }
        __syncwarp();
        if (cq == 0)
        {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc)
                #pragma unroll
                for (int r = 0; r < RW; ++r)
                {
                    rs0[r] += red[red_idx(wave, cc, g8, r)];
                    rs1[r] += red[red_idx(wave, cc, g8 + 8, r)];
                }
        }
        __syncwarp();
    }
    if (cq == 0)
    {
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            if constexpr (CF32) { ((float*) C)[c_idx(r, size_n, tile, g8)] = rs0[r]; ((float*) C)[c_idx(r, size_n, tile, g8 + 8)] = rs1[r]; }
            else { ((half*) C)[c_idx(r, size_n, tile, g8)] = __float2half_rn(rs0[r]); ((half*) C)[c_idx(r, size_n, tile, g8 + 8)] = __float2half_rn(rs1[r]); }
        }
    }
}

template <int BITS, bool CF32, int KS, int ABL>
__device__ __forceinline__ void mf7_tile_rows(int rows, const uint32_t* A32, const uint32_t* Bt, long ss, void* C, int size_n, int tile, float* red, int wave, int lane)
{
    if (rows == 1) mf7_tile<BITS, CF32, 1, KS, ABL>(A32, Bt, ss, C, size_n, tile, red, wave, lane);
    else if (rows == 2) mf7_tile<BITS, CF32, 2, KS, ABL>(A32, Bt, ss, C, size_n, tile, red, wave, lane);
    else if (rows == 3) mf7_tile<BITS, CF32, 3, KS, ABL>(A32, Bt, ss, C, size_n, tile, red, wave, lane);
    else mf7_tile<BITS, CF32, 4, KS, ABL>(A32, Bt, ss, C, size_n, tile, red, wave, lane);
}


template <class S, int ABL>
__device__ void ph_gateup7(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, NA = R * TOPK, NAR = NA + R;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NU = s.ngroups + 1;
    constexpr int TPP = DM::TPP;
    const bool nat = p.native != 0;
    const long long W = split_total(s.urows, NU, 2, TPP);
    const int lo = split_tile(s.urows, NU, 2, TPP, (long long) blockIdx.x * W / p.G), hi = split_tile(s.urows, NU, 2, TPP, (long long) (blockIdx.x + 1) * W / p.G);
    int i = lo;
    while (i < hi)
    {
        const int segA = i / TPP, endA = min(hi, (segA + 1) * TPP);
        const int iB = endA, segB = iB / TPP;
        const bool hasB = iB < hi;
        const int endB = hasB ? min(hi, (segB + 1) * TPP) : iB;
        const int nA = endA - i, nB = endB - iB;
        __syncthreads();
        for (int w2 = 0; w2 < (hasB ? 2 : 1); ++w2)
        {
            const int seg = w2 ? segB : segA;
            const Seg g = seginfo<S>(p, s, seg, true);
            half* Abuf = w2 ? s.u.g.A2 : s.u.g.A;
            const half* suh = g.sv[g.proj == 0 ? 0 : 2];
            for (int task = wave; task < g.rows * DM::CMB_BLOCKS; task += 16)
            {
                const int row = task / DM::CMB_BLOCKS, blk = task % DM::CMB_BLOCKS;
                const int token = g.shared ? row : s.tok[g.p0 + row];
                half4 v = ((const half4*) (p.mixed + mixed_row<S>(token) + blk * 128))[lane];
                v = had_regs_h<true, false>(v, suh + blk * 128, lane);
                ((half4*) (Abuf + a_row_gu<S>(row, blk)))[lane] = v;
            }
        }
        __syncthreads();
        for (int n = wave; n < nA + nB; n += 16)
        {
            const bool inB = n >= nA;
            const int seg = inB ? segB : segA;
            const int tile = (inB ? iB + (n - nA) : i + n) - seg * TPP;
            const Seg g = seginfo<S>(p, s, seg, true);
            const uint32_t* Ab = (const uint32_t*) (inB ? s.u.g.A2 : s.u.g.A);
            half* C = p.gu + gu_out<S>(g.proj, NAR, g.p0);
            const uint32_t* Wm = p.wt[3 * (g.shared ? DM::NEXP : g.e) + g.proj];
            if (g.shared)
                mf7_tile_rows<DM::SB, false, DM::KSG, ABL>(g.rows, Ab, Wm + tile_off(nat, DM::SB, DM::KSG, TPP, tile), tile_ss(nat, DM::SB, TPP), C, INTER, tile, s.u.g.red, wave, lane);
            else
                mf7_tile_rows<DM::RB, false, DM::KSG, ABL>(g.rows, Ab, Wm + tile_off(nat, DM::RB, DM::KSG, TPP, tile), tile_ss(nat, DM::RB, TPP), C, INTER, tile, s.u.g.red, wave, lane);
        }
        i = endB;
    }
}

template <class S, int ABL>
__device__ void ph_down7(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, NA = R * TOPK, NAR = NA + R;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NU = s.ngroups + 1;
    constexpr int TPU = DM::TPU;
    const bool nat = p.native != 0;
    const long long W = split_total(s.urows, NU, 1, TPU);
    const int lo = split_tile(s.urows, NU, 1, TPU, (long long) blockIdx.x * W / p.G), hi = split_tile(s.urows, NU, 1, TPU, (long long) (blockIdx.x + 1) * W / p.G);
    int i = lo;
    while (i < hi)
    {
        const int segA = i / TPU, endA = min(hi, (segA + 1) * TPU);
        const int iB = endA, segB = iB / TPU;
        const bool hasB = iB < hi;
        const int endB = hasB ? min(hi, (segB + 1) * TPU) : iB;
        const int nA = endA - i, nB = endB - iB;
        __syncthreads();
        for (int w2 = 0; w2 < (hasB ? 2 : 1); ++w2)
        {
            const Seg g = seginfo<S>(p, s, w2 ? segB : segA, false);
            half* Abuf = w2 ? s.u.g.A2 : s.u.g.A;
            const half* const* sv = g.sv;
            for (int task = wave; task < g.rows * DM::INT_BLOCKS; task += 16)
            {
                const int row = task / DM::INT_BLOCKS, blk = task % DM::INT_BLOCKS;
                half4 gg = ((const half4*) (p.gu + gu_read<S>(NAR, g.p0 + row, blk, 0)))[lane];
                half4 uu = ((const half4*) (p.gu + gu_read<S>(NAR, g.p0 + row, blk, 1)))[lane];
                gg = had_regs_h<false, true>(gg, sv[1] + blk * 128, lane);
                uu = had_regs_h<false, true>(uu, sv[3] + blk * 128, lane);
                half4 a;
                {
                    float f0 = __half2float(__low2half(gg.x)), f1 = __half2float(__high2half(gg.x));
                    float f2 = __half2float(__low2half(gg.y)), f3 = __half2float(__high2half(gg.y));
                    half h0 = __float2half_rn(f0 / (1.0f + expf(-f0))), h1 = __float2half_rn(f1 / (1.0f + expf(-f1)));
                    half h2 = __float2half_rn(f2 / (1.0f + expf(-f2))), h3 = __float2half_rn(f3 / (1.0f + expf(-f3)));
                    a.x = __halves2half2(__hmul(h0, __low2half(uu.x)), __hmul(h1, __high2half(uu.x)));
                    a.y = __halves2half2(__hmul(h2, __low2half(uu.y)), __hmul(h3, __high2half(uu.y)));
                }
                a = had_regs_h<true, false>(a, sv[4] + blk * 128, lane);
                ((half4*) (Abuf + a_row_dn<S>(row, blk)))[lane] = a;
            }
        }
        __syncthreads();
        for (int n = wave; n < nA + nB; n += 16)
        {
            const bool inB = n >= nA;
            const int seg = inB ? segB : segA;
            const int tile = (inB ? iB + (n - nA) : i + n) - seg * TPU;
            const Seg g = seginfo<S>(p, s, seg, false);
            const uint32_t* Ab = (const uint32_t*) (inB ? s.u.g.A2 : s.u.g.A);
            float* C = p.dn + dn_row<S>(g.p0);
            const uint32_t* Wm = p.wt[3 * (g.shared ? DM::NEXP : g.e) + 2];
            if (g.shared)
                mf7_tile_rows<DM::SB, true, DM::KSD, ABL>(g.rows, Ab, Wm + tile_off(nat, DM::SB, DM::KSD, TPU, tile), tile_ss(nat, DM::SB, TPU), C, D, tile, s.u.g.red, wave, lane);
            else
                mf7_tile_rows<DM::RB, true, DM::KSD, ABL>(g.rows, Ab, Wm + tile_off(nat, DM::RB, DM::KSD, TPU, tile), tile_ss(nat, DM::RB, TPU), C, D, tile, s.u.g.red, wave, lane);
        }
        i = endB;
    }
}


}  // namespace mf7
