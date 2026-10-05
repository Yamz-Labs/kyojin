#pragma once
// VALU grouped-MoE decode matvec (gfx11.5, opt-in EXL3_MOE_VALU=1). Replaces the WMMA/LDS body of the grouped and
// expert-dedup decode launches for K2 and K4 experts, one launch per projection as before.
//
// Numerics: v_wmma_f32_16x16x16_f16 equals a chain of 8 v_dot2_f32_f16 over k ascending from the running accumulator, so a lane that
// owns a whole 16-k column per k-slice reproduces the engine's WMMA body bit for bit. The 16 k-chunks (WK = 16) are reduced in chunk
// order from 0.0f exactly like the engine's `sum += red[j]`. One reduction order for every row count: a row's result does not depend
// on how many rows share the launch, so decode (R=1) and verify rounds (R>1) give identical bits.
//
// Mapping: one wave = one 16-column tile, all k. lane = cq * 8 + g8: g8 = column pair (g8, g8 + 8), cq = chunk lane; 4 rounds x 4
// chunk lanes = the 16 k-chunks. Each lane reads its own `bits` trellis words plus one wrap word per slice straight from global and
// decodes with the engine's own dq8_regs_* helpers. Rows run in passes of at most 4.
#include "exl3_moe_valu_idx.h"

#if defined(USE_ROCM) || defined(__HIPCC__)
typedef _Float16 mv_h2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float mv_dot2(uint32_t a, uint32_t b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(mv_h2, a), __builtin_bit_cast(mv_h2, b), c, false);
}

#ifndef EXL3_MOE_VALU_W
#define EXL3_MOE_VALU_W 4          // waves (tiles) per block
#endif

// One tile (16 columns) of RW rows. VU = k-slices per pipeline stage (register ring of VU slices, crossing round boundaries).
template <int BITS, bool CF32, int RW, int VU>
__device__ __forceinline__ void mv_tile
(
    const uint32_t* __restrict__ A32, int size_k, const uint32_t* __restrict__ B32, int ntiles,
    void* __restrict__ C, int size_n, int tile, float* red, int wave, int lane
)
{
    constexpr int NWD = BITS + 1;
    const int ks = size_k / 16, CH = mv::chunk(ks), NRND = mv::NCH / 4;
    const int g8 = lane & 7, cq = lane >> 3;
    float rs0[RW], rs1[RW];
    #pragma unroll
    for (int r = 0; r < RW; ++r) { rs0[r] = 0.f; rs1[r] = 0.f; }
    uint32_t wa[VU][NWD], wb[VU][NWD];
    auto ld = [&](uint32_t (&w)[NWD], int s)
    {
        w[0] = B32[mv::wrap(BITS, ntiles, tile, g8, s)];
        const uint32_t* p = B32 + mv::own(BITS, ntiles, tile, g8, s);
        if constexpr (BITS == 2) { const uint2 v = *(const uint2*) p; w[1] = v.x; w[2] = v.y; }
        else { const uint4 v = *(const uint4*) p; w[1] = v.x; w[2] = v.y; w[3] = v.z; w[4] = v.w; }
    };
    auto ldst = [&](uint32_t (&w)[VU][NWD], int rnd, int sg0)
    {
        const int cc = 4 * rnd + cq, s0n = cc * CH, m = mv::myn(ks, cc);
        #pragma unroll
        for (int u = 0; u < VU; ++u) if (sg0 + u < m) ld(w[u], s0n + sg0 + u);
    };
    ldst(wa, 0, 0);
    #pragma unroll 1
    for (int round = 0; round < NRND; ++round)
    {
        const int c = 4 * round + cq;
        const int s0 = c * CH;
        const int myn = mv::myn(ks, c);
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
                if (sb + u < myn)
                {
                    FragB f0[4], f1[4];
                    #pragma unroll
                    for (int t = 0; t < 4; ++t)
                    {
                        if constexpr (BITS == 2) exl3_gemv_ns::dq8_regs_2bits<2>(wa[u][t >> 1], wa[u][1 + (t >> 1)], t << 3, f0[t], f1[t]);
                        else exl3_gemv_ns::dq8_regs_4bits<2>(wa[u][t], wa[u][t + 1], f0[t], f1[t]);
                    }
                    #pragma unroll
                    for (int r = 0; r < RW; ++r)
                    {
                        const long ai = mv::a_idx(r, size_k, s0 + sb + u);
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
            red[mv::red_idx(wave, cq, g8, r)] = acc0[r];
            red[mv::red_idx(wave, cq, g8 + 8, r)] = acc1[r];
        }
        __syncwarp();
        if (cq == 0)
        {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc)
                #pragma unroll
                for (int r = 0; r < RW; ++r)
                {
                    rs0[r] += red[mv::red_idx(wave, cc, g8, r)];
                    rs1[r] += red[mv::red_idx(wave, cc, g8 + 8, r)];
                }
        }
        __syncwarp();
    }
    if (cq == 0)
    {
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            if constexpr (CF32)
            {
                ((float*) C)[mv::c_idx(r, size_n, tile, g8)] = rs0[r];
                ((float*) C)[mv::c_idx(r, size_n, tile, g8 + 8)] = rs1[r];
            }
            else
            {
                ((half*) C)[mv::c_idx(r, size_n, tile, g8)] = __float2half_rn(rs0[r]);
                ((half*) C)[mv::c_idx(r, size_n, tile, g8 + 8)] = __float2half_rn(rs1[r]);
            }
        }
    }
}

// DEDUP: blockIdx.y is a run of equal experts in the sorted assignment list (rows = run length); otherwise blockIdx.y is the
// assignment slot (one row). Same buffer layout and zero-fill of invalid experts as the WMMA grouped kernels.
template <int BITS, bool CF32, bool DEDUP, bool TWO, int VU>
__global__ __launch_bounds__(EXL3_MOE_VALU_W * 32)
void moe_valu_kernel
(
    const half* A, const int64_t* selected, const int* group_start, const int* num_groups,
    const int64_t* B_table_0, const int64_t* B_table_1, void* C,
    int size_k, int size_n, int experts, int assignments
)
{
    constexpr int W = EXL3_MOE_VALU_W;
    __shared__ float red[mv::red_size(W)];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ntiles = size_n / 16;
    const int tile = mv::tile_of(blockIdx.x, wave, W);
    int p0, rows;
    if constexpr (DEDUP)
    {
        const int g = blockIdx.y;
        if (g >= *num_groups) return;
        p0 = group_start[g];
        rows = group_start[g + 1] - p0;
    }
    else { p0 = blockIdx.y; rows = 1; }
    if (tile >= ntiles) return;
    const int projection = TWO ? blockIdx.z : 0;
    const int64_t expert = selected[p0];
    const int64_t* B_table = projection == 0 ? B_table_0 : B_table_1;
    const size_t matrix = (size_t) projection * assignments + p0;
    void* C_rows = CF32 ? static_cast<void*>(reinterpret_cast<float*>(C) + matrix * size_n)
                        : static_cast<void*>(reinterpret_cast<half*>(C) + matrix * size_n);
    const bool valid_expert = expert >= 0 && expert < experts;
    const int64_t B_ptr = valid_expert ? B_table[expert] : 0;
    if (!B_ptr)
    {
        if (lane < 16)
            for (int r = 0; r < rows; ++r)
            {
                const size_t o = mv::c_idx(r, size_n, tile, lane);
                if constexpr (CF32) reinterpret_cast<float*>(C_rows)[o] = 0.0f;
                else reinterpret_cast<half*>(C_rows)[o] = __float2half_rn(0.0f);
            }
        return;
    }
    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B_ptr);
    const uint32_t* A_rows = reinterpret_cast<const uint32_t*>(A + matrix * size_k);
    for (int r0 = 0; r0 < rows; r0 += mv::MAXPASS)
    {
        const int n = mv::pass_rows(rows, r0);
        const uint32_t* Ap = A_rows + (size_t) r0 * (size_k / 2);
        void* Cp = CF32 ? static_cast<void*>(reinterpret_cast<float*>(C_rows) + (size_t) r0 * size_n)
                        : static_cast<void*>(reinterpret_cast<half*>(C_rows) + (size_t) r0 * size_n);
        if (n == 1) mv_tile<BITS, CF32, 1, VU>(Ap, size_k, B32, ntiles, Cp, size_n, tile, red, wave, lane);
        else if (n == 2) mv_tile<BITS, CF32, 2, VU>(Ap, size_k, B32, ntiles, Cp, size_n, tile, red, wave, lane);
        else if (n == 3) mv_tile<BITS, CF32, 3, VU>(Ap, size_k, B32, ntiles, Cp, size_n, tile, red, wave, lane);
        else mv_tile<BITS, CF32, 4, VU>(Ap, size_k, B32, ntiles, Cp, size_n, tile, red, wave, lane);
    }
}
#endif  // USE_ROCM
