#pragma once

// Grouped trellis quantizer for gfx11 (RDNA3 / RDNA3.5), K = 2..4. Produces the same indices as
// quantize_tiles_kernel (same half arithmetic, same first-minimum tie rule, same argmin ranking).
//
// The 2^K states S*g .. S*g + S-1 of a group g share the same S = 2^K predecessors
// (c << (16 - 2K)) | g, so one thread owns whole groups: it loads the S predecessor costs once
// and evaluates S x S candidates from decoded values cached in registers. The per-state minimum
// uses packed (cost << 16 | branch) keys. Costs are non-negative halves, so unsigned key order is
// cost order, and equal costs resolve to the lowest branch, i.e. the reference's strict-less
// first-minimum rule over increasing branches (infinite costs included).
//
// Costs stay in LDS: two arrays when they fit in 64 KiB (K >= 3), otherwise one array with the
// new costs staged in registers between two barriers (K = 2). Traceback history is bit-packed,
// K bits per state, one word per group and step (K=2: 8 bits, K=3: 24 of 32, K=4: 64 bits), so a
// tile writes 1 MiB (K=2/3) instead of the byte-per-state layout's 4/2 MiB.

#include <type_traits>
#include "quantize_tiles_kernel.cuh"

template <int K>
using qt_rdna_hword = std::conditional_t<K == 2, uint8_t, std::conditional_t<K == 3, uint32_t, uint64_t>>;

template <int K, int L>
__host__ __device__ constexpr bool qt_rdna_double_buffer()
{
    return 2 * (65536 >> K) * 2 + L * 2 + 128 <= 65536;
}

template <int K, int L>
__host__ __device__ constexpr int qt_rdna_shmem()
{
    return L * 2 + 128 + (qt_rdna_double_buffer<K, L>() ? 2 : 1) * (65536 >> K) * 2;
}

template <int K, int L>
__host__ __device__ constexpr int64_t qt_rdna_history_bytes()
{
    return (int64_t) L * (65536 >> (2 * K)) * sizeof(qt_rdna_hword<K>);
}

template <int K, int cb, int L, int NT>
__global__ __launch_bounds__(NT) void quantize_tiles_rdna_kernel(
    const float* __restrict__ input_tiles_ptr, float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr, half* __restrict__ temp_costs_ptr,
    uint16_t* __restrict__ temp_edges_ptr, const half2* __restrict__ lut)
{
    static_assert(K >= 2 && K <= 4 && L % 2 == 0);
    using hword = qt_rdna_hword<K>;
    constexpr int S = 1 << K;
    constexpr int edges = 65536 >> K;
    constexpr int groups = edges / S;
    constexpr int NG = groups / NT;
    static_assert(NG >= 1 && groups % NT == 0, "block must tile the groups");
    constexpr int NW = NT / 32;
    constexpr int Kr = 16 - K;
    constexpr int pred_shift = 16 - 2 * K;
    constexpr bool dbl = qt_rdna_double_buffer<K, L>();

    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    hword* hist = reinterpret_cast<hword*>(
        reinterpret_cast<uint8_t*>(temp_edges_ptr) + qt_rdna_history_bytes<K, L>() * tile_idx);

    extern __shared__ uint8_t shbuf[];
    half* sh_input_tile = reinterpret_cast<half*>(shbuf);
    int* sh_idx = reinterpret_cast<int*>(shbuf + L * 2);
    half* costs_a = reinterpret_cast<half*>(shbuf + L * 2 + 128);
    half* costs_b = dbl ? costs_a + edges : costs_a;

    for (int i = thread; i < L; i += NT)
        sh_input_tile[i] = __float2half_rn(input_tile[i]);

    // Decoded candidate values: values[j][p][c] is the pair (S*g + 2p, S*g + 2p + 1) reached
    // through branch c, for group g = thread + j * NT
    half2 values[NG][S / 2][S];
    #pragma unroll
    for (int j = 0; j < NG; ++j)
    {
        const int g = thread + j * NT;
        #pragma unroll
        for (int p = 0; p < S / 2; ++p)
        {
            #pragma unroll
            for (int c = 0; c < S; ++c)
            {
                const int state = (c << Kr) | (S * g + 2 * p);
                values[j][p][c] = decode_3inst_2<cb>(state, state + 1);
            }
        }
    }
    __syncthreads();

    auto ring = [&](int i, int roll)
    {
        int ri = i + roll;
        if (ri >= L) ri -= L;
        return ri;
    };

    // Returns the buffer that holds the last step's costs
    auto forward = [&](int roll, int pre_state) -> half*
    {
        half* cur = costs_a;
        half* nxt = costs_b;
        for (int i = 0; i < L; ++i)
        {
            const int ri = ring(i, roll);
            const half2 w = __half2half2(sh_input_tile[ri]);
            const bool write_hist = pre_state >= 0 || ri < L / 2;
            half2 staged[dbl ? 1 : NG][S / 2];

            #pragma unroll
            for (int j = 0; j < NG; ++j)
            {
                const int g = thread + j * NT;
                half cost[S];
                if (i > 0)
                {
                    #pragma unroll
                    for (int c = 0; c < S; ++c)
                        cost[c] = cur[(c << pred_shift) | g];
                }

                uint32_t key_lo[S / 2], key_hi[S / 2];
                #pragma unroll
                for (int p = 0; p < S / 2; ++p)
                {
                    #pragma unroll
                    for (int c = 0; c < S; ++c)
                    {
                        const half2 d = __hsub2(values[j][p][c], w);
                        half2 err;
                        if (i == 0)
                        {
                            err = __hmul2(d, d);
                            if (pre_state >= 0 && ((c << pred_shift) | g) != pre_state)
                                err = __half2half2(H_INF);
                        }
                        else
                            err = __hfma2(d, d, __half2half2(cost[c]));
                        const uint32_t e = __builtin_bit_cast(uint32_t, err);
                        const uint32_t kl = (e << 16) | (uint32_t) c;
                        const uint32_t kh = (e & 0xffff0000u) | (uint32_t) c;
                        key_lo[p] = c == 0 ? kl : min(key_lo[p], kl);
                        key_hi[p] = c == 0 ? kh : min(key_hi[p], kh);
                    }
                }

                half2 best[S / 2];
                hword h = 0;
                #pragma unroll
                for (int p = 0; p < S / 2; ++p)
                {
                    const uint32_t packed = (key_lo[p] >> 16) | (key_hi[p] & 0xffff0000u);
                    best[p] = __builtin_bit_cast(half2, packed);
                    h |= (hword) (key_lo[p] & (S - 1)) << (K * (2 * p));
                    h |= (hword) (key_hi[p] & (S - 1)) << (K * (2 * p + 1));
                }
                if (write_hist)
                    hist[ri * groups + g] = h;

                if constexpr (dbl)
                {
                    #pragma unroll
                    for (int p = 0; p < S / 2; ++p)
                        reinterpret_cast<half2*>(nxt)[(S * g) / 2 + p] = best[p];
                }
                else
                {
                    #pragma unroll
                    for (int p = 0; p < S / 2; ++p)
                        staged[dbl ? 0 : j][p] = best[p];
                }
            }

            if constexpr (!dbl)
            {
                __syncthreads();
                #pragma unroll
                for (int j = 0; j < NG; ++j)
                {
                    const int g = thread + j * NT;
                    #pragma unroll
                    for (int p = 0; p < S / 2; ++p)
                        reinterpret_cast<half2*>(cur)[(S * g) / 2 + p] = staged[j][p];
                }
            }
            __syncthreads();
            if constexpr (dbl)
            {
                half* t = cur;
                cur = nxt;
                nxt = t;
            }
        }
        return cur;
    };

    // Same ranking as quantize_tiles_kernel's argmin_cost (block-size independent tie order)
    auto argmin_cost = [&](const half* costs)
    {
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges; e += NT)
        {
            unsigned v = e & 1023;
            unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
            unsigned key = ((uint32_t) __half_as_ushort(costs[e]) << 16) | rank;
            best = min(best, key);
        }
        best = qt_warp_min(best);
        if ((thread & 31) == 0)
            ((uint32_t*) sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32)
        {
            best = thread < NW ? ((uint32_t*) sh_idx)[thread] : 0x7c00ffffu;
            best = qt_warp_min(best);
        }
        unsigned rank = best & 65535;
        unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
        return best >= 0x7c000000u ? 0 : (int) (((rank & 31) << 10) | v);
    };

    // Traceback is a single-thread serial dependency chain (each step's read address depends on
    // the previous step's result), but WHICH rows it visits is static: roll==0 walks rows
    // L-1..0 (write pass), roll==L/2 walks rows L/2-1..0 (probe pass, ring(i,L/2) hits ri==0 at
    // i==L/2 and the original loop broke there). Both are a contiguous descending physical-row
    // range with no wrap, so instead of thread 0 issuing L serialized, latency-bound global loads
    // (one hword per row, no thread-level parallelism to hide the latency), all NT threads
    // cooperatively stage 8 rows at a time into the LDS freed by argmin_cost (costs_a/costs_b, or
    // just costs_a when K=2 is single-buffered: 8 * groups * sizeof(hword) == that region's size
    // for every K here, so it always fits exactly). Thread 0 then walks those 8 rows out of LDS
    // (cheap) before the next chunk overwrites the cache. Same reads, same values, same order;
    // only the memory path changes.
    auto backward = [&](int roll, bool write, int edge)
    {
        hword* cache = reinterpret_cast<hword*>(costs_a);
        const int start_ri = write ? L - 1 : L / 2 - 1;  // matches roll==0 / roll==L/2 call sites
        for (int base = (start_ri / 8) * 8; base >= 0; base -= 8)
        {
            for (int idx = thread; idx < 8 * groups; idx += NT)
                cache[idx] = hist[(int64_t) base * groups + idx];
            __syncthreads();
            if (thread == 0)
            {
                for (int r = 7; r >= 0; --r)
                {
                    const int ri = base + r;
                    const hword h = cache[r * groups + (edge >> K)];
                    const int c = (int) (h >> (K * (edge & (S - 1)))) & (S - 1);
                    const int prev_edge = (c << pred_shift) | (edge >> K);
                    const int encoded = (prev_edge << K) | edge;
                    edge = prev_edge;
                    if (write)
                    {
                        output_indices[ri] = (uint16_t) encoded;
                        output_tile[ri] = __half2float(decode_3inst<cb>(encoded));
                    }
                }
            }
            __syncthreads();
        }
        if (thread == 0) sh_idx[0] = edge;
        __syncthreads();
        return sh_idx[0];
    };

    const half* last = forward(L / 2, -1);
    const int end_state = backward(L / 2, false, argmin_cost(last));
    forward(0, end_state);
    backward(0, true, end_state);
}
