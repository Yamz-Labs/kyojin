// Minimal CPU emulation of the HIP kernel model for tests/dsv41 (NOT a HIP implementation): every thread of a block is a real
// std::thread, blocks run one after the other, __syncthreads is a block barrier, __shfl_* exchange through shared slots with a
// per-warp (32 lanes) barrier, __shared__ is a function-local static. Enough to run dsv41_hip.hip unmodified on the CPU.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <barrier>
#include <thread>
#include <vector>
#include <memory>
#include <utility>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static

struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
struct float4 { float x, y, z, w; };
static inline float4 make_float4(float a, float b, float c, float d) { return float4{a, b, c, d}; }

extern thread_local dim3 threadIdx, blockIdx;
extern dim3 blockDim, gridDim;

typedef _Float16 __half;
static inline __half __ushort_as_half(unsigned short u) { __half h; std::memcpy(&h, &u, 2); return h; }
static inline unsigned short __half_as_ushort(__half h) { unsigned short u; std::memcpy(&u, &h, 2); return u; }
static inline float __half2float(__half h) { return (float) h; }
static inline __half __float2half_rn(float v) { return (__half) v; }

struct EmuBlock
{
    std::unique_ptr<std::barrier<>> block;
    std::vector<std::unique_ptr<std::barrier<>>> warp;
    unsigned xchg[32][32];
};
extern EmuBlock* g_emu;
extern thread_local unsigned g_tid;

static inline void __syncthreads() { g_emu->block->arrive_and_wait(); }

template <class T>
static inline T __shfl_xor(T v, int o, int w = 32)
{
    unsigned wid = g_tid / 32, lane = g_tid % 32;
    unsigned u; std::memcpy(&u, &v, 4);
    g_emu->xchg[wid][lane] = u;
    g_emu->warp[wid]->arrive_and_wait();
    unsigned r = g_emu->xchg[wid][lane ^ o];
    g_emu->warp[wid]->arrive_and_wait();
    T out; std::memcpy(&out, &r, 4);
    return out;
}
template <class T>
static inline T __shfl(T v, int src, int w = 32)
{
    unsigned wid = g_tid / 32, lane = g_tid % 32;
    unsigned u; std::memcpy(&u, &v, 4);
    g_emu->xchg[wid][lane] = u;
    g_emu->warp[wid]->arrive_and_wait();
    unsigned r = g_emu->xchg[wid][src];
    g_emu->warp[wid]->arrive_and_wait();
    T out; std::memcpy(&out, &r, 4);
    return out;
}

// run kernel k over grid x block with the argument pack of hipModuleLaunchKernel (array of pointers to the values)
template <class... A, size_t... I>
static void emu_call(void (*k)(A...), void** p, std::index_sequence<I...>) { k(*reinterpret_cast<A*>(p[I])...); }

template <class... A>
static void emu_launch(void (*k)(A...), int gx, int gy, int gz, int bx, void** params)
{
    gridDim = dim3(gx, gy, gz);
    blockDim = dim3(bx, 1, 1);
    for (int z = 0; z < gz; z++)
        for (int y = 0; y < gy; y++)
            for (int x = 0; x < gx; x++)
            {
                EmuBlock blk;
                blk.block = std::make_unique<std::barrier<>>(bx);
                for (int w = 0; w < (bx + 31) / 32; w++) blk.warp.push_back(std::make_unique<std::barrier<>>(std::min(32, bx - w * 32)));
                g_emu = &blk;
                std::vector<std::thread> th;
                for (int t = 0; t < bx; t++)
                    th.emplace_back([&, t] {
                        g_tid = t; threadIdx = dim3(t, 0, 0); blockIdx = dim3(x, y, z);
                        emu_call(k, params, std::index_sequence_for<A...>{});
                    });
                for (auto& t : th) t.join();
            }
}
