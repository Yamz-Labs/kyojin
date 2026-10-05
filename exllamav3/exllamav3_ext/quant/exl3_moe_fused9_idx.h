#pragma once
// core9: index plan of the cache "touch" (prefetch by read-only loads) added in exl3_moe_fused9.cuh. Pure C++, shared by the kernel and by the CPU bounds checker (bounds9.cpp).
// A touch reads one dword per lane (global_load_b32 into a sink register, result never used): lane l of chunk c reads the dword at byte c * 2048 + 64 l, so a wave step covers
// 2 KB (every 64-byte granule once). Ranges are multiples of 2048 B for the real shapes (the guard handles any tail).
// The waves of the grid (gw = block * 16 + wave, NW = G * 16) share a range round-robin: wave gw reads chunks gw, gw + NW, ...
#include "exl3_moe_fused7_idx.h"
namespace mf9 {
using namespace mf;
constexpr int TOUCH_CHUNK = 2048;     // bytes covered per wave per step
constexpr int TOUCH_LANE = 64;        // byte stride between lanes
constexpr int TOUCH_W = 4;            // bytes loaded per lane
MF_FN long touch_nch(long bytes)                       { return (bytes + TOUCH_CHUNK - 1) / TOUCH_CHUNK; }
MF_FN long touch_off(long chunk, int lane)             { return chunk * TOUCH_CHUNK + (long) lane * TOUCH_LANE; }
MF_FN bool touch_in(long bytes, long chunk, int lane)  { return touch_off(chunk, lane) + TOUCH_W <= bytes; }
// the address a lane loads: its own dword, or byte 0 of the range for a lane past the end (no divergence: every lane of the wave issues the load)
MF_FN long touch_addr(long bytes, long chunk, int lane) { return touch_in(bytes, chunk, lane) ? touch_off(chunk, lane) : 0; }
// byte sizes of the ranges the touches read (the same tensors the kernel already reads in full)
template <class S> MF_FN long upt_bytes()              { return (long) Dm<S>::H * Dm<S>::D4 * Dm<S>::LR * 4; }
template <class S> MF_FN long router_bytes()           { return (long) Dm<S>::NEXP * Dm<S>::D * 2; }
template <class S> MF_FN long sh_gu_bytes()            { return 4L * mat_words(Dm<S>::SB, Dm<S>::KSG, Dm<S>::TPP); }
template <class S> MF_FN long sh_dn_bytes()            { return 4L * mat_words(Dm<S>::SB, Dm<S>::KSD, Dm<S>::TPU); }
template <class S> MF_FN long rt_dn_bytes()            { return 4L * mat_words(Dm<S>::RB, Dm<S>::KSD, Dm<S>::TPU); }
constexpr int TOUCH_DOWN_CAP = 32;    // loads per wave for the down-weight touch (stays under the 63-load vmcnt window)
}
