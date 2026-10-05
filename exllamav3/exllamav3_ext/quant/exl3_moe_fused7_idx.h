#pragma once
// core7: index plan of the phases rewritten in exl3_moe_fused7.cuh. Pure C++, shared by the kernel and by the CPU bounds checker (bounds7.cpp).
#include "exl3_moe_fused_idx.h"
namespace mf7 {
using namespace mf;
// dots phase: (h, j) items, h-major: item i = h * (MR + 1) + j, j = MR is the sum-of-squares row. Team t (128 threads) of NT owns [dots_lo(t), dots_lo(t + 1)).
constexpr int DOTS_MAXIT = 24;    // upper bound of items per team (checked at launch and in the CPU proof)
constexpr int DOTS_GRP = 4;       // items of the same h processed together (loads hoisted, x rows shared)
MF_FN int dots_lo(long long total, int nt, int t)                   { return (int) (total * t / nt); }
// LDS scratch of one team: [item slot][row][warp] floats, inside Smem.u.g.red (RED_SIZE floats)
MF_FN int dots_scratch(int team, int m, int r, int warp)            { return ((team * DOTS_MAXIT + m) * MAXR + r) * 4 + warp; }
constexpr int DOTS_SCRATCH_MAX = 4 * DOTS_MAXIT * MAXR * 4;
static_assert(DOTS_SCRATCH_MAX <= RED_SIZE, "dots scratch fits in red");
}
