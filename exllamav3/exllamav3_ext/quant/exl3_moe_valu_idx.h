#pragma once
// Index arithmetic of the VALU grouped-MoE decode matvec (EXL3_MOE_VALU). Pure C++: shared by the kernel
// (exl3_moe_valu.cuh) and by the CPU bounds checker (tests/moe_valu_bounds.cpp). Every global, LDS and output index the
// kernel computes goes through these helpers. Weights are in the engine's native trellis layout:
// u32 index = kslice * (ntiles * TW) + tile * TW + word, TW = 8 * bits.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define MV_FN constexpr __host__ __device__ __forceinline__
#else
#define MV_FN constexpr inline
#endif
namespace mv {
constexpr int NCH = 16;      // k-chunks: the engine's warp k-split (WK = 16, EXL3_MOE_CFG = 0)
constexpr int MAXPASS = 4;   // rows per pass; more rows run as extra passes over the same expert
MV_FN int tw(int bits)                 { return 8 * bits; }
MV_FN long sst(int bits, int ntiles)   { return (long) ntiles * tw(bits); }                       // u32 per k-slice row
MV_FN int chunk(int ks)                { return (ks + NCH - 1) / NCH; }                           // k-slices per chunk
MV_FN int myn(int ks, int c)           { int ch = chunk(ks); int r = ks - c * ch; r = r < ch ? r : ch; return r > 0 ? r : 0; }
// u32 offset in the expert region of the lane's own words (bits of them, contiguous) and of its wrap word
MV_FN long own(int bits, int ntiles, int tile, int g8, int s)  { return (long) s * sst(bits, ntiles) + (long) tile * tw(bits) + bits * g8; }
MV_FN long wrap(int bits, int ntiles, int tile, int g8, int s) { return (long) s * sst(bits, ntiles) + (long) tile * tw(bits) + ((bits * g8 - 1) & (tw(bits) - 1)); }
// A (fp16, row-major, K halves per row), as u32: row r, k-slice s covers 8 u32
MV_FN long a_idx(int r, int k, int s)  { return (long) r * (k / 2) + 8 * s; }
// wave-private reduction scratch [waves][4 chunk lanes][16 cols][4 rows] floats
MV_FN int red_idx(int wave, int cq, int col, int row) { return ((wave * 4 + cq) * 16 + col) * 4 + row; }
MV_FN int red_size(int waves)          { return waves * 4 * 16 * 4; }
MV_FN long c_idx(int row, int n, int tile, int col16) { return (long) row * n + (long) tile * 16 + col16; }
MV_FN int tile_of(int bx, int wave, int W) { return bx * W + wave; }
MV_FN int grid_x(int ntiles, int W)    { return (ntiles + W - 1) / W; }
MV_FN int pass_rows(int rows, int r0)  { int n = rows - r0; return n < MAXPASS ? n : MAXPASS; }
}
