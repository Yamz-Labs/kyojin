#pragma once
// Plan constants and index arithmetic of the fused MoE half-layer (EXL3_MOE_FUSED). Pure C++: shared by the kernel
// (exl3_moe_fused.cuh) and by the CPU bounds checker (tests/moe_fused/moe_fused_bounds.cpp). Every global-memory and LDS index the kernel
// computes in a phase goes through these helpers.
//
// Pack format (repacked at load, `moe_fused.py`): one 32-column group = 2 adjacent 16-column tiles of the engine's trellis, k-slice major:
//   u32 index in the group = kslice * (2 * 8 * bits) + tt * (8 * bits) + word,   tt = tile & 1,  bits = K of the projection (2, 3, 4).
// A group holds KS = K_dim/16 k-slices. Per expert: [gate groups INTER/32][up groups INTER/32][down groups D/32].
// Experts are indexed by expert id; the shared expert has the same layout with its own bits (SB) and its own base pointer.
// Since core5 the kernel reaches every projection matrix through a pointer table (p.wt[3 * unit + proj]) and a layout flag:
//   native (the engine's own trellis tensor (K/16, N/16, 16*bits) int16, no copy): tile t, k-slice s at u32 (s * NT + t) * tw,  stride between k-slices NT * tw
//   repack (core4 layout above, one repacked copy): tile t at (t / 2) * KS * 2 * tw + (t & 1) * tw,  stride between k-slices 2 * tw
// tw = 8 * bits u32 per tile. A lane's words are own = bits * g8 (bits words) and the wrap word.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define MF_FN constexpr __host__ __device__ __forceinline__
#else
#define MF_FN constexpr inline
#endif
namespace mf {

constexpr int MAXR = 4;       // rows per launch (R = 1..4)
constexpr int THREADS = 512;  // 16 waves of 32

// A shape is a struct with static constexpr: D hidden, H hyper-connection streams, LR hc rank, NEXP experts, TOPK, INTER expert width,
// RB routed expert bits, SB shared expert bits.
struct QwenShape { static constexpr int D = 2560, H = 4, LR = 320, NEXP = 512, TOPK = 10, INTER = 640, RB = 2, SB = 4; };
// small synthetic shape for the first GPU run (random weights, canary-checked): same code, 2 K2 / K4 mixes, 64 experts
struct SmallTestShape { static constexpr int D = 256, H = 4, LR = 32, NEXP = 64, TOPK = 4, INTER = 128, RB = 2, SB = 4; };

// K3 synthetic shape (routed and shared experts 3 bit) for the whole-kernel K3 run
struct SmallK3Shape { static constexpr int D = 256, H = 4, LR = 32, NEXP = 64, TOPK = 4, INTER = 128, RB = 3, SB = 3; };

// the Qwen 3.8 Flash shape with the routed / shared bit widths of our packs (routed K3 / K4, shared K4 / K5), and small twins for the GPU parity run
struct QwenK3S4 { static constexpr int D = 2560, H = 4, LR = 320, NEXP = 512, TOPK = 10, INTER = 640, RB = 3, SB = 4; };
struct QwenK3S5 { static constexpr int D = 2560, H = 4, LR = 320, NEXP = 512, TOPK = 10, INTER = 640, RB = 3, SB = 5; };
struct QwenK4S4 { static constexpr int D = 2560, H = 4, LR = 320, NEXP = 512, TOPK = 10, INTER = 640, RB = 4, SB = 4; };
struct QwenK4S5 { static constexpr int D = 2560, H = 4, LR = 320, NEXP = 512, TOPK = 10, INTER = 640, RB = 4, SB = 5; };
struct SmallK3S4 { static constexpr int D = 256, H = 4, LR = 32, NEXP = 64, TOPK = 4, INTER = 128, RB = 3, SB = 4; };
struct SmallK3S5 { static constexpr int D = 256, H = 4, LR = 32, NEXP = 64, TOPK = 4, INTER = 128, RB = 3, SB = 5; };
struct SmallK4S4 { static constexpr int D = 256, H = 4, LR = 32, NEXP = 64, TOPK = 4, INTER = 128, RB = 4, SB = 4; };
struct SmallK4S5 { static constexpr int D = 256, H = 4, LR = 32, NEXP = 64, TOPK = 4, INTER = 128, RB = 4, SB = 5; };

template <class S>
struct Dm
{
    static constexpr int D = S::D, H = S::H, LR = S::LR, NEXP = S::NEXP, TOPK = S::TOPK, INTER = S::INTER, RB = S::RB, SB = S::SB;
    static constexpr int MR = LR + H;                   // hc projection rows (rank + H gates)
    static constexpr int D4 = D / 4;
    static constexpr int KSG = D / 16;                  // k-slices of gate/up (K dim = D)
    static constexpr int KSD = INTER / 16;              // k-slices of down (K dim = INTER)
    static constexpr int NGU = INTER / 32;              // 32-column groups per gate or up projection
    static constexpr int NGD = D / 32;                  // groups of down
    static constexpr int TPP = INTER / 16;              // tiles per (unit, proj) of gate/up
    static constexpr int TPU = D / 16;                  // tiles per unit of down
    static constexpr int GU_GRP = KSG * 16 * RB;        // u32 per gate/up group, routed
    static constexpr int DN_GRP = KSD * 16 * RB;
    static constexpr int SH_GU_GRP = KSG * 16 * SB;     // shared
    static constexpr int SH_DN_GRP = KSD * 16 * SB;
    static constexpr long EXPERT_U32 = 2L * NGU * GU_GRP + (long) NGD * DN_GRP;
    static constexpr long SHARED_U32 = 2L * NGU * SH_GU_GRP + (long) NGD * SH_DN_GRP;
    static constexpr int SV_STRIDE = 3 * (D + INTER);   // halves per expert: gsuh gsvh usuh usvh dsuh dsvh
    static constexpr int SV_GSUH = 0, SV_GSVH = D, SV_USUH = D + INTER, SV_USVH = 2 * D + INTER, SV_DSUH = 2 * D + 2 * INTER, SV_DSVH = 2 * D + 3 * INTER;
    static constexpr int CMB_BLOCKS = D / 128;          // 128-column blocks of the combine / A prep
    static constexpr int INT_BLOCKS = INTER / 128;
    static constexpr int FIN_CHUNKS = D / 32;           // finalize chunk_cols = 32
    static constexpr int KEYS = NEXP / 32;              // top-k keys per lane

    static_assert(D % 128 == 0 && INTER % 128 == 0, "128-wide Hadamard blocks");
    static_assert(INTER % 32 == 0 && D % 32 == 0, "32-column groups");
    static_assert(NEXP % 32 == 0 && NEXP <= 1024, "top-k lanes");
    static_assert(TOPK >= 1 && TOPK <= 15, "combine uses wave <= TOPK, 16 waves");
    static_assert(MAXR * TOPK <= THREADS && MAXR * TOPK <= 40, "sort/group arrays");
    static_assert((RB >= 2 && RB <= 5) && (SB >= 2 && SB <= 5), "K2, K3, K4, K5");
    static_assert(H >= 1 && H <= 4, "dred/rmr scratch");
    static_assert(LR % 1 == 0 && LR <= 1024, "t scratch");
    // dynamic-free LDS: A + A2 for MAXR rows of the larger of D / INTER
    static constexpr int LDS_A_HALVES = 2 * MAXR * D;
};

// ---- load-time plan checks that do not depend on the kernel
MF_FN long expert_base(long expert_u32, int e)                     { return (long) e * expert_u32; }
// group base (u32 offset in an expert region) for gate/up: proj in {0,1}, group gi in [0, NGU)
template <class S> MF_FN long gu_group(int proj, int gi)           { return ((long) proj * Dm<S>::NGU + gi) * Dm<S>::GU_GRP; }
template <class S> MF_FN long dn_group(int gi)                     { return 2L * Dm<S>::NGU * Dm<S>::GU_GRP + (long) gi * Dm<S>::DN_GRP; }
template <class S> MF_FN long sh_gu_group(int proj, int gi)        { return ((long) proj * Dm<S>::NGU + gi) * Dm<S>::SH_GU_GRP; }
template <class S> MF_FN long sh_dn_group(int gi)                  { return 2L * Dm<S>::NGU * Dm<S>::SH_GU_GRP + (long) gi * Dm<S>::SH_DN_GRP; }

// ---- tile-level indices (same arithmetic as the engine's native layout with ntiles = 2, base at the group)
MF_FN int tw(int bits)                                              { return 8 * bits; }
MF_FN int chunk(int ks)                                             { return (ks + 15) / 16; }
MF_FN int myn(int ks, int c)                                        { int ch = chunk(ks); int r = ks - c * ch; r = r < ch ? r : ch; return r > 0 ? r : 0; }
// u32 offset of tile `tile` (k-slice 0) from the matrix base, and u32 stride between two k-slices of that tile; ks = K/16, nt = N/16
MF_FN long tile_off(bool nat, int bits, int ks, int nt, int tile)  { return nat ? (long) tile * tw(bits) : (long) (tile >> 1) * ks * 2 * tw(bits) + (long) (tile & 1) * tw(bits); }
MF_FN long tile_ss(bool nat, int bits, int nt)                      { return nat ? (long) nt * tw(bits) : 2L * tw(bits); }
MF_FN long mat_words(int bits, int ks, int nt)                      { return (long) ks * nt * tw(bits); }
// a lane's own words (bits of them, from g8) and its wrap word (the word before them, cyclic over the tile: any tw, K3 has tw = 24), relative to the tile base
// K5: lane (g8, t) is the engine's lane L = 4 g8 + t. Its two 4-value groups g (values 4g .. 4g + 3, engine dq4<5>) start at bit
// (8L + 4g + 257) * 5 - 16 = 160 g8 + 40 t + 20 g - 11 (mod 1280); relative to the first bit of the lane set's wrap word (global word 5 g8 - 1, cyclic) that is
// 40 t + 20 g + 21. Both words of the funnel shift are local indices 0..5 of {wrap, own0 .. own4}.
MF_FN int k5_rel(int t, int g)                                      { return 40 * t + 20 * g + 21; }
MF_FN int k5_i0(int t, int g)                                       { return k5_rel(t, g) / 32; }
MF_FN int k5_i2(int t, int g)                                       { return (k5_rel(t, g) + 30) / 32; }
MF_FN int k5_s2(int t, int g)                                       { return (k5_i2(t, g) + 1) * 32 - (k5_rel(t, g) + 31); }
MF_FN long own_i(int bits, int g8, int s, long ss)                  { return s * ss + (long) bits * g8; }
MF_FN long wrp_i(int bits, int g8, int s, long ss)                  { return s * ss + (bits * g8 > 0 ? bits * g8 - 1 : tw(bits) - 1); }
MF_FN long a_idx(int r, int k, int s)                               { return (long) r * (k / 2) + 8 * s; }  // u32 in the A staging buffer
MF_FN int red_idx(int wave, int cq, int col, int row)               { return ((wave * 4 + cq) * 16 + col) * 4 + row; }
constexpr int RED_SIZE = 16 * 4 * 16 * 4;                           // floats
MF_FN long c_idx(int row, int n, int tile, int col16)               { return (long) row * n + (long) tile * 16 + col16; }

// ---- phase indices (elements of the stated type)
// HC dots: fn int8 [MR][H][D], xin f32 [R][H][D], dots f32 [R][MR+1][H]
template <class S> MF_FN long fn_row(int j, int h)                  { return ((long) j * Dm<S>::H + h) * Dm<S>::D; }
template <class S> MF_FN long xin_row(int r, int h)                 { return ((long) r * Dm<S>::H + h) * Dm<S>::D; }
template <class S> MF_FN long dots_idx(int r, int j, int h)         { return ((long) r * (Dm<S>::MR + 1) + j) * Dm<S>::H + h; }
// finalize: upt int8 [H][D4][LR][4] read as u32 at ((h*D4 + c) * LR + ii)
template <class S> MF_FN long upt_u32(int h, int c, int ii)         { return ((long) h * Dm<S>::D4 + c) * Dm<S>::LR + ii; }
template <class S> MF_FN long upsc_idx(int h, int c)                { return (long) h * Dm<S>::D + 4L * c; }   // up_scale f32 / w_h half, 4 wide
// group scales (EXL3_GR_GS): the int8 codes keep their layout; the per-row scale vectors are replaced by group scales, and a kernel
// is in group mode when fn_scale.numel() != MR (host sets Params::gs). Weight = half(code * scale) as fp32, accumulated in the old order.
//   fn: groups of GS_FN = 128 consecutive k over the flattened (h, d) axis; fn_gs f32 [MR][H * D / 128]; c = 8-element block (int2 load index)
//   up: groups of GS_UP = 64 consecutive ranks per output column; up_gs f32 [H][D4][ceil(LR / 64)][4] (4 = the column quad)
constexpr int GS_FN = 128, GS_UP = 64;
template <class S> MF_FN bool gs_ok()                               { return S::D % GS_FN == 0; }
template <class S> MF_FN long fn_scale_n(bool gs)                   { return gs ? (long) Dm<S>::MR * (S::H * S::D / GS_FN) : (long) Dm<S>::MR; }
template <class S> MF_FN long up_scale_n(bool gs)                   { return gs ? (long) S::H * S::D * ((S::LR + GS_UP - 1) / GS_UP) : (long) S::H * S::D; }
template <class S> MF_FN long fngs_idx(int j, int h, int c8)        { return (long) j * (Dm<S>::H * Dm<S>::D / GS_FN) + (long) h * (Dm<S>::D / GS_FN) + (c8 >> 4); }
template <class S> MF_FN long upgs_idx(int h, int c, int ii)        { return (((long) h * Dm<S>::D4 + c) * ((Dm<S>::LR + GS_UP - 1) / GS_UP) + (ii >> 6)) * 4; }
// router
template <class S> MF_FN long router_row(int e)                    { return (long) e * Dm<S>::D; }
template <class S> MF_FN long scores_idx(int r, int e)              { return (long) r * Dm<S>::NEXP + e; }
template <class S> MF_FN long mixed_row(int r)                      { return (long) r * Dm<S>::D; }
// gate/up output (half): [proj][NAR][INTER]; down output (f32) [NAR][D]
template <class S> MF_FN long gu_out(int proj, int nar, int row0)   { return ((long) proj * nar + row0) * Dm<S>::INTER; }
template <class S> MF_FN long gu_read(int nar, int row, int blk, int which) { return ((long) which * nar + row) * Dm<S>::INTER + (long) blk * 128; }
template <class S> MF_FN long dn_row(int row)                       { return (long) row * Dm<S>::D; }
// A staging buffers (half): row-major, K halves per row
template <class S> MF_FN long a_row_gu(int row, int blk)            { return (long) row * Dm<S>::D + blk * 128; }
template <class S> MF_FN long a_row_dn(int row, int blk)            { return (long) row * Dm<S>::INTER + blk * 128; }
// combine
template <class S> MF_FN long xout_idx(int r, int h, int col)       { return ((long) r * Dm<S>::H + h) * Dm<S>::D + col; }

// ---- debug buffer (seldbg, int32): [0, 40) selected experts, [64, 104) weights, [STAMP_BASE, ...) per-block phase time stamps (MF_TIMING builds)
constexpr int SELDBG_INTS = 1024;
constexpr int STAMP_BASE = 256, STAMP_STRIDE = 16, STAMP_SLOTS = 12, STAMP_MAXB = 40;
MF_FN int stamp_idx(int bx, int k)                                  { return STAMP_BASE + bx * STAMP_STRIDE + k; }

// ---- top-k expert id sanitising: an id outside [0, NEXP) (all-NaN scores leave the sentinel) becomes 0, never a wild pointer
MF_FN int clamp_expert(int e, int nexp)                             { return (e >= 0 && e < nexp) ? e : 0; }

// ---- task decomposition
// ---- cost-weighted split of the gate/up and down tile lists: a tile of a segment whose group has `rows` rows costs seg_weight(rows) (3.5 VALU
// instructions per weight to decode + 0.5 per row), so the shared expert (R rows) and multi-row groups get fewer tiles per block.
// urows[u] = rows of unit u (the shared expert is the last unit); nu units x pu projections (2 gate/up, 1 down), tp tiles per segment, segments unit-major.
MF_FN int seg_weight(int rows)                                      { return 7 + rows; }
MF_FN long long split_total(const int* urows, int nu, int pu, int tp)
{
    long long w = 0;
    for (int u = 0; u < nu; ++u) w += (long long) pu * tp * seg_weight(urows[u]);
    return w;
}
// first tile index whose exclusive weight prefix is >= x, x in [0, W]; split_tile(0) = 0, split_tile(W) = nu * pu * tp. Block b owns [split(b W / G), split((b + 1) W / G)).
MF_FN int split_tile(const int* urows, int nu, int pu, int tp, long long x)
{
    long long acc = 0;
    for (int u = 0; u < nu; ++u)
    {
        const long long w = seg_weight(urows[u]), segw = w * tp;
        for (int pj = 0; pj < pu; ++pj)
        {
            if (x < acc + segw) { const long long t = (x - acc + w - 1) / w; return (u * pu + pj) * tp + (int) t; }
            acc += segw;
        }
    }
    return nu * pu * tp;
}
MF_FN int blk_lo(int bx, int total, int G)                          { return (int) ((long long) bx * total / G); }
MF_FN int blk_hi(int bx, int total, int G)                          { return (int) ((long long) (bx + 1) * total / G); }
}
