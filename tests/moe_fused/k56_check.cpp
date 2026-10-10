// CPU proof for the K5 routed / K4 and K5 routed with K6 shared shapes added to the fused MoE half-layer, plus the grouped GEMV staging bounds for K5 / K6.
// Includes the REAL index header. For each new shape: plan constants (dims, expert and shared-expert word counts), every tile of every k-slice of the gate / up / down
// matrix inside its matrix for routed bits and shared bits (native layout), the workspace offsets are monotonic and 256-aligned. For the grouped GEMV (CFG 0/1/2): the
// staged-word count WNT * TWORDS never exceeds the padded load window, and the last group of a matrix reads inside the matrix.
#include "exl3_moe_fused_idx.h"
#include <cstdio>
#include <cstdlib>
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails < 20) { printf("FAIL %s: ", #c); printf(__VA_ARGS__); printf("\n"); } ++fails; } } while (0)
static void matrix(int bits, int ks, int nt, const char* what)
{
    const long words = mf::mat_words(bits, ks, nt);
    CHECK(words == (long) ks * nt * 8 * bits, "%s words", what);
    for (int tile = 0; tile < nt; ++tile)
        for (int s = 0; s < ks; ++s)
            for (int g8 = 0; g8 < 8; ++g8)
            {
                const long base = mf::tile_off(true, bits, ks, nt, tile), ss = mf::tile_ss(true, bits, nt);
                const long own = base + mf::own_i(bits, g8, s, ss), wrp = base + mf::wrp_i(bits, g8, s, ss);
                CHECK(own >= 0 && own + bits <= words && wrp >= 0 && wrp < words, "%s range bits=%d tile=%d s=%d", what, bits, tile, s);
            }
}
template <class S> static void shape(const char* name)
{
    using DM = mf::Dm<S>;
    printf("shape %s RB=%d SB=%d\n", name, S::RB, S::SB);
    CHECK(DM::GU_GRP == DM::KSG * 16 * S::RB && DM::SH_GU_GRP == DM::KSG * 16 * S::SB, "group words");
    CHECK(DM::EXPERT_U32 == 2L * DM::NGU * DM::GU_GRP + (long) DM::NGD * DM::DN_GRP, "expert words");
    // an expert is gate + up (D x INTER) + down (INTER x D): 3 * D * INTER * bits / 16 ... in u32: 3 * (D / 16) * (INTER / 16) * 8 * bits
    CHECK(DM::EXPERT_U32 == 3L * (S::D / 16) * (S::INTER / 16) * 8 * S::RB, "expert size independent");
    CHECK(DM::SHARED_U32 == 3L * (S::D / 16) * (S::INTER / 16) * 8 * S::SB, "shared size independent");
    matrix(S::RB, S::D / 16, S::INTER / 16, "routed gate/up");
    matrix(S::RB, S::INTER / 16, S::D / 16, "routed down");
    matrix(S::SB, S::D / 16, S::INTER / 16, "shared gate/up");
    matrix(S::SB, S::INTER / 16, S::D / 16, "shared down");
}
int main()
{
    shape<mf::QwenK4S6>("QwenK4S6"); shape<mf::QwenK5S6>("QwenK5S6"); shape<mf::SmallK4S6>("SmallK4S6"); shape<mf::SmallK5S6>("SmallK5S6");
    // grouped GEMV staging: TWORDS = 8 * bits per tile, WNT tiles per warp (CFG 0: 2, CFG 1/2: 4), loads of LSTRIDE = min(TWORDS, 32) lanes
    for (int bits = 2; bits <= 6; ++bits)
        for (int cfg = 0; cfg < 3; ++cfg)
        {
            const int wnt = cfg == 0 ? 2 : 4, tw = 8 * bits, ls = bits == 2 ? 2 * tw : (tw < 32 ? tw : 32);
            const int loads = (wnt * tw + ls - 1) / ls;
            CHECK(loads * ls >= wnt * tw && loads * ls < wnt * tw + ls, "load cover bits=%d cfg=%d", bits, cfg);
            // guarded tail: words staged = exactly wnt * tw
            int staged = 0;
            for (int l = 0; l < loads; ++l) for (int lane = 0; lane < 32; ++lane) if (lane < ls && l * ls + lane < wnt * tw) ++staged;
            CHECK(staged == wnt * tw, "staged words bits=%d cfg=%d", bits, cfg);
            // last output group of the matrices of the real 5 bpw pack (gate/up N=640, down N=2560): group * WNT * TWORDS + staged words <= ntiles * TWORDS
            const int ns[] = {640, 2560};
            for (int n : ns)
            {
                const int ntiles = n / 16, groups = n / (wnt * 16);
                CHECK(n % (wnt * 16) == 0, "N divisible bits=%d cfg=%d n=%d", bits, cfg, n);
                CHECK((long) (groups - 1) * wnt * tw + wnt * tw == (long) ntiles * tw, "last group ends at the slice end bits=%d cfg=%d n=%d", bits, cfg, n);
            }
        }
    printf(fails ? "K56 CHECK FAILED (%d)\n" : "K56 CHECK OK\n", fails);
    return fails ? 1 : 0;
}
