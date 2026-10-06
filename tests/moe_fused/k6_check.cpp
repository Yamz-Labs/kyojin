// CPU proof for the K6 (and K5 reference) lane word sets of the fused MoE half-layer. Includes the REAL index header of the kernel.
// For every lane of a trellis tile (32 lanes x 2 groups) it compares the kernel's word extraction (mf::kn_words on the lane set {wrap, own0..}) with the engine's
// dq4<bits> arithmetic (exl3_dq.cuh, copied formula by formula, cyclic over the tile), checks every index is inside the lane set / the tile, the 64 bit shifts stay < 64,
// the group stays inside two words, the lane sets of the 8 g8 cover the tile once, and the 8 byte alignment of the uint2 loads.
#include "exl3_moe_fused_idx.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails < 20) { printf("FAIL %s: ", #c); printf(__VA_ARGS__); printf("\n"); } ++fails; } } while (0)
static uint32_t fshift_eng(uint32_t b, uint32_t a, int shift) { uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b; return (uint32_t) (merged >> shift); }
int main(int argc, char** argv)
{
    int bitsv[] = {5, 6};
    for (int bits : bitsv)
    {
        const int tw = mf::tw(bits);
        srand(7 + bits);
        for (int trial = 0; trial < 20; ++trial)
        {
            uint32_t tile[64];
            for (int i = 0; i < tw; ++i) tile[i] = ((uint32_t) rand() << 16) ^ (uint32_t) rand() ^ ((uint32_t) rand() << 3);
            int cover[64] = {0};
            for (int g8 = 0; g8 < 8; ++g8)
            {
                const long own = mf::own_i(bits, g8, 0, 0), wrp = mf::wrp_i(bits, g8, 0, 0);
                CHECK(own >= 0 && own + bits <= tw, "own range bits=%d g8=%d", bits, g8);
                CHECK(wrp >= 0 && wrp < tw, "wrap range");
                CHECK(wrp == (own + tw - 1) % tw, "wrap is the word before own (cyclic)");
                CHECK(((own * 4) % 8) == 0 || bits % 2 == 1, "uint2 alignment of own words, bits=%d g8=%d", bits, g8);
                for (int k = 0; k < bits; ++k) cover[own + k]++;
                uint32_t w[7] = {0, 0, 0, 0, 0, 0, 0};
                w[0] = tile[wrp];
                for (int k = 0; k < bits; ++k) w[1 + k] = tile[own + k];
                for (int t = 0; t < 4; ++t)
                {
                    unsigned v[8];
                    mf::kn_words(bits, w, t, v);
                    const int L = 4 * g8 + t;
                    for (int g = 0; g < 2; ++g)
                    {
                        // engine dq4<bits>(ptr, t_offset = 8 L + 4 g, frag)
                        const int t_offset = 8 * L + 4 * g;
                        int b0 = (t_offset + 257) * bits - 16, b1 = b0 + 3 * bits, b2 = b1 + 16;
                        int i0 = b0 / 32, i2 = (b2 - 1) / 32, s2 = (i2 + 1) * 32 - b2;
                        uint32_t a = tile[i0 % tw], b = tile[i2 % tw];
                        uint32_t ew[4] = {fshift_eng(b, a, s2 + bits * 3) & 0xffff, fshift_eng(b, a, s2 + bits * 2) & 0xffff, fshift_eng(b, a, s2 + bits) & 0xffff, fshift_eng(b, a, s2) & 0xffff};
                        for (int j = 0; j < 4; ++j) CHECK(v[4 * g + j] == ew[j], "word mismatch bits=%d L=%d g=%d j=%d", bits, L, g, j);
                        // indices
                        const int l0 = mf::kn_i0(bits, t, g), l2 = mf::kn_i2(bits, t, g), ls = mf::kn_s2(bits, t, g);
                        CHECK(l0 >= 0 && l0 <= bits && l2 >= 0 && l2 <= bits, "local index range bits=%d L=%d g=%d (%d %d)", bits, L, g, l0, l2);
                        CHECK(l2 - l0 == 0 || l2 - l0 == 1, "group inside two words");
                        CHECK(s2 == ls && (l0 == l2) == (i0 == i2), "engine shift / word match");
                        CHECK(ls >= 0 && ls + 3 * bits < 64, "shift < 64");
                        const int gl0 = (l0 == 0 ? (int) wrp : (int) own + l0 - 1), gl2 = (l2 == 0 ? (int) wrp : (int) own + l2 - 1);
                        CHECK(gl0 == i0 % tw && gl2 == i2 % tw, "local words are the engine's words bits=%d L=%d g=%d", bits, L, g);
                    }
                }
            }
            for (int i = 0; i < tw; ++i) CHECK(cover[i] == 1, "own words cover the tile once bits=%d word=%d cover=%d", bits, i, cover[i]);
        }
        // matrix level: for the real Qwen shapes every tile of every k-slice of every expert matrix is inside its matrix (native layout)
        const int shapes[][2] = {{160, 40}, {40, 160}};   // (KS, NT): gate/up K2560 N640 -> KS 160 NT 40 ; down K640 N2560 -> KS 40 NT 160
        for (auto& sh : shapes)
        {
            const int ks = sh[0], nt = sh[1];
            for (int nat = 0; nat < 2; ++nat)
            {
                if (!nat && nt % 2) continue;
                const long words = mf::mat_words(bits, ks, nt);
                for (int tile = 0; tile < nt; ++tile)
                    for (int s = 0; s < ks; ++s)
                        for (int g8 = 0; g8 < 8; ++g8)
                        {
                            const long base = mf::tile_off(nat, bits, ks, nt, tile), ss = mf::tile_ss(nat, bits, nt);
                            const long own = base + mf::own_i(bits, g8, s, ss), wrp = base + mf::wrp_i(bits, g8, s, ss);
                            CHECK(own >= 0 && own + bits <= words && wrp >= 0 && wrp < words, "matrix range bits=%d nat=%d tile=%d s=%d", bits, nat, tile, s);
                            CHECK(bits != 6 || own % 2 == 0, "uint2 loads 8-byte aligned in the matrix (even word offset) bits=%d", bits);
                        }
            }
        }
    }
    printf(fails ? "K6 CHECK FAILED (%d)\n" : "K6 CHECK OK\n", fails);
    return fails ? 1 : 0;
}
