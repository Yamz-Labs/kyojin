// CPU proof of the K7 (KB2 = 14) tile format of the batch-1 / R-row decode kernels (exl3_dec.cu, Fmt<KB2>). Re-implements Fmt::wstart and window() word for word and compares every
// 16 bit window of a 256-weight trellis tile with the engine's dq2x2<7> arithmetic (exl3_dq.cuh, cyclic over the 56 tile words) for random tiles. Also checks the word indices stay
// inside the tile and that the tile is 56 words = 14 vectors of 4 words. Run for K5 / K6 too, so the check is anchored on formats that are already in production.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails < 20) { printf("FAIL %s: ", #c); printf(__VA_ARGS__); printf("\n"); } ++fails; } } while (0)
template <int KB2> struct Fmt
{
    static constexpr int KA = KB2 / 2, HALF = KB2 & 1, BPB = 16 * KA + 8 * HALF, L = 16 * BPB, WORDS = L / 32, VEC = WORDS / 4;
    static constexpr int send(int p) { return (p >> 4) * BPB + ((p & 15) + 1) * KA + (HALF ? ((p & 15) + 1) / 2 : 0); }
    static constexpr int wstart(int p) { return ((send(p) - 16) % L + L) % L; }
};
static uint32_t ubfe(uint32_t x, int off, int w) { return (x >> off) & ((1u << w) - 1u); }
static uint32_t alignbit(uint32_t hi, uint32_t lo, int s) { return (uint32_t) ((((uint64_t) hi << 32) | lo) >> s); }
static uint32_t fshift_eng(uint32_t b, uint32_t a, int shift) { return (uint32_t) ((((uint64_t) a << 32) | (uint64_t) b) >> shift); }
template <int KB2> static void run()
{
    using F = Fmt<KB2>;
    const int bits = KB2 / 2, tw = bits * 256 / 32;
    CHECK(F::WORDS == tw && F::WORDS % 4 == 0 && F::VEC * 4 == F::WORDS, "format KB2=%d", KB2);
    srand(11 + KB2);
    for (int trial = 0; trial < 50; ++trial)
    {
        uint32_t w[64];
        for (int i = 0; i < tw; ++i) w[i] = ((uint32_t) rand() << 16) ^ (uint32_t) rand() ^ ((uint32_t) rand() << 3);
        for (int p = 0; p < 256; ++p)
        {
            const int s = F::wstart(p), i = s >> 5, o = s & 31;
            CHECK(i >= 0 && i < F::WORDS && (i + 1) % F::WORDS < F::WORDS, "index KB2=%d p=%d", KB2, p);
            const uint32_t got = (o <= 16 ? ubfe(w[i], 16 - o, 16) : alignbit(w[i], w[(i + 1) % F::WORDS], 48 - o)) & 0xffffu;
            // engine: value p of the tile; dq2x2 / dq4 numbering: t_offset + j with end bit (t_offset + j + 257) * bits
            const int e = p;   // position p of the ring is weight p of the tile
            const int b2 = (e + 257) * bits;                 // end of the window
            const int b0 = b2 - 16;
            const int i0 = b0 / 32, i2 = (b2 - 1) / 32, s2 = (i2 + 1) * 32 - b2;
            const uint32_t a = w[i0 % tw], b = w[i2 % tw];
            const uint32_t want = fshift_eng(b, a, s2) & 0xffffu;
            CHECK(got == want, "window mismatch KB2=%d p=%d got=%x want=%x", KB2, p, got, want);
        }
    }
}
int main()
{
    run<10>(); run<12>(); run<14>();
    printf(fails ? "DEC K7 CHECK FAILED (%d)\n" : "DEC K7 CHECK OK\n", fails);
    return fails ? 1 : 0;
}
