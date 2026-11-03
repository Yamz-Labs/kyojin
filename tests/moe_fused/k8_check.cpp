// CPU-only check of the actual fused word-extraction helper. No HIP runtime.
// Oracle 1 follows exl3_dq.cuh::dq4<8>; oracle 2 extracts each 16-bit word
// independently from the cyclic tile, without the fused lane-local helpers.
#include "exl3_moe_fused_idx.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <cstdlib>

static long checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "K8 CHECK FAILED: %s\n", what); std::exit(1); }
}

static void decode_tile(const std::array<uint32_t, 64>& tile) {
    constexpr int bits = 8;
    for (int g8 = 0; g8 < 8; ++g8) {
        uint32_t w[9];
        const long own = mf::own_i(bits, g8, 0, 0);
        const long wrap = mf::wrp_i(bits, g8, 0, 0);
        check(own >= 0 && own + 8 <= 64 && wrap >= 0 && wrap < 64, "lane load bounds");
        check(own % 4 == 0 && (own + 4) % 4 == 0, "two uint4 loads aligned");
        w[0] = tile[wrap];
        for (int j = 0; j < 8; ++j) w[j + 1] = tile[own + j];
        for (int t = 0; t < 4; ++t) {
            unsigned values[8];
            mf::kn_words(bits, w, t, values);
            for (int group = 0; group < 2; ++group) {
                const int offset = 8 * (4 * g8 + t) + 4 * group;
                const int b0 = (offset + 257) * bits - 16;
                const int b2 = b0 + 3 * bits + 16;
                const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
                const int shift = (i2 + 1) * 32 - b2;
                const uint64_t pair = (uint64_t(tile[i0 % 64]) << 32) | tile[i2 % 64];
                const int l0 = mf::kn_i0(bits, t, group), l2 = mf::kn_i2(bits, t, group);
                check(l0 >= 0 && l0 < 9 && l2 >= 0 && l2 < 9, "local word bounds");
                check(l2 == l0 + 1, "K8 group spans exactly two words");
                check(shift == mf::kn_s2(bits, t, group), "engine shift matches");
                for (int j = 0; j < 4; ++j) {
                    const int sh = shift + bits * (3 - j);
                    check(sh >= 0 && sh < 64, "valid shift");
                    const unsigned engine_word = unsigned(pair >> sh) & 0xffffu;
                    check(values[4 * group + j] == engine_word, "dq4 oracle mismatch");
                    // Tiles are a cyclic stream of MSB-first uint32 words.
                    const int pos = (b0 + bits * j) % 2048;
                    unsigned bit_word = 0;
                    for (int bit = 0; bit < 16; ++bit) {
                        const int p = (pos + bit) % 2048;
                        bit_word = (bit_word << 1) | ((tile[p / 32] >> (31 - p % 32)) & 1u);
                    }
                    check(values[4 * group + j] == bit_word, "independent bit oracle mismatch");
                }
            }
        }
    }
}

template<class S> static void matrices() {
    using DM = mf::Dm<S>;
    check(DM::SHARED_U32 == 3L * S::D * S::INTER * S::SB / 32, "shared matrix size");
    check(DM::EXPERT_U32 == 3L * S::D * S::INTER * S::RB / 32, "routed matrix size");
    for (int bits : {S::RB, S::SB}) {
        for (auto dims : {std::array<int, 2>{S::D / 16, S::INTER / 16},
                          std::array<int, 2>{S::INTER / 16, S::D / 16}}) {
            const int ks = dims[0], nt = dims[1];
            const long size = mf::mat_words(bits, ks, nt);
            for (bool native : {false, true})
                for (int tile = 0; tile < nt; ++tile)
                    for (int slice = 0; slice < ks; ++slice)
                        for (int lane = 0; lane < 8; ++lane) {
                            const long base = mf::tile_off(native, bits, ks, nt, tile);
                            const long stride = mf::tile_ss(native, bits, nt);
                            const long own = base + mf::own_i(bits, lane, slice, stride);
                            const long wrap = base + mf::wrp_i(bits, lane, slice, stride);
                            check(own >= 0 && own + bits <= size && wrap >= 0 && wrap < size, "matrix load bounds");
                            check(own % (bits == 8 ? 4 : 2) == 0, "matrix vector-load alignment");
                        }
        }
    }
}

int main() {
    std::array<uint32_t, 64> tile{};
    decode_tile(tile);
    tile.fill(0xffffffffu); decode_tile(tile);
    // Every input bit individually exercises both the wrapped first lane and last word.
    for (int bit = 0; bit < 2048; ++bit) {
        tile.fill(0); tile[bit / 32] = uint32_t(1) << (bit % 32); decode_tile(tile);
    }
    std::mt19937 random(8060);
    for (int n = 0; n < 200; ++n) {
        for (auto& word : tile) word = random();
        decode_tile(tile);
    }
    matrices<mf::SmallK6S8>();
    matrices<mf::QwenK6S8>();
    std::printf("K8 CHECK OK (%ld checks)\n", checks);
}
