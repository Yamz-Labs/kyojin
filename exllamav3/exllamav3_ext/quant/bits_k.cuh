#pragma once

#include <c10/util/Exception.h>

struct BitsK
{
    int bits;
    bool half;
};

inline BitsK bits_from_K(float K)
{
    const int bits = static_cast<int>(K);
    const float frac = K - static_cast<float>(bits);
    TORCH_CHECK(bits >= 1 && bits <= 8 &&
                (frac == 0.0f || (frac == 0.5f && bits <= 3)),
                "Unsupported EXL3 bitrate ", K,
                " (integer 1..8, or 1.5 / 2.5 / 3.5)");
    return {bits, frac == 0.5f};
}

inline int k2_from_K(float K)
{
    const BitsK bk = bits_from_K(K);
    return 2 * bk.bits + (bk.half ? 1 : 0);
}
