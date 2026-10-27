// CPU build of exllamav3/modules/dsv41_fn/dsv41_num.h for tests/dsv41/test_dsv41_num.py (ctypes). No GPU.
#include "../../exllamav3/modules/dsv41_fn/dsv41_num.h"

extern "C" {

// in-place fake-quant of `rows` rows of n floats; mode 0 = fp8 e4m3/ue8m0 (group 32), 1 = fp4 ue8m0 (32), 2 = fp4 e4m3 scale (16)
void ds_fq_rows(float* x, long rows, long n, int mode)
{
    int g = mode == 2 ? 16 : 32;
    for (long r = 0; r < rows; r++)
        for (long b = 0; b < n; b += g)
        {
            float* p = x + r * n + b;
            float amax = 0.0f;
            for (int i = 0; i < g; i++) amax = fmaxf(amax, fabsf(p[i]));
            float s = mode == 0 ? ds_fp8_scale(amax) : (mode == 1 ? ds_fp4_scale_ue8m0(amax) : ds_fp4_scale_e4m3(amax));
            for (int i = 0; i < g; i++) p[i] = mode == 0 ? ds_fp8_q(p[i], s) : ds_fp4_q(p[i], s);
        }
}

void ds_e4m3_round_arr(const float* x, float* y, long n) { for (long i = 0; i < n; i++) y[i] = ds_e4m3_round(x[i]); }

// raw [rows, 136] -> out [rows, 256]: 128 packed bytes (low nibble first) + 8 e8m0 scales per 32 values
void ds_mxfp4_decode(const uint8_t* raw, float* out, long rows)
{
    for (long r = 0; r < rows; r++)
        for (int j = 0; j < 256; j++)
        {
            uint8_t byte = raw[r * 136 + j / 2];
            uint32_t nib = (j & 1) ? (byte >> 4) : (byte & 15);
            out[r * 256 + j] = ds_e2m1(nib) * ds_e8m0(raw[r * 136 + 128 + j / 32]);
        }
}

// one engram position: cache int64 [*], returns out[(ng-1)*nh]
void ds_engram_pos(const int64_t* cache, int64_t pos, int ng, int nh, int64_t pad, const int64_t* mult,
                   const int64_t* primes, const int64_t* offsets, int64_t* out)
{
    int64_t toks[8];
    ds_engram_toks(cache, pos, ng, pad, toks);
    ds_engram_hash(toks, ng, nh, mult, primes, offsets, out);
}

}
