#pragma once
// index arithmetic of row-padded activation buffers, shared by the kernels (device) and the
// CPU bounds proof (tests/check_pad_bounds.cpp). One definition, so the proof checks the code that ships.
//
// Why: an fp16 activation with K=6144 has a 12288 B row pitch; the dense GEMM reads such rows at about half rate
// (rocBLAS, gfx1151). A pitch of K + 64 elements runs the same product bit-exactly at 87-90 % of peak.
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define PAD_HD __host__ __device__ __forceinline__
#else
#define PAD_HD inline
#endif

// Pitch (elements) of a padded row of k elements: k + PAD_ELEMS, or k (no padding) when the row is not a pitch hazard.
constexpr int PAD_ELEMS = 64;
PAD_HD int64_t pad_pitch_for(int64_t k) { return (k * 2) % 1024 == 0 ? k + PAD_ELEMS : k; }

// Gated norm output: row = token * hpt + head (hpt heads per token, dim elements each), y token row pitch = pitch.
// Element i of that row lands at pad_norm_off(...) + i.
PAD_HD size_t pad_norm_off(size_t row, int hpt, size_t pitch, int dim)
{
    return (row / (size_t) hpt) * pitch + (row % (size_t) hpt) * (size_t) dim;
}

// Flat element e of a (rows, k) tensor lands at pad_flat_off(e, k, pitch) in a (rows, pitch) buffer.
PAD_HD size_t pad_flat_off(size_t e, size_t k, size_t pitch)
{
    return (e / k) * pitch + (e % k);
}

// pad_copy (one 16-byte chunk = 8 two-byte elements per thread): a chunk never straddles a row and every chunk start is 16-byte aligned.
PAD_HD bool pad_copy_ok(int64_t k, int64_t pitch, int64_t storage_offset)
{
    return k >= 8 && k % 8 == 0 && pitch >= k && pitch % 8 == 0 && storage_offset % 8 == 0;
}

// A view (storage_offset, m rows of k elements at row pitch lda) fits a storage of storage_elems elements.
PAD_HD bool pad_view_fits(int64_t storage_offset, int64_t m, int64_t lda, int64_t k, int64_t storage_elems)
{
    return lda >= k && storage_offset + (m - 1) * lda + k <= storage_elems;
}
