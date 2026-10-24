#pragma once

#if defined(USE_ROCM)

#include <ATen/Tensor.h>

// QSA selection with MANY blocks per row: bitwise identical to dsa_topk_gfx12 (one block per row,
// four row sweeps) on the same inputs, but the row sweeps run on the whole GPU, which is what a
// decode-batch-1 selection of a 32K..512K-pool row needs. Contract and bounds proof: dsa_topk_par.cu.
// nb = 0 picks the block count from the row width; pass nb > 0 only to sweep it.
void dsa_topk_par(const at::Tensor& scores, at::Tensor& indices, int64_t k, int64_t nb = 0);

#endif