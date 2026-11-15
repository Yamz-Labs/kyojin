# Third-party notices

## llama.cpp / ggml (MIT License)

exllamav3/exllamav3_ext/rocm/gmmq/ (routed MMQ prefill MoE for HIP) is adapted from Gufo (https://github.com/gufo-org/gufo,
commit 07e1191, path src/models/qwen38_flash_next/kernels/rocm/mmq/, MIT License, Copyright (c) 2026 Gufo contributors), itself
adapted from llama.cpp (commit 5c0e946, ggml/src/ggml-cuda mmq / mmid / quantize, MIT License, Copyright (c) 2023-2026 The ggml
authors). Each file keeps its licence header and lists its changes. Gufo's MIT licence text is the same as the one below.

exllamav3/exllamav3_ext/rocm/routed_f16/ (routed F16-WMMA prefill MoE for HIP gfx1151) is adapted from Gufo (https://github.com/gufo-org/gufo,
commit 07e1191, src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp: RoutedPairF16Kernel, RoutedDownF16Kernel, RoutedPadBoundsKernel,
RoutedScatterKernel and their device helpers, and MoeEpilogueVec4Kernel for fq_routed_f16_combine and fq_routed_f16_combine_shared (its shared-expert term and gate), MIT License, Copyright (c) 2026 gufo contributors). The file keeps its licence header and lists
its changes.

exllamav3/modules/attention_fn/qsa_prefill_wmma.hip (4-row union WMMA sparse prefill attention for HIP) follows the structure of Gufo's
WmmaCausalAttentionKernel (https://github.com/gufo-org/gufo, path src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp, MIT License,
Copyright (c) 2026 Gufo contributors): several query rows per block, 16-key tiles gathered once from the union of the rows' selections,
K / V tile prefetch in registers, V staged transposed in LDS. The code itself is written for Kyojin's index-list selection (LDS bitonic
union, per-row visibility masks); no Gufo source is copied.

Parts of exllamav3/exllamav3_ext/foreign_quant/ggml_mmvq/ and foreign_quant/fq_mmvq.cu are derived from
llama.cpp (https://github.com/ggml-org/llama.cpp, commit 7182ef7):

- ggml_iq_tables.cuh: the IQ lookup tables (iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid, iq3s_grid, kvalues_iq4nl) and block layouts
  from ggml/src/ggml-common.h, copied unchanged.
- ggml_vecdotq_iq.cuh: the vec_dot_iq*_q8_1 functions and helpers from ggml/src/ggml-cuda/vecdotq.cuh (only ggml_cuda_dp4a renamed to __dp4a,
  CUDA-only helpers).
- fq_mmvq.cu: kernel structure after ggml/src/ggml-cuda/mmvq.cu (mul_mat_vec_q, mul_mat_vec_q_moe).
- ggml_vecdotq_k46.cuh: vec_dot_q4_K_q8_1 and vec_dot_q6_K_q8_1 (with their _impl helpers) and the block_q4_K / block_q6_K layouts,
  from llama.cpp commit 61dbf49 (only ggml_cuda_dp4a renamed to ggml_dp4a).
- fq_mmq_few.cu: the few-row MMQ (mul_mat_q_few, its Q8_0 / Q4_K / Q5_K / Q6_K decoders and quantize_mmq_few_q8_1) of llama.cpp PR #31533
  (ggml/src/ggml-cuda/mmq-few.cu, quantize.cu; merge commit 0fb8f15770, release v1.0.0), with unpack_scales_q45_K (mmq-load-tiles.cuh) and
  the int8 mma / cp.async PTX of mma.cuh / cp-async.cuh inlined. Renamed: ggml_type -> local MF_* ids, tile<16, 8, int> -> mf_c_tile;
  one channel, half input, half or float output, no PDL / DGX Spark prefetch.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
