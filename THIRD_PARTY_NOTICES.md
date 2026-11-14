# Third-party notices

## llama.cpp / ggml (MIT License)

Parts of exllamav3/exllamav3_ext/foreign_quant/ggml_mmvq/ and foreign_quant/fq_mmvq.cu are derived from
llama.cpp (https://github.com/ggml-org/llama.cpp, commit 7182ef7):

- ggml_iq_tables.cuh: the IQ lookup tables (iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid, iq3s_grid, kvalues_iq4nl) and block layouts
  from ggml/src/ggml-common.h, copied unchanged.
- ggml_vecdotq_iq.cuh: the vec_dot_iq*_q8_1 functions and helpers from ggml/src/ggml-cuda/vecdotq.cuh (only ggml_cuda_dp4a renamed to __dp4a,
  CUDA-only helpers).
- fq_mmvq.cu: kernel structure after ggml/src/ggml-cuda/mmvq.cu (mul_mat_vec_q, mul_mat_vec_q_moe).
- ggml_vecdotq_k46.cuh: vec_dot_q4_K_q8_1 and vec_dot_q6_K_q8_1 (with their _impl helpers) and the block_q4_K / block_q6_K layouts,
  from llama.cpp commit 61dbf49 (only ggml_cuda_dp4a renamed to ggml_dp4a).

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
