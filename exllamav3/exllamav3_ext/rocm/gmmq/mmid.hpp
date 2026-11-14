// SPDX-License-Identifier: MIT
// Adapted from Gufo (github.com/gufo-org/gufo, 07e1191, MIT, Copyright (c) 2026 Gufo contributors), path
// src/models/qwen38_flash_next/kernels/rocm/mmq/mmid.hpp, itself adapted from llama.cpp 5c0e946 ggml/src/ggml-cuda
// (github.com/ggml-org/llama.cpp, MIT, Copyright (c) 2023-2026 The ggml authors). See THIRD_PARTY_NOTICES.md.
// Changes: caller-supplied scan scratch (gmmq_ids_scratch_elems).
#pragma once

size_t gmmq_ids_scratch_elems(int n_experts, int n_tokens, int n_expert_used);

void ggml_hip_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1,
        int32_t * scratch, hipStream_t stream);
