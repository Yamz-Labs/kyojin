// SPDX-License-Identifier: MIT
// Kyojin: one explicit mul_mat_q_case<type> instantiation per translation unit (gmmq_case_*.cu), like llama.cpp's
// ggml-cuda/template-instances/mmq-instance-*.cu, so the four routed formats compile in parallel.
// The kernels come from rocm/gmmq/mmq.hpp: adapted from Gufo (github.com/gufo-org/gufo, 07e1191, MIT), itself adapted
// from llama.cpp 5c0e946 (MIT, Copyright (c) 2023-2026 The ggml authors). See THIRD_PARTY_NOTICES.md.
#pragma once
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "common.hpp"
#include "mmq.hpp"
}  // namespace qfn_mmq

#define GMMQ_DECL_CASE(T) \
    namespace qfn_mmq { extern template void mul_mat_q_case<T>(ggml_backend_hip_context &, const mmq_args &, hipStream_t); }
#define GMMQ_INST_CASE(T) \
    namespace qfn_mmq { template void mul_mat_q_case<T>(ggml_backend_hip_context &, const mmq_args &, hipStream_t); }
