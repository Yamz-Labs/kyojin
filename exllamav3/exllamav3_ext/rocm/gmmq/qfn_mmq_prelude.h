// SPDX-License-Identifier: MIT
// Adapted from Gufo (github.com/gufo-org/gufo, 07e1191, MIT, Copyright (c) 2026 Gufo contributors), path
// src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq_prelude.h, itself adapted from llama.cpp 5c0e946 ggml/src/ggml-cuda
// (github.com/ggml-org/llama.cpp, MIT, Copyright (c) 2023-2026 The ggml authors). See THIRD_PARTY_NOTICES.md.
// Changes: #undef __HIP_NO_HALF_OPERATORS__ / __HIP_NO_HALF_CONVERSIONS__ (set by torch's HIP flags).
// Every system header the vendored tier includes, pulled in at global scope
// before the tier is wrapped in namespace qfn_mmq. The tier's own includes
// then hit include guards, so its definitions land in the namespace while
// the runtime headers stay global. The wrap keeps this copy from clashing
// with the identical tier linked for DeepSeek V4 Flash.
#pragma once
// kyojin: torch's HIP extension flags define these two; the vendored kernels rely on half operators and
// conversions (Gufo / llama.cpp build without them). Must run before the first HIP header.
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
