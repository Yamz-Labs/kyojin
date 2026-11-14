// SPDX-License-Identifier: MIT
// Adapted from Gufo (github.com/gufo-org/gufo, 07e1191, MIT, Copyright (c) 2026 Gufo contributors), path
// src/models/qwen38_flash_next/kernels/rocm/mmq/ggml.h, itself adapted from llama.cpp 5c0e946 ggml/src/ggml-cuda
// (github.com/ggml-org/llama.cpp, MIT, Copyright (c) 2023-2026 The ggml authors). See THIRD_PARTY_NOTICES.md.
// Changes: none.
// Thin redirect so the vendored common.hpp / vecdotq.hpp / mmid.cu / etc.
// can write #include "ggml.h" unchanged.
#pragma once
#include "qfn_ggml_stubs.h"
