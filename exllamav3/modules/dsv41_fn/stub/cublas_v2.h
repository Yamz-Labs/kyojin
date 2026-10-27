// Stub for the standalone DS4.1 kernel modules: exllamav3_ext/util.cuh includes cublas_v2.h unguarded (the extension build gets
// it from hipify). Only the names util.cuh itself uses are provided; no kernel here calls cuBLAS.
#pragma once
#include <hip/hip_runtime.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
typedef int cublasStatus_t;
enum { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE,
       CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR,
       CUBLAS_STATUS_NOT_SUPPORTED };
