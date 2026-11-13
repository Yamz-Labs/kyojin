#pragma once

// RDNA fdot2 GEMV for the dense single-matrix forms, shared by RDNA4
// (gfx1200/gfx1201) and RDNA3.5 (gfx1150/1151/1152):
// M == 1, and M == 2..8 split-K.
//
// The generic GEMV kernel (quant/exl3_gemv_kernel.cuh) is built on m16n8k16
// mma.sync and reaches 107-220 GB/s on the small dense shapes on gfx1201.
// The RDNA dot core here runs one warp per 16-wide output tile and consumes
// the decoded trellis through V_DOT2_F32_F16, which sustains 1.3x that on the
// same shapes (measured against upstream exllamav3 v1.6.6). For M == 1 the
// output Hadamard is fused into the store epilogue when the per-device
// segment counters fit the width; otherwise, and for every M == 2..8 call,
// the input and output Hadamards stay the caller's had_r_128 launches.

#if defined(USE_ROCM)

#include <cstdint>
#include <cuda_fp16.h>
#include <hip/hip_runtime.h>

// True on gfx1200/gfx1201 (RDNA4) and gfx1150/1151/1152 (RDNA3.5), the parts
// whose decode was measured with this shared core.
bool exl3_gemv_rdna_arch(int device);

// Launch the dot kernel for M == 1..8. A is the already-rotated input
// (A_had), B is the trellis. For M == 1, with fuse_out and room in the
// per-device segment counters, C receives the finished output (output
// Hadamard applied) and *out_rotated is set; otherwise C holds the
// pre-rotation values and the caller applies the output Hadamard. M > 1
// always returns *out_rotated = false, so the caller rotates the whole C.
// Returns false when the shape is outside the kernel's hard constraints,
// leaving nothing launched (and *out_rotated = false).
bool exl3_gemv_rdna_launch
(
    const half* A,
    const uint16_t* B,
    void* C,
    const half* svh,
    int device,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    hipStream_t stream,
    bool fuse_out,
    bool* out_rotated
);

#endif  // USE_ROCM
