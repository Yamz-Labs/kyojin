#pragma once

// RDNA (gfx1200/gfx1201) fdot2 GEMV for m == 1.
//
// The generic GEMV kernel (quant/exl3_gemv_kernel.cuh) is built on m16n8k16
// mma.sync and reaches 107-220 GB/s on the small dense shapes on gfx1201.
// The RDNA dot core here runs one warp per 16-wide output tile and consumes
// the decoded trellis through V_DOT2_F32_F16, which sustains 1.3x that on the
// same shapes (measured against upstream exllamav3 v1.6.6). The input and output Hadamards stay the caller's
// had_r_128 launches, so this replaces only the middle of the three-launch
// form.

#if defined(USE_ROCM)

#include <cstdint>
#include <cuda_fp16.h>
#include <hip/hip_runtime.h>

// True on gfx1200/gfx1201 (the parts whose decode was measured with this core)
bool exl3_gemv_rdna_arch(int device);

// Launch the m == 1 dot kernel. A is the already-rotated input row (A_had),
// B is the trellis. With fuse_out and room in the per-device segment
// counters, C receives the finished output (output Hadamard applied) and
// *out_rotated is set; otherwise C holds the pre-rotation values and the
// caller applies the output Hadamard. Returns false when the shape is
// outside the kernel's hard constraints, leaving nothing launched.
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
