#if defined(USE_ROCM)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
// c10/hip/HIPGuard.h does NOT define the MasqueradingAsCUDA guards that
// hipify rewrites c10::cuda::OptionalCUDAGuard into; pull in the ATen header
// that actually declares them (this file is ROCm-only, see build_config.py).
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <cstdint>

#include "activation.cuh"
#include "util.h"
#include "util.cuh"

#define NUM_THREADS_P 1024

// x * sigmoid(y @ w) + z -> z
__global__ __launch_bounds__(NUM_THREADS_P)
void add_sigmoid_proj_kernel_f
(
    const float* __restrict__ px,
    const half* __restrict__ py,
    float* __restrict__ pz,
    const half* __restrict__ pw,
    const size_t dim
)
{
    const size_t b = blockIdx.x;
    const int t = threadIdx.x;
    const float* pxb = px + dim * b;
    const half* pyb = py + dim * b;
    float* pzb = pz + dim * b;

    float yw = 0.0f;
    for (size_t idx = t; idx < dim; idx += NUM_THREADS_P)
        yw += __half2float(pw[idx]) * __half2float(pyb[idx]);

    __shared__ float sums[NUM_THREADS_P];
    sums[t] = yw;
    __syncthreads();
    for (int stride = NUM_THREADS_P / 2; stride > 0; stride >>= 1)
    {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    const float syw = 1.0f / (1.0f + __expf(-sums[0]));
    if (syw < 1e-8f) return;

    for (size_t idx = t; idx < dim; idx += NUM_THREADS_P)
        pzb[idx] += pxb[idx] * syw;
}

void add_sigmoid_gate_proj
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const at::Tensor& w
)
{
    constexpr const char* op = "add_sigmoid_gate_proj";
    TORCH_CHECK(x.is_cuda(), op, ": x must be a CUDA tensor");
    const at::Device device = x.device();
    const auto check_device = [&](const at::Tensor& tensor)
    {
        TORCH_CHECK(tensor.is_cuda(), op, ": all tensors must be CUDA tensors");
        TORCH_CHECK(tensor.device() == device, op, ": all tensors must be on the same device");
    };
    check_device(y);
    check_device(z);
    check_device(w);

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_DTYPE(z, kFloat);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK(x.dim() >= 1, op, ": x must have shape [..., D]");
    TORCH_CHECK(x.sizes() == y.sizes() && x.sizes() == z.sizes(),
                op, ": x, y and z must have identical shapes [..., D]");
    TORCH_CHECK(w.dim() == 2, op, ": w must have shape [D, 1]");

    const int64_t dim = x.size(-1);
    TORCH_CHECK(dim > 0, op, ": D must be greater than zero");
    TORCH_CHECK(w.size(0) == dim && w.size(1) == 1, op, ": w must have shape [D, 1]");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && z.is_contiguous() && w.is_contiguous(),
                op, ": all tensors must be contiguous");

    hipDeviceProp_t prop;
    cuda_check(hipGetDeviceProperties(&prop, device.index()));
    TORCH_CHECK(prop.maxThreadsPerBlock >= NUM_THREADS_P,
                op, ": device does not support the required block size");

    const size_t rows = static_cast<size_t>(x.numel() / dim);
    if (rows == 0) return;
    TORCH_CHECK(rows <= static_cast<size_t>(prop.maxGridSize[0]), op, ": too many rows");

    const c10::cuda::OptionalCUDAGuard device_guard(device);
    hipStream_t stream = c10::hip::getCurrentHIPStream(device.index()).stream();
    add_sigmoid_proj_kernel_f<<<rows, NUM_THREADS_P, 0, stream>>>
    (
        static_cast<const float*>(x.data_ptr()),
        static_cast<const half*>(y.data_ptr()),
        static_cast<float*>(z.data_ptr()),
        static_cast<const half*>(w.data_ptr()),
        static_cast<size_t>(dim)
    );
    cuda_check(hipPeekAtLastError());
}

#endif // USE_ROCM
