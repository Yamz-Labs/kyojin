// RDNA (gfx1200/gfx1201) m=1 GEMV launcher. See exl3_gemv_rdna.cuh.
//
// The kernel cores are adapted from exllamav3's RDNA GEMV (rocm/quant/
// exl3_gemv_kernel_rdna.cuh, MIT) and use the same dq chain as the rest of
// this tree. Only the dense single-matrix m == 1 form is built here; the
// multi-row and multi-matrix forms stay on the generic WMMA path.
//
// Env switches (measurement / A-B only; all default to the measured choice):
//   EXL3_GEMV=0                 keep the generic WMMA GEMV
//   EXL3_ROCM_GEMV_TILES=0      barrier-free direct core instead of the tiles core
//   EXL3_GEMV_LDS=1             LDS unswizzle core
//   EXL3_GEMV_SPLITK=0          pin the single-warp form
//   EXL3_GEMV_SPLITK_WARPS=4|8|16
//   EXL3_GEMV_WARPS=4|8|16      force the single-warp block size
//   EXL3_GEMV_FUSE_OUT=0        keep the separate output Hadamard launcher

#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "../../util.h"
#include "../../util.cuh"
#include "exl3_gemv_rdna.cuh"
#include "exl3_gemv_kernel_rdna.cuh"

// One arrival counter per 128-wide output segment. Self-resetting: the warp
// that observes the final arrival zeroes it again (see
// exl3_gemv_fused_epilogue). Widths above the capacity keep the separate
// output-Hadamard launch.
#ifndef EXL3_GEMV_SEG_COUNTERS
#define EXL3_GEMV_SEG_COUNTERS 1024
#endif

static int env_int(const char* name, int def)
{
    const char* env = std::getenv(name);
    return env ? atoi(env) : def;
}

bool exl3_gemv_lds_core()
{
    return env_int("EXL3_GEMV_LDS", 0) != 0;
}

int exl3_gemv_core_mode()
{
    if (exl3_gemv_lds_core()) return EXL3_GEMV_CORE_LDS;
    return env_int("EXL3_ROCM_GEMV_TILES", 1) == 0
        ? EXL3_GEMV_CORE_DIRECT
        : EXL3_GEMV_CORE_TILES;
}

static bool splitk_enabled()
{
    return env_int("EXL3_GEMV_SPLITK", 1) != 0;
}

static int warps_for(int n_tiles, int k_blocks)
{
    const int forced = env_int("EXL3_GEMV_WARPS", 0);
    if (forced == 4 || forced == 8 || forced == 16) return forced;
    if (n_tiles <= 32) return 16;
    if (n_tiles <= 96 && k_blocks <= 256) return 8;
    return 4;
}

static int splitk_warps(int k_tiles, int n_tiles)
{
    int w = env_int("EXL3_GEMV_SPLITK_WARPS", 0);
    if (w == 4 || w == 8 || w == 16) return w;
    const int blocks = n_tiles;
    if (k_tiles <= 128) return 4;
    if (blocks >= 1024) return 4;
    if (blocks < 128 || k_tiles >= 1024) return 16;
    return 8;
}

// Per-device output-Hadamard arrival counters. Allocated once; never freed
// (the process owns the device for its lifetime, as DevCtx does).
static int* seg_counters(int device)
{
    static std::mutex mtx;
    static int* blocks[64] = {};
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    if (!blocks[device])
    {
        void* p = nullptr;
        if (hipMalloc(&p, EXL3_GEMV_SEG_COUNTERS * sizeof(int)) != hipSuccess)
        {
            (void) hipGetLastError();
            return nullptr;
        }
        if (hipMemset(p, 0, EXL3_GEMV_SEG_COUNTERS * sizeof(int)) != hipSuccess)
        {
            (void) hipGetLastError();
            hipFree(p);
            return nullptr;
        }
        blocks[device] = (int*) p;
    }
    return blocks[device];
}

bool exl3_gemv_rdna_arch(int device)
{
    static int cache[64] = {};
    if (device < 0 || device >= 64) return false;
    if (cache[device] < 0) return false;
    if (cache[device] > 0) return true;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess)
    {
        cache[device] = -1;
        return false;
    }
    const bool ok =
        strncmp(prop.gcnArchName, "gfx1200", 7) == 0 ||
        strncmp(prop.gcnArchName, "gfx1201", 7) == 0;
    cache[device] = ok ? 1 : -1;
    return ok;
}

// -----------------------------------------------------------------------------
// Dot kernel with the output Hadamard fused into the store (one launch fewer
// than the three-kernel form; the input Hadamard stays the caller's launch).
// Every output is identical to the separate rotation: the epilogue runs the
// same arithmetic once the last of the segment's eight tiles has arrived.
// -----------------------------------------------------------------------------

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_rdna_dot_fused
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const half* __restrict__ svh,
    int* __restrict__ seg,
    const int size_k,
    const int size_n,
    const int core
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int n_tiles = size_n / 16;
    if (tile_n >= n_tiles) return;

    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    const float accum = exl3_gemv_dot_tile_sel<bits, cb, false, true>
    (
        core, A, B, size_k, n_tiles, tile_n, lane, my_sh_b, my_sh_b_quant,
        0, size_k / 16
    );

    exl3_gemv_fused_epilogue<c_fp32>
    (
        accum, C, 0, size_n, tile_n, lane, svh, 0.088388347648f,
        seg + tile_n / 8, nullptr, 0, 1
    );
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_rdna_dot_fused_splitk
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const half* __restrict__ svh,
    int* __restrict__ seg,
    const int size_k,
    const int size_n,
    const int core
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile_n = blockIdx.x;
    const int n_tiles = size_n / 16;

    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    float* sh_red = (float*) (sh_b_quant + WARPS_PER_BLOCK * EXL3_GEMV_SH_QUANT_U16);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    const float accum = exl3_gemv_dot_tile_splitk<bits, cb, WARPS_PER_BLOCK>
    (
        A, B, size_k, n_tiles, tile_n, warp_id, lane,
        my_sh_b, my_sh_b_quant, sh_red, core
    );

    if (warp_id != 0) return;
    exl3_gemv_fused_epilogue<c_fp32>
    (
        accum, C, 0, size_n, tile_n, lane, svh, 0.088388347648f,
        seg + tile_n / 8, nullptr, 0, 1
    );
}

bool exl3_gemv_rdna_launch
(
    const half* A,
    const uint16_t* B,
    void* C,
    const half* svh,
    int device,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    hipStream_t stream,
    bool fuse_out,
    bool* out_rotated
)
{
    if (out_rotated) *out_rotated = false;
    if (K < 1 || K > 8) return false;
    if (cb < 0 || cb > 2) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (env_int("EXL3_GEMV", 1) == 0) return false;

    const int n_tiles = size_n / 16;
    const int k_blocks = size_k / 16;

    bool splitk = splitk_enabled() && n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES;
    int warps = warps_for(n_tiles, k_blocks);
    int blocks = (n_tiles + warps - 1) / warps;
    size_t smem = exl3_gemv_smem_bytes(warps);
    if (splitk)
    {
        warps = splitk_warps(k_blocks, n_tiles);
        blocks = n_tiles;
        smem = exl3_gemv_smem_bytes_splitk(warps);
    }
    if (smem > 64 * 1024) return false;

    const int core = exl3_gemv_core_mode();
    int* seg = nullptr;
    if (fuse_out && env_int("EXL3_GEMV_FUSE_OUT", 1) != 0 &&
        size_n / 128 <= EXL3_GEMV_SEG_COUNTERS)
        seg = seg_counters(device);
    const bool fused = seg != nullptr;
    if (out_rotated) *out_rotated = fused;

    #define LAUNCH_FUSED(bits_val, codebook, w, kern)                              \
        do {                                                                       \
            if (c_fp32) hipLaunchKernelGGL((kern<bits_val, true,  codebook, w>),   \
                dim3(blocks), dim3(w * 32), smem, stream, A, B, C, svh, seg,       \
                size_k, size_n, core);                                             \
            else        hipLaunchKernelGGL((kern<bits_val, false, codebook, w>),   \
                dim3(blocks), dim3(w * 32), smem, stream, A, B, C, svh, seg,       \
                size_k, size_n, core);                                             \
        } while (0)

    #define LAUNCH_PLAIN_K(bits_val, codebook, w, kern)                            \
        do {                                                                       \
            if (c_fp32) hipLaunchKernelGGL((kern<bits_val, true,  codebook, w>),   \
                dim3(blocks), dim3(w * 32), smem, stream, A, B, C, size_k, size_n, core); \
            else        hipLaunchKernelGGL((kern<bits_val, false, codebook, w>),   \
                dim3(blocks), dim3(w * 32), smem, stream, A, B, C, size_k, size_n, core); \
        } while (0)

    #define LAUNCH_FOR_W(bits_val, codebook, w, fused_kern, plain_kern)            \
        do {                                                                       \
            if (w != 4 && w != 8 && w != 16) return false;                         \
            if (fused)                                                                 \
            {                                                                      \
                if (w == 4)  LAUNCH_FUSED(bits_val, codebook, 4,  fused_kern);      \
                if (w == 8)  LAUNCH_FUSED(bits_val, codebook, 8,  fused_kern);      \
                if (w == 16) LAUNCH_FUSED(bits_val, codebook, 16, fused_kern);      \
            }                                                                      \
            else                                                                   \
            {                                                                      \
                if (w == 4)  LAUNCH_PLAIN_K(bits_val, codebook, 4,  plain_kern);    \
                if (w == 8)  LAUNCH_PLAIN_K(bits_val, codebook, 8,  plain_kern);    \
                if (w == 16) LAUNCH_PLAIN_K(bits_val, codebook, 16, plain_kern);    \
            }                                                                      \
        } while (0)

    #define LAUNCH_CB(bits_val, codebook)                                          \
        do {                                                                       \
            if (splitk) LAUNCH_FOR_W(bits_val, codebook, warps,                         \
                exl3_gemv_rdna_dot_fused_splitk, exl3_gemv_dot_kernel_splitk);          \
            else        LAUNCH_FOR_W(bits_val, codebook, warps,                         \
                exl3_gemv_rdna_dot_fused, exl3_gemv_dot_kernel);                        \
        } while (0)

    #define LAUNCH_BITS(bits_val)                                                  \
        do {                                                                       \
            switch (cb) {                                                          \
                case 0: LAUNCH_CB(bits_val, 0); break;                             \
                case 1: LAUNCH_CB(bits_val, 1); break;                             \
                case 2: LAUNCH_CB(bits_val, 2); break;                             \
                default: return false;                                             \
            }                                                                      \
        } while (0)

    switch (K)
    {
        case 1: LAUNCH_BITS(1); break;
        case 2: LAUNCH_BITS(2); break;
        case 3: LAUNCH_BITS(3); break;
        case 4: LAUNCH_BITS(4); break;
        case 5: LAUNCH_BITS(5); break;
        case 6: LAUNCH_BITS(6); break;
        case 7: LAUNCH_BITS(7); break;
        case 8: LAUNCH_BITS(8); break;
        default: return false;
    }

    #undef LAUNCH_BITS
    #undef LAUNCH_CB
    #undef LAUNCH_FOR_W
    #undef LAUNCH_PLAIN_K
    #undef LAUNCH_FUSED

    return true;
}

#endif  // USE_ROCM
