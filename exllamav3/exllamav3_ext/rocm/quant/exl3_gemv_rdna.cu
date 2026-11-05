// RDNA (gfx1200/gfx1201) GEMV launcher. See exl3_gemv_rdna.cuh.
//
// The kernel cores are adapted from exllamav3's RDNA GEMV (rocm/quant/
// exl3_gemv_kernel_rdna.cuh, MIT) and use the same dq chain as the rest of
// this tree. Two dense single-matrix forms are built here: M == 1 with the
// output Hadamard fused into the store epilogue, and M == 2..8 as a split-K
// multi-row kernel whose C the caller rotates. The multi-matrix and expert
// forms are not built here and stay on the generic WMMA path.
//
// Env switches (measurement / A-B only; all default to the measured choice):
//   EXL3_GEMV=0                 keep the generic WMMA GEMV
//   EXL3_ROCM_GEMV_TILES=0      barrier-free direct core instead of the tiles core
//   EXL3_GEMV_LDS=1             LDS unswizzle core
//   EXL3_GEMV_SPLITK=0          pin the single-warp form
//   EXL3_GEMV_SPLITK_WARPS=4|8|16
//   EXL3_GEMV_WARPS=4|8|16      force the single-warp block size
//   EXL3_GEMV_FUSE_OUT=0        keep the separate output Hadamard launcher
//
// EXL3_GEMV_SPLITK, EXL3_GEMV_SPLITK_WARPS and EXL3_GEMV_WARPS select the
// M == 1 form only; the M == 2..8 path always uses the split-K MR kernel and
// ignores them. EXL3_GEMV_FUSE_OUT likewise applies to M == 1 only.

#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

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

// One preallocated counter slab per device; each stream owns one slice. The
// previous lazy table was capped at 64 entries, so keep that cap here.
static constexpr int MAX_SEG_COUNTER_STREAMS = 64;

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

// Per-(device, stream) output-Hadamard arrival counters. A single device-wide
// buffer would race when two streams overlap: both kernels index the same
// segment counters and one can rotate a segment the other is still writing.
// An entry is never freed (the process owns the device); if the table fills,
// the caller falls back to the separate output Hadamard.
//
// The counter slab is preallocated and zeroed on the first eager call, then
// handed out in per-stream slices. It must never be allocated inside a
// captured stream: hipMalloc/hipMemset are not permitted during capture, and
// the block graph captures on a dedicated stream that the eager warmups never
// touch. If the slab does not exist yet while capturing, decline and let the
// caller use the plain kernel plus the separate output Hadamard.
struct SegCounterSlot
{
    int device;
    hipStream_t stream;
    int* ptr;
};

struct SegCounterPool
{
    int device;
    int* base;
    int used;
};

static bool stream_is_capturing(hipStream_t stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &status) != hipSuccess) return false;
    return status != hipStreamCaptureStatusNone;
}

static int* seg_counters(int device, hipStream_t stream)
{
    static std::mutex mtx;
    static std::vector<SegCounterSlot> slots;
    static std::vector<SegCounterPool> pools;
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    for (const auto& s : slots)
        if (s.device == device && s.stream == stream) return s.ptr;
    if (slots.size() >= (size_t) MAX_SEG_COUNTER_STREAMS) return nullptr;

    SegCounterPool* pool = nullptr;
    for (auto& p : pools)
        if (p.device == device) { pool = &p; break; }

    if (!pool)
    {
        if (stream_is_capturing(stream)) return nullptr;
        const size_t bytes = (size_t) MAX_SEG_COUNTER_STREAMS * EXL3_GEMV_SEG_COUNTERS * sizeof(int);
        void* base = nullptr;
        if (hipMalloc(&base, bytes) != hipSuccess)
        {
            (void) hipGetLastError();
            return nullptr;
        }
        if (hipMemset(base, 0, bytes) != hipSuccess)
        {
            (void) hipGetLastError();
            hipFree(base);
            return nullptr;
        }
        pools.push_back({device, (int*) base, 0});
        pool = &pools.back();
    }

    if (pool->used >= MAX_SEG_COUNTER_STREAMS) return nullptr;
    int* p = pool->base + (size_t) pool->used * EXL3_GEMV_SEG_COUNTERS;
    pool->used++;
    slots.push_back({device, stream, p});
    return p;
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
    // This path is tuned for gfx1200/gfx1201. Keep the arch test explicit so
    // another RDNA family is opted in only after its own measurement and
    // tuning.
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
        accum, C, 0, tile_n, lane, svh, 0.088388347648f,
        seg + tile_n / 8
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
        accum, C, 0, tile_n, lane, svh, 0.088388347648f,
        seg + tile_n / 8
    );
}

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
)
{
    if (out_rotated) *out_rotated = false;
    if (K < 1 || K > 8) return false;
    if (cb < 0 || cb > 2) return false;
    if (size_m < 1 || size_m > 8) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (env_int("EXL3_GEMV", 1) == 0) return false;

    const int n_tiles = size_n / 16;
    const int k_blocks = size_k / 16;
    const int core = exl3_gemv_core_mode();

    // Rows 2..8: one block per N-tile, warps split K, the decoded tile is
    // applied to every row before it is discarded. No output-Hadamard fusion
    // (the caller's had_r_128 covers all rows), so C holds the raw products.
    if (size_m > 1)
    {
        if (K < 2 || K > 6 || (cb != 1 && cb != 2)) return false;
        const int M = size_m <= 2 ? 2 : (size_m == 3 ? 3 : (size_m == 4 ? 4 : 8));
        const int warps = splitk_warps(k_blocks, n_tiles);
        const size_t smem_mr = exl3_gemv_smem_bytes_splitk_mr(warps, M);
        if (smem_mr > 64 * 1024) return false;

        #define LAUNCH_MR(bits_val, codebook, m_val, w)                                \
            do {                                                                       \
                if (c_fp32) hipLaunchKernelGGL((exl3_gemv_dot_kernel_splitk_mr<bits_val, true,  codebook, m_val, w>), \
                    dim3(n_tiles), dim3(w * 32), smem_mr, stream, A, B, C, size_k, size_n, size_m, core); \
                else        hipLaunchKernelGGL((exl3_gemv_dot_kernel_splitk_mr<bits_val, false, codebook, m_val, w>), \
                    dim3(n_tiles), dim3(w * 32), smem_mr, stream, A, B, C, size_k, size_n, size_m, core); \
            } while (0)

        #define LAUNCH_MR_W(bits_val, codebook, m_val)                                 \
            do {                                                                       \
                if (warps == 4)       LAUNCH_MR(bits_val, codebook, m_val, 4);         \
                else if (warps == 8)  LAUNCH_MR(bits_val, codebook, m_val, 8);         \
                else if (warps == 16) LAUNCH_MR(bits_val, codebook, m_val, 16);        \
                else return false;                                                     \
            } while (0)

        #define LAUNCH_MR_CB(bits_val, m_val)                                          \
            do {                                                                       \
                if (cb == 1)      LAUNCH_MR_W(bits_val, 1, m_val);                     \
                else if (cb == 2) LAUNCH_MR_W(bits_val, 2, m_val);                     \
                else return false;                                                     \
            } while (0)

        #define LAUNCH_MR_K(m_val)                                                     \
            do {                                                                       \
                switch (K) {                                                           \
                    case 2: LAUNCH_MR_CB(2, m_val); break;                             \
                    case 3: LAUNCH_MR_CB(3, m_val); break;                             \
                    case 4: LAUNCH_MR_CB(4, m_val); break;                             \
                    case 5: LAUNCH_MR_CB(5, m_val); break;                             \
                    case 6: LAUNCH_MR_CB(6, m_val); break;                             \
                    default: return false;                                             \
                }                                                                      \
            } while (0)

        if (M == 2) LAUNCH_MR_K(2);
        else if (M == 3) LAUNCH_MR_K(3);
        else if (M == 4) LAUNCH_MR_K(4);
        else LAUNCH_MR_K(8);

        #undef LAUNCH_MR_K
        #undef LAUNCH_MR_CB
        #undef LAUNCH_MR_W
        #undef LAUNCH_MR
        return true;
    }

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

    int* seg = nullptr;
    if (fuse_out && env_int("EXL3_GEMV_FUSE_OUT", 1) != 0 &&
        size_n / 128 <= EXL3_GEMV_SEG_COUNTERS)
        seg = seg_counters(device, stream);
    const bool fused = seg != nullptr;

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

    // Publish only now: every LAUNCH_* path above can return false, and the
    // caller must fall back and still run its output Hadamard in that case.
    if (out_rotated) *out_rotated = fused;

    return true;
}

#endif  // USE_ROCM
