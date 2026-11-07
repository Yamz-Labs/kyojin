#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include "pad_pitch.h"
#include <limits>
#include <cstdlib>
#include <type_traits>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

// row stride (elements) of the activation operand. Contiguous -> k (unchanged behaviour).
// A row-padded view (stride > k, columns contiguous, leading dims collapsible) is accepted so a producer can leave
// a non-power-of-two row pitch: K=6144 fp16 rows are 12288 B apart and the GEMM reads them at ~half rate.
// Returns false if the view cannot be expressed as one (m, k) matrix with a single row pitch or if the
// storage cannot hold (m-1)*lda + k elements (the bound that keeps the library from reading outside the buffer).
static bool a_row_pitch(const at::Tensor& a, int64_t& lda)
{
    const int dim = a.dim();
    if (dim < 1) return false;
    const int64_t k = a.size(-1);
    if (k < 1) return false;
    if (dim == 1) { lda = k; return true; }
    if (a.stride(-1) != 1) return false;
    int64_t m = a.size(-2);
    for (int i = dim - 3; i >= 0; --i)
    {
        if (a.size(i + 1) != 1 && a.stride(i) != a.stride(i + 1) * a.size(i + 1)) return false;
        m *= a.size(i);
    }
    lda = m == 1 ? k : a.stride(-2);
    if (lda < k || lda > std::numeric_limits<int>::max()) return false;
    return pad_view_fits(a.storage_offset(), m, lda, k, (int64_t) (a.storage().nbytes() / sizeof(at::Half)));
}

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t a_pitch = 0;
    TORCH_CHECK(a_row_pitch(a, a_pitch), "hgemm: a must be contiguous or a row-padded view inside its storage");
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, (int) a_pitch,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

#if defined(USE_ROCM)

/*
Tuned dense GEMM for ROCm prefill (EXL3_DENSE_GEMM_TUNE=1, read on every call so an in-process
A/B can toggle it). The default hipblas -> rocBLAS/hipBLASLt heuristic picks MT128x128x16 for
every large fp16 GEMM on gfx1151 (~24-27 TFLOPS); a rocBLAS solution chosen per shape by
timing runs the same GEMMs 20-35% faster. Per (n, k, m class, out dtype) key, every solution
rocblas_gemm_ex_get_solutions returns is screened on the live operands, the best 4 are re-timed
against the untuned hipblas path and must beat it by 3%; the winner (or 0 = untuned) is kept in
memory and appended to a cache file (keyed by rocBLAS version and arch), so a process tunes each shape once ever. Own rocblas handle with atomics disabled:
split-K solutions stay deterministic. Never tunes under stream capture; any failure falls back
to the untuned path.
*/

#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <algorithm>
#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <mutex>
#include <cstdio>
#include <sys/stat.h>

namespace dtune
{

constexpr int MIN_M = 256;             // rows below this: untuned path (decode, short tails)
constexpr int NUM_DEV = MAX_DEVICES;

typedef std::tuple<int, int, int, int, int> Key;   // arch-local: n, k, m class, ldc == n, out fp32
static std::map<Key, int> g_cache;     // solution index, 0 = default heuristic wins
static bool g_loaded = false;
static std::mutex g_mutex;
static rocblas_handle g_handle[NUM_DEV] = {};
static std::string g_tag;

static bool enabled()
{
    const char* e = getenv("EXL3_DENSE_GEMM_TUNE");
    return !e || *e != '0';   // on by default: winners are bit-exact with the untuned path
}

// m class: tuned winners at 2048 and 1792 rows coincide; round up to 256 so one key serves
// every chunk length in the class
static int m_class(int m) { return ((m + 255) / 256) * 256; }

static std::string cache_path()
{
    const char* p = getenv("EXL3_DENSE_GEMM_TUNE_FILE");
    if (p && *p) return p;
    const char* h = getenv("HOME");
    std::string d = std::string(h ? h : "/tmp") + "/.cache/exllamav3";
    mkdir((std::string(h ? h : "/tmp") + "/.cache").c_str(), 0755);
    mkdir(d.c_str(), 0755);
    return d + "/dense_gemm_tune.txt";
}

static void load(int device)
{
    if (g_loaded) return;
    g_loaded = true;
    char ver[256] = {};
    rocblas_get_version_string(ver, sizeof(ver));
    hipDeviceProp_t prop;
    cudaGetDeviceProperties(&prop, device);
    g_tag = std::string(ver) + "|" + prop.gcnArchName;
    for (auto& ch : g_tag) if (ch == ' ') ch = '_';
    g_tag += "|exact1";   // only bit-exact winners; older free-pick lines no longer match
    FILE* f = fopen(cache_path().c_str(), "r");
    if (!f) return;
    char tag[512]; int n, k, mc, ldn, f32, sol;
    while (fscanf(f, "%511s %d %d %d %d %d %d", tag, &n, &k, &mc, &ldn, &f32, &sol) == 7)
        if (g_tag == tag) g_cache[Key(n, k, mc, ldn, f32)] = sol;
    fclose(f);
}

static void store(const Key& key, int sol)
{
    FILE* f = fopen(cache_path().c_str(), "a");
    if (!f) return;
    fprintf(f, "%s %d %d %d %d %d %d\n", g_tag.c_str(), std::get<0>(key), std::get<1>(key),
            std::get<2>(key), std::get<3>(key), std::get<4>(key), sol);
    fclose(f);
}

static rocblas_handle handle(int device)
{
    if (!g_handle[device])
    {
        rocblas_create_handle(&g_handle[device]);
        rocblas_set_atomics_mode(g_handle[device], rocblas_atomics_not_allowed);
        rocblas_set_pointer_mode(g_handle[device], rocblas_pointer_mode_host);
    }
    return g_handle[device];
}

struct Call
{
    rocblas_handle h; int m, n, k, ldc, lda; const void* a; const void* b; void* c; rocblas_datatype ct;
    rocblas_status run(rocblas_gemm_algo algo, int sol) const
    {
        float alpha = 1.0f, beta = 0.0f;
        // column-major: C^T[n, m] = B^T[n, k] @ A^T[k, m], same mapping as hgemm_gemmex_impl
        return rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none, n, m, k,
                               &alpha, b, rocblas_datatype_f16_r, n, a, rocblas_datatype_f16_r, lda,
                               &beta, c, ct, ldc, c, ct, ldc, rocblas_datatype_f32_r,
                               algo, sol, rocblas_gemm_flags_none);
    }
};

template <typename F>
static float time_fn(F&& fn, cudaStream_t stream, int reps)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    if (!fn()) { cudaEventDestroy(e0); cudaEventDestroy(e1); return 1e30f; }
    cudaEventRecord(e0, stream);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, e0, e1) != cudaSuccess) ms = 1e30f * reps;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / reps;
}

static float time_ms(const Call& call, rocblas_gemm_algo algo, int sol, cudaStream_t stream, int reps)
{
    return time_fn([&]{ return call.run(algo, sol) == rocblas_status_success; }, stream, reps);
}

// def_time(reps): times the untuned path (hipblas GemmEx, may route to hipBLASLt), the bar to beat
// same(): true if c now holds exactly the untuned path's output. Only bit-identical solutions
// qualify, so tuning never changes numerics (a free pick moved PPL +0.11% at m=4096).
template <typename T, typename S>
static int tune(const Call& call, cudaStream_t stream, T&& def_time, S&& same)
{
    float alpha = 1.0f, beta = 0.0f;
    rocblas_int count = 0;
    auto st = rocblas_gemm_ex_get_solutions(call.h, rocblas_operation_none, rocblas_operation_none,
        call.n, call.m, call.k, &alpha, call.b, rocblas_datatype_f16_r, call.n, call.a,
        rocblas_datatype_f16_r, call.k, &beta, call.c, call.ct, call.ldc, call.c, call.ct, call.ldc,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none, nullptr, &count);
    if (st != rocblas_status_success || count <= 0) return 0;
    std::vector<rocblas_int> sols(count);
    rocblas_gemm_ex_get_solutions(call.h, rocblas_operation_none, rocblas_operation_none,
        call.n, call.m, call.k, &alpha, call.b, rocblas_datatype_f16_r, call.n, call.a,
        rocblas_datatype_f16_r, call.k, &beta, call.c, call.ct, call.ldc, call.c, call.ct, call.ldc,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none, sols.data(), &count);
    // screen every solution with one timed rep, then re-time the best 4 (and the default)
    std::vector<std::pair<float, int>> t;
    for (int i = 0; i < count; ++i)
        t.push_back({time_ms(call, rocblas_gemm_algo_solution_index, sols[i], stream, 1), sols[i]});
    std::sort(t.begin(), t.end());
    // finalists: best 4 screened + the untuned path, timed in 3 interleaved rounds of 5 reps,
    // min per candidate (clock noise on a shared box made single timings pick losers)
    std::vector<int> fin;
    for (int i = 0; i < (int) t.size() && i < 16 && fin.size() < 4; ++i)
    {
        if (t[i].first >= 1e29f) break;
        if (call.run(rocblas_gemm_algo_solution_index, t[i].second) != rocblas_status_success) continue;
        if (same()) fin.push_back(t[i].second);
    }
    std::vector<float> tm(fin.size(), 1e30f);
    float best_def = 1e30f;
    for (int r = 0; r < 3; ++r)
    {
        best_def = fminf(best_def, def_time(5));   // not std::min: HIP's int min() overload wins
        for (size_t i = 0; i < fin.size(); ++i)
            tm[i] = fminf(tm[i], time_ms(call, rocblas_gemm_algo_solution_index, fin[i], stream, 5));
    }
    float best = best_def * 0.97f;
    int best_sol = 0;
    for (size_t i = 0; i < fin.size(); ++i) if (tm[i] < best) { best = tm[i]; best_sol = fin[i]; }
    if (getenv("EXL3_DENSE_GEMM_TUNE_DEBUG"))
    {
        fprintf(stderr, "[dtune] n=%d k=%d m=%d exact=%d/%d def=%.3f", call.n, call.k, call.m,
                (int) fin.size(), count, best_def);
        for (size_t i = 0; i < fin.size(); ++i) fprintf(stderr, " %d:%.3f", fin[i], tm[i]);
        fprintf(stderr, " -> %d\n", best_sol);
    }
    return best_sol;
}

// true if handled
template <typename D>
static bool try_launch(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream, D&& def_run)
{
    if (!enabled()) return false;
    const int k = a.size(-1);
    const int m = a.numel() / k;
    const int n = b.size(-1);
    if (m < MIN_M) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kHalf && c.dtype() != at::kFloat) return false;
    if (b.dim() != 2 || !b.is_contiguous() || c.stride(-1) != 1) return false;
    int64_t lda = 0;
    if (!a_row_pitch(a, lda)) return false;
    const int64_t ldc = c.stride(-2);
    if (ldc > std::numeric_limits<int>::max()) return false;
    int device;
    cudaGetDevice(&device);
    if (device < 0 || device >= NUM_DEV) return false;

    std::lock_guard<std::mutex> lock(g_mutex);
    load(device);
    const bool f32 = c.dtype() == at::kFloat;
    // bit 1 of the ldc field: padded A pitch (its winner is screened on the padded layout)
    Key key(n, k, m_class(m), (ldc == n ? 1 : 0) | (lda != k ? 2 : 0), f32 ? 1 : 0);
    Call call{handle(device), m, n, k, (int) ldc, (int) lda, a.data_ptr(), b.data_ptr(), c.data_ptr(),
              f32 ? rocblas_datatype_f32_r : rocblas_datatype_f16_r};
    rocblas_set_stream(call.h, stream);
    auto it = g_cache.find(key);
    int sol;
    if (it != g_cache.end()) sol = it->second;
    else
    {
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        hipStreamIsCapturing(stream, &cs);
        if (cs != hipStreamCaptureStatusNone) return false;
        def_run();
        at::Tensor ref = c.clone();
        sol = tune(call, stream, [&](int reps){ return time_fn([&]{ def_run(); return true; }, stream, reps); },
                   [&]{ return c.equal(ref); });
        ref = at::Tensor();
        g_cache[key] = sol;
        store(key, sol);
    }
    if (sol == 0) return false;   // caller runs the untuned path
    // final run writes the real result (tuning reps overwrite c with the same product)
    if (call.run(rocblas_gemm_algo_solution_index, sol) != rocblas_status_success) return false;
    return true;
}

} // namespace dtune

/*
Skinny fp16 GEMM for ROCm: a [m, k] @ b [k, n] -> c [m, n] fp32/fp16, tiny m (decode rows) and
tiny n. hipblaslt answers these shapes (the GDN b/a projections, k=2560 n=48, m<=8) with a
split-K solution plus a PostGSU reduce, ~60 us for a 245 KB weight that streams in ~1 us.
36 GDN layers x 2 calls per forward made that ~6% of decode device time on gfx1151.

Split-K: grid (n/32, k/128) blocks write fp32 partials to a per-device workspace, then a
one-block reduce sums them in fixed order (deterministic, no atomics). The m rows share every
b load (b is the traffic, a is L2-resident), so cost is ~flat in m. A first version with one
block per 32 columns and no k-split was latency-bound at ~85 us (2 blocks on 20 CUs, 320
dependent load rounds per lane) - slower than hipblaslt. Parallelism first, then bandwidth.
*/

namespace skinny
{

constexpr int COLS = 32;         // output columns per block (one lane each)
constexpr int WARPS = 8;         // warps per block, each a k-sub-slice of the block's k-chunk
constexpr int THREADS = WARPS * 32;
constexpr int EXACT_M = 8;       // rows served by one exact-M instantiation (and by skinny_cat)
constexpr int MAX_M = 32;        // 9..32 rows run padded instantiations (12/16/24/32), same per-row reduction
constexpr int MAX_N = 128;       // above this, let hipblaslt have it
constexpr int K_CHUNK = 128;     // k rows per block -> 16 k's per warp, 20 blocks on k=2560
constexpr int MAX_KSPLIT = 64;   // k <= 8192
constexpr int NUM_DEV = MAX_DEVICES;   // from exl3_devctx.cuh

// Partials workspace [KSPLIT][M][N] fp32 per device, allocated once. Stream-ordered use only
// (the partials kernel and the reduce run back to back on the caller's stream), like the
// cuBLAS workspace this path replaces.
static float* g_partials[NUM_DEV] = {};
static int* g_counters[NUM_DEV] = {};   // one per 32-column group, zeroed once; the last block resets its own

static int* counters(int device)
{
    if (device < 0 || device >= NUM_DEV) return nullptr;
    if (!g_counters[device])
    {
        cuda_check(cudaMalloc((void**) &g_counters[device], (MAX_N / COLS) * sizeof(int)));
        cuda_check(cudaMemset(g_counters[device], 0, (MAX_N / COLS) * sizeof(int)));
        cuda_check(cudaDeviceSynchronize());
    }
    return g_counters[device];
}

static float* partials(int device)
{
    if (device < 0 || device >= NUM_DEV) return nullptr;
    if (!g_partials[device])
    {
        cuda_check(cudaMalloc((void**) &g_partials[device], (size_t) MAX_KSPLIT * MAX_M * MAX_N * sizeof(float)));
    }
    return g_partials[device];
}

// grid (ceil(n / 32), ksplit). Block (x, y) covers columns [32x, 32x+32) and k rows
// [K_CHUNK*y, K_CHUNK*(y+1)); warp w takes every WARPS-th k of that chunk so the 8 warps
// stream interleaved 64-byte rows. Each lane holds M fp32 accumulators and issues its loads
// back to back (the k loop is fully unrolled: K_CHUNK / WARPS = 16 independent loads in
// flight per lane), which is what the one-block-per-32-columns version lacked.
//
// ONEPASS (r35): the last block to finish a column group (atomic ticket) sums that group's
// k-split partials itself in skinny_reduce_kernel's fixed y order, so the result is bit-identical
// to the two-launch form without the reduce launch. Every writer fences before the ticket; the
// last block fences (acquire) before reading. It resets its counter for the next launch.
template <int M, bool ONEPASS, typename C_T, int CPL = 1>
__global__ __launch_bounds__(THREADS)
void skinny_partials_kernel
(
    const half* __restrict__ a,   // [M, k], row stride lda
    const half* __restrict__ b,   // [k, n], row stride n
    float* __restrict__ part,     // [ksplit, M, MAX_N]
    const int k,
    const int n,
    const int lda,
    int* __restrict__ cnt,        // ONEPASS: [MAX_N / COLS] tickets
    C_T* __restrict__ c,          // ONEPASS: output [M, ldc]
    const int ldc,
    const int m_rt                // rows really used: M for the exact instantiations (<= 8), <= M for the padded ones
)
{
    // M > EXACT_M: padded instantiation, rows >= m_rt are skipped. A row's accumulation and reduction order do not
    // depend on M or m_rt, so rows 9..32 are bit-identical to the same rows in an exact <= 8-row launch.
    constexpr bool PAD = M > EXACT_M;
    const int mr = PAD ? m_rt : M;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col = blockIdx.x * (COLS * CPL) + lane * CPL;
    const bool active = col < n;
    const int kbase = blockIdx.y * K_CHUNK;

    float acc[M][CPL];
    #pragma unroll
    for (int i = 0; i < M; ++i)
        #pragma unroll
        for (int c = 0; c < CPL; ++c) acc[i][c] = 0.0f;

    if (active)
    {
        constexpr int PER_WARP = K_CHUNK / WARPS;
        float bv[PER_WARP][CPL];
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            if (kk < k)
            {
                if constexpr (CPL == 2)   // 4 B = two adjacent columns; n even and col even are host-checked
                {
                    const half2 v = *reinterpret_cast<const half2*>(b + (size_t) kk * n + col);
                    bv[j][0] = __low2float(v);
                    bv[j][1] = __high2float(v);
                }
                else bv[j][0] = __half2float(b[(size_t) kk * n + col]);
            }
            else
                #pragma unroll
                for (int c = 0; c < CPL; ++c) bv[j][c] = 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            if (kk < k)
            {
                #pragma unroll
                for (int i = 0; i < M; ++i)
                    if (!PAD || i < mr)
                    {
                        const float av = __half2float(a[(size_t) i * lda + kk]);
                        #pragma unroll
                        for (int c = 0; c < CPL; ++c) acc[i][c] = fmaf(av, bv[j][c], acc[i][c]);
                    }
            }
        }
    }

    __shared__ float red[WARPS][M][COLS * CPL + 1];    // 33: coprime with the 32 dword banks
    #pragma unroll
    for (int i = 0; i < M; ++i)
        #pragma unroll
        for (int c = 0; c < CPL; ++c) red[warp][i][lane * CPL + c] = acc[i][c];
    __syncthreads();

    if (warp == 0 && active)
    {
        float* out = part + ((size_t) blockIdx.y * mr) * MAX_N;
        #pragma unroll
        for (int i = 0; i < M; ++i)
        {
            if (PAD && i >= mr) continue;
            #pragma unroll
            for (int c = 0; c < CPL; ++c)
            {
                float s = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) s += red[w][i][lane * CPL + c];
                out[(size_t) i * MAX_N + col + c] = s;
            }
        }
    }

    if constexpr (ONEPASS)
    {
        __shared__ int last_s;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            const int t = atomicAdd(&cnt[blockIdx.x], 1);
            last_s = t == (int) gridDim.y - 1;
        }
        __syncthreads();
        if (!last_s) return;
        __threadfence();
        const int ksplit = gridDim.y;
        for (int idx = threadIdx.x; idx < mr * COLS * CPL; idx += THREADS)
        {
            const int i = idx / (COLS * CPL);
            const int cc = blockIdx.x * (COLS * CPL) + idx % (COLS * CPL);
            if (cc >= n) continue;
            float s = 0.0f;
            for (int y = 0; y < ksplit; ++y)
                s += __builtin_nontemporal_load(&part[((size_t) y * mr + i) * MAX_N + cc]);
            if constexpr (std::is_same_v<C_T, float>)
                c[(size_t) i * ldc + cc] = s;
            else
                c[(size_t) i * ldc + cc] = __float2half(s);
        }
        if (threadIdx.x == 0) cnt[blockIdx.x] = 0;
    }
}

// One block, thread per (row, col): fixed-order sum over the k-splits -> deterministic.
template <typename C_T>
__global__ void skinny_reduce_kernel
(
    const float* __restrict__ part,   // [ksplit, M, MAX_N]
    C_T* __restrict__ c,
    const int m,
    const int n,
    const int ksplit,
    const int ldc
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    const int i = idx / n, col = idx % n;
    float s = 0.0f;
    for (int y = 0; y < ksplit; ++y)
        s += part[((size_t) y * m + i) * MAX_N + col];
    if constexpr (std::is_same_v<C_T, float>)
        c[(size_t) i * ldc + col] = s;
    else
        c[(size_t) i * ldc + col] = __float2half(s);
}

template <bool ONEPASS, typename C_T, int CPL>
static void launch_partials_cpl(int m, const half* a, const half* b, float* part, int k, int n, int lda,
                                dim3 grid, cudaStream_t stream, int* cnt, C_T* c, int ldc)
{
    switch (m)
    {
        case 1: skinny_partials_kernel<1, ONEPASS, C_T, CPL><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); break;
        case 2: skinny_partials_kernel<2, ONEPASS, C_T, CPL><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); break;
        case 3: skinny_partials_kernel<3, ONEPASS, C_T, CPL><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); break;
        case 4: skinny_partials_kernel<4, ONEPASS, C_T, CPL><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); break;
        case 5: skinny_partials_kernel<5, ONEPASS, C_T, CPL><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); break;
        case 6: skinny_partials_kernel<6, ONEPASS, C_T, CPL><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); break;
        case 7: skinny_partials_kernel<7, ONEPASS, C_T, CPL><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); break;
        case 8: skinny_partials_kernel<8, ONEPASS, C_T, CPL><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); break;
        default:
            // rows 9..32 keep the one-column form (rows > 8 are the prefill/rows-scan tail, not the hot path)
            if constexpr (CPL == 1)
            {
                if (m >= 9 && m <= 12) { skinny_partials_kernel<12, ONEPASS, C_T><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); return; }
                if (m >= 13 && m <= 16) { skinny_partials_kernel<16, ONEPASS, C_T><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); return; }
                if (m >= 17 && m <= 24) { skinny_partials_kernel<24, ONEPASS, C_T><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); return; }
                if (m >= 25 && m <= 32) { skinny_partials_kernel<32, ONEPASS, C_T><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc, m); return; }
            }
            else
                launch_partials_cpl<ONEPASS, C_T, 1>(m, a, b, part, k, n, lda, grid, stream, cnt, c, ldc);
            break;
    }
}

template <bool ONEPASS, typename C_T>
static void launch_partials(int m, const half* a, const half* b, float* part, int k, int n, int lda,
                            dim3 grid, cudaStream_t stream, int* cnt = nullptr, C_T* c = nullptr, int ldc = 0,
                            int cpl = 1)
{
    if (cpl == 2) launch_partials_cpl<ONEPASS, C_T, 2>(m, a, b, part, k, n, lda, grid, stream, cnt, c, ldc);
    else launch_partials_cpl<ONEPASS, C_T, 1>(m, a, b, part, k, n, lda, grid, stream, cnt, c, ldc);
}

// EXL3_SKINNY_W2 (default 0 = inert): two ADJACENT columns per lane, so a warp's weight load is one
// 128 B line instead of half of one (the neighbouring column group re-reads the other half) and half
// the load instructions. Bit-identical: a column's k order, warp-sum order and k-split sum are the
// same values in the same order; only which lane owns which column changes.
// DEFAULT ON (a micro on the GLM KDA cat shape: 25.88 -> 21.55 us/call, -16.7 %, and no measured shape slower;
// Qwen's b/a cat -8.6 %, n=128 projections -3.4 %, n=32 untouched). Read per call (a getenv is ~0.1 us against
// ~70 skinny calls per decode step) so a per-arm env flip in ONE loaded process switches it; EXL3_SKINNY_W2=0
// turns it off and skinny_set_w2 forces either form without touching the environment.
static int g_w2 = -1;
static bool w2on()
{
    if (g_w2 >= 0) return g_w2 == 1;
    const char* e = getenv("EXL3_SKINNY_W2");
    return !(e && *e == '0');
}

// EXL3_SKINNY_1PASS (default 1): fold the reduce into the partials launch. Runtime-settable
// (skinny_set_onepass) so one process can A/B it; graphs bake the choice at capture.
static int g_onepass = -1;
static bool onepass()
{
    if (g_onepass < 0)
    {
        const char* e = getenv("EXL3_SKINNY_1PASS");
        g_onepass = (e && *e == '0') ? 0 : 1;
    }
    return g_onepass == 1;
}

static bool enabled()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* e = getenv("EXL3_HIP_SKINNY_GEMM");
        cached = (e && *e == '0') ? 0 : 1;
    }
    return cached == 1;
}

// Rows above this are left to the caller's GEMM (EXL3_HIP_SKINNY_MAX_M, default MAX_M; 8 = the old behaviour).
static int max_rows()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* e = getenv("EXL3_HIP_SKINNY_MAX_M");
        int v = e && *e ? atoi(e) : MAX_M;
        cached = v < 1 ? 1 : (v > MAX_M ? MAX_M : v);
    }
    return cached;
}

// true if handled
static bool try_launch(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream)
{
    if (!enabled()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kFloat && c.dtype() != at::kHalf) return false;
    if (b.dim() != 2 || !b.is_contiguous()) return false;
    const int k = a.size(-1);
    const int m = a.numel() / k;
    const int n = b.size(-1);
    if (m < 1 || m > max_rows() || n < 1 || n > MAX_N || k < 1) return false;
    const int ksplit = (k + K_CHUNK - 1) / K_CHUNK;
    if (ksplit > MAX_KSPLIT) return false;
    if (a.stride(-1) != 1 || c.stride(-1) != 1) return false;
    if (a.dim() > 2 && !a.is_contiguous()) return false;
    const int lda = a.dim() == 1 ? k : (int) a.stride(-2);
    const int ldc = (int) c.stride(-2);
    float* part = partials(a.device().index());
    if (!part) return false;
    int* cnt = counters(a.device().index());   // allocated on the first (eager) call, never under capture

    // W2: two ADJACENT columns per lane needs an even row stride (4 B aligned pair) and one full warp of pairs.
    // Rows 9..32 use the padded one-column instantiations (the branch below); sizing the grid
    // for CPL=2 there left columns 64..127 unwritten because that path falls back to CPL=1.
    const int cpl = (w2on() && m <= EXACT_M && (n % 2 == 0) && n >= 2 * COLS) ? 2 : 1;
    dim3 grid((n + COLS * cpl - 1) / (COLS * cpl), ksplit);
    if (onepass())
    {
        if (cnt)
        {
            if (c.dtype() == at::kFloat)
                launch_partials<true, float>(m, (const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, grid, stream,
                                             cnt, (float*) c.data_ptr(), ldc, cpl);
            else
                launch_partials<true, half>(m, (const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, grid, stream,
                                            cnt, (half*) c.data_ptr(), ldc, cpl);
            cuda_check(cudaPeekAtLastError());
            return true;
        }
    }
    launch_partials<false, float>(m, (const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, grid, stream,
                                  nullptr, nullptr, 0, cpl);
    const int total = m * n;
    const int rthreads = total < 256 ? ((total + 31) / 32) * 32 : 256;
    const int rblocks = (total + rthreads - 1) / rthreads;
    if (c.dtype() == at::kFloat)
        skinny_reduce_kernel<float><<<rblocks, rthreads, 0, stream>>>(part, (float*) c.data_ptr(), m, n, ksplit, ldc);
    else
        skinny_reduce_kernel<half><<<rblocks, rthreads, 0, stream>>>(part, (half*) c.data_ptr(), m, n, ksplit, ldc);
    cuda_check(cudaPeekAtLastError());
    return true;
}

// skinny_cat (kda-dec): several small-N projections of the SAME a, concatenated along n into one
// weight, in ONE launch. Each column's reduction order is exactly skinny_partials_kernel's
// (16 interleaved k per warp, warps summed 0..7, k-splits summed in y order), independent of n,
// so every output is bit-identical to a separate skinny launch per projection. Each column
// range [start[s], start[s+1]) lands in its own [M, width] output, fp16 (RNE) or fp32.
constexpr int CAT_MAX_N = 512;
constexpr int CAT_MAX_SEG = 4;
static float* g_cat_partials[NUM_DEV] = {};
static int* g_cat_counters[NUM_DEV] = {};

struct CatSeg
{
    void* ptr[CAT_MAX_SEG];
    const half* w[CAT_MAX_SEG];   // segment s weight, column 0 (concat: b + start[s]; separate: its own tensor)
    int ldw[CAT_MAX_SEG];         // row stride of w[s] (concat: n; separate: segment width)
    int is_half[CAT_MAX_SEG];
    int start[CAT_MAX_SEG + 1];
    int nseg;
};

template <int M, int CPL = 1>
__global__ __launch_bounds__(THREADS)
void skinny_cat_kernel
(
    const half* __restrict__ a,
    float* __restrict__ part,     // [ksplit, M, CAT_MAX_N]
    const int k,
    const int n,
    const int lda,
    int* __restrict__ cnt,
    const CatSeg seg
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col = blockIdx.x * (COLS * CPL) + lane * CPL;
    const bool active = col < n;
    const int kbase = blockIdx.y * K_CHUNK;

    float acc[M][CPL];
    #pragma unroll
    for (int i = 0; i < M; ++i)
        #pragma unroll
        for (int c = 0; c < CPL; ++c) acc[i][c] = 0.0f;

    if (active)
    {
        constexpr int PER_WARP = K_CHUNK / WARPS;
        int cs = 0;
        while (cs + 1 < seg.nseg && col >= seg.start[cs + 1]) ++cs;
        const half* bw = seg.w[cs] + (col - seg.start[cs]);
        const int ldb = seg.ldw[cs];
        float bv[PER_WARP][CPL];
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            if (kk < k)
            {
                if constexpr (CPL == 2)   // the pair stays inside one segment: host checks even widths/strides
                {
                    const half2 v = *reinterpret_cast<const half2*>(bw + (size_t) kk * ldb);
                    bv[j][0] = __low2float(v);
                    bv[j][1] = __high2float(v);
                }
                else bv[j][0] = __half2float(bw[(size_t) kk * ldb]);
            }
            else
                #pragma unroll
                for (int c = 0; c < CPL; ++c) bv[j][c] = 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            if (kk < k)
            {
                #pragma unroll
                for (int i = 0; i < M; ++i)
                {
                    const float av = __half2float(a[(size_t) i * lda + kk]);
                    #pragma unroll
                    for (int c = 0; c < CPL; ++c) acc[i][c] = fmaf(av, bv[j][c], acc[i][c]);
                }
            }
        }
    }

    __shared__ float red[WARPS][M][COLS * CPL + 1];
    #pragma unroll
    for (int i = 0; i < M; ++i)
        #pragma unroll
        for (int c = 0; c < CPL; ++c) red[warp][i][lane * CPL + c] = acc[i][c];
    __syncthreads();

    if (warp == 0 && active)
    {
        float* out = part + ((size_t) blockIdx.y * M) * CAT_MAX_N;
        #pragma unroll
        for (int i = 0; i < M; ++i)
        {
            #pragma unroll
            for (int c = 0; c < CPL; ++c)
            {
                float s = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) s += red[w][i][lane * CPL + c];
                out[(size_t) i * CAT_MAX_N + col + c] = s;
            }
        }
    }

    __shared__ int last_s;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int t = atomicAdd(&cnt[blockIdx.x], 1);
        last_s = t == (int) gridDim.y - 1;
    }
    __syncthreads();
    if (!last_s) return;
    __threadfence();
    const int ksplit = gridDim.y;
    for (int idx = threadIdx.x; idx < M * COLS * CPL; idx += THREADS)
    {
        const int i = idx / (COLS * CPL);
        const int cc = blockIdx.x * (COLS * CPL) + idx % (COLS * CPL);
        if (cc >= n) continue;
        float s = 0.0f;
        for (int y = 0; y < ksplit; ++y)
            s += __builtin_nontemporal_load(&part[((size_t) y * M + i) * CAT_MAX_N + cc]);
        int sg = 0;
        while (sg + 1 < seg.nseg && cc >= seg.start[sg + 1]) ++sg;
        const int w = seg.start[sg + 1] - seg.start[sg];
        const int lc = cc - seg.start[sg];
        if (seg.is_half[sg])
            ((half*) seg.ptr[sg])[(size_t) i * w + lc] = __float2half(s);
        else
            ((float*) seg.ptr[sg])[(size_t) i * w + lc] = s;
    }
    if (threadIdx.x == 0) cnt[blockIdx.x] = 0;
}

} // namespace skinny

// a: [M, k] fp16 (M <= 8, contiguous rows). Weights either one concatenated b: [k, n] (skinny_cat) or
// one contiguous [k, w_s] tensor per output (skinny_cat_w: no concat copy, same values in the same order,
// so bit-identical). n = sum(w_s) <= 512, k <= 8192. outs: contiguous [M, w_s] fp16 or fp32, column order.
static void skinny_cat_impl(at::Tensor a, const at::Tensor* b, const std::vector<at::Tensor>* ws,
                            std::vector<at::Tensor> outs)
{
    using namespace skinny;
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK(a.dim() >= 1 && a.size(-1) >= 1, "skinny_cat: a must have non-zero K");
    const int k = a.size(-1);
    const int m = a.numel() / k;
    TORCH_CHECK(a.is_contiguous(), "skinny_cat: a must be contiguous");
    TORCH_CHECK((int) outs.size() >= 1 && (int) outs.size() <= CAT_MAX_SEG, "skinny_cat: 1..4 outputs");
    TORCH_CHECK(!ws || ws->size() == outs.size(), "skinny_cat_w: one weight per output");
    if (m == 0) return;
    CatSeg seg;
    seg.nseg = (int) outs.size();
    int acc = 0;
    for (int s = 0; s < seg.nseg; ++s)
    {
        const at::Tensor& o = outs[s];
        TORCH_CHECK(o.is_contiguous(), "skinny_cat: outputs must be contiguous");
        TORCH_CHECK(o.dtype() == at::kHalf || o.dtype() == at::kFloat, "skinny_cat: outputs fp16/fp32");
        const int w = (int) (o.numel() / m);
        TORCH_CHECK((int64_t) w * m == o.numel() && o.size(-1) == w, "skinny_cat: output shape");
        seg.ptr[s] = o.data_ptr();
        seg.is_half[s] = o.dtype() == at::kHalf;
        seg.start[s] = acc;
        if (ws)
        {
            const at::Tensor& ww = (*ws)[s];
            TORCH_CHECK_DTYPE(ww, kHalf);
            TORCH_CHECK(ww.dim() == 2 && ww.is_contiguous() && ww.size(0) == k && ww.size(1) == w,
                        "skinny_cat_w: weight s must be contiguous [k, w_s]");
            seg.w[s] = (const half*) ww.data_ptr();
            seg.ldw[s] = w;
        }
        acc += w;
    }
    seg.start[seg.nseg] = acc;
    const int n = acc;
    if (b)
    {
        TORCH_CHECK_DTYPE((*b), kHalf);
        TORCH_CHECK(b->dim() == 2 && b->is_contiguous(), "skinny_cat: b must be contiguous [k, n]");
        TORCH_CHECK(b->size(0) == k, "skinny_cat: k mismatch");
        TORCH_CHECK(b->size(1) == n, "skinny_cat: output widths must sum to n");
        for (int s = 0; s < seg.nseg; ++s)
        {
            seg.w[s] = (const half*) b->data_ptr() + seg.start[s];
            seg.ldw[s] = n;
        }
    }
    TORCH_CHECK(m >= 1 && m <= EXACT_M && n >= 1 && n <= CAT_MAX_N, "skinny_cat: shape out of range");
    const int ksplit = (k + K_CHUNK - 1) / K_CHUNK;
    TORCH_CHECK(ksplit <= MAX_KSPLIT, "skinny_cat: k too large");

    const int dev = a.device().index();
    TORCH_CHECK(dev >= 0 && dev < NUM_DEV, "skinny_cat: device");
    if (!g_cat_partials[dev])
        cuda_check(cudaMalloc((void**) &g_cat_partials[dev], (size_t) MAX_KSPLIT * EXACT_M * CAT_MAX_N * sizeof(float)));
    if (!g_cat_counters[dev])
    {
        cuda_check(cudaMalloc((void**) &g_cat_counters[dev], (CAT_MAX_N / COLS) * sizeof(int)));
        cuda_check(cudaMemset(g_cat_counters[dev], 0, (CAT_MAX_N / COLS) * sizeof(int)));
        cuda_check(cudaDeviceSynchronize());
    }
    // W2: two ADJACENT columns per lane; needs every segment width and row stride even (the pair stays
    // inside one segment and bw + kk*ldb stays 4 B aligned) and one full warp of pairs.
    int cpl = 1;
    if (w2on() && n % 2 == 0 && n >= 2 * COLS)
    {
        cpl = 2;
        for (int s = 0; s < seg.nseg; ++s)
            if ((seg.start[s + 1] - seg.start[s]) % 2 || seg.ldw[s] % 2) { cpl = 1; break; }
    }
    dim3 grid((n + COLS * cpl - 1) / (COLS * cpl), ksplit);
    const half* ap = (const half*) a.data_ptr();
    float* pp = g_cat_partials[dev];
    int* ccp = g_cat_counters[dev];
    // one launch per macro line: hipify rewrites the <<< >>> form in place and mangles multi-line macros
    #define SC1(MM) case MM: skinny_cat_kernel<MM><<<grid, THREADS, 0, stream>>>(ap, pp, k, n, k, ccp, seg); break;
    #define SC2(MM) case MM: skinny_cat_kernel<MM, 2><<<grid, THREADS, 0, stream>>>(ap, pp, k, n, k, ccp, seg); break;
    if (cpl == 2)
    {
        switch (m)
        {
            SC2(1) SC2(2) SC2(3) SC2(4) SC2(5) SC2(6) SC2(7) SC2(8)
            default: break;
        }
    }
    else
    {
        switch (m)
        {
            SC1(1) SC1(2) SC1(3) SC1(4) SC1(5) SC1(6) SC1(7) SC1(8)
            default: break;
        }
    }
    #undef SC1
    #undef SC2
    cuda_check(cudaPeekAtLastError());
}

void skinny_cat(at::Tensor a, at::Tensor b, std::vector<at::Tensor> outs) { skinny_cat_impl(a, &b, nullptr, outs); }
void skinny_cat_w(at::Tensor a, std::vector<at::Tensor> ws, std::vector<at::Tensor> outs) { skinny_cat_impl(a, nullptr, &ws, outs); }

void skinny_set_onepass(int64_t on) { skinny::g_onepass = on ? 1 : 0; }
void skinny_set_w2(int64_t on) { skinny::g_w2 = on ? 1 : 0; }

#endif // USE_ROCM

#if !defined(USE_ROCM)

/*
Tuned dense GEMM for CUDA prefill (EXL3_CUDA_GEMM_TUNE, default on; 0 = off, read on every call so an in-process A/B
can toggle it). cublasGemmEx's heuristic picks a 128x128 tile for some large prefill shapes on Ampere: Qwen3.8-27B
mlp down (k=17408, n=5120, 4096 rows) runs at ~100 TFLOPS on a 3090 against ~115 for a 256x128 / 128x256 cuBLASLt
config of the same shape (-13 %). Per (n, k, m class, layout, out dtype) key, every cuBLASLt config (algo id x tile x
stages x CTA swizzle, split-K 1, tiles of at least 64x128) is screened on the live operands with two reps; the 24
fastest are run once more and only the ones whose output is BIT-IDENTICAL to cublasGemmEx's stay (no split-K, same
fp32 accumulation order: every big-tile config of these shapes matched); the best 6 are re-timed against cublasGemmEx
in 5 interleaved rounds (device drained before each) and must beat it by 3 %. The winner (or 0 = untuned) is kept in memory and appended to a cache
file keyed by the cuBLASLt version and the device, so a process tunes a shape class once ever. Never tunes under
stream capture or inside a graph; any failure falls back to cublasGemmEx.
*/

#include <cublasLt.h>
#include <algorithm>
#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <mutex>
#include <cstdio>
#include <sys/stat.h>

namespace ltune
{

constexpr int MIN_M = 256;             // rows below this: untuned path (decode, verify, short tails)
constexpr int NUM_DEV = MAX_DEVICES;

typedef std::tuple<int, int, int, int, int> Key;   // n, k, m class, layout bits, out fp32
struct Cfg { int id, tile, stages, swz; };
static std::map<Key, Cfg> g_cache;     // id < 0: cublasGemmEx wins
static bool g_loaded = false;
static std::mutex g_mutex;
static cublasLtHandle_t g_handle[NUM_DEV] = {};
static std::string g_tag;

static bool enabled()
{
    const char* e = getenv("EXL3_CUDA_GEMM_TUNE");
    return !e || *e != '0';   // on by default: winners are bit-exact with the untuned path
}

static int m_class(int m) { return ((m + 255) / 256) * 256; }

static std::string cache_path()
{
    const char* p = getenv("EXL3_CUDA_GEMM_TUNE_FILE");
    if (p && *p) return p;
    const char* h = getenv("HOME");
    std::string d = std::string(h ? h : "/tmp") + "/.cache/exllamav3";
    mkdir((std::string(h ? h : "/tmp") + "/.cache").c_str(), 0755);
    mkdir(d.c_str(), 0755);
    return d + "/cuda_gemm_tune.txt";
}

static void load(int device)
{
    if (g_loaded) return;
    g_loaded = true;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    g_tag = "lt" + std::to_string(cublasLtGetVersion()) + "|" + prop.name + "|sm" + std::to_string(prop.major * 10 + prop.minor) + "|exact1";
    for (auto& ch : g_tag) if (ch == ' ') ch = '_';
    FILE* f = fopen(cache_path().c_str(), "r");
    if (!f) return;
    char tag[512]; int n, k, mc, lay, f32, id, tile, stages, swz;
    while (fscanf(f, "%511s %d %d %d %d %d %d %d %d %d", tag, &n, &k, &mc, &lay, &f32, &id, &tile, &stages, &swz) == 10)
        if (g_tag == tag) g_cache[Key(n, k, mc, lay, f32)] = Cfg{id, tile, stages, swz};
    fclose(f);
}

static void store(const Key& key, const Cfg& c)
{
    FILE* f = fopen(cache_path().c_str(), "a");
    if (!f) return;
    fprintf(f, "%s %d %d %d %d %d %d %d %d %d\n", g_tag.c_str(), std::get<0>(key), std::get<1>(key),
            std::get<2>(key), std::get<3>(key), std::get<4>(key), c.id, c.tile, c.stages, c.swz);
    fclose(f);
}

static cublasLtHandle_t handle(int device)
{
    if (!g_handle[device]) cublasLtCreate(&g_handle[device]);
    return g_handle[device];
}

// tile edge product; 0 = unknown / too small to win a large prefill GEMM
static int tile_area(int t)
{
    switch (t)
    {
        case CUBLASLT_MATMUL_TILE_64x128: case CUBLASLT_MATMUL_TILE_128x64: return 64 * 128;
        case CUBLASLT_MATMUL_TILE_64x256: case CUBLASLT_MATMUL_TILE_256x64: return 64 * 256;
        case CUBLASLT_MATMUL_TILE_128x128: return 128 * 128;
        case CUBLASLT_MATMUL_TILE_64x512: case CUBLASLT_MATMUL_TILE_512x64: return 64 * 512;
        case CUBLASLT_MATMUL_TILE_128x256: case CUBLASLT_MATMUL_TILE_256x128: return 128 * 256;
        case CUBLASLT_MATMUL_TILE_256x256: return 256 * 256;
        default: return 0;
    }
}

// row-major C[m, n] = A[m, k] @ B[k, n] as column-major C^T = B^T A^T
struct Call
{
    cublasLtHandle_t h;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t lb = nullptr, la = nullptr, lc = nullptr;
    const void* a; const void* b; void* c;
    void* ws; size_t ws_size;
    Call(cublasLtHandle_t h_, int m, int n, int k, int lda, int ldc, bool f32, const void* a_, const void* b_, void* c_, void* ws_)
        : h(h_), a(a_), b(b_), c(c_), ws(ws_), ws_size(WORKSPACE_SIZE)
    {
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, n, k, n);
        cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, k, m, lda);
        cublasLtMatrixLayoutCreate(&lc, f32 ? CUDA_R_32F : CUDA_R_16F, n, m, ldc);
    }
    ~Call()
    {
        if (lc) cublasLtMatrixLayoutDestroy(lc);
        if (la) cublasLtMatrixLayoutDestroy(la);
        if (lb) cublasLtMatrixLayoutDestroy(lb);
        if (op) cublasLtMatmulDescDestroy(op);
    }
    bool algo(const Cfg& g, cudaDataType_t ct, cublasLtMatmulAlgo_t& al) const
    {
        if (cublasLtMatmulAlgoInit(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, ct, ct, g.id, &al) != CUBLAS_STATUS_SUCCESS) return false;
        int one = 1;
        cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_TILE_ID, &g.tile, sizeof(g.tile));
        cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_STAGES_ID, &g.stages, sizeof(g.stages));
        cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &g.swz, sizeof(g.swz));
        cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof(one));
        cublasLtMatmulHeuristicResult_t hr;
        if (cublasLtMatmulAlgoCheck(h, op, lb, la, lc, lc, &al, &hr) != CUBLAS_STATUS_SUCCESS) return false;
        return hr.workspaceSize <= ws_size;
    }
    bool run(const cublasLtMatmulAlgo_t& al, cudaStream_t s) const
    {
        float alpha = 1.0f, beta = 0.0f;
        return cublasLtMatmul(h, op, &alpha, b, lb, a, la, &beta, c, lc, c, lc, &al, ws, ws_size, s) == CUBLAS_STATUS_SUCCESS;
    }
};

template <typename F>
static float time_fn(F&& fn, cudaStream_t stream, int reps)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    if (!fn()) { cudaEventDestroy(e0); cudaEventDestroy(e1); return 1e30f; }
    cudaEventRecord(e0, stream);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, e0, e1) != cudaSuccess) ms = 1e30f * reps;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / reps;
}

template <typename T, typename S>
static Cfg tune(const Call& call, cudaDataType_t ct, cudaStream_t stream, int m, int n, int k, T&& def_time, S&& same)
{
    Cfg none{-1, 0, 0, 0};
    int ids[128]; int nid = 0;
    if (cublasLtMatmulAlgoGetIds(call.h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, ct, ct, 128, ids, &nid)
        != CUBLAS_STATUS_SUCCESS || nid <= 0) return none;
    // screen: every config with a big enough tile, two timed reps. Work queued on other streams (the side streams of
    // a server) would land inside the event window and pick losers: drain the device first, and before each round.
    cudaDeviceSynchronize();
    std::vector<std::pair<float, Cfg>> t;
    for (int ii = 0; ii < nid; ++ii)
    {
        cublasLtMatmulAlgo_t al;
        if (cublasLtMatmulAlgoInit(call.h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, ct, ct, ids[ii], &al) != CUBLAS_STATUS_SUCCESS) continue;
        size_t sz = 0;
        std::vector<int> tiles, stages;
        cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz);
        if (!sz) continue;
        tiles.resize(sz / sizeof(int));
        cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz);
        sz = 0; cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz);
        if (sz) { stages.resize(sz / sizeof(int)); cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_STAGES_IDS, stages.data(), sz, &sz); }
        else stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
        int swmax = 0;
        cublasLtMatmulAlgoCapGetAttribute(&al, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swmax, sizeof(swmax), &sz);
        for (int tile : tiles)
        {
            if (tile_area(tile) < 64 * 128) continue;
            for (int st : stages) for (int sw = 0; sw <= (swmax ? 1 : 0); ++sw)
            {
                Cfg g{ids[ii], tile, st, sw};
                cublasLtMatmulAlgo_t a2;
                if (!call.algo(g, ct, a2)) continue;
                float ms = time_fn([&]{ return call.run(a2, stream); }, stream, 2);
                cudaGetLastError();
                if (ms < 1e29f) t.push_back({ms, g});
            }
        }
    }
    std::sort(t.begin(), t.end(), [](const std::pair<float, Cfg>& x, const std::pair<float, Cfg>& y) { return x.first < y.first; });
    // finalists: up to 6 of the 24 fastest whose output is bit-identical to cublasGemmEx
    std::vector<Cfg> fin;
    for (int i = 0; i < (int) t.size() && i < 24 && fin.size() < 6; ++i)
    {
        cublasLtMatmulAlgo_t al;
        if (!call.algo(t[i].second, ct, al) || !call.run(al, stream)) continue;
        if (same()) fin.push_back(t[i].second);
    }
    std::vector<float> tm(fin.size(), 1e30f);
    float best_def = 1e30f;
    for (int r = 0; r < 5; ++r)
    {
        cudaDeviceSynchronize();
        best_def = fminf(best_def, def_time(5));
        for (size_t i = 0; i < fin.size(); ++i)
        {
            cublasLtMatmulAlgo_t al;
            if (!call.algo(fin[i], ct, al)) continue;
            tm[i] = fminf(tm[i], time_fn([&]{ return call.run(al, stream); }, stream, 5));
        }
    }
    float best = best_def * 0.97f;
    Cfg win = none;
    for (size_t i = 0; i < fin.size(); ++i) if (tm[i] < best) { best = tm[i]; win = fin[i]; }
    if (getenv("EXL3_CUDA_GEMM_TUNE_DEBUG"))
    {
        fprintf(stderr, "[ltune] n=%d k=%d m=%d screened=%d exact=%d def=%.3f", n, k, m, (int) t.size(), (int) fin.size(), best_def);
        for (size_t i = 0; i < fin.size(); ++i) fprintf(stderr, " %d/%d/%d/%d:%.3f", fin[i].id, fin[i].tile, fin[i].stages, fin[i].swz, tm[i]);
        fprintf(stderr, " -> %d/%d/%d/%d\n", win.id, win.tile, win.stages, win.swz);
    }
    return win;
}

// true if handled
template <typename D>
static bool try_launch(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream, D&& def_run)
{
    if (!enabled()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kHalf && c.dtype() != at::kFloat) return false;
    if (b.dim() != 2 || !b.is_contiguous() || c.dim() < 2 || c.stride(-1) != 1) return false;
    const int k = a.size(-1);
    if (k < 1) return false;
    const int m = a.numel() / k;
    const int n = b.size(-1);
    if (m < MIN_M || b.size(0) != k || c.size(-1) != n) return false;
    int64_t lda = 0;
    if (!a_row_pitch(a, lda)) return false;
    const int64_t ldc = c.stride(-2);
    if (ldc < n || ldc > std::numeric_limits<int>::max()) return false;
    if (c.numel() / n != m) return false;
    int device;
    cudaGetDevice(&device);
    if (device < 0 || device >= NUM_DEV) return false;

    std::lock_guard<std::mutex> lock(g_mutex);
    load(device);
    const bool f32 = c.dtype() == at::kFloat;
    const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16F;
    Key key(n, k, m_class(m), (ldc == n ? 1 : 0) | (lda != k ? 2 : 0), f32 ? 1 : 0);
    Call call(handle(device), m, n, k, (int) lda, (int) ldc, f32, a.data_ptr(), b.data_ptr(), c.data_ptr(),
              DevCtx::instance().get_ws(device));
    auto it = g_cache.find(key);
    Cfg g;
    if (it != g_cache.end()) g = it->second;
    else
    {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cs);
        if (cs != cudaStreamCaptureStatusNone) return false;
        def_run();
        at::Tensor ref = c.clone();
        g = tune(call, ct, stream, m, n, k, [&](int reps){ return time_fn([&]{ def_run(); return true; }, stream, reps); },
                 [&]{ return c.equal(ref); });
        ref = at::Tensor();
        g_cache[key] = g;
        store(key, g);
        cudaGetLastError();
    }
    if (g.id < 0) return false;   // caller runs cublasGemmEx
    cublasLtMatmulAlgo_t al;
    if (!call.algo(g, ct, al)) return false;
    return call.run(al, stream);   // the final run writes the real result (tuning reps wrote the same product)
}

} // namespace ltune

#endif // !USE_ROCM

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
#if defined(USE_ROCM)
    if (skinny::try_launch(a, b, c, stream)) return;
    if (!graph && dtune::try_launch(a, b, c, stream, [&]{ hgemm_gemmex_impl(a, b, c, stream); })) return;
#else
    if (!graph && ltune::try_launch(a, b, c, stream, [&]{ hgemm_gemmex_impl(a, b, c, stream); })) return;
#endif
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}

/*
Strided-batched row-major matmul, a[b] @ w[b] -> c[b] for b in [0, B), fp16 inputs with fp32
accumulation (same cuBLAS setup as hgemm). a: [B, m, k], w: [B, k, n], c: [B, m, n], all
contiguous; c fp16 or fp32. Used by the batched expert reconstruct path (moe_batch_recon.py).
*/
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
{
    // Reconstruct-path GEMM: the fp16-accumulator kernel where it pays (GeForce), else cuBLAS
    if (hgemm_f16acc_try(a, w, c)) return;

    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    bool output_fp32 = c.dtype() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.dtype() == at::kHalf, "hgemm_batched: c must be float32 or float16");
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(), "hgemm_batched: tensors must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w.data_ptr(), CUDA_R_16F, size_n, (long long) size_k * size_n,
                 a.data_ptr(), CUDA_R_16F, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), output_fp32 ? CUDA_R_32F : CUDA_R_16F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}
