// HIP (ROCm) port of cpu/moe_handoff.cu: GPU publish/wait flag kernels over the parent's
// registered SHM region and the persistent CPU worker loop. Compiled only for ROCm builds
// (the whole TU is empty elsewhere); cpu/moe_handoff.cu remains the CUDA implementation.
//
// Design constraints on ROCm:
//  - Kernel waits only. hipStreamWaitValue32 requires a hipMallocSignalMemory pointer and
//    that allocation fails on gfx1201, so the CUDA build's cuStreamWaitValue32 memops path
//    is NOT ported. exl3_moe_cpu_set_memops is an accepted no-op so the Python side stays
//    uniform.
//  - Release/acquire flag pairing at system scope: __hip_atomic_load/store with
//    __HIP_MEMORY_SCOPE_SYSTEM.
//  - __nanosleep is undeclared on the supported ROCm toolchains and __builtin_amdgcn_s_sleep
//    rejects runtime arguments: backoff uses literal immediate buckets only.
//  - s_sleep is cycle-based, so the ~30 s wait timeout maps to a poll budget calibrated at
//    startup by measuring the kernel's no-publisher cadence once. On timeout the wait kernel
//    release-stores abort=1 and returns, so a stream can never hang on a dead/hung worker;
//    the host-side watchdog remains the recovery path.
//  - Bounded-failure contract: a timed-out wait sets the abort flag; the current request
//    fails at the next host-side abort check (per-step sampling sync).

#if defined(USE_ROCM)

#include "moe_handoff.h"
#include "moe_mul1.h"

#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <thread>

// -------------------------------------------------------------------------------------------
//   Device helpers: system-scope release/acquire on mapped host memory
// -------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t moe_acquire_sys_u32(const uint32_t* p)
{
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void moe_release_sys_u32(uint32_t* p, uint32_t v)
{
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// One backoff epoch of literal immediate sleeps (immediate operands only on gfx1201):
// 8 * (64 + 128 + 256 + 512 + 1024 + 2048) sleeps per epoch.
__device__ __forceinline__ void moe_backoff_epoch_hip()
{
    #pragma unroll
    for (int i = 0; i < 8; ++i) __builtin_amdgcn_s_sleep(64);
    #pragma unroll
    for (int i = 0; i < 8; ++i) __builtin_amdgcn_s_sleep(128);
    #pragma unroll
    for (int i = 0; i < 8; ++i) __builtin_amdgcn_s_sleep(256);
    #pragma unroll
    for (int i = 0; i < 8; ++i) __builtin_amdgcn_s_sleep(512);
    #pragma unroll
    for (int i = 0; i < 8; ++i) __builtin_amdgcn_s_sleep(1024);
    #pragma unroll
    for (int i = 0; i < 8; ++i) __builtin_amdgcn_s_sleep(2048);
}

// GEQ cyclic predicate, hard-spin phase first, then backoff epochs until the poll budget
// is exhausted; on timeout, release-store abort=1 and return.
__device__ __forceinline__ bool moe_wait_geq_hip
(
    const uint32_t* flag,
    uint32_t value,
    uint32_t* abort_flag,
    unsigned long long max_polls
)
{
    for (int i = 0; i < 64; ++i)
    {
        uint32_t v = moe_acquire_sys_u32(flag);
        if ((int32_t)(v - value) >= 0) return true;
    }
    unsigned long long polls = 64;
    while (true)
    {
        moe_backoff_epoch_hip();
        uint32_t v = moe_acquire_sys_u32(flag);
        if ((int32_t)(v - value) >= 0) return true;
        if (++polls >= max_polls)
        {
            moe_release_sys_u32(abort_flag, 1u);
            return false;
        }
    }
}

__global__ void moe_flag_write_kernel_hip(uint32_t* flag, uint32_t value)
{
    moe_release_sys_u32(flag, value);
}

__global__ void moe_flag_wait_kernel_hip
(
    uint32_t* flag,
    uint32_t value,
    uint32_t* abort_flag,
    unsigned long long max_polls
)
{
    moe_wait_geq_hip(flag, value, abort_flag, max_polls);
}

// Fine-grained variant (EXL3_MOE_HIP_WAIT_FINE=1): one short s_sleep per poll instead of a
// ~32K-cycle backoff epoch. The epoch was sized for multi-second GPU-side waits on gfx1201;
// in the per-layer expert split the worker answers in ~0.2 ms, so a coarse epoch adds up to a
// full epoch of latency to every layer of every decode step. Budget is calibrated separately.
__device__ __forceinline__ bool moe_wait_geq_fine_hip
(
    const uint32_t* flag,
    uint32_t value,
    uint32_t* abort_flag,
    unsigned long long max_polls
)
{
    unsigned long long polls = 0;
    while (true)
    {
        uint32_t v = moe_acquire_sys_u32(flag);
        if ((int32_t)(v - value) >= 0) return true;
        __builtin_amdgcn_s_sleep(2);
        if (++polls >= max_polls)
        {
            moe_release_sys_u32(abort_flag, 1u);
            return false;
        }
    }
}

__global__ void moe_flag_wait_fine_kernel_hip
(
    uint32_t* flag,
    uint32_t value,
    uint32_t* abort_flag,
    unsigned long long max_polls
)
{
    moe_wait_geq_fine_hip(flag, value, abort_flag, max_polls);
}

// -------------------------------------------------------------------------------------------
//   Poll-budget calibration: s_sleep is cycle-based, so convert the wait timeout (ns) to a
//   poll budget by measuring the kernel once. Runs on the first flag_wait enqueue; costs one
//   small allocation and one bounded wait (~tens of ms) at startup.
// -------------------------------------------------------------------------------------------

static std::atomic<bool> g_polls_ready { false };
static std::atomic<unsigned long long> g_max_polls { 0 };
static std::atomic<bool> g_polls_ready_fine { false };
static std::atomic<unsigned long long> g_max_polls_fine { 0 };
static constexpr unsigned long long MOE_WAIT_TIMEOUT_NS = 30000000000ull;

static bool moe_wait_fine_enabled()
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("EXL3_MOE_HIP_WAIT_FINE"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v == 1;
}

static unsigned long long moe_wait_poll_budget_impl(hipStream_t stream, bool fine);

static unsigned long long moe_wait_poll_budget(hipStream_t stream)
{
    return moe_wait_poll_budget_impl(stream, moe_wait_fine_enabled());
}

static unsigned long long moe_wait_poll_budget_impl(hipStream_t stream, bool fine)
{
    std::atomic<unsigned long long>& g_max = fine ? g_max_polls_fine : g_max_polls;
    std::atomic<bool>& g_ready = fine ? g_polls_ready_fine : g_polls_ready;
    unsigned long long polls = g_max.load(std::memory_order_relaxed);
    if (polls) return polls;
    if (g_ready.exchange(true, std::memory_order_acq_rel))
    {
        // Another thread is calibrating (or failed); fall back to a conservative estimate:
        // ~3 us per poll epoch at gfx1201 sclk (~9.5 us worst case; the
        // watchdog unblock path bounds any error).
        return 64 + MOE_WAIT_TIMEOUT_NS / 10000ull;
    }
    uint32_t* scratch = nullptr;
    if (hipMalloc(&scratch, 8) != hipSuccess)
    {
        (void) hipGetLastError();
        g_ready.store(false, std::memory_order_relaxed);
        return 64 + MOE_WAIT_TIMEOUT_NS / 10000ull;
    }
    (void) hipMemset(scratch, 0, 8);
    uint32_t* abort_scratch = scratch + 1;
    // Measure a known poll budget with no publisher; timeout scales linearly in polls.
    const unsigned long long probe_polls = fine ? 2000000ull : 20000ull;
    if (fine)
        moe_flag_wait_fine_kernel_hip<<<1, 1, 0, stream>>>(scratch, 0x7FFFFFFFu, abort_scratch,
                                                          probe_polls);
    else
        moe_flag_wait_kernel_hip<<<1, 1, 0, stream>>>(scratch, 0x7FFFFFFFu, abort_scratch,
                                                     probe_polls);
    auto t0 = std::chrono::steady_clock::now();
    hipError_t e = hipStreamSynchronize(stream);
    double ns_per_poll = 10000.0;   // fallback estimate
    if (e == hipSuccess)
    {
        double elapsed_ns =
            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed_ns > 0) ns_per_poll = elapsed_ns / (double) probe_polls;
    }
    else
    {
        (void) hipGetLastError();
        (void) hipStreamSynchronize(stream);
    }
    hipFree(scratch);
    unsigned long long budget = 64 + (unsigned long long)(MOE_WAIT_TIMEOUT_NS / ns_per_poll);
    g_max.store(budget, std::memory_order_relaxed);
    if (getenv("EXL3_MOE_HANDOFF_PROF")) printf(" -- hip wait calib: fine=%d ns/poll=%.1f budget=%llu\n", (int) fine, ns_per_poll, budget);
    return budget;
}

// -------------------------------------------------------------------------------------------
//   GPU flag ops on the current torch stream (addresses inside the registered region)
// -------------------------------------------------------------------------------------------

void exl3_moe_flag_write(uintptr_t flag, int64_t value)
{
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    moe_flag_write_kernel_hip<<<1, 1, 0, stream>>>
    (
        reinterpret_cast<uint32_t*>(flag),
        static_cast<uint32_t>(value)
    );
}

void exl3_moe_flag_wait(uintptr_t flag, int64_t value, uintptr_t abort_flag)
{
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    unsigned long long budget = moe_wait_poll_budget(stream);
    if (moe_wait_fine_enabled())
        moe_flag_wait_fine_kernel_hip<<<1, 1, 0, stream>>>
        (
            reinterpret_cast<uint32_t*>(flag),
            static_cast<uint32_t>(value),
            reinterpret_cast<uint32_t*>(abort_flag),
            budget
        );
    else
        moe_flag_wait_kernel_hip<<<1, 1, 0, stream>>>
        (
            reinterpret_cast<uint32_t*>(flag),
            static_cast<uint32_t>(value),
            reinterpret_cast<uint32_t*>(abort_flag),
            budget
        );
}

// No-op on ROCm: native stream memory operations are not used (see file header). The Python
// side calls this unconditionally, so keep the symbol.
void exl3_moe_cpu_set_memops(bool enabled)
{
    (void) enabled;
}

// -------------------------------------------------------------------------------------------
//   Child worker loop (identical job-ring semantics to cpu/moe_handoff.cu)
// -------------------------------------------------------------------------------------------

namespace {

inline void cpu_pause_()
{
#ifdef __x86_64__
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}

inline uint32_t load_acquire_u32(const uint32_t* p)
{
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

inline void store_release_u32(uint32_t* p, uint32_t v)
{
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

inline size_t align64(size_t x) { return x + 63 & ~(size_t) 63; }

} // namespace

void exl3_moe_cpu_worker_run
(
    uintptr_t shm_base,
    int64_t num_slots,
    int64_t slot_size,
    int64_t cap_rows,
    int64_t max_hi,
    int64_t max_ho,
    int64_t max_topk,
    int64_t wstage_offset,
    int64_t num_wslots,
    int64_t wslot_size,
    int64_t threads,
    int64_t stage_threads
)
{
    // GIL is released by the binding's call_guard; do not release it again here
    uint8_t* base = reinterpret_cast<uint8_t*>(shm_base);
    uint32_t* quit = reinterpret_cast<uint32_t*>(base + 0);
    uint32_t* pass_wake = reinterpret_cast<uint32_t*>(base + 64);
    uint32_t* ready = reinterpret_cast<uint32_t*>(base + 192);
    uint32_t* jobs_tail = reinterpret_cast<uint32_t*>(base + 256);
    uint32_t* jobs_head = reinterpret_cast<uint32_t*>(base + 320);
    MoeJob* jobs = reinterpret_cast<MoeJob*>(base + MOE_CTRL_JOBS_OFFSET);
    uint32_t* data_ready = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET);
    uint32_t* done = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS);
    uint32_t* stage_done = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS);
    uint32_t* pinned_free = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS + 64 * MOE_MAX_WSLOTS);
    uint32_t* stage_tail = reinterpret_cast<uint32_t*>(base + MOE_STAGE_TAIL_OFFSET);
    uint32_t* stage_head = reinterpret_cast<uint32_t*>(base + MOE_STAGE_HEAD_OFFSET);
    MoeJob* stage_jobs = reinterpret_cast<MoeJob*>(base + MOE_STAGE_JOBS_OFFSET);
    uint8_t* data = base + MOE_CTRL_SIZE;

    // Dedicated stager: consumes the stage ring so weight memcpys run concurrently with the
    // compute pool's work on the token tail
    int stage_threads_ = stage_threads > 0 ? (int) stage_threads : 1;
    std::thread stager([&]()
    {
        uint32_t shead = load_acquire_u32(stage_head);
        int s_idle = 0;
        while (true)
        {
            if (load_acquire_u32(quit)) return;
            if (load_acquire_u32(stage_tail) == shead)
            {
                if (++s_idle < 65536) { cpu_pause_(); continue; }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            s_idle = 0;

            const MoeJob job = stage_jobs[shead % MOE_STAGE_RING];
            shead++;
            store_release_u32(stage_head, shead);

            // Wait until the slot's previous tenant has been DMA'd out
            uint32_t* pf = pinned_free + size_t(job.slot) * 16;
            while ((int32_t)(load_acquire_u32(pf) - job.prev_seq) < 0)
            {
                if (load_acquire_u32(quit)) return;
                cpu_pause_();
            }
            exl3_moe_cpu_stage_experts(
                static_cast<int64_t>(job.layer),
                job.experts,
                static_cast<int>(job.rows),
                base + wstage_offset + size_t(job.slot) * wslot_size,
                stage_threads_
            );
            store_release_u32(stage_done + size_t(job.slot) * 16, job.seq);
        }
    });

    // Fixed slot section offsets from the registered maxima
    const size_t off_x = 0;
    const size_t off_sel = align64(size_t(cap_rows) * max_hi * 2);
    const size_t off_w = align64(off_sel + size_t(cap_rows) * max_topk * 4);
    const size_t off_out = align64(off_w + size_t(cap_rows) * max_topk * 2);

    // Handoff profiling (EXL3_MOE_HANDOFF_PROF)
    const bool hprof = getenv("EXL3_MOE_HANDOFF_PROF") != nullptr;
    double hp_gap = 0.0, hp_spin = 0.0, hp_comp = 0.0;
    double hp_gap_mx = 0.0, hp_spin_mx = 0.0, hp_comp_mx = 0.0;
    long hp_jobs = 0, hp_empty = 0, hp_assign = 0, hp_rows = 0;
    auto hp_prev_end = std::chrono::steady_clock::now();

    store_release_u32(ready, 1);

    uint32_t head = load_acquire_u32(jobs_head);
    uint32_t last_wake = 0;
    int idle = 0;
    while (true)
    {
        if (load_acquire_u32(quit)) break;

        const uint32_t wake = load_acquire_u32(pass_wake);
        if (wake != last_wake) { last_wake = wake; idle = 0; }

        if (load_acquire_u32(jobs_tail) == head)
        {
            // Spin hard in-pass (jobs arrive within microseconds of the GPU reaching the
            // layer), back off to naps when the queue has been dry for a while
            if (++idle < 65536) { cpu_pause_(); continue; }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        idle = 0;

        const MoeJob job = jobs[head % MOE_JOB_RING];
        head++;
        store_release_u32(jobs_head, head);

        const auto hp_t0 = std::chrono::steady_clock::now();

        // Wait for the GPU to publish the staged inputs for this seq
        uint32_t* drdy = data_ready + size_t(job.slot) * 16;
        while ((int32_t)(load_acquire_u32(drdy) - job.seq) < 0)
        {
            if (load_acquire_u32(quit)) goto out;
            cpu_pause_();
        }

        const auto hp_t1 = std::chrono::steady_clock::now();

        {
            uint8_t* slot = data + size_t(job.slot) * slot_size;
            bool run = true;
            if (job.kind == MOE_JOB_KIND_COMPUTE_GATED)
            {
                // Fused-issue job: an all-inactive job is a pure no-op
                const int32_t* selp = reinterpret_cast<const int32_t*>(slot + off_sel);
                const int total = (int) job.rows * (int) job.topk;
                if (hprof)
                {
                    int n = 0;
                    for (int i = 0; i < total; ++i)
                        if (selp[i] >= 0) n++;
                    run = n > 0;
                    hp_assign += n;
                    hp_rows += (int) job.rows;
                    if (!run) hp_empty++;
                }
                else
                {
                    run = false;
                    for (int i = 0; i < total; ++i)
                        if (selp[i] >= 0) { run = true; break; }
                }
            }
            if (run)
                exl3_moe_cpu_forward_raw
                (
                    static_cast<int64_t>(job.layer),
                    reinterpret_cast<const at::Half*>(slot + off_x),
                    reinterpret_cast<const int32_t*>(slot + off_sel),
                    reinterpret_cast<const at::Half*>(slot + off_w),
                    reinterpret_cast<float*>(slot + off_out),
                    static_cast<int>(job.rows),
                    static_cast<int>(job.topk),
                    static_cast<int>(threads)
                );
        }

        store_release_u32(done + size_t(job.slot) * 16, job.seq);

        if (hprof)
        {
            const auto hp_t2 = std::chrono::steady_clock::now();
            auto ms = [](auto a, auto b)
                { return std::chrono::duration<double, std::milli>(b - a).count(); };
            double g = ms(hp_prev_end, hp_t0), s = ms(hp_t0, hp_t1), c = ms(hp_t1, hp_t2);
            hp_gap += g; hp_spin += s; hp_comp += c;
            if (g > hp_gap_mx) hp_gap_mx = g;
            if (s > hp_spin_mx) hp_spin_mx = s;
            if (c > hp_comp_mx) hp_comp_mx = c;
            hp_prev_end = hp_t2;
            if (++hp_jobs % 64 == 0)
            {
                printf(" -- handoff prof (%ld jobs, ms/job avg|max): gap %.3f|%.3f "
                       "spin %.3f|%.3f compute %.3f|%.3f | empty %ld/64, "
                       "cpu-assign/row %.2f\n",
                       hp_jobs, hp_gap / 64, hp_gap_mx, hp_spin / 64, hp_spin_mx,
                       hp_comp / 64, hp_comp_mx, hp_empty,
                       hp_rows ? (double) hp_assign / (double) hp_rows : 0.0);
                fflush(stdout);
                hp_gap = hp_spin = hp_comp = 0.0;
                hp_gap_mx = hp_spin_mx = hp_comp_mx = 0.0;
                hp_empty = 0; hp_assign = 0; hp_rows = 0;
            }
        }
    }
    out:;
    stager.join();
}

#endif // USE_ROCM