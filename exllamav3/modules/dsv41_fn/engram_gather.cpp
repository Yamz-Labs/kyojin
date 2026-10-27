// Threaded O_DIRECT row gather for the Engram tables (plain C ABI, loaded with ctypes by engram_io.py; no torch, no extension rebuild).
// A row is the concatenation of `nseg` pieces, piece s = file fds[s] bytes [bases[s] + row * strides[s], + nbytes[s]) (MXFP4 release:
// 128 B of e2m1 pairs from the weight file and 8 B of e8m0 scales from the scale file). Every piece is read with one aligned pread of
// the covering 4 KiB sector(s) into a per-thread aligned buffer and copied to out[i * row_bytes + col]; a persistent pool of
// worker threads keeps `threads` reads in flight.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
constexpr int64_t SECTOR = 4096;

struct Job
{
    const int* fds; const int64_t* bases; const int64_t* strides; const int* nbytes; int nseg;
    const int64_t* rows; int64_t n; uint8_t* out; int64_t row_bytes;
    std::atomic<int64_t> next{0}; std::atomic<int> err{0}; std::atomic<int> pending{0};
};

struct Pool
{
    std::vector<std::thread> th;
    std::mutex m; std::condition_variable cv, done_cv;
    Job* job = nullptr; uint64_t gen = 0; bool stop = false;

    void run(int tid, uint64_t seen)
    {
        void* buf = nullptr;
        if (posix_memalign(&buf, SECTOR, 4 * SECTOR)) return;
        for (;;)
        {
            Job* j;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stop || gen != seen; });
                if (stop) break;
                seen = gen; j = job;
            }
            for (;;)
            {
                // one work item per (row, piece): the pieces of a row (values file, scales file) are read in parallel too
                const int64_t it = j->next.fetch_add(1);
                if (it >= j->n * j->nseg) break;
                const int64_t i = it / j->nseg;
                const int s = (int) (it % j->nseg);
                if (j->err.load()) continue;
                int64_t col = 0;
                for (int q = 0; q < s; q++) col += j->nbytes[q];
                const int64_t off = j->bases[s] + j->rows[i] * j->strides[s];
                const int64_t a = off / SECTOR * SECTOR;
                const int64_t e = (off + j->nbytes[s] + SECTOR - 1) / SECTOR * SECTOR;
                if (e - a > 4 * SECTOR) { j->err = 2; continue; }
                ssize_t got = pread(j->fds[s], buf, e - a, a);
                if (got < off + j->nbytes[s] - a) { j->err = 1; continue; }
                memcpy(j->out + i * j->row_bytes + col, (uint8_t*) buf + (off - a), j->nbytes[s]);
            }
            if (j->pending.fetch_sub(1) == 1) { std::lock_guard<std::mutex> lk(m); done_cv.notify_all(); }
        }
        free(buf);
    }
};

Pool* g_pool = nullptr;
std::mutex g_call;

int run_job(Pool& p, std::mutex& call_mutex, int threads, const int* fds, const int64_t* bases, const int64_t* strides,
            const int* nbytes, int nseg, const int64_t* rows, int64_t n, uint8_t* out, int64_t row_bytes)
{
    std::lock_guard<std::mutex> call(call_mutex);
    if (n <= 0) return 0;
    while ((int) p.th.size() < threads)
    {
        int t = (int) p.th.size();
        uint64_t g0;
        { std::lock_guard<std::mutex> lk(p.m); g0 = p.gen; }      // a new worker must not pick up an earlier (finished) job
        p.th.emplace_back([&p, t, g0] { p.run(t, g0); });
    }
    Job j;
    j.fds = fds; j.bases = bases; j.strides = strides; j.nbytes = nbytes; j.nseg = nseg; j.rows = rows; j.n = n;
    j.out = out; j.row_bytes = row_bytes;
    j.pending = (int) p.th.size();
    {
        std::lock_guard<std::mutex> lk(p.m);
        p.job = &j; p.gen++;
    }
    p.cv.notify_all();
    std::unique_lock<std::mutex> lk(p.m);
    p.done_cv.wait(lk, [&] { return j.pending.load() == 0; });
    return j.err.load();
}

struct Handle { Pool pool; std::mutex call; int threads; };
}

// Shared pool (capped at 512 threads), one call at a time.
extern "C" int ds_engram_gather(const int* fds, const int64_t* bases, const int64_t* strides, const int* nbytes, int nseg,
                                const int64_t* rows, int64_t n, uint8_t* out, int64_t row_bytes, int threads)
{
    threads = std::max(1, std::min<int>(threads, 512));
    if (!g_pool) g_pool = new Pool();
    return run_job(*g_pool, g_call, threads, fds, bases, strides, nbytes, nseg, rows, n, out, row_bytes);
}

// Private pool per reader: two readers (one per Engram layer) can gather at the same time from two Python threads.
extern "C" void* ds_engram_pool_new(int threads)
{
    Handle* h = new Handle();
    h->threads = std::max(1, std::min<int>(threads, 512));
    return h;
}

extern "C" int ds_engram_pool_gather(void* handle, const int* fds, const int64_t* bases, const int64_t* strides, const int* nbytes,
                                     int nseg, const int64_t* rows, int64_t n, uint8_t* out, int64_t row_bytes)
{
    Handle* h = (Handle*) handle;
    return run_job(h->pool, h->call, h->threads, fds, bases, strides, nbytes, nseg, rows, n, out, row_bytes);
}

extern "C" void ds_engram_pool_free(void* handle)
{
    Handle* h = (Handle*) handle;
    { std::lock_guard<std::mutex> lk(h->pool.m); h->pool.stop = true; }
    h->pool.cv.notify_all();
    for (auto& t : h->pool.th) t.join();
    delete h;
}
