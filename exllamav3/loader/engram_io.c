/* engram_io.c: GIL-free parallel row reads for the Engram table (same bytes as engram_gpu._read_rows: one pread per distinct row into a flat buffer).
   build: gcc -O2 -shared -fPIC -pthread engram_io.c -o libengram_io.so */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>

typedef struct {
    const int *fds; const int64_t *bases; const int64_t *rows; int64_t lo, hi, off0, rb; uint8_t *out; int err;
} job_t;

static void *work(void *p) {
    job_t *j = (job_t *)p;
    for (int64_t k = j->lo; k < j->hi; k++) {
        int64_t o = j->off0 + j->rows[k] * j->rb;
        int fi = (int)(o >> 48);
        int64_t pos = j->bases[fi] + (o & ((1LL << 48) - 1));
        uint8_t *dst = j->out + k * j->rb;
        int64_t got = 0;
        while (got < j->rb) {
            ssize_t n = pread(j->fds[fi], dst + got, (size_t)(j->rb - got), (off_t)(pos + got));
            if (n <= 0) { j->err = n < 0 ? errno : -1; return NULL; }
            got += n;
        }
    }
    return NULL;
}

/* rows: sorted distinct row ids [n]; returns 0 or an errno / -1 (short read) */
int engram_read_rows(const int *fds, const int64_t *bases, const int64_t *rows, int64_t n, int64_t off0, int64_t rb, uint8_t *out, int nthreads) {
    if (n <= 0) return 0;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 256) nthreads = 256;
    if (nthreads > n) nthreads = (int)n;
    pthread_t th[256]; job_t jobs[256];
    int64_t step = (n + nthreads - 1) / nthreads;
    int started = 0;
    for (int t = 0; t < nthreads; t++) {
        int64_t lo = t * step, hi = lo + step < n ? lo + step : n;
        if (lo >= hi) break;
        jobs[t] = (job_t){fds, bases, rows, lo, hi, off0, rb, out, 0};
        if (pthread_create(&th[t], NULL, work, &jobs[t]) != 0) { work(&jobs[t]); th[t] = 0; }
        started++;
    }
    int err = 0;
    for (int t = 0; t < started; t++) { if (th[t]) pthread_join(th[t], NULL); if (jobs[t].err && !err) err = jobs[t].err; }
    return err;
}
