// CPU-side bounds test of the gather address math (sp_idx.h) with mmap guard pages: every virtual key of every tile of every query is
// resolved through sp_row and the whole latent row [0, 576) is read; gidx holds exactly ns entries followed by a PROT_NONE page.
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdint.h>
#include "sp_idx.h"
static void* guarded(size_t bytes, void** base, size_t* tot)
{
    const size_t pg = sysconf(_SC_PAGESIZE), body = (bytes + pg - 1) / pg * pg, total = body + 2 * pg;
    char* b = (char*)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    mprotect(b, pg, PROT_NONE); mprotect(b + pg + body, pg, PROT_NONE);
    *base = b; *tot = total;
    return b + pg + body - bytes;   // data ends exactly at the trailing guard page
}
template <int KP> static long run(int N, int T, int qstart, int ns_cap, int tail, unsigned seed)
{
    srand(seed);
    void *b1, *b2; size_t t1, t2;
    int16_t* kc = (int16_t*)guarded((size_t)N * 576 * 2, &b1, &t1);
    long chk = 0;
    for (int t = 0; t < T; t++) {
        const int pos = qstart + t, lo_t = sp_lo_t(pos, KP, tail), ntail = pos - lo_t + 1;
        const int P = (pos / KP) > 0 ? pos / KP : 1;
        int ns = rand() % (ns_cap + 1); if (ns > P) ns = P;
        int* gi = (int*)guarded((size_t)(ns ? ns : 1) * 4, &b2, &t2);
        for (int i = 0; i < ns; i++) gi[i] = (int)((long)i * P / (ns ? ns : 1));   // ascending, < P (groups fully before pos)
        const int nv = ns * KP + ntail, ntiles = (nv + 15) >> 4;
        for (int tau = 0; tau < ntiles; tau++)
            for (int k = 0; k < 16; k++) {
                const int row = sp_row<KP>(tau * 16 + k, nv, ns, lo_t, gi, N);
                if (row < 0 || row >= N) { printf("OOB row %d (N %d)\n", row, N); exit(1); }
                chk += kc[(long)row * 576] + kc[(long)row * 576 + 575];   // touches first and last element of the row
            }
        munmap(b2, t2);
    }
    munmap(b1, t1);
    return chk;
}
int main()
{
    int cases = 0;
    const int Ns[] = {1, 3, 4, 5, 17, 40, 255, 2049, 3000, 33001};
    for (int N : Ns)
        for (int tail = 0; tail < 2; tail++) {
            const int T = N < 64 ? N : 64;
            for (int qs : {0, N - T}) {
                run<4>(N, T, qs, 511, tail, N + qs); run<2>(N, T, qs, 511, tail, N + 1); run<8>(N, T, qs, 300, tail, N + 2); cases += 3;
            }
        }
    printf("guard-page host test OK (%d cases)\n", cases);
    return 0;
}
