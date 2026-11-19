// Address math of the sparse MLA gather, shared by the HIP kernel and the host guard-page test.
// Virtual key v of query t: groups 0..ns-1 are the selected groups (gidx), then the own open group (tail keys lo_t .. pos).
// nv = ns*KP + ntail virtual keys are valid; v >= nv is clamped to nv-1 (the row is read, its probability is forced to 0).
#pragma once
#ifdef __HIPCC__
#define SP_HD __host__ __device__
#else
#define SP_HD
#endif
template <int KP>
SP_HD inline int sp_row(int v, int nv, int ns, int lo_t, const int* gi /* ns valid entries */, int N)
{
    v = v < nv ? v : nv - 1;
    const int g = v / KP, e = v % KP;
    int row = g < ns ? gi[g] * KP + e : lo_t + e;
    row = row < 0 ? 0 : row;
    row = row >= N ? N - 1 : row;
    return row;
}
SP_HD inline int sp_lo_t(int pos, int KP, int tail) { return tail ? pos - pos % KP : pos - ((pos + 1) % KP) + 1; }
