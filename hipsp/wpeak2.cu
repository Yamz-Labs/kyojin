#include <hip/hip_runtime.h>
#include <cstdio>
typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 h16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float f8;
template <int NACC> __global__ __launch_bounds__(512) void k(float* out, int iters) {
  h16 a[4], b[4]; unsigned x = threadIdx.x * 2654435761u + blockIdx.x; for (int j = 0; j < 4; j++) for (int i = 0; i < 16; i++) { x = x * 1664525u + 1013904223u; a[j][i] = (__fp16)(((x >> 16) & 0xffff) / 65536.0f - 0.5f); x = x * 1664525u + 1013904223u; b[j][i] = (__fp16)(((x >> 16) & 0xffff) / 65536.0f - 0.5f); }
  f8 acc[NACC]; for (int i = 0; i < NACC; i++) acc[i] = (f8){0,0,0,0,0,0,0,0};
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < NACC; i++) acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i & 3], b[(i + it) & 3], acc[i]);
  }
  float s = 0; for (int i = 0; i < NACC; i++) for (int j = 0; j < 8; j++) s += acc[i][j];
  out[blockIdx.x * 512 + threadIdx.x] = s;
}
template <int NACC> void run(int waves_per_block) {
  float* o; hipMalloc(&o, 4 * 512 * 4096);
  int iters = 4096 / NACC * 16; int blocks = 400;
  k<NACC><<<blocks, 512>>>(o, 10); hipDeviceSynchronize();
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
  k<NACC><<<blocks, 512>>>(o, iters); hipEventRecord(e1); hipEventSynchronize(e1);
  float ms; hipEventElapsedTime(&ms, e0, e1);
  double fl = (double)blocks * 16 * iters * NACC * 8192.0;
  printf("NACC %d: %.1f TF\n", NACC, fl / ms / 1e9);
}
int main() { run<1>(16); run<2>(16); run<4>(16); run<8>(16); }
