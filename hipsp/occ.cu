#include "sp_mla.cu"
#include <cstdio>
int main() {
  int nb = 0; hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)sp_mla<4, true>, 512, 0);
  hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
  printf("max blocks/CU %d, CUs %d, lds/block max %zu, lds/CU %zu, clock %d kHz, regs/block %d\n", nb, p.multiProcessorCount, p.sharedMemPerBlock, p.maxSharedMemoryPerMultiProcessor, p.clockRate, p.regsPerBlock);
}
