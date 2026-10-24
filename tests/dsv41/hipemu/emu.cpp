// CPU emulator build of exllamav3/modules/dsv41_fn/dsv41_hip.hip: every kernel gets a C entry `emu_<name>(gx, gy, gz, block, params)`.
#include <hip/hip_runtime.h>
thread_local dim3 threadIdx, blockIdx;
dim3 blockDim, gridDim;
EmuBlock* g_emu;
thread_local unsigned g_tid;
#include "../../../exllamav3/modules/dsv41_fn/dsv41_hip.hip"

#define EMU(name) extern "C" void emu_##name(int gx, int gy, int gz, int bx, void** p) { emu_launch(name, gx, gy, gz, bx, p); }
EMU(ds41_kv_prep) EMU(ds41_rmsnorm_rows) EMU(ds41_fq_rows) EMU(ds41_idx_q_prep) EMU(ds41_idx_k_store) EMU(ds41_pool_store)
EMU(ds41_compress) EMU(ds41_compress_state) EMU(ds41_hc_mix_part) EMU(ds41_hc_mix_part_h) EMU(ds41_hc_mix_fin) EMU(ds41_hc_cn)
EMU(ds41_router_scores) EMU(ds41_router_topk) EMU(ds41_engram_comp) EMU(ds41_engram_hash) EMU(ds41_mxfp4_decode) EMU(ds41_fp8_decode)
EMU(ds41_engram_gate)
