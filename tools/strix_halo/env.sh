#!/usr/bin/env bash
# Environment for exllamav3 on a Strix Halo machine (AMD Strix Halo / gfx1151).
#
#   RUN:    source tools/strix_halo/env.sh        (from the repository root)
#   BUILD:  source tools/strix_halo/env.sh build
#   CHECK:  bash tools/strix_halo/env.sh --check
#
# ===========================================================================
# FIVE TRAPS, all verified on this box (2026-09-16). Do not "simplify" these.
# ===========================================================================
#
# TRAP 1 -- torch's bundled HSA runtime SEGFAULTS on gfx1151.
#   torch 2.10.0+rocm7.0 lists gfx1151 in get_arch_list() and
#   torch.cuda.is_available() returns True, but the FIRST device allocation
#   segfaults -- even torch.zeros(4, device="cuda"):
#       segfault at 34 ... in libhsa-runtime64.so (error 4)  -> exit 139
#   The fault is inside libhsa-runtime64.so, NOT PyTorch. No env var avoids it
#   (HSA_ENABLE_SDMA=0, HSA_ENABLE_IPC=0, AMD_SERIALIZE_KERNEL=3,
#   TORCH_BLAS_PREFER_HIPBLASLT=0 all still segfault).
#   The kernel is fine: KFD reports gfx_target_version 110501 on node 1.
#   FIX: preload the system ROCm 7 libhsa-runtime64 (1.18) over torch's copy.
#   Verified: without -> exit 139; with -> 31.5 TFLOP/s fp16 4096^3.
#
# TRAP 2 -- ROCm 6.4 has NO gfx1151 code object (HIP error: invalid device
#   function on every kernel). Requires ROCm 7.0+. HSA_OVERRIDE_GFX_VERSION
#   must stay UNSET: gfx1151 is native on 7.0 and overriding corrupts kernel
#   selection.
#
# TRAP 3 -- the torch wheel ships runtime libs but no hipcc, and reports
#   ROCM_HOME=None / IS_HIP_EXTENSION=False, so the build dies with
#   "CUDA_HOME environment variable is not set". We borrow the HIP toolchain
#   from the AMD gfx1151 nightly's rocm-sdk-devel wheel.
#
# TRAP 4 -- those wheels put device bitcode at lib/llvm/amdgcn/bitcode, not
#   the $ROCM_PATH/amdgcn/bitcode clang probes ("cannot find ROCm device
#   library"). Passed explicitly below. Also note _rocm_sdk_core lacks
#   hipsparse.h and thrust/, so ROCM_HOME must be _rocm_sdk_devel.
#
# TRAP 5 -- those same SDK paths must NOT be exported at RUNTIME. The SDK
#   ships its own HSA runtime that wins over the LD_PRELOAD and re-triggers
#   the Trap 1 segfault. Build-time only, hence the `build` guard below.
# ===========================================================================

# Needs a ROCm 7 libhsa-runtime64.so.1 (HSA 1.18 or newer). Ubuntu 24.04's own libhsa-runtime64-1 is ROCm 5.7 (1.11)
# and is too old: torch then fails with "undefined symbol: hsa_amd_enable_logging". Install ROCm 7 from AMD's apt
# repository (it provides /opt/rocm/lib/libhsa-runtime64.so.1), or point EXL3_HSA_LIB at your copy. The rocm-sdk-devel wheel ships one too (EXL3_ROCM_SDK/lib). It is tried first because it matches the ROCm torch wheel: a system /opt/rocm with an older HSA fails with "undefined symbol: hsa_ext_image_create_v2".
_HSA=""
for _c in "${EXL3_HSA_LIB:-}" "${EXL3_ROCM_SDK:-/nonexistent}/lib/libhsa-runtime64.so.1" /opt/rocm/lib/libhsa-runtime64.so.1 /usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1; do
    if [ -n "$_c" ] && [ -e "$_c" ]; then _HSA="$_c"; break; fi
done
if [ -n "$_HSA" ]; then
    export LD_PRELOAD="$_HSA"
else
    echo "[env.sh] WARNING: no libhsa-runtime64.so.1 found. Install ROCm 7, set EXL3_ROCM_SDK or set EXL3_HSA_LIB, or torch will segfault on gfx1151 (Trap 1)." >&2
fi

# Repository root = two levels above this file (works for any clone location).
_SELF="${BASH_SOURCE[0]:-$0}"
EXL3_ROOT="${EXL3_ROOT:-$(cd "$(dirname "$_SELF")/../.." && pwd)}"
EXL3_VENV="${EXL3_VENV:-$EXL3_ROOT/.venv}"
export PATH="$EXL3_VENV/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$EXL3_ROOT:${PYTHONPATH:-}"   # exllamav3 and the built exllamav3_ext*.so live in the repo root

# Uncomment to force reconstruct+hgemm instead of HIP GEMV (debugging):
# export EXL3_GEMV=0

# --- BUILD-ONLY toolchain (Trap 5: never export these for inference) ---
if [ "${1:-}" = "build" ] || [ "${EXL3_BUILD:-0}" = "1" ]; then
    _SDK="${EXL3_ROCM_SDK:-$EXL3_ROOT/.venv-gfx1151/lib/python3.12/site-packages/_rocm_sdk_devel}"
    export ROCM_HOME="$_SDK"
    export ROCM_PATH="$_SDK"
    export HIP_PATH="$_SDK"
    export HIPCXX="$_SDK/bin/hipcc"
    export PATH="$_SDK/bin:$_SDK/llvm/bin:$_SDK/lib/llvm/bin:$PATH"
    if [ -d "$_SDK/lib/llvm/amdgcn/bitcode" ]; then
        HIP_DEVICE_LIB_PATH="$_SDK/lib/llvm/amdgcn/bitcode"
    else
        HIP_DEVICE_LIB_PATH="$_SDK/amdgcn/bitcode"
    fi
    export HIP_DEVICE_LIB_PATH
    GCC_INSTALL_DIR="${EXL3_GCC_INSTALL_DIR:-$(dirname "$(gcc -print-libgcc-file-name)")}"
    export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=$GCC_INSTALL_DIR --rocm-device-lib-path=$HIP_DEVICE_LIB_PATH --rocm-path=$_SDK"
    if [ -n "${EXL3_ROCM_DEV_INCLUDE:-}" ]; then
        export HIPCC_COMPILE_FLAGS_APPEND="$HIPCC_COMPILE_FLAGS_APPEND -I$EXL3_ROCM_DEV_INCLUDE"
    fi
    export PYTORCH_ROCM_ARCH=gfx1151
    export MAX_JOBS="${MAX_JOBS:-6}"        # 32 threads exist, but unbounded ninja OOMs the box
    echo "[env.sh] BUILD mode: ROCM_HOME=$(basename "$_SDK") arch=$PYTORCH_ROCM_ARCH jobs=$MAX_JOBS"
fi

if [ "${1:-}" = "--check" ]; then
    if [ ! -x "$EXL3_VENV/bin/python" ]; then
        echo "[env.sh] no venv at $EXL3_VENV. Create it (python3 -m venv .venv) or set EXL3_VENV." >&2
        exit 1
    fi
    "$EXL3_VENV/bin/python" - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  hip {torch.version.hip}")
print(f"{p.name} / {p.gcnArchName} / warp={p.warp_size} / {p.total_memory/2**30:.1f} GiB / {p.multi_processor_count} CUs")
a = torch.randn(512, 512, dtype=torch.float16, device="cuda")
(a @ a).sum().item()
torch.cuda.synchronize()
print("GPU compute: OK")
try:
    import exllamav3_ext
    print("exllamav3_ext: OK")
    print("exl3_gemv_supported(0):", exllamav3_ext.exl3_gemv_supported(0),
          "<- False on gfx1151 => reconstruct+hgemm fallback (expected)")
except ImportError as e:
    print("exllamav3_ext: NOT BUILT --", e)
import exllamav3
print("exllamav3:", exllamav3.__version__ if hasattr(exllamav3, "__version__") else "imported")
from exllamav3.util import hip_compiler
hip_compiler.report(print)
PY
fi
