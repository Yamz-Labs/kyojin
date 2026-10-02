#!/bin/bash
# Build the ROCm (gfx1151, Strix Halo) extension in this checkout.
# Needs: a Python venv with a ROCm PyTorch build (see README.md). Override the defaults below from the environment.
cd "$(dirname "$0")"
export EXL3_ROOT="$PWD"
export EXL3_VENV="${EXL3_VENV:-$PWD/.venv}"
# The ROCm SDK devel tree that ships hipsparse/ and thrust/ headers (the _rocm_sdk_devel wheel, or /opt/rocm if it has them).
export EXL3_ROCM_SDK="${EXL3_ROCM_SDK:-$PWD/.venv-gfx1151/lib/python3.12/site-packages/_rocm_sdk_devel}"
export EXL3_ROCM_DEV_INCLUDE="${EXL3_ROCM_DEV_INCLUDE:-}"
export MAX_JOBS="${MAX_JOBS:-6}" PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1151}"
# Preflight: fail early with a readable message instead of "CUDA_HOME environment variable is not set".
if [ ! -x "$EXL3_VENV/bin/python" ]; then
    echo "[build] no venv at $EXL3_VENV. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2; exit 1
fi
if [ -z "$("$EXL3_VENV/bin/python" -c 'import torch; print(torch.version.hip or "")' 2>/dev/null)" ]; then
    echo "[build] the venv's PyTorch is not a ROCm build (torch.version.hip is empty). Install the ROCm wheel for gfx1151 first, see README.md." >&2; exit 1
fi
if [ ! -f "$EXL3_ROCM_SDK/include/hipsparse/hipsparse.h" ] && [ ! -f "$EXL3_ROCM_SDK/include/hipsparse.h" ] \
   && [ ! -f "${EXL3_ROCM_DEV_INCLUDE:-/nonexistent}/hipsparse/hipsparse.h" ] && [ ! -f "${EXL3_ROCM_DEV_INCLUDE:-/nonexistent}/hipsparse.h" ]; then
    echo "[build] EXL3_ROCM_SDK=$EXL3_ROCM_SDK and EXL3_ROCM_DEV_INCLUDE have no hipsparse headers. Point EXL3_ROCM_SDK at the _rocm_sdk_devel directory (or EXL3_ROCM_DEV_INCLUDE at an include dir that has hipsparse/ and thrust/), see README.md." >&2; exit 1
fi
exec bash tools/strix_halo/rebuild.sh
