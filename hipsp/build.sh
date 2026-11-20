#!/bin/bash
# build.sh SRC.cu OUT.so [VENV] : HIP library for gfx1151 (SDK toolchain that matches torch). Prints resource usage.
# Fails loudly: any error stops the script with a non-zero status, and the old OUT is removed first so a failed build never leaves a stale library behind.
set -euo pipefail
SRC=${1:?usage: build.sh SRC.cu OUT.so [VENV]}; OUT=${2:?usage: build.sh SRC.cu OUT.so [VENV]}; VENV=${3:-${VIRTUAL_ENV:-.venv}}
SDK=$(PATH=$VENV/bin:$PATH rocm-sdk path --root)
[ -x "$SDK/bin/hipcc" ] || { echo "build.sh: no hipcc in '$SDK' (rocm-sdk of the venv '$VENV'): install the ROCm devel wheel as in doc/install.md" >&2; exit 1; }
GCCD=$(dirname "$(gcc -print-libgcc-file-name)")
BC=$SDK/lib/llvm/amdgcn/bitcode; [ -d "$BC" ] || BC=$SDK/amdgcn/bitcode
export ROCM_PATH=$SDK HIP_PATH=$SDK
rm -f "$OUT"
"$SDK/bin/hipcc" -x hip --offload-arch=gfx1151 -O3 -std=c++17 -shared -fPIC --gcc-install-dir="$GCCD" --rocm-device-lib-path="$BC" --rocm-path="$SDK" -I"$(dirname "$SRC")" -Rpass-analysis=kernel-resource-usage "$SRC" -o "$OUT" 2>&1
[ -s "$OUT" ] || { echo "build.sh: hipcc left no library at $OUT" >&2; exit 1; }
ls -la "$OUT"
