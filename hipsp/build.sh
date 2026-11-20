#!/bin/bash
# build.sh SRC.cu OUT.so [VENV] : HIP library for gfx1151 (SDK toolchain that matches torch). Prints resource usage.
SRC=$1; OUT=$2; VENV=${3:-${VIRTUAL_ENV:-.venv}}
SDK=$(PATH=$VENV/bin:$PATH rocm-sdk path --root)
GCCD=$(dirname "$(gcc -print-libgcc-file-name)")
BC=$SDK/lib/llvm/amdgcn/bitcode; [ -d $BC ] || BC=$SDK/amdgcn/bitcode
export ROCM_PATH=$SDK HIP_PATH=$SDK
$SDK/bin/hipcc -x hip --offload-arch=gfx1151 -O3 -std=c++17 -shared -fPIC --gcc-install-dir=$GCCD --rocm-device-lib-path=$BC --rocm-path=$SDK -I$(dirname $SRC) -Rpass-analysis=kernel-resource-usage "$SRC" -o "$OUT" 2>&1 | grep -E "error|VGPRs:|Spill|LDS Size|Occupancy|Function Name|warning: " | head -60
ls -la "$OUT"
