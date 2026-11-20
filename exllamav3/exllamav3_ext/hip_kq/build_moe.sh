#!/bin/bash
# build_moe.sh SRC.cu OUT.co : device-only code object for gfx1151 with the SDK toolchain that matches torch (build-only env, separate process)
SRC=$1; OUT=$2
ENG=${KYOJIN_ENGINE_DIR:-$PWD}
SDK=$(PATH=$ENG/.venv/bin:$PATH rocm-sdk path --root)
GCCD=$(dirname "$(gcc -print-libgcc-file-name)")
BC=$SDK/lib/llvm/amdgcn/bitcode; [ -d $BC ] || BC=$SDK/amdgcn/bitcode
export ROCM_PATH=$SDK HIP_PATH=$SDK
$SDK/bin/hipcc -x hip --offload-arch=gfx1151 --offload-device-only -O3 -std=c++17 --gcc-install-dir=$GCCD --rocm-device-lib-path=$BC --rocm-path=$SDK -save-temps=obj -Rpass-analysis=kernel-resource-usage "$SRC" -o "$OUT" 2>&1 | grep -E "error|warning: |VGPRs|Spill|SGPRs|LDS|Occupancy|Function Name" | head -400
ls -la "$OUT"
