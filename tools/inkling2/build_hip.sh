#!/bin/bash
# build_hip.sh SRC OUT  (gfx1151 shared library from a .hip / .cu source; EXTRA="-DWPB=8 -DSTAGE_FIRST=1" adds compile flags)
# ROCM_SDK = ROCm root (bin/hipcc); default: `rocm-sdk path --root` when that tool is on PATH, else /opt/rocm
SRC=$1; OUT=$2
SDK=${ROCM_SDK:-$(rocm-sdk path --root 2>/dev/null || echo /opt/rocm)}
GCCD=$(dirname "$(gcc -print-libgcc-file-name)")
BC=$SDK/lib/llvm/amdgcn/bitcode; [ -d $BC ] || BC=$SDK/amdgcn/bitcode
export ROCM_PATH=$SDK HIP_PATH=$SDK
$SDK/bin/hipcc -x hip --offload-arch=gfx1151 -O3 -std=c++17 -shared -fPIC --gcc-install-dir=$GCCD --rocm-device-lib-path=$BC --rocm-path=$SDK "$SRC" -o "$OUT" ${EXTRA}
