#!/bin/bash
# Dump gfx1151 ISA of exl3_dec kernels -> build/devext/exl3_dec.s
WT=${WT:-$(cd "$(dirname "$0")/../../.." && pwd)}
export EXL3_ROOT=$WT EXL3_VENV=$HOME/kyojin/.venv EXL3_ROCM_SDK=/opt/rocm-7.2.4 EXL3_ROCM_DEV_INCLUDE=$WT/.sdk-deps/root/opt/rocm-7.2.4/include
source $WT/tools/strix_halo/env.sh build > /dev/null
F=$(grep "^cuda_cflags" $WT/build/devext/exl3_dec_dev/build.ninja | sed 's/^cuda_cflags = //')
$WT/tools/decode/devext/hipify.sh
cd $WT/build/devext
/opt/rocm-7.2.4/bin/hipcc $F $DEV_DEFINES --cuda-device-only -S $WT/exllamav3/exllamav3_ext/quant/exl3_dec.hip -o exl3_dec.s 2>&1 | grep -v warning | head
