#!/bin/bash
# JIT-build only the exl3_dec kernels as a standalone extension (fast iteration).
WT=${WT:-$(cd "$(dirname "$0")/../../.." && pwd)}
export EXL3_ROOT=$WT EXL3_VENV=$HOME/kyojin/.venv EXL3_ROCM_SDK=/opt/rocm-7.2.4
export EXL3_ROCM_DEV_INCLUDE=$WT/.sdk-deps/root/opt/rocm-7.2.4/include
source $WT/tools/strix_halo/env.sh build > /dev/null
export PYTHONNOUSERSITE=1 TORCH_EXTENSIONS_DIR=$WT/build/devext MAX_JOBS=4
mkdir -p $TORCH_EXTENSIONS_DIR
cd $WT
exec $EXL3_VENV/bin/python - "$@" <<'PY'
import os, sys, torch
from torch.utils.cpp_extension import load
wt = os.environ["EXL3_ROOT"]
src = wt + "/exllamav3/exllamav3_ext"
defs = [f"-D{d}" for d in os.environ.get("DEV_DEFINES", "").split()]
flags = ["-O3", "--offload-arch=gfx1151", "-I" + src, "-I" + os.environ["EXL3_ROCM_DEV_INCLUDE"]] + defs
m = load(name = os.environ.get("DEV_NAME", "exl3_dec_dev"), sources = [wt + "/tools/decode/devext/bind.cpp", src + "/quant/exl3_dec.cu"],
         extra_include_paths = [src], extra_cflags = ["-O2", "-DUSE_ROCM"] + defs, extra_cuda_cflags = flags,
         verbose = "-v" in sys.argv)
print("built", m.__file__)
PY
