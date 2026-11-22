#!/bin/bash
# build_experts_cuda.sh SRC OUT  (CPU only, nice 19; sm_86; shared cudart so torch's libcudart is reused)
# CUDA_TC = CUDA toolkit root (bin/nvcc, lib/), e.g. the nvidia/cu13 directory of a pip install; default CUDA_HOME, then /usr/local/cuda
SRC=$1; OUT=$2
N=${CUDA_TC:-${CUDA_HOME:-/usr/local/cuda}}
export CUDA_HOME=$N CUDA_PATH=$N PATH=$N/bin:$PATH LD_LIBRARY_PATH=$N/lib
nice -n 19 $N/bin/nvcc -arch=sm_86 -O3 -std=c++17 -shared -Xcompiler -fPIC --cudart shared -Xptxas -v -lineinfo "$SRC" -o "$OUT" ${EXTRA}
