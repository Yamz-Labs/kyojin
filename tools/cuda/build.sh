#!/bin/bash
# Build the Kyojin extension for NVIDIA (sm_86 by default) with nvcc from NVIDIA's PyPI wheels; no system CUDA needed.
# usage: tools/cuda/build.sh TOOLCHAIN_DIR [TORCH_PYTHON]      (run from anywhere; builds in place in this tree)
#   TOOLCHAIN_DIR: a CUDA_HOME made by tools/cuda/README.md (bin/nvcc, include, nvvm, lib64 with unversioned .so links)
#   TORCH_PYTHON:  python of a venv with a CUDA torch whose CUDA major matches nvcc (torch 2.10 cu130 <-> nvcc 13.0)
# env: ARCH (default 8.6), MAX_JOBS (default 10), EXL3_PORT_DEC=1 to also compile the HIP decode kernels (long: see README),
#      SCRATCH (TMPDIR for nvcc, default <tree>/build/tmp)
set -eu
E=$(cd "$(dirname "$0")/../.." && pwd); T=$(readlink -f "$1"); PY=${2:-python3}
export CUDA_HOME=$T TORCH_CUDA_ARCH_LIST=${ARCH:-8.6} MAX_JOBS=${MAX_JOBS:-10} TMPDIR=${SCRATCH:-$E/build/tmp}
export PATH=$CUDA_HOME/bin:$(dirname "$PY"):$PATH
mkdir -p "$TMPDIR"; cd "$E"
nvcc --version | tail -2
nice -n 10 "$PY" -I setup.py build_ext --inplace -j "$MAX_JOBS"
CUDA_VISIBLE_DEVICES= "$PY" -I -c "import sys; sys.path.insert(0, '$E'); import torch, exllamav3_ext as e; print('ok', len([n for n in dir(e) if not n.startswith('_')]), 'names')"
