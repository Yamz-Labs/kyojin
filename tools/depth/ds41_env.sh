# source from anywhere: DS4.1 box env on this tree (reuses a built extension .so next to the package)
export EXL3_VENV=./ds41_work/v15g/run/engine/.venv
export EXL3_ROCM_SDK=$EXL3_VENV/lib/python3.12/site-packages/_rocm_sdk_devel
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$D"; source tools/strix_halo/env.sh >/dev/null 2>&1
export PATH=$EXL3_VENV/bin:$PATH PYTHONPATH=$D:./ds41_work/ds41box/pylib
unset CUDA_VISIBLE_DEVICES; export HIP_VISIBLE_DEVICES=0
