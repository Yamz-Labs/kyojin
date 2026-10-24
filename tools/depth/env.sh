# Source from the repo root (or anywhere): sets the ROCm runtime + venv for the depth tools.
export EXL3_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$EXL3_ROOT/.venv/bin/activate"
export EXL3_ROCM_SDK=$(rocm-sdk path --root)
source "$EXL3_ROOT/tools/strix_halo/env.sh"
export PYTHONPATH="$EXL3_ROOT" TQDM_DISABLE=1
