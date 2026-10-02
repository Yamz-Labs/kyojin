#!/usr/bin/env bash
# Lane launcher for GLM-5.3-Flash EXL3 (MTP): the exact flags and speed envs of our serving lane.
#   serve_glm.sh --port N --model-id ID -c CTX [serve.py args...]
# Flags: medium chat template, model-card sampling (T1.0, top_p 0.95), reasoning effort medium, MTP n=2.
# Knobs (env):
#   GLM_MODEL        pack dir (required unless ~/models/GLM-5.3-Flash-EXL3-Yamz exists; fails fast when missing)
#   GLM_ROOT         engine tree with the built extension (default: this checkout)
#   GLM_ABLIT        edit_spec.json for the runtime uncensor hook (EXL3_ABLIT_RUNTIME). Unset = hook off.
#   EXL3_VENV        venv with the ROCm torch (default <root>/.venv)
#   EXL3_HSA_LIB     libhsa-runtime64.so.1 to preload (default: what tools/strix_halo/env.sh picks)
#   LANE_STATE       scratch dir (default ~/cache/glm-lane)
#   LANE_MIN_AVAIL_GIB  refuse to start below this MemAvailable (default 100)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${GLM_ROOT:-$(cd "$HERE/../.." && pwd)}
MODEL=${GLM_MODEL:-$HOME/models/GLM-5.3-Flash-EXL3-Yamz}
STATE=${LANE_STATE:-$HOME/cache/glm-lane}
die() { echo "serve_glm: $*" >&2; exit 3; }
[ -f "$MODEL/config.json" ] || die "model pack missing: $MODEL (set GLM_MODEL)"
[ -f "$ROOT/tools/glm/serve.py" ] || die "engine tree missing: $ROOT"
grep -q -- "--default-temperature" "$ROOT/tools/glm/serve.py" || die "$ROOT serve.py lacks --default-temperature (lane would serve greedy and loop)"
ls "$ROOT"/exllamav3_ext*.so >/dev/null 2>&1 || die "built extension (exllamav3_ext*.so) missing in $ROOT (run build.sh)"
need_gib=${LANE_MIN_AVAIL_GIB:-100}
avail_gib=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
[ "$avail_gib" -ge "$need_gib" ] || die "MemAvailable ${avail_gib} GiB < ${need_gib} GiB"
# Runtime shell: build variables must not leak in (they make the first kernel launch segfault).
unset ROCM_HOME ROCM_PATH HIP_PATH HIPCXX HIP_DEVICE_LIB_PATH HIPCC_COMPILE_FLAGS_APPEND PYTORCH_ROCM_ARCH LD_LIBRARY_PATH HSA_OVERRIDE_GFX_VERSION
export EXL3_ROOT="$ROOT"
source "$ROOT/tools/strix_halo/env.sh"
[ -z "${EXL3_HSA_LIB:-}" ] || export LD_PRELOAD="$EXL3_HSA_LIB"
export PYTHONNOUSERSITE=1
mkdir -p "$STATE/tmp" "$STATE/slots"
export EXL3_REPO="$ROOT" PYTHONPATH="$ROOT" TMPDIR="$STATE/tmp" TQDM_DISABLE=1
# Speed defaults of the lane (decode/prefill wins measured on gfx1151; serve.py sets the rest via setdefault).
export EXL3_MOE_CFG="${EXL3_MOE_CFG:-2}"
export EXL3_HIP_PREFILL_MIN_ROWS="${EXL3_HIP_PREFILL_MIN_ROWS:-2}"
export EXL3_LOAD_DEVICE="${EXL3_LOAD_DEVICE-cuda:0}"
export EXL3_DENSE_GEMM_TUNE_FILE=${EXL3_DENSE_GEMM_TUNE_FILE:-$STATE/tune.txt}
[ -e "$EXL3_DENSE_GEMM_TUNE_FILE" ] || cp "$HERE/assets/tune_seed.txt" "$EXL3_DENSE_GEMM_TUNE_FILE"
hook=off
if [ -n "${GLM_ABLIT:-}" ]; then
  [ -f "$GLM_ABLIT" ] || die "GLM_ABLIT set but not a file: $GLM_ABLIT"
  export EXL3_ABLIT_RUNTIME="$GLM_ABLIT"; hook=on
elif [ -n "${EXL3_ABLIT_RUNTIME:-}" ]; then
  hook="env:$EXL3_ABLIT_RUNTIME"   # explicit env wins; "off" disables a bundled spec
elif [ -f "$MODEL/uncensor_spec.json" ]; then
  hook="bundled"                   # the engine applies <model>/uncensor_spec.json by itself
fi
echo "serve_glm: model=$MODEL root=$ROOT hook=$hook avail=${avail_gib}GiB" >&2
exec "${EXL3_VENV:-$ROOT/.venv}/bin/python" -u "$ROOT/tools/glm/serve.py" --model "$MODEL" \
  --num-draft 2 --max-history 2 --chat-template "$HERE/assets/glm53-template-medium.jinja" \
  --default-temperature 1.0 --default-top-p 0.95 --default-reasoning-effort medium "$@"
