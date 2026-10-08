#!/usr/bin/env bash
# llama-swap / standalone lane wrapper for MiMo-V2.6-Flash EXL3.
#   serve_mimo.sh --port N --model-id ID -c CTX [serve.py args...]
# DFlash speculation is on by default (MIMO_SPEC=1): 4 bpw EXL3 drafter, ndt 7, confidence-truncated draft length.
# Greedy output is not token-identical to plain decode (near-tied logits can flip under the batched verify);
# MIMO_SPEC=0 gives plain decode. With no drafter found the lane logs one line and decodes plain.
# Sampling: model-card / generation_config.json defaults T1.0, top_p 0.95 (agent clients often send neither; greedy decoding is not recommended).
# Chat template: tools/mimo/chat_template.jinja, the model's template plus a medium effort level (serve.py default).
# Tool calls: serve.py parses <parameter=k>v</parameter> XML and JSON bodies, typed by the request's tool schema.
# Knobs (env):
#   MIMO_MODEL   pack dir (default ~/models/MiMo-V2.6-Flash-MOPD-EXL3-Yamz; MUST exist: fail fast)
#   MIMO_ROOT       engine worktree (exllamav3 + built ext). Default = this checkout
#   MIMO_ABLIT      uncensor hook, OFF by default. Set to the edit_spec.json path (its .safetensors direction next to it)
#                   to enable EXL3_ABLIT_RUNTIME. A set-but-missing path is an error, never a silent "off".
#   MIMO_CTX_MAX    hard ceiling for -c (default computed below)
#   MIMO_RESERVE_GIB non-weight, non-KV memory kept for kernels/graphs/workspace/SWA rings (default 10)
#   LANE_MIN_FREE_GIB free memory that must remain after load at full context (default 10)
#   EXL3_VENV       venv with the ROCm torch (default <root>/.venv)
#   EXL3_HSA_LIB    libhsa-runtime64.so.1 to preload (default: what tools/strix_halo/env.sh picks)
#   MIMO_SPEC       DFlash speculation, default 1: ndt MIMO_SPEC_NDT (7), confidence MIMO_SPEC_CONF (0.6). 0 = plain decode,
#                   token-exact reproducible.
#   MIMO_DRAFTER    drafter dir (default <MIMO_MODEL>/drafter, then <MIMO_MODEL>-drafter next to it)
#   LANE_STATE      scratch dir (default ~/cache/mimo-lane)
# Context budget (computed): MemTotal 124.9 GiB - weights 97.8 GiB - reserve 10 - free floor 10 = 7.1 GiB of KV.
#   KV = 9 global-attention layers x (4 kv heads x (192 K + 128 V)) x 2 B = 23040 B/token (fp16); the 39 SWA layers
#   use a fixed 128-token window ring per slot. 7.1 GiB / 23040 B = ~330K tokens -> default ceiling 327680.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LANE_ROOT=${MIMO_ROOT:-$(cd "$HERE/../.." && pwd)}
MODEL=${MIMO_MODEL:-$HOME/models/MiMo-V2.6-Flash-MOPD-EXL3-Yamz}
SERVE=$LANE_ROOT/tools/mimo/serve.py
STATE=${LANE_STATE:-$HOME/cache/mimo-lane}
RESERVE_GIB=${MIMO_RESERVE_GIB:-10}
FREE_FLOOR_GIB=${LANE_MIN_FREE_GIB:-10}
KV_BYTES_PER_TOKEN=23040
die() { echo "serve_mimo: $*" >&2; exit 3; }
[ -f "$MODEL/config.json" ] || die "model pack missing: $MODEL"
[ -f "$SERVE" ] || die "serve.py missing: $SERVE"
[ -d "$LANE_ROOT/exllamav3" ] || die "engine worktree missing: $LANE_ROOT"
grep -q -- "--default-temperature" "$SERVE" || die "$SERVE lacks --default-temperature (lane would be greedy)"
grep -q "_schema_types" "$SERVE" || die "$SERVE lacks the typed tool-call parser (wrong branch)"
ls "$LANE_ROOT"/exllamav3_ext*.so >/dev/null 2>&1 || die "built extension (exllamav3_ext*.so) missing in $LANE_ROOT"

# --- context budget: parse -c / --ctx from the args, refuse what memory cannot hold
ctx=""; args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in -c|--ctx) ctx=${args[$((i+1))]:-};; --ctx=*) ctx=${args[$i]#--ctx=};; esac
done
weights_b=$(du -sb --apparent-size "$MODEL"/*.safetensors | awk '{s+=$1} END {print s}')
total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
cap=$(awk -v t="$total_kb" -v w="$weights_b" -v r="$RESERVE_GIB" -v f="$FREE_FLOOR_GIB" -v k="$KV_BYTES_PER_TOKEN" \
  'BEGIN {kv=t*1024-w-(r+f)*2^30; c=int(kv/k/4096)*4096; if (c<0) c=0; print c}')
cap=${MIMO_CTX_MAX:-$cap}
[ -n "$ctx" ] || die "pass -c CTX (computed ceiling for this box: $cap tokens)"
[ "$ctx" -le "$cap" ] || die "-c $ctx exceeds the memory ceiling $cap (weights $((weights_b>>20)) MiB, reserve ${RESERVE_GIB} GiB, floor ${FREE_FLOOR_GIB} GiB)"
need_kb=$(awk -v w="$weights_b" -v r="$RESERVE_GIB" -v f="$FREE_FLOOR_GIB" -v k="$KV_BYTES_PER_TOKEN" -v c="$ctx" \
  'BEGIN {printf "%d", (w+c*k+(r+f)*2^30)/1024}')
[ "$avail_kb" -ge "$need_kb" ] || die "MemAvailable $((avail_kb>>20)) GiB < needed $((need_kb>>20)) GiB for -c $ctx (another lane loaded?)"

unset ROCM_HOME ROCM_PATH HIP_PATH HIPCXX HIP_DEVICE_LIB_PATH HIPCC_COMPILE_FLAGS_APPEND PYTORCH_ROCM_ARCH LD_LIBRARY_PATH HSA_OVERRIDE_GFX_VERSION
export EXL3_ROOT="$LANE_ROOT"
source "$LANE_ROOT/tools/strix_halo/env.sh"
[ -z "${EXL3_HSA_LIB:-}" ] || export LD_PRELOAD="$EXL3_HSA_LIB"
export PYTHONNOUSERSITE=1
ROOT=$LANE_ROOT
export EXL3_MOE_CFG="${EXL3_MOE_CFG:-2}" EXL3_HIP_PREFILL_MIN_ROWS="${EXL3_HIP_PREFILL_MIN_ROWS:-2}"
export EXL3_LOAD_DEVICE="${EXL3_LOAD_DEVICE-cuda:0}"
mkdir -p "$STATE/tmp" "$STATE/slots"
export EXL3_ROOT="$ROOT" EXL3_REPO="$ROOT" PYTHONPATH="$ROOT" TMPDIR="$STATE/tmp" TQDM_DISABLE=1
hook=off
if [ -n "${MIMO_ABLIT:-}" ]; then
  [ -f "$MIMO_ABLIT" ] || die "MIMO_ABLIT set but not a file: $MIMO_ABLIT"
  [ -f "$ROOT/exllamav3/modules/ablit_runtime.py" ] || die "engine tree has no ablit_runtime.py"
  export EXL3_ABLIT_RUNTIME="$MIMO_ABLIT"; hook=on
elif [ -n "${EXL3_ABLIT_RUNTIME:-}" ]; then
  hook="env:$EXL3_ABLIT_RUNTIME"   # explicit env wins; "off" disables a bundled spec
elif [ -f "$MODEL/uncensor_spec.json" ]; then
  hook="bundled"                   # the engine applies <model>/uncensor_spec.json by itself
fi
echo "serve_mimo: model=$MODEL root=$ROOT ctx=$ctx cap=$cap hook=$hook avail=$((avail_kb>>20))GiB" >&2
# Bit-exact host/prefill speed envs (token ids identical to the unflagged lane); an exported value wins.
for kv in EXL3_HOST_LEAN=1 EXL3_HOST_CUTS=1 EXL3_PF_MSPLIT=2048 EXL3_MPW2X=2; do
  k=${kv%%=*}; [ -n "${!k+x}" ] || export "$kv"
done
spec_args=(--no-dflash)
if [ "${MIMO_SPEC:-1}" = "1" ]; then
  drafter=""
  if [ -n "${MIMO_DRAFTER:-}" ]; then
    [ -f "$MIMO_DRAFTER/model.safetensors" ] || die "MIMO_DRAFTER set but no model.safetensors in: $MIMO_DRAFTER"
    drafter=$MIMO_DRAFTER
  else
    for d in "$MODEL/drafter" "${MODEL%/}-drafter"; do
      [ -f "$d/model.safetensors" ] && { drafter=$d; break; }
    done
  fi
  if [ -n "$drafter" ]; then
    spec_args=(--dflash --no-spec-gate --dynamic-draft --draft-confidence "${MIMO_SPEC_CONF:-0.6}" --drafter "$drafter" --ndt "${MIMO_SPEC_NDT:-7}")
  else
    echo "serve_mimo: no drafter found (looked in $MODEL/drafter); decoding without speculation" >&2
  fi
fi
exec "${EXL3_VENV:-$LANE_ROOT/.venv}/bin/python" -u "$SERVE" --model "$MODEL" "${spec_args[@]}" \
  --slot-save-path "$STATE/slots" --default-temperature 1.0 --default-top-p 0.95 "$@"
