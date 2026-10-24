#!/bin/bash
# DS4.1 quality gate: top-1 agreement + KLD of a pack against the original model's logits (see gate.py).
#   tools/ds41qual/run_gate.sh PACK_DIR|BASE_URL [REF_DIR] [extra gate.py args]
# PACK_DIR: serves it with tools/ds41/serve.py (loader ds41_serving) on PORT (default 8399) under the GPU lock, then stops it.
# BASE_URL (http...): scores an already running server. REF_DIR defaults to data/gate_ref (copy of OUT/gate_ref written
# by the conversion hook tools/ds41/evalref.py). env: PY, GPU_LOCK, PORT, CTX (default 16384), SERVE_ARGS, OUT.
# GPU rule: DS4.1 GPU jobs run on the second machine only (iGPU lock ~/ds41_work/igpu.gpu.lock), never on the box GPU.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$(dirname "$HERE")")
TARGET=${1:?usage: run_gate.sh PACK_DIR|BASE_URL [REF_DIR]}; shift
REF=${1:-$HERE/data/gate_ref}; [ $# -gt 0 ] && shift
PY=${PY:-python3}; PORT=${PORT:-8399}; OUT=${OUT:-$PWD/gate_result.json}
[ -f "$REF/manifest.json" ] || { echo "no reference at $REF (the conversion writes OUT/gate_ref)"; exit 2; }
case "$TARGET" in
  http*) exec $PY "$HERE/gate.py" --base-url "$TARGET" --ref "$REF" --out "$OUT" "$@" ;;
esac
LOCK=${GPU_LOCK:-$HOME/ds41_work/igpu.gpu.lock}
LOG=${LOG:-$PWD/gate_serve.log}
flock "$LOCK" bash -c '
  cd "$0/tools/ds41" || exit 1
  '"$PY"' serve.py --model "$1" --loader ds41_serving:load_model -c "$2" --port "$3" $4 > "$5" 2>&1 &
  SPID=$!
  trap "kill $SPID 2>/dev/null; wait $SPID 2>/dev/null" EXIT
  for i in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$3/health" >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "server died, see $5"; exit 3; }
    sleep 5
  done
  '"$PY"' "$0/tools/ds41qual/gate.py" --base-url "http://127.0.0.1:$3" --ref "$6" --out "$7" "${@:8}"
' "$ROOT" "$TARGET" "${CTX:-16384}" "$PORT" "${SERVE_ARGS:-}" "$LOG" "$REF" "$OUT" "$@"
