#!/usr/bin/env bash
# Start one lane launcher, run parity_check.py save against it, stop it.
#   parity_run.sh OUT.json MODEL_ID -- <launcher> [launcher args...]
# The launcher gets --port 18091 --model-id MODEL_ID -c 8192 appended (override the port with PARITY_PORT).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; MID=$2; shift 2; [ "$1" = "--" ] && shift
PORT=${PARITY_PORT:-18091}
"$@" --port "$PORT" --model-id "$MID" -c 8192 > "${OUT%.json}.server.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 1800); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { tail -20 "${OUT%.json}.server.log"; exit 4; }
  sleep 1
done
python3 "$HERE/parity_check.py" save --port "$PORT" --model-id "$MID" --out "$OUT"
