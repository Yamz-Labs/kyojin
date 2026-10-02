#!/usr/bin/env bash
set -euo pipefail
SRV_ROOT="${GLM_SRV_ROOT:-$PWD}"
export TMPDIR="$SRV_ROOT/scratch/glm-serve"
mkdir -p "$TMPDIR"
PORT="${GLM_SRV_PORT:-18080}"
BASE="http://127.0.0.1:$PORT/v1"
source "$SRV_ROOT/tools/strix_halo/env.sh"
export EXL3_MOE_UNION_V2=1 EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 EXL3_BLOCK_GRAPH=1 EXL3_BLOCK_GRAPH_MLA=2
PY="${EXL3_VENV:-$SRV_ROOT/.venv}/bin/python"
trap 'kill "$PID" 2>/dev/null || true' EXIT
"$PY" "$SRV_ROOT/tools/glm/serve.py" --model "${GLM_SRV_MODEL:-$HOME/models/glm53-exl3-td205}" --port "$PORT" >"$TMPDIR/server.log" 2>&1 & PID=$!
for _ in $(seq 1 900); do curl -sS --fail-with-body "$BASE/models" >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || { tail -30 "$TMPDIR/server.log"; exit 4; }; sleep 1; done; echo "ready after ${SECONDS}s"
curl -sS --fail-with-body "$BASE/chat/completions" -H content-type:application/json -d '{"model":"glm-5.3-exl3","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":1024,"temperature":0}'
printf '\n-- streamed --\n'
curl -N -sS --fail-with-body "$BASE/chat/completions" -H content-type:application/json -d '{"model":"glm-5.3-exl3","messages":[{"role":"user","content":"Say hello."}],"max_tokens":1024,"temperature":0,"stream":true}' > sse.out; S=sse.out; echo "chunks $(grep -c "^data: {" "$S")"; head -c 600 "$S"; echo; tail -c 700 "$S"
printf '\n-- tool --\n'
curl -sS --fail-with-body "$BASE/chat/completions" -H content-type:application/json -d '{"model":"glm-5.3-exl3","messages":[{"role":"user","content":"Use the weather tool for Paris."}],"tools":[{"type":"function","function":{"name":"weather","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":1024,"temperature":0}'
for i in 1 2; do t0=$(date +%s.%N); curl -sS --fail-with-body "$BASE/chat/completions" -H content-type:application/json -d "{\"model\":\"glm-5.3-exl3\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 300-word story about a lighthouse.\"}],\"max_tokens\":400,\"temperature\":0}" | python3 -c "import json,sys;d=json.load(sys.stdin);print(\"usage\",d.get(\"usage\"))"; echo "bench wall $(echo "$(date +%s.%N) - $t0" | bc)s"; done
printf '\n-- server log tail --\n'
tail -20 "$TMPDIR/server.log"
