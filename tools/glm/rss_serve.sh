#!/bin/bash
# rssleak1 served-path harness. Runs inside a resource-limited scope (env already sourced,
# cwd = worktree) or locally under gpu-lease:
#   tools/glm/rss_serve.sh <model> <outdir> [n_short] [n_long] [port]
# Starts serve.py on its own port with the lane's -c, drives the brief's request mix through
# HTTP, and leaves the server's own per-request RSS/device/recurrent-cache log in <outdir>.
set -u
M=$1; O=$2; NS=${3:-30}; NL=${4:-10}; PORT=${5:-8299}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=$(cd "$HERE/../.." && pwd)
mkdir -p "$O"
export EXL3_SERVE_RSS_LOG="$O/rss_server.jsonl"
# Diagnostic on the LAST request only (rss_probe counts requests itself)
export EXL3_SERVE_RSS_FINISH=$((NS + NL))
export EXL3_SERVE_DTUNE_PRIME_S=${EXL3_SERVE_DTUNE_PRIME_S:-0}   # skip the 363 s prime: not the variable
export TQDM_DISABLE=1
unset EXL3_SERVE_WARMUP

echo "=== $(date -Is) start serve.py :$PORT ctx ${MAX_CTX:-524288} ==="
python -u "$W/tools/glm/serve.py" --model "$M" --port "$PORT" --max-ctx "${MAX_CTX:-524288}" \
    > "$O/serve.log" 2>&1 &
SP=$!
echo "serve pid $SP"
for i in $(seq 1 180); do
    curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    kill -0 $SP 2>/dev/null || { echo "serve died:"; tail -30 "$O/serve.log"; exit 1; }
    sleep 2
done
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | head -c 200; echo
# baseline RSS after load + warm-up, before any request
awk '/VmRSS|VmHWM/{printf "baseline %s %d MB\n", $1, $2/1024}' /proc/$SP/status

python -u "$W/tools/glm/rss_client.py" --pid $SP --out "$O/rss_client.json" --port "$PORT" \
    --n-short "$NS" --n-long "$NL" --seed "${SEED:-7}"
RC=$?
awk '/VmRSS|VmHWM/{printf "final %s %d MB\n", $1, $2/1024}' /proc/$SP/status
kill -INT $SP 2>/dev/null; sleep 3; kill -9 $SP 2>/dev/null
echo "rss_serve rc=$RC  $(date -Is)"
