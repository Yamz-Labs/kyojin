#!/usr/bin/env bash
# llama-swap lane wrapper for the GLM EXL3 server: serve_lane.sh --port N -c CTX [serve.py args...]
set -euo pipefail
LANE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$LANE_ROOT/tools/strix_halo/env.sh"
export EXL3_ROOT="$LANE_ROOT" EXL3_REPO="$LANE_ROOT" PYTHONPATH="$LANE_ROOT" TMPDIR="$LANE_ROOT/scratch/glm-serve" TQDM_DISABLE=1
mkdir -p "$TMPDIR"

# External supervisor (exllamav3/guard.py): the reliable backstop for a unified pool.
# auto = arm on Linux when /proc/meminfo is readable. Set MEMGUARD=0 to run without
# a supervisor (the old lane behavior); the supervisor is otherwise on by default.
MEMGUARD="${MEMGUARD:-auto}"
MODEL=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --model) MODEL="${args[$((i + 1))]:-}" ;;
        --model=*) MODEL="${args[$i]#--model=}" ;;
    esac
done
use_mg=0
case "$MEMGUARD" in
    1) use_mg=1 ;;
    0) ;;
    auto)
        if [ -r /proc/meminfo ]; then use_mg=1; fi
        ;;
esac
MG=()
if [ "$use_mg" = "1" ] && [ -f "$LANE_ROOT/exllamav3/guard.py" ]; then
    MG=(python "$LANE_ROOT/exllamav3/guard.py" --label glm
        --log "$TMPDIR/memguard.log" --sample-every 5 --fadvise-every 5)
    [ -n "$MODEL" ] && [ -d "$MODEL" ] && MG+=(--fadvise-dir "$MODEL")
    MG+=(--)
    echo "serve_lane: MEMGUARD=on (host floor MEMGUARD_FLOOR/default, log $TMPDIR/memguard.log)" >&2
elif [ "$MEMGUARD" = "1" ]; then
    echo "serve_lane: MEMGUARD=1 requested but exllamav3/guard.py is missing" >&2
    exit 1
fi
if [ ${#MG[@]} -gt 0 ]; then
    exec "${MG[@]}" "${EXL3_VENV:-$LANE_ROOT/.venv}/bin/python" "$LANE_ROOT/tools/glm/serve.py" "$@"
fi
exec "${EXL3_VENV:-$LANE_ROOT/.venv}/bin/python" "$LANE_ROOT/tools/glm/serve.py" "$@"
