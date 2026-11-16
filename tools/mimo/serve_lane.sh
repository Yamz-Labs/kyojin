#!/usr/bin/env bash
# llama-swap lane wrapper for the GLM EXL3 server: serve_lane.sh --port N -c CTX [serve.py args...]
set -euo pipefail
LANE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$LANE_ROOT/tools/strix_halo/env.sh"
export EXL3_ROOT="$LANE_ROOT" EXL3_REPO="$LANE_ROOT" PYTHONPATH="$LANE_ROOT" TMPDIR="$LANE_ROOT/scratch/msrv-lane" TQDM_DISABLE=1
mkdir -p "$TMPDIR"

# External supervisor: unified-memory boxes do not reliably OOM-kill.
MEMGUARD="${MEMGUARD:-auto}"
MODEL=""
mg_args=("$@")
for ((i = 0; i < ${#mg_args[@]}; i++)); do
    case "${mg_args[$i]}" in
        --model) MODEL="${mg_args[$((i + 1))]:-}" ;;
        --model=*) MODEL="${mg_args[$i]#--model=}" ;;
    esac
done
use_mg=0
case "$MEMGUARD" in
    1) use_mg=1 ;;
    auto)
        if compgen -G "/sys/class/drm/card*/device/mem_info_gtt_total" > /dev/null; then
            use_mg=1
        elif [ -n "$MODEL" ] && [ -d "$MODEL" ]; then
            model_bytes=$(du -sb --apparent-size "$MODEL"/*.safetensors 2> /dev/null | awk '{s+=$1} END {print s+0}' || true)
            if [ "${model_bytes:-0}" -ge $((48 * 1024 * 1024 * 1024)) ]; then use_mg=1; fi
        fi
        ;;
esac
MG=()
if [ "$use_mg" = "1" ] && [ -f "$LANE_ROOT/exllamav3/guard.py" ]; then
    MG=(python "$LANE_ROOT/exllamav3/guard.py" --label mimo
        --log "$TMPDIR/memguard.log" --sample-every 5 --fadvise-every 5)
    [ -n "$MODEL" ] && [ -d "$MODEL" ] && MG+=(--fadvise-dir "$MODEL")
    MG+=(--)
    echo "serve_lane: MEMGUARD=on (host floor ${KYOJIN_OOM_GUARD_HOST_FREE_GIB:-7} GiB, log $TMPDIR/memguard.log)" >&2
elif [ "$MEMGUARD" = "1" ]; then
    echo "serve_lane: MEMGUARD=1 requested but exllamav3/guard.py is missing" >&2
    exit 1
fi
if [ ${#MG[@]} -gt 0 ]; then
    exec "${MG[@]}" "${EXL3_VENV:-$LANE_ROOT/.venv}/bin/python" "$LANE_ROOT/tools/mimo/serve.py" "$@"
fi
exec "${EXL3_VENV:-$LANE_ROOT/.venv}/bin/python" "$LANE_ROOT/tools/mimo/serve.py" "$@"
