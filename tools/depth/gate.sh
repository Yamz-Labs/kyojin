#!/usr/bin/env bash
# Depth gate for one model:  tools/depth/gate.sh MODEL_DIR glm|qwen|none [extra profile.py args]
#   env: TOL (default 0.05)  FLOOR_MS (0.3)  DEPTHS (8192,...,500000)  NDT (2 glm / 3 qwen)  CACHE_BITS (16)  OUT (scratch/depth)
# Builds the ideal curve from config + pack, profiles the model in-process at synthetic depths inside the GPU lock
# (quiet.sh), and exits 0 only if measured per-step cost stays within TOL of "shallow cost + ideal growth" everywhere.
set -euo pipefail
MODEL=$1; SERVE=${2:-none}; shift 2 || shift $#
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
DEPTHS=${DEPTHS:-8192,32768,65536,131072,262144,500000}
NDT=${NDT:-$([ "$SERVE" = qwen ] && echo 3 || echo 2)}
CB=${CACHE_BITS:-16}
OUT=${OUT:-$EXL3_ROOT/scratch/depth}; mkdir -p "$OUT"
TAG=$(basename "$MODEL")-cb$CB
cd "$HOME"                                        # cwd outside the agent trees: the ideal step needs no GPU
python "$HERE/ideal.py" "$MODEL" --cache-bits "$CB" --ndt "$NDT" --depths "$DEPTHS" --json "$OUT/$TAG.ideal.json" > "$OUT/$TAG.ideal.txt"
cat "$OUT/$TAG.ideal.txt"
cd "$EXL3_ROOT"
flock "${GPU_LOCK:-$HOME/ds41_work/gpu.lock}" "${QUIET:-$HOME/ds41_work/cur1510/quiet.sh}" \
  python "$HERE/depth_profile.py" "$MODEL" --serve "$SERVE" -o "$OUT/$TAG.profile.json" --depths "$DEPTHS" --cache-bits "$CB" --ndt "$NDT" --kernels "$@"
python "$HERE/compare.py" "$OUT/$TAG.ideal.json" "$OUT/$TAG.profile.json" --tol "${TOL:-0.05}" --floor-ms "${FLOOR_MS:-0.3}" --json "$OUT/$TAG.gate.json"
