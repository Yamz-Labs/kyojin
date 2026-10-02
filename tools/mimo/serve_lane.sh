#!/usr/bin/env bash
# llama-swap lane wrapper for the GLM EXL3 server: serve_lane.sh --port N -c CTX [serve.py args...]
set -euo pipefail
LANE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$LANE_ROOT/tools/strix_halo/env.sh"
export EXL3_ROOT="$LANE_ROOT" EXL3_REPO="$LANE_ROOT" PYTHONPATH="$LANE_ROOT" TMPDIR="$LANE_ROOT/scratch/msrv-lane" TQDM_DISABLE=1
mkdir -p "$TMPDIR"
exec "${EXL3_VENV:-$LANE_ROOT/.venv}/bin/python" "$LANE_ROOT/tools/mimo/serve.py" "$@"
