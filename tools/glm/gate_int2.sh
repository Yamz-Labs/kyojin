#!/bin/bash
# Usage (test box): tools/glm/gate_int2.sh <out.json>   -- env as glm-next r3 gate
cd "$(dirname "$0")/../.."
source "$PWD/tools/strix_halo/env.sh" > /dev/null
export EXL3_ROOT="$PWD" EXL3_REPO="$PWD" PYTHONPATH="$PWD" TQDM_DISABLE=1
export EXL3_BLOCK_GRAPH=1 EXL3_BLOCK_GRAPH_MLA=2
mkdir -p "$(dirname "$1")"
T0=$(date +%s)
"${EXL3_VENV:-$PWD/.venv}/bin/python" tools/glm/gate_int2.py "$1" 2>&1 | tee "${1%.json}.log" | grep --line-buffered -E "^(loaded|\[knobs|\[smoke|\[speed|\[ppl [a-z]+\] [0-9]|\[gtt|SUMMARY|DONE)|Error|error|Traceback"
echo "gate rc=${PIPESTATUS[0]} wall=$(( $(date +%s) - T0 ))s"
