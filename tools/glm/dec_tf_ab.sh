#!/bin/bash
# Decode-path A/B, one load (tools/glm/dec_tf_ab.py). Env: D (log dir), TAG, ARMS (json), DEPTHS,
# R38_REPS/R38_NDEC/R38_NNLL, TF_DEPTH/TF_REPS. Run under gpu-lease.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
D=${D:-logs-r39}; mkdir -p "$D"
source $HOME/kyojin/tools/strix_halo/env.sh > /dev/null
export EXL3_ROOT="$PWD" EXL3_REPO="$PWD" PYTHONPATH="$PWD"
export TQDM_DISABLE=1 EXL3_BLOCK_GRAPH=${BG:-1} EXL3_BLOCK_GRAPH_MLA=2
PY=$HOME/kyojin/.venv/bin/python
$PY tools/glm/dec_tf_ab.py $D/${TAG:-ab}.json "$ARMS" ${DEPTHS:-4096,32768} > $D/${TAG:-ab}.log 2>&1
echo "${TAG:-ab} rc=$?" >> $D/status
