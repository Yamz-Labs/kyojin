#!/usr/bin/env bash
# Determinism check for the skinny GEMM. Log: /tmp/batch2.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
G=tools/strix_halo/greedy_ab.py
echo "=== noise floor, MTP off: skinny=1 vs skinny=1 ==="
NOMTP=1 .venv/bin/python $G EXL3_HIP_SKINNY_GEMM 1 1 2>&1 | grep -v Warning; sleep 6
echo "=== A/B, MTP off: skinny=0 vs skinny=1 ==="
NOMTP=1 .venv/bin/python $G EXL3_HIP_SKINNY_GEMM 0 1 2>&1 | grep -v Warning; sleep 6
echo "=== noise floor, MTP on: skinny=1 vs skinny=1 ==="
.venv/bin/python $G EXL3_HIP_SKINNY_GEMM 1 1 2>&1 | grep -v Warning
echo DONE
