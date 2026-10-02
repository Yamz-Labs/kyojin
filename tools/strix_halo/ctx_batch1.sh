#!/usr/bin/env bash
# Find the max cache size that loads, then measure decode/prefill vs prompt depth. Log: /tmp/ctx.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
S=tools/strix_halo/ctx_sweep.py
echo "=== phase 1: does it load? (memory after load) ==="
for cs in 32768 65536 131072 262144; do
  CS=$cs PROMPTS=1024 NTOK=32 .venv/bin/python $S 2>&1 | grep "^{" ; sleep 6
done
echo DONE1
