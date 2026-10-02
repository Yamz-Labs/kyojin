#!/usr/bin/env bash
# Decode/prefill vs prompt depth at the max context config (q4 cache, 262144). Log: /tmp/ctx4.log
# Prefill at 350 tok/s means a 200k prompt takes ~10 min; total ~35 min.
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
S=tools/strix_halo/ctx_sweep.py
echo "=== q4 262144: depth sweep ==="
CQ=4 CS=262144 PROMPTS=1024,8192,32768,65536,131072,200000,250000 NTOK=128 .venv/bin/python $S 2>&1 | grep "^{"
echo "=== fp16 98304 (max fp16): depth sweep ==="
CS=98304 PROMPTS=1024,32768,90000 NTOK=128 .venv/bin/python $S 2>&1 | grep "^{"
echo DONE4
