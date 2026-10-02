#!/usr/bin/env bash
# Max context: fp16 cache bisect + quantized cache, then depth sweep at the max. Log: /tmp/ctx3.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
S=tools/strix_halo/ctx_sweep.py
echo "=== fp16 cache: 104k? ==="
CS=106496 PROMPTS=1024 NTOK=16 .venv/bin/python $S 2>&1 | grep "^{" | head -1; sleep 6
echo "=== q4 cache: how far? ==="
for cs in 131072 196608 262144; do
  CQ=4 CS=$cs PROMPTS=1024 NTOK=16 .venv/bin/python $S 2>&1 | grep "^{" | head -1; sleep 6
done
echo "=== q8 cache 131072 ==="
CQ=8 CS=131072 PROMPTS=1024 NTOK=16 .venv/bin/python $S 2>&1 | grep "^{" | head -1; sleep 6
echo DONE3
