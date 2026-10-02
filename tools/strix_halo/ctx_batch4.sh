#!/usr/bin/env bash
# Double-confirm max context. Log: /tmp/ctx5.log
#  A) q4 262144: fill the cache to the brim with a RANDOM (uncacheable, non-repeating) prompt -> true cold prefill + decode at the edge
#  B) does q4 327680 / 393216 even load (beyond max_position_embeddings; for the record only)
#  C) fp16 98304 depth (previous batch was killed)
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
S=tools/strix_halo/ctx_sweep.py
echo "=== A: q4 262144, random prompt, fill to the brim ==="
CQ=4 CS=262144 PROMPTS=4096 FILL=1 RANDOM_PROMPT=1 NTOK=128 .venv/bin/python $S 2>&1 | grep "^{"
sleep 6
echo "=== B: beyond 262144 (load only) ==="
for cs in 327680 393216 524288; do CQ=4 CS=$cs PROMPTS=1024 NTOK=16 .venv/bin/python $S 2>&1 | grep "^{" | head -1; sleep 6; done
echo "=== C: fp16 98304 depth ==="
CS=98304 PROMPTS=1024,32768 FILL=1 NTOK=128 .venv/bin/python $S 2>&1 | grep "^{"
echo DONE5
