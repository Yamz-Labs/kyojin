#!/usr/bin/env bash
# Prefill chunk size sweep (bigger chunk = better GEMM shapes, fewer per-chunk fixed costs).
# Log: /tmp/chunk.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 EXL3_HIP_F32OUT_VIA_F16=1
for c in 1024 2048 4096 8192; do
  echo "=== chunk $c ==="
  CHUNK=$c P=8192 .venv/bin/python tools/strix_halo/prefill_prof.py 2>&1 | grep -E "^prefill"
  sleep 5
done
echo DONECHUNK
