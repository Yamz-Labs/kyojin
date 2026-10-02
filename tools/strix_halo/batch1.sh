#!/usr/bin/env bash
# Batch: greedy A/B for the skinny GEMM, then a fine ndt/dc sweep on the default prompt.
# Logs on the remote box; tail /tmp/batch1.log.
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
A=tools/strix_halo/ab.sh
echo "=== greedy token A/B: EXL3_HIP_SKINNY_GEMM ==="
.venv/bin/python tools/strix_halo/greedy_ab.py EXL3_HIP_SKINNY_GEMM 2>&1 | grep -v Warning
sleep 6
echo "=== ndt/dc sweep, n=512, greedy, skinny on ==="
for cfg in "3 0.5" "3 0.6" "3 0.7" "4 0.6" "4 0.7" "5 0.7"; do
  set -- $cfg
  bash $A sw_ndt$1_dc$2 "" -n 512 -ndt $1 -dds -g -dc $2
done
echo DONE
