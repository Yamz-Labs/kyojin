#!/usr/bin/env bash
# 2.05bpw pack: can it clear 60 tok/s? Log: /tmp/bpw205.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
M=~/models/Qwen3.8-Flash-Next-EXL3-2.05
echo "=== 2.05bpw greedy ndt=3 dc=0.6 n=512 ==="
.venv/bin/python tools/strix_halo/bench_mtp.py -m "$M" -n 512 -ndt 3 -dds -g -dc 0.6
echo "=== 2.05bpw greedy ndt=2 n=512 ==="
.venv/bin/python tools/strix_halo/bench_mtp.py -m "$M" -n 512 -ndt 2 -dds -g -dc 0.4
echo "=== 3.05bpw reference ==="
.venv/bin/python tools/strix_halo/bench_mtp.py -n 512 -ndt 3 -dds -g -dc 0.6
echo DONE205
