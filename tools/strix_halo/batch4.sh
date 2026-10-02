#!/usr/bin/env bash
# int8 gr_mix: decode-path fidelity + 6-prompt sweeps. Log: /tmp/batch4.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
echo "=== tie_check EXL3_HIP_GR_MIX_Q8 0 vs 1 (decode path, teacher-forced) ==="
NTOK=96 .venv/bin/python tools/strix_halo/tie_check.py EXL3_HIP_GR_MIX_Q8 0 1 2>&1 | grep -v Warning
sleep 6
echo "=== 6-prompt sweep q8 on, ndt=2 dc=0.4 ==="
EXL3_HIP_GR_MIX_Q8=1 NDT=2 DC=0.4 .venv/bin/python tools/strix_halo/prompt_sweep.py 2>&1 | tail -4
echo "=== 6-prompt sweep q8 on, ndt=3 dc=0.6 ==="
EXL3_HIP_GR_MIX_Q8=1 NDT=3 DC=0.6 .venv/bin/python tools/strix_halo/prompt_sweep.py 2>&1 | tail -4
echo DONE
