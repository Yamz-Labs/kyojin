#!/usr/bin/env bash
# int8 gr_mix: e2e A/B then PPL. Log: /tmp/batch3.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
A=tools/strix_halo/ab.sh
bash $A q8_on  "EXL3_HIP_GR_MIX_Q8=1" -n 512 -ndt 3 -dds -g -dc 0.6
bash $A q8_off "EXL3_HIP_GR_MIX_Q8=0" -n 512 -ndt 3 -dds -g -dc 0.6
bash $A q8_on_ndt2 "EXL3_HIP_GR_MIX_Q8=1" -n 512 -ndt 2 -dds -g
echo "=== PPL q8 on ==="
EXL3_HIP_GR_MIX_Q8=1 .venv/bin/python eval/ppl.py -m ~/models/Qwen3.8-Flash-Next-EXL3 -r 20 -l 1024 > /tmp/ppl_q8.raw 2>&1
grep -iE "perplexity" /tmp/ppl_q8.raw | tail -1
echo DONE
