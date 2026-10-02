#!/usr/bin/env bash
# Does static -ndt 2 beat the -ndt 3 -dds -dc 0.6 production default on the SIX-PROMPT MEAN?
# A single favourable prompt said yes (47.3 vs 44.7); that is exactly the mistake the recipe
# warns about, so judge it on the distribution. Log: /tmp/ndt_sweep.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
S=tools/strix_halo/prompt_sweep.py

echo "########## A: ndt=2 STATIC (the new candidate) ##########"
LABEL=ndt2_static NDT=2 DDS=0 .venv/bin/python $S
sleep 6
echo "########## B: ndt=3 dynamic dc=0.6 (current recipe default) ##########"
LABEL=ndt3_dyn_dc06 NDT=3 DC=0.6 DDS=1 .venv/bin/python $S
sleep 6
echo "########## C: ndt=2 dynamic dc=0.4 (historical sweep config) ##########"
LABEL=ndt2_dyn_dc04 NDT=2 DC=0.4 DDS=1 .venv/bin/python $S
sleep 6
echo "########## D: ndt=3 STATIC (isolate static-vs-dynamic at fixed depth) ##########"
LABEL=ndt3_static NDT=3 DDS=0 .venv/bin/python $S
echo DONENDTSWEEP
