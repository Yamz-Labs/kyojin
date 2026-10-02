#!/bin/bash
cd $HOME/kyojin
source $HOME/kyojin/tools/strix_halo/env.sh
export PYTHONPATH=$PWD EXL3_REPO=$PWD
rm -rf profiles/kprof; /opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv -d profiles/kprof -- python tools/decode/kprof.py ${1:-dec_r2} > /dev/null 2>&1
f=$(find profiles/kprof -name "*kernel_stats.csv" | head -1); python3 -c "
import csv,sys
for r in csv.DictReader(open('$f')):
    n=r['Name'][:90]
    if 'exl3dec' in n or 'router' in n: print(f\"{n:<90} calls {r['Calls']:>4} avg {float(r['AverageNs'])/1e3:8.1f} us min {float(r['MinNs'])/1e3:8.1f}\")
"
