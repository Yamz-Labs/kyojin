#!/usr/bin/env bash
# Graph-capture A/B. Log: /tmp/graph60.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
A=tools/strix_halo/ab.sh
echo "=== graph OFF ==="
bash $A g_off "EXL3_BLOCK_GRAPH=0" -n 512 -ndt 3 -dds -g -dc 0.6
echo "=== graph ON slots=16 (eager fallback, no debug) ==="
bash $A g_on16 "EXL3_BLOCK_GRAPH=1 EXL3_BLOCK_GRAPH_MAX_SLOTS=16 EXL3_BLOCK_GRAPH_CAPTURE_FALLBACK=eager" -n 512 -ndt 3 -dds -g -dc 0.6
echo "=== graph ON slots=64 ==="
bash $A g_on64 "EXL3_BLOCK_GRAPH=1 EXL3_BLOCK_GRAPH_MAX_SLOTS=64 EXL3_BLOCK_GRAPH_CAPTURE_FALLBACK=eager" -n 512 -ndt 3 -dds -g -dc 0.6
echo DONEGRAPH
