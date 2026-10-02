#!/bin/bash
# Run tools/decode/dev.py on the full MiMo pack with this worktree's extension.
#   run.sh [--prof OUTDIR] <dev.py args...>     (caller wraps with big-gpu-run.sh)
WT=$HOME/kyojin
source $HOME/kyojin/tools/strix_halo/env.sh
export PYTHONPATH=$WT EXL3_REPO=$WT
cd $WT
PROF=()
if [ "$1" = "--prof" ]; then PROF=(/opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d "$2" --); shift 2; fi
exec "${PROF[@]}" python tools/decode/dev.py -m $HOME/models/mimo26-exl3 "$@"
