#!/bin/bash
# Small GPU test runner (caller wraps with gpu-run.sh): runtime env + this worktree.
WT=$HOME/kyojin
source $HOME/kyojin/tools/strix_halo/env.sh
export PYTHONPATH=$WT EXL3_REPO=$WT
cd $WT
exec python "$@"
