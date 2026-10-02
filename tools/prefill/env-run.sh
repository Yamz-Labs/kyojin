#!/bin/bash
# Runtime env for this worktree: run <cmd...> with the worktree's extension on PYTHONPATH.
W="$(cd "$(dirname "$0")/../.." && pwd)"
source $W/tools/strix_halo/env.sh
export PYTHONPATH=$W EXL3_REPO=$W
cd $W
exec "$@"
