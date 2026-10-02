#!/bin/bash
# Usage: tools/glm/run.sh <fast|ref|prof> <out.json> [extra harness args]
# Run on an otherwise idle GPU (the harness loads the full model).
cd "$(dirname "$0")/../.."
# shared runtime env (the one every MiMo run uses): /opt/rocm HSA preload, MOE_CFG=2, PREFILL_MIN_ROWS=2
source $HOME/kyojin/tools/strix_halo/env.sh > /dev/null
export EXL3_ROOT="$PWD" EXL3_REPO="$PWD" PYTHONPATH="$PWD"
mode=$1; out=$2; shift 2
if [ "$mode" = ref ]; then
  # generic path: every fast kernel knob off
  export EXL3_DEC=0 EXL3_DEC_MOE=0 EXL3_MLA_DEC=0 EXL3_MOE_WMMA=0 EXL3_HIP_GROUPED_MOE=0 EXL3_HIP_GROUPED_MOE_PREFILL=0 \
         EXL3_GEMV=0 EXL3_BC_ATTN=0 EXL3_BC_DSA=0 EXL3_HIP_ROUTER=0 EXL3_DEC_ROUTER=0 EXL3_HIP_GR_MIX_Q8=0
fi
if [ "$mode" = prof ]; then
  d="${out%.json}.rocprof"; rm -rf "$d"
  exec /opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv -d "$d" -- \
    $HOME/kyojin/.venv/bin/python tools/glm/glm_base.py prof -o "$out" "$@"
fi
exec $HOME/kyojin/.venv/bin/python tools/glm/glm_base.py "$mode" -o "$out" "$@"
