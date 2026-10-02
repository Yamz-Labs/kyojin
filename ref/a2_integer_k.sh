#!/usr/bin/env bash
# Phase A2 -- validate ref/frac_reconstruct.py against the fork's HIP reconstruct on INTEGER K.
#
# The fork's build has no fractional kernels, so the only tensors both sides can decode are the
# integer-K ones of the MiMo pack: K = 2.0 (18432 groups), 3.0 (3), 4.0 (192), 6.0 (1).
# The reference reads the format right only if it agrees with the kernel on those, bit-for-bit on
# the raw path and to fp16-rounding tolerance on the hadamard path.
#
#   bash ref/a2_integer_k.sh            (small tiles: 8x8, one tensor at a time)
#
# Environment (see tools/strix_halo/env.sh, Trap 1/Trap 5): a working LD_PRELOAD is mandatory at
# runtime; the ROCm SDK paths must NOT be set.
#
# CORRECTION (this box, verified 2026-09-23): env.sh and the recipe both preload
# /usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1, which on this install is Ubuntu's 1.11.0
# (ROCm 5.7 era) and does NOT know gfx1151 -- HSA then prints "Agent creation failed. The GPU node
# has an unrecognized id." and torch reports "No HIP GPUs are available". The 1.18 the recipe
# describes lives in /opt/rocm-7.2.4/lib. Measured, same venv, same process type:
#   1.11.0 (/usr/lib) -> No HIP GPUs are available
#   1.18   (/opt/rocm-7.2.4) -> gfx1151, 120.0 GiB, fp16 matmul OK
#   1.21   (/opt/rocm-10)    -> gfx1151, 120.0 GiB, fp16 matmul OK
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=$HOME/kyojin
PY="$BASE/venv/bin/python"
MODEL="${MODEL:-$HOME/models/mimo26-exl3}"

export LD_PRELOAD="${LD_PRELOAD_HSA:-/opt/rocm-7.2.4/lib/libhsa-runtime64.so.1}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export EXL3_EXT_DIR="$ROOT"

cd "$ROOT"
rc=0
run() {  # run <label> <tensor> <mode>
    echo "==================================================================="
    echo "### $1  $2  mode=$3"
    echo "==================================================================="
    "$PY" ref/validate_hip.py \
        --model "$MODEL" --tensor "$2" --mode "$3" --kt 0 8 --nt 0 8 || rc=1
}
# K = 4.0 : attention projections (192 groups)
run "K4.0 q_proj"  model.layers.0.self_attn.q_proj raw
run "K4.0 q_proj"  model.layers.0.self_attn.q_proj had
# K = 3.0 : layer 0 MLP (3 groups)
run "K3.0 gate"    model.layers.0.mlp.gate_proj     raw
run "K3.0 gate"    model.layers.0.mlp.gate_proj     had
exit $rc
