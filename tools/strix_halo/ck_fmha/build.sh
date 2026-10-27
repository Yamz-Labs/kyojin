#!/usr/bin/env bash
# Build libckpf.so: Composable Kernel fmha_fwd (stock gfx115 fp16 tiles, hd128 + hd256, batch mode, causal/none) behind
# a C ABI (ckpf_fwd) used by EXL3_PF_CK=1. Nothing is fetched here: CK_SRC is a local ROCm/composable_kernel checkout
# (tested at 6f44fdf90f69adee742ec3660b95d32963b9bd2d). Output: $EXL3_CK_OUT (default ~/.cache/kyojin/ck-fmha/libckpf.so).
# Needs the ROCm clang (ROCM_CLANG, default /opt/rocm/bin/amdclang++) and a libstdc++ GCC install dir (GCC_DIR, default 13).
set -euo pipefail
: "${CK_SRC:?set CK_SRC to a composable_kernel checkout}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${EXL3_CK_OUT:-${XDG_CACHE_HOME:-$HOME/.cache}/kyojin/ck-fmha/libckpf.so}
CXX=${ROCM_CLANG:-/opt/rocm/bin/amdclang++}
GCC_DIR=${GCC_DIR:-/usr/lib/gcc/x86_64-linux-gnu/13}
F=$CK_SRC/example/ck_tile/01_fmha
B=$(mktemp -d); trap 'rm -rf "$B"' EXIT
mkdir -p "$B/gen" "$B/obj" "$(dirname "$OUT")"
python3 "$F/generate.py" --targets gfx1151 --api fwd --optdim 128,256 -r 0 \
    -f "*fp16*batch*_nlogits_nbias_*_nlse_ndropout_nskip_nqscale*_nsink*" -o "$B/gen" >/dev/null
FLAGS="--gcc-install-dir=$GCC_DIR -std=c++20 -O3 -fPIC -x hip --offload-arch=gfx1151 -I$CK_SRC/include -I$F -I$B/gen \
 -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero -DCK_TILE_FMHA_FWD_SPLITKV_API=0 -DCK_TILE_FMHA_FWD_APPENDKV_API=0 \
 -DCK_TILE_FMHA_FWD_PAGEDKV_API=0 -DCK_TILE_FMHA_FWD_BATCH_PREFILL_API=0 -w"
ls "$B"/gen/*.cpp "$HERE/ckpf_shim.cpp" | nice -n 19 taskset -c 16-31 xargs -P 8 -I{} sh -c "$CXX $FLAGS -c {} -o $B/obj/\$(basename {} .cpp).o"
"$CXX" --gcc-install-dir="$GCC_DIR" -shared -fPIC --offload-arch=gfx1151 "$B"/obj/*.o -o "$OUT"
echo "built $OUT"
