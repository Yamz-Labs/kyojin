#!/usr/bin/env bash
# Build libckmla.so: CK fmha_fwd fp16 batch, hd 256 (gfx1151 tile), causal / window mask, elementwise bias, learnable sink, behind ckmla_fwd.
# Same env as tools/strix_halo/ck_fmha/build.sh (CK_SRC checkout at the pinned commit, EXL3_ROCM_SDK, GCC_DIR). OUT=$EXL3_CKMLA_OUT.
set -euo pipefail
: "${CK_SRC:?set CK_SRC}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${EXL3_CKMLA_OUT:-${XDG_CACHE_HOME:-$HOME/.cache}/kyojin/ck-fmha/libckmla.so}
CXX=${ROCM_CLANG:-}
for c in "${EXL3_ROCM_SDK:-/nonexistent}/lib/llvm/bin/amdclang++" "${EXL3_ROCM_SDK:-/nonexistent}/llvm/bin/amdclang++" /opt/rocm/bin/amdclang++; do
    if [ -z "$CXX" ] && [ -x "$c" ]; then CXX=$c; fi
done
: "${CXX:?no amdclang++ found: set ROCM_CLANG}"
GCC_DIR=${GCC_DIR:-/usr/lib/gcc/x86_64-linux-gnu/13}
F=$CK_SRC/example/ck_tile/01_fmha
B=$(mktemp -d); trap 'rm -rf "$B"' EXIT
mkdir -p "$B/gen" "$B/obj" "$(dirname "$OUT")"
python3 "$F/generate.py" --targets gfx1151 --api fwd --optdim ${CKMLA_HD:-256} -r 0 \
    -f "*fp16*batch*_nlogits_*_nlse_ndropout_nskip_nqscale*" -o "$B/gen" >/dev/null
FLAGS="--gcc-install-dir=$GCC_DIR -std=c++20 -O3 -fPIC -x hip --offload-arch=gfx1151 -I$CK_SRC/include -I$F -I$B/gen \
 -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero -DCK_TILE_FMHA_FWD_SPLITKV_API=0 -DCK_TILE_FMHA_FWD_APPENDKV_API=0 \
 -DCK_TILE_FMHA_FWD_PAGEDKV_API=0 -DCK_TILE_FMHA_FWD_BATCH_PREFILL_API=0 -w"
ls "$B"/gen/*.cpp "$HERE/ckmla_shim.cpp" | xargs -P "${CK_JOBS:-8}" -I{} sh -c "$CXX $FLAGS -c {} -o $B/obj/\$(basename {} .cpp).o"
"$CXX" --gcc-install-dir="$GCC_DIR" -shared -fPIC --offload-arch=gfx1151 "$B"/obj/*.o -o "$OUT"
echo "built $OUT ($(ls "$B"/gen/*.cpp | wc -l) instance files)"
