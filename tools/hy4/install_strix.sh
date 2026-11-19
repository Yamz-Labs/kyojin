#!/usr/bin/env bash
# One-time build of the Hy4-flash kernels on a Strix Halo machine (gfx1151). Run from anywhere; needs the repository venv (see doc/install.md).
#   bash tools/hy4/install_strix.sh
# Builds: hipsp/libspmla.so (sparse prefill attention), ~/.cache/exl3_gm_ext/exl3_gm_ext.so (MoE expert kernels), ~/.cache/kyojin/ck-fmha/libckmla.so (CK latent attention).
# Env: EXL3_VENV / VIRTUAL_ENV (default <repo>/.venv), CK_SRC (a composable_kernel checkout at the pinned commit; default: fetched once into ~/.cache/kyojin/ck-src),
#      EXL3_GM_BUILD_DIR, EXL3_CKMLA_OUT (other output places; set HY4_GM_EXT / EXL3_CKMLA_SO to the same paths when serving).
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT" || exit 1
die() { echo "install_strix: $*" >&2; exit 1; }
export EXL3_VENV=${EXL3_VENV:-${VIRTUAL_ENV:-$ROOT/.venv}}; export VIRTUAL_ENV=$EXL3_VENV
[ -x "$EXL3_VENV/bin/python" ] || die "no venv at $EXL3_VENV: create it as in doc/install.md or set EXL3_VENV"
export PATH=$EXL3_VENV/bin:$PATH
ls "$ROOT"/exllamav3_ext*.so >/dev/null 2>&1 || die "the extension of the repository is not built: run ./build.sh as in doc/install.md first"
command -v rocm-sdk >/dev/null || die "rocm-sdk not found in the venv: install the ROCm devel wheel as in doc/install.md"
export EXL3_ROCM_SDK=${EXL3_ROCM_SDK:-$(rocm-sdk path --root)}
CK_PIN=6f44fdf90f69adee742ec3660b95d32963b9bd2d
if [ -z "${CK_SRC:-}" ]; then
    CK_SRC=${XDG_CACHE_HOME:-$HOME/.cache}/kyojin/ck-src
    if [ "$(git -C "$CK_SRC" rev-parse HEAD 2>/dev/null)" != "$CK_PIN" ]; then
        echo "install_strix: fetching composable_kernel $CK_PIN into $CK_SRC"
        mkdir -p "$CK_SRC" && git -C "$CK_SRC" init -q \
            && git -C "$CK_SRC" fetch -q --depth 1 https://github.com/ROCm/composable_kernel.git "$CK_PIN" \
            && git -C "$CK_SRC" checkout -q FETCH_HEAD || die "could not fetch composable_kernel: set CK_SRC to a checkout of $CK_PIN"
    fi
fi
export CK_SRC
source tools/strix_halo/env.sh build || die "env.sh build failed"
set -e
echo "install_strix: [1/3] sparse prefill attention (hipsp/libspmla.so)"
bash hipsp/build.sh hipsp/sp_mla.cu hipsp/libspmla.so
echo "install_strix: [2/3] MoE expert kernels (exl3_gm_ext)"
EXL3_GM_BUILD=1 python -c "from exllamav3 import gm_ext; assert gm_ext.get() is not None, \"gm_ext build failed\""
echo "install_strix: [3/3] CK latent attention (libckmla.so)"
bash tools/strix_halo/ck_fmha/build_mla.sh
GM=${EXL3_GM_BUILD_DIR:-$HOME/.cache/exl3_gm_ext}; CK=${EXL3_CKMLA_OUT:-${XDG_CACHE_HOME:-$HOME/.cache}/kyojin/ck-fmha/libckmla.so}
for f in "$ROOT/hipsp/libspmla.so" "$GM/exl3_gm_ext.so" "$CK"; do [ -s "$f" ] || die "missing after build: $f"; done
echo "install_strix: OK"
