#!/usr/bin/env bash
# Worktree-local rebuild after HIP source/header edits. Build and bench in separate shells.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export EXL3_ROOT="${EXL3_ROOT:-$ROOT}"
source "$SCRIPT_DIR/env.sh" build > /dev/null
BUILD_PYTHON="${EXL3_BUILD_PYTHON:-$EXL3_VENV/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export EXL3_HIP_DEFINES="${EXL3_HIP_DEFINES:-EXL3_HIP_STG_PAD}"
cd "$ROOT"

# Remove generated hipify twins and their cached objects. These are build products, not sources.
python3 - "$ROOT" "${FULL:-0}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
ext = root / "exllamav3/exllamav3_ext"
full = sys.argv[2] == "1"
for rel in ("hgemm", "hc_mix", "fused_elt_rocm", "gdn", "quant/exl3_gemv", "quant/exl3_moe_prefill"):
    for suffix in (".hip", "_hip.cuh"):
        p = ext / f"{rel}{suffix}"
        p.unlink(missing_ok=True)
    build = root / "build"
    if build.exists():
        for p in build.rglob(f"{Path(rel).name}*.o"):
            p.unlink()
(ext / "bindings_hip.cpp").unlink(missing_ok=True)
build = root / "build"
if build.exists():
    for p in build.rglob("bindings*.o"):
        p.unlink()
if full:
    for pattern in ("*.hip", "*_hip.cuh", "*_hip.*"):
        for p in ext.rglob(pattern):
            if pattern == "*_hip.*" and "hip" in p.relative_to(ext).parts:
                continue
            p.unlink()
    if build.exists():
        import shutil
        shutil.rmtree(build)
PY

TMP_DIR="$(mktemp -d "$ROOT/.rebuild.XXXXXX")"
cleanup() {
    rc=$?
    if [ "$rc" -eq 0 ]; then
        rm -rf "$TMP_DIR"
    else
        printf '[build] failed; log preserved at %s/build.log\n' "$TMP_DIR" >&2
    fi
}
trap cleanup EXIT
printf '[build] start defines=%s\n' "$EXL3_HIP_DEFINES"
"$BUILD_PYTHON" -m pip install --no-build-isolation --no-deps --target "$TMP_DIR/site" "$ROOT" > "$TMP_DIR/build.log" 2>&1
for so in "$TMP_DIR"/site/exllamav3_ext*.so; do cp "$so" "$ROOT/.$(basename "$so").new" && mv -f "$ROOT/.$(basename "$so").new" "$ROOT/$(basename "$so")"; done  # atomic: running processes keep the old inode
printf '[build] extension refreshed in %s\n' "$ROOT"
