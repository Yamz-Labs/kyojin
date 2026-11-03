#!/usr/bin/env bash
# Build the optional gfx1151 MiMo d192 paged-KV backend, never in the donor tree.
# No implicit fetch, environment sourcing, device probe, native load or install.
# EXL3_CK_SOURCE is a read-only local Git donor; dirty worktree bytes are ignored.
# EXL3_CK_CACHE must be a new or matching builder-owned directory, disjoint from it.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec /usr/bin/python3 -I -S "$SCRIPT_DIR/ck_pagedkv_build.py" "$@"
