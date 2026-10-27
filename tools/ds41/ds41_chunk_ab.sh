#!/usr/bin/env bash
# usage: [DS41_PACK=.. DS41_ENGRAM_DIR=.. DS41_BUDGET_GIB=70 LENS=32768,65536 CHUNKS=16384,32768,65536 BASE_LENS=32768 BASE_CHUNKS=16384,32768] ds41_chunk_ab.sh
# Run it under gpurun. Arm 1: row blocking on (this branch). Arm 2: blocking off (= the 16K-era buffers) for the baseline peak/tokens.
set -u
export DS41_VALIDATE=0 DS41_VALIDATED=1 DS41_BUDGET_GIB=${DS41_BUDGET_GIB:-70}
source "$(dirname "${BASH_SOURCE[0]}")/ds41_env.sh" || exit $?
export EXL3_DSV41_ENGRAM_DIR=$DS41_ENGRAM_DIR
cd "$ROOT" || exit 2
OUT=${OUT:-$H/ds41chunk_out}; mkdir -p "$OUT"
python tools/ds41/ds41_chunk_ab.py "$DS41_PACK" --lens "${LENS:-32768,65536}" --chunks "${CHUNKS:-16384,32768,65536}" --blocking 1 --engram-dir "$DS41_ENGRAM_DIR" --out "$OUT" 2>&1 | grep -v Warning
python tools/ds41/ds41_chunk_ab.py "$DS41_PACK" --lens "${BASE_LENS:-32768}" --chunks "${BASE_CHUNKS:-16384,32768}" --blocking 0 --engram-dir "$DS41_ENGRAM_DIR" --out "$OUT" 2>&1 | grep -v Warning
