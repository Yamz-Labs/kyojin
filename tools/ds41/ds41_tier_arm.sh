#!/usr/bin/env bash
# usage: [DS41_BUDGET_GIB=.. (or DS41_BUDGET_MINUS=N: MemAvailable-N) DS41_WARM=.. DS41_CHUNK=.. EXL3_EXPERT_READERS=.. EXL3_EXPERT_RING=..] ds41_tier_arm.sh ARM [lens]
# Re-executes itself under gpurun so MemAvailable (and the tier budget) is read after the GPU ticket is granted.
set -u
H=${H:-$HOME/ds41_work}
if [ -z "${DS41_INTICKET:-}" ]; then
  DS41_INTICKET=1 exec "$H/bin/gpurun" --need "${DS41_NEED_GB:-20}" --who ${DS41_WHO:-ds41pfov} bash "${BASH_SOURCE[0]}" "$@"
fi
ARM=$1; LENS=${2:-2048,8192,16384}
export DS41_VALIDATE=0 DS41_VALIDATED=1 DS41_CHUNK=${DS41_CHUNK:-32768}   # serving prefills tier prompts in 32768-row chunks (EXL3_TIER_PF_CHUNK)
if [ -n "${DS41_BUDGET_MINUS:-}" ]; then DS41_BUDGET_GIB=$(( $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) - DS41_BUDGET_MINUS )); export DS41_BUDGET_GIB; fi
source "$(dirname "${BASH_SOURCE[0]}")/ds41_env.sh" || exit $?
export EXL3_DSV41_ENGRAM_DIR=$DS41_ENGRAM_DIR
export EXL3_TIER_PF_CHUNK=$DS41_CHUNK
cd "$ROOT" || exit 2
OUT=${DS41_OUT:-$H/ds41perf_out/arms.jsonl}
${DS41_WRAP:-} python tools/ds41/ds41_tier_ab.py "$DS41_PACK" --arm "$ARM" --out "$OUT" \
  --chunk "$DS41_CHUNK" --lens "$LENS" --new "${DS41_NEW:-512}" --profile "${DS41_PROFILE:-0}" --engram-dir "$DS41_ENGRAM_DIR" 2>&1 | grep -v Warning
