#!/usr/bin/env bash
# usage: same env knobs as ds41_tier_arm.sh; ds41_session_arm.sh ARM [extra ds41_session.py flags]
set -u
H=${H:-$HOME/ds41_work}
if [ -z "${DS41_INTICKET:-}" ]; then DS41_INTICKET=1 exec "$H/bin/gpurun" --need "${DS41_NEED_GB:-20}" --grace 600 --who "${DS41_WHO:-ds41sess}" bash "${BASH_SOURCE[0]}" "$@"; fi
ARM=$1; shift
export DS41_VALIDATE=0 DS41_VALIDATED=1
if [ -n "${DS41_BUDGET_MINUS:-}" ]; then DS41_BUDGET_GIB=$(( $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) - DS41_BUDGET_MINUS )); export DS41_BUDGET_GIB; fi
source "$(dirname "${BASH_SOURCE[0]}")/ds41_env.sh" || exit $?
export EXL3_DSV41_ENGRAM_DIR=$DS41_ENGRAM_DIR EXL3_TIER_PF_CHUNK=$DS41_CHUNK
cd "$ROOT" || exit 2
python tools/ds41/${DS41_SESSION_PY:-ds41_session.py} "$DS41_PACK" --arm "$ARM" --out "${DS41_OUT:-$H/ds41perf_out/sessions.jsonl}" --chunk "$DS41_CHUNK" --engram-dir "$DS41_ENGRAM_DIR" "$@" 2>&1 | grep -v Warning
