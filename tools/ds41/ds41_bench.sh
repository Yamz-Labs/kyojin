#!/usr/bin/env bash
# ds41_bench.sh: one command bench of the converted DS4.1 pack. One gpurun ticket per stage (<= 600 s each), results in
# $DS41_OUT (JSON) and a summary at the end. Stages: prefill (2048 / 16384 tok/s), decode (plain), dspark (tok/s, accepted
# per round, speedup), quality (top-1 + KL vs the teacher files), longctx (needle in a 16K haystack).
#   tools/ds41/ds41_bench.sh [stage ...]            default: all, in that order
#   DS41_TEACHER=<dir with teacher/row_XXX.npz>   default ~/ds41_work/ds41ready_data/teacher
#       (worker: ~/ds41_work/recipe/teacher/out/teacher/ via: rsync -a worker:ds41_work/recipe/teacher/out/teacher $DS41_TEACHER/)
#   DS41_OUT=<json>  DS41_DRAFT=4  DS41_NEW=64  DS41_QROWS=<rows per quality ticket, 0 = budget only>  DS41_QMAX=<tickets, 12>
#   DS41_LAYERS=N first N trunk layers; DS41_EXTRA="--lens 512,1024 --ctx 4096 --trunc 512" extra ds41_bench.py flags (dry runs);
#   DS41_TICKET=0 no gpurun per stage
set -u
source "$(dirname "${BASH_SOURCE[0]}")/ds41_env.sh" || exit $?
DS41_OUT=${DS41_OUT:-$H/ds41ready_scratch/bench_$(date +%m%d_%H%M).json}
DS41_TEACHER=${DS41_TEACHER:-$H/ds41ready_data/teacher}
STAGES=${*:-prefill decode dspark quality longctx}
cd "$ROOT" || exit 2
DS41_DATA=${DS41_DATA:-}
common=(--out "$DS41_OUT" --chunk "$DS41_CHUNK" --teacher "$DS41_TEACHER" --draft "${DS41_DRAFT:-4}" --new "${DS41_NEW:-64}")
[ -n "$DS41_DATA" ] && common+=(--data "$DS41_DATA")
[ -n "${DS41_LAYERS:-}" ] && common+=(--layers "$DS41_LAYERS")
[ -n "${DS41_ENGRAM_DIR:-}" ] && common+=(--engram-dir "$DS41_ENGRAM_DIR")
[ -n "${DS41_EXTRA:-}" ] && common+=($DS41_EXTRA)
# DS41_TICKET=0: no gpurun ticket per stage (the caller already holds one window, e.g. a dry run of every stage in one ticket)
tick() {
  if [ "${DS41_TICKET:-1}" = 0 ]; then python tools/ds41/ds41_bench.py "$DS41_PACK" "$@" "${common[@]}"
  else "$H/bin/gpurun" --need "${DS41_NEED_GB:-20}" --max 600 --who "$(basename "$ROOT")" python tools/ds41/ds41_bench.py "$DS41_PACK" "$@" "${common[@]}"; fi
}
# a failed stage is never silent: quality / prefill / decode / maketeacher failures (gate inputs) make the script exit 1;
# a failed dspark or longctx stage (gate WARN rows, "missing" in the table) is listed loudly at the end and exits 0
failed=""; soft=""
for s in $STAGES; do
  echo "== stage $s" >&2
  if [ "$s" = quality ]; then
    [ -d "$DS41_TEACHER/teacher" ] || { echo "ds41: no teacher files in $DS41_TEACHER/teacher: quality stage FAILED" >&2; failed="$failed quality"; continue; }
    rc=75
    for i in $(seq 1 "${DS41_QMAX:-12}"); do
      tick --stage quality --budget 450 ${DS41_QROWS:+--rows $DS41_QROWS}; rc=$?
      [ $rc = 75 ] || break
    done
    [ $rc = 0 ] || { echo "ds41: stage quality FAILED (rc $rc$([ $rc = 75 ] && echo ", not finished after ${DS41_QMAX:-12} tickets: raise DS41_QMAX and rerun"))" >&2; failed="$failed quality"; }
  else
    tick --stage "$s"; rc=$?
    if [ $rc != 0 ]; then
      echo "ds41: stage $s FAILED (rc $rc)" >&2
      case $s in dspark|longctx) soft="$soft $s" ;; *) failed="$failed $s" ;; esac
    fi
  fi
done
echo "== summary ($DS41_OUT)" >&2
[ -f "$DS41_OUT" ] && python - "$DS41_OUT" <<'P'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("prefill", "decode", "dspark", "quality_summary", "longctx"):
    if k in d: print(k, json.dumps(d[k]))
P
[ -z "$soft" ] || echo "ds41: WARNING, stages FAILED (not gate-hard, numbers missing):$soft" >&2
[ -z "$failed" ] || { echo "ds41: FAILED STAGES:$failed" >&2; exit 1; }
