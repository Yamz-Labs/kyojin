#!/usr/bin/env bash
# check_all.sh: one command to validate a DS4.1 tree. ~6 min CPU stage (parallel), + GPU smoke when BOX=1.
#   BOX=1   also run tools/ds41/smoke_pack.py on the synthetic pack through ~/ds41_work/bin/gpurun (one window)
#   FAST=1  skip the files that take > 60 s (dspark_sampling, ds41_model, e2e, serving, spec, stream, dspark_official)
#   JOBS=8  parallel pytest files      PACK=...  synthetic pack dir      OUT=dir  logs (default $ROOT/.check_logs)
# Traps handled here:
#  - extension .so missing -> exllamav3.ext would JIT-compile (torch extension lock): copy the built .so from EXT_SRC.
#  - stale torch-extension lock files with no live compiler are removed.
#  - CPU stage runs with HIP_VISIBLE_DEVICES= (no GPU use outside gpurun) and OMP_NUM_THREADS=4.
#  - gpurun's quiet.sh freezes (SIGSTOP) agent CPU jobs during any GPU window: wall times of the CPU stage inflate then.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
H=${H:-$HOME/ds41_work}
EXT_SRC=${EXT_SRC:-$H/ds41box/mk}
PACK=${PACK:-$H/ds41box/pack3}
JOBS=${JOBS:-8}
OUT=${OUT:-$ROOT/.check_logs}
cd "$ROOT" || exit 2
mkdir -p "$OUT"; rm -f "$OUT"/*.log "$OUT"/*.json
# shellcheck disable=SC1091
source "$H/ds41box/pyenv.sh" "$ROOT"
SO=$(ls "$ROOT"/exllamav3_ext*.so 2>/dev/null | head -1)
if [ -z "$SO" ]; then
  src=$(ls "$EXT_SRC"/exllamav3_ext*.so 2>/dev/null | head -1)
  [ -n "$src" ] || { echo "check_all: no built exllamav3_ext*.so in $ROOT or $EXT_SRC (build under flock $H/box.build.lock)"; exit 2; }
  cp "$src" "$ROOT"/ && echo "check_all: copied $(basename "$src") from $EXT_SRC (Python-only checks; rebuild if C++/HIP changed)"
fi
if ! pgrep -f "ninja|hipcc|cc1plus" >/dev/null; then
  find "$HOME/.cache/torch_extensions" -name lock -delete 2>/dev/null
fi
T0=$(date +%s)
SLOW="tools/ds41/test_ds41_dspark_sampling.py tools/ds41/test_ds41_model.py tests/dsv41/test_dsv41_e2e.py tests/dsv41/test_dsv41_serving.py tests/dsv41/test_dsv41_spec.py tests/dsv41/test_dsv41_stream.py tools/ds41/test_ds41_dspark_official.py"
FASTF="tests/dsv41/test_dsv41_calib.py tests/dsv41/test_dsv41_engram_io.py tests/dsv41/test_dsv41_framework.py tests/dsv41/test_dsv41_loaders.py tests/dsv41/test_dsv41_num.py tests/dsv41/test_dsv41_ops.py tests/dsv41/test_dsv41_real_headers.py tests/dsv41/test_dsv41_real_slices.py tools/ds41/test_ds41_chat.py tools/ds41/test_ds41_dspark.py tools/ds41/test_ds41_parse.py tools/ds41/test_ds41_serve.py tools/ds41/test_ds41_serve_dspark.py tools/ds41/test_ds41_serve_dspark_auto.py"
# Known failures on the base (default loader is now the real model, test expects EngineUnavailable): deselected, listed in the report.
DESELECT="--deselect tools/ds41/test_ds41_serve.py::test_load_model_stub_and_missing_loader"
FILES="$SLOW $FASTF"; [ "${FAST:-0}" = 1 ] && FILES="$FASTF"
# only files that exist in this tree (other branches add/remove tests)
FILES=$(for f in $FILES; do [ -f "$f" ] && echo "$f"; done)
# new test files from merged branches that are not in the lists above
EXTRA=$(ls tests/dsv41/test_*.py tools/ds41/test_*.py 2>/dev/null | grep -v -E "dsa_gpu|test_dsv41_hip(k)?\.py|spec_real" | while read -r f; do case " $SLOW $FASTF " in *" $f "*) ;; *) echo "$f";; esac; done)
FILES="$FILES $EXTRA"
export HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4
echo "== CPU stage: $(echo $FILES | wc -w) files, $JOBS parallel"
echo $FILES | tr ' ' '\n' | xargs -P "$JOBS" -I{} bash -c 'f={}; s=$(date +%s); timeout -k 5 '"${TMO:-800}"' nice python -m pytest $f -q -p no:cacheprovider '"$DESELECT"' > "'"$OUT"'/$(basename $f).log" 2>&1; echo "$? $(( $(date +%s)-s )) $f" >> "'"$OUT"'/rc.log"'
FAIL=0
printf '%-48s %-6s %5s  %s\n' file rc sec summary
sort -k3 "$OUT/rc.log" | while read -r rc sec f; do
  printf '%-48s %-6s %5s  %s\n' "$f" "$rc" "$sec" "$(tail -1 "$OUT/$(basename $f).log" | tr -d '=' | cut -c1-60)"
done
# rc 0 = pass, 5 = nothing collected (all skipped) is fine
BAD=$(awk '$1!=0 && $1!=5' "$OUT/rc.log" | wc -l)
[ "$BAD" -gt 0 ] && FAIL=1
echo "CPU stage: $([ $BAD -eq 0 ] && echo PASS || echo "FAIL ($BAD files, logs in $OUT)") in $(( $(date +%s)-T0 )) s; deselected known failure: test_load_model_stub_and_missing_loader"
# spec == greedy lives in tests/dsv41/test_dsv41_spec.py and tools/ds41/test_ds41_dspark*.py (all in the lists above)
if [ "${BOX:-0}" = 1 ]; then
  echo "== GPU stage (gpurun, one window)"
  unset HIP_VISIBLE_DEVICES; export HIP_VISIBLE_DEVICES=0
  "$H/bin/gpurun" --need "${NEED_GB:-20}" --max 900 --who "$(basename "$ROOT")" \
    python tools/ds41/smoke_pack.py "$PACK" --layers 1,2,3 --prefill 2048 --reps 5 --json "$OUT/smoke.json" 2>&1 | tee "$OUT/smoke.log" | grep -E "^(EXACT|SPEED|\{)|rror|gpurun" 
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU stage: FAIL (see $OUT/smoke.log)"; FAIL=1; }
fi
echo "check_all: $([ $FAIL -eq 0 ] && echo PASS || echo FAIL) total $(( $(date +%s)-T0 )) s"
exit $FAIL
