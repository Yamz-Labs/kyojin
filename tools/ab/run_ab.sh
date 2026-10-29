#!/bin/bash
# run_ab.sh: ONE server life, interleaved in-process A/B (ab_inproc.py). All settings come from the environment:
#   AB_LOCK    3090 | igpu | none           (3090: rtx1/gpu3090.lock + wait for an empty card; igpu: strix-bench/gpu.lock then strixfix/kigpu.lock)
#   AB_TREE    checkout that has tools/ab and /admin/knobs (tools/harn commits)      AB_PY  python of that tree's venv
#   AB_SERVER  server command after the python, e.g. "tools/qwen/serve.py --model M --port 8671 ..." (--admin-knobs is appended)
#   AB_PORT    port of that server              AB_OUT  prefix for AB_OUT.srv.log / .ab.txt / .json
#   AB_ARGS    ab_inproc.py arguments after PORT, e.g. "--arm A:EXL3_GEMV_TRI=0 --arm B:EXL3_GEMV_TRI=1 --order ABBAAB --synth 4096"
# Extra environment (EXL3_*, PYTHONPATH...) passes through to the server. Kills only the server it started, by PID.
: "${AB_LOCK:=none}" "${AB_TREE:?}" "${AB_PY:?}" "${AB_SERVER:?}" "${AB_PORT:?}" "${AB_OUT:?}" "${AB_ARGS:?}"
case $AB_LOCK in
  3090) exec 9>$HOME/hermes-work/rtx1/gpu3090.lock; flock -w 28800 9 || { echo "lockfail"; exit 1; }
        for w in $(seq 120); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -le 2000 ] && break; sleep 3; done ;;
  igpu) exec 8>$HOME/strix-bench/gpu.lock; flock -w 28800 8 || exit 1; exec 7>$HOME/hermes-work/strixfix/kigpu.lock; flock -w 28800 7 || exit 1 ;;
esac
cd "$AB_TREE" || exit 1
export PYTHONPATH=$AB_TREE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
$AB_PY -u $AB_SERVER --admin-knobs > $AB_OUT.srv.log 2>&1 9>&- 8>&- 7>&- &
P=$!
trap 'kill $P 2>/dev/null' EXIT
for i in $(seq 900); do
  [ "$(curl -s -m3 -o /dev/null -w '%{http_code}' localhost:$AB_PORT/health)" = 200 ] && break
  kill -0 $P 2>/dev/null || { echo "server died, see $AB_OUT.srv.log"; exit 2; }
  sleep 3
done
if ! curl -sf -m5 localhost:$AB_PORT/admin/knobs > /dev/null; then echo "no /admin/knobs: $AB_TREE lacks the tools/harn commits (git cherry-pick fb0f3b216..tools/harn)"; exit 3; fi
python3 -I "$(dirname "$(readlink -f "$0")")/ab_inproc.py" $AB_PORT $AB_ARGS --json $AB_OUT.json 2>&1 | tee $AB_OUT.ab.txt
rc=${PIPESTATUS[0]}
kill $P; wait $P 2>/dev/null
exit $rc
