#!/usr/bin/env bash
# One arm of the 131K decode study: start a GLM server (-c 272384, own HOME) with the per-round log on, replay the
# points, stop the server. The whole script is one GPU window. Environment:
#   KYOJIN_GLM_PACK    the GLM-5.3-Flash EXL3 pack directory (required)
#   ROUND_DOC          text file the prompts are cut from (required)
#   ROUND_CHARS        served curve json with the 131072 / 262144 points (required)
#   tools/glm/round_replay.sh ARM [round_replay.py args...]
# ARM names the single variable: base | dyn0 (EXL3_GLM_DYN=0) | nd1 | nd3 (--num-draft) | nongram (GLM_NGRAM=off)
set -u
ARM=${1:?arm}; shift
KY=$(cd "$(dirname "$0")/../.." && pwd)
PACK=${KYOJIN_GLM_PACK:?set KYOJIN_GLM_PACK to the GLM pack directory}
DOC=${ROUND_DOC:?set ROUND_DOC}; CHARS=${ROUND_CHARS:?set ROUND_CHARS}
export HOME=$KY/home; mkdir -p $HOME $KY/scratch
cd $KY; source .venv/bin/activate; export EXL3_ROCM_SDK=$(rocm-sdk path --root); source tools/strix_halo/env.sh
ND=2
case $ARM in
  base) ;;
  dyn0) export EXL3_GLM_DYN=0 ;;
  nd1) ND=1; export EXL3_GLM_DYN=0 ;;   # a depth is only a depth with the dynamic stop off; compare with nd2dyn0 (=dyn0)
  nd3) ND=3; export EXL3_GLM_DYN=0 ;;
  nongram) export GLM_NGRAM=off ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
TAG=round_$ARM; RL=$KY/scratch/$TAG.rounds.jsonl; : > $RL
export EXL3_ROUND_LOG=$RL KYOJIN_REQLOG_DIR=$KY/scratch/$TAG.reqlog; rm -rf $KYOJIN_REQLOG_DIR
LOG=$KY/scratch/$TAG.serve.log
python -u tools/glm/serve.py --model $PACK --port 8013 -c 272384 --num-draft $ND > $LOG 2>&1 & SRV=$!
for i in $(seq 1 240); do curl -s -m 10 localhost:8013/health | grep -qE '"status": *"ok"' && break; kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 15; done
grep -E "two-stage draft head" $LOG
python3 tools/glm/round_replay.py --base http://localhost:8013/v1 --roundlog $RL --reqlog-dir $KYOJIN_REQLOG_DIR \
   --doc "$DOC" --chars-from "$CHARS" --out $KY/scratch/$TAG.json "$@" 2>&1 | tee $KY/scratch/$TAG.out
kill $SRV; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 5; done; kill -9 $SRV 2>/dev/null
echo ARMDONE >> $KY/scratch/$TAG.out
