# Bench: one ndt/flag config per process, all under one GPU-lock acquisition.
# usage: bench27.sh TAG:UNION:ATTN_LOOP:NDT[:extra-args] ...
source $HOME/kyojin/tools/strix_halo/env.sh
cd $HOME/kyojin
export PYTHONPATH=$PWD EXL3_REPO=$PWD
for spec in "$@"; do
  IFS=: read -r tag u al ndt extra <<< "$spec"
  echo "=== $tag UNION=$u ATTN_LOOP=$al ndt=$ndt $extra $(date +%T)"
  EXL3_DEC_MOE_UNION=$u EXL3_VERIFY_ATTN_LOOP=$al python scripts/dflash-bench.py \
    -m ~/models/mimo26-exl3 -dm ${DM:-~/models/mimo26-exl3/dflash} -ndt $ndt -cs 2304 $extra \
    --reps ${REPS:-3} --kinds ${KINDS:-code,chat,prose} --out scratch/b27-$tag.json 2>&1 \
    | grep -vE 'UserWarning|torch.empty|dflash_debug'
done
