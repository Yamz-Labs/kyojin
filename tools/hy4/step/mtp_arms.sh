#!/bin/bash
# arms.sh "ARMNAME:K=V,K=V" ...: one mtp_items round timing per arm (same process env otherwise), step-1 output checked against arm A
cd "${TREE:-$(git rev-parse --show-toplevel)}"   # env: GGUF (model path) + the HY4_DRAFT_* arms
REF=/tmp/mtp_ref_$$.pt
for arm in "$@"; do
  n=${arm%%:*}; kv=${arm#*:}; envs=""; [ -n "$kv" ] && envs=$(echo $kv | tr ',' ' ')
  echo "=== ARM $n ($kv)"
  if [ "$n" = "A" ]; then env $envs python -u tools/hy4/step/mtp_items.py --save $REF 2>&1 | grep -E "MTP_ROUND|Error|error" ; else env $envs python -u tools/hy4/step/mtp_items.py --ref $REF 2>&1 | grep -E "MTP_ROUND|Error|error"; fi
done
rm -f $REF
