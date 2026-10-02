#!/usr/bin/env bash
# Cost of verify WIDTH, measured through the production path.
# At -ndt k the verify forward carries k+1 rows. If ms/forward is flat in k, extra rows are
# ~free and a TREE draft (which buys accepted-tokens-per-forward instead of relying on a
# single linear chain's acceptance) can convert width into tok/s. If ms/forward climbs
# linearly, rows cost full price and width is a dead end.
#
# ms/forward is derived: steps = tokens - accepted ; ms/fwd = wall/steps
# Log: /tmp/width_ndt.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
echo "=== no MTP (1-row forward baseline) ==="
.venv/bin/python tools/strix_halo/bench_mtp.py -n 256 --no-mtp -g 2>&1 | grep -E "decode:|draft:"
sleep 6
for k in 1 2 3 4 5; do
  echo "=== ndt=$k (verify carries $((k+1)) rows), static draft so width is fixed ==="
  .venv/bin/python tools/strix_halo/bench_mtp.py -n 512 -ndt $k -g 2>&1 | grep -E "decode:|draft:"
  sleep 6
done
echo DONEWIDTH
