#!/usr/bin/env bash
# llama-swap lane launcher for the DS4.1-Flash-1017 pack (PREPARED, not wired into any live config; see ds41_golive_report.md).
#   ds41_lane.sh PORT [extra serve.py flags]
# llama-swap starts it as `cmd:` and proxies http://localhost:PORT. The pack is mostly on SSD (expert tier), so the memory
# guard only asks for the resident part (~30 GiB: non-expert weights ~16, KV + runtime ~8, tier floor 8) instead of the
# whole pack. The uncensor spec sits in the pack folder (uncensor_spec.json): ON by default, DS41_UNCENSOR=0 turns it off.
set -euo pipefail
PORT=${1:?port}; shift
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
H=${H:-$HOME/ds41_work}
NEED=${DS41_LANE_NEED_GIB:-40}
pf=$(journalctl -k -b 0 --no-pager 2>/dev/null | grep -c 'amdgpu.*page fault' || true)
[ "$pf" -eq 0 ] || { echo "ds41_lane: REFUSED, $pf amdgpu page fault(s) this boot" >&2; exit 97; }
for _ in $(seq 90); do av=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$av" -ge "$NEED" ] && break; sleep 1; done
[ "$av" -ge "$NEED" ] || { echo "ds41_lane: REFUSED, MemAvailable ${av} GiB < $NEED" >&2; exit 98; }
[ "${DS41_UNCENSOR:-1}" = 1 ] || export EXL3_ABLIT_RUNTIME=off
echo "$(date '+%d/%m %H:%M:%S') ds41-lane port=$PORT avail=${av}G uncensor=${DS41_UNCENSOR:-1}" >> "$H/.gpu-guard.log"
export DS41_PORT=$PORT DS41_PACK=${DS41_PACK:-$HOME/models/ds41-1017-yamz-pack} DS41_VALIDATE=0
exec "$ROOT/tools/ds41/ds41_serve.sh" --model-id DeepSeek-V4.1-Flash-Yamz "$@"
