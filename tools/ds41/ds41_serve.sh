#!/usr/bin/env bash
# ds41_serve.sh: OpenAI server for the converted DS4.1 pack. SSD expert tier + Engram prefetch + DSpark, one command.
#   tools/ds41/ds41_serve.sh [extra serve.py flags]          (env knobs: see ds41_env.sh)
#   DS41_PORT=8000 DS41_CTX=131072 DS41_DRAFT=4 DS41_DRAFT_AUTO=1   (DS41_DRAFT=0: plain decode)
#   DS41_GPURUN=1   wrap in the box gpu ticket (max 900 s: dry runs only; a real serve is started by the orchestrator, no ticket)
# Engram prefetch is always on (PrefetchRowStore: the rows of a pass are read while the layers before the Engram layer compute).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/ds41_env.sh" || exit $?
PORT=${DS41_PORT:-8000}; CTX=${DS41_CTX:-131072}; DRAFT=${DS41_DRAFT:-4}
args=(--model "$DS41_PACK" --loader ds41_serving:load_model --port "$PORT" -c "$CTX" --engine-opt "prefill_chunk=$DS41_CHUNK")
[ -n "${DS41_ENGRAM_DIR:-}" ] && args+=(--engine-opt "engram_dir=$DS41_ENGRAM_DIR")
if [ "$DRAFT" -gt 0 ]; then
  args+=(--num-draft "$DRAFT"); [ "${DS41_DRAFT_AUTO:-1}" = 1 ] && args+=(--draft-auto)
fi
cd "$ROOT" || exit 2
run=(python tools/ds41/serve.py "${args[@]}" "$@")
[ "${DS41_GPURUN:-0}" = 1 ] && run=("$H/bin/gpurun" --need 20 --max 900 --who "$(basename "$ROOT")" "${run[@]}")
echo "ds41: ${run[*]}" >&2
exec "${run[@]}"
