# Shared setup of ds41_serve.sh and ds41_bench.sh (sourced). Sets the engine env, the SSD expert tier and the Engram table
# directory for the converted DS4.1 pack. Every knob is an environment variable; the defaults are the morning settings.
#
#   DS41_PACK          pack folder                               ~/models/ds41-1017-yamz
#   DS41_ENGRAM_DIR    official FP8 Engram shards (47, 48)       ~/models/ds41-1017/engram_fp8  ("" = tables inside the pack)
#   DS41_TIER          auto | 1 | 0   SSD expert tier            auto: on when the pack's shards exceed the RAM budget
#   DS41_BUDGET_GIB    slot-pool size of the tier in GiB         MemAvailable - 30 (resident non-expert weights ~16, KV +
#                                                                runtime ~8, pinned staging ring ~2, margin ~4)
#   DS41_CHUNK         prefill chunk in tokens                   4096 (a chunk touches ~every expert: bigger chunk = fewer
#                                                                full reads per prompt token; resident pack: 2048)
#   DS41_WARM          routing-count json that seeds the hot set (EXL3_EXPERT_STATS_OUT of an earlier run, or plan.py freq)
#   DS41_VALIDATE      1 = run validate_pack.py first (default 1)
# Tier knobs passed through (defaults here): EXL3_EXPERT_HALFLIFE=16 (decayed-use half life in forwards), EXL3_EXPERT_READERS=8,
# EXL3_EXPERT_RING=48 (pinned staging entries), EXL3_EXPERT_AHEAD=0 (router-ahead prefetch needs the generic router: off),
# EXL3_EXPERT_STATS_OUT=<json> (write routing counts at exit: feed it back as DS41_WARM), EXL3_EXPERT_PIN_FRAC=0.
# Not tiered: the 3 DSpark stages and any layer with pruned experts stay resident. DSV41_DECODE_GRAPH is forced to 0 with the tier.
# Drafter knobs (off by default, engine-side, exact): DS41_TUNER=timed (draft length from per-position acceptance and measured round time,
# plain step allowed), DS41_DRAFT_WANT=1 (drafter chains only the passes a round needs), DS41_NGRAM=strong|extend (prompt lookup merge;
# DS41_NGRAM_MIN match length, default 6, DS41_NGRAM_N key length, default 3). Measured: briefs/ds41_draft.md.
# DS41_DRAFT_SWITCH=1: request-level choice (tool block in prompt -> auto cap 4; else timed tuner + want-cap). Off by default.
# Serve default decided 20/10 (briefs/ds41_draftdecide.md, warm runner, 2 ABBA rounds, noisy box): keep draft 4 + auto (best on tool/after-tool, +10 %);
# no opt-in arm is within 5 % of plain on every workload, so none is made default. Tool context: auto cap 4. No-tool code/follow-up: DS41_DRAFT_WANT=1 was best (+0..16 %).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
H=${H:-$HOME/ds41_work}
DS41_PACK=${DS41_PACK:-$HOME/models/ds41-1017-yamz}
DS41_ENGRAM_DIR=${DS41_ENGRAM_DIR-$HOME/models/ds41-1017/engram_fp8}
DS41_TIER=${DS41_TIER:-auto}
DS41_VALIDATE=${DS41_VALIDATE:-1}
# Python env: DS41_PYENV=<script> (sourced with the repo root) if set, else the repo's standard Strix Halo env (tools/strix_halo/env.sh,
# venv at $EXL3_VENV or <repo>/.venv; needs ml_dtypes installed in it).
# shellcheck disable=SC1090,SC1091
if [ -n "${DS41_PYENV:-}" ]; then
  source "$DS41_PYENV" "$ROOT" || { echo "ds41: python env $DS41_PYENV failed" >&2; return 5 2>/dev/null || exit 5; }
else
  source "$ROOT/tools/strix_halo/env.sh" >/dev/null || { echo "ds41: tools/strix_halo/env.sh failed" >&2; return 5 2>/dev/null || exit 5; }
  export PATH="$EXL3_VENV/bin:$PATH" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  unset CUDA_VISIBLE_DEVICES
fi
export HIP_VISIBLE_DEVICES=0
[ -f "$DS41_PACK/config.json" ] || { echo "ds41: no pack at $DS41_PACK (config.json missing)" >&2; return 2 2>/dev/null || exit 2; }
if ! ls "$ROOT"/exllamav3_ext*.so >/dev/null 2>&1; then
  src=$(ls "$H"/ds41integ2/exllamav3_ext*.so 2>/dev/null | head -1)
  [ -n "$src" ] && cp "$src" "$ROOT"/ && echo "ds41: copied $(basename "$src") from ds41integ2" >&2
fi
mem_gib() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
free_disk_gib() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
pack_gib=$(du -cbL "$DS41_PACK"/*.safetensors 2>/dev/null | tail -1 | awk '{printf "%d", $1/1073741824}')
avail=$(mem_gib)
budget_set=${DS41_BUDGET_GIB:+1}
# DS41_BUDGET_GIB=auto: the pool takes what is free AFTER the non-expert weights are loaded, minus a reserve for what grows later:
# prefill workspace (chunk/4096 + 6 GiB: peak torch use above the weights, measured 3-7 GiB at chunk 4096 on 8K-80K prompts, 8 GiB at chunk 32768), prefix snapshots (DS41_PFX_GIB, 4), runtime = what the serve allocates AFTER the pool is sized: KV, Engram and drafter buffers (3.1 GiB of GTT) + python heap growth (1.4 GiB), measured at ctx 131072 draft 5 (DS41_RUNTIME_GIB, 4.5), margin (DS41_AUTO_MARGIN_GIB, 2)
# Default since 20/10 (briefs/ds41_pool.md): auto, 86 GiB pool vs 78 on a quiet box, -13.5 % GB read per session, tokhash identical.
if [ "${DS41_BUDGET_GIB:-auto}" = auto ]; then
  DS41_AUTO=1; DS41_BUDGET_GIB=$(( avail - 30 ))
fi
DS41_BUDGET_GIB=${DS41_BUDGET_GIB:-$(( avail - 30 ))}
if [ "$DS41_TIER" = auto ]; then
  if [ "$pack_gib" -gt $(( avail - 10 )) ]; then DS41_TIER=1; else DS41_TIER=0; fi
fi
if [ "$DS41_TIER" = 1 ]; then
  [ -n "$budget_set" ] || [ "$DS41_BUDGET_GIB" -ge 8 ] || { echo "ds41: MemAvailable ${avail} GiB leaves a tier budget of ${DS41_BUDGET_GIB} GiB: free memory first" >&2; return 3 2>/dev/null || exit 3; }
  export EXL3_EXPERT_BUDGET_GIB=$DS41_BUDGET_GIB
  export EXL3_EXPERT_RING=${EXL3_EXPERT_RING:-128}   # pinned staging entries (12.7 MiB each): 128 = 1.6 GiB instead of 4 GiB at the old 320
  if [ "${DS41_AUTO:-}" = 1 ]; then
    export EXL3_EXPERT_BUDGET_GIB=auto
    export EXL3_EXPERT_RESERVE_GIB=$(LC_ALL=C awk -v c="${DS41_CHUNK:-4096}" -v p="${DS41_PFX_GIB:-4}" -v m="${DS41_AUTO_MARGIN_GIB:-2}" -v r="${DS41_RUNTIME_GIB:-4.5}" 'BEGIN{printf "%.1f", c/4096+6+p+r+m}')
  fi
  export EXL3_EXPERT_HALFLIFE=${EXL3_EXPERT_HALFLIFE:-16}
  export EXL3_EXPERT_AHEAD=${EXL3_EXPERT_AHEAD:-0}
  # scan-resistant admission + prompt-tail promotion (cache policy only, greedy tokens identical): see briefs/ds41_hot_protect.md
  export EXL3_EXPERT_PROTECT=${EXL3_EXPERT_PROTECT:-1} EXL3_EXPERT_HOT_HALFLIFE=${EXL3_EXPERT_HOT_HALFLIFE:-8} EXL3_EXPERT_TAIL_ROWS=${EXL3_EXPERT_TAIL_ROWS:-16}
  # 20/10 read path A/B (prefill 16K: 277-321 -> 410 tok/s, reads fully hidden): 12 reader threads, 4 MiB read chunks
  export EXL3_EXPERT_READERS=${EXL3_EXPERT_READERS:-12} EXL3_EXPERT_CHUNK_KIB=${EXL3_EXPERT_CHUNK_KIB:-4096}
  [ -n "${DS41_WARM:-}" ] && export EXL3_EXPERT_WARM=$DS41_WARM
  export DSV41_DECODE_GRAPH=0
  DS41_CHUNK=${DS41_CHUNK:-4096}
else
  unset EXL3_EXPERT_BUDGET_GIB
  DS41_CHUNK=${DS41_CHUNK:-2048}
fi
[ -n "$DS41_ENGRAM_DIR" ] && DS41_ENGRAM_OPT="--engine-opt engram_dir=$DS41_ENGRAM_DIR"
echo "ds41: pack $DS41_PACK (${pack_gib} GiB of shards) | MemAvailable ${avail} GiB | tier $([ "$DS41_TIER" = 1 ] && echo "ON budget ${DS41_BUDGET_GIB} GiB halflife ${EXL3_EXPERT_HALFLIFE}" || echo off) | chunk $DS41_CHUNK | engram ${DS41_ENGRAM_DIR:-in pack} | disk free $(free_disk_gib "$DS41_PACK") GiB" >&2
if [ "$DS41_VALIDATE" = 1 ] && [ -z "${DS41_VALIDATED:-}" ]; then
  vout=$(python "$ROOT/tools/ds41/validate_pack.py" "$DS41_PACK" ${DS41_ENGRAM_DIR:+--engram-dir "$DS41_ENGRAM_DIR"} 2>&1); vrc=$?
  echo "$vout" | tail -12 >&2
  [ "$vrc" -eq 0 ] || { echo "ds41: validate_pack found errors (DS41_VALIDATE=0 skips the check)" >&2; return 4 2>/dev/null || exit 4; }
fi
