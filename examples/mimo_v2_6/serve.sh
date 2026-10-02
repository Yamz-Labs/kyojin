#!/usr/bin/env bash
#
# Serve a MiMo-V2.6-Flash EXL3 quant over an OpenAI-compatible HTTP API, with the
# shipped DFlash drafter for speculative decoding.
#
#   ./examples/mimo_v2_6/serve.sh /path/to/mimo-2.25bpw-hq
#   DRAFT_DIR=/path/to/mimo-dflash-draft ./examples/mimo_v2_6/serve.sh /path/to/mimo-2.25bpw-hq
#   PORT=8080 MAX_SEQ_LEN=131072 CACHE_MODE=8,8 ./examples/mimo_v2_6/serve.sh <dir>
#
# ExLlamaV3 ships no HTTP server, so this drives TabbyAPI
# (github.com/theroyallab/tabbyAPI) against whatever `exllamav3` the active Python
# environment resolves -- which should be an editable install of this checkout.
# Use the matching TabbyAPI branch:
#
#   github.com/benthecarman/tabbyAPI  branch  mimo-v2.6-flash
#
# It carries one patch: honour EXL3_LOAD_DEVICE (see doc/mimo_v2_6.md).
#
# See doc/mimo_v2_6.md for the full runbook (build, knobs, memory budget).
#
# ---------------------------------------------------------------------------
# Paths
#   $1                    EXL3 model directory (required)
#   TABBY_DIR             TabbyAPI checkout      (default: ./tabbyAPI, else $PWD/tabbyAPI)
#   DRAFT_DIR             staged DFlash drafter, from util/prepare_dflash_draft.py
#   CONFIG_OUT            where the generated YAML is written
#                         (default: <TABBY_DIR>/mimo-tabby-config.yml)
#
# Network
#   HOST=127.0.0.1        listen address; leave on loopback unless you add auth
#   PORT=8080             listen port
#   DISABLE_AUTH=1        0 requires an API key from <TABBY_DIR>/api_tokens.yml
#
# Model
#   MAX_SEQ_LEN=262144    max context. On a 128 GB unified-memory box (DGX Spark)
#                         393216 is the largest clean setting with the drafter attached.
#   CACHE_SIZE=$MAX_SEQ_LEN   paged-cache tokens (multiple of 256)
#   CACHE_MODE=FP16       or 8,8 / 6,6 / Q8 / Q6 -- quantizes the GLOBAL-attention
#                         paged cache only; the 39 SWA layers' ring is always fp16
#   MAX_BATCH_SIZE=1      concurrent slots; each costs one 175.5 MiB SWA ring
#   CHUNK_SIZE=2048       prefill chunk
#   GPU_SPLIT=            e.g. "110" to hard-set the per-device GB allowance
#   WARMUP=false          true pays kernel autotuning at load instead of first request
#   REASONING_START/_END  reasoning tags (default <think> / </think>)
#
# Speculative decoding
#   DRAFT_TOKENS=         tokens drafted per step (DFlash default 7, n-gram 4)
#   DYNAMIC_DRAFT=1       shrink the draft window from observed acceptance.
#                         ON by default: best worst case and best mean. Set 0 for
#                         peak coding throughput (1.58x vs 1.38x).
#   DRAFT_CACHE_MODE=FP16 draft KV cache mode. Only meaningful with DRAFT_RING=0;
#                         the ring is always FP16 and always 35.0 MiB/slot.
#   DRAFT_RING=1          the DFlash drafter's 5 sliding-window layers keep a fixed
#                         35.0 MiB/slot ring instead of a paged cache sized for
#                         MAX_SEQ_LEN (2.50 GiB at 128K, 7.50 GiB at 384K). 0
#                         restores the paged cache for A/B.
#   NGRAM_MIN=            n-gram match length; selects draft_mode: ngram when
#                         DRAFT_DIR is unset. Measured 0.87-0.98x on this model --
#                         not a useful default here.
#
# Unified-memory safety (GB10 / DGX Spark and anything else sharing one pool)
#   EXL3_LOAD_DEVICE=cuda:0   single-device load, bypassing the autosplit budget
#                         arithmetic (which reads MemFree, not MemAvailable).
#                         Set to "" to restore stock autosplit.
#   MEMGUARD=auto         auto | 1 | 0. `auto` enables util/memguard.py on Linux.
#   MEMGUARD_FLOOR=8      SIGKILL the server below this MemAvailable, in GiB
#   MIN_AVAIL_GIB=12      refuse to start below this MemAvailable, in GiB
#   FOREGROUND=1          0 backgrounds it and writes $LOG
#
# EXL3_MIMO_FP32_MLP_LAYERS / EXL3_MIMO_MLP_ACT_LIMIT / EXL3_MIMO_FP32_MLP_FUSED /
# EXL3_KEEP_PAGE_CACHE are read by exllamav3 itself; export them before running
# this script. Defaults are right for the published quant.
# ---------------------------------------------------------------------------

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"          # the exllamav3 checkout

MODEL_DIR="${1:-}"
if [ -z "$MODEL_DIR" ]; then
    echo "usage: $0 <exl3_model_dir>" >&2
    exit 2
fi
MODEL_DIR="$(readlink -f "$MODEL_DIR")"
[ -f "$MODEL_DIR/config.json" ] || { echo "no config.json in $MODEL_DIR" >&2; exit 2; }

if [ -z "${TABBY_DIR:-}" ]; then
    # Guess: a sibling of this checkout, else a subdirectory of the working directory.
    TABBY_DIR="$REPO/../tabbyAPI"
    [ -f "$TABBY_DIR/main.py" ] || TABBY_DIR="$PWD/tabbyAPI"
fi
if [ ! -f "$TABBY_DIR/main.py" ]; then
    echo "TabbyAPI not found. Set TABBY_DIR=/path/to/tabbyAPI" >&2
    echo "  git clone -b mimo-v2.6-flash https://github.com/benthecarman/tabbyAPI" >&2
    exit 2
fi
TABBY_DIR="$(readlink -f "$TABBY_DIR")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
DISABLE_AUTH="${DISABLE_AUTH:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-262144}"
CACHE_SIZE="${CACHE_SIZE:-$MAX_SEQ_LEN}"
CACHE_MODE="${CACHE_MODE:-FP16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
GPU_SPLIT="${GPU_SPLIT:-}"
WARMUP="${WARMUP:-false}"
REASONING_START="${REASONING_START:-<think>}"
REASONING_END="${REASONING_END:-</think>}"
DRAFT_DIR="${DRAFT_DIR:-}"
NGRAM_MIN="${NGRAM_MIN:-}"
DRAFT_TOKENS="${DRAFT_TOKENS:-}"
DYNAMIC_DRAFT="${DYNAMIC_DRAFT:-1}"
DRAFT_CACHE_MODE="${DRAFT_CACHE_MODE:-FP16}"
DRAFT_RING="${DRAFT_RING:-1}"
MEMGUARD="${MEMGUARD:-auto}"
MEMGUARD_FLOOR="${MEMGUARD_FLOOR:-8}"
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-12}"
FOREGROUND="${FOREGROUND:-1}"
CONFIG_OUT="${CONFIG_OUT:-$TABBY_DIR/mimo-tabby-config.yml}"
LOG="${LOG:-$TABBY_DIR/mimo-serve.log}"

# ------------------------------------------------------------------ preflight
if command -v systemctl > /dev/null 2>&1; then
    # Anything else holding most of a unified pool must be stopped first.
    for unit in sglang vllm; do
        if [ "$(systemctl is-active "$unit.service" 2>/dev/null || true)" = "active" ]; then
            echo "REFUSING: $unit.service is active; it and an 86 GiB model cannot share one pool."
            exit 1
        fi
    done
fi
if command -v ss > /dev/null 2>&1 && ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
    echo "REFUSING: something is already listening on $PORT:"; ss -ltnp "sport = :$PORT"
    exit 1
fi
if [ -r /proc/meminfo ]; then
    avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
    if [ "$avail" -lt "$MIN_AVAIL_GIB" ]; then
        echo "REFUSING: MemAvailable ${avail} GiB < ${MIN_AVAIL_GIB} GiB floor."
        exit 1
    fi
fi

# ------------------------------------------------------------------ config
PARENT="$(dirname "$MODEL_DIR")"
NAME="$(basename "$MODEL_DIR")"

draft_block="draft_mode: disabled"
draft_desc="off"
if [ -n "$DRAFT_DIR" ]; then
    DRAFT_DIR="$(readlink -f "$DRAFT_DIR")"
    [ -f "$DRAFT_DIR/config.json" ] || { echo "no config.json in DRAFT_DIR=$DRAFT_DIR" >&2; exit 2; }
    draft_block="draft_mode: model
  draft_model_dir: $(dirname "$DRAFT_DIR")
  draft_model_name: $(basename "$DRAFT_DIR")
  draft_cache_mode: $DRAFT_CACHE_MODE
  dynamic_draft: $([ "$DYNAMIC_DRAFT" = "1" ] && echo true || echo false)"
    [ -n "$DRAFT_TOKENS" ] && draft_block="$draft_block
  draft_num_tokens: $DRAFT_TOKENS"
    draft_desc="DFlash model $(basename "$DRAFT_DIR")${DRAFT_TOKENS:+ (${DRAFT_TOKENS} tok)}"
elif [ -n "$NGRAM_MIN" ]; then
    draft_block="draft_mode: ngram
  ngram_match_min: $NGRAM_MIN
  dynamic_draft: $([ "$DYNAMIC_DRAFT" = "1" ] && echo true || echo false)"
    [ -n "$DRAFT_TOKENS" ] && draft_block="$draft_block
  draft_num_tokens: $DRAFT_TOKENS"
    draft_desc="n-gram, match_min $NGRAM_MIN"
fi

gpu_split_line="gpu_split: []"
gpu_split_auto="true"
if [ -n "$GPU_SPLIT" ]; then
    gpu_split_line="gpu_split: [$GPU_SPLIT]"
    gpu_split_auto="false"
fi

cat > "$CONFIG_OUT" <<YAML
# GENERATED by examples/mimo_v2_6/serve.sh -- edit the script, not this file.
# A hand-editable version of the same thing: examples/mimo_v2_6/tabby-config.yml
network:
  host: $HOST
  port: $PORT
  disable_auth: $([ "$DISABLE_AUTH" = "1" ] && echo true || echo false)
  allowed_origins: []
  api_servers: ["OAI"]

logging:
  log_prompt: false
  log_generation_params: false
  log_requests: false
  log_live_status: false

model:
  model_dir: $PARENT
  model_name: $NAME
  max_seq_len: $MAX_SEQ_LEN
  cache_size: $CACHE_SIZE
  cache_mode: $CACHE_MODE
  chunk_size: $CHUNK_SIZE
  max_batch_size: $MAX_BATCH_SIZE
  gpu_split_auto: $gpu_split_auto
  $gpu_split_line
  warmup: $WARMUP
  vision: false
  # The MiMo template emits <think>...</think> and offers enable_thinking; TabbyAPI's
  # reasoning parser splits that into reasoning_content on the OAI response.
  reasoning: true
  reasoning_start_token: "$REASONING_START"
  reasoning_end_token: "$REASONING_END"
  start_in_reasoning: auto
  # prompt_template left unset -> TabbyAPI picks up
  # $MODEL_DIR/chat_template.jinja, and auto-detects tool format qwen3_coder.

draft_model:
  $draft_block

developer:
  unsafe_launch: false
YAML

echo "############################################################################"
echo "# TabbyAPI  ->  http://$HOST:$PORT/v1"
echo "#   model        $MODEL_DIR"
echo "#   tabbyAPI     $TABBY_DIR"
echo "#   max_seq_len  $MAX_SEQ_LEN   cache $CACHE_SIZE tok @ $CACHE_MODE   slots $MAX_BATCH_SIZE"
echo "#   config       $CONFIG_OUT"
echo "#   auth         $([ "$DISABLE_AUTH" = "1" ] && echo 'disabled (loopback only)' || echo 'api_tokens.yml')"
echo "#   drafting     $draft_desc"
echo "############################################################################"

# _load_autosplit's headroom check reads MemFree, not MemAvailable, and can refuse an
# 86 GiB model that fits on a unified-memory box. EXL3_LOAD_DEVICE= (empty) opts out.
# The DFlash draft ring: a fixed per-slot window ring for sliding-window draft layers
# instead of a draft cache sized for the whole context. EXL3_DFLASH_RING=0 restores it.
export EXL3_DFLASH_RING="$DRAFT_RING"
export EXL3_LOAD_DEVICE="${EXL3_LOAD_DEVICE-cuda:0}"
[ -n "$EXL3_LOAD_DEVICE" ] && echo " -- EXL3_LOAD_DEVICE=$EXL3_LOAD_DEVICE (single-device load)"

# memguard: SIGKILL the server's process group if MemAvailable falls below the floor,
# and fadvise(DONTNEED) the shards every 5 s so the ~86 GiB of buffered-pread page
# cache never shadows the same 86 GiB of weights in one unified pool.
use_mg=0
case "$MEMGUARD" in
    1) use_mg=1 ;;
    auto) [ -r /proc/meminfo ] && use_mg=1 ;;
esac
if [ "$use_mg" = "1" ] && [ -f "$REPO/util/memguard.py" ]; then
    MG=(python "$REPO/util/memguard.py" --floor-gib "$MEMGUARD_FLOOR"
        --fadvise-dir "$MODEL_DIR" --fadvise-every 5 --label tabbyapi --)
else
    MG=()
    [ "$use_mg" = "1" ] || echo " !! memguard disabled (MEMGUARD=$MEMGUARD)"
fi

cd "$TABBY_DIR"
if [ "$FOREGROUND" = "1" ]; then
    exec "${MG[@]}" python main.py --config "$CONFIG_OUT"
else
    setsid nohup "${MG[@]}" python main.py --config "$CONFIG_OUT" > "$LOG" 2>&1 < /dev/null &
    echo " -- backgrounded, pid $!, log $LOG"
fi
