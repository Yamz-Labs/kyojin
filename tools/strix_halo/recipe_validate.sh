#!/usr/bin/env bash
# =============================================================================
# recipe_validate.sh -- adversarial validation of the published recipe.
#
# Runs unattended, writes JSON lines to $OUT/results.jsonl plus a human report
# at $OUT/REPORT.md. Every measurement is tagged with the page-cache state and
# repeat index so first-run effects cannot hide.
#
#   bash recipe_validate.sh            # full pipeline (~60 min)
#   PHASES="clone cold" bash ...       # subset
#
# Phases:
#   clone   fresh clone of the PUBLISHED recipe from GitHub (not the local tree)
#   smoke   setup.sh's own smoke assertions, run standalone
#   cold    drop page cache -> run -> measure; the 8 tok/s report
#   warm    same configs with the model already in page cache, x3 each
#   sweep   cache-size sweep, warm, greedy, fixed prompt
#   claims  re-measure every headline number the README asserts
# =============================================================================
set -u
OUT="${OUT:-/tmp/recipe_validate}"
PHASES="${PHASES:-clone smoke cold warm sweep claims}"
RECIPE_URL="${RECIPE_URL:-https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe}"
RECIPE_DIR="${RECIPE_DIR:-/tmp/recipe_pub}"
REPO_DIR="${REPO_DIR:-$HOME/kyojin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
PROMPT="${PROMPT:-Explain gradient descent in two sentences.}"
MAXR="${MAXR:-120}"
mkdir -p "$OUT"
JSONL="$OUT/results.jsonl"
: > "$JSONL"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/pipeline.log"; }

# emit a json line: emit key=val ...  (values are strings unless numeric)
emit() {
  python3 - "$@" >> "$JSONL" <<'PY'
import sys, json
d = {}
for a in sys.argv[1:]:
    k, _, v = a.partition("=")
    try:
        d[k] = int(v) if v.isdigit() else float(v)
    except ValueError:
        d[k] = v
print(json.dumps(d))
PY
}

mem_line() { free -g | sed -n 2p | awk '{print "total="$2" used="$3" free="$4" buffcache="$6}'; }
gtt_used()  { cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1; }
vram_used() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1; }

drop_caches() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && sleep 3; }

# Run the published run.sh once; parse chat.py's summary line.
#   run_one <tag> <cache> <cq> <cachestate> <rep> [extra run.sh args...]
run_one() {
  local tag=$1 cs=$2 cq=$3 state=$4 rep=$5; shift 5
  local raw="$OUT/run_${tag}_${cs}_${cq:-fp16}_${state}_${rep}.raw"
  local t0=$(date +%s.%N)
  REPO_DIR="$REPO_DIR" MODEL_DIR="$MODEL_DIR" CACHE="$cs" CQ="$cq" \
    timeout 1200 bash "$RECIPE_DIR/scripts/run.sh" -prompt "$PROMPT" -maxr "$MAXR" -no_think "$@" \
    > "$raw" 2>&1
  local rc=$? t1=$(date +%s.%N)
  local wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")
  # "Context: 32 new tokens at 112.05 t/s ... Generate: 51 tokens at 28.29 t/s - Draft: 28 / 69 accepted (40.58%)"
  local line=$(grep -E "^Context:.*Generate:" "$raw" | tail -1)
  local ttft_tps gen_tps acc oom
  ttft_tps=$(sed -n 's/.*Context: [0-9]* new tokens at \([0-9.]*\) t\/s.*/\1/p' <<<"$line")
  gen_tps=$(sed -n 's/.*Generate: [0-9]* tokens at \([0-9.]*\) t\/s.*/\1/p' <<<"$line")
  acc=$(sed -n 's/.*accepted (\([0-9.]*\)%).*/\1/p' <<<"$line")
  oom=$(grep -c "Insufficient VRAM" "$raw")
  emit phase="$tag" cache="$cs" cq="${cq:-fp16}" cachestate="$state" rep="$rep" rc="$rc" \
       wall_s="$wall" prefill_tps="${ttft_tps:-0}" decode_tps="${gen_tps:-0}" \
       acceptance="${acc:-0}" oom="$oom" gtt_kb="$(gtt_used)" vram_kb="$(vram_used)" raw="$raw"
  log "  $tag cs=$cs cq=${cq:-fp16} $state rep$rep -> decode=${gen_tps:-FAIL} t/s acc=${acc:-?}% oom=$oom rc=$rc"
}

has() { grep -qw "$1" <<<"$PHASES"; }

# ---------------------------------------------------------------- clone
if has clone; then
  log "PHASE clone: fresh clone of the PUBLISHED recipe"
  rm -rf "$RECIPE_DIR"
  git clone -q "$RECIPE_URL" "$RECIPE_DIR" && log "  cloned $(git -C "$RECIPE_DIR" rev-parse --short HEAD)"
  for f in README.md AGENTS.md scripts/setup.sh scripts/download.sh scripts/run.sh LICENSE; do
    [[ -f "$RECIPE_DIR/$f" ]] && log "  present: $f" || { log "  MISSING: $f"; emit phase=clone missing="$f"; }
  done
  # every script must at least parse
  for s in "$RECIPE_DIR"/scripts/*.sh; do
    bash -n "$s" && log "  syntax ok: $(basename $s)" || { log "  SYNTAX ERROR: $s"; emit phase=clone syntax_error="$s"; }
  done
  emit phase=clone commit="$(git -C "$RECIPE_DIR" rev-parse --short HEAD)" ok=1
fi

# ---------------------------------------------------------------- smoke
if has smoke; then
  log "PHASE smoke: does the documented environment actually hold?"
  cd "$REPO_DIR" && source env.sh
  .venv/bin/python - > "$OUT/smoke.raw" 2>&1 <<'PY'
import torch, exllamav3_ext as e, json
r = {"arch_list": torch.cuda.get_arch_list(), "torch": torch.__version__, "hip": torch.version.hip,
     "gemv_supported": bool(e.exl3_gemv_supported(0)), "wmma_family": int(e.exl3_gemv_wmma_family(0))}
a = torch.randn(512, 512, dtype=torch.float16, device="cuda"); (a @ a).sum().item(); torch.cuda.synchronize()
r["compute_ok"] = True
print("SMOKEJSON " + json.dumps(r))
PY
  grep SMOKEJSON "$OUT/smoke.raw" | sed 's/SMOKEJSON //' >> "$JSONL" || emit phase=smoke failed=1
  log "  $(grep SMOKEJSON "$OUT/smoke.raw" || echo 'SMOKE FAILED')"
fi

# ---------------------------------------------------------------- cold
# The user report: "followed AGENTS.md, got 8 tok/s". Hypothesis: on a 122 GB box the
# 80 GB of safetensors fill the page cache, and the GPU's ~59 GB GTT allocation comes
# out of the SAME pool, so the first run after a fresh boot/download reclaims while
# it runs. Test: drop caches, run published default vs small cache, then repeat warm.
if has cold; then
  log "PHASE cold: page-cache-cold runs (the 8 tok/s report)"
  for cfg in "204800:4" "32768:"; do
    cs=${cfg%%:*}; cq=${cfg#*:}
    drop_caches; log "  caches dropped: $(mem_line)"
    run_one cold "$cs" "$cq" cold 1
    log "  after run:       $(mem_line)"
  done
fi

# ---------------------------------------------------------------- warm
if has warm; then
  log "PHASE warm: same configs, page cache warm, 3 reps each (variance from sampling)"
  for cfg in "204800:4" "32768:"; do
    cs=${cfg%%:*}; cq=${cfg#*:}
    for rep in 1 2 3; do run_one warm "$cs" "$cq" warm "$rep"; sleep 5; done
  done
fi

# ---------------------------------------------------------------- sweep
if has sweep; then
  log "PHASE sweep: cache size vs decode, warm"
  for cfg in "32768:" "32768:4" "65536:4" "131072:4" "204800:4" "262144:4"; do
    cs=${cfg%%:*}; cq=${cfg#*:}
    run_one sweep "$cs" "$cq" warm 1; sleep 5
  done
fi

# ---------------------------------------------------------------- claims
# Re-measure the README's headline numbers with the harness, greedy, so sampling
# variance cannot flatter or damn them.
if has claims; then
  log "PHASE claims: re-measure README headline numbers (greedy)"
  cd "$REPO_DIR" && source tools/strix_halo/env.sh >/dev/null 2>&1
  export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
  for ndt in 2 3; do
    NDT=$ndt DC=0.6 .venv/bin/python tools/strix_halo/prompt_sweep.py > "$OUT/sweep_ndt$ndt.raw" 2>&1
    m=$(grep -oE "mean[ =:]+[0-9.]+" "$OUT/sweep_ndt$ndt.raw" | tail -1 | grep -oE "[0-9.]+$")
    [[ -z "${m:-}" ]] && m=$(grep -oE "^[0-9.]+ tok/s" "$OUT/sweep_ndt$ndt.raw" | grep -oE "^[0-9.]+" | python3 -c "
import sys; v=[float(x) for x in sys.stdin]; print(f'{sum(v)/len(v):.2f}' if v else '')" )
    emit phase=claims metric="six_prompt_mean_ndt$ndt" value="${m:-0}"
    log "  six-prompt mean ndt=$ndt: ${m:-PARSE-FAIL}"
  done
  # single best prompt, greedy
  .venv/bin/python tools/strix_halo/bench_mtp.py -n 512 -ndt 3 -dds -g -dc 0.6 > "$OUT/bench_best.raw" 2>&1
  b=$(grep -oE "decode: +[0-9.]+" "$OUT/bench_best.raw" | grep -oE "[0-9.]+")
  emit phase=claims metric=best_prompt_greedy value="${b:-0}"
  log "  best-prompt greedy: ${b:-PARSE-FAIL} tok/s"
  # perplexity gate
  .venv/bin/python eval/ppl.py -m "$MODEL_DIR" -r 20 -l 1024 > "$OUT/ppl.raw" 2>&1
  p=$(grep -oE "Perplexity: [0-9.]+" "$OUT/ppl.raw" | grep -oE "[0-9.]+")
  emit phase=claims metric=ppl value="${p:-0}" gate=4.225935
  log "  perplexity: ${p:-PARSE-FAIL} (gate 4.225935)"
fi

# ---------------------------------------------------------------- report
log "Writing $OUT/REPORT.md"
python3 - "$JSONL" "$OUT/REPORT.md" <<'PY'
import json, sys, collections, statistics
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out = open(sys.argv[2], "w")
W = out.write
W("# Recipe validation report\n\n")
env = [r for r in rows if "wmma_family" in r]
if env:
    e = env[0]
    W(f"Environment: torch {e.get('torch')} / HIP {e.get('hip')}, wmma_family={e.get('wmma_family')} "
      f"(want 2), gemv_supported={e.get('gemv_supported')}\n\n")
runs = [r for r in rows if "decode_tps" in r]
if runs:
    W("## Runs\n\n| phase | cache | cq | page cache | rep | decode t/s | prefill t/s | accept % | OOM | wall s |\n")
    W("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in runs:
        W(f"| {r['phase']} | {r['cache']} | {r['cq']} | {r['cachestate']} | {r['rep']} | "
          f"**{r['decode_tps']}** | {r['prefill_tps']} | {r['acceptance']} | {r['oom']} | {r['wall_s']} |\n")
    W("\n### Cold vs warm (same config)\n\n")
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        if r["decode_tps"]:
            by[(r["cache"], r["cq"])][r["cachestate"]].append(r["decode_tps"])
    W("| cache | cq | cold | warm (mean of reps) | ratio |\n|---|---|---|---|---|\n")
    for (cs, cq), d in sorted(by.items()):
        cold = statistics.mean(d["cold"]) if d.get("cold") else None
        warm = statistics.mean(d["warm"]) if d.get("warm") else None
        ratio = f"{warm/cold:.2f}x" if cold and warm else "-"
        W(f"| {cs} | {cq} | {cold if cold else '-'} | {warm:.1f} | {ratio} |\n" if warm
          else f"| {cs} | {cq} | {cold} | - | - |\n")
claims = [r for r in rows if r.get("phase") == "claims"]
if claims:
    W("\n## README claims re-measured\n\n| metric | measured |\n|---|---|\n")
    for c in claims:
        W(f"| {c['metric']} | {c['value']} |\n")
bad = [r for r in runs if r.get("oom") or not r.get("decode_tps")]
W(f"\n## Failures\n\n{len(bad)} run(s) failed or OOMed.\n")
for r in bad:
    W(f"- {r['phase']} cache={r['cache']} cq={r['cq']} {r['cachestate']}: rc={r['rc']} raw={r['raw']}\n")
out.close()
print(open(sys.argv[2]).read())
PY
log "PIPELINE DONE"
echo VALIDATE_DONE
