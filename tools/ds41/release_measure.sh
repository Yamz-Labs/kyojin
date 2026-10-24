#!/usr/bin/env bash
# DS4.1 release measurement: quality gate + served speed table for one converted pack. Markdown table on stdout.
#   tools/ds41/release_measure.sh PACK_DIR [-- options]
# Steps (each GPU step is its own gpurun window; finished steps are kept in $OUT and skipped on a re-run):
#   1. quality gate      tools/ds41qual/gate.py        -> top-1 / KLD vs the reference logits  (needs GATE_REF)
#   2. Lucebox eval      tools/ds41/luce_eval.py       -> KL top-256 / top-1 / ppl            (needs LUCE_DATA)
#   3. speed             one window per depth and mode (plain, dspark): cold prefill and decode, median of REPS
# Env (defaults in brackets):
#   OUT [./ds41_release_out]  results folder          GATE_REF [<repo>/tools/ds41qual/data/gate_ref]
#   LUCE_DATA [unset = skip]  folder with eval/ and results/ of the public Lucebox set
#   DEPTHS ["8192 32768 131072 524288 1048576"]  prompt sizes in tokens      REPS [5]  repeats per cell
#   NUM_DRAFT [5]  draft tokens for the dspark rows (0..5)       PORT [8398]    DECODE_TOKENS [256]
#   WINDOW_S [900]  gpurun cap per window (gpurun accepts at most 900 s)
#   STEPS ["gate luce speed"]
#   DRYRUN=1  no GPU: stub engine, no gpurun, tiny depths. Checks the plumbing only; the numbers mean nothing.
# A cold prefill of 512K or 1M tokens takes longer than one 900 s window. Those cells are reported as
# "window cap" unless the owner raises the cap (WINDOW_S above 900 is clamped by gpurun).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(dirname "$(dirname "$HERE")")}
GPURUN=${GPURUN:-$HOME/ds41_work/bin/gpurun}
PY=${PY:-python3}
OUT=${OUT:-$PWD/ds41_release_out}
PORT=${PORT:-8398}
REPS=${REPS:-5}
NUM_DRAFT=${NUM_DRAFT:-5}
DECODE_TOKENS=${DECODE_TOKENS:-256}
WINDOW_S=${WINDOW_S:-900}
STEPS=${STEPS:-"gate luce speed"}
GATE_REF=${GATE_REF:-$ROOT/tools/ds41qual/data/gate_ref}
DEPTHS=${DEPTHS:-"8192 32768 131072 524288 1048576"}
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" = 1 ]; then DEPTHS=${DEPTHS_DRY:-"512 2048"}; REPS=${REPS_DRY:-3}; DECODE_TOKENS=${DECODE_TOKENS_DRY:-24}; fi

# ---- window mode: runs INSIDE one gpurun window (server up, one client, server down) ----
if [ "${1:-}" = "__window" ]; then
  kind=$2; pack=$3; ctx=$4; draft=$5; out=$6; depth=${7:-0}
  log="$out.serve.log"
  if [ "$DRYRUN" = 1 ]; then
    srv=("$PY" serve.py --model "$pack" --loader stub -c "$ctx" --port "$PORT" --engine-opt replies="$(printf 'word %.0s' $(seq 1 400))")
  else
    loader=ds41_model:load_model; [ "$kind" != speed ] && loader=ds41_serving:load_model
    srv=("$PY" serve.py --model "$pack" --loader "$loader" -c "$ctx" --port "$PORT")
    [ "$draft" -gt 0 ] && srv+=(--num-draft "$draft")
  fi
  (cd "$ROOT/tools/ds41" && exec "${srv[@]}") > "$log" 2>&1 &
  spid=$!
  trap 'kill $spid 2>/dev/null; wait $spid 2>/dev/null' EXIT
  ok=0
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { ok=1; break; }
    kill -0 $spid 2>/dev/null || break
    sleep 5
  done
  [ $ok = 1 ] || { echo "server did not come up, see $log" >&2; exit 3; }
  url="http://127.0.0.1:$PORT"
  case "$kind" in
    gate)  "$PY" "$ROOT/tools/ds41qual/gate.py" --base-url "$url" --ref "$GATE_REF" --out "$out" ;;
    luce)  "$PY" "$ROOT/tools/ds41/luce_eval.py" run --data "$LUCE_DATA" --base-url "$url" --label "$(basename "$pack")" \
             --bytes "$(du -sb "$pack" | cut -f1)" --out "$out" --cache "$out.cache" ;;
    speed) "$PY" -I - "$url" "$depth" "$REPS" "$DECODE_TOKENS" "$ROOT" "$out" <<'PYEOF'
import json, statistics as st, sys, time, urllib.request, pathlib
url, depth, reps, ntok, root, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]

def post(path, body, timeout=3600):
    r = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

# natural, non-repeating text: the repository's own sources, in a fixed order
text = []
for p in sorted(pathlib.Path(root).rglob("*")):
    if p.suffix in (".md", ".py", ".cpp", ".h", ".sh") and p.is_file() and ".git" not in p.parts:
        try: text.append(p.read_text(errors="ignore"))
        except OSError: pass
text = "\n".join(text)

def msgs(t):
    return [{"role": "user", "content": t}]

def count(t):  # prompt length of the chat request, template included
    return post("/tokenize", {"messages": msgs(t)})["count"]

cpt = 3.3  # characters per token, refined below
def build(n, off):
    """A slice of the corpus whose chat prompt has n tokens (within 1 %)."""
    global cpt
    for _ in range(5):
        t = text[off:off + int(n * cpt)]
        c = count(t)
        if abs(c - n) <= max(2, n // 100): return t, c
        cpt *= n / c
    return t, c

def run(t, n, salt):
    body = {"messages": msgs(t), "max_tokens": n, "temperature": 0, "thinking": False, "cache_salt": salt}
    r = post("/v1/chat/completions", body)
    return r["timings"]

if len(text) < depth * 3.5 + 4096 * (reps + 1):
    sys.exit(f"corpus has {len(text)} characters, too short for {depth} tokens")
res = {"depth": depth, "reps": reps, "prefill": [], "decode": [], "accepted_per_step": [], "decode_tokens": []}
# cold prefill: another slice of the corpus each time and another salt, so no cache hit
for k in range(reps):
    t, c = build(depth, k * 4096)
    tm = run(t, 1, f"pf{k}")
    if tm.get("cache_n", 0): sys.exit("prefix cache hit during a cold prefill measurement")
    res["prefill"].append(tm["prompt_per_second"])
    if depth >= 524288: break   # one cold prefill at this size already fills a window
# decode at this depth: the prompt stays cached; the last words differ so every run is a new request
base, c = build(depth, 0)
run(base, 1, "dec")
for k in range(reps):
    tm = run(base + f"\nQuestion {k}: continue the text above in the same style.", ntok, "dec")
    if tm["predicted_n"] < ntok // 2: print(f"warning: only {tm['predicted_n']} tokens decoded", file=sys.stderr)
    res["decode"].append(tm["predicted_per_second"]); res["decode_tokens"].append(tm["predicted_n"])
    if "accepted_per_step" in tm: res["accepted_per_step"].append(tm["accepted_per_step"])
json.dump(res, open(out, "w"))
print(json.dumps(res))
PYEOF
    ;;
  esac
  exit $?
fi

# ---- driver ----
mkdir -p "$OUT"
PACK=${1:?usage: release_measure.sh PACK_DIR}
[ -d "$PACK" ] || { echo "no pack at $PACK" >&2; exit 2; }
PACK=$(cd "$PACK" && pwd)
export ROOT PY PORT REPS DECODE_TOKENS DRYRUN GATE_REF LUCE_DATA="${LUCE_DATA:-}"

window() {  # window KIND CTX DRAFT OUTFILE [DEPTH]: one GPU window, skipped when OUTFILE exists
  local kind=$1 ctx=$2 draft=$3 of=$4 depth=${5:-0}
  [ -s "$of" ] && return 0
  if [ "$DRYRUN" = 1 ]; then
    bash "$0" __window "$kind" "$PACK" "$ctx" "$draft" "$of" "$depth" > /dev/null
  else
    "$GPURUN" --quiet --max "$WINDOW_S" --grace 600 bash "$0" __window "$kind" "$PACK" "$ctx" "$draft" "$of" "$depth" > /dev/null
  fi
  local rc=$?
  [ $rc = 0 ] || { echo "step $kind ($of) failed, rc=$rc" >&2; rm -f "$of"; }
  return $rc
}

for s in $STEPS; do
  case $s in
    gate) window gate 16384 0 "$OUT/gate.json" ;;
    luce) [ -n "${LUCE_DATA:-}" ] && window luce 4096 0 "$OUT/luce.json" ;;
    speed)
      for d in $DEPTHS; do
        window speed $((d + DECODE_TOKENS + 64)) 0 "$OUT/speed_plain_$d.json" "$d"
        window speed $((d + DECODE_TOKENS + 64)) "$NUM_DRAFT" "$OUT/speed_dspark_$d.json" "$d"
      done ;;
  esac
done

# ---- report (CPU only) ----
"$PY" -I - "$OUT" "$(basename "$PACK")" "$(du -sb "$PACK" | cut -f1)" $DEPTHS <<'PYEOF'
import json, os, statistics as st, sys
out, name, size, depths = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4:]]
def load(f):
    p = os.path.join(out, f)
    return json.load(open(p)) if os.path.exists(p) else None
def med(v): return f"{st.median(v):.1f}" if v else "n/a"
def lab(d): return f"{d // 1024}K" if d >= 1024 else str(d)
print(f"## {name}\n\nPack size: {size / 2**30:.2f} GiB\n")
g = load("gate.json")
print("### Quality\n")
if g:
    print("| Measure | Value |\n|---|---|")
    for k, v in g.items():
        if isinstance(v, (int, float, str)): print(f"| gate {k} | {v} |")
else: print("Quality gate: not run.")
l = load("luce.json")
if l:
    for k in ("kl_top256", "top1_agree", "ppl", "teacher_nll", "tokens"):
        if k in l: print(f"| Lucebox set {k} | {l[k]} |")
print("\n### Speed (tok/s, median of repeats)\n")
print("| Depth | Prefill | Decode plain | Decode dspark | dspark accepted per step | Repeats |\n|---|---|---|---|---|---|")
for d in depths:
    p, s = load(f"speed_plain_{d}.json"), load(f"speed_dspark_{d}.json")
    pf = (p or s or {}).get("prefill")
    pre = med(pf) if pf else ("window cap" if d >= 524288 else "n/a")
    n = len(pf) if pf else 0
    print(f"| {lab(d)} | {pre} | {med(p['decode']) if p else 'n/a'} | {med(s['decode']) if s else 'n/a'} | "
          f"{med(s['accepted_per_step']) if s else 'n/a'} | {n} prefill, {len(p['decode']) if p else 0} decode |")
PYEOF
