#!/usr/bin/env python3
"""Turn the publication measurements into doc/figures.json, then regenerate the charts and the README tables.

Inputs, in the --scratch directory:
  curve_{glm,qwen}_c272384.json  served depth curve: points["8192"].."262144"  (timings, usage)
  curve_{glm,qwen}_c524288.json  served depth curve: points["499712"] (the 500K point)
  bench_{glm,qwen}_c272384.md    tools/bench.py --json output: card prompts (fenced JSON block); not read with --keep-card

Writes doc/figures.json, keeping its schema: the previous current values become "before", the new
values replace prefill / decode / spec, "table" is recomputed, "date" is the bench run date and
"commit" is the measured commit. Then runs tools/make_charts.py and tools/make_readme_table.py.
Refuses to write anything when an expected point or card value is missing or inconsistent.

usage: python3 tools/pub_to_figures.py --scratch DIR [--commit HASH] [--dry-run] [--no-render] [--add-500k]
                                       [--keep-card] [--skip-500k] [--models glm,qwen]
"""
import argparse
import copy
import difflib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("qwen", "glm")
SHORT_FILE, LONG_FILE = "c272384", "c524288"
POINTS = (8192, 32768, 65536, 131072, 262144)
POINT_500K = 499712
CARD = ("prose", "chat", "code")
CARD_MAX_TOKENS = 256
TOL_TOKENS = 64  # calibration lands on or just above the target: allowed |prompt_tokens - target|


class InputError(Exception):
    pass


def load_curve(path, targets, client_timed=False):
    """Return {target: (prefill_tok_s, decode_tok_s)}: server-reported timings; client_timed replaces the decode
    figure by the client first-to-last-token one when the curve has it (the published method for Qwen)."""
    if not os.path.exists(path):
        raise InputError(f"missing input {path}")
    with open(path, encoding="utf-8") as f:
        points = json.load(f).get("points", {})
    out = {}
    for t in targets:
        p = points.get(str(t))
        if not p:
            raise InputError(f"{os.path.basename(path)}: no point {t}")
        tm = p.get("timings") or {}
        pre, dec = tm.get("prompt_per_second"), tm.get("predicted_per_second")
        if not all(isinstance(v, (int, float)) and v > 0 for v in (pre, dec)):
            raise InputError(f"{os.path.basename(path)}: point {t} has no server-reported prefill/decode")
        ptok = (p.get("usage") or {}).get("prompt_tokens")
        if not isinstance(ptok, int) or abs(ptok - t) > max(TOL_TOKENS, t * 0.001):
            raise InputError(f"{os.path.basename(path)}: point {t} has prompt_tokens={ptok}")
        if client_timed:  # doc/benchmarks.md: Qwen decode is timed by the client, first to last streamed token
            cli = p.get("client_decode_tok_s")
            if isinstance(cli, (int, float)) and cli > 0:
                dec = cli
        out[t] = (float(pre), float(dec))
    return out


def load_bench(path):
    """Return ({prose, chat, code: tok/s}, date) from the fenced JSON block of tools/bench.py --json."""
    if not os.path.exists(path):
        raise InputError(f"missing input {path}")
    with open(path, encoding="utf-8") as f:
        m = re.search(r"```json\n(.*?)\n```", f.read(), re.S)
    if not m:
        raise InputError(f"{os.path.basename(path)}: no JSON block (bench not finished?)")
    d = json.loads(m.group(1))
    dec, spec = d.get("decode") or {}, {}
    for k in CARD:
        c = dec.get(k) or {}
        v = c.get("tok_s")
        if not isinstance(v, (int, float)) or v <= 0:
            raise InputError(f"{os.path.basename(path)}: no decode {k} tok_s")
        if c.get("max_tokens") != CARD_MAX_TOKENS:
            raise InputError(f"{os.path.basename(path)}: decode {k} max_tokens={c.get('max_tokens')}")
        spec[k] = round(float(v), 1)
    if not d.get("date"):
        raise InputError(f"{os.path.basename(path)}: no date")
    return spec, d["date"]


def curve_date(path):
    """Day of the curve run (the "started" field), or None."""
    with open(path, encoding="utf-8") as f:
        started = json.load(f).get("started") or ""
    return started[:10] or None


def collect(scratch, models=MODELS, card=True, long=True):
    """Return ({model: {prefill: {int: v}, decode: {int: v}, spec: {...} or None}}, date).

    card=False: the card prompts were not re-measured; spec is None and the date is the curve date.
    long=False: the 500K curve is not read."""
    data, dates = {}, []
    for mid in models:
        short_path = os.path.join(scratch, f"curve_{mid}_{SHORT_FILE}.json")
        pts = dict(load_curve(short_path, POINTS, mid == "qwen"))
        if long:
            pts.update(load_curve(os.path.join(scratch, f"curve_{mid}_{LONG_FILE}.json"), [POINT_500K], mid == "qwen"))
        if card:
            spec, date = load_bench(os.path.join(scratch, f"bench_{mid}_{SHORT_FILE}.md"))
        else:
            spec, date = None, curve_date(short_path)
            if not date:
                raise InputError(f"{os.path.basename(short_path)}: no started date")
        data[mid] = {
            "prefill": {t: pts[t][0] for t in pts},
            "decode": {t: pts[t][1] for t in pts},
            "spec": spec,
        }
        dates.append(date)
    return data, max(dates)


def build(F, data, date, commit, add_500k):
    """Return (new figures dict, warnings). Does not touch the input dict."""
    G = copy.deepcopy(F)
    warnings = []
    ids = {m["id"] for m in G["models"]}
    for mid in data:
        if mid not in ids:
            raise InputError(f"figures.json has no model {mid!r}")
    for m in G["models"]:
        if m["id"] not in data:
            continue
        d = data[m["id"]]
        m["before"] = {"prefill": m["prefill"], "decode": m["decode"], "spec": m["spec"]}
        published = set(m["prefill"]) | set(m["decode"])
        keys = list(POINTS) + ([POINT_500K] if (add_500k or str(POINT_500K) in published) else [])
        if POINT_500K not in keys and POINT_500K in d["prefill"]:
            warnings.append(f"{m['id']}: 500K point measured ({d['prefill'][POINT_500K]:.1f} prefill, "
                            f"{d['decode'][POINT_500K]:.1f} decode) but figures.json has no {POINT_500K} key; "
                            f"not published (use --add-500k)")
        m["prefill"] = {str(t): round(d["prefill"][t]) for t in keys}
        m["decode"] = {str(t): round(d["decode"][t], 1) for t in keys}
        if d["spec"] is not None:   # card prompts re-measured; otherwise the published card values stay
            m["spec"] = dict(d["spec"])
        vals = list(m["spec"].values())
        m["table"] = {"prefill": f"{m['prefill']['8192']} at 8K",
                      "decode": f"{min(vals):.1f} to {max(vals):.1f}"}
    G["date"] = date
    G["commit"] = commit
    return G, warnings


def dump(F):
    return json.dumps(F, indent=1, ensure_ascii=False) + "\n"


def git_head(root):
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render(root):
    for script in ("make_charts.py", "make_readme_table.py"):
        subprocess.run([sys.executable, os.path.join(root, "tools", script)], cwd=root, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scratch", required=True, help="directory with the curve_*.json (and bench_*.md) inputs")
    ap.add_argument("--models", default=",".join(MODELS), help="comma list of models to update (default: all)")
    ap.add_argument("--keep-card", action="store_true",
                    help="do not read bench_*.md: keep the published card values (spec, decode range); date = curve date")
    ap.add_argument("--skip-500k", action="store_true", help="do not read the 500K curves (partial dry run)")
    ap.add_argument("--root", default=ROOT, help="repository root (doc/figures.json, tools/, README templates)")
    ap.add_argument("--commit", help="short hash of the measured commit (default: git HEAD of --root)")
    ap.add_argument("--dry-run", action="store_true", help="print the diff of doc/figures.json; write nothing")
    ap.add_argument("--no-render", action="store_true", help="update figures.json only; skip the charts and tables")
    ap.add_argument("--add-500k", action="store_true",
                    help="publish the 500K point under key 499712 even if figures.json has no such key yet")
    a = ap.parse_args(argv)
    a.root = os.path.abspath(a.root)

    fig_path = os.path.join(a.root, "doc", "figures.json")
    try:
        with open(fig_path, encoding="utf-8") as f:
            old_text = f.read()
        commit = a.commit or git_head(a.root)
        if not commit:
            raise InputError("no --commit given and git HEAD is unavailable")
        models = tuple(m for m in a.models.split(",") if m)
        unknown = [m for m in models if m not in MODELS]
        if not models or unknown:
            raise InputError(f"--models: expected a subset of {','.join(MODELS)}, got {a.models!r}")
        if a.skip_500k and a.add_500k:
            raise InputError("--skip-500k and --add-500k contradict each other")
        data, date = collect(a.scratch, models, card=not a.keep_card, long=not a.skip_500k)
        G, warnings = build(json.loads(old_text), data, date, commit, a.add_500k)
    except (InputError, OSError, json.JSONDecodeError) as e:
        print(f"pub_to_figures: {e}", file=sys.stderr)
        print("pub_to_figures: nothing written", file=sys.stderr)
        return 1

    new_text = dump(G)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    diff = list(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                     "a/doc/figures.json", "b/doc/figures.json"))
    if a.dry_run:
        sys.stdout.write("".join(diff) if diff else "no changes\n")
        return 0
    if new_text != old_text:
        with open(fig_path, "w", encoding="utf-8") as f:
            f.write(new_text)
        print(f"wrote {fig_path} ({len(diff)} diff lines)")
    else:
        print("figures.json unchanged")
    if not a.no_render:
        try:
            render(a.root)
        except subprocess.CalledProcessError as e:
            print(f"pub_to_figures: render failed ({e.cmd[1]}, rc {e.returncode}); doc/figures.json is already "
                  "updated: fix the renderer, then run the renderers again", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
