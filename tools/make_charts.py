#!/usr/bin/env python3
"""Build the README charts (SVG, standard library only) from doc/figures.json.
usage: python3 tools/make_charts.py [figures.json] [out_dir]
Four line charts, one idea each: prefill against context length (log scale), speculative decode by kind of text, and speculative decode against context length (Kyojin now and before, rivals where they publish a figure at that context), and prefill against context with the rivals. Higher is better.
Dark card background, so they read the same on GitHub light and dark themes.
A model without published values is listed in the legend and not drawn."""
import json, sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "doc", "figures.json")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "doc", "img")
F = json.load(open(src)); M = F["models"]
CARD, GRID, TXT, SUB = "#18181F", "#2E2E38", "#F4F2EC", "#9A98A3"
COL = {"qwen": "#9C8CFF", "glm": "#F0B04A", "mimo": "#3FD0B0"}
SHORT = {"qwen": "Qwen3.8-Flash-Next", "glm": "GLM-5.3-Flash", "mimo": "MiMo-V2.6-Flash"}
FONT = "font-family=\"'Ubuntu Sans','DejaVu Sans',Helvetica,Arial,sans-serif\""
def fmt(v): return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"
def ck(c): return f"{int(c) // 1024}K"
def t(x, y, s, size, fill=TXT, anchor="start", weight="400"):
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'

import math
def frame(headline, sub):
    W, H = 560, 380
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT}>'
            f'<rect width="{W}" height="{H}" rx="16" fill="{CARD}" stroke="{GRID}"/>'
            + t(26, 44, headline, 25, TXT, "start", "700") + t(26, 70, sub, 16, SUB)), W, H
def draw_lines(headline, footer, xs, label, pos, series, note=None):
    """xs: sorted x keys; pos: x key -> pixel; series: [{name, color, pts {x: value}, dash, hollow}].
    A series without points gets a muted legend line and is not drawn."""
    s, W, H = frame(headline, "Higher is better.")
    x0, x1, y0, y1 = 64, W - 28, 176, 316
    vals = [v for se in series for v in se["pts"].values() if v]
    top = max(vals) * 1.15 if vals else 100
    step = next(st for st in (5, 10, 20, 50, 100, 200, 250, 500) if top / st <= 4)
    top = step * -(-top // step)
    for i in range(int(top // step) + 1):
        v = i * step; y = y1 - (y1 - y0) * v / top
        s += f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" stroke="{GRID}"/>' + t(x0 - 10, y + 5, f"{v:,.0f}", 15, SUB, "end")
    px = pos(x0 + 28, x1 - 28)
    for c in xs: s += t(px[c], y1 + 28, label(c), 16, SUB, "middle")
    for i, se in enumerate(series):
        col = se["color"]; g = se["pts"]
        pts = [(px[c], y1 - (y1 - y0) * g[c] / top, g[c]) for c in xs if g.get(c)]
        ly = 98 + 22 * i
        txt = se["name"] + ("" if pts else " (" + (note or "see table") + ")")
        dash = ' stroke-dasharray="8 7"' if se.get("dash") else ""
        sw = f'<rect x="26" y="{ly-12}" width="14" height="14" rx="3" fill="{col if pts else GRID}"/>'
        if se.get("dash") and pts: sw = f'<rect x="27" y="{ly-11}" width="12" height="12" rx="3" fill="none" stroke="{col}" stroke-width="2.5" stroke-dasharray="4 3"/>'
        if se.get("hollow") and pts: sw = f'<rect x="27" y="{ly-11}" width="12" height="12" rx="6" fill="{CARD}" stroke="{col}" stroke-width="3"/>'
        s += sw + t(48, ly, txt, 16, TXT if pts else SUB)
        if len(pts) > 1: s += f'<polyline fill="none" stroke="{col}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"{dash} points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, _ in pts)}"/>'
        for k, (a, b, v) in enumerate(pts):
            s += (f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" fill="{CARD}" stroke="{col}" stroke-width="3"/>' if se.get("hollow")
                  else f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5.5" fill="{col}"/>')
            lab = (k == len(pts) - 1 and se.get("last_label", True)) or (k == 0 and se.get("first_label", True))
            if lab and not se.get("dash"): s += t(a, b + (26 if se.get("below") else -12), fmt(v), 16, col, "middle", "700")
    return s + t(26, H - 18, footer, 14, SUB) + "</svg>"

def draw(key, headline, footer, xs, label, pos, getter, note=None):
    """One series per model; getter(model) -> {x key: value}."""
    return draw_lines(headline, footer, xs, label, pos,
                      [{"name": SHORT[m["id"]], "color": COL[m["id"]], "pts": getter(m)} for m in M], note)

def prefill_chart():
    ticks = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    allx = sorted({int(c) for m in M for c, v in m["prefill"].items() if v})
    xs = sorted(set(ticks) | set(allx)); lo, hi = math.log2(ticks[0]), math.log2(ticks[-1])
    pos = lambda a, b: {c: a + (b - a) * (math.log2(c) - lo) / (hi - lo) for c in xs}
    qv = [v for v in M[0]["prefill"].values() if v]; cmax = max(int(c) for c, v in M[0]["prefill"].items() if v)
    head = f"Qwen prefill: {fmt(min(qv))}+ tok/s to {ck(cmax)}"
    foot = f'Prompt tokens per second against context; one {F["machine"].split(",")[0]}, 128 GB.'
    return draw("prefill", head, foot, xs, lambda c: ck(c) if c in ticks else "", pos, lambda m: {int(c): v for c, v in m["prefill"].items() if v})

def decode_chart():
    kinds = ["prose", "chat", "code"]
    pos = lambda a, b: {k: a + 40 + (b - a - 80) * i / 2 for i, k in enumerate(kinds)}
    allv = [v for m in M for v in m["spec"].values() if v]
    head = f"Qwen decode: {fmt(min(M[0]['spec'].values()))} to {fmt(max(M[0]['spec'].values()))} tok/s"
    foot = "Speculative decode by kind of text; same machine, 128 GB."
    return draw("decode", head, foot, kinds, str.capitalize, pos, lambda m: {k: v for k, v in m["spec"].items() if v}, "per context in the table")

def prefill_context_chart():
    """Prefill against context length: Kyojin and the rivals where a figure exists (Strata by the independent lab, Gufo measured by us)."""
    q = M[0]; R = F.get("rivals", {})
    ticks = [8192, 16384, 32768, 65536, 131072, 262144]
    ints = lambda d: {int(c): v for c, v in d.items() if v}
    series = [{"name": "Kyojin", "color": COL["qwen"], "pts": ints(q["prefill"]), "below": True, "first_label": False}]
    if R.get("strata"):
        series.append({"name": "Strata (" + R["strata"]["independent"]["lab"] + ")", "color": "#E8745C", "pts": ints(R["strata"]["prefill_independent"]), "hollow": True, "dash": True, "last_label": False})
    if R.get("gufo"): series.append({"name": "Gufo (measured by us)", "color": "#8FA3C4", "pts": ints(R["gufo"]["prefill_measured"]), "first_label": False, "last_label": False})
    xs = sorted(set(ticks) | {c for se in series for c in se["pts"]}); lo, hi = math.log2(ticks[0]), math.log2(ticks[-1])
    pos = lambda a, b: {c: a + (b - a) * (math.log2(c) - lo) / (hi - lo) for c in xs}
    pts = ints(q["prefill"]); cmax = max(pts) if pts else 0
    head = f"Qwen prefill: {fmt(pts[cmax])} tok/s at {ck(cmax)}" if pts else "Qwen prefill by context"
    foot = "Prompt tokens per second against context; same machine, 128 GB."
    return draw_lines(head, foot, xs, lambda c: ck(c) if c in ticks else "", pos, series)

def decode_context_chart():
    """Speculative decode against context length: Kyojin now and before, and the rivals where a figure exists at that context."""
    q = M[0]; R = F.get("rivals", {})
    ticks = [8192, 16384, 32768, 65536, 131072, 262144]
    lo, hi = math.log2(ticks[0]), math.log2(ticks[-1])
    pos = lambda a, b: {c: a + (b - a) * (math.log2(c) - lo) / (hi - lo) for c in ticks}
    ints = lambda d: {int(c): v for c, v in d.items() if v}
    series = [{"name": "Kyojin", "color": COL["qwen"], "pts": ints(q["decode"]), "below": True, "first_label": False}]
    if q.get("before"): series.append({"name": "Kyojin, previous release", "color": COL["qwen"], "pts": ints(q["before"]["decode"]), "dash": True})
    if R.get("strata"): series.append({"name": "Strata (measured by us)", "color": "#E8745C", "pts": ints(R["strata"]["decode_measured"]), "first_label": False, "last_label": False})
    if R.get("gufo"): series.append({"name": "Gufo (measured by us)", "color": "#8FA3C4", "pts": ints(R["gufo"]["decode_measured"]), "first_label": False, "last_label": False})
    pts = ints(q["decode"]); cmax = max(pts) if pts else 0
    head = f"Qwen decode: {fmt(pts[cmax])} tok/s at {ck(cmax)}" if pts else "Qwen decode by context"
    foot = "Speculative decode against context; same machine, 128 GB."
    return draw_lines(head, foot, ticks, lambda c: ck(c), pos, series)

os.makedirs(out, exist_ok=True)
for name, svg in (("prefill", prefill_chart()), ("decode", decode_chart()), ("decode_context", decode_context_chart()), ("prefill_context", prefill_context_chart())): open(os.path.join(out, name + ".svg"), "w").write(svg)
print("wrote", out)
