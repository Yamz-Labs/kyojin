#!/usr/bin/env python3
"""Build the homepage charts and platform cards (SVG, standard library only), each in a light and a dark variant.
Data: doc/figures.json (Strix Halo) and doc/figures_platforms.json (RTX, Mac, engine duels).
usage: python3 tools/make_home_charts.py [out_dir]     (default doc/img/home)
Files: card-<platform>, decode-<platform>, prefill-<platform>, duel-<name>, each as -light.svg and -dark.svg.
One accent colour per platform (AMD red, NVIDIA green, Apple blue); the other lines are neutral."""
import json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "doc", "img", "home")
F = json.load(open(os.path.join(ROOT, "doc", "figures.json")))
P = json.load(open(os.path.join(ROOT, "doc", "figures_platforms.json")))

THEMES = {
    "light": dict(bg="#FFFFFF", edge="#D0D7DE", txt="#1F2328", sub="#59636E", grid="#E4E8EC", n1="#444C56", n2="#8C959F",
                  amd="#D6001C", nv="#4E8F00", apple="#3D6FB6"),
    "dark": dict(bg="#0D1117", edge="#30363D", txt="#E6EDF3", sub="#9DA7B3", grid="#262C34", n1="#C9D1D9", n2="#7D8590",
                 amd="#FF5A6B", nv="#76B900", apple="#79A8F0"),
}
ACCENT = {"strix": "amd", "rtx": "nv", "mac": "apple"}
FONT = "font-family=\"-apple-system,'Segoe UI','Ubuntu Sans','DejaVu Sans',Helvetica,Arial,sans-serif\""
W, H = 600, 400


def ctx_label(c):
    c = int(c)
    if c == 499712: return "500K"
    return f"{c // 1024}K" if c % 1024 == 0 else f"{round(c / 1000)}K"


def fmt(v): return f"{v:,.0f}" if v >= 100 else (f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}")


def tx(x, y, s, size, fill, anchor="start", weight="400"):
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def frame(th, w, h, title, sub_lines):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT}>'
         f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{th["bg"]}" stroke="{th["edge"]}"/>')
    if title: s += tx(24, 42, title, 28, th["txt"], "start", "700")
    for i, l in enumerate(sub_lines): s += tx(24, 70 + 23 * i, l, 17, th["sub"])
    return s


def spread(ys, gap, lo, hi):
    """Push label baselines apart (top to bottom) so they do not overlap; keep them inside [lo, hi]."""
    order = sorted(range(len(ys)), key=lambda i: ys[i]); out = list(ys); prev = None
    for i in order:
        y = min(max(out[i], lo), hi)
        if prev is not None and y < prev + gap: y = prev + gap
        out[i] = y; prev = y
    return out


def line_chart(th, title, sub_lines, accent, series, xs, top=None, legend=False):
    """series: [{name, color, pts [(x, value, hollow)], dash}]. Log-scale x, y from 0."""
    s = frame(th, W, H, title, sub_lines)
    x0, x1, y0, y1 = 30, W - 78, (128 + 22 * len(series)) if legend else 128, 336
    vals = [v for se in series for _, v, _ in se["pts"]]
    top = top or max(vals) * 1.12
    lo, hi = math.log2(xs[0]), math.log2(xs[-1])
    px = lambda c: x0 + 44 + (x1 - x0 - 44) * (math.log2(c) - lo) / (hi - lo)
    py = lambda v: y1 - (y1 - y0) * v / top
    for k in range(4):
        y = y1 - (y1 - y0) * k / 3
        s += f'<line x1="{x0}" x2="{W - 24}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["grid"]}" stroke-width="1.5"/>'
    for c in xs: s += tx(px(c), y1 + 30, ctx_label(c), 18, th["sub"], "middle")
    if legend:
        for i, se in enumerate(series):
            d = f' stroke-dasharray="{se["dash"]}"' if se.get("dash") else ""
            ly = 108 + 22 * i
            s += f'<line x1="24" x2="50" y1="{ly}" y2="{ly}" stroke="{se["color"]}" stroke-width="4" stroke-linecap="round"{d}/>'
            s += tx(58, ly + 6, se["name"], 17, th["txt"])
    starts, ends = [], []
    for se in series:
        pts = se["pts"]; col = se["color"]
        coords = [(px(c), py(v), v, h) for c, v, h in pts]
        for (a, b, _, h1), (c2, d, _, h2) in zip(coords, coords[1:]):
            dash = se.get("dash") or ("8 7" if h2 else None)
            ds = f' stroke-dasharray="{dash}"' if dash else ""
            s += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c2:.1f}" y2="{d:.1f}" stroke="{col}" stroke-width="4.5" stroke-linecap="round"{ds}/>'
        for a, b, v, h in coords:
            if h: s += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="6" fill="{th["bg"]}" stroke="{col}" stroke-width="3.5"/>'
            else: s += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="6.5" fill="{col}"/>'
        starts.append((coords[0][0] - 12, coords[0][1] + 6, coords[0][2], col))
        ends.append((coords[-1][0] + 12, coords[-1][1] + 6, coords[-1][2], col))
    sy = spread([y for _, y, _, _ in starts], 22, y0 - 8, y1)
    ey = spread([y for _, y, _, _ in ends], 22, y0 - 8, y1)
    for (a, _, v, col), y in zip(starts, sy): s += tx(a, y, fmt(v), 20, col, "end", "700")
    for (a, _, v, col), y in zip(ends, ey): s += tx(a, y, fmt(v), 20, col, "start", "700")
    return s


def bar_chart(th, title, sub_lines, accent, groups, names, vals, foot):
    """groups: x labels; names: one per bar in a group (first = Kyojin); vals[k][g] = value."""
    s = frame(th, W, H, title, sub_lines)
    cols = [accent, th["n1"], th["n2"]][:len(names)]
    x0, x1, y0, y1 = 24, W - 24, 150, 312
    top = max(v for row in vals for v in row) * 1.1
    gw = (x1 - x0) / len(groups); bw = min(46, gw / (len(names) + 1.1))
    s += f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" stroke="{th["grid"]}" stroke-width="2"/>'
    lx = 24
    for n, c in zip(names, cols):
        s += f'<rect x="{lx}" y="102" width="16" height="16" rx="3" fill="{c}"/>' + tx(lx + 24, 115, n, 17, th["txt"])
        lx += 24 + 9.4 * len(n) + 22
    for g, gl in enumerate(groups):
        cx = x0 + gw * (g + 0.5); left = cx - bw * len(names) / 2
        for k in range(len(names)):
            v = vals[k][g]; h = (y1 - y0) * v / top; x = left + bw * k
            s += f'<rect x="{x + 2:.1f}" y="{y1 - h:.1f}" width="{bw - 4:.1f}" height="{h:.1f}" rx="3" fill="{cols[k]}"/>'
            s += tx(x + bw / 2, y1 - h - 7, fmt(v), 16, th["txt"] if k == 0 else th["sub"], "middle", "700" if k == 0 else "400")
        best = max(vals[k][g] for k in range(1, len(names)))
        s += tx(cx, y1 + 28, gl, 18, th["sub"], "middle")
        s += tx(cx, y1 + 52, f"{vals[0][g] / best:.1f}x", 20, accent, "middle", "700")
    return s + tx(24, H - 12, foot, 15, th["sub"])


def svg_end(s): return s + "</svg>"


def card(th, key, name, spec, big, unit, model, extra):
    acc = th[ACCENT[key]]
    w, h = 320, 236
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT}>'
         f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{th["bg"]}" stroke="{th["edge"]}"/>'
         f'<rect x="0.5" y="0.5" width="{w - 1}" height="8" rx="4" fill="{acc}"/>')
    s += tx(22, 48, name, 26, th["txt"], "start", "700") + tx(22, 74, spec, 16, th["sub"])
    s += tx(22, 134, big, 64, acc, "start", "700") + tx(22, 158, unit, 17, th["sub"])
    s += tx(22, 190, model, 18, th["txt"], "start", "700") + tx(22, 214, extra, 17, th["sub"])
    return svg_end(s)


def write(name, fn):
    os.makedirs(OUT, exist_ok=True)
    for tn, th in THEMES.items():
        with open(os.path.join(OUT, f"{name}-{tn}.svg"), "w") as f: f.write(svg_end(fn(th)))


# ---------- data ----------
M = F["models"]; SHORT = {"qwen": "Flash-Next", "glm": "GLM-5.3", "mimo": "MiMo"}
sv, sd = F["version"], F["date"]
STRIX_SUB = [f'{F["machine"].split(",")[0]}, 128 GB, tok/s', f"Kyojin {sv}, {sd}"]


FULL = {"qwen": "Qwen3.8-Flash-Next", "glm": "GLM-5.3-Flash"}


def featured():
    return [next(m for m in M if m["id"] == i) for i in ("glm", "qwen")]


def strix_series(th, key):
    out = []
    for m in featured():
        pts = [(int(c), v, False) for c, v in m[key].items() if v]
        if not pts: continue
        col = {"qwen": th["amd"], "glm": th["n1"]}[m["id"]]
        out.append(dict(name=f'{FULL[m["id"]]} (Strix Halo)', color=col, pts=sorted(pts)))
    return out


def strix_xs(key):
    return sorted({int(c) for m in featured() for c, v in m[key].items() if v})


def rtx_series(th, key, which):
    d = P["rtx"][which]; base = d["points"][0]["version"]
    pts = [(p["ctx"], p[key], p["version"] != base) for p in d["points"]]
    return [dict(name=f'{d["name"]} (RTX 3090)', color=th["nv"], pts=pts)], [p["ctx"] for p in d["points"]], d


def mac_series(th, key):
    m = P["mac"]
    return [dict(name=f'{m["name"]} (Mac Studio)', color=th["apple"], pts=[(p["ctx"], p[key], False) for p in m["points"]])], [p["ctx"] for p in m["points"]]


def make_all():
    # Strix Halo
    for key, label in (("decode", "Decode"), ("prefill", "Prefill")):
        sub = STRIX_SUB if key == "decode" else STRIX_SUB
        def fn(th, key=key, label=label):
            return line_chart(th, f"{label}: GLM-5.3 and Flash-Next", sub, "amd", strix_series(th, key), strix_xs(key), legend=True)
        write(f"{key}-strix", fn)
    # RTX: Qwen3.8-27B and Flint, drawn in one chart per metric would mix scales, so Qwen3.8-27B carries the RTX charts
    for key, label in (("decode", "Decode"), ("prefill", "Prefill")):
        def fn(th, key=key, label=label):
            ser, xs, d = rtx_series(th, key, "qwen27")
            sub = [f'{P["rtx"]["machine"]}, tok/s', f'Kyojin {d["points"][0]["version"]}, {d["points"][0]["date"]}']
            return line_chart(th, f"{label}: {d['name']}", sub, "nv", ser, xs, legend=True)
        write(f"{key}-rtx", fn)
    for key, label in (("decode", "Decode"), ("prefill", "Prefill")):
        def fn(th, key=key, label=label):
            ser, xs, d = rtx_series(th, key, "flint")
            sub = [f'{P["rtx"]["machine"]}, tok/s', f'Kyojin {d["points"][0]["version"]}, {d["points"][0]["date"]}']
            return line_chart(th, f"{label}: {d['name']}", sub, "nv", ser, xs, legend=True)
        write(f"{key}-rtx-flint", fn)
    # Mac
    for key, label in (("decode", "Decode"), ("prefill", "Prefill")):
        def fn(th, key=key, label=label):
            ser, xs = mac_series(th, key); m = P["mac"]
            return line_chart(th, f"{label}: {m['name']}", [f'{m["machine"]}, tok/s', f'Kyojin {m["version"]}, {m["date"]}'], "apple", ser, xs, legend=True)
        write(f"{key}-mac", fn)
    # Duels
    R = F["rivals"]; q = M[0]
    ctx3 = [8192, 32768, 131072]
    def duel_rtx(which, title):
        d = P["rtx"][which]
        def fn(th):
            sub = [f'{P["rtx"]["machine"]}, decode, no speculation' if which == "duel_qwen27" else f'{P["rtx"]["machine"]}, decode',
                   f'Kyojin {P["rtx"]["duel_version"]}, {P["rtx"]["duel_date"]}, same prompts']
            return bar_chart(th, title, sub, th["nv"], [ctx_label(c) for c in d["ctx"]], ["Kyojin", d["rival"].split(",")[0]],
                             [d["kyojin_decode"], d["rival_decode"]], d["rival"] + ". Multiple of the rival.")
        return fn
    write("duel-rtx-qwen27", duel_rtx("duel_qwen27", "Qwen3.8-27B vs llama.cpp"))
    write("duel-rtx-flint", duel_rtx("duel_flint", "Flint V1 vs llama.cpp"))
    def duel_strix_decode(th):
        names = ["Kyojin", R["strata"]["name"], R["gufo"]["name"]]
        vals = [[q["decode"][str(c)] for c in ctx3], [R["strata"]["decode_measured"][str(c)] for c in ctx3], [R["gufo"]["decode_measured"][str(c)] for c in ctx3]]
        sub = [f'{F["machine"].split(",")[0]}, 128 GB, Qwen3.8-Flash-Next', f"Kyojin {sv}, {sd}, speculative decode"]
        return bar_chart(th, "Flash-Next decode vs others", sub, th["amd"], [ctx_label(c) for c in ctx3], names, vals, "Rivals measured by us, same machine. Multiple of the best rival.")
    def duel_strix_prefill(th):
        names = ["Kyojin", R["gufo"]["name"]]
        vals = [[q["prefill"][str(c)] for c in ctx3], [R["gufo"]["prefill_measured"][str(c)] for c in ctx3]]
        sub = [f'{F["machine"].split(",")[0]}, 128 GB, Qwen3.8-Flash-Next', f"Kyojin {sv}, {sd}"]
        return bar_chart(th, "Flash-Next prefill vs others", sub, th["amd"], [ctx_label(c) for c in ctx3], names, vals, "Gufo measured by us, same machine. Multiple of the rival.")
    write("duel-strix-decode", duel_strix_decode)
    write("duel-strix-prefill", duel_strix_prefill)
    # Cards
    qd = {int(c): v for c, v in q["decode"].items() if v}; qlast = max(qd)
    r = P["rtx"]["qwen27"]["points"]; mp = P["mac"]["points"]
    g = next(m for m in M if m["id"] == "glm"); gd = {int(c): v for c, v in g["decode"].items() if v}
    cards = (("strix", "Strix Halo", "Ryzen AI Max+ 395, 128 GB", f"{gd[8192]:.0f} / {qd[8192]:.0f}", "tok/s decode at 8K", "GLM-5.3-Flash / Flash-Next", f"GLM to {ctx_label(max(gd))} of context"),
             ("rtx", "RTX 3090", "NVIDIA, 24 GB", fmt(r[0]["decode"]), "tok/s decode at 5K", "Qwen3.8-27B", f'{fmt(r[3]["decode"])} at {ctx_label(r[3]["ctx"])}'),
             ("mac", "Mac Studio", "M5 Ultra, 256 GB", fmt(mp[0]["decode"]), "tok/s decode at 8K", P["mac"]["name"], f'{fmt(mp[-1]["decode"])} at {ctx_label(mp[-1]["ctx"])}'))
    os.makedirs(OUT, exist_ok=True)
    for key, *args in cards:
        for tn, th in THEMES.items():
            open(os.path.join(OUT, f"card-{key}-{tn}.svg"), "w").write(card(th, key, *args))


if __name__ == "__main__":
    make_all()
    print("wrote", OUT)
