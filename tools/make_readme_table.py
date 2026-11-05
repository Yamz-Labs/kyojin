#!/usr/bin/env python3
"""Fill the {{TABLE}}, {{LONG}}, {{DECODE}} and {{LADDER}} fields of doc/strix.md and doc/benchmarks.md from doc/figures.json.
Templates: doc/strix.tmpl.md, doc/benchmarks.tmpl.md. README.md is written by hand. usage: python3 tools/make_readme_table.py"""
import json, os, re
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(os.path.join(root, "doc", "figures.json"))); HP, HC = F["headline_prompt"], F["headline_prefill_ctx"]
def f(v): return "n/a" if v is None else (f"{v:.1f}" if v < 100 else f"{v:.0f}")
def ck(c): return "500K" if int(c) == 499712 else f"{int(c) // 1024}K"   # 499712 = the 500K point
def b(m, v): return v if m.get("earlier") else f"<b>{v}</b>"
def label(m): return f'{m["name"]} <sub>({m["earlier"]})</sub>' if m.get("earlier") else f'<b>{m["name"]}</b>'
rows = "".join(f'<tr><td>{label(m)}</td><td align="right">{m["pack_gb"]:g} GB</td><td align="right">{m["table"]["prefill"]}</td><td align="right">{b(m, m["table"]["decode"])}</td></tr>\n' for m in F["models"])
rows += "".join(r + "\n" for r in F.get("readme_extra_rows", []))   # rows with no chart curves (DeepSeek-V4.1-Flash)
table = ('<table align="center">\n<tr><th align="left">Model</th><th>Pack</th><th>Prefill</th><th>Speculative decode</th></tr>\n' + rows + '</table>')
CT = sorted({int(c) for m in F["models"] for c in m["prefill"]})
in_main = F["models"]
def row(m): return f'| {m["name"]} | {m["pack_gb"]:g} GB | ' + " | ".join(f(m["prefill"].get(str(c))) if str(c) in m["prefill"] else "-" for c in CT) + f' | {f(m["spec"]["prose"])} | {f(m["spec"]["chat"])} | {f(m["spec"]["code"])} |\n'
full = "| Model | Pack | " + " | ".join(f"Prefill {ck(c)}" for c in CT) + " | Spec prose | Spec chat | Spec code |\n|" + "---|" * (len(CT) + 5) + "\n" + "".join(row(m) for m in in_main)

dec_block = ("### Speculative decode by context\n\n![Speculative decode against context, three models](img/decode_models.svg)\n\n| Model | " + " | ".join(ck(c) for c in CT) + " |\n|" + "---|" * (len(CT) + 1) + "\n"
             + "".join(f'| {m["name"]} | ' + " | ".join(f(m["decode"][str(c)]) if m["decode"].get(str(c)) else "-" for c in CT) + " |\n" for m in F["models"]))

def long_block():
    """README table of the longest points (128K, 256K, 500K): prefill / decode, from figures.json."""
    pts = [131072, 262144, 499712]
    ms = [m for m in F["models"] if all(str(c) in m["prefill"] and m["decode"].get(str(c)) for c in pts)]
    if not ms: return ""
    # the release the figures were measured on (figures.json), not the version of the tree the README is built from
    vf = os.path.join(root, "exllamav3", "version.py")
    ver = F.get("version") or (re.search(r'"([^"]+)"', open(vf).read()).group(1) if os.path.exists(vf) else "")
    t = '<table align="center">\n<tr><th align="left">Model</th>' + "".join(f"<th>{ck(c)}</th>" for c in pts) + "</tr>\n"
    for m in ms:
        t += f'<tr><td>{label(m)}</td>' + "".join(f'<td align="right">{f(m["prefill"][str(c)])} / {f(m["decode"][str(c)])}</td>' for c in pts) + "</tr>\n"
    return ("<p align=\"center\"><b>Long context: up to 500K tokens</b></p>\n\n" + t + "</table>\n\n"
            f'<p align="center"><sub>Prefill / speculative decode, tokens per second, Kyojin {ver} on one Ryzen AI Max+ 395 with 128 GB. '
            'One real document read as a growing prompt (500K = 499,712 tokens); prefill as reported by the server. '
            'Both models recalled two facts hidden at 40 % and 85 % of a ~500K-token prompt (needle test, Kyojin 1.5.3, 15/10).</sub></p>\n\n'
            "500K needs <code>-c 524288</code> at launch. Qwen also needs <code>--cache-bits 8</code>: past 262K it uses the 8-bit cache.")

reps = F.get("method_ladder_reps", 1)
ladder = "one request per depth" if reps == 1 else f"median of {reps} requests per depth"
for tmpl, dst, tb in (("doc/strix.tmpl.md", "doc/strix.md", table), ("doc/benchmarks.tmpl.md", "doc/benchmarks.md", full)):
    s = open(os.path.join(root, tmpl)).read().replace("{{TABLE}}", tb).replace("{{LONG}}", long_block()).replace("{{DECODE}}", dec_block).replace("{{LADDER}}", ladder)
    open(os.path.join(root, dst), "w").write(s)
print("ok")
