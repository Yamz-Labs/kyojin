#!/usr/bin/env python3
"""Fill the {{TABLE}} field of README.md and doc/benchmarks.md from doc/figures.json.
Templates: README.tmpl.md, doc/benchmarks.tmpl.md. usage: python3 tools/make_readme_table.py"""
import json, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(os.path.join(root, "doc", "figures.json"))); HP, HC = F["headline_prompt"], F["headline_prefill_ctx"]
def f(v): return "n/a" if v is None else (f"{v:.1f}" if v < 100 else f"{v:.0f}")
rows = "".join(f'<tr><td><b>{m["name"]}</b></td><td align="right">{m["pack_gb"]:g} GB</td><td align="right">{m["table"]["prefill"]}</td><td align="right"><b>{m["table"]["decode"]}</b></td></tr>\n' for m in F["models"])
table = ('<table align="center">\n<tr><th align="left">Model</th><th>Pack</th><th>Prefill</th><th>Speculative decode</th></tr>\n' + rows + '</table>')
CT = sorted({int(c) for m in F["models"] if m["id"] != "glm" for c in m["prefill"]})
def row(m): return f'| {m["name"]} | {m["pack_gb"]:g} GB | ' + " | ".join(f(m["prefill"].get(str(c))) if str(c) in m["prefill"] else "-" for c in CT) + f' | {f(m["spec"]["prose"])} | {f(m["spec"]["chat"])} | {f(m["spec"]["code"])} |\n'
full = "| Model | Pack | " + " | ".join(f"Prefill {c//1024}K" for c in CT) + " | Spec prose | Spec chat | Spec code |\n|" + "---|" * (len(CT) + 5) + "\n" + "".join(row(m) for m in F["models"] if m["id"] != "glm")
for tmpl, dst, tb in (("README.tmpl.md", "README.md", table), ("doc/benchmarks.tmpl.md", "doc/benchmarks.md", full)):
    s = open(os.path.join(root, tmpl)).read().replace("{{TABLE}}", tb)
    open(os.path.join(root, dst), "w").write(s)
print("ok")
