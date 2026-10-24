#!/usr/bin/env python3
"""Fill the {{TABLE}}, {{LONG}}, {{DECODE}}, {{RIVALS}} and {{LADDER}} fields of README.md and doc/benchmarks.md from doc/figures.json.
Templates: README.tmpl.md, doc/benchmarks.tmpl.md. usage: python3 tools/make_readme_table.py"""
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

def rivals():
    R = F.get("rivals")
    if not R: return "Comparison with other engines: not in this version of the figures file.\n"
    q = F["models"][0]; S, G = R["strata"], R["gufo"]; Q = F["quality"]
    cols = [8192, 32768, 65536, 131072, 262144]
    def cell(d, c): return f(d[str(c)]) if d.get(str(c)) else "-"
    head = "| tok/s | " + " | ".join(ck(c) for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    def line(name, d): return f"| {name} | " + " | ".join(cell(d, c) for c in cols) + " |\n"
    pre = head + line(f"Kyojin", q["prefill"]) + line(f"{G['name']} {G['version']}, measured by us", G["prefill_measured"])
    dec = head + line("Kyojin", q["decode"]) + line(f"{G['name']}, measured by us", G["decode_measured"]) + line(f"{S['name']}, measured by us", S["decode_measured"])
    ind = S["independent"]; ip = S["prefill_independent"]
    out = ("Same machine, one method for every engine we ran: non-thinking chat, an essay request on cold prompts of 8K to 256K tokens, 256 greedy tokens, speculation on, engine-reported prefill. Rival figures are the ones we measured on our machine, with " + f"{S['name']} {S['version']} ({S['pack']} pack) and {G['name']} {G['version']} ({G['pack']} pack)" + ", plus the independent test cited below. Each project also publishes its own figures, on its own machine and with its own method: " + f"[{S['name']}]({S['source_url']}), [{G['name']}]({G['source_url']})" + ".\n\n"
           "### Prefill\n\n![Prefill against context](img/prefill_context.svg)\n\n" + pre + "\n"
           f"{S['name']} prefill, independent measurement by {ind['lab']} ([report]({ind['url']}), {ind['date']}; cold prompts of {ind['sizes']}; a different machine and different prompts; the lab states that it did not reproduce the publisher's exact workload and that the cause of the gap is not established): " + " / ".join(f(ip[k]) for k in ("32768", "65536", "120000")) + " tok/s at 32K / 64K / 120K.\n\n"
           "### Speculative decode\n\n![Speculative decode against context](img/decode_context.svg)\n\n" + dec + "\n"
           f"Card prompts (chat / prose / code): Kyojin {f(q['spec']['chat'])} / {f(q['spec']['prose'])} / {f(q['spec']['code'])}; {G['name']} {f(G['card_measured']['chat'])} / {f(G['card_measured']['prose'])} / {f(G['card_measured']['code'])}; {S['name']} {f(S['card_measured']['chat'])} / {f(S['card_measured']['prose'])} / {f(S['card_measured']['code'])} (measured by us, client-timed). {G['name']} could not take the 256K prompt: " + G["note_256K"] + ".\n\n"
           "### Fidelity to the original model\n\n"
           f"Top-1 agreement and KL divergence against the official FP8 release: {Q['texts']} public neutral texts, {Q['positions']} scored positions, fixed-seed points, one scorer for all three engines, row-cluster bootstrap. The rival engines were patched only to write out their top-20 log-probabilities; their GGUF packs come from the BF16 weights. Run-to-run variation is the KL between two runs of the same engine on the same points.\n\n"
           "| Engine | Pack | Top-1 | KL | KL run-to-run |\n|---|---|---|---|---|\n"
           + "".join(f"| {n} | {Q[k]['pack_gb']:g} GB | {Q[k]['top1']:.2f} % | {Q[k]['kl']:.4f} | {Q[k]['kl_variation']:.4f} |\n" for n, k in (("Kyojin", "kyojin"), (S["name"], "strata"), (G["name"], "gufo")))
           + f"\nSources: [{S['name']}]({S['source_url']}) (v{S['version'].lstrip('v')}, seen {S['seen']}), [{G['name']}]({G['source_url']}) ({G['version']}, seen {G['seen']}). Scripts, prompts and raw outputs are in the repository.\n")
    return out

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
for tmpl, dst, tb in (("README.tmpl.md", "README.md", table), ("doc/benchmarks.tmpl.md", "doc/benchmarks.md", full)):
    s = open(os.path.join(root, tmpl)).read().replace("{{TABLE}}", tb).replace("{{LONG}}", long_block()).replace("{{DECODE}}", dec_block).replace("{{RIVALS}}", rivals()).replace("{{LADDER}}", ladder)
    open(os.path.join(root, dst), "w").write(s)
print("ok")
