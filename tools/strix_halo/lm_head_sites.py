#!/usr/bin/env python
"""Attribute lm_head GEMV launches in /tmp/gemm_trace.json (from gemm_sites.py) to call sites."""
import json, collections, sys
tr = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gemm_trace.json"))["traceEvents"]
ks = [e for e in tr if e.get("cat") == "kernel" and "exl3_gemv_kernel" in e["name"] and e["args"].get("grid", [0])[0] == 3880]
tot = sum(e["dur"] for e in tr if e.get("cat") == "kernel")
d = sum(e["dur"] for e in ks)
print(f"{len(ks)} lm_head launches, {d/1000:.1f} ms = {100*d/tot:.1f}% of device; per call {d/len(ks):.0f} us")
print(collections.Counter(e["name"][:48] for e in ks))
by_corr = collections.defaultdict(list)
for e in tr:
    c = e.get("args", {}).get("correlation")
    if c is not None: by_corr[c].append(e)
pyf_by_tid = collections.defaultdict(list)
for e in tr:
    if e.get("cat") == "python_function" and "exllamav3/" in e["name"]:
        pyf_by_tid[e["tid"]].append(e)
def site(k):
    rt = [e for e in by_corr[k["args"]["correlation"]] if e is not k]
    if not rt: return "?"
    r = rt[0]; ts = r["ts"]; tid = r["tid"]
    fr = [e for e in pyf_by_tid[tid] if e["ts"] <= ts <= e["ts"] + e.get("dur", 0)]
    fr.sort(key=lambda e: e["ts"])
    return " <- ".join(e["name"].split("exllamav3/")[-1].split("(")[0] + ":" + e["name"].split(": ")[-1] for e in fr[-7:-2])
c = collections.Counter(site(k) for k in ks)
for s, n in c.most_common(): print(n, s)
