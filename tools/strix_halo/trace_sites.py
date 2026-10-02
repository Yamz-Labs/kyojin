#!/usr/bin/env python
"""Attribute kernels in a chrome trace to python call sites via the runtime launch's correlation id.
Usage: trace_sites.py /tmp/gemm_trace.json PATTERN [PATTERN2 ...]
"""
import json, collections, sys
tr = json.load(open(sys.argv[1]))["traceEvents"]
pats = sys.argv[2:] or ["Cijk"]
by_corr = collections.defaultdict(list)
for e in tr:
    c = e.get("args", {}).get("correlation")
    if c is not None: by_corr[c].append(e)
pyf_by_tid = collections.defaultdict(list)
for e in tr:
    if e.get("cat") == "python_function":
        pyf_by_tid[e["tid"]].append(e)
ops_by_tid = collections.defaultdict(list)
for e in tr:
    if e.get("cat") == "cpu_op":
        ops_by_tid[e["tid"]].append(e)
def site(k):
    rt = [e for e in by_corr[k["args"]["correlation"]] if e is not k and e.get("cat") != "kernel"]
    if not rt: return ("?", "")
    r = rt[0]; ts = r["ts"]; tid = r["tid"]
    fr = [e for e in pyf_by_tid[tid] if e["ts"] <= ts <= e["ts"] + e.get("dur", 0) and "exllamav3/" in e["name"]]
    fr.sort(key=lambda e: e["ts"])
    st = " <- ".join(e["name"].split("exllamav3/")[-1].split("(")[0] + ":" + e["name"].split(": ")[-1] for e in fr[-6:-1])
    ops = [e for e in ops_by_tid[tid] if e["ts"] <= ts <= e["ts"] + e.get("dur", 0)]
    ops.sort(key=lambda e: e["ts"])
    leaf = ops[-1] if ops else None
    shape = (leaf["name"] + " " + str(leaf.get("args", {}).get("Input Dims", ""))[:90]) if leaf else ""
    return (st, shape)
tot = sum(e["dur"] for e in tr if e.get("cat") == "kernel")
for pat in pats:
    ks = [e for e in tr if e.get("cat") == "kernel" and pat in e["name"]]
    d = sum(e["dur"] for e in ks)
    print(f"\n### {pat}: {len(ks)} launches, {d/1000:.0f} ms = {100*d/tot:.1f}% of device")
    agg = collections.defaultdict(lambda: [0, 0.0, set()])
    for k in ks:
        st, shape = site(k)
        a = agg[st]; a[0] += 1; a[1] += k["dur"]; a[2].add(shape)
    for st, (n, dd, shapes) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:6]:
        print(f"  {n:5d}x {dd/1000:8.0f} ms  {dd/n:8.0f} us/call")
        print(f"        {st}")
        for sh in sorted(shapes)[:3]: print(f"        {sh}")
