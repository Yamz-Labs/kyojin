#!/usr/bin/env python
"""Join a depth_profile.py --kernels JSON with a rocprofv3 kernel trace CSV: adds "kernels" to every window point.
  kernels.py PROFILE.json KERNEL_TRACE.csv        (rewrites PROFILE.json)"""
import collections, csv, json, re, sys

pj, tr = sys.argv[1], sys.argv[2]
P = json.load(open(pj))
rows = [(int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(open(tr))]


def short(n):
    return re.sub(r"\(.*$", "", n.replace("(anonymous namespace)::", ""))[:110]


for d, pt in P["points"].items():
    for key, v in pt.items():
        w = v.get("window") if isinstance(v, dict) else None
        if not w:
            continue
        best = []
        for clk in ("mono", "boot"):                 # rocprofv3 timestamps follow one of the two clocks
            sel = [r for r in rows if w["start"][clk] <= r[0] <= w["end"][clk]]
            if len(sel) > len(best):
                best = sel
        tab, cnt = collections.Counter(), collections.Counter()
        for s, e, n in best:
            tab[short(n)] += (e - s) / 1e3 / w["steps"]
            cnt[short(n)] += 1 / w["steps"]
        v["kernels"] = {"busy_us": sum(tab.values()), "kernels_per_step": sum(cnt.values()), "wall_us": v["ms"] * 1e3 * (v.get("chunk", 1) and 1),
                        "top": [[k, round(u, 1), round(cnt[k], 2)] for k, u in tab.most_common(60)]}
json.dump(P, open(pj, "w"), indent=1)
print("joined", sum(1 for pt in P["points"].values() for v in pt.values() if isinstance(v, dict) and "kernels" in v), "windows")
