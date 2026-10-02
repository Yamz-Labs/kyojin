#!/usr/bin/env python3
"""
Per-kernel attribution of ONE served MTP round (n1f2) at two context depths, from a
depth_curve_mtp.py prof run under rocprofv3.

  usage: depth_mtp_table.py <prof.json> <run.rocprof dir|kernel_trace.csv> [top_n]

Windows come from the json (prof[*].start/end in monotonic ns and boottime ns; the trace clock
that selects more kernels wins, same as depth_prof_table.py). Output: ms/round wall from the
untraced timing, GPU busy, per-family us/round at each depth ranked by growth, then the top
growing kernels. Families are the prefill-tuned regexes in prof_families.py, reused as they are;
the raw kernel table below them is unfiltered, so nothing hides behind a regex.
"""
import collections, csv, glob, json, os, re, sys

pj, prof_dir = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 12

# prof_families.py runs main() at import, so take its FAMILIES table by ast instead of importing
import ast
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prof_families.py")).read()
FAMILIES = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FAMILIES")

J = json.load(open(pj))
wins = J["prof"]
if os.path.isdir(prof_dir):
    f = glob.glob(f"{prof_dir}/**/*kernel_trace.csv", recursive=True)[0]
else:
    f = prof_dir
rows = []
with open(f) as fh:
    for r in csv.DictReader(fh):
        try:
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
        except (KeyError, ValueError):
            pass


def short(n):
    return re.sub(r"\(.*$", "", n.replace("(anonymous namespace)::", ""))[:78]


def family(n):
    for name, pat in FAMILIES:
        if re.search(pat, n):
            return name
    return "norms / elementwise / copies"


per = []
for w in wins:
    best = None
    for clk in ("mono", "boot"):
        a, b = w["start"][clk], w["end"][clk]
        sel = [r for r in rows if a <= r[0] <= b]
        if best is None or len(sel) > len(best):
            best = sel
    if not best:
        print(f"depth {w['depth']}: NO kernels matched the window. trace range "
              f"[{min((r[0] for r in rows), default=0)}, {max((r[0] for r in rows), default=0)}] ns, "
              f"window mono [{w['start']['mono']}, {w['end']['mono']}] boot "
              f"[{w['start']['boot']}, {w['end']['boot']}]")
    nr = w["rounds"]
    c = collections.Counter()
    for s, e, n in best:
        c[short(n)] += (e - s) / 1e3 / nr          # us per round
    fam = collections.Counter()
    for n, us in c.items():
        fam[family(n)] += us
    per.append((w, c, fam, len(best) / nr))
    print(f"depth {w['depth']}: {len(best)/nr:.0f} kernels/round, GPU busy {sum(c.values())/1e3:.2f} ms/round, "
          f"wall {w['ms_per_round']:.2f} ms/round, tok/round {w['tok_per_round']}, accept {w['accept']}, "
          f"stash {w['stashed_state_mb']:.0f} MB")

(w0, c0, f0, k0), (w1, c1, f1, k1) = per[0], per[-1]
d0, d1 = w0["depth"], w1["depth"]
print(f"\nwall {w0['ms_per_round']:.2f} -> {w1['ms_per_round']:.2f} ms/round "
      f"({100*(w1['ms_per_round']/w0['ms_per_round']-1):+.1f} %); GPU busy "
      f"{sum(f0.values())/1e3:.2f} -> {sum(f1.values())/1e3:.2f} ms/round; kernels/round {k0:.0f} -> {k1:.0f}")
print(f"\n| family | us/round @{d0} | us/round @{d1} | growth us | share of growth |")
print("|---|---|---|---|---|")
tot = sum(max(v1 - v0, 0) for v0, v1 in zip(f0.values(), f1.values())) or 1.0
for name in sorted(set(f0) | set(f1), key=lambda n: f1.get(n, 0) - f0.get(n, 0), reverse=True):
    a, b = f0.get(name, 0.0), f1.get(name, 0.0)
    print(f"| {name} | {a:.0f} | {b:.0f} | {b-a:+.0f} | {100*max(b-a,0)/tot:.0f} % |")

names = set(c0) | set(c1)
g = sorted(names, key=lambda n: c1[n] - c0[n], reverse=True)
print(f"\n| kernel | us/round @{d0} | us/round @{d1} | growth us |")
print("|---|---|---|---|")
for n in g[:top]:
    print(f"| {n} | {c0[n]:.1f} | {c1[n]:.1f} | {c1[n]-c0[n]:+.1f} |")
top_g = g[0] if g and c1[g[0]] > c0[g[0]] else None
print(f"\ntop growing kernel: {top_g} {c0[top_g]:.1f} -> {c1[top_g]:.1f} us/round "
      f"({c1[top_g]-c0[top_g]:+.1f})" if top_g else "\nno kernel grows with depth")
