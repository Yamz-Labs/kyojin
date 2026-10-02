# Per-kernel decode cost at two depths from a depth_curve.py prof run under rocprofv3.
# usage: depth_prof_table.py <prof.json> <kernel_trace.csv> [top_n]
# Prints us/token per kernel name at each window's depth, ranked by growth (deep - shallow).
import csv, json, re, sys, collections

pj, tr = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 15
wins = json.load(open(pj))["prof"]
rows = [(int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(open(tr))]


def short(n):
    n = re.sub(r"\(.*$", "", n.replace("(anonymous namespace)::", ""))
    return n[:70]


per = []
for w in wins:
    best = None
    for clk in ("mono", "boot"):
        a, b = w["start"][clk], w["end"][clk]
        sel = [r for r in rows if a <= r[0] <= b]
        if best is None or len(sel) > len(best):
            best = sel
    c = collections.Counter()
    for s, e, n in best:
        c[short(n)] += (e - s) / 1e3 / w["steps"]
    per.append((w, c, len(best)))
    print(f"depth {w['depth']}: {len(best)/w['steps']:.0f} kernels/tok, GPU busy {sum(c.values())/1e3:.2f} ms/tok, wall {w['ms_per_tok']} ms/tok")

(w0, c0, _), (w1, c1, _) = per[0], per[-1]
names = set(c0) | set(c1)
g = sorted(names, key=lambda n: c1[n] - c0[n], reverse=True)
print(f"| kernel | us/tok @{w0['depth']} | us/tok @{w1['depth']} | growth us |")
print("|---|---|---|---|")
for n in g[:top]:
    print(f"| {n} | {c0[n]:.1f} | {c1[n]:.1f} | {c1[n]-c0[n]:+.1f} |")
