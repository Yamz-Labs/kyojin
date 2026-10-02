# per-kernel stats of the census-free prefill@4K window (glm_base.py prof -> prof_prefill_window) from a rocprofv3 kernel_trace.csv
# usage: prof_prefill_topk.py <run.rocprof dir> <run.json> [top N]
import csv, json, sys, glob, collections
d, js = sys.argv[1], sys.argv[2]
W = json.load(open(js))["prof_prefill_window"]
f = glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True)[0]
rows = list(csv.DictReader(open(f)))
st = [int(r["Start_Timestamp"]) for r in rows]
lo, hi = min(st), max(st)
clk = next(c for c in ("boot", "mono") if lo <= W["start"][c] <= hi)
a, b = W["start"][clk], W["end"][clk]
agg = collections.defaultdict(lambda: [0, 0])
busy = 0; first = None; last = None
for r in rows:
    s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
    if s < a or e > b: continue
    n = r["Kernel_Name"].split("(")[0][:110]
    agg[n][0] += e - s; agg[n][1] += 1; busy += e - s
    first = s if first is None else min(first, s); last = e if last is None else max(last, e)
wall = (b - a) / 1e6
print(f"clock={clk} window={wall:.1f} ms  gpu-busy={busy/1e6:.1f} ms  first->last={(last-first)/1e6:.1f} ms  ttft={W['ttft']}")
tot = busy
for n, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:int(sys.argv[3]) if len(sys.argv) > 3 else 25]:
    print(f"{t/1e6:9.1f} ms {c:6d} calls {100*t/tot:5.1f}%  {t/1e3/c:8.1f} us/call  {n}")
