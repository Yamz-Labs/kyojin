"""analyze_trace.py KERNEL_TRACE.csv [--steps N] [--layer-dump] : cut the measured window (MARK file or tail), group kernels.
usage: analyze_trace.py CSV LOG LAYERS"""
import csv, sys, re, collections
csvf, log, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
mk = [l for l in open(log) if l.startswith("MARK")][-1].split(); t0, t1, n = int(mk[1]), int(mk[2]), int(mk[4])
rows = []
for r in csv.DictReader(open(csvf)):
    s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
    rows.append((s, e, r["Kernel_Name"]))
rows.sort()
sel = [r for r in rows if t0 - 2_000_000 <= r[0] <= t1 + 2_000_000]
if not sel:
    print("no kernels in window; clock mismatch? first", rows[0][0], "t0", t0); sys.exit(1)
print(f"{len(sel)} kernels in window, {len(sel)/n:.0f}/step, window {(t1-t0)/n/1e6:.2f} ms/step")
busy = sum(e - s for s, e, _ in sel) / n / 1e3
span = (sel[-1][1] - sel[0][0]) / n / 1e3
print(f"GPU busy {busy:.0f} us/step, span {span:.0f} us/step, gaps {span-busy:.0f} us/step")
def short(nm): return re.sub(r"\(.*", "", nm)[:70]
agg = collections.defaultdict(lambda: [0, 0])
for s, e, nm in sel:
    a = agg[short(nm)]; a[0] += e - s; a[1] += 1
print("by kernel (us/step, count/step, avg us)")
for k, (d, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:45]:
    print(f"  {d/n/1e3:8.1f} {c/n:6.1f} {d/c/1e3:7.1f}  {k}")
# sequence of the last step, middle layer region
per = len(sel) // n
last = sel[-per:]
print("last step sequence: idx start_us dur_us gap_us name")
prev = last[0][0]
for i, (s, e, nm) in enumerate(last):
    print(f"{i:4d} {(s-last[0][0])/1e3:8.1f} {(e-s)/1e3:6.1f} {(s-prev)/1e3:6.1f} {short(nm)}")
    prev = e
