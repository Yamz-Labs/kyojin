"""sections.py KERNEL_TRACE.csv PROF_LOG LAYERS : per-section decode table (us/layer, kernels/layer) from a rocprofv3 trace.
Each kernel is charged its duration plus the idle gap before it (launch gaps land on the kernel that waited), so the buckets add
up to the step time. Sections: hc, attention (EXL3 linears / HIP kernels / torch glue), router, routed experts, shared expert +
tail, embed/head."""
import csv, re, sys, collections
csvf, log, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
mk = [l for l in open(log) if l.startswith("MARK")][-1].split(); t0, t1, n = int(mk[1]), int(mk[2]), int(mk[4])
rows = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), re.sub(r"\(.*", "", r["Kernel_Name"])) for r in csv.DictReader(open(csvf)))
sel = [r for r in rows if t0 - 2e6 <= r[0] <= t1 + 2e6]
per = len(sel) // n
steps = [sel[i * per:(i + 1) * per] for i in range(n)]
agg = collections.defaultdict(lambda: [0.0, 0.0])      # bucket -> [busy us, kernels]
isexl = lambda nm: "exl3" in nm or nm.startswith("had_hf")
def ffn_buckets(names):
    it = max((i for i, nm in enumerate(names) if "router_topk" in nm), default=-1)
    lv = max((i for i, nm in enumerate(names) if "moe_valu" in nm or "ds41m_valu" in nm), default=-1)
    end = lv
    for j in range(lv + 1, len(names)):
        if isexl(names[j]) and "had_hf" not in names[j]: break
        end = j
    out = []
    for i, nm in enumerate(names):
        if nm.startswith("ds41_hc"): out.append("hc")
        elif "router" in nm: out.append("router")
        elif it < i <= end: out.append("routed experts")
        else: out.append("shared expert + tail")
    return out
# walk the kernels in start order; ds41_hc_cn marks section changes: per step 2 per layer + the final norm
segs, cur, c = [], [], 0
for s_, e_, nm in sel:
    if e_ - s_ > 1_000_000:                  # the lm head GEMM (its recorded start is unreliable): own bucket, not part of a section
        agg["lm head GEMM"][0] += e_ - s_; agg["lm head GEMM"][1] += 1
        continue
    if nm == "ds41_hc_cn":
        segs.append(cur); cur = []
    cur.append((s_, e_, nm))
segs.append(cur)
# segs[0] = kernels before the first cn of the window (embed + pre of the first step / tail of an earlier step)
CN = 2 * L + 1
for k, seg in enumerate(segs):
    if k == 0:
        for s_, e_, nm in seg: agg["embed + pre"][0] += (e_ - s_); agg["embed + pre"][1] += 1
        continue
    pos = (k - 1) % CN                       # cn index inside its step
    names = [x[2] for x in seg]
    if pos == 2 * L:
        bs = ["final norm + head"] * len(seg)
    elif pos % 2 == 0:
        bs = ["hc" if nm.startswith("ds41_hc") else "attention: EXL3 linears" if isexl(nm) else "attention: HIP kernels" if nm.startswith("ds41_") else "attention: torch glue" for nm in names]
    else:
        bs = ffn_buckets(names)
    for (s_, e_, nm), b in zip(seg, bs):
        agg[b][0] += (e_ - s_) / 1e3 if False else (e_ - s_); agg[b][1] += 1
wall = (t1 - t0) / n / 1e3
busy = sum(v[0] for v in agg.values()) / n / 1e3
print(f"wall {wall:.0f} us/step, GPU busy {busy:.0f} us/step ({L} layers), {per if False else len(sel)/n:.0f} kernels/step")
print(f"{'section':32s} {'us/step':>9s} {'us/layer':>9s} {'kern/layer':>10s}")
fixed = ("embed + pre", "final norm + head", "lm head GEMM")
for b, (us, cnt) in sorted(agg.items(), key=lambda x: -x[1][0]):
    d = us / n / 1e3
    if b in fixed: print(f"{b:32s} {d:9.0f} {'':>9s} {'':>10s}")
    else: print(f"{b:32s} {d:9.0f} {d / L:9.0f} {cnt / n / L:10.1f}")
print(f"{'idle (launch gaps, overlap)':32s} {wall - busy:9.0f} {(wall - busy) / L:9.0f}")
