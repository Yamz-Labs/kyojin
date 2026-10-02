#!/usr/bin/env python
"""Cut a rocprofv3 kernel trace to the decode window between the two marker fills and summarise.

  trace_split.py <kernel_trace.csv> [ntokens]
Prints wall time of the window, GPU busy time, top kernels by total time, launches per token,
and a coarse category split (MoE GEMV, dense GEMV, hadamard, router, attention, norm, other).
"""
import csv, sys, re, collections

path = sys.argv[1]
ntok = int(sys.argv[2]) if len(sys.argv) > 2 else None
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                     int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"]),
                     int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"])))
rows.sort()
MARK = 7777777
marks = [i for i, r in enumerate(rows) if "FillFunctor<float>" in r[2] and r[3] >= MARK // 4 and r[3] <= MARK * 2]
# the marker fill of 7777777 floats: pick the two latest candidates with that exact thread count range
cands = [i for i in marks]
if len(cands) < 2:
    print("markers not found", len(cands)); sys.exit(1)
a, b = cands[-2], cands[-1]
win = rows[a + 1:b]
t0, t1 = rows[a][1], rows[b][0]
wall = (t1 - t0) / 1e3
busy = sum(r[1] - r[0] for r in win) / 1e3
print(f"window: {len(win)} kernels, wall {wall:.0f} us, GPU busy {busy:.0f} us ({100*busy/wall:.1f} %)")

def short(n):
    n = re.sub(r"\(.*", "", n)
    return n[:110]

def cat(n):
    l = n.lower()
    if "dec_" in l or "exl3dec" in l:
        if "moe" in l: return "moe (dec)"
        if "router" in l: return "router (dec)"
        return "gemv (dec)"
    if "moe" in l and "gemv" in l: return "moe gemv"
    if "exl3_gemv" in l or "gemv_kernel" in l: return "gemv"
    if "had" in l: return "hadamard"
    if "topk" in l or "sort" in l or "gather" in l and "moe" not in l: return "router/topk/sort"
    if "gemm" in l or "cijk" in l: return "blas gemm"
    if "attn" in l or "attention" in l or "flash" in l or "softmax" in l: return "attention"
    if "norm" in l or "rsqrt" in l: return "norm"
    if "rope" in l: return "rope"
    if "copy" in l or "memcpy" in l: return "copy"
    if "elementwise" in l or "reduce" in l: return "torch elementwise/reduce"
    return "other"

tot = collections.Counter(); cnt = collections.Counter()
ctot = collections.Counter(); ccnt = collections.Counter()
for r in win:
    d = (r[1] - r[0]) / 1e3
    k = short(r[2]); tot[k] += d; cnt[k] += 1
    c = cat(r[2]); ctot[c] += d; ccnt[c] += 1
div = ntok or 1
print(f"\nper token (/{div}): kernels {len(win)/div:.0f}, wall {wall/div:.0f} us, busy {busy/div:.0f} us")
print("\ncategory                      us/tok   launches/tok")
for c, t in ctot.most_common():
    print(f"  {c:<28} {t/div:8.0f} {ccnt[c]/div:8.1f}")
print("\ntop kernels                                                                                     us/tok  n/tok")
for k, t in tot.most_common(30):
    print(f"  {k:<100} {t/div:7.0f} {cnt[k]/div:6.1f}")
# gaps
gaps = [(win[i+1][0] - win[i][1]) / 1e3 for i in range(len(win) - 1)]
gaps = [g for g in gaps if g > 0]
gaps.sort()
if gaps:
    print(f"\ngaps: total {sum(gaps)/div:.0f} us/tok, median {gaps[len(gaps)//2]:.1f} us, >50us: {sum(g for g in gaps if g > 50)/div:.0f} us/tok")
