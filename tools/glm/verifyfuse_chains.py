# verifyfuse1: rank GPU-side gaps inside verify_fwd_R2 by kernel and by chain, from a D1-1 rounddecomp trace.
#   usage: verifyfuse_chains.py <run.rocprof dir> <round_decomp.json> [phase]
import csv, glob, json, re, sys, collections
d, jf = sys.argv[1], sys.argv[2]; PH = sys.argv[3] if len(sys.argv) > 3 else "verify_fwd_R2"
EV = json.load(open(jf))["events"]
ops = []
for f in glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True):
    for r in csv.DictReader(open(f)):
        ops.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], int(r.get("Grid_Size_X") or 0),
                    int(r.get("Workgroup_Size_X") or 0)))
ops.sort()
MK = re.compile(r"FillFunctor<c10::complex")
raw = [(s, e, g // 256) for s, e, n, g, w in ops if MK.search(n)]
want = [e[0] for e in EV]
off = next(i for i, m in enumerate(raw) if m[2] == want[0])
marks = raw[off:off + len(EV)]
assert [m[2] for m in marks] == want
spans, st = [], []
for i, ev in enumerate(EV):
    if ev[2] == "in": st.append((ev[1], i))
    else:
        lab, j = st.pop(); spans.append((lab, marks[j][1], marks[i][0]))
vs = sorted((s, e) for l, s, e in spans if l == PH)
def short(n):
    n = re.sub(r"\(.*", "", n); n = re.sub(r"^void ", "", n)
    return n[:90]
import bisect
starts = [o[0] for o in ops]
K = collections.Counter(); KG = collections.defaultdict(float); KB = collections.defaultdict(float)
P = collections.defaultdict(float); PC = collections.Counter()
seqs = []
for s, e in vs:
    i = bisect.bisect_left(starts, s); seq = []
    while i < len(ops) and ops[i][1] <= e:
        if not MK.search(ops[i][2]): seq.append(ops[i])
        i += 1
    seqs.append(seq)
    for j, b in enumerate(seq):
        nb = short(b[2]); K[nb] += 1; KB[nb] += b[1] - b[0]
        if j:
            a = seq[j - 1]; gap = max(0, b[0] - a[1]); KG[nb] += gap
            P[(short(a[2]), nb)] += gap; PC[(short(a[2]), nb)] += 1
nr = len(vs)
print(f"{nr} {PH} spans; launches/round {sum(len(q) for q in seqs)/nr:.1f}; busy {sum(KB.values())/nr/1e6:.3f} ms; gaps-before {sum(KG.values())/nr/1e6:.3f} ms/round")
print("\nper kernel (gap BEFORE it = what fusing it into its predecessor removes):")
print(f"{'launch/rd':>9s} {'gap ms/rd':>9s} {'busy ms/rd':>10s} {'us/call':>7s}  kernel")
for n in sorted(K, key=lambda n: -KG[n])[:45]:
    print(f"{K[n]/nr:9.1f} {KG[n]/nr/1e6:9.3f} {KB[n]/nr/1e6:10.3f} {KB[n]/K[n]/1e3:7.1f}  {n}")
print("\ntop adjacent pairs by gap:")
for p in sorted(P, key=lambda p: -P[p])[:30]:
    print(f"{PC[p]/nr:7.1f} {P[p]/nr/1e6:7.3f}  {p[0][:55]} -> {p[1][:55]}")
json.dump([[(short(o[2]), o[1]-o[0], o[3], o[4]) for o in q] for q in seqs[:3]], open(sys.argv[4] if len(sys.argv) > 4 else "seq.json", "w"))

# ---- layer-family split: hc_mix_norm opens each half-layer (attn, ffn); tail = head ----
def cat(n):
    if n.startswith("had_"): return "hadamard"
    if n.startswith("hc_"): return "mHC"
    if n.startswith("at::native") or n == "": return "torch-glue"
    if "gemv" in n or "skinny" in n: return "gemv"
    if "moe_" in n or "router" in n: return "moe/router"
    return "other"
FL = collections.defaultdict(lambda: [0, 0.0]); FC = collections.defaultdict(lambda: [0, 0.0])
for seq in seqs:
    halves, cur = [], ("pre", [])
    for j, o in enumerate(seq):
        n = short(o[2])
        if n.startswith("hc_mix_norm"):
            halves.append(cur); cur = (len(halves), [])
        cur[1].append(j)
    halves.append(cur)
    nh = len(halves) - 1
    for hi, (tag, idx) in enumerate(halves):
        names = [short(seq[j][2]) for j in idx]
        if tag == "pre": fam = "pre"
        else:
            h = tag - 1; layer = h // 2; part = "attn" if h % 2 == 0 else "ffn"
            lay_names = []
            for t2, ix2 in halves:
                if t2 != "pre" and (t2 - 1) // 2 == layer: lay_names += [short(seq[j][2]) for j in ix2]
            kind = "KDA" if any("recurrent_gated_delta" in x for x in lay_names) else \
                   "DSA" if any(x.startswith(("_mla", "_dsa")) for x in lay_names) else "other"
            fam = f"{kind}-{part}"
            if hi == nh and part == "ffn":
                # last half-layer also carries the head (final norm, lm_head); split at the last moe_combine/add
                pass
        for j in idx:
            if j == 0: continue
            gap = max(0, seq[j][0] - seq[j - 1][1]); n = short(seq[j][2])
            FL[fam][0] += 1; FL[fam][1] += gap
            FC[(fam, cat(n))][0] += 1; FC[(fam, cat(n))][1] += gap
print("\nper layer family (launches/round, gap-before ms/round):")
for f in sorted(FL, key=lambda f: -FL[f][1]):
    print(f"  {f:10s} {FL[f][0]/nr:7.1f} {FL[f][1]/nr/1e6:7.3f}   " +
          "  ".join(f"{c}:{FC[(f,c)][0]/nr:.0f}/{FC[(f,c)][1]/nr/1e6:.3f}" for c in
                    ("hadamard", "mHC", "torch-glue", "gemv", "moe/router", "other") if FC[(f, c)][0]))
