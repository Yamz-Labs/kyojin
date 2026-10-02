# pfDepth1 table from depth_curve.py pfprof JSON: exclusive per-family GPU ms of one prefill chunk per depth,
# delta vs the shallowest depth, and growth floors (indexer scores FLOPs @47 TF, K-pool bytes @230 GB/s).
# usage: pf_depth_table.py <pf.json>
import json, sys
J = json.load(open(sys.argv[1]))
P = J["pfprof"]; C = J["args"]["chunk"]
PEAK, BW, L, H, D, KP, SLAB = 47e12, 230e9, 11, 32, 128, 4, 256


def fam(m):
    g = lambda k, f="gpu_ms": m.get(k, {}).get(f, 0.0)
    sc, tk, kp = g("idx.scores"), g("idx.dsa_topk"), g("mla.indexer_topk_kpool")
    sp, mla, kda, moe, top = g("mla.attend_sparse"), g("mla"), g("kda"), g("moe"), g("top")
    ik, pp, pre = g("mla.indexer_keys"), g("mla.update_pool_plane"), g("mla.attend_pre")
    other_sub = sum(v["gpu_ms"] for k, v in m.items() if isinstance(v, dict) and k.startswith(("attn:", "mlp:")))
    return {
        "DSA indexer scores": sc,
        "DSA top-k (dsa_topk kernel)": tk,
        "DSA top-k merge + q/w proj (kpool rest)": kp - sc - tk,
        "DSA idx keys + pool plane": ik + pp,
        "sparse MLA attention": sp,
        "MLA dense (proj, rope, post)": mla - kp - sp - ik - pp,
        "KDA": kda,
        "MoE": moe,
        "dense MLP / other sublayers": other_sub,
        "other (norms, hc, embed, head, host)": m["plain_ms"] - top + (top - mla - kda - moe - other_sub),
    }, {"topk_kpool_cpu_ms": g("mla.indexer_topk_kpool", "cpu_ms"), "topk_kpool_gpu_ms": kp}


def floors(d):
    pools = [(d + r + 1) // KP for r in range(C)]
    fl = sum(pools) * H * D * 2 * L                                   # scores FLOPs
    by = sum(((d + min(r0 + SLAB, C)) // KP) * D * 2 for r0 in range(0, C, SLAB)) * L   # K-pool re-read per slab
    sm = sum(pools) * 2 * 2 * L                                       # fp16 score matrix write + read
    return fl / PEAK * 1e3, by / BW * 1e3, sm / BW * 1e3


ds = sorted(P, key=int)
rows = {d: fam(P[d]["median"]) for d in ds}
d0, d1 = ds[0], ds[-1]
print(f"chunk {C} rows; plain wall ms: " + ", ".join(f"{d}: {P[d]['median']['plain_ms']} ({C/P[d]['median']['plain_ms']*1e3:.0f} t/s)" for d in ds))
print("instr overhead ms: " + ", ".join(f"{d}: {P[d]['median']['instr_ms']-P[d]['median']['plain_ms']:.1f}" for d in ds))
print("| family | " + " | ".join(f"ms @{int(d)//1024}K" for d in ds) + f" | delta {int(d0)//1024}K->{int(d1)//1024}K |")
print("|---|" + "---|" * (len(ds) + 1))
for k in rows[d0][0]:
    v = [rows[d][0][k] for d in ds]
    print(f"| {k} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {v[-1]-v[0]:+.1f} |")
tot = [P[d]["median"]["plain_ms"] for d in ds]
print("| **chunk wall (plain)** | " + " | ".join(f"{x:.1f}" for x in tot) + f" | {tot[-1]-tot[0]:+.1f} |")
for d in ds:
    f, b, s = floors(int(d))
    print(f"floor @{d}: scores FLOPs {f:.1f} ms, K-pool bytes {b:.1f} ms, score matrix bytes {s:.1f} ms; "
          f"topk_kpool cpu {rows[d][1]['topk_kpool_cpu_ms']:.1f} vs gpu {rows[d][1]['topk_kpool_gpu_ms']:.1f} ms; "
          f"calls scores {P[d]['median'].get('idx.scores',{}).get('calls')} topk {P[d]['median'].get('idx.dsa_topk',{}).get('calls')}")
