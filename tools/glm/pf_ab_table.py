# pfDepth2 A/B table from depth_curve.py pfprof --pf-arms JSON: per arm x depth, plain chunk wall,
# dsa_topk GPU ms, kpool top-k total (scores + topk + merge), and the check block (bitwise / greedy).
# usage: pf_ab_table.py <pf.json>
import json, sys
J = json.load(open(sys.argv[1]))
P = J["pfprof"]; C = J["args"]["chunk"]
arms = sorted({k.split("@")[0] for k in P}); ds = sorted({int(k.split("@")[1]) for k in P})
g = lambda m, k: m.get(k, {}).get("gpu_ms", 0.0)
print("| depth | arm | chunk wall ms (samples) | t/s | dsa_topk | merge+proj (kpool - scores - topk) | kpool total |")
print("|---|---|---|---|---|---|---|")
for d in ds:
    for a in arms:
        r = P[f"{a}@{d}"]; m = r["median"]; kp, sc, tk = g(m, "mla.indexer_topk_kpool"), g(m, "idx.scores"), g(m, "idx.dsa_topk")
        print(f"| {d} | {a} | {m['plain_ms']:.1f} ({', '.join(f'{x*1e3:.0f}' for x in r['plain_s'])}) | {C/m['plain_ms']*1e3:.0f} | "
              f"{tk:.1f} | {kp-sc-tk:.1f} | {kp:.1f} |")
for k, v in J.get("pfab_check", {}).items():
    print(f"check {k}: bitwise {v['bitwise_calls']}/{v['calls']} set {v['set_equal_calls']} rows_differ {v['rows_differ']} "
          f"logits_max_abs {v['logits_max_abs']:.3g} greedy {v['greedy_same']}/32")
