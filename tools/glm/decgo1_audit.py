#!/usr/bin/env python3
"""Audit one GLM plain-decode rocprofv3 trace against 230 GB/s."""
import argparse, collections, csv, glob, json, re, statistics

FLOOR = 230e9

def norm(name):
    return re.sub(r"<.*|\(.*", "", name).replace("void ", "").strip()[:160]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.profile)); w = d["prof_decode_window"]; p = d["prof_decode"]
    t0 = w["start"]["mono"] + int(p["ttft"] * 1e9); t1 = w["end"]["mono"]
    trace = glob.glob(a.profile.removesuffix(".json") + ".rocprof/*/*kernel_trace.csv")[0]
    rows = []
    for r in csv.DictReader(open(trace)):
        s = int(r["Start_Timestamp"])
        if t0 <= s <= t1:
            rows.append((s, int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    lmi = [i for i, r in enumerate(rows) if "gemv_kernel<10" in r[2]]
    steps = len(lmi) - 1
    kda = {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44}
    ndense = 3
    # Bytes/call for GLM td205 decode. Quantized projections use packed safetensor sizes;
    # recurrent state, sparse-latent traffic, and small fp16 projections use shapes.
    def assign(name, layer, half, occurrence):
        if "gemv_kernel<10" in name: return 397e6
        if "hc_mix_norm" in name: return 1.6e6
        if "router_kernel" in name: return 2.36e6
        if "moe_gu" in name: return 8 * 2 * (603979776 / 288)
        if "moe_down" in name: return 8 * (603979776 / 288)
        if layer < 0: return 0
        if half == 1:
            if "gemv_kernel<6" in name: return 18.9e6
            if "gemv_kernel<8" in name: return 4.2e6
        elif layer in kda:
            if "gemv_kernel<8" in name: return 50.4e6 if occurrence == 0 else 16.8e6
            if "recurrent_gated_delta" in name: return 64 * 128 * 128 * 4 * 2
            if "skinny" in name: return 0.52e6
            if "Cijk" in name: return 2.1e6
            if "conv1d" in name: return 0.2e6
        else:
            if "gemv_kernel<8" in name:
                return [(4.2e6, "MLA q_a/kv_a"), (12.6e6, "MLA q_b"),
                        (3.1e6, "DSA indexer wq_b"), (33.5e6, "MLA o")][min(occurrence, 3)][0]
            if "_mla_absorb" in name: return 16.8e6
            if "_mla_unfold" in name: return 16.8e6
            if "_dsa_attn_split" in name: return 2048 * 576 * 2
            if "skinny" in name: return 1.05e6
            if "_dsa_indexer" in name: return 0.5e6
        return 0

    time_ns = collections.Counter(); calls = collections.Counter()
    byte_ns: collections.Counter[str] = collections.Counter()
    gap = collections.Counter()
    for x, y in zip(lmi, lmi[1:]):
        seg = -1; prev_end = rows[x][1]; occ = collections.Counter()
        for s, e, raw in rows[x+1:y+1]:
            if "hc_mix_norm" in raw: seg += 1; occ = collections.Counter()
            key = norm(raw)
            layer = seg // 2 if seg >= 0 else -1
            half = seg % 2
            by = assign(raw, layer, half, occ[key])
            occ[key] += 1
            time_ns[key] += e-s; calls[key] += 1; byte_ns[key] += by
            gap[key] += max(0, s-prev_end); prev_end = max(prev_end, e)
    def per_tok(v): return v / steps
    wall = (rows[lmi[-1]][1] - rows[lmi[0]][1]) / 1e6 / steps
    busy = per_tok(sum(time_ns.values())) / 1e6
    dispatch = per_tok(sum(gap.values())) / 1e6
    total_bytes = per_tok(sum(byte_ns.values()))
    total_floor = total_bytes / FLOOR * 1e3
    ranked = []
    for k in time_ns:
        us = per_tok(time_ns[k]) / 1e3; by = per_tok(byte_ns[k]); floor_us = by / FLOOR * 1e6
        ranked.append((us-floor_us, k, us, per_tok(calls[k]), by, floor_us))
    out = ["# decgo1 byte-floor audit", "",
           f"Trace: `{a.profile}`. Decode context 4K, 64 generated tokens; {steps} complete inter-lm-head intervals.",
           f"Flags: `EXL3_DEC_MOE_FOLD=1 EXL3_DSA_DEC_DT=1`. Floor: 230 GB/s. Wall is profiler-observed, not an unprofiled A/B.", "",
           "## Top 15 by gap", "",
           "| rank | kernel | calls/token | us/token | bytes read/token | achieved GB/s | floor us | gap us |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, (g, k, us, n, by, fl) in enumerate(sorted(ranked, reverse=True)[:15], 1):
        bw = f"{by/us/1e3:.2f}" if by else "—"
        out.append(f"| {i} | `{k}` | {n:.1f} | {us:.2f} | {by/1e9:.6f} | {bw} | {fl:.2f} | {g:.2f} |")
    out += ["", "## Totals", "", "| metric | value |", "|---|---:|",
            f"| kernels/launches per token | {sum(calls.values())/steps:.1f} |",
            f"| summed measured kernel time | {busy:.2f} ms/token |",
            f"| summed accounted byte floor | {total_floor:.2f} ms/token |",
            f"| accounted bytes | {total_bytes/1e9:.6f} GB/token |",
            f"| profiler wall | {wall:.2f} ms/token |",
            f"| summed idle gaps between kernels | {dispatch:.2f} ms/token |", "",
            "## Every kernel name", "",
            "| kernel | calls/token | us/token | bytes read/token | achieved GB/s | gap us |", "|---|---:|---:|---:|---:|---:|"]
    for g, k, us, n, by, fl in sorted(ranked, key=lambda x: x[1]):
        bw = f"{by/us/1e3:.2f}" if by else "—"
        out.append(f"| `{k}` | {n:.1f} | {us:.2f} | {by/1e9:.6f} | {bw} | {g:.2f} |")
    out += ["", "## Accounting notes", "",
            "- Bytes are assigned from td205 packed weight sizes and decode tensor shapes; zero-byte families are norms, routing control, copies, and top-k glue.",
            "- KDA small-projection names are classified by launch shape: skinny fp16/cat or Cijk low-rank GEMM.",
            "- Idle time is the direct sum of non-overlapping gaps between consecutive trace kernels inside lm-head-to-lm-head intervals. It is a lower bound for end-to-end launch/sync overhead because boundary and profiler costs are excluded.",
            "- The 8.6 GB/token / 30 ms brief is not matched by the directly accounted shapes: this trace accounts for the numbers in the totals table. Unaccounted state, cache, and transient traffic is not invented into the floor."]
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"steps={steps} wall_ms={wall:.3f} busy_ms={busy:.3f} floor_ms={total_floor:.3f} idle_ms={dispatch:.3f} launches_tok={sum(calls.values())/steps:.1f}")
if __name__ == "__main__": main()
