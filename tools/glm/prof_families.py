# Per-kernel-family prefill profile from a rocprofv3 kernel_trace.csv, sliced on glm_base.py prof windows.
# usage: prof_families.py <run.rocprof dir> <run.json> [depth ...]
# Windows: json "prof_windows" {depth: {start, end, ttft, chunks}} (ab-sets --prof-depths) or "prof_prefill_window" (prof mode).
# Floors (GLM-5.3 text, gfx1151): peak 47 TFLOP/s WMMA fp16, 256 GB/s DRAM; per-token FLOP/bytes basis in FLOOR below.
import csv, json, sys, glob, re, collections

PEAK, BW = 47e12, 256e9
L_MOE, L_MLA, L_KDA, HC_CALLS = 41, 11, 34, 90
FAMILIES = [  # first match wins
    ("MoE grouped GEMM", r"^mpw_gemm|mpw2_gemm"),
    ("MoE aux (gather/act/reduce/route)", r"^mpw_|moe_gu|moe_down|router_kernel|gatherTopK|sigmoid_kernel"),
    ("dense GEMM (hipBLASLt)", r"^Cijk_|skinny|gemv_kernel"),
    ("dense weight reconstruct", r"^reconstruct|had_hf_r"),
    ("KDA (conv, chunk, solve, o)", r"kda|gla_|gated_delta|causal_conv1d|local_cumsum|l2norm"),
    ("MLA/DSA attention", r"^_mla_|^_dsa_attn"),
    ("DSA indexer + top-k", r"^_dsa_index|radixSort|rocprim|sbtopk|topk"),
    ("mHC stream mix/apply", r"^hc_"),
    ("norms / elementwise / copies", r"."),
]

BWFAM = {"dense weight reconstruct", "KDA (conv, chunk, solve, o)", "mHC stream mix/apply"}

def floors(T, chunks):
    """ms floor per family for T prompt tokens."""
    # MoE: 8 experts x (4096x4096 gate/up + 2048x4096 down) x 2 FLOP x 41 layers = 16.5 GFLOP/token
    moe = 8 * (4096 * 4096 + 2048 * 4096) * 2 * L_MOE * T / PEAK
    # dense: 28 GFLOP/token total model minus MoE ~ 11.5 GFLOP/token (projections, shared experts, 3 dense MLPs)
    dense = 11.5e9 * T / PEAK
    # dense weight reconstruct: ~6 G params x 2 B fp16 write + ~0.6 B read, once per chunk
    recon = len(chunks) * 6e9 * 2.6 / BW
    # KDA: read q,k,v,g (4 x 8192 bf16) + write o (8192 bf16) per token per layer = 80 KB
    kda = 80e3 * L_KDA * T / BW
    # MLA/DSA: absorbed latent 512+512, 64 heads: 131 kFLOP per query-key; keys = min(ctx, 2048 top-k)
    fl, pos = 0.0, 0
    for c in chunks:
        for q in range(pos, pos + c, 64):
            fl += 64 * min(q + 1, 2048) * 131072
        pos += c
    attn = fl * L_MLA / PEAK
    # indexer: 32 heads x 128 dim over ctx/4 pooled keys
    fi, pos = 0.0, 0
    for c in chunks:
        for q in range(pos, pos + c, 64):
            fi += 64 * (q + 1) / 4 * 32 * 128 * 2
        pos += c
    idx = fi * L_MLA / PEAK
    # mHC: fp32 4 x 4096 residual stream, ~224 KB moved per token per sublayer (mix read + apply read/write)
    hc = 224e3 * HC_CALLS * T / BW
    return {"MoE grouped GEMM": moe, "MoE aux (gather/act/reduce/route)": None, "dense GEMM (hipBLASLt)": dense,
            "dense weight reconstruct": recon, "KDA (conv, chunk, solve, o)": kda, "MLA/DSA attention": attn,
            "DSA indexer + top-k": idx, "mHC stream mix/apply": hc, "norms / elementwise / copies": None}

def main():
    d, js = sys.argv[1], sys.argv[2]
    J = json.load(open(js))
    W = J.get("prof_windows") or {"4096": {**J["prof_prefill_window"], "chunks": [2048, 1792, 255]}}
    want = sys.argv[3:] or list(W)
    f = glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True)[0]
    fam_re = [(n, re.compile(p)) for n, p in FAMILIES]
    cache = {}
    def fam(k):
        if k not in cache:
            base = k.split("(")[0].replace("void ", "").strip()
            cache[k] = next((n for n, p in fam_re if p.search(base)), FAMILIES[-1][0])
        return cache[k]
    # one streaming pass (the trace can be GBs); clock = the one whose window start is near the trace stamps
    agg = {dep: collections.defaultdict(float) for dep in want}; busy = dict.fromkeys(want, 0.0); win = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
            if not win:
                for dep in want:
                    clk = min(("boot", "mono"), key=lambda c: abs(W[dep]["start"][c] - s))
                    win[dep] = (W[dep]["start"][clk], W[dep]["end"][clk])
            for dep, (a, b) in win.items():
                if a <= s and e <= b:
                    agg[dep][fam(r["Kernel_Name"])] += (e - s) / 1e6; busy[dep] += (e - s) / 1e6
    for dep in want:
        w = W[dep]; a, b = win[dep]; agg_d = agg[dep]; busy_d = busy[dep]
        T = int(dep); wall = (b - a) / 1e6
        agg_d["host gaps (window - GPU busy)"] = wall - busy_d
        fl = floors(T, w["chunks"])
        print(f"\n## prefill@{T}: TTFT {w['ttft']:.2f} s = {T / w['ttft']:.1f} t/s, window {wall:.0f} ms, GPU busy {busy_d:.0f} ms, chunks {w['chunks']}")
        print("| family | ms | share | floor ms | achieved | % of floor speed | gap ms |")
        print("|---|---|---|---|---|---|---|")
        out = []
        for n, ms in agg_d.items():
            f_ = fl.get(n)
            fms = f_ * 1e3 if f_ is not None else 0.0
            out.append((ms - fms, n, ms, f_))
        for gap, n, ms, f_ in sorted(out, reverse=True):
            fs = f"{f_ * 1e3:.0f}" if f_ is not None else "0 (fuse)"
            pct = f"{100 * f_ * 1e3 / ms:.0f}%" if f_ is not None and ms > 0 else "-"
            ach = "-" if not f_ or ms <= 0 else (f"{f_ * 1e3 / ms * BW / 1e9:.0f} GB/s" if n in BWFAM else f"{f_ * 1e3 / ms * PEAK / 1e12:.1f} TF/s")
            print(f"| {n} | {ms:.0f} | {100 * ms / wall:.1f}% | {fs} | {ach} | {pct} | {gap:.0f} |")
        tf = sum(v for v in fl.values() if v)
        print(f"sum of floors {tf * 1e3:.0f} ms -> {T / tf:.0f} t/s; 1000 t/s needs TTFT {T / 1000:.2f} s")

main()
