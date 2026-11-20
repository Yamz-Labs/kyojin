"""mla_micro.py (cwd = tree): MODEL-FREE micro A/B of the shipped Hy4 decode attention of the full-attention (indexer) layers (HY4_LC_SPARSE: lc_sparse.sparse_mla_attend):
H = 32 heads, R 512, P 64, rows T = 1..6, context N = 4K / 32K, selection like the indexer (512 groups of 4 keys per row, rows overlap ~90 %, + the open tail group).
Arms: f32 = fp32 K cache, q8 = fp8 cache with the one-head kernel (shipped HY4_KV_FP8), q8hb = fp8 cache head-blocked (HY4_LC_HB=1), plus any arm added below.
NL distinct layer caches are cycled (one cache re-read in a loop would sit in the 32 MB MALL). Prints ms per call and ms per verify round (x NL_ROUND layers) and the
max |diff| of every arm against dq32 (q8 must be 0.0: same dequantised values).
env: MLA_CPU=1 (TRITON_INTERPRET, tiny shapes, correctness only), MLA_NL (default 12), MLA_NS ("1000,4000,32000"), MLA_TS ("1,2,3,4,6"), MLA_ARMS ("f32,dq32,q8"), HY4_MLA_HB (shipped serve flag)."""
import os, sys, time
CPU = os.environ.get("MLA_CPU") == "1"
if CPU:
    os.environ["TRITON_INTERPRET"] = "1"
os.environ.setdefault("HY4_REF_FUSED", "1")
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import torch
import engine as _eng
_eng._stub_package(tree)
from exllamav3.loader import ref_fused as rf, kv_fp8 as kv8, lc_sparse as LS
dev = torch.device("cpu" if CPU else "cuda")
R, P, H = 512, 64, 32
NL = 2 if CPU else int(os.environ.get("MLA_NL", "12"))
NS = [256] if CPU else [int(x) for x in os.environ.get("MLA_NS", "4000,32000").split(",")]
TS = [1, 3] if CPU else [int(x) for x in os.environ.get("MLA_TS", "1,2,3,4,6").split(",")]
ARMS = os.environ.get("MLA_ARMS", "f32,q8,q8hb,q8hbs2,q8hbs4,q8hbs8,q8hbs16").split(",")
LAYERS_PER_ROUND = 12
TOPK, KP = (64 if CPU else 2048), 4
torch.manual_seed(0)


def ev(fn, n=20):
    if CPU:
        fn(); return 0.0
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def make_sel(T, N, qpos):
    """SparseSel: per row n_top groups (rows share ~90 % of them), nsel all valid (groups below the open tail group)"""
    nt = TOPK // KP - 1
    base = torch.randperm(max(1, N // KP - 2), device=dev)[:nt]
    idx = torch.empty(T, TOPK // KP - 1, dtype=torch.int32, device=dev)
    for t in range(T):
        r = base.clone()
        k = nt // 10
        r[torch.randperm(nt, device=dev)[:k]] = torch.randint(0, max(1, int(qpos[t].item()) // KP - 1), (k,), device=dev)
        idx[t] = r.int()
    nsel = torch.full((T,), nt, dtype=torch.int32, device=dev)
    return LS.SparseSel(idx, nsel, KP, True)


for N in NS:
    cap = N + 64
    caches = []
    for _ in range(NL):
        k = torch.randn(cap, R + P, device=dev) * 1.5
        f = kv8.Fp8KV(cap, R, P, dev)
        f.write(0, k)
        caches.append((k, f))
    sinks = torch.randn(H, device=dev)
    scale = (R + P) ** -0.5 * 1.7
    for T in TS:
        qpos = torch.arange(N - T, N, device=dev)
        q = torch.randn(T, H, R, device=dev) * 0.5
        qp = torch.randn(T, H, P, device=dev) * 0.5
        sel = make_sel(T, N, qpos)

        def run(arm, i):
            k, f = caches[i % NL]
            if arm == "f32":
                return LS.sparse_mla_attend(q, qp, k[:, :R], k[:, R:], qpos, sel, scale, sinks)
            LS.HB_ON = arm.startswith("q8hb")
            LS.HB_SPLIT = int(arm[5:]) if arm.startswith("q8hbs") else 0
            return LS.sparse_mla_attend(q, qp, f.lat, f.pe, qpos, sel, scale, sinks, scl=f.sc)

        ref = run("q8", 0)
        row = []
        for arm in ARMS:
            o = run(arm, 0)
            d = (o - ref).abs().max().item()
            cnt = [0]

            def f():
                run(arm, cnt[0]); cnt[0] += 1
            ms = ev(f)
            row.append(f"{arm} {ms:.3f} ms/layer ({ms * LAYERS_PER_ROUND:.2f}/round, maxdiff vs q8 {d:.1e})")
        print(f"N {N} T {T}: " + " | ".join(row), flush=True)
    del caches


# ---- decode attention of the layers below the sparse threshold (allow None, HY4_MLA_HB kernels): shipped split kernel vs the rows kernel (HY4_MLA_HB_ROWS)
rf.HB = True
if os.environ.get("MLA_ROWS", "1") == "1":
    RARMS = os.environ.get("MLA_RARMS", "hb,r8x4,r8x2,r16x4,r16x2,r8x4n4").split(",")
    for (Hh, win, Nn) in ((64, int(os.environ.get("MLA_WIN", "128")), 1000), (32, 0, 1000), (32, 0, 2000)):
        cap = Nn + 64
        kc = [torch.randn(cap, R + P, device=dev) * 1.5 for _ in range(NL)]
        sinks = torch.randn(Hh, device=dev)
        kpos = torch.arange(cap, device=dev)
        for T in [t for t in TS if t >= 1]:
            qpos = torch.arange(Nn - T, Nn, device=dev)
            q = torch.randn(T, Hh, R, device=dev) * 0.5
            qp = torch.randn(T, Hh, P, device=dev) * 0.5
            scale = (R + P) ** -0.5 * 1.7

            def runr(arm, i):
                k = kc[i % NL]
                rf.HB_ROWS, rf.HB_ROWS_NW = 0, 4
                if arm != "hb":      # rNxM[nW]: BH = N heads, RG = M rows per program, nW num_warps
                    bh, rg = arm[1:].split("x")
                    nw = 8
                    if "n" in rg:
                        rg, nw = rg.split("n")
                    rf.HB_ROWS, rf.HB_ROWS_BH, rf.HB_ROWS_RG, rf.HB_ROWS_NW = 8, int(bh), int(rg), int(nw)
                return rf.mla_sink_attend(q, qp, k[:, :R], k[:, R:], qpos, kpos, scale, sinks, win, None, kpos_is_arange=True)

            ref = runr("hb", 0)
            row = []
            for arm in RARMS:
                if T == 1 and arm != "hb":
                    continue
                o = runr(arm, 0)
                d = (o - ref).abs().max().item()
                cnt = [0]

                def f():
                    runr(arm, cnt[0]); cnt[0] += 1
                ms = ev(f)
                row.append(f"{arm} {ms * 1000:.0f}us (diff {d:.1e})")
            print(f"ROWS H {Hh} win {win} N {Nn} T {T}: " + " | ".join(row), flush=True)
