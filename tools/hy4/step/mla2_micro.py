"""mla2_micro.py (cwd = tree): MODEL-FREE microbench + check of the row-batched MLA decode attention (ref_fused HY4_MLA_ROWS2) vs the DEPLOYED kernel (HY4_MLA_HB=1: head-batched split-K, one program per (row, 16 heads, split)).
Real Hy4 shapes (R 512, P 64): class A = 12 full layers (H 32, no window), class B = 30 window layers (H 64, window 512); ctx 1K / 4K / 16K; T = 1..8 verify rows.
Per point: ms per layer call (HIP graph replay of a cycle over distinct per-layer fp32 KV buffers, like opbench mla_rows), max|err| and rel err (norm ratio) vs the f32 torch reference
(mla_sink_attend_ref), for the deployed kernel and every mla2 config; ROUND lines = 12 x A + 30 x B (ms per verify round) for deployed vs the best config per class.
env: MLA_CPU=1 (TRITON_INTERPRET, tiny shapes, correctness only); MLA2_CTX ("1024,4096,16384"); MLA2_TS ("1,2,3,4,5,6,7,8"); MLA2_CFGS (semicolon list "BH,RG,BN,BK,NW,PROG,MINB"); MLA2_OUT (json path)."""
import os, sys, time, json, math, statistics, importlib.util
CPU = os.environ.get("MLA_CPU") == "1"
if CPU:
    os.environ["TRITON_INTERPRET"] = "1"
import torch
tree = os.getcwd()
spec = importlib.util.spec_from_file_location("ref_fused", os.path.join(tree, "exllamav3", "loader", "ref_fused.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
dev = torch.device("cpu" if CPU else "cuda")
R, P = 512, 64
CLASSES = (("A_H32_full", 32, 0, 12), ("B_H64_win512", 64, 512, 30))
CTX = [200] if CPU else [int(x) for x in os.environ.get("MLA2_CTX", "1024,4096,16384").split(",")]
TS = [1, 3] if CPU else [int(x) for x in os.environ.get("MLA2_TS", "1,2,3,4,5,6,7,8").split(",")]
DEF = "16,2,16,64,8,64,2;16,2,32,64,8,64,2;16,1,16,64,4,64,2;16,1,32,64,4,64,2;16,4,16,64,8,64,2;8,2,16,64,4,64,2;8,4,16,64,8,64,2;16,2,16,64,4,64,2;16,2,16,32,8,64,2;16,2,16,64,8,32,2;16,2,16,64,8,128,2;16,2,16,64,8,64,4"
CFGS = [tuple(int(x) for x in c.split(",")) for c in os.environ.get("MLA2_CFGS", "16,2,16,64,8,64,2;16,1,16,64,4,64,2" if CPU else DEF).split(";")]
NREP = 1 if CPU else 30
torch.manual_seed(0)


def setcfg(c):
    rf.R2_BH, rf.R2_RG, rf.R2_BN, rf.R2_BK, rf.R2_NW, rf.R2_PROG, rf.R2_MINB = c


def graph_ms(fns):
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(3):
        g.replay()
    ts = []
    for _ in range(NREP):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts) / len(fns)


def err(a, ref):
    d = (a - ref)
    return d.abs().max().item(), (d.norm() / ref.norm()).item()


res = []          # (cls, ctx, T, name, ms, maxabs, rel)
for ctx in CTX:
    for cname, H, win, nl in CLASSES:
        cap = ctx + 16
        nb = 2 if CPU else max(2, min(24, int(48e6 // (cap * (R + P) * 4)) + 1))
        kvs = [torch.randn(cap, R + P, device=dev) * 1.5 for _ in range(nb)]
        sinks = torch.randn(H, device=dev)
        scale = (R + P) ** -0.5 * 1.7
        kpos = torch.arange(cap, device=dev)
        for T in TS:
            q = torch.randn(T, H, R, device=dev) * 0.5
            qp = torch.randn(T, H, P, device=dev) * 0.5
            qpos = torch.arange(ctx - T, ctx, device=dev)
            ref = rf.mla_sink_attend_ref(q, qp, kvs[0][:ctx, :R], kvs[0][:ctx, R:], qpos, kpos[:ctx], scale, sinks, win, None)

            def call(i):
                kc = kvs[i % nb]
                return rf.mla_sink_attend(q, qp, kc[:ctx, :R], kc[:ctx, R:], qpos, kpos[:ctx], scale, sinks, win, None, kpos_is_arange=True)
            arms = [("deployed", None)] + [("mla2:" + ",".join(map(str, c)), c) for c in CFGS]
            for name, c in arms:
                if c is None:
                    rf.ROWS2, rf.HB, rf.HB_ROWS = False, True, 0
                else:
                    rf.ROWS2 = True
                    setcfg(c)
                    if c[0] * min(c[1], 1 << (T - 1).bit_length()) < 16 or H % c[0]:
                        continue
                try:
                    o = call(0)
                    ea, er = err(o, ref)
                    ms = 0.0 if CPU else graph_ms([(lambda i=i: call(i)) for i in range(nb)])
                except Exception as e:      # noqa: BLE001
                    print(f"MLA2 {cname} ctx {ctx} T {T} {name}: FAIL {type(e).__name__} {str(e)[:100]}", flush=True)
                    continue
                res.append((cname, ctx, T, name, ms, ea, er))
                print(f"MLA2 {cname} ctx {ctx} T {T} {name:28s} ms/layer {ms:.4f}  maxabs {ea:.2e} rel {er:.2e}", flush=True)
            rf.ROWS2 = False
        del kvs
        if not CPU:
            torch.cuda.empty_cache()
# verify-round table: 12 x A + 30 x B, per ctx and T: deployed vs best mla2 config per class (min ms) and vs one fixed config
print("\nROUND ms per verify round (12 x A + 30 x B attention layers):")
for ctx in CTX:
    for T in TS:
        row = {}
        for cname, H, win, nl in CLASSES:
            pts = [r for r in res if r[0] == cname and r[1] == ctx and r[2] == T]
            dep = [r for r in pts if r[3] == "deployed"]
            new = sorted([r for r in pts if r[3] != "deployed"], key=lambda r: r[4])
            if dep and new:
                row[cname] = (nl, dep[0], new[0])
        if len(row) == 2:
            d = sum(nl * a[4] for nl, a, b in row.values()); n = sum(nl * b[4] for nl, a, b in row.values())
            print(f"ROUND ctx {ctx} T {T}: deployed {d:.2f} ms -> mla2 best {n:.2f} ms ({n - d:+.2f}); " + " | ".join(f"{k}: {b[3]} rel {b[6]:.1e} (dep rel {a[6]:.1e})" for k, (nl, a, b) in row.items()), flush=True)
out = os.environ.get("MLA2_OUT")
if out:
    json.dump(res, open(out, "w"))
print("MLA2_DONE", flush=True)
