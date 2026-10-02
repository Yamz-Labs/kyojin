# Step 11 round 2: (a) fla intra with gfx1151-pinned configs (tp BH16 w16, inter_solve BK16 w2 / w8), whole
# chunk_kda o/state vs fla autotune picks; (b) one-kernel intra sweep incl. fp16 inter (F16) / solve (S16) dots,
# each config's Aqk/Akk fed to the fused h kernel for an o diff vs reference.
import os, sys, json, itertools, argparse, torch, triton
sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)
from exllamav3.vendor.fla import RCP_LN2
from exllamav3.vendor.fla.l2norm import l2norm_fwd
from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
import exllamav3.vendor.fla.kda_chunk_intra as KI
import exllamav3.vendor.fla.kda_chunk_intra_token_parallel as TP
import exllamav3.vendor.fla.kda_intra_fused as KF
from exllamav3.vendor.fla.kda_fused_h import chunk_kda_fwd_h_fused

ap = argparse.ArgumentParser()
ap.add_argument("--caps", default="scratch/step10/cap")
ap.add_argument("--cases", default="t4096,t2048,t1792")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--bk", default="16,32")
ap.add_argument("--bkd", default="8,16,32")
ap.add_argument("--warps", default="2,4,8")
ap.add_argument("-o", "--out", default=None)
args = ap.parse_args()
dev = "cuda:0"
TPJ, ISJ, FJ = TP.chunk_kda_fwd_kernel_intra_token_parallel.fn.fn, KI.chunk_kda_fwd_kernel_inter_solve_fused.fn.fn, KF.chunk_kda_fwd_kernel_intra_fused.fn
L = lambda s: [int(x) for x in s.split(",")]


def timeit(fn, it):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()
out = []
for name in args.cases.split(","):
    d = torch.load(f"{args.caps}/{name}.pt")
    d = {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in d.items()}
    q, _ = l2norm_fwd(d["q"]); k, _ = l2norm_fwd(d["k"]); v = d["v"]; beta = d["beta"]
    g = chunk_local_cumsum(d["g"], chunk_size=64, scale=RCP_LN2)
    B, T, H, K = q.shape; HV = g.shape[2]; scale = K ** -0.5; NT = triton.cdiv(T, 64)
    h0 = d["initial_state"]

    def tail(Aqk, Akk):
        _, _, o, s = chunk_kda_fwd_h_fused(q=q, k=k, v=v, gk=g, beta=beta, Akk=Akk, Aqk=Aqk, scale=scale,
                                           initial_state=None if h0 is None else h0.clone(), output_final_state=True,
                                           chunk_size=64, fuse_o=True)
        return o.to(q.dtype), s
    fr = lambda: KI.chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, chunk_size=64, skip_wu=True)[4:]
    o0, s0 = tail(*fr())
    row = {"case": name, "T": T, "ref_ms": round(timeit(fr, args.iters), 3)}

    def fla_pinned(tbh, tw, bk, iw):
        Aqk = torch.empty(B, T, HV, 64, device=dev, dtype=k.dtype); Akk = torch.zeros_like(Aqk)
        Akkd = torch.empty(B, T, HV, 16, device=dev, dtype=torch.float32)
        TPJ[(B * T, triton.cdiv(HV, tbh))](q, k, g, beta, Aqk, Akkd, scale, None, B, T, H=H, HV=HV, K=K, BT=64, BC=16,
                                           BH=tbh, IS_VARLEN=False, num_warps=tw, num_stages=1)
        ISJ[(NT, B * HV)](q, k, g, beta, Aqk, Akkd, Akk, scale, None, None, T, H=H, HV=HV, K=K, BT=64, BC=16, NC=4,
                          BK=bk, IS_VARLEN=False, USE_SAFE_GATE=False, num_warps=iw, num_stages=1)
        return Aqk, Akk
    for cfg in [(8, 8, 32, 2), (16, 16, 16, 2), (16, 16, 16, 4), (16, 16, 16, 8), (32, 16, 16, 2)]:
        o1, s1 = tail(*fla_pinned(*cfg))
        row[f"fla{cfg}"] = {"ms": round(timeit(lambda: fla_pinned(*cfg), args.iters), 3), "o_bitwise": torch.equal(o1, o0),
                            "s_bitwise": torch.equal(s1, s0), "o_rel": rel(o1, o0)}
    print(json.dumps(row), flush=True)
    res = []
    for f16, s16, bk, bkd, nw in itertools.product((0, 1), (0, 1), L(args.bk), L(args.bkd), L(args.warps)):
        def run():
            Aqk = torch.empty(B, T, HV, 64, device=dev, dtype=k.dtype); Akk = torch.empty_like(Aqk)
            FJ[(NT, B * HV)](q, k, g, beta, Aqk, Akk, scale, T, H=H, HV=HV, K=K, BT=64, BC=16, BK=bk, BKD=bkd,
                             F16=bool(f16), S16=bool(s16), num_warps=nw, num_stages=1)
            return Aqk, Akk
        try:
            o1, s1 = tail(*run())
            res.append({"ms": round(timeit(run, args.iters), 3), "F16": f16, "S16": s16, "BK": bk, "BKD": bkd, "w": nw,
                        "o_rel": rel(o1, o0), "s_rel": rel(s1, s0), "finite": bool(torch.isfinite(o1).all())})
        except Exception as e:
            print(f"  cfg f{f16} s{s16} {bk} {bkd} {nw}: {type(e).__name__} {str(e)[-400:]}", flush=True)
    for f16, s16 in itertools.product((0, 1), (0, 1)):
        row[f"fused_f{f16}s{s16}"] = sorted([r for r in res if r["F16"] == f16 and r["S16"] == s16], key=lambda r: r["ms"])[:3]
    out.append(row)
    print(json.dumps({kk: vv for kk, vv in row.items() if kk.startswith("fused")}), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    del d, q, k, v, g, beta
    torch.cuda.empty_cache()
