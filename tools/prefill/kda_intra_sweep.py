# Step 11: chunk_kda_fwd_intra (token_parallel + inter_solve_fused) microbench on captured GLM inputs.
# Times each kernel, sweeps configs with autotune bypassed, prints the autotune picks and diffs vs reference.
import os, sys, glob, json, itertools, argparse, torch, triton
sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)
from exllamav3.vendor.fla import RCP_LN2
from exllamav3.vendor.fla.l2norm import l2norm_fwd
from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
import exllamav3.vendor.fla.kda_chunk_intra as KI
import exllamav3.vendor.fla.kda_chunk_intra_token_parallel as TP

ap = argparse.ArgumentParser()
ap.add_argument("--caps", default="scratch/step10/cap")
ap.add_argument("--cases", default="t4096,t2048,t1792")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--sweep", default="tp,is")
ap.add_argument("-o", "--out", default=None)
args = ap.parse_args()
dev = "cuda:0"
TPJ = TP.chunk_kda_fwd_kernel_intra_token_parallel.fn.fn
ISJ = KI.chunk_kda_fwd_kernel_inter_solve_fused.fn.fn


def timeit(fn, it):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def mad(a, b):
    return (a.float() - b.float()).abs().max().item()


out = []
for name in args.cases.split(","):
    d = torch.load(f"{args.caps}/{name}.pt")
    q, _ = l2norm_fwd(d["q"].to(dev)); k, _ = l2norm_fwd(d["k"].to(dev)); v = d["v"].to(dev)
    beta = d["beta"].to(dev)
    g = chunk_local_cumsum(d["g"].to(dev), chunk_size=64, scale=RCP_LN2)
    B, T, H, K = q.shape; HV = g.shape[2]; BT, BC = 64, 16; NT = triton.cdiv(T, BT); NC = 4
    scale = K ** -0.5
    f = lambda: KI.chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, chunk_size=BT, skip_wu=True)
    *_, Aqk0, Akk0 = f()
    *_, Aqk1, Akk1 = f()
    row = {"case": name, "T": T, "repeat_bitwise": torch.equal(Aqk0, Aqk1) and torch.equal(Akk0, Akk1)}
    row["intra_ms"] = round(timeit(f, args.iters), 3)
    Aqk = torch.empty_like(Aqk0); Akkd = torch.empty(B, T, HV, BC, device=dev, dtype=torch.float32)
    Akk = torch.zeros_like(Akk0)
    ftp = lambda: TP.chunk_kda_fwd_intra_token_parallel(q=q, k=k, gk=g, beta=beta, Aqk=Aqk, Akk=Akkd, scale=scale)
    fis = lambda: KI.chunk_kda_fwd_kernel_inter_solve_fused[(NT, B * HV)](
        q=q, k=k, g=g, beta=beta, Aqk=Aqk, Akkd=Akkd, Akk=Akk, scale=scale, cu_seqlens=None, chunk_indices=None,
        T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, NC=NC, USE_SAFE_GATE=False)
    row["tp_ms"] = round(timeit(ftp, args.iters), 3)
    row["is_ms"] = round(timeit(fis, args.iters), 3)
    row["zeros_ms"] = round(timeit(lambda: torch.zeros_like(Akk0), args.iters), 3)
    row["picks"] = {"tp": [str(v) for v in TP.chunk_kda_fwd_kernel_intra_token_parallel.fn.cache.values()],
                    "is": [str(v) for v in KI.chunk_kda_fwd_kernel_inter_solve_fused.fn.cache.values()]}
    Akkd_ref = Akkd.clone()
    if "tp" in args.sweep:
        res = []
        for bh, nw, ns in itertools.product([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16], [1, 2]):
            run = lambda: TPJ[(B * T, triton.cdiv(HV, bh))](q, k, g, beta, Aqk, Akkd, scale, None, B, T, H=H, HV=HV, K=K,
                                                            BT=BT, BC=BC, BH=bh, IS_VARLEN=False, num_warps=nw, num_stages=ns)
            try:
                t = timeit(run, args.iters)
                res.append((round(t, 3), bh, nw, ns, torch.equal(Akkd, Akkd_ref)))
            except Exception as e:
                pass
        row["tp_top"] = sorted(res)[:5]
    if "is" in args.sweep:
        ftp(); res = []
        for bk, nw, ns in itertools.product([16, 32, 64, 128], [1, 2, 4, 8], [1, 2, 3]):
            run = lambda: ISJ[(NT, B * HV)](q, k, g, beta, Aqk, Akkd, Akk, scale, None, None, T, H=H, HV=HV, K=K, BT=BT,
                                             BC=BC, NC=NC, BK=bk, IS_VARLEN=False, USE_SAFE_GATE=False,
                                             num_warps=nw, num_stages=ns)
            try:
                t = timeit(run, args.iters)
                res.append((round(t, 3), bk, nw, ns, torch.equal(Akk, Akk0), mad(Akk, Akk0)))
            except Exception as e:
                pass
        row["is_top"] = sorted(res)[:5]
    # safe_gate path (sub_chunk dot kernel for the diagonal blocks): other numerics, report diffs
    fs = lambda: KI.chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, chunk_size=BT, skip_wu=True, safe_gate=True)
    *_, Aqks, Akks = fs()
    row["safe_ms"] = round(timeit(fs, args.iters), 3)
    row["safe_mad"] = [mad(Aqks, Aqk0), mad(Akks, Akk0)]
    out.append(row)
    print(json.dumps(row), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    del d, q, k, v, g, beta
    torch.cuda.empty_cache()
