# Step 11: one-kernel KDA intra (EXL3_KDA_INTRA=1) vs fla's token_parallel + inter_solve_fused, on captured GLM
# inputs. Prints max-abs-diff of Aqk (lower triangle) / Akk, whole chunk_kda o/state diffs, ms, config sweep.
import os, sys, json, itertools, argparse, torch, triton
sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)
import exllamav3.vendor.fla as F
from exllamav3.vendor.fla import RCP_LN2
from exllamav3.vendor.fla.l2norm import l2norm_fwd
from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
import exllamav3.vendor.fla.kda_chunk_intra as KI
import exllamav3.vendor.fla.kda_intra_fused as KF

ap = argparse.ArgumentParser()
ap.add_argument("--caps", default="scratch/step10/cap")
ap.add_argument("--cases", default="t4096,t2048,t1792")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--sweep", type=int, default=1)
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--arm", default="1", help="EXL3_KDA_INTRA value compared against 0")
args = ap.parse_args()
dev = "cuda:0"
FJ = KF.chunk_kda_fwd_kernel_intra_fused.fn


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


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


out = []
for name in args.cases.split(","):
    d = torch.load(f"{args.caps}/{name}.pt")
    d = {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in d.items()}
    q, _ = l2norm_fwd(d["q"]); k, _ = l2norm_fwd(d["k"]); v = d["v"]; beta = d["beta"]
    g = chunk_local_cumsum(d["g"], chunk_size=64, scale=RCP_LN2)
    B, T, H, K = q.shape; HV = g.shape[2]; scale = K ** -0.5
    fr = lambda: KI.chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, chunk_size=64, skip_wu=True)[4:]
    ff = (lambda: KF.chunk_kda_fwd_intra_pinned(q, k, g, beta, scale)) if args.arm == "2" else (lambda: KF.chunk_kda_fwd_intra_fused(q, k, g, beta, scale))
    Aqk0, Akk0 = fr(); Aqk1, Akk1 = ff()
    m = (torch.arange(64, device=dev)[None, :] <= (torch.arange(T, device=dev) % 64)[:, None])[None, :, None, :]
    z = torch.zeros((), device=dev, dtype=Aqk0.dtype)
    row = {"case": name, "T": T,
           "Aqk_mad": mad(torch.where(m, Aqk1, z), torch.where(m, Aqk0, z)),
           "Aqk_bitwise": torch.equal(torch.where(m, Aqk1, z), torch.where(m, Aqk0, z)),
           "Akk_mad": mad(Akk1, Akk0), "Akk_bitwise": torch.equal(Akk1, Akk0), "Akk_rel": rel(Akk1, Akk0),
           "Akk_neq_frac": (Akk1 != Akk0).float().mean().item()}

    def ck(a):
        os.environ["EXL3_KDA_INTRA"] = a
        s0 = d["initial_state"].clone() if d["initial_state"] is not None else None
        return F.chunk_kda(d["q"], d["k"], d["v"], g=d["g"], beta=d["beta"], initial_state=s0,
                           output_final_state=bool(d["output_final_state"]), use_qk_l2norm_in_kernel=bool(d["use_qk_l2norm_in_kernel"]))
    o0, s0 = ck("0"); o1, s1 = ck(args.arm)
    row.update({"o_mad": mad(o1, o0), "o_rel": rel(o1, o0), "o_bitwise": torch.equal(o1, o0),
                "state_rel": rel(s1, s0) if s0 is not None else None, "finite": bool(torch.isfinite(o1).all())})
    for rep in range(2):
        row[f"ref_ms{rep}"] = round(timeit(fr, args.iters), 3)
        row[f"fus_ms{rep}"] = round(timeit(ff, args.iters), 3)
        row[f"kda0_ms{rep}"] = round(timeit(lambda: ck("0"), args.iters), 3)
        row[f"kda1_ms{rep}"] = round(timeit(lambda: ck(args.arm), args.iters), 3)
    row["pick"] = [str(x) for x in KF.chunk_kda_fwd_kernel_intra_fused.cache.values()]
    if args.sweep:
        Aqk = torch.empty_like(Aqk0); Akk = torch.empty_like(Akk0); res = []
        for bk, bkd, nw, ns in itertools.product([32, 64, 128], [8, 16, 32], [1, 2, 4, 8], [1, 2]):
            run = lambda: FJ[(triton.cdiv(T, 64), B * HV)](q, k, g, beta, Aqk, Akk, scale, T, H=H, HV=HV, K=K, BT=64, BC=16,
                                                        BK=bk, BKD=bkd, num_warps=nw, num_stages=ns)
            try:
                res.append((round(timeit(run, args.iters), 3), bk, bkd, nw, ns, mad(Akk, Akk0)))
            except Exception as e:
                print(f"  cfg {bk} {bkd} {nw} {ns}: {type(e).__name__} {str(e)[:100]}", flush=True)
        row["top"] = sorted(res)[:6]
    out.append(row)
    print(json.dumps(row), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    del d, q, k, v, g, beta
    torch.cuda.empty_cache()
