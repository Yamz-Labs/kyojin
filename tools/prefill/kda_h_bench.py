# Step 12: fused_h (chunk_kda_fwd_kernel_h_fused) microbench on captured GLM inputs. Intra runs once per case;
# then the fused state loop alone is timed per config / variant, with regs/spills and bitwise o/state vs default.
import os, sys, json, itertools, argparse, torch, triton
sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)
from exllamav3.vendor.fla import RCP_LN2
from exllamav3.vendor.fla.l2norm import l2norm_fwd
from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
from exllamav3.vendor.fla.kda_intra_fused import chunk_kda_fwd_intra_pinned
import exllamav3.vendor.fla.kda_fused_h as KH

ap = argparse.ArgumentParser()
ap.add_argument("--caps", default="scratch/step10/cap")
ap.add_argument("--cases", default="t4096")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--cfgs", default="128:8:2", help="BV:warps:stages list, comma separated")
ap.add_argument("--variants", default="base")
ap.add_argument("--asm", default=None, help="dump amdgcn of the first cfg to this path")
ap.add_argument("--heads", default="0", help="comma list: slice the capture to the first N heads (0 = all)")
ap.add_argument("-o", "--out", default=None)
args = ap.parse_args()
dev = "cuda:0"


def timeit(fn, it):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


res = []
for case, nh in [(c, int(n)) for c in args.cases.split(",") for n in args.heads.split(",")]:
    d = torch.load(f"{args.caps}/{case}.pt")
    d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    if nh:
        for kk in ("q", "k", "v", "g", "beta"):
            d[kk] = d[kk][:, :, :nh].contiguous()
        if d["initial_state"] is not None:
            d["initial_state"] = d["initial_state"][:, :nh].contiguous()
    q, k, v, beta = d["q"], d["k"], d["v"], d["beta"]
    if d["use_qk_l2norm_in_kernel"]:
        q, _ = l2norm_fwd(q); k, _ = l2norm_fwd(k)
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    scale = K ** -0.5
    g = chunk_local_cumsum(d["g"], chunk_size=64, scale=RCP_LN2)
    Aqk, Akk = chunk_kda_fwd_intra_pinned(q=q, k=k, gk=g, beta=beta, scale=scale, chunk_size=64)
    h0 = d["initial_state"]
    print(f"{case}: T={T} H={H} HV={HV} K={K} V={V} dtypes q {q.dtype} g {g.dtype} beta {beta.dtype} Akk {Akk.dtype} "
          f"Aqk {Aqk.dtype} h0 {None if h0 is None else h0.dtype}", flush=True)
    _, _, ref_o, ref_s = KH.chunk_kda_fwd_h_fused(q, k, v, g, beta, Akk, Aqk, scale, initial_state=h0,
                                                   output_final_state=True, fuse_o=True)
    # floor: FLOPs and bytes of the fused loop (o fused; state in registers, h0/ht once)
    NT = triton.cdiv(T, 64)
    fl = 2 * NT * HV * (64 * 64 * K + 64 * K * V + 64 * K * V + 64 * 64 * V + 64 * 64 * V + K * 64 * V)
    by = sum(t.numel() * t.element_size() for t in (q, k, v, g, beta, Akk, Aqk)) + ref_o.numel() * ref_o.element_size()
    by += 2 * HV * K * V * 4
    print(f"  floor: {fl / 1e9:.2f} GFLOP -> {fl / 47e12 * 1e3:.3f} ms @47TF; {by / 1e6:.1f} MB -> {by / 230e9 * 1e3:.3f} ms @230GB/s", flush=True)
    for var in args.variants.split(","):
        # variant "name[:KEY=int...]": base = the autotuned kernel's jit, else KH.chunk_kda_fwd_kernel_h_<name>
        vn_, *kv = var.split(":")
        extra = {a.split("=")[0]: int(a.split("=")[1]) for a in kv}
        if vn_ == "hip":  # hand-written HIP kernel (exllamav3/vendor/fla/hip), no Triton config
            from exllamav3.vendor.fla.hip.kda_fused_h_hip import chunk_kda_fwd_h_fused_hip
            o = torch.empty_like(v); ht = k.new_zeros(B, HV, K, V, dtype=torch.float32)
            run = lambda: chunk_kda_fwd_h_fused_hip(q, k, v, g, beta, Akk, Aqk, scale, initial_state=h0,
                                                    output_final_state=True, o=o, final_state=ht,
                                                    defs=" ".join(f"{a}={b}" for a, b in extra.items()))
            run(); torch.cuda.synchronize()
            ms = min(timeit(run, args.iters), timeit(run, args.iters))
            dif = (o.float() - ref_o.float()).abs()
            row = {"case": case, "heads": HV, "var": var, "cfg": "-", "ms": round(ms, 3), "bit_o": torch.equal(o, ref_o),
                   "bit_s": torch.equal(ht, ref_s), "n_diff_o": int((o != ref_o).sum()), "max_o": dif.max().item(),
                   "rel_o": ((o.float() - ref_o.float()).norm() / ref_o.float().norm()).item(),
                   "rel_s": ((ht - ref_s).norm() / ref_s.norm()).item()}
            if extra.get("PROF"):  # per-phase shader-cycle share, mean over WGs and waves, per chunk
                from exllamav3.vendor.fla.hip.kda_fused_h_hip import read_prof
                run(); pr = read_prof(" ".join(f"{a}={b}" for a, b in extra.items()), B * HV).astype("float64")
                ph = pr.mean(axis=(0, 1)); row["prof_cyc_per_chunk"] = [round(x / NT) for x in ph]
                row["prof_share"] = [round(x / ph.sum(), 3) for x in ph]
                row["prof_wave_max_share"] = [round(x, 3) for x in (pr.max(axis=1).mean(axis=0) / ph.sum())]
            res.append(row); print("  " + json.dumps(row), flush=True)
            if args.out:
                json.dump(res, open(args.out, "w"), indent=1)
            continue
        kern = KH.chunk_kda_fwd_kernel_h_fused.fn.fn if vn_ == "base" else getattr(KH, "chunk_kda_fwd_kernel_h_" + vn_)
        for cfg in args.cfgs.split(","):
            bv, nw, ns = (int(x) for x in cfg.split(":"))
            o = torch.empty_like(v); ht = k.new_zeros(B, HV, K, V, dtype=torch.float32)
            def run():
                return kern[(triton.cdiv(V, bv), B * HV)](q, k, v, g, beta, Akk, Aqk, None, None, o, h0, ht, scale, T,
                    H=H, HV=HV, K=K, V=V, BT=64, BV=bv, FUSE_O=True, USE_INITIAL_STATE=h0 is not None,
                    STORE_FINAL_STATE=True, num_warps=nw, num_stages=ns, **extra)
            try:
                ck = run(); torch.cuda.synchronize()
                ms = min(timeit(run, args.iters), timeit(run, args.iters))
            except Exception as e:
                print(f"  {var} {cfg}: {type(e).__name__} {str(e)[:200]}", flush=True); continue
            md = ck.metadata if hasattr(ck, "metadata") else None
            regs = getattr(ck, "n_regs", None); sp = getattr(ck, "n_spills", None)
            row = {"case": case, "heads": HV, "var": var, "cfg": cfg, "ms": round(ms, 3), "regs": regs, "spills": sp,
                   "lds": getattr(md, "shared", None), "bit_o": torch.equal(o, ref_o), "bit_s": torch.equal(ht, ref_s),
                   "rel_o": ((o.float() - ref_o.float()).norm() / ref_o.float().norm()).item()}
            res.append(row); print("  " + json.dumps(row), flush=True)
            if args.asm and cfg == args.cfgs.split(",")[0]:
                open(f"{args.asm}.{var.replace(':', '_')}.s", "w").write(ck.asm.get("amdgcn", ""))
            if args.out:
                json.dump(res, open(args.out, "w"), indent=1)
