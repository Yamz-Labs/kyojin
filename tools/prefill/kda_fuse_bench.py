# Step 10 KDA fusion microbench: chunk_kda with EXL3_KDA_FUSE=0 (fla chain) vs 1 (w/u/kg fused) vs 2 (+ o fused).
# Inputs: scratch/step10/cap/*.pt (captured GLM calls) or synthetic (--synth T,...). Prints relL2, bitwise, ms/call.
import os, sys, glob, json, time, argparse, torch
sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)
import exllamav3.vendor.fla as F

ap = argparse.ArgumentParser()
ap.add_argument("--synth", default=None, help="comma list of T for synthetic GLM-shaped inputs (H=64, D=128)")
ap.add_argument("--caps", default="scratch/step10/cap")
ap.add_argument("--arms", default="0,1,2")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("-o", "--out", default=None)
args = ap.parse_args()
dev = "cuda:0"


def synth(T, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=g)
    H, D = 64, 128
    return {"q": r(1, T, H, D).bfloat16(), "k": r(1, T, H, D).bfloat16(), "v": r(1, T, H, D).bfloat16(),
            "g": (-5.0 * torch.sigmoid(r(1, T, H, D) - 2.0)).float(), "beta": torch.sigmoid(r(1, T, H)).bfloat16(),
            "initial_state": (0.1 * r(1, H, D, D)).float(), "output_final_state": True, "use_qk_l2norm_in_kernel": True}


cases = []
if args.synth:
    for T in args.synth.split(","):
        cases.append((f"synth{T}", synth(int(T))))
else:
    for p in sorted(glob.glob(f"{args.caps}/*.pt")):
        d = torch.load(p)
        cases.append((os.path.basename(p)[:-3], {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in d.items()}))


def call(d, arm):
    os.environ["EXL3_KDA_FUSE"] = arm
    s0 = d["initial_state"].clone() if d["initial_state"] is not None else None
    return F.chunk_kda(d["q"], d["k"], d["v"], g=d["g"], beta=d["beta"], initial_state=s0,
                       output_final_state=bool(d["output_final_state"]), use_qk_l2norm_in_kernel=bool(d["use_qk_l2norm_in_kernel"]))


def rel(a, b):
    if a is None or b is None:
        return None
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def timeit(fn, it):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


res = []
arms = args.arms.split(",")
for name, d in cases:
    T = d["q"].shape[1]
    ref_o, ref_s = call(d, "0")
    ref_o2, _ = call(d, "0")
    row = {"case": name, "T": T, "ref_repeat_bitwise": torch.equal(ref_o, ref_o2)}
    for a in arms:
        o, s = call(d, a)
        row[f"a{a}"] = {"relL2_o": rel(o, ref_o), "relL2_state": rel(s, ref_s), "bitwise_o": torch.equal(o, ref_o),
                        "bitwise_state": (s is None and ref_s is None) or (s is not None and torch.equal(s, ref_s)),
                        "finite": bool(torch.isfinite(o).all())}
    for rep in range(2):  # interleaved A B C A B C
        for a in arms:
            row[f"a{a}"][f"ms{rep}"] = round(timeit(lambda: call(d, a), args.iters), 3)
    for a in arms:
        row[f"a{a}"]["ms"] = min(row[f"a{a}"]["ms0"], row[f"a{a}"]["ms1"])
    res.append(row)
    print(json.dumps(row), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)

try:
    from exllamav3.vendor.fla.kda_fused_h import chunk_kda_fwd_kernel_h_fused as K
    print("fused autotune picks:", {str(k): str(v) for k, v in K.fn.cache.items()}, flush=True)
except Exception as e:
    print("autotune picks n/a", repr(e))
