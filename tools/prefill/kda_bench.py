"""GLM KDA prefill intra kernels in isolation: chunk_kda_fwd_kernel_inter_solve_fused and
recompute_w_u_fwd_kda_kernel at the GLM shape (T 2048, H = HV = 64, K = V = 128, BT 64, bf16
q/k/v, fp32 cumsum gates). Variants = values of one env knob, alternated per rep in one process;
outputs compared against the first variant and against an fp64 torch reference of Akk/w/u."""
import argparse, os, statistics, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
p = argparse.ArgumentParser()
p.add_argument("--T", type = int, default = 2048)
p.add_argument("--var", type = str, default = "EXL3_KDA_FP16DOT")
p.add_argument("--vals", type = str, nargs = "+", default = ["0", "1"])
p.add_argument("--reps", type = int, default = 9)
p.add_argument("--inner", type = int, default = 4)
p.add_argument("--meta", action = "store_true")
args = p.parse_args()

from exllamav3.vendor.fla import kda_chunk_intra as KI, kda_wy_fast as KW
from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
from exllamav3.vendor.fla.l2norm import l2norm_fwd

DEV = "cuda:0"; B = 1; T = args.T; H = 64; K = 128; V = 128; BT = 64
torch.manual_seed(0)
q = l2norm_fwd(torch.randn(B, T, H, K, device = DEV).bfloat16())[0]
k = l2norm_fwd(torch.randn(B, T, H, K, device = DEV).bfloat16())[0]
v = torch.randn(B, T, H, V, device = DEV).bfloat16()
g_raw = -F.softplus(torch.randn(B, T, H, K, device = DEV) * 2 + 2)   # log-decay, like the gate
g_raw = g_raw.clamp(min = -5.0)                                     # gate_lower_bound
g = chunk_local_cumsum(g_raw, chunk_size = BT, scale = 1.4426950408889634)
beta = torch.sigmoid(torch.randn(B, T, H, device = DEV)).bfloat16()
scale = K ** -0.5


def run():
    return KI.chunk_kda_fwd_intra(q = q, k = k, v = v, gk = g, beta = beta, scale = scale, chunk_size = BT)


def ref_akk():
    # Akk_inv = (I + tril(diag(beta) K_g K_g^T, -1))^-1 per chunk, gates relative (fp64)
    kk = k.double().view(B, T // BT, BT, H, K); gg = g.double().view(B, T // BT, BT, H, K)
    bb = beta.double().view(B, T // BT, BT, H)
    A = torch.einsum("bnihk,bnjhk->bnhij", kk * gg.exp2(), kk * (-gg).exp2())
    A = A * bb.permute(0, 1, 3, 2)[..., None]
    A = torch.tril(A, -1)
    I = torch.eye(BT, dtype = torch.float64, device = DEV)
    Ai = torch.linalg.inv(I + A)
    return Ai.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)


def timed(f):
    e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
    e0.record()
    for _ in range(args.inner): f()
    e1.record(); e1.synchronize(); return e0.elapsed_time(e1) / args.inner


Aref = ref_akk()
outs, ts = {}, {vv: {"intra": [], "wu": []} for vv in args.vals}
for vv in args.vals:
    os.environ[args.var] = vv
    for _ in range(3): o = run()
    outs[vv] = [x.float() if x is not None else None for x in o]
Akk0 = outs[args.vals[0]][5].to(k.dtype)
for _ in range(args.reps):
    for vv in args.vals:
        os.environ[args.var] = vv
        ts[vv]["intra"].append(timed(run))
        ts[vv]["wu"].append(timed(lambda: KW.recompute_w_u_fwd(k = k, v = v, beta = beta, A = Akk0, gk = g)))
names = ["w", "u", "qg", "kg", "Aqk", "Akk"]
b0 = statistics.median(ts[args.vals[0]]["intra"])
for vv in args.vals:
    o = outs[vv]; o0 = outs[args.vals[0]]
    diffs = " ".join(f"{n} {(a - b).abs().max().item():.1e}" for n, a, b in zip(names, o, o0) if a is not None)
    ea = ((o[5].double() - Aref).norm() / Aref.norm()).item()
    ti = statistics.median(ts[vv]["intra"]); tw = statistics.median(ts[vv]["wu"])
    print(f"kda T {T} {args.var}={vv:>6s}: intra {ti:6.3f} ms (x{b0 / ti:4.2f}) wu {tw:6.3f} ms  "
          f"Akk relL2 vs fp64 {ea:.2e}  maxabs vs[{args.vals[0]}]: {diffs}", flush = True)
if args.meta:
    for kern in (KI.chunk_kda_fwd_kernel_inter_solve_fused, KW.recompute_w_u_fwd_kda_kernel):
        fn = kern
        while hasattr(fn, "fn") and not hasattr(fn, "device_caches"):
            fn = fn.fn
        for dev, c in fn.device_caches.items():
            for key, ck in c[0].items():
                print("META", getattr(fn, "__name__", "?"), "regs", ck.n_regs, "spills", ck.n_spills,
                      "lds", ck.metadata.shared, "warps", ck.metadata.num_warps, flush = True)
        if hasattr(kern, "best_config"):
            print("BEST", getattr(fn, "__name__", "?"), kern.best_config, flush = True)
