# KDA chunk-chain kernel sweep (synthetic GLM shapes, H=64, D=128): times the fla kernels the fusion replaces
# (recompute_w_u, fwd_h, gla_o) and the fused h kernel per config (BV, num_warps, num_stages, FUSE_O), autotune bypassed.
import os, sys, itertools, argparse, torch, triton
sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)
from exllamav3.vendor.fla import RCP_LN2
from exllamav3.vendor.fla.l2norm import l2norm_fwd
from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
from exllamav3.vendor.fla.kda_chunk_intra import chunk_kda_fwd_intra
from exllamav3.vendor.fla.kda_wy_fast import recompute_w_u_fwd
from exllamav3.vendor.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from exllamav3.vendor.fla.gla_chunk import chunk_gla_fwd_o_gk
from exllamav3.vendor.fla.kda_fused_h import chunk_kda_fwd_kernel_h_fused as HK

ap = argparse.ArgumentParser()
ap.add_argument("--T", default="255,2048,4096")
ap.add_argument("--bv", default="16,32,64,128")
ap.add_argument("--warps", default="1,2,4,8")
ap.add_argument("--stages", default="1,2")
ap.add_argument("--iters", type=int, default=10)
args = ap.parse_args()
dev = "cuda:0"
JIT = HK.fn.fn


def timeit(fn, it):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


for T in [int(x) for x in args.T.split(",")]:
    g_ = torch.Generator(device=dev).manual_seed(0)
    r = lambda *s: torch.randn(*s, device=dev, generator=g_)
    H, D, BT = 64, 128, 64
    q, _ = l2norm_fwd(r(1, T, H, D).bfloat16()); k, _ = l2norm_fwd(r(1, T, H, D).bfloat16())
    v = r(1, T, H, D).bfloat16(); beta = torch.sigmoid(r(1, T, H)).bfloat16()
    g = chunk_local_cumsum((-5.0 * torch.sigmoid(r(1, T, H, D) - 2.0)).float(), chunk_size=BT, scale=RCP_LN2)
    h0 = (0.1 * r(1, H, D, D)).float(); scale = D ** -0.5
    _, _, _, _, Aqk, Akk = chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, chunk_size=BT, skip_wu=True)
    w, u, _, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Akk, gk=g)
    h, vn, _ = chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=h0, output_final_state=True, chunk_size=BT)
    t_wu = timeit(lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=Akk, gk=g), args.iters)
    t_h = timeit(lambda: chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=h0, output_final_state=True, chunk_size=BT), args.iters)
    t_o = timeit(lambda: chunk_gla_fwd_o_gk(q=q, v=vn, g=g, A=Aqk, h=h, scale=scale, chunk_size=BT), args.iters)
    t_intra = timeit(lambda: chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, chunk_size=BT, skip_wu=True), args.iters)
    print(f"T={T}: fla recompute_w_u {t_wu:.3f} ms, fwd_h {t_h:.3f}, gla_o {t_o:.3f} (sum {t_wu + t_h + t_o:.3f}); intra w/o wu {t_intra:.3f}", flush=True)
    NT = triton.cdiv(T, BT)
    hb = k.new_empty(1, NT, H, D, D); vnb = torch.empty_like(v); ob = torch.empty_like(v); ht = torch.zeros(1, H, D, D, device=dev)
    res = []
    for fo, bv, nw, ns in itertools.product((0, 1), *[[int(x) for x in a.split(",")] for a in (args.bv, args.warps, args.stages)]):
        def run():
            JIT[(triton.cdiv(D, bv), H)](q, k, v, g, beta, Akk, Aqk, None if fo else hb, None if fo else vnb, ob if fo else None,
                                          h0, ht, scale, T, H=H, HV=H, K=D, V=D, BT=BT, BV=bv, FUSE_O=bool(fo),
                                          USE_INITIAL_STATE=True, STORE_FINAL_STATE=True, num_warps=nw, num_stages=ns)
        try:
            res.append((timeit(run, args.iters), fo, bv, nw, ns))
        except Exception as e:
            print(f"  cfg fo{fo} bv{bv} w{nw} s{ns}: {type(e).__name__} {str(e)[:120]}", flush=True)
    for fo in (0, 1):
        rr = sorted(x for x in res if x[1] == fo)[:4]
        print(f"  fused FUSE_O={fo}: " + ", ".join(f"{t:.3f} ms (BV{bv} w{nw} s{ns})" for t, _, bv, nw, ns in rr), flush=True)
