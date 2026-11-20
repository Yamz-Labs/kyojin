# hy4_ops_bench.py: gm_moe_kq_decode at Hy4 shapes (E=512 K=I=1536 S=16): stock vs gm plain vs gm expert-major, per layer ms (events, 50 reps); plus lim sensitivity check.
import os, sys, torch
sys.path.insert(0, os.environ.get("EXL3_ROOT", os.getcwd()))
torch.set_grad_enabled(False)
from exllamav3 import gm_ext
from exllamav3.ext import exllamav3_ext as ext
dev = "cuda:0"; gm = gm_ext.get(); g = torch.Generator(device=dev); g.manual_seed(3)
BB = {0: 144, 8: 110}
def rnd(lead, nblk, fmt):
    bb = BB[fmt]; t = torch.randint(0, 256, (*lead, nblk, bb), dtype=torch.uint8, device=dev, generator=g)
    t[..., (0 if fmt == 0 else 108):(2 if fmt == 0 else 110)] = (torch.rand(*lead, nblk, 1, device=dev, generator=g) * 0.02 + 0.001).half().view(torch.uint8)
    if fmt == 0: t[..., 2:4] = (torch.rand(*lead, nblk, 1, device=dev, generator=g) * 0.02).half().view(torch.uint8)
    return t.reshape(*lead, nblk * bb).contiguous()
E, K, I, S = 512, 1536, 1536, 16
def tm(f, n=40):
    for _ in range(5): f()
    torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n
for fu in (0, 8):
    gate, up, down = rnd((E, I), K // 256, fu), rnd((E, I), K // 256, fu), rnd((E, K), I // 256, 0)
    mk = lambda t: torch.tensor([t[e].data_ptr() for e in range(E)], dtype=torch.int64, device=dev)
    gp, upp, dpp = mk(gate), mk(up), mk(down); gs, ds = gate.shape[-1], down.shape[-1]
    for R in (1, 2, 4, 8):
        h = (torch.randn(R, K, device=dev) * 0.7).half()
        sel = torch.stack([torch.randperm(E, device=dev)[:S] for _ in range(R)]).long().contiguous()
        if R > 1:   # Hy4 verify-like overlap: later rows share ~25 percent of the first row's experts
            for r in range(1, R): sel[r, :4] = sel[0, :4]
        wt = torch.softmax(torch.randn(R, S, device=dev), -1).half().contiguous()
        part = torch.empty(R * S, K, dtype=torch.float, device=dev); ref = torch.empty(R, K, dtype=torch.float, device=dev)
        act = torch.empty(8 * S, I, dtype=torch.half, device=dev); qb = torch.empty(8 * K // 32 * 36, dtype=torch.uint8, device=dev)
        pt = torch.empty(8 * S, K, dtype=torch.float, device=dev); out = torch.empty(R, K, dtype=torch.float, device=dev)
        meta = torch.zeros(8 * S * 9, dtype=torch.int32, device=dev); nometa = torch.empty(0, dtype=torch.int32, device=dev); nr = torch.empty(0, dtype=torch.float, device=dev)
        ext.fq_moe_kq_set_swiglu_limit(10.0)
        f_stock = lambda: ext.fq_moe_kq_grouped_mixed_q8(h, sel, wt, gp, upp, dpp, gs, gs, ds, fu, 0, part, ref)
        f_gm = lambda m: (lambda: gm.gm_moe_kq_decode(h, sel.reshape(-1), wt.reshape(-1), gp, upp, dpp, gs, gs, ds, 0, act, qb, pt, out, nr, fu, 10.0, m))
        wb = (len(set(sel.flatten().tolist())) * (2 * gs * I + ds * K))
        r = [f"gate/up fmt {fu} R={R}: distinct experts {len(set(sel.flatten().tolist()))}/{R * S}", f"stock {tm(f_stock):.3f} ms", f"gm plain {tm(f_gm(nometa)):.3f} ms"]
        if R > 1: r.append(f"gm expert-major {tm(f_gm(meta)):.3f} ms")
        r.append(f"(unique weight bytes {wb / 1e6:.0f} MB = {wb / 215e9 * 1e3:.3f} ms at 215 GB/s)")
        print("  " + "  ".join(r), flush=True)
# clamp sensitivity: lim 0.3 must change the result vs 10, and equal the stock kernel
R = 2; h = (torch.randn(R, K, device=dev) * 0.7).half(); sel = torch.stack([torch.randperm(E, device=dev)[:S] for _ in range(R)]).long().contiguous(); wt = torch.softmax(torch.randn(R, S, device=dev), -1).half().contiguous()
part = torch.empty(R * S, K, dtype=torch.float, device=dev); res = {}
for lim in (0.0, 0.3, 10.0):
    ext.fq_moe_kq_set_swiglu_limit(lim); o = torch.empty(R, K, dtype=torch.float, device=dev)
    ext.fq_moe_kq_grouped_mixed_q8(h, sel, wt, gp, upp, dpp, gs, gs, ds, 8, 0, part, o); res[lim] = o
    act = torch.empty(8 * S, I, dtype=torch.half, device=dev); qb = torch.empty(8 * K // 32 * 36, dtype=torch.uint8, device=dev); pt = torch.empty(8 * S, K, dtype=torch.float, device=dev); o2 = torch.empty(R, K, dtype=torch.float, device=dev)
    gm.gm_moe_kq_decode(h, sel.reshape(-1), wt.reshape(-1), gp, upp, dpp, gs, gs, ds, 0, act, qb, pt, o2, torch.empty(0, device=dev), 8, lim, torch.empty(0, dtype=torch.int32, device=dev))
    print(f"  lim {lim}: gm == stock {torch.equal(o, o2)}", flush=True)
print("  lim 0.3 differs from lim 0:", not torch.equal(res[0.0], res[0.3]))
