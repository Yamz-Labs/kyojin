# hy4_ops_test.py: bit-exact checks of the Hy4 variants of the gm_ext decode ops against the stock fq kernels (Q3_K / Q4_K gate-up, routed clamp, expert-major
# order, Q8_0 projections, fused shared swiglu + down, generic router). Run under tools/gguf_moe/window.sh (GPU lock) with EXL3_GM_BUILD_DIR set.
import os, sys, torch
sys.path.insert(0, os.environ.get("EXL3_ROOT", os.getcwd()))
torch.set_grad_enabled(False)
from exllamav3 import gm_ext
from exllamav3.ext import exllamav3_ext as ext
dev = "cuda:0"; gm = gm_ext.get()
assert gm is not None
BB = {0: (144, 0), 8: (110, 108), 2: (210, 208), 3: (34, 0)}   # fmt -> (block bytes, offset of the fp16 d)
def rnd(lead, nblk, fmt, g):
    bb, dp = BB[fmt]
    t = torch.randint(0, 256, (*lead, nblk, bb), dtype=torch.uint8, device=dev, generator=g)
    d = ((torch.rand(*lead, nblk, 1, device=dev, generator=g) * 0.02 + 0.001) * DS[0]).half().view(torch.uint8)
    t[..., dp:dp + 2] = d
    if fmt == 0: t[..., 2:4] = (torch.rand(*lead, nblk, 1, device=dev, generator=g) * 0.02 * DS[0]).half().view(torch.uint8)
    return t.reshape(*lead, nblk * bb).contiguous()
FID = {0: 0, 8: 8, 2: 2}
DS = [1.0]
ok_all = True
def report(name, a, b):
    global ok_all
    eq = torch.equal(a, b); ok_all &= eq
    print(f"  {name}: {'BIT-EXACT' if eq else 'MISMATCH max diff %g, n diff %d' % ((a.float() - b.float()).abs().max().item(), (a != b).sum().item())}", flush=True)
g = torch.Generator(device=dev); g.manual_seed(1)

def moe_case(tag, E, K, I, S, fu, fd, lim, rows=(1, 2, 4, 8), pool=24):
    print(f"MoE {tag}: E={E} K={K} I={I} S={S} gate/up fmt {fu} down fmt {fd} lim {lim}", flush=True)
    gw = rnd((E, I), K // 256 * (1 if fu == 0 else 1), fu, g) if False else None
    nbu = K // 256; nbd = I // 256
    gate = rnd((E, I), nbu, fu, g); up = rnd((E, I), nbu, fu, g); down = rnd((E, K), nbd, fd, g)
    gs, ds = gate.shape[-1], down.shape[-1]
    mk = lambda t: torch.tensor([t[e].data_ptr() for e in range(E)], dtype=torch.int64, device=dev)
    gp, upp, dpp = mk(gate), mk(up), mk(down)
    for R in rows:
        torch.manual_seed(100 + R)
        h = (torch.randn(R, K, device=dev) * 0.7).half()
        sel = torch.stack([torch.randperm(E, device=dev)[:S] if pool is None else torch.randperm(pool, device=dev)[:S] for _ in range(R)]).long().contiguous()
        wt = torch.softmax(torch.randn(R, S, device=dev), -1).half().contiguous()
        part = torch.empty(R * S, K, dtype=torch.float, device=dev); ref = torch.empty(R, K, dtype=torch.float, device=dev)
        ext.fq_moe_kq_set_swiglu_limit(float(lim))
        ext.fq_moe_kq_grouped_mixed_q8(h, sel, wt.reshape(-1).contiguous() if False else wt, gp, upp, dpp, gs, gs, ds, FID[fu], FID[fd], part, ref)
        def run(meta):
            act = torch.empty(8 * S, I, dtype=torch.half, device=dev); qb = torch.empty(8 * K // 32 * 36, dtype=torch.uint8, device=dev)
            pt = torch.empty(8 * S, K, dtype=torch.float, device=dev); out = torch.empty(R, K, dtype=torch.float, device=dev)
            gm.gm_moe_kq_decode(h, sel.reshape(-1).contiguous(), wt.reshape(-1).contiguous(), gp, upp, dpp, gs, gs, ds, FID[fd], act, qb, pt, out,
                                torch.empty(0, dtype=torch.float, device=dev), FID[fu], float(lim), meta)
            return out
        report(f"R={R} plain vs stock", run(torch.empty(0, dtype=torch.int32, device=dev)), ref)
        if R >= 2:
            meta = torch.zeros(8 * S * 9, dtype=torch.int32, device=dev)
            o = run(meta)
            nuniq = int((meta.view(-1, 9)[:, 0] > 0).sum())
            report(f"R={R} expert-major ({nuniq} distinct of {R * S}) vs stock", o, ref)
    del gate, up, down

moe_case("Hy4 Q4_K/Q4_K/Q4_K", 512, 1536, 1536, 16, 0, 0, 10.0)
moe_case("Hy4 Q3_K/Q3_K/Q4_K", 512, 1536, 1536, 16, 8, 0, 10.0)
moe_case("Q3_K no clamp", 512, 1536, 1536, 16, 8, 0, 0.0, rows=(1, 4))
DS[0] = 0.02
moe_case("Qwen3 Q4_K/Q4_K/Q6_K", 128, 2048, 768, 8, 0, 2, 0.0, pool=20)

print("Q8_0 projections", flush=True)
for K, Ns in ((3072, (1024, 576, 8192)), (1536, (3072,)), (3072, (1536, 1536))):
    for R in (1, 4):
        torch.manual_seed(7 + R)
        ws3 = [rnd((N,), K // 32, 3, g) for N in Ns]
        ws = [w.view(N, K // 32 * 34) for w, N in zip(ws3, Ns)]
        x = (torch.randn(R, K, device=dev) * 0.8).half()
        qb = torch.empty(R, K // 32 * 36, dtype=torch.uint8, device=dev); ext.fq_quant_q8_1_rows(x, qb)
        for od in (torch.half, torch.float):
            ref = torch.cat([(lambda o: (ext.fq_gemv_q8_0_q8(qb, w.view(N, K // 32, 34), o), o)[1])(torch.empty(R, N, dtype=od, device=dev)).reshape(-1) for w, N in zip(ws, Ns)])
            out = torch.empty(R * sum(Ns), dtype=od, device=dev); sc = torch.empty(8 * K // 32 * 36, dtype=torch.uint8, device=dev)
            gm.gm_qkv_decode(x, ws, [3] * len(ws), out.view(1, -1), sc)
            report(f"K={K} N={Ns} R={R} out {od} half-x", out, ref)
        xf = (torch.randn(R, K, device=dev) * 0.8)
        qbf = torch.empty(R, K // 32 * 36, dtype=torch.uint8, device=dev); ext.fq_quant_q8_1_rows(xf, qbf)
        ref = torch.cat([(lambda o: (ext.fq_gemv_q8_0_q8(qbf, w.view(N, K // 32, 34), o), o)[1])(torch.empty(R, N, dtype=torch.float, device=dev)).reshape(-1) for w, N in zip(ws, Ns)])
        out = torch.empty(R * sum(Ns), dtype=torch.float, device=dev)
        gm.gm_qkv_decode(xf, ws, [3] * len(ws), out.view(1, -1), sc)
        report(f"K={K} N={Ns} R={R} out float float-x", out, ref)

print("Q8_0 swiglu + down (+ accumulate)", flush=True)
I, N = 1536, 3072
for R in (1, 4):
    for gd in (torch.half, torch.float):
        wd = rnd((N,), I // 32, 3, g).view(N, I // 32 * 34)
        gg = (torch.randn(R, I, device=dev) * 2).to(gd); uu = (torch.randn(R, I, device=dev)).to(gd)
        act = torch.nn.functional.silu(gg.float()) * uu.float()       # fp32 product (Hy4 shared expert), quantized from fp32
        qb = torch.empty(R, I // 32 * 36, dtype=torch.uint8, device=dev); ext.fq_quant_q8_1_rows(act, qb)
        ref = torch.empty(R, N, dtype=torch.float, device=dev); ext.fq_gemv_q8_0_q8(qb, wd.view(N, I // 32, 34), ref)
        out = torch.empty(R, N, dtype=torch.float, device=dev); gm.gm_q8_swiglu_down(gg, uu, wd, out, False)
        report(f"R={R} g/u {gd} overwrite", out, ref)
        base = torch.randn(R, N, device=dev); o2 = base.clone(); gm.gm_q8_swiglu_down(gg, uu, wd, o2, True)
        report(f"R={R} g/u {gd} accumulate", o2, base + ref)

print("Router", flush=True)
for (H, E, k, gating, norm, scale, xd, gd, bias) in ((3072, 512, 16, 2, True, 3.998, torch.float, torch.float, True), (3072, 512, 16, 2, True, 3.998, torch.half, torch.float, True),
                                                      (2048, 128, 8, 0, True, 1.0, torch.half, torch.half, False), (2816, 128, 8, 0, False, 1.0, torch.half, torch.half, False),
                                                      (2880, 32, 4, 0, True, 1.0, torch.half, torch.float, False), (2048, 64, 6, 2, True, 1.8, torch.half, torch.float, True)):
    for R in (1, 4, 8):
        x = (torch.randn(R, H, device=dev) * 1.0).to(xd); gate = (torch.randn(E, H, device=dev) * 0.05).to(gd)
        b = (torch.randn(E, device=dev) * 0.1) if bias else torch.empty(0, device=dev)
        sc = torch.empty(R, E, dtype=torch.float, device=dev); sel = torch.empty(R, k, dtype=torch.long, device=dev); w = torch.empty(R, k, dtype=torch.half, device=dev)
        gm.gm_route(x.contiguous(), gate.contiguous(), b, sc, sel, w, gating, norm, scale, 6.103515625e-5)
        lg = x.float() @ gate.float().T
        if gating == 2:
            p = torch.sigmoid(lg); key = p + (b if bias else 0)
            rs = torch.sort(key, dim=-1, descending=True, stable=True).indices[:, :k]; rw = torch.gather(p, 1, rs)
            if norm: rw = rw / rw.sum(-1, keepdim=True).clamp(min=6.103515625e-5)
        else:
            p = torch.softmax(lg, -1); rs = torch.sort(p, dim=-1, descending=True, stable=True).indices[:, :k]; rw = torch.gather(p, 1, rs)
            if norm: rw = rw / rw.sum(-1, keepdim=True)
        rw = rw * scale
        same = torch.equal(sel, rs); err = ((w.float() - rw).abs() / rw.abs().clamp(min=1e-6)).max().item()
        ok_all &= same and err < 2e-3
        print(f"  H={H} E={E} k={k} gating={gating} norm={norm} R={R} x={xd} gate={gd}: sel {'same' if same else 'DIFFERENT'}, max rel w err {err:.2e}", flush=True)
print("ALL BIT-EXACT / OK" if ok_all else "SOME CHECKS FAILED")
