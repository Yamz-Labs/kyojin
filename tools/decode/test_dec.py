#!/usr/bin/env python
"""Unit tests + microbenchmarks for the exl3_dec kernels on synthetic trellis data.

  test_dec.py [--ext dev|main] [--bench] [--moe]

Correctness: kernel vs CPU reference (ref/frac_reconstruct.py) for K in 2, 2.5, 3, 4, 6, rel-L2.
Speed: event-timed median with an L2/MALL flush between iterations, GB/s of trellis streamed.
"""
import argparse
import os
import sys
import time

import torch

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "ref"))
import frac_reconstruct as fr  # noqa: E402


def load_ext(which):
    if which != "main":
        import importlib
        sys.path.insert(0, os.path.join(WT, "build/devext", which))
        m = importlib.import_module(which)
        return dict(gemv = m.gemv, gemv_multi = m.gemv_multi, moe = m.moe, router = getattr(m, "router", None), rms_norm = getattr(m, "rms_norm", None),
                    gemv_strided = getattr(m, "gemv_strided", None), router_norm = getattr(m, "router_norm", None))
    import exllamav3_ext as m
    return dict(gemv = m.exl3_dec_gemv, gemv_multi = m.exl3_dec_gemv_multi, moe = m.exl3_dec_moe, router = m.exl3_dec_router, rms_norm = getattr(m, "exl3_dec_rms_norm", None),
                gemv_strided = getattr(m, "exl3_dec_gemv_strided", None), router_norm = getattr(m, "exl3_dec_router_norm", None))


def rand_trellis(k, n, K, gen):
    words16 = int(round(16 * K))
    return torch.randint(-32768, 32767, (k // 16, n // 16, words16), dtype = torch.int16, generator = gen)


def rand_sign_scale(n, gen):
    # suh/svh in real packs are signed per-channel scales, not only +-1
    s = torch.randn(n, generator = gen).abs() * 0.5 + 0.5
    sign = torch.randint(0, 2, (n,), generator = gen) * 2 - 1
    return (s * sign).half()


def ref_gemv(x, tr, suh, svh, K):
    W = fr.reconstruct(tr, K, fr.CB_MUL1, suh, svh, had = True, out_dtype = torch.float32)
    return x.float() @ W.float()


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def workspace(dev):
    scratch = torch.zeros(4 << 20, dtype = torch.float, device = dev)
    counters = torch.zeros(4096, dtype = torch.int, device = dev)
    return scratch, counters


def test_gemv(ext, dev):
    gen = torch.Generator().manual_seed(1)
    scratch, counters = workspace(dev)
    ok = True
    for K in (2, 2.5, 3, 4, 6):
        for (k, n) in ((512, 768), (1024, 1536)):
            tr = rand_trellis(k, n, K, gen)
            suh = rand_sign_scale(k, gen)
            svh = rand_sign_scale(n, gen)
            x = (torch.randn(1, k, generator = gen) * 0.5).half()
            ref = ref_gemv(x, tr, suh, svh, K)
            outs = []
            for dt in (torch.half, torch.float):
                out = torch.empty(1, n, dtype = dt, device = dev)
                ext["gemv"](x.to(dev), tr.to(dev), suh.to(dev), svh.to(dev), out, scratch, counters, float(K))
                outs.append(out.cpu())
            e = rel_l2(outs[1], ref)
            eh = rel_l2(outs[0], ref)
            ctr = int(counters.abs().sum())
            good = e < 2e-3 and eh < 3e-3 and ctr == 0
            ok &= good
            print(f"gemv K={K:<4} {k}x{n}: rel-L2 fp32 {e:.2e} fp16 {eh:.2e} counters {ctr} {'OK' if good else 'FAIL'}")
    return ok


def test_ktw(ext, dev):
    """Large k with every forced k-tiles-per-wave value (ring and cross-block reduction paths)."""
    gen = torch.Generator().manual_seed(5)
    scratch, counters = workspace(dev)
    ok = True
    for K in (2, 2.5, 4):
        k, n = 4096, 1024
        tr = rand_trellis(k, n, K, gen)
        suh = rand_sign_scale(k, gen)
        svh = rand_sign_scale(n, gen)
        x = (torch.randn(1, k, generator = gen) * 0.5).half()
        ref = ref_gemv(x, tr, suh, svh, K)
        for ktw in (0, 4, 8, 16, 32):
            os.environ["EXL3_DEC_KTW"] = str(ktw)
            out = torch.empty(1, n, dtype = torch.float, device = dev)
            ext["gemv"](x.to(dev), tr.to(dev), suh.to(dev), svh.to(dev), out, scratch, counters, float(K))
            e = rel_l2(out.cpu(), ref)
            good = e < 2e-3 and int(counters.abs().sum()) == 0
            ok &= good
            print(f"gemv K={K} {k}x{n} ktw={ktw}: rel-L2 {e:.2e} {'OK' if good else 'FAIL'}")
        os.environ.pop("EXL3_DEC_KTW")
    return ok


def test_kbs(ext, dev):
    """Ragged k split: forced blocks-per-strip values, including odd per-wave tile counts."""
    gen = torch.Generator().manual_seed(6)
    scratch, counters = workspace(dev)
    ok = True
    for K in (2, 3, 4):
        k, n = 4096, 1536
        tr = rand_trellis(k, n, K, gen)
        suh = rand_sign_scale(k, gen)
        svh = rand_sign_scale(n, gen)
        x = (torch.randn(1, k, generator = gen) * 0.5).half()
        ref = ref_gemv(x, tr, suh, svh, K)
        for kbs in (1, 3, 5, 7, 12, 32):
            os.environ["EXL3_DEC_KBS"] = str(kbs)
            out = torch.empty(1, n, dtype = torch.float, device = dev)
            ext["gemv"](x.to(dev), tr.to(dev), suh.to(dev), svh.to(dev), out, scratch, counters, float(K))
            e = rel_l2(out.cpu(), ref)
            good = e < 2e-3 and int(counters.abs().sum()) == 0
            ok &= good
            print(f"gemv K={K} {k}x{n} kbs={kbs}: rel-L2 {e:.2e} {'OK' if good else 'FAIL'}")
        os.environ.pop("EXL3_DEC_KBS")
    if ext.get("gemv_strided") is not None:
        heads, hd, K = 64, 192, 4
        k, n = heads * 128, 4096
        tr = rand_trellis(k, n, K, gen)
        suh = rand_sign_scale(k, gen)
        svh = rand_sign_scale(n, gen)
        xf = (torch.randn(1, heads * hd, generator = gen) * 0.5).half()
        ref = ref_gemv(xf.view(heads, hd)[:, :128].reshape(1, -1), tr, suh, svh, K)
        out = torch.empty(1, n, dtype = torch.float, device = dev)
        ext["gemv_strided"](xf.to(dev), tr.to(dev), suh.to(dev), svh.to(dev), out, scratch, counters, float(K), k, hd)
        e = rel_l2(out.cpu(), ref)
        good = e < 2e-3 and int(counters.abs().sum()) == 0
        ok &= good
        print(f"gemv strided {heads}x{hd}->128 x {n}: rel-L2 {e:.2e} {'OK' if good else 'FAIL'}")
    for a_, b_ in ((3, 3), (5, 2), (1, 1)):
        os.environ["EXL3_DEC_MOE_KBS_A"] = str(a_)
        os.environ["EXL3_DEC_MOE_KBS_B"] = str(b_)
        print(f"moe kbs A={a_} B={b_}:")
        ok &= test_moe(ext, dev)
    os.environ.pop("EXL3_DEC_MOE_KBS_A")
    os.environ.pop("EXL3_DEC_MOE_KBS_B")
    return ok


def test_router_norm(ext, dev):
    """Fused MLP pre-norm + router vs rms_norm_res_in fallback + torch routing."""
    sys.path.insert(0, WT)
    from exllamav3 import ext_fallbacks as FB
    gen = torch.Generator().manual_seed(8)
    scratch, counters = workspace(dev)
    ok = True
    for (H, E, k, xdt) in ((4096, 256, 8, torch.half), (4096, 256, 8, torch.float), (2048, 128, 6, torch.half)):
        xa = (torch.randn(1, H, generator = gen) * 2).to(xdt).to(dev)
        r = (torch.randn(1, H, generator = gen) * 5).to(dev)
        w = (torch.randn(H, generator = gen) * 0.3 + 1).bfloat16().to(dev)
        G = (torch.randn(H, E, generator = gen) * 0.02).half().to(dev)
        bias = (torch.randn(E, generator = gen) * 0.1).to(dev)
        y_ref = torch.empty(1, H, dtype = torch.half, device = dev)
        r_ref = r.clone()
        FB.rms_norm_res_in(xa, w, y_ref, r_ref, 1e-6, 0.0, 1.0)
        y = torch.empty(1, H, dtype = torch.half, device = dev)
        sel = torch.empty(1, k, dtype = torch.long, device = dev)
        wt = torch.empty(1, k, dtype = torch.half, device = dev)
        ext["router_norm"](xa, r, w, 1e-6, 0.0, 1.0, y, G, bias, sel, wt, scratch, counters, 2.5)
        ydiff = int((y.float() != y_ref.float()).sum())
        rdiff = (r - r_ref).abs().max().item()
        logits = (y_ref.float() @ G.float()).half().float()
        sc = torch.sigmoid(logits)
        sel_ref = torch.topk(sc + bias, k, dim = -1).indices
        s_ok = sorted(sel.cpu()[0].tolist()) == sorted(sel_ref.cpu()[0].tolist())
        good = ydiff <= H // 500 and rdiff == 0.0 and s_ok and int(counters.abs().sum()) == 0
        ok &= good
        print(f"router_norm H={H} E={E} xa={str(xdt)[6:]}: y mismatches {ydiff}/{H}, r max diff {rdiff:.1e}, selection {'match' if s_ok else 'MISMATCH'} {'OK' if good else 'FAIL'}")
    return ok


def test_router(ext, dev):
    gen = torch.Generator().manual_seed(6)
    scratch, counters = workspace(dev)
    ok = True
    for (H, E, k) in ((4096, 256, 8), (2048, 128, 6)):
        x = (torch.randn(1, H, generator = gen)).half()
        G = (torch.randn(H, E, generator = gen) * 0.02).half()
        bias = torch.randn(E, generator = gen) * 0.1
        logits = (x.float() @ G.float()).half().float()
        scores = torch.sigmoid(logits)
        sel_ref = torch.topk(scores + bias, k, dim = -1).indices
        w_ref = scores.gather(1, sel_ref)
        w_ref = w_ref / (w_ref.sum(-1, keepdim = True) + 1e-20) * 2.5
        sel = torch.empty(1, k, dtype = torch.long, device = dev)
        w = torch.empty(1, k, dtype = torch.half, device = dev)
        ext["router"](x.to(dev), G.to(dev), bias.to(dev), sel, w, scratch, counters, 2.5)
        s_ok = sorted(sel.cpu()[0].tolist()) == sorted(sel_ref[0].tolist())
        d = {int(a): float(b) for a, b in zip(sel.cpu()[0], w.cpu()[0].float())}
        werr = max(abs(d.get(int(a), 0.0) - float(b)) for a, b in zip(sel_ref[0], w_ref[0]))
        good = s_ok and werr < 2e-3
        ok &= good
        print(f"router H={H} E={E} k={k}: selection {'match' if s_ok else 'MISMATCH'}, max weight err {werr:.2e} {'OK' if good else 'FAIL'}")
    return ok


def test_norm(ext, dev):
    from exllamav3 import ext_fallbacks as fb
    gen = torch.Generator().manual_seed(7)
    ok = True
    for rows, dim, xdt, ydt, wdt in ((1, 4096, torch.float, torch.half, torch.bfloat16), (3, 4096, torch.half, torch.half, torch.half),
                                     (1, 192, torch.half, torch.half, None), (5, 8192, torch.float, torch.float, torch.bfloat16)):
        x = (torch.randn(rows, dim, generator = gen) * 3).to(xdt).to(dev)
        w = (torch.randn(dim, generator = gen) * 0.1 + 1).to(wdt).to(dev) if wdt else None
        for mode in (0, 1):
            y0 = (torch.randn(rows, dim, generator = gen)).to(ydt).to(dev)
            y1 = y0.clone()
            fb.rms_norm(x, w, y0, 1e-6, 0.0, 1.0, False, mode == 1)
            ext["rms_norm"](x, w, y1, None, 1e-6, 0.0, 1.0, mode)
            e = rel_l2(y1.cpu(), y0.cpu())
            good = e < 2e-3
            ok &= good
            print(f"norm rows={rows} dim={dim} {xdt}->{ydt} w={wdt} mode={mode}: rel-L2 {e:.2e} {'OK' if good else 'FAIL'}")
        if ydt == torch.half:
            r0 = (torch.randn(rows, dim, generator = gen) * 2).float().to(dev)
            r1 = r0.clone()
            y0 = torch.empty(rows, dim, dtype = torch.half, device = dev)
            y1 = torch.empty_like(y0)
            fb.rms_norm_res_in(x, w, y0, r0, 1e-6, 0.0, 1.0)
            ext["rms_norm"](x, w, y1, r1, 1e-6, 0.0, 1.0, 2)
            e = max(rel_l2(y1.cpu(), y0.cpu()), rel_l2(r1.cpu(), r0.cpu()))
            good = e < 2e-3
            ok &= good
            print(f"norm res_in rows={rows} dim={dim}: rel-L2 {e:.2e} {'OK' if good else 'FAIL'}")
    return ok


def test_multi(ext, dev):
    gen = torch.Generator().manual_seed(2)
    scratch, counters = workspace(dev)
    k = 1024
    ns = (1536, 256, 128)
    K = 4
    trs = [rand_trellis(k, n, K, gen) for n in ns]
    suhs = [rand_sign_scale(k, gen) for _ in ns]
    svhs = [rand_sign_scale(n, gen) for n in ns]
    x = (torch.randn(1, k, generator = gen) * 0.5).half()
    outs = [torch.empty(1, n, dtype = torch.half, device = dev) for n in ns]
    ext["gemv_multi"](x.to(dev), [t.to(dev) for t in trs], [s.to(dev) for s in suhs], [s.to(dev) for s in svhs],
                      outs, scratch, counters, [float(K)] * 3)
    ok = True
    for i, n in enumerate(ns):
        e = rel_l2(outs[i].cpu(), ref_gemv(x, trs[i], suhs[i], svhs[i], K))
        ok &= e < 3e-3
        print(f"multi[{i}] n={n}: rel-L2 {e:.2e}")
    return ok


def ref_moe(x, sel, w, G, U, D, Kg, Kd):
    out = torch.zeros(1, x.shape[1])
    for s, e in enumerate(sel.tolist()):
        g = ref_gemv(x, *G[e], Kg).half().float()
        u = ref_gemv(x, *U[e], Kg).half().float()
        a = (torch.nn.functional.silu(g).half().float() * u).half()
        d = ref_gemv(a, *D[e], Kd)
        out += d * float(w[s])
    return out


def test_moe(ext, dev, H = 1024, I = 512, E = 6, topk = 4):
    gen = torch.Generator().manual_seed(3)
    scratch, counters = workspace(dev)
    ok = True
    for Kg, Kd in ((2, 2), (2.5, 2.5), (3, 4)):
        G = [(rand_trellis(H, I, Kg, gen), rand_sign_scale(H, gen), rand_sign_scale(I, gen)) for _ in range(E)]
        U = [(rand_trellis(H, I, Kg, gen), rand_sign_scale(H, gen), rand_sign_scale(I, gen)) for _ in range(E)]
        D = [(rand_trellis(I, H, Kd, gen), rand_sign_scale(I, gen), rand_sign_scale(H, gen)) for _ in range(E)]
        dG = [tuple(t.to(dev) for t in g) for g in G]
        dU = [tuple(t.to(dev) for t in g) for g in U]
        dD = [tuple(t.to(dev) for t in g) for g in D]
        tab = lambda L, i: torch.tensor([t[i].data_ptr() for t in L], dtype = torch.long, device = dev)
        x = (torch.randn(1, H, generator = gen) * 0.5).half()
        sel = torch.tensor([[4, 1, 5, 0][:topk]], dtype = torch.long)
        w = torch.tensor([[0.4, 0.3, 0.2, 0.1][:topk]]).half()
        out = torch.empty(1, H, dtype = torch.float, device = dev)
        act = torch.empty(topk, I, dtype = torch.half, device = dev)
        for _ in range(2):  # twice: counters must be left clean
            ext["moe"](x.to(dev), out, sel.to(dev), w.to(dev),
                       tab(dG, 0), tab(dG, 1), tab(dG, 2), tab(dU, 0), tab(dU, 1), tab(dU, 2),
                       tab(dD, 0), tab(dD, 1), tab(dD, 2), act, scratch, counters, I, float(Kg), float(Kd), False)
        ref = ref_moe(x, sel[0], w[0], G, U, D, Kg, Kd)
        e = rel_l2(out.cpu(), ref)
        resid = torch.randn(1, H, generator = gen)
        out.copy_(resid.to(dev))
        ext["moe"](x.to(dev), out, sel.to(dev), w.to(dev),
                   tab(dG, 0), tab(dG, 1), tab(dG, 2), tab(dU, 0), tab(dU, 1), tab(dU, 2),
                   tab(dD, 0), tab(dD, 1), tab(dD, 2), act, scratch, counters, I, float(Kg), float(Kd), True)
        ea = rel_l2(out.cpu() - resid, ref)
        e = max(e, ea)
        ctr = int(counters.abs().sum())
        good = e < 3e-3 and ctr == 0
        ok &= good
        print(f"moe Kgu={Kg} Kd={Kd} H={H} I={I} topk={topk}: rel-L2 {e:.2e} counters {ctr} {'OK' if good else 'FAIL'}")
    return ok


def bench(fn, nbytes, dev, iters = 30):
    flush = torch.empty(256 << 20, dtype = torch.uint8, device = dev)
    ts = []
    for i in range(iters + 3):
        flush.fill_(i & 0xff)
        s = torch.cuda.Event(enable_timing = True)
        e = torch.cuda.Event(enable_timing = True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if i >= 3:
            ts.append(s.elapsed_time(e) * 1e3)
    ts.sort()
    us = ts[len(ts) // 2]
    return us, nbytes / us / 1e3


def bench_all(ext, dev):
    gen = torch.Generator().manual_seed(4)
    scratch, counters = workspace(dev)
    rows = []
    for (name, k, n, K) in (("q_proj", 4096, 12288, 4), ("o_proj", 8192, 4096, 4), ("kv_swa", 4096, 1536, 4),
                            ("expert_gu K2", 4096, 2048, 2), ("expert_gu K2.5", 4096, 2048, 2.5),
                            ("expert_dn K2", 2048, 4096, 2), ("mlp0 K3", 4096, 16384, 3),
                            ("head K6", 4096, 152576, 6)):
        tr = rand_trellis(k, n, K, gen).to(dev)
        suh = rand_sign_scale(k, gen).to(dev)
        svh = rand_sign_scale(n, gen).to(dev)
        x = (torch.randn(1, k, generator = gen) * 0.5).half().to(dev)
        out = torch.empty(1, n, dtype = torch.half, device = dev)
        f = lambda: ext["gemv"](x, tr, suh, svh, out, scratch, counters, float(K))
        us, gbs = bench(f, tr.numel() * 2, dev)
        print(f"bench {name:<16} {k}x{n} K={K}: {us:8.1f} us  {gbs:6.1f} GB/s", flush = True)
        del tr
    # router and RMSNorm at MiMo shapes (no L2 flush needed to show launch-bound behaviour)
    x = torch.randn(1, 4096, generator = gen).half().to(dev)
    G = (torch.randn(4096, 256, generator = gen) * 0.02).half().to(dev)
    bias = (torch.randn(256, generator = gen) * 0.1).to(dev)
    sel = torch.empty(1, 8, dtype = torch.long, device = dev)
    w = torch.empty(1, 8, dtype = torch.half, device = dev)
    us, _ = bench(lambda: ext["router"](x, G, bias, sel, w, scratch, counters, 2.5), 4096 * 256 * 2, dev)
    print(f"bench router 4096x256 top8: {us:8.1f} us", flush = True)
    if ext["rms_norm"] is not None:
        xr = torch.randn(1, 4096, generator = gen).to(dev)
        r = torch.randn(1, 4096, generator = gen).to(dev)
        wn = torch.randn(4096, generator = gen).bfloat16().to(dev)
        y = torch.empty(1, 4096, dtype = torch.half, device = dev)
        us, _ = bench(lambda: ext["rms_norm"](xr, wn, y, None, 1e-6, 0.0, 1.0, 0), 4096 * 6, dev)
        print(f"bench rms_norm mode0 1x4096: {us:8.1f} us", flush = True)
        us, _ = bench(lambda: ext["rms_norm"](xr, wn, y, r, 1e-6, 0.0, 1.0, 2), 4096 * 10, dev)
        print(f"bench rms_norm res_in 1x4096: {us:8.1f} us", flush = True)
    # MoE layer at MiMo shapes
    H, I, E, topk = 4096, 2048, 32, 8
    for Kg in (2, 2.5):
        L = {}
        for nm, (kk, nn) in (("g", (H, I)), ("u", (H, I)), ("d", (I, H))):
            L[nm] = [(rand_trellis(kk, nn, Kg, gen).to(dev), rand_sign_scale(kk, gen).to(dev), rand_sign_scale(nn, gen).to(dev)) for _ in range(E)]
        tab = lambda nm, i: torch.tensor([t[i].data_ptr() for t in L[nm]], dtype = torch.long, device = dev)
        tabs = [tab(nm, i) for nm in "gud" for i in range(3)]
        x = (torch.randn(1, H, generator = gen) * 0.5).half().to(dev)
        sel = torch.tensor([[3, 17, 5, 30, 11, 2, 25, 8]], dtype = torch.long, device = dev)
        w = torch.full((1, topk), 0.125, dtype = torch.half, device = dev)
        out = torch.empty(1, H, dtype = torch.float, device = dev)
        act = torch.empty(topk, I, dtype = torch.half, device = dev)
        f = lambda: ext["moe"](x, out, sel, w, *tabs, act, scratch, counters, I, float(Kg), float(Kg), False)
        nbytes = topk * 3 * H * I * Kg / 8
        us, gbs = bench(f, nbytes, dev)
        print(f"bench moe K={Kg} top{topk} {H}/{I}: {us:8.1f} us  {gbs:6.1f} GB/s", flush = True)
        del L


def sweep_kbs(ext, dev):
    """Median time per blocks-per-strip value: dense shapes, then MoE gate/up (A) and down (B)."""
    gen = torch.Generator().manual_seed(4)
    scratch, counters = workspace(dev)
    for (name, k, n, K, vals) in (("q_proj", 4096, 12288, 4, (1, 2, 3, 4, 5, 6, 8)),
                                  ("o_proj", 8192, 4096, 4, (2, 4, 5, 6, 8, 10, 12, 16, 20)),
                                  ("kv_swa", 4096, 2560, 4, (2, 4, 6, 8, 10, 12, 16)),
                                  ("qkv_full", 4096, 13568, 4, (1, 2, 3, 4, 6))):
        tr = rand_trellis(k, n, K, gen).to(dev)
        suh = rand_sign_scale(k, gen).to(dev)
        svh = rand_sign_scale(n, gen).to(dev)
        x = (torch.randn(1, k, generator = gen) * 0.5).half().to(dev)
        out = torch.empty(1, n, dtype = torch.half, device = dev)
        f = lambda: ext["gemv"](x, tr, suh, svh, out, scratch, counters, float(K))
        line = []
        for v in (0,) + vals:
            if v: os.environ["EXL3_DEC_KBS"] = str(v)
            us, gbs = bench(f, tr.numel() * 2, dev, iters = 15)
            line.append(f"{v or 'def'}:{us:.0f}us/{gbs:.0f}")
            os.environ.pop("EXL3_DEC_KBS", None)
        print(f"sweep {name:<9} {k}x{n}: " + "  ".join(line), flush = True)
        del tr
    H, I, E, topk = 4096, 2048, 32, 8
    for Kg in (2, 2.5):
        L = {}
        for nm, (kk, nn) in (("g", (H, I)), ("u", (H, I)), ("d", (I, H))):
            L[nm] = [(rand_trellis(kk, nn, Kg, gen).to(dev), rand_sign_scale(kk, gen).to(dev), rand_sign_scale(nn, gen).to(dev)) for _ in range(E)]
        tab = lambda nm, i: torch.tensor([t[i].data_ptr() for t in L[nm]], dtype = torch.long, device = dev)
        tabs = [tab(nm, i) for nm in "gud" for i in range(3)]
        x = (torch.randn(1, H, generator = gen) * 0.5).half().to(dev)
        sel = torch.tensor([[3, 17, 5, 30, 11, 2, 25, 8]], dtype = torch.long, device = dev)
        w = torch.full((1, topk), 0.125, dtype = torch.half, device = dev)
        out = torch.empty(1, H, dtype = torch.float, device = dev)
        act = torch.empty(topk, I, dtype = torch.half, device = dev)
        f = lambda: ext["moe"](x, out, sel, w, *tabs, act, scratch, counters, I, float(Kg), float(Kg), False)
        nbytes = topk * 3 * H * I * Kg / 8
        for var, vals in (("EXL3_DEC_MOE_KBS_A", (1, 2, 3, 4, 5, 6, 8)), ("EXL3_DEC_MOE_KBS_B", (1, 2, 3, 4, 5, 6))):
            line = []
            for v in (0,) + vals:
                if v: os.environ[var] = str(v)
                us, gbs = bench(f, nbytes, dev, iters = 15)
                line.append(f"{v or 'def'}:{us:.0f}us")
                os.environ.pop(var, None)
            print(f"sweep moe K={Kg} {var[-1]}: " + "  ".join(line), flush = True)
        del L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default = "exl3_dec_dev")
    ap.add_argument("--bench", action = "store_true")
    ap.add_argument("--no-test", action = "store_true")
    ap.add_argument("--sweep", action = "store_true")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    ext = load_ext(a.ext)
    ok = True
    if not a.no_test:
        ok &= test_gemv(ext, dev)
        ok &= test_multi(ext, dev)
        ok &= test_moe(ext, dev)
        ok &= test_ktw(ext, dev)
        ok &= test_kbs(ext, dev)
        if ext["router"] is not None:
            ok &= test_router(ext, dev)
        if ext.get("router_norm") is not None:
            ok &= test_router_norm(ext, dev)
        if ext["rms_norm"] is not None:
            ok &= test_norm(ext, dev)
        print("ALL OK" if ok else "SOME FAILED", flush = True)
    if a.sweep:
        sweep_kbs(ext, dev)
    if a.bench:
        bench_all(ext, dev)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
