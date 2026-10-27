"""Micro-bench of the DS4.1 HIP kernels vs the torch reference ops on the device (decode R=1, verify R=6, prefill R=2048 rows).

Device time comes from a HIP graph of N back-to-back launches (so the ~30 us ctypes launch cost of the python wrapper is not in
the number); the torch reference is timed the same way where it is capturable, else wall clock of an eager loop (flagged).
    flock ~/ds41_work/gpu.lock python tools/ds41kern/bench_hipk.py          (<= 20 min window, see tools/ds41kern/gpu_window.sh)
    DSV41_DEVICE=emu python tools/ds41kern/bench_hipk.py --smoke                  (CPU emulation, one launch each, no timing)
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "dsv41"))
from exllamav3.dsv41 import ops, quant
from exllamav3.modules.dsv41_fn import dsv41_hip as K

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--n", type=int, default=50)
args = ap.parse_args()
EMU = os.environ.get("DSV41_DEVICE") == "emu"
if EMU:
    import hipemu.build_emu as emu
    emu.install(K)
D = "cpu" if EMU else "cuda"
EPS = 1e-20


def graph_time(fn, n):
    """us per call of fn(), measured as a captured graph of n calls (device time)."""
    if EMU or args.smoke:
        fn()
        return float("nan")
    fn(); fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                fn()
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(5):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (5 * n) * 1e3


def eager_time(fn, n=20):
    if EMU or args.smoke:
        fn()
        return float("nan")
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6


def row(name, R, us, gb=None, ref=None):
    s = f"{name:34s} R={R:5d} {us:9.1f} us"
    if gb:
        s += f"  {gb / us * 1e-3:7.1f} GB/s"
    if ref:
        s += f"   torch ref {ref:9.1f} us ({ref / us:5.1f}x)"
    print(s, flush=True)


g = torch.Generator().manual_seed(1)
rnd = lambda *s, dt=torch.float: (torch.randn(*s, generator=g) * 0.5).to(dt).to(D)
fcc = ops.rope_tables(64, 4096, 65536, 160000.0, 16.0, 32, 1).to(D)
fc = K.rope_table_f32(fcc)
Rs = (1, 6, 2048)
for R in Rs:
    if args.smoke and R > 6:
        continue
    pos = torch.arange(R, dtype=torch.int32).to(D)
    slot = (torch.arange(R, dtype=torch.int32) % 128).to(D)
    # hyper-connections: mix + apply/collapse/norm vs the torch chain
    h = rnd(R, 4, 5120)
    fn = rnd(24, 20480) * 0.04
    scale, base = rnd(3), rnd(24)
    pre, post, comb = (torch.empty(R, n, device=D) for n in (4, 4, 16))
    part = torch.empty(R * K.HC_PART_CH * 25, device=D)
    us = graph_time(lambda: K.hc_mix(h, fn, scale, base, pre, post, comb, part, 20, EPS, 1e-6), args.n)
    ref = eager_time(lambda: ops.hc_mixes(h.unsqueeze(0), fn, scale, base, 4, 20, 1e-6, EPS)) if R <= 6 else None
    row("hc_mix (fn fp32)", R, us, 2 * 1024 * 1024 + R * 80 * 1024, ref)
    fnh = fn.half()
    us = graph_time(lambda: K.hc_mix(h, fnh, scale, base, pre, post, comb, part, 20, EPS, 1e-6), args.n)
    row("hc_mix (fn fp16)", R, us, 1024 * 1024 + R * 80 * 1024)
    y, w, out = rnd(R, 5120, dt=torch.half), rnd(5120), torch.empty(R, 5120, dtype=torch.half, device=D)
    us = graph_time(lambda: K.hc_cn(h, y, post, comb, pre, w, out, EPS, True), args.n)
    ref = eager_time(lambda: ops.rmsnorm(ops.hc_pre(ops.hc_post(y.float().unsqueeze(0), h.unsqueeze(0), post.unsqueeze(0),
                                          comb.view(R, 4, 4).unsqueeze(0)), pre.unsqueeze(0)).half(), w.half(), EPS)) if R <= 6 else None
    row("hc_cn apply+collapse+norm", R, us, R * (2 * 80 * 1024 + 10240 + 10240), ref)
    # router
    x = rnd(R, 5120, dt=torch.half)
    for wdt in (torch.float, torch.half):
        W = rnd(384, 5120, dt=wdt)
        bias = rnd(384)
        sel = torch.empty(R, 6, dtype=torch.int64, device=D)
        wts = torch.empty(R, 6, dtype=torch.half, device=D)
        sc = torch.empty(R, 384, device=D)
        us = graph_time(lambda: K.router(x, W, bias, sel, wts, sc, 6, 1.5), args.n)
        ref = eager_time(lambda: ops.route(x, W.float(), bias, 6, "sqrtsoftplus", True, 1.5)) if R <= 6 else None
        row(f"router sqrtsp ({str(wdt)[6:]} gate)", R, us, 384 * 5120 * W.element_size(), ref)
    # attention-side small ops
    kv = rnd(R, 512, dt=torch.half)
    w512, ring = rnd(512), torch.zeros(128, 512, dtype=torch.half, device=D)
    o = torch.empty(R, 512, dtype=torch.half, device=D)
    us = graph_time(lambda: K.kv_prep(kv, w512, fc, pos, slot, o, ring, EPS), args.n)
    def ref_kv():
        x = ops.rmsnorm(kv.unsqueeze(0), w512.half(), EPS)
        ops.rope_apply_(x[..., -64:], fcc[pos.long()])
        quant.act_quant_fp8_(x, 32)
    ref = eager_time(ref_kv) if R <= 6 else None
    row("kv_prep (norm+rope+fp8+ring)", R, us, None, ref)
    oo = rnd(R, 64 * 512, dt=torch.half)
    us = graph_time(lambda: K.fq_rows(oo, K.FQ_FP8), args.n)
    row("fq_rows fp8 (o, 32768/row)", R, us, R * 64 * 512 * 4)
    q = rnd(R, 32 * 128, dt=torch.half)
    us = graph_time(lambda: K.idx_q_prep(q, fc, pos), args.n)
    row("idx_q_prep (rope+fp4)", R, us, R * 32 * 128 * 4)
    k = rnd(R, 128, dt=torch.half)
    pool = torch.zeros(4096, 128, dtype=torch.half, device=D)
    dst = (torch.arange(R, dtype=torch.int32) % 4096).to(D)
    us = graph_time(lambda: K.idx_k_store(k, w512[:128].contiguous(), fc, pos, dst, pool, EPS), args.n)
    row("idx_k_store", R, us)
    lat = rnd(R, 512, dt=torch.half)
    pool2 = torch.zeros(4096, 512, dtype=torch.half, device=D)
    us = graph_time(lambda: K.pool_store(lat, fc, pos, dst, pool2), args.n)
    row("pool_store (rope+fp4 e4m3/16)", R, us)
    kv32, sc32 = rnd(R, 512), rnd(R, 512)
    kst, sst = rnd(2, 512), rnd(2, 512)
    outc = torch.empty(R + 1, 512, dtype=torch.half, device=D)
    us = graph_time(lambda: K.compress(kv32, sc32, kst, sst, w512, outc, 0, 2, EPS), args.n)
    row("compress ratio 2", R, us)
    # engram
    raw = torch.randint(0, 256, (R * 24, 136), dtype=torch.uint8).to(D)
    emb = torch.empty(R * 24, 256, dtype=torch.half, device=D)
    us = graph_time(lambda: K.mxfp4_decode(raw, emb), args.n)
    row("mxfp4_decode (24 rows/token)", R, us, R * 24 * 136)
    hh = rnd(R, 4, 5120)
    kvg, qk = rnd(R, 25600, dt=torch.half), rnd(4, 5120)
    us = graph_time(lambda: K.engram_gate(hh, kvg, qk, EPS), args.n)
    row("engram_gate", R, us, R * (2 * 80 * 1024 + 51200))

# host launch cost of the python paths (decode issues ~30 launches per layer): wrapper vs prepared pack vs a replayed HIP graph
if not EMU and not args.smoke:
    R = 1
    h = rnd(R, 4, 5120); y = rnd(R, 5120, dt=torch.half); w = rnd(5120)
    post, comb, pre = (torch.rand(R, n, device=D) for n in (4, 16, 4))
    out = torch.empty(R, 5120, dtype=torch.half, device=D)
    pl = K.prepare("ds41_hc_cn", (R,), 256, (h, y, 0, post, comb, pre, w, out, 1, float(EPS)))
    for label, fn in (("wrapper K.hc_cn", lambda: K.hc_cn(h, y, post, comb, pre, w, out, EPS, True)), ("prepared launch", pl)):
        fn(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(500):
            fn()
        host = (time.perf_counter() - t) / 500 * 1e6
        torch.cuda.synchronize()
        print(f"hc_cn launch, {label:16s}: {host:7.1f} us host per call", flush=True)
