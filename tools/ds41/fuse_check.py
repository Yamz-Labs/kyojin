"""GPU exactness + timing of the decode glue fusion (dsv41_fuse): fused vs unfused, bit for bit, then ms/step.
usage (under gpurun): fuse_check.py PACK [--layers 3] [--steps 24] [--ctx 512] [--skip-model]"""
import argparse, gc, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--ctx", type=int, default=512); ap.add_argument("--skip-model", action="store_true")
a = ap.parse_args()
from exllamav3.modules.dsv41_fn import dsv41_fuse as FZ, dsv41_hip as K
from exllamav3.dsv41 import ops
dev = "cuda:0"
bad = []

def same(name, x, y):
    ok = x.shape == y.shape and torch.equal(x.view(torch.int16) if x.dtype == torch.half else x, y.view(torch.int16) if y.dtype == torch.half else y)
    if not ok and x.dtype in (torch.half, torch.float) and bool((x.isnan() == y.isnan()).all()):          # NaN payloads
        m = ~x.isnan()
        ok = torch.equal(x[m], y[m])
    print(f"  {name}: {'EXACT' if ok else 'DIFF max_abs=' + str((x.float() - y.float()).abs().max().item())}", flush=True)
    if not ok and x.shape == y.shape:
        d = (x.view(torch.int16) != y.view(torch.int16)) if x.dtype == torch.half else (x != y)
        d &= ~(x.isnan() & y.isnan())
        idx = d.flatten().nonzero().flatten()[:6].tolist()
        print(f"    {int(d.sum())} mismatches; first:", [(i, x.flatten()[i].item(), y.flatten()[i].item()) for i in idx], flush=True)
    if not ok: bad.append(name)

g = torch.Generator(device=dev).manual_seed(1)
print("kernel level")
for n in (5120, 2048 * 5, 12288):
    for sc in (0.02, 1.0, 6.0, 40.0):
        gate = (torch.randn(1, n, device=dev, generator=g) * sc).half(); up = (torch.randn(1, n, device=dev, generator=g) * sc).half()
        gate[0, :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), 65504.0], device=dev).half()
        up[0, 4:8] = torch.tensor([float("inf"), -float("inf"), float("nan"), -65504.0], device=dev).half()
        for lim in (10.0, 0.0):
            ref = ops.swiglu(gate, up, lim).to(torch.half)
            o = FZ.swiglu_h(gate, up, lim); print(f"  swiglu n={n} sc={sc} lim={lim}: {int((o.view(torch.int16) != ref.view(torch.int16)).sum())} fp16 mismatches (informational, kernel not used by default)")
x = torch.randn(1, 5120, device=dev, generator=g).half()
for _ in range(3):
    o = [torch.randn(1, 5120, device=dev, generator=g) * 3 for _ in range(3)]
    sh = (torch.randn(1, 5120, device=dev, generator=g) * 2).half()
    for ng in (1, 2, 3):
        y = o[0].clone()
        for t in o[1:ng]: y.add_(t)
        same(f"moe_tail ng={ng}", FZ.moe_tail(o[:ng], sh), (y + sh.float()).type_as(sh))
for wd in (torch.bfloat16, torch.half):
    for E in (384, 256):
        W = (torch.randn(E, 5120, device=dev, generator=g) * 5120 ** -0.5).to(wd)
        bias = torch.randn(E, device=dev, generator=g) * 0.2
        for _ in range(4):
            xx = torch.randn(1, 5120, device=dev, generator=g).half()
            sel = torch.empty(1, 6, dtype=torch.int64, device=dev); wts = torch.empty(1, 6, dtype=torch.half, device=dev)
            wf = torch.empty(1, 6, dtype=torch.float, device=dev); sc0 = torch.empty(1, E, dtype=torch.float, device=dev)
            K.router(xx, W, bias, sel, wts, sc0, 6, 1.5, 1.0, True, wf)
            sc1 = torch.empty(1, E, dtype=torch.float, device=dev); sel1 = torch.empty_like(sel); w1h = torch.empty_like(wts); w1f = torch.empty_like(wf)
            lut = torch.randint(-1, 40, (3, E), device=dev, generator=g)
            gsel = torch.empty(3, 1, 6, dtype=torch.int64, device=dev)
            FZ.router_scores_d(xx, W, sc1, 1.0)
            FZ.router_topk2(sc1, bias, sel1, w1h, w1f, 6, True, 1.5, lut, gsel)
            same(f"router scores {wd} E={E}", sc1, sc0); same("router sel", sel1, sel); same("router wts16", w1h, wts); same("router wts32", w1f, wf)
            ref = torch.stack([lut[i].index_select(0, sel.reshape(-1)).view(sel.shape) for i in range(3)])
            same("router gsel", gsel, ref)
if bad:
    print("KERNEL LEVEL FAIL", bad, flush=True)
else:
    print("kernel level ALL EXACT", flush=True)
if a.skip_model: sys.exit(0)

from ds41_serving import build_model

def set_fuse(m, on):
    m.fuse = on
    os.environ["DSV41_FUSE"] = "1" if on else "0"

def run_model(nl):
    m = build_model(a.pack, device=dev, dtype="float16", layers=nl, mtp=False, moe="auto", prefill_chunk=a.ctx)
    print("backends", sorted({type(b).__name__ for b in m.moe_backends.values()}), flush=True)
    ids = torch.randint(0, m.cfg.vocab_size - 1, (1, a.ctx), device=dev, generator=torch.Generator(device=dev).manual_seed(5))
    outs, times = {}, {}
    if nl == a.layers:
        for l in range(nl):
            for _ in range(3):
                xx = (torch.randn(1, 1, 5120, device=dev, generator=g) * 1.5).half()
                set_fuse(m, False); r0 = m._moe(l, xx).clone()
                set_fuse(m, True); r1 = m._moe(l, xx).clone()
                same(f"_moe layer {l}", r1, r0)
    for mode in (False, True, False, True):
        set_fuse(m, mode)
        st = m.new_state(1, a.ctx + a.steps + 64)
        lg, _ = m.prefill(ids, st, 0, chunk=a.ctx)
        tok = lg.argmax(-1).view(1, 1); pos = a.ctx; seq = []; lgs = []
        for _ in range(8):
            lg, _ = m.forward(tok, pos, st); pos += 1; tok = lg.argmax(-1).view(1, 1)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(a.steps):
            lg, _ = m.forward(tok, pos, st); pos += 1; tok = lg.argmax(-1).view(1, 1); lgs.append(lg.clone())
        torch.cuda.synchronize(); dt = (time.time() - t) / a.steps * 1e3
        times.setdefault(mode, []).append(dt)
        outs.setdefault(mode, torch.stack(lgs))
    same(f"logits {nl}L x{a.steps} steps", outs[True], outs[False])
    print(f"TIME layers={nl} unfused {min(times[False]):.3f} ms/step  fused {min(times[True]):.3f} ms/step", flush=True)
    r = (nl, min(times[False]), min(times[True]))
    del m; gc.collect(); torch.cuda.empty_cache()
    return r

rows = [run_model(1), run_model(a.layers)]
(l0, u0, f0), (l1, u1, f1) = rows
print(f"SLOPE unfused {(u1-u0)/(l1-l0):.3f} ms/layer + {u0-(u1-u0)/(l1-l0)*l0:.2f} fixed;  fused {(f1-f0)/(l1-l0):.3f} ms/layer + {f0-(f1-f0)/(l1-l0)*l0:.2f} fixed")
print("FAIL" if bad else "ALL EXACT", bad)
sys.exit(1 if bad else 0)
