#!/usr/bin/env python
"""Does torch.cuda.CUDAGraph capture/replay the exl3_dec kernels correctly on gfx1151?

Standalone probe, no model: synthetic trellis at the real MiMo shapes, one kernel per graph, then a
capture/replay vs a plain run comparison (bit-exact). Answers "is HIP graph capture usable at all
on this kernel set" separately from "is the MiMo decode step capturable".

    python tools/decode/graph_probe.py            # all shapes
    python tools/decode/graph_probe.py head       # one shape
"""
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dec import load_ext, rand_trellis, rand_sign_scale, workspace  # noqa: E402

SHAPES = {
    "q":    (4096, 12288, 4.0),
    "o":    (4096, 8192, 4.0),
    "mlp0": (4096, 16384, 3.0),
    "head": (4096, 152576, 6.0),
}


def one(name, k, n, K, dev, gen):
    ext = load_ext("main")
    tr = rand_trellis(k, n, K, gen).to(dev)
    suh = rand_sign_scale(k, gen).to(dev)
    svh = rand_sign_scale(n, gen).to(dev)
    x = (torch.randn(1, k, generator = gen) * 0.5).half().to(dev)
    out = torch.empty(1, n, dtype = torch.half, device = dev)
    scratch, counters = workspace(dev)

    def call():
        counters.zero_()
        ext["gemv"](x, tr, suh, svh, out, scratch, counters, float(K))

    call()
    torch.cuda.synchronize()
    ref = out.clone()

    # plain repeat, to separate "replay differs" from "the kernel is nondeterministic"
    call()
    torch.cuda.synchronize()
    plain = out.clone()
    plain_diff = (plain.float() - ref.float()).abs().max().item()

    g = torch.cuda.CUDAGraph()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            call()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
    except Exception:
        print(f"{name:<6} WARMUP-SIDE-STREAM FAILED\n{traceback.format_exc()}")
        return
    try:
        with torch.cuda.graph(g, capture_error_mode = "thread_local"):
            call()
    except Exception:
        print(f"{name:<6} CAPTURE FAILED: {traceback.format_exc().splitlines()[-1]}")
        return
    try:
        torch.cuda.synchronize()
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
    except Exception:
        print(f"{name:<6} REPLAY FAILED: {traceback.format_exc().splitlines()[-1]}")
        return
    d = (out.float() - ref.float()).abs().max().item()
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    print(f"{name:<6} k={k} n={n} K={K}: capture OK, replay max|d| {d:.3e} rel {rel:.2e} "
          f"(plain repeat max|d| {plain_diff:.3e})")


def staging_test(dev, gen):
    """The generator's staging pattern inside a graph: pinned CPU buffer -> .to(device) -> kernel.
    Rewritten between replays, exactly like Generator._staging + get_for_device."""
    ext = load_ext("main")
    k, n, K = 4096, 8192, 4.0
    tr = rand_trellis(k, n, K, gen).to(dev)
    suh = rand_sign_scale(k, gen).to(dev)
    svh = rand_sign_scale(n, gen).to(dev)
    out = torch.empty(1, n, dtype = torch.half, device = dev)
    scratch, counters = workspace(dev)
    ids = torch.zeros(1, k, dtype = torch.half, pin_memory = True)
    ids.normal_(generator = gen)
    print(f"staging: pinned={ids.is_pinned()} device={ids.device}", flush = True)

    def call(nb):
        x = ids.to(dev, non_blocking = nb)
        counters.zero_()
        ext["gemv"](x, tr, suh, svh, out, scratch, counters, float(K))
        return x

    call(True)
    torch.cuda.synchronize()
    ref = out.clone()
    for name, nb, mode in (("blocking", False, "thread_local"), ("async", True, "thread_local"),
                           ("async-global", True, "global")):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            call(nb)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(g, capture_error_mode = mode):
                call(nb)
        except Exception as e:
            print(f"staging({name}): CAPTURE FAILED {type(e).__name__}: {e}".splitlines()[0])
            continue
        try:
            ids.normal_(generator = gen)     # rewrite the pinned buffer, as the generator does
            call(nb)
            torch.cuda.synchronize()
            ref = out.clone()
            g.replay()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"staging({name}): REPLAY FAILED {type(e).__name__}: {e}".splitlines()[0])
            continue
        d = (out.float() - ref.float()).abs().max().item()
        print(f"staging({name}): capture+replay OK, max|d| vs plain {d:.3e}")


def main():
    dev = torch.device("cuda:0")
    print("torch", torch.__version__, "hip", torch.version.hip, flush = True)
    gen = torch.Generator().manual_seed(4)
    want = sys.argv[1:] or list(SHAPES) + ["staging"]
    for nm in want:
        if nm == "staging":
            try:
                staging_test(dev, gen)
            except Exception:
                print(f"staging ERROR\n{traceback.format_exc()}")
            continue
        k, n, K = SHAPES[nm]
        try:
            one(nm, k, n, K, dev, gen)
        except Exception:
            print(f"{nm:<6} ERROR\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
