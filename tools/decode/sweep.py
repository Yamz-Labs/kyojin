#!/usr/bin/env python
"""ktw sweep of the exl3_dec kernels at MiMo shapes (synthetic trellis), one line per config."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dec import load_ext, rand_trellis, rand_sign_scale, bench, workspace

def main():
    dev = torch.device("cuda:0")
    names = sys.argv[1].split(",")
    gen = torch.Generator().manual_seed(4)
    scratch, counters = workspace(dev)
    shapes = [("q_proj", 4096, 12288, 4), ("o_proj", 8192, 4096, 4), ("kv_swa", 4096, 1536, 4),
              ("gu_K2", 4096, 2048, 2), ("mlp0_K3", 4096, 16384, 3), ("head_K6", 4096, 152576, 6)]
    data = {}
    for (nm, k, n, K) in shapes:
        data[nm] = (rand_trellis(k, n, K, gen).to(dev), rand_sign_scale(k, gen).to(dev), rand_sign_scale(n, gen).to(dev),
                    (torch.randn(1, k, generator = gen) * 0.5).half().to(dev), torch.empty(1, n, dtype = torch.half, device = dev), K)
    H, I, E, topk = 4096, 2048, 24, 8
    moe = {}
    for Kg in (2, 2.5):
        L = {}
        for p, (kk, nn) in (("g", (H, I)), ("u", (H, I)), ("d", (I, H))):
            L[p] = [(rand_trellis(kk, nn, Kg, gen).to(dev), rand_sign_scale(kk, gen).to(dev), rand_sign_scale(nn, gen).to(dev)) for _ in range(E)]
        tabs = [torch.tensor([t[i].data_ptr() for t in L[p]], dtype = torch.long, device = dev) for p in "gud" for i in range(3)]
        moe[Kg] = (L, tabs)
    x = (torch.randn(1, H, generator = gen) * 0.5).half().to(dev)
    sel = torch.tensor([[3, 17, 5, 20, 11, 2, 23, 8]], dtype = torch.long, device = dev)
    w = torch.full((1, topk), 0.125, dtype = torch.half, device = dev)
    out = torch.empty(1, H, dtype = torch.float, device = dev)
    act = torch.empty(topk, I, dtype = torch.half, device = dev)
    for name in names:
        ext = load_ext(name)
        for nm, (tr, suh, svh, xx, o, K) in data.items():
            row = []
            for ktw in (4, 8, 16, 32):
                os.environ["EXL3_DEC_KTW"] = str(ktw)
                try:
                    us, gbs = bench(lambda: ext["gemv"](xx, tr, suh, svh, o, scratch, counters, float(K)), tr.numel() * 2, dev, iters = 20)
                    row.append(f"{ktw}:{us:7.1f}us/{gbs:5.1f}")
                except RuntimeError as e:
                    row.append(f"{ktw}:  n/a")
            os.environ.pop("EXL3_DEC_KTW")
            print(f"{name:<14} {nm:<8} " + "  ".join(row), flush = True)
        for Kg, (L, tabs) in moe.items():
            nbytes = topk * 3 * H * I * Kg / 8
            for ka in (4, 8, 16, 32):
                row = []
                for kb in (4, 8, 16):
                    os.environ["EXL3_DEC_MOE_KTW_A"] = str(ka); os.environ["EXL3_DEC_MOE_KTW_B"] = str(kb)
                    try:
                        us, gbs = bench(lambda: ext["moe"](x, out, sel, w, *tabs, act, scratch, counters, I, float(Kg), float(Kg)), nbytes, dev, iters = 20)
                        row.append(f"B{kb}:{us:6.1f}us/{gbs:5.1f}")
                    except RuntimeError:
                        row.append(f"B{kb}: n/a")
                print(f"{name:<14} moe K={Kg} A{ka:<2} " + "  ".join(row), flush = True)
            os.environ.pop("EXL3_DEC_MOE_KTW_A"); os.environ.pop("EXL3_DEC_MOE_KTW_B")

if __name__ == "__main__":
    main()
