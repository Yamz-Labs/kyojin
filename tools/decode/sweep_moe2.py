#!/usr/bin/env python
"""Brief 11: k-split knob sweep for the exl3_dec MoE and dense GEMV at the exact MiMo-V2.6
batch-1 shapes (H=4096, I=2048, E=24, topk=8; expert K in {2, 2.5}, dense K 4).
The env knob is global, so the winner per side is the combo with the lowest summed time over
both expert K values. Prints the table and writes the winning env assignments to argv[1]."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dec import load_ext, rand_trellis, rand_sign_scale, bench, workspace

H, I, E, topk = 4096, 2048, 24, 8
OUT = sys.argv[1] if len(sys.argv) > 1 else "logs/moebest.env"
CLEAR = ("EXL3_DEC_MOE_KTW_A", "EXL3_DEC_MOE_KBS_A", "EXL3_DEC_MOE_KTW_B", "EXL3_DEC_MOE_KBS_B",
         "EXL3_DEC_KTW", "EXL3_DEC_KBS")


def clear():
    for v in CLEAR:
        os.environ.pop(v, None)


def main():
    dev = torch.device("cuda:0")
    ext = load_ext("main")
    gen = torch.Generator().manual_seed(4)
    scratch, counters = workspace(dev)
    x = (torch.randn(1, H, generator = gen) * 0.5).half().to(dev)
    sel = torch.tensor([[3, 17, 5, 20, 11, 2, 23, 8]], dtype = torch.long, device = dev)
    w = torch.full((1, topk), 0.125, dtype = torch.half, device = dev)
    out = torch.empty(1, H, dtype = torch.float, device = dev)
    act = torch.empty(topk, I, dtype = torch.half, device = dev)

    total = {}      # (side, ktw, kbs) -> summed us over both expert K values
    for K in (2, 2.5):
        L = {}
        for nm, (kk, nn) in (("g", (H, I)), ("u", (H, I)), ("d", (I, H))):
            L[nm] = [(rand_trellis(kk, nn, K, gen).to(dev), rand_sign_scale(kk, gen).to(dev),
                      rand_sign_scale(nn, gen).to(dev)) for _ in range(E)]
        tabs = [torch.tensor([t[i].data_ptr() for t in L[nm]], dtype = torch.long, device = dev)
                for nm in "gud" for i in range(3)]
        f = lambda: ext["moe"](x, out, sel, w, *tabs, act, scratch, counters, I, float(K), float(K), False)
        nbytes = topk * 3 * H * I * K / 8
        for side, ktwv, kbsv in (("A", (4, 8, 16), (0, 4, 5, 6, 8)), ("B", (4, 8, 16, 32), (0, 4, 6, 8))):
            res = []
            for ktw in ktwv:
                for kbs in kbsv:
                    clear()
                    os.environ[f"EXL3_DEC_MOE_KTW_{side}"] = str(ktw)
                    if kbs:
                        os.environ[f"EXL3_DEC_MOE_KBS_{side}"] = str(kbs)
                    try:
                        us, gbs = bench(f, nbytes, dev, iters = 12)
                    except RuntimeError:
                        continue
                    res.append((us, ktw, kbs, gbs))
                    total[(side, ktw, kbs)] = total.get((side, ktw, kbs), 0.0) + us
            res.sort()
            ref = [r for r in res if r[1] == (8 if side == "A" else 16) and r[2] == 0][0]
            print(f"moe K={K} {side}: best ktw={res[0][1]} kbs={res[0][2]} {res[0][0]:.0f}us {res[0][3]:.0f}GB/s | "
                  f"default ktw={ref[1]} {ref[0]:.0f}us {ref[3]:.0f}GB/s | {100 * (ref[0] - res[0][0]) / ref[0]:+.1f}%",
                  flush = True)
            for us, ktw, kbs, gbs in res[:5]:
                print(f"     ktw={ktw} kbs={kbs}: {us:.1f}us {gbs:.0f}GB/s", flush = True)
        del L
    clear()

    lines = []
    for side in "AB":
        cands = sorted((v, k[1], k[2]) for k, v in total.items() if k[0] == side)
        v, ktw, kbs = cands[0]
        dflt = total[(side, 8 if side == "A" else 16, 0)]
        print(f"moe {side} summed winner ktw={ktw} kbs={kbs}: {v:.1f}us vs default {dflt:.1f}us "
              f"({100 * (dflt - v) / dflt:+.1f}%)", flush = True)
        lines.append(f"EXL3_DEC_MOE_KTW_{side}={ktw}")
        if kbs:
            lines.append(f"EXL3_DEC_MOE_KBS_{side}={kbs}")

    # dense GEMV: qkv multi + o_proj are ~30 % of the token
    for (nm, k, n, K) in (("q", 4096, 12288, 4), ("o", 8192, 4096, 4)):
        tr = rand_trellis(k, n, K, gen).to(dev)
        suh = rand_sign_scale(k, gen).to(dev)
        svh = rand_sign_scale(n, gen).to(dev)
        xx = (torch.randn(1, k, generator = gen) * 0.5).half().to(dev)
        o = torch.empty(1, n, dtype = torch.half, device = dev)
        f = lambda: ext["gemv"](xx, tr, suh, svh, o, scratch, counters, float(K))
        res = []
        for ktw in (0, 4, 8, 16):
            for kbs in (0, 2, 4, 6, 8):
                clear()
                if ktw:
                    os.environ["EXL3_DEC_KTW"] = str(ktw)
                if kbs:
                    os.environ["EXL3_DEC_KBS"] = str(kbs)
                try:
                    us, gbs = bench(f, tr.numel() * 2, dev, iters = 12)
                except RuntimeError:
                    continue
                res.append((us, ktw, kbs, gbs))
        good = sorted(res)
        dflt = [r for r in res if r[1] == 0 and r[2] == 0][0]
        print(f"gemv {nm} {k}x{n} K{K}: best ktw={good[0][1]} kbs={good[0][2]} {good[0][0]:.0f}us {good[0][3]:.0f}GB/s | "
              f"default {dflt[0]:.0f}us {dflt[3]:.0f}GB/s | {100 * (dflt[0] - good[0][0]) / dflt[0]:+.1f}%",
              flush = True)
        for us, ktw, kbs, gbs in good[:5]:
            print(f"     ktw={ktw} kbs={kbs}: {us:.1f}us {gbs:.0f}GB/s", flush = True)
        clear()
        del tr

    print("WINNER " + " ".join(lines), flush = True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
