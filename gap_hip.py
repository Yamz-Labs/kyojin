"""Gap-matrix micro (Strix / HIP): stored GB/s of ForeignLinear.forward per format and row count on the 27B MLP up shape 17408 x 5120.
Random payload bytes (values do not change the read cost), 8 matrices cycled, 1 s ramp, median of 15. LinearFP8 for fp8 / fp8 grid."""
import statistics as st, time
import torch
from exllamav3.modules.foreign_linear import ForeignLinear
from exllamav3.modules.quant.fp8 import LinearFP8

N, K, COUNT = 17408, 5120, 8
BLK = {"q8_0": (34, 32), "q4_0": (18, 32), "q5_0": (22, 32), "iq4_nl": (18, 32), "mxfp4": (17, 32), "q4_k": (144, 256), "q5_k": (176, 256),
       "q6_k": (210, 256), "iq4_xs": (136, 256), "iq2_xxs": (66, 256), "iq3_xxs": (98, 256), "iq3_s": (110, 256)}


def make(fmt):
    if fmt in ("fp8", "fp8grid"):
        grid = fmt == "fp8grid"
        q = torch.randint(0, 120, (N, K), dtype=torch.uint8, device="cuda")
        s = (torch.rand((N // 128, K // 128)) if grid else torch.rand(N)).cuda() * 1e-3
        return LinearFP8(K, N, q, s, grid, (128, 128), None, torch.half), N * K
    fl = ForeignLinear(None, "x", K, N, "nvfp4" if fmt == "nvfp4" else fmt)
    if fmt == "nvfp4":
        fl.packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        fl.scales = torch.randint(0x28, 0x46, (N, K // 16), dtype=torch.uint8, device="cuda")
        fl.global_scale = 1e-3
        return fl, N * K // 2 + N * K // 16
    bb, be = BLK[fmt]
    fl.packed = torch.randint(0, 256, (N, K // be * bb), dtype=torch.uint8, device="cuda")
    for c in range(0, K // be * bb, bb):                             # keep the fp16 block scale finite and small
        fl.packed[:, c:c + 2] = torch.tensor([0x00, 0x14], dtype=torch.uint8, device="cuda")
    return fl, N * (K // be) * bb


import os
for fmt in (os.environ["FMTS"].split(",") if os.environ.get("FMTS") else ["fp8", "fp8grid", "nvfp4"] + list(BLK)):
    try:
        mods = [make(fmt) for _ in range(COUNT)]
    except Exception as e:
        print(f"{fmt:8s} not built: {type(e).__name__}: {str(e)[:80]}", flush=True); continue
    nb = mods[0][1]
    out = []
    for R in (1, 4, 8):
        x = torch.randn((R, K), device="cuda", dtype=torch.half)
        try:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 1.0:
                [m.forward(x, {}) for m, _ in mods]; torch.cuda.synchronize()
            ts = []
            for _ in range(15):
                t = time.perf_counter(); [m.forward(x, {}) for m, _ in mods]; torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            out.append(f"R{R} {COUNT * nb / st.median(ts) / 1e9:4.0f} GB/s")
        except Exception as e:
            out.append(f"R{R} ERR {type(e).__name__}: {str(e)[:50]}")
    print(f"{fmt:8s} {nb / (N * K):.3f} B/w  " + " | ".join(out), flush=True)
    del mods
    torch.cuda.empty_cache()
