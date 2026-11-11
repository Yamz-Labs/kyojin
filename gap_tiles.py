"""Strix tiling sweep of triton_gap._gemv32 (decision: per-format (BN, num_warps) on this GPU). 17408 x 5120, random payload, 8 matrices, GB/s."""
import statistics as st, time
import torch
from exllamav3.foreign_quant import triton_gap as TG

N, K, C = 17408, 5120, 8
for fmt in ("q4_0", "mxfp4", "q5_0", "iq4_nl"):
    bb = TG._BB[fmt]
    raws = []
    for _ in range(C):
        r = torch.randint(0, 256, (N, K // 32 * bb), dtype=torch.uint8, device="cuda").view(N, K // 32, bb)
        if fmt == "mxfp4":
            r[:, :, 0] = 120                                     # E8M0 exponent near 1
        else:
            r[:, :, 0] = 0; r[:, :, 1] = 0x14                    # small fp16 d
        raws.append(r.reshape(N, -1).contiguous())
    nb = raws[0].numel()
    for R in (1, 4):
        x = torch.randn((R, K), device="cuda", dtype=torch.half)
        res = []
        for BN in (16, 32, 64, 128):
            for nw in (1, 2, 4):
                try:
                    f = lambda: [TG.gemv(fmt, x, w, BN=BN, num_warps=nw) for w in raws]
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < 0.5:
                        f(); torch.cuda.synchronize()
                    ts = []
                    for _ in range(9):
                        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
                    res.append((round(C * nb / st.median(ts) / 1e9), BN, nw))
                except Exception as e:
                    res.append((0, BN, nw))
        res.sort(reverse=True)
        cur = [r for r in res if (r[1], r[2]) == TG._CFG[fmt]]
        print(f"{fmt:7s} R{R} best {res[:3]}  current {TG._CFG[fmt]} -> {cur[0][0] if cur else '?'} GB/s", flush=True)
