"""triton_gap NB unroll check + micro: NB=4 vs the committed NB=1 on the same payload (same accumulation order -> bit-identical), then GB/s at
17408 x 5120 for R1 / R4 with the per-arch config."""
import statistics as st, time
import torch
from exllamav3.foreign_quant import triton_gap as TG

def payload(fmt, N, K):
    bb = TG._BB[fmt]
    r = torch.randint(0, 256, (N, K // 32, bb), dtype=torch.uint8, device="cuda")
    if fmt == "mxfp4":
        r[:, :, 0] = 120
    else:
        r[:, :, 0] = 0; r[:, :, 1] = 0x14
    return r.reshape(N, -1).contiguous()

ok = bad = 0
for fmt in ("q4_0", "q5_0", "iq4_nl", "mxfp4"):
    w = payload(fmt, 256, 2048)
    for R in (1, 4):
        x = torch.randn((R, 2048), device="cuda", dtype=torch.half)
        TG.GAP_NB = 1; a = TG.gemv(fmt, x, w)
        TG.GAP_NB = 4; b = TG.gemv(fmt, x, w)
        if torch.equal(a, b):
            ok += 1
        else:
            bad += 1; print("DIFF", fmt, R, (a.float() - b.float()).abs().max().item())
print(f"{ok} identical, {bad} differ", flush=True)

N, K, C = 17408, 5120, 8
for fmt in ("q4_0", "q5_0", "iq4_nl", "mxfp4"):
    ws = [payload(fmt, N, K) for _ in range(C)]
    nb = ws[0].numel()
    for R in (1, 4):
        x = torch.randn((R, K), device="cuda", dtype=torch.half)
        line = []
        for v in (1, 2, 4, 8):
            TG.GAP_NB = v
            f = lambda: [TG.gemv(fmt, x, w) for w in ws]
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.5:
                f(); torch.cuda.synchronize()
            ts = []
            for _ in range(9):
                t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            line.append(f"NB{v} {C * nb / st.median(ts) / 1e9:4.0f}")
        print(f"{fmt:7s} R{R} " + " | ".join(line), flush=True)
