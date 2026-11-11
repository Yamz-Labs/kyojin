"""triton_f16gemv vs torch.matmul on the gpt-oss decode shapes: correctness (R 1 / 4) and a BN x BK x NW sweep at R1 vs hipBLAS."""
import os, sys, subprocess, time
import torch

SHAPES = ((2880, 4096), (2880, 512), (4096, 2880))

if len(sys.argv) > 1 and sys.argv[1] == "one":
    from exllamav3.foreign_quant import triton_f16gemv as F
    out = []
    for K, N in SHAPES:
        ws = [(torch.randn((K, N), device="cuda") * 0.02).half() for _ in range(8)]
        x = torch.randn((1, K), device="cuda").half()
        i = [0]
        def f():
            i[0] += 1; F.gemv(x, ws[i[0] % 8])
        for _ in range(10): f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(300): f()
        torch.cuda.synchronize(); us = (time.perf_counter() - t) / 300 * 1e6
        out.append(f"{K}x{N}:{K * N * 2 / us / 1e3:.0f}")
    print("RES", " ".join(out), flush=True)
    sys.exit(0)

from exllamav3.foreign_quant import triton_f16gemv as F
ok = bad = 0
for K, N in SHAPES:
    w = (torch.randn((K, N), device="cuda") * 0.02).half()
    for R in (1, 4, 16):
        x = torch.randn((R, K), device="cuda").half()
        ref = (x.float() @ w.float())
        y = F.gemv(x, w).float()
        rel = ((y - ref).norm() / ref.norm()).item()
        if rel < 2e-3: ok += 1
        else: bad += 1; print("DIFF", K, N, R, rel)
print(f"{ok} passed, {bad} failed", flush=True)
for bn in (32, 64, 128):
    for bk in (32, 64, 128):
        for nw in (2, 4, 8):
            env = dict(os.environ, EXL3_F16GEMV_BN=str(bn), EXL3_F16GEMV_BK=str(bk), EXL3_F16GEMV_NW=str(nw))
            r = subprocess.run([sys.executable, "-u", __file__, "one"], env=env, capture_output=True, text=True, timeout=200)
            line = [l for l in r.stdout.splitlines() if l.startswith("RES")]
            print(bn, bk, nw, line[0] if line else "FAIL " + (r.stderr or "")[-150:].replace("\n", " "), flush=True)
