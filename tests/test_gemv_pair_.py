"""
exl3_gemv_int8_pair: two same-shape mul1 GEMVs over one activation in a single sq launch, bit-identical
to two single launches (EXL3_GEMV_PAIR=0 disables the pair: the call then returns False).
Needs a CUDA device and EXL3_INT8_GEMV on (default on CUDA).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.ext import exllamav3_ext as ext

mode = int(os.environ.get("EXL3_GEMV_PAIR", "1"))
dev = torch.device("cuda:0")
g = torch.Generator(device = dev).manual_seed(7)
R = lambda *s: torch.randn(*s, dtype = torch.half, device = dev, generator = g)

taken = 0
for bits, k, n, m in [(4, 2048, 6144, 1), (4, 5120, 5120, 2), (3, 5120, 17408, 1), (4, 5120, 17408, 1), (5, 5120, 17408, 1), (5, 5120, 17408, 2)]:
    mk = lambda: torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype = torch.short, device = dev, generator = g)
    w1, w2 = mk(), mk()
    suh1, suh2, svh1, svh2 = R(k), R(k), R(n), R(n)
    a = R(m, k); ah = torch.empty_like(a)
    for dtype in (torch.half, torch.float):
        r1 = torch.zeros(m, n, dtype = dtype, device = dev); r2 = torch.zeros_like(r1)
        ext.exl3_gemm(a, w1, r1, suh1, ah, svh1, 0, False, True, 0)
        ext.exl3_gemm(a, w2, r2, suh2, ah, svh2, 0, False, True, 0)
        c1 = torch.zeros_like(r1); c2 = torch.zeros_like(r1)
        ok = ext.exl3_gemv_int8_pair(a, w1, c1, suh1, svh1, w2, c2, suh2, svh2)
        torch.cuda.synchronize()
        if mode == 0: assert not ok
        if not ok: continue
        taken += 1
        assert torch.equal(c1, r1) and torch.equal(c2, r2), (bits, k, n, m, dtype)
        # shared activations and input signs: one staged slice serves both matrices
        d1 = torch.zeros_like(r1); d2 = torch.zeros_like(r1); r3 = torch.zeros_like(r1)
        assert ext.exl3_gemv_int8_pair(a, w1, d1, suh1, svh1, w2, d2, suh1, svh2)
        ext.exl3_gemm(a, w2, r3, suh1, ah, svh2, 0, False, True, 0)
        torch.cuda.synchronize()
        assert torch.equal(d2, r3)
assert mode == 0 or taken > 0
print(f"gemv pair mode {mode}: ok, {taken} pairs taken")
