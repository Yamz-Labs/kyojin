"""IQ3_XXS q8_1 dp4a GEMV (fq_gemv_iq3_xxs_q8, port of ggml vec_dot_iq3_xxs_q8_1) vs the torch reference (gap_formats.dequant @ x in fp32)
on real IQ3_XXS rows of a GGUF (default: the 8B UD-IQ3_XXS ffn_gate of layer 0). The kernel quantizes x to q8_1 like llama.cpp, so it is
compared by relative error; the Triton decode (fp16 activations) is printed as the second reference. A wrong port (layout, signs, scale)
gives errors of order 1. Usage: python tests/test_iq3xxs_q8.py [GGUF]"""
import os, struct, sys

import numpy as np
import torch

import exllamav3_ext as ext
from exllamav3.foreign_quant import gap_formats, triton_iq

GGUF = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("IQ3XXS_GGUF", "model.gguf")
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def tensor(path, want):
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]; u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    def s(): return f.read(u64()).decode()
    def skip(t):
        if t == 8: f.seek(u64(), 1)
        elif t == 9:
            at, n = u32(), u64()
            if at in SZ: f.seek(SZ[at] * n, 1)
            else:
                for _ in range(n): skip(at)
        else: f.seek(SZ[t], 1)
    assert f.read(4) == b"GGUF"; u32(); nt = u64(); nkv = u64(); align = 32
    for _ in range(nkv):
        k = s(); t = u32()
        if k == "general.alignment": align = u32(); continue
        skip(t)
    infos = []
    for _ in range(nt):
        n = s(); nd = u32(); dims = [u64() for _ in range(nd)]; t = u32(); off = u64(); infos.append((n, dims, t, off))
    base = (f.tell() + align - 1) // align * align
    for n, dims, t, off in infos:
        if n == want:
            assert t == 18, f"{n}: type {t}, not IQ3_XXS"
            K, N = dims[0], dims[1]
            f.seek(base + off)
            return np.frombuffer(f.read(N * (K // 256) * 98), dtype=np.uint8).reshape(N, -1), K, N
    raise KeyError(want)


def test_iq3xxs_q8_matches_reference():
    raw, K, N = tensor(GGUF, "blk.0.ffn_gate.weight")
    n = min(N, 1024)
    w = torch.from_numpy(raw[:n].copy()).cuda()
    g = torch.Generator(device="cpu").manual_seed(0)
    worst = 0.0
    for R in (1, 4, 8):
        x = (torch.randn(R, K, generator=g) * 0.5).half().cuda()
        ref = x.float() @ gap_formats.dequant("iq3_xxs", w).float().reshape(n, K).t()
        out = torch.empty((R, n), dtype=torch.half, device="cuda")
        qb = ext.fq_q8_scratch(torch.cuda.current_device(), R, K)
        ext.fq_quant_q8_1_rows(x, qb)
        ext.fq_gemv_iq3_xxs_q8(qb, w, out)
        rel = float((out.float() - ref).norm() / ref.norm())
        tri = triton_iq.gemv("iq3_xxs", x, w)
        rel_t = float((tri.float() - ref).norm() / ref.norm())
        print(f"R={R} rel err q8 kernel {rel:.4f}  triton {rel_t:.4f}")
        worst = max(worst, rel)
    assert worst < 0.02, f"relative error {worst:.4f} (q8_1 activations alone give ~0.005)"


if __name__ == "__main__":
    test_iq3xxs_q8_matches_reference()
    print("PASS")
