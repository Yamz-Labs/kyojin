"""Single-row q8_1 dp4a GEMV for dense IQ3_XXS / IQ2_S / IQ3_S (fq_gemv_iq3_xxs_q8 / fq_gemv_iq2_s_q8 / fq_gemv_iq3_s_q8, ggml vecdotq ports)
vs the torch reference (gap_formats.dequant @ x in fp32) on real rows of a GGUF (default: the 8B UD-IQ3_XXS, first tensor of each type).
q8_1 activations -> relative error ~0.5 %; a wrong port gives errors of order 1. Usage: python tests/test_dense_iq_q8.py [GGUF]"""
import os, sys

import numpy as np
import torch

sys.path.insert(0, "tests")
from test_moe_iq_q8 import index  # noqa: E402

import exllamav3_ext as ext  # noqa: E402
from exllamav3.foreign_quant import gap_formats  # noqa: E402

GGUF = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DENSE_IQ_GGUF", "model.gguf")
TYPES = {18: ("iq3_xxs", 98, "fq_gemv_iq3_xxs_q8"), 22: ("iq2_s", 82, "fq_gemv_iq2_s_q8"), 21: ("iq3_s", 110, "fq_gemv_iq3_s_q8")}


def test_dense_iq_q8_matches_reference():
    f, infos, base = index(GGUF)
    worst = 0.0
    for t, (fmt, bb, fn) in TYPES.items():
        name = next(n for n, v in infos.items() if v[1] == t and n.endswith(".weight") and len(v[0]) == 2)
        dims, _, off = infos[name]
        K, N = dims[0], dims[1]
        n = min(N, 1024)
        f.seek(base + off)
        w = torch.from_numpy(np.frombuffer(f.read(n * (K // 256) * bb), dtype=np.uint8).reshape(n, -1).copy()).cuda()
        x = (torch.randn(1, K, generator=torch.Generator().manual_seed(t)) * 0.5).half().cuda()
        ref = x.float() @ gap_formats.dequant(fmt, w).float().reshape(n, K).t()
        out = torch.empty((1, n), dtype=torch.half, device="cuda")
        qb = ext.fq_q8_scratch(torch.cuda.current_device(), 1, K)
        ext.fq_quant_q8_1_rows(x, qb)
        getattr(ext, fn)(qb, w, out)
        rel = float((out.float() - ref).norm() / ref.norm())
        print(f"{name} {fmt} [{n}, {K}] rel err {rel:.4f}")
        worst = max(worst, rel)
    assert worst < 0.02, f"relative error {worst:.4f}"


if __name__ == "__main__":
    test_dense_iq_q8_matches_reference()
    print("PASS")
