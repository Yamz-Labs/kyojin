"""GPU check for the legacy-block fq_mmvq kernels (Q4_0 / Q4_1 / Q5_0 / Q5_1, 1 row): y = fq_mmvq(q8_1(x)) vs dequant(W) @ x.
The tolerance covers the q8_1 activation noise (llama.cpp quantizes the activations the same way)."""
import os
import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or bool(torch.version.hip), reason="needs CUDA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fq_legacy_cpu import BB, deq_np, rand_blocks  # noqa: E402

SHAPES = [(64, 256), (96, 1024), (256, 512)]      # (N out, K in)


@pytest.mark.parametrize("fmt", ["q4_0", "q4_1", "q5_0", "q5_1"])
@pytest.mark.parametrize("N,K", SHAPES)
def test_mmvq_legacy_vs_dequant(fmt, N, K):
    from exllamav3.ext import exllamav3_ext as ext
    fn = getattr(ext, "fq_mmvq_" + fmt, None)
    if fn is None:
        pytest.skip("extension built without the legacy mmvq kernels")
    dev = torch.device("cuda:0")
    rng = np.random.default_rng(1)
    raw = rand_blocks(fmt, N, K // 32, rng)
    packed = torch.from_numpy(raw).to(dev)
    W = torch.from_numpy(deq_np(fmt, raw)).to(dev)                       # fp32 reference dequant (ggml dequantize_row_q*)
    x = torch.randn(1, K, device=dev, dtype=torch.half)
    ref = (W.double() @ x.double().t()).t().float()                      # [1, N]
    qb = ext.fq_q8_scratch(0, 1, K)
    out = torch.empty(1, N, dtype=torch.half, device=dev)
    ext.fq_quant_q8_1_rows(x, qb)
    fn(qb, packed, out)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 3e-2, (fmt, N, K, rel)
