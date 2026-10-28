"""Draft head restricted to a set of 128-row blocks: column j of the restricted EXL3 linear is column blocks[j // 128] * 128 + j % 128
of the full one, to fp16 rounding (the accumulation order follows n), at 1..16 rows (the draft steps of N requests). Needs a CUDA device."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import pytest
except ImportError:      # plain-python run (the serving venv ships no pytest)
    pytest = None
import torch
from hardware import require
require("ext")
from exllamav3.architecture.qwen3_5_mtp import exl3_block_subhead
from exllamav3.modules.quant.exl3 import LinearEXL3

if pytest is not None:
    pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

def check(K, mul1):
    torch.manual_seed(K * 10 + mul1)
    k, nb = 512, 24
    n = nb * 128
    dev = "cuda"
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype = torch.short, device = dev)
    suh = (torch.randint(0, 2, (k,), device = dev) * 2 - 1).half()
    svh = torch.randn((n,), device = dev).half()
    code = torch.ones((), dtype = torch.int32, device = dev)
    full = LinearEXL3(None, k, n, suh = suh, svh = svh, trellis = trellis, mcg = None if mul1 else code, mul1 = code if mul1 else None)
    blocks = torch.tensor([0, 3, 4, 11, 17, 23])
    sub = exl3_block_subhead(full, blocks)
    cols = (blocks[:, None] * 128 + torch.arange(128))[:].reshape(-1).to(dev)
    for rows in (1, 2, 4, 8, 16):
        x = (torch.randn((rows, k), device = dev) * 1e-2).half()
        yf = full.forward(x.clone(), {})
        ys = sub.forward(x.clone(), {})
        assert ys.shape == (rows, cols.numel())
        d = (ys.float() - yf[:, cols].float()).abs().max().item()
        # the split-K decomposition of the GEMV / the kernel family follows n: same products, fp16 accumulation order differs
        assert d < 5e-3 * max(1.0, yf.float().abs().max().item()), f"K={K} rows={rows}: max diff {d}"


if pytest is not None:
    test_block_subhead_columns = pytest.mark.parametrize("K", [4, 5])(pytest.mark.parametrize("mul1", [True, False])(check))

if __name__ == "__main__":
    for K in (4, 5):
        for mul1 in (True, False):
            check(K, mul1); print("ok", K, mul1)
