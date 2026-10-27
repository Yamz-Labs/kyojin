import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

# Swapped-operand 24-row tile (kernel shape 6, CUDA): 17..24 rows against the 32-row tile (shape 5) and the 16-row tiles

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "CUDA only")

SWAP_SHAPE = 6
REF_SHAPE = 5


def run(m, k, n, K, shape, fp32, mul1 = True):
    dev = "cuda:0"
    g = torch.Generator(device = dev).manual_seed(k * 7 + n + K)
    trel = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype = torch.short, device = dev, generator = g)
    suh = torch.randn((k,), dtype = torch.half, device = dev, generator = g)
    svh = torch.randn((n,), dtype = torch.half, device = dev, generator = g)
    torch.manual_seed(m)
    a = torch.randn((m, k), dtype = torch.half, device = dev)
    ah = torch.empty_like(a)
    c = torch.zeros((m, n), dtype = torch.float if fp32 else torch.half, device = dev)
    ext.exl3_gemm(a, trel, c, suh, ah, svh, shape, not mul1, mul1, 0)
    torch.cuda.synchronize()
    return c.float()


def test_compat():
    for m in (1, 8, 16):
        assert not ext.exl3_gemm_shape_compat(SWAP_SHAPE, m, 2048, 512, 4)
    for m in range(17, 25):
        assert ext.exl3_gemm_shape_compat(SWAP_SHAPE, m, 2048, 512, 4)
    for m in (25, 32, 48):
        assert not ext.exl3_gemm_shape_compat(SWAP_SHAPE, m, 2048, 512, 4)
    assert not ext.exl3_gemm_shape_compat(SWAP_SHAPE, 20, 2048, 384, 4)    # N must divide the 256-column tile


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("K", [2, 3, 4, 6])
@pytest.mark.parametrize("m", [17, 18, 20, 23, 24])
@pytest.mark.parametrize("kn", [(2048, 512), (5120, 6144), (6144, 5120)])
def test_matches_32_row_tile(kn, m, K, fp32):
    k, n = kn
    ref = run(m, k, n, K, REF_SHAPE, fp32)
    out = run(m, k, n, K, SWAP_SHAPE, fp32)
    assert not torch.isnan(out).any()
    # Same products; the k16 tile only splits the fp16 accumulation span differently from the k32 tile
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel


@pytest.mark.parametrize("m", [17, 21, 24])
def test_rows_independent(m):
    # An output row only depends on its own input row: the first 17 rows of a 24-row call equal a 17-row call
    k, n, K = 2048, 512, 4
    dev = "cuda:0"
    g = torch.Generator(device = dev).manual_seed(k * 7 + n + K)
    trel = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype = torch.short, device = dev, generator = g)
    suh = torch.randn((k,), dtype = torch.half, device = dev, generator = g)
    svh = torch.randn((n,), dtype = torch.half, device = dev, generator = g)
    torch.manual_seed(5)
    a = torch.randn((24, k), dtype = torch.half, device = dev)
    outs = []
    for rows in (24, m):
        ah = torch.empty_like(a[:rows]); c = torch.zeros((rows, n), dtype = torch.half, device = dev)
        ext.exl3_gemm(a[:rows].contiguous(), trel, c, suh, ah, svh, SWAP_SHAPE, False, True, 0)
        outs.append(c)
    assert torch.equal(outs[0][:m], outs[1])
