"""dsa_topk_gfx12_v2 (row read once into registers) must be bitwise identical to dsa_topk_gfx12 (v1)."""
from __future__ import annotations

import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm dsa_topk tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext

if not (hasattr(ext, "dsa_topk_gfx12") and hasattr(ext, "dsa_topk_gfx12_v2")):
    pytest.skip("dsa_topk_gfx12_v2 binding is unavailable (rebuild the extension)", allow_module_level = True)

DEV = torch.device("cuda")
WIDTHS = [1, 7, 8, 9, 15, 16, 127, 128, 129, 511, 512, 513, 1000, 1023, 1024, 1025, 4095, 4096, 4097, 8191, 8193,
          16384, 24576, 32001, 32767, 32768]


def _storage(rows, width, kind, seed):
    g = torch.Generator(device = DEV).manual_seed(seed)
    stride = -(-width // 128) * 128
    if kind == "randn":
        x = torch.randn((rows, stride), generator = g, device = DEV, dtype = torch.half)
    elif kind == "ties":      # few distinct values: heavy threshold ties
        x = (torch.randint(0, 6, (rows, stride), generator = g, device = DEV).half() - 2) * 0.25
    elif kind == "const":
        x = torch.full((rows, stride), 0.5, device = DEV, dtype = torch.half)
    elif kind == "ninf":
        x = torch.full((rows, stride), -float("inf"), device = DEV, dtype = torch.half)
    elif kind == "sparse":    # mostly -inf (causal mask look), some valid
        x = torch.randn((rows, stride), generator = g, device = DEV, dtype = torch.half)
        keep = torch.rand((rows, stride), generator = g, device = DEV) < 0.02
        x = torch.where(keep, x, torch.full_like(x, -float("inf")))
    elif kind == "special":   # +-inf, +-NaN, +-0, subnormals mixed
        raw = torch.randint(-32768, 32767, (rows, stride), generator = g, device = DEV, dtype = torch.int16)
        x = raw.view(torch.half)
    elif kind == "causal":    # row r sees only the first (width * (r + 1)) // rows columns
        x = torch.randn((rows, stride), generator = g, device = DEV, dtype = torch.half)
        lim = ((torch.arange(rows, device = DEV) + 1) * width // rows).clamp_min(1)
        col = torch.arange(stride, device = DEV)
        x = torch.where(col.unsqueeze(0) < lim.unsqueeze(1), x, torch.full_like(x, -float("inf")))
    else:
        raise ValueError(kind)
    return x[:, :width]


@pytest.mark.parametrize("kind", ["randn", "ties", "const", "ninf", "sparse", "special", "causal"])
@pytest.mark.parametrize("threads", [512, 1024])
@torch.inference_mode()
def test_v2_matches_v1_bitwise(kind, threads):
    for width in WIDTHS:
        for k, out_w in ((512, 512), (1, 8), (17, 32), (256, 256), (511, 512), (300, 512)):
            rows = 6 if width > 8192 else 24
            scores = _storage(rows, width, kind, seed = width * 7 + k)
            a = torch.empty((rows, out_w), device = DEV, dtype = torch.int32)
            b = torch.empty_like(a)
            ext.dsa_topk_gfx12(scores, a, k)
            ext.dsa_topk_gfx12_v2(scores, b, k, threads)
            torch.cuda.synchronize()
            assert torch.equal(a, b), (kind, threads, width, k, out_w)


@torch.inference_mode()
def test_v2_rejects_unsupported_shapes():
    ok = torch.zeros((2, 32768), device = DEV, dtype = torch.half)
    out = torch.empty((2, 512), device = DEV, dtype = torch.int32)
    with pytest.raises(RuntimeError):
        ext.dsa_topk_gfx12_v2(torch.zeros((2, 32896), device = DEV, dtype = torch.half), out, 512, 512)
    with pytest.raises(RuntimeError):
        ext.dsa_topk_gfx12_v2(ok, out, 512, 256)
    with pytest.raises(RuntimeError):   # unaligned base: a view offset by one element
        base = torch.zeros((2 * 32768 + 256,), device = DEV, dtype = torch.half)
        ext.dsa_topk_gfx12_v2(base[1:1 + 2 * 32768].view(2, 32768), out, 512, 512)
