"""
Real-bytes cross-check of the two weight layouts (needs the ~21 MB of range-fetched slices: run
tests/dsv41/fetch_real_slices.py once; skipped otherwise). Official fp8/fp4 vs antirez Q8_0/IQ2_XXS/Q2_K of the
SAME tensors must agree up to requantisation error, and the Engram rows must be byte-identical.
That validates tile/scale layouts, nibble order, ne-order of the 3-D expert tensors and the row packing on real data.
"""
import json, os, sys
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.dsv41 import quant, gguf as G

R = os.environ.get("DSV41_REAL_DIR", "./ds41_work/ds41port/scratch/real")
pytestmark = pytest.mark.skipif(not os.path.exists(f"{R}/engram_rows.json"), reason="real slices not fetched")


def rd(f):
    return torch.frombuffer(bytearray(open(f"{R}/{f}", "rb").read()), dtype=torch.uint8)


def corr(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (((a - a.mean()) * (b - b.mean())).mean() / (a.std(unbiased=False) * b.std(unbiased=False))).item()


def test_dense_fp8_vs_q8_0_wkv():
    off = quant.dequant_fp8_block(rd("off_wkv.bin").view(torch.float8_e4m3fn).reshape(512, 5120),
                                  rd("off_wkv_s.bin").view(torch.float8_e8m0fnu).reshape(16, 160))
    gg = G.decode(rd("gg_wkv.bin"), 8, (512, 5120))
    assert 0.01 < off.std() < 0.1
    assert (off - gg).norm() / off.norm() < 0.01              # Q8_0 requantisation of the fp8 source
    assert corr(off, gg) > 0.9999


@pytest.mark.parametrize("w,o,i,gt", [("w1", 2304, 5120, 16), ("w2", 5120, 2304, 10)])
def test_routed_expert_fp4_vs_iq2_xxs_q2_k(w, o, i, gt):
    a = quant.dequant_fp4_rows(rd(f"off_e3_{w}.bin").view(torch.int8).reshape(o, i // 2),
                               rd(f"off_e3_{w}_s.bin").view(torch.float8_e8m0fnu).reshape(o, i // 32))
    b = G.decode(rd(f"gg_e3_{w}.bin"), gt, (o, i))
    assert corr(a, b) > 0.9                                    # measured 0.926 (IQ2_XXS) / 0.947 (Q2_K)
    assert abs(a.std() / b.std() - 1) < 0.1
    # controls: a wrong orientation / wrong nibble order / shifted rows decorrelates completely
    assert abs(corr(a, b.flatten().roll(1).reshape(o, i))) < 0.05
    swapped = quant.dequant_fp4_rows(
        ((rd(f"off_e3_{w}.bin") >> 4) | (rd(f"off_e3_{w}.bin") << 4)).view(torch.int8).reshape(o, i // 2),
        rd(f"off_e3_{w}_s.bin").view(torch.float8_e8m0fnu).reshape(o, i // 32))
    assert corr(swapped, b) < 0.5


def test_engram_rows_official_vs_gguf_byte_identical():
    d = json.load(open(f"{R}/engram_rows.json"))
    assert d["rows"][-1] == 384006167                          # last row of the 384 006 168-row table
    for r in d["rows"]:
        assert d["off"][str(r)] == d["gg"][str(r)]
    from exllamav3.dsv41.engram import dequant_rows
    row = torch.tensor(list(bytes.fromhex(d["gg"]["123456789"])), dtype=torch.uint8)
    v = dequant_rows(row)
    assert v.shape == (256,) and torch.isfinite(v).all() and v.abs().max() > 0
