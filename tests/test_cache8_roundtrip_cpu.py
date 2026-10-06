import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "qwen", "cache8"))
from ref8 import pack8, dequant8

def _rel(x):
    w, s = pack8(x)
    return ((dequant8(w, s) - x.float()).pow(2).mean().sqrt() / x.float().pow(2).mean().sqrt()).item()

def test_roundtrip_8bit_error():
    g = torch.Generator().manual_seed(0)
    assert _rel(torch.randn(512, 512, generator=g).half()) < 7e-3
    x = torch.randn(512, 512, generator=g); x[:, ::37] *= 30
    assert _rel(x.half()) < 7e-3

def test_zero_group():
    w, s = pack8(torch.zeros(4, 512).half())
    assert dequant8(w, s).abs().max() < 1e-6
