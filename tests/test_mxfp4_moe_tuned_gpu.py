"""GPU check: EXL3_MXFP4_MOE_TUNED=1 (fused gate+up+bias+swiglu_oai, BN16/NW4) == flag off, bit for bit, R == 1."""
import importlib.util
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or bool(torch.version.hip), reason="needs CUDA")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H, K, I, ND, NE, S, LIM = 2880, 2944, 2880, 2880, 32, 4, 7.0   # gpt-oss-20b shapes (K = H padded to 128)


def _load():
    d = os.path.join(ROOT, "exllamav3", "foreign_quant")
    spec = importlib.util.spec_from_file_location("fq_mxg", os.path.join(d, "__init__.py"), submodule_search_locations=[d])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["fq_mxg"] = pkg
    spec.loader.exec_module(pkg)
    import fq_mxg.triton_moe_mxfp4 as MX
    return MX


def _experts(seed, N, Kk, dev):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Kb = Kk // 32
    w = torch.randint(0, 256, (NE, N, Kb, 17), dtype=torch.uint8, generator=g)      # GGUF block_mxfp4: [e8m0][16 bytes]
    w[..., 0] = torch.randint(118, 127, (NE, N, Kb), dtype=torch.uint8, generator=g)
    w = w.reshape(NE, N, Kb * 17).to(dev)
    b = (torch.randn(NE, N, generator=g) * 0.1).half().to(dev)
    wt = torch.tensor([w.data_ptr() + e * w[0].numel() for e in range(NE)], dtype=torch.int64, device=dev)
    bt = torch.tensor([b.data_ptr() + e * b[0].numel() * 2 for e in range(NE)], dtype=torch.int64, device=dev)
    return w, b, wt, bt


def test_tuned_equals_untuned(monkeypatch):
    MX = _load()
    try:
        from exllamav3.ext import exllamav3_ext as ext
        oai = ext.silu_oai_mul
    except Exception:
        pytest.skip("needs the built extension (silu_oai_mul)")
    dev = torch.device("cuda")
    g_, u_, d_ = _experts(1, I, K, dev), _experts(2, I, K, dev), _experts(3, ND, I, dev)
    h = torch.nn.functional.pad(torch.randn(1, H, device=dev).half(), (0, K - H)).contiguous()
    sel = torch.tensor([3, 17, 5, 28], dtype=torch.int32, device=dev)

    def layer():
        if MX.tuned():
            a = MX.gu_fused(h, g_[2], g_[3], u_[2], u_[3], sel, I, K, LIM)
        else:
            g = MX.moe_gemv(h, g_[2], g_[3], sel, I, K, S).half()
            u = MX.moe_gemv(h, u_[2], u_[3], sel, I, K, S).half()
            a = torch.empty_like(g)
            oai(g, u, a, LIM)
        return MX.moe_gemv(a, d_[2], d_[3], sel, ND, I, 1)

    monkeypatch.setenv("EXL3_MXFP4_MOE_TUNED", "0")
    off = layer()
    monkeypatch.setenv("EXL3_MXFP4_MOE_TUNED", "1")
    on = layer()
    torch.cuda.synchronize()
    assert torch.isfinite(on).all()
    assert (on - off).abs().max().item() == 0.0
