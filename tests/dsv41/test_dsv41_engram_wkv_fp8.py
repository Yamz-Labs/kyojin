"""The converter copies the Engram projection `layers.N.engram.wkv` raw (F8_E4M3 weight + F8_E8M0 32x32 scale grid, never
quantized). The serving block builds it as a framework Linear: it must load through the fp8 + e8m0 dequant path and
reproduce weight * 2^(scale - 127) block by block."""
import os, sys
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.loader import SafetensorsCollection
from exllamav3.modules import Linear


class _Cfg:
    pass


def test_engram_wkv_fp8_loads_and_dequantizes(tmp_path):
    out_f, in_f = 160, 192                                    # real: 25600 x 6144 with a [800, 192] grid (32 x 32 blocks)
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(out_f, in_f, generator=g) * 0.5).to(torch.float8_e4m3fn)
    sc = torch.randint(122, 130, (out_f // 32, in_f // 32), generator=g, dtype=torch.uint8)
    save_file({"layers.1.engram.wkv.weight": w, "layers.1.engram.wkv.scale": sc.view(torch.float8_e8m0fnu)},
              str(tmp_path / "layers.1.safetensors"))
    cfg = _Cfg()
    cfg.stc = SafetensorsCollection(str(tmp_path))
    lin = Linear(cfg, "layers.1.engram.wkv", in_f, out_f, qmap="layers.1.engram.wkv", qbits_key="bits", out_dtype=torch.half,
                 trim_padded_out=True)
    lin.load(torch.device("cpu"))
    assert getattr(lin.inner, "quant_type", None) == "fp16"
    ref = (w.float().view(out_f // 32, 32, in_f // 32, 32) * (sc.float() - 127).exp2().view(out_f // 32, 1, in_f // 32, 1)).view(out_f, in_f)
    x = torch.randn(3, in_f, generator=g)
    from exllamav3.dsv41.pack import PackLin
    y = PackLin(lin)(x)
    assert torch.allclose(y.float(), x @ ref.half().float().T, rtol=2e-2, atol=2e-2), (y.float() - x @ ref.T).abs().max()
