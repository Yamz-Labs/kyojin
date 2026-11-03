"""GPU: ForeignLinear over the GGUF gap formats (dequant + matmul) vs the gguf-py dequant, R = 1, 4, 40. Needs the built extension and the gguf package."""
import sys, types, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import numpy as np, gguf
from gguf import GGMLQuantizationType as T
from exllamav3.modules.foreign_linear import ForeignLinear
N, K = 192, 512
names = {"q4_0": T.Q4_0, "q5_0": T.Q5_0, "iq4_nl": T.IQ4_NL, "iq3_xxs": T.IQ3_XXS, "iq3_s": T.IQ3_S, "mxfp4": T.MXFP4, "nvfp4_gguf": T.NVFP4}
dev = torch.device("cuda:0")
class STC:
    def __init__(s, raw): s.raw = raw
    def get_tensor(s, key, device, **kw): return s.raw.to(device)
for fmt, ty in names.items():
    rng = np.random.default_rng(1)
    qk, bb = {2:(32,18),6:(32,22),20:(32,18),18:(256,98),21:(256,110),39:(32,17),40:(64,36)}[ty.value]
    raw = rng.integers(0, 256, size=(N, K // qk, bb), dtype=np.uint8)
    if fmt == "mxfp4": raw[:, :, 0] = rng.integers(120, 135, size=(N, K // qk))
    elif fmt == "nvfp4_gguf": raw[:, :, :4] = rng.integers(40, 90, size=(N, K // qk, 4))
    else: raw[:, :, :2] = rng.uniform(0.001, 0.05, size=(N, K // qk)).astype(np.float16).view(np.uint8).reshape(N, K // qk, 2)
    raw = raw.reshape(N, -1)
    ref_w = torch.from_numpy(gguf.quants.dequantize(raw, ty)).to(dev).half()
    cfg = types.SimpleNamespace(stc=STC(torch.from_numpy(raw)))
    lin = ForeignLinear(cfg, "blk.0.ffn", K, N, fmt); lin.load(dev)
    for R in (1, 4, 40):
        x = torch.randn(R, K, device=dev, dtype=torch.half)
        y = lin.forward(x, {})
        ref = x.float() @ ref_w.float().T
        err = ((y.float() - ref).norm() / ref.norm()).item()
        assert err < 2e-3, (fmt, R, err)
    print(fmt, "ok", err)
