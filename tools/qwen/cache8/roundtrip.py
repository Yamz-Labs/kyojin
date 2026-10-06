import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import torch
from ref8 import pack8, dequant8
torch.manual_seed(0)
def stats(name, x):
    w, s = pack8(x); y = dequant8(w, s)
    xf = x.float(); err = (y - xf)
    rel = (err.pow(2).mean().sqrt() / xf.pow(2).mean().sqrt()).item()
    G = x.shape[1] // 32
    amax = xf.view(x.shape[0], G, 32).abs().amax(-1, keepdim=True)
    mx = (err.view(x.shape[0], G, 32).abs() / amax).max().item()
    f16 = ((xf.half().float() - xf).pow(2).mean().sqrt() / xf.pow(2).mean().sqrt()).item()
    print(f"{name:28s} rel RMS {rel:.2e}  max err/groupmax {mx:.2e}  (fp16 rounding itself {f16:.2e})")
    return rel, mx
N, D = 4096, 512
stats("gaussian", torch.randn(N, D).half())
x = torch.randn(N, D) * torch.exp(torch.randn(1, D) * 1.0); stats("per-channel scale spread", x.half())
x = torch.randn(N, D); x[:, ::37] *= 30; stats("outlier channels x30", x.half())
x = torch.randn(N, D) ** 3; stats("heavy tail (cubed)", x.half())
x = torch.zeros(N, D).half(); w, s = pack8(x); assert (dequant8(w, s).abs().max() < 1e-6), "zero group"; print("zero group ok")
# theory: step = 2/256 of the rotated group absmax; uniform error rms = step/sqrt(12) = 2.3e-3 of absmax; group absmax ~ 3 sigma-ish => ~ 0.1 % of RMS
