"""row invariance of ext.hc_mix_norm / hc_mix / hc_apply at R=1..32 (EXL3_ROOT=<worktree>): row j of an R-row call vs the same row alone."""
import os, sys
sys.path.insert(0, os.environ["EXL3_ROOT"])
import torch
from exllamav3.ext import exllamav3_ext as ext
torch.manual_seed(1); dev = "cuda"
D = 4096; H = 4; M = 24
fn = (torch.randn(M, H * D, device=dev) * 0.02).half()
base = torch.randn(M, device=dev); scale = torch.randn(3, device=dev) * 0.1
w = torch.randn(D, device=dev).half()
Rmax = 32
st = torch.randn(Rmax, H, D, device=dev)
def norm_call(s):
    R = s.shape[0]
    chunks = ext.hc_mix_num_chunks(R, H * D)
    part = torch.zeros(R, chunks, M + 1, device=dev); post = torch.zeros(R, H, device=dev); comb = torch.zeros(R, H, H, device=dev)
    y = torch.zeros(R, D, device=dev, dtype=torch.half)
    ext.hc_mix_norm(s.contiguous(), fn, base, scale, 1e-6, 1e-6, 20, part, post, comb, y, w, 1e-6, 0.0, 1.0)
    return post, comb, y
def mix_call(s):
    R = s.shape[0]
    chunks = ext.hc_mix_num_chunks(R, H * D)
    part = torch.zeros(R, chunks, M + 1, device=dev); post = torch.zeros(R, H, device=dev); comb = torch.zeros(R, H, H, device=dev)
    y = torch.zeros(R, D, device=dev, dtype=torch.half)
    ext.hc_mix(s.contiguous(), fn, base, scale, 1e-6, 1e-6, 20, part, post, comb, y)
    return post, comb, y
bad = 0
for name, call in (("hc_mix_norm", norm_call), ("hc_mix", mix_call)):
    ref = [call(st[j:j + 1]) for j in range(Rmax)]
    for R in list(range(2, 33)):
        out = call(st[:R])
        eq = []
        for j in range(R):
            eq.append(all(torch.equal(out[k][j:j + 1], ref[j][k]) for k in range(3)))
        if not all(eq):
            bad += 1; print(name, "R", R, "rows not equal to R=1:", [j for j, e in enumerate(eq) if not e][:8], flush=True)
print("DONE bad", bad)
