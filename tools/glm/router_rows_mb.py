"""round2 microbench: exl3_dec_router_rows (one launch, NR rows) vs NR exl3_dec_router launches.
Rotates 42 distinct GLM-shaped gates (H 4096, E 288, top-8, fp32 bias) so G comes from DRAM, like the
42 MoE layers of one verify pass. Gate: sel and wts bitwise equal. Prints us per layer for each."""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import dec_workspace
torch.manual_seed(0)
dev = torch.device("cuda:0")
H, E, K, L = 4096, 288, 8, 42
gates = [(torch.randn(H, E, device=dev) * 0.02).half().contiguous() for _ in range(L)]
biases = [(torch.randn(E, device=dev) * 0.01).float().contiguous() for _ in range(L)]
scratch, counters = dec_workspace(dev)
for NR in (2, 3, 4):
    x = torch.randn(NR, H, device=dev).half().contiguous()
    s1 = torch.zeros(NR, K, dtype=torch.long, device=dev); w1 = torch.zeros(NR, K, dtype=torch.half, device=dev)
    s2 = torch.zeros_like(s1); w2 = torch.zeros_like(w1)
    def loop(l):
        for r in range(NR):
            ext.exl3_dec_router(x[r:r + 1], gates[l], biases[l], s1[r], w1[r], scratch, counters, 2.5)
    def rows(l):
        ext.exl3_dec_router_rows(x, gates[l], biases[l], s2, w2, scratch, counters, 2.5)
    ok = True
    for l in range(L):
        loop(l); rows(l); torch.cuda.synchronize()
        ok &= torch.equal(s1, s2) and torch.equal(w1.view(torch.short), w2.view(torch.short))
    res = {}
    for name, f in (("loop", loop), ("rows", rows)):
        for _ in range(3):
            for l in range(L): f(l)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        N = 20
        e0.record()
        for _ in range(N):
            for l in range(L): f(l)
        e1.record(); torch.cuda.synchronize()
        res[name] = 1000 * e0.elapsed_time(e1) / (N * L)
    print(f"NR={NR} bitwise={ok} loop {res['loop']:.2f} us/layer rows {res['rows']:.2f} us/layer "
          f"saved {res['loop'] - res['rows']:.2f} us/layer = {42 * (res['loop'] - res['rows']) / 1000:.3f} ms/round", flush=True)
