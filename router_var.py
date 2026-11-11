import time, torch
from exllamav3.ext import exllamav3_ext as ext
H, E, Kt = 2048, 256, 8
torch.manual_seed(0)
y = torch.randn((1, H), device="cuda", dtype=torch.half)
gate = (torch.randn((H, E), device="cuda", dtype=torch.half) * 0.03).contiguous()
gt = gate.T.contiguous(); gtf = gt.float()
lg = torch.empty((1, E), device="cuda", dtype=torch.half); se = torch.empty((1, Kt), device="cuda", dtype=torch.long); rw = torch.empty((1, Kt), device="cuda", dtype=torch.half)
def T(f, n=300):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
from exllamav3.modules.block_sparse_mlp_routing import _routing_std_torch
class C: pass
cfg = C(); cfg.gate_tensor = gate; cfg.num_experts_per_tok = Kt; cfg.per_expert_scale = None; cfg.router_bias = None; cfg.num_experts = E
f0 = lambda: _routing_std_torch(cfg, y, {})
print("_routing_std_torch  %.0f us" % T(f0))
def f1():
    l = (y.float() * gtf).sum(-1).view(1, E)
    v, i = torch.topk(l, Kt, dim=-1)
    return i, torch.softmax(v, dim=-1).half()
print("mulsum+topk+softmax %.0f us" % T(f1))
def f2():
    l = torch.matmul(y, gate)
    return torch.topk(l.float(), Kt, dim=-1)
print("fp16 matmul alone   %.0f us" % T(lambda: torch.matmul(y, gate)))
print("mulsum alone        %.0f us" % T(lambda: (y.float() * gtf).sum(-1)))
print("topk+softmax alone  %.0f us" % T(lambda: torch.softmax(torch.topk(lg.float(), Kt, dim=-1)[0], -1)))
s0, r0 = f0(); i1, w1 = f1(); se = s0; rw = r0
print("same experts", sorted(se[0].tolist()) == sorted(i1[0].tolist()), "wdiff", (torch.sort(rw.float(), descending=True)[0] - torch.sort(w1.float(), descending=True)[0]).abs().max().item())
