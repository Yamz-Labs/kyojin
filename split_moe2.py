import sys, time, torch
from exllamav3 import Config, Model
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
config = Config.from_directory(sys.argv[1]); model = Model.from_config(config); model.load(progressbar=False)
mlps, stack = [], list(model.modules)
while stack:
    m = stack.pop()
    if isinstance(m, BlockSparseMLP): mlps.append(m); continue
    stack.extend(getattr(m, "modules", []) or [])
m = mlps[0]; K = m.gates[0].in_features; S = m.num_experts_per_tok
x = torch.randn((1, 1, K), device="cuda", dtype=torch.half) * 0.1
sel = torch.randperm(m.num_experts)[:S].view(1, S).cuda()
wts = torch.full((1, S), 1.0 / S, device="cuda", dtype=torch.half)
def T(f, n=200):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
print("S", S, "E", m.num_experts, "K", K, "I", m.intermediate_size, "fmt", m.gates[0].format, m.downs[0].format)
print("full mlp forward      %.0f us" % T(lambda: m.forward(x, {})))
print("grouped kq (routed+sh) %.0f us" % T(lambda: m._forward_foreign_grouped_kq(x, x.view(1, K), sel, wts, {}, None, None)))
print("shared expert         %.0f us" % T(lambda: m.shared_experts.forward(x, {})))

print("routing_fn            %.0f us" % T(lambda: m.routing_fn(1, m.routing_cfg, x.view(1, K), {})))
gw = m.shared_gate.inner.weight
print("shared gate (torch)   %.0f us" % T(lambda: torch.sigmoid((x.to(torch.float) @ gw.to(torch.float)))))
r = torch.zeros((1, K), device="cuda", dtype=torch.float); se = torch.zeros((1, 1, K), device="cuda", dtype=torch.half)
g = torch.sigmoid((x.to(torch.float) @ gw.to(torch.float)))
print("gate fold + cast      %.0f us" % T(lambda: (r + se.to(torch.float) * g).to(torch.half)))
print("gate dtype", gw.dtype, tuple(gw.shape))
