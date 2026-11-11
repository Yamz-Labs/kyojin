import time, torch
K = 2048
x = torch.randn((1, 1, K), device="cuda", dtype=torch.half) * 0.1
gw = torch.randn((K, 1), device="cuda", dtype=torch.half) * 0.05
gwf = gw.float(); gwr = gw.view(1, K).contiguous()
def T(f, n=300):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
ref = torch.sigmoid(x.float() @ gw.float())
print("current             %.0f us" % T(lambda: torch.sigmoid((x.to(torch.float) @ gw.to(torch.float)))))
print("cached fp32 weight  %.0f us" % T(lambda: torch.sigmoid(x.float() @ gwf)))
v = lambda: torch.sigmoid((x.float() * gwr.float()).sum(-1, keepdim=True))
print("mul+sum             %.0f us" % T(v))
w = lambda: torch.sigmoid(torch.matmul(x, gw).float())
print("fp16 matmul         %.0f us   maxdiff %.2e" % (T(w), (w() - ref).abs().max().item()))
print("cached maxdiff %.2e" % (torch.sigmoid(x.float() @ gwf) - ref).abs().max().item())
