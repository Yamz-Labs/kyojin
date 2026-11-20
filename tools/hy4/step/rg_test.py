"""rg_test.py: router gemv fp32 vs fp16 weights: logits error, top-16 set agreement on random x, time on GPU (DEV=cuda)."""
import os, statistics, importlib.util
import torch
tree = os.getcwd()
sp = importlib.util.spec_from_file_location("router_gemv", os.path.join(tree, "exllamav3", "loader", "router_gemv.py")); rg = importlib.util.module_from_spec(sp); sp.loader.exec_module(rg)
dev = os.environ.get("DEV", "cuda"); CPU = dev == "cpu"
N, K = (64, 256) if CPU else (512, 3072)
torch.manual_seed(0)
W = torch.randn(N, K, device=dev) * 0.03
W16 = W.half().contiguous()
rg.ENABLED = True
def bench(fn, n=50):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
for T in (1, 2, 6):
    x = torch.randn(T, K, device=dev)
    ref = (x.double() @ W.double().T).float()
    a, b = rg.mm(x, W), rg.mm(x, W16)
    ea, eb = ((a - ref).norm() / ref.norm()).item(), ((b - ref).norm() / ref.norm()).item()
    k = 8 if CPU else 16
    same = (a.topk(k, -1).indices.sort(-1).values == b.topk(k, -1).indices.sort(-1).values).all().item()
    line = f"T {T}: rel err fp32 {ea:.1e} fp16w {eb:.1e} topk same {same}"
    if not CPU: line += f"; fp32 {bench(lambda: rg.mm(x, W)):.0f} us fp16 {bench(lambda: rg.mm(x, W16)):.0f} us"
    print(line, flush=True)
print("RG_DONE", flush=True)
