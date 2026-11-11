import time, torch
def T(f, n=200):
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
for K, N in ((2880, 4096), (2880, 512), (4096, 2880), (2880, 32), (2880, 201088)):
    w = (torch.randn((K, N), device="cuda") * 0.02).half(); x = torch.randn((1, K), device="cuda").half()
    ws = [ (torch.randn((K, N), device="cuda") * 0.02).half() for _ in range(max(1, min(8, int(4e8 // (K * N * 2))))) ]
    i = [0]
    def f():
        i[0] += 1; torch.matmul(x, ws[i[0] % len(ws)])
    us = T(f, 100 if N > 100000 else 300)
    print(f"x[1,{K}] @ W[{K},{N}]: {us:7.1f} us  {K * N * 2 / us / 1e3:6.1f} GB/s", flush=True)
