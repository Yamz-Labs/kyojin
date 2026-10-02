import os, sys, torch, importlib
WT = "~/kyojin"
sys.path.insert(0, WT + "/ref"); import frac_reconstruct as fr
name = sys.argv[1]
sys.path.insert(0, WT + "/build/devext/" + name); m = importlib.import_module(name)
dev = "cuda:0"
scratch = torch.zeros(1 << 20, device = dev); counters = torch.zeros(4096, dtype = torch.int, device = dev)
g = torch.Generator().manual_seed(0)
for K in (2, 4):
    k, n = 512, 512
    tr = torch.randint(-32768, 32767, (k // 16, n // 16, int(16 * K)), dtype = torch.int16, generator = g)
    W = fr.reconstruct(tr, K, fr.CB_MUL1).float()
    one = torch.ones(max(k, n), dtype = torch.half)
    for label, x in (("e0", torch.eye(k)[0:1]), ("e1", torch.eye(k)[1:2]), ("e17", torch.eye(k)[17:18]), ("rand", torch.randn(1, k) * 0.5)):
        x = x.half()
        out = torch.empty(1, n, dtype = torch.float, device = dev)
        m.gemv(x.to(dev), tr.to(dev), one[:k].to(dev), one[:n].to(dev), out, scratch, counters, float(K))
        ref = x.float() @ W
        o = out.cpu()
        err = (o - ref).abs()
        print(f"K={K} {label}: maxerr {err.max():.3e} ref absmax {ref.abs().max():.3e} out[:6] {o[0,:6].tolist()} ref[:6] {ref[0,:6].tolist()}")
        if label == "e0":
            bad = (err[0] > 1e-2).nonzero().flatten()[:10].tolist()
            print("   bad cols", bad, "n_bad", int((err[0] > 1e-2).sum()))
