import sys, types, time, importlib.util, os, torch, numpy as np
sys.path.insert(0, "pylib")
D = "/home/kamel/hermes-work/kyolane/src_v185/kyojin/exllamav3/foreign_quant"
spec = importlib.util.spec_from_file_location("fq_gap", D + "/__init__.py", submodule_search_locations=[D])
m = importlib.util.module_from_spec(spec); sys.modules["fq_gap"] = m; spec.loader.exec_module(m)
import fq_gap.gap_formats as gf, fq_gap.triton_gap as tg
N, K = 12288, 4096
dev = "cuda:0"
for fmt in ("q4_0", "iq4_nl", "mxfp4", "q5_0"):
    qk, bb = gf.BLOCKS[fmt]
    rng = np.random.default_rng(2)
    raw = rng.integers(0, 256, size=(N, K // qk, bb), dtype=np.uint8)
    if fmt == "mxfp4": raw[:, :, 0] = rng.integers(120, 135, size=(N, K // qk))
    else: raw[:, :, :2] = rng.uniform(0.001, 0.05, size=(N, K // qk)).astype(np.float16).view(np.uint8).reshape(N, K // qk, 2)
    raw = torch.from_numpy(raw.reshape(N, -1)).to(dev)
    w = gf.dequant(fmt, raw)
    for R in (1, 4, 16):
        x = torch.randn(R, K, device=dev, dtype=torch.half)
        y = tg.gemv(fmt, x, raw); ref = x.float() @ w.float().T
        err = ((y.float() - ref).norm() / ref.norm()).item(); assert err < 2e-3, (fmt, R, err)
        def t(f, n=30):
            for _ in range(5): f()
            torch.cuda.synchronize(); a = time.perf_counter()
            for _ in range(n): f()
            torch.cuda.synchronize(); return (time.perf_counter() - a) / n * 1e3
        tf = t(lambda: tg.gemv(fmt, x, raw)); tb = t(lambda: x @ gf.dequant(fmt, raw).T)
        print(f"{fmt} R={R} err={err:.1e} fused {tf:.3f} ms ({raw.numel()/tf/1e6:.0f} GB/s) vs dequant+matmul {tb:.3f} ms")
