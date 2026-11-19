"""bg_dot_test.py: HY4_BG_DOT kernel (k_b: H 64, N 512, K 128; v_b: H 64, N 128, K 512) vs the shipped fp32 row-batched kernel and an fp64 reference. Error always, time on GPU (DEV=cuda)."""
import os, statistics, importlib.util
import numpy as np, torch
tree = os.getcwd()
sp = importlib.util.spec_from_file_location("mla_bgemv", os.path.join(tree, "exllamav3", "loader", "mla_bgemv.py")); bg = importlib.util.module_from_spec(sp); sp.loader.exec_module(bg)
dev = os.environ.get("DEV", "cuda"); CPU = dev == "cpu"
torch.manual_seed(0); rng = np.random.default_rng(0)
def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
for name, H, N, K in (("k_b", 8 if CPU else 64, 64 if CPU else 512, 128), ("v_b", 8 if CPU else 64, 32 if CPU else 128, 256 if CPU else 512)):
    raw = np.zeros((H * N, K // 32, 34), dtype=np.uint8)
    raw[:, :, 0:2] = (rng.standard_normal((H * N, K // 32)) * 0.02).astype(np.float16).view(np.uint8).reshape(H * N, K // 32, 2)
    raw[:, :, 2:] = rng.integers(-127, 128, (H * N, K // 32, 32), dtype=np.int16).astype(np.int8).view(np.uint8)
    m = bg.Q8Batched(raw, H, N, K, dev)
    w = (m.q.double() * m.d.double().repeat_interleave(32, dim=2))
    for T in (2, 3, 6):
        x = torch.randn(T, H, K, device=dev)
        ref = torch.einsum("hnk,thk->thn", w, x.double()).float()
        bg.DOT = False; a = m.mv(x)
        for bn, nw in (((16, 4),) if CPU else ((64, 4), (32, 4), (128, 4), (64, 8), (16, 2))):
            bg.DOT, bg.DOT_BN, bg.DOT_NW = True, bn, nw
            try: b = m.mv(x)
            except Exception as e: print(f"{name} T {T} BN {bn} NW {nw}: FAIL {type(e).__name__} {str(e)[:80]}", flush=True); continue
            ea = ((a - ref).norm() / ref.norm()).item(); eb = ((b - ref).norm() / ref.norm()).item()
            line = f"{name} T {T} BN {bn} NW {nw}: rel err old {ea:.1e} dot {eb:.1e}"
            if not CPU:
                bg.DOT = False; ta = bench(lambda: m.mv(x)); bg.DOT = True; tb = bench(lambda: m.mv(x)); line += f"; old {ta:.0f} us dot {tb:.0f} us"
            print(line, flush=True)
print("BG_DOT_DONE", flush=True)
