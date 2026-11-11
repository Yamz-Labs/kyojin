"""triton_iq.gemv vs gap_formats.dequant + matmul on real rows of each IQ / Q3_K type in the two unsloth Qwen3-8B GGUFs; then micro GB/s."""
import time, torch, numpy as np
from exllamav3.loader.gguf_collection import GGUFCollection
from exllamav3.foreign_quant.gap_formats import dequant
from exllamav3.foreign_quant import triton_iq as TI
TY = {18: "iq3_xxs", 21: "iq3_s", 17: "iq2_xs", 22: "iq2_s", 11: "q3_k", 10: "q2_k"}
BB = {"iq3_xxs": 98, "iq3_s": 110, "iq2_xs": 74, "iq2_s": 82, "q3_k": 110, "q2_k": 84, "iq2_xxs": 66}
got = {}
for m in ("iq3xxs", "iq2m", "iq2xxs"):
    g = GGUFCollection(f"/home/yaniss/models/foreign/q3_8b_{m}_cfg/model.gguf")
    for k, (ty, dims, off) in g.infos.items():
        f = TY.get(ty)
        if f and f not in got and len(dims) == 2:
            K, N = dims
            n = min(N, 4096)
            t = g.get_tensor(k, device="cuda")
            got[f] = t.reshape(N, -1)[:n].contiguous()
            print(f, k, tuple(got[f].shape), K)
ok = bad = 0
for f, raw in got.items():
    K = raw.shape[1] // BB[f] * 256
    w = dequant(f, raw).float()
    for R in (1, 4):
        x = (torch.randn((R, K), device="cuda") * 0.5).half()
        ref = x.float() @ w.T
        y = TI.gemv(f, x, raw).float()
        rel = ((y - ref).norm() / ref.norm()).item()
        if rel < 2e-3: ok += 1
        else: bad += 1; print("DIFF", f, R, rel)
    x = torch.randn((1, K), device="cuda").half()
    for _ in range(5): TI.gemv(f, x, raw)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50): TI.gemv(f, x, raw)
    torch.cuda.synchronize(); us = (time.perf_counter() - t0) / 50 * 1e6
    print(f"{f:8s} N {raw.shape[0]} K {K}: {us:7.1f} us  {raw.numel() / us / 1e3:6.1f} GB/s (R1)")
print(f"{ok} passed, {bad} failed")
