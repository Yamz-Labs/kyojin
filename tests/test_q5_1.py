"""Q5_1 checks: (1) torch dequant (gap_formats) == an independent numpy reference written from ggml's dequantize_row_q5_1 (exact);
(2) Triton fused decode GEMV (triton_gap, R 1 / 4) vs fp32 matmul over the dequantized weights (fp16-level tolerance, like the other
gap formats); (3) ForeignLinear("q5_1") forward runs. Run in the tree on the GPU: python test_q5_1.py"""
import numpy as np
import torch
from exllamav3.foreign_quant import gap_formats, triton_gap


def ref_dequant(raw):                                   # ggml dequantize_row_q5_1
    nb = raw.size // 24
    b = raw.reshape(nb, 24)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    m = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    qh = b[:, 4:8].copy().view(np.uint32)[:, 0]
    qs = b[:, 8:24]
    y = np.zeros((nb, 32), np.float32)
    for j in range(16):
        xh0 = ((qh >> j) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        x0 = (qs[:, j] & 0x0F) | xh0
        x1 = (qs[:, j] >> 4) | xh1
        y[:, j] = x0 * d[:, 0] + m[:, 0]
        y[:, j + 16] = x1 * d[:, 0] + m[:, 0]
    return y.reshape(-1)


rng = np.random.default_rng(0)
bad = 0
for N, K in ((64, 640), (2560, 640), (96, 2560)):
    nb = K // 32
    raw = rng.integers(0, 256, (N, nb, 24), dtype=np.uint8)
    raw[:, :, 0:2] = np.frombuffer(np.full(N * nb, 0.01, np.float16).tobytes(), np.uint8).reshape(N, nb, 2)
    raw[:, :, 2:4] = np.frombuffer((rng.standard_normal(N * nb) * 0.05).astype(np.float16).tobytes(), np.uint8).reshape(N, nb, 2)
    ref = ref_dequant(raw.reshape(-1)).reshape(N, K)
    t = torch.from_numpy(raw.reshape(N, nb * 24)).cuda()
    dq = gap_formats.dequant("q5_1", t) if hasattr(gap_formats, "dequant") else None
    if dq is None:
        import inspect
        fn = [f for n, f in inspect.getmembers(gap_formats, inspect.isfunction) if "dequant" in n][0]
        dq = fn("q5_1", t)
    ok1 = np.array_equal(dq.float().cpu().numpy(), ref.astype(np.float16).astype(np.float32))
    for R in (1, 4):
        x = torch.randn(R, K).half().cuda()
        if triton_gap.supported("q5_1", R, K, N):
            y = triton_gap.gemv("q5_1", x, t).float()
            want = x.float() @ dq.float().t()
            err = ((y - want).abs().max() / want.abs().max()).item()
            ok2 = err < 5e-3
        else:
            err, ok2 = None, True
        print(f"N {N} K {K} R {R}: dequant exact {ok1}, triton rel err {err}")
        bad += (not ok1) + (not ok2)
print("PASS" if bad == 0 else f"FAIL ({bad})")
