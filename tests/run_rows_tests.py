"""triton_rows (Q8_0 / NVFP4, 2..16 rows) == ForeignLinear dequant + matmul; real-valued payloads."""
import torch
from exllamav3.modules.foreign_linear import ForeignLinear
from exllamav3.foreign_quant import triton_rows as TR

ok = bad = 0
N, K = 640, 2048
g = torch.Generator().manual_seed(3)
for fmt in ("q8_0", "nvfp4"):
    fl = ForeignLinear(None, "x", K, N, fmt)
    if fmt == "q8_0":
        q = torch.randint(-127, 128, (N, K // 32, 32), dtype=torch.int8, generator=g)
        d = (torch.rand((N, K // 32, 1), generator=g) * 1e-2 + 1e-3).half()
        fl.packed = torch.cat([d.view(torch.uint8), q.view(torch.uint8)], dim=2).reshape(N, -1).cuda()
    else:
        fl.packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g).cuda()
        fl.scales = torch.randint(0x28, 0x46, (N, K // 16), dtype=torch.uint8, generator=g).cuda()
        fl.global_scale = 0.01
    w = fl._dequant_rows(0, N).float()
    for R in (2, 4, 9, 16):
        x = torch.randn((R, K), device="cuda", dtype=torch.half)
        ref = x.float() @ w.T
        y = TR.gemv(fmt, x, fl.packed, fl.scales, fl.global_scale if fl.global_scale is not None else 1.0)
        err = (y - ref).abs().max().item()
        if err <= 3e-3 * ref.abs().max().item() + 1e-3:
            ok += 1
        else:
            bad += 1; print("FAIL", fmt, R, err, ref.abs().max().item())
print(f"{ok} passed, {bad} failed")
