"""IQ2_XXS rows 2..16 through triton_iq (FMT 5) vs the C++ dequant (ext.fq_dequant_iq2_xxs) + fp32 matmul, random blocks with sane scales."""
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.foreign_quant import triton_iq as T
torch.manual_seed(0)
ok = bad = 0
for N, K in ((512, 2048), (1024, 4096), (96, 768 * 4)):
    Kb = K // 256
    p = torch.randint(0, 256, (N, Kb, 66), dtype=torch.uint8)
    p[:, :, 0:2] = torch.full((N, Kb, 1), 2e-3, dtype=torch.float16).view(torch.uint8).view(N, Kb, 2)
    p = p.cuda()
    w = torch.empty((N, K), dtype=torch.half, device="cuda")
    ext.fq_dequant_iq2_xxs(p.contiguous(), w)
    for R in (2, 4, 8, 16):
        x = torch.randn((R, K), device="cuda").half()
        ref = x.float() @ w.float().t()
        y = T.gemv("iq2_xxs", x, p.reshape(N, -1)).float()
        rel = ((y - ref).norm() / ref.norm()).item()
        if rel < 3e-3: ok += 1
        else: bad += 1; print("DIFF", N, K, R, rel)
print(f"{ok} passed, {bad} failed", flush=True)
