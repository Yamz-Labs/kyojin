"""OpenAI MXFP4 -> GGUF MXFP4 repack: ForeignLinear('mxfp4') over the repacked bytes == _mxfp4_dequant reference (dequant rows + matmul)."""
import torch
from exllamav3.modules.linear import _mxfp4_dequant, _mxfp4_to_gguf
from exllamav3.modules.foreign_linear import ForeignLinear

ok = bad = 0
g = torch.Generator().manual_seed(9)
for N, K in ((96, 2880), (256, 512)):
    blocks = torch.randint(0, 256, (N, K // 32, 16), dtype=torch.uint8, generator=g).cuda()
    scales = torch.randint(118, 132, (N, K // 32), dtype=torch.uint8, generator=g).cuda()
    w = _mxfp4_dequant(blocks, scales).float()                 # (N, K)
    fl = ForeignLinear(None, "x", K, N, "mxfp4")
    fl.packed = _mxfp4_to_gguf(blocks, scales)
    d = fl._dequant_rows(0, N).float()
    if torch.equal(d, w):
        ok += 1
    else:
        bad += 1; print("DEQ DIFF", N, K, (d - w).abs().max().item())
    for R in (1, 4):
        x = torch.randn((R, K), device="cuda", dtype=torch.half)
        y = fl.forward(x, {}).float()
        ref = x.float() @ w.T
        err = (y - ref).abs().max().item()
        if err <= 3e-3 * ref.abs().max().item() + 1e-2:
            ok += 1
        else:
            bad += 1; print("FWD", N, K, R, err, ref.abs().max().item())
print(f"{ok} passed, {bad} failed")
