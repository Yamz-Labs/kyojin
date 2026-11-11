"""moe_gemv (grouped FP8 expert GEMV through pointer tables) == per-slot dequant + matmul; grid and per-row scales, xdiv 4 and 1."""
import torch
from exllamav3.modules.quant import fp8 as F

ok = bad = 0
for grid in (True, False):
    for (N, K) in ((768, 2048), (2048, 768)):
        g = torch.Generator().manual_seed(N + K + grid)
        E = 8
        qs = [torch.randint(0, 120, (N, K), dtype=torch.uint8, generator=g).cuda() for _ in range(E)]
        ss = [((torch.rand((-(-N // 128), -(-K // 128)), generator=g) if grid else torch.rand(N, generator=g)) * 2e-3 + 1e-3).cuda() for _ in range(E)]
        qtab = torch.tensor([q.data_ptr() for q in qs], dtype=torch.int64, device="cuda")
        stab = torch.tensor([s.data_ptr() for s in ss], dtype=torch.int64, device="cuda")
        for xdiv in (4, 1):
            sel = torch.tensor([3, 0, 7, 5, 1, 1, 6, 2], dtype=torch.int32, device="cuda")
            rows = 8 // xdiv
            x = torch.randn((rows, K), device="cuda", dtype=torch.half)
            y = F.moe_gemv(x, qtab, stab, sel, N, K, grid, (128, 128), xdiv, ss[0].shape[1] if grid else 1)
            ref = torch.stack([x[i // xdiv].float() @ F.dequant(qs[int(e)], ss[int(e)], grid, (128, 128)).float().T for i, e in enumerate(sel.tolist())])
            err = (y - ref).abs().max().item()
            if err <= 2e-3 * ref.abs().max().item() + 1e-3:
                ok += 1
            else:
                bad += 1; print("FAIL", grid, N, K, xdiv, err)
print(f"{ok} passed, {bad} failed")
