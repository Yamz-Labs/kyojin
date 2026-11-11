"""moe_fused (gate|up launch + down with SwiGLU staging and weighted fold) == dequant reference of a SwiGLU MoE layer, 2 rows x 4 slots."""
import torch
from exllamav3.modules.quant import fp8 as F

ok = bad = 0
for grid in (True, False):
    I, K, E, S, R = 768, 2048, 8, 4, 2
    g = torch.Generator().manual_seed(7 + grid)
    def mk(n, k):
        q = torch.randint(0, 120, (n, k), dtype=torch.uint8, generator=g).cuda()
        s = ((torch.rand((-(-n // 128), -(-k // 128)), generator=g) if grid else torch.rand(n, generator=g)) * 2e-3 + 1e-3).cuda()
        return q, s
    G = [mk(I, K) for _ in range(E)]; U = [mk(I, K) for _ in range(E)]; D = [mk(K, I) for _ in range(E)]
    tab = lambda L, i: torch.tensor([t[i].data_ptr() for t in L], dtype=torch.int64, device="cuda")
    tabs = (tab(G, 0), tab(G, 1), tab(U, 0), tab(U, 1), tab(D, 0), tab(D, 1))
    h = torch.randn((R, K), device="cuda", dtype=torch.half)
    sel = torch.tensor([3, 0, 7, 5, 1, 6, 2, 4], dtype=torch.int32, device="cuda")
    wts = torch.rand(R * S, device="cuda").half()
    y = F.moe_fused(h, tabs, sel, wts, I, K, grid, (128, 128), S, G[0][1].shape[1] if grid else 1, D[0][1].shape[1] if grid else 1)
    deq = lambda t: F.dequant(t[0], t[1], grid, (128, 128)).float()
    ref = torch.zeros((R, K), device="cuda")
    for r in range(R):
        for j in range(S):
            e = int(sel[r * S + j])
            a = (torch.nn.functional.silu(h[r].float() @ deq(G[e]).T) * (h[r].float() @ deq(U[e]).T)).half().float()
            ref[r] += float(wts[r * S + j]) * (a @ deq(D[e]).T)
    err = (y - ref).abs().max().item()
    if err <= 3e-3 * ref.abs().max().item() + 1e-3:
        ok += 1
    else:
        bad += 1; print("FAIL fused", grid, err, ref.abs().max().item())
print(f"{ok} passed, {bad} failed")
