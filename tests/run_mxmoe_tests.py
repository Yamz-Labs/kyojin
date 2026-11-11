"""Grouped MXFP4 expert GEMV == per-expert ForeignLinear('mxfp4') forward + bias, for every (row, expert) slot."""
import torch
from exllamav3.modules.linear import _mxfp4_to_gguf
from exllamav3.modules.foreign_linear import ForeignLinear
from exllamav3.foreign_quant import triton_moe_mxfp4 as MX

g = torch.Generator().manual_seed(3)
ok = bad = 0
for E, N, K, R, S in ((8, 2944, 2944, 1, 4), (8, 384, 512, 3, 2)):
    fls, bs = [], []
    for _ in range(E):
        blocks = torch.randint(0, 256, (N, K // 32, 16), dtype=torch.uint8, generator=g).cuda()
        scales = torch.randint(118, 128, (N, K // 32), dtype=torch.uint8, generator=g).cuda()
        fl = ForeignLinear(None, "x", K, N, "mxfp4")
        fl.packed = _mxfp4_to_gguf(blocks, scales)
        fls.append(fl)
        bs.append(torch.randn(N, device="cuda", dtype=torch.half))
    wtab = torch.tensor([f.packed.data_ptr() for f in fls], dtype=torch.int64, device="cuda")
    btab = torch.tensor([b.data_ptr() for b in bs], dtype=torch.int64, device="cuda")
    sel = torch.randint(0, E, (R * S,), generator=g).to(torch.int32).cuda()
    x = torch.randn((R, K), device="cuda", dtype=torch.half)
    for has_b in (False, True):
        y = MX.moe_gemv(x, wtab, btab if has_b else None, sel, N, K, S)
        for slot in range(R * S):
            e = int(sel[slot])
            ref = fls[e]._dequant_rows(0, N).float() @ x[slot // S].float()
            if has_b:
                ref = ref + bs[e].float()
            err = (y[slot] - ref).abs().max().item()
            if err <= 2e-3 * ref.abs().max().item() + 1e-2:
                ok += 1
            else:
                bad += 1
                print("DIFF", E, N, K, R, S, has_b, slot, err, ref.abs().max().item())
print(f"{ok} passed, {bad} failed")
