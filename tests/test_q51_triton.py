"""Q5_1 dequant: Triton kernel vs the torch chain, bit-identical on random blocks (finite fp16 d / m, all qh / qs bit patterns)."""
import os, torch
os.environ["EXL3_Q51_TRITON"] = "1"
import exllamav3.foreign_quant.gap_formats as gf

torch.manual_seed(0)
bad = 0
for rows, nb in ((1, 1), (3, 5), (640, 80), (2560, 20), (7, 513)):
    raw = torch.randint(0, 256, (rows, nb, 24), dtype = torch.uint8)
    dm = (torch.randn(rows, nb, 2) * 0.01).half().view(torch.uint8).reshape(rows, nb, 4)
    raw[:, :, 0:4] = dm
    raw = raw.reshape(rows, nb * 24).cuda()
    gf._Q51_TRITON = True
    a = gf.dequant("q5_1", raw)
    gf._Q51_TRITON = False
    b = gf.dequant("q5_1", raw)
    same = torch.equal(a.view(torch.int16), b.view(torch.int16))
    bad += not same
    print(f"rows {rows} nb {nb}: {'equal' if same else 'DIFF max ' + str((a.float() - b.float()).abs().max().item())}")
print("PASS" if bad == 0 else f"FAIL ({bad})")
