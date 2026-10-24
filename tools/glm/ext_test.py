import os, sys, torch
from exllamav3.modules.attention_fn import mla_triton as MT
from exllamav3.ext import exllamav3_ext as ext
assert hasattr(ext, "mla_absorb_dec")
dev = torch.device("cuda:0"); torch.manual_seed(1); DN = DV = 256
QK = int(os.environ.get("QK", "256"))   # GLM-5.3: NoPE, q head = 256; 320 = a rope-padded head
def both(f, *a):
    os.environ["EXL3_MLA_DEC_VALU"] = "1"; x = f(*a); os.environ["EXL3_MLA_DEC_VALU"] = "0"; y = f(*a); os.environ["EXL3_MLA_DEC_VALU"] = "1"
    torch.cuda.synchronize(); return torch.equal(x.view(torch.short), y.view(torch.short))
def check(H, DC):
    ok = True
    for sc in (0.02, 1.0):
        wuk = (torch.randn(DC, H*DN, device=dev)*sc).half(); wuv = (torch.randn(DC, H*DV, device=dev)*sc).half()
        for R in range(1, 9):
            qf = torch.randn(R + 2, H, QK, device=dev).half(); q = qf[1:1+R]   # offset view: nonzero storage offset
            o = torch.randn(H, R, DC, device=dev).half()
            a = both(MT.mla_absorb, q, wuk, H, DN); b = both(MT.mla_unfold, o, wuv, DV); ok &= a and b
            if not (a and b): print("MISMATCH", H, DC, sc, R, a, b)
    # row identity: R rows == R single-row calls
    q = torch.randn(8, H, QK, device=dev).half(); o = torch.randn(H, 8, DC, device=dev).half()
    full = MT.mla_absorb(q, wuk, H, DN); fu = MT.mla_unfold(o, wuv, DV)
    for r in range(8):
        ok &= torch.equal(MT.mla_absorb(q[r:r+1], wuk, H, DN)[:, 0].view(torch.short), full[:, r].view(torch.short))
        ok &= torch.equal(MT.mla_unfold(o[:, r:r+1].contiguous(), wuv, DV)[0].view(torch.short), fu[r].view(torch.short))
    print(f"QK={QK} H={H} DC={DC}: ext ops vs Triton, R=1..8, 2 scales, row-identity: {'ALL EQUAL' if ok else 'FAIL'}", flush=True); return ok
if not check(2, 256) or not check(64, 512): sys.exit(1)
H, DC, L = 64, 512, 11
wuk = [(torch.randn(DC, H*DN, device=dev)*0.02).half() for _ in range(L)]; wuv = [(torch.randn(DC, H*DV, device=dev)*0.02).half() for _ in range(L)]
def timeit(f, N=20):
    for l in range(L): f(l)
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(N):
        for l in range(L): f(l)
    b.record(); torch.cuda.synchronize(); return 1000*a.elapsed_time(b)/(N*L)
for R in (1, 2, 3, 4, 8):
    q = torch.randn(R, H, QK, device=dev).half(); o = torch.randn(H, R, DC, device=dev).half()
    for m in ("0", "1"):
        os.environ["EXL3_MLA_DEC_VALU"] = m
        print(f"R={R} VALU={m} absorb {timeit(lambda l: MT.mla_absorb(q, wuk[l], H, DN)):6.1f} us  unfold {timeit(lambda l: MT.mla_unfold(o, wuv[l], DV)):6.1f} us", flush=True)
