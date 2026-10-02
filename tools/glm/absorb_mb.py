"""round2 microbench: MLA absorb/unfold tile sweep at GLM shapes (H 64, D_nope 256, D_c 512, D_v 256).
Rotates 11 distinct W_UK/W_UV (16.8 MB each, one per DSA layer) so weights come from DRAM.
absorb: EXL3_MLA_ABSORB_CFG=bn,warps,stages; unfold: EXL3_MLA_UNFOLD_CFG=bk,bn,stages,warps (bk kept at the
default 128 so the K order is unchanged). Gate: output bitwise vs the legacy tile. Prints us/call and GB/s."""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exllamav3.modules.attention_fn import mla_triton as MT
torch.manual_seed(0)
dev = torch.device("cuda:0")
H, DN, DC, DV, L = 64, 256, 512, 256, 11
wuk = [(torch.randn(DC, H * DN, device=dev) * 0.02).half() for _ in range(L)]
wuv = [(torch.randn(DC, H * DV, device=dev) * 0.02).half() for _ in range(L)]
WB = DC * H * DN * 2

def timeit(f, N=10):
    for _ in range(2):
        for l in range(L): f(l)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(N):
        for l in range(L): f(l)
    e1.record(); torch.cuda.synchronize()
    return 1000 * e0.elapsed_time(e1) / (N * L)

AB = os.environ.get("R2_ABSORB_SWEEP", "legacy 128,4,2 128,8,2 64,4,2 64,2,2 64,8,2 32,4,2 32,2,2 32,1,2 16,2,2 16,1,2 64,4,1 32,4,1").split()
UF = os.environ.get("R2_UNFOLD_SWEEP", "legacy 128,256,1,8 128,128,2,4 128,128,1,4 128,64,2,4 128,64,1,4 128,64,2,2 128,32,2,2 128,32,1,2 128,128,2,8").split()
for R in (1, 2, 3):
    q = torch.randn(R, H, DN, device=dev).half()
    ol = torch.randn(H, R, DC, device=dev).half()
    os.environ["EXL3_MLA_ABSORB_CFG"] = "legacy"
    ref = [MT.mla_absorb(q, wuk[l], H, DN).clone() for l in range(L)]
    for c in AB:
        os.environ["EXL3_MLA_ABSORB_CFG"] = c
        try:
            ok = all(torch.equal(MT.mla_absorb(q, wuk[l], H, DN), ref[l]) for l in range(L))
            t = timeit(lambda l: MT.mla_absorb(q, wuk[l], H, DN))
            print(f"absorb R={R} cfg {c:10s} {t:7.1f} us {WB / t / 1e3:6.1f} GB/s bitwise={ok}", flush=True)
        except Exception as ex:
            print(f"absorb R={R} cfg {c} FAIL {type(ex).__name__}: {str(ex)[:120]}", flush=True)
    os.environ["EXL3_MLA_ABSORB_CFG"] = "legacy"
    os.environ["EXL3_MLA_UNFOLD_CFG"] = "legacy"
    ref = [MT.mla_unfold(ol, wuv[l], DV).clone() for l in range(L)]
    for c in UF:
        os.environ["EXL3_MLA_UNFOLD_CFG"] = c
        try:
            ok = all(torch.equal(MT.mla_unfold(ol, wuv[l], DV), ref[l]) for l in range(L))
            t = timeit(lambda l: MT.mla_unfold(ol, wuv[l], DV))
            print(f"unfold R={R} cfg {c:12s} {t:7.1f} us {WB / t / 1e3:6.1f} GB/s bitwise={ok}", flush=True)
        except Exception as ex:
            print(f"unfold R={R} cfg {c} FAIL {type(ex).__name__}: {str(ex)[:120]}", flush=True)
    os.environ["EXL3_MLA_UNFOLD_CFG"] = "legacy"
