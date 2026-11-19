"""indexer vg ring: pool_fused.pool_update with a ring vgb (VR rows) == with the full-length vgb, positions beyond VR, batch of rows (T = 8 and a 300-row call)"""
import importlib, os, sys, types
import torch
DEV = os.environ.get("DEV", "cpu")
here = os.path.dirname(os.path.abspath(__file__))
for n in ("exllamav3", "exllamav3.loader"):
    m = types.ModuleType(n); m.__path__ = [os.path.join(here, "..", "exllamav3") if n == "exllamav3" else os.path.join(here, "..", "exllamav3", "loader")]; sys.modules[n] = m
PF = importlib.import_module("exllamav3.loader.pool_fused")
sys.path.insert(0, here)
from test_lc_sparse import guarded

def run(T, start, VR, k=4, d=128, rd=64, cap=2048):
    g = torch.Generator().manual_seed(start)
    full = torch.randn(cap, 2 * d, generator=g).to(DEV)
    ring = torch.zeros(VR, 2 * d, device=DEV)
    for j in range(cap):
        if j < start + T:
            ring[j % VR] = full[j]
    class IX: pass
    ape = torch.randn(k, d, generator=g).to(DEV); pn = torch.randn(d, generator=g).to(DEV)
    cs = torch.randn(cap, rd // 2, generator=g).to(DEV); sn = torch.randn(cap, rd // 2, generator=g).to(DEV)
    pos = torch.arange(start, start + T, device=DEV)
    outs = []
    for vgb, vr in ((full, 0), (ring, VR)):
        ix = IX(); ix.w = {"ape": ape, "pool_norm": pn}; ix.eps = 1e-5; ix.k = k; ix.d = d; ix.rope_dim = rd; ix.pool = torch.zeros(cap // k, d, device=DEV)
        ix._max_pos = cap; ix._table = lambda n, cs=cs, sn=sn: (cs, sn); ix.vg_ring = vr
        vgb_g, chk = guarded(vgb)
        ix.vgb = vgb_g
        PF.pool_update(ix, pos)
        chk(); outs.append(ix.pool.clone())
    done = torch.arange(cap // k, device=DEV) * k + (k - 1) <= start + T - 1      # groups complete after the call (an open group is partial data, masked by the indexer)
    done &= torch.arange(cap // k, device=DEV) >= start // k
    assert int(done.sum()) >= 1 and torch.equal(outs[0][done], outs[1][done]), (T, start)
for T, start, VR in ((1, 1003, 64), (8, 1001, 64), (300, 700, 4096), (8, 5, 64), (2048 - 2048 + 16, 1500, 64)):
    run(T, start, VR)
print("ALL OK")
