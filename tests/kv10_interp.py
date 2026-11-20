"""Triton interpreter checks of the kv10 kernel changes on CPU tensors (TRITON_INTERPRET=1, no GPU): rope rows vs the full table, fp16 pool store. Run by tests/test_kv10.py in a subprocess."""
import importlib
import os
import sys
import types

os.environ["TRITON_INTERPRET"] = "1"
import torch

LOADER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3", "loader")
pkg = types.ModuleType("kvpk")
pkg.__path__ = [LOADER]
sys.modules["kvpk"] = pkg
RR = importlib.import_module("kvpk.rope_rows")
KF = importlib.import_module("kvpk.kv_fused")
PF = importlib.import_module("kvpk.pool_fused")
torch.manual_seed(0)
BASE, P, R, T, N = 1e7, 64, 512, 5, 4096
pos = torch.tensor([3, 4090, 777, 778, 1500], dtype=torch.long)
tab = RR.attn_rows(P, BASE, torch.arange(N))                               # the full table (what attn_eager._build holds)
rows = RR.attn_rows(P, BASE, pos)
assert torch.equal(rows, tab[pos])

# kv write: table indexed by position == rows indexed by row, ring on and off
c = torch.randn(T, R + P)
w = torch.randn(R)
for ring in (0, 1024):
    k0, k1 = torch.zeros(N, R + P), torch.zeros(N, R + P)
    KF.write_row(c, w, tab, pos, k0, R, P, 1e-6, ring=ring)
    KF.write_row(c, w, rows, pos, k1, R, P, 1e-6, ring=ring, per_row=True)
    assert torch.equal(k0, k1) and k0.abs().sum() > 0, ring
# q rope
H, NOPE = 4, 128
q = torch.randn(T, H, NOPE + P)
assert torch.equal(KF.rope_q(q, tab, pos, NOPE, P), KF.rope_q(q, rows, pos, NOPE, P, True))

# pool update: cos / sin of the group start by position vs by row; fp16 pool store
k, d, rd, VR = 4, 128, 64, 0
vgb = torch.randn(N, 2 * d)
ape = torch.randn(k, d)
pnw = torch.randn(d)
cs_t, sn_t = RR.ix_rows(BASE, rd, torch.arange(N))                          # table [N, 32] (cos | sin)
cs_r, sn_r = RR.ix_rows(BASE, rd, pos, k)                                   # rows of the group starts of `pos`
assert torch.equal(cs_r, cs_t[(pos // k) * k]) and torch.equal(sn_r, sn_t[(pos // k) * k])
p32, p32r, p16 = torch.zeros(N // k, d), torch.zeros(N // k, d), torch.zeros(N // k, d, dtype=torch.float16)
PF._pool_kernel[(T,)](vgb, pos, ape, pnw, cs_t, sn_t, p32, 1e-6, k=k, d=d, rd=rd, VR=VR)
PF._pool_kernel[(T,)](vgb, pos, ape, pnw, cs_r, sn_r, p32r, 1e-6, k=k, d=d, rd=rd, VR=VR, TROW=1)
PF._pool_kernel[(T,)](vgb, pos, ape, pnw, cs_r, sn_r, p16, 1e-6, k=k, d=d, rd=rd, VR=VR, TROW=1)
g = pos // k
assert torch.equal(p32[g], p32r[g]) and p32[g].abs().sum() > 0
assert torch.equal(p16[g], p32r[g].half())                                   # the fp16 pool holds the fp32 value rounded to nearest
# the sparse selection kernel on an fp16 pool (what HY4_IX_POOL16 feeds it) vs the fp32 pool: group sets of the top-k
LS = importlib.import_module("kvpk.lc_sparse")
Pb, Hh, Tt, n_top = 1024, 16, 4, 127
pool = torch.randn(Pb, 128)
pool = pool / pool.pow(2).mean(-1, keepdim=True).sqrt()
iq = torch.randn(Tt, Hh, 128)
wt = torch.rand(Tt, Hh) / 45
qpos = torch.tensor([Pb * 4 + 8] * Tt, dtype=torch.long)
i32, n32 = LS.select_groups(iq, wt, pool, qpos, Pb, n_top, 4, True)
i16, n16 = LS.select_groups(iq, wt, pool.half().contiguous(), qpos, Pb, n_top, 4, True)
ov = [len(set(i32[t, :int(n32[t])].tolist()) & set(i16[t, :int(n16[t])].tolist())) / max(int(n32[t]), 1) for t in range(Tt)]
assert min(ov) > 0.95 and int(n32[0]) == n_top, ov
print("KV10_SELECT_OVERLAP min %.4f" % min(ov))
print("KV10_INTERP_OK")
