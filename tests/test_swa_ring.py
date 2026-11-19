"""HY4_SWA_RING unit tests (CPU interpreter or GPU): window layers keep their K rows in a ring buffer.
 1. ref_fused.mla_sink_attend(ring=RW) over the ring == the same call over the full-length buffer (exact), for decode / verify rows;
 2. kv_fused.write_row(ring=RW) writes slot pos % RW and the same values as the plain row write;
 3. step_nosync.append_kv_ring: scratch = window - 1 history rows + the new rows; the ring holds the last RW positions.
Guard-poisoned buffers detect any out-of-bounds access.   CPU: TRITON_INTERPRET=1 DEV=cpu python tests/test_swa_ring.py"""
import importlib.util, os, sys, types
import torch

DEV = os.environ.get("DEV", "cpu")
here = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(here, "..", "exllamav3", "loader")
for n in ("exllamav3", "exllamav3.loader"):
    m = types.ModuleType(n); m.__path__ = [os.path.join(here, "..", "exllamav3") if n == "exllamav3" else L]; sys.modules[n] = m
import importlib
RF = importlib.import_module("exllamav3.loader.ref_fused")
KVF = importlib.import_module("exllamav3.loader.kv_fused")
NS = importlib.import_module("exllamav3.loader.step_nosync")
sys.path.insert(0, here)
from test_lc_sparse import guarded


def test_attend(N, RW, window, qpos_list, H=4, R=512, P=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(N, R + P, generator=g).to(DEV)
    ring = torch.zeros(RW, R + P, device=DEV)
    for j in range(N):
        ring[j % RW] = full[j]
    T = len(qpos_list)
    ql = torch.randn(T, H, R, generator=g).to(DEV); qp = torch.randn(T, H, P, generator=g).to(DEV)
    sinks = torch.randn(H, generator=g).to(DEV)
    pos = torch.tensor(qpos_list, dtype=torch.long, device=DEV)
    # the full buffer only has to hold positions <= max(qpos): rows past it are what the ring never wrote
    Nf = max(qpos_list) + 1
    want = RF.mla_sink_attend(ql, qp, full[:Nf, :R], full[:Nf, R:], pos, torch.arange(Nf, device=DEV), 0.0625, sinks, window, None, kpos_is_arange=True)
    ring_g, c1 = guarded(ring)
    ql_g, c2 = guarded(ql); qp_g, c3 = guarded(qp); pos_g, c4 = guarded(pos)
    got = RF.mla_sink_attend(ql_g, qp_g, ring_g[:, :R], ring_g[:, R:], pos_g, torch.arange(RW, device=DEV), 0.0625, sinks, window, None, kpos_is_arange=True, ring=RW)
    for c in (c1, c2, c3, c4):
        c()
    assert torch.isfinite(got).all()
    err = (got - want).abs().max().item()
    assert err < 1e-6, err
    return err


def test_write(RW=1024, R=512, P=64):
    g = torch.Generator().manual_seed(3)
    c = torch.randn(1, R + P, generator=g).to(DEV)
    w = torch.randn(R, generator=g).to(DEV)
    tab = torch.randn(5000, 2, P // 2, 2, generator=g).to(DEV)
    for p in (0, 5, 1023, 1024, 3001):
        pos = torch.tensor([p], dtype=torch.long, device=DEV)
        a = torch.zeros(4096, R + P, device=DEV)
        KVF.write_row(c, w, tab, pos, a, R, P, 1e-6)
        b = torch.zeros(RW, R + P, device=DEV)
        b_g, chk = guarded(b)
        KVF.write_row(c, w, tab, pos, b_g, R, P, 1e-6, ring=RW)
        chk()
        assert torch.equal(a[p], b_g[p % RW]), p
        assert int((b_g.abs().sum(1) > 0).sum()) == 1, "only one slot may be written"


def test_append_ring(RW=1024, window=512, W=576):
    class FW: pass
    fw = FW(); fw.dev = torch.device(DEV)
    st = NS.state(fw); st.kv = {0: torch.zeros(RW, W, device=DEV)}
    full = torch.zeros(5000, W, device=DEV)
    g = torch.Generator().manual_seed(9)
    start = 0
    for T in (100, 700, 2048, 3, 1, 1, 8, 1000):
        kn = torch.randn(T, W, generator=g).to(DEV)
        scratch, base = NS.append_kv_ring(fw, 0, kn, start, RW, window)
        full[start:start + T] = kn
        hist = min(start, window - 1)
        assert base == start - hist and scratch.shape[0] == hist + T
        assert torch.equal(scratch, full[base:start + T]), (start, T)
        start += T
        for j in range(max(0, start - RW), start):
            assert torch.equal(st.kv[0][j % RW], full[j])


if __name__ == "__main__":
    print("attend", test_attend(1500, 1024, 512, [1499]), test_attend(3000, 1024, 512, [2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997], H=2),
          test_attend(700, 1024, 512, [690, 699]))
    test_write(); print("write ok")
    test_append_ring(); print("append ring ok")
    print("ALL OK")
