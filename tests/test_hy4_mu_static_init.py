"""CPU check of the static indexer init on a slab view (multi-session start_graph_state): the session binds its slab view as ix.vgb BEFORE the init, so the init must not
assume it created the buffer. 20/11 served crash: a prompt that left an open indexer group (length % k != 0) died with AttributeError "vg_ring" at the second request."""
import os
import sys
import types
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _n, _s in (("exllamav3", ""), ("exllamav3.loader", "loader")):      # the loader package without exllamav3/__init__ (no C++ extension build on a CPU box)
    _m = sys.modules.get(_n) or types.ModuleType(_n)
    _m.__path__ = [os.path.join(ROOT, "exllamav3", _s)]
    sys.modules[_n] = _m

from exllamav3.loader import mu_slots as MS  # noqa: E402
from exllamav3.loader import step_graph as SG  # noqa: E402


def fake_ix(k=4, d=8):
    return SimpleNamespace(k=k, d=d, dev=torch.device("cpu"), pool=None, vgb=None, vg=None, n=0, P=0, _table=lambda n: None)


def test_static_init_on_a_slab_view_with_an_open_group():
    k, d, cap, S = 4, 8, 64, 3
    ix = fake_ix(k, d)
    sl = MS.IxSlabs(ix, S, cap)
    for slot in range(S):
        for n in (60, 61, 63):                                   # 60 % k == 0: no open group (the only case that worked); 61 / 63: open group of 1 / 3 rows
            ix.pool, ix.vgb = sl.view(slot)
            sl.vgb.zero_()
            m = n % k
            vg = torch.randn(m, 2 * d) if m else None
            ix.n, ix.P, ix.vg = n, n // k, vg
            if hasattr(ix, "vg_ring"):
                del ix.vg_ring                                   # the next session binds a fresh view on the same eager object
            SG.indexer_static_init(ix, cap)
            assert ix.vg_ring == 0 and ix.vgb.shape[0] == cap
            if vg is not None:
                assert torch.equal(ix.vgb[n - m:n], vg)
                assert torch.count_nonzero(sl.vgb) == torch.count_nonzero(vg)                                       # nothing outside this session's rows
                assert torch.count_nonzero(sl.vgb[slot * cap:(slot + 1) * cap]) == torch.count_nonzero(vg)


def test_static_init_solo_still_creates_the_buffer():
    ix = fake_ix()
    ix.pool = torch.zeros(4, 8)
    ix.n, ix.P, ix.vg = 2, 0, torch.randn(2, 16)
    SG.indexer_static_init(ix, 64)
    assert ix.vg_ring == (SG.VG_RING if (SG.VG_RING and SG.VG_RING < 64) else 0)
    assert ix.pool.shape[0] == 16 and ix.vgb.shape[0] == (ix.vg_ring or 64)
    assert torch.equal(ix.vgb[torch.arange(0, 2) % ix.vg_ring] if ix.vg_ring else ix.vgb[0:2], ix.vg)


if __name__ == "__main__":
    test_static_init_on_a_slab_view_with_an_open_group()
    test_static_init_solo_still_creates_the_buffer()
    print("ok")
