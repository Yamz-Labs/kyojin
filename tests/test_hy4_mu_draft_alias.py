"""CPU check of MTPGraphMU.run with fake captured graphs that emulate the shared memory pool: the (slot, N) draft graphs capture into ONE pool, so a graph's output can sit in a
block an earlier-captured graph uses for intermediates. Replaying the earlier graph AFTER the later one (two sessions whose slots were first drafted in the other order, the
speculative round of a 3rd / later request pair) overwrote the later graph's drafts before the host read them (garbage ids -> HSA 0x1016 in the verify replay).
run() must hand back a private copy: every session's drafts read after both replays equal what its own replay wrote."""
import os
import sys
import types
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _n, _s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    _m = sys.modules.get(_n) or types.ModuleType(_n)
    _m.__path__ = [os.path.join(ROOT, "exllamav3", _s)]
    sys.modules[_n] = _m

from exllamav3.loader import mtp_head as MH  # noqa: E402
from exllamav3.loader import step_mu  # noqa: E402

K = 3
GARBAGE = 10 ** 9


class AliasedGraph:
    """replay writes the drafts of the bound slot (token + j .. ) into `out`; `clobbers` = the outputs of later-captured graphs that live in this graph's intermediate blocks"""

    def __init__(self, out, clobbers=()):
        self.out, self.clobbers, self.slot_tok = out, list(clobbers), None

    def replay(self):
        for c in self.clobbers:
            c.fill_(GARBAGE)                       # the intermediates of this graph overwrite whatever later captures put in the shared blocks
        self.out.copy_(self.inp[0] + torch.arange(1, K + 1))      # drafts = f(token): distinguishes the sessions


def make():
    pool = torch.zeros(2 * K, dtype=torch.long)
    out1, out2 = pool[:K], pool[K:]               # graph 1 (captured first) and graph 2 (captured second)
    d = object.__new__(step_mu.MTPGraphMU)
    d.mt = SimpleNamespace(cap=4096)
    d.k = K
    d.hin = torch.zeros(2, dtype=torch.long)
    d.inp = torch.zeros(2, dtype=torch.long)
    d.h0 = torch.zeros(1, 4)
    d.ev, d.ev_set = SimpleNamespace(synchronize=lambda: None, record=lambda: None), False
    g1, g2 = AliasedGraph(out1, clobbers=[out2]), AliasedGraph(out2)
    for g in (g1, g2):
        g.inp = d.inp
    d.graphs = {(1, 512): (g1, out1), (2, 512): (g2, out2)}
    d._graph = lambda N: d.graphs[(d.slot, N)]
    return d


def drafts_after(run, reverse):
    d = make()
    order = [(2, 100), (1, 200)] if reverse else [(1, 200), (2, 100)]
    outs = {}
    for slot, tok in order:
        d.slot = slot
        outs[slot] = run(d, tok, torch.zeros(1, 4), 5)
    return {s: o.tolist() for s, o in outs.items()}


def test_naive_pool_aliasing_is_real_in_the_emulation():
    naive = lambda d, tok, h, j: MH.MTPGraph.run(d, tok, h, j)
    assert drafts_after(naive, reverse=False) == {1: [201, 202, 203], 2: [101, 102, 103]}      # capture order: safe
    got = drafts_after(naive, reverse=True)                                                      # graph 2 replayed, then graph 1: graph 2's drafts are garbage
    assert got[1] == [201, 202, 203] and got[2] == [GARBAGE] * K


def test_run_returns_private_drafts_in_either_replay_order():
    fixed = lambda d, tok, h, j: step_mu.MTPGraphMU.run(d, tok, h, j)
    for reverse in (False, True):
        assert drafts_after(fixed, reverse) == {1: [201, 202, 203], 2: [101, 102, 103]}, reverse


if __name__ == "__main__":
    test_naive_pool_aliasing_is_real_in_the_emulation()
    test_run_returns_private_drafts_in_either_replay_order()
    print("ok")
