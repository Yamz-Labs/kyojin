"""CPU check of MultiDecoder.step_slots with a fake captured graph: GraphDecoder._graph returns (list of graph segments, outputs) (the Engram split of the solo step),
so step_slots must replay every segment, in capture order, once per step (the 1.9.0 clone called .replay() on the list and died at the first spec round)."""
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

from exllamav3.loader import step_mu  # noqa: E402


class FakeCG:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def replay(self):
        self.log.append(self.name)


def make(nseg):
    log = []
    out = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    d = object.__new__(step_mu.MultiDecoder)
    d.fw = SimpleNamespace(_h=None)
    d.gs = SimpleNamespace(inp=torch.zeros(3, step_mu.TM, dtype=torch.long), Nb=0)
    d.hin = torch.zeros(3, step_mu.TM, dtype=torch.long)
    d.cap, d.prepared, d.ev_set = 4096, True, False
    d.stats = {"replays": 0}
    d.ev = SimpleNamespace(record=lambda: log.append("ev"))
    d._stage_rows = lambda rows: log.append("stage")
    d._graph = lambda Nb, T: ([FakeCG(f"seg{i}", log) for i in range(nseg)], out)
    return d, log


def sess(slot, n=5):
    return SimpleNamespace(slot=slot, n=n, tokens=list(range(n)))


def test_step_slots_replays_every_segment_in_order():
    for nseg in (1, 3):
        d, log = make(nseg)
        a, b = sess(0), sess(1, 9)
        r = d.step_slots([(a, [7, 8]), (b, [9])])
        assert log == ["stage"] + [f"seg{i}" for i in range(nseg)] + ["ev"], log
        assert (a.n, b.n) == (7, 10) and d.stats["replays"] == 1 and len(r) == 4
        assert d.hin[1, :3].tolist() == [5, 6, 9] and d.hin[2, :3].tolist() == [0, 0, d.cap]


if __name__ == "__main__":
    test_step_slots_replays_every_segment_in_order()
    print("ok")
