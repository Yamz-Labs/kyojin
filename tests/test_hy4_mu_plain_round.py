"""CPU check of MuEngine.plain_round with a fake decoder: one token per session, and the per-session state a later speculative round starts from (hcur = trunk state of the
position just fed, n-gram history) stays current. Plain rounds run above HY4_MU_SPEC_MAXB sessions; when sessions finish, the same sessions go back to spec_round."""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "hy4"))
import mu_engine as ME  # noqa: E402


class FakeNG:
    def __init__(self):
        self.seen = []

    def add(self, toks):
        self.seen += list(toks)


def make(D=4):
    eng = object.__new__(ME.MuEngine)
    log = []

    def step_slots(rows):
        T = sum(len(t) for _, t in rows)
        log.append([(s.slot, list(t)) for s, t in rows])
        am = torch.arange(100, 100 + T)
        hv = torch.arange(T * D, dtype=torch.float32).reshape(T, D)
        return None, am, None, hv

    eng.GR = SimpleNamespace(step_slots=step_slots)
    return eng, log


class Sess:
    def __init__(self, slot, y, ng=None):
        self.slot, self.y, self.ng, self.hcur = slot, y, ng, None


def sess(slot, y, ng=None):
    return Sess(slot, y, ng)


def test_plain_round_one_row_each_and_state_current():
    eng, log = make()
    a, b, c = sess(0, 7, FakeNG()), sess(1, 8), sess(2, 9, FakeNG())
    out = eng.plain_round([a, b, c])
    assert log == [[(0, [7]), (1, [8]), (2, [9])]]
    assert out == {a: [100], b: [101], c: [102]} and (a.y, b.y, c.y) == (100, 101, 102)
    assert torch.equal(b.hcur, torch.arange(4, 8, dtype=torch.float32)[None])        # the trunk state of ITS row, not of row 0
    assert a.ng.seen == [100] and c.ng.seen == [102] and b.ng is None
    eng.plain_round([c])                                                              # single row: shapes of the T = 1 step
    assert c.y == 100 and c.ng.seen == [102, 100]


if __name__ == "__main__":
    test_plain_round_one_row_each_and_state_current()
    print("ok")
