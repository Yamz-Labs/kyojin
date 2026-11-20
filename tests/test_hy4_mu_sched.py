"""MuScheduler logic with a fake engine (no torch, no GPU): admission into free slots, queueing past the slot count, slot reuse, one prefill chunk between decode rounds,
cancel, max_new, context error, error propagation."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "hy4"))
import mu_engine as ME  # noqa: E402


class Sess:
    def __init__(self, slot):
        self.slot, self.pf_pos, self.y, self.n = slot, 0, 100 + slot, 0


class FakeEng:
    k, ctx = 3, 4096

    def __init__(self, slots=2, boom=False):
        self.slots, self.log, self.boom = slots, [], boom

    def phase(self, _):
        pass

    def open(self, slot):
        self.log.append(("open", slot))
        return Sess(slot)

    def prefill_begin(self, s, ids, chunk):
        s.ids, s.chunk = ids, chunk

    def prefill_step(self, s):
        s.pf_pos = min(len(s.ids), s.pf_pos + s.chunk)
        self.log.append(("pf", s.slot, s.pf_pos))
        time.sleep(0.002)
        return s.pf_pos >= len(s.ids)

    def plain_round(self, sess):
        if self.boom:
            raise RuntimeError("boom")
        self.log.append(("dec", tuple(s.slot for s in sess)))
        time.sleep(0.002)
        out = {}
        for s in sess:
            s.y += 1
            out[s] = [s.y]
        return out

    def spec_round(self, sess):
        out = self.plain_round(sess)
        self.log[-1] = ("spec", self.log[-1][1])
        return out


def run(reqs, eng, chunk=4, spec=False):
    sch = ME.MuScheduler(eng, chunk=chunk, spec=spec)
    sch.start()
    for r in reqs:
        sch.submit(r)
    for r in reqs:
        assert r.done.wait(10), "request did not finish"
    sch.shutdown()
    sch.join(5)
    return sch


def test_queue_past_slots_reuse_and_max_new():
    eng = FakeEng(slots=2)
    reqs = [ME.Request(list(range(10)), 5) for _ in range(5)]
    run(reqs, eng)
    assert all(len(r.out) == 5 and not r.error for r in reqs)
    opens = [x[1] for x in eng.log if x[0] == "open"]
    assert len(opens) == 5 and set(opens) == {0, 1}
    assert max(len(x[1]) for x in eng.log if x[0] == "dec") == 2          # never more rows than slots


def test_prefill_chunk_is_interleaved_with_decode_rounds():
    eng = FakeEng(slots=2)
    a = ME.Request(list(range(4)), 40)
    b = ME.Request(list(range(40)), 3)                                       # 10 chunks of 4
    sch = ME.MuScheduler(eng, chunk=4)
    sch.start()
    sch.submit(a)
    while not a.out:
        time.sleep(0.001)
    sch.submit(b)
    b.done.wait(10)
    a.done.wait(10)
    sch.shutdown()
    seq = [("dec" if x[0] == "spec" else x[0]) for x in eng.log if x[0] in ("pf", "dec", "spec")]
    i0 = seq.index("pf", 1)                                                  # b's chunks (a's single chunk comes first)
    run_pf = 0
    longest = 0
    for x in seq[i0:i0 + 25]:
        run_pf = run_pf + 1 if x == "pf" else 0
        longest = max(longest, run_pf)
    assert longest <= 1, "two prefill chunks ran back to back while a session was decoding"


def test_cancel_context_and_error():
    eng = FakeEng(slots=2)
    r = ME.Request(list(range(4)), 3000)
    r.on_tokens = lambda new: (len(r.out) > 20) and r.cancel.set()
    big = ME.Request(list(range(5000)), 4)
    done = []
    r.on_done = lambda: done.append(1)
    run([r, big], eng)
    assert 20 < len(r.out) < 40 and done == [1] and big.error == "context"
    bad = ME.Request(list(range(4)), 5)
    run([bad], FakeEng(boom=True))
    assert bad.error and "boom" in bad.error


def test_speculation_only_while_few_sessions_decode():
    eng = FakeEng(slots=4)
    reqs = [ME.Request(list(range(4)), 30) for _ in range(4)]
    run(reqs, eng, spec=True)
    kinds = {(x[0], len(x[1])) for x in eng.log if x[0] in ("spec", "dec")}
    assert all(not (k == "spec" and n > 2) for k, n in kinds) and ("spec", 1) in kinds or ("spec", 2) in kinds
    assert any(k == "dec" and n >= 3 for k, n in kinds)
