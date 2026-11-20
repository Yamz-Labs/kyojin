"""CPU tests of HY4_MU_HOSTFAST (batched host path of a multi-session decode round), stub engine / stub decoder, no GPU:
  1. plain_round / spec_round produce the SAME tokens, session state (y, ng history, hcur values) and verify rows with the flag on and off (greedy stub trunk, drafts that depend on hcur,
     real NGram, TM 8 and TM 16, plain -> spec -> plain transitions, 1 / 2 / 4 / 8 sessions);
  2. the numpy EngramStager hash + staging of several sessions == the per-row python staging of the baseline `_stage_rows` (bytes read from a fake table file);
  3. step_slots with the flag stages each Engram layer group just before its own graph segment (overlap), without it everything before the first replay; same rows either way;
  4. the FIRES counters fire."""
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "hy4"))
for _n, _s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    _m = sys.modules.get(_n) or types.ModuleType(_n)
    _m.__path__ = [os.path.join(ROOT, "exllamav3", _s)]
    sys.modules[_n] = _m

import mu_engine as ME  # noqa: E402
from ngram import NGram  # noqa: E402

K = 3
D = 3


def f_next(t, pos):
    """the stub trunk: next token after token t at position pos (short cycles, so the n-gram index finds matches)"""
    return (t * 5 + 1 + (pos // 11)) % 13


class Sess:
    def __init__(self, slot, prompt):
        self.slot, self.ids, self.ng, self.hcur, self.extra = slot, list(prompt), None, None, {}
        self.tokens, self.n = [], 0
        self.y = None


class FakeGR:
    """step_slots: argmax of a row = f_next(token, position); trunk state of a row = [token, position, argmax]"""

    def __init__(self, tm):
        self.tm, self.binds, self.steps, self.rows_log = tm, 0, 0, []

    def step_slots(self, rows):
        am, hv = [], []
        T = sum(len(t) for _, t in rows)
        assert 1 <= T <= self.tm
        for s, toks in rows:
            for i, t in enumerate(toks):
                p = s.n + i
                a = f_next(t, p)
                am.append(a)
                hv.append([float(t), float(p), float(a)])
            s.tokens.extend(toks)
            s.n += len(toks)
        self.rows_log.append([(s.slot, list(t)) for s, t in rows])
        self.steps += 1
        return None, torch.tensor(am), None, torch.tensor(hv)

    def rewind(self, s, n):
        del s.tokens[n:]
        s.n = n

    def bind(self, s):
        self.binds += 1


class FakeMG:
    """draft chain: the true greedy chain, corrupted by the trunk state it is given (a wrong hcur changes the drafts, so the streams would diverge)"""

    def __init__(self):
        self.slot = 0
        self.graphs = set()
        self.runs = 0

    def captured(self, slot, j):
        return (slot, j // 512) in self.graphs

    def run(self, tok, h, j):
        self.graphs.add((self.slot, j // 512))
        self.runs += 1
        t, out = tok, []
        pos = int(h[0, 1].item()) + 1
        for i in range(K):
            t = f_next(t, pos + i)
            out.append(t + 1 if (int(h[0, 0].item()) + int(h[0, 2].item()) + i) % 4 == 0 else t)
        return torch.tensor(out)


def make(tm, ng_ext=4, slots=8):
    eng = object.__new__(ME.MuEngine)
    eng.torch, eng.dev, eng.k, eng.slots = torch, "cpu", K, slots
    eng.GR, eng.MG, eng.MT = FakeGR(tm), FakeMG(), SimpleNamespace(step=lambda *a, **k: None, ar=torch.arange(64))
    eng.SG = SimpleNamespace(TM=tm)
    eng.ngram_ext, eng.ngram_min, eng.ngram_gate, eng.ngram_adapt, eng.ngram_fill, eng.ngram_long = ng_ext, 2, True, False, False, 0
    return eng


def start(eng, n):
    out = []
    for slot in range(n):
        s = Sess(slot, [(slot * 3 + i * 2) % 13 for i in range(20 + slot)])
        s.tokens = list(s.ids)
        s.n = len(s.ids)
        s.y = (slot * 7 + 1) % 13
        s.hcur = torch.tensor([[float(s.ids[-1]), float(s.n - 1), 0.0]])
        out.append(s)
    return out


def drive(fast, tm, n, plan, monkeypatch):
    """plan = sequence of 'p' / 's' rounds; -> everything observable"""
    monkeypatch.setenv("HY4_MU_HOSTFAST", "1" if fast else "0")
    eng = make(tm)
    sess = start(eng, n)
    emitted = {s.slot: [] for s in sess}
    for kind in plan:
        out = eng.plain_round(sess) if kind == "p" else eng.spec_round(sess)
        for s in sess:
            emitted[s.slot] += out[s]
    for s in sess:
        eng._ng_flush(s)
    return {"emitted": emitted, "y": [s.y for s in sess], "n": [s.n for s in sess], "tokens": [list(s.tokens) for s in sess], "hcur": [s.hcur.clone().tolist() for s in sess],
            "ng": [list(s.ng.seq) if s.ng is not None else None for s in sess], "rows": eng.GR.rows_log, "gate": [dict(s.extra) for s in sess]}, eng


@pytest.mark.parametrize("tm", [8, 16])
@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_rounds_identical_with_and_without_hostfast(tm, n, monkeypatch):
    plan = "pppsssspsspppsssss"
    if n * 2 > tm:                                   # spec needs >= 2 rows per session (the scheduler runs plain above MAXB)
        plan = "pppp" + "p" * 14
    a, ea = drive(False, tm, n, plan, monkeypatch)
    b, eb = drive(True, tm, n, plan, monkeypatch)
    assert a == b
    assert ea.GR.binds >= eb.GR.binds
    if "s" in plan:
        assert eb.GR.binds < ea.GR.binds             # the captured (slot, bucket) draft graphs run without bind
        assert ME.FIRES["bind_skip"] > 0 and ME.FIRES["draft_cat"] > 0 and ME.FIRES["spec"] > 0
    assert ME.FIRES["plain"] > 0 and ME.FIRES["hcur_idx"] > 0


def test_hcur_rows_are_isolated_and_values_are_the_continuation_row(monkeypatch):
    monkeypatch.setenv("HY4_MU_HOSTFAST", "1")
    eng = make(8)
    sess = start(eng, 3)
    eng.spec_round(sess)
    for s in sess:
        want = [float(s.tokens[-1]), float(s.n - 1), float(f_next(s.tokens[-1], s.n - 1))]
        assert s.hcur.shape == (1, D) and s.hcur[0].tolist() == want
    sess[0].hcur.zero_()                              # a view of ITS slot row only
    assert all(float(s.hcur.abs().sum()) > 0 for s in sess[1:])


def test_plain_round_defers_the_history_append_but_never_loses_it(monkeypatch):
    monkeypatch.setenv("HY4_MU_HOSTFAST", "1")
    eng = make(8)
    s = start(eng, 1)[0]
    s.ng = NGram(2, 4)
    s.ng.reset(list(s.ids) + [s.y])
    s.extra.update(rate=0.7, since=0)
    n0 = len(s.ng.seq)
    eng.plain_round([s])
    assert len(s.ng.seq) == n0                       # appended while the NEXT round runs
    eng.plain_round([s])
    assert len(s.ng.seq) == n0 + 1
    eng.spec_round([s])                              # a spec round flushes before it proposes
    assert s.ng_pend is not None


# ------------------------------------------------------------------------------------------------------------------ Engram staging
@pytest.fixture
def decoder(tmp_path):
    if not os.path.exists("/proc/self/fd"):
        pytest.skip("Linux only (page-cache reads through /proc/self/fd)")
    from exllamav3.loader import step_mu
    from exllamav3.loader.engram_stage import EngramStager
    ng, nh, nE, TMX, rb = 3, 2, 3, 8, 16
    nc = (ng - 1) * nh
    n_tm = 40
    rng = np.random.RandomState(7)
    primes = [101, 103, 107, 109]
    offs = np.cumsum([0] + primes[:-1]).tolist()
    nrows = sum(primes) + 4
    path = tmp_path / "table.bin"
    path.write_bytes(rng.randint(1, 255, size=nrows * rb, dtype=np.uint8).tobytes())
    fd = os.open(str(path), os.O_RDONLY)
    g = SimpleNamespace(fds=[fd], bases=[0])
    table = SimpleNamespace(off=0, row_bytes=rb, rows=nrows, name="t", g=g)
    token_map = rng.randint(0, 29, size=n_tm).tolist()
    mods = []
    for e in range(nE):
        mods.append(SimpleNamespace(e=e, token_map=token_map, pad_id=1, n_gram=ng, n_head=nh, n_cols=nc, mult=[int(x) for x in rng.randint(1, 2 ** 31, size=ng)],
                                    primes=primes, offs=offs))
    d = object.__new__(step_mu.MultiDecoder)
    d.fw = SimpleNamespace(ple=[2, 5, 9][:nE], eng={2: mods[0], 5: mods[1], 9: mods[2]}, table=table, g=g, n_layers=12)
    d.rpin = torch.zeros(nE, TMX, nc, rb, dtype=torch.uint8)
    d.vpin = torch.zeros(nE, TMX, nc, dtype=torch.float32)
    d.rnp = d.rpin.numpy()
    d._rmv = None
    d.gs = SimpleNamespace(eng_raw=torch.zeros(nE, TMX, nc, rb, dtype=torch.uint8), eng_valid=torch.zeros(nE, TMX, nc))
    d.stager = EngramStager(mods, table, d.rnp, d.vpin.numpy(), pool=None)
    d.stager.nowait = False
    yield d
    os.close(fd)


def sessions_for_rows(spec, n_tm=40):
    rng = np.random.RandomState(3)
    rows = []
    for slot, (n, T) in enumerate(spec):
        s = Sess(slot, [])
        hist = [int(x) for x in rng.randint(0, n_tm + 10, size=n + T)]      # ids past the token map: dead rows
        s.tokens, s.n = hist, n
        rows.append((s, hist[n:]))
    return rows


@pytest.mark.parametrize("spec", [[(0, 1)], [(1, 2), (20, 1)], [(5, 3), (0, 2), (31, 2), (2, 1)]])
def test_vectorised_staging_equals_the_python_baseline(decoder, spec):
    d = decoder
    rows = sessions_for_rows(spec)
    d._stage_rows(rows)                                     # baseline: per row, per layer python hash
    want_r, want_v = d.rpin.clone(), d.vpin.clone()
    want_g = (d.gs.eng_raw.clone(), d.gs.eng_valid.clone())
    d.rpin.zero_()
    d.vpin.zero_()
    d.gs.eng_raw.zero_()
    d.gs.eng_valid.zero_()
    hs = d._hash_rows(rows)
    nE = d.rpin.shape[0]
    d._stage_group_mu(hs, 0, 1)                             # group by group, as the overlapped replay does
    d._stage_group_mu(hs, 1, nE)
    T = sum(len(t) for _, t in rows)
    assert torch.equal(d.rpin[:, :T], want_r[:, :T]) and torch.equal(d.vpin[:, :T], want_v[:, :T])
    assert torch.equal(d.gs.eng_raw[:, :T], want_g[0][:, :T]) and torch.equal(d.gs.eng_valid[:, :T], want_g[1][:, :T])


# ------------------------------------------------------------------------------------------------------------------ step_slots overlap
class FakeSeg:
    def __init__(self, d, si, log, expect):
        self.d, self.si, self.log, self.expect = d, si, log, expect

    def replay(self):
        raw = self.d.gs.eng_raw
        staged = [bool(raw[e].any()) for e in range(raw.shape[0])]
        self.log.append((self.si, staged))


def make_step(d, rows_spec, monkeypatch, fast):
    from exllamav3.loader import step_mu
    monkeypatch.setenv("HY4_MU_HOSTFAST", "1" if fast else "0")
    d.hin = torch.zeros(3, 8, dtype=torch.long)
    d.hnp = d.hin.numpy()
    d.gs.inp = torch.zeros(3, 8, dtype=torch.long)
    d.gs.Nb = d.gs.T = 0
    d.cap, d.prepared, d.ev_set, d.ev = 4096, True, False, SimpleNamespace(synchronize=lambda: None, record=lambda: None)
    d.stats = {"replays": 0}
    d.split = True
    log = []
    out = (None, torch.zeros(1), torch.zeros(1), torch.zeros(1))
    d.graphs = {}
    segs = [FakeSeg(d, i, log, None) for i in range(3)]
    d._graph = lambda Nb, T: (segs, out)
    return step_mu, log


@pytest.mark.parametrize("fast", [False, True])
def test_step_slots_stages_before_the_segment_that_reads_them(decoder, monkeypatch, fast):
    d = decoder
    step_mu, log = make_step(d, None, monkeypatch, fast)
    rows = sessions_for_rows([(5, 2), (9, 1)])
    key = (512, 3, len(d.bounds()))
    if fast:
        d.graphs[key] = True                                   # the graph exists: overlapped staging
    f0 = dict(step_mu.FIRES)
    d.step_slots(rows)
    assert [si for si, _ in log] == [0, 1, 2]
    if fast:
        assert log[0][1] == [False, False, False] and log[1][1] == [True, False, False] and log[2][1] == [True, True, True]
        assert step_mu.FIRES["stage_ov"] == f0["stage_ov"] + 1 and step_mu.FIRES["stage_vec"] == f0["stage_vec"] + 1
    else:
        assert log[0][1] == [True, True, True]                 # baseline: everything staged before the first replay
        assert step_mu.FIRES["stage_ov"] == f0["stage_ov"]
    assert [s.n for s, _ in rows] == [7, 10] and d.stats["replays"] == 1


def test_step_slots_rows_equal_in_both_modes(decoder, monkeypatch):
    d = decoder
    got = {}
    for fast in (False, True):
        step_mu, _ = make_step(d, None, monkeypatch, fast)
        d.rpin.zero_()
        d.vpin.zero_()
        d.gs.eng_raw.zero_()
        d.gs.eng_valid.zero_()
        rows = sessions_for_rows([(5, 2), (9, 1), (0, 2)])
        d.graphs = {(512, 5, len(d.bounds())): True} if fast else {}
        d.step_slots(rows)
        got[fast] = (d.gs.eng_raw[:, :5].clone(), d.gs.eng_valid[:, :5].clone(), d.hin[:, :5].clone())
    assert all(torch.equal(a, b) for a, b in zip(got[False], got[True]))


def test_fires_line_format_is_what_the_driver_parses():
    import re
    from exllamav3.loader import step_mu
    eng = SimpleNamespace(SM=step_mu)
    snap = {}
    ME.HT["rounds"] += 100
    ME.HT["rows"] += 400
    line = ME.fires_line(eng, snap)
    assert line.startswith("MUFAST rounds=")
    got = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)(?=[ ])", line.split(" | ")[0] + " ")}
    for k in ("fast", "plain", "spec", "hcur_idx", "bind_skip", "draft_cat", "stage_vec", "stage_ov", "inp_np", "rows_gt8", "steps"):
        assert k in got, k
    assert "rows/round=4.00" in line
    assert ME.fires_line(SimpleNamespace(), None).startswith("MUFAST")
