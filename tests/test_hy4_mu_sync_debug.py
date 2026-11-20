"""CPU check of the HY4_MU_SYNC_DEBUG instrumentation (loader/mu_debug.py): off = null context, on = begin line per stage + a FAULT line when a stage raises, probe_wrap
logs / restores wrapped methods (instance attribute removed again, original untouched)."""
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _n, _s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    _m = sys.modules.get(_n) or types.ModuleType(_n)
    _m.__path__ = [os.path.join(ROOT, "exllamav3", _s)]
    sys.modules[_n] = _m

from exllamav3.loader import mu_debug as D  # noqa: E402


def test_off_is_a_null_context(capsys, monkeypatch):
    monkeypatch.delenv("HY4_MU_SYNC_DEBUG", raising=False)
    with D.stage("x"):
        pass
    assert capsys.readouterr().out == ""


def test_on_logs_begin_and_fault(capsys, monkeypatch):
    monkeypatch.setenv("HY4_MU_SYNC_DEBUG", "1")
    with D.stage("a"):
        pass
    with pytest.raises(RuntimeError):
        with D.stage("b"):
            raise RuntimeError("boom")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "mu-sync begin a" and out[1] == "mu-sync begin b" and out[2].startswith("mu-sync FAULT in b")


def test_probe_wrap_logs_layer_index_and_restores(capsys, monkeypatch):
    monkeypatch.setenv("HY4_MU_SYNC_DEBUG", "1")

    class Fw:
        def layer(self, i, x):
            return x + i

    fw = Fw()
    with D.probe_wrap([(fw, "layer", "layer")]):
        assert fw.layer(3, 10) == 13
    assert "layer" not in fw.__dict__ and fw.layer(1, 1) == 2
    assert capsys.readouterr().out.splitlines() == ["mu-sync begin layer L3"]


def test_step_slots_probe_runs_once_per_slot_set_and_restores(capsys, monkeypatch):
    monkeypatch.setenv("HY4_MU_SYNC_DEBUG", "1")
    import torch
    from types import SimpleNamespace
    from exllamav3.loader import step_mu

    log = []

    class Mla:
        def attend(self, *a):
            return a[0]

    class Fw:
        n_layers = 2
        _gs = None
        _h = None

        def __init__(self):
            self.mlas = [Mla(), Mla()]

        def _layer(self, i):
            return SimpleNamespace(mla=self.mlas[i])

        def layer(self, i, x, *a):
            self.mlas[i].attend(i)
            self._ffn(i, None, x)
            log.append(("layer", i, self._gs is not None))
            return x

        def _ffn(self, i, L, x):
            return x

    class Gs(SimpleNamespace):
        def indexer(self, fw, i, *a):
            return None

        def append_kv(self, i, kn):
            return kn

        def engram(self, fw, i, x):
            return x

    out = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    d = object.__new__(step_mu.MultiDecoder)
    d.fw = Fw()
    d.gs = Gs(inp=torch.zeros(3, step_mu.TM, dtype=torch.long), Nb=0, T=0)
    d.hin = torch.zeros(3, step_mu.TM, dtype=torch.long)
    d.cap, d.prepared, d.ev_set = 4096, True, False
    d.stats = {"replays": 0}
    d.ev = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    d._stage_rows = lambda rows: None
    d._graph = lambda Nb, T: ([SimpleNamespace(replay=lambda: None)], out)
    d.bounds = lambda: [(0, 2)]
    d.body = lambda bnds: [d.fw.layer(i, None) for i in range(2)]
    s = lambda slot: SimpleNamespace(slot=slot, n=5, tokens=[1] * 5)
    d.step_slots([(s(6), [1])])
    d.step_slots([(s(6), [1])])          # same key: no second probe
    d.step_slots([(s(7), [1])])          # new slot set: probed again
    lines = capsys.readouterr().out.splitlines()
    assert sum("probe eager body" in l for l in lines) == 2
    assert "mu-sync begin layer L1" in lines and "mu-sync begin L0 mla.attend L0" in lines and "mu-sync begin ffn L1" in lines
    assert [l for l in lines if "replay segment" in l] and len([1 for e in log if e[0] == "layer"]) == 4
    assert not any(k in d.fw.__dict__ for k in ("layer", "_ffn")) and "attend" not in d.fw.mlas[0].__dict__ and "indexer" not in d.gs.__dict__


def test_sync_only_filters_stages_and_skips_the_probe(capsys, monkeypatch):
    monkeypatch.setenv("HY4_MU_SYNC_DEBUG", "1")
    monkeypatch.delenv("HY4_MU_SYNC_ONLY", raising=False)
    assert D.probe_enabled()
    monkeypatch.setenv("HY4_MU_SYNC_ONLY", "replay segment, draft chain")
    assert not D.probe_enabled()
    for name in ("step_slots T=1 replay segment 0/3", "spec_round draft chain slot 2 j 9", "prefill trunk step slot 1", "plain_round hcur clone slot 0 row 0"):
        with D.stage(name):
            pass
    assert capsys.readouterr().out.splitlines() == ["mu-sync begin step_slots T=1 replay segment 0/3", "mu-sync begin spec_round draft chain slot 2 j 9"]
    monkeypatch.setenv("HY4_MU_SYNC_ONLY", "probe")
    assert D.probe_enabled()
    monkeypatch.delenv("HY4_MU_SYNC_DEBUG")
    assert not D.probe_enabled()
