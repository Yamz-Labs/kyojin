"""CPU checks of run-time A/B knobs (exllamav3.util.knobs) and POST /admin/knobs (tools/admin_knobs.py)."""
import asyncio
import contextlib
import os
import re
import sys
from pathlib import Path

import pytest

from exllamav3.util import knobs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture(autouse = True)
def clean_env():
    saved = {n: os.environ.get(n) for n in knobs.KNOBS}
    for n in knobs.KNOBS:
        os.environ.pop(n, None)
    knobs.set_knobs({})
    yield
    for n, v in saved.items():
        if v is None:
            os.environ.pop(n, None)
        else:
            os.environ[n] = v
    knobs.invalidate()


def test_defaults_match_call_sites():
    """the default each knob()/knob_int()/knob_on() call passes is the one the table advertises"""
    pat = re.compile(r'knob(?:_int|_on)?\(\s*"(EXL3_[A-Z0-9_]+)"(?:\s*,\s*([^)]*?))?\s*\)')
    seen = {}
    for p in (ROOT / "exllamav3").rglob("*.py"):
        for name, dflt in pat.findall(p.read_text()):
            if name in knobs.KNOBS and dflt:
                seen.setdefault(name, set()).add(dflt.strip().strip('"'))
    for name, dflts in seen.items():
        assert dflts == {knobs.KNOBS[name]}, (name, dflts, knobs.KNOBS[name])
    assert {"EXL3_PF_CK", "EXL3_QDEC_ROWS", "EXL3_PF_WAVES_PER_EU"} <= set(seen)


def test_every_knob_is_read_through_the_helper():
    """no module reads a switchable knob with a bare os.environ (it would not follow set_knobs)"""
    bare = []
    for p in (ROOT / "exllamav3").rglob("*.py"):
        if p.name == "knobs.py":
            continue
        for n in knobs.KNOBS:
            if re.search(r'os\.environ(?:\.get)?[\[(]\s*"%s"' % n, p.read_text()):
                bare.append((p.name, n))
    assert not bare, bare


def test_module_state_follows_and_restores():
    from exllamav3.modules.attention_fn import bc_attn, qdec_cuda, triton_paged as tp
    base = (bc_attn._kvwin_bc, qdec_cuda.ENABLED, qdec_cuda.ROWS_MODE, qdec_cuda.GRAPH_ENABLED, tp._pf_waves_per_eu, tp._pf_ck_mode)
    eff = knobs.set_knobs({"EXL3_BC_KVWIN": "0", "EXL3_QDEC": "0", "EXL3_QDEC_ROWS": "2", "EXL3_QDEC_GRAPH": "0",
                           "EXL3_PF_WAVES_PER_EU": 4, "EXL3_PF_CK": "0"})
    assert (eff["EXL3_BC_KVWIN"], eff["EXL3_PF_WAVES_PER_EU"]) == ("0", "4")
    assert (bc_attn._kvwin_bc, qdec_cuda.ENABLED, qdec_cuda.ROWS_MODE, qdec_cuda.GRAPH_ENABLED, tp._pf_waves_per_eu, tp._pf_ck_mode) \
        == (False, False, "2", False, 4, "0")
    assert tp._pf_ck is False
    knobs.set_knobs({n: None for n in eff})
    assert (bc_attn._kvwin_bc, qdec_cuda.ENABLED, qdec_cuda.ROWS_MODE, qdec_cuda.GRAPH_ENABLED, tp._pf_waves_per_eu, tp._pf_ck_mode) == base
    assert knobs.effective() == knobs.KNOBS


def test_call_time_reads():
    from exllamav3.modules.attention_fn import qdec_cuda
    assert knobs.knob("EXL3_GEMV_TRI", "1") == "1"
    knobs.set_knobs({"EXL3_GEMV_TRI": 0, "EXL3_QDEC_FASTDQ": 0})
    assert knobs.knob("EXL3_GEMV_TRI", "1") == "0"
    assert qdec_cuda._compile_knobs()[1] == "0"            # part of the kernel cache key: the flipped cubin is its own entry


def test_unknown_knob_applies_nothing():
    with pytest.raises(ValueError, match="EXL3_NOPE"):
        knobs.set_knobs({"EXL3_GEMV_TRI": "0", "EXL3_NOPE": "1"})
    assert "EXL3_GEMV_TRI" not in os.environ


def test_invalidate_only_on_a_real_change(monkeypatch):
    calls = []
    monkeypatch.setattr(knobs, "invalidate", lambda: calls.append(1))
    knobs.set_knobs({"EXL3_GEMV_TRI": "1"})               # equals the default: nothing changes
    knobs.set_knobs({"EXL3_GEMV_TRI": "0"})
    knobs.set_knobs({"EXL3_GEMV_TRI": "0"})
    knobs.set_knobs({"EXL3_GEMV_TRI": None})
    assert len(calls) == 2


def test_invalidate_runs_graph_purges_and_hooks(monkeypatch):
    import exllamav3.model.cuda_graph as cg
    import exllamav3.modules.block_graph as bg
    hit = []
    monkeypatch.setattr(cg, "purge", lambda model = None: hit.append("cuda"))
    monkeypatch.setattr(bg, "purge", lambda: hit.append("block"))
    knobs.on_change(lambda: hit.append("hook"))
    try:
        knobs.set_knobs({"EXL3_DEC_MOE_KBS_A": 9})
        assert hit == ["cuda", "block", "hook"]
    finally:
        knobs._hooks.pop()


def _serve(handler_engine, exclusive, calls):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    import admin_knobs
    app = web.Application()
    h = admin_knobs.make_handler(handler_engine, exclusive)
    app.router.add_get("/admin/knobs", h)
    app.router.add_post("/admin/knobs", h)
    return TestClient(TestServer(app))


def test_admin_endpoint():
    order = []

    class Engine:
        async def run_exclusive(self, fn, *a):
            order.append("run")
            return fn(*a)

        async def reset_prefix_cache(self):
            order.append("reset")

    @contextlib.asynccontextmanager
    async def exclusive():
        order.append("drain")
        yield
        order.append("release")

    async def go():
        async with _serve(Engine(), exclusive, order) as c:
            r = await c.get("/admin/knobs")
            assert (await r.json())["effective"] == knobs.KNOBS
            r = await c.post("/admin/knobs", json = {"EXL3_GEMV_TRI": 0, "EXL3_BC_KVWIN": False})
            eff = (await r.json())["effective"]
            assert r.status == 200 and eff["EXL3_GEMV_TRI"] == "0" and eff["EXL3_BC_KVWIN"] == "0"
            assert order == ["drain", "run", "reset", "release"] and (await r.json())["prefix_cache_reset"] is True
            order.clear()
            r = await c.post("/admin/knobs", json = {"EXL3_GEMV_TRI": 0})           # no change: no reset
            assert order == ["drain", "run", "release"] and (await r.json())["prefix_cache_reset"] is False
            r = await c.post("/admin/knobs", json = {"EXL3_GEMV_TRI": None})
            assert (await r.json())["effective"]["EXL3_GEMV_TRI"] == "1"
            r = await c.post("/admin/knobs", json = {"EXL3_NOPE": 1})
            assert r.status == 400 and "EXL3_NOPE" in (await r.json())["error"]["message"]
            r = await c.post("/admin/knobs", data = b"[1]")
            assert r.status == 400
            r = await c.post("/admin/knobs", json = {"EXL3_GEMV_TRI": 0}, headers = {"X-Forwarded-For": "10.0.0.2"})
            assert r.status == 403 and os.environ.get("EXL3_GEMV_TRI") is None
    asyncio.run(go())


def test_tracked_dicts_are_emptied_on_a_change():
    class Holder:
        def __init__(self):
            self.cache = {"built": 1}
    h = Holder()
    knobs.clear_on_change(h, "cache")
    knobs.clear_on_change(h, "cache")                      # registered once
    assert sum(1 for r, a in knobs._tracked if r() is h) == 1
    knobs.set_knobs({"EXL3_GEMV_TRI": "1"})                # same as the default: kept
    assert h.cache
    knobs.set_knobs({"EXL3_GEMV_TRI": "0"})
    assert h.cache == {}
    del h
    import gc
    gc.collect()
    knobs.set_knobs({"EXL3_GEMV_TRI": None})               # a dead object is skipped, not an error
