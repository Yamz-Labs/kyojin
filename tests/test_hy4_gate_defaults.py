"""zero-GPU check: 1.9.3 ships the depth gate (learn mode) and the row-batched MLA decode attention on; the gate steps aside where it cannot run
(slots > 1, num-draft != 3, a draft policy); an explicit env value wins; the engine boots with the shipped gate."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))
KEYS = ("HY4_SPEC2_GATE", "HY4_SPEC2_MODE", "HY4_MLA_ROWS2", "HY4_DRAFT_POLICY", "HY4_SPEC2_KB")


def _apply(env, **kw):
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    import serve
    serve.apply_env(16384, **kw)
    return {k: os.environ.get(k) for k in KEYS}


def test_defaults_on():
    e = _apply({})
    assert (e["HY4_SPEC2_GATE"], e["HY4_SPEC2_MODE"], e["HY4_MLA_ROWS2"]) == ("1", "learn", "1")


def test_gate_and_mla2_off_with_slots():
    e = _apply({}, slots=2)
    assert (e["HY4_SPEC2_GATE"], e["HY4_MLA_ROWS2"]) == ("0", "0")


def test_gate_off_when_it_cannot_run():
    assert _apply({}, num_draft=4)["HY4_SPEC2_GATE"] == "0"
    assert _apply({"HY4_DRAFT_POLICY": "cost"})["HY4_SPEC2_GATE"] == "0"
    assert _apply({}, num_draft=4)["HY4_MLA_ROWS2"] == "1"


def test_explicit_env_wins():
    e = _apply({"HY4_SPEC2_GATE": "0", "HY4_MLA_ROWS2": "0"})
    assert (e["HY4_SPEC2_GATE"], e["HY4_MLA_ROWS2"]) == ("0", "0")


def test_engine_builds_with_the_shipped_gate():
    _apply({})
    from engine import Hy4Engine
    e = Hy4Engine(TREE, "/nonexistent.gguf", ctx=16384, k=3)
    assert e.gate is not None and e.gate_kb == 2
