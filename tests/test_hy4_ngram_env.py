"""zero-GPU check: the shipped environment (serve.SHIPPED_ENV through apply_env) turns the n-gram rows on in the engine, with the gate; an explicit env value wins."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))


def _engine(env):
    for k in [k for k in os.environ if k.startswith("HY4_NGRAM")]:
        del os.environ[k]
    os.environ.update(env)
    import serve
    serve.apply_env(16384)
    from engine import Hy4Engine
    return Hy4Engine(TREE, "/nonexistent.gguf", ctx=16384, k=3)


def test_shipped_flags_enable_ngram():
    e = _engine({})
    assert (e.ngram_ext, e.ngram_min, e.ngram_gate) == (4, 8, True)
    assert e.ngram_adapt is False and e.ngram_fill is False


def test_explicit_env_wins_and_zero_disables():
    e = _engine({"HY4_NGRAM_EXT": "0"})
    assert e.ngram_ext == 0
    e = _engine({"HY4_NGRAM_EXT": "6", "HY4_NGRAM_MIN": "5", "HY4_NGRAM_GATE": "0"})
    assert (e.ngram_ext, e.ngram_min, e.ngram_gate) == (6, 5, False)
