"""zero-GPU check: the load warm-up prompt is clamped to the context (a requested HY4_WARMUP_TOKENS above ctx no longer raises ValueError("context"))"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))

import engine as E


def _fake(ctx, k=3):
    """an engine without model or GPU: generate() applies the real context bound and records the prompt length"""
    e = object.__new__(E.Hy4Engine)
    e.ctx, e.k, e.seen = ctx, k, []

    def generate(ids, max_new, on_tokens, **kw):
        if len(ids) + max_new + e.k + 2 > e.ctx:
            raise ValueError("context")
        e.seen.append(len(ids))
        return {}

    e.generate = generate
    e.reset = lambda: None
    e.warmup_tails = lambda: None
    return e


def _run(monkeypatch, ctx, n, k=3):
    for v in ("HY4_PRECAPTURE", "HY4_LC_PRIME", "HY4_WARMUP_TAILS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("HY4_WARMUP_TAILS", "0")
    if hasattr(E.Hy4Engine, "_warmup"):           # builds that stage the Engram rows (HY4_WARMUP_NO_ENGRAM) import the GPU loader: stub it
        eg = types.SimpleNamespace(READ={"bytes": 0, "dummy_bytes": 0}, DUMMY=[False])
        for name in ("exllamav3", "exllamav3.loader"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "exllamav3.loader.engram_gpu", eg)
        sys.modules["exllamav3.loader"].engram_gpu = eg
        monkeypatch.setattr(E, "_proc_io", lambda: (0, 0), raising=False)
    e = _fake(ctx, k)
    e.warmup(n)
    return e.seen


def test_clamp_value():
    assert E.warmup_clamp(8493, 8192, 3) == 8192 - (8 + 3 + 2 + E.WARMUP_PAD)
    assert E.warmup_clamp(8493, 70000, 3) == 8493
    assert E.warmup_clamp(8163, 8192, 3) == 8163                 # exactly at the bound: unchanged
    assert E.warmup_clamp(8493, 200, 3) == 0                     # ctx too small: skip


def test_small_ctx_warmup_runs(monkeypatch):
    seen = _run(monkeypatch, 8192, 8493)
    assert seen == [E.warmup_clamp(8493, 8192, 3)] and seen[0] + 8 + 3 + 2 <= 8192


def test_large_ctx_unchanged(monkeypatch):
    assert _run(monkeypatch, 70000, 8493) == [8493]


def test_tiny_ctx_skips(monkeypatch, capsys):
    assert _run(monkeypatch, 200, 8493) == []
    assert "warm-up skipped: ctx too small" in capsys.readouterr().out


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
