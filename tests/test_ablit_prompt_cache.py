"""The ablit tools fetch the public prompt sets at first use into a user cache dir (tools/ablit/prompt_cache.py + build_prompts.build).
CPU only, synthetic two-row sets from a monkeypatched fetcher: no network, no real prompt text."""
import ast, json, os, sys
import pytest

ABLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "ablit")
sys.path.insert(0, ABLIT)
import build_prompts as BP
import prompt_cache as PC


def _fake(monkeypatch, calls):
    def mk(src):
        def get(n, *a):
            calls.append(src)
            return [{"text": f"synthetic {src} row {i}", "source": src} for i in range(2)]
        return get
    for fn, src in (("get_advbench", "advbench"), ("get_harmbench", "harmbench"), ("get_jbb", "jbb"), ("get_xstest", "xstest"), ("get_alpaca", "alpaca")):
        monkeypatch.setattr(BP, fn, mk(src))
    real_build = BP.build
    monkeypatch.setattr(BP, "build", lambda out, seed=0: real_build(out, seed, n=4, hold=1))


def test_ensure_fetches_once_into_the_cache(monkeypatch, tmp_path):
    calls = []
    _fake(monkeypatch, calls)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = PC.ensure()
    assert d == str(tmp_path / "kyojin" / "ablit_prompts") == PC.cache_dir()
    for n in PC.NAMES:
        rows = [json.loads(l) for l in open(os.path.join(d, n))]
        assert rows and all(set(r) >= {"text", "source", "split"} for r in rows)
    assert os.path.isfile(os.path.join(d, "SOURCES.md"))
    n_calls = len(calls)
    assert n_calls == 5
    assert PC.ensure() == d and len(calls) == n_calls                 # second use: no fetch
    assert not [x for x in os.listdir(tmp_path / "kyojin") if ".part" in x]


def test_legacy_dir_wins_when_complete(monkeypatch, tmp_path):
    calls = []
    _fake(monkeypatch, calls)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    leg = tmp_path / "leg"; leg.mkdir()
    for n in PC.NAMES:
        (leg / n).write_text(json.dumps({"text": "synthetic", "source": "x", "split": "s"}) + "\n")
    assert PC.ensure(str(leg)) == str(leg) and not calls
    assert PC.default_file("harmful_heldout.jsonl", str(leg)) == str(leg / "harmful_heldout.jsonl")


def test_cached_files_load_through_the_tools(monkeypatch, tmp_path):
    calls = []
    _fake(monkeypatch, calls)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    p = PC.default_file("harmful_heldout.jsonl")
    try:
        import eval_ablit as E
    except Exception as e:
        pytest.skip(f"eval_ablit not importable here: {e}")
    rows = E.load_prompts(p, 10)
    assert rows and all("synthetic" in (r if isinstance(r, str) else r.get("text", "")) for r in rows)


def test_readers_compile_and_use_the_cache():
    for f in ("search_ds41.py", "search_ablit4.py", "search_ablit5.py", "eval_runtime.py", "collect_dirs.py", "collect_ds41.py", "build_prompts.py", "prompt_cache.py"):
        ast.parse(open(os.path.join(ABLIT, f)).read(), f)
    for f in ("search_ds41.py", "search_ablit4.py", "search_ablit5.py", "eval_runtime.py", "collect_dirs.py", "collect_ds41.py"):
        assert "prompt_cache" in open(os.path.join(ABLIT, f)).read(), f
    assert not [f for f in os.listdir(os.path.join(ABLIT, "prompts")) if f.endswith(".jsonl")]     # the sets are never stored in the tree
