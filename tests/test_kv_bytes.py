"""zero-GPU checks of the KV footprint levers (1.9.2): analytic bytes per token, HY4_KV_COMPACT defaults, the fp8 draft-layer cache"""
import ast
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))
sys.path.insert(0, TREE)


def _fn(path, name, cls=None):
    tree = ast.parse(open(os.path.join(TREE, path)).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and cls and node.name != cls:
            continue
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(open(os.path.join(TREE, path)).read(), node)
    raise AssertionError(name)


def _kv8():
    import importlib.util
    sp = importlib.util.spec_from_file_location("kv_fp8_t", os.path.join(TREE, "exllamav3", "loader", "kv_fp8.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def test_bytes_per_token_table():
    import kv_bytes as KB
    t, c = KB.total(KB.per_token())
    assert t == 115344 and c == 0                            # 1.9.1 default: fp32 latent everywhere, window layers at full context (measured slope of the release table: 98-118 KB)
    t8, _ = KB.total(KB.per_token(fp8=True))
    assert t8 == 115344 - 12 * (2304 - 644)
    tc, cc = KB.total(KB.per_token(fp8=True, fp8_mtp=True, swa_ring=True))
    assert tc == 7728 + 644 + 1664 + 16 + 1280 and 11000 < tc < 11500
    assert cc == 30 * 1024 * 2304 + 13 * 4096 * 1024                # window rings + indexer value|gate rings: constant
    tp, _ = KB.total(KB.per_token(fp8=True, fp8_mtp=True, swa_ring=True, pool_bytes=2))
    assert tc - tp == 13 * 32 * 2 and tp < 10600


def test_fp8_row_bytes_is_644():
    torch = pytest.importorskip("torch")
    kv_fp8 = _kv8()
    kv = kv_fp8.Fp8KV(10, 512, 64, "cpu")
    assert kv.nbytes() == 10 * 644


def test_compact_sets_defaults_and_keeps_user_values(monkeypatch):
    src = _fn("tools/hy4/serve.py", "apply_env")
    ns = {"os": os, "SHIPPED_ENV": {}, "TREE": TREE, "glob": None}
    exec(compile(src, "apply_env", "exec"), ns)
    for k in ("HY4_KV_COMPACT", "HY4_KV_FP8", "HY4_KV_FP8_MTP", "HY4_SWA_RING"):
        monkeypatch.delenv(k, raising=False)
    ns["apply_env"](65536)
    assert not any(k in os.environ for k in ("HY4_KV_FP8", "HY4_KV_FP8_MTP", "HY4_SWA_RING"))      # off by default
    monkeypatch.setenv("HY4_KV_COMPACT", "1")
    monkeypatch.setenv("HY4_SWA_RING", "2048")                                                        # a value set by hand wins
    ns["apply_env"](262144)
    assert os.environ["HY4_KV_FP8"] == "1" and os.environ["HY4_KV_FP8_MTP"] == "1" and os.environ["HY4_SWA_RING"] == "2048"


def test_mtp_k_put_fp8_roundtrip():
    torch = pytest.importorskip("torch")
    kv_fp8 = _kv8()
    src = _fn("exllamav3/loader/mtp_head.py", "_kput")
    ns = {"_KV8": kv_fp8}
    exec(compile("class M:\n" + "\n".join("    " + ln for ln in src.splitlines()), "kput", "exec"), ns)
    m = ns["M"]()
    m.kbuf = kv_fp8.Fp8KV(16, 512, 64, "cpu")
    kn = torch.randn(3, 576)
    m._kput(torch.tensor([2, 5, 9]), kn)
    back = m.kbuf.dequant32()[[2, 5, 9]]
    assert (back[:, :512] - kn[:, :512]).abs().max() < 0.07 * kn[:, :512].abs().max()
    assert (back[:, 512:] - kn[:, 512:]).abs().max() < 0.02 * kn[:, 512:].abs().max()
    m.kbuf = torch.zeros(16, 576)                                       # fp32 cache: the old index_copy_
    m._kput(torch.tensor([1, 4, 7]), kn)
    assert torch.equal(m.kbuf[[1, 4, 7]], kn)


def test_mtp_fp8_needs_trunk_fp8_and_is_off_by_default():
    src = open(os.path.join(TREE, "exllamav3/loader/mtp_head.py")).read()
    assert 'os.environ.get("HY4_KV_FP8_MTP", "0") == "1" and os.environ.get("HY4_KV_FP8", "0") == "1"' in src
    assert "HY4_KV_FP8_MTP" in open(os.path.join(TREE, "tools/hy4/mu_engine.py")).read()
