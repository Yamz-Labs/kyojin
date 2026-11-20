"""zero-GPU checks of the 1.9.2 KV10 levers: HY4_IX_POOL16 (fp16 indexer pool), HY4_ROPE_POS (rope rows from positions), bytes per token <= 10 KB, top-k set overlap fp32 vs fp16 pool"""
import ast
import importlib.util
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
LOADER = os.path.join(TREE, "exllamav3", "loader")
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))
torch = pytest.importorskip("torch")


def _mod(name):
    sp = importlib.util.spec_from_file_location(name + "_t", os.path.join(LOADER, name + ".py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _src(path, name, cls=None):
    text = open(os.path.join(TREE, path)).read()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(name)


def test_bytes_table_under_10kb():
    import kv_bytes as KB
    d = KB.per_token(fp8=True, fp8_mtp=True, swa_ring=True, pool_bytes=2, rope_pos=True)
    t, c = KB.total(d)
    assert t == 7728 + 644 + 832 + 16 and t == 9220
    assert KB.at_1m(d) < 10 * 1024 and KB.at_1m(d) < 10000          # owner rule: <= 10 KB / token at 1M (both readings of KB)
    assert KB.total(KB.per_token(fp8=True, fp8_mtp=True, swa_ring=True))[0] == 11332          # bcfb9e5 unchanged without the new levers
    assert KB.total(KB.per_token(fp8=True, fp8_mtp=True, swa_ring=True, pool_bytes=2))[0] > 10 * 1024      # the pool alone is not enough
    assert KB.total(KB.per_token(fp8=True, fp8_mtp=True, swa_ring=True, rope_pos=True))[0] > 10000        # nor are the rope rows alone


def test_attn_rope_rows_equal_the_table():
    RR = _mod("rope_rows")
    build = {}
    exec(_src("exllamav3/loader/attn_eager.py", "_build"), {"torch": torch}, build)
    n = 100000
    for base in (1e7, 10000.0):
        tab = build["_build"](64, base, n, "cpu")
        pos = torch.cat([torch.tensor([0, 1, n - 1, 65535, 65536]), torch.randint(0, n, (4000,))])
        assert torch.equal(RR.attn_rows(64, base, pos), tab[pos])


def test_indexer_rope_rows_equal_the_table():
    RR = _mod("rope_rows")
    src = _src("exllamav3/loader/indexer_fused.py", "_table")
    ns = {"torch": torch}
    exec(compile("class M:\n    _TABS = {}\n" + "\n".join("    " + ln for ln in src.splitlines()), "t", "exec"), ns)
    m = ns["M"]()
    m.base, m.rope_dim, m.dev, m._cs = 1e7, 64, "cpu", None
    cos, sin = m._table(50000)
    pos = torch.cat([torch.tensor([0, 3, 4, 49999]), torch.randint(0, 50000, (3000,))])
    c, s = RR.ix_rows(1e7, 64, pos)
    assert torch.equal(c, cos[pos]) and torch.equal(s, sin[pos])
    c, s = RR.ix_rows(1e7, 64, pos, 4)                              # rows of the group start (the pool rope position)
    g0 = (pos // 4) * 4
    assert torch.equal(c, cos[g0]) and torch.equal(s, sin[g0])


def test_rows_memo_follows_object_and_version():
    RR = _mod("rope_rows")
    pos = torch.tensor([5, 6, 7])
    a = RR.ix_rows(1e7, 64, pos)
    n0 = RR.STATS["ix"]
    assert RR.ix_rows(1e7, 64, pos)[0] is a[0] and RR.STATS["ix"] == n0           # same object, same version: shared by the layers
    pos.copy_(torch.tensor([9, 10, 11]))                                           # a static graph input overwritten in place
    b = RR.ix_rows(1e7, 64, pos)
    assert RR.STATS["ix"] == n0 + 1 and not torch.equal(a[0], b[0])
    assert RR.ix_rows(1e7, 64, pos.clone())[0] is not b[0]                         # a new object never hits


def test_pool16_topk_set_overlap_vs_fp32():
    """the selection set from an fp16-rounded pool vs the fp32 pool: random pooled keys (rms-normalised, 128-d), 16 heads, n_top 511 groups; real indexer inputs are gated by the GPU job (qual_kv10.py IXOVL)"""
    g = torch.Generator().manual_seed(1)
    rows = []
    for Pb in (4096, 16384, 65536):
        pool = torch.randn(Pb, 128, generator=g)
        pool = pool / pool.pow(2).mean(-1, keepdim=True).sqrt()
        iq = torch.randn(8, 16, 128, generator=g)
        wt = torch.rand(8, 16, generator=g) / 45
        def sel(pk):
            sc = (torch.relu(torch.einsum("nhd,pd->nhp", iq, pk)) * wt[:, :, None]).sum(1)
            return torch.topk(sc, 511, dim=1).indices
        a, b = sel(pool), sel(pool.half().float())
        ov = [len(set(a[t].tolist()) & set(b[t].tolist())) / 511 for t in range(8)]
        rows.append((Pb, min(ov), sum(ov) / 8))
    for Pb, mn, mean in rows:
        assert mean > 0.99 and mn > 0.985, (Pb, mn, mean)       # the cut-off neighbours swap; the bulk of the set is the same


def test_pool16_store_and_cast_sources():
    sg = open(os.path.join(LOADER, "step_graph.py")).read()
    assert 'HY4_IX_POOL16' in sg and "pooled.to(ix.pool.dtype)" in sg and 'ix.pool[:Pb].to(iq.dtype)' in sg
    assert "if not _RR.ENABLED:" in sg and "ix._table(cap)" in sg
    ixf = open(os.path.join(LOADER, "indexer_fused.py")).read()
    assert "dtype=POOL_DT" in ixf and "dtype=self.pool.dtype" in ixf and "self.pool[:P].to(iq.dtype)" in ixf
    assert "dtype=self.pool.dtype" in open(os.path.join(LOADER, "prefill_kernels.py")).read()
    assert "pool[:Pb].to(iq.dtype)" in open(os.path.join(LOADER, "lc_sparse.py")).read()


def test_compact_sets_the_kv10_levers_and_user_values_win(monkeypatch):
    text = open(os.path.join(TREE, "tools/hy4/serve.py")).read()
    node = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "apply_env"][0]
    ns = {"os": os, "SHIPPED_ENV": {}, "TREE": TREE, "glob": None}
    exec(compile(ast.get_source_segment(text, node), "apply_env", "exec"), ns)
    for k in ("HY4_KV_COMPACT", "HY4_KV_FP8", "HY4_KV_FP8_MTP", "HY4_SWA_RING", "HY4_IX_POOL16", "HY4_ROPE_POS"):
        monkeypatch.delenv(k, raising=False)
    ns["apply_env"](65536)
    assert "HY4_IX_POOL16" not in os.environ and "HY4_ROPE_POS" not in os.environ              # off without COMPACT
    monkeypatch.setenv("HY4_KV_COMPACT", "1")
    monkeypatch.setenv("HY4_IX_POOL16", "0")
    ns["apply_env"](262144)
    assert os.environ["HY4_IX_POOL16"] == "0" and os.environ["HY4_ROPE_POS"] == "1"
    for k in ("HY4_IX_POOL16", "HY4_ROPE_POS"):
        monkeypatch.delenv(k)
    ns["apply_env"](262144)
    assert os.environ["HY4_IX_POOL16"] == "1" and os.environ["HY4_ROPE_POS"] == "1"
    assert "HY4_IX_POOL16" in open(os.path.join(TREE, "tools/hy4/mu_engine.py")).read()


def test_gate_regex_matches_the_qual_line():
    line = "QKL_VS_NOISE w32k kl 0.012345 x noise-null max 0.8 | |dNLL| 0.0007 vs null max 0.004 | prefill tok/s ref 800 cand 790"
    m = re.match(r"QKL_VS_NOISE (\w+) kl ([\d.eE-]+) x noise-null max ([\d.]+) \| \|dNLL\| ([\d.eE-]+) vs null max ([\d.eE-]+)", line)
    assert m and m.group(1) == "w32k" and float(m.group(3)) == 0.8
    assert 'print("QKL_VS_NOISE", k, "kl", s["kl_mean"], "x noise-null max"' in open(os.path.join(TREE, "tools/hy4/step/qual_kv10.py")).read()


def test_triton_interpreter_rows_and_fp16_store():
    pytest.importorskip("triton")
    r = subprocess.run([sys.executable, os.path.join(HERE, "kv10_interp.py")], capture_output=True, text=True, timeout=900)
    assert "KV10_INTERP_OK" in r.stdout, r.stdout[-800:] + r.stderr[-1500:]


def _probe():
    sp = importlib.util.spec_from_file_location("kvp_t", os.path.join(TREE, "tools", "hy4", "kv_bytes_probe.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def test_probe_touch_sets_t0_and_reads_the_tables_back(capsys):
    P = _probe()
    seen = {}

    class GRStub:
        def step_rows(self, ids):
            seen.setdefault("steps", []).append(len(ids))

    class Eng:
        GR = GRStub()

        def reset(self):
            assert hasattr(self, "_t0")          # prefill() reads _t0: the touch must set it first (engine.py)
            seen["reset"] = True

        def prefill(self, ids):
            assert hasattr(self, "_t0") and len(ids) == 2048
            seen["prefill"] = True

        def phase(self, name):
            seen["phase"] = name

    class T:
        shape = (7, 2)

    class IXF:
        class FusedKPoolIndexer:
            _TABS = {}

    class RR:
        ENABLED = True
        STATS = {"attn": 3, "ix": 2}

    class AE:
        _TAB = {}

    assert P.touch(Eng(), 10, lambda: None, lambda: 25, AE, IXF, RR) == []
    assert seen["steps"] == [4, 4] and seen["phase"] == "decode"
    out = capsys.readouterr().out
    assert out.startswith("KVTOUCH ") and '"alloc_after_B": 25' in out and '"rope_table_rows_attn": 0' in out
    AE._TAB = {("k",): T()}                      # a full-context table left after the touch while HY4_ROPE_POS is on -> failure string
    assert "full-context rope table" in P.touch(Eng(), 10, lambda: None, lambda: 25, AE, IXF, RR)[0]


def test_probe_parent_keeps_the_measurement_of_a_failed_child_and_prints_touch_slope(monkeypatch, capsys):
    import json
    P = _probe()
    monkeypatch.setenv("K3_ARMS", "kv10:1000,3000:HY4_KV_COMPACT=1")
    monkeypatch.setenv("K3_OUT", os.path.join(os.environ.get("TMPDIR", "/tmp"), "kv10_probe_test.json"))

    class R:
        def __init__(self, c, rc):
            c = int(c)
            self.returncode, self.stderr = rc, "boom"
            self.stdout = ("KVBYTES " + json.dumps({"ctx": c, "cap": c, "alloc_B": c * 9220, "comp_B": {"a": c * 9000}}) + "\nKVTOUCH "
                           + json.dumps({"alloc_after_B": c * 9220 + 500}) + "\nFAIL x\n")

    monkeypatch.setattr(P.subprocess, "run", lambda cmd, env, **kw: R(env["K3_CHILD"], 3))
    with pytest.raises(SystemExit) as e:
        P.parent()
    out = capsys.readouterr().out
    assert e.value.code == 1 and "DRIVER_FAIL" in out
    assert "SLOPE [kv10] ctx 1000->3000: total 9220 B/token" in out and "SLOPE_TOUCH [kv10] ctx 1000->3000: 9220 B/token" in out


def test_driver_gate_lines_and_fires_kv10():
    d = open(os.path.join(TREE, "tools/hy4/step/kv10_driver.py")).read()
    m = re.search(r'm = re.match\(r"(IXOVL[^"]+)", ln\)', d)
    assert m and re.match(m.group(1), "IXOVL rows 40 mean_overlap 0.9991 min_overlap 0.9900 exact_rows 12")
    assert "KV10_GATE topk" in d and 'fires.get("kv10", 0)' in d
    q = open(os.path.join(TREE, "tools/hy4/step/qual_kv10.py")).read()
    assert 'print(f"FIRES kv10 {min(flagged)}"' in q and 'IXOVL rows' in q
