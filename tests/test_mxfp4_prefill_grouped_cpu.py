"""CPU checks for the MXFP4 grouped prefill (foreign_quant/moe_mxfp4_prefill.py), EXL3_MXFP4_PREFILL_GROUPED. No GPU, no extension build.
1. numpy reference of the dequant kernel == gap_formats.dequant('mxfp4') on the same bytes (bitwise, all 256 scale bytes)
2. scratch budget grouping (split_groups)
3. dispatch: flag 0/1, min rows, forced low min, HIP / no CUDA; BlockSparseMLP._forward_mxfp4_prefill_grouped with a stub self (run() stubbed).
Modules are loaded by path and the method is cut out of the source by ast: nothing imports exllamav3.ext."""
import ast
import importlib.util
import os
import sys
import types

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "exllamav3")


def _load_all():
    """Load gap_formats, triton_gap (if triton), fq_moe_prefill (if triton) and moe_mxfp4_prefill by path under empty parent packages,
    so that exllamav3/__init__ (which builds the C++ extension) never runs. Returns (modules dict, parent packages)."""
    pk = {n: types.ModuleType(n) for n in ("exllamav3", "exllamav3.foreign_quant", "exllamav3.modules")}
    pk["exllamav3"].__path__ = [PKG]
    pk["exllamav3.foreign_quant"].__path__ = [os.path.join(PKG, "foreign_quant")]
    pk["exllamav3.modules"].__path__ = [os.path.join(PKG, "modules")]
    for k, v in pk.items():
        sys.modules[k] = v
    try:
        import triton  # noqa: F401
        have = True
    except Exception:
        have = False
    out = {}
    for name, rel, needs in (("exllamav3.foreign_quant.gap_formats", "foreign_quant/gap_formats.py", False),
                             ("exllamav3.foreign_quant.triton_gap", "foreign_quant/triton_gap.py", True),
                             ("exllamav3.modules.fq_moe_prefill", "modules/fq_moe_prefill.py", True),
                             ("exllamav3.foreign_quant.moe_mxfp4_prefill", "foreign_quant/moe_mxfp4_prefill.py", False)):
        if needs and not have:
            continue
        spec = importlib.util.spec_from_file_location(name, os.path.join(PKG, rel))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        parent, _, leaf = name.rpartition(".")
        setattr(pk[parent], leaf, m)
        out[leaf] = m
    return out


def _mlp_method(name="_forward_mxfp4_prefill_grouped", fq_max_rows=128):
    """The BlockSparseMLP method under test, compiled from the source file without importing the module."""
    tree = ast.parse(open(os.path.join(PKG, "modules/block_sparse_mlp.py")).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BlockSparseMLP")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch, "FQ_GROUPED_MAX_ROWS": fq_max_rows, "__name__": "exllamav3.modules.block_sparse_mlp",
          "__package__": "exllamav3.modules"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "block_sparse_mlp.py", "exec"), ns)
    return ns[name]


@pytest.fixture(scope="module")
def mods():
    saved = {k: sys.modules.get(k) for k in list(sys.modules) if k == "exllamav3" or k.startswith("exllamav3.")}
    try:
        yield _load_all()
    finally:
        for k in [k for k in sys.modules if k == "exllamav3" or k.startswith("exllamav3.")]:
            sys.modules.pop(k)
        sys.modules.update({k: v for k, v in saved.items() if v is not None})


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("EXL3_MXFP4_PREFILL_GROUPED", "EXL3_MXFP4_PREFILL_GROUPED_MIN"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_dequant_reference_matches_engine(mods):
    MP, gf = mods["moe_mxfp4_prefill"], mods["gap_formats"]
    rng = np.random.default_rng(0)
    for N, Kb in ((8, 7), (3, 90), (16, 1)):
        raw = rng.integers(0, 256, size=(N, Kb * 17), dtype=np.uint8)
        raw.reshape(N, Kb, 17)[:, :, 0] = rng.integers(100, 150, size=(N, Kb))
        ref = MP.ref_dequant_np(raw)
        eng = gf.dequant("mxfp4", torch.from_numpy(raw)).numpy()
        assert ref.shape == eng.shape and ref.dtype == eng.dtype and np.array_equal(ref.view(np.uint16), eng.view(np.uint16))
    raw = np.zeros((256, 17), dtype=np.uint8)
    raw[:, 0] = np.arange(256)
    raw[:, 1:] = rng.integers(0, 256, size=(256, 16), dtype=np.uint8)
    eng = gf.dequant("mxfp4", torch.from_numpy(raw)).numpy()
    assert np.array_equal(MP.ref_dequant_np(raw).view(np.uint16), eng.view(np.uint16))
    assert tuple(MP.FP4_X2) == tuple(gf._FP4_HALF)


@pytest.mark.parametrize("n,bud,exp", [(0, 1 << 30, 0), (1, 1, 1), (32, 1536 << 20, 1), (32, 600 << 20, 2), (32, 100 << 20, 11), (5, 1, 5)])
def test_split_groups(mods, n, bud, exp):
    MP = mods["moe_mxfp4_prefill"]
    pe = max(2 * 2880 * 2880, 2880 * 2880) * 2
    g = MP.split_groups(n, pe, bud)
    assert len(g) == exp
    if g:
        assert g[0][0] == 0 and g[-1][1] == n and all(g[i][1] == g[i + 1][0] for i in range(len(g) - 1))
    cap = max(1, bud // pe)
    assert all(b - a <= cap and b > a for a, b in g)
    assert max([b - a for a, b in g], default=0) - min([b - a for a, b in g], default=0) <= 1


def test_budget_env(mods, clean_env):
    MP = mods["moe_mxfp4_prefill"]
    assert MP.budget_bytes() == 1536 << 20
    clean_env.setenv("EXL3_MXFP4_PREFILL_SCRATCH_MB", "64")
    assert MP.budget_bytes() == 64 << 20


def test_use_prefill(mods, clean_env):
    MP = mods["moe_mxfp4_prefill"]
    clean_env.setattr(torch.version, "cuda", "12.8")
    clean_env.setattr(torch.version, "hip", None)
    assert not MP.use_prefill(8192, 128)                                    # unset -> off
    clean_env.setenv("EXL3_MXFP4_PREFILL_GROUPED", "0")
    assert not MP.use_prefill(8192, 128)
    clean_env.setenv("EXL3_MXFP4_PREFILL_GROUPED", "1")
    assert MP.use_prefill(8192, 128) and MP.use_prefill(129, 128)
    assert not MP.use_prefill(128, 128) and not MP.use_prefill(1, 128)
    clean_env.setenv("EXL3_MXFP4_PREFILL_GROUPED_MIN", "2")
    assert MP.use_prefill(34, 128) and not MP.use_prefill(1, 128)
    clean_env.setattr(torch.version, "hip", "6.2")
    assert not MP.use_prefill(8192, 128)
    clean_env.setattr(torch.version, "hip", None)
    clean_env.setattr(torch.version, "cuda", None)
    assert not MP.use_prefill(8192, 128)


def _stub(ok=True, bias=True, E=4, H=64, I=32):
    ns = types.SimpleNamespace

    def lin(i, o):
        return ns(inner=ns(fl=ns(in_features=i, out_features=o, packed=torch.zeros(4, dtype=torch.uint8)),
                           bias=torch.zeros(o, dtype=torch.half) if bias else None))
    s = ns(_mxfp4_grouped_ok=lambda y: ok, num_local_experts=None, num_experts=E, num_experts_per_tok=2, shared_experts=None, act_limit=7.0,
           activation_fn_call=lambda g, u, a, l: None)
    s.gates, s.ups, s.downs = [lin(H, I) for _ in range(E)], [lin(H, I) for _ in range(E)], [lin(I, H) for _ in range(E)]
    return s


def test_dispatch(mods, clean_env):
    MP = mods["moe_mxfp4_prefill"]
    clean_env.setattr(torch.version, "cuda", "12.8")
    clean_env.setattr(torch.version, "hip", None)
    meth = _mlp_method()
    calls = []

    def fake_run(h, sel, wts, tbl, E, I, K, N, act_fn, lim, budget=None):
        calls.append((tuple(h.shape), tuple(sel.shape), wts.dtype, E, I, K, N, lim))
        return torch.zeros(h.shape[0], N)
    clean_env.setattr(MP, "run", fake_run)

    def call(rows, **kw):
        y = torch.zeros(rows, 64, dtype=torch.half)
        sel = torch.zeros(rows, 2, dtype=torch.long)
        return meth(_stub(**kw), y, y, sel, torch.ones(rows, 2), {}, None, (rows, 64))

    assert call(300) is None and not calls                                  # flag off
    clean_env.setenv("EXL3_MXFP4_PREFILL_GROUPED", "1")
    r = call(300)
    assert r is not None and len(calls) == 1
    assert calls[0] == ((300, 64), (300, 2), torch.half, 4, 32, 64, 64, 7.0)
    calls.clear()
    assert call(128) is None and not calls                                  # rows == decode max
    assert call(300, ok=False) is None and not calls                        # layer not eligible
    clean_env.setenv("EXL3_MXFP4_PREFILL_GROUPED_MIN", "2")
    assert call(34) is not None and len(calls) == 1
