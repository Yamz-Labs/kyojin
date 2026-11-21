"""zero-GPU checks of the Strix decode levers HY4_DENSE_V2, HY4_MOE_V2, HY4_DRAFT_MOE_TILES, HY4_DRAFT_MLA_BG, HY4_DRAFT_IX_GV:
env switches default off ("0"), the fallback paths are the stock ones, every fallback reason and every hit is counted (loader/lever_fires.py),
the draft layer's quantiser follows the Q8_0 scale rule, and indexer_keys_static keeps its @torch.no_grad()."""
import ast
import importlib
import os
import re
import sys
import types
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
LOADER = os.path.join(TREE, "exllamav3", "loader")
os.environ.setdefault("HY4_ALLOW_SLOW", "1")


def _src(name):
    return open(os.path.join(LOADER, name)).read()


@pytest.fixture(scope="module")
def mods():
    """the loader modules by path, without the exllamav3 package import (which builds the extension); the stub entries are removed again"""
    added = []
    if "exllamav3" not in sys.modules:
        for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
            m = types.ModuleType(n)
            m.__path__ = [os.path.join(TREE, "exllamav3", s)]
            sys.modules[n] = m
            added.append(n)
    before = set(sys.modules)
    out = SimpleNamespace(**{k: importlib.import_module(f"exllamav3.loader.{k}") for k in ("lever_fires", "hy4_forward", "moe_i8", "mla_bgemv", "step_graph")})
    yield out
    if added:
        for n in set(sys.modules) - before | set(added):
            if n.startswith("exllamav3"):
                sys.modules.pop(n, None)


@pytest.fixture(autouse=True)
def fresh(mods):
    for f in mods.lever_fires.REG.values():
        f.reset()
    yield


# ---- switches ----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("flag,path", [("HY4_DENSE_V2", "hy4_forward.py"), ("HY4_MOE_V2", "hy4_forward.py"), ("HY4_DRAFT_MOE_TILES", "moe_i8.py"),
                                       ("HY4_DRAFT_MLA_BG", "mtp_head.py"), ("HY4_DRAFT_IX_GV", "mtp_head.py")])
def test_flag_is_an_env_switch_defaulting_off(flag, path):
    assert re.search(r'environ\.get\("%s", "0"\) == "1"' % flag, _src(path)), flag


def test_lever_flags_unset_means_off(mods):
    if not any(k in os.environ for k in ("HY4_DENSE_V2", "HY4_MOE_V2", "HY4_DRAFT_MOE_TILES")):
        assert mods.hy4_forward.DENSE_V2 == [False] and mods.hy4_forward.MOE_V2 is False and mods.moe_i8.TILES is False


def test_draft_moe_tiles_follow_the_flag(mods, monkeypatch):
    try:
        monkeypatch.delenv("HY4_DRAFT_MOE_TILES", raising=False)
        m = importlib.reload(mods.moe_i8)
        assert (m.GU_T, m.DN_T) == ((16, 128, 4), (8, 128, 4))
        m.FIRES.reset()
        with pytest.raises(Exception):                      # no GPU: the launch fails after the counter
            m.experts_forward(SimpleNamespace(N=16, K=32, q=None, s=None), SimpleNamespace(q=None, s=None), SimpleNamespace(N=16, K=16, q=None, s=None), torch.zeros(32), torch.zeros(2, dtype=torch.long), torch.zeros(2), 10.0)
        assert m.FIRES.hits == 0 and m.FIRES.fallback == {"flag_off": 1}
        monkeypatch.setenv("HY4_DRAFT_MOE_TILES", "1")
        m = importlib.reload(mods.moe_i8)
        assert (m.GU_T, m.DN_T) == ((8, 512, 8), (16, 512, 4))
        m.FIRES.reset()
        with pytest.raises(Exception):
            m.experts_forward(SimpleNamespace(N=16, K=32, q=None, s=None), SimpleNamespace(q=None, s=None), SimpleNamespace(N=16, K=16, q=None, s=None), torch.zeros(32), torch.zeros(2, dtype=torch.long), torch.zeros(2), 10.0)
        assert m.FIRES.hits == 1 and m.FIRES.fallback == {}
    finally:
        monkeypatch.undo()
        importlib.reload(mods.moe_i8)


# ---- DENSE_V2 ----------------------------------------------------------------------------------------------------------------------------

def _stock(calls):
    def fq_gemv_q8_0(xf, packed, out):
        calls.append("stock")
    return fq_gemv_q8_0


class _Gm:
    def __init__(self, ok=True):
        self.ok, self.n = ok, 0

    def gm_dense_q8(self, qb, packed, out, cpb):
        self.n += 1
        return self.ok


class _Ext:
    def fq_q8_scratch(self, dev, R, K):
        return object()

    def fq_quant_q8_1_rows(self, xf, qb):
        pass


def test_dense_gemv_flag_off_is_stock_and_silent(mods, monkeypatch):
    hf, calls = mods.hy4_forward, []
    monkeypatch.setattr(hf, "DENSE_V2", [False])
    hf.dense_gemv(_stock(calls), torch.zeros(4, 256), None, None)
    assert calls == ["stock"] and hf.DENSE_FIRES.snapshot() == {"v2": 0, "stock": {}}


@pytest.mark.parametrize("shape,name,reason", [((1, 256), "fq_gemv_q8_0", "rows<2"), ((4, 8192 + 32), "fq_gemv_q8_0", "K>kmax"), ((4, 256), "other_gemv", "not_q8_0")])
def test_dense_gemv_stock_reasons(mods, monkeypatch, shape, name, reason):
    hf, calls = mods.hy4_forward, []
    monkeypatch.setattr(hf, "DENSE_V2", [True])
    g = _stock(calls)
    g.__name__ = name
    hf.dense_gemv(g, torch.zeros(*shape), None, None)
    assert calls == ["stock"] and hf.DENSE_FIRES.snapshot() == {"v2": 0, "stock": {reason: 1}}


def test_dense_gemv_no_extension_falls_back(mods, monkeypatch):
    hf, calls = mods.hy4_forward, []
    monkeypatch.setattr(hf, "DENSE_V2", [True])
    monkeypatch.setattr(hf, "_gm", lambda: None)
    monkeypatch.setattr(hf, "_ext", lambda: None)
    hf.dense_gemv(_stock(calls), torch.zeros(4, 256), None, None)
    assert calls == ["stock"] and hf.DENSE_FIRES.snapshot()["stock"] == {"no_gm_ext": 1}


@pytest.mark.parametrize("ok", [True, False])
def test_dense_gemv_v2_and_rejected(mods, monkeypatch, ok):
    hf, calls, gm = mods.hy4_forward, [], _Gm(ok)
    monkeypatch.setattr(hf, "DENSE_V2", [True])
    monkeypatch.setattr(hf, "_gm", lambda: gm)
    monkeypatch.setattr(hf, "_ext", lambda: _Ext())
    hf.dense_gemv(_stock(calls), torch.zeros(4, 256), None, None)
    assert gm.n == 1
    if ok:
        assert calls == [] and hf.DENSE_FIRES.snapshot() == {"v2": 1, "stock": {}}
    else:
        assert calls == ["stock"] and hf.DENSE_FIRES.snapshot() == {"v2": 0, "stock": {"kernel_rejected": 1}}


# ---- MOE_V2 ------------------------------------------------------------------------------------------------------------------------------

class _MoeDec:
    def __init__(self, dev, T, topk, K, nff, fmt, E):
        self.topk, self.args = topk, (T, K, nff, fmt, E)


def _rs(**kw):
    d = dict(fu=8, fd=0, nout=1536, nff=1536, strides=(1536, 2048, 4096), bufs={"gate": torch.zeros(64)}, E=512)
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture
def moe(mods, monkeypatch):
    hf = mods.hy4_forward
    md = SimpleNamespace(MoeDec=_MoeDec)
    monkeypatch.setattr(hf, "MOE_V2", True)
    monkeypatch.setattr(hf, "_MD", [md, {}])
    return hf


def test_moe_v2_flag_off_is_silent(mods, monkeypatch):
    hf = mods.hy4_forward
    monkeypatch.setattr(hf, "MOE_V2", False)
    assert hf._moe_v2(_rs(), 1, 1536) is None and hf.MOE_V2_FIRES.snapshot() == {"calls": 0, "fallback": {}}


def test_moe_v2_happy_path(moe):
    h = moe._moe_v2(_rs(), 2, 1536)
    assert h is not None and h.args == (2, 1536, 1536, "q3_k", 512)
    assert moe._moe_v2(_rs(fu=0), 2, 1536).args[3] == "q4_k"
    assert moe.MOE_V2_FIRES.snapshot() == {"calls": 2, "fallback": {}}
    assert moe._moe_v2(_rs(), 2, 1536) is moe._MD[1][(2, "q3_k", 1536, 1536)]          # one holder per (T, fmt, K, nff)


@pytest.mark.parametrize("args,rs,reason", [
    ((1, 1536, 8), {}, "topk!=16"),
    ((99, 1536, 16), {}, "T>maxr"),
    ((1, 1536, 16), {"fu": 5}, "gate_up_fmt"),
    ((1, 1536, 16), {"fd": 1}, "down_fmt"),
    ((1, 1536, 16), {"nout": 1024}, "nout!=K"),
    ((1, 1024, 16), {"nout": 1024}, "shape"),
    ((1, 1536, 16), {"nff": 2048}, "shape"),
    ((1, 1536, 16), {"strides": (1536, 2051, 4096)}, "stride_align"),             # Q3_K gate / up need 2-byte rows: an odd row is refused
    ((1, 1536, 16), {"strides": (1536, 2048, 4098)}, "stride_align"),             # down is Q4_K: 16-byte rows
    ((1, 1536, 16), {"fu": 0, "strides": (1536, 2050, 4096)}, "stride_align"),    # Q4_K gate / up: 16-byte rows
])
def test_moe_v2_fallback_reasons(moe, args, rs, reason):
    T, K, S = args
    assert moe._moe_v2(_rs(**rs), T, K, S) is None
    assert moe.MOE_V2_FIRES.snapshot() == {"calls": 0, "fallback": {reason: 1}}


def test_moe_v2_pointer_alignment_and_missing_code_object(moe, monkeypatch):
    t = torch.zeros(64)
    odd = t[1:]                                       # 4-byte offset: not 16-byte aligned
    assert moe._moe_v2(_rs(fu=0, bufs={"gate": odd}), 1, 1536) is None            # Q4_K gate: uint4 loads need 16 bytes
    monkeypatch.setattr(moe, "_MD", [None, {}])
    assert moe._moe_v2(_rs(), 1, 1536) is None
    assert moe.MOE_V2_FIRES.snapshot()["fallback"] == {"ptr_align": 1, "no_code_object": 1}


def test_moe_v2_topk_other_than_16_is_refused(moe, monkeypatch):
    class Odd(_MoeDec):
        def __init__(self, *a):
            super().__init__(*a)
            self.topk = 8
    monkeypatch.setattr(moe._MD[0], "MoeDec", Odd)
    assert moe._moe_v2(_rs(), 1, 1536) is None and moe.MOE_V2_FIRES.snapshot()["fallback"] == {"topk": 1}


# ---- DRAFT_MLA_BG ------------------------------------------------------------------------------------------------------------------------

def test_from_float_follows_the_q8_0_rule(mods):
    BG = mods.mla_bgemv
    torch.manual_seed(0)
    w = torch.randn(3, 5, 64) * 2
    o = BG.Q8Batched.from_float(w)
    assert (o.H, o.N, o.K) == (3, 5, 64) and o.q.shape == (3, 5, 64) and o.d.shape == (3, 5, 2)
    assert o.q.dtype == torch.int8 and o.d.dtype == torch.float16 and int(o.q.abs().max()) <= 127
    blk = w.reshape(3, 5, 2, 32)
    assert torch.equal(o.d.float(), (blk.abs().amax(-1) / 127.0).half().float())          # scale = amax / 127, stored fp16
    deq = o.q.float().reshape(3, 5, 2, 32) * o.d.float()[..., None]
    assert float((deq - blk).abs().max()) <= float(o.d.float().max()) * 0.51                 # round-trip error <= half a step (+ fp16 scale rounding)
    assert o.draft is True and BG.Q8Batched.draft is False


def test_from_float_zero_block_is_finite(mods):
    o = mods.mla_bgemv.Q8Batched.from_float(torch.zeros(1, 2, 32))
    assert torch.isfinite(o.d.float()).all() and int(o.q.abs().max()) == 0


def test_draft_matrix_mv_is_counted(mods):
    BG = mods.mla_bgemv
    o = BG.Q8Batched.from_float(torch.randn(2, 4, 32))
    with pytest.raises(Exception):                    # no GPU: the Triton launch fails; the counter is incremented before it
        o.mv(torch.randn(1, 2, 32))
    assert BG.DRAFT_FIRES.hits == 1


def test_mtp_head_sets_the_draft_matrices_only_with_the_flag():
    src = _src("mtp_head.py")
    assert re.search(r'if self\.fast and os\.environ\.get\("HY4_DRAFT_MLA_BG", "0"\) == "1":\s*#?[^\n]*\n\s+from \. import mla_bgemv as _BGM\n\s+mla\.bg_kb, mla\.bg_vb = ', src)


# ---- DRAFT_IX_GV + the no_grad regression ------------------------------------------------------------------------------------------------

def test_mmw_without_gv_is_the_gemm_bit_for_bit(mods):
    SG = mods.step_graph
    a, w = torch.randn(3, 16), torch.randn(8, 16)
    ix = SimpleNamespace(w={"x": w})
    assert torch.equal(SG._mmw(ix, a, "x"), a @ w.T) and SG.IX_GV_FIRES.snapshot() == {"gv": 0, "gemm": {}}          # the trunk's indexer: not counted
    ix.gv = False
    assert torch.equal(SG._mmw(ix, a, "x"), a @ w.T) and SG.IX_GV_FIRES.snapshot() == {"gv": 0, "gemm": {"flag_off": 1}}


def test_mmw_with_gv_takes_the_matvec(mods, monkeypatch):
    SG = mods.step_graph
    a, w = torch.randn(1, 16), torch.randn(8, 16)
    ix = SimpleNamespace(w={"x": w}, gv=True)
    monkeypatch.setattr(SG, "_RG", SimpleNamespace(usable=lambda a, w: True, logits=lambda a, w: "mv", MAXT=8))
    assert SG._mmw(ix, a, "x") == "mv"
    monkeypatch.setattr(SG, "_RG", SimpleNamespace(usable=lambda a, w: False, logits=lambda a, w: "mv", MAXT=8))
    assert torch.equal(SG._mmw(ix, a, "x"), a @ w.T)
    assert SG.IX_GV_FIRES.snapshot() == {"gv": 1, "gemm": {"unusable_shape": 1}}
    a9 = torch.randn(9, 16)                                                      # more rows than the mat-vec takes (draft-cache fill chunks): counted as the natural reason
    assert torch.equal(SG._mmw(ix, a9, "x"), a9 @ w.T)
    assert SG.IX_GV_FIRES.snapshot() == {"gv": 1, "gemm": {"unusable_shape": 1, "rows>maxt": 1}}


def test_indexer_keys_static_keeps_no_grad(mods):
    SG = mods.step_graph
    assert hasattr(SG.indexer_keys_static, "__wrapped__")                        # torch.no_grad() decorates it (b2 619c92f put it on _mmw by mistake)
    tree = ast.parse(_src("step_graph.py"))
    deco = {n.name: [ast.unparse(d) for d in n.decorator_list] for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "torch.no_grad()" in deco["indexer_keys_static"] and deco["_mmw"] == []


# ---- the fires line ----------------------------------------------------------------------------------------------------------------------

def test_fires_line(mods):
    LF = mods.lever_fires
    assert "none" in LF.line()
    mods.hy4_forward.DENSE_FIRES.hit()
    mods.hy4_forward.DENSE_FIRES.miss("rows<2")
    mods.moe_i8.FIRES.hit()
    ln = LF.line()
    assert "FIRES dense_v2 {'v2': 1, 'stock': {'rows<2': 1}}" in ln and "FIRES draft_moe_tiles {'calls': 1, 'fallback': {}}" in ln


# ---- MOE_V2 alignment contract (22 of 42 MoE layers of the Q3_K GGUF have 660-byte Q3_K gate / up rows) ------------------------------------------------------------------

def test_moe_v2_q3k_660_byte_rows_take_the_code_object(moe):
    strides = (660 * 1, 660, 864)                                                  # real Q3_K gate / up (K 1536: 6 blocks x 110 B) with Q4_K down (6 x 144 B)
    assert 660 % 16 != 0 and 864 % 16 == 0
    assert moe._moe_v2(_rs(fu=8, strides=strides, bufs={"gate": torch.zeros(64), "up": torch.zeros(64), "down": torch.zeros(64)}), 1, 1536) is not None
    assert moe.MOE_V2_FIRES.snapshot() == {"calls": 1, "fallback": {}}


def test_moe_v2_q4k_layers_keep_the_16_byte_rule(moe):
    assert moe._moe_v2(_rs(fu=0, strides=(864, 864, 864)), 1, 1536) is not None
    assert moe._moe_v2(_rs(fu=0, strides=(660, 864, 864)), 1, 1536) is None
    assert moe.MOE_V2_FIRES.snapshot() == {"calls": 1, "fallback": {"stride_align": 1}}
