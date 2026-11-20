"""GPU checks for the MXFP4 grouped prefill (EXL3_MXFP4_PREFILL_GROUPED): small random packed MXFP4 experts.
1. _dq_mx (Triton) == gap_formats.dequant('mxfp4') exactly
2. BlockSparseMLP._forward_mxfp4_prefill_grouped output vs the per-expert dequant + fp16 matmul path, rel L2 < 1e-2.
Modules are loaded by path (no extension build); the activation is the pure-torch silu_oai_mul of ext_fallbacks."""
import ast
import importlib.util
import os
import sys
import types

import pytest
import torch

try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="needs CUDA and triton")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "exllamav3")


def _load_all():
    """Load gap_formats, triton_gap (if triton), fq_moe_prefill (if triton) and moe_mxfp4_prefill by path under empty parent packages,
    so that exllamav3/__init__ (which builds the C++ extension) never runs. Returns (modules dict, parent packages)."""
    pk = {n: types.ModuleType(n) for n in ("exllamav3", "exllamav3.foreign_quant", "exllamav3.modules", "exllamav3.util")}
    pk["exllamav3"].__path__ = [PKG]
    pk["exllamav3.foreign_quant"].__path__ = [os.path.join(PKG, "foreign_quant")]
    pk["exllamav3.modules"].__path__ = [os.path.join(PKG, "modules")]
    pk["exllamav3.util"].__path__ = [os.path.join(PKG, "util")]
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
        m = _load_all()
        spec = importlib.util.spec_from_file_location("exllamav3.ext_fallbacks", os.path.join(PKG, "ext_fallbacks.py"))
        fb = importlib.util.module_from_spec(spec)
        sys.modules["exllamav3.ext_fallbacks"] = fb
        spec.loader.exec_module(fb)
        m["fallbacks"] = fb
        yield m
    finally:
        for k in [k for k in sys.modules if k == "exllamav3" or k.startswith("exllamav3.")]:
            sys.modules.pop(k)
        sys.modules.update({k: v for k, v in saved.items() if v is not None})


def _rand_packed(n, k, gen, dev):
    """[n, k/32 * 17] uint8: realistic e8m0 scale bytes + random nibbles."""
    nb = k // 32
    raw = torch.randint(0, 256, (n, nb, 17), dtype=torch.uint8, generator=gen)
    raw[:, :, 0] = torch.randint(118, 128, (n, nb), dtype=torch.uint8, generator=gen)
    return raw.reshape(n, nb * 17).to(dev)


def test_dq_mx_equals_gap_dequant(mods):
    MP, gf = mods["moe_mxfp4_prefill"], mods["gap_formats"]
    dev, gen = "cuda", torch.Generator().manual_seed(1)
    E, N, K = 5, 48, 96
    w = [_rand_packed(N, K, gen, dev) for _ in range(E)]
    tab = torch.tensor([t.data_ptr() for t in w], dtype=torch.int64, device=dev)
    ids = torch.tensor([3, 0, 4], dtype=torch.int32, device=dev)
    out = torch.empty((3, N, K), dtype=torch.half, device=dev)
    MP.dequant_group(tab, ids, out, N, K)
    for i, e in enumerate(ids.tolist()):
        ref = gf.dequant("mxfp4", w[e])
        assert torch.equal(out[i].view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("scratch_mb", [None, "0"])      # "0": every expert in its own group
def test_prefill_matches_reference(mods, monkeypatch, scratch_mb):
    MP, gf, fb = mods["moe_mxfp4_prefill"], mods["gap_formats"], mods["fallbacks"]
    dev, gen = "cuda", torch.Generator().manual_seed(2)
    E, S, H, I, R, LIM = 8, 2, 64, 64, 300, 7.0
    ns = types.SimpleNamespace

    def lin(i, o):
        return ns(inner=ns(fl=ns(in_features=i, out_features=o, packed=_rand_packed(o, i, gen, dev)),
                           bias=(torch.randn(o, generator=gen) * 0.1).half().to(dev)))
    st = ns(_mxfp4_grouped_ok=lambda y: True, num_local_experts=None, num_experts=E, num_experts_per_tok=S, shared_experts=None,
            act_limit=LIM, activation_fn_call=fb.silu_oai_mul)
    st.gates, st.ups, st.downs = [[lin(i, o) for _ in range(E)] for i, o in ((H, I), (H, I), (I, H))]
    monkeypatch.setenv("EXL3_MXFP4_PREFILL_GROUPED", "1")
    monkeypatch.setattr(torch.version, "hip", None)
    if scratch_mb is not None:
        monkeypatch.setenv("EXL3_MXFP4_PREFILL_SCRATCH_MB", scratch_mb)
    y = (torch.randn(R, H, generator=gen) * 0.5).half().to(dev)
    sel = torch.stack([torch.randperm(E, generator=gen)[:S] for _ in range(R)]).to(dev)
    wts = torch.softmax(torch.randn(R, S, generator=gen), -1).half().to(dev)

    meth = _mlp_method()
    got = meth(st, y, y, sel, wts, {}, None, (R, H))
    assert got is not None and got.shape == (R, H)

    # reference: per-expert dequant + fp16 matmul + bias, same activation, routing weight, fp32 sum
    def lw(m):
        return gf.dequant("mxfp4", m.inner.fl.packed), m.inner.bias
    ref = torch.zeros(R, H, dtype=torch.float32, device=dev)
    for e in range(E):
        rr, ss = (sel == e).nonzero(as_tuple=True)
        if rr.numel() == 0:
            continue
        x = y[rr]
        (wg, bg), (wu, bu), (wd, bd) = lw(st.gates[e]), lw(st.ups[e]), lw(st.downs[e])
        g, u = x @ wg.T + bg, x @ wu.T + bu
        a = torch.empty_like(g)
        fb.silu_oai_mul(g, u, a, LIM)
        d = (a @ wd.T + bd) * wts[rr, ss, None]
        ref.index_add_(0, rr, d.float())
    rel = (got.float() - ref).norm() / ref.norm()
    assert rel < 1e-2, rel.item()
