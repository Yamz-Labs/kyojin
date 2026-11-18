"""CPU checks for EXL3_FQ_MMVQ_MXFP4 (no GPU, no kernel launch, no extension build).

1. e8m0 convention: ggml (kvalues x ggml_e8m0_to_fp32(e) * 0.5) == engine dequant (exp2(e - 128) x fp4) on the same bytes
2. numpy emulation of vec_dot_mxfp4_q8_1 (+ the 4-warp lane partition of the kernel) vs dequant(W) @ x, rel < 1e-2
3. numpy emulation of the gate/up epilogue (bias, fp16 rounding, clamp, swiglu_oai)
4. dispatch with stubs: flag 0 -> Triton calls only (identical to today), flag 1 -> quant, gu, quant, down, fold;
   R > 1 / other activation / no bias -> old path

Modules are loaded by path (gap_formats, triton_moe_mxfp4) or cut out of the source by ast (the two BlockSparseMLP methods), so that
nothing imports exllamav3.ext (which would start the extension build).
"""
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

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int64)   # kvalues_mxfp4 (= doubled e2m1)


class _Dummy:
    """Stands in for triton / triton.language when only the module-level code of triton_moe_mxfp4 has to run."""
    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else _Dummy()
    def __getattr__(self, n):
        return _Dummy()
    def __getitem__(self, k):
        return _Dummy()


def _load(name, rel, extra_modules=None):
    """Load exllamav3/<rel> by path as exllamav3.<name> under empty parent packages (their __init__ is never run)."""
    saved = {}
    mods = {"exllamav3": types.ModuleType("exllamav3"), "exllamav3.foreign_quant": types.ModuleType("exllamav3.foreign_quant")}
    mods["exllamav3"].__path__ = [PKG]
    mods["exllamav3.foreign_quant"].__path__ = [os.path.join(PKG, "foreign_quant")]
    mods.update(extra_modules or {})
    for k, v in mods.items():
        saved[k] = sys.modules.get(k)
        sys.modules[k] = v
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(PKG, rel))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    finally:
        sys.modules.pop(name, None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _gap_formats():
    return _load("exllamav3.foreign_quant.gap_formats", "foreign_quant/gap_formats.py")


def _mx():
    return _load("exllamav3.foreign_quant.triton_moe_mxfp4", "foreign_quant/triton_moe_mxfp4.py",
                 {"triton": _Dummy(), "triton.language": _Dummy()})


def _mlp_methods():
    """The two BlockSparseMLP methods under test, compiled from the source file without importing the module."""
    src = open(os.path.join(PKG, "modules/block_sparse_mlp.py")).read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BlockSparseMLP")
    want = ("_mxfp4_mmvq_forward", "_forward_mxfp4_grouped")
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert len(fns) == 2
    mod = ast.Module(body=fns, type_ignores=[])
    ns = {"os": os, "torch": torch, "FQ_GROUPED_MAX_ROWS": 8, "ext": None, "__name__": "exllamav3.modules.block_sparse_mlp"}
    exec(compile(ast.fix_missing_locations(mod), "block_sparse_mlp.py", "exec"), ns)
    return ns


def e8m0_ggml(e):
    """ggml_cuda_e8m0_to_fp32 (common.cuh:830): bf16 raw e8m0 = 2^(e-127); e == 0 -> 2^-127 (bits 0x00400000)."""
    bits = np.where(e == 0, np.uint32(0x00400000), (e.astype(np.uint32) << 23)).astype(np.uint32)
    return bits.view(np.float32)


def dequant_np(raw, nb):
    """raw uint8 [N, nb*17] -> float32 [N, nb*32] with the ggml convention: kvalues * (e8m0 * 0.5)."""
    b = raw.reshape(raw.shape[0], nb, 17)
    d = (e8m0_ggml(b[..., 0]) * np.float32(0.5))[..., None]
    qs = b[..., 1:]
    v = np.concatenate([KV[qs & 15], KV[qs >> 4]], axis=-1).astype(np.float32)
    return (d * v).reshape(raw.shape[0], nb * 32)


def quant_q8_1(x):
    """per-32 q8_1: d = amax / 127 (fp16), q = round(x / d); returns (d fp32 [nb], q int8 [nb, 32])."""
    xb = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(xb).max(1)
    d = (amax / 127.0).astype(np.float16).astype(np.float32)
    dd = np.where(d == 0, 1.0, d)
    q = np.clip(np.rint(xb / dd[:, None]), -127, 127).astype(np.int8)
    return d, q


def vec_dot_np(row_bytes, nb, d8, q8):
    """The kernel's work split: 4 warps x 32 lanes = 128 threads, blocks_per_iter = vdr(2) * 128 / qi(4) = 64, kbx = tid / 2 (+ 64 per iter),
    iqs = vdr * (tid % 2); each call = vec_dot_mxfp4_q8_1 (2 ints x 2 dp4a). Returns per-thread partial sums (128,) -> sum = row result."""
    b = row_bytes.reshape(nb, 17)
    part = np.zeros(128, dtype=np.float64)
    for tid in range(128):
        for kbx in range(tid // 2, nb, 64):
            iqs = 2 * (tid % 2)
            e, qs = b[kbx, 0], b[kbx, 1:]
            sumi = 0
            for l in range(2):
                four = qs[4 * (iqs + l): 4 * (iqs + l) + 4]
                vx, vy = KV[four & 15], KV[four >> 4]          # get_int_from_table_16: v.x = low nibbles, v.y = high nibbles
                sumi += int((vx * q8[kbx, 4 * (iqs + l): 4 * (iqs + l) + 4]).sum())
                sumi += int((vy * q8[kbx, 16 + 4 * (iqs + l): 16 + 4 * (iqs + l) + 4]).sum())
            part[tid] += float(e8m0_ggml(np.array([e]))[0] * np.float32(0.5) * np.float32(d8[kbx])) * sumi
    return part


def rnd16(v):
    return np.float32(np.float16(v))


def swiglu_oai(g, u, lim):
    g = np.minimum(g, lim)
    u = np.minimum(np.maximum(u, -lim), lim)
    return (u + 1.0) * g * (1.0 / (1.0 + np.exp(-1.702 * g)))


def test_e8m0_convention_matches_engine_dequant():
    GF = _gap_formats()
    rng = np.random.default_rng(0)
    nb, N = 8, 6
    raw = rng.integers(0, 256, (N, nb, 17), dtype=np.uint8)
    raw[..., 0] = rng.integers(112, 136, (N, nb), dtype=np.uint8)
    raw[0, 0, 0] = 0
    raw[0, 1, 0] = 1                                    # denormal corner (fp16 flushes these to 0 on both sides)
    eng = GF.dequant("mxfp4", torch.from_numpy(raw.reshape(N, -1))).float().numpy()
    ref = dequant_np(raw.reshape(N, -1), nb)
    assert np.array_equal(eng, ref.astype(np.float16).astype(np.float32))
    allb = np.arange(0, 255, dtype=np.uint8)            # full e8m0 range: ggml value == 2^(e-128) (what Triton / engine use), in fp32
    assert np.array_equal(e8m0_ggml(allb) * np.float32(0.5), np.exp2(allb.astype(np.float64) - 128.0).astype(np.float32))


@pytest.mark.parametrize("K", [2944, 2880])           # 92 blocks (gate/up shape), 90 blocks (down shape)
def test_vec_dot_emulation(K):
    rng = np.random.default_rng(K)
    nb = K // 32
    W = rng.integers(0, 256, (3, nb * 17), dtype=np.uint8).reshape(3, nb, 17)
    W[..., 0] = rng.integers(118, 127, (3, nb), dtype=np.uint8)
    W = W.reshape(3, -1)
    x = (rng.standard_normal(K) * 0.5).astype(np.float16).astype(np.float32)
    d8, q8 = quant_q8_1(x)
    wd = dequant_np(W, nb)
    exact = wd.astype(np.float64) @ x.astype(np.float64)
    for r in range(3):
        got = vec_dot_np(W[r], nb, d8, q8).sum()
        rel = abs(got - exact[r]) / (np.linalg.norm(wd[r]) * np.linalg.norm(x) + 1e-30)
        assert rel < 1e-2, (K, r, got, exact[r], rel)
    cover = np.zeros(nb)
    for tid in range(128):
        for kbx in range(tid // 2, nb, 64):
            cover[kbx] += 1
    assert (cover == 2).all(), "every block must be taken by exactly 2 lanes (iqs 0 and 2)"


def test_epilogue_swiglu_oai():
    rng = np.random.default_rng(1)
    lim = 7.0
    ag, au = rng.standard_normal(4096) * 6, rng.standard_normal(4096) * 6
    bg, bu = (rng.standard_normal(4096) * 0.1).astype(np.float16), (rng.standard_normal(4096) * 0.1).astype(np.float16)
    g = rnd16(ag + bg.astype(np.float32))
    u = rnd16(au + bu.astype(np.float32))
    a_kernel = np.float16(swiglu_oai(g, u, lim))
    gg, uu = np.minimum(g, lim), np.clip(u, -lim, lim)
    a_ref = np.float16((uu + 1.0) * (gg * (1.0 / (1.0 + np.exp(-1.702 * gg)))))
    assert np.abs(a_kernel.astype(np.float32) - a_ref.astype(np.float32)).max() <= 2 ** -9 * np.abs(a_ref.astype(np.float32)).max()


def test_flag_read_at_call_time(monkeypatch):
    MX = _mx()
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    for v, want in (("1", True), ("0", False), (None, False)):
        monkeypatch.delenv("EXL3_FQ_MMVQ_MXFP4", raising=False)
        if v is not None:
            monkeypatch.setenv("EXL3_FQ_MMVQ_MXFP4", v)
        assert MX.mmvq_mxfp4() is want
    monkeypatch.setenv("EXL3_FQ_MMVQ_MXFP4", "1")
    monkeypatch.setattr(torch.version, "hip", "6.1")
    assert MX.mmvq_mxfp4() is False                      # off on HIP


def test_dispatch_with_stubs(monkeypatch):
    MX = _mx()
    ns = _mlp_methods()
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    rec = []

    class Ext:
        def fq_q8_scratch(self, dev, M, K):
            rec.append(("scratch", M, K))
            return torch.empty(M, K // 32 * 36, dtype=torch.uint8)
        def fq_quant_q8_1_rows(self, x, qb): rec.append(("quant", tuple(x.shape)))
        def fq_mxfp4_moe_gu(self, qb, sel, gw, gb, uw, ub, a, I, K, lim): rec.append(("gu", I, K, lim, tuple(a.shape), sel.dtype))
        def fq_mxfp4_moe_down(self, qa, sel, dw, db, d, N, K): rec.append(("down", N, K, tuple(d.shape), db.numel()))

    ns["ext"] = Ext()

    def stub_gemv(*a, **k):
        rec.append(("tri_gemv", a[4], a[5]))
        return torch.zeros(a[3].numel(), a[4])
    MX.moe_gemv = stub_gemv
    # the method does `from ..foreign_quant import triton_moe_mxfp4 as MX`: resolve it to this module
    pkg = types.ModuleType("exllamav3.modules"); pkg.__path__ = []
    fq = types.ModuleType("exllamav3.foreign_quant"); fq.triton_moe_mxfp4 = MX; fq.__path__ = []
    root = types.ModuleType("exllamav3"); root.__path__ = []
    for k, v in (("exllamav3", root), ("exllamav3.modules", pkg), ("exllamav3.foreign_quant", fq),
                 ("exllamav3.foreign_quant.triton_moe_mxfp4", MX)):
        monkeypatch.setitem(sys.modules, k, v)
    ns["__package__"] = "exllamav3.modules"

    def mk(H=2880, K=2944, I=2880, bias=True, act="swiglu_oai", R=1):
        def mod(n, k):
            t = torch.zeros(8, n * (k // 32) * 17, dtype=torch.uint8)
            return types.SimpleNamespace(inner=types.SimpleNamespace(
                fl=types.SimpleNamespace(packed=t[0], in_features=k, out_features=n, format="mxfp4"),
                bias=torch.zeros(n, dtype=torch.float16) if bias else None))
        s = types.SimpleNamespace(num_experts_per_tok=4, activation_fn=act, act_limit=7.0, shared_experts=None,
                                  gates=[mod(I, K)] * 4, ups=[mod(I, K)] * 4, downs=[mod(2880, I)] * 4)
        s._mxfp4_grouped_ok = lambda y: True
        s._mxfp4_mmvq_forward = types.MethodType(ns["_mxfp4_mmvq_forward"], s)
        s.activation_fn_call = lambda g, u, a, lim: rec.append(("act_call",))
        y = torch.zeros(R, H, dtype=torch.float16)
        return s, y, torch.zeros(R, 4, dtype=torch.int64), torch.full((R, 4), 0.25)

    def run(**kw):
        rec.clear()
        s, y, sel, rw = mk(**kw)
        out = ns["_forward_mxfp4_grouped"](s, y, y, sel, rw, {}, None, None)
        return [r[0] for r in rec], out

    monkeypatch.setenv("EXL3_FQ_MMVQ_MXFP4", "0")
    names, _ = run()
    assert names == ["tri_gemv", "tri_gemv", "act_call", "tri_gemv"], names
    monkeypatch.setenv("EXL3_FQ_MMVQ_MXFP4", "1")
    fires = MX.MMVQ_FIRES
    names, out = run()
    assert names == ["scratch", "quant", "gu", "scratch", "quant", "down"], names
    assert MX.MMVQ_FIRES == fires + 1
    assert out.shape[-1] == 2880, out.shape
    g = [r for r in rec if r[0] == "gu"][0]
    d = [r for r in rec if r[0] == "down"][0]
    assert g[1:6] == (2880, 2944, 7.0, (4, 2880), torch.int32) and d[1:4] == (2880, 2880, (4, 2880)) and d[4] == 4
    assert [r for r in rec if r[0] == "quant"] == [("quant", (1, 2944)), ("quant", (4, 2880))]
    for kw in (dict(R=2), dict(act="silu"), dict(bias=False)):        # outside the branch conditions -> today's code
        names, _ = run(**kw)
        assert not any(n in ("gu", "down") for n in names) and any(n.startswith("tri") for n in names), (kw, names)
