"""CPU checks for EXL3_FQ_MMVQ_EPI: bias in the fq_mmvq epilogue (A1) + one q8_1 quantization for q / k / v (A2).
No GPU, no kernel launch: the extension is replaced by stubs. Needs the repo importable (exllamav3 package, built extension present so the
import does not start a build); skipped otherwise. Run: CUDA_VISIBLE_DEVICES= python -m pytest tests/test_fq_mmvq_epi_cpu.py -q"""
import glob
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not glob.glob(os.path.join(ROOT, "exllamav3_ext*.so")):
    pytest.skip("extension not built in " + ROOT + " (importing exllamav3 would start the build)", allow_module_level=True)
sys.path.insert(0, ROOT)
import torch



def epi_half(v, b):                 # kernel: h = half(v); out = half(float(h) + float(b))
    h = v.astype(np.float16)
    return (h.astype(np.float32) + b.astype(np.float32)).astype(np.float16)

def epi_float(v, b):
    return v.astype(np.float32) + b.astype(np.float32)


def test_bias_rounding():
    import torch
    rng = np.random.default_rng(1)
    v = (rng.standard_normal(4096) * 3).astype(np.float32)
    b = rng.standard_normal(4096).astype(np.float16)
    t = (torch.from_numpy(v).half() + torch.from_numpy(b)).numpy()           # torch: y(half) + bias(half)
    assert np.array_equal(epi_half(v, b), t), "half epilogue != torch half add"
    tf = (torch.from_numpy(v) + torch.from_numpy(b)).numpy()                 # float y + half bias
    assert np.array_equal(epi_float(v, b), tf), "float epilogue != torch float add"
    # the wrong order (add in fp32 then round once) differs on some elements: the test is not vacuous
    wrong = (v + b.astype(np.float32)).astype(np.float16)
    ndiff = int((wrong != t).sum())
    print(f"bias epilogue: half order == torch (4096/4096), float == torch; single-rounding order would differ on {ndiff} elements")
    assert ndiff > 0


def test_q8_reuse_numeric():
    import torch
    # q8_1 quantize emulation (ggml quantize_q8_1: d = amax / 127, q = round(x / d), s = d * sum q): deterministic, same input -> same bytes
    def q8(x):
        x = x.reshape(-1, 32).astype(np.float32)
        amax = np.abs(x).max(1, keepdims=True); d = amax / 127.0
        inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0)
        q = np.rint(x * inv).astype(np.int8)
        return d.astype(np.float16), (d * q.sum(1, keepdims=True)).astype(np.float16), q
    rng = np.random.default_rng(2)
    x = rng.standard_normal(2880).astype(np.float16)
    a = q8(x); b = q8(x.copy()); c = q8(x)
    for i in range(3):
        assert np.array_equal(a[i], b[i]) and np.array_equal(a[i], c[i])
    W = [rng.standard_normal((16, 2880)).astype(np.float32) for _ in range(3)]
    dq = lambda t: (t[2].astype(np.float32) * t[0].astype(np.float32)).reshape(-1)
    y1 = [w @ dq(a) for w in W]; y2 = [w @ dq(q8(x)) for w in W]            # 3 quantizations vs 1 shared
    assert all(np.array_equal(p, q) for p, q in zip(y1, y2))
    print("q8_1 reuse: quantize twice == once, the three gemv inputs bit-equal")


class _FakeCudaBias:
    """Stand-in for a CUDA half bias on a CPU box."""
    is_cuda = True
    dtype = torch.half
    def __init__(self, n): self.n = n
    def is_contiguous(self): return True
    def numel(self): return self.n


@pytest.fixture(params=["0", "1"])
def flag(request, monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return request.param == "1"

def test_dispatch(flag, monkeypatch):
    from exllamav3.modules import foreign_linear as fl
    from exllamav3.modules import linear as lin
    monkeypatch.setenv("EXL3_FQ_MMVQ_EPI", "1" if flag else "0")
    log = []
    qcount = [0]
    ext = types.SimpleNamespace(
        fq_q8_scratch=lambda dev, R, K: torch.zeros(R, K // 32 * 36, dtype=torch.uint8),
        fq_quant_q8_1_rows=lambda x, qb: qcount.__setitem__(0, qcount[0] + 1),
        fq_mmvq_epi_abi=lambda: 1)
    ext.fq_mmvq_q5_0 = lambda qb, p, out, bias=None: (out.zero_(), log.append(("mmvq", bias is not None)))[1]
    monkeypatch.setattr(fl, "ext", ext)
    monkeypatch.setattr(fl, "_MMVQ_LEGACY", True)
    monkeypatch.setitem(fl._MMVQ_FN, "q5_0", "fq_mmvq_q5_0")
    monkeypatch.setitem(fl._MMVQ_QK, "q5_0", 32)
    FB = _FakeCudaBias(8)

    def mk():
        m = object.__new__(fl.ForeignLinear); m.format = "q5_0"; m.in_features = 64; m.out_features = 8; m.key = "t"
        m.packed = torch.zeros(8, 2 * 22, dtype=torch.uint8); m.scales = None; m.global_scale = None
        m.out_dtype = torch.half; m.post_scale = 1.0; m.softcap = 0.0; m.pre_scale = 1.0
        return m
    m = mk(); x = torch.zeros(1, 64, dtype=torch.half)
    # (A2) q / k / v of one layer: quantize count
    qcount[0] = 0
    fl.q8_share_begin()
    for _ in range(3): m.forward(x, {})
    fl.q8_share_end()
    assert qcount[0] == (1 if flag else 3), ("shared q8 quantizations", qcount[0])
    qcount[0] = 0
    for _ in range(3): m.forward(x, {})                     # outside the bracket: always one quantization per call
    assert qcount[0] == 3, qcount[0]
    qcount[0] = 0
    fl.q8_share_begin(); m.forward(x, {}); x2 = torch.zeros(1, 64, dtype=torch.half); m.forward(x2, {})   # other tensor: re-quantize
    x.add_(1); m.forward(x, {}); fl.q8_share_end()          # in-place write bumps the version: re-quantize
    assert qcount[0] == 3, qcount[0]
    # (A1) bias: kernel gets it only with the flag, R == 1 and a CUDA half contiguous bias
    log.clear(); m.forward(x, {}, None, bias=FB)
    assert log == [("mmvq", flag)], log
    assert m._bias_fused == flag
    log.clear(); m.forward(torch.zeros(1, 64, dtype=torch.half), {}, None, bias=torch.zeros(8, dtype=torch.half))   # CPU tensor bias: never fused
    assert log == [("mmvq", False)] and not m._bias_fused, log
    inner = lin._ForeignInner(m, torch.ones(8, dtype=torch.half))
    y = inner.forward(torch.zeros(1, 64, dtype=torch.half), {})
    assert torch.equal(y, torch.ones(1, 8, dtype=torch.half)), "bias must be added exactly once (CPU bias is never fused)"
