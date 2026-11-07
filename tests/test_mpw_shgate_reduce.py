"""EXL3_SHGATE_REDUCE: the sigmoid-gated shared expert summed in the grouped MoE prefill reduce epilogue.

exl3_moe_prefill_wmma(..., addend = y, addend_gate = s) must be bit-identical to the unfused sequence the
Qwen3.8 MoE block ran before: exl3_moe_prefill_wmma(..., no addend) then gr_pf.shgate_add(out, y, s)
(out += float(y) * s, unfused fp32 multiply + add). Synthetic weights (packed random trellis states), no model.
Also checks the ungated addend path (GLM, EXL3_NC2_SHARED_FUSE) is unchanged, addend rows shorter than the
output (aligned-layout dummy rows), and the argument checks.

Run (ROCm, gfx115x/gfx12): PYTHONPATH=<repo> python -m pytest -q tests/test_mpw_shgate_reduce.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.environ.get("EXL3_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hardware import require  # noqa: E402
require("ext")
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

DEV = "cuda:0"
H, I, TOPK, EXPERTS = 2560, 640, 10, 24      # Qwen3.8-Flash-Next expert shapes (N 640: the tail-block path)


def _require():
    if not (torch.version.hip and torch.cuda.is_available()):
        pytest.skip("ROCm build / device not available")
    if not hasattr(ext, "EXL3_MPW_ADDEND_GATE"):
        pytest.skip("this build has no addend_gate")
    if not ext.exl3_gemv_supported(torch.cuda.current_device()):
        pytest.skip("exl3 kernels not supported on this device")


def _ptrs(tensors):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.long, device=DEV)


def _trellis(k, n, K, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    states = torch.randint(0, 1 << K, (k // 16, n // 16, 256), generator=gen, device=DEV, dtype=torch.int16)
    packed = torch.zeros((k // 16, n // 16, 16 * K), dtype=torch.int16, device=DEV)
    ext.pack_trellis(packed, states, K)
    g = torch.Generator(device=DEV).manual_seed(seed * 7 + 1)
    suh = (torch.randint(0, 2, (k,), generator=g, device=DEV) * 2 - 1).half()
    svh = (torch.randint(0, 2, (n,), generator=g, device=DEV) * 2 - 1).half()
    return packed, suh, svh


def _weights(K, seed):
    proj = {"gate": [_trellis(H, I, K, seed + 1 + 97 * e) for e in range(EXPERTS)],
            "up": [_trellis(H, I, K, seed + 2 + 97 * e) for e in range(EXPERTS)],
            "down": [_trellis(I, H, K, seed + 3 + 97 * e) for e in range(EXPERTS)]}
    tabs = []
    for p in ("gate", "up", "down"):
        tabs += [_ptrs([t for t, _, _ in proj[p]]), _ptrs([s for _, s, _ in proj[p]]), _ptrs([v for _, _, v in proj[p]])]
    return proj, tabs


def _run(x, sel, w, tabs, K, addend=None, gate=None):
    rows, topk = sel.shape
    A = rows * topk
    order = sel.reshape(-1).argsort(stable=True)
    count = torch.zeros(EXPERTS + 1, dtype=torch.long, device=DEV)
    count.scatter_add_(0, sel.reshape(-1), torch.ones(A, dtype=torch.long, device=DEV))
    gu_had = torch.empty((2 * A * H + 128 * H,), dtype=torch.half, device=DEV)
    gu_out = torch.empty((2 * A + 128, I), dtype=torch.half, device=DEV)
    out = torch.empty((rows, H), dtype=torch.float, device=DEV)
    kw = {}
    if addend is not None: kw["addend"] = addend
    if gate is not None: kw["addend_gate"] = gate
    ext.exl3_moe_prefill_wmma(
        x, out, sel, w, order, count, *tabs, float(K), float(K),
        gu_had[:2 * A * H].view(2 * A, H), gu_out, gu_had[:2 * A * H].view(torch.float).view(A, H),
        torch.empty(EXPERTS + 1, dtype=torch.long, device=DEV), torch.empty(A, dtype=torch.long, device=DEV),
        torch.empty(A // 64 + EXPERTS + 1, dtype=torch.int32, device=DEV), torch.empty(1, dtype=torch.int32, device=DEV),
        0.0, False, **kw)
    torch.cuda.synchronize()
    return out


def _inputs(rows, seed, add_rows=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(rows, H, generator=g, device=DEV) * 0.35).half()
    sel = torch.stack([torch.randperm(EXPERTS, generator=g, device=DEV)[:TOPK] for _ in range(rows)]).to(torch.long)
    w = torch.rand(rows, TOPK, generator=g, device=DEV)
    w = (w / w.sum(dim=-1, keepdim=True)).contiguous()
    ar = rows if add_rows is None else add_rows
    y = (torch.randn(ar, H, generator=g, device=DEV) * 3.0).half()
    z = torch.randn(ar, 1, generator=g, device=DEV) * 2.0
    return x, sel, w, y, torch.sigmoid(z).view(-1).contiguous()


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("rows", [40, 333, 1024])
def test_gated_reduce_bit_identical(K, rows):
    _require()
    from exllamav3.modules.gated_delta_net_fn.gr_pf import shgate_add
    _, tabs = _weights(K, 500 + K)
    x, sel, w, y, s = _inputs(rows, rows * 3 + K)
    ref = _run(x, sel, w, tabs, K)
    shgate_add(ref, y, s)
    torch.cuda.synchronize()
    got = _run(x, sel, w, tabs, K, addend=y, gate=s)
    assert torch.isfinite(ref).all()
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), \
        f"K={K} rows={rows}: max |d| {float((got - ref).abs().max()):.3e}"
    # torch form of the same arithmetic (what add_sigmoid_gate's fallback computes)
    ref2 = _run(x, sel, w, tabs, K)
    ref2.add_(y.float() * s.view(-1, 1))
    assert torch.equal(got.view(torch.int32), ref2.view(torch.int32))


def test_gated_reduce_short_addend_rows():
    """Aligned layout: dummy rows past the addend rows get no shared term (gate not read there)."""
    _require()
    from exllamav3.modules.gated_delta_net_fn.gr_pf import shgate_add
    K, rows, real = 4, 300, 291
    _, tabs = _weights(K, 900)
    x, sel, w, y, s = _inputs(rows, 77, add_rows=real)
    ref = _run(x, sel, w, tabs, K)
    shgate_add(ref[:real], y, s)
    got = _run(x, sel, w, tabs, K, addend=y, gate=s)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))


def test_ungated_addend_unchanged():
    """GLM path (EXL3_NC2_SHARED_FUSE): addend without gate = out + float(y), as before."""
    _require()
    K, rows = 2, 200
    _, tabs = _weights(K, 1300)
    x, sel, w, y, _ = _inputs(rows, 5)
    ref = _run(x, sel, w, tabs, K)
    ref += y.float()
    got = _run(x, sel, w, tabs, K, addend=y)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))


def test_gate_argument_checks():
    _require()
    K, rows = 2, 64
    _, tabs = _weights(K, 1700)
    x, sel, w, y, s = _inputs(rows, 9)
    with pytest.raises(RuntimeError):
        _run(x, sel, w, tabs, K, gate=s)                        # gate without addend
    with pytest.raises(RuntimeError):
        _run(x, sel, w, tabs, K, addend=y, gate=s[:-1])         # wrong length
    with pytest.raises(RuntimeError):
        _run(x, sel, w, tabs, K, addend=y, gate=s.half())       # wrong dtype
