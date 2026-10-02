"""GLM-5.3-Flash gates on the fused EXL3 decode, synthetic weights, no model.

Gate 1: clamped SwiGLU (act_limit / GLM swiglu_limit = 10.0) in exl3_dec_moe / exl3_dec_moe_union.
Gate 2: the mcg codebook in those same kernels (they were mul1-only).

Synthetic weights: random EXL3 trellis *states* packed with ext.pack_trellis. The quantizer
(ext.quantize_tiles) is CUDA-only in this fork, but packing random states
produces exactly the same class of tensors the loader hands the kernels -- every state is a valid
codebook index for both the mul1 and the mcg codebook, and the reference reconstruct() decodes
them the same way it decodes a real pack.

Reference path = what the non-fused route runs: ext.reconstruct + ext.hgemm with the same suh/svh
hadamard stages, ext.silu_mul for the activation, per-expert, fp32 weighted combine. Fused path =
ext.exl3_dec_moe (batch 1) / ext.exl3_dec_moe_union (R rows). Both deterministic.

At H=512 the pre-activations are ~N(0, 45), so silu(gate) and |up| land far past a limit of 10: a
kernel that ignores act_limit disagrees with the reference by O(1) relative. The tests also run
act_limit = 0 as the control, where the two paths must agree to fp16 noise.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.environ.get("EXL3_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

HIDDEN = 512          # % 512 == 0 (exl3_dec_moe requirement)
INTERMEDIATE = 512    # % 512 == 0
NUM_EXPERTS = 16
TOP_K = 8
ACT_LIMIT = 10.0      # GLM-5.3-Flash swiglu_limit
DEV = "cuda:0"
# pybind11 renders the annotated signature in __doc__, so this is the capability probe
_MCG_READY = "mcg" in (getattr(ext.exl3_dec_moe, "__doc__", "") or "")


def _require_dec(mcg: bool):
    if not (torch.version.hip and torch.cuda.is_available()):
        pytest.skip("ROCm build / device not available")
    for name in ("exl3_dec_moe", "exl3_dec_moe_union", "reconstruct", "hgemm", "had_r_128",
                 "pack_trellis", "silu_mul"):
        assert hasattr(ext, name), f"missing ext.{name}"
    if not ext.exl3_gemv_supported(torch.cuda.current_device()):
        pytest.skip("exl3_dec kernels not supported on this device")
    if mcg and not _MCG_READY:
        pytest.skip("this build's exl3_dec_moe has no mcg codebook support")


def _ptrs(tensors):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.long, device=DEV)


def _projection(k: int, n: int, K: int, seed: int, experts: int = NUM_EXPERTS):
    """(trellis per expert, suh per expert, svh per expert) with random trellis states."""
    kb = 16 * K                       # packed int16 per 16x16 tile
    trellises, suhs, svhs = [], [], []
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for e in range(experts):
        states = torch.randint(0, 1 << K, (k // 16, n // 16, 256), generator=gen,
                               device=DEV, dtype=torch.int16)
        packed = torch.zeros((k // 16, n // 16, kb), dtype=torch.int16, device=DEV)
        ext.pack_trellis(packed, states, K)
        trellises.append(packed)
        g = torch.Generator(device=DEV).manual_seed(seed * 1000 + e)
        suhs.append((torch.randint(0, 2, (k,), generator=g, device=DEV) * 2 - 1).half())
        svhs.append((torch.randint(0, 2, (n,), generator=g, device=DEV) * 2 - 1).half())
    return trellises, suhs, svhs


def _ref_linear(x, trellis, suh, svh, K, mcg=False):
    """reconstruct + hadamard + hgemm, the reference per-expert linear."""
    k = suh.numel()
    n = svh.numel()
    rows = x.shape[0]
    xh = torch.empty((rows, k), dtype=torch.half, device=DEV)
    y = torch.empty((rows, n), dtype=torch.half, device=DEV)
    yh = torch.empty((rows, n), dtype=torch.half, device=DEV)
    w = torch.empty((k, n), dtype=torch.half, device=DEV)
    ext.had_r_128(x.contiguous(), xh, suh, None, 1.0)
    ext.reconstruct(w, trellis, K, mcg, not mcg)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, yh, None, svh, 1.0)
    return yh


def _ref_moe(x, proj, K, selected, weights, act_limit, mcg=False):
    """Per-row reference: reconstruct+matmul per selected expert, ext.silu_mul, fp32 combine."""
    gt, gs, gv = proj["gate"]
    ut, us, uv = proj["up"]
    dt, ds, dv = proj["down"]
    rows = x.shape[0]
    out = torch.zeros((rows, HIDDEN), dtype=torch.float, device=DEV)
    a = torch.empty((rows, INTERMEDIATE), dtype=torch.half, device=DEV)
    for r in range(rows):
        xr = x[r:r + 1].half()
        for k in range(TOP_K):
            e = int(selected[r, k])
            g = _ref_linear(xr, gt[e], gs[e], gv[e], K, mcg)
            u = _ref_linear(xr, ut[e], us[e], uv[e], K, mcg)
            ext.silu_mul(g, u, a, act_limit)
            d = _ref_linear(a, dt[e], ds[e], dv[e], K, mcg)
            out[r] += float(weights[r, k]) * d[0].float()
    return out


def _synthetic(seed: int, K: int):
    torch.manual_seed(seed)
    gate = _projection(HIDDEN, INTERMEDIATE, K, seed + 11)
    up = _projection(HIDDEN, INTERMEDIATE, K, seed + 23)
    down = _projection(INTERMEDIATE, HIDDEN, K, seed + 37)
    return {"gate": gate, "up": up, "down": down}


def _inputs(rows: int, seed: int, scale: float = 1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=g, device=DEV) * scale).half()
    sel = torch.stack([torch.randperm(NUM_EXPERTS, generator=g, device=DEV)[:TOP_K]
                       for _ in range(rows)]).to(torch.long)
    w = torch.rand(rows, TOP_K, generator=g, device=DEV)
    w = (w / w.sum(dim=-1, keepdim=True)).half()
    return x, sel, w


# Unclamped runs need a smaller input: the reference stores the down projection in fp16
# (ext.hgemm), and with random mcg weights (larger than mul1's) silu(g)*u at full scale
# overflows fp16 there -- the reference goes inf/nan while the fused kernel stays finite in fp32
# (measured: ref 111 nan / 277 inf, fused max 2.4e4). The act_limit > 0 runs keep full scale so
# the clamp is actually exercised.
OPEN_SCALE = 0.35


def _workspace(rows: int):
    scratch = torch.zeros(rows * TOP_K * 16 * max(2 * INTERMEDIATE, HIDDEN),
                          dtype=torch.float, device=DEV)
    counters = torch.zeros(4096, dtype=torch.int32, device=DEV)
    return scratch, counters


def _run_batch1(x, proj, K, sel, w, act_limit, mcg):
    scratch, counters = _workspace(1)
    out = torch.zeros((1, HIDDEN), dtype=torch.float, device=DEV)
    act = torch.empty((TOP_K, INTERMEDIATE), dtype=torch.half, device=DEV)
    gt, gs, gv = proj["gate"]
    ut, us, uv = proj["up"]
    dt, ds, dv = proj["down"]
    ext.exl3_dec_moe(
        x.contiguous().view(-1), out.view(-1), sel.contiguous(), w.contiguous(),
        _ptrs(gt), _ptrs(gs), _ptrs(gv),
        _ptrs(ut), _ptrs(us), _ptrs(uv),
        _ptrs(dt), _ptrs(ds), _ptrs(dv),
        act, scratch, counters, INTERMEDIATE, float(K), float(K), False, act_limit, mcg,
    )
    torch.cuda.synchronize()
    return out


def _run_union(x, proj, K, sel, w, act_limit, mcg):
    rows = x.shape[0]
    scratch, counters = _workspace(rows)
    out = torch.zeros((rows, HIDDEN), dtype=torch.float, device=DEV)
    act = torch.empty((rows * TOP_K, INTERMEDIATE), dtype=torch.half, device=DEV)
    down_part = torch.empty((rows * TOP_K, HIDDEN), dtype=torch.float, device=DEV)
    gt, gs, gv = proj["gate"]
    ut, us, uv = proj["up"]
    dt, ds, dv = proj["down"]
    ext.exl3_dec_moe_union(
        x.contiguous(), out, sel.contiguous(), w.contiguous(),
        _ptrs(gt), _ptrs(gs), _ptrs(gv),
        _ptrs(ut), _ptrs(us), _ptrs(uv),
        _ptrs(dt), _ptrs(ds), _ptrs(dv),
        act, down_part, scratch, counters, INTERMEDIATE, float(K), float(K), False, act_limit, mcg,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("mcg", [False, True], ids=["mul1", "mcg"])
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("act_limit", [0.0, ACT_LIMIT])
def test_batch1_decode_matches_reference(K, act_limit, mcg):
    _require_dec(mcg)
    proj = _synthetic(4100 + K + 100 * mcg, K)
    x, sel, w = _inputs(1, 900 + K, 1.0 if act_limit else OPEN_SCALE)
    ref = _ref_moe(x, proj, K, sel, w, act_limit, mcg)
    got = _run_batch1(x, proj, K, sel, w, act_limit, mcg)
    assert torch.isfinite(ref).all(), "reference overflowed -- lower OPEN_SCALE"
    assert torch.isfinite(got).all(), "fused kernel produced a non-finite value"
    scale = ref.abs().max().item()
    err = (got - ref).abs().max().item()
    print(f"\nbatch1 K={K} act_limit={act_limit} mcg={mcg} max|ref|={scale:.4f} "
          f"max|d|={err:.4e} rel={err / scale:.2e}")
    assert scale > 1.0, "degenerate reference (test would not discriminate)"
    assert err <= 2e-2 * scale, f"K={K} act_limit={act_limit} mcg={mcg}: {err / scale:.2e} rel"


@pytest.mark.parametrize("mcg", [False, True], ids=["mul1", "mcg"])
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("act_limit", [0.0, ACT_LIMIT])
def test_union_rows_match_reference(K, act_limit, mcg):
    _require_dec(mcg)
    proj = _synthetic(4300 + K + 100 * mcg, K)
    x, sel, w = _inputs(4, 920 + K, 1.0 if act_limit else OPEN_SCALE)
    ref = _ref_moe(x, proj, K, sel, w, act_limit, mcg)
    got = _run_union(x, proj, K, sel, w, act_limit, mcg)
    assert torch.isfinite(ref).all(), "reference overflowed -- lower OPEN_SCALE"
    assert torch.isfinite(got).all(), "fused kernel produced a non-finite value"
    scale = ref.abs().max().item()
    err = (got - ref).abs().max().item()
    print(f"\nunion K={K} act_limit={act_limit} mcg={mcg} max|ref|={scale:.4f} "
          f"max|d|={err:.4e} rel={err / scale:.2e}")
    assert err <= 2e-2 * scale, f"K={K} act_limit={act_limit} mcg={mcg}: {err / scale:.2e} rel"


@pytest.mark.parametrize("mcg", [False, True], ids=["mul1", "mcg"])
@pytest.mark.parametrize("K", [2, 3])
def test_decode_is_deterministic(K, mcg):
    _require_dec(mcg)
    proj = _synthetic(4200 + K + 100 * mcg, K)
    x, sel, w = _inputs(1, 910 + K)
    a = _run_batch1(x, proj, K, sel, w, ACT_LIMIT, mcg)
    b = _run_batch1(x, proj, K, sel, w, ACT_LIMIT, mcg)
    assert torch.equal(a, b)


@pytest.mark.parametrize("mcg", [False, True], ids=["mul1", "mcg"])
def test_act_limit_actually_clamps(mcg):
    """Sanity: with these inputs the clamp is active, so ignoring it cannot pass the tests."""
    _require_dec(mcg)
    K = 3
    proj = _synthetic(4400 + K + 100 * mcg, K)
    x, sel, w = _inputs(1, 930 + K)
    ref_clamped = _ref_moe(x, proj, K, sel, w, ACT_LIMIT, mcg)
    ref_open = _ref_moe(x, proj, K, sel, w, 0.0, mcg)
    rel = (ref_clamped - ref_open).abs().max().item() / ref_open.abs().max().item()
    print(f"\nclamp effect (mcg={mcg}): max|clamped-open|/max|open| = {rel:.3f}")
    assert rel > 0.05, "inputs do not exercise the clamp"


def test_mcg_is_not_the_mul1_codebook():
    """Sanity: the two codebooks decode the same states to different weights."""
    _require_dec(True)
    K = 3
    proj = _synthetic(4500 + K, K)
    gt, gs, gv = proj["gate"]
    w_mul1 = torch.empty((HIDDEN, INTERMEDIATE), dtype=torch.half, device=DEV)
    w_mcg = torch.empty((HIDDEN, INTERMEDIATE), dtype=torch.half, device=DEV)
    ext.reconstruct(w_mul1, gt[0], K, False, True)
    ext.reconstruct(w_mcg, gt[0], K, True, False)
    rel = (w_mul1 - w_mcg).abs().max().item() / w_mul1.abs().max().item()
    print(f"\ncodebook spread: max|mul1-mcg|/max|mul1| = {rel:.3f}")
    assert rel > 0.1, "mcg and mul1 decode identically -- wrong codebook flag somewhere"
