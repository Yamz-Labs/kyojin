"""Routed (MoE) HIP MMQ (ext.fq_mmq_moe, rocm/gmmq: llama.cpp MMQ via Gufo) vs a torch fp32 reference.

Random raw GGUF blocks (finite fp16 scales) for Q4_K / Q5_K (gate/up shape: K = 2560, M = 640) and Q5_1 / Q8_0 (down
shape: K = 640, M = 2560, K % 256 != 0 so the tail over-read path runs), each expert in its own allocation with a zeroed
tail pad (see rocm/gmmq/gmmq_api.h), random top-k ids. Reference: Kyojin's dequant (ext.fq_dequant_q4_k / q5_k,
foreign_quant.gap_formats.dequant for q5_1 / q8_0) then fp32 matmul. MMQ quantizes the activations to q8_1, so the
gate is max|err| / max|ref| < 2e-2 per case; the error against a q8_1-emulated reference is printed too.
Runs under pytest or as a plain script (ROCm only; skips elsewhere)."""
import os
import sys
import time

import torch

FMTS = {  # name: (ggml type id, block values, block bytes, K, M)
    "q4_k": (12, 256, 144, 2560, 640),
    "q5_k": (13, 256, 176, 2560, 640),
    "q5_1": (7, 32, 24, 640, 2560),
    "q8_0": (8, 32, 34, 640, 2560),
}
TOL = 2e-2
PAD = 1024   # zeroed bytes after each expert (>= the MMQ K-tail over-read)


def _ext():
    if not torch.cuda.is_available() or not torch.version.hip:
        return None
    try:
        import exllamav3_ext as ext
    except ImportError:
        return None
    return ext if hasattr(ext, "fq_mmq_moe") else None


def _f16_bytes(t):
    return t.to(torch.float16).contiguous().view(torch.uint8).reshape(*t.shape, 2)


def make_raw(fmt, E, M, K, g):
    """uint8 [E, M, nb * bb] of random blocks with finite fp16 scale fields."""
    _, qk, bb, _, _ = FMTS[fmt]
    nb = K // qk
    raw = torch.randint(0, 256, (E, M, nb, bb), dtype=torch.uint8, generator=g)
    sh = (E, M, nb)
    d = torch.empty(sh).uniform_(0.002, 0.01, generator=g)
    raw[..., 0:2] = _f16_bytes(d)
    if fmt in ("q4_k", "q5_k"):
        raw[..., 2:4] = _f16_bytes(torch.empty(sh).uniform_(0.0, 0.01, generator=g))     # dmin
    elif fmt == "q5_1":
        raw[..., 2:4] = _f16_bytes(torch.empty(sh).uniform_(-0.05, 0.05, generator=g))   # m
    return raw.reshape(E, M, nb * bb)


def dequant(ext, fmt, raw_e):
    """raw_e uint8 [M, nb * bb] (device) -> fp32 [M, K] with Kyojin's dequant helpers."""
    _, qk, bb, _, _ = FMTS[fmt]
    M, w = raw_e.shape
    if fmt in ("q4_k", "q5_k"):
        out = torch.empty(M, w // bb * qk, dtype=torch.half, device=raw_e.device)
        (ext.fq_dequant_q4_k if fmt == "q4_k" else ext.fq_dequant_q5_k)(raw_e.contiguous(), out)
        return out.float()
    from exllamav3.foreign_quant.gap_formats import dequant as gap_dequant
    return gap_dequant(fmt, raw_e.contiguous()).float()


def q8_1_emulate(x):
    """Per-32 symmetric int8 round trip of the activations (what quantize_mmq_q8_1 feeds the integer dots)."""
    T, K = x.shape
    xb = x.reshape(T, K // 32, 32)
    d = xb.abs().amax(-1, keepdim=True) / 127.0
    q = torch.where(d > 0, torch.round(xb / torch.where(d > 0, d, torch.ones_like(d))), torch.zeros_like(xb))
    return (q * d).reshape(T, K)


def run_case(ext, fmt, E, T, n_used, x_dtype, seed, timing=False):
    tid, qk, bb, K, M = FMTS[fmt]
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    raw = make_raw(fmt, E, M, K, g)
    bufs, views = [], []
    for e in range(E):   # one allocation per expert (Kyojin stores each expert as its own tensor)
        n = raw[e].numel()
        b = torch.zeros(n + PAD, dtype=torch.uint8, device=dev)
        b[:n] = raw[e].reshape(-1).to(dev)
        bufs.append(b)
        views.append(b[:n].view(M, -1))
    w_ptrs = torch.tensor([v.data_ptr() for v in views], dtype=torch.int64, device=dev)
    x32 = torch.randn(T, K, generator=g)
    ids = torch.stack([torch.randperm(E, generator=g)[:n_used] for _ in range(T)]).to(torch.int32)
    x = x32.to(x_dtype).to(dev)
    xr = x.float()
    out = torch.full((T * n_used, M), float("nan"), dtype=torch.float, device=dev)
    ext.fq_mmq_moe(x, ids.to(dev).contiguous(), w_ptrs, tid, M, out)
    torch.cuda.synchronize()
    ref = torch.zeros(T * n_used, M, dtype=torch.float, device=dev)
    refq = torch.zeros_like(ref)
    xq = q8_1_emulate(xr)
    flat = ids.reshape(-1)
    for e in flat.unique().tolist():
        rows = (flat == e).nonzero().reshape(-1).to(dev)          # r = t * n_used + slot
        W = dequant(ext, fmt, views[e])
        tok = rows // n_used
        ref[rows] = xr[tok] @ W.T
        refq[rows] = xq[tok] @ W.T
    scale = ref.abs().max().item()
    err = (out - ref).abs()
    errq = (out - refq).abs().max().item()
    finite = bool(torch.isfinite(out).all().item())
    rel = err.max().item() / scale
    ok = finite and rel < TOL
    msg = (f"{fmt:5s} E={E:3d} T={T:4d} n_used={n_used:2d} x={str(x_dtype).split('.')[-1]:7s} K={K} M={M}: "
           f"max_abs={err.max().item():.4e} max_rel={rel:.3e} mean_rel={err.mean().item() / scale:.3e} "
           f"vs_q8emu_rel={errq / scale:.3e} finite={finite} -> {'PASS' if ok else 'FAIL'}")
    if timing:
        for _ in range(2):
            ext.fq_mmq_moe(x, ids.to(dev).contiguous(), w_ptrs, tid, M, out)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            ext.fq_mmq_moe(x, ids.to(dev).contiguous(), w_ptrs, tid, M, out)
        torch.cuda.synchronize()
        msg += f"  [{(time.perf_counter() - t0) / 10 * 1e3:.3f} ms/call]"
    print(msg, flush=True)
    return ok


CASES = [  # (fmt, E, T, n_used, x dtype, timing)
    *[(f, 64, T, 10, torch.float, False) for f in FMTS for T in (1, 37, 300)],
    *[(f, 64, 129, 8, torch.half, False) for f in FMTS],
    *[(f, 512, 512, 10, torch.float, True) for f in FMTS],
]


def run_all():
    ext = _ext()
    if ext is None:
        print("SKIP: needs a ROCm torch and an exllamav3_ext with fq_mmq_moe")
        return None
    fails = 0
    for i, (fmt, E, T, n_used, dt, timing) in enumerate(CASES):
        fails += not run_case(ext, fmt, E, T, n_used, dt, seed=100 + i, timing=timing)
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {len(CASES) - fails}/{len(CASES)} cases")
    return fails == 0


def test_hip_gmmq():
    import pytest
    res = run_all()
    if res is None:
        pytest.skip("ROCm-only op (fq_mmq_moe) not available")
    assert res


if __name__ == "__main__":
    res = run_all()
    sys.exit(0 if res in (True, None) else 1)
