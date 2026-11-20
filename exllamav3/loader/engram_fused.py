"""HY4_ENGRAM_FUSED=1: the Engram block of a layer in TWO Triton launches around its one GEMM (old: ~35 torch launches per Engram layer: rms / pow / mean / rsqrt / abs / clamp /
sqrt / neg / compare / sigmoid chains, the dominant part of the decode step's torch glue).
  pre(rows, lookup_norm, valid)          r = rms(rows) * lookup_norm * valid                       [T, n_cols * hd]
  (torch)  kv = r @ kv_w.T
  post(kv, kv_b, x, gate, hc_scale, pre_gamma, valid)  x + delta                                    [T, hc, D]   (same maths as Hy4Engram.apply, M:5292-5353)
Shapes come from the tensors; the reduction order differs from torch (1e-7 relative)."""
import os
import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("HY4_ENGRAM_FUSED", "0") == "1"


@triton.jit
def _k_pre(rows_ptr, ln_ptr, valid_ptr, o_ptr, HD, eps, BH: tl.constexpr, NC: tl.constexpr):
    pid = tl.program_id(0)                                  # row t * NC + col
    d = tl.arange(0, BH)
    m = d < HD
    x = tl.load(rows_ptr + pid * HD + d, mask=m, other=0.0)
    rc = tl.rsqrt(tl.sum(x * x, axis=0) / HD + eps)
    w = tl.load(ln_ptr + (pid % NC) * HD + d, mask=m, other=0.0)
    v = tl.load(valid_ptr + pid)
    tl.store(o_ptr + pid * HD + d, x * rc * w * v, mask=m)


@triton.jit
def _k_post(kv_ptr, kvb_ptr, x_ptr, gate_ptr, hcs_ptr, pg_ptr, valid_ptr, o_ptr, D, eps, inv_sqrt_d,
            HC: tl.constexpr, NC: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, BD)
    m = d < D
    kvt = kv_ptr + t * (HC + 1) * D
    val = tl.load(kvt + HC * D + d, mask=m, other=0.0) + tl.load(kvb_ptr + HC * D + d, mask=m, other=0.0)
    sig = val * tl.rsqrt(tl.sum(val * val, axis=0) / D + eps) * tl.load(pg_ptr + d, mask=m, other=0.0)
    vs = 0.0
    for c in range(NC):
        vs += tl.load(valid_ptr + t * NC + c)
    anyv = tl.where(vs > 0, 1.0, 0.0)
    for h in range(HC):
        xh = tl.load(x_ptr + (t * HC + h) * D + d, mask=m, other=0.0)
        key = tl.load(kvt + h * D + d, mask=m, other=0.0) + tl.load(kvb_ptr + h * D + d, mask=m, other=0.0)
        rx = xh * tl.rsqrt(tl.sum(xh * xh, axis=0) / D + eps)
        rk = key * tl.rsqrt(tl.sum(key * key, axis=0) / D + eps)
        dot = tl.sum(rx * rk * tl.load(gate_ptr + h * D + d, mask=m, other=0.0), axis=0) * inv_sqrt_d
        mag = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
        sgn = tl.where(dot < 0, -1.0, 1.0)
        g = 1.0 / (1.0 + tl.exp(-(mag * sgn)))
        delta = sig * (g + tl.load(hcs_ptr + h * D + d, mask=m, other=0.0))
        tl.store(o_ptr + (t * HC + h) * D + d, xh + delta * anyv, mask=m)


def apply(e, x, rows, valid):
    """== Hy4Engram.apply for fp32 [T, hc, D] x, rows [T, n_cols, hd], valid [T, n_cols] float"""
    T, hc, D = x.shape
    nc, hd = rows.shape[1], rows.shape[2]
    rows = rows.contiguous()
    r = torch.empty(T, nc * hd, dtype=torch.float32, device=x.device)
    _k_pre[(T * nc,)](rows, e.lookup_norm, valid.contiguous(), r, hd, e.eps, BH=triton.next_power_of_2(hd), NC=nc, num_warps=1)
    kv = r @ e.kv.T
    out = torch.empty_like(x)
    _k_post[(T,)](kv.contiguous(), e.kv_b, x.contiguous(), e.gate, e.hc_scale, e.pre_gamma, valid.contiguous(), out, D, e.eps, D ** -0.5,
                  HC=hc, NC=nc, BD=triton.next_power_of_2(D), num_warps=8)
    return out
