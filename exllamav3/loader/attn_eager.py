"""HY4_ATTN_EAGER: fewer eager launches in the attention projections (flag default OFF; old path untouched when off).

  HY4_ATTN_EAGER=1  (bit-exact vs the old code)
     rope_norm : cos/sin built ONCE per (dim, base, device) in float64 with the SAME expression as the per-call path, cast to fp32,
                 stored as [n, 2, P/2, 2] = [(c, s), (-s, c)]. A call = 1 index_select + 2 mul + 1 add (old: 18 launches, +1 contiguous
                 in q). out = x0*(c,s) + x1*(-s,c) is the same fp32 ops as x0*c - x1*s / x0*s + x1*c (a - b == a + (-b), no FMA in eager).
     kv()      : rms*w and rope write straight into one preallocated [T, R+rope] buffer (no torch.cat, no copy_ into out[..., 0::2]).
  HY4_ATTN_EAGER=2  = 1 + rmsw(): rms(x)*w as ONE Triton launch (kv_a, q_a, post-attn norm). The reduction order differs from torch's
                 mean(), so NOT bit-exact (~1e-7 rel): gate it with teacher forcing before keeping.
Generic by shape: P, base, R, H come from the caller / tensors."""
from __future__ import annotations
import os
import torch
from . import mla_bgemv as _BG
from . import kv_fused as _KVF

_F = os.environ.get("HY4_ATTN_EAGER", "0")
ENABLED = _F in ("1", "2")
RMS = _F == "2"
_NEED = [0]            # highest position + 1 the caller announced (host int, set by ensure())
_TAB = {}              # (P, base, device) -> [n, 2, P/2, 2]

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None


def ensure(n: int):
    """host-side bound on positions (start + T); tables grow lazily on the next rope call. No device sync."""
    if n > _NEED[0]:
        _NEED[0] = n


def _build(P, base, n, dev):
    h = P // 2
    pos = torch.arange(n, device=dev)
    th = pos.double()[:, None] * (base ** (-2.0 * torch.arange(h, dtype=torch.float64, device=dev) / P))[None, :]   # == rope_norm
    c, s = th.cos().float(), th.sin().float()
    tab = torch.empty(n, 2, h, 2, dtype=torch.float32, device=dev)
    tab[:, 0, :, 0], tab[:, 0, :, 1], tab[:, 1, :, 0], tab[:, 1, :, 1] = c, s, -s, c
    return tab


def table(P, base, dev, pos):
    key = (P, float(base), str(dev))
    tab = _TAB.get(key)
    need = _NEED[0]
    if need == 0:                                    # nobody announced a bound: derive it (one sync; CPU: free)
        need = int(pos.max()) + 1
    if tab is None or tab.shape[0] < need:
        tab = _TAB[key] = _build(P, base, max(need, 2 * (tab.shape[0] if tab is not None else 2048)), dev)
    return tab


def rope(x, pos, base, out=None):
    """== mla_hy4.rope_norm (bit-exact). x [T, ..., P] with unit last stride (views / slices allowed); out: optional [T, ..., P] view"""
    P = x.shape[-1]
    h = P // 2
    g = table(P, base, x.device, pos).index_select(0, pos.long())            # [T, 2, h, 2]
    shp = (x.shape[0],) + (1,) * (x.dim() - 2) + (h, 2)
    a, b = g[:, 0].reshape(shp), g[:, 1].reshape(shp)
    xv = x.unflatten(-1, (h, 2))
    ov = torch.empty(xv.shape, dtype=x.dtype, device=x.device) if out is None else out.unflatten(-1, (h, 2))
    torch.add(xv[..., 0:1] * a, xv[..., 1:2] * b, out=ov)
    return ov.flatten(-2) if out is None else out


def kv(m, cur, pos):
    """Hy4MLA.kv: [rms(kv_a)*w | rope(k_pe)] written into one buffer"""
    c = m._q(cur, "wkv_a") @ m.wkv_a.T
    R = m.R
    buf = torch.empty(c.shape[0], R + m.rope_dim, dtype=c.dtype, device=c.device)
    if RMS:
        rmsw(c[:, :R], m.kv_a_norm, m.eps, out=buf[:, :R])
    else:
        from .engram_rows import rms
        torch.mul(rms(c[:, :R], m.eps), m.kv_a_norm, out=buf[:, :R])
    rope(c[:, R:], pos, m.theta, out=buf[:, R:])
    return m._h(buf)


def q(m, cur, pos):
    """Hy4MLA.q: same maths, rope reads the strided slice (no .contiguous())"""
    T, H = cur.shape[0], m.H
    qa = m._q(cur, "wq_a") @ m.wq_a.T
    if RMS:
        qr = rmsw(qa, m.q_a_norm, m.eps)
    else:
        from .engram_rows import rms
        qr = rms(qa, m.eps) * m.q_a_norm
    qq = (m._q(qr, "wq_b") @ m.wq_b.T).view(T, H, m.hd)
    qn = qq[..., :m.nope]
    if _KVF.ENABLED and T == 1 and qq.is_cuda:         # HY4_KV_FUSED: the pe rotation in one launch (no slice copies)
        qp = _KVF.rope_q(qq, table(m.rope_dim, m.theta, qq.device, pos), pos, m.nope, m.rope_dim)
    else:
        qp = rope(qq[..., m.nope:], pos, m.theta)
    qlat = m.bg_kb.mv(m._q(qn, "kb")) if (_BG.ENABLED and T == 1 and getattr(m, "bg_kb", None) is not None) else torch.einsum("hrk,thk->thr", m.wk_b, m._q(qn, "kb"))
    return m._h(qlat, "q"), m._h(qp, "q")


if triton is not None:
    @triton.jit
    def _k_rmsw(x_ptr, w_ptr, o_ptr, D, sx, so, eps, BD: tl.constexpr):
        r = tl.program_id(0)
        d = tl.arange(0, BD)
        dm = d < D
        x = tl.load(x_ptr + r * sx + d, mask=dm, other=0.0)
        rc = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        tl.store(o_ptr + r * so + d, x * rc * tl.load(w_ptr + d, mask=dm, other=0.0), mask=dm)


def rmsw(x, w, eps, out=None):
    """rms(x, eps) * w over the last dim of a [R, D] fp32 tensor (row stride free, unit column stride): one launch"""
    from .engram_rows import rms
    usable = (triton is not None and x.dtype == torch.float32 and x.dim() == 2 and x.stride(1) == 1 and w.is_contiguous()
              and (x.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1") and x.shape[1] <= 8192
              and (out is None or (out.stride(1) == 1 and out.dtype == torch.float32)))
    if not usable:
        r = rms(x, eps) * w
        if out is None:
            return r
        out.copy_(r)
        return out
    o = torch.empty(x.shape, dtype=x.dtype, device=x.device) if out is None else out
    D = x.shape[1]
    _k_rmsw[(x.shape[0],)](x, w, o, D, x.stride(0), o.stride(0), eps, BD=triton.next_power_of_2(D), num_warps=8)
    return o
