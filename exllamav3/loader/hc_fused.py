"""Fused hyper-connection sites + norms (Triton). Flag HY4_HC_FUSED=1 (default off; hy4_forward only calls in here when ENABLED).

Generic by SHAPE: n_streams H (any), hidden D, fn [NO, H*D] fp32 with NO = 2H (pre+post, "site") or H (pre only, "head"); no model name.
The math is exactly the current torch path of hy4_forward.py (_site / layer / _head), fp32:
    mix  = rms(x.flat, eps) @ fn.T
    pre  = sigmoid(mix[:H] * scale[0] + base[:H]) + hc_eps
    post = magnitude * sigmoid(mix[H:] * scale[1] + base[H:]) + hc_eps
    cur  = rms((x * pre[:, None]).sum(streams), eps) * w
Launches per layer: site A = 2, residual+site B = 2, residual C = 1  (old path: ~80-100 torch launches).
  K1 hc_mix_partials : per (row, chunk) of the H*D row: [optional s = x + post_prev * y written out], sumsq, NO dot partials with fn
  K2 hc_mix_collapse : per row: reduce partials -> pre/post (store post), collapse streams by pre, rms, * norm weight -> cur
  K3 hc_resid        : per row: f = rms(f) * w ; s = s + post * f  (all streams)
"""
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("HY4_HC_FUSED", "0") == "1"
ONEPASS = os.environ.get("HY4_PF_FUSE", "0") == "1"      # one program per row: [optional residual add], mix dots, sigmoid, collapse, norm in ONE read of the stream (replaces K1 + K2)
BK = int(os.environ.get("HY4_HC_BK", "512"))      # K1 chunk (elements of the flattened H*D row)
EMIT16 = [os.environ.get("HY4_PF_FUSE", "0") == "1"]     # the one-pass kernel also writes the saturating fp16 copy of cur (registered with to_f16)
PF_MIN_ROWS = 16                                    # rows <= 16 are decode / verify steps (never take the prefill-only kernels)
MAX_D = 16384                                       # K2/K3 keep one row of D in registers


def _p2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


@triton.jit
def _k1_partials(x_ptr, y_ptr, post_ptr, s_ptr, fn_ptr, part_ptr, sq_ptr,
                 K, D, NO, NCH,
                 HAS_RES: tl.constexpr, NOP: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    k = c * BK + tl.arange(0, BK)
    km = k < K
    xv = tl.load(x_ptr + r * K + k, mask=km, other=0.0)
    if HAS_RES:
        h = k // D
        d = k - h * D
        yv = tl.load(y_ptr + r * D + d, mask=km, other=0.0)
        pv = tl.load(post_ptr + r * (K // D) + h, mask=km, other=0.0)
        xv = xv + pv * yv
        tl.store(s_ptr + r * K + k, xv, mask=km)
    o = tl.arange(0, NOP)
    fm = (o[:, None] < NO) & km[None, :]
    fv = tl.load(fn_ptr + o[:, None] * K + k[None, :], mask=fm, other=0.0)
    acc = tl.sum(fv * xv[None, :], axis=1)
    tl.store(part_ptr + (r * NCH + c) * NOP + o, acc, mask=o < NO)
    tl.store(sq_ptr + r * NCH + c, tl.sum(xv * xv, axis=0))


@triton.jit
def _k2_collapse(x_ptr, part_ptr, sq_ptr, base_ptr, scale_ptr, w_ptr, post_ptr, cur_ptr,
                 D, K, NCH, eps, hc_eps, mag,
                 H: tl.constexpr, HP: tl.constexpr, NOP: tl.constexpr, NCHP: tl.constexpr,
                 HAS_POST: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    ch = tl.arange(0, NCHP)
    hv = tl.arange(0, HP)
    cm = ch < NCH
    ss = tl.sum(tl.load(sq_ptr + r * NCH + ch, mask=cm, other=0.0), axis=0)
    rr = tl.math.rsqrt(ss / K + eps)
    pm = tl.sum(tl.load(part_ptr + (r * NCH + ch[:, None]) * NOP + hv[None, :],
                        mask=cm[:, None] & (hv[None, :] < H), other=0.0), axis=0)          # [HP] mix[:H]
    s0 = tl.load(scale_ptr)
    pre = tl.sigmoid(pm * rr * s0 + tl.load(base_ptr + hv, mask=hv < H, other=0.0)) + hc_eps
    if HAS_POST:
        qm = tl.sum(tl.load(part_ptr + (r * NCH + ch[:, None]) * NOP + H + hv[None, :],
                            mask=cm[:, None] & (hv[None, :] < H), other=0.0), axis=0)      # [HP] mix[H:]
        s1 = tl.load(scale_ptr + 1)
        post = mag * tl.sigmoid(qm * rr * s1 + tl.load(base_ptr + H + hv, mask=hv < H, other=0.0)) + hc_eps
        tl.store(post_ptr + r * H + hv, post, mask=hv < H)
    d = tl.arange(0, BD)
    dm = d < D
    col = tl.zeros([BD], dtype=tl.float32)
    for h in tl.static_range(H):
        ph = tl.sum(tl.where(hv == h, pre, 0.0), axis=0)
        xh = tl.load(x_ptr + r * K + h * D + d, mask=dm, other=0.0)
        col = col + xh * ph
    rc = tl.math.rsqrt(tl.sum(col * col, axis=0) / D + eps)
    w = tl.load(w_ptr + d, mask=dm, other=0.0)
    tl.store(cur_ptr + r * D + d, col * rc * w, mask=dm)


@triton.jit
def _k12_row(x_ptr, y_ptr, post_ptr, s_ptr, fn_ptr, base_ptr, scale_ptr, w_ptr, postout_ptr, cur_ptr, cur16_ptr,
             D, K, eps, hc_eps, mag,
             H: tl.constexpr, HP: tl.constexpr, NO: tl.constexpr, HAS_RES: tl.constexpr, EMIT16: tl.constexpr, BD: tl.constexpr):
    """K1 + K2 in one program per row: the whole [H, D] row is held in registers (read once, residual written once)"""
    r = tl.program_id(0)
    hv = tl.arange(0, HP)
    d = tl.arange(0, BD)
    off = hv[:, None] * D + d[None, :]
    xm = (hv[:, None] < H) & (d[None, :] < D)
    x = tl.load(x_ptr + r * K + off, mask=xm, other=0.0)
    if HAS_RES:
        yv = tl.load(y_ptr + r * D + d, mask=d < D, other=0.0)
        pv = tl.load(post_ptr + r * H + hv, mask=hv < H, other=0.0)
        x = x + pv[:, None] * yv[None, :]
        tl.store(s_ptr + r * K + off, x, mask=xm)
    rr = tl.math.rsqrt(tl.sum(tl.sum(x * x, axis=1), axis=0) / K + eps)
    s0 = tl.load(scale_ptr)
    pre_v = tl.zeros([HP], dtype=tl.float32)
    for o in tl.static_range(NO):
        fnt = tl.load(fn_ptr + o * K + off, mask=xm, other=0.0)
        m = tl.sum(tl.sum(fnt * x, axis=1), axis=0)
        if o < H:
            val = tl.sigmoid(m * rr * s0 + tl.load(base_ptr + o)) + hc_eps
            pre_v = tl.where(hv == o, val, pre_v)
        else:
            tl.store(postout_ptr + r * H + (o - H), mag * tl.sigmoid(m * rr * tl.load(scale_ptr + 1) + tl.load(base_ptr + o)) + hc_eps)
    col = tl.sum(x * pre_v[:, None], axis=0)
    rc = tl.math.rsqrt(tl.sum(col * col, axis=0) / D + eps)
    c = col * rc * tl.load(w_ptr + d, mask=d < D, other=0.0)
    tl.store(cur_ptr + r * D + d, c, mask=d < D)
    if EMIT16:
        tl.store(cur16_ptr + r * D + d, tl.minimum(tl.maximum(c, -65504.0), 65504.0).to(tl.float16), mask=d < D)


@triton.jit
def _k3_resid(s_ptr, f_ptr, post_ptr, w_ptr, o_ptr, D, eps,
              H: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BD)
    dm = d < D
    f = tl.load(f_ptr + r * D + d, mask=dm, other=0.0)
    rc = tl.math.rsqrt(tl.sum(f * f, axis=0) / D + eps)
    fn = f * rc * tl.load(w_ptr + d, mask=dm, other=0.0)
    for h in tl.static_range(H):
        ph = tl.load(post_ptr + r * H + h)
        sv = tl.load(s_ptr + (r * H + h) * D + d, mask=dm, other=0.0)
        tl.store(o_ptr + (r * H + h) * D + d, sv + ph * fn, mask=dm)


@triton.jit
def _k4_rmsw(x_ptr, w_ptr, o_ptr, o16_ptr, D, eps, EMIT16: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BD)
    dm = d < D
    x = tl.load(x_ptr + r * D + d, mask=dm, other=0.0)
    rc = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    o = x * rc * tl.load(w_ptr + d, mask=dm, other=0.0)
    tl.store(o_ptr + r * D + d, o, mask=dm)
    if EMIT16:
        tl.store(o16_ptr + r * D + d, tl.minimum(tl.maximum(o, -65504.0), 65504.0).to(tl.float16), mask=dm)


def ok(x, hw, H):
    """the fused kernels apply: fp32 contiguous streams [T, H, D], fn [NO, H*D] fp32 with NO in (H, 2H), D within the register budget"""
    if not ENABLED or x.dim() != 3 or x.dtype != torch.float32 or not x.is_contiguous() or x.shape[1] != H:
        return False
    fn = hw[0]
    if not (fn.is_contiguous() and hw[1].is_contiguous() and hw[2].is_contiguous() and hw[1].dtype == torch.float32 and hw[2].dtype == torch.float32):
        return False
    return fn.dtype == torch.float32 and fn.shape[1] == H * x.shape[2] and fn.shape[0] in (H, 2 * H) and x.shape[2] <= MAX_D


def _mix(x, y, post_prev, hw, w, eps, hc_eps, mag):
    T, H, D = x.shape
    fn, base, scale = hw
    NO = fn.shape[0]
    K = H * D
    NOP = _p2(NO)
    NCH = triton.cdiv(K, BK)
    dev = x.device
    if ONEPASS and H <= 8 and T > PF_MIN_ROWS:      # prefill rows only: the decode / verify rows keep the two-kernel path of the decode graph
        has_res = y is not None
        s = torch.empty_like(x) if has_res else x
        post = torch.empty(T, H, dtype=torch.float32, device=dev) if NO == 2 * H else None
        cur = torch.empty(T, D, dtype=torch.float32, device=dev)
        cur16 = torch.empty(T, D, dtype=torch.float16, device=dev) if EMIT16[0] else cur
        _k12_row[(T,)](x, y if has_res else x, post_prev if has_res else x, s, fn, base, scale, w, post if post is not None else cur, cur, cur16,
                       D, K, eps, hc_eps, mag, H=H, HP=_p2(H), NO=NO, HAS_RES=has_res, EMIT16=EMIT16[0], BD=_p2(D), num_warps=8)
        if EMIT16[0]:
            from . import prefill_kernels as _pk
            _pk.register_f16(cur, cur16)
        return s, post, cur
    part = torch.empty(T, NCH, NOP, dtype=torch.float32, device=dev)
    sq = torch.empty(T, NCH, dtype=torch.float32, device=dev)
    has_res = y is not None
    s = torch.empty_like(x) if has_res else x
    _k1_partials[(T, NCH)](x, y if has_res else x, post_prev if has_res else x, s, fn, part, sq, K, D, NO, NCH,
                           HAS_RES=has_res, NOP=NOP, BK=BK)
    has_post = NO == 2 * H
    post = torch.empty(T, H, dtype=torch.float32, device=dev) if has_post else None
    cur = torch.empty(T, D, dtype=torch.float32, device=dev)
    _k2_collapse[(T,)](s, part, sq, base, scale, w, post if has_post else cur, cur, D, K, NCH,
                       eps, hc_eps, mag, H=H, HP=_p2(H), NOP=NOP, NCHP=_p2(NCH),
                       HAS_POST=has_post, BD=_p2(D), num_warps=8)
    return s, post, cur


def site(x, hw, w_norm, eps, hc_eps, mag):
    """site A / head: x [T,H,D] -> (post [T,H] or None for a pre-only head fn, cur [T,D] = rms(sum_h pre_h x_h) * w_norm)"""
    _, post, cur = _mix(x, None, None, hw, w_norm.contiguous(), eps, hc_eps, mag)
    return post, cur


def res_site(x, y, post_prev, hw, w_norm, eps, hc_eps, mag):
    """s = x + post_prev[:, :, None] * y[:, None, :]; then the site on s -> (s, post, cur)"""
    return _mix(x, y.float().contiguous(), post_prev.float().contiguous(), hw, w_norm.contiguous(), eps, hc_eps, mag)


def resid(s, f, post, w_post, eps):
    """s + post[:, :, None] * (rms(f) * w_post)[:, None, :]  -> new [T,H,D]"""
    T, H, D = s.shape
    o = torch.empty_like(s)
    _k3_resid[(T,)](s, f.float().contiguous(), post.float().contiguous(), w_post.contiguous(), o, D, eps, H=H, BD=_p2(D), num_warps=8)
    return o


def rmsw(x, w, eps, emit16=False):
    """rms(x, eps) * w over the last dim, fp32 [R, D] (one launch instead of 5). emit16: also writes the saturating fp16 copy and registers it with to_f16"""
    x = x.float().contiguous()
    emit16 = emit16 and x.shape[0] > PF_MIN_ROWS
    o = torch.empty_like(x)
    o16 = torch.empty(x.shape, dtype=torch.float16, device=x.device) if emit16 else o
    _k4_rmsw[(x.shape[0],)](x, w.contiguous(), o, o16, x.shape[1], eps, EMIT16=bool(emit16), BD=_p2(x.shape[1]), num_warps=8)
    if emit16:
        from . import prefill_kernels as _pk
        _pk.register_f16(o, o16)
    return o
