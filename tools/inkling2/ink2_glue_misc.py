"""Inkling 2 Mini glue fusions (Triton; runs on CUDA, and on CPU tensors under TRITON_INTERPRET=1 for the unit test): the small ops of the step that are launch / bandwidth bound.

  rmsw(x, w, eps)                  x * rsqrt(mean(x^2) + eps) [* w]                      1 launch   (torch: 4-5)
  mix(states, query, eps)          block-depth attention of Inkling2Forward.mix           2 launches (torch: ~9)   states [S, T, D] raw (rms is inside), query [Q, D] -> [Q, T, D]
  control(cur, msg, wd, wg, wf)    (1 - tanh(F h)) cur + tanh(G h) msg, h = D [rms(cur); rms(msg)]   2 launches (torch: ~14)
T <= 8 rows, fp32 in / out, D = 2304 (any D works: masked power-of-two blocks)."""
import math

import torch
import triton
import triton.language as tl

EPS_MIX = 1e-6
FIRES = {"rmsw": 0, "mix": 0, "control": 0}


@triton.jit
def _rmsw_k(X, W, OUT, C, eps, BLOCK: tl.constexpr, HAS_W: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < C
    x = tl.load(X + t * C + c, mask=m, other=0.0)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / C + eps)
    if HAS_W:
        y = y * tl.load(W + c, mask=m, other=0.0)
    tl.store(OUT + t * C + c, y, mask=m)


def rmsw(x, w, eps):
    """x [T, C] fp32 (row stride C after .contiguous()), w [C] fp32 or None"""
    x = x.contiguous()
    T, C = x.shape
    out = torch.empty_like(x)
    FIRES["rmsw"] += 1
    _rmsw_k[(T,)](x, w if w is not None else x, out, C, float(eps), BLOCK=triton.next_power_of_2(C), HAS_W=w is not None, num_warps=4)
    return out


@triton.jit
def _mix_scores_k(ST, QU, SC, INV, S, T, D, eps, scale, BLOCK: tl.constexpr, QN: tl.constexpr):
    s = tl.program_id(0)
    t = tl.program_id(1)
    c = tl.arange(0, BLOCK)
    m = c < D
    x = tl.load(ST + (s * T + t) * D + c, mask=m, other=0.0)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    tl.store(INV + t * S + s, inv)
    for q in tl.static_range(QN):
        qv = tl.load(QU + q * D + c, mask=m, other=0.0)
        tl.store(SC + (q * T + t) * S + s, tl.sum(x * qv, 0) * inv * scale)


@triton.jit
def _mix_out_k(ST, SC, INV, OUT, T, D, S: tl.constexpr, SP: tl.constexpr, BC: tl.constexpr, QN: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BC + tl.arange(0, BC)
    cm = cols < D
    idx = tl.arange(0, SP)
    im = idx < S
    inv = tl.load(INV + t * S + idx, mask=im, other=0.0)
    s0 = tl.load(SC + (0 * T + t) * S + idx, mask=im, other=float("-inf"))
    e0 = tl.exp(s0 - tl.max(s0, 0))
    p0 = e0 / tl.sum(e0, 0) * inv
    if QN > 1:
        s1 = tl.load(SC + (1 * T + t) * S + idx, mask=im, other=float("-inf"))
        e1 = tl.exp(s1 - tl.max(s1, 0))
        p1 = e1 / tl.sum(e1, 0) * inv
    if QN > 2:
        s2 = tl.load(SC + (2 * T + t) * S + idx, mask=im, other=float("-inf"))
        e2 = tl.exp(s2 - tl.max(s2, 0))
        p2 = e2 / tl.sum(e2, 0) * inv
    a0 = tl.zeros([BC], dtype=tl.float32)
    a1 = tl.zeros([BC], dtype=tl.float32)
    a2 = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        x = tl.load(ST + (s * T + t) * D + cols, mask=cm, other=0.0)
        a0 += tl.sum(tl.where(idx == s, p0, 0.0), 0) * x
        if QN > 1:
            a1 += tl.sum(tl.where(idx == s, p1, 0.0), 0) * x
        if QN > 2:
            a2 += tl.sum(tl.where(idx == s, p2, 0.0), 0) * x
    tl.store(OUT + (0 * T + t) * D + cols, a0, mask=cm)
    if QN > 1:
        tl.store(OUT + (1 * T + t) * D + cols, a1, mask=cm)
    if QN > 2:
        tl.store(OUT + (2 * T + t) * D + cols, a2, mask=cm)


def mix(states, query, eps=EPS_MIX):
    """states [S, T, D] fp32 contiguous (the raw blocks + cur, stacked), query [Q, D] fp32 (Q <= 3) -> [Q, T, D]; = softmax_s(q . rms(state_s) / sqrt(D)) weighted sum of rms(state_s)"""
    S, T, D = states.shape
    Q = query.shape[0]
    assert Q <= 3 and T <= 8 and S <= 32 and states.is_contiguous() and query.is_contiguous()
    sc = torch.empty(Q, T, S, device=states.device, dtype=torch.float32)
    inv = torch.empty(T, S, device=states.device, dtype=torch.float32)
    out = torch.empty(Q, T, D, device=states.device, dtype=torch.float32)
    FIRES["mix"] += 1
    _mix_scores_k[(S, T)](states, query, sc, inv, S, T, D, float(eps), 1.0 / math.sqrt(D), BLOCK=triton.next_power_of_2(D), QN=Q, num_warps=4)
    _mix_out_k[(T, triton.cdiv(D, 512))](states, sc, inv, out, T, D, S=S, SP=triton.next_power_of_2(S), BC=512, QN=Q, num_warps=4)
    return out


@triton.jit
def _tanh(x):
    x = tl.minimum(tl.maximum(x, -20.0), 20.0)
    e = tl.exp(2.0 * x)
    return (e - 1.0) / (e + 1.0)


@triton.jit
def _ctl_h_k(CUR, MSG, WD, HP, D, eps, R: tl.constexpr, BK: tl.constexpr, NCH: tl.constexpr):
    t = tl.program_id(0)
    kb = tl.program_id(1)
    cc = tl.arange(0, NCH)[:, None] * BK + tl.arange(0, BK)[None, :]      # one wide load of both rows (the per-chunk loop was a serial dependent chain)
    mm = cc < D
    xc = tl.load(CUR + t * D + cc, mask=mm, other=0.0)
    xm = tl.load(MSG + t * D + cc, mask=mm, other=0.0)
    ss_c = tl.sum(tl.sum(xc * xc, 1), 0)
    ss_m = tl.sum(tl.sum(xm * xm, 1), 0)
    ic = tl.rsqrt(ss_c / D + eps)
    im = tl.rsqrt(ss_m / D + eps)
    k = kb * BK + tl.arange(0, BK)
    km = k < 2 * D
    xv = tl.where(k < D, tl.load(CUR + t * D + k, mask=km & (k < D), other=0.0) * ic, tl.load(MSG + t * D + (k - D), mask=km & (k >= D), other=0.0) * im)
    r = tl.arange(0, R)
    w = tl.load(WD + r[:, None] * (2 * D) + k[None, :], mask=km[None, :], other=0.0)
    tl.store(HP + (t * tl.num_programs(1) + kb) * R + r, tl.sum(w * xv[None, :], 1))


@triton.jit
def _ctl_out_k(HP, WG, WF, CUR, MSG, OUT, D, NB: tl.constexpr, R: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    r = tl.arange(0, R)
    h = tl.zeros([R], dtype=tl.float32)
    for b in tl.static_range(NB):
        h += tl.load(HP + (t * NB + b) * R + r)
    n = nb * BN + tl.arange(0, BN)
    nm = n < D
    wg = tl.load(WG + n[:, None] * R + r[None, :], mask=nm[:, None], other=0.0)
    wf = tl.load(WF + n[:, None] * R + r[None, :], mask=nm[:, None], other=0.0)
    gain = _tanh(tl.sum(wg * h[None, :], 1))
    forget = _tanh(tl.sum(wf * h[None, :], 1))
    cur = tl.load(CUR + t * D + n, mask=nm, other=0.0)
    msg = tl.load(MSG + t * D + n, mask=nm, other=0.0)
    tl.store(OUT + t * D + n, (1.0 - forget) * cur + gain * msg, mask=nm)


def control(cur, msg, wd, wg, wf, eps=EPS_MIX):
    """cur / msg [T, D] fp32; wd [R, 2D] (down), wg / wf [D, R] (gain / forget) fp32 contiguous -> [T, D]"""
    cur, msg = cur.contiguous(), msg.contiguous()
    T, D = cur.shape
    R = wd.shape[0]
    assert T <= 8 and wd.shape[1] == 2 * D and wg.shape == (D, R) and wf.shape == (D, R) and R & (R - 1) == 0
    BK = 128
    NB = triton.cdiv(2 * D, BK)
    hp = torch.empty(T, NB, R, device=cur.device, dtype=torch.float32)
    out = torch.empty_like(cur)
    FIRES["control"] += 1
    _ctl_h_k[(T, NB)](cur, msg, wd, hp, D, float(eps), R=R, BK=BK, NCH=triton.next_power_of_2(triton.cdiv(D, BK)), num_warps=4)
    _ctl_out_k[(T, triton.cdiv(D, 32))](hp, wg, wf, cur, msg, out, D, NB=NB, R=R, BN=32, num_warps=2)
    return out
