"""Flash-style MLA prefill attention (Triton, absorbed latent form): one program = one query position x a block of heads.

scores[h, t, j] = scale * (q_lat[t, h] . ckv[j] + q_pe[t, h] . k_pe[j]);  visible: kpos[j] <= qpos[t], window (0 = none), optional bool mask `allow` [T, N];
a per-head sink joins the softmax denominator only (running max / sum start at the sink logit); o[t, h] = softmax . ckv.
The latent K rows [N, R + P] are shared by every head (MQA form): the same tile is the key (all R + P columns) and the value (first R columns).
Same maths as modules/attention_fn/mla_sinks.mla_sink_attend; fp16 operands, fp32 accumulation / softmax. Keys are visited only inside
[qpos - window + 1, qpos], so a sliding-window layer costs O(window), not O(N). Generic by (R, P, H, window); nothing model specific.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
CFG = {"HB": 32, "BK": 32, "warps": 4, "stages": 1}


@triton.jit
def _mla_kernel(Q, QP, KC, SINK, ALLOW, O, T, N, qstart, kbase, H, scale,
                R: tl.constexpr, P: tl.constexpr, HB: tl.constexpr, BK: tl.constexpr, WIN: tl.constexpr,
                HAS_ALLOW: tl.constexpr, HAS_SINK: tl.constexpr):
    # R = 4 * DC: the latent dim is walked in 4 chunks (q / k loaded per chunk, 4 separate fp32 accumulators): keeps the dot-operand fragments
    # of one chunk live instead of the whole [rows, R] tile (a full-width tile needs ~256 VGPRs of duplicated WMMA operand fragments -> spills)
    DC: tl.constexpr = R // 4
    t = tl.program_id(0)
    hb = tl.program_id(1)
    pos = qstart + t                                   # absolute position of this query
    h = hb * HB + tl.arange(0, HB)
    rowq = (t * H + h).to(tl.int64)
    cd = tl.arange(0, DC)
    cp = tl.arange(0, P)
    qp = tl.load(QP + rowq[:, None] * P + cp[None, :])
    if HAS_SINK:
        m = tl.load(SINK + h).to(tl.float32) * LOG2E
        l = tl.full((HB,), 1.0, dtype=tl.float32)
    else:
        m = tl.full((HB,), -1e30, dtype=tl.float32)
        l = tl.zeros((HB,), dtype=tl.float32)
    a0 = tl.zeros((HB, DC), dtype=tl.float32)
    a1 = tl.zeros((HB, DC), dtype=tl.float32)
    a2 = tl.zeros((HB, DC), dtype=tl.float32)
    a3 = tl.zeros((HB, DC), dtype=tl.float32)
    hi = pos + 1
    lo = 0
    if WIN > 0:
        lo = tl.maximum(0, pos - WIN + 1)
    lo = (lo // BK) * BK
    sc = scale * LOG2E
    for k0 in range(lo, hi, BK):
        kidx = k0 + tl.arange(0, BK)
        kin = (kidx < hi) & (kidx >= 0)
        krow = (kidx - kbase).to(tl.int64) * (R + P)
        kp = tl.load(KC + krow[:, None] + R + cp[None, :], mask=kin[:, None], other=0.0)     # [BK, P]
        s = tl.dot(qp, tl.trans(kp))
        for c in tl.static_range(4):
            qc = tl.load(Q + rowq[:, None] * R + c * DC + cd[None, :], volatile=True)
            kc = tl.load(KC + krow[:, None] + c * DC + cd[None, :], mask=kin[:, None], other=0.0)
            s = tl.dot(qc, tl.trans(kc), s)
        s = s * sc
        vis = kin
        if WIN > 0:
            vis = vis & (kidx > pos - WIN)
        if HAS_ALLOW:
            al = tl.load(ALLOW + t.to(tl.int64) * N + kidx, mask=kin, other=0)
            vis = vis & (al != 0)
        s = tl.where(vis[None, :], s, -1e30)
        mn = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp2(m - mn)
        p = tl.exp2(s - mn[:, None])
        p = tl.where(vis[None, :], p, 0.0)
        l = l * alpha + tl.sum(p, 1)
        p16 = p.to(tl.float16)
        v0 = tl.load(KC + krow[:, None] + 0 * DC + cd[None, :], mask=kin[:, None], other=0.0)
        a0 = tl.dot(p16, v0, a0 * alpha[:, None])
        v1 = tl.load(KC + krow[:, None] + 1 * DC + cd[None, :], mask=kin[:, None], other=0.0)
        a1 = tl.dot(p16, v1, a1 * alpha[:, None])
        v2 = tl.load(KC + krow[:, None] + 2 * DC + cd[None, :], mask=kin[:, None], other=0.0)
        a2 = tl.dot(p16, v2, a2 * alpha[:, None])
        v3 = tl.load(KC + krow[:, None] + 3 * DC + cd[None, :], mask=kin[:, None], other=0.0)
        a3 = tl.dot(p16, v3, a3 * alpha[:, None])
        m = mn
    inv = 1.0 / tl.maximum(l, 1e-30)
    tl.store(O + rowq[:, None] * R + 0 * DC + cd[None, :], a0 * inv[:, None])
    tl.store(O + rowq[:, None] * R + 1 * DC + cd[None, :], a1 * inv[:, None])
    tl.store(O + rowq[:, None] * R + 2 * DC + cd[None, :], a2 * inv[:, None])
    tl.store(O + rowq[:, None] * R + 3 * DC + cd[None, :], a3 * inv[:, None])


def mla_flash(q_lat, q_pe, kc, qstart, scale, sinks=None, window=0, allow=None, cfg=None):
    """q_lat [T, H, R], q_pe [T, H, P] (any float dtype), kc [N, R + P] latent K rows of ALL positions 0..N-1 (N = qstart + T), qstart = position of query 0,
    sinks [H] or None, window 0 / int, allow bool [T, N] or None -> o_lat [T, H, R] fp32"""
    c = cfg or CFG
    T, H, R = q_lat.shape
    P = q_pe.shape[-1]
    N = kc.shape[0]
    HB = min(c["HB"], H)
    assert H % HB == 0 and N == qstart + T
    klo = max(0, qstart - window + 1) if window else 0
    klo = (klo // c["BK"]) * c["BK"]
    kc16 = kc[klo:].clamp(-65504.0, 65504.0).half().contiguous()
    q16 = q_lat.clamp(-65504.0, 65504.0).half().contiguous()
    qp16 = q_pe.clamp(-65504.0, 65504.0).half().contiguous()
    o = torch.empty(T, H, R, dtype=torch.float32, device=q_lat.device)
    sk = sinks.float().contiguous() if sinks is not None else q16
    al = allow.to(torch.int8).contiguous() if allow is not None else q16
    _mla_kernel[(T, H // HB)](q16, qp16, kc16, sk, al, o, T, N, qstart, klo, H, float(scale), R=R, P=P, HB=HB, BK=c["BK"], WIN=int(window),
                              HAS_ALLOW=allow is not None, HAS_SINK=sinks is not None, num_warps=c["warps"], num_stages=c["stages"])
    return o


# ------------------------------------------------------------------------------------------------ banded pipeline (default)
# The one-pass flash kernel above needs a [rows, 512] accumulator plus dot-operand fragments of the whole latent width: it spills hundreds of VGPRs on
# gfx11. Pipeline instead, per block of Bq query positions (all heads as rows, keys = the visible band [klo, last position]):
#   1. S = scale * Q K^T  (fp32, Triton GEMM over the concatenated [latent | rope] width, visibility mask -> -inf)
#   2. P = softmax(S, sink)  per row (Triton row kernel, fp16 out)
#   3. O = P V            (Triton GEMM, V = first R columns of the same K rows)
@triton.jit
def _qk_kernel(Q, K, S, ALLOW, M, NK, NKP, row0, t0, qstart, klo, kbase, N, H, scale_l2, TT,
               D: tl.constexpr, WIN: tl.constexpr, HAS_ALLOW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HT: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    nn = offs_n < NK
    kb = tl.arange(0, BK)
    if HT:        # Q rows laid out [H, TT, D] (row = t * H + h)
        gq = (row0 + offs_m).to(tl.int64)
        qrow = ((gq % H) * TT + gq // H) * D
    else:
        qrow = (row0 + offs_m).to(tl.int64) * D
    krow = (klo - kbase + offs_n).to(tl.int64) * D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for d in range(0, D, BK):
        a = tl.load(Q + qrow[:, None] + d + kb[None, :], mask=mm[:, None], other=0.0)
        b = tl.load(K + krow[None, :] + d + kb[:, None], mask=nn[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    pos = qstart + t0 + offs_m // H
    kabs = klo + offs_n
    vis = nn[None, :] & (kabs[None, :] <= pos[:, None]) & mm[:, None]
    if WIN > 0:
        vis = vis & (kabs[None, :] > pos[:, None] - WIN)
    if HAS_ALLOW:
        al = tl.load(ALLOW + (t0 + offs_m // H).to(tl.int64)[:, None] * N + kabs[None, :], mask=vis, other=0)
        vis = vis & (al != 0)
    s = tl.where(vis, acc * scale_l2, float("-inf"))
    tl.store(S + offs_m.to(tl.int64)[:, None] * NKP + offs_n[None, :], s, mask=mm[:, None] & (offs_n[None, :] < NKP))


@triton.jit
def _softmax_kernel(S, Pout, SINK, NKP, H, BLOCK: tl.constexpr, HAS_SINK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    cm = c < NKP
    x = tl.load(S + r.to(tl.int64) * NKP + c, mask=cm, other=float("-inf"))
    m = tl.max(x, 0)
    if HAS_SINK:
        sk = tl.load(SINK + r % H).to(tl.float32) * 1.4426950408889634
        m = tl.maximum(m, sk)
    e = tl.exp2(x - m)
    den = tl.sum(e, 0)
    if HAS_SINK:
        den = den + tl.exp2(sk - m)
    tl.store(Pout + r.to(tl.int64) * NKP + c, (e / den).to(tl.float16), mask=cm)


@triton.jit
def _pv_kernel(Pm, V, O, M, NK, NKP, row0, klo, kbase, H, TT, D: tl.constexpr, R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, O16: tl.constexpr, HT: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    kb = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, NKP, BK):
        kk = k0 + kb
        a = tl.load(Pm + offs_m.to(tl.int64)[:, None] * NKP + kk[None, :], mask=mm[:, None] & (kk[None, :] < NKP), other=0.0)
        b = tl.load(V + (klo - kbase + kk).to(tl.int64)[:, None] * D + offs_n[None, :], mask=(kk[:, None] < NK), other=0.0)
        acc = tl.dot(a, b, acc)
    if O16:       # saturating fp16 store (the consumer GEMM takes fp16 operands: same value as clamp + half of the fp32 result)
        acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0).to(tl.float16)
    go = (row0 + offs_m).to(tl.int64)
    if HT:
        orow = (go % H) * TT + go // H
    else:
        orow = go
    tl.store(O + orow[:, None] * R + offs_n[None, :], acc, mask=mm[:, None])


BCFG = {"BM": 128, "BN": 128, "BK": 32, "warps": 8, "PBN": 128, "PBK": 32, "pwarps": 8, "S_BYTES": 48 << 20}


# measured (test_pf2, 2048 queries): sliding window H=64 best 7.9 TFLOPS at S=8 MiB; full + allow mask H=32 best 17.7 at S=16 MiB with a 64-wide key tile
BCFG_SWA = dict(BCFG, S_BYTES=8 << 20, PBN=64, pwarps=4, qs=2)       # random config sweep (gb/sweep_att.py, 8K-class shapes): 14.3 -> 16.6 TFLOPS, bit-identical output
BCFG_FULL = dict(BCFG, S_BYTES=16 << 20, BM=64, BN=64, warps=4, PBN=64, PBK=128, pwarps=4, qs=2)       # sweep: 19.6 -> 24.6 TFLOPS, bit-identical


def mla_banded(q_lat, q_pe, kc, qstart, scale, sinks=None, window=0, allow=None, cfg=None, qcat=None, o16=False, ht=False):
    """same contract as mla_flash: q_lat [T, H, R], q_pe [T, H, P], kc [N, R + P] (N = qstart + T) -> o_lat [T, H, R] fp32.
    qcat (fp16 [T, H, R + P], already saturated; q_lat = its first R columns, q_pe is None) skips the fp32 concat; o16 returns o_lat as saturated fp16."""
    c = cfg or (BCFG_SWA if window else BCFG_FULL)
    if qcat is not None:
        if ht:
            H, T, D = qcat.shape
        else:
            T, H, D = qcat.shape
        R = q_lat
        Pd = D - R
    else:
        T, H, R = q_lat.shape
        Pd = q_pe.shape[-1]
        D = R + Pd
    N = kc.shape[0]
    assert N == qstart + T
    dev = kc.device
    klo_all = max(0, qstart - window + 1) if window else 0
    kc16 = kc[klo_all:].clamp(-65504.0, 65504.0).half().contiguous()
    if qcat is None:
        qcat = torch.cat([q_lat, q_pe], -1).clamp(-65504.0, 65504.0).half().reshape(T * H, D).contiguous()
    else:
        qcat = qcat.reshape(T * H, D)
    o = torch.empty(*((H, T, R) if ht else (T, H, R)), dtype=torch.float16 if o16 else torch.float32, device=dev)
    al = allow.to(torch.int8).contiguous() if allow is not None else qcat
    sk = sinks.float().contiguous() if sinks is not None else qcat
    nk_max = min(N, (window + 0) if window else N) + 0
    t0 = 0
    sbuf = pbuf = None
    while t0 < T:
        # block of Bq positions: S [Bq*H, nkp] fp32 must stay under S_BYTES
        last_pos0 = qstart + t0
        nk_guess = (min(last_pos0 + 1, window + 0 if window else last_pos0 + 1))
        Bq = 1
        while Bq < T - t0:
            nb = Bq * 2
            hi_ = min(N, qstart + t0 + nb)
            lo_ = max(0, qstart + t0 - window + 1) if window else 0
            nkp_ = ((hi_ - lo_ + 63) // 64) * 64
            if nb * H * nkp_ * 4 > c["S_BYTES"]:
                break
            Bq = nb
        Bq = min(Bq, T - t0)
        hi = min(N, qstart + t0 + Bq)
        lo = max(0, qstart + t0 - window + 1) if window else 0
        NK = hi - lo
        NKP = ((NK + 63) // 64) * 64
        M = Bq * H
        need = M * NKP
        if sbuf is None or sbuf.numel() < need:
            sbuf = torch.empty(need, dtype=torch.float32, device=dev)
            pbuf = torch.empty(need, dtype=torch.float16, device=dev)
        row0 = t0 * H
        _qk_kernel[(triton.cdiv(M, c["BM"]), triton.cdiv(NKP, c["BN"]))](qcat, kc16, sbuf, al, M, NK, NKP, row0, t0, qstart, lo, klo_all, N, H, float(scale) * 1.4426950408889634, T,
                                                                          D=D, WIN=int(window), HAS_ALLOW=allow is not None, BM=c["BM"], BN=c["BN"], BK=c["BK"], HT=ht,
                                                                          num_warps=c["warps"], num_stages=c.get("qs", 1))
        _softmax_kernel[(M,)](sbuf, pbuf, sk, NKP, H, BLOCK=triton.next_power_of_2(NKP), HAS_SINK=sinks is not None, num_warps=4)
        _pv_kernel[(triton.cdiv(M, c["BM"]), triton.cdiv(R, c["PBN"]))](pbuf, kc16, o, M, NK, NKP, row0, lo, klo_all, H, T, D=D, R=R, BM=c["BM"], BN=c["PBN"], BK=c["PBK"], O16=o16, HT=ht,
                                                                         num_warps=c["pwarps"], num_stages=c.get("qs", 1))
        t0 += Bq
    return o
