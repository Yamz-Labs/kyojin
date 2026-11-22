"""ink2_glue_clip: l2 clip of Q (per-dim scale) and R (per-dim scale) + optional tau (global layers) in ONE launch, replacing ~7 aten launches per tensor
(mul, vector_norm, add, reciprocal / div, mul, clamp, mul, [mul tau]) = ~700 launches per decode token.
Op class: per-head row clip  y = x * scale;  y *= clamp(thr / (|y|_2 + 1e-6), 0, 1)  [* tau[t]].  Same fp32 math as ink2 forward l2_clip.
Design note: Loom fuses the same prep (g_ink_qr_prep: clip + rope-free prep of q / r in one kernel). Generic HIP / CUDA Triton, any T (grid T x heads).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _clip_row(X, S, OUT, thr, tau, D: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.arange(0, BLOCK)
    m = k < D
    x = tl.load(X + k, mask=m, other=0.0)
    s = tl.load(S + k, mask=m, other=1.0)
    y = x * s
    n = tl.sqrt(tl.sum(y * y, 0)) + 1e-6
    f = tl.minimum(tl.maximum(thr / n, 0.0), 1.0)
    tl.store(OUT + k, (y * f) * tau, mask=m)


@triton.jit
def _qr_clip_kernel(Q, R, SQ, SR, OQ, OR, TAU, thr_q, thr_r, s_sq, s_sr, HAS_TAU: tl.constexpr,
                    NH: tl.constexpr, HD: tl.constexpr, NREL: tl.constexpr, BQ: tl.constexpr, BR: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    tau = 1.0
    if HAS_TAU:
        tau = tl.load(TAU + t)
    _clip_row(Q + (t * NH + h) * HD, SQ + h * s_sq, OQ + (t * NH + h) * HD, thr_q, tau, HD, BQ)
    _clip_row(R + (t * NH + h) * NREL, SR + h * s_sr, OR + (t * NH + h) * NREL, thr_r, tau, NREL, BR)


def qr_clip_ref(Q, R, sq, sr, thr_q, thr_r, nh, hd, nrel, tau=None):
    def clip(x, s, thr):
        x = x * s
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-6
        return x * (thr / n).clamp(0.0, 1.0)
    T = Q.shape[0]
    q = clip(Q.reshape(T, nh, hd), sq.reshape(-1, hd), thr_q)
    r = clip(R.reshape(T, nh, nrel), sr.reshape(-1, nrel), thr_r)
    if tau is not None:
        q, r = q * tau.reshape(T, 1, 1), r * tau.reshape(T, 1, 1)
    return q, r


def qr_clip(Q, R, sq, sr, thr_q, thr_r, nh, hd, nrel, tau=None, out_q=None, out_r=None):
    """Q [T, nh*hd], R [T, nh*nrel] fp32 contiguous; sq / sr per-dim scales ([hd] or [nh, hd] / [nrel] or [nh, nrel], fp32); tau [T] fp32 or None -> (q [T, nh, hd], r [T, nh, nrel]) fp32"""
    T = Q.shape[0]
    Q, R = Q.contiguous(), R.contiguous()
    sq, sr = sq.float().contiguous(), sr.float().contiguous()
    oq = out_q if out_q is not None else torch.empty(T, nh, hd, dtype=torch.float32, device=Q.device)
    orr = out_r if out_r is not None else torch.empty(T, nh, nrel, dtype=torch.float32, device=Q.device)
    s_sq = 0 if sq.numel() == hd else hd
    s_sr = 0 if sr.numel() == nrel else nrel
    _qr_clip_kernel[(T, nh)](Q, R, sq, sr, oq, orr, tau if tau is not None else Q, thr_q, thr_r, s_sq, s_sr, HAS_TAU=tau is not None,
                             NH=nh, HD=hd, NREL=nrel, BQ=triton.next_power_of_2(hd), BR=triton.next_power_of_2(nrel), num_warps=1)
    return oq, orr
