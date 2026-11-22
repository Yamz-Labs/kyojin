"""Inkling 2 Mini fused controller + rms-norm glue ops (Triton, gfx1151; same file runs on CPU with TRITON_INTERPRET=1).

control(cur, msg) = (1 - tanh(F h)) * cur + tanh(G h) * msg,  h = Down [rms(cur); rms(msg)]   (Down [64, 4608], G / F [2304, 64], rms eps 1e-6)
torch chain = 20 aten launches per call (2 x rms 10, cat, 3 GEMV, 2 tanh, rsub, 2 mul, add), 96 calls per token (24 applications x 4 controllers), 2.36 MB of fp32 weights per call.
Fused = 2 launches:
  _down_kernel  grid 64 (one program per row of Down): both row sums of squares, the dot with the raw rows, the rms scale applied to the dot -> h [T, 64] fp32 scratch
  _out_kernel   grid 2304 / BD: tanh gates from the G / F tiles and h, combine with cur / msg -> out
`rms_w(x, w, out, eps)` = rms(x) * w in ONE launch (6 aten launches; 104 input norms + 48 ffn norms per token), also used as the epilogue launch after `control`.
Weights may be fp32 (default), fp16 or bf16 COPIES made at load (`to_ctrl_dtype`): the kernels convert to fp32 on load, so the 227 MB / token of controller weights halve. Gate: relL2 of the output vs the fp32 path (micro job).
T <= 8 rows per call. No host sync, no allocation inside a call when `scratch` is given: graph safe.
"""
import torch
import triton
import triton.language as tl

R = 64            # controller rank
BLK = 4096        # power of two >= D (2304), masked


@triton.jit
def _tanh(x):
    e = tl.exp(-2.0 * tl.abs(x))
    t = (1.0 - e) / (1.0 + e)
    return tl.where(x < 0, -t, t)


@triton.jit
def _down_kernel(CUR, MSG, WD, H, T, s_cur, s_msg, eps, D: tl.constexpr, BLOCK: tl.constexpr, RK: tl.constexpr):
    r = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    km = k < D
    dc = tl.load(WD + r * 2 * D + k, mask=km, other=0.0).to(tl.float32)
    dm = tl.load(WD + r * 2 * D + D + k, mask=km, other=0.0).to(tl.float32)
    for t in range(T):
        xc = tl.load(CUR + t * s_cur + k, mask=km, other=0.0)
        xm = tl.load(MSG + t * s_msg + k, mask=km, other=0.0)
        rc = tl.rsqrt(tl.sum(xc * xc, 0) / D + eps)
        rm = tl.rsqrt(tl.sum(xm * xm, 0) / D + eps)
        tl.store(H + t * RK + r, tl.sum(xc * dc, 0) * rc + tl.sum(xm * dm, 0) * rm)


@triton.jit
def _out_kernel(CUR, MSG, WG, WF, H, OUT, T, s_cur, s_msg, s_out, BD: tl.constexpr, RK: tl.constexpr):
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    r = tl.arange(0, RK)
    g_tile = tl.load(WG + d[:, None] * RK + r[None, :]).to(tl.float32)
    f_tile = tl.load(WF + d[:, None] * RK + r[None, :]).to(tl.float32)
    for t in range(T):
        h = tl.load(H + t * RK + r)
        g = _tanh(tl.sum(g_tile * h[None, :], 1))
        f = _tanh(tl.sum(f_tile * h[None, :], 1))
        c = tl.load(CUR + t * s_cur + d)
        m = tl.load(MSG + t * s_msg + d)
        tl.store(OUT + t * s_out + d, (1.0 - f) * c + g * m)


@triton.jit
def _rmsw_kernel(X, W, OUT, s_x, s_out, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    km = k < D
    x = tl.load(X + t * s_x + k, mask=km, other=0.0)
    w = tl.load(W + k, mask=km, other=0.0)
    r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    tl.store(OUT + t * s_out + k, x * r * w, mask=km)


def to_ctrl_dtype(w, dtype):
    """one-time copy of a controller matrix to the storage dtype (fp32 / fp16 / bf16), contiguous"""
    return w.to(dtype).contiguous()


def control_fused(cur, msg, wdown, wgain, wforget, out, scratch=None, eps=1e-6):
    """cur / msg [T, D] fp32 contiguous; wdown [64, 2D], wgain / wforget [D, 64] (fp32 / fp16 / bf16); out [T, D] fp32; scratch [8, 64] fp32 (allocated here when None). Returns out."""
    T, D = cur.shape
    assert T <= 8 and wdown.shape == (R, 2 * D) and wgain.shape == (D, R) and wforget.shape == (D, R) and D % 32 == 0
    assert cur.stride(1) == 1 and msg.stride(1) == 1 and out.stride(1) == 1
    h = scratch if scratch is not None else torch.empty(8, R, dtype=torch.float32, device=cur.device)
    _down_kernel[(R,)](cur, msg, wdown, h, T, cur.stride(0), msg.stride(0), eps, D=D, BLOCK=BLK, RK=R, num_warps=4)
    BD = 32
    _out_kernel[(D // BD,)](cur, msg, wgain, wforget, h, out, T, cur.stride(0), msg.stride(0), out.stride(0), BD=BD, RK=R, num_warps=2)
    return out


def rms_w(x, w, out, eps):
    T, D = x.shape
    assert T <= 8 and x.stride(1) == 1 and w.shape == (D,) and out.stride(1) == 1
    _rmsw_kernel[(T,)](x, w, out, x.stride(0), out.stride(0), eps, D=D, BLOCK=BLK, num_warps=4)
    return out


def control_ref(cur, msg, wdown, wgain, wforget, eps=1e-6):
    rms = lambda x: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    h = torch.cat([rms(cur), rms(msg)], -1) @ wdown.float().T
    return (1.0 - torch.tanh(h @ wforget.float().T)) * cur + torch.tanh(h @ wgain.float().T) * msg


def rms_w_ref(x, w, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w
