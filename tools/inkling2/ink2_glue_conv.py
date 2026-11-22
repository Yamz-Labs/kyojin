"""Inkling 2 Mini fused short-conv glue op (Triton, gfx1151; the same file runs on CPU through TRITON_INTERPRET=1).

ONE launch replaces, per application and per stream (K, V, attn-out, ffn-out), the torch chain
    cat(hist, x) / hist update / zeros / 4 x (mul, add) [+ 4 x (mul, add) dynamic taps] / add / div   (13 or 21 aten launches)
and optionally the epilogue that follows it in the step:
    EPI_NONE   conv output                                  (post-attn conv at a block opener)
    EPI_ADD    residual + conv output                       (post-attn conv inside a block, post-ffn conv: `inp_sa + cur`, `ffn_inp + cur`)
    EPI_L2CLIP l2 clip over each 128-channel head           (K: x * clamp(thr / (|x| + 1e-6), 0, 1))
    EPI_RMS    rms over each 128-channel head               (V: x * rsqrt(mean(x^2) + eps))
and the output cast (fp32 / fp16 / bf16 by the dtype of `out`) and the destination row mapping: out row = (pos0 + t) % out_mod when out_mod > 0
(local K / V: write straight into the bf16 attention ring KR / VR, no lattn_append, no casts) else t.

Pre-conv history = ring `[RING = 16][n_chan]` fp32 INDEXED BY POSITION (row p % 16), not a shifted [3, n_chan] buffer: a T = 8 verify writes rows pos0 .. pos0 + 7 and
leaves the 3 history rows before pos0 intact (3 + 8 = 11 <= 16), so spec rollback = reset pos0 (device int32 scalar, HIP-graph safe; ops_lattn.md correction 21/11).
Row p is read from `x` when p >= pos0 (this call) and from the ring when p < pos0; positions < 0 read the zero ring (reset zero-fills it).
Semantics = Inkling2Forward.conv (taps oldest -> current, static w [C, 4]; dynamic dyn [T, nh, 4] with lag 0 = current), divisor sqrt((2 if dyn else 1) * 4 + 1).
T <= 8 per call.
"""
import math
import torch
import triton
import triton.language as tl

RING = 16
EPI_NONE, EPI_ADD, EPI_L2CLIP, EPI_RMS = 0, 1, 2, 3
HEAD = 128            # channels per program = one kv head (so the K / V epilogue reductions stay inside the program); post convs use the same tile (18 programs for D = 2304)


@triton.jit
def _conv_kernel(X, RINGP, W, DYN, RES, OUT, POS,
                 T, s_x, s_ring, s_res, s_dyn, s_orow, s_ohead, out_mod, thr, eps, inv,
                 BLOCK: tl.constexpr, HAS_DYN: tl.constexpr, DYN_GROUP: tl.constexpr, EPI: tl.constexpr, RING: tl.constexpr):
    pid = tl.program_id(0)
    c = pid * BLOCK + tl.arange(0, BLOCK)
    pos0 = tl.load(POS)
    w0 = tl.load(W + c * 4 + 0)
    w1 = tl.load(W + c * 4 + 1)
    w2 = tl.load(W + c * 4 + 2)
    w3 = tl.load(W + c * 4 + 3)
    for t in range(T):
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        xt = tl.load(X + t * s_x + c)
        for lag in tl.static_range(4):                       # tap j = 3 - lag multiplies position p = pos0 + t - lag
            tt = t - lag                                     # row of x when >= 0, else history position pos0 + tt
            xv = tl.load(X + tt * s_x + c, mask=(tt >= 0) & (tt < T), other=0.0)
            hr = (pos0 + tt + RING * 2) % RING
            hv = tl.load(RINGP + hr * s_ring + c, mask=(tt < 0) & (pos0 + tt >= 0), other=0.0)
            v = xv + hv
            if lag == 0:
                acc += v * w3
            elif lag == 1:
                acc += v * w2
            elif lag == 2:
                acc += v * w1
            else:
                acc += v * w0
            if HAS_DYN:
                d = tl.load(DYN + t * s_dyn + (c // DYN_GROUP) * 4 + lag)
                acc += v * d
        y = (xt + acc) * inv
        if EPI == 1:
            y += tl.load(RES + t * s_res + c)
        elif EPI == 2:
            n = tl.sqrt(tl.sum(y * y, 0)) + 1e-6
            y = y * tl.minimum(tl.maximum(thr / n, 0.0), 1.0)
        elif EPI == 3:
            y = y * tl.rsqrt(tl.sum(y * y, 0) / BLOCK + eps)
        if out_mod > 0:
            orow = (pos0 + t) % out_mod
        else:
            orow = t
        hd = c // 128
        tl.store(OUT + orow * s_orow + hd * s_ohead + (c % 128), y)
        tl.store(RINGP + ((pos0 + t) % RING) * s_ring + c, xt)    # after the reads of this row: positions needed later are >= pos0 - 3, rows written are <= pos0 + 7 - 16


def conv_fused(x, ring, w, pos0, out, dyn=None, res=None, epi=EPI_NONE, thr=0.0, eps=0.0, out_mod=0, out_strides=None, k=4):
    """x [T, C] fp32 (contiguous, C % 128 == 0); ring [16, n_chan] fp32 view starting at this stream's column (row stride = ring.stride(0)); w [C, 4] fp32;
    pos0 device int32 [1]; out: tensor whose (row, head, d) address = row * out_strides[0] + head * out_strides[1] + d (default [T, C] contiguous); dtype of out = output dtype.
    dyn [T, nh, 4] fp32 (global K / V); res [T, C] fp32 (EPI_ADD); thr (EPI_L2CLIP), eps (EPI_RMS)."""
    T, C = x.shape
    assert k == 4 and T <= 8 and C % HEAD == 0 and x.stride(1) == 1 and ring.shape[0] == RING and ring.stride(1) == 1 and w.shape == (C, 4)
    if out_strides is None:
        out_strides = (C, HEAD)
    has_dyn = dyn is not None
    nh = dyn.shape[1] if has_dyn else 1
    if has_dyn:
        assert dyn.stride(1) == 4 and dyn.stride(2) == 1 and dyn.shape[2] == 4, "dyn must be [T, nh, 4] with head stride 4, lag stride 1"
    inv = 1.0 / math.sqrt((2 if has_dyn else 1) * k + 1)
    if epi == EPI_ADD:
        assert res is not None and res.stride(1) == 1
    _conv_kernel[(C // HEAD,)](
        x, ring, w, dyn if has_dyn else x, res if res is not None else x, out, pos0,
        T, x.stride(0), ring.stride(0), res.stride(0) if res is not None else 0, dyn.stride(0) if has_dyn else 0,
        out_strides[0], out_strides[1], out_mod, float(thr), float(eps), inv,
        BLOCK=HEAD, HAS_DYN=has_dyn, DYN_GROUP=C // nh, EPI=epi, RING=RING, num_warps=1)
    return out


def conv_ref(x, ring, w, pos0, dyn=None, res=None, epi=EPI_NONE, thr=0.0, eps=0.0, k=4):
    """torch reference of the same op on the same ring state (updates `ring` in place like the kernel). Returns the fp32 [T, C] result before the cast."""
    T, C = x.shape
    p0 = int(pos0)
    hist = torch.stack([ring[(p0 - (k - 1) + i) % RING] if p0 - (k - 1) + i >= 0 else torch.zeros(C) for i in range(k - 1)], 0)
    ext = torch.cat([hist, x], 0)
    out = torch.zeros_like(x)
    for j in range(k):
        out = out + ext[j:j + T] * w[:, j]
    if dyn is not None:
        nh = dyn.shape[1]
        for lag in range(k):
            win = ext[k - 1 - lag:k - 1 - lag + T].reshape(T, nh, C // nh)
            out = out + (win * dyn[:, :, lag:lag + 1]).reshape(T, C)
    y = (x + out) / math.sqrt((2 if dyn is not None else 1) * k + 1)
    for t in range(T):
        ring[(p0 + t) % RING] = x[t]
    if epi == EPI_ADD:
        y = res + y
    elif epi == EPI_L2CLIP:
        yh = y.reshape(T, C // HEAD, HEAD)
        y = (yh * (thr / (torch.linalg.vector_norm(yh, dim=-1, keepdim=True) + 1e-6)).clamp(0.0, 1.0)).reshape(T, C)
    elif epi == EPI_RMS:
        yh = y.reshape(T, C // HEAD, HEAD)
        y = (yh * torch.rsqrt(yh.square().mean(-1, keepdim=True) + eps)).reshape(T, C)
    return y


# ------------------------------------------------------------------------------------------------------------------ step interface (INK_GLUE_FUSE=conv)
import os
FUSE = {s for s in os.environ.get("INK_GLUE_FUSE", "").replace("all", "conv").split(",") if s}
FIRES = {"conv": 0}


class ConvFuse:
    """Per-application pre-conv rings by position + the four conv call sites of the step. The forward owns `pos0` (the device int32 scalar `gl.pos0`, set by gl.begin) and calls:
        cf.kv(il, off, x, w, dyn, kind, out, out_mod=0, out_strides=None)   K (kind 'k', l2 clip) / V (kind 'v', rms); `out` = [T, C] fp32 / fp16, or the bf16 ring KR / VR of a local slot (out_mod = 1088)
        cf.post(il, off, x, w, res, out)                                    post-attn / post-ffn conv (+ residual add when res is not None)
    Same channel offsets as slot.hist (k 0, v kvd, attn-post 2 kvd, ffn-post 2 kvd + D); the rings replace slot.hist."""
    def __init__(self, n_app, n_chan, device, pos0, qk_clip, eps):
        self.rings = [torch.zeros(RING, n_chan, device=device) for _ in range(n_app)]
        self.pos0, self.qk_clip, self.eps = pos0, qk_clip, eps

    def reset(self):
        for r in self.rings:
            r.zero_()

    def kv(self, il, off, x, w, dyn, kind, out, out_mod=0, out_strides=None):
        FIRES["conv"] += 1
        C = x.shape[1]
        return conv_fused(x, self.rings[il][:, off:off + C], w, self.pos0, out, dyn=dyn, epi=EPI_L2CLIP if kind == "k" else EPI_RMS,
                          thr=self.qk_clip, eps=self.eps, out_mod=out_mod, out_strides=out_strides)

    def post(self, il, off, x, w, res, out):
        FIRES["conv"] += 1
        C = x.shape[1]
        return conv_fused(x, self.rings[il][:, off:off + C], w, self.pos0, out, res=res, epi=EPI_ADD if res is not None else EPI_NONE)
