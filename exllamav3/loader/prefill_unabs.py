"""Un-absorbed MLA prefill helpers (generic by (H, nope, rope, R, vd)): q rotary in place on the raw wq_b output, and the per-head K / V decompression
weight  W [R, H * S]  with slot S = 512 per head = [k_nope (nope) | pad to 256 (filled with the shared rope key) | v (vd)], so ONE GEMM
latent [N, R] @ W gives K / V as strided views (row stride H * S, head stride S) that the CK fmha reads without a copy."""
import torch
import triton
import triton.language as tl


@triton.jit
def _qrope_kernel(Q, COS, SIN, H, HD: tl.constexpr, NOPE: tl.constexpr, P: tl.constexpr, HB: tl.constexpr):
    """one program = HB heads of one row t of the raw fp16 wq_b output [T, H * HD], in place, coalesced [HB, HD] tile: nan -> 0 and saturate everything,
    rotary (consecutive pairs) on the last P columns: out[c] = x[c] * cos + (c even ? -x[c + 1] : x[c - 1]) * sin (same rounding points as rope_norm + clamp)"""
    t = tl.program_id(0)
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    c = tl.arange(0, HD)
    hm = (h < H)[:, None]
    base = (t.to(tl.int64) * H + h)[:, None] * HD
    x = tl.load(Q + base + c[None, :], mask=hm, other=0.0).to(tl.float32)
    x = tl.where(x != x, 0.0, tl.minimum(tl.maximum(x, -65504.0), 65504.0))
    xp = tl.load(Q + base + (c ^ 1)[None, :], mask=hm, other=0.0).to(tl.float32)
    xp = tl.where(xp != xp, 0.0, tl.minimum(tl.maximum(xp, -65504.0), 65504.0))
    pe = c >= NOPE
    j = tl.where(pe, (c - NOPE) // 2, 0)
    cs = tl.load(COS + t * (P // 2) + j)
    sn = tl.load(SIN + t * (P // 2) + j)
    sg = tl.where((c % 2) == 0, -sn, sn)
    r = tl.minimum(tl.maximum(x * cs[None, :] + xp * sg[None, :], -65504.0), 65504.0)
    o = tl.where(pe[None, :], r, x)
    tl.store(Q + base + c[None, :], o.to(tl.float16), mask=hm)


def qrope_inplace(q, cos, sin, H, hd, nope, P):
    T = q.shape[0]
    HB = 8
    _qrope_kernel[(T, triton.cdiv(H, HB))](q, cos, sin, H, HD=hd, NOPE=nope, P=P, HB=HB, num_warps=4)
    return q


def kv_weight(wk16, wv16, slot=512, kpad=256):
    """wk16 [H, R, nope] fp16, wv16 [H, vd, R] fp16 -> W [R, H * slot] fp16: head h columns [0:nope] = wk16[h], [kpad:kpad + vd] = wv16[h].T, rest 0"""
    H, R, nope = wk16.shape
    vd = wv16.shape[1]
    W = torch.zeros(R, H, slot, dtype=torch.float16, device=wk16.device)
    W[:, :, :nope] = wk16.permute(1, 0, 2)
    W[:, :, kpad:kpad + vd] = wv16.permute(2, 0, 1)
    return W.reshape(R, H * slot)


def kv_decompress(lat16, pe16, W, H, hd=256, slot=512):
    """lat16 [N, R] fp16, pe16 [N, P] fp16 (rope already applied), W from kv_weight -> (K [N, H, hd] view, V [N, H, hd] view) of one [N, H, slot] slab;
    the rope key is broadcast over heads into the pad columns of K"""
    N = lat16.shape[0]
    P = pe16.shape[1]
    slab = torch.mm(lat16, W).view(N, H, slot)
    slab[:, :, hd - P:hd] = pe16[:, None, :]
    return slab[:, :, :hd], slab[:, :, hd:hd + hd]
