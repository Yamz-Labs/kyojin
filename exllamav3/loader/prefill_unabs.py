"""Un-absorbed MLA prefill helpers (generic by (H, nope, rope, R, vd)): q rotary in place on the raw wq_b output, and the per-head K / V decompression
weight  W [R, H * S]  with slot S = 512 per head = [k_nope (nope) | pad to 256 (filled with the shared rope key) | v (vd)], so ONE GEMM
latent [N, R] @ W gives K / V as strided views (row stride H * S, head stride S) that the CK fmha reads without a copy."""
import os
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


@triton.jit
def _qrope_hm_kernel(Q, OUT, COS, SIN, T, H, HD: tl.constexpr, NOPE: tl.constexpr, P: tl.constexpr, HB: tl.constexpr):
    """_qrope_kernel writing HEAD-MAJOR into OUT [H, T, HD] (same values, same rounding points; no extra pass)"""
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
    ob = (h.to(tl.int64)[:, None] * T + t) * HD
    tl.store(OUT + ob + c[None, :], o.to(tl.float16), mask=hm)


def qo_head_major():
    """HY4_PF_QO_HM=1 (read live, default off): the CK fmha reads Q and writes O head-major ([H, T, hd], seen as a [T, H, hd] view): a tile of rows is one contiguous
    run instead of a 32 KB-stride gather (sliding-window layer, 8K rows: 22 -> 13.7 ms); the rope kernel writes Q head-major and the gate kernel reads O head-major,
    so no extra pass"""
    return os.environ.get("HY4_PF_QO_HM", "0") == "1"


def qrope_hm(q, cos, sin, H, hd, nope, P):
    """q [T, H * hd] raw fp16 -> [T, H, hd] VIEW of a head-major [H, T, hd] buffer, rotary applied (qrope_inplace maths)"""
    T = q.shape[0]
    HB = 8
    out = torch.empty(H, T, hd, dtype=torch.float16, device=q.device)
    _qrope_hm_kernel[(T, triton.cdiv(H, HB))](q, out, cos, sin, T, H, HD=hd, NOPE=nope, P=P, HB=HB, num_warps=4)
    return out.permute(1, 0, 2)


@triton.jit
def _gate_hm_kernel(O, G, OUT, T, n, H: tl.constexpr, VD: tl.constexpr, BLK: tl.constexpr):
    """OUT [T, H * VD] = O (head-major [H, T, VD]) * sigmoid(G [T, H * VD]); the maths of prefill_kernels._gate_ew_kernel"""
    i = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
    m = i < n
    g = tl.load(G + i, mask=m, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, tl.minimum(tl.maximum(g, -65504.0), 65504.0))
    t = i // (H * VD)
    cc = i % (H * VD)
    oi = (cc // VD) * T * VD + t * VD + cc % VD
    kv = tl.load(O + oi, mask=m, other=0.0).to(tl.float32)
    o = kv * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(OUT + i, tl.minimum(tl.maximum(o, -65504.0), 65504.0).to(tl.float16), mask=m)


def gate16_hm(o, g, H, VD):
    """o [T, H, VD] view of a head-major buffer, g [T, H * VD] raw gate GEMM output -> gated [T, H * VD] fp16"""
    T = o.shape[0]
    out = torch.empty(T, H * VD, dtype=torch.float16, device=o.device)
    n = out.numel()
    _gate_hm_kernel[(triton.cdiv(n, 4096),)](o, g, out, T, n, H=H, VD=VD, BLK=4096, num_warps=4)
    return out


def qrope_inplace(q, cos, sin, H, hd, nope, P):
    T = q.shape[0]
    HB = 8
    _qrope_kernel[(T, triton.cdiv(H, HB))](q, cos, sin, H, HD=hd, NOPE=nope, P=P, HB=HB, num_warps=4)
    return q


def kv_head_major():
    """HY4_PF_KV_HM=1 (read live, default off): K / V decompressed head-major ([H, N, slot], one batched GEMM) instead of [N, H, slot]: the CK fmha then reads a
    tile of keys as one contiguous run (the [N, H, 512] slab has a 64 KB row stride: channel camping, 33 -> 22 ms per sliding-window layer at 8K, dense 180 -> 104 ms);
    same values (bit-equal outputs), same footprint (the weight is kept in ONE layout, chosen when it is built)"""
    return os.environ.get("HY4_PF_KV_HM", "0") == "1"


def kv_weight(wk16, wv16, slot=512, kpad=256):
    """wk16 [H, R, nope] fp16, wv16 [H, vd, R] fp16 -> W [R, H * slot] fp16: head h columns [0:nope] = wk16[h], [kpad:kpad + vd] = wv16[h].T, rest 0
    (HY4_PF_KV_HM=1: W [H, R, slot], the batched-GEMM layout of kv_decompress)"""
    H, R, nope = wk16.shape
    vd = wv16.shape[1]
    W = torch.zeros(R, H, slot, dtype=torch.float16, device=wk16.device)
    W[:, :, :nope] = wk16.permute(1, 0, 2)
    W[:, :, kpad:kpad + vd] = wv16.permute(2, 0, 1)
    if kv_head_major():
        return W.permute(1, 0, 2).contiguous()
    return W.reshape(R, H * slot)


def kv_decompress(lat16, pe16, W, H, hd=256, slot=512):
    """lat16 [N, R] fp16, pe16 [N, P] fp16 (rope already applied), W from kv_weight -> (K [N, H, hd] view, V [N, H, hd] view) of one [N, H, slot] slab;
    the rope key is broadcast over heads into the pad columns of K"""
    N = lat16.shape[0]
    P = pe16.shape[1]
    if W.dim() == 3:        # head-major (HY4_PF_KV_HM): slab [H, N, slot]; K / V are [N, H, hd] views with row stride slot, head stride N * slot
        slab = torch.bmm(lat16.unsqueeze(0).expand(H, N, lat16.shape[1]), W)
        slab[:, :, hd - P:hd] = pe16[None]
        return slab[:, :, :hd].permute(1, 0, 2), slab[:, :, hd:hd + hd].permute(1, 0, 2)
    slab = torch.mm(lat16, W).view(N, H, slot)
    slab[:, :, hd - P:hd] = pe16[:, None, :]
    return slab[:, :, :hd], slab[:, :, hd:hd + hd]
