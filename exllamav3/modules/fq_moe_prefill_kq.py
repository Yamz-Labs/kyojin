"""Grouped MoE prefill for K-quant experts (Q4_K / Q5_K / Q6_K): the weights are decoded inside the K loop of a Triton grouped
GEMM over expert-sorted rows (no fp16 expert scratch, no per-expert launches). Same fp16-rounded weights and fp32 accumulation
as the per-expert dequant + matmul path; only the K summation order differs. Tile map: fq_moe_prefill.build_tiles."""
import os
import torch
import triton
import triton.language as tl

FMT_Q4K, FMT_Q5K, FMT_Q6K = 0, 1, 2
PF_BM = int(os.environ.get("MCPM_FQ_PF_BM", "64"))   # rows per tile: weight decode is shared by BM rows
BLOCK_BYTES = {FMT_Q4K: 144, FMT_Q5K: 176, FMT_Q6K: 210}


@triton.jit
def _f16(blk, off, nmask):
    b0 = tl.load(blk + off, mask=nmask, other=0).to(tl.uint16)
    b1 = tl.load(blk + off + 1, mask=nmask, other=0).to(tl.uint16)
    return (b0 | (b1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _k4_scales(blk, jj, nmask):
    """get_scale_min_k4 for sub-block indices jj [4] of every row in the tile: (sc, m), each [BN, 4] fp32."""
    sa = tl.load(blk[:, None] + 4 + jj[None, :], mask=nmask[:, None], other=0).to(tl.int32)
    sb = tl.load(blk[:, None] + 4 + (jj + 4)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
    sc_ = tl.load(blk[:, None] + 4 + tl.maximum(jj - 4, 0)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
    lowj = (jj < 4)[None, :]
    sc = tl.where(lowj, sa & 63, (sb & 15) | ((sc_ >> 6) << 4))
    m = tl.where(lowj, sb & 63, (sb >> 4) | ((sa >> 6) << 4))
    return sc.to(tl.float32), m.to(tl.float32)


@triton.jit
def _spread32(x, BN: tl.constexpr):
    """[BN, 4] per-sub-block values -> [BN, 128], each repeated over its 32 lanes."""
    return tl.reshape(tl.broadcast_to(x[:, :, None], (BN, 4, 32)), (BN, 128))


@triton.jit
def _q45k_sb(blk, nmask, HAS_QH: tl.constexpr, BN: tl.constexpr):
    """One Q4_K / Q5_K super-block per row: (wl, wh) [128, BN] fp16. wl[c] = low-nibble element of qs byte c = K index
    64*(c//32) + c%32 in the super-block, wh[c] = high-nibble element, K index + 32. Op order as fq_dq_q4_k_kernel."""
    c = tl.arange(0, 128)
    g4 = tl.arange(0, 4)
    d = _f16(blk, 0, nmask)
    dmin = _f16(blk, 2, nmask)
    qs_off = 48 if HAS_QH else 16
    q = tl.load(blk[:, None] + qs_off + c[None, :], mask=nmask[:, None], other=0).to(tl.int32)
    lo = q & 15
    hi = q >> 4
    if HAS_QH:
        qh = tl.load(blk[:, None] + 16 + (c & 31)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
        lo = lo + (((qh >> (2 * (c >> 5))[None, :]) & 1) << 4)
        hi = hi + (((qh >> (2 * (c >> 5) + 1)[None, :]) & 1) << 4)
    scl, ml = _k4_scales(blk, 2 * g4, nmask)
    sch, mh = _k4_scales(blk, 2 * g4 + 1, nmask)
    d1l = _spread32(d[:, None] * scl, BN)
    d1h = _spread32(d[:, None] * sch, BN)
    m1l = _spread32(dmin[:, None] * ml, BN)
    m1h = _spread32(dmin[:, None] * mh, BN)
    wl = (d1l * lo.to(tl.float32) - m1l).to(tl.float16)
    wh = (d1h * hi.to(tl.float32) - m1h).to(tl.float16)
    return tl.trans(wl), tl.trans(wh)


@triton.jit
def _q6k_half(blk, h, nmask, BN: tl.constexpr):
    """Half h (0/1) of a Q6_K super-block per row: w [128, BN] fp16 in natural K order (K = 128*h + c). Op order as
    fq_dq_q6_k_kernel: (d * sc) * (q - 32)."""
    c = tl.arange(0, 128)
    grp = c >> 5
    l = c & 31
    d = _f16(blk, 208, nmask)
    ql = tl.load(blk[:, None] + (64 * h + l + 32 * (grp & 1))[None, :], mask=nmask[:, None], other=0).to(tl.int32)
    qh = tl.load(blk[:, None] + (128 + 32 * h + l)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
    q = tl.where((grp >> 1)[None, :] != 0, ql >> 4, ql & 15) | (((qh >> (2 * grp)[None, :]) & 3) << 4)
    sraw = tl.load(blk[:, None] + (192 + 8 * h + 2 * grp + (l >> 4))[None, :], mask=nmask[:, None], other=0)
    s = sraw.to(tl.int8, bitcast=True).to(tl.float32)
    w = ((d[:, None] * s) * (q - 32).to(tl.float32)).to(tl.float16)
    return tl.trans(w)


@triton.jit
def _gmm_kq(a_ptr, wptrs, out_ptr, tile_e, tile_r0, offs_ptr, wrow_ptr,
            N, K, stride_am, stride_om, E,
            FMT: tl.constexpr, SWIGLU: tl.constexpr, SCALE_ROWS: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    """out[r, n] = sum_k A[r, k] * W_e[n, k] for the rows of tile t's expert (A in natural K order). SWIGLU: wptrs holds
    gate ptrs [0:E] and up ptrs [E:2E]; out = silu(g) * u with g, u rounded to fp16 first. SCALE_ROWS: multiply the fp16
    result by wrow[r] (routing weight)."""
    BB: tl.constexpr = 144 if FMT == 0 else (176 if FMT == 1 else 210)
    pid_n = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.load(tile_e + t)
    if e >= E:
        return
    r0 = tl.load(tile_r0 + t)
    r_end = tl.load(offs_ptr + e)
    offs_m = r0 + tl.arange(0, BM)
    mmask = offs_m < r_end
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N
    c = tl.arange(0, 128)
    kb = K // 256
    wb = tl.load(wptrs + e).to(tl.pointer_type(tl.uint8))
    bp_n = wb + offs_n.to(tl.int64) * (kb * BB)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if SWIGLU:
        wb2 = tl.load(wptrs + E + e).to(tl.pointer_type(tl.uint8))
        bp_n2 = wb2 + offs_n.to(tl.int64) * (kb * BB)
        acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    a_row = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am
    for sb in range(0, kb):
        if FMT == 2:
            for h in range(0, 2):
                ah = tl.load(a_row + (sb * 256 + 128 * h + c)[None, :], mask=mmask[:, None], other=0.0)
                acc = tl.dot(ah, _q6k_half(bp_n + sb * BB, h, nmask, BN), acc)
                if SWIGLU:
                    acc2 = tl.dot(ah, _q6k_half(bp_n2 + sb * BB, h, nmask, BN), acc2)
        else:
            ka = sb * 256 + 64 * (c >> 5) + (c & 31)
            al = tl.load(a_row + ka[None, :], mask=mmask[:, None], other=0.0)
            ah = tl.load(a_row + (ka + 32)[None, :], mask=mmask[:, None], other=0.0)
            wl, wh = _q45k_sb(bp_n + sb * BB, nmask, FMT == 1, BN)
            acc = tl.dot(al, wl, acc)
            acc = tl.dot(ah, wh, acc)
            if SWIGLU:
                wl2, wh2 = _q45k_sb(bp_n2 + sb * BB, nmask, FMT == 1, BN)
                acc2 = tl.dot(al, wl2, acc2)
                acc2 = tl.dot(ah, wh2, acc2)
    if SWIGLU:
        g = acc.to(tl.float16).to(tl.float32)
        u = acc2.to(tl.float16).to(tl.float32)
        res = (g / (1.0 + tl.exp(-g)) * u).to(tl.float16)
    else:
        res = acc.to(tl.float16)
        if SCALE_ROWS:
            wr = tl.load(wrow_ptr + offs_m, mask=mmask, other=0.0).to(tl.float32)
            res = (res.to(tl.float32) * wr[:, None]).to(tl.float16)
    o = out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
    tl.store(o, res, mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _dq_kq(ptrs, out, e0, N, K, FMT: tl.constexpr, BN: tl.constexpr):
    """Decode expert e0 + program_id(2) (row tile program_id(0), super-block program_id(1)) into fp16 out[e, n, k]."""
    BB: tl.constexpr = 144 if FMT == 0 else (176 if FMT == 1 else 210)
    pid_n = tl.program_id(0)
    sb = tl.program_id(1)
    ec = tl.program_id(2)
    kb = K // 256
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N
    wb = tl.load(ptrs + e0 + ec).to(tl.pointer_type(tl.uint8))
    blk = wb + offs_n.to(tl.int64) * (kb * BB) + sb * BB
    c = tl.arange(0, 128)
    o = out + (ec.to(tl.int64) * N + offs_n.to(tl.int64))[None, :] * K + sb * 256     # [1, BN] row starts
    if FMT == 2:
        for h in range(0, 2):
            w = _q6k_half(blk, h, nmask, BN)                                          # [128, BN]
            tl.store(o + (128 * h + c)[:, None], w, mask=nmask[None, :])
    else:
        wl, wh = _q45k_sb(blk, nmask, FMT == 1, BN)
        ka = 64 * (c >> 5) + (c & 31)
        tl.store(o + ka[:, None], wl, mask=nmask[None, :])
        tl.store(o + (ka + 32)[:, None], wh, mask=nmask[None, :])


@triton.jit
def _gmm_f16(a_ptr, w_ptr, w2_ptr, out_ptr, tile_e, tile_r0, offs_ptr, wrow_ptr,
             N, K, stride_am, stride_om, e_lo, e_hi,
             SWIGLU: tl.constexpr, SCALE_ROWS: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Like _gmm_kq over fp16 expert weights w[(e - e_lo), n, k] (a chunk of experts [e_lo, e_hi)); SWIGLU also reads w2."""
    pid_n = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.load(tile_e + t)
    if (e < e_lo) | (e >= e_hi):
        return
    r0 = tl.load(tile_r0 + t)
    r_end = tl.load(offs_ptr + e)
    offs_m = r0 + tl.arange(0, BM)
    mmask = offs_m < r_end
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N
    kk = tl.arange(0, BK)
    wbase = (e - e_lo).to(tl.int64) * N * K
    a_row = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am
    w_col = w_ptr + wbase + offs_n.to(tl.int64)[None, :] * K
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if SWIGLU:
        w2_col = w2_ptr + wbase + offs_n.to(tl.int64)[None, :] * K
        acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_row + (k + kk)[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(w_col + (k + kk)[:, None], mask=nmask[None, :], other=0.0)
        acc = tl.dot(a, w, acc)
        if SWIGLU:
            w2 = tl.load(w2_col + (k + kk)[:, None], mask=nmask[None, :], other=0.0)
            acc2 = tl.dot(a, w2, acc2)
    if SWIGLU:
        g = acc.to(tl.float16).to(tl.float32)
        u = acc2.to(tl.float16).to(tl.float32)
        res = (g / (1.0 + tl.exp(-g)) * u).to(tl.float16)
    else:
        res = acc.to(tl.float16)
        if SCALE_ROWS:
            wr = tl.load(wrow_ptr + offs_m, mask=mmask, other=0.0).to(tl.float32)
            res = (res.to(tl.float32) * wr[:, None]).to(tl.float16)
    o = out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
    tl.store(o, res, mask=mmask[:, None] & nmask[None, :])


_scratch = {}


def _buf(dev, name, numel):
    b = _scratch.get((dev, name))
    if b is None or b.numel() < numel:
        b = _scratch[(dev, name)] = torch.empty(numel, dtype=torch.half, device=dev)
    return b


def moe_kq_prefill(y_sorted, gu_ptrs, dn_ptrs, a, dn, tile_e, tile_r0, offs, wrow, E, I, K, fmt_gu, fmt_dn, bm,
                   chunk=None):
    """Dequant a chunk of experts to fp16 scratch (one launch per matrix kind), then fused gate/up/SwiGLU and down GEMMs
    over the rows of those experts. Scratch = chunk x (2 I K + K I) halves."""
    dev = y_sorted.device
    chunk = chunk or int(os.environ.get("MCPM_FQ_PF_CHUNK", "64"))
    BNd = 32
    for e0 in range(0, E, chunk):
        ec = min(chunk, E - e0)
        wg = _buf(dev, "g", ec * I * K)
        wu = _buf(dev, "u", ec * I * K)
        wd = _buf(dev, "d", ec * K * I)
        _dq_kq[(triton.cdiv(I, BNd), K // 256, ec)](gu_ptrs, wg, e0, I, K, FMT=fmt_gu, BN=BNd, num_warps=4)
        _dq_kq[(triton.cdiv(I, BNd), K // 256, ec)](gu_ptrs, wu, E + e0, I, K, FMT=fmt_gu, BN=BNd, num_warps=4)
        _dq_kq[(triton.cdiv(K, BNd), I // 256, ec)](dn_ptrs, wd, e0, K, I, FMT=fmt_dn, BN=BNd, num_warps=4)
        T = tile_e.shape[0]
        BN, BK, nw, ns = 128, 64, 8, (2 if torch.version.hip else 3)   # 64 KiB LDS on gfx11
        _gmm_f16[(triton.cdiv(I, BN), T)](y_sorted, wg, wu, a, tile_e, tile_r0, offs, offs, I, K, y_sorted.stride(0),
                                          a.stride(0), e0, e0 + ec, SWIGLU=True, SCALE_ROWS=False,
                                          BM=bm, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
        _gmm_f16[(triton.cdiv(K, BN), T)](a, wd, wd, dn, tile_e, tile_r0, offs, wrow, K, I, a.stride(0),
                                          dn.stride(0), e0, e0 + ec, SWIGLU=False, SCALE_ROWS=True,
                                          BM=bm, BN=BN, BK=BK, num_warps=nw, num_stages=ns)


def gmm_kq(a, wptrs, out, tile_e, tile_r0, offs, N, K, E, fmt, swiglu=False, wrow=None, BM=64, BN=64,
           num_warps=4, num_stages=1):
    BN = int(os.environ.get("MCPM_FQ_PF_BN", BN))
    num_warps = int(os.environ.get("MCPM_FQ_PF_WARPS", num_warps))
    T = tile_e.shape[0]
    grid = (triton.cdiv(N, BN), T)
    _gmm_kq[grid](a, wptrs, out, tile_e, tile_r0, offs, wrow if wrow is not None else offs,
                  N, K, a.stride(0), out.stride(0), E,
                  FMT=fmt, SWIGLU=swiglu, SCALE_ROWS=wrow is not None, BM=BM, BN=BN,
                  num_warps=num_warps, num_stages=num_stages)
