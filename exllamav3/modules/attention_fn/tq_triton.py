"""TurboQuant cache kernels (Triton, ROCm + CUDA): paged window dequant for the prefill staging scratch.

The pool layout is CacheLayer_quant's (32-value groups, power-of-two bit planes, plane order 8 4 2 1) with one fp16 scale
per head vector instead of per group; see cache/tq.py. Values stay in the rotated domain."""
import torch
import triton
import triton.language as tl


@triton.jit
def _tq_dequant_window_kernel(
    words, scales, cb, out, block_table, cache_seqlens,
    bonus, npps,
    BITS: tl.constexpr,
    PAGE: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    TB: tl.constexpr,
):
    """Program (page slot of the block table x TB-token tile, kv head): dequantize TB tokens to fp16 (compact scratch layout:
    scratch page b * npps + p holds table slot p of sequence b)."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    TPP: tl.constexpr = PAGE // TB
    bp = pid // TPP
    sub = pid - bp * TPP
    b = bp // npps
    p = bp - b * npps
    limit = tl.load(cache_seqlens + b) + bonus
    toff = sub * TB + tl.arange(0, TB)
    tmask = (p * PAGE + toff) < limit
    phys = tl.load(block_table + b * npps + p)
    row = phys * PAGE + toff
    d = tl.arange(0, D)
    g = d // 32
    r = d - g * 32
    G: tl.constexpr = D // 32
    base = row[:, None] * (HKV * G * BITS) + h * (G * BITS) + g[None, :] * BITS
    m2 = tmask[:, None]
    raw = tl.zeros((TB, D), tl.int32)
    pbase = 0
    if BITS & 8:
        w = tl.load(words + base + pbase + (r // 4)[None, :], mask = m2, other = 0)
        raw = (raw << 8) | ((w >> ((r % 4) * 8)[None, :]) & 255)
        pbase += 8
    if BITS & 4:
        w = tl.load(words + base + pbase + (r // 8)[None, :], mask = m2, other = 0)
        raw = (raw << 4) | ((w >> ((r % 8) * 4)[None, :]) & 15)
        pbase += 4
    if BITS & 2:
        w = tl.load(words + base + pbase + (r // 16)[None, :], mask = m2, other = 0)
        raw = (raw << 2) | ((w >> ((r % 16) * 2)[None, :]) & 3)
        pbase += 2
    if BITS & 1:
        w = tl.load(words + base + pbase + (r // 32)[None, :], mask = m2, other = 0)
        raw = (raw << 1) | ((w >> (r % 32)[None, :]) & 1)
    sc = tl.load(scales + row * HKV + h, mask = tmask, other = 0.0).to(tl.float32)
    val = (tl.load(cb + raw) * sc[:, None]).to(tl.float16)
    dst = ((bp * PAGE + toff)[:, None].to(tl.int64) * HKV + h) * D + d[None, :]
    tl.store(out + dst, val, mask = m2)


def tq_dequant_window(words, scales, cb, out, cache_seqlens, block_table, page_size, bonus_len, bits, n_kv_heads, head_dim):
    """Dequantize the table-mapped pages of every sequence (positions < cache_seqlens + bonus_len) into the compact scratch
    `out` [>= bsz * npps pages, page_size, kvh, hd] fp16, rotated domain."""
    bsz, npps = block_table.shape
    tb = 16
    with torch.cuda.device(words.device):
        _tq_dequant_window_kernel[(bsz * npps * (page_size // tb), n_kv_heads)](
            words, scales, cb, out, block_table, cache_seqlens, bonus_len, npps,
            bits, page_size, n_kv_heads, head_dim, tb,
            num_warps = 4, num_stages = 1,
        )


@triton.jit
def _tq_write_kernel(
    x, words, scales, cb, mid, block_table, cache_seqlens,
    length, npps,
    BITS: tl.constexpr,
    PAGE: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
):
    """One program per (sequence b, new position t, kv head): y -> unit vector -> nearest Lloyd-Max centroid per coordinate
    (count of midpoints below it) -> bit planes (8 4 2 1, high bits first) + fp16 scale = norm / ||centroid vector||."""
    bt = tl.program_id(0)
    h = tl.program_id(1)
    b = bt // length
    t = bt - b * length
    tok = tl.load(cache_seqlens + b) + t
    phys = tl.load(block_table + b * npps + tok // PAGE)
    row = phys.to(tl.int64) * PAGE + tok % PAGE
    d = tl.arange(0, D)
    y = tl.load(x + (bt.to(tl.int64) * HKV + h) * D + d).to(tl.float32)
    nrm = tl.maximum(tl.sqrt(tl.sum(y * y, axis = 0)), 1e-12)
    u = y / nrm
    code = tl.zeros((D,), tl.int32)
    for i in range((1 << BITS) - 1):
        code += (u > tl.load(mid + i)).to(tl.int32)
    c = tl.load(cb + code)
    cn = tl.maximum(tl.sqrt(tl.sum(c * c, axis = 0)), 1e-12)
    tl.store(scales + row * HKV + h, (nrm / cn).to(tl.float16))
    G: tl.constexpr = D // 32
    wbase = words + row * (HKV * G * BITS) + h * (G * BITS)
    if BITS & 8:
        _tq_pack_plane(wbase, code, 8, BITS, 0, BITS - 8, G, D)
    if BITS & 4:
        _tq_pack_plane(wbase, code, 4, BITS, (BITS & 8), BITS - (BITS & 8) - 4, G, D)
    if BITS & 2:
        _tq_pack_plane(wbase, code, 2, BITS, (BITS & 12), BITS - (BITS & 12) - 2, G, D)
    if BITS & 1:
        _tq_pack_plane(wbase, code, 1, BITS, (BITS & 14), 0, G, D)


@triton.jit
def _tq_pack_plane(wbase, code, W: tl.constexpr, BITS: tl.constexpr, pbase: tl.constexpr, shift: tl.constexpr,
                   G: tl.constexpr, D: tl.constexpr):
    VPW: tl.constexpr = 32 // W
    nib = (code >> shift) & ((1 << W) - 1)
    nib2 = tl.reshape(nib, (G * W, VPW))
    sh = (tl.arange(0, VPW) * W)[None, :]
    word = tl.sum(nib2 << sh, axis = 1)
    wi = tl.arange(0, G * W)
    tl.store(wbase + (wi // W) * BITS + pbase + (wi % W), word)


def tq_write(x, words, scales, cb, mid, cache_seqlens, block_table, bits, n_kv_heads, head_dim, page_size):
    """x: rotated fp16 [bsz, length, kvh, hd] contiguous -> pool rows at cache_seqlens .. + length."""
    bsz, length = x.shape[0], x.shape[1]
    with torch.cuda.device(x.device):
        _tq_write_kernel[(bsz * length, n_kv_heads)](
            x, words, scales, cb, mid, block_table, cache_seqlens,
            length, block_table.shape[1],
            bits, page_size, n_kv_heads, head_dim,
            num_warps = 2, num_stages = 1,
        )


@triton.jit
def _tq_rotate_kernel(
    x0, x1, x2, y0, y1, y2, rot_t,
    n0, n1, n2,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """y = x @ rot_t (rows of D fp16) for up to three tensors in one launch: program (row block over the concatenated rows,
    BN-wide column slice). Replaces three tiny GEMMs per attention call."""
    pid = tl.program_id(0)
    cb = tl.program_id(1)
    nb0 = tl.cdiv(n0, BM)
    nb1 = tl.cdiv(n1, BM)
    is0 = pid < nb0
    is1 = (pid >= nb0) & (pid < nb0 + nb1)
    is2 = pid >= nb0 + nb1
    blk = tl.where(is0, pid, tl.where(is1, pid - nb0, pid - nb0 - nb1))
    rows = blk * BM + tl.arange(0, BM)
    m0 = (rows < n0) & is0
    m1 = (rows < n1) & is1
    m2 = (rows < n2) & is2
    ks = tl.arange(0, D)
    cols = cb * BN + tl.arange(0, BN)
    ro = rows[:, None].to(tl.int64) * D
    xt = tl.load(x0 + ro + ks[None, :], mask = m0[:, None], other = 0.0)
    xt += tl.load(x1 + ro + ks[None, :], mask = m1[:, None], other = 0.0)
    xt += tl.load(x2 + ro + ks[None, :], mask = m2[:, None], other = 0.0)
    rt = tl.load(rot_t + ks[:, None] * D + cols[None, :])
    acc = tl.dot(xt, rt).to(tl.float16)
    tl.store(y0 + ro + cols[None, :], acc, mask = m0[:, None])
    tl.store(y1 + ro + cols[None, :], acc, mask = m1[:, None])
    tl.store(y2 + ro + cols[None, :], acc, mask = m2[:, None])


def tq_rotate(rot_t, *xs):
    """Rotate each fp16 tensor (any leading shape, last dim = head_dim) with one launch; returns the rotated tensors (same shapes)."""
    D = rot_t.shape[0]
    cnt = len(xs)
    xs = [x.contiguous() for x in xs]
    ys = [torch.empty_like(x) for x in xs]
    ns = [x.numel() // D for x in xs]
    while len(xs) < 3:
        xs.append(xs[0]); ys.append(ys[0]); ns.append(0)
    bm = 16
    nblk = sum(triton.cdiv(n, bm) for n in ns)
    with torch.cuda.device(rot_t.device):
        _tq_rotate_kernel[(nblk, D // 64)](xs[0], xs[1], xs[2], ys[0], ys[1], ys[2], rot_t, ns[0], ns[1], ns[2], D, bm, 64,
                                           num_warps = 4, num_stages = 1)
    return ys[:cnt]
