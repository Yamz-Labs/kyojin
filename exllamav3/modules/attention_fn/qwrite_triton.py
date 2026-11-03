"""Native (Triton) 8-bit paged cache writer. Replaces the PyTorch fallback of ext.quant_cache_paged on ROCm
(cache/q_cache.cu is not part of the ROCm build): same format, same float32 operation order
(unnormalised H32 butterfly, * 1/sqrt(32), scale = absmax + 1e-10, codes = floor(v/scale*128 + 128),
four codes per int32 word, low byte first), one launch for K and V instead of ~70 torch launches."""
import torch
import triton
import triton.language as tl


@triton.jit
def _h32_stage(x, G: tl.constexpr, W: tl.constexpr):
    N: tl.constexpr = 16 // W
    y = tl.reshape(x, (G, N, 2, W))
    y = tl.permute(y, (0, 1, 3, 2))
    lo, hi = tl.split(y)
    y = tl.join(lo + hi, lo - hi)
    y = tl.permute(y, (0, 1, 3, 2))
    return tl.reshape(y, (G, 32))


@triton.jit
def _quant_row8(inp, out, sc, src, dst, G: tl.constexpr):
    og = tl.arange(0, G)[:, None]
    oi = tl.arange(0, 32)[None, :]
    x = tl.load(inp + src * (G * 32) + og * 32 + oi).to(tl.float32)
    x = _h32_stage(x, G, 1)
    x = _h32_stage(x, G, 2)
    x = _h32_stage(x, G, 4)
    x = _h32_stage(x, G, 8)
    x = _h32_stage(x, G, 16)
    x = x * 0.17677669529663688
    s = tl.max(tl.abs(x), axis=1) + 1.0e-10
    inv = tl.math.div_rn(tl.full((G,), 1.0, tl.float32), s)
    v = x * inv[:, None]
    c = tl.floor(v * 128.0 + 128.0)
    c = tl.minimum(tl.maximum(c, 0.0), 255.0).to(tl.int32)
    c = tl.reshape(c, (G, 8, 4))
    sh = (tl.arange(0, 4) * 8)[None, None, :]
    words = tl.sum(c << sh, axis=2)
    ow = tl.arange(0, 8)[None, :]
    tl.store(out + dst * (G * 8) + og * 8 + ow, words)
    tl.store(sc + dst * G + tl.arange(0, G), s.to(tl.float16))


@triton.jit
def _quant_cache_paged8_kernel(
    k_in, v_in, k_out, v_out, k_sc, v_sc, seqlens, btable,
    seq_len, pages_per_seq, in_contig: tl.constexpr, G: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    t = row % seq_len
    tok = tl.load(seqlens + b) + t
    page = tl.load(btable + b * pages_per_seq + tok // 256)
    dst = page.to(tl.int64) * 256 + tok % 256
    if in_contig:
        src = row.to(tl.int64)
    else:
        src = dst
    _quant_row8(k_in, k_out, k_sc, src, dst, G)
    _quant_row8(v_in, v_out, v_sc, src, dst, G)


def quant_cache_paged8(k_in, k_out, k_sc, v_in, v_out, v_sc, cache_seqlens, block_table, seq_len, in_contiguous):
    dim = k_sc.shape[-1] * 32
    G = dim // 32
    bsz = block_table.shape[0]
    _quant_cache_paged8_kernel[(bsz * seq_len,)](
        k_in, v_in, k_out, v_out, k_sc, v_sc, cache_seqlens, block_table,
        seq_len, block_table.shape[1], in_contig=bool(in_contiguous), G=G, num_warps=1)


@triton.jit
def _quant_row_bits(inp, out, sc, src, dst, G: tl.constexpr, BITS: tl.constexpr):
    """_quant_row8 for any width 2..8: codes = floor(v/scale * 2^(B-1) + 2^(B-1)) clamped to [0, 2^B), packed as the cache's
    power-of-two bit planes, highest plane first (8, 4, 2, 1 bits per value, the planes B contains); plane of width W is W words,
    value i of the group in word i // (32 / W) at bit (i % (32 / W)) * W."""
    og = tl.arange(0, G)[:, None]
    oi = tl.arange(0, 32)[None, :]
    x = tl.load(inp + src * (G * 32) + og * 32 + oi).to(tl.float32)
    x = _h32_stage(x, G, 1)
    x = _h32_stage(x, G, 2)
    x = _h32_stage(x, G, 4)
    x = _h32_stage(x, G, 8)
    x = _h32_stage(x, G, 16)
    x = x * 0.17677669529663688
    s = tl.max(tl.abs(x), axis=1) + 1.0e-10
    inv = tl.math.div_rn(tl.full((G,), 1.0, tl.float32), s)
    v = x * inv[:, None]
    M: tl.constexpr = float(1 << (BITS - 1))
    c = tl.floor(v * M + M)
    c = tl.minimum(tl.maximum(c, 0.0), float((1 << BITS) - 1)).to(tl.int32)
    base = dst * (G * BITS)
    po = 0
    if BITS & 8:
        f = tl.reshape((c >> (BITS - 8)) & 255, (G, 8, 4))
        words = tl.sum(f << ((tl.arange(0, 4) * 8)[None, None, :]), axis=2)
        tl.store(out + base + og * BITS + po + tl.arange(0, 8)[None, :], words)
        po += 8
    if BITS & 4:
        f = tl.reshape((c >> (BITS & 3)) & 15, (G, 4, 8))
        words = tl.sum(f << ((tl.arange(0, 8) * 4)[None, None, :]), axis=2)
        tl.store(out + base + og * BITS + po + tl.arange(0, 4)[None, :], words)
        po += 4
    if BITS & 2:
        f = tl.reshape((c >> (BITS & 1)) & 3, (G, 2, 16))
        words = tl.sum(f << ((tl.arange(0, 16) * 2)[None, None, :]), axis=2)
        tl.store(out + base + og * BITS + po + tl.arange(0, 2)[None, :], words)
        po += 2
    if BITS & 1:
        f = tl.reshape(c & 1, (G, 1, 32))
        words = tl.sum(f << (tl.arange(0, 32)[None, None, :]), axis=2)
        tl.store(out + base + og * BITS + po + tl.arange(0, 1)[None, :], words)
    tl.store(sc + dst * G + tl.arange(0, G), s.to(tl.float16))


@triton.jit
def _quant_cache_paged_bits_kernel(
    k_in, v_in, k_out, v_out, k_sc, v_sc, seqlens, btable,
    seq_len, pages_per_seq, in_contig: tl.constexpr, G: tl.constexpr, KB: tl.constexpr, VB: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    t = row % seq_len
    tok = tl.load(seqlens + b) + t
    page = tl.load(btable + b * pages_per_seq + tok // 256)
    dst = page.to(tl.int64) * 256 + tok % 256
    if in_contig:
        src = row.to(tl.int64)
    else:
        src = dst
    _quant_row_bits(k_in, k_out, k_sc, src, dst, G, KB)
    _quant_row_bits(v_in, v_out, v_sc, src, dst, G, VB)


def quant_cache_paged_bits(k_in, k_out, k_sc, v_in, v_out, v_sc, cache_seqlens, block_table, seq_len, in_contiguous, k_bits, v_bits):
    G = k_sc.shape[-1]
    bsz = block_table.shape[0]
    _quant_cache_paged_bits_kernel[(bsz * seq_len,)](
        k_in, v_in, k_out, v_out, k_sc, v_sc, cache_seqlens, block_table,
        seq_len, block_table.shape[1], in_contig=bool(in_contiguous), G=G, KB=k_bits, VB=v_bits, num_warps=1)
