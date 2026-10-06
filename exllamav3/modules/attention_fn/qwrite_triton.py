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
