"""Paged quantized cache (k/v bits 2..8, 32-value groups) -> compact fp16 staging window, one launch per plane.
Same arithmetic and operation order as the PyTorch fallback of dequant_cache_paged_window (ext_fallbacks.py): codes from the
high-to-low power-of-two bit planes, (code - (mid - 0.5)) * ((scale * r32) / mid) in fp32, then the unnormalised H32 butterfly
(widths 1..16, the writer's _h32_stage), rounded once to fp16. Page p of batch row b lands at row b * pages_per_seq * 256 of
the scratch; rows at or past cache_seqlens + bonus_len are left untouched."""
import torch
import triton
import triton.language as tl
from .qwrite_triton import _h32_stage

BT = 4   # tokens per program


@triton.jit
def _deq_window_kernel(q, sc, out, seqlens, btable, bonus, pps,
                       G: tl.constexpr, BITS: tl.constexpr, BT: tl.constexpr):
    tb = tl.program_id(0)
    b = tl.program_id(1)
    lim = tl.minimum(tl.maximum(tl.load(seqlens + b) + bonus, 0), pps * 256)
    if tb * BT >= lim:
        return
    R: tl.constexpr = BT * G
    r = tl.arange(0, R)
    tok = tb * BT + r // G
    g = r % G
    valid = tok < lim
    page = tl.load(btable + b * pps + tok // 256, mask = valid, other = 0).to(tl.int64)
    src = page * 256 + tok % 256
    dst = b.to(tl.int64) * pps * 256 + tok
    codes = tl.zeros((R, 32), dtype = tl.int32)
    base = (src * (G * BITS) + g * BITS)[:, None]
    po = 0
    if BITS & 8:
        w = tl.load(q + base + po + tl.arange(0, 8)[None, :], mask = valid[:, None], other = 0)
        x = (w[:, :, None] >> ((tl.arange(0, 4) * 8)[None, None, :])) & 255
        codes = (codes << 8) | tl.reshape(x, (R, 32))
        po += 8
    if BITS & 4:
        w = tl.load(q + base + po + tl.arange(0, 4)[None, :], mask = valid[:, None], other = 0)
        x = (w[:, :, None] >> ((tl.arange(0, 8) * 4)[None, None, :])) & 15
        codes = (codes << 4) | tl.reshape(x, (R, 32))
        po += 4
    if BITS & 2:
        w = tl.load(q + base + po + tl.arange(0, 2)[None, :], mask = valid[:, None], other = 0)
        x = (w[:, :, None] >> ((tl.arange(0, 16) * 2)[None, None, :])) & 3
        codes = (codes << 2) | tl.reshape(x, (R, 32))
        po += 2
    if BITS & 1:
        w = tl.load(q + base + po + tl.arange(0, 1)[None, :], mask = valid[:, None], other = 0)
        x = (w[:, :, None] >> (tl.arange(0, 32)[None, None, :])) & 1
        codes = (codes << 1) | tl.reshape(x, (R, 32))
    MID: tl.constexpr = float(1 << (BITS - 1))
    s = tl.load(sc + src * G + g, mask = valid, other = 0.0).to(tl.float32)
    mult = tl.math.div_rn(s * 0.17677669529663688, tl.full((R,), MID, tl.float32))
    v = (codes.to(tl.float32) - (MID - 0.5)) * mult[:, None]
    v = _h32_stage(v, R, 1)
    v = _h32_stage(v, R, 2)
    v = _h32_stage(v, R, 4)
    v = _h32_stage(v, R, 8)
    v = _h32_stage(v, R, 16)
    o = (dst * (G * 32) + g * 32)[:, None] + tl.arange(0, 32)[None, :]
    tl.store(out + o, v.to(tl.float16), mask = valid[:, None])


def dequant_window_supported(k_in, k_sc, k_out, v_in, v_sc, v_out, cache_seqlens, block_table, page_size, compand_a):
    if compand_a != 0.0 or page_size != 256 or not torch.cuda.is_available():
        return False
    if not (k_in.is_cuda and k_in.dim() == 3 and v_in.dim() == 3 and k_out.dim() in (3, 4)):
        return False
    G = k_sc.shape[-1]
    if G < 1 or G > 64 or G & (G - 1) or k_out[0].numel() != page_size * G * 32 or v_out[0].numel() != page_size * G * 32:
        return False
    kb, vb = k_in.shape[-1] // G, v_in.shape[-1] // G
    return (2 <= kb <= 8 and 2 <= vb <= 8 and k_in.shape[-1] == G * kb and v_in.shape[-1] == G * vb and
            k_in.dtype == torch.int32 and v_in.dtype == torch.int32 and k_sc.dtype == torch.float16 and v_sc.dtype == torch.float16 and
            k_out.dtype == torch.float16 and v_out.dtype == torch.float16 and
            k_in.is_contiguous() and v_in.is_contiguous() and k_sc.is_contiguous() and v_sc.is_contiguous() and
            k_out.is_contiguous() and v_out.is_contiguous() and block_table.dtype == torch.int32 and
            block_table.is_contiguous() and cache_seqlens.is_cuda and k_in.shape[:2] == k_sc.shape[:2] == v_in.shape[:2])


def dequant_window(k_in, k_sc, k_out, v_in, v_sc, v_out, cache_seqlens, block_table, bonus_len):
    bsz, pps = block_table.shape
    G = k_sc.shape[-1]
    grid = (triton.cdiv(pps * 256, BT), bsz)
    seq32 = cache_seqlens if cache_seqlens.dtype == torch.int32 else cache_seqlens.to(torch.int32)
    for q, sc, out in ((k_in, k_sc, k_out), (v_in, v_sc, v_out)):
        _deq_window_kernel[grid](q, sc, out, seq32, block_table, int(bonus_len), pps,
                                 G = G, BITS = q.shape[-1] // G, BT = BT, num_warps = 4, num_stages = 1, enable_fp_fusion = False)
