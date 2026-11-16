"""Grouped prefill MoE for IQ experts (unsloth UD-IQ packs: gate/up IQ2_S, down IQ3_XXS or IQ4_XS), same flow as fq_moe_prefill_kq.moe_kq_prefill:
dequant a chunk of experts to fp16 scratch (one Triton launch per matrix kind, decode as triton_iq / gap_formats.dequant), then the fused
gate/up/SwiGLU and down GEMMs (_gmm_f16) over the rows routed to those experts. Replaces the per-expert loop (torch dequant + mm per
expert, hundreds of small launches per layer). Formats: 0 iq3_xxs, 3 iq2_s, 7 iq4_xs."""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ..foreign_quant.triton_iq import _tabs, _b, _h
from .fq_moe_prefill_kq import _gmm_f16, _buf

FMT_ID = {"iq3_xxs": 0, "iq2_s": 3, "iq4_xs": 7}
_KV = {}


def _kv(dev):
    k = str(dev)
    if k not in _KV:
        _KV[k] = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                              dtype=torch.float32, device=dev)
    return _KV[k]


@triton.jit
def _dq_iq(ptrs, out, e0, N, K, t3x, tks, t2s, kv, FMT: tl.constexpr, BN: tl.constexpr):
    """Decode super-block program_id(1) of rows tile program_id(0) of expert e0 + program_id(2) into fp16 out[e, n, k]."""
    BB: tl.constexpr = 98 if FMT == 0 else (82 if FMT == 3 else 136)
    pid_n = tl.program_id(0)
    sb = tl.program_id(1)
    ec = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < N
    m2 = mn[:, None]
    e = tl.arange(0, 32)[None, :]
    w_ptr = tl.load(ptrs + e0 + ec).to(tl.pointer_type(tl.uint8))
    blk = rn.to(tl.int64) * ((K // 256) * BB) + sb * BB
    b2 = blk[:, None]
    d = _h(tl.load(w_ptr + blk, mask=mn, other=0), tl.load(w_ptr + blk + 1, mask=mn, other=0))
    d2 = d[:, None]
    o = out + (ec.to(tl.int64) * N + rn.to(tl.int64))[:, None] * K + sb * 256
    for ib in tl.static_range(8):
        if FMT == 0:      # iq3_xxs
            idx = _b(w_ptr, b2 + 2 + 8 * ib + e // 4, m2)
            g = tl.load(t3x + idx * 4 + e % 4).to(tl.float32)
            aw = blk + 66 + 4 * ib
            aux = _b(w_ptr, aw, mn) | (_b(w_ptr, aw + 1, mn) << 8) | (_b(w_ptr, aw + 2, mn) << 16) | (_b(w_ptr, aw + 3, mn) << 24)
            db = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.5
            sidx = (aux[:, None] >> (7 * (e // 8))) & 127
            sg = (tl.load(tks + sidx).to(tl.int32) >> (e % 8)) & 1
            w = db[:, None] * g * (1.0 - 2.0 * sg.to(tl.float32))
        elif FMT == 3:    # iq2_s
            l = e // 8
            idx = _b(w_ptr, b2 + 2 + 4 * ib + l, m2) | ((_b(w_ptr, blk + 66 + ib, mn)[:, None] << (8 - 2 * l)) & 0x300)
            g = tl.load(t2s + idx * 8 + e % 8).to(tl.float32)
            sg = (_b(w_ptr, b2 + 34 + 4 * ib + l, m2) >> (e % 8)) & 1
            sc = (_b(w_ptr, blk + 74 + ib, mn)[:, None] >> (4 * (l // 2))) & 15
            w = d2 * (0.5 + sc.to(tl.float32)) * 0.25 * g * (1.0 - 2.0 * sg.to(tl.float32))
        else:             # iq4_xs: d, scales_h u16 @2, scales_l[4] @4, qs[128] @8 (sub-block ib: 16 bytes, low nibbles 0..15, high 16..31)
            sh = _b(w_ptr, blk + 2, mn) | (_b(w_ptr, blk + 3, mn) << 8)
            ls = ((_b(w_ptr, blk + 4 + ib // 2, mn) >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32).to(tl.float32)
            q = (_b(w_ptr, b2 + 8 + 16 * ib + e % 16, m2) >> (4 * (e // 16))) & 15
            w = dl[:, None] * tl.load(kv + q)
        tl.store(o + 32 * ib + e, w.to(tl.float16), mask=m2)


def moe_iq_prefill(y_sorted, gu_ptrs, dn_ptrs, a, dn, tile_e, tile_r0, offs, wrow, E, I, K, fmt_gu, fmt_dn, bm, chunk=None):
    dev = y_sorted.device
    chunk = chunk or int(os.environ.get("MCPM_FQ_PF_CHUNK", "64"))
    t3x, _, tks, _, t2s, _ = _tabs(dev)
    kv = _kv(dev)
    BNd = 32
    for e0 in range(0, E, chunk):
        ec = min(chunk, E - e0)
        wg = _buf(dev, "g", ec * I * K)
        wu = _buf(dev, "u", ec * I * K)
        wd = _buf(dev, "d", ec * K * I)
        _dq_iq[(triton.cdiv(I, BNd), K // 256, ec)](gu_ptrs, wg, e0, I, K, t3x, tks, t2s, kv, FMT=fmt_gu, BN=BNd, num_warps=4)
        _dq_iq[(triton.cdiv(I, BNd), K // 256, ec)](gu_ptrs, wu, E + e0, I, K, t3x, tks, t2s, kv, FMT=fmt_gu, BN=BNd, num_warps=4)
        _dq_iq[(triton.cdiv(K, BNd), I // 256, ec)](dn_ptrs, wd, e0, K, I, t3x, tks, t2s, kv, FMT=fmt_dn, BN=BNd, num_warps=4)
        T = tile_e.shape[0]
        BN, BK, nw, ns = 128, 64, 8, (2 if torch.version.hip else 3)
        _gmm_f16[(triton.cdiv(I, BN), T)](y_sorted, wg, wu, a, tile_e, tile_r0, offs, offs, I, K, y_sorted.stride(0),
                                          a.stride(0), e0, e0 + ec, SWIGLU=True, SCALE_ROWS=False,
                                          BM=bm, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
        _gmm_f16[(triton.cdiv(K, BN), T)](a, wd, wd, dn, tile_e, tile_r0, offs, wrow, K, I, a.stride(0),
                                          dn.stride(0), e0, e0 + ec, SWIGLU=False, SCALE_ROWS=True,
                                          BM=bm, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
