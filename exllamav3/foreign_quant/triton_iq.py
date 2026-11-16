"""Fused decode GEMV (<= 16 rows) for the GGUF 256-element blocks IQ3_XXS, IQ3_S, IQ2_XS, IQ2_S and Q3_K, read in their native layout.

One program = BN output rows; the K loop walks 32-element units (8 per 256 block): each unit is decoded into a (BN, 32) fp16 tile
(grid / sign tables gathered from small device tables) and multiplied with tl.dot against up to 16 activation rows. Values equal
gap_formats.dequant (the torch reference) up to the fp32 accumulation order. Formats: FMT 0 iq3_xxs, 1 iq3_s, 2 iq2_xs, 3 iq2_s, 4 q3_k, 5 iq2_xxs (rows 2..16 only: the C++ GEMV serves 1 row).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

FUSED = {"iq3_xxs": 0, "iq3_s": 1, "iq2_xs": 2, "iq2_s": 3, "q3_k": 4, "iq2_xxs": 5, "q2_k": 6}
_BB = {"iq3_xxs": 98, "iq3_s": 110, "iq2_xs": 74, "iq2_s": 82, "q3_k": 110, "iq2_xxs": 66, "q2_k": 84}
MAX_ROWS = 16
BN = int(os.environ.get("EXL3_IQ_BN", "16"))
# Strix tile sweep 11/11 (iq_tiles.py): nw 1 iq2_s +8 %, iq3_s +13 % at R1; served UD-IQ2_M 15.4 -> 17.3 tok/s
NW = int(os.environ.get("EXL3_IQ_NW", "1"))
NB = int(os.environ.get("EXL3_IQ_NB", "0"))     # 32-units per loop iteration (static unroll); 0 = per-format default below
# Strix sweep 11/11 (iq_tiles.py, R1): iq3_xxs NB 4 68 -> 92 GB/s, iq2_s NB 4 61 -> 69, iq3_s NB 2 114 -> 124
_NB = {"iq3_xxs": 4, "iq2_s": 4, "iq3_s": 2}
_TABS: dict = {}


def _tabs(device):
    key = str(device)
    if key not in _TABS:
        from . import gap_formats as G
        from .gguf_tables import IQ3_XXS_GRID, IQ3_S_GRID, KSIGNS
        from .iq_grids_gen import iq2_grid
        u8 = lambda b: torch.frombuffer(bytearray(b), dtype=torch.uint8)
        _TABS[key] = (u8(IQ3_XXS_GRID).to(device), u8(IQ3_S_GRID).to(device), u8(KSIGNS).to(device),
                      torch.from_numpy(iq2_grid("xs").copy()).reshape(-1).to(device),
                      torch.from_numpy(iq2_grid("s").copy()).reshape(-1).to(device),
                      torch.from_numpy(iq2_grid("xxs").copy()).reshape(-1).to(device))
    return _TABS[key]


@triton.jit
def _b(p, off, m):
    return tl.load(p + off, mask=m, other=0).to(tl.int32)


@triton.jit
def _gemv_iq(x_ptr, w_ptr, o_ptr, t3x, t3s, tks, t2x, t2s, t2xx, R, K, N, FMT: tl.constexpr, BB: tl.constexpr, BN: tl.constexpr,
             RP: tl.constexpr, NB: tl.constexpr = 1):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    m2 = mn[:, None]
    rr = tl.arange(0, RP)
    e = tl.arange(0, 32)[None, :]
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    row = rn.to(tl.int64) * ((K // 256) * BB)
    for u0 in range(0, K // 32, NB):
      for t_ in tl.static_range(NB):
        u = u0 + t_
        blk = row + (u // 8) * BB
        ib = u % 8
        b2 = blk[:, None]
        if FMT == 4:
            d = _h(tl.load(w_ptr + blk + 108, mask=mn, other=0), tl.load(w_ptr + blk + 109, mask=mn, other=0))
        elif FMT == 6:
            d = _h(tl.load(w_ptr + blk + 80, mask=mn, other=0), tl.load(w_ptr + blk + 81, mask=mn, other=0))
        else:
            d = _h(tl.load(w_ptr + blk, mask=mn, other=0), tl.load(w_ptr + blk + 1, mask=mn, other=0))
        d2 = d[:, None]
        if FMT == 0:      # iq3_xxs: qs[64] grid idx (4 values each), then 8 x uint32 (4 x 7-bit sign idx | 4-bit scale << 28)
            idx = _b(w_ptr, b2 + 2 + 8 * ib + e // 4, m2)
            g = tl.load(t3x + idx * 4 + e % 4).to(tl.float32)
            aw = blk + 66 + 4 * ib
            aux = _b(w_ptr, aw, mn) | (_b(w_ptr, aw + 1, mn) << 8) | (_b(w_ptr, aw + 2, mn) << 16) | (_b(w_ptr, aw + 3, mn) << 24)
            db = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.5
            sidx = (aux[:, None] >> (7 * (e // 8))) & 127
            sg = (tl.load(tks + sidx).to(tl.int32) >> (e % 8)) & 1
            w = db[:, None] * g * (1.0 - 2.0 * sg.to(tl.float32))
        elif FMT == 1:    # iq3_s: qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106
            l = e // 4
            idx = _b(w_ptr, b2 + 2 + 8 * ib + l, m2) | (((_b(w_ptr, blk + 66 + ib, mn)[:, None] >> l) & 1) << 8)
            g = tl.load(t3s + idx * 4 + e % 4).to(tl.float32)
            sc = (_b(w_ptr, blk + 106 + ib // 2, mn) >> (4 * (ib % 2))) & 15
            db = d * (1.0 + 2.0 * sc.to(tl.float32))
            sg = (_b(w_ptr, b2 + 74 + 4 * ib + e // 8, m2) >> (e % 8)) & 1
            w = db[:, None] * g * (1.0 - 2.0 * sg.to(tl.float32))
        elif FMT == 2:    # iq2_xs: 32 x uint16 (9-bit grid idx | 7-bit sign idx) @2, scales[8] @66
            l = e // 8
            qo = b2 + 2 + 2 * (4 * ib + l)
            q = _b(w_ptr, qo, m2) | (_b(w_ptr, qo + 1, m2) << 8)
            g = tl.load(t2x + (q & 511) * 8 + e % 8).to(tl.float32)
            sg = (tl.load(tks + (q >> 9)).to(tl.int32) >> (e % 8)) & 1
            sc = (_b(w_ptr, blk + 66 + ib, mn)[:, None] >> (4 * (l // 2))) & 15
            w = d2 * (0.5 + sc.to(tl.float32)) * 0.25 * g * (1.0 - 2.0 * sg.to(tl.float32))
        elif FMT == 3:    # iq2_s: grid low bytes[32] @2, signs[32] @34, qh[8] @66, scales[8] @74
            l = e // 8
            idx = _b(w_ptr, b2 + 2 + 4 * ib + l, m2) | ((_b(w_ptr, blk + 66 + ib, mn)[:, None] << (8 - 2 * l)) & 0x300)
            g = tl.load(t2s + idx * 8 + e % 8).to(tl.float32)
            sg = (_b(w_ptr, b2 + 34 + 4 * ib + l, m2) >> (e % 8)) & 1
            sc = (_b(w_ptr, blk + 74 + ib, mn)[:, None] >> (4 * (l // 2))) & 15
            w = d2 * (0.5 + sc.to(tl.float32)) * 0.25 * g * (1.0 - 2.0 * sg.to(tl.float32))
        elif FMT == 5:    # iq2_xxs: per 32-unit 8 bytes @2 + 8 ib: 4 x 8-bit grid idx, then uint32 (4 x 7-bit sign idx | 4-bit scale << 28)
            l = e // 8
            idx = _b(w_ptr, b2 + 2 + 8 * ib + l, m2)
            g = tl.load(t2xx + idx * 8 + e % 8).to(tl.float32)
            aw = blk + 6 + 8 * ib
            aux = _b(w_ptr, aw, mn) | (_b(w_ptr, aw + 1, mn) << 8) | (_b(w_ptr, aw + 2, mn) << 16) | (_b(w_ptr, aw + 3, mn) << 24)
            db = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.25
            sidx = (aux[:, None] >> (7 * l)) & 127
            sg = (tl.load(tks + sidx).to(tl.int32) >> (e % 8)) & 1
            w = db[:, None] * g * (1.0 - 2.0 * sg.to(tl.float32))
        elif FMT == 6:    # q2_k: scales[16] @0, qs[64] @16, d @80, dmin @82; unit ib = 32 * (4 * n + j)
            n2 = ib // 4
            j = ib % 4
            q = (_b(w_ptr, b2 + 16 + 32 * n2 + e, m2) >> (2 * j)) & 3
            sc = _b(w_ptr, b2 + 8 * n2 + 2 * j + e // 16, m2)
            dm = _h(tl.load(w_ptr + blk + 82, mask=mn, other=0), tl.load(w_ptr + blk + 83, mask=mn, other=0))
            w = d2 * (sc & 15).to(tl.float32) * q.to(tl.float32) - dm[:, None] * (sc >> 4).to(tl.float32)
        else:             # q3_k: hmask[32] @0, qs[64] @32, scales[12] @96, d @108; unit ib = 32 * (4 * hh + j)
            hh = ib // 4
            j = ib % 4
            half = e // 16
            q = (_b(w_ptr, b2 + 32 + 32 * hh + e, m2) >> (2 * j)) & 3
            hm = (_b(w_ptr, b2 + e, m2) >> (4 * hh + j)) & 1
            v = q - 4 * (1 - hm)
            i = 8 * hh + 2 * j + half                      # scale index 0..15, (1, 32)
            bq = i % 4
            gq = i // 4
            lo = (_b(w_ptr, b2 + 96 + bq + 4 * (gq % 2), m2) >> (4 * (gq // 2))) & 15
            hi = (_b(w_ptr, b2 + 104 + bq, m2) >> (2 * gq)) & 3
            sc = (lo | (hi << 4)) - 32
            w = d2 * sc.to(tl.float32) * v.to(tl.float32)
        wt = w.to(tl.float16)
        xv = tl.load(x_ptr + rr[:, None] * K + u * 32 + e, mask=rr[:, None] < R, other=0.0)
        acc = tl.dot(xv, tl.trans(wt), acc)
    tl.store(o_ptr + rr[:, None] * N + rn[None, :], acc.to(tl.float16), mask=(rr[:, None] < R) & mn[None, :])


@triton.jit
def _h(lo, hi):
    return (lo.to(tl.int16) | (hi.to(tl.int16) << 8)).to(tl.float16, bitcast=True).to(tl.float32)


def supported(fmt: str, rows: int, k: int, n: int) -> bool:
    # iq2_xxs: every width comes here. The C++ fq_gemv_iq2_xxs returns wrong rows on real GGUF blocks (rel 1.5 vs its own dequant,
    # Qwen3-8B-UD-IQ2_XXS up_proj) and its header scratch covers K <= 8192 only.
    lo = 1
    return fmt in FUSED and lo <= rows <= MAX_ROWS and k % 256 == 0 and os.environ.get("EXL3_IQ_FUSED", "1") != "0"


def gemv(fmt: str, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """x fp16 [R, K] contiguous, raw uint8 [N, K/256*bb] -> fp16 [R, N]."""
    R, K = x.shape
    N = raw.shape[0]
    assert R <= MAX_ROWS and K % 256 == 0 and raw.shape[1] == K // 256 * _BB[fmt]
    out = torch.empty((R, N), dtype=torch.float16, device=x.device)
    t3x, t3s, tks, t2x, t2s, t2xx = _tabs(x.device)
    nb = NB or _NB.get(fmt, 1)
    nb = nb if (K // 32) % nb == 0 else 1
    _gemv_iq[(triton.cdiv(N, BN),)](x, raw, out, t3x, t3s, tks, t2x, t2s, t2xx, R, K, N, FUSED[fmt], _BB[fmt], BN, 16, nb, num_warps=NW)
    return out
