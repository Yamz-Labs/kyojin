"""Multi-row decode GEMV (2..16 rows: verify / multi-user) for Q8_0 and NVFP4 read once for all rows (tl.dot over 16 padded rows).
The C++ fq_gemv kernels re-read the weights for every row (Strix 17408 x 5120: R4 ~50 GB/s, R8 ~25 vs ~200 at R1); R1 stays on them.
Values match gap dequant + matmul up to the fp32 accumulation order. BOLT_FQ_ROWS_TRITON=0 disables."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

FORMATS = ("q8_0", "nvfp4", "q4_k", "q6_k", "q5_k", "iq4_xs")
MAX_ROWS = 16
BN, BK = 32, 64


@triton.jit
def _e4m3(b):
    return ((((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True) * 256.0).to(tl.float32)


@triton.jit
def _e2m1(n):
    # E2M1 nibble -> value by bit placement in fp16: sign | exp(2) | mant(1) -> fp16 bits s eeeee m.. with bias fix (x 2^14),
    # subnormal 0b0001 (0.5) handled by the same placement (fp16 subnormal 2^-15 * 2^14 = 0.5)
    h = (((n & 8) << 12) | ((n & 7) << 9)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    return h * 16384.0


@triton.jit
def _rows_q8_0(x_ptr, p_ptr, y_ptr, R, K, N, BN: tl.constexpr, BK: tl.constexpr, RP: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    rr = tl.arange(0, RP)
    mr = rr < R
    rowb = rn.to(tl.int64) * ((K // 32) * 34)
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        off = (rk // 32) * 34
        q = tl.load(p_ptr + rowb[:, None] + off[None, :] + 2 + (rk % 32)[None, :], mask=mn[:, None], other=0).to(tl.int8).to(tl.float32)
        lo = tl.load(p_ptr + rowb[:, None] + off[None, :], mask=mn[:, None], other=0).to(tl.int16)
        hi = tl.load(p_ptr + rowb[:, None] + off[None, :] + 1, mask=mn[:, None], other=0).to(tl.int16)
        d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        w = (q * d).to(tl.float16)
        xv = tl.load(x_ptr + rr[:, None] * K + rk[None, :], mask=mr[:, None], other=0.0)
        acc += tl.dot(xv, tl.trans(w), out_dtype=tl.float32)
    tl.store(y_ptr + rr[:, None] * N + rn[None, :], acc, mask=mr[:, None] & mn[None, :])


@triton.jit
def _rows_nvfp4(x_ptr, p_ptr, s_ptr, y_ptr, R, K, N, G, BN: tl.constexpr, BK: tl.constexpr, RP: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    rr = tl.arange(0, RP)
    mr = rr < R
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        rb = k0 // 2 + tl.arange(0, BK // 2)
        byte = tl.load(p_ptr + rn[:, None].to(tl.int64) * (K // 2) + rb[None, :], mask=mn[:, None], other=0).to(tl.int32)
        v = tl.reshape(tl.join(_e2m1(byte & 15), _e2m1(byte >> 4)), (BN, BK))       # each byte once: low nibble = even k
        rs = k0 // 16 + tl.arange(0, BK // 16)
        sc = _e4m3(tl.load(s_ptr + rn[:, None].to(tl.int64) * (K // 16) + rs[None, :], mask=mn[:, None], other=0).to(tl.int16))
        w = tl.reshape(tl.reshape(v, (BN, BK // 16, 16)) * sc[:, :, None], (BN, BK)).to(tl.float16)
        xv = tl.load(x_ptr + rr[:, None] * K + rk[None, :], mask=mr[:, None], other=0.0)
        acc += tl.dot(xv, tl.trans(w), out_dtype=tl.float32)
    tl.store(y_ptr + rr[:, None] * N + rn[None, :], acc * G, mask=mr[:, None] & mn[None, :])


@triton.jit
def _h16(lo, hi):
    return (lo.to(tl.int16) | (hi.to(tl.int16) << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _rows_q4_k(x_ptr, p_ptr, y_ptr, R, K, N, BN: tl.constexpr, RP: tl.constexpr, Q5: tl.constexpr = False):
    """Q4_K super-block 144 B / 256: d, dmin (fp16), 12 B of 6-bit scales / mins, 128 B of nibbles (64-chunk i: low = sub 2i, high = sub 2i+1)."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    rr = tl.arange(0, RP)
    mr = rr < R
    l = tl.arange(0, 32)
    BB: tl.constexpr = 176 if Q5 else 144
    QS: tl.constexpr = 48 if Q5 else 16
    row = p_ptr + rn.to(tl.int64) * ((K // 256) * BB)
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, 64):
        b = row + (k0 // 256) * BB
        i = (k0 % 256) // 64
        d = _h16(tl.load(b, mask=mn, other=0), tl.load(b + 1, mask=mn, other=0))
        dm = _h16(tl.load(b + 2, mask=mn, other=0), tl.load(b + 3, mask=mn, other=0))
        sb = b + 4
        out = []
        for h in tl.static_range(2):
            j = 2 * i + h
            if j < 4:
                sc = tl.load(sb + j, mask=mn, other=0).to(tl.int32) & 63
                m = tl.load(sb + j + 4, mask=mn, other=0).to(tl.int32) & 63
            else:
                a = tl.load(sb + j + 4, mask=mn, other=0).to(tl.int32)
                sc = (a & 15) | ((tl.load(sb + j - 4, mask=mn, other=0).to(tl.int32) >> 6) << 4)
                m = (a >> 4) | ((tl.load(sb + j, mask=mn, other=0).to(tl.int32) >> 6) << 4)
            q = tl.load(b[:, None] + QS + 32 * i + l[None, :], mask=mn[:, None], other=0).to(tl.int32)
            q = (q >> (4 * h)) & 15
            if Q5:
                qh = tl.load(b[:, None] + 16 + l[None, :], mask=mn[:, None], other=0).to(tl.int32)
                q = q | (((qh >> (2 * i + h)) & 1) << 4)
            w = (q.to(tl.float32) * (d * sc.to(tl.float32))[:, None] - (dm * m.to(tl.float32))[:, None]).to(tl.float16)
            xv = tl.load(x_ptr + rr[:, None] * K + (k0 + 32 * h + l)[None, :], mask=mr[:, None], other=0.0)
            acc += tl.dot(xv, tl.trans(w), out_dtype=tl.float32)
    tl.store(y_ptr + rr[:, None] * N + rn[None, :], acc, mask=mr[:, None] & mn[None, :])


@triton.jit
def _rows_q6_k(x_ptr, p_ptr, y_ptr, R, K, N, BN: tl.constexpr, RP: tl.constexpr):
    """Q6_K super-block 210 B / 256: ql[128], qh[64], int8 scales[16] (one per 16), d fp16 at 208. Half n of 128: 4 groups of 32."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    rr = tl.arange(0, RP)
    mr = rr < R
    l = tl.arange(0, 32)
    row = p_ptr + rn.to(tl.int64) * ((K // 256) * 210)
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, 128):
        b = row + (k0 // 256) * 210
        n = (k0 % 256) // 128
        d = _h16(tl.load(b + 208, mask=mn, other=0), tl.load(b + 209, mask=mn, other=0))
        m2 = mn[:, None]
        ql0 = tl.load(b[:, None] + 64 * n + l[None, :], mask=m2, other=0).to(tl.int32)
        ql1 = tl.load(b[:, None] + 64 * n + 32 + l[None, :], mask=m2, other=0).to(tl.int32)
        qh = tl.load(b[:, None] + 128 + 32 * n + l[None, :], mask=m2, other=0).to(tl.int32)
        for g in tl.static_range(4):
            src = ql0 if g % 2 == 0 else ql1
            q = (((src >> (4 * (g // 2))) & 15) | (((qh >> (2 * g)) & 3) << 4)) - 32
            sc = tl.load(b[:, None] + 192 + 8 * n + 2 * g + (l // 16)[None, :], mask=m2, other=0).to(tl.int8).to(tl.float32)
            w = (q.to(tl.float32) * sc * d[:, None]).to(tl.float16)
            xv = tl.load(x_ptr + rr[:, None] * K + (k0 + 32 * g + l)[None, :], mask=mr[:, None], other=0.0)
            acc += tl.dot(xv, tl.trans(w), out_dtype=tl.float32)
    tl.store(y_ptr + rr[:, None] * N + rn[None, :], acc, mask=mr[:, None] & mn[None, :])


@triton.jit
def _rows_iq4_xs(x_ptr, p_ptr, kv_ptr, y_ptr, R, K, N, BN: tl.constexpr, RP: tl.constexpr):
    """IQ4_XS super-block 136 B / 256: d fp16, scales_h u16, scales_l[4], qs[128] (sub-block ib: 16 B, low nibbles = first 16, high = next 16)."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    rr = tl.arange(0, RP)
    mr = rr < R
    l = tl.arange(0, 16)
    row = p_ptr + rn.to(tl.int64) * ((K // 256) * 136)
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, 32):
        b = row + (k0 // 256) * 136
        ib = (k0 % 256) // 32
        d = _h16(tl.load(b, mask=mn, other=0), tl.load(b + 1, mask=mn, other=0))
        sh = tl.load(b + 2, mask=mn, other=0).to(tl.int32) | (tl.load(b + 3, mask=mn, other=0).to(tl.int32) << 8)
        sl = tl.load(b + 4 + ib // 2, mask=mn, other=0).to(tl.int32)
        ls = ((sl >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32).to(tl.float32)
        q = tl.load(b[:, None] + 8 + 16 * ib + l[None, :], mask=mn[:, None], other=0).to(tl.int32)
        for h in tl.static_range(2):
            v = tl.load(kv_ptr + ((q >> (4 * h)) & 15))
            w = (v * dl[:, None]).to(tl.float16)
            xv = tl.load(x_ptr + rr[:, None] * K + (k0 + 16 * h + l)[None, :], mask=mr[:, None], other=0.0)
            acc += tl.dot(xv, tl.trans(w), out_dtype=tl.float32)
    tl.store(y_ptr + rr[:, None] * N + rn[None, :], acc, mask=mr[:, None] & mn[None, :])


_KV = {}


def _kvalues(dev):
    t = _KV.get(dev)
    if t is None:
        t = _KV[dev] = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.float32, device=dev)
    return t


def supported(fmt: str, R: int, K: int) -> bool:
    return fmt in FORMATS and 2 <= R <= MAX_ROWS and K % (256 if fmt.endswith("_k") or fmt == "iq4_xs" else BK) == 0


def gemv(fmt: str, x: torch.Tensor, packed: torch.Tensor, scales=None, global_scale: float = 1.0) -> torch.Tensor:
    """x half [R, K] contiguous -> float32 [R, N]."""
    R, K = x.shape
    N = packed.shape[0]
    y = torch.empty((R, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, BN),)
    if fmt == "q8_0":
        _rows_q8_0[grid](x, packed, y, R, K, N, BN=BN, BK=BK, RP=16, num_warps=4)
    elif fmt == "q4_k":
        _rows_q4_k[grid](x, packed, y, R, K, N, BN=BN, RP=16, num_warps=4)
    elif fmt == "q5_k":
        _rows_q4_k[grid](x, packed, y, R, K, N, BN=BN, RP=16, Q5=True, num_warps=4)
    elif fmt == "iq4_xs":
        _rows_iq4_xs[grid](x, packed, _kvalues(x.device), y, R, K, N, BN=BN, RP=16, num_warps=4)
    elif fmt == "q6_k":
        _rows_q6_k[grid](x, packed, y, R, K, N, BN=BN, RP=16, num_warps=4)
    else:
        _rows_nvfp4[grid](x, packed, scales, y, R, K, N, float(global_scale), BN=BN, BK=BK, RP=16, num_warps=4)
    return y
