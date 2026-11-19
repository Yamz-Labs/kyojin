"""Prefill kernels (Triton, ROCm gfx11 WMMA through tl.dot): fused dequant + GEMM on the STORED bytes of GGUF weights.

  moe_prefill  grouped / sorted-token MoE experts: tokens sorted by expert, one launch for gate+up(+SwiGLU), one for down(+weight);
               Q4_K / Q3_K blocks are dequantised in registers once per (weight tile, expert block), never per token.
  qdense       dense Q8_0 [N, K/32, 34] bytes  x [M, K] -> [M, N] fp32 (prefill rows; decode rows keep the GEMV kernels).
Generic by stored format (FMT constexpr) and shape; nothing keyed on a model name. Activations are fp16 (saturating clamp on the cast),
accumulation fp32. Formats follow ggml: Q4_K 144 B / 256, Q3_K 110 B / 256, Q8_0 34 B / 32.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

FMT_Q4K, FMT_Q3K, FMT_Q8 = 0, 1, 2
BLK_BYTES = {FMT_Q4K: 144, FMT_Q3K: 110, FMT_Q8: 34}
BLK_ELEMS = {FMT_Q4K: 256, FMT_Q3K: 256, FMT_Q8: 32}
# kyojin grouped-kernel fmt id (fq_moe_kq_grouped_mixed_q8) -> FMT here
FROM_GROUPED = {0: FMT_Q4K, 8: FMT_Q3K}


@triton.jit
def _b(W, off):
    return tl.load(W + off).to(tl.int32)


@triton.jit
def _f16(W, off):
    return tl.load((W + off).to(tl.pointer_type(tl.float16))).to(tl.float32)


@triton.jit
def _wtile(W, wrow, sb, FMT: tl.constexpr, BN: tl.constexpr):
    """dequantised fp16 weight tile [BN, 32] of sub-block `sb` (runtime index, 32 K values) of the [BN] rows at byte offsets wrow.
    Q4_K / Q3_K: sb = 8 * superblock + s; Q8_0: sb = block. Runtime (not unrolled) so the register footprint stays one sub-block."""
    lane = tl.arange(0, 32)
    if FMT == 0:                                   # Q4_K: fp16 d, fp16 dmin, 12 scale bytes, 128 nibble bytes
        kb = sb // 8
        s = sb % 8
        blk = wrow + kb * 144
        d = _f16(W, blk)
        dm = _f16(W, blk + 2)
        j = s % 4
        x0 = _b(W, blk + 4 + j)
        x4 = _b(W, blk + 8 + j)
        x8 = _b(W, blk + 12 + j)
        hi = s >= 4
        sc = tl.where(hi, (x8 & 15) | ((x0 >> 6) << 4), x0 & 63).to(tl.float32)
        mn = tl.where(hi, (x8 >> 4) | ((x4 >> 6) << 4), x4 & 63).to(tl.float32)
        qs = tl.load(W + blk[:, None] + (16 + (s // 2) * 32) + lane[None, :]).to(tl.int32)
        q = ((qs >> ((s % 2) * 4)) & 15).to(tl.float32)
        w = (q * (d * sc)[:, None] - (dm * mn)[:, None]).to(tl.float16)
    elif FMT == 1:                                 # Q3_K: 32 high-bit bytes, 64 2-bit bytes, 12 scale bytes, fp16 d
        kb = sb // 8
        s = sb % 8
        half = s // 4
        j = s % 4
        blk = wrow + kb * 110
        d = _f16(W, blk + 108)
        mA = 2 * s
        mB = 2 * s + 1
        lA = (_b(W, blk + 96 + (mA & 7)) >> (4 * (mA >> 3))) & 15
        hA = (_b(W, blk + 104 + (mA & 3)) >> (2 * (mA >> 2))) & 3
        lB = (_b(W, blk + 96 + (mB & 7)) >> (4 * (mB >> 3))) & 15
        hB = (_b(W, blk + 104 + (mB & 3)) >> (2 * (mB >> 2))) & 3
        scA = ((lA | (hA << 4)) - 32).to(tl.float32)
        scB = ((lB | (hB << 4)) - 32).to(tl.float32)
        sc = tl.where(lane[None, :] < 16, scA[:, None], scB[:, None])
        qs = tl.load(W + blk[:, None] + (32 + half * 32) + lane[None, :]).to(tl.int32)
        hm = tl.load(W + blk[:, None] + lane[None, :]).to(tl.int32)
        q = ((qs >> (2 * j)) & 3).to(tl.float32)
        hb = ((hm >> s) & 1).to(tl.float32)
        w = ((q - 4.0 * (1.0 - hb)) * sc * d[:, None]).to(tl.float16)
    else:                                          # Q8_0: fp16 d, 32 int8
        blk = wrow + sb * 34
        d = _f16(W, blk)
        q = tl.load(W + blk[:, None] + 2 + lane[None, :]).to(tl.int8).to(tl.float32)
        w = (q * d[:, None]).to(tl.float16)
    return w


@triton.jit
def _gu_kernel(X, Wg, Wu, H, sorted_pairs, block_expert, P, topk, E, ebytes, lim,
               K: tl.constexpr, N: tl.constexpr, ROWB: tl.constexpr, FMT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert + pid_m)
    if e < E:
        offs_m = pid_m * BM + tl.arange(0, BM)
        pair = tl.load(sorted_pairs + offs_m)
        valid = pair < P
        xrow = (pair // topk).to(tl.int64) * K
        offs_n = pid_n * BN + tl.arange(0, BN)
        wrow = e.to(tl.int64) * ebytes + offs_n.to(tl.int64) * ROWB
        ag = tl.zeros((BM, BN), dtype=tl.float32)
        au = tl.zeros((BM, BN), dtype=tl.float32)
        lane = tl.arange(0, 32)
        for sb in range(0, K // 32):
            x = tl.load(X + xrow[:, None] + sb * 32 + lane[None, :], mask=valid[:, None], other=0.0)
            ag = tl.dot(x, tl.trans(_wtile(Wg, wrow, sb, FMT, BN)), ag)
            au = tl.dot(x, tl.trans(_wtile(Wu, wrow, sb, FMT, BN)), au)
        g = tl.minimum(ag, lim)
        u = tl.minimum(tl.maximum(au, -lim), lim)
        h = g / (1.0 + tl.exp(-g)) * u
        tl.store(H + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], h.to(tl.float16))


@triton.jit
def _down_kernel(Hin, Wd, Out, wts, sorted_pairs, block_expert, P, E, ebytes,
                 K: tl.constexpr, N: tl.constexpr, ROWB: tl.constexpr, FMT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert + pid_m)
    if e < E:
        offs_m = pid_m * BM + tl.arange(0, BM)
        pair = tl.load(sorted_pairs + offs_m)
        valid = pair < P
        xrow = offs_m.to(tl.int64) * K
        offs_n = pid_n * BN + tl.arange(0, BN)
        wrow = e.to(tl.int64) * ebytes + offs_n.to(tl.int64) * ROWB
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        lane = tl.arange(0, 32)
        for sb in range(0, K // 32):
            x = tl.load(Hin + xrow[:, None] + sb * 32 + lane[None, :], mask=valid[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(_wtile(Wd, wrow, sb, FMT, BN)), acc)
        w = tl.load(wts + pair, mask=valid, other=0.0)
        tl.store(Out + pair[:, None].to(tl.int64) * N + offs_n[None, :], acc * w[:, None], mask=valid[:, None])


@triton.jit
def _dense_kernel(X, W, Out, M, N, K: tl.constexpr, ROWB: tl.constexpr, FMT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < M
    xrow = offs_m.to(tl.int64) * K
    offs_n = pid_n * BN + tl.arange(0, BN)
    nm = offs_n < N
    wrow = tl.where(nm, offs_n, 0).to(tl.int64) * ROWB
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    lane = tl.arange(0, 32)
    for sb in range(0, K // 32):
        x = tl.load(X + xrow[:, None] + sb * 32 + lane[None, :], mask=mm[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(_wtile(W, wrow, sb, FMT, BN)), acc)
    tl.store(Out + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], acc, mask=mm[:, None] & nm[None, :])


# ------------------------------------------------------------------------------------------------ int8 path (MMQ-like): Q8_0 weights x Q8 activations on int8 WMMA
@triton.jit
def _quant_x_kernel(X, XQ, XD, M, K, BM: tl.constexpr):
    """per row and 32-block: d = amax / 127, q = round(x / d) (half away from zero, like ggml quantize_row_q8_0); X [M, K] fp16 / fp32"""
    pid_m = tl.program_id(0)
    sb = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    mm = offs_m < M
    x = tl.load(X + offs_m.to(tl.int64)[:, None] * K + sb * 32 + lane[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    d = amax / 127.0
    inv = tl.where(d > 0.0, 1.0 / d, 0.0)
    v = x * inv[:, None]
    q = tl.where(v >= 0.0, v + 0.5, v - 0.5).to(tl.int8)
    tl.store(XQ + offs_m.to(tl.int64)[:, None] * K + sb * 32 + lane[None, :], q, mask=mm[:, None])
    tl.store(XD + offs_m.to(tl.int64) * (K // 32) + sb, d, mask=mm)


@triton.jit
def _dense_i8_kernel(XQ, XD, W, Out, M, N, K: tl.constexpr, ROWB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < M
    xrow = offs_m.to(tl.int64) * K
    xdrow = offs_m.to(tl.int64) * (K // 32)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nm = offs_n < N
    wrow = tl.where(nm, offs_n, 0).to(tl.int64) * ROWB
    lane = tl.arange(0, 32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for sb in range(0, K // 32):
        xq = tl.load(XQ + xrow[:, None] + sb * 32 + lane[None, :], mask=mm[:, None], other=0)
        dx = tl.load(XD + xdrow + sb, mask=mm, other=0.0)
        blk = wrow + sb * 34
        dw = _f16(W, blk)
        wq = tl.load(W + blk[:, None] + 2 + lane[None, :]).to(tl.int8)
        ai = tl.dot(xq, tl.trans(wq))
        acc += ai.to(tl.float32) * dx[:, None] * dw[None, :]
    tl.store(Out + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], acc, mask=mm[:, None] & nm[None, :])


# ------------------------------------------------------------------------------------------------ int8 grouped MoE (llama.cpp mmq style: Q8_K activations)
@triton.jit
def _quant_k_kernel(X, XQ, XD, XS, M, K, BM: tl.constexpr):
    """Q8_K-like activation quantisation: per row and 256-block d = amax / 127, q = round(x / d); also the int sum of q per 32-block (the mins term of Q4_K)"""
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    c = tl.arange(0, 256)
    mm = offs_m < M
    rows = offs_m.to(tl.int64)
    x = tl.load(X + rows[:, None] * K + kb * 256 + c[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    d = amax / 127.0
    inv = tl.where(d > 0.0, 1.0 / d, 0.0)
    v = x * inv[:, None]
    q = tl.where(v >= 0.0, v + 0.5, v - 0.5).to(tl.int8)
    tl.store(XQ + rows[:, None] * K + kb * 256 + c[None, :], q, mask=mm[:, None])
    tl.store(XD + rows * (K // 256) + kb, d, mask=mm)
    qs = tl.sum(tl.reshape(q.to(tl.int32), (BM, 8, 32)), 2)
    s8 = tl.arange(0, 8)
    tl.store(XS + rows[:, None] * (K // 32) + kb * 8 + s8[None, :], qs, mask=mm[:, None])


@triton.jit
def _quant_h_kernel(G, U, XQ, XD, XS, M, K, BM: tl.constexpr):
    """h = silu(g) * u (g, u fp16, already clamped) then the same Q8_K quantisation: the down-projection input without a separate h tensor"""
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    c = tl.arange(0, 256)
    mm = offs_m < M
    rows = offs_m.to(tl.int64)
    ptr = rows[:, None] * K + kb * 256 + c[None, :]
    g = tl.load(G + ptr, mask=mm[:, None], other=0.0).to(tl.float32)
    u = tl.load(U + ptr, mask=mm[:, None], other=0.0).to(tl.float32)
    x = g / (1.0 + tl.exp(-g)) * u
    amax = tl.max(tl.abs(x), 1)
    d = amax / 127.0
    inv = tl.where(d > 0.0, 1.0 / d, 0.0)
    v = x * inv[:, None]
    q = tl.where(v >= 0.0, v + 0.5, v - 0.5).to(tl.int8)
    tl.store(XQ + ptr, q, mask=mm[:, None])
    tl.store(XD + rows * (K // 256) + kb, d, mask=mm)
    qs = tl.sum(tl.reshape(q.to(tl.int32), (BM, 8, 32)), 2)
    s8 = tl.arange(0, 8)
    tl.store(XS + rows[:, None] * (K // 32) + kb * 8 + s8[None, :], qs, mask=mm[:, None])


@triton.jit
def _w4_i8(W, wrow, kb, s, BN: tl.constexpr):
    """Q4_K sub-block s of superblock kb: nibbles as int8 [BN, 32] (0..15), scale / min as int32 [BN]"""
    lane = tl.arange(0, 32)
    blk = wrow + kb * 144
    j = s % 4
    x0 = _b(W, blk + 4 + j)
    x4 = _b(W, blk + 8 + j)
    x8 = _b(W, blk + 12 + j)
    hi = s >= 4
    sc = tl.where(hi, (x8 & 15) | ((x0 >> 6) << 4), x0 & 63)
    mn = tl.where(hi, (x8 >> 4) | ((x4 >> 6) << 4), x4 & 63)
    qs = tl.load(W + blk[:, None] + (16 + (s // 2) * 32) + lane[None, :]).to(tl.int32)
    wq = ((qs >> ((s % 2) * 4)) & 15).to(tl.int8)
    return wq, sc, mn


@triton.jit
def _w3_i8(W, wrow, kb, s, BN: tl.constexpr):
    """Q3_K sub-block s of superblock kb: int8 [BN, 32] = (q - 4 (1 - hb)) * scale (scale folded in; |value| <= 128 is clamped to 127)"""
    lane = tl.arange(0, 32)
    blk = wrow + kb * 110
    half = s // 4
    j = s % 4
    mA = 2 * s
    mB = 2 * s + 1
    lA = (_b(W, blk + 96 + (mA & 7)) >> (4 * (mA >> 3))) & 15
    hA = (_b(W, blk + 104 + (mA & 3)) >> (2 * (mA >> 2))) & 3
    lB = (_b(W, blk + 96 + (mB & 7)) >> (4 * (mB >> 3))) & 15
    hB = (_b(W, blk + 104 + (mB & 3)) >> (2 * (mB >> 2))) & 3
    scA = (lA | (hA << 4)) - 32
    scB = (lB | (hB << 4)) - 32
    sc = tl.where(lane[None, :] < 16, scA[:, None], scB[:, None])
    qs = tl.load(W + blk[:, None] + (32 + half * 32) + lane[None, :]).to(tl.int32)
    hm = tl.load(W + blk[:, None] + lane[None, :]).to(tl.int32)
    qv = ((qs >> (2 * j)) & 3) + 4 * ((hm >> s) & 1) - 4
    return tl.minimum(tl.maximum(qv * sc, -127), 127).to(tl.int8)


@triton.jit
def _gmm_i8_kernel(XQ, XD, XS, W, OUT, wts, sorted_pairs, block_expert, P, topk, E, ebytes, lim,
                   K: tl.constexpr, N: tl.constexpr, ROWB: tl.constexpr, FMT: tl.constexpr, MODE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """grouped (sorted-token) int8 GEMM over one stored expert tensor. MODE 0: gate (x row = pair // topk, min(acc, lim) -> fp16 at the padded row),
    1: up (clamp +-lim -> fp16), 2: down (x row = padded row, acc * wts[pair] -> fp32 at row `pair`)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert + pid_m)
    if e < E:
        offs_m = pid_m * BM + tl.arange(0, BM)
        pair = tl.load(sorted_pairs + offs_m)
        valid = pair < P
        if MODE == 2:
            xr = offs_m.to(tl.int64)
        else:
            xr = (pair // topk).to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        wrow = e.to(tl.int64) * ebytes + offs_n.to(tl.int64) * ROWB
        lane = tl.arange(0, 32)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, K // 256):
            acc_i = tl.zeros((BM, BN), dtype=tl.int32)
            if FMT == 0:
                mn_i = tl.zeros((BM, BN), dtype=tl.int32)
            for s in range(0, 8):
                xq = tl.load(XQ + xr[:, None] * K + (kb * 8 + s) * 32 + lane[None, :], mask=valid[:, None], other=0)
                if FMT == 0:
                    wq, sc, mn = _w4_i8(W, wrow, kb, s, BN)
                    ai = tl.dot(xq, tl.trans(wq), out_dtype=tl.int32)
                    acc_i += ai * sc[None, :]
                    xs = tl.load(XS + xr * (K // 32) + kb * 8 + s, mask=valid, other=0)
                    mn_i += xs[:, None] * mn[None, :]
                else:
                    wq = _w3_i8(W, wrow, kb, s, BN)
                    acc_i = tl.dot(xq, tl.trans(wq), acc_i, out_dtype=tl.int32)
            dx = tl.load(XD + xr * (K // 256) + kb, mask=valid, other=0.0)
            if FMT == 0:
                d = _f16(W, wrow + kb * 144)
                dm = _f16(W, wrow + kb * 144 + 2)
                acc += dx[:, None] * (d[None, :] * acc_i.to(tl.float32) - dm[None, :] * mn_i.to(tl.float32))
            else:
                d = _f16(W, wrow + kb * 110 + 108)
                acc += dx[:, None] * d[None, :] * acc_i.to(tl.float32)
        if MODE == 0:
            tl.store(OUT + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], tl.minimum(acc, lim).to(tl.float16))
        elif MODE == 1:
            tl.store(OUT + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], tl.minimum(tl.maximum(acc, -lim), lim).to(tl.float16))
        else:
            w = tl.load(wts + pair, mask=valid, other=0.0)
            tl.store(OUT + pair[:, None].to(tl.int64) * N + offs_n[None, :], acc * w[:, None], mask=valid[:, None])


@triton.jit
def _deq_q8_kernel(W, OUT, N, K: tl.constexpr, ROWB: tl.constexpr, BN: tl.constexpr, SPP: tl.constexpr):
    """stored Q8_0 [N, K/32, 34] -> fp16 [N, K]; one program = BN rows x SPP sub-blocks"""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nm = offs_n < N
    wrow = tl.where(nm, offs_n, 0).to(tl.int64) * ROWB
    lane = tl.arange(0, 32)
    for i in range(0, SPP):
        sb = pid_k * SPP + i
        w = _wtile(W, wrow, sb, 2, BN)
        tl.store(OUT + offs_n.to(tl.int64)[:, None] * K + sb * 32 + lane[None, :], w, mask=nm[:, None])


# ------------------------------------------------------------------------------------------------ host side
F16MAX = 65504.0


def to_f16(x):
    """saturating fp32 / fp16 -> fp16 (hidden streams keep the clamp)"""
    return x.float().clamp(-F16MAX, F16MAX).half() if x.dtype != torch.float16 else x


MOE_CFG = None            # None = chosen from the rows per expert (see moe_cfg)
DENSE_CFG = {"BM": 128, "BN": 128, "warps": 8, "stages": 1}


def moe_cfg(T, S, E):
    """tile rows follow the expected rows per expert (tokens * topk / experts): a 128-row tile on 64 rows wastes half the MMA work"""
    r = T * S / max(E, 1)
    if r >= 96:
        return {"BM": 128, "BN": 64, "warps": 8, "stages": 1}
    if r >= 40:
        return {"BM": 64, "BN": 64, "warps": 4, "stages": 1}
    return {"BM": 32, "BN": 64, "warps": 4, "stages": 1}


def route_blocks(sel, E, BM):
    """sel [T, topk] expert ids -> (sorted_pairs int32 [maxb*BM] padded with P, block_expert int32 [maxb] (E = idle block), maxb). No host sync."""
    T, S = sel.shape
    P = T * S
    ex = sel.reshape(-1)
    order = torch.argsort(ex, stable=True)
    counts = torch.bincount(ex, minlength=E)
    nblk = (counts + (BM - 1)) // BM
    csum = torch.cumsum(nblk, 0)
    bstart = csum - nblk
    cstart = torch.cumsum(counts, 0) - counts
    es = ex[order]
    ppos = bstart[es] * BM + (torch.arange(P, device=sel.device) - cstart[es])
    maxb = (P + BM - 1) // BM + E
    sp = torch.full((maxb * BM,), P, dtype=torch.int32, device=sel.device)
    sp[ppos] = order.to(torch.int32)
    be = torch.searchsorted(csum, torch.arange(maxb, device=sel.device), right=True).to(torch.int32)
    return sp, be, maxb


def moe_prefill(z, sel, w, bufs, fmts, lim, nff, nout, rowb, ebytes, E, cfg=None):
    """z [T, K] activations (fp32 / fp16), sel [T, topk] int, w [T, topk] fp32 routing weights.
    bufs = (gate, up, down) raw uint8 expert buffers (expert-major, row-major, stored blocks), fmts = (fmt gate/up, fmt down) as FMT ids,
    rowb = (row bytes gate/up, row bytes down), ebytes = (bytes per expert gate/up, down), nff = rows of gate / up (= K of down), nout = rows of down.
    Returns sum_k w[t, k] * down_e(silu(min(gate_e z, lim)) * clamp(up_e z, +-lim)) as fp32 [T, nout]."""
    T, S = sel.shape
    c = cfg or MOE_CFG or moe_cfg(T, S, E)
    BM, BN = c["BM"], c["BN"]
    P = T * S
    K = z.shape[1]
    sp, be, maxb = route_blocks(sel.to(torch.int64), E, BM)
    x = to_f16(z).contiguous()
    H = torch.empty(maxb * BM, nff, dtype=torch.float16, device=z.device)
    _gu_kernel[(maxb, nff // BN)](x, bufs[0], bufs[1], H, sp, be, P, S, E, ebytes[0], float(lim), K=K, N=nff, ROWB=rowb[0], FMT=fmts[0], BM=BM, BN=BN,
                                  num_warps=c["warps"], num_stages=c["stages"])
    out = torch.empty(P, nout, dtype=torch.float32, device=z.device)
    _down_kernel[(maxb, nout // BN)](H, bufs[2], out, w.reshape(-1).float().contiguous(), sp, be, P, E, ebytes[1], K=nff, N=nout, ROWB=rowb[1], FMT=fmts[1],
                                     BM=BM, BN=BN, num_warps=c["warps"], num_stages=c["stages"])
    return out.view(T, S, nout).sum(1)


def qdense(x, packed, fmt, cfg=None):
    """x [.., K] -> [.., N] fp32 = x @ W.T, W stored as packed [N, K/blk, bytes] (uint8)"""
    c = cfg or DENSE_CFG
    N, nb, bb = packed.shape
    K = nb * BLK_ELEMS[fmt]
    lead = x.shape[:-1]
    x2 = to_f16(x.reshape(-1, K)).contiguous()
    M = x2.shape[0]
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, c["BM"]), triton.cdiv(N, c["BN"]))
    _dense_kernel[grid](x2, packed, out, M, N, K=K, ROWB=nb * bb, FMT=fmt, BM=c["BM"], BN=c["BN"], num_warps=c["warps"], num_stages=c["stages"])
    return out.reshape(*lead, N)


def qdense_i8(x, packed, cfg=None):
    """x [.., K] -> [.., N] fp32 = x @ W.T for stored Q8_0 W [N, K/32, 34]: activations quantised to Q8 (per row, 32-block) and multiplied on int8 WMMA"""
    c = cfg or DENSE_CFG
    N, nb, bb = packed.shape
    K = nb * 32
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    xq = torch.empty(M, K, dtype=torch.int8, device=x.device)
    xd = torch.empty(M, nb, dtype=torch.float32, device=x.device)
    _quant_x_kernel[(triton.cdiv(M, 32), nb)](x2, xq, xd, M, K, BM=32, num_warps=1)
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    _dense_i8_kernel[(triton.cdiv(M, c["BM"]), triton.cdiv(N, c["BN"]))](xq, xd, packed, out, M, N, K=K, ROWB=nb * bb, BM=c["BM"], BN=c["BN"],
                                                                         num_warps=c["warps"], num_stages=c["stages"])
    return out.reshape(*lead, N)


I8_CFG = {"BM": 64, "BN": 64, "warps": 8, "stages": 1}


def quant_k(x):
    """x [M, K] -> (q int8 [M, K], d fp32 [M, K/256], sums int32 [M, K/32]); K a multiple of 256"""
    M, K = x.shape
    x = x.contiguous()
    q = torch.empty(M, K, dtype=torch.int8, device=x.device)
    d = torch.empty(M, K // 256, dtype=torch.float32, device=x.device)
    s = torch.empty(M, K // 32, dtype=torch.int32, device=x.device)
    _quant_k_kernel[(triton.cdiv(M, 8), K // 256)](x, q, d, s, M, K, BM=8, num_warps=4)
    return q, d, s


def moe_prefill_i8(z, sel, w, bufs, fmts, lim, nff, nout, rowb, ebytes, E, cfg=None):
    """same contract as moe_prefill; activations quantised to Q8_K (llama.cpp's vec_dot_type for k-quants), int8 WMMA, scales applied per superblock"""
    T, S = sel.shape
    c = cfg or I8_CFG
    BM, BN = c["BM"], c["BN"]
    P = T * S
    K = z.shape[1]
    sp, be, maxb = route_blocks(sel.to(torch.int64), E, BM)
    zq, zd, zs = quant_k(z.float())
    rows = maxb * BM
    G = torch.empty(rows, nff, dtype=torch.float16, device=z.device)
    U = torch.empty(rows, nff, dtype=torch.float16, device=z.device)
    kw = dict(num_warps=c["warps"], num_stages=c["stages"])
    _gmm_i8_kernel[(maxb, nff // BN)](zq, zd, zs, bufs[0], G, G, sp, be, P, S, E, ebytes[0], float(lim), K=K, N=nff, ROWB=rowb[0], FMT=fmts[0], MODE=0, BM=BM, BN=BN, **kw)
    _gmm_i8_kernel[(maxb, nff // BN)](zq, zd, zs, bufs[1], U, U, sp, be, P, S, E, ebytes[0], float(lim), K=K, N=nff, ROWB=rowb[0], FMT=fmts[0], MODE=1, BM=BM, BN=BN, **kw)
    hq = torch.empty(rows, nff, dtype=torch.int8, device=z.device)
    hd = torch.empty(rows, nff // 256, dtype=torch.float32, device=z.device)
    hs = torch.empty(rows, nff // 32, dtype=torch.int32, device=z.device)
    _quant_h_kernel[(triton.cdiv(rows, 8), nff // 256)](G, U, hq, hd, hs, rows, nff, BM=8, num_warps=4)
    out = torch.empty(P, nout, dtype=torch.float32, device=z.device)
    _gmm_i8_kernel[(maxb, nout // BN)](hq, hd, hs, bufs[2], out, w.reshape(-1).float().contiguous(), sp, be, P, S, E, ebytes[1], float(lim), K=nff, N=nout, ROWB=rowb[1],
                                       FMT=fmts[1], MODE=2, BM=BM, BN=BN, **kw)
    return out.view(T, S, nout).sum(1)


DENSE_TM_MIN_M = 1024        # rows from which dequantise-to-fp16 + vendor GEMM beats the fused kernel (the dequant pass is ~1 % of the GEMM at M = 8192)


def mm_f32(a, b):
    """a [M, K] fp16 @ b [K, N] fp16 -> fp32 [M, N]. fp16 GEMM output (fp32 accumulation inside) upcast with the saturating clamp: the out_dtype=float32
    variant of torch.mm falls off the fast kernel on this box (measured: attention projections 1.7 TFLOPS instead of 15-35)."""
    return torch.nan_to_num(torch.mm(a, b).float(), nan=0.0, posinf=F16MAX, neginf=-F16MAX)


def dequant_q8(packed):
    """stored Q8_0 [N, K/32, 34] uint8 -> fp16 [N, K]"""
    N, nb, bb = packed.shape
    K = nb * 32
    out = torch.empty(N, K, dtype=torch.float16, device=packed.device)
    spp = 8 if nb % 8 == 0 else 1
    _deq_q8_kernel[(triton.cdiv(N, 64), nb // spp)](packed, out, N, K=K, ROWB=nb * bb, BN=64, SPP=spp, num_warps=4)
    return out


def qdense_tm(x, packed):
    """x [.., K] -> [.., N] fp32 = x @ W.T: dequantise the stored Q8_0 weight to fp16 once per call, then the vendor fp16 GEMM"""
    N, nb, bb = packed.shape
    K = nb * 32
    lead = x.shape[:-1]
    x2 = to_f16(x.reshape(-1, K)).contiguous()
    out = mm_f32(x2, dequant_q8(packed).T)
    return out.reshape(*lead, N)
