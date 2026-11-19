"""Prefill kernels (Triton, ROCm gfx11 WMMA through tl.dot): fused dequant + GEMM on the STORED bytes of GGUF weights.

  moe_prefill  grouped / sorted-token MoE experts: tokens sorted by expert, one launch for gate+up(+SwiGLU), one for down(+weight);
               Q4_K / Q3_K blocks are dequantised in registers once per (weight tile, expert block), never per token.
  qdense       dense Q8_0 [N, K/32, 34] bytes  x [M, K] -> [M, N] fp32 (prefill rows; decode rows keep the GEMV kernels).
Generic by stored format (FMT constexpr) and shape; nothing keyed on a model name. Activations are fp16 (saturating clamp on the cast),
accumulation fp32. Formats follow ggml: Q4_K 144 B / 256, Q3_K 110 B / 256, Q8_0 34 B / 32.
"""
from __future__ import annotations
import contextlib
import os
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
               K: tl.constexpr, N: tl.constexpr, ROWB: tl.constexpr, FMT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NF: tl.constexpr):
    pid_m = tl.program_id(1 if NF else 0)         # NF: grid (N tiles, row blocks): the N tiles of one row block (and the blocks of one expert) run back to back
    pid_n = tl.program_id(0 if NF else 1)
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
                 K: tl.constexpr, N: tl.constexpr, ROWB: tl.constexpr, FMT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NF: tl.constexpr):
    pid_m = tl.program_id(1 if NF else 0)
    pid_n = tl.program_id(0 if NF else 1)
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
    """stored Q8_0 [N, K/32, 34] -> fp16 W^T [K, N]; one program = BN rows x SPP sub-blocks"""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nm = offs_n < N
    wrow = tl.where(nm, offs_n, 0).to(tl.int64) * ROWB
    lane = tl.arange(0, 32)
    for i in range(0, SPP):
        sb = pid_k * SPP + i
        w = _wtile(W, wrow, sb, 2, BN)
        # transposed store: OUT is W^T [K, N] row-major (the vendor GEMM runs about 2x faster on the NN layout than on the NT view of [N, K])
        tl.store(OUT + (sb * 32 + lane).to(tl.int64)[:, None] * N + offs_n[None, :], tl.trans(w), mask=nm[None, :])


# ------------------------------------------------------------------------------------------------ k-pool indexer scoring (prefill rows)
IX_ALLOW_CPU = [False]      # tests only
CK_BIAS = [os.environ.get("HY4_PF_CK", "0") == "1"]      # ix_select also writes the CK attention bias


def ix_scores(iq, pk, wt):
    """sc [T, P] = sum_h wt[t, h] * relu(iq[t, h, :] . pk[p, :]): iq [T, H, d], pk [P, d], wt [T, H] (fp32 in / out). fp16 GEMM with fp32 accumulation instead of
    an fp32 einsum plus two [T, H, P] fp32 temporaries (about 2 GB each at T = 8192); the weighted head sum is one batched GEMM."""
    T, H, d = iq.shape
    P = pk.shape[0]
    kq = torch.mm(to_f16(iq.reshape(T * H, d)), to_f16(pk).T.contiguous())          # [T*H, P] fp16
    kq.relu_()
    sc = torch.bmm(to_f16(wt)[:, None, :], kq.view(T, H, P))                         # [T, 1, P]
    return sc[:, 0].float()


def _pf_indexer_forward(self, x, qr, n_keys=None):
    """FusedKPoolIndexer.forward with ix_scores for T > 16 rows (same maths otherwise; see indexer_fused.py)"""
    import math
    base = type(self).__mro__[1]
    if x.shape[0] <= 16 or (not x.is_cuda and not IX_ALLOW_CPU[0]):
        return base.forward(self, x, qr, n_keys)
    mod = __import__(base.__module__, fromlist=["_rms"])
    _rms = mod._rms
    T, k, d, H = x.shape[0], self.k, self.d, self.H
    tm = getattr(self, "_tm", None)
    t = tm.t if tm is not None and tm.on else (lambda n: contextlib.nullcontext())
    start = self.n
    self._max_pos = start + T
    pos = torch.arange(start, start + T, device=self.dev)
    with t("f ix: key pool"):
        with t("f ixk: attn_k mm"):
            ikr = x @ self.w["attn_k"].T
        with t("f ixk: norm"):
            ik = _rms(ikr, self.w["k_norm"], self.eps)
        with t("f ixk: value/gate mm"):
            vg_new = torch.cat([ik @ self.w["value"].T, ik @ self.w["gate"].T], -1)
        buf = vg_new if self.vg is None else torch.cat([self.vg, vg_new], 0)
        m = buf.shape[0] // k
        if m:
            if self.P + m > self.pool.shape[0]:
                big = torch.empty(max(2 * self.pool.shape[0], self.P + m), d, device=self.dev)
                big[:self.P] = self.pool[:self.P]
                self.pool = big
            with t("f ixk: pool groups"):
                self.pool[self.P:self.P + m] = self._pool_groups(buf[:m * k], self.P)
            self.P += m
        self.vg = buf[m * k:] if buf.shape[0] - m * k else None
        self.n += T
    with t("f ix: q_b + rope + proj"):
        iq = self._rope((qr @ self.w["q_b"].T).reshape(T, H, d), pos)
        wt = (x.float() @ self.w["proj"].T) / math.sqrt(d * H)
    P = self.P
    if PF_FUSE and PF_SPARSE and IX_BLOCKED:      # blocked score + linear-time top-k: no [T, P] fp32 matrix, cost linear in N
        N = self.n if n_keys is None else n_keys
        last = start + T - 1
        if (last // k if self.tail else (last + 1) // k) <= self.n_top:
            return None
        if N >= SPARSE_MINN and N >= P * k:
            with t("f ix: scores + select (blocked)"):
                self.last_scores = None
                return ix_select_blocked(to_f16(iq), to_f16(self.pool[:P]), wt, start, k, self.tail, self.n_top, N)
    with t("f ix: scores"):
        if PF_FUSE:
            sc = ix_scores_fused(to_f16(iq), to_f16(self.pool[:P]), wt, start, k, self.tail)
        else:
            sc = ix_scores(iq, self.pool[:P], wt)
            gend = torch.arange(P, device=self.dev) * k + (k - 1)
            vis = (gend[None] < pos[:, None]) if self.tail else (gend[None] <= pos[:, None])
            sc = torch.where(vis, sc, torch.full_like(sc, -math.inf))
        self.last_scores = sc
    last = start + T - 1
    nvis_max = last // k if self.tail else (last + 1) // k
    if nvis_max <= self.n_top:
        return None
    N = self.n if n_keys is None else n_keys
    if PF_FUSE and N >= P * k:
        with t("f ix: select + allow + bias"):
            sp = PF_SPARSE and N >= SPARSE_MINN
            return ix_select(sc, start, N, self.n_top, k, self.tail, want_bias=CK_BIAS[0] and not sp, sparse=sp)
    with t("f ix: topk"):
        idx = torch.topk(sc, min(self.n_top, P), dim=1).indices
    with t("f ix: allow mask"):
        gsel = torch.zeros(T, P, dtype=torch.bool, device=self.dev)
        gsel.scatter_(1, idx, True)
        gsel &= vis
        allow = torch.zeros(T, N, dtype=torch.bool, device=self.dev)
        allow[:, :P * k] = gsel.repeat_interleave(k, dim=1)
        j = torch.arange(N, device=self.dev)[None]
        lo = (pos - (pos % k)) if self.tail else (pos - ((pos + 1) % k) + 1)
        allow |= (j >= lo[:, None]) & (j <= pos[:, None])
    return allow


def pf_indexer(obj, tm=None):
    """re-class a FusedKPoolIndexer instance so its forward uses ix_scores for prefill rows (tm = optional stage timers)"""
    obj._tm = tm
    if not getattr(type(obj), "_pf", False):
        obj.__class__ = type("PFIndexer", (type(obj),), {"forward": _pf_indexer_forward, "_pf": True})
    return obj


# ------------------------------------------------------------------------------------------------ host side
F16MAX = 65504.0


@triton.jit
def _sat_half_kernel(X, O, n, BLK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
    m = i < n
    x = tl.load(X + i, mask=m, other=0.0)
    tl.store(O + i, tl.minimum(tl.maximum(x, -65504.0), 65504.0).to(tl.float16), mask=m)


_F16_MEMO = [None, None]      # (source tensor, converted) and its key: the fp32 stream of one layer feeds several GEMMs (wq_a, wkv_a, wgate / router, rdown, shared gate + up)


def to_f16(x):
    """saturating fp32 / fp16 -> fp16 (hidden streams keep the clamp): one fused pass, and the conversion of the last fp32 input is reused while that tensor is
    unchanged (same storage, same in-place version)"""
    if x.dtype == torch.float16:
        return x
    if x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous() or x.numel() < (1 << 20):
        return x.float().clamp(-F16MAX, F16MAX).half()
    key = (x.data_ptr(), x.shape, x._version)
    if _F16_MEMO[0] is not None and _F16_MEMO[1] == key:      # same storage address + shape + in-place version; the memo holds the source, so the storage cannot be reused
        return _F16_MEMO[0][1]
    o = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    n = x.numel()
    _sat_half_kernel[(triton.cdiv(n, 4096),)](x, o, n, BLK=4096, num_warps=4)
    _F16_MEMO[0], _F16_MEMO[1] = (x, o), key
    return o


# ------------------------------------------------------------------------------------------------ fused cast helpers (HY4_PF_FUSE=1)
PF_FUSE = os.environ.get("HY4_PF_FUSE", "0") == "1"


@triton.jit
def _f32_sat_kernel(X, O, n, BLK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
    m = i < n
    x = tl.load(X + i, mask=m, other=0.0).to(tl.float32)
    x = tl.where(x != x, 0.0, tl.minimum(tl.maximum(x, -65504.0), 65504.0))
    tl.store(O + i, x, mask=m)


def f32_sat(o):
    """fp16 GEMM output -> fp32 with nan -> 0 and +-inf -> +-65504: torch.nan_to_num(o.float(), ...) in ONE pass instead of three"""
    if not PF_FUSE or o.dtype != torch.float16 or not o.is_cuda or not o.is_contiguous():
        return torch.nan_to_num(o.float(), nan=0.0, posinf=F16MAX, neginf=-F16MAX)
    out = torch.empty(o.shape, dtype=torch.float32, device=o.device)
    n = o.numel()
    _f32_sat_kernel[(triton.cdiv(n, 4096),)](o, out, n, BLK=4096, num_warps=4)
    return out


def register_f16(x, x16):
    """tell to_f16 that x16 is the saturating fp16 copy of the fp32 tensor x (a producer kernel wrote both)"""
    if x.dtype == torch.float32 and x.is_contiguous() and x.numel() >= (1 << 20):
        _F16_MEMO[0], _F16_MEMO[1] = (x, x16), (x.data_ptr(), x.shape, x._version)


# ------------------------------------------------------------------------------------------------ glue fusions (each behind its own env flag, default OFF; read per call so one boot can A/B arms; a fused topk-sum + fp16 copy was measured: not bit-equal, -0.2 ms per layer, dropped)
def _flag(name):
    return os.environ.get(name, "0") == "1"


@triton.jit
def _swiglu16_kernel(G, U, O, n, BLK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
    m = i < n
    g = tl.load(G + i, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + i, mask=m, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, tl.minimum(tl.maximum(g, -65504.0), 65504.0))
    u = tl.where(u != u, 0.0, tl.minimum(tl.maximum(u, -65504.0), 65504.0))
    h = g / (1.0 + tl.exp(-g)) * u
    tl.store(O + i, tl.minimum(tl.maximum(h, -65504.0), 65504.0).to(tl.float16), mask=m)


def swiglu16(g16, u16):
    """silu(f32_sat(g)) * f32_sat(u) -> saturating fp16, from the two raw fp16 GEMM outputs in ONE pass (was: 2 x f32_sat, silu, mul, sat_half = 5 passes over fp32)"""
    o = torch.empty(g16.shape, dtype=torch.float16, device=g16.device)
    n = o.numel()
    _swiglu16_kernel[(triton.cdiv(n, 4096),)](g16, u16, o, n, BLK=4096, num_warps=4)
    return o


@triton.jit
def _add_sat2_kernel(A, B, O, n, BLK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
    m = i < n
    a = tl.load(A + i, mask=m, other=0.0).to(tl.float32)
    b = tl.load(B + i, mask=m, other=0.0).to(tl.float32)
    a = tl.where(a != a, 0.0, tl.minimum(tl.maximum(a, -65504.0), 65504.0))
    b = tl.where(b != b, 0.0, tl.minimum(tl.maximum(b, -65504.0), 65504.0))
    tl.store(O + i, a + b, mask=m)


def add_sat2(a16, b16):
    """f32_sat(a) + f32_sat(b) -> fp32 in one pass (same values as the two casts and the fp32 add)"""
    o = torch.empty(a16.shape, dtype=torch.float32, device=a16.device)
    n = o.numel()
    _add_sat2_kernel[(triton.cdiv(n, 4096),)](a16, b16, o, n, BLK=4096, num_warps=4)
    return o


def ffn_tail_fused(lat, x, rup_p, gate_p, up_p, down_p, tm=None):
    """MoE layer tail on stored Q8_0 matrices: rup(lat) + shared_down(silu(shared_gate(x)) * shared_up(x)) as fp32 [T, D], from raw fp16 GEMM outputs.
    HY4_PF_G_SWIGLU=1: silu * mul + casts in one kernel; HY4_PF_G_ADD=1: both saturating casts and the add in one kernel."""
    y16 = qdense_raw16(lat, rup_p, tm)
    g16 = qdense_raw16(x, gate_p, tm)
    u16 = qdense_raw16(x, up_p, tm)
    if _flag("HY4_PF_G_SWIGLU"):
        h = swiglu16(g16, u16)
    else:
        h = torch.nn.functional.silu(f32_sat(g16)) * f32_sat(u16)
    s16 = qdense_raw16(h, down_p, tm)
    if _flag("HY4_PF_G_ADD"):
        return add_sat2(y16, s16)
    return f32_sat(y16) + f32_sat(s16)


def dense_ffn_fused(x, gate_p, up_p, down_p, tm=None):
    """dense-layer MLP (stored Q8_0): down(silu(gate(x)) * up(x)) -> fp32 [T, D] with the SwiGLU glue in one kernel (HY4_PF_G_SWIGLU=1)"""
    h = swiglu16(qdense_raw16(x, gate_p, tm), qdense_raw16(x, up_p, tm))
    return qdense_tm(h, down_p, tm)


# ------------------------------------------------------------------------------------------------ fused indexer scorer + exact top-k selection (HY4_PF_FUSE=1)
@triton.jit
def _ix_score_kernel(IQ, PK, WT, SC, T, P, start, H: tl.constexpr, D: tl.constexpr, KP: tl.constexpr, TAIL: tl.constexpr, BT: tl.constexpr, BP: tl.constexpr):
    """SC[t, p] = sum_h WT[t, h] * relu(IQ[t, h, :] . PK[p, :]) for the groups visible to row t (group end < pos with TAIL, <= pos without), -inf elsewhere.
    IQ [T, H, D] fp16, PK [P, D] fp16, WT [T, H] fp32, SC [T, P] fp32. Tiles that no row of the block can see are skipped."""
    pt = tl.program_id(0)
    pp = tl.program_id(1)
    t = pt * BT + tl.arange(0, BT)
    p = pp * BP + tl.arange(0, BP)
    pos = start + t
    gend = p * KP + (KP - 1)
    if TAIL:
        vis = gend[None, :] < pos[:, None]
    else:
        vis = gend[None, :] <= pos[:, None]
    tm = t < T
    out = SC + t[:, None].to(tl.int64) * P + p[None, :]
    om = tm[:, None] & (p[None, :] < P)
    pmax = start + tl.minimum(pt * BT + BT - 1, T - 1)
    first_end = pp * BP * KP + (KP - 1)
    skip = (first_end >= pmax) if TAIL else (first_end > pmax)
    if skip:
        tl.store(out, tl.full([BT, BP], float("-inf"), dtype=tl.float32), mask=om)
    else:
        dd = tl.arange(0, D)
        pk = tl.load(PK + p[None, :].to(tl.int64) * D + dd[:, None], mask=(p[None, :] < P), other=0.0)      # [D, BP]
        acc = tl.zeros([BT, BP], dtype=tl.float32)
        for h in range(H):
            q = tl.load(IQ + (t[:, None].to(tl.int64) * H + h) * D + dd[None, :], mask=tm[:, None], other=0.0)      # [BT, D]
            w = tl.load(WT + t * H + h, mask=tm, other=0.0)
            s = tl.dot(q, pk)
            acc += tl.maximum(s, 0.0) * w[:, None]
        tl.store(out, tl.where(vis, acc, float("-inf")), mask=om)


def ix_scores_fused(iq16, pk16, wt, start, kpool, tail):
    """iq16 [T, H, d] fp16, pk16 [P, d] fp16, wt [T, H] fp32 -> sc [T, P] fp32 (invisible groups = -inf); fp32 accumulation, no [T*H, P] temporaries"""
    T, H, d = iq16.shape
    P = pk16.shape[0]
    sc = torch.empty(T, P, dtype=torch.float32, device=iq16.device)
    _ix_score_kernel[(triton.cdiv(T, 64), triton.cdiv(P, 64))](iq16.contiguous(), pk16.contiguous(), wt.contiguous(), sc, T, P, start, H=H, D=d, KP=kpool,
                                                                 TAIL=int(bool(tail)), BT=64, BP=64, num_warps=4)
    return sc


@triton.jit
def _ix_select_kernel(SC, ALLOW, BIAS, GIDX, NSEL, P, N, NP, start, n_top, PP: tl.constexpr, KP: tl.constexpr, TAIL: tl.constexpr, WRITE_BIAS: tl.constexpr, SPARSE: tl.constexpr):
    """one row per program: the n_top highest scores (ties: lowest group index first; invisible groups never selected) -> group mask expanded to keys,
    OR the row's own open group (tail); writes allow [T, N] (int8) and optionally the CK bias [T, NP] fp16 (0 / -30000)"""
    r = tl.program_id(0)
    pos = start + r
    g = tl.arange(0, PP)
    gm = g < P
    sc = tl.load(SC + r.to(tl.int64) * P + g, mask=gm, other=float("-inf"))
    gend = g * KP + (KP - 1)
    vis = gm & ((gend < pos) if TAIL else (gend <= pos))
    b = (sc + 0.0).to(tl.int32, bitcast=True)      # + 0.0: -0.0 becomes +0.0 (equal floats must give equal keys)
    key = tl.where(b < 0, b ^ 0x7FFFFFFF, b).to(tl.int64)
    lo = tl.full([1], -2147483648, dtype=tl.int64)
    hi = tl.full([1], 2147483647, dtype=tl.int64)
    for _ in range(32):
        mid = lo + (hi - lo + 1) // 2
        cnt = tl.sum((key >= mid).to(tl.int32), axis=0)
        ge = cnt >= n_top
        lo = tl.where(ge, mid, lo)
        hi = tl.where(ge, hi, mid - 1)
    gt = key > lo
    eq = key == lo
    need = n_top - tl.sum(gt.to(tl.int32), axis=0)
    rank = tl.cumsum(eq.to(tl.int32), axis=0)
    sel = (gt | (eq & (rank <= need))) & vis
    if SPARSE:      # compact group ids (ascending) + count; the own open group is added by the attention kernel
        si = sel.to(tl.int32)
        tl.store(GIDX + r.to(tl.int64) * n_top + tl.cumsum(si, axis=0) - 1, g, mask=sel)
        tl.store(NSEL + r, tl.sum(si, axis=0))
        return
    selx = tl.reshape(tl.broadcast_to(sel.to(tl.int8)[:, None], [PP, KP]), [PP * KP])
    j = tl.arange(0, PP * KP)
    if TAIL:
        lo_t = pos - (pos % KP)
    else:
        lo_t = pos - ((pos + 1) % KP) + 1
    al = (selx != 0) | ((j >= lo_t) & (j <= pos))
    tl.store(ALLOW + r.to(tl.int64) * N + j, al.to(tl.int8), mask=j < N)
    if WRITE_BIAS:
        tl.store(BIAS + r.to(tl.int64) * NP + j, tl.where(al, 0.0, -30000.0).to(tl.float16), mask=j < NP)
        j2 = PP * KP + tl.arange(0, 8)
        tl.store(BIAS + r.to(tl.int64) * NP + j2, tl.full([8], -30000.0, dtype=tl.float16), mask=j2 < NP)


SPARSE_MINN = int(os.environ.get("HY4_PF_SPARSE_MINN", "0"))     # HY4_PF_SPARSE=1 (per-query sparse gather attention) only when the key count N >= this
PF_SPARSE = os.environ.get("HY4_PF_SPARSE", "0") == "1"


class PfSparse:
    """indexer selection of a prefill call without the dense mask: gidx [T, n_top] int32 group ids (ascending), nsel [T] int32 valid counts, kpool keys per group,
    tail = the own-open-group convention of the indexer. Visible keys of row t = its groups + the open group (mla_sparse); to_dense() gives the equivalent allow mask"""

    def __init__(self, gidx, nsel, kpool, tail, start, N):
        self.gidx, self.nsel, self.kpool, self.tail, self.start, self.N = gidx, nsel, kpool, bool(tail), start, N

    def to_dense(self):
        T, k = self.gidx.shape[0], self.kpool
        dev = self.gidx.device
        P = (self.N + k - 1) // k
        ok = torch.arange(self.gidx.shape[1], device=dev)[None] < self.nsel[:, None]
        gs = torch.zeros(T, P + 1, dtype=torch.bool, device=dev)
        gs.scatter_(1, torch.where(ok, self.gidx.long(), P), True)
        allow = gs[:, :P].repeat_interleave(k, dim=1)[:, :self.N].clone()
        pos = self.start + torch.arange(T, device=dev)
        lo = (pos - (pos % k)) if self.tail else (pos - ((pos + 1) % k) + 1)
        j = torch.arange(self.N, device=dev)[None]
        allow |= (j >= lo[:, None]) & (j <= pos[:, None])
        return allow


def ix_select(sc, start, N, n_top, kpool, tail, want_bias=True, sparse=False):
    """sc [T, P] fp32 (-inf = invisible) -> allow [T, N] bool (+ its CK bias [T, ceil8(N)] fp16 attached as allow._ck_bias when want_bias);
    sparse=True -> PfSparse (compact group ids, no [T, N] mask)"""
    T, P = sc.shape
    PP = 1 << max(1, int(P).bit_length())          # strictly above P: PP * kpool >= N + 1
    NP = (N + 7) // 8 * 8
    if sparse:
        gidx = torch.empty(T, n_top, dtype=torch.int32, device=sc.device)
        nsel = torch.empty(T, dtype=torch.int32, device=sc.device)
        _ix_select_kernel[(T,)](sc, sc, sc, gidx, nsel, P, N, NP, start, n_top, PP=PP, KP=kpool, TAIL=int(bool(tail)), WRITE_BIAS=False, SPARSE=True, num_warps=8)
        return PfSparse(gidx, nsel, kpool, tail, start, N)
    allow = torch.empty(T, N, dtype=torch.bool, device=sc.device)
    bias = torch.empty(T, NP, dtype=torch.float16, device=sc.device) if want_bias else allow
    _ix_select_kernel[(T,)](sc, allow.view(torch.int8), bias, sc, sc, P, N, NP, start, n_top, PP=PP, KP=kpool, TAIL=int(bool(tail)), WRITE_BIAS=bool(want_bias), SPARSE=False, num_warps=8)
    if want_bias:
        allow._ck_bias = bias
    return allow


# ------------------------------------------------------------------------------------------------ blocked score + linear-time exact top-k (HY4_PF_SPARSE=1)
@triton.jit
def _fkey(x):
    """fp32 -> int32 key with the same order (-0.0 == +0.0)"""
    b = (x + 0.0).to(tl.int32, bitcast=True)
    return tl.where(b < 0, b ^ 0x7FFFFFFF, b)


@triton.jit
def _kth_key(key, valid, k):
    """largest L with count(valid & key >= L) >= k (int32 keys of one vector; -2^31 when fewer than k valid entries)"""
    zero = tl.sum(tl.zeros([1], dtype=tl.int32), axis=0).to(tl.int64)
    lo = zero - 2147483648
    hi = zero + 2147483647
    k64 = key.to(tl.int64)
    for _ in range(32):
        mid = lo + (hi - lo + 1) // 2
        cnt = tl.sum(((k64 >= mid) & valid).to(tl.int32), axis=0)
        ge = cnt >= k
        lo = tl.where(ge, mid, lo)
        hi = tl.where(ge, hi, mid - 1)
    return lo


@triton.jit
def _ix_topk_kernel(SC, CK, CI, GIDX, NSEL, P, start, n_top, KP: tl.constexpr, TAIL: tl.constexpr, BLK: tl.constexpr, SM: tl.constexpr, CAP: tl.constexpr,
                    TARGET: tl.constexpr, NATT: tl.constexpr):
    """One row per program, cost linear in the visible groups nv (no power-of-two row vector, no 32 passes):
    1. sample SM groups (stride ceil(nv / SM)), threshold = the key giving ~TARGET expected hits; one counting pass; accept when n_top <= count <= CAP (NATT tries, the
       expected count is rescaled after a miss);
    2. accepted: one pass compacts the candidates (ascending group order) into the row scratch CK / CI; exact n_top-th key on that buffer (ties: lowest index first);
    3. not accepted (heavy ties / odd score law): exact 32-step bisection with tiled passes, then one compacting pass.
    nv <= CAP skips the sampling (every visible group is a candidate). Output = GIDX [T, n_top] ascending group ids + NSEL, same selection as _ix_select_kernel."""
    r = tl.program_id(0)
    pos = start + r
    if TAIL:
        nv = pos // KP
    else:
        nv = (pos + 1) // KP
    nv = tl.minimum(nv, P)
    row = SC + r.to(tl.int64) * P
    zero = tl.sum(tl.zeros([1], dtype=tl.int32), axis=0)
    thr = zero.to(tl.int64) - 2147483648
    ok = zero + 1
    if nv > CAP:
        ok = zero
        s = (nv + SM - 1) // SM
        sidx = tl.arange(0, SM) * s
        sm = sidx < nv
        skey = _fkey(tl.load(row + sidx, mask=sm, other=float("-inf")))
        m = tl.sum(sm.to(tl.int32), axis=0)
        kk = tl.maximum((TARGET * m + nv - 1) // nv, 1)
        for _a in range(NATT):
            if ok == 0:
                ta = _kth_key(skey, sm, kk)
                cnt = zero
                for t0 in range(0, nv, BLK):
                    g = t0 + tl.arange(0, BLK)
                    gm = g < nv
                    key = _fkey(tl.load(row + g, mask=gm, other=float("-inf")))
                    cnt += tl.sum(((key.to(tl.int64) >= ta) & gm).to(tl.int32), axis=0)
                good = (cnt >= n_top) & (cnt <= CAP)
                thr = tl.where(good, ta, thr)
                ok = good.to(tl.int32)
                # rescale the sample rank so the next try expects CAP / 2 hits (too many) or 2 * n_top hits (too few)
                want = tl.where(cnt > CAP, CAP // 2, 2 * n_top)
                kk = tl.minimum(tl.maximum(kk * want // tl.maximum(cnt, 1), 1), m)
    if ok != 0:
        crow = r.to(tl.int64) * CAP
        off = zero
        for t0 in range(0, nv, BLK):
            g = t0 + tl.arange(0, BLK)
            gm = g < nv
            key = _fkey(tl.load(row + g, mask=gm, other=float("-inf")))
            sel = gm & (key.to(tl.int64) >= thr)
            si = sel.to(tl.int32)
            pb = off + tl.cumsum(si, axis=0) - 1
            tl.store(CK + crow + pb, key, mask=sel)
            tl.store(CI + crow + pb, g, mask=sel)
            off += tl.sum(si, axis=0)
        tl.debug_barrier()
        c = tl.arange(0, CAP)
        cm = c < off
        bk = tl.load(CK + crow + c, mask=cm, other=0)
        bi = tl.load(CI + crow + c, mask=cm, other=0)
        lo = _kth_key(bk, cm, n_top)
        k64 = bk.to(tl.int64)
        gt = cm & (k64 > lo)
        eq = cm & (k64 == lo)
        need = n_top - tl.sum(gt.to(tl.int32), axis=0)
        sel = gt | (eq & (tl.cumsum(eq.to(tl.int32), axis=0) <= need))
        si = sel.to(tl.int32)
        tl.store(GIDX + r.to(tl.int64) * n_top + tl.cumsum(si, axis=0) - 1, bi, mask=sel)
        tl.store(NSEL + r, tl.sum(si, axis=0))
    else:
        lo = zero.to(tl.int64) - 2147483648
        hi = zero.to(tl.int64) + 2147483647
        for _b in range(32):
            mid = lo + (hi - lo + 1) // 2
            cnt = zero
            for t0 in range(0, nv, BLK):
                g = t0 + tl.arange(0, BLK)
                gm = g < nv
                key = _fkey(tl.load(row + g, mask=gm, other=float("-inf")))
                cnt += tl.sum(((key.to(tl.int64) >= mid) & gm).to(tl.int32), axis=0)
            ge = cnt >= n_top
            lo = tl.where(ge, mid, lo)
            hi = tl.where(ge, hi, mid - 1)
        ngt = zero
        for t0 in range(0, nv, BLK):
            g = t0 + tl.arange(0, BLK)
            gm = g < nv
            key = _fkey(tl.load(row + g, mask=gm, other=float("-inf")))
            ngt += tl.sum(((key.to(tl.int64) > lo) & gm).to(tl.int32), axis=0)
        need = n_top - ngt
        ce = zero
        off = zero
        for t0 in range(0, nv, BLK):
            g = t0 + tl.arange(0, BLK)
            gm = g < nv
            key = _fkey(tl.load(row + g, mask=gm, other=float("-inf")))
            gt = gm & (key.to(tl.int64) > lo)
            eq = gm & (key.to(tl.int64) == lo)
            sel = gt | (eq & ((ce + tl.cumsum(eq.to(tl.int32), axis=0)) <= need))
            si = sel.to(tl.int32)
            tl.store(GIDX + r.to(tl.int64) * n_top + off + tl.cumsum(si, axis=0) - 1, g, mask=sel)
            ce += tl.sum(eq.to(tl.int32), axis=0)
            off += tl.sum(si, axis=0)
        tl.store(NSEL + r, off)


IX_BLOCKED = os.environ.get("HY4_PF_IXBLK", "1") == "1"      # HY4_PF_SPARSE=1 only: 0 = the unblocked score matrix + ix_select
IX_BLOCK_ELEMS = 1 << 27      # fp32 score elements of one query block (512 MB): the [T, P] matrix is never materialised, so the cost per layer stays linear in N


def ix_select_blocked(iq16, pk16, wt, start, kpool, tail, n_top, N, rows=None, cap=4096, blk=2048, sm=4096, target=2048, natt=3):
    """score + exact top-k per query block -> PfSparse (same selection as ix_scores_fused + ix_select(sparse=True)); iq16 [T, H, d] fp16, pk16 [P, d] fp16, wt [T, H] fp32"""
    T = iq16.shape[0]
    P = pk16.shape[0]
    TB = rows or max(64, min(T, (IX_BLOCK_ELEMS // max(P, 1)) // 64 * 64))
    dev = iq16.device
    gidx = torch.empty(T, n_top, dtype=torch.int32, device=dev)
    nsel = torch.empty(T, dtype=torch.int32, device=dev)
    ck = torch.empty(min(TB, T), cap, dtype=torch.int32, device=dev)
    ci = torch.empty(min(TB, T), cap, dtype=torch.int32, device=dev)
    for t0 in range(0, T, TB):
        t1 = min(T, t0 + TB)
        sc = ix_scores_fused(iq16[t0:t1], pk16, wt[t0:t1], start + t0, kpool, tail)
        _ix_topk_kernel[(t1 - t0,)](sc, ck, ci, gidx[t0:t1], nsel[t0:t1], P, start + t0, n_top, KP=kpool, TAIL=int(bool(tail)), BLK=blk, SM=sm, CAP=cap, TARGET=target,
                                    NATT=natt, num_warps=8)
        del sc
    return PfSparse(gidx, nsel, kpool, tail, start, N)


MOE_NF = None             # tests: force the grid order (None = env HY4_MOE_NFIRST)
LAST_PATH = ""           # path the last moe_prefill call ran (triton / hip_wmma / gm_pfmoe_shared); gmmq is not called from this module
MOE_HIP = None            # tests: force (None = env HY4_MOE_HIP=1); HIP / WMMA expert kernels (moe_hip.cu, 128-row blocks, Q4_K / Q3_K); off = Triton
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


def moe_prefill(z, sel, w, bufs, fmts, lim, nff, nout, rowb, ebytes, E, cfg=None, tm=None):
    """z [T, K] activations (fp32 / fp16), sel [T, topk] int, w [T, topk] fp32 routing weights.
    bufs = (gate, up, down) raw uint8 expert buffers (expert-major, row-major, stored blocks), fmts = (fmt gate/up, fmt down) as FMT ids,
    rowb = (row bytes gate/up, row bytes down), ebytes = (bytes per expert gate/up, down), nff = rows of gate / up (= K of down), nout = rows of down.
    Returns sum_k w[t, k] * down_e(silu(min(gate_e z, lim)) * clamp(up_e z, +-lim)) as fp32 [T, nout]."""
    T, S = sel.shape
    c = cfg or MOE_CFG or moe_cfg(T, S, E)
    BM, BN = c["BM"], c["BN"]
    P = T * S
    K = z.shape[1]
    t = tm.t if tm is not None and tm.on else (lambda n: contextlib.nullcontext())
    nf = c.get("NF", MOE_NF if MOE_NF is not None else os.environ.get("HY4_MOE_NFIRST", "1") == "1")
    with t("f moe: route + sort"):
        sp, be, maxb = route_blocks(sel.to(torch.int64), E, BM)
        x = to_f16(z).contiguous()
    use_hip = BM == 128 and nf and (MOE_HIP if MOE_HIP is not None else os.environ.get("HY4_MOE_HIP", "0") in ("1", "2")) and K % 64 == 0 and nff % 128 == 0 and nout % 128 == 0 and fmts[0] in (FMT_Q4K, FMT_Q3K) and fmts[1] == FMT_Q4K
    global LAST_PATH
    LAST_PATH = "triton"
    if use_hip:
        from . import kq_moe_hip as KH
        sh = os.environ.get("HY4_MOE_HIP", "0") == "2"
        LAST_PATH = "gm_pfmoe_shared" if sh else "hip_wmma"       # 2 = the shared gguf-moe gm_ext kernel (same source family), 1 = the local code object
        with t("f moe: gate+up kernel"):
            H = (KH.gate_up_shared if sh else KH.gate_up)(x, bufs[0], bufs[1], sp, be, P, S, E, ebytes[0], float(lim), K, nff, rowb[0], 0 if fmts[0] == FMT_Q4K else 1, maxb)
        with t("f moe: down kernel"):
            out = (KH.down_shared if sh else KH.down)(H, bufs[2], w.reshape(-1).float().contiguous(), sp, be, P, E, ebytes[1], nff, nout, rowb[1], maxb)
        with t("f moe: sum over topk"):
            return out.view(T, S, nout).sum(1)
    H = torch.empty(maxb * BM, nff, dtype=torch.float16, device=z.device)
    with t("f moe: gate+up kernel"):
        _gu_kernel[((nff // BN, maxb) if nf else (maxb, nff // BN))](x, bufs[0], bufs[1], H, sp, be, P, S, E, ebytes[0], float(lim), K=K, N=nff, ROWB=rowb[0], FMT=fmts[0], BM=BM, BN=BN,
                                                                      NF=nf, num_warps=c["warps"], num_stages=c["stages"])
    out = torch.empty(P, nout, dtype=torch.float32, device=z.device)
    with t("f moe: down kernel"):
        _down_kernel[((nout // BN, maxb) if nf else (maxb, nout // BN))](H, bufs[2], out, w.reshape(-1).float().contiguous(), sp, be, P, E, ebytes[1], K=nff, N=nout, ROWB=rowb[1], FMT=fmts[1],
                                                                          BM=BM, BN=BN, NF=nf, num_warps=c["warps"], num_stages=c["stages"])
    with t("f moe: sum over topk"):
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


def moe_prefill_i8(z, sel, w, bufs, fmts, lim, nff, nout, rowb, ebytes, E, cfg=None, tm=None):
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
    return f32_sat(torch.mm(a, b))


def dequant_q8(packed):
    """stored Q8_0 [N, K/32, 34] uint8 -> fp16 W^T [K, N] (contiguous)"""
    N, nb, bb = packed.shape
    K = nb * 32
    out = torch.empty(K, N, dtype=torch.float16, device=packed.device)
    spp = 8 if nb % 8 == 0 else 1
    _deq_q8_kernel[(triton.cdiv(N, 64), nb // spp)](packed, out, N, K=K, ROWB=nb * bb, BN=64, SPP=spp, num_warps=4)
    return out


def mm_rows(x2, w):
    """torch.mm(x2 [M, K] fp16, w [K, N] fp16) in row pieces of HY4_PF_MM_SPLIT (default 8192, 0 = off): the TunableOp results only cover M = 4096 / 8192, so a merged
    tail chunk (8192 + up to 2048 rows) fell back to the default GEMM solution (35 TF instead of 42, wo K = 16384 24 instead of 34): the first 8192 rows take the tuned entry"""
    M = x2.shape[0]
    S = int(os.environ.get("HY4_PF_MM_SPLIT", "8192"))
    if S <= 0:
        return torch.mm(x2, w)
    fine = os.environ.get("HY4_PF_MM_FINE", "0") == "1" and S >= 8192       # 1: after the 8192-row pieces, 4096-row pieces too (the other tuned M): chunks of 4096 * k rows and of 4096..8192 rows hit tuned entries
    if M <= S and not (fine and M > 4096):
        return torch.mm(x2, w)
    out = torch.empty(M, w.shape[1], dtype=x2.dtype, device=x2.device)
    a = 0
    while a < M:
        r = M - a
        n = S if r > S else (4096 if (fine and r > 4096) else r)
        torch.mm(x2[a:a + n], w, out=out[a:a + n])
        a += n
    return out


def qdense_tm(x, packed, tm=None):
    """x [.., K] -> [.., N] fp32 = x @ W.T: dequantise the stored Q8_0 weight to fp16 once per call, then the vendor fp16 GEMM (tm = optional stage timers)"""
    N, nb, bb = packed.shape
    K = nb * 32
    lead = x.shape[:-1]
    t = tm.t if tm is not None and tm.on else (lambda n: contextlib.nullcontext())
    with t(f"f dense N={N} K={K}: to_f16"):
        x2 = to_f16(x.reshape(-1, K)).contiguous()
    with t(f"f dense N={N} K={K}: dequant"):
        w16 = dequant_q8(packed)
    with t(f"f dense N={N} K={K}: mm"):
        o = mm_rows(x2, w16)
    with t(f"f dense N={N} K={K}: float"):
        out = f32_sat(o)
    return out.reshape(*lead, N)


@triton.jit
def _gate_kernel(KV, G, OUT, T, HV, VD: tl.constexpr, BC: tl.constexpr):
    """OUT[t, c] = KV[h, t, d] * sigmoid(G[t, c]) with c = h * VD + d: the Hy4 output gate on fp16 operands (fp32 maths, nan_to_num on the gate logits,
    saturating fp16 result); KV is the batch-over-heads GEMM output [H, T, VD], G / OUT are [T, H * VD]"""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    cm = c < HV
    h = c // VD
    d = c % VD
    g = tl.load(G + t.to(tl.int64) * HV + c, mask=cm, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, tl.minimum(tl.maximum(g, -65504.0), 65504.0))
    kv = tl.load(KV + (h.to(tl.int64) * T + t) * VD + d, mask=cm, other=0.0).to(tl.float32)
    o = kv * (1.0 / (1.0 + tl.exp(-g)))
    o = tl.minimum(tl.maximum(o, -65504.0), 65504.0)
    tl.store(OUT + t.to(tl.int64) * HV + c, o.to(tl.float16), mask=cm)


def gate16(kv, g):
    """kv [H, T, VD] fp16 (bmm output), g [T, H * VD] fp16 (raw gate GEMM output) -> gated [T, H * VD] fp16"""
    H, T, VD = kv.shape
    out = torch.empty(T, H * VD, dtype=torch.float16, device=kv.device)
    _gate_kernel[(T, triton.cdiv(H * VD, 1024))](kv, g, out, T, H * VD, VD=VD, BC=1024, num_warps=4)
    return out



@triton.jit
def _gate_ew_kernel(KV, G, OUT, n, BLK: tl.constexpr):
    """OUT = KV * sigmoid(G) elementwise, the maths of _gate_kernel (gate logits nan -> 0 and clamped, saturating fp16 result); all three operands in the SAME layout"""
    i = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
    m = i < n
    g = tl.load(G + i, mask=m, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, tl.minimum(tl.maximum(g, -65504.0), 65504.0))
    kv = tl.load(KV + i, mask=m, other=0.0).to(tl.float32)
    o = kv * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(OUT + i, tl.minimum(tl.maximum(o, -65504.0), 65504.0).to(tl.float16), mask=m)


def gate16t(kvt, gt):
    """transposed-layout gate: kvt [H * VD, T] (wv_b GEMM output with the head rows stacked), gt [H * VD, T] (raw gate GEMM output, transposed) -> gated [H * VD, T] fp16"""
    out = torch.empty_like(kvt)
    n = out.numel()
    _gate_ew_kernel[(triton.cdiv(n, 4096),)](kvt, gt, out, n, BLK=4096, num_warps=4)
    return out


def qdense_raw16_t(x, packed, tm=None):
    """qdense_raw16 with a TRANSPOSED result: x [M, K] -> [N, M] fp16 = (x @ W.T).T, from the GEMM W.T-view @ x.T-view (both operands read in their stored layout)"""
    N, nb, bb = packed.shape
    t = tm.t if tm is not None and tm.on else (lambda n: contextlib.nullcontext())
    with t(f"f dense N={N} K={nb * 32}: to_f16"):
        x2 = to_f16(x.reshape(-1, nb * 32)).contiguous()
    with t(f"f dense N={N} K={nb * 32}: dequant"):
        w16 = dequant_q8(packed)
    with t(f"f dense N={N} K={nb * 32}: mm"):
        return torch.mm(w16.t(), x2.t())


def qdense_t(xt, packed, tm=None):
    """qdense_tm for an activation stored TRANSPOSED: xt [K, M] fp16 contiguous -> [M, N] fp32 = xt.T @ W.T (the GEMM reads A column-major)"""
    N, nb, bb = packed.shape
    K = nb * 32
    t = tm.t if tm is not None and tm.on else (lambda n: contextlib.nullcontext())
    with t(f"f dense N={N} K={K}: dequant"):
        w16 = dequant_q8(packed)
    with t(f"f dense N={N} K={K}: mm"):
        o = torch.mm(xt.t(), w16)
    with t(f"f dense N={N} K={K}: float"):
        return torch.nan_to_num(o.float(), nan=0.0, posinf=F16MAX, neginf=-F16MAX)


def qdense_raw16(x, packed, tm=None, tag=""):
    """qdense_tm without the fp32 upcast: x [M, K] (fp16 passes through, else saturating cast) -> raw fp16 GEMM output [M, N]"""
    N, nb, bb = packed.shape
    t = tm.t if tm is not None and tm.on else (lambda n: contextlib.nullcontext())
    with t(f"f dense N={N} K={nb * 32}: to_f16"):
        x2 = to_f16(x.reshape(-1, nb * 32)).contiguous()
    with t(f"f dense N={N} K={nb * 32}: dequant"):
        w16 = dequant_q8(packed)
    with t(f"f dense N={N} K={nb * 32}: mm"):
        return mm_rows(x2, w16)


@triton.jit
def _qprep_kernel(Q, COS, SIN, XH, QC, T, H, HD: tl.constexpr, NOPE: tl.constexpr, NOPEP: tl.constexpr, P: tl.constexpr, D: tl.constexpr, R: tl.constexpr):
    """one program = one (t, h) row of the wq_b output [T, H * HD] (fp16 raw GEMM output): q_nope -> XH [H, T, NOPE] fp16 (nan -> 0, saturating),
    q_pe -> rotary (consecutive pairs, cos / sin [T, P / 2] fp32) -> QC[h, t, R:R + P] fp16 (same rounding points as rope_norm + clamp + half)"""
    t = tl.program_id(0)
    h = tl.program_id(1)
    base = (t.to(tl.int64) * H + h) * HD
    dst = (h.to(tl.int64) * T + t)
    c = tl.arange(0, NOPEP)
    x = tl.load(Q + base + c, mask=c < NOPE, other=0.0).to(tl.float32)
    x = tl.where(x != x, 0.0, tl.minimum(tl.maximum(x, -65504.0), 65504.0))
    tl.store(XH + dst * NOPE + c, x.to(tl.float16), mask=c < NOPE)
    i = tl.arange(0, P // 2)
    x0 = tl.load(Q + base + NOPE + 2 * i).to(tl.float32)
    x1 = tl.load(Q + base + NOPE + 2 * i + 1).to(tl.float32)
    x0 = tl.where(x0 != x0, 0.0, tl.minimum(tl.maximum(x0, -65504.0), 65504.0))
    x1 = tl.where(x1 != x1, 0.0, tl.minimum(tl.maximum(x1, -65504.0), 65504.0))
    cs = tl.load(COS + t * (P // 2) + i)
    sn = tl.load(SIN + t * (P // 2) + i)
    o0 = tl.minimum(tl.maximum(x0 * cs - x1 * sn, -65504.0), 65504.0)
    o1 = tl.minimum(tl.maximum(x0 * sn + x1 * cs, -65504.0), 65504.0)
    tl.store(QC + dst * D + R + 2 * i, o0.to(tl.float16))
    tl.store(QC + dst * D + R + 2 * i + 1, o1.to(tl.float16))


def qprep(q, cos, sin, H, hd, nope, P, R):
    """q [T, H * hd] fp16 (raw wq_b GEMM output) -> (xh [H, T, nope] fp16, qc [H, T, R + P] fp16 with the rotary q_pe in columns R:). The caller's batched GEMM
    fills qc[..., :R] (xh @ wk_b^T)."""
    T = q.shape[0]
    xh = torch.empty(H, T, nope, dtype=torch.float16, device=q.device)
    qc = torch.empty(H, T, R + P, dtype=torch.float16, device=q.device)
    _qprep_kernel[(T, H)](q, cos, sin, xh, qc, T, H, HD=hd, NOPE=nope, NOPEP=triton.next_power_of_2(nope), P=P, D=R + P, R=R, num_warps=2)
    return xh, qc
