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


# ------------------------------------------------------------------------------------------------ host side
F16MAX = 65504.0


def to_f16(x):
    """saturating fp32 / fp16 -> fp16 (hidden streams keep the clamp)"""
    return x.float().clamp_(-F16MAX, F16MAX).half() if x.dtype != torch.float16 else x


MOE_CFG = {"BM": 64, "BN": 64, "warps": 4, "stages": 1}
DENSE_CFG = {"BM": 64, "BN": 64, "warps": 4, "stages": 1}


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
    c = cfg or MOE_CFG
    BM, BN = c["BM"], c["BN"]
    T, S = sel.shape
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
