"""MXFP4 routed experts in the grouped MoE prefill (CUDA only), EXL3_MXFP4_PREFILL_GROUPED=1 (default off).

Per prefill chunk: the experts routed in this chunk (and only those) are dequantized to an fp16 scratch [n, N, K] with ONE Triton
launch per projection and expert group (the 17-byte GGUF block_mxfp4: [e8m0][16 bytes, elements j | j + 16 << 4], same decode as
triton_gap FMT 2), then one grouped fp16 GEMM per projection over the expert-sorted rows (bias in the epilogue, gate+up in one
launch), the layer's own activation kernel (swiglu_oai clamp), and a down GEMM that adds the bias and scales the rows by the
routing weight. The scratch is freed at the end of the call: nothing stays resident between chunks. When the routed experts need more
than EXL3_MXFP4_PREFILL_SCRATCH_MB (default 1536) of scratch, they are processed in groups.

Value contract (same as the per-expert path it replaces, ForeignLinear gap_dequant + fp16 matmul + fp16 bias add + fp16 weight mul):
weights = fp16(2^(e-128) * fp4x2 value), g/u = fp16(acc) + fp16 bias rounded to fp16, a = activation(g, u) (same kernel),
dn = fp16(fp16(acc) + bias) * routing weight rounded to fp16; only the fp32 summation order of the top-k picks and the GEMM accumulate
order differ.
Env: EXL3_MXFP4_PREFILL_GROUPED (0/1), EXL3_MXFP4_PREFILL_GROUPED_MIN (min rows; default FQ_GROUPED_MAX_ROWS + 1, the rows above the
decode grouped path; a lower value forces the new path for short prompts, used by the oracle cell only), EXL3_MXFP4_PREFILL_SCRATCH_MB,
EXL3_MXFP4_PREFILL_BM / _BN / _BK / _NW / _NS (GEMM tile).
"""
from __future__ import annotations

import os

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    from .triton_gap import _fp4half
    HAVE_TRITON = True
except Exception:       # CPU-only checks without triton
    HAVE_TRITON = False

FP4_X2 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)      # gap_formats._FP4_HALF (fp4 value x 2; the scale is 2^(e-128))
FIRES = 0                                                               # python-level count of run() calls (profiling / tests)


def flag_on() -> bool:
    """EXL3_MXFP4_PREFILL_GROUPED=1 on CUDA (not HIP). Read at call time."""
    return os.environ.get("EXL3_MXFP4_PREFILL_GROUPED", "0") == "1" and bool(torch.version.cuda) and not torch.version.hip


def use_prefill(rows: int, decode_max_rows: int) -> bool:
    """Dispatch: flag on, and rows >= EXL3_MXFP4_PREFILL_GROUPED_MIN (default decode_max_rows + 1) and >= 2."""
    if not flag_on():
        return False
    lo = int(os.environ.get("EXL3_MXFP4_PREFILL_GROUPED_MIN", str(decode_max_rows + 1)))
    return rows >= max(2, lo)


def budget_bytes() -> int:
    return int(float(os.environ.get("EXL3_MXFP4_PREFILL_SCRATCH_MB", "1536")) * (1 << 20))


def split_groups(n: int, per_expert_bytes: int, budget: int) -> list:
    """[(a, b)) ranges over n routed experts, each group's scratch <= budget (at least one expert per group), sizes balanced."""
    if n <= 0:
        return []
    cap = max(1, int(budget) // max(1, int(per_expert_bytes)))
    ng = -(-n // cap)
    base, extra = divmod(n, ng)
    out, a = [], 0
    for i in range(ng):
        b = a + base + (1 if i < extra else 0)
        out.append((a, b))
        a = b
    return out


def ref_dequant_np(raw: np.ndarray) -> np.ndarray:
    """Numpy reference of the kernel: raw uint8 [N, Kb * 17] -> fp16 [N, Kb * 32] (what _dq_mx writes for one expert)."""
    n = raw.shape[0]
    b = raw.reshape(n, -1, 17)
    d = np.exp2(b[:, :, 0].astype(np.float32) - 128.0)[:, :, None]
    qs = b[:, :, 1:17]
    v = np.asarray(FP4_X2, dtype=np.float32)[np.concatenate([qs & 15, qs >> 4], axis=2).astype(np.int64)]
    return (d * v).reshape(n, -1).astype(np.float16)


if HAVE_TRITON:
    @triton.jit
    def _dq_mx(wtab, eids, out, NB, BLK: tl.constexpr):
        pid = tl.program_id(0)
        s = tl.program_id(1)
        e = tl.load(eids + s).to(tl.int64)
        w = tl.load(wtab + e).to(tl.pointer_type(tl.uint8))
        b = pid.to(tl.int64) * BLK + tl.arange(0, BLK)          # block index inside the expert, row n = b // Kb
        m = b < NB
        j = tl.arange(0, 16)
        base = b * 17
        d = tl.exp2(tl.load(w + base, mask=m, other=0).to(tl.float32) - 128.0)
        qs = tl.load(w + base[:, None] + 1 + j[None, :], mask=m[:, None], other=0).to(tl.int32)
        lo = (_fp4half(qs & 15) * d[:, None]).to(tl.float16)
        hi = (_fp4half(qs >> 4) * d[:, None]).to(tl.float16)
        o = out + s.to(tl.int64) * NB * 32 + b[:, None] * 32 + j[None, :]
        tl.store(o, lo, mask=m[:, None])
        tl.store(o + 16, hi, mask=m[:, None])

    @triton.jit
    def _gmm_mx(a_ptr, w_ptr, w2_ptr, out_ptr, out2_ptr, btab, b2tab, slot_ptr, tile_e, tile_r0, offs_ptr, wrow_ptr,
                N, K, stride_am, stride_om,
                DUAL: tl.constexpr, HAS_B: tl.constexpr, SCALE_ROWS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_n = tl.program_id(0)
        t = tl.program_id(1)
        e = tl.load(tile_e + t)
        slot = tl.load(slot_ptr + e)                              # scratch slot of expert e in this group, -1 = not in it (or unused tile)
        if slot < 0:
            return
        r0 = tl.load(tile_r0 + t)
        r_end = tl.load(offs_ptr + e)
        offs_m = r0 + tl.arange(0, BM)
        mmask = offs_m < r_end
        offs_n = pid_n * BN + tl.arange(0, BN)
        nmask = offs_n < N
        kk = tl.arange(0, BK)
        wbase = slot.to(tl.int64) * N * K
        a_row = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am
        w_col = w_ptr + wbase + offs_n.to(tl.int64)[None, :] * K
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if DUAL:
            w2_col = w2_ptr + wbase + offs_n.to(tl.int64)[None, :] * K
            acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_row + (k + kk)[None, :], mask=mmask[:, None], other=0.0)
            w = tl.load(w_col + (k + kk)[:, None], mask=nmask[None, :], other=0.0)
            acc = tl.dot(a, w, acc)
            if DUAL:
                w2 = tl.load(w2_col + (k + kk)[:, None], mask=nmask[None, :], other=0.0)
                acc2 = tl.dot(a, w2, acc2)
        res = acc.to(tl.float16)
        if HAS_B:
            bp = tl.load(btab + e).to(tl.pointer_type(tl.float16))
            bv = tl.load(bp + offs_n, mask=nmask, other=0.0).to(tl.float32)
            res = (res.to(tl.float32) + bv[None, :]).to(tl.float16)
        if SCALE_ROWS:
            wr = tl.load(wrow_ptr + offs_m, mask=mmask, other=0.0).to(tl.float32)
            res = (res.to(tl.float32) * wr[:, None]).to(tl.float16)
        o = out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
        tl.store(o, res, mask=mmask[:, None] & nmask[None, :])
        if DUAL:
            res2 = acc2.to(tl.float16)
            if HAS_B:
                bp2 = tl.load(b2tab + e).to(tl.pointer_type(tl.float16))
                bv2 = tl.load(bp2 + offs_n, mask=nmask, other=0.0).to(tl.float32)
                res2 = (res2.to(tl.float32) + bv2[None, :]).to(tl.float16)
            o2 = out2_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
            tl.store(o2, res2, mask=mmask[:, None] & nmask[None, :])


def dequant_group(wtab: torch.Tensor, ids: torch.Tensor, out: torch.Tensor, N: int, K: int) -> None:
    """ONE launch: experts ids (int32 [n]) of the int64 pointer table wtab -> fp16 out[n, N, K]."""
    NB = N * (K // 32)
    BLK = 128
    _dq_mx[(triton.cdiv(NB, BLK), ids.numel())](wtab, ids, out, NB, BLK=BLK, num_warps=4)


def _gemm(a, w, w2, out, out2, btab, b2tab, slot_of, tiles, wrow, N, K, bm):
    tile_e, tile_r0, offs = tiles
    dual = w2 is not None
    bn = int(os.environ.get("EXL3_MXFP4_PREFILL_BN", "64" if dual else "128"))
    bk = int(os.environ.get("EXL3_MXFP4_PREFILL_BK", "32"))
    nw = int(os.environ.get("EXL3_MXFP4_PREFILL_NW", "8"))
    ns = int(os.environ.get("EXL3_MXFP4_PREFILL_NS", "3"))
    assert K % bk == 0, (K, bk)
    has_b = btab is not None
    z = offs
    _gmm_mx[(triton.cdiv(N, bn), tile_e.shape[0])](
        a, w, w2 if dual else w, out, out2 if dual else out, btab if has_b else z, b2tab if (has_b and dual) else z,
        slot_of, tile_e, tile_r0, offs, wrow if wrow is not None else offs, N, K, a.stride(0), out.stride(0),
        DUAL=dual, HAS_B=has_b, SCALE_ROWS=wrow is not None, BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns)


def run(h, sel, wts, tbl, E, I, K, N, act_fn, act_limit, budget=None):
    """h half [R, K] contiguous (zero-padded to the experts' K); sel long [R, top_k]; wts half [R, top_k];
    tbl = the decode path's pointer tables (_, dev, gw, gb, uw, ub, dw, db). Returns fp32 [R, N] (routing-weighted sum of the experts)."""
    global FIRES
    from ..modules.fq_moe_prefill import build_tiles
    FIRES += 1
    R, top_k = sel.shape
    dev = h.device
    M = R * top_k
    _, _, gw, gb, uw, ub, dw, db = tbl
    bm = int(os.environ.get("EXL3_MXFP4_PREFILL_BM", "128"))
    flat_e = sel.reshape(-1)
    order = flat_e.argsort(stable=True)
    token_sorted = order // top_k
    wrow = wts.reshape(-1)[order].contiguous()
    counts = torch.bincount(flat_e, minlength=E)
    routed = torch.nonzero(counts).flatten().to(torch.int32)          # one host sync per layer (n routed experts)
    n = routed.numel()
    xs = h.index_select(0, token_sorted)
    tiles = build_tiles(counts, M, bm)
    groups = split_groups(n, max(2 * I * K, N * I) * 2, budget if budget is not None else budget_bytes())
    slot_of = torch.empty(E + 1, dtype=torch.int32, device=dev)
    ar = torch.arange(n, dtype=torch.int32, device=dev)

    def set_slots(a, b):
        slot_of.fill_(-1)
        slot_of[routed[a:b].long()] = ar[:b - a]
        return routed[a:b]

    g = torch.empty((M, I), dtype=torch.half, device=dev)
    u = torch.empty((M, I), dtype=torch.half, device=dev)
    for a, b in groups:
        ids = set_slots(a, b)
        wg = torch.empty((b - a, I, K), dtype=torch.half, device=dev)
        wu = torch.empty((b - a, I, K), dtype=torch.half, device=dev)
        dequant_group(gw, ids, wg, I, K)
        dequant_group(uw, ids, wu, I, K)
        _gemm(xs, wg, wu, g, u, gb, ub, slot_of, tiles, None, I, K, bm)
        del wg, wu
    act = torch.empty_like(g)
    act_fn(g, u, act, act_limit)
    del g, u, xs
    dn = torch.empty((M, N), dtype=torch.half, device=dev)
    for a, b in groups:
        ids = set_slots(a, b)
        wd = torch.empty((b - a, N, I), dtype=torch.half, device=dev)
        dequant_group(dw, ids, wd, N, I)
        _gemm(act, wd, None, dn, None, db, None, slot_of, tiles, wrow, N, I, bm)
        del wd
    del act
    inv = torch.empty_like(order)
    inv[order] = torch.arange(M, device=dev)
    return dn.index_select(0, inv).view(R, top_k, N).float().sum(1)
