"""EXL3_LEGACY_PREFILL_DQ: one Triton launch dequantizing GGUF Q4_0 / Q5_0 rows to fp16 (prefill attention projections, rows > 1).

Replaces gap_formats.dequant (about 15 torch ops per projection). Bit-identical to it: element j of a block is d * v in fp32 (one rounded multiply, fp16 d
widened), then rounded to fp16. Q4_0 block = [d fp16][16 nibble bytes] (v = q - 8); Q5_0 = [d fp16][qh u32][16 nibble bytes] (v = (q | bit << 4) - 16);
elements 0..15 are the low nibbles, 16..31 the high nibbles. Q5_1 already has its own Triton dequant in gap_formats and is not routed here.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:       # pragma: no cover
    HAVE_TRITON = False

BB = {"q4_0": 18, "q5_0": 22}
QOFF = {"q4_0": 2, "q5_0": 6}


def ref_dequant_np(fmt: str, raw: np.ndarray) -> np.ndarray:
    """Numpy reference: raw uint8 [rows, nb * BB] -> fp16 [rows, nb * 32] (float32 d * v, one rounding to fp16)."""
    n = raw.shape[0]
    b = raw.reshape(-1, BB[fmt])
    d = np.ascontiguousarray(b[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1, 1)
    qs = b[:, QOFF[fmt]:QOFF[fmt] + 16].astype(np.int32)
    q = np.concatenate([qs & 15, qs >> 4], axis=1)
    if fmt == "q4_0":
        v = (q - 8).astype(np.float32)
    else:
        qh = np.ascontiguousarray(b[:, 2:6]).view("<u4").reshape(-1, 1).astype(np.uint32)
        hb = ((qh >> np.arange(32, dtype=np.uint32)[None, :]) & 1).astype(np.int32)
        v = ((q | (hb << 4)) - 16).astype(np.float32)
    return (d * v).astype(np.float16).reshape(n, -1)


if HAVE_TRITON:
    @triton.jit
    def _dq_legacy(src, dst, nblk, HAS_QH: tl.constexpr, BBYTES: tl.constexpr, QO: tl.constexpr, BLOCKS: tl.constexpr):
        pid = tl.program_id(0)
        bi = pid * BLOCKS + tl.arange(0, BLOCKS)
        ok = bi < nblk
        j = tl.arange(0, 32)
        base = bi.to(tl.int64) * BBYTES
        b0 = tl.load(src + base, mask=ok, other=0).to(tl.uint32)
        b1 = tl.load(src + base + 1, mask=ok, other=0).to(tl.uint32)
        d = (b0 | (b1 << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        qs = tl.load(src + base[:, None] + QO + (j % 16)[None, :], mask=ok[:, None], other=0).to(tl.uint32)
        lo = tl.where((j < 16)[None, :], qs & 15, qs >> 4).to(tl.int32)
        if HAS_QH:
            b2 = tl.load(src + base + 2, mask=ok, other=0).to(tl.uint32)
            b3 = tl.load(src + base + 3, mask=ok, other=0).to(tl.uint32)
            b4 = tl.load(src + base + 4, mask=ok, other=0).to(tl.uint32)
            b5 = tl.load(src + base + 5, mask=ok, other=0).to(tl.uint32)
            qh = b2 | (b3 << 8) | (b4 << 16) | (b5 << 24)
            hb = ((qh[:, None] >> j[None, :].to(tl.uint32)) & 1).to(tl.int32)
            v = ((lo | (hb << 4)) - 16).to(tl.float32)
        else:
            v = (lo - 8).to(tl.float32)
        o = d[:, None] * v
        tl.store(dst + bi.to(tl.int64)[:, None] * 32 + j[None, :], o.to(tl.float16), mask=ok[:, None])


def dequant(fmt: str, raw: torch.Tensor) -> torch.Tensor:
    """raw uint8 [rows, nb * BB] (CUDA) -> fresh fp16 [rows, nb * 32]; ONE launch. The caller frees it after its matmul."""
    bb = BB[fmt]
    rows, w = raw.shape
    assert w % bb == 0
    nblk = rows * (w // bb)
    out = torch.empty((rows, (w // bb) * 32), dtype=torch.half, device=raw.device)
    B = 64
    _dq_legacy[(triton.cdiv(nblk, B),)](raw.contiguous(), out, nblk, HAS_QH=(fmt == "q5_0"), BBYTES=bb, QO=QOFF[fmt], BLOCKS=B,
                                         enable_fp_fusion=False)
    return out
