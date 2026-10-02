"""
CPU (pure-torch) reference decoder for EXL3 trellis tensors -- integer and half-integer bitrate K.

This is a line-by-line translation of the decode side of exllamav3, taken from upstream
turboderp-org/exllamav3 commit 07b8a2e "Fractional trellis support" plus the integer path that
the AMD fork already ships:

  * quant/exl3_dq.cuh        -- window extraction (bit indices) + the 3-instruction codebook
  * quant/codebook.cuh       -- the three codebooks, reimplemented in integer/half arithmetic
  * quant/frac.cu            -- fractional ring storage: D(i), S(i), 16-bit window at [S(i)-16, S(i))
  * quant/reconstruct.cu     -- ring position -> row-major tile element, suh/svh, fused H128
  * modules/quant/exl3_lib/quantize.py::tensor_core_perm -- the permutation itself

Storage conventions (all verified against the sources above):

  * A tile is 256 weights. The bit stream of a tile is `bpb` bits, MSB-first inside each 32-bit
    word, words in memory order (little-endian u16 pairs make the u32 word).
  * Integer K: D(i) = K for all i, bpb = 16K.
    Half-integer K = KA + 0.5: D(i) = KA + ((MASK >> (i & 15)) & 1) with MASK = 0xAAAA, i.e. odd
    ring positions carry the extra bit; bpb = 16KA + 8. (Derived from dq8_half: w7 -> w6 shifts by
    KA+1, w6 -> w5 by KA, w5 -> w4 by KA+1, i.e. positions 7,5,3 carry the extra bit.)
  * Ring position p has its 16-bit window at ring bits [S(p) - 16, S(p)) mod L, where
    S(p) = (p >> 4) * bpb + sum_{j=0}^{p & 15} D(j) and L = 16 * bpb is the tile length. For integer
    K this reduces to the kernel's S(p) = (p+1)*K (dq8 uses b1 = (t_offset + 257) * bits).
  * Ring position p is the row-major tile element tensor_core_perm[p].
  * Decoded tile (kt, nt) lands at rows [16kt, 16kt+16) and columns [16nt, 16nt+16) of W, which is
    stored (k, n) = (in_features, out_features) -- y = x @ W.
  * The hadamard variant emits W = diag(suh) . H128 . W_hat . H128 . diag(svh), H128 the
    natural-order Sylvester Hadamard normalised by 1/sqrt(128), one 128x128 block at a time.

Everything is computed in fp32/fp64 integer maths and rounded once to fp16 where the kernels round,
so the output is bit-identical to the kernels' fp16 arithmetic except for ties inside a fused
multiply-add (the kernel uses __hfma; this reference rounds the fp64 product-sum once).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional

import torch

# ---------------------------------------------------------------------------------------------
# constants taken from the kernels

MASK_HALF = 0xAAAA  # decode-side fractional mask (odd ring positions carry the extra bit)

CB_PLAIN, CB_MCG, CB_MUL1 = 0, 1, 2
CB_NAME = {CB_PLAIN: "0 (plain)", CB_MCG: "1 (mcg)", CB_MUL1: "2 (mul1)"}

_MUL_CB0 = 89226354
_ADD_CB0 = 64248484
_MUL_CB1 = 0xCBAC1FED
_MUL_CB2 = 0x83DCD12D
_LOP3_AND = 0x8FFF8FFF
_LOP3_XOR = 0x3B603B60
_DP4A_ACC = 0x6400

_M32 = 0xFFFFFFFF


def _f16_from_bits(u: torch.Tensor) -> torch.Tensor:
    """Reinterpret an int64 tensor of 16-bit patterns as fp16 (bit-exact, no conversion)."""
    return u.to(torch.int32).to(torch.uint16).view(torch.float16)


# fp16 constants the mul1 codebook uses (k_inv = 1/147.7, k_bias = (-1024 - 510) * k_inv)
K_INV = float(_f16_from_bits(torch.tensor([0x1EEE], dtype=torch.int64))[0])
K_BIAS = float(_f16_from_bits(torch.tensor([0xC931], dtype=torch.int64))[0])


# ---------------------------------------------------------------------------------------------
# layout: ring positions, bit budget, windows


def group_bits(K: float, mask: int | None = None) -> torch.Tensor:
    """D(0..15): bits consumed by each of the 16 ring positions of one ring period."""
    ka = int(math.floor(K + 1e-6))
    frac = round(K - ka, 1)
    if mask is None:
        mask = 0 if frac == 0.0 else MASK_HALF
    if frac == 0.0:
        if mask:
            raise ValueError("integer K takes no fractional mask")
        return torch.full((16,), ka, dtype=torch.long)
    if frac != 0.5:
        raise ValueError(f"unsupported fractional part {frac} (only .5 is implemented upstream)")
    return torch.tensor([ka + ((mask >> j) & 1) for j in range(16)], dtype=torch.long)


def tile_layout(K: float, mask: int | None = None):
    """-> (starts, L, bpb): window start bit per ring position p (0..255), tile bit length, bits/tile."""
    d16 = group_bits(K, mask)
    bpb = int(d16.sum())
    if bpb % 2:
        raise ValueError(f"bpb = {bpb} is odd: fractional tiles must be whole 32-bit words")
    L = 16 * bpb
    cum = torch.cumsum(d16, 0)
    S = (torch.arange(256, dtype=torch.long) // 16) * bpb + cum.repeat(16)
    return (S - 16) % L, L, bpb


def tensor_core_perm() -> torch.Tensor:
    """Row-major element -> ring position (inverse of quantize.py::tensor_core_perm)."""
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        perm[t * 8 + 0] = r0 * 16 + c0
        perm[t * 8 + 1] = (r0 + 1) * 16 + c0
        perm[t * 8 + 2] = (r0 + 8) * 16 + c0
        perm[t * 8 + 3] = (r0 + 9) * 16 + c0
        perm[t * 8 + 4] = r0 * 16 + c0 + 8
        perm[t * 8 + 5] = (r0 + 1) * 16 + c0 + 8
        perm[t * 8 + 6] = (r0 + 8) * 16 + c0 + 8
        perm[t * 8 + 7] = (r0 + 9) * 16 + c0 + 8
    # quantize.py's permutation is ring -> row-major. Decoding needs the inverse gather:
    # for each row-major output element, select its source ring position.
    return torch.argsort(torch.tensor(perm, dtype=torch.long))


PERM = tensor_core_perm()


# ---------------------------------------------------------------------------------------------
# bits -> windows -> codebook


def _u32_words(tiles_u16: torch.Tensor) -> torch.Tensor:
    """(nt, bpb) uint16 row -> (nt, bpb/4) int32, the little-endian word the kernels load as uint32."""
    x = tiles_u16.to(torch.int32)
    return x[:, 0::2] + (x[:, 1::2] << 16)


def bits_of_row(tiles_u16: torch.Tensor, L: int) -> torch.Tensor:
    """(nt, bpb) uint16 -> (nt, L) uint8, MSB-first inside each 32-bit word."""
    w = _u32_words(tiles_u16)
    j = torch.arange(L, dtype=torch.long)
    word = w[:, j >> 5]
    return ((word >> (31 - (j & 31)).to(torch.int32)) & 1).to(torch.uint8)


def windows_of_row(tiles_u16: torch.Tensor, K: float, mask: int | None = None) -> torch.Tensor:
    """(nt, bpb) uint16 -> (nt, 256) int64 of the raw 16-bit window per ring position."""
    starts, L, bpb = tile_layout(K, mask)
    if tiles_u16.shape[-1] != bpb:
        raise ValueError(f"packed width {tiles_u16.shape[-1]} != bpb {bpb} for K = {K}")
    bit = bits_of_row(tiles_u16, L)
    padded = torch.cat([bit, bit[:, :16]], dim=1)  # +16 bits so the wrap-around window is a slice
    idx = starts[:, None] + torch.arange(16, dtype=torch.long)
    wb = padded[:, idx].to(torch.int64)
    weights = (1 << torch.arange(15, -1, -1, dtype=torch.long))
    return (wb * weights).sum(-1)


def decode_windows(w: torch.Tensor, cb: int = CB_MUL1) -> torch.Tensor:
    """16-bit windows -> fp16 codebook values, arithmetic per codebook.cuh::decode_3inst_2."""
    x = w.to(torch.int64) & 0xFFFF
    if cb == CB_PLAIN or cb == CB_MCG:
        mul = _MUL_CB0 if cb == CB_PLAIN else _MUL_CB1
        x = (x * mul) & _M32
        if cb == CB_PLAIN:
            x = (x + _ADD_CB0) & _M32
        x = _LOP3_XOR ^ (x & _LOP3_AND)
        lo = _f16_from_bits(x & 0xFFFF).to(torch.float32)
        hi = _f16_from_bits(x >> 16).to(torch.float32)
        return (lo + hi).to(torch.float16)
    if cb == CB_MUL1:
        x = (x * _MUL_CB2) & _M32
        s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF) + _DP4A_ACC
        h = _f16_from_bits(s & 0xFFFF).to(torch.float32)
        return (h * K_INV + K_BIAS).to(torch.float16)
    raise ValueError(f"unknown codebook {cb}")


# ---------------------------------------------------------------------------------------------
# tile -> matrix


def decode_tile_row(tiles_u16: torch.Tensor, K: float, cb: int, mask: int | None = None) -> torch.Tensor:
    """(nt, bpb) uint16 of one tile row -> (16, 16*nt) fp16, row-major rows/cols of W for that row."""
    nt = tiles_u16.shape[0]
    v = decode_windows(windows_of_row(tiles_u16, K, mask), cb)  # (nt, 256) ring order
    rm = v[:, PERM]                                             # (nt, 256) row-major in tile
    blk = rm.view(nt, 16, 16).permute(1, 0, 2).reshape(16, nt * 16)
    return blk


def decode_block(trellis: torch.Tensor, K: float, cb: int, mask: int | None = None,
                 kt0: int = 0, kt1: Optional[int] = None, nt0: int = 0, nt1: Optional[int] = None,
                 progress: bool = False) -> torch.Tensor:
    """Decode tiles [kt0:kt1] x [nt0:nt1] -> fp16 (16*(kt1-kt0), 16*(nt1-nt0)), no scales/hadamard."""
    kt1 = trellis.shape[0] if kt1 is None else kt1
    nt1 = trellis.shape[1] if nt1 is None else nt1
    t = trellis.contiguous().view(torch.uint16)
    out = torch.empty((16 * (kt1 - kt0), 16 * (nt1 - nt0)), dtype=torch.float16)
    for kt in range(kt0, kt1):
        out[16 * (kt - kt0):16 * (kt - kt0) + 16] = decode_tile_row(t[kt, nt0:nt1], K, cb, mask)
        if progress and (kt - kt0) % 64 == 63:
            print(f"    kt {kt - kt0 + 1}/{kt1 - kt0}", flush=True)
    return out


def hadamard128() -> torch.Tensor:
    """Natural-order Sylvester Hadamard, normalised by 1/sqrt(128) (the kernels' r_scale per side)."""
    h = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(7):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(128)).to(torch.float32)


def reconstruct(trellis: torch.Tensor, K: float, cb: int = CB_MUL1,
                suh: Optional[torch.Tensor] = None, svh: Optional[torch.Tensor] = None,
                had: bool = False, mask: int | None = None,
                kt0: int = 0, kt1: Optional[int] = None, nt0: int = 0, nt1: Optional[int] = None,
                progress: bool = False, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    EXL3 tensor -> weight matrix (k, n) = (in_features, out_features), y = x @ W.

    had=False reproduces exllamav3_ext.reconstruct / reconstruct_slice (raw decoded values, the
    caller applies the Hadamards to x and y). had=True reproduces reconstruct_had_slice, i.e.
    W = diag(suh) . H128 . W_hat . H128 . diag(svh) in the original basis.
    """
    kt1 = trellis.shape[0] if kt1 is None else kt1
    nt1 = trellis.shape[1] if nt1 is None else nt1
    k, n = 16 * (kt1 - kt0), 16 * (nt1 - nt0)

    raw = decode_block(trellis, K, cb, mask, kt0, kt1, nt0, nt1, progress)
    if not had:
        return raw.to(out_dtype)

    if suh is None or svh is None:
        raise ValueError("had=True needs suh and svh")
    if k % 128 or n % 128:
        raise ValueError("had recontruction needs both dims multiples of 128 (as the kernels do)")

    H = hadamard128()
    svf = svh.to(torch.float32)                                        # already offset by the caller
    if svf.numel() < n:
        raise ValueError(f"svh too short for n = {n} (slice: pass svh already offset)")
    out = torch.empty((k, n), dtype=torch.float16)
    for kb in range(k // 128):
        rows = raw[128 * kb:128 * kb + 128].to(torch.float32)          # (128, n)
        # right transform, per 128-column block (the kernel does both sides in one pass)
        for nb in range(n // 128):
            sl = slice(128 * nb, 128 * nb + 128)
            rows[:, sl] = rows[:, sl] @ H
        rows = H @ rows
        s = suh[128 * kb:128 * kb + 128].to(torch.float32)[:, None]
        out[128 * kb:128 * kb + 128] = (rows * s * svf[:n][None, :]).to(torch.float16)
    return out.to(out_dtype)


# ---------------------------------------------------------------------------------------------
# checkpoint access


def codebook_of(model_dir: str, key: str) -> int:
    """0/1/2 from the {plain, mcg, mul1} marker tensors that sit next to the trellis."""
    keys = _index(model_dir)
    if f"{key}.mul1" in keys:
        return CB_MUL1
    if f"{key}.mcg" in keys:
        return CB_MCG
    return CB_PLAIN


_INDEX_CACHE: dict[str, dict] = {}


def _index(model_dir: str) -> dict:
    if model_dir not in _INDEX_CACHE:
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            _INDEX_CACHE[model_dir] = json.load(f)["weight_map"]
    return _INDEX_CACHE[model_dir]


def load_tensor(model_dir: str, key: str, device: str = "cpu") -> torch.Tensor:
    """Lazy single-tensor read (safe_open): only this tensor is materialised."""
    from safetensors import safe_open
    shard = _index(model_dir)[key]
    with safe_open(os.path.join(model_dir, shard), framework="pt", device=device) as f:
        return f.get_tensor(key)


def load_layer(model_dir: str, key: str, device: str = "cpu"):
    """trellis + suh/svh (+ codebook) for one quantised tensor group."""
    return (load_tensor(model_dir, f"{key}.trellis", device),
            load_tensor(model_dir, f"{key}.suh", device),
            load_tensor(model_dir, f"{key}.svh", device),
            codebook_of(model_dir, key))


def K_of(trellis: torch.Tensor) -> float:
    return round(trellis.shape[-1] / 16, 4)


# ---------------------------------------------------------------------------------------------
# CLI: stats for one tensor without loading the model


def stats(w: torch.Tensor) -> dict:
    wf = w.to(torch.float32)
    return {
        "shape": tuple(w.shape),
        "nan": int(torch.isnan(wf).sum()),
        "inf": int(torch.isinf(wf).sum()),
        "rms": float(wf.pow(2).mean().sqrt()),
        "absmax": float(wf.abs().max()),
        "rms_row_med": float(wf.pow(2).mean(1).sqrt().median()),
        "rms_col_med": float(wf.pow(2).mean(0).sqrt().median()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="~/models/mimo26-exl3")
    ap.add_argument("--tensor", required=True, help="e.g. model.layers.0.mlp.gate_proj")
    ap.add_argument("--had", action="store_true", help="fused suh/H/W/H/svh (original basis)")
    ap.add_argument("--cb", type=int, default=None, help="override the codebook (0/1/2)")
    ap.add_argument("--mask", type=lambda s: int(s, 0), default=None, help="fractional mask override")
    ap.add_argument("--kt", type=int, nargs=2, default=None, help="tile rows [a b)")
    ap.add_argument("--nt", type=int, nargs=2, default=None, help="tile cols [a b)")
    ap.add_argument("--row-stats", type=int, default=0, help="print the N most extreme rows by RMS")
    ap.add_argument("--hist", action="store_true", help="histogram of decoded pre-scale values")
    args = ap.parse_args(argv)

    tr, suh, svh, cb = load_layer(args.model, args.tensor)
    K = K_of(tr)
    cb = args.cb if args.cb is not None else cb
    kt = tuple(args.kt) if args.kt else (0, tr.shape[0])
    nt = tuple(args.nt) if args.nt else (0, tr.shape[1])
    print(f"{args.tensor}: K = {K}  trellis {tuple(tr.shape)}  suh {suh.numel()}  svh {svh.numel()}"
          f"  codebook {CB_NAME[cb]}  mask {hex(args.mask if args.mask is not None else (MASK_HALF if K % 1 else 0))}")

    w = reconstruct(tr, K, cb, suh, svh, had=args.had, mask=args.mask,
                    kt0=kt[0], kt1=kt[1], nt0=nt[0], nt1=nt[1])
    s = stats(w)
    print(f"  {s['shape']}  rms {s['rms']:.6f}  absmax {s['absmax']:.6f}"
          f"  nan {s['nan']}  inf {s['inf']}  row-rms med {s['rms_row_med']:.6f}"
          f"  col-rms med {s['rms_col_med']:.6f}")
    if args.row_stats:
        rms = w.to(torch.float32).pow(2).mean(1).sqrt()
        top = torch.topk(rms, min(args.row_stats, rms.numel())).indices
        print("  worst rows (global idx, rms):", [(int(i), round(float(rms[i]), 5)) for i in top[:16]])
    if args.hist:
        raw = reconstruct(tr, K, cb, mask=args.mask, kt0=kt[0], kt1=kt[1], nt0=nt[0], nt1=nt[1])
        r = raw.to(torch.float32).flatten()
        cnt = torch.histc(r, bins=21, min=-1.5, max=1.5)
        print("  pre-scale histogram [-1.5,1.5] 21 bins:", [int(c) for c in cnt])
        print(f"  distinct values {len(torch.unique(r))}  min {float(r.min()):.4f} max {float(r.max()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
