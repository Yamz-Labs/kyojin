#!/usr/bin/env python3
"""
Phase A2 diagnostic -- recover the exact ring-position -> tile-element mapping from the fork's own
HIP reconstruct, on one real integer-K tile, instead of trusting a hand-transcribed permutation.

Why: ref/frac_reconstruct.py first disagreed with ext.reconstruct on 98% of elements while agreeing
on the overall RMS, which is the signature of a placement error, not a codebook error. The kernel is
deterministic (kernel vs kernel = bit-exact), so it is the ground truth for the mapping.

The kernel cannot emit ring order, so the test is in two steps per 16x16 tile:
  1. multiset test: sort(kernel 16x16 block) vs sort(reference 256 ring values). Equal => the
     window extraction + codebook are right and only the placement is wrong.
  2. if equal, match each kernel element to the ring position holding the same bit pattern and
     print the resulting permutation (and its inverse), flagging values that occur more than once.

The kernel needs at least 128 output columns (reconstruct_slice TORCH_CHECK), so one call covers 8
tiles of one k-row: kt = 0..1, nt = 0..8 -> (16, 128).

  LD_PRELOAD=/opt/rocm-7.2.4/lib/libhsa-runtime64.so.1 \
  PYTHONPATH=~/kyojin \
  ~/kyojin/.venv/bin/python ref/probe_tile.py \
      --tensor model.layers.0.self_attn.q_proj
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

REF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REF_DIR)

import frac_reconstruct as R  # noqa: E402

EXT_DIR = os.environ.get("EXL3_EXT_DIR", os.path.dirname(REF_DIR))


def get_ext():
    sys.path.insert(0, EXT_DIR)
    import exllamav3_ext as ext  # type: ignore
    return ext


def as_bits(x: torch.Tensor) -> torch.Tensor:
    """fp16 -> int16 bit patterns as int32, so equality is exact and NaN-safe."""
    return x.contiguous().view(torch.int16).to(torch.int32)


def match_perm(blk_bits: torch.Tensor, ring_bits: torch.Tensor):
    """blk_bits (256,) row-major tile, ring_bits (256,) ring order. -> (perm[ring_pos] = rowmajor
    index, ambiguous count). None if the multisets differ."""
    if sorted(blk_bits.tolist()) != sorted(ring_bits.tolist()):
        return None, 0
    pos = {}
    for i, v in enumerate(ring_bits.tolist()):
        pos.setdefault(v, []).append(i)
    perm, amb = [], 0
    for v in blk_bits.tolist():
        cand = pos[v]
        if len(cand) > 1:
            amb += 1
        perm.append(cand.pop(0))
    return perm, amb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="~/models/mimo26-exl3")
    ap.add_argument("--tensor", default="model.layers.0.self_attn.q_proj")
    ap.add_argument("--nt", type=int, nargs=2, default=(0, 8), help="tile cols [a b), b-a >= 8")
    args = ap.parse_args(argv)

    ext = get_ext()
    tr, suh, svh, cb = R.load_layer(args.model, args.tensor)
    K = R.K_of(tr)
    assert float(K).is_integer(), f"{args.tensor}: K = {K} is fractional"
    kt, nt = (0, 1), tuple(args.nt)
    packed = tr[kt[0]:kt[1], nt[0]:nt[1]].contiguous()
    mcg, mul1 = cb == R.CB_MCG, cb == R.CB_MUL1
    print(f"{args.tensor}: K={K} cb={cb} packed {tuple(packed.shape)}")

    w = torch.empty((16, 16 * (nt[1] - nt[0])), dtype=torch.half, device="cuda")
    ext.reconstruct(w, packed.to("cuda"), int(K), mcg, mul1)
    torch.cuda.synchronize()
    wb = as_bits(w.cpu())

    t16 = tr.contiguous().view(torch.uint16)
    for j in range(nt[1] - nt[0]):
        # windows_of_row takes a 2D (nt, bpb) row block
        ring = R.decode_windows(R.windows_of_row(t16[kt[0], nt[0] + j:nt[0] + j + 1], K, None),
                                cb)[0]
        blk = wb[:, 16 * j:16 * j + 16].reshape(-1)
        rb = as_bits(ring)
        n_common = int((torch.sort(blk).values == torch.sort(rb).values).sum())
        perm, amb = match_perm(blk, rb)
        print(f"\n--- tile (kt={kt[0]}, nt={nt[0] + j})")
        print(f"  multiset: {n_common}/256 positions equal after sorting"
              f"  ({'IDENTICAL MULTISET' if n_common == 256 else 'MISMATCH'})")
        if perm is None:
            print("  -> decoded values differ, not just placement; codebook/window bug")
            continue
        print(f"  exact placement match: {sum(1 for i, p in enumerate(perm) if i == p)}/256 ring "
              f"positions land where the reference puts them; {amb} ambiguous (duplicate) values")
        print(f"  perm[ring_pos] = rowmajor_index:\n  {perm}")
        inv = [0] * 256
        for p, i in enumerate(perm):
            inv[i] = p
        print(f"  inv[rowmajor_index] = ring_pos:\n  {inv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
