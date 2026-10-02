#!/usr/bin/env python3
"""
Phase A2 -- recover the exact tile-element -> ring-position map from the fork's own HIP reconstruct,
using SYNTHETIC random tiles so that the 256 decoded values of a tile are (almost) all distinct.

On real weights the greedy value-match is ambiguous (93 of 256 values duplicated in one K=4 tile),
which is why probe_tile.py could not pin the permutation. Random packed data removes that: every
ring position gets its own codebook value, so a cell's value identifies its ring position. Three
seeds are used and only positions that agree across all seeds are kept (a duplicate in one seed
cannot survive in another).

The kernel is the ground truth for the mapping: it is deterministic (kernel vs kernel is bit-exact)
and it is what the fork's own loader/forward pass uses.

  LD_PRELOAD=/opt/rocm-7.2.4/lib/libhsa-runtime64.so.1 \
  PYTHONPATH=~/kyojin \
  ~/kyojin/.venv/bin/python ref/probe_perm.py --K 4

Output: inv[rowmajor_index] = ring_position, plus ref/perm_K<K>.json.
"""

from __future__ import annotations

import argparse
import json
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


def bits16(x: torch.Tensor) -> list:
    return x.contiguous().view(torch.int16).to(torch.int32).tolist()


def one_seed(ext, K: int, seed: int, nt: int = 8, cb: int = R.CB_MUL1):
    """-> list of 8 tiles, each a list of 256 ring positions indexed by row-major cell."""
    gen = torch.Generator().manual_seed(seed)
    u16 = torch.randint(0, 1 << 16, (1, nt, 16 * K), dtype=torch.int32, generator=gen)
    packed = u16.to(torch.uint16).view(torch.int16).contiguous()

    w = torch.empty((16, 16 * nt), dtype=torch.half, device="cuda")
    ext.reconstruct(w, packed.to("cuda"), K, False, cb == R.CB_MUL1)
    torch.cuda.synchronize()
    wb = bits16(w.cpu())

    t16 = packed.view(torch.uint16)
    out = []
    for t in range(nt):
        ring = bits16(R.decode_windows(R.windows_of_row(t16[0, t:t + 1], float(K), None), cb)[0])
        blk = [wb[r][16 * t + c] for r in range(16) for c in range(16)]
        if sorted(blk) != sorted(ring):
            raise SystemExit(f"seed {seed} tile {t}: multiset mismatch -- decode bug, not placement")
        pos: dict = {}
        for p, v in enumerate(ring):
            pos.setdefault(v, []).append(p)
        out.append([pos[v].pop(0) for v in blk])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--cb", type=int, default=R.CB_MUL1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ext = get_ext()
    runs = [one_seed(ext, args.K, s, cb=args.cb) for s in args.seeds]

    inv, unconfirmed = [], 0
    for cell in range(256):
        # consensus over every seed and every one of the 8 tiles: duplicates resolve differently
        # per seed, so a position that is stable everywhere is the true one
        cand = {run[t][cell] for run in runs for t in range(len(run))}
        if len(cand) == 1:
            inv.append(cand.pop())
        else:
            inv.append(min(cand))
            unconfirmed += 1
    print(f"K={args.K} cb={args.cb} seeds={args.seeds}: {256 - unconfirmed}/256 cells resolved "
          f"unambiguously across all seeds and all 8 tiles of each seed")
    if unconfirmed:
        print(f"  {unconfirmed} cells still ambiguous (duplicate codebook values in every seed): "
              f"{[i for i, (a, b) in enumerate(zip(inv, inv)) if False]}")
    print(f"  bijection: {len(set(inv)) == 256}")
    print(f"  inv[rowmajor_index] = ring_pos:\n  {inv}")
    out = args.out or os.path.join(REF_DIR, f"perm_K{args.K}.json")
    with open(out, "w") as f:
        json.dump({"K": args.K, "cb": args.cb, "seeds": args.seeds, "inv": inv,
                   "unconfirmed": unconfirmed}, f)
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
