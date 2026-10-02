#!/usr/bin/env python3
"""
List every quantised EXL3 tensor group in a checkpoint with its bitrate K, read straight from the
safetensors headers (8-byte length + JSON header) -- no torch, no tensor materialisation.

K is implied by the packed trellis width: the loader stores `.trellis` as uint16 with last dim
`16 * K`, so K = width / 16 (2.5 -> 40, 3.0 -> 48, 4.0 -> 64).

  python ref/list_k.py --model ~/models/mimo26-exl3 [--K 4.0]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from collections import Counter


def header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="~/models/mimo26-exl3")
    ap.add_argument("--K", type=float, default=None, help="only print groups with this K")
    args = ap.parse_args(argv)

    groups: dict[str, dict] = {}
    for shard in sorted(glob.glob(os.path.join(args.model, "*.safetensors"))):
        for name, meta in header(shard).items():
            if name == "__metadata__":
                continue
            if name.endswith(".trellis"):
                groups.setdefault(name[:-len(".trellis")], {})["trellis"] = (meta["dtype"],
                                                                             tuple(meta["shape"]),
                                                                             os.path.basename(shard))
            else:
                key, _, suffix = name.rpartition(".")
                groups.setdefault(key, {})[suffix] = (meta["dtype"], tuple(meta["shape"]),
                                                      os.path.basename(shard))

    by_k: Counter = Counter()
    rows = []
    for key, t in groups.items():
        if "trellis" not in t:
            continue
        w = t["trellis"][1][-1]
        K = w / 16
        by_k[K] += 1
        cb = next((s for s in ("mul1", "mcg") if s in t), "plain")
        rows.append((K, key, t["trellis"][1], cb, t["trellis"][2]))

    print(f"{args.model}: {len(rows)} quantised groups, K histogram "
          f"{ {k: v for k, v in sorted(by_k.items())} }")
    for K, key, shape, cb, shard in sorted(rows, key=lambda r: (r[0], r[1])):
        if args.K is not None and abs(K - args.K) > 1e-6:
            continue
        print(f"  K={K:<5} {key:<55} trellis {shape} cb={cb} {shard}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
