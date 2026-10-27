#!/usr/bin/env python3
"""Bitwise comparison of k2_micro.py --dump fixtures (same tensors, different builds).

Usage: cmp_dumps.py <file...>  (even count; files sharing the same
   _k<K>_<shape>_<rows>.pt suffix are paired base-vs-new).
"""
import re
import sys
import torch


def suffix(f):
    m = re.search(r"_k(\d+(?:p5)?)_(.+)$", f)
    return m.group(0) if m else f


def main():
    files = sys.argv[1:]
    if len(files) < 2:
        print("usage: cmp_dumps.py base_k2_ds_2.pt new_k2_ds_2.pt ...")
        sys.exit(1)
    groups = {}
    for f in files:
        groups.setdefault(suffix(f), []).append(f)
    pairs = []
    for suf, fs in sorted(groups.items()):
        if len(fs) == 2:
            pairs.append(tuple(fs))
        else:
            print(f"{suf}: {len(fs)} files (need exactly 2), skipping: {fs}")
    if not pairs:  # fall back to consecutive pairing
        args = files
        pairs = [(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]
    bad = 0
    for fa, fb in pairs:
        A = torch.load(fa, weights_only=False)
        B = torch.load(fb, weights_only=False)
        diffs = []
        for key in ("out", "dn", "gu"):
            ta, tb = A[key], B[key]
            if ta.shape != tb.shape:
                diffs.append(f"{key}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
                continue
            iview = torch.int32 if ta.element_size() == 4 else torch.int16
            n = int((ta.view(iview) != tb.view(iview)).sum())
            if n:
                mx = float((ta.float() - tb.float()).abs().max())
                diffs.append(f"{key}: {n}/{ta.numel()} elements differ, max|d| {mx:.3e}")
        tag = "BITWISE-IDENTICAL" if not diffs else "MISMATCH"
        print(f"{fa} vs {fb}: {tag}" + ("  " + "; ".join(diffs) if diffs else ""))
        bad += bool(diffs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
