#!/usr/bin/env python3
"""prefetch_ds41.py -- fetch the DS4.1 source layers for collect_ds41.py (--external-prefetch), outside the GPU window.

Keeps at most --lookahead layers fetched ahead of <out>/next_layer.txt (written by collect_ds41 after each layer), puts the
tensors in --cache (use /dev/shm: no SSD writes), then drops <cache>/layers.N.ready. Global tensors (embed, norm, head)
are fetched first. Resumable: a layer with a .ready marker is skipped, a half-fetched layer resumes tensor by tensor.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ds41"))
LOG = lambda m: print(f"[{time.strftime('%H:%M:%S')}] [prefetch_ds41] {m}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="0-39")
ap.add_argument("--lookahead", type=int, default=3)
ap.add_argument("--threads", type=int, default=24)
a = ap.parse_args()
import capture as CAP  # noqa: E402
import fetch as F  # noqa: E402

tmap = json.load(open(a.map))
os.makedirs(a.cache, exist_ok=True)
fetcher = F.TensorFetcher(tmap, cache=a.cache)
lo, _, hi = a.layers.partition("-")
lo, hi = int(lo), int(hi or lo)
for g in ("embed.weight", "embed", "norm.weight", "norm"):
    if g in tmap["tensors"]:
        fetcher.fetch_bytes(g)
        LOG(f"global {g} cached")


def next_layer():
    try:
        return int(open(os.path.join(a.out, "next_layer.txt")).read())
    except Exception:
        return lo


for l in range(lo, hi + 1):
    while l - next_layer() >= a.lookahead:
        time.sleep(15)
    if os.path.exists(os.path.join(a.cache, f"layers.{l}.ready")):
        continue
    for attempt in range(30):
        try:
            CAP.prefetch_layer(fetcher, tmap, f"layers.{l}", a.threads, LOG)
            break
        except Exception as e:  # network resets: tensors already cached are kept, retry the rest
            LOG(f"layer {l} attempt {attempt}: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(10)
    else:
        raise SystemExit(f"layer {l}: giving up")
    open(os.path.join(a.cache, f"layers.{l}.ready"), "w").write("1")
LOG("all layers fetched")
