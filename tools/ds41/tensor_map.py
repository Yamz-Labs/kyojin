"""Range-scan all shard headers of the checkpoint into one JSON map.

    python tools/ds41/tensor_map.py --out tensor_map.json [--workers 8]

Map layout:
    {"repo": ..., "revision": ...,
     "shards": {file: {"header_len": int, "file_size": int}},
     "tensors": {name: {"file", "dtype", "shape", "offset", "length"}}}

`offset` is the absolute byte offset of the tensor data inside the file. Only the 8-byte length
prefix and the JSON header of each shard are downloaded. The scan is resumable: shards already
present in the output file are skipped, and the file is rewritten after every shard.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def scan_shard(fname, repo, revision):
    hdr, n, total = C.read_header(C.file_url(fname, repo, revision))
    base = 8 + n
    tensors = {}
    spans = []
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        length = e - b
        want = C.expected_nbytes(meta["dtype"], meta["shape"])
        if length != want:
            raise ValueError(f"{fname}:{name} size {length} != {want} for {meta['dtype']}{meta['shape']}")
        tensors[name] = {"file": fname, "dtype": meta["dtype"], "shape": meta["shape"],
                         "offset": base + b, "length": length}
        spans.append((b, e))
    spans.sort()
    pos = 0
    for b, e in spans:
        if b != pos:
            raise ValueError(f"{fname}: gap or overlap at {b} (expected {pos})")
        pos = e
    if base + pos != total:
        raise ValueError(f"{fname}: data ends at {base + pos} but file is {total} bytes")
    return tensors, {"header_len": n, "file_size": total}


def verify(m, index):
    """Completeness checks against the repo index. Returns a list of problems (empty = ok)."""
    problems = []
    wm = index["weight_map"]
    names = set(m["tensors"])
    if names != set(wm):
        problems.append(f"name mismatch: {len(set(wm) - names)} missing, {len(names - set(wm))} extra")
    for n, f in wm.items():
        t = m["tensors"].get(n)
        if t and t["file"] != f:
            problems.append(f"{n}: file {t['file']} != index {f}")
    total = sum(t["length"] for t in m["tensors"].values())
    want = index.get("metadata", {}).get("total_size")
    if want is not None and total != want:
        problems.append(f"total bytes {total} != index total_size {want}")
    return problems


def build(out, repo=C.REPO, revision=C.REVISION, workers=8, log=print):
    index = C.fetch_json("model.safetensors.index.json", repo, revision)
    files = sorted(set(index["weight_map"].values()))
    m = {"repo": repo, "revision": revision, "shards": {}, "tensors": {}}
    if os.path.exists(out):
        old = json.load(open(out))
        if old.get("repo") == repo and old.get("revision") == revision:
            m = old
    todo = [f for f in files if f not in m["shards"]]
    log(f"{len(files) - len(todo)}/{len(files)} shards already scanned")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {f: ex.submit(scan_shard, f, repo, revision) for f in todo}
        for f in todo:
            tensors, info = futs[f].result()
            m["tensors"].update(tensors)
            m["shards"][f] = info
            C.atomic_write_json(out, m)
            log(f"scanned {f}: {len(tensors)} tensors")
    problems = verify(m, index)
    if problems:
        raise SystemExit("verification failed:\n  " + "\n  ".join(problems))
    log(f"ok: {len(m['tensors'])} tensors, {sum(t['length'] for t in m['tensors'].values())} bytes")
    return m


def load(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tensor_map.json")
    ap.add_argument("--repo", default=C.REPO)
    ap.add_argument("--revision", default=C.REVISION)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(a.out, a.repo, a.revision, a.workers)
