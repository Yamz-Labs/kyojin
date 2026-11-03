"""Range-scan the safetensors headers of any HF repo into a tensor map (same layout as tools/ds41/tensor_map.py).

    python -m bigconv.tensor_map --repo MiniMaxAI/MiniMax-M3 --revision <sha> --out tensor_map.json [--workers 8]

Only the 8-byte length prefix and the JSON header of each shard are read (no weights). Resumable.
Map: {"repo", "revision", "shards": {file: {header_len, file_size}},
      "tensors": {name: {file, dtype, shape, offset, length}}}   (offset = absolute byte offset of the data)
"""
import argparse
import http.client
import json
import os
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "U16": 2, "U32": 4, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F4": 0.5}


def _get(url, a, b, retries=6, token=None):
    last = None
    for k in range(retries):
        try:
            h = {"Range": f"bytes={a}-{b}", "User-Agent": "bigconv-tensor-map/1"}
            if token:
                h["Authorization"] = f"Bearer {token}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                data = r.read()
                cr = r.headers.get("Content-Range", "")
            if len(data) != b - a + 1:
                raise IOError("short read")
            return data, (int(cr.rsplit("/", 1)[1]) if "/" in cr else None)
        except (urllib.error.URLError, IOError, TimeoutError, http.client.HTTPException) as e:   # IncompleteRead
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise
            last = e
            time.sleep(min(30, 1.5 * 2 ** k))
    raise IOError(f"{url}: {last}")


def scan_shard(url, fname, token=None):
    raw, total = _get(url, 0, 7, token=token)
    n = struct.unpack("<Q", raw)[0]
    hdr = json.loads(_get(url, 8, 8 + n - 1, token=token)[0])
    base, tensors, spans = 8 + n, {}, []
    for name, m in hdr.items():
        if name == "__metadata__":
            continue
        b, e = m["data_offsets"]
        numel = 1
        for d in m["shape"]:
            numel *= d
        if e - b != int(numel * DTYPE_BYTES[m["dtype"]]):
            raise ValueError(f"{fname}:{name}: {e - b} bytes != {m['dtype']}{m['shape']}")
        tensors[name] = {"file": fname, "dtype": m["dtype"], "shape": m["shape"], "offset": base + b, "length": e - b}
        spans.append((b, e))
    spans.sort()
    pos = 0
    for b, e in spans:
        if b != pos:
            raise ValueError(f"{fname}: gap or overlap at {b}")
        pos = e
    if base + pos != total:
        raise ValueError(f"{fname}: data ends at {base + pos}, file is {total}")
    return tensors, {"header_len": n, "file_size": total}


def build(repo, revision, out, workers=8, token=None, log=print):
    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    req = urllib.request.Request(f"{base}/model.safetensors.index.json", headers={"User-Agent": "bigconv/1"})
    index = json.loads(urllib.request.urlopen(req, timeout=120).read())
    files = sorted(set(index["weight_map"].values()))
    m = {"repo": repo, "revision": revision, "shards": {}, "tensors": {}}
    if os.path.exists(out):
        old = json.load(open(out))
        if old.get("repo") == repo and old.get("revision") == revision:
            m = old
    todo = [f for f in files if f not in m["shards"]]
    with ThreadPoolExecutor(workers) as ex:
        futs = {f: ex.submit(scan_shard, f"{base}/{f}", f, token) for f in todo}
        for f in todo:
            t, info = futs[f].result()
            m["tensors"].update(t)
            m["shards"][f] = info
    tmp = out + ".tmp"
    json.dump(m, open(tmp, "w"))
    os.replace(tmp, out)
    names = set(m["tensors"])
    if names != set(index["weight_map"]):
        raise SystemExit(f"name mismatch vs index: {len(set(index['weight_map']) - names)} missing")
    tot = sum(t["length"] for t in m["tensors"].values())
    want = index.get("metadata", {}).get("total_size")
    log(f"ok: {len(names)} tensors, {tot} bytes" + (f" (index total_size {want}{'' if want == tot else ' MISMATCH'})" if want else ""))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(a.repo, a.revision, a.out, a.workers, os.environ.get("HF_TOKEN"))
