"""Fetch only the safetensors HEADERS (dtype + shape + offsets, ~ a few MB total) of every shard of the official
DeepSeek-V4.1-Flash repo into DSV41_REAL_DIR/headers/. Used by test_dsv41_real_headers (no weight bytes)."""
import json, os, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
REAL = os.environ.get("DSV41_REAL_DIR", "./ds41_work/ds41port/scratch/real")
HF = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/"
INDEX = "./ds41_work/ds41port/ref/model.safetensors.index.json"


def rng(url, a, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{a + n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = r.read()
    assert len(d) == n
    return d


def one(shard):
    out = f"{REAL}/headers/{shard}.json"
    if os.path.exists(out):
        return shard
    n = struct.unpack("<Q", rng(HF + shard, 0, 8))[0]
    h = rng(HF + shard, 8, n)
    open(out, "wb").write(h)
    return shard


if __name__ == "__main__":
    os.makedirs(f"{REAL}/headers", exist_ok=True)
    shards = sorted(set(json.load(open(INDEX))["weight_map"].values()))
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, shards))
    print(len(shards), "headers,", sum(os.path.getsize(f"{REAL}/headers/{s}.json") for s in shards), "bytes")
