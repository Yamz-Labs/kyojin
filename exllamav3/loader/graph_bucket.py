"""Context buckets of the captured decode graphs (step_graph.GraphDecoder, mtp_head.MTPGraph, step_mu): a graph is built per bucket Nb and the kernels mask the rows past the
position, so a bucket only costs the masked rows it adds.

Fixed 512-row buckets mean one graph build (2 eager bodies + the capture, 0.5 - 1 s of host time) every 512 decoded tokens and a new build for every (bucket, rows) a fresh
prompt lands in: at 256K that is 512 buckets, the first request of a process captured its own bucket inside a 47-token decode (256K spec: 11 tok/s, the same checkpoint
decoded warm: 16). Above HY4_GRAPH_BUCKET_FROM rows the bucket grows with the context: 1 / HY4_GRAPH_BUCKET_REL of its power-of-two octave (default 32: at most 3 % masked
rows, 32 buckets per octave: 8192 rows at 256K, 16384 at 512K, 32768 at 1M). HY4_GRAPH_BUCKET_REL=0 keeps the fixed bucket."""
import os

BASE = int(os.environ.get("HY4_GRAPH_BUCKET", "512"))
REL = int(os.environ.get("HY4_GRAPH_BUCKET_REL", "32"))
FROM = int(os.environ.get("HY4_GRAPH_BUCKET_FROM", "16384"))


def size(n: int, base: int = BASE, rel: int = REL, start: int = FROM) -> int:
    """bucket granule for a step that needs n rows"""
    if rel <= 0 or n <= start:
        return base
    return max(base, (1 << (n.bit_length() - 1)) // rel // base * base)


def bucket(n: int, cap: int, base: int = BASE, rel: int = REL, start: int = FROM) -> int:
    """smallest bucket >= n rows (a multiple of base), at most cap"""
    s = size(n, base, rel, start)
    return min(cap, -(-n // s) * s)


def buckets(cap: int, base: int = BASE, rel: int = REL, start: int = FROM) -> list[int]:
    """every bucket a step can use, ascending (what precapture builds)"""
    out, n = [], 1
    while True:
        b = bucket(n, cap, base, rel, start)
        out.append(b)
        if b >= cap:
            return out
        n = b + 1
