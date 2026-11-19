"""dense_requant.py GGUF OUTDIR KINDS [THREADS]: requantize the dense Q8_0 2-D tensors of a GGUF to Q6_K / Q5_K, CPU only (no GPU, no lock).

The bytes are what `llama-quantize --allow-requantize --tensor-type <name>=<type>` writes for these tensors without an imatrix: ggml_quantize_chunk
(libggml-base) on the fp32 dequantisation of the stored Q8_0 blocks. One file per tensor: OUTDIR/<kind>/<gguf tensor name>.bin = rows of
[K / 256] blocks (Q6_K 210 B, Q5_K 176 B). Converted: 2-D Q8_0 tensors with K % 256 == 0 (not token_embd: a lookup, requant costs quality and saves no bytes
per token). Skipped (stay Q8_0): K % 256 != 0 (k_b, compressors), 3-D tensors.
Used by exllamav3/loader/dense_req.py (HY4_DENSE_REQ_ARMS=1). KINDS = q6_k,q5_k."""
import ctypes
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np

KIND = {"q6_k": (14, 210), "q5_k": (13, 176)}       # ggml type id, bytes per 256-element block
LIB = os.environ.get("GGML_LIB", "/path/to/libggml-base.so")


def _collection(tree):
    for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
        m = types.ModuleType(n)
        m.__path__ = [os.path.join(tree, "exllamav3", s)]
        sys.modules[n] = m
    from exllamav3.loader.gguf_collection import GGUFCollection
    return GGUFCollection


def q8_0_to_f32(raw: np.ndarray, N: int, K: int) -> np.ndarray:
    b = raw.reshape(N, K // 32, 34)
    d = np.ascontiguousarray(b[:, :, :2]).view(np.float16)[:, :, 0].astype(np.float32)
    q = np.ascontiguousarray(b[:, :, 2:]).view(np.int8).astype(np.float32)
    return (q * d[:, :, None]).reshape(N, K)


def quantize(lib, kind, w: np.ndarray, pool, rows_per=64) -> np.ndarray:
    ty, bb = KIND[kind]
    N, K = w.shape
    rb = K // 256 * bb
    dst = np.empty((N, rb), dtype=np.uint8)
    w = np.ascontiguousarray(w, dtype=np.float32)

    def run(r0):
        r1 = min(N, r0 + rows_per)
        n = lib.ggml_quantize_chunk(ty, ctypes.c_void_p(w[r0:].ctypes.data), ctypes.c_void_p(dst[r0:].ctypes.data), 0, r1 - r0, K, None)
        assert n == (r1 - r0) * rb, (n, r1 - r0, rb)

    list(pool.map(run, range(0, N, rows_per)))
    return dst


def main():
    gguf, out, kinds = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    nth = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    tree = os.environ.get("TREE") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    g = _collection(tree)(gguf)
    lib = ctypes.CDLL(LIB)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    for k in kinds:
        os.makedirs(os.path.join(out, k), exist_ok=True)
    pool = ThreadPoolExecutor(nth)
    t0, tot_in, tot = time.time(), 0, {k: 0 for k in kinds}
    done = skipped = 0
    for name, (ty, dims, off) in sorted(g.infos.items()):
        if ty != 8 or len(dims) != 2 or dims[0] % 256 or name == "token_embd.weight":
            if ty == 8:
                skipped += 1
            continue
        K, N = dims
        raw = np.asarray(g._pread(off, N * (K // 32) * 34))
        w = q8_0_to_f32(raw, N, K)
        tot_in += raw.nbytes
        for k in kinds:
            p = os.path.join(out, k, name + ".bin")
            if os.path.exists(p) and os.path.getsize(p) == N * (K // 256) * KIND[k][1]:
                tot[k] += os.path.getsize(p)
                continue
            q = quantize(lib, k, w, pool)
            q.tofile(p + ".tmp")
            os.replace(p + ".tmp", p)
            tot[k] += q.nbytes
        done += 1
        if done % 25 == 0:
            print(f"{time.time() - t0:.0f}s {done} tensors", flush=True)
    print(f"DONE tensors={done} skipped_q8={skipped} q8_bytes={tot_in / 1e9:.3f} GB " + " ".join(f"{k}={v / 1e9:.3f} GB" for k, v in tot.items()) + f" in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
