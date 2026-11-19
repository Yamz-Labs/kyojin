"""dense_requant.py GGUF OUTDIR KINDS [THREADS]: requantize the dense Q8_0 2-D tensors of a GGUF to Q6_K / Q5_K, CPU only (no GPU, no lock).

The bytes are what `llama-quantize --allow-requantize --tensor-type <name>=<type>` writes for these tensors without an imatrix: ggml_quantize_chunk
(libggml-base) on the fp32 dequantisation of the stored Q8_0 blocks. One file per tensor: OUTDIR/<kind>/<gguf tensor name>.bin = rows of
[K / 256] blocks (Q6_K 210 B, Q5_K 176 B). Converted: 2-D Q8_0 tensors with K % 256 == 0 (not token_embd: a lookup, requant costs quality and saves no bytes
per token). Skipped (stay Q8_0): K % 256 != 0 (k_b, compressors), 3-D tensors.
Used by exllamav3/loader/dense_req.py (HY4_DENSE_REQ_ARMS=1). KINDS = q6_k,q5_k."""
import os
import sys
import time
import types

import numpy as np

KIND = {"q6_k": (14, 210), "q5_k": (13, 176)}       # ggml type id, bytes per 256-element block


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


def main():
    gguf, out, kinds = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    nth = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    tree = os.environ.get("TREE") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    g = _collection(tree)(gguf)
    from exllamav3.loader.dense_req import quantize_rows
    for k in kinds:
        os.makedirs(os.path.join(out, k), exist_ok=True)
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
            q = quantize_rows(w, k, nth)
            q.tofile(p + ".tmp")
            os.replace(p + ".tmp", p)
            tot[k] += q.nbytes
        done += 1
        if done % 25 == 0:
            print(f"{time.time() - t0:.0f}s {done} tensors", flush=True)
    print(f"DONE tensors={done} skipped_q8={skipped} q8_bytes={tot_in / 1e9:.3f} GB " + " ".join(f"{k}={v / 1e9:.3f} GB" for k, v in tot.items()) + f" in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
