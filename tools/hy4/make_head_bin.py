"""make_head_bin.py GGUF OUT.bin [THREADS]: Q5_K bytes of the bf16 output head (ggml_quantize_chunk, no imatrix) for HY4_MTP_HEAD=q5k. CPU only."""
import os, sys, types
import numpy as np
tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    m = types.ModuleType(n); m.__path__ = [os.path.join(tree, "exllamav3", s)]; sys.modules[n] = m
from exllamav3.loader.gguf_collection import GGUFCollection
from exllamav3.loader.dense_req import quantize_rows
g = GGUFCollection(sys.argv[1])
ty, dims, off = g.infos["output.weight"]
assert ty == 30, ty
K, N = dims
raw = np.asarray(g._pread(off, N * K * 2)).view(np.uint16)
out = []
for a in range(0, N, 8192):
    w = (raw[a * K:min(N, a + 8192) * K].astype(np.uint32) << 16).view(np.float32).reshape(-1, K)
    out.append(quantize_rows(w, "q5_k", int(sys.argv[3]) if len(sys.argv) > 3 else 12))
q = np.concatenate(out)
assert q.shape == (N, K // 256 * 176), q.shape
q.tofile(sys.argv[2])
print("OK", q.shape, q.nbytes)
