# Two-stage draft head (EXL3_DRAFT2S), offline pack builder (CPU).
# Tier 1 preview: a data-aware rotation of the head input (512 axes, fit in the head's metric on captured draft-head inputs)
#   and 4-bit codes of the rotated head rows, one scale per row: score[v] = pscale[v] * sum_j (code[v, j] - 7.5) * (h @ E)[j].
# Tier 2 row copy: the head rows after a 128-point input Hadamard per group, 3 bits per weight, one fp16 scale per (row,
#   group): row-addressable, unlike the EXL3 head (its output Hadamard mixes 128 rows).
# Tier 3 is the EXL3 head itself (exact logits of the short-listed 128-row blocks): nothing to pack.
# Maths checked offline (recall 0.9958 at m = 512 on held-out steps).
# usage: draft2s_pack.py W.npy HDIR OUT.safetensors     (W.npy = the dequantized head (in_features, vocab); HDIR = rec.json +
#        h_*.npy of a capture; only the "train" split is used for the fit)
import os, sys, json
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "8")
import numpy as np
import torch
from safetensors.torch import save_file

AXES, GROUP = int(os.environ.get("DRAFT2S_AXES", 512)), 128
M = int(os.environ.get("DRAFT2S_M", 512))   # tier-1 target count; GLM 4096-wide head: 1024 (measured)

def hadamard(n):
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n: h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)

def build(W, Htr):
    d, V = W.shape
    G = W @ W.T; ew, U = np.linalg.eigh(G.astype(np.float64)); U = U[:, ::-1].astype(np.float32); ew = ew[::-1]
    C = (Htr.T @ Htr) / len(Htr)
    S = (U * np.sqrt(np.maximum(ew, 1e-12)).astype(np.float32)) @ U.T
    em, Q = np.linalg.eigh((S @ C @ S).astype(np.float64)); Q = Q[:, ::-1].astype(np.float32)
    Sinv = (U / np.sqrt(np.maximum(ew, 1e-12)).astype(np.float32)) @ U.T
    E = (S @ Q)[:, :AXES]; B = W.T @ (Sinv @ Q)[:, :AXES]; del G
    step = 0.335 * (B.std(0) + 1e-12)
    q = np.floor(B / step).clip(-8, 7)
    Bq = (q + 0.5) * step
    al = (B * Bq).sum(1) / np.maximum((Bq * Bq).sum(1), 1e-20)
    code = (q + 8).astype(np.uint32)                                     # [V, 512] in [0, 15]
    pcodes = np.zeros((V, AXES // 8), np.uint32)
    for i in range(8): pcodes |= code[:, i::8] << (4 * i)                # nibble i of word w = axis 8 w + i
    del B, Bq, q, code
    HAD = hadamard(GROUP)
    Wg = np.einsum("ij,bjv->biv", HAD, W.reshape(d // GROUP, GROUP, V))   # [groups, 128, V]
    s3 = (np.abs(Wg).max(1) / 3).astype(np.float16)                      # [groups, V]
    s3f = s3.astype(np.float32); s3f[s3f == 0] = 1
    c3 = (np.clip(np.round(Wg / s3f[:, None, :]), -4, 3) + 4).astype(np.uint32).reshape(d, V).T   # [V, d] in [0, 7]
    del Wg
    qw = -(-d // 10)                                                    # words per row; d % 10 != 0 (GLM 4096): pad code 4 (= 0)
    if qw * 10 > d: c3 = np.concatenate([c3, np.full((V, qw * 10 - d), 4, np.uint32)], 1)
    q3 = np.zeros((V, qw), np.uint32)
    for i in range(10): q3 |= c3[:, i::10] << (3 * i)                    # bits 3i..3i+2 of word w = input 10 w + i
    del c3
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return {"E": t((E * step[None, :]).T.astype(np.float16)),             # [512, d]: xr = E @ h (the code step folded in)
            "pcodes": t(pcodes.view(np.int32)), "pscale": t(al.astype(np.float32)),
            "q3": t(q3.view(np.int32)), "q3s": t(s3.T.copy()), "had": t(HAD.astype(np.float16))}

if __name__ == "__main__":
    WP, HD, OUT = sys.argv[1:4]
    R = json.load(open(f"{HD}/rec.json"))
    V = R["head"]["vocab"]
    W = np.ascontiguousarray(np.load(WP, mmap_mode="r")[:, :V], dtype=np.float32)
    Htr = np.concatenate([np.load(f"{HD}/h_{i}.npy").astype(np.float32) for i, r in enumerate(R["rec"]) if r["split"] == "train"])
    P = build(W, Htr)
    meta = {"format": "draft2s-v1", "axes": str(AXES), "group": str(GROUP), "m": str(M), "vocab": str(V), "in_features": str(W.shape[0]),
            "fit_rows": str(len(Htr))}
    save_file(P, OUT, metadata=meta)
    print(OUT, {k: (tuple(v.shape), str(v.dtype)) for k, v in P.items()}, f"{sum(v.numel() * v.element_size() for v in P.values()) / 1e6:.1f} MB")
