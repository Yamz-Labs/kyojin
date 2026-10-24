#!/usr/bin/env python3
"""hook_ablate.py -- inference-time directional ablation hooks (ablit4).

Implements EXACTLY the weight edit
    W_writer <- W_writer - w_comp(L) * r_hat (r_hat^T W_writer)
on every residual writer (each layer's attn o_proj + all expert/shared/dense
down_proj) as an activation hook:
    after each writer output y:  y <- y - w_comp(L) * (y . r_hat) * r_hat
Proof of equivalence: for y = W x,
    W' x = Wx - w r_hat (r_hat^T W x) = y - w (y . r_hat) r_hat.  Exact for any x.
Hooking the summed attn output / summed MLP output of a block equals editing
each writer of that block with the same w (linearity of the sum). Per-expert
strengths are NOT in this search space (uniform treatment, P1 style).

Heretic-style search space (exact formulas from p-e-w/heretic v1.2.0 + docs):
  weight kernel per component c in {attn, mlp}, over layer index L:
      d = |L - peak| ; w(L) = 0 if d > width
      else w(L) = max_w + (d / width) * (min_w - max_w)   (linear taper)
  direction: mode "perlayer" (each layer its own r_L) or float dir_index f in
      [0, n_layers-1]: r = normalize((1-frac)*r_i + frac*r_{i+1}), i=floor(f).
      ("single layer" == interp with frac == 0.)

mHC note: the 4x4 stream mixer acts on the stream index, never rotating inside
the hidden dim, so ablating each stream's y along r_hat commutes with mixing;
a hook before the mixer equals the weight edit. The hook wraps block
attn/mlp forward entry points, i.e. OUTSIDE all fused kernels / internal CUDA
graphs, so fused decode kernels (exl3_dec_moe, BC graphs) are still exact.
Block-level HIP graphs (EXL3_BLOCK_GRAPH, default off) would bypass Python
wrappers -- keep them off during search.

CPU proof: --prove runs a fp32 random-Linear equivalence test + kernel/dir
unit tests, no model, no GPU.
"""
import argparse
import math

N_LAYERS = 45
HIDDEN = 4096


# ------------------------------------------------------------- Heretic kernel
def heretic_kernel(n_layers, max_w, peak, min_w, width):
    """Linear-taper weight kernel, Heretic model.py formula. Returns list[float]."""
    w = []
    for L in range(n_layers):
        d = abs(L - peak)
        if d > width:
            w.append(0.0)
        else:
            w.append(max_w + (d / width) * (min_w - max_w) if width > 0 else max_w)
    return w


def interp_direction(R, f):
    """R: (n, d) array-like of per-layer directions. f: float index.
    Returns normalized interpolated vector (single direction for all layers)."""
    import numpy as np
    R = np.asarray(R, dtype=np.float64)
    n = len(R)
    f = min(max(f, 0.0), n - 1.0)
    i = int(math.floor(f))
    frac = f - i
    j = min(i + 1, n - 1)
    v = (1.0 - frac) * R[i] + frac * R[j]
    return v / (float(__import__("numpy").linalg.norm(v)) + 1e-12)


def build_edit(R_mats, mode, dir_index, attn_p, mlp_p):
    """R_mats: (45, 4096) float array (raw diff-means, any scale).
    attn_p/mlp_p: dicts {max_w, peak, min_w, width}.
    Returns dict {attn_w: (45,), mlp_w: (45,), r: (45, 4096) normalized}."""
    import numpy as np
    R = np.asarray(R_mats, dtype=np.float64)
    n = len(R)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    attn_w = heretic_kernel(n, attn_p["max_w"], attn_p["peak"],
                            attn_p["min_w"], attn_p["width"])
    mlp_w = heretic_kernel(n, mlp_p["max_w"], mlp_p["peak"],
                           mlp_p["min_w"], mlp_p["width"])
    if mode == "perlayer":
        r = Rn
    else:
        v = interp_direction(R, dir_index)
        r = (v / (np.linalg.norm(v) + 1e-12))
        r = np.repeat(r[None, :], n, axis=0)
    return {"attn_w": np.asarray(attn_w), "mlp_w": np.asarray(mlp_w), "r": r}


def projected_direction(R_mats, f, mu_h):
    """grimjim-style: single interp direction with the harmless-mean component
    removed. v = interp(R, f); v' = normalize(v - (v.mu_hat) mu_hat).
    Returns (d,) float64 unit vector orthogonal to mu_hat."""
    import numpy as np
    v = interp_direction(np.asarray(R_mats, dtype=np.float64), f)
    m = np.asarray(mu_h, dtype=np.float64)
    m = m / (np.linalg.norm(m) + 1e-12)
    v = v - float(v @ m) * m
    return v / (np.linalg.norm(v) + 1e-12)


def build_single_edit(direction, attn_p, mlp_p, n_layers=N_LAYERS):
    """Same kernels as build_edit but one direction vector for every layer
    (projected arm). direction: (d,) array-like, any scale (normalized here)."""
    import numpy as np
    v = np.asarray(direction, dtype=np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)
    attn_w = heretic_kernel(n_layers, attn_p["max_w"], attn_p["peak"],
                            attn_p["min_w"], attn_p["width"])
    mlp_w = heretic_kernel(n_layers, mlp_p["max_w"], mlp_p["peak"],
                           mlp_p["min_w"], mlp_p["width"])
    return {"attn_w": np.asarray(attn_w), "mlp_w": np.asarray(mlp_w),
            "r": np.repeat(v[None, :], n_layers, axis=0)}


def pca_subspace(D, k):
    """Top-k orthonormal basis (rows) of the diff matrix D: (n, d).
    Returns (U (k, d) float64, explained (k,) variance ratios)."""
    import numpy as np
    D = np.asarray(D, dtype=np.float64)
    Dc = D - D.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    var = S ** 2
    tot = float(var.sum()) + 1e-12
    return Vt[:k].copy(), (var[:k] / tot).copy()


def build_subspace_edit(U, attn_p, mlp_p, n_layers=N_LAYERS):
    """Ablate the k-dim subspace spanned by orthonormal rows U: (k, d):
    y <- y - w * U^T (U y), == W <- W - w U^T U W per writer.
    Same U at every layer; kernels searched as usual."""
    import numpy as np
    U = np.asarray(U, dtype=np.float64)
    k = U.shape[0]
    # re-orthonormalize (QR) for numerical safety
    Q, _ = np.linalg.qr(U.T)
    U = Q.T[:k]
    attn_w = heretic_kernel(n_layers, attn_p["max_w"], attn_p["peak"],
                            attn_p["min_w"], attn_p["width"])
    mlp_w = heretic_kernel(n_layers, mlp_p["max_w"], mlp_p["peak"],
                           mlp_p["min_w"], mlp_p["width"])
    r = np.repeat(U[None, :, :], n_layers, axis=0)
    return {"attn_w": np.asarray(attn_w), "mlp_w": np.asarray(mlp_w), "r": r}


# ------------------------------------------------------------------ GPU hooks
class AblationHooks:
    """Wraps each decoder block's attn/mlp forward with the hook ablation.

    Config is pushed per trial via set_edit() with zero reload cost; enabled
    flag toggles original vs edited in the same process (for KL pairs).
    Covers layer_idx 0..n_layers-1 blocks that expose .attn / .mlp (MTP head untouched).
    """

    def __init__(self, model, n_layers=N_LAYERS, hidden=HIDDEN):
        import torch
        self.torch = torch
        self.n_layers = n_layers
        self.hidden = hidden
        from exllamav3.modules.transformer import TransformerBlock
        self.model = model
        self.blocks = [m for m in model.modules
                       if isinstance(m, TransformerBlock)
                       and getattr(m, "layer_idx", -1) is not None
                       and 0 <= m.layer_idx < n_layers]
        assert len(self.blocks) == n_layers, \
            f"expected {n_layers} blocks, got {len(self.blocks)}"
        self.enabled = False
        self.attn_w = [0.0] * n_layers
        self.mlp_w = [0.0] * n_layers
        self.r = None  # torch (n_layers, hidden) fp32, moved to device on first use
        self._r_dev = {}
        self._wrapped = False

    def _r_on(self, device):
        r = self._r_dev.get(str(device))
        if r is None:
            r = self.r.to(device)
            self._r_dev[str(device)] = r
        return r

    def _ablate(self, y, L, comp):
        if not self.enabled or y is None:
            return y
        w = self.attn_w[L] if comp == "attn" else self.mlp_w[L]
        if w == 0.0:
            return y
        r = self._r_on(y.device)[L]
        yf = y.float()
        if r.dim() == 1:
            coeff = (yf * r).sum(dim=-1, keepdim=True)
            return (yf - w * coeff * r).to(y.dtype)
        else:
            # subspace: r is (k, d) orthonormal rows;
            # y - w U^T(U y) == (W - w U^T U W) x
            coeff = yf @ r.T  # (..., k)
            return (yf - w * (coeff @ r)).to(y.dtype)

    def install(self):
        if self._wrapped:
            return
        for blk in self.blocks:
            L = blk.layer_idx
            if getattr(blk, "attn", None) is not None:
                orig = blk.attn.forward

                def attn_fwd(x, params, out_dtype=None, _o=orig, _L=L, _s=self):
                    return _s._ablate(_o(x, params, out_dtype), _L, "attn")

                blk.attn.forward = attn_fwd
            if getattr(blk, "mlp", None) is not None:
                orig = blk.mlp.forward

                def mlp_fwd(x, params, out_dtype=None, _o=orig, _L=L, _s=self):
                    return _s._ablate(_o(x, params, out_dtype), _L, "mlp")

                blk.mlp.forward = mlp_fwd
        self._wrapped = True

    def set_edit(self, edit):
        import numpy as np
        import torch
        self.attn_w = [float(v) for v in edit["attn_w"]]
        self.mlp_w = [float(v) for v in edit["mlp_w"]]
        self.r = torch.as_tensor(np.asarray(edit["r"], dtype=np.float32))
        assert self.r.shape == (self.n_layers, self.hidden) or \
            (self.r.dim() == 3 and self.r.shape[0] == self.n_layers
             and self.r.shape[2] == self.hidden), tuple(self.r.shape)
        self._r_dev = {}


# ------------------------------------------------------------------ CPU proof
def prove_equivalence():
    import numpy as np
    rng = np.random.default_rng(0)
    # 1. hook == explicit weight edit, fp32, small layer (out=hidden toy dim)
    for trial in range(5):
        out_dim, in_dim, bsz = 32, 16, 7
        W = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        X = rng.standard_normal((in_dim, bsz)).astype(np.float32)
        r = rng.standard_normal(out_dim).astype(np.float32)
        r /= np.linalg.norm(r)
        w = float(rng.uniform(0.5, 1.5))
        Y = W @ X
        coef = (r[None, :] @ Y)  # (1, bsz): r_hat^T y per column
        Y_hook = Y - w * (coef * r[:, None])  # y - w (y.r) r, broadcast over rows
        W_edit = W - w * np.outer(r, r @ W)
        Y_edit = W_edit @ X
        err = float(np.abs(Y_hook - Y_edit).max())
        assert err < 1e-5, f"trial {trial}: hook vs weight-edit maxabs {err}"
    print(f"PROOF-1 hook==weight-edit: 5/5 random fp32 layers, maxabs < 1e-5")
    # 2. summed-writer linearity: editing each expert == hooking the sum
    E, d_out, d_in = 8, 24, 12
    Ws = rng.standard_normal((E, d_out, d_in)).astype(np.float32)
    xs = rng.standard_normal((E, d_in, 5)).astype(np.float32)
    r = rng.standard_normal(d_out).astype(np.float32)
    r /= np.linalg.norm(r)
    w = 1.2
    Y_sum = sum(Ws[e] @ xs[e] for e in range(E))
    coef = r[None, :] @ Y_sum
    Y_hooked_sum = Y_sum - w * (coef * r[:, None])
    Y_per_expert = sum((Ws[e] - w * np.outer(r, r @ Ws[e])) @ xs[e] for e in range(E))
    err = float(np.abs(Y_hooked_sum - Y_per_expert).max())
    assert err < 1e-5, f"summed-writer maxabs {err}"
    print(f"PROOF-2 block-sum hook==per-expert edits: maxabs {err:.2e}")
    # 3. kernel endpoints: peak->max_w, edge->min_w, outside->0
    k = heretic_kernel(45, 1.3, 30.0, 0.2, 6.0)
    assert abs(k[30] - 1.3) < 1e-12, k[30]
    assert abs(k[36] - 0.2) < 1e-12 and abs(k[24] - 0.2) < 1e-12, (k[24], k[36])
    assert k[23] == 0.0 and k[37] == 0.0 and k[0] == 0.0
    assert abs(k[33] - (1.3 + 0.5 * (0.2 - 1.3))) < 1e-12, k[33]
    print("PROOF-3 kernel endpoints/linearity OK (peak=1.3, edge=0.2, outside=0)")
    # 4. direction interp endpoints: int index == that layer; perlayer passthrough
    R = rng.standard_normal((45, 64))
    v = interp_direction(R, 29.0)
    assert float(np.dot(v, R[29] / np.linalg.norm(R[29]))) > 1 - 1e-12
    v = interp_direction(R, 29.5)
    mid = (R[29] + R[30]) / 2
    mid /= np.linalg.norm(mid)
    assert float(np.dot(v, mid)) > 1 - 1e-9
    e = build_edit(R, "perlayer", 0.0,
                   {"max_w": 1, "peak": 30, "min_w": 0, "width": 5},
                   {"max_w": 1, "peak": 30, "min_w": 0, "width": 5})
    assert e["r"].shape == (45, 64) and abs(np.linalg.norm(e["r"][7]) - 1) < 1e-9
    print("PROOF-4 direction interp endpoints OK (f=29 -> L29, f=29.5 -> midpoint)")
    # 5. subspace hook == explicit subspace weight edit, fp32
    d_out, d_in, k = 24, 12, 3
    W = rng.standard_normal((d_out, d_in)).astype(np.float32)
    X = rng.standard_normal((d_in, 5)).astype(np.float32)
    U, _ = np.linalg.qr(rng.standard_normal((d_out, k)).astype(np.float32))
    U = U.T.copy()  # (k, d) orthonormal rows
    w = 2.2  # widened range reaches 3.0
    Y = W @ X
    Y_hook = Y - w * (U.T @ (U @ Y))
    W_edit = W - w * (U.T @ (U @ W))
    Y_edit = W_edit @ X
    err = float(np.abs(Y_hook - Y_edit).max())
    assert err < 1e-5, f"subspace hook vs weight-edit maxabs {err}"
    print(f"PROOF-5 subspace hook==weight-edit (k=3, w=2.2): maxabs {err:.2e}")
    # 6. projected direction: unit norm + orthogonal to harmless mean
    Rm = rng.standard_normal((45, 64))
    mu = rng.standard_normal(64)
    vp = projected_direction(Rm, 29.34, mu)
    assert abs(np.linalg.norm(vp) - 1) < 1e-9, np.linalg.norm(vp)
    assert abs(float(vp @ (mu / np.linalg.norm(mu)))) < 1e-9
    print("PROOF-6 projected direction unit + orthogonal to mu_h OK")
    # 7. PCA recovers a planted 2-dim subspace
    n, d = 300, 64
    B, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    B = B.T  # (2, d) planted rows
    C = rng.standard_normal((n, 2)) * [5.0, 3.0]
    D = C @ B + rng.standard_normal((n, d)) * 0.3
    U2, expl = pca_subspace(D, 2)
    assert U2.shape == (2, d) and len(expl) == 2 and expl[0] > 0.5 and expl[1] > 0.15
    # subspace overlap: mean canonical correlation with planted B
    M = U2 @ B.T
    s = np.linalg.svd(M, compute_uv=False)
    assert float(s.min()) > 0.99, s
    print(f"PROOF-7 PCA recovers planted 2-dim subspace (min canon corr {s.min():.4f})")
    # 8. MiMo-sized geometry (48 layers): kernel + builders, every block index covered, MTP idx >= 48 absent
    k = heretic_kernel(48, 1.3, 33.0, 0.2, 9.0)
    assert len(k) == 48 and abs(k[33] - 1.3) < 1e-12 and k[23] == 0.0 and k[43] == 0.0
    R48 = rng.standard_normal((48, 64))
    e = build_edit(R48, "perlayer", 0.0,
                   {"max_w": 1, "peak": 33, "min_w": 0, "width": 9},
                   {"max_w": 1, "peak": 33, "min_w": 0, "width": 9})
    assert e["r"].shape == (48, 64) and len(e["attn_w"]) == 48 and len(e["mlp_w"]) == 48
    e = build_single_edit(R48[31], {"max_w": 1, "peak": 33, "min_w": 0, "width": 9},
                          {"max_w": 1, "peak": 33, "min_w": 0, "width": 9}, n_layers=48)
    assert e["r"].shape == (48, 64)
    print("PROOF-8 48-layer (MiMo) kernel/builders OK")
    print("ALL CPU PROOFS PASS (fp32, no model, no GPU)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prove", action="store_true")
    a = ap.parse_args()
    if a.prove:
        prove_equivalence()
    else:
        ap.error("nothing to do (use --prove)")
