"""Decensor edit hook for the Hy4 forward (spec = edit_spec.json + edit_spec.safetensors, loaded by load_spec; armed by HY4_ABLIT_SPEC in tools/hy4/serve.py).

Runtime formula at a writer output y of layer L and site s (0 = attention output, 1 = MLP / MoE block output), fp32:
    y' = y - w_{L,s} * ((r_L . y) - mu_{L,s}) * r_L          (w = 0: y is returned untouched, bit for bit)
Sites on the Hy4 stream (the values that are ADDED to the hyper-connection streams):
    attn : the MLA output projection `out` [T, D] (no norm follows it)
    mlp  : rms(f) * post_ffw_norm, i.e. the dense MLP / routed+shared MoE block output AFTER its post norm [T, D]
Layers >= n_layers (the MTP draft block, blk 42) are never edited. Spec fields gain.* / suhs.* (weight-side levers) are not supported
by the GGUF runtime: a spec that carries them is rejected.

Everything the kernels read lives in static device buffers (W, MU, R), so a HIP graph captured with the hook keeps working after
`set` / `load_spec` rewrites them in place. `sites[L][s]` is a python flag read at trace time (capture): only flagged sites run a kernel.
`ST is None` (default) = the forward is the unmodified one.
"""
import json
import os

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except Exception:               # no triton: the torch reference runs everywhere
    triton = None

ST = None                       # the active EditState (module global read by hy4_forward.layer)
MAX_D = 16384


def _p2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def edit_torch(y, r, w, mu):
    """reference: y [T, D] fp32, r [D], w / mu 0-dim tensors (any device); w == 0 returns y itself, values unchanged"""
    d = (y * r).sum(-1, keepdim=True)
    return torch.where(w == 0, y, y - w * (d - mu) * r)


if triton is not None:
    @triton.jit
    def _k_edit(y_ptr, r_ptr, w_ptr, mu_ptr, o_ptr, D, BD: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, BD)
        dm = d < D
        y = tl.load(y_ptr + row * D + d, mask=dm, other=0.0)
        r = tl.load(r_ptr + d, mask=dm, other=0.0)
        w = tl.load(w_ptr)
        mu = tl.load(mu_ptr)
        dot = tl.sum(y * r, axis=0)
        e = y - w * (dot - mu) * r
        tl.store(o_ptr + row * D + d, tl.where(w == 0.0, y, e), mask=dm)


def _usable(y):
    return triton is not None and (y.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1") and y.shape[-1] <= MAX_D


class EditState:
    def __init__(self, n_layers, D, device):
        self.n, self.D, self.dev = n_layers, D, torch.device(device)
        self.W = torch.zeros(n_layers, 2, dtype=torch.float32, device=self.dev)
        self.MU = torch.zeros(n_layers, 2, dtype=torch.float32, device=self.dev)
        self.R = torch.zeros(n_layers, D, dtype=torch.float32, device=self.dev)
        self.sites = [[False, False] for _ in range(n_layers)]
        self.pinned = False                 # True: every site stays flagged (search mode: weights change, the captured graph does not)

    def pin_all_sites(self):
        self.pinned = True
        self.sites = [[True, True] for _ in range(self.n)]

    def set(self, attn_w, mlp_w, attn_mu, mlp_mu, r):
        """in-place update; r is (D,) (one direction for every layer) or (n, D) (per layer)"""
        n = self.n
        w = np.stack([np.asarray(attn_w, np.float32)[:n], np.asarray(mlp_w, np.float32)[:n]], 1)
        mu = np.stack([np.asarray(attn_mu, np.float32)[:n], np.asarray(mlp_mu, np.float32)[:n]], 1)
        r = np.asarray(r, np.float32)
        r = np.broadcast_to(r, (n, self.D)) if r.ndim == 1 else r[:n]
        self.W.copy_(torch.from_numpy(np.ascontiguousarray(w)))
        self.MU.copy_(torch.from_numpy(np.ascontiguousarray(mu)))
        self.R.copy_(torch.from_numpy(np.ascontiguousarray(r)))
        if not self.pinned:
            self.sites = [[bool(w[L, 0] != 0), bool(w[L, 1] != 0)] for L in range(n)]

    def clear(self):
        self.W.zero_(); self.MU.zero_(); self.R.zero_()
        if not self.pinned:
            self.sites = [[False, False] for _ in range(self.n)]

    def args(self, i, s):
        return self.R[i], self.W.view(-1)[i * 2 + s:], self.MU.view(-1)[i * 2 + s:]

    def edit(self, y, i, s):
        """y [T, D] -> edited fp32 contiguous copy (the input is never modified)"""
        y = y.float().contiguous()
        r, w, mu = self.args(i, s)
        if _usable(y):
            o = torch.empty_like(y)
            _k_edit[(y.shape[0],)](y, r, w, mu, o, y.shape[1], BD=_p2(y.shape[1]), num_warps=8)
            return o
        return edit_torch(y, r, w[0], mu[0])


def install(n_layers, D, device):
    global ST
    if ST is None or ST.n != n_layers or ST.D != D or ST.dev != torch.device(device):
        ST = EditState(n_layers, D, device)
    return ST


def uninstall():
    global ST
    ST = None


def load_spec(path, device, n_layers=None, D=None, pin=False):
    """load edit_spec.json (+ .safetensors next to it) into the active state; -> the spec dict"""
    from safetensors.numpy import load_file
    spec = json.load(open(path))
    for k in ("hidden", "n_layers", "attn_w", "mlp_w"):
        if k not in spec:
            raise ValueError(f"spec has no field {k!r}")
    nl = int(spec["n_layers"])
    for k in ("attn_w", "mlp_w", "attn_mu", "mlp_mu"):
        if k in spec and (len(spec[k]) != nl or not np.isfinite(np.asarray(spec[k], np.float64)).all()):
            raise ValueError(f"spec field {k!r}: need {nl} finite values")
    t = load_file(os.path.splitext(path)[0] + ".safetensors")
    bad = [k for k in t if k.startswith(("gain.", "suhs."))]
    if bad:
        raise ValueError(f"spec carries weight-side levers ({len(bad)} tensors): not supported by the Hy4 GGUF runtime")
    if "r" not in t:
        raise ValueError("spec safetensors has no tensor 'r'")
    n = min(nl, n_layers or nl)
    D = D or spec["hidden"]
    if spec["hidden"] != D:
        raise ValueError(f"spec hidden {spec['hidden']} != model {D}")
    r = t["r"]
    if r.ndim == 2 and not spec.get("r_per_layer"):
        raise ValueError("a (k, hidden) subspace spec is not supported")
    if r.ndim not in (1, 2) or r.shape[-1] != D or (r.ndim == 2 and r.shape[0] < n) or not np.isfinite(r).all():
        raise ValueError(f"direction tensor 'r' has shape {tuple(r.shape)}: need ({D},) or (>= {n}, {D}), finite")
    st = install(n_layers or nl, D, device)
    if pin:
        st.pin_all_sites()
    z = [0.0] * nl
    st.set(spec["attn_w"], spec["mlp_w"], spec.get("attn_mu", z), spec.get("mlp_mu", z), r)
    return spec
