"""GPU Engram for hy4_forward (flag HY4_ENGRAM_GPU=1, patched in by hy4/patch_engram.py).

Same maths as engram_rows.Hy4Engram.apply, but nothing of the hidden state leaves the device:
  * the big row table stays on the CPU side (pread of the DISTINCT needed rows only, threaded, one pinned staging buffer),
  * the raw bytes are copied to the device once per call and decoded there (E4M3 [* E8M0 scale] table gather, exact),
  * engram_kv stays as stored bytes (QLinear, GEMV kernel for small T) when `lin` gives one, else a dense matrix on the device,
  * norms / gate / signal are torch ops on the device (no x.cpu(), no .to(dev) round trip).
The hash (integer, token ids only) stays on the CPU: it never touches activations.
Generic: config comes from the GGUF header keys, the row form from F8Rows.layout; nothing keyed on a model name.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from .engram_rows import Hy4Engram, F8Rows, load_dense, rms, E4M3, E8M0, QB

_POOL = ThreadPoolExecutor(max_workers=16)
_TABS: dict = {}


def _tabs(dev):
    k = str(dev)
    if k not in _TABS:
        _TABS[k] = (E4M3.to(dev), E8M0.to(dev))
    return _TABS[k]


def fetch_rows(table: F8Rows, ids, dev) -> torch.Tensor:
    """ids (any int shape) -> float32 [*shape, width] on `dev`. Distinct rows are pread once (threaded), copied once, decoded on `dev`."""
    ids = np.asarray(torch.as_tensor(ids, dtype=torch.int64).numpy())
    uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
    rb = table.row_bytes
    buf = torch.empty(len(uniq), rb, dtype=torch.uint8).pin_memory() if torch.device(dev).type == "cuda" else torch.empty(len(uniq), rb, dtype=torch.uint8)
    view = buf.numpy()

    def one(a):
        j, r = a
        view[j] = table.raw_row(int(r))
    if len(uniq) > 8:
        list(_POOL.map(one, enumerate(uniq)))
    else:
        for a in enumerate(uniq):
            one(a)
    u = buf.to(dev, non_blocking=True)
    t4, t8 = _tabs(dev)
    w = table.width
    if table.layout == "plain":
        rows = t4[u.long()]
    else:                                                    # width e4m3 values, then width/32 e8m0 scales (same as F8Rows.decode_raw)
        rows = (t4[u[:, :w].long()].reshape(len(uniq), -1, QB) * t8[u[:, w:].long()][:, :, None]).reshape(len(uniq), w)
    sel = torch.from_numpy(inv.astype(np.int64)).to(dev)
    return rows[sel].reshape(*ids.shape, w)


class Hy4EngramGPU:
    def __init__(self, g, layer: int, dev, table: F8Rows | None = None, lin=None, eps: float = 1.1920929e-07):
        m = g.meta
        p = m["general.architecture"] + "."
        pl = list(m[p + "ple.layers"])
        self.layer, self.e, self.dev, self.eps = layer, pl.index(layer), torch.device(dev), eps
        self.n_gram, self.n_head = m[p + "ple.ngram_size"], m[p + "ple.heads_per_ngram"]
        self.n_cols = (self.n_gram - 1) * self.n_head
        e, nc, ng = self.e, self.n_cols, self.n_gram
        self.mult = list(m[p + "ple.layer_multipliers"])[e * ng:(e + 1) * ng]
        self.primes = list(m[p + "ple.head_vocab_sizes"])[e * nc:(e + 1) * nc]
        self.offs = list(m[p + "ple.head_offsets"])[e * nc:(e + 1) * nc]
        self.token_map, self.pad_id = m[p + "engram.token_map"], m[p + "engram.pad_token_id"]
        self.head_dim = m[p + "embedding_length_per_layer_input"]
        self.n_embd, self.hc = m[p + "embedding_length"], m[p + "hyper_connection.count"]
        b = f"blk.{layer}."
        f = lambda n: load_dense(g, b + n).to(self.dev)
        kv = lin(b + "engram_kv.weight") if lin is not None else load_dense(g, b + "engram_kv.weight")
        self.kv = kv if not torch.is_tensor(kv) else kv.to(self.dev)
        self.kv_b, self.lookup_norm, self.gate = f("engram_kv_b.weight"), f("engram_lookup_norm.weight"), f("engram_gate.weight")
        self.hc_scale, self.pre_gamma = f("engram_hc_scale.weight"), f("engram_pre_gamma.weight")
        self.table = table if table is not None else F8Rows(g)
        assert self.table.width == self.head_dim

    row_ids = Hy4Engram.row_ids

    def _kv(self, r):
        return self.kv(r) if callable(self.kv) else r @ self.kv.T

    def apply(self, x, rows, valid):
        T, eps = x.shape[0], self.eps
        r = rms(rows, eps) * self.lookup_norm * valid[:, :, None]
        r = r.reshape(T, -1)
        kv = self._kv(r).reshape(T, self.hc + 1, self.n_embd) + self.kv_b
        key, value = kv[:, :self.hc], kv[:, self.hc]
        dot = (rms(x, eps) * rms(key, eps) * self.gate).sum(-1) / self.n_embd ** 0.5
        mag = dot.abs().clamp_min(1e-6).sqrt()
        sign = 1.0 - 2.0 * (-dot > 0).float()
        gate = torch.sigmoid(mag * sign)
        signal = rms(value, eps) * self.pre_gamma
        delta = signal[:, None, :] * (gate[:, :, None] + self.hc_scale)
        anyv = (valid.sum(-1) > 0).float()
        return x + delta * anyv[:, None, None]

    def __call__(self, x: torch.Tensor, tokens, start: int = 0) -> torch.Tensor:
        ids, valid = self.row_ids(tokens, start, x.shape[0])
        rows = fetch_rows(self.table, ids, self.dev)
        return self.apply(x, rows, valid.to(self.dev, non_blocking=True))
