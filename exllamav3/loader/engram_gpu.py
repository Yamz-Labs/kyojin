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
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from .engram_rows import Hy4Engram, F8Rows, load_dense, rms, E4M3, E8M0, QB

_IO_THREADS = int(os.environ.get("HY4_ENGRAM_IO_THREADS", "96"))
_POOL = ThreadPoolExecutor(max_workers=_IO_THREADS)
_TABS: dict = {}


def _tabs(dev):
    k = str(dev)
    if k not in _TABS:
        _TABS[k] = (E4M3.to(dev), E8M0.to(dev))
    return _TABS[k]


_NAT = [None, False]          # [ctypes lib or None, tried]


def _native_lib():
    """HY4_ENGRAM_NATIVE_IO=1: the row reads of one fetch run in C threads (engram_io.c: pread per distinct row, no GIL). Built once with gcc into the cache dir; None = Python path."""
    if os.environ.get("HY4_ENGRAM_NATIVE_IO", "0") != "1":
        return None
    if _NAT[0] is not None or _NAT[1]:
        return _NAT[0]
    _NAT[1] = True
    import ctypes, hashlib, shutil, subprocess
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engram_io.c")
    try:
        tag = hashlib.sha1(open(src, "rb").read()).hexdigest()[:10]
        d = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "kyojin", "engram_io")
        so = os.path.join(d, f"libengram_io_{tag}.so")
        if not os.path.exists(so):
            os.makedirs(d, exist_ok=True)
            tmp = so + f".{os.getpid()}.tmp"
            subprocess.run([shutil.which("gcc") or "gcc", "-O2", "-shared", "-fPIC", "-pthread", src, "-o", tmp], check=True, capture_output=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.engram_read_rows.restype = ctypes.c_int
        lib.engram_read_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
        _NAT[0] = lib
    except Exception as ex:      # noqa: BLE001 - speed only: the Python reader below does the same reads
        print(f"engram native io unavailable ({type(ex).__name__}: {ex}); Python reader", flush=True)
    return _NAT[0]


def _read_rows_native(lib, table: F8Rows, uniq, view):
    g = table.g
    fds = np.ascontiguousarray(np.asarray(g.fds, dtype=np.int32))
    bases = np.ascontiguousarray(np.asarray(g.bases, dtype=np.int64))
    rows = np.ascontiguousarray(uniq, dtype=np.int64)
    nt = int(os.environ.get("HY4_ENGRAM_NATIVE_THREADS", "32"))
    rc = lib.engram_read_rows(fds.ctypes.data, bases.ctypes.data, rows.ctypes.data, len(rows), int(table.off), int(table.row_bytes), view.ctypes.data, nt)
    if rc != 0:
        raise IOError(f"{table.name}: native row read failed (rc {rc})")


def fetch_rows_host(table: F8Rows, ids):
    """CPU half of fetch_rows: distinct rows pread once (threaded) into one pinned buffer. -> (shape, uniq, inv, buf). Safe to run in a worker thread."""
    ids = np.asarray(torch.as_tensor(ids, dtype=torch.int64).numpy())
    uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
    rb = table.row_bytes
    buf = torch.empty(len(uniq), rb, dtype=torch.uint8).pin_memory() if torch.cuda.is_available() else torch.empty(len(uniq), rb, dtype=torch.uint8)
    view = buf.numpy()

    if len(uniq) and (int(uniq[0]) < 0 or int(uniq[-1]) >= table.rows):
        raise IndexError(f"{table.name}: row outside 0..{table.rows - 1}")
    lib = _native_lib() if len(uniq) > 8 else None
    if lib is not None:
        _read_rows_native(lib, table, uniq, view)
    elif len(uniq) > 8:
        # one task per slice of the sorted row list (a task per row cost more in future / GIL hand-offs than the read itself); preadv straight into the pinned buffer
        nw = _IO_THREADS
        step = (len(uniq) + nw - 1) // nw
        list(_POOL.map(lambda lo: _read_rows(table, uniq, view, lo, min(len(uniq), lo + step)), range(0, len(uniq), step)))
    else:
        _read_rows(table, uniq, view, 0, len(uniq))
    return ids.shape, uniq, inv, buf


def _read_rows(table: F8Rows, uniq, view, lo: int, hi: int):
    g, rb, off0 = table.g, table.row_bytes, table.off
    for j in range(lo, hi):
        o = off0 + int(uniq[j]) * rb
        fi, o = o >> 48, o & ((1 << 48) - 1)
        fd, pos = g.fds[fi], g.bases[fi] + o
        mv = memoryview(view[j])
        got = 0
        while got < rb:
            n = os.preadv(fd, [mv[got:]], pos + got)
            if n <= 0:
                raise IOError(f"short read at row {int(uniq[j])}")
            got += n


def rows_to_dev(table: F8Rows, host, dev) -> torch.Tensor:
    """device half of fetch_rows: one H2D copy + exact E4M3 [* E8M0] decode + gather -> float32 [*shape, width]"""
    shape, uniq, inv, buf = host
    u = buf.to(dev, non_blocking=True)
    t4, t8 = _tabs(dev)
    w = table.width
    if table.layout == "plain":
        rows = t4[u.long()]
    else:                                                    # width e4m3 values, then width/32 e8m0 scales (same as F8Rows.decode_raw)
        rows = (t4[u[:, :w].long()].reshape(len(uniq), -1, QB) * t8[u[:, w:].long()][:, :, None]).reshape(len(uniq), w)
    sel = torch.from_numpy(inv.astype(np.int64)).to(dev)
    return rows[sel].reshape(*shape, w)


def fetch_rows(table: F8Rows, ids, dev) -> torch.Tensor:
    """ids (any int shape) -> float32 [*shape, width] on `dev`. Distinct rows are pread once (threaded), copied once, decoded on `dev`."""
    return rows_to_dev(table, fetch_rows_host(table, ids), dev)


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

    def host_rows(self, tokens, start: int, n: int):
        """CPU part for positions start..start+n-1: hash + row pread. -> (valid, host) for rows_to_dev. Worker-thread safe (no CUDA compute)."""
        ids, valid = self.row_ids(tokens, start, n)
        return valid, fetch_rows_host(self.table, ids)

    def __call__(self, x: torch.Tensor, tokens, start: int = 0, pre=None) -> torch.Tensor:
        valid, host = pre if pre is not None else self.host_rows(tokens, start, x.shape[0])
        rows = rows_to_dev(self.table, host, self.dev)
        return self.apply(x, rows, valid.to(self.dev, non_blocking=True))
