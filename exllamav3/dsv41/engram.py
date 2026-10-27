"""
Engram n-gram memory of DeepSeek-V4.1.

hash layer   : token ids -> compressed ids -> rolling XOR of (id * multiplier) -> % prime + offset
                (reference inference/engram.py; parity with the antirez GGUF primes/multipliers is tested)
row store    : (layer, row) -> 264 bytes = 256 e4m3 + 8 ue8m0 scales (32 elems each). Rows are read on
                demand; the on-disk stores use O_DIRECT, never mmap.
forward      : gated n-gram injection into the hc_mult residual streams (model.py:328)
"""
from __future__ import annotations
import os
import mmap
import subprocess
from dataclasses import dataclass
import numpy as np
import torch

ROW_VALUES = 256
ROW_SCALES = 8
ROW_BYTES = ROW_VALUES + ROW_SCALES


# -------------------------------------------------------------------------------------- primes
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin (exact for n < 3.3e24)."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple
    num_embeddings: tuple
    primes: tuple          # [layer][ngram-1][head]
    n_heads: int
    head_dim: int

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def build(cls, layer_ids, num_embeddings, max_ngram_size, vocab_size, n_heads, head_dim, primes=None):
        """primes=None derives them like the reference (one run of primes above vocab_size per
        (layer, ngram), never reused); pass a [layer][ngram][head] nest to use stored ones (GGUF)."""
        layer_ids = tuple(layer_ids)
        if primes is None:
            primes, seen = [], set()
            for _ in layer_ids:
                per = []
                for _ in range(max_ngram_size - 1):
                    cur, sizes = vocab_size - 1, []
                    for _ in range(n_heads):
                        cur = find_next_prime(cur, seen)
                        seen.add(cur)
                        sizes.append(cur)
                    per.append(tuple(sizes))
                primes.append(tuple(per))
        primes = tuple(tuple(tuple(int(p) for p in h) for h in l) for l in primes)
        if not num_embeddings:
            num_embeddings = [sum(p for ng in l for p in ng) for l in primes]
        return cls(max_ngram_size, layer_ids, tuple(int(n) for n in num_embeddings), primes, n_heads, head_dim)

    def offsets(self, layer_pos: int):
        flat = [p for ng in self.primes[layer_pos] for p in ng]
        return np.concatenate([[0], np.cumsum(flat[:-1])]).astype(np.int64)


def compute_multipliers(layer_ids, max_ngram_size: int, compressed_vocab: int) -> np.ndarray:
    """[layer, max_ngram] odd int64 multipliers (per-layer numpy PCG64 stream seeded 10007*layer_id)."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rows = []
    for lid in layer_ids:
        g = np.random.default_rng(10007 * lid)
        v = g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(v * 2 + 1)
    return np.stack(rows)


def build_compressed_token_map(tokenizer):
    """token id -> compressed id; tokens that normalise alike (NFKC/NFD, strip accents, lowercase,
    collapse whitespace) share an id. Returns (list, n_ids). `tokenizer` needs .backend_tokenizer and len()."""
    from tokenizers import Regex, normalizers
    sentinel = "\ue000"
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    be = tokenizer.backend_tokenizer
    key_to_id, out = {}, [0] * len(tokenizer)
    for tid in range(len(tokenizer)):
        text = be.decode([tid], skip_special_tokens=False)
        if "\ufffd" in text:
            key = be.id_to_token(tid)
        else:
            n = norm.normalize_str(text)
            key = n if n else text
        nid = key_to_id.get(key)
        if nid is None:
            nid = len(key_to_id)
            key_to_id[key] = nid
        out[tid] = nid
    return out, len(key_to_id)


DEAD = -1


class NgramHasher:
    """Maps positions to hash ids of the n-grams ending there. Holds the compressed-id history so
    prefill/decode splits and chunking give the same ids as one pass."""

    def __init__(self, layout: EngramLayout, token_map, pad_token_id: int, compressed_vocab: int,
                 multipliers=None, max_batch: int = 1, max_seq: int = 4096, device="cpu"):
        self.layout = layout
        self.device = device
        tm = torch.as_tensor(token_map, dtype=torch.int64)
        self.token_map = tm.to(device)
        self.pad = int(tm[pad_token_id])
        nl, ng = len(layout.layer_ids), layout.max_ngram_size
        if multipliers is None:
            multipliers = compute_multipliers(layout.layer_ids, ng, compressed_vocab)
        self.mult = torch.as_tensor(np.asarray(multipliers, dtype=np.int64), device=device)  # [nl, ng]
        self.primes = torch.tensor(layout.primes, dtype=torch.int64, device=device)           # [nl, ng-1, nh]
        self.offsets = torch.tensor(np.stack([layout.offsets(i) for i in range(nl)]),
                                    dtype=torch.int64, device=device)                        # [nl, (ng-1)*nh]
        self.cache = torch.full((max_batch, max_seq), DEAD, dtype=torch.int64, device=device)

    def reset(self):
        self.cache.fill_(DEAD)

    def __call__(self, input_ids: torch.Tensor, start_pos: int, token_mask=None) -> torch.Tensor:
        """-> [B, L, n_layers, n_hash_cols] int64 table rows."""
        B, L = input_ids.shape
        ng = self.layout.max_ngram_size
        comp = self.token_map[input_ids]
        if token_mask is not None:
            comp = torch.where(token_mask, comp, DEAD)
        self.cache[:B, start_pos:start_pos + L] = comp
        pos = torch.arange(start_pos, start_pos + L, device=input_ids.device).expand(B, L)
        toks, blocked = [], torch.zeros_like(pos, dtype=torch.bool)
        for sh in range(ng):
            src = self.cache[:B].gather(1, (pos - sh).clamp_min(0))
            blocked = blocked | (pos < sh) | (src == DEAD)
            toks.append(torch.where(blocked, self.pad, src))
        toks = torch.stack(toks, dim=-1)                                 # [B,L,ng]
        prod = toks.unsqueeze(2) * self.mult                              # [B,L,nl,ng]
        rolling, hashes = prod[..., 0], []
        for i in range(1, ng):
            rolling = torch.bitwise_xor(rolling, prod[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


# -------------------------------------------------------------------------------------- row stores
class EngramRowStore:
    """Interface: fetch(layer_pos, rows int64 [n]) -> uint8 [n, 264] (256 e4m3 bytes + 8 e8m0 bytes)."""

    def fetch(self, layer_pos: int, rows: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def num_rows(self, layer_pos: int) -> int:  # pragma: no cover
        raise NotImplementedError


class MemoryEngramStore(EngramRowStore):
    """Tables held as tensors: values [N,256] (fp8/uint8) + scales [N,8] (e8m0/uint8)."""

    def __init__(self, values, scales, fmt: str = "fp8"):
        self.fmt = fmt
        self.v = [t.view(torch.uint8) for t in values]
        self.s = [t.view(torch.uint8) for t in scales]

    def num_rows(self, layer_pos):
        return self.v[layer_pos].size(0)

    def fetch(self, layer_pos, rows):
        r = rows.to(self.v[layer_pos].device)
        return torch.cat([self.v[layer_pos][r], self.s[layer_pos][r]], dim=-1)


_SECTOR = 4096


class DirectRowReader:
    """Random 'row' reads from files with O_DIRECT (page cache bypassed, no mmap).

    A row is described by `segments` = [(path, base_offset, stride, nbytes), ...]; its bytes are the
    concatenation of file[base + i*stride : +nbytes] over the segments (official layout: values and
    scales are two separate tensors, possibly in different shards; GGUF: one interleaved 264-byte segment).
    Each read is widened to aligned 4 KiB sectors into an aligned anonymous buffer; the last window stays
    valid, so rows sharing a sector cost one read.
    O_DIRECT is mandatory unless allow_buffered=True (filesystems without it), in which case
    POSIX_FADV_DONTNEED is issued after each batch instead.
    """

    def __init__(self, segments, allow_buffered: bool = False, buf_bytes: int = 1 << 20):
        self.segments = [(str(s[0]),) + tuple(int(v) for v in s[1:]) for s in segments]
        self.row_bytes = sum(s[3] for s in self.segments)
        self.direct = True
        self.fds = {}
        for path in dict.fromkeys(s[0] for s in self.segments):
            try:
                self.fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
            except OSError:
                if not allow_buffered:
                    raise
                self.fds[path] = os.open(path, os.O_RDONLY)
                self.direct = False
        self._buf = mmap.mmap(-1, max(buf_bytes, 2 * _SECTOR))      # anonymous, page aligned
        self.reads = 0
        self.bytes_read = 0
        self._win = None

    @property
    def fd(self):
        return next(iter(self.fds.values()))

    def close(self):
        try:
            self._buf.close()
            for fd in self.fds.values():
                os.close(fd)
        except OSError:
            pass
        self.fds = {}

    def _pread(self, path: str, off: int, n: int) -> memoryview:
        """Aligned read covering [off, off+n); returns a view of exactly those bytes. The last window
        stays valid in the buffer, so rows sharing a 4 KiB sector cost one read."""
        a = off // _SECTOR * _SECTOR
        e = -(-(off + n) // _SECTOR) * _SECTOR
        if self._win is not None and self._win[0] == path and self._win[1] <= off and off + n <= self._win[2]:
            base = self._win[1]
        else:
            if e - a > len(self._buf):
                self._buf.close()
                self._buf = mmap.mmap(-1, e - a)
            want = e - a
            self._win = None
            got = os.preadv(self.fds[path], [memoryview(self._buf)[:want]], a)
            self.reads += 1
            self.bytes_read += want
            if got < off + n - a:
                raise IOError(f"short read at {off}: got {got}, need {off + n - a}")
            self._win = (path, a, a + got)         # bytes of the buffer that are valid
            base = a
        return memoryview(self._buf)[off - base: off - base + n]

    def read_rows(self, rows) -> torch.Tensor:
        rows = [int(r) for r in rows]
        out = torch.empty(len(rows), self.row_bytes, dtype=torch.uint8)
        order = sorted(range(len(rows)), key=lambda i: rows[i])
        for i in order:
            c = 0
            for path, base, stride, nb in self.segments:
                mv = self._pread(path, base + rows[i] * stride, nb)
                out[i, c:c + nb] = torch.frombuffer(bytearray(mv), dtype=torch.uint8)
                c += nb
        if not self.direct:
            for fd in self.fds.values():
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
        return out


def make_reader(segments, allow_buffered: bool = False, kind: str = None):
    """Row reader for `segments` (see DirectRowReader). kind (default $EXL3_DSV41_ENGRAM_READER, else "auto"):
    "native"   C++ thread pool, O_DIRECT, no GIL (modules/dsv41_fn/engram_io.NativeRowReader; needs g++)
    "parallel" Python thread pool of O_DIRECT preadv calls (ParallelRowReader)
    "direct"   one O_DIRECT read at a time (DirectRowReader, no dependencies)
    "auto"     native, else parallel, else direct. Every reader returns the same bytes. Without O_DIRECT support on the
               filesystem the readers fall back to buffered reads when `allow_buffered`."""
    kind = kind or os.environ.get("EXL3_DSV41_ENGRAM_READER", "auto")
    order = {"auto": ("native", "parallel", "direct"), "native": ("native", "direct"),
             "parallel": ("parallel", "direct"), "direct": ("direct",)}[kind]
    err = None
    for k in order:
        try:
            if k == "direct":
                return DirectRowReader(segments, allow_buffered=allow_buffered)
            from ..modules.dsv41_fn.engram_io import NativeRowReader, ParallelRowReader
            if k == "native":
                th = int(os.environ.get("EXL3_DSV41_ENGRAM_THREADS", "0"))
                return NativeRowReader(segments, allow_buffered=allow_buffered, **({"threads": th} if th else {}))
            return ParallelRowReader(segments, allow_buffered=allow_buffered)
        except (ImportError, OSError, subprocess.SubprocessError) as ex:     # no compiler / no O_DIRECT: next reader
            err = ex
            if k == "direct":
                raise
    raise err


class FileEngramStore(EngramRowStore):
    """One DirectRowReader per engram layer."""

    def __init__(self, readers, n_rows, fmt: str = "fp8"):
        self.fmt = fmt
        self.readers = list(readers)
        self.n = list(n_rows)

    def num_rows(self, layer_pos):
        return self.n[layer_pos]

    def fetch(self, layer_pos, rows):
        assert int(rows.max()) < self.n[layer_pos] and int(rows.min()) >= 0, "engram row out of range"
        return self.readers[layer_pos].read_rows(rows.tolist())

    def close(self):
        for r in self.readers:
            r.close()


class PrefetchRowStore(EngramRowStore):
    """Overlaps the row reads with compute. `prefetch(layer_pos, rows)` queues the read on a background thread and returns
    at once; `fetch(layer_pos, rows)` (called by `engram_inject` at the engram layer) hands back that result when the rows
    are the same ones, else reads synchronously. One reader thread (the underlying readers are not shared between
    threads); the reads release the GIL, so the model keeps computing the layers in front of the engram layer."""

    def __init__(self, store: EngramRowStore, max_pending: int = 16):
        from concurrent.futures import ThreadPoolExecutor
        self.store = store
        self.fmt = getattr(store, "fmt", "fp8")
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-prefetch")
        self.pending = {}
        self.max_pending = max_pending
        self.hits = self.misses = 0

    def num_rows(self, layer_pos):
        return self.store.num_rows(layer_pos)

    @staticmethod
    def _key(layer_pos, rows):
        return layer_pos, rows.numel(), hash(rows.numpy().tobytes())

    def prefetch(self, layer_pos: int, rows: torch.Tensor):
        rows = rows.detach().to("cpu", torch.int64).contiguous()
        k = self._key(layer_pos, rows)
        if k in self.pending:
            return
        while len(self.pending) >= self.max_pending:                  # stale entries (rows nobody fetched)
            self.pending.pop(next(iter(self.pending))).cancel()
        self.pending[k] = self.pool.submit(self.store.fetch, layer_pos, rows)

    def fetch(self, layer_pos, rows):
        r = rows.detach().to("cpu", torch.int64).contiguous()
        fut = self.pending.pop(self._key(layer_pos, r), None)
        if fut is not None:
            self.hits += 1
            return fut.result()
        self.misses += 1
        return self.pool.submit(self.store.fetch, layer_pos, r).result()     # same thread as the prefetches

    def close(self):
        self.pool.shutdown(wait=True)
        getattr(self.store, "close", lambda: None)()


E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_rows(raw: torch.Tensor, dtype=torch.float32, fmt: str = "fp8") -> torch.Tensor:
    """uint8 rows -> [..., hd] values. fmt "fp8": [hd + hd/32] bytes (e4m3 values then e8m0 per 32), hd=256 -> 264.
    fmt "mxfp4": [hd/2 + hd/32] bytes (two e2m1 per byte, low nibble first, then e8m0 per 32), hd=256 -> 136."""
    if fmt == "mxfp4":
        nb = raw.size(-1) * 16 // 17                    # packed value bytes = hd / 2
        u = raw[..., :nb].contiguous()
        lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=raw.device)
        v = torch.stack((lut[(u & 15).long()], lut[(u >> 4).long()]), dim=-1).flatten(-2)
    else:
        nv = raw.size(-1) * 32 // 33                    # head_dim values + head_dim/32 scale bytes
        v = raw[..., :nv].contiguous().view(torch.float8_e4m3fn).float()
        nb = nv
    s = raw[..., nb:].contiguous().view(torch.float8_e8m0fnu).float()
    out = v.unflatten(-1, (-1, 32)) * s.unsqueeze(-1)
    return out.flatten(-2).to(dtype)


def detect_engram_format(directory: str, head_dim: int = 256) -> str:
    """"safetensors" (official shards, tensors read in place; fmt follows their dtype/shape), "mxfp4" (plain .bin files,
    the transcode), or "fp8" for plain .bin files of e4m3 rows (a `.meta` with dtype F8_E4M3 or a 1:1 byte/scale ratio)."""
    import glob
    if glob.glob(os.path.join(directory, "*.safetensors")):
        return "safetensors"
    ws = sorted(glob.glob(os.path.join(directory, "layers.*.engram.embed.weight.bin")))
    if not ws:
        raise FileNotFoundError(f"no Engram tables (*.safetensors or layers.N.engram.embed.weight.bin) in {directory}")
    w = ws[0]
    sc = w.replace(".weight.bin", ".scale.bin")
    n = os.path.getsize(sc) // (head_dim // 32)
    return "fp8" if os.path.getsize(w) == n * head_dim else "mxfp4"


class RawBinEngramStore(FileEngramStore):
    """Engram tables kept as plain files, one pair per layer: `layers.N.engram.embed.weight.bin` (rows of
    `row_values` bytes) and `layers.N.engram.embed.scale.bin` (8 e8m0 bytes per row), e.g. the MXFP4 transcode
    (128 + 8 bytes per row). Rows are read with O_DIRECT (DirectRowReader), never mmap."""

    @classmethod
    def open(cls, directory: str, layer_ids, head_dim: int = 256, fmt: str = "auto", allow_buffered: bool = False):
        """fmt "auto" (default, also None): the official FP8 shards (`*.safetensors`, read in place) when the directory has
        them, else the plain `.bin` files (MXFP4 unless fmt says fp8)."""
        if fmt in (None, "", "auto") or not os.path.exists(os.path.join(directory, f"layers.{layer_ids[0]}.engram.embed.weight.bin")):
            fmt = detect_engram_format(directory, head_dim)
        if fmt == "safetensors":
            from .safetensors_io import SafetensorsIndex, engram_store_from_index
            return engram_store_from_index(SafetensorsIndex(directory), layer_ids, allow_buffered)
        wb = head_dim // 2 if fmt == "mxfp4" else head_dim
        sb = head_dim // 32
        readers, rows = [], []
        for lid in layer_ids:
            w = os.path.join(directory, f"layers.{lid}.engram.embed.weight.bin")
            s = os.path.join(directory, f"layers.{lid}.engram.embed.scale.bin")
            n = os.path.getsize(s) // sb
            assert os.path.getsize(w) == n * wb, (w, os.path.getsize(w), n, wb)
            readers.append(make_reader([(w, 0, wb, wb), (s, 0, sb, sb)], allow_buffered=allow_buffered))
            rows.append(n)
        return cls(readers, rows, fmt)


# -------------------------------------------------------------------------------------- forward
def engram_inject(h: torch.Tensor, hash_ids: torch.Tensor, store: EngramRowStore, layer_pos: int, wkv,
                  qk_weight: torch.Tensor, hc: int, dim: int, eps: float, token_mask=None,
                  clamp_value: float = 1e-6) -> torch.Tensor:
    """h [B,L,hc,dim] (residual streams) + hashes [B,L,n_cols] -> updated streams (model.py:350).
    wkv: callable [.., n_cols*256] -> [.., dim*(hc+1)] ; qk_weight = q_weight*k_weight [hc, dim]."""
    B, L, nc = hash_ids.shape
    flat = hash_ids.reshape(-1)
    uniq, inv = torch.unique(flat, return_inverse=True)
    rows = dequant_rows(store.fetch(layer_pos, uniq), h.dtype, getattr(store, "fmt", "fp8")).to(h.device)
    emb = rows[inv].view(B, L, nc, rows.size(-1))
    kv = wkv(emb.flatten(-2))
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc, dim))
    hf = h.float()
    rstd = torch.rsqrt(hf.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (hf * qk_weight.float() * key).sum(-1) * rstd * dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hf + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)
