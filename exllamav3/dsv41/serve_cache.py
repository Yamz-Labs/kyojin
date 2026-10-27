"""
Per-sequence cache of the DS4.1 serving model.

Layout (everything is recurrent-free: positions never need recomputation):

  win[l]            sliding-window ring [B, 128, 512] for EVERY layer (trunk 0..39 and the dspark stages 40..42);
                    position p lives in slot p % 128
  comp_kv[s]        compressed-KV pool [B, rows, 512] only for the kv SOURCE layers (2, 8, 14, 20); the layers that read a
                    shared pool (3..7, 9..13, ...) own nothing. Entries hold the fp4 fake-quant values (lossless fp16)
  idx_k[s]          indexer key pool [B, rows, 128], same source layers (the index SOURCE layers 24..36 only score the
                    keys of the nearest kv source, they own no keys)
  comp_kv_state[s]  open compressor group (kv / score rows of the positions of the unfinished group), ratio > 1 sources
  hasher            Engram n-gram history (compressed token ids)

Pools are paged: capacity grows in `page` rows, only up to what the context needs (`ensure(n_tokens)`), so an idle
or short session costs a few MiB instead of the full-context reservation. Rows of a pool are contiguous per sequence,
which is what the gather / top-k kernels read; growth reallocates and copies (amortised, the pools are small: ~0.4 GB at 128K).

The block journal of `DSV41State` (begin_block / rollback / end_block) is inherited unchanged: a verify block of
L <= window tokens can be rolled back to any accepted length without recomputation.
"""
from __future__ import annotations
import os
import torch
from .model import DSV41State, DSV41Model
from .engram import NgramHasher

PAGE_ROWS = 256


class PagedPool:
    """Growable row pool [B, capacity, width]; capacity is a multiple of `page` and never above `max_rows`."""

    def __init__(self, batch: int, width: int, dtype, device, max_rows: int, page: int = PAGE_ROWS, fill: float = 0.0):
        self.batch, self.width, self.dtype, self.device = batch, width, dtype, device
        self.max_rows, self.page, self.fill = max(int(max_rows), 1), page, fill
        self.t = torch.full((batch, 0, width), fill, dtype=dtype, device=device)

    @property
    def capacity(self) -> int:
        return self.t.size(1)

    def ensure(self, rows: int) -> torch.Tensor:
        """Capacity for at least `rows` rows (clamped to max_rows); returns the (possibly reallocated) tensor."""
        rows = min(int(rows), self.max_rows)
        if rows > self.capacity:
            cap = min(-(-rows // self.page) * self.page, self.max_rows)
            t = torch.full((self.batch, cap, self.width), self.fill, dtype=self.dtype, device=self.device)
            t[:, :self.capacity] = self.t
            self.t = t
        return self.t

    def nbytes(self) -> int:
        return self.t.numel() * self.t.element_size()


class DSV41Cache(DSV41State):
    """`DSV41State` with paged pools. Same attributes the graph reads (win, comp_kv, comp_kv_state, comp_score_state,
    idx_k, hasher, max_seq, journal), plus ensure / reset / fork / nbytes. `length` is the number of positions the
    cache has seen (advanced by the serving model)."""

    def __init__(self, model: DSV41Model, batch: int = 1, max_seq: int = 4096, page: int = PAGE_ROWS):
        cfg, dev, dt = model.cfg, model.device, model.dtype
        self.batch, self.max_seq, self.page = batch, max_seq, page
        self.journal = None
        self.length = 0
        n = cfg.num_hidden_layers
        W, hd = cfg.sliding_window, cfg.head_dim
        # pools and the ring hold fake-quant values: fp16 is lossless for them (and what the kernels read)
        pdt = dt if dt in (torch.float16, torch.float32) else torch.float16
        self.pool_dtype = pdt
        self.win = {l: torch.zeros(batch, W, hd, dtype=pdt, device=dev) for l in range(n + cfg.n_mtp_layers)}
        self._kv, self._ik = {}, {}
        self.comp_kv, self.idx_k = {}, {}
        self.packed = os.environ.get("DSV41_IDX_PACKED", "0") == "1"   # the packed copy exists only when the kernels read it
        self._iq, self._is = {}, {}                  # packed fp4 copy of the indexer keys (64 B + 4 B scales per key)
        self.idx_kq, self.idx_ks = {}, {}
        self.comp_kv_state, self.comp_score_state = {}, {}
        for l in cfg.kv_source_layer_ids:
            if l >= n:
                continue
            r = cfg.ratio(l)
            rows = max_seq // r
            self._kv[l] = PagedPool(batch, hd, pdt, dev, rows, page)
            self._ik[l] = PagedPool(batch, cfg.index_head_dim, pdt, dev, rows, page)
            self.comp_kv[l], self.idx_k[l] = self._kv[l].t, self._ik[l].t
            if self.packed:
                self._iq[l] = PagedPool(batch, cfg.index_head_dim // 2, torch.uint8, dev, rows, page)
                self._is[l] = PagedPool(batch, cfg.index_head_dim // 32, torch.uint8, dev, rows, page)
                self.idx_kq[l], self.idx_ks[l] = self._iq[l].t, self._is[l].t
            if r > 1:
                self.comp_kv_state[l] = torch.zeros(batch, r, hd, dtype=torch.float32, device=dev)
                self.comp_score_state[l] = torch.full((batch, r, hd), -torch.inf, dtype=torch.float32, device=dev)
        self.hasher = None
        if model.engram_layout is not None:
            self.hasher = NgramHasher(model.engram_layout, model.token_map, cfg.engram_pad_token_id,
                                      model.compressed_vocab, model.multipliers, batch, max_seq, dev)
        self._cfg = cfg

    # ------------------------------------------------------------------ growth
    def ensure(self, n_tokens: int):
        """Make the pools big enough for positions 0 .. n_tokens - 1."""
        if n_tokens > self.max_seq:
            raise ValueError(f"{n_tokens} tokens exceed the cache capacity of {self.max_seq}")
        for l, pool in self._kv.items():
            r = self._cfg.ratio(l)
            self.comp_kv[l] = pool.ensure(-(-n_tokens // r))
            self.idx_k[l] = self._ik[l].ensure(-(-n_tokens // r))
            if self.packed:
                self.idx_kq[l] = self._iq[l].ensure(-(-n_tokens // r))
                self.idx_ks[l] = self._is[l].ensure(-(-n_tokens // r))

    def repack_idx(self, step: int = 1 << 16):
        """Rebuild the packed fp4 copy of the indexer keys from the fp16 pools (after a bulk fill / restore)."""
        if not self.packed:
            return
        from ..modules.dsv41_fn.dsv41_idx import pack_keys
        for l in self._ik:
            t = self._ik[l].t
            for a in range(0, t.size(1), step):
                for b in range(t.size(0)):
                    kq, ks = pack_keys(t[b, a:a + step])
                    self._iq[l].t[b, a:a + step] = kq
                    self._is[l].t[b, a:a + step] = ks

    def reset(self):
        for t in self.win.values():
            t.zero_()
        for l in self._kv:
            self._kv[l].t.zero_()
            self._ik[l].t.zero_()
            self._iq[l].t.zero_()
            self._is[l].t.zero_()
        for l in self.comp_kv_state:
            self.comp_kv_state[l].zero_()
            self.comp_score_state[l].fill_(-torch.inf)
        if self.hasher is not None:
            self.hasher.reset()
        self.journal = None
        self.length = 0

    def nbytes(self) -> int:
        n = sum(t.numel() * t.element_size() for t in self.win.values())
        n += sum(p.nbytes() for p in self._kv.values()) + sum(p.nbytes() for p in self._ik.values()) + sum(p.nbytes() for p in self._iq.values()) + sum(p.nbytes() for p in self._is.values())
        n += sum(t.numel() * 4 for t in self.comp_kv_state.values()) * 2
        if self.hasher is not None:
            n += self.hasher.cache.numel() * 8
        return n

    def fork(self) -> "DSV41Cache":
        """Deep copy (prefix reuse: fork after the shared prompt was prefilled)."""
        c = object.__new__(DSV41Cache)
        c.__dict__.update({k: v for k, v in self.__dict__.items()
                           if k not in ("win", "_kv", "_ik", "_iq", "_is", "idx_kq", "idx_ks", "comp_kv", "idx_k", "comp_kv_state", "comp_score_state",
                                        "hasher", "journal")})
        c.journal = None
        c.win = {l: t.clone() for l, t in self.win.items()}
        c._kv, c._ik, c._iq, c._is = {}, {}, {}, {}
        for src, dst in ((self._kv, c._kv), (self._ik, c._ik), (self._iq, c._iq), (self._is, c._is)):
            for l, p in src.items():
                q = PagedPool(p.batch, p.width, p.dtype, p.device, p.max_rows, p.page, p.fill)
                q.t = p.t.clone()
                dst[l] = q
        c.comp_kv = {l: p.t for l, p in c._kv.items()}
        c.idx_k = {l: p.t for l, p in c._ik.items()}
        c.idx_kq = {l: p.t for l, p in c._iq.items()}
        c.idx_ks = {l: p.t for l, p in c._is.items()}
        c.comp_kv_state = {l: t.clone() for l, t in self.comp_kv_state.items()}
        c.comp_score_state = {l: t.clone() for l, t in self.comp_score_state.items()}
        c.hasher = None
        if self.hasher is not None:
            h = object.__new__(type(self.hasher))
            h.__dict__.update(self.hasher.__dict__)
            h.cache = self.hasher.cache.clone()
            c.hasher = h
        return c


def cache_bytes(cfg, tokens: int, batch: int = 1, itemsize: int = 2) -> int:
    """Planner: bytes of a DSV41Cache holding `tokens` positions (pools + rings + compressor state + engram history)."""
    n = cfg.num_hidden_layers
    total = (n + cfg.n_mtp_layers) * batch * cfg.sliding_window * cfg.head_dim * itemsize
    for l in cfg.kv_source_layer_ids:
        if l >= n:
            continue
        r = cfg.ratio(l)
        rows = -(-tokens // r)
        total += batch * rows * (cfg.head_dim + cfg.index_head_dim) * itemsize
        if os.environ.get("DSV41_IDX_PACKED", "0") == "1":
            total += batch * rows * (cfg.index_head_dim // 2 + cfg.index_head_dim // 32)
        if r > 1:
            total += batch * r * cfg.head_dim * 4 * 2
    if cfg.engram_layer_ids:
        total += batch * tokens * 8
    return total
