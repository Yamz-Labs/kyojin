"""HY4_GRAPH: the whole Hy4 decode step (T = 1) captured as ONE HIP graph per context bucket (night 18/11, perf/hy4-night-graph).

Why: the eager step is host / launch bound (~196 ms for ~4000 small launches; the GPU work itself is a fraction). One replay launches the
whole step with no Python and no per-kernel host cost. Everything that changes between tokens lives in STATIC device buffers:
  inp[0] token id, inp[1] position       written by the host before each replay (one H2D of 16 bytes)
  eng_raw / eng_valid                    the Engram rows of the 4 Engram layers for THIS token (host: hash + threaded pread, one H2D)
  K buffers [cap, 576] per layer         row `pos` written with index_copy_ (device index), attention reads the first Nb rows (bucket)
  indexer pool / ring                    pooled key of group pos // 4 written every step; groups not complete yet are masked (gend < pos)
The attention kernels see a fixed number of rows Nb (bucket = next multiple of HY4_GRAPH_BUCKET above pos + 1): rows past pos are masked
by the existing `kpos <= qpos` test, so the maths is the eager maths. One graph is captured lazily per bucket (shared memory pool).
Prefill (T > 1) stays eager. After the first graph step the eager indexer state is stale: reset (new sequence) before an eager step again.
Generic: shapes and config come from the Hy4Forward object; nothing is keyed on a model name.
"""
from __future__ import annotations
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import step_nosync as _ns
from .engram_rows import QB, E4M3, E8M0

BUCKET = int(os.environ.get("HY4_GRAPH_BUCKET", "512"))
MASK64 = (1 << 64) - 1
_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("HY4_GRAPH_IO", "16")))


# ---------------------------------------------------------------------------------------------------- static indexer
def _rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def indexer_static_init(ix, cap):
    """FusedKPoolIndexer -> fixed buffers: pool [cap // k, d], per-position (value | gate) rows vgb [cap, 2d] (open group filled from the eager state), rope table to cap"""
    k, d = ix.k, ix.d
    rows = cap // k
    if ix.pool.shape[0] < rows:
        big = torch.zeros(rows, d, device=ix.dev)
        big[:ix.P] = ix.pool[:ix.P]
        ix.pool = big
    vgb = getattr(ix, "vgb", None)             # per-position (value | gate) rows [cap, 2d]: a rejected / re-run position just overwrites its row
    if vgb is None:
        vgb = ix.vgb = torch.zeros(cap, 2 * d, device=ix.dev)
    if ix.vg is not None:                      # open group of an eager prefill (< k rows)
        m = ix.vg.shape[0]
        vgb[ix.n - m:ix.n] = ix.vg
    ix._max_pos = cap
    ix._table(cap)


@torch.no_grad()
def indexer_forward_static(ix, x, qr, pos, Nb, ar):
    """x [1, E], qr [1, R], pos [1] long (device), Nb rows -> allow [1, Nb] bool. Same maths and op order as FusedKPoolIndexer.forward for one
    token; no host value is read: group slot and rope position come from `pos`, the visible groups from a mask (gend < pos)."""
    k, d, H = ix.k, ix.d, ix.H
    ik = _rms(x @ ix.w["attn_k"].T, ix.w["k_norm"], ix.eps)
    vg_new = torch.cat([ik @ ix.w["value"].T, ik @ ix.w["gate"].T], -1)
    ix.vgb.index_copy_(0, pos, vg_new)
    g0 = (pos // k) * k
    rows = ix.vgb[g0[:, None] + ar[:k][None, :]]                       # [1, k, 2d]: the (partial) group of this position
    v, g = rows[..., :d].reshape(-1, k, d), rows[..., d:].reshape(-1, k, d)
    p = (torch.softmax(g + ix.w["ape"][None], dim=1) * v).sum(1)
    pooled = ix._rope(_rms(p, ix.w["pool_norm"], ix.eps), g0)
    ix.pool.index_copy_(0, pos // k, pooled)            # a group still open holds partial data here; it stays masked until it is complete
    iq = ix._rope((qr @ ix.w["q_b"].T).reshape(1, H, d), pos)
    wt = (x.float() @ ix.w["proj"].T) / math.sqrt(d * H)
    Pb = Nb // k
    kq = torch.relu(torch.einsum("nhd,pd->nhp", iq, ix.pool[:Pb]))
    sc = (kq * wt[:, :, None]).sum(1)
    gend = ar[:Pb] * k + (k - 1)
    vis = (gend[None] < pos[:, None]) if ix.tail else (gend[None] <= pos[:, None])
    sc = torch.where(vis, sc, torch.full_like(sc, -math.inf))
    idx = torch.topk(sc, min(ix.n_top, Pb), dim=1).indices
    gsel = torch.zeros(1, Pb, dtype=torch.bool, device=ix.dev)
    gsel.scatter_(1, idx, True)
    gsel &= vis
    allow = torch.zeros(1, Nb, dtype=torch.bool, device=ix.dev)
    allow[:, :Pb * k] = gsel.repeat_interleave(k, dim=1)
    lo = (pos - (pos % k)) if ix.tail else (pos - ((pos + 1) % k) + 1)
    j = ar[:Nb][None]
    allow |= (j >= lo[:, None]) & (j <= pos[:, None])
    return allow


# ---------------------------------------------------------------------------------------------------- Engram, host side
def engram_row_ids(e, tokens, p):
    """engram_rows.engram_hash for ONE position p (same integer maths), with the token map converted once -> (ids [n_cols] int, valid [n_cols] float)"""
    tm = getattr(e, "_tm", None)
    if tm is None:
        tm = e._tm = [int(t) for t in e.token_map]
        e._tpad = int(tm[e.pad_id])
        e._mult = [int(m) for m in e.mult]
        e._primes = [int(m) for m in e.primes]
        e._offs = [int(m) for m in e.offs]
    n_tm, pad = len(tm), e._tpad
    cid = lambda t: -1 if t < 0 or t >= n_tm else tm[t]
    c = cid(tokens[p])
    dead = c == -1
    toks = [pad if dead else c]
    for s in range(1, e.n_gram):
        c = -1 if p - s < 0 else cid(tokens[p - s])
        toks.append(pad if c == -1 else c)
    rolling = (toks[0] * e._mult[0]) & MASK64
    ids, valid = [0] * e.n_cols, [0.0 if dead else 1.0] * e.n_cols
    for o in range(1, e.n_gram):
        rolling ^= (toks[o] * e._mult[o]) & MASK64
        for h in range(e.n_head):
            col = (o - 1) * e.n_head + h
            ids[col] = rolling % e._primes[col] + e._offs[col]
    return ids, valid


# ---------------------------------------------------------------------------------------------------- static context seen by Hy4Forward.layer
class _GS:
    def __init__(self, fw, cap):
        self.fw, self.cap, self.Nb = fw, cap, 0
        dev = fw.dev
        self.inp = torch.zeros(2, dtype=torch.long, device=dev)         # [token, position]
        self.tok, self.pos = self.inp[0:1], self.inp[1:2]
        self.ar = torch.arange(cap, device=dev)
        self.eng_raw = self.eng_valid = None

    # -- Engram: rows are staged on the host before the replay, decoded + applied inside the graph
    def engram(self, fw, i, x):
        e = fw.eng[i]
        u = self.eng_raw[e.e]
        tab = e.table
        t4, t8 = _tabs(fw.dev)
        if tab.layout == "plain":
            rows = t4[u.long()]
        else:
            w = tab.width
            rows = (t4[u[:, :w].long()].reshape(u.shape[0], -1, QB) * t8[u[:, w:].long()][:, :, None]).reshape(u.shape[0], w)
        return e.apply(x, rows[None], self.eng_valid[e.e][None])

    def append_kv(self, i, kn):
        b = _ns.state(self.fw).kv[i]
        b.index_copy_(0, self.pos, kn)
        return b[:self.Nb]

    def indexer(self, fw, i, cur, qr, N):
        return indexer_forward_static(fw._indexer_fused(i), cur, qr, self.pos, N, self.ar)


_T = {}


def _tabs(dev):
    k = str(dev)
    if k not in _T:
        _T[k] = (E4M3.to(dev), E8M0.to(dev))
    return _T[k]


class GraphDecoder:
    def __init__(self, fw, cap=None):
        self.fw = fw
        self.st = _ns.state(fw)
        self.cap = cap or self.st.cap
        assert self.cap % BUCKET == 0 and self.cap % 4 == 0
        self.gs = _GS(fw, self.cap)
        self.graphs, self.pool = {}, torch.cuda.graph_pool_handle()
        self.prepared = False
        self.expect_n = None
        self.hin = torch.zeros(2, dtype=torch.long).pin_memory()
        self.stats = {"captures": 0, "replays": 0}
        self.ev = torch.cuda.Event()
        self.ev_set = False

    # -- one-time setup (eager): weights, K buffers, indexer state, Engram modules / staging
    def prepare(self):
        fw, gs, st = self.fw, self.gs, self.st
        assert fw.cache_weights and _ns.active(fw) and len(fw.res) == fw.n_layers - fw.n_dense, "graph mode needs cache_weights, HY4_NOSYNC=1 and every MoE layer resident"
        st.ensure(self.cap)
        if st.table is None:
            st.table = _ns.load_table(fw.g, "token_embd.weight", fw.dev)
        try:
            from . import attn_eager as _AE
            _AE.ensure(self.cap)
        except ImportError:
            pass
        for i in range(fw.n_layers):
            fw._layer(i)
            b = st.kv.get(i)
            if b is None or b.shape[0] < self.cap:
                nb = torch.zeros(self.cap, 512 + 64, dtype=torch.float32, device=fw.dev) if b is None else torch.zeros(self.cap, b.shape[1], dtype=b.dtype, device=fw.dev)
                if b is not None:
                    nb[:b.shape[0]] = b
                st.kv[i] = nb
        if os.environ.get("HY4_ENGRAM_GPU", "0") not in ("1", "2"):
            raise RuntimeError("graph mode needs HY4_ENGRAM_GPU=1")
        from .engram_gpu import Hy4EngramGPU
        for i in sorted(fw.ple):
            if i not in fw.eng:
                fw.eng[i] = Hy4EngramGPU(fw.g, i, fw.dev, table=fw.table, lin=None if os.environ.get("HY4_ENGRAM_GPU") == "2" else fw._lin)
        e0 = fw.eng[min(fw.ple)]
        nE, nc, rb = len(fw.ple), e0.n_cols, e0.table.row_bytes
        gs.eng_raw = torch.zeros(nE, nc, rb, dtype=torch.uint8, device=fw.dev)
        gs.eng_valid = torch.zeros(nE, nc, dtype=torch.float32, device=fw.dev)
        self.rpin = torch.zeros(nE, nc, rb, dtype=torch.uint8).pin_memory()
        self.vpin = torch.zeros(nE, nc, dtype=torch.float32).pin_memory()
        self.rnp = self.rpin.numpy()
        for i in range(fw.n_layers):
            if fw._layer(i).ix:
                indexer_static_init(fw._indexer_fused(i), self.cap)
        self.prepared = True

    def clear(self):
        """drop every captured graph (a flag changed) and start a fresh memory pool (a pool handle must not outlive its graphs)"""
        self.graphs.clear()
        self.pool = torch.cuda.graph_pool_handle()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self):
        """new sequence: zero the static state (buffers keep their addresses so captured graphs stay valid)"""
        fw = self.fw
        self.expect_n = None
        for b in self.st.kv.values():
            b.zero_()
        for ix in getattr(fw, "ixf", {}).values():
            ix.n, ix.P, ix.vg = 0, 0, None
            ix.pool.zero_()
            if getattr(ix, "vgb", None) is not None:
                ix.vgb.zero_()

    # -- the step body (eager warm-up and capture run exactly this)
    def body(self):
        fw, gs = self.fw, self.gs
        e = _ns.gather_dequant_q8_0(self.st.table, gs.tok)
        x = e[:, None, :].expand(-1, fw.hc, -1).contiguous()
        for i in range(fw.n_layers):
            x = fw.layer(i, x, gs.pos, None, 0)
        fw._ns_dev = True
        try:
            lg = fw._head(x, False)
        finally:
            fw._ns_dev = False
        return lg, torch.argmax(lg), x

    def _graph(self, Nb):
        g = self.graphs.get(Nb)
        if g is not None:
            return g
        fw, gs = self.fw, self.gs
        gs.Nb = Nb
        fw._gs = gs
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.body()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg, pool=self.pool):
                out = self.body()
        finally:
            fw._gs = None
        g = self.graphs[Nb] = (cg, out)
        self.stats["captures"] += 1
        return g

    def stage(self, start):
        """host side of one token: Engram hash + threaded pread of this token's rows into pinned staging, one H2D each"""
        fw = self.fw
        jobs = []
        toks = fw.tokens
        for i in sorted(fw.ple):
            e = fw.eng[i]
            ids, valid = engram_row_ids(e, toks, start)
            self.vpin[e.e] = torch.tensor(valid)
            jobs += [(e, c, r) for c, r in enumerate(ids)]
        def rd(j):
            e, c, r = j
            self.rnp[e.e, c] = e.table.raw_row(r)
        list(_POOL.map(rd, jobs))
        self.gs.eng_raw.copy_(self.rpin, non_blocking=True)
        self.gs.eng_valid.copy_(self.vpin, non_blocking=True)

    def step(self, tok):
        """append host token id `tok` at position fw.n; returns the DEVICE logits [V] of this position (valid until the next step)"""
        fw = self.fw
        if not self.prepared:
            self.prepare()
        start = fw.n
        assert start < self.cap, "context past the graph cap (HY4_NOSYNC_CAP)"
        if self.expect_n != start:            # an eager prefill / reset ran since the last graph step: refill the indexer vg rows from its state
            for i in range(fw.n_layers):
                if fw._layer(i).ix:
                    indexer_static_init(fw._indexer_fused(i), self.cap)
        if self.ev_set:
            self.ev.synchronize()           # the previous replay has read the pinned staging buffers: safe to overwrite them
        fw.tokens.append(int(tok))
        Nb = min(self.cap, (start // BUCKET + 1) * BUCKET)
        self.gs.Nb = Nb
        self.hin[0], self.hin[1] = int(tok), start
        self.gs.inp.copy_(self.hin, non_blocking=True)
        self.stage(start)
        cg, _out = self._graph(Nb)
        lg = _out[0]
        self.x_out = _out[2]
        cg.replay()
        self.ev.record()
        self.ev_set = True
        fw.n += 1
        self.expect_n = fw.n
        self.stats["replays"] += 1
        return lg
