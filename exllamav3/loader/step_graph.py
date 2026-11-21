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
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import step_nosync as _ns
from .engram_rows import QB, E4M3, E8M0
from . import attn_eager as _ae
from . import router_gemv as _RG
from . import lever_fires as _LVF
from .engram_stage import EngramStager

from . import graph_bucket as _gb
BUCKET = _gb.BASE
TM = int(os.environ.get("HY4_GRAPH_TM", "8"))          # max rows of one graph step (verify rows)
MASK64 = (1 << 64) - 1
PERT = None                      # [cap, D] fp32 noise table (measurement arms only)
_IO_W = int(os.environ.get("HY4_GRAPH_IO", "64"))
PRIME_IX = os.environ.get("HY4_LC_PRIME", "0") == "1"      # draft-cache fill rows (T > 16) select with the prefill scorer (fp16 GEMM, exact top-k)
PRIME_SPARSE = os.environ.get("HY4_PRIME_SPARSE", "1") == "1"      # draft-cache fill above PRIME_IX_BLK groups: the selection stays a PfSparse (gather kernel in Hy4MLA.attend), never a dense [T, N] mask
PRIME_IX_BLK = int(os.environ.get("HY4_PRIME_IX_BLK", "16384"))           # draft-cache fill: pooled groups above this use the blocked linear-time selection
VG_RING = 4096 if int(os.environ.get("HY4_SWA_RING", "0")) else 0      # rows of the indexer (value | gate) ring: one prime call (2048 rows) + the open group
RING_ROWS = int(os.environ.get("HY4_SWA_RING", "0"))      # >= 1: K rows of window layers in a ring buffer of this many rows (e.g. 1024) instead of cap rows
_POOL = ThreadPoolExecutor(max_workers=_IO_W)
SPLIT = os.environ.get("HY4_GRAPH_SPLIT", "1") == "1"      # 3 graphs per step, cut before the first two Engram layers: the rows of the later layers are staged while the GPU runs the earlier ones
STAGE_VEC = os.environ.get("HY4_STAGE_VEC", "1") == "1"   # vectorised Engram hash + a few big read chunks (engram_stage); 0 = the per-row python path


# ---------------------------------------------------------------------------------------------------- static indexer
def _rms(x, w, eps):
    x = x.float()
    if _ae.RMS and x.dim() == 2 and x.is_cuda:        # G2: one launch instead of five
        return _ae.rmsw(x, w, eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


from . import pool_fused as _PF
from . import lc_sparse as _LS
_PF.ENABLED = os.environ.get("HY4_IDX_POOLK", "0") == "1"
from . import kv_fp8 as _KV8
from . import rope_rows as _RR
POOL16 = os.environ.get("HY4_IX_POOL16", "0") == "1"      # indexer pooled keys stored in fp16 (half the bytes; the prefill scorer already scores with the fp16 copy; selection reads fp16, top-k set overlap gate in tests/test_kv10.py)
KV_FP8 = os.environ.get("HY4_KV_FP8", "0") == "1"      # latent K cache of the full-attention layers in fp8 e4m3 + per-token scale (+ bf16 rope dims)


def indexer_static_init(ix, cap):
    """FusedKPoolIndexer -> fixed buffers: pool [cap // k, d], per-position (value | gate) rows vgb [cap, 2d] (open group filled from the eager state), rope table to cap"""
    k, d = ix.k, ix.d
    rows = cap // k
    if ix.pool.shape[0] < rows:
        big = torch.zeros(rows, d, device=ix.dev, dtype=torch.float16 if POOL16 else ix.pool.dtype)
        big[:ix.P] = ix.pool[:ix.P]
        ix.pool = big
    elif POOL16 and ix.pool.dtype != torch.float16:
        ix.pool = ix.pool.half()
    vgb = getattr(ix, "vgb", None)             # per-position (value | gate) rows [cap, 2d]: a rejected / re-run position just overwrites its row
    if vgb is None:
        ix.vg_ring = VG_RING if (VG_RING and VG_RING < cap) else 0       # HY4_SWA_RING: only the open group (+ the rows of one batch call) is ever read: slot = position % VG_RING
        vgb = ix.vgb = torch.zeros(ix.vg_ring or cap, 2 * d, device=ix.dev)
    elif not hasattr(ix, "vg_ring"):
        ix.vg_ring = 0                         # a buffer bound from outside (multi-session slab view [cap, 2d]) is full length: positions index it directly, never a ring
    if ix.vg is not None:                      # open group of an eager prefill (< k rows)
        m = ix.vg.shape[0]
        vgb[(torch.arange(ix.n - m, ix.n, device=ix.dev) % ix.vg_ring) if ix.vg_ring else slice(ix.n - m, ix.n)] = ix.vg
    ix._max_pos = cap
    if not _RR.ENABLED:             # HY4_ROPE_POS: the rope rows of a call come from its positions, no table to the full context
        ix._table(cap)


IX_GV_FIRES = _LVF.Fires("draft_ix_gv", hit="gv", miss="gemm")      # draft indexer matrices (ix has the gv attribute): gv = mat-vec calls, gemm = {reason: n} of the calls that kept the GEMM (the trunk's indexer has no gv attribute and is not counted)


def _mmw(ix, a, name):
    """a @ ix.w[name].T; the draft layer's indexer (ix.gv, HY4_DRAFT_IX_GV=1) runs its small fp32 matrices through the router mat-vec (rocBLAS SGEMM at M = 1 runs these at 12-60 GB/s)"""
    w = ix.w[name]
    gv = getattr(ix, "gv", None)
    if gv is None:
        return a @ w.T
    if gv and _RG.usable(a, w):
        IX_GV_FIRES.hit()
        return _RG.logits(a, w)
    IX_GV_FIRES.miss("flag_off" if not gv else "rows>maxt" if (a.dim() == 2 and a.shape[0] > _RG.MAXT) else "unusable_shape")      # rows>maxt: draft-cache fill / prefill chunks (natural); unusable_shape: any other refusal
    return a @ w.T


@torch.no_grad()
def indexer_keys_static(ix, x, pos, ar):
    """key side of indexer_forward_static: the per-position (value | gate) rows and the pool groups they complete (x [T, E], pos [T] long device, ascending). Needs no query and no selection."""
    k, d = ix.k, ix.d
    ik = _rms(_mmw(ix, x, "attn_k"), ix.w["k_norm"], ix.eps)
    vg_new = torch.cat([_mmw(ix, ik, "value"), _mmw(ix, ik, "gate")], -1)
    vr = getattr(ix, "vg_ring", 0)
    assert not vr or x.shape[0] + k <= vr, "vg ring smaller than one call"
    ix.vgb.index_copy_(0, pos % vr if vr else pos, vg_new)
    if _PF.ENABLED:                                     # one Triton kernel for the whole pool update (HY4_IDX_POOLK)
        _PF.pool_update(ix, pos)
    else:
        g0 = (pos // k) * k
        gi = g0[:, None] + ar[:k][None, :]
        rows = ix.vgb[gi % vr if vr else gi]                                # [1, k, 2d]: the (partial) group of this position
        v, g = rows[..., :d].reshape(-1, k, d), rows[..., d:].reshape(-1, k, d)
        p = (torch.softmax(g + ix.w["ape"][None], dim=1) * v).sum(1)
        pooled = ix._rope(_rms(p, ix.w["pool_norm"], ix.eps), g0)
        ix.pool.index_copy_(0, pos // k, pooled.to(ix.pool.dtype))            # a group still open holds partial data here; it stays masked until it is complete


@torch.no_grad()
def indexer_forward_static(ix, x, qr, pos, Nb, ar):
    """x [T, E], qr [T, R], pos [T] long (device, ascending), Nb rows -> allow [T, Nb] bool (T = 1 decode, T > 1 verify rows). Same maths and op order as FusedKPoolIndexer.forward for one
    token; no host value is read: group slot and rope position come from `pos`, the visible groups from a mask (gend < pos).
    Returns None (all keys causal) while the bucket is small enough that no query can see more than n_top groups: only the pool state is updated."""
    k, d, H = ix.k, ix.d, ix.H
    indexer_keys_static(ix, x, pos, ar)
    if Nb <= (ix.n_top + 1) * k:        # every position of this bucket sees <= n_top groups: the selection is all-causal (the eager path returns None)
        return None
    T = x.shape[0]
    iq = ix._rope(_mmw(ix, qr, "q_b").reshape(T, H, d), pos)
    wt = _mmw(ix, x.float(), "proj") / math.sqrt(d * H)
    Pb = Nb // k
    if PRIME_IX and T > 16 and ix.tail and iq.is_cuda:       # draft-cache fill of a long prompt (T up to 2048 rows): fused fp16 scorer + exact select instead of the fp32 [T, 16, Pb] einsum
        off = (pos.data_ptr() - ar.data_ptr()) // ar.element_size()
        if pos.dim() == 1 and 0 <= off < ar.shape[0] and Nb == Pb * k:
            from . import prefill_kernels as _pkm
            if Pb > PRIME_IX_BLK:       # long prompt: the unblocked ix_select keeps a pow2(P)-wide row in registers (P = 65K groups at 256K: the Triton compile alone ran > 20 min at 100 % CPU)
                sel = _pkm.ix_select_blocked(_pkm.to_f16(iq), _pkm.to_f16(ix.pool[:Pb]), wt, int(off), k, ix.tail, ix.n_top, Nb)
                return sel if PRIME_SPARSE else sel.to_dense()
            sc = _pkm.ix_scores_fused(_pkm.to_f16(iq), _pkm.to_f16(ix.pool[:Pb]), wt, int(off), k, ix.tail)
            return _pkm.ix_select(sc, int(off), Nb, ix.n_top, k, ix.tail, want_bias=False)
    if _LS.ENABLED and ix.tail and T <= 16 and _LS._usable(iq):        # HY4_LC_SPARSE: exact top-k over the pools + sparse attend (no [T, Nb] mask, no dense key scan)
        idx, nsel = _LS.select_groups(iq.contiguous(), wt.contiguous(), ix.pool, pos, Pb, ix.n_top, k, ix.tail)
        sel = _LS.SparseSel(idx, nsel, k, ix.tail)
        if _LS.CHECK and not torch.cuda.is_current_stream_capturing():
            _LS.check_rows(iq, wt, ix.pool, pos, Pb, ix.n_top, k, sel)
        return sel
    kq = torch.relu(torch.einsum("nhd,pd->nhp", iq, ix.pool[:Pb].to(iq.dtype)))
    sc = (kq * wt[:, :, None]).sum(1)
    gend = ar[:Pb] * k + (k - 1)
    vis = (gend[None] < pos[:, None]) if ix.tail else (gend[None] <= pos[:, None])
    sc = torch.where(vis, sc, torch.full_like(sc, -math.inf))
    idx = torch.topk(sc, min(ix.n_top, Pb), dim=1).indices
    gsel = torch.zeros(T, Pb, dtype=torch.bool, device=ix.dev)
    gsel.scatter_(1, idx, True)
    gsel &= vis
    allow = torch.zeros(T, Nb, dtype=torch.bool, device=ix.dev)
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
        self.fw, self.cap, self.Nb, self.T = fw, cap, 0, 1
        dev = fw.dev
        self.inp = torch.zeros(2, TM, dtype=torch.long, device=dev)     # row 0 tokens, row 1 positions (first T columns used)
        self.ar = torch.arange(cap, device=dev)
        self.eng_raw = self.eng_valid = None
        self._pv = None

    @property
    def tok(self):
        return self.inp[0, :self.T]

    @property
    def pos(self):
        """the position row of this step: ONE view object per segment body (new_pos() starts a new one), so the rope rows computed from it (HY4_ROPE_POS, memo on the object) are shared by the layers"""
        if self._pv is None or self._pv.shape[0] != self.T:
            self._pv = self.inp[1, :self.T]
            self._pv._rope_ok = True
        return self._pv

    def new_pos(self):
        self._pv = None

    # -- Engram: rows are staged on the host before the replay, decoded + applied inside the graph
    def engram(self, fw, i, x):
        e = fw.eng[i]
        T = self.T
        u = self.eng_raw[e.e, :T]                                       # [T, n_cols, row_bytes]
        tab = e.table
        t4, t8 = _tabs(fw.dev)
        if tab.layout == "plain":
            rows = t4[u.long()]
        else:
            w = tab.width
            rows = (t4[u[..., :w].long()].reshape(T, u.shape[1], -1, QB) * t8[u[..., w:].long()][..., None]).reshape(T, u.shape[1], w)
        return e.apply(x, rows, self.eng_valid[e.e, :T])

    def append_kv(self, i, kn):
        b = _ns.state(self.fw).kv[i]
        ring = getattr(self.fw._layer(i).mla, "kv_ring", 0)
        if ring:                                  # window layer: ring buffer, slot = position % ring; attention reads the whole ring through absolute positions
            b.index_copy_(0, self.pos % ring, kn)
            return b
        if b.__class__.__name__ == "Fp8KV":           # HY4_KV_FP8: quantise + scatter the rows
            b.write_pos(self.pos, kn)
            return b[:self.Nb]
        b.index_copy_(0, self.pos, kn)
        return b[:self.Nb]

    def kv_row_fused(self, i, mla, cur, pos):
        """K row of this token written by ONE kernel (kv_fused): rms * w, rope from the attn_eager table, row `pos` of the layer's K buffer"""
        from . import attn_eager as AE, kv_fused as KVF
        b = _ns.state(self.fw).kv[i]
        c = mla._q(cur, "wkv_a") @ mla.wkv_a.T
        tab, per_row = AE.table_rows(mla.rope_dim, mla.theta, self.fw.dev, pos)
        ring = getattr(mla, "kv_ring", 0)
        KVF.write_row(c, mla.kv_a_norm, tab, pos, b, mla.R, mla.rope_dim, mla.eps, ring=ring, per_row=per_row)
        return b if ring else b[:self.Nb]

    def indexer(self, fw, i, cur, qr, N):
        # T rows in one pass: every row writes its vg row first, then the pooled groups are computed (a group completed inside this round gets its
        # final value, a group still open stays masked), then each row selects with its own visibility (gend < pos): the state of T sequential steps
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
        self.hin = torch.zeros(2, TM, dtype=torch.long).pin_memory()
        self.hnp = self.hin.numpy()
        self.stager = None
        self.split = SPLIT
        self.t_launch = 0.0
        self._pre = None
        self.stats = {"captures": 0, "replays": 0, "capture_s": 0.0}
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
            m = fw._layer(i).mla
            if RING_ROWS and m.swa and m.window and not hasattr(m, "kv_ring") and m.window + TM + 8 <= RING_ROWS:
                m.kv_ring = RING_ROWS                  # exact: a window-W layer never reads a key older than W - 1 positions (RING_ROWS >= W + the verify rows)
            b = st.kv.get(i)
            if getattr(m, "kv_ring", 0):
                if b is None or b.shape[0] != m.kv_ring:
                    st.kv[i] = torch.zeros(m.kv_ring, 512 + 64, dtype=torch.float32, device=fw.dev)
                continue
            if KV_FP8 and not m.swa and fw._layer(i).ix and not isinstance(b, _KV8.Fp8KV):          # full-attention layer: fp8 latent cache
                nb = _KV8.Fp8KV(self.cap, 512, 64, fw.dev)
                if b is not None:
                    nb.write(0, b[:min(b.shape[0], self.cap)])
                st.kv[i] = nb
                continue
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
        gs.eng_raw = torch.zeros(nE, TM, nc, rb, dtype=torch.uint8, device=fw.dev)
        gs.eng_valid = torch.zeros(nE, TM, nc, dtype=torch.float32, device=fw.dev)
        self.rpin = torch.zeros(nE, TM, nc, rb, dtype=torch.uint8).pin_memory()
        self.vpin = torch.zeros(nE, TM, nc, dtype=torch.float32).pin_memory()
        self.rnp = self.rpin.numpy()
        self._rmv = None
        if STAGE_VEC:
            self.stager = EngramStager([fw.eng[i] for i in sorted(fw.ple)], fw.table, self.rnp, self.vpin.numpy(), pool=_POOL)
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
        self._drain_pre()
        for b in self.st.kv.values():
            b.zero_()
        for ix in getattr(fw, "ixf", {}).values():
            ix.n, ix.P, ix.vg = 0, 0, None
            ix.pool.zero_()
            if getattr(ix, "vgb", None) is not None:
                ix.vgb.zero_()

    # -- the step body (eager warm-up and capture run exactly this)
    def bounds(self):
        """layer ranges of the graph segments. Engram rows are applied at the START of an Engram layer, so the cuts sit before the first two Engram layers:
        [embedding + layers before ple[0]) | [ple[0], ple[1]) | [ple[1], end + head). The rows of ple[0] are staged while segment 0 runs, those of the later
        layers while segment 1 runs (the host never holds the GPU back for them). One segment when split is off or the model has < 2 Engram layers."""
        fw = self.fw
        ple = sorted(fw.ple)
        if self.split and self.stager is not None and len(ple) >= 2 and 0 < ple[0] < ple[1]:
            return [(0, ple[0]), (ple[0], ple[1]), (ple[1], fw.n_layers)]
        return [(0, fw.n_layers)]

    def groups(self, bnds):
        """per segment: the Engram module range [a, b) whose rows must be on the device before the segment starts (None = none)"""
        ple = sorted(self.fw.ple)
        out = []
        for lo, hi in bnds:
            es = [e for e, L in enumerate(ple) if lo <= L < hi]
            out.append((es[0], es[-1] + 1) if es else None)
        return out

    def body_seg(self, si, x, bnds):
        fw, gs = self.fw, self.gs
        lo, hi = bnds[si]
        gs.new_pos()                  # every segment body (eager warm-up, graph capture) starts its own position view: no rope rows memo outlives a capture
        if si == 0:
            e = _ns.gather_dequant_q8_0(self.st.table, gs.tok)
            if PERT is not None:                      # noise-ensemble null arm: 1e-6 fixed-seed perturbation of the embedding output (quality gate, never in production)
                e = e + PERT[gs.pos]
            x = e[:, None, :].expand(-1, fw.hc, -1).contiguous()
        for i in range(lo, hi):
            x = fw.layer(i, x, gs.pos, None, 0)
        if si < len(bnds) - 1:
            return x
        fw._ns_dev = True
        try:
            lg = fw._head(x, gs.T > 1)              # T > 1: logits of every row [T, V]
        finally:
            fw._ns_dev = False
        return lg, torch.argmax(lg, -1), x, fw._h

    def body(self, bnds=None):
        bnds = bnds or self.bounds()
        x = None
        for si in range(len(bnds)):
            x = self.body_seg(si, x, bnds)
        return x

    def bucket(self, start, T=1):
        """context bucket Nb of a step of T rows at positions start .. start + T - 1 (rows past the last position are masked inside the graph)"""
        return _gb.bucket(start + T, self.cap)

    def graph(self, Nb, T=1):
        """the captured (graph, outputs) of a T-row step at context bucket Nb (captured now if not yet). Row-generic: T = rows of ONE sequence (verify) or any
        batch of rows that the body maps onto positions (inp[1, :T])."""
        return self._graph(Nb, T)

    def precapture(self, nbs=None, ts=None, progress=None):
        """capture every (Nb, T) graph now (a capture inside a decode round costs 2 eager bodies + the capture: 0.5 - 1 s of host time that lands in one round).
        nbs: iterable of buckets (default every bucket up to the cap), ts: row counts (default 1 .. TM). Returns [(Nb, T, ms)] of the graphs built. The static
        inputs keep their last values: call it between requests (the K / indexer rows it rewrites are the same ones the last step wrote) and reset() after."""
        import time
        nbs = sorted(set(nbs)) if nbs is not None else _gb.buckets(self.cap)
        ts = sorted(set(ts)) if ts is not None else list(range(1, TM + 1))
        if not self.prepared:
            self.prepare()
        out = []
        for T in ts:
            assert 1 <= T <= TM
            for Nb in nbs:
                assert Nb % BUCKET == 0 and Nb <= self.cap
                if (Nb, T, len(self.bounds())) in self.graphs:
                    continue
                t = time.perf_counter()
                self._graph(Nb, T)
                out.append((Nb, T, round((time.perf_counter() - t) * 1000)))
                if progress:
                    progress(f"graph Nb={Nb} T={T} {out[-1][2]} ms")
        return out

    def _graph(self, Nb, T=1):
        key = (Nb, T, len(self.bounds()))
        g = self.graphs.get(key)
        if g is not None:
            return g
        import time
        t_cap = time.perf_counter()
        fw, gs = self.fw, self.gs
        gs.Nb, gs.T = Nb, T
        fw._gs = gs
        bnds = self.bounds()
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.body(bnds)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            cgs, x = [], None
            for si in range(len(bnds)):
                cg = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cg, pool=self.pool):
                    x = self.body_seg(si, x, bnds)
                cgs.append(cg)
        finally:
            fw._gs = None
            gs.new_pos()
        g = self.graphs[key] = (cgs, x)
        self.stats["captures"] += 1
        self.stats["capture_s"] += time.perf_counter() - t_cap
        return g

    def _jobs(self, start, t0, t1, toks):
        fw = self.fw
        jobs = []
        for i in sorted(fw.ple):
            e = fw.eng[i]
            for t in range(t0, t1):
                ids, valid = engram_row_ids(e, toks, start + t)
                self.vpin[e.e, t] = torch.tensor(valid)
                jobs += [(e, t, c, r) for c, r in enumerate(ids)]
        return jobs

    def _submit(self, jobs):
        """spread the row reads over the thread pool; returns futures (preadv releases the GIL)"""
        fw = self.fw
        tab = fw.table
        fi = tab.off >> 48
        fd, o0, rb, nrows = fw.g.fds[fi], fw.g.bases[fi] + (tab.off & ((1 << 48) - 1)), tab.row_bytes, tab.rows
        mvs = self._rmv
        if mvs is None:
            mvs = self._rmv = {}
        def rd(chunk):
            for e, t, c, r in chunk:
                if not 0 <= r < nrows:
                    raise IndexError(f"{tab.name}: row {r} outside 0..{nrows - 1}")
                mv = mvs.get((e.e, t, c))
                if mv is None:
                    mv = mvs[(e.e, t, c)] = memoryview(self.rnp[e.e, t, c])
                if os.preadv(fd, [mv], o0 + r * rb) != rb:
                    raise IOError(f"{tab.name}: short read of row {r}")
        per = max(1, -(-len(jobs) // _IO_W))
        return [_POOL.submit(rd, jobs[a:a + per]) for a in range(0, len(jobs), per)]

    def _drain_pre(self):
        """wait for the reads of a staged row that was never used (they write into the shared pinned staging)"""
        pre, self._pre = self._pre, None
        if pre is not None:
            for f in pre[2]:
                f.result()

    def prestage(self, tok):
        """HY4_STAGE_OV: row 0 of the next verify (token tok at position fw.n) does not depend on the drafts: start its Engram reads now, while the draft chain runs on the GPU"""
        fw = self.fw
        self._drain_pre()
        if self.ev_set:
            self.ev.synchronize()           # the previous replay has finished reading the pinned staging rows
        start = fw.n
        if self.stager is not None:
            futs = self.stager.stage(fw.tokens, start, 0, 1, extra=[int(tok)])
        else:
            futs = self._submit(self._jobs(start, 0, 1, fw.tokens + [int(tok)]))
        self._pre = (start, int(tok), futs)

    def _stage_begin(self, start, T):
        """host side of a step, part 1: match the prestaged row 0 (token fw.tokens[start]) and hash every (layer, row) still to read"""
        fw = self.fw
        pre, self._pre = self._pre, None
        t0 = 0
        if pre is not None and pre[0] == start and pre[1] == fw.tokens[start]:
            t0 = 1
        elif pre is not None:
            for f in pre[2]:                # stale staged row: its reads still write into the staging rows below
                f.result()
            pre = None
        hs = self.stager.hash(fw.tokens, start, t0, T) if self.stager is not None and T > t0 else None
        self._sp = [start, T, t0, pre, hs]

    def _stage_group(self, a, b):
        """part 2: read the rows of the Engram modules a .. b - 1 into the pinned staging, then ONE H2D of those layers (stream-ordered before the next replay)"""
        sp = self._sp
        start, T, t0, pre, hs = sp
        fw = self.fw
        if self.stager is not None:
            futs = self.stager.submit(hs[0], hs[1], t0, es=range(a, b)) if hs is not None else []
        else:
            futs = self._submit(self._jobs(start, t0, T, fw.tokens)) if T > t0 else []
        if pre is not None:
            futs = futs + pre[2]
            sp[3] = None
        for f in futs:
            f.result()
        if a == 0 and b == self.rpin.shape[0]:
            self.gs.eng_raw.copy_(self.rpin, non_blocking=True)
            self.gs.eng_valid.copy_(self.vpin, non_blocking=True)
        else:
            self.gs.eng_raw[a:b].copy_(self.rpin[a:b], non_blocking=True)
            self.gs.eng_valid[a:b].copy_(self.vpin[a:b], non_blocking=True)

    def stage(self, start, T=1):
        """host side of T tokens, all layers at once (fw.tokens holds them at start .. start + T - 1)"""
        self._stage_begin(start, T)
        self._stage_group(0, self.rpin.shape[0])

    def step(self, tok):
        """append host token id `tok` at position fw.n; returns the DEVICE logits [V] of this position (valid until the next step)"""
        return self.step_rows([int(tok)])[0]

    def step_rows(self, toks):
        """append T host token ids at positions fw.n .. fw.n + T - 1 in ONE graph replay (T = 1: a plain decode step). Returns (logits, argmax, x, h) as DEVICE
        tensors (T = 1: logits [V], argmax scalar; T > 1: [T, V], [T]). fw.n / fw.tokens advance by T: a verifier that rejects rows rewinds them (K rows and
        indexer state are written by position, so the rejected rows are simply overwritten by the next step)."""
        fw = self.fw
        T = len(toks)
        assert 1 <= T <= TM
        if not self.prepared:
            self.prepare()
        start = fw.n
        assert start + T <= self.cap, "context past the graph cap (HY4_NOSYNC_CAP)"
        if self.expect_n != start:            # an eager prefill / reset ran since the last graph step: refill the indexer vg rows from its state
            for i in range(fw.n_layers):
                if fw._layer(i).ix:
                    indexer_static_init(fw._indexer_fused(i), self.cap)
        if self.ev_set:
            self.ev.synchronize()           # the previous replay has read the pinned staging buffers: safe to overwrite them
        fw.tokens.extend(int(t) for t in toks)
        Nb = self.bucket(start, T)
        self.gs.Nb = Nb
        self.hnp[0, :T] = toks
        self.hnp[1, :T] = np.arange(start, start + T)
        self.gs.inp.copy_(self.hin, non_blocking=True)
        bnds = self.bounds()
        g = self.graphs.get((Nb, T, len(bnds)))
        full = g is None
        if full:                                  # first use of this (bucket, rows): stage everything, then build the graphs (their warm-up bodies see the real inputs)
            self.stage(start, T)
            g = self._graph(Nb, T)
        cgs, _out = g
        self.x_out = _out[2]
        fw._h = _out[3]
        grp = self.groups(bnds)
        if not full:
            self._stage_begin(start, T)
        for si, cg in enumerate(cgs):
            if not full and grp[si] is not None:
                self._stage_group(*grp[si])       # segment si - 1 is already running on the GPU
            cg.replay()
            if si == 0:
                self.t_launch = time.perf_counter()
        self.ev.record()
        self.ev_set = True
        fw.n += T
        self.expect_n = fw.n
        self.stats["replays"] += 1
        return _out[0], _out[1], _out[2], _out[3]

    def rewind(self, n):
        """verifier rejected rows: the sequence continues from position n (<= fw.n); stale K / indexer rows past n are overwritten by position"""
        fw = self.fw
        del fw.tokens[n:]
        fw.n = self.expect_n = n
