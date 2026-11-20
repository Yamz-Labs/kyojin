"""Multi-session Hy4 decode: rows of S sessions in ONE graph replay (perf/hy4-strix-mu).

The solo graph step (step_graph.GraphDecoder) appends T consecutive tokens of ONE sequence. Here the T rows of a step may belong to different sessions: each row
carries (token, position, slab base) in the static input, every per-session buffer is a slab of a flat tensor (mu_slots), and the ops that read or write per-session
state (K row write, indexer update + scoring, MLA core, Engram row staging) address their own slab. Dense matrices, router, experts (union of the rows' experts), hyper
connections and the output head are row-wise already, so the rows share every weight read.

A session binds to the eager objects (Hy4Forward n / tokens, the indexers' pool / vgb / n / P / vg, the MTP head's K buffer) by `bind`: the solo code of prefill and of the draft
head runs on a session unchanged, on the slab views. Graphs are keyed (bucket, rows): the row -> session mapping is device data, so any mix of sessions reuses them."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from . import mu_debug as DBG
from . import mu_slots as MS
from . import step_graph as SG
from . import step_nosync as _ns
from .engram_rows import QB  # noqa: F401  (re-export for callers)

TM = SG.TM

# HY4_MU_HOSTFAST=1: the Engram rows of a step are hashed with the numpy EngramStager (one pass per session), read with the cached-row fast path, and each layer group is staged
# right before its own graph segment so the reads overlap the GPU (the solo step_rows pipeline); the step inputs are written through a numpy view of the pinned buffer.
# HY4_MU_HOSTPROF=1: perf_counter splits per step (HT, seconds) for the server log line of mu_engine.
FIRES = {"stage_vec": 0, "stage_ov": 0, "inp_np": 0, "rows_gt8": 0, "steps": 0}      # path counters (FIRES-CHECK): a flagged arm with 0 stage_ov FAILS
HT = {"stage": 0.0, "launch": 0.0, "inputs": 0.0, "gpu": 0.0, "bubble": 0.0}      # gpu = summed replay time of the segments, bubble = GPU idle between segments (events, HOSTPROF only)


def hostfast() -> bool:
    return os.environ.get("HY4_MU_HOSTFAST", "0") not in ("", "0")


def hostprof() -> bool:
    return os.environ.get("HY4_MU_HOSTPROF", "0") not in ("", "0")


class _GSMU(SG._GS):
    """static context of a multi-session step: inp rows 0 token, 1 position, 2 slab base (slot * cap)"""
    slots = True

    def __init__(self, fw, cap):
        super().__init__(fw, cap)
        self.inp = torch.zeros(3, TM, dtype=torch.long, device=fw.dev)
        self.kvflat = {}
        self.slabs = {}

    @property
    def base(self):
        return self.inp[2, :self.T]

    def append_kv(self, i, kn):
        b = self.kvflat[i]
        b.index_copy_(0, self.base + self.pos, kn)
        return b

    def indexer(self, fw, i, cur, qr, N):
        from . import pool_fused as _PF
        return MS.indexer_forward_slots(fw._indexer_fused(i), self.slabs[i], cur, qr, self.pos, self.base, N, self.ar, poolk=_PF.ENABLED)


class Session:
    """host state of one session; the device state lives in the slabs of the decoder"""

    def __init__(self, slot):
        self.slot = slot
        self.tokens: list[int] = []
        self.n = 0                      # positions written (== len(tokens) between steps)
        self.ixs: dict = {}             # trunk indexer layer -> (n, P, vg) eager state
        self.mt_ix = (0, 0, None)       # MTP head indexer eager state (unused after the static init: kept for symmetry)
        self.y = None                   # next token to feed (host int)
        self.hcur = None                # trunk state [1, D] of the last accepted position (device)
        self.ids: list[int] = []
        self.pf_pos = 0
        self.H: list = []
        self.ng = None
        self.extra: dict = {}


class MultiDecoder(SG.GraphDecoder):
    def __init__(self, fw, nslots, cap=None):
        super().__init__(fw, cap)
        self.S = nslots
        self.gs = _GSMU(fw, self.cap)
        self.hin = torch.zeros(3, TM, dtype=torch.long).pin_memory()
        self.hnp = self.hin.numpy()
        self.mt = None

    # -- one-time setup: flat slabs, then the solo prepare on slot 0's views
    def prepare(self):
        fw, st, gs, cap, S = self.fw, self.st, self.gs, self.cap, self.S
        st.ensure(cap)
        for i in range(fw.n_layers):
            mla = fw._layer(i).mla
            flat = torch.zeros(S * cap, mla.R + mla.rope_dim, dtype=torch.float32, device=fw.dev)
            gs.kvflat[i] = flat
            st.kv[i] = flat[:cap]
        for i in range(fw.n_layers):
            if fw._layer(i).ix:
                ix = fw._indexer_fused(i)
                sl = gs.slabs[i] = MS.IxSlabs(ix, S, cap)
                ix.pool, ix.vgb = sl.view(0)
        super().prepare()
        for ix in getattr(fw, "ixf", {}).values():       # slabs are zero: eager state of every session starts empty
            ix.n, ix.P, ix.vg = 0, 0, None

    def attach_mtp(self, mt):
        """per-session K buffer + indexer slabs of the MTP draft head"""
        self.mt = mt
        S, cap = self.S, self.cap
        mt.kflat = torch.zeros(S * cap, mt.mla.R + mt.mla.rope_dim, dtype=torch.float32, device=mt.dev)
        mt.sl = MS.IxSlabs(mt.ix, S, cap)
        mt.kbuf = mt.kflat[:cap]
        mt.ix.pool, mt.ix.vgb = mt.sl.view(0)

    # -- sessions
    def new_session(self, slot):
        with DBG.stage(f"new_session slot {slot}"):
            return self._new_session(slot)

    def _new_session(self, slot):
        s = Session(slot)
        c = self.cap
        for b in self.gs.kvflat.values():
            b[slot * c:(slot + 1) * c].zero_()
        for sl in self.gs.slabs.values():
            p, v = sl.view(slot)
            p.zero_()
            v.zero_()
        if self.mt is not None:
            self.mt.kflat[slot * c:(slot + 1) * c].zero_()
            p, v = self.mt.sl.view(slot)
            p.zero_()
            v.zero_()
        return s

    def bind(self, s):
        """make the eager objects (Hy4Forward, indexers, MTP head) work on session s"""
        fw, st, c = self.fw, self.st, self.cap
        for i, flat in self.gs.kvflat.items():
            st.kv[i] = flat[s.slot * c:(s.slot + 1) * c]
        for i, sl in self.gs.slabs.items():
            ix = fw._indexer_fused(i)
            ix.pool, ix.vgb = sl.view(s.slot)
            ix.n, ix.P, ix.vg = s.ixs.get(i, (0, 0, None))
        fw.n, fw.tokens, fw._x, fw._eng_pf = s.n, s.tokens, None, getattr(fw, "_eng_pf", {})
        if self.mt is not None:
            mt = self.mt
            mt.kbuf = mt.kflat[s.slot * c:(s.slot + 1) * c]
            mt.ix.pool, mt.ix.vgb = mt.sl.view(s.slot)
            mt.ix.n, mt.ix.P, mt.ix.vg = s.mt_ix

    def save(self, s):
        fw = self.fw
        s.n = fw.n
        for i in self.gs.slabs:
            ix = fw._indexer_fused(i)
            s.ixs[i] = (ix.n, ix.P, ix.vg)
        if self.mt is not None:
            s.mt_ix = (self.mt.ix.n, self.mt.ix.P, self.mt.ix.vg)

    def start_graph_state(self, s):
        """after the eager prefill of s: fold the open indexer group of each layer into the static vg rows (the solo `expect_n != start` branch)"""
        fw = self.fw
        with DBG.stage(f"start_graph_state slot {s.slot} n {s.n}"):
            self.bind(s)
            for i in self.gs.slabs:
                SG.indexer_static_init(fw._indexer_fused(i), self.cap)
            self.save(s)

    def _probe(self, Nb, T, rows):
        """HY4_MU_SYNC_DEBUG: the first step of each (bucket, rows, slot set) runs ONE eager body (captured graph or not) with a device sync + log around every layer and op
        (mu_debug.probe_wrap), on the static inputs of this very step: a fault that depends on the slot index is caught on the first step that uses it"""
        key = (Nb, T, tuple(sorted({s.slot for s, _ in rows})))
        seen = self.__dict__.setdefault("_probed", set())
        if key in seen:
            return
        seen.add(key)
        fw, gs = self.fw, self.gs
        tg = [(fw, "layer", "layer"), (fw, "_ffn", "ffn"), (gs, "indexer", "indexer"), (gs, "append_kv", "append_kv"), (gs, "engram", "engram")]
        tg += [(fw._layer(i).mla, "attend", f"L{i} mla.attend") for i in range(fw.n_layers)]
        gs.Nb, gs.T = Nb, T
        old = fw._gs
        fw._gs = gs
        try:
            with DBG.probe_wrap(tg), DBG.stage(f"probe eager body Nb={Nb} T={T} slots={key[2]}"):
                self.body(self.bounds())
        finally:
            fw._gs = old

    # -- host side of the Engram rows of a multi-session step
    def _stage_rows(self, rows):
        fw = self.fw
        jobs, r = [], 0
        for s, toks in rows:
            for t in range(len(toks)):
                p = s.n + t
                for i in sorted(fw.ple):
                    e = fw.eng[i]
                    ids, valid = SG.engram_row_ids(e, s.tokens, p)
                    self.vpin[e.e, r] = torch.tensor(valid)
                    jobs += [(e, r, c, rr) for c, rr in enumerate(ids)]
                r += 1
        for f in self._submit(jobs):
            f.result()
        self.gs.eng_raw.copy_(self.rpin, non_blocking=True)
        self.gs.eng_valid.copy_(self.vpin, non_blocking=True)

    # -- HY4_MU_HOSTFAST: vectorised staging of the Engram rows (numpy hash per session, rows concatenated along T), group by group
    def _hash_rows(self, rows):
        """ids int64 [nE, T, nc] and valid float32 [T] of every row of the step, rows in step order (EngramStager.hash per session: same integer maths as engram_row_ids)"""
        parts = [self.stager.hash(s.tokens, s.n, 0, len(toks)) for s, toks in rows]
        if len(parts) == 1:
            return parts[0]
        return np.concatenate([p[0] for p in parts], axis=1), np.concatenate([p[1] for p in parts])

    def _stage_group_mu(self, hs, a, b):
        """read the rows of the Engram modules a .. b - 1 into the pinned staging, then ONE H2D of those layers (stream-ordered before the next replay)"""
        for f in self.stager.submit(hs[0], hs[1], 0, es=range(a, b)):
            f.result()
        if a == 0 and b == self.rpin.shape[0]:
            self.gs.eng_raw.copy_(self.rpin, non_blocking=True)
            self.gs.eng_valid.copy_(self.vpin, non_blocking=True)
        else:
            self.gs.eng_raw[a:b].copy_(self.rpin[a:b], non_blocking=True)
            self.gs.eng_valid[a:b].copy_(self.vpin[a:b], non_blocking=True)

    def step_slots(self, rows):
        """rows = [(session, [tokens])]: append each token list at that session's next positions, ONE graph replay for all rows. Returns DEVICE (logits, argmax, x, h)
        with the rows concatenated in order (single row: logits [V], argmax scalar). session.n and session.tokens advance; a verifier that rejects rows calls `rewind`."""
        fw, gs = self.fw, self.gs
        T = sum(len(t) for _, t in rows)
        assert 1 <= T <= TM
        if not self.prepared:
            self.prepare()
        fast = hostfast() and self.stager is not None
        prof = hostprof()
        pc = time.perf_counter
        t0 = pc() if prof else 0.0
        if self.ev_set:
            self.ev.synchronize()
            if prof and getattr(self, "_tev_n", 0):          # the previous step is done: its per-segment event pairs give the GPU time and the idle gaps between segments
                pr = self._tev_pool[:self._tev_n]
                gpu = sum(a.elapsed_time(b) for a, b in pr)
                HT["gpu"] += gpu / 1000
                HT["bubble"] += (pr[0][0].elapsed_time(pr[-1][1]) - gpu) / 1000
                self._tev_n = 0
        tk, ps, bs, top = [], [], [], 0
        for s, toks in rows:
            assert s.n + len(toks) <= self.cap, "context past the slab"
            s.tokens.extend(int(t) for t in toks)
            tk += [int(t) for t in toks]
            ps += list(range(s.n, s.n + len(toks)))
            bs += [s.slot * self.cap] * len(toks)
            top = max(top, s.n + len(toks))
        Nb = min(self.cap, ((top - 1) // SG.BUCKET + 1) * SG.BUCKET)
        gs.Nb = Nb
        what = f"step_slots T={T} Nb={Nb} rows={[(s.slot, s.n, len(t)) for s, t in rows]}" if (not fast or DBG.enabled()) else "step_slots"
        bnds = self.bounds() if fast else None
        ov = fast and (Nb, T, len(bnds)) in self.graphs        # graph already captured: stage group by group while the earlier segments run
        hs = None
        with DBG.stage(what + " inputs + Engram rows"):
            if fast:
                self.hnp[0, :T] = tk
                self.hnp[1, :T] = ps
                self.hnp[2, :T] = bs
                FIRES["inp_np"] += 1
            else:
                self.hin[0, :T] = torch.tensor(tk)
                self.hin[1, :T] = torch.tensor(ps)
                self.hin[2, :T] = torch.tensor(bs)
            gs.inp.copy_(self.hin, non_blocking=True)
            if prof:
                t1 = pc()
                HT["inputs"] += t1 - t0
                t0 = t1
            if fast:
                hs = self._hash_rows(rows)
                FIRES["stage_vec"] += 1
                if not ov:                                   # first use of this (bucket, rows): stage everything, the capture's warm-up bodies read the rows
                    self._stage_group_mu(hs, 0, self.rpin.shape[0])
            else:
                self._stage_rows(rows)
            if prof:
                t1 = pc()
                HT["stage"] += t1 - t0
                t0 = t1
        if DBG.probe_enabled():
            self._probe(Nb, T, rows)
        with DBG.stage(what + " graph (capture if new)"):
            cgs, _out = self._graph(Nb, T)       # the solo capture: a LIST of graph segments (Engram split), replayed in order like step_rows
        self.x_out = _out[2]
        fw._h = _out[3]
        grp = self.groups(bnds) if ov else None
        for si, cg in enumerate(cgs):        # staged up front (_stage_rows / first use); with `ov` each group is read just before its own segment
            if ov and grp[si] is not None:
                t2 = pc() if prof else 0.0
                self._stage_group_mu(hs, *grp[si])
                if prof:
                    dt = pc() - t2
                    HT["stage"] += dt
                    t0 += dt                                 # launch = the replay calls only
            if prof:
                if getattr(self, "_tev_pool", None) is None:
                    self._tev_pool = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(8)]
                self._tev_pool[si][0].record()
            with DBG.stage(f"{what} replay segment {si}/{len(cgs)}"):
                cg.replay()
            if prof:
                self._tev_pool[si][1].record()
        if ov:
            FIRES["stage_ov"] += 1
        if T > 8:
            FIRES["rows_gt8"] += 1
        FIRES["steps"] += 1
        if prof:
            self._tev_n = len(cgs)
        self.ev.record()
        self.ev_set = True
        for s, toks in rows:
            s.n += len(toks)
        self.stats["replays"] += 1
        if prof:
            HT["launch"] += pc() - t0
        return _out[0], _out[1], _out[2], _out[3]

    @staticmethod
    def rewind(s, n):
        """session s continues from position n (<= s.n): stale K / indexer rows past n are overwritten by position"""
        del s.tokens[n:]
        s.n = n


from . import mtp_head as _MH


class MTPGraphMU(_MH.MTPGraph):
    """the draft chain graph per (slot, bucket): the captured graph holds the addresses of the slot's K buffer / indexer slab views (set `slot` and bind the session first)"""
    slot = 0

    def captured(self, slot, j):
        """True when the draft graph of (slot, the bucket of head position j) exists: run() then needs no `bind` (a replay touches no eager state)"""
        return (slot, min(self.mt.cap, ((j + self.k) // self.BUCKET + 1) * self.BUCKET)) in self.graphs

    def run(self, tok, h, j):
        """drafts of the bound session as a PRIVATE device tensor. The (slot, N) graphs share one memory pool and a graph's output can live in a block that an
        earlier-captured graph used for intermediates: replaying graph A after graph B (two sessions whose slots were first drafted in the other order) overwrote
        the drafts of B before the host read them (garbage token ids -> out-of-range embedding gather -> HSA 0x1016 in the verify replay). The clone is stream-ordered
        right after its own replay, so no later replay can touch it."""
        return super().run(tok, h, j).clone()

    def _graph(self, N):
        key = (self.slot, N)
        g = self.graphs.get(key)
        if g is not None:
            return g
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.body(N)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg, pool=self.pool):
            out = self.body(N)
        g = self.graphs[key] = (cg, out)
        return g
