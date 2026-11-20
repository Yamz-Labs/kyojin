"""Multi-session Hy4 engine + scheduler (perf/hy4-strix-mu): 2-4 sessions decoded in ONE batched trunk step, new requests prefilled in chunks between decode rounds.

  MuEngine     Hy4Engine with a MultiDecoder (loader/step_mu.py): per-session K / indexer slabs, per-session MTP head state, per-session n-gram + Engram history.
  MuScheduler  one thread: admit requests into free slots, one prefill chunk of the oldest prefilling request, then one decode round of every decoding session
               (plain: one row each; speculative: the MTP drafts + n-gram rows of each session share ONE verify replay, TM = 8 rows split evenly).
Greedy: a session's stream equals plain greedy decoding of the trunk (the verifier keeps only the rows the trunk agrees with)."""
from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import deque

from engine import Hy4Engine, PRIME_ROWS, warmup_clamp


# HY4_MU_HOSTFAST=1 (default OFF): the batched host path of a decode round (see MuEngine._hcur_set, plain_round, spec_round, step_mu.step_slots).
# HY4_MU_HOSTPROF=1: perf_counter splits per round, one `MUFAST` line per 100 rounds in the server log (also the FIRES-CHECK path counters).
FIRES = {"plain": 0, "spec": 0, "hcur_idx": 0, "bind_skip": 0, "draft_cat": 0}
HT = {"rounds": 0, "rows": 0, "draft": 0.0, "wait": 0.0, "post": 0.0, "emit": 0.0, "round": 0.0}


def _fast() -> bool:
    return os.environ.get("HY4_MU_HOSTFAST", "0") not in ("", "0")


def _prof() -> bool:
    return os.environ.get("HY4_MU_HOSTPROF", "0") not in ("", "0")


def fires_line(eng, last: dict | None = None) -> str:
    """one log line: path counters (cumulative) + mean ms per decode round since the previous line (`last` = the previous snapshot, updated in place)"""
    sm = getattr(eng, "SM", None)
    fs = dict(FIRES)
    ht = dict(HT)
    if sm is not None:
        fs.update(sm.FIRES)
        ht.update({"stage": sm.HT["stage"], "launch": sm.HT["launch"], "inputs": sm.HT["inputs"]})
    last = {} if last is None else last
    nr = max(1, ht["rounds"] - last.get("rounds", 0))
    ms = {k: 1000.0 * (ht.get(k, 0.0) - last.get(k, 0.0)) / nr for k in ("round", "inputs", "stage", "launch", "draft", "wait", "post", "emit")}
    rows = (ht["rows"] - last.get("rows", 0)) / nr
    last.clear()
    last.update(ht)
    return ("MUFAST rounds=%d fast=%d " % (ht["rounds"], int(_fast())) + " ".join(f"{k}={v}" for k, v in fs.items()) +
            " | ms/round " + " ".join(f"{k}={v:.2f}" for k, v in ms.items()) + f" rows/round={rows:.2f}")


def _stage(name):
    """HY4_MU_SYNC_DEBUG=1: log the stage and sync the device when it ends (loader/mu_debug.py); a null context otherwise"""
    if os.environ.get("HY4_MU_SYNC_DEBUG", "0") in ("", "0"):
        return contextlib.nullcontext()
    from exllamav3.loader import mu_debug
    return mu_debug.stage(name)


class MuEngine(Hy4Engine):
    def __init__(self, *a, slots: int = 4, **kw):
        for f in ("HY4_SWA_RING", "HY4_LC_SPARSE"):        # long-context levers act on the single-session K / indexer buffers, not on the slot slabs
            if os.environ.get(f, "0") not in ("", "0"):
                raise RuntimeError(f + " is not supported with --slots > 1")
        super().__init__(*a, **kw)
        self.slots = slots

    def _decoder(self, M):
        from exllamav3.loader import step_mu
        self.SM = step_mu
        return step_mu.MultiDecoder(M, self.slots)

    def _mtp_graph(self):
        self.GR.attach_mtp(self.MT)
        return self.SM.MTPGraphMU(self.MT, self.k)

    # ------------------------------------------------------------------------------------------ sessions
    def open(self, slot):
        return self.GR.new_session(slot)

    def prefill_begin(self, s, ids, chunk):
        s.ids, s.pf_pos, s.H, s.chunk = list(ids), 0, [], min(chunk, PRIME_ROWS)

    def prefill_step(self, s):
        """one chunk of the prompt of s (eager prefill set) + the draft-head cache rows of that chunk (so no single long stall at the end of the prompt).
        -> True when the prompt is done: s.y = first token, s.hcur = trunk state of the last prompt row"""
        torch, M, GR, MT = self.torch, self.M, self.GR, self.MT
        a = s.pf_pos
        n = len(s.ids)
        b = min(n, a + s.chunk)
        self.phase("prefill")
        GR.bind(s)
        if a == 0:
            M.prefetch_engram(s.ids, s.chunk)      # the row prefetch list is global: one prefill is in flight at a time, it starts with its first chunk
            s.tok_dev = torch.tensor(s.ids, dtype=torch.long, device=self.dev)
        with _stage(f"prefill trunk step slot {s.slot} rows {a}:{b} of {n}"):
            lg = M.step(s.ids[a:b])
            hs = M.head_state(M._x)
        M._x = None
        e_end = min(b, n - 1)                       # head pairs (token j + 1, trunk state h_j) for j = a .. min(b, n - 1) - 1
        if e_end > a:
            f16 = os.environ.get("HY4_PF_F16")
            os.environ["HY4_PF_F16"] = "0"          # the fp16 attention chain is tuned for the trunk's matrices; the draft layer takes the generic prefill path
            try:
                with _stage(f"prefill draft prime slot {s.slot} rows {a}:{e_end}"):
                    MT.prime(s.tok_dev[a + 1:e_end + 1], hs[:e_end - a], MT.ar[a:e_end], ((e_end + 3) // 4) * 4)
            finally:
                if f16 is None:
                    os.environ.pop("HY4_PF_F16", None)
                else:
                    os.environ["HY4_PF_F16"] = f16
        torch.cuda.synchronize()
        GR.save(s)
        s.pf_pos = b
        if b < n:
            return False
        s.y = int(torch.argmax(lg))
        s.hcur = hs[-1:].clone()
        s.tok_dev = None
        GR.start_graph_state(s)
        self.phase("decode")
        return True

    # ------------------------------------------------------------------------------------------ decode rounds
    def plain_round(self, sess):
        """one row per session -> {session: [new token]}"""
        if _fast():
            return self._plain_round_fast(sess)
        rows = [(s, [s.y]) for s in sess]
        _lg, am, _x, hv = self.GR.step_slots(rows)
        amh = am.reshape(-1).tolist()
        out = {}
        for r, s in enumerate(sess):
            s.y = amh[r]
            with _stage(f"plain_round hcur clone slot {s.slot} row {r}"):
                s.hcur = hv[r:r + 1].clone()           # a later spec_round (fewer sessions decode) drafts from the trunk state of the last fed position and the n-gram history
            if s.ng is not None:
                s.ng.add([s.y])
            out[s] = [s.y]
        return out

    # ---- HY4_MU_HOSTFAST: batched host work of a round
    def _hcur_set(self, sess, hv, rows=None):
        """s.hcur of every session in ONE indexed device copy into a preallocated [slots, D] buffer (no per-slot clone); s.hcur is then a [1, D] view of the session's row.
        hv = the step's trunk states [T, D]; rows = the row of hv each session continues from (None: row r of session r, T == len(sess))."""
        torch = self.torch
        D = hv.shape[-1]
        hb = getattr(self, "_hbuf", None)
        if hb is None:
            dev = hv.device
            hb = self._hbuf = torch.zeros(self.slots, D, dtype=hv.dtype, device=dev)
            self._hviews = [hb[i:i + 1] for i in range(self.slots)]
            self._sidx = {}
            n = max(self.slots, 16)
            self._rpin = torch.zeros(n, dtype=torch.long).pin_memory() if dev.type == "cuda" else torch.zeros(n, dtype=torch.long)
            self._rdev = torch.zeros(n, dtype=torch.long, device=dev)
        key = tuple(s.slot for s in sess)
        idx = self._sidx.get(key)
        if idx is None:                                 # once per distinct slot tuple (a blocking pageable H2D): never in a steady round
            idx = self._sidx[key] = torch.tensor(key, dtype=torch.long, device=hv.device)
        src = hv.reshape(-1, D)
        if rows is None:
            src = src[:len(key)]
        else:
            n = len(rows)
            self._rpin[:n] = torch.tensor(rows, dtype=torch.long)
            self._rdev[:n].copy_(self._rpin[:n], non_blocking=True)      # the previous copy of this pinned slice was enqueued before the last readback (a sync): safe to overwrite
            src = src.index_select(0, self._rdev[:n])
        hb.index_copy_(0, idx, src)
        views = self._hviews
        for s in sess:
            s.hcur = views[s.slot]
        FIRES["hcur_idx"] += 1

    @staticmethod
    def _ng_flush(s):
        """append the tokens of the previous round to the n-gram history (deferred out of the round's critical path)"""
        p = getattr(s, "ng_pend", None)
        if p and s.ng is not None:
            s.ng.add(p)
            p.clear()

    def _plain_round_fast(self, sess):
        pc = time.perf_counter
        prof = _prof()
        t0 = pc()
        rows = [(s, [s.y]) for s in sess]
        _lg, am, _x, hv = self.GR.step_slots(rows)           # launched, the GPU runs from here
        for s in sess:
            self._ng_flush(s)                                # history appends of the previous round overlap the replay
        self._hcur_set(sess, hv)                             # stream-ordered after the replay: no host wait
        t1 = pc()
        amh = am.reshape(-1).tolist()                        # the ONE readback
        t2 = pc()
        out = {}
        for r, s in enumerate(sess):
            y = amh[r]
            s.y = y
            if s.ng is not None:
                p = getattr(s, "ng_pend", None)
                if p is None:
                    p = s.ng_pend = []
                p.append(y)
            out[s] = [y]
        FIRES["plain"] += 1
        if prof:
            t3 = pc()
            HT["wait"] += t2 - t1
            HT["post"] += t3 - t2
        return out


    def _ng(self, s):
        if s.ng is None and self.ngram_ext > 0:
            from ngram import NGram
            s.ng = NGram(self.ngram_min, self.ngram_ext)
            s.ng.reset(list(s.ids) + [s.y])
            s.extra.update(rate=0.7, since=0)
        return s.ng

    def spec_round(self, sess):
        """speculative round of every session in sess: drafts per session (own head state), ONE verify replay over all rows. -> {session: [new tokens]}
        HY4_MU_HOSTFAST: no bind where the (slot, bucket) draft graph exists, ONE readback of all drafts, hcur by one indexed copy, history appends deferred"""
        torch, GR, MG, MT, k = self.torch, self.GR, self.MG, self.MT, self.k
        fast = _fast()
        prof = _prof()
        pc = time.perf_counter
        t0 = pc()
        TM = self.SG.TM
        Rs = max(1, TM // len(sess))               # verify rows per session
        ks = min(k, Rs - 1)
        plan = []
        for s in sess:
            if Rs == 1:
                plan.append((s, [s.y], False, 0, None))
                continue
            j = s.n - 1
            ng = self._ng(s)
            if fast:
                self._ng_flush(s)
            skip, cl = False, []
            if ng is not None and self.ngram_long > 0:
                cl, ml = ng.propose([], Rs - 1)
                skip = bool(cl) and ml >= self.ngram_long
            if skip:
                plan.append((s, [s.y] + cl, True, j, s.hcur if fast else s.hcur.clone()))       # fast: s.hcur is only replaced after the round
                continue
            cap = getattr(MG, "captured", None)
            if fast and cap is not None and cap(s.slot, j):
                FIRES["bind_skip"] += 1                # a replay of a captured (slot, bucket) graph reads no eager state
            else:
                GR.bind(s)
            MG.slot = s.slot
            with _stage(f"spec_round draft chain slot {s.slot} j {j}"):
                plan.append((s, None, False, j, MG.run(s.y, s.hcur, j)))      # drafts launched; read below
        drafts = {}
        if fast:
            pend = [p for p in plan if p[1] is None]
            if pend:
                flat = torch.cat([p[4] for p in pend]).tolist()               # ONE readback for the drafts of every session
                L = len(flat) // len(pend)
                for i, p in enumerate(pend):
                    drafts[p[0]] = flat[i * L:i * L + ks]
                FIRES["draft_cat"] += 1
        rows, meta = [], []
        for s, toks, skip, j, h in plan:
            if toks is None:
                d = drafts[s] if fast else h.tolist()[:ks]
                ext = []
                ng = s.ng
                if ng is not None:
                    lim = min(self.ngram_ext, Rs - 1 - ks)
                    if self.ngram_gate:
                        st = s.extra
                        st["since"] += 1
                        rate, since = st["rate"], st["since"]
                        lim = lim if rate >= 0.6 else (2 if rate >= 0.4 else (lim if since >= 8 else 0))
                    ext, ml = ng.propose(d, lim) if lim > 0 else ([], 0)
                    if self.ngram_adapt and ext:
                        ext = ext[:max(1, (ml - self.ngram_min + 1) * 2)]
                toks = [s.y] + d + ext
                meta.append((s, toks, False, j, len(ext), ks))
            else:
                meta.append((s, toks, skip, j, 0, ks))
            rows.append((s, toks))
        t1 = pc()
        starts = {s: s.n for s, _ in rows}
        _lg, amv, _x, hv = GR.step_slots(rows)
        t2 = pc()
        amh_all = amv.reshape(-1).tolist()
        t3 = pc()
        out, o = {}, 0
        accrow = []
        for s, toks, skip, j, next_, kk in meta:
            T = len(toks)
            amh = amh_all[o:o + T]
            hs = hv[o:o + T]
            acc, nd = 0, T - 1
            while acc < nd and toks[acc + 1] == amh[acc]:
                acc += 1
            st = s.extra
            if next_ and not skip:
                st["rate"] = 0.75 * st["rate"] + 0.25 * (max(0, acc - kk) / next_)
                st["since"] = 0
            new = toks[1:acc + 1] + [amh[acc]]
            GR.rewind(s, starts[s] + 1 + acc)
            if skip:                                   # the chain wrote no head rows: pairs (token m, state before it) for head positions j .. j + acc
                GR.bind(s)
                N = ((j + acc + 1 + 3) // 4) * 4
                sts = torch.cat([skip_h(plan, s), hs[:acc]]) if acc > 0 else skip_h(plan, s)
                with _stage(f"spec_round draft catch-up (skip) slot {s.slot} j {j} acc {acc} N {N}"):
                    MT.step(torch.tensor(toks[:acc + 1], dtype=torch.long, device=self.dev), sts, MT.ar[j:j + acc + 1], N)
            elif self.ngram_fill and acc >= kk and kk > 0:
                GR.bind(s)
                N = ((j + acc + 1 + 3) // 4) * 4
                with _stage(f"spec_round draft catch-up (fill) slot {s.slot} j {j} acc {acc} N {N}"):
                    MT.step(torch.tensor(toks[kk:acc + 1], dtype=torch.long, device=self.dev), hs[kk - 1:acc], MT.ar[j + kk:j + acc + 1], N)
            if s.ng is not None:
                if fast:
                    p = getattr(s, "ng_pend", None)
                    if p is None:
                        p = s.ng_pend = []
                    p.extend(new)
                else:
                    s.ng.add(new)
            if fast:
                s.y = amh[acc]
                accrow.append(o + acc)
            else:
                s.y, s.hcur = amh[acc], hs[acc:acc + 1].clone()
            o += T
            s.extra["hist"] = s.extra.get("hist", 0) + 1
            s.extra["acc_sum"] = s.extra.get("acc_sum", 0) + acc
            out[s] = new
        if fast:
            self._hcur_set([m[0] for m in meta], hv, accrow)
            FIRES["spec"] += 1
        if prof:
            t4 = pc()
            HT["draft"] += t1 - t0
            HT["wait"] += t3 - t2
            HT["post"] += t4 - t3
        return out


def skip_h(plan, s):
    for p in plan:
        if p[0] is s:
            return p[4]
    raise KeyError


class Request:
    def __init__(self, ids, max_new, speculative=True, on_tokens=None):
        self.ids, self.max_new, self.speculative = list(ids), max_new, speculative
        self.on_tokens = on_tokens
        self.out: list[int] = []
        self.times: list[tuple[float, int]] = []      # (time, tokens emitted by this event)
        self.t_submit = time.perf_counter()
        self.t_first = self.t_done = None
        self.done = threading.Event()
        self.cancel = threading.Event()               # set by the server when the client went away
        self.on_done = None                           # called once from the scheduler thread when the request ends (any reason)
        self.error = None
        self.sess = None
        self.stats: dict = {}
        self.rounds = 0

    def engine_stats(self, k):
        """the stats dict the single-session engine returns from generate() (what serve.py prints / reports as timings)"""
        t_first = self.t_first or self.t_done
        dec = max((self.t_done or t_first) - t_first, 1e-9)
        n = max(len(self.out) - 1, 0)
        return {"prompt_tokens": len(self.ids), "speculative": self.speculative, "k": k, "prefill_s": t_first - self.t_submit, "ttft_s": t_first - self.t_submit,
                "decode_s": dec, "decode_tokens": n, "rounds": self.rounds, "accept_hist": [], "tau": (n / self.rounds) if self.rounds else None,
                "prefill_trunk_s": self.stats.get("pf_s", 0.0), "prime_s": 0.0, "generated": len(self.out), "cancelled": self.cancel.is_set()}


class MuScheduler(threading.Thread):
    """admit -> prefill chunk (oldest prefilling) -> one decode round of all decoding sessions -> repeat"""

    def __init__(self, eng: MuEngine, chunk: int = 1024, spec: bool = True, spec_max_b: int | None = None):
        super().__init__(daemon=True)
        self.eng, self.chunk, self.spec = eng, chunk, spec
        # cost policy (box A/B 19/11): a verify row costs about what a new session's row costs (+7.7 ms), so speculation only pays while rows are shared among few sessions
        self.spec_max_b = int(os.environ.get("HY4_MU_SPEC_MAXB", "2")) if spec_max_b is None else spec_max_b
        self.cv = threading.Condition()
        self.pending: deque = deque()
        self.free = list(range(eng.slots))
        self.pin = int(os.environ.get("HY4_MU_SLOT_PIN", "-1"))      # HY4_MU_SLOT_PIN=k: slot k is allocated first whenever it is free (tests one slot index: a single session always lands on it)
        if self.pin >= eng.slots:
            raise ValueError(f"HY4_MU_SLOT_PIN={self.pin} is past --slots {eng.slots}")
        self.pf: deque = deque()
        self.dec: list = []
        self.stop_flag = False
        self.stats = {"rounds": 0, "round_rows": 0, "pf_chunks": 0, "pf_s": 0.0, "dec_s": 0.0}
        self.trace: list = []                       # (t0, t1, kind, n) per scheduler action

    def _take_slot(self):
        if self.pin in self.free:
            self.free.remove(self.pin)
            return self.pin
        return self.free.pop(0)

    def submit(self, req: Request):
        with self.cv:
            self.pending.append(req)
            self.cv.notify()
        return req

    def shutdown(self):
        with self.cv:
            self.stop_flag = True
            self.cv.notify()

    def _emit(self, req, new, now):
        room = req.max_new - len(req.out)
        new = new[:room]
        req.out += new
        req.times.append((now, len(new)))
        stop = bool(req.on_tokens(new)) if req.on_tokens else False
        return stop or len(req.out) >= req.max_new or req.cancel.is_set()

    def _finish(self, req):
        req.t_done = time.perf_counter()
        if req.sess is not None:
            self.free.append(req.sess.slot)
        if req in self.dec:
            self.dec.remove(req)
        req.done.set()
        if req.on_done is not None:
            req.on_done()

    def run(self):
        eng, pc = self.eng, time.perf_counter
        prof, snap = _prof(), {}
        try:
            eng.phase("decode")
            while True:
                with self.cv:
                    while not (self.pending or self.pf or self.dec) and not self.stop_flag:
                        self.cv.wait()
                    if self.stop_flag:
                        return
                    while self.pending and self.free:
                        req = self.pending.popleft()
                        req.sess = eng.open(self._take_slot())
                        req.stats["admit"] = pc()
                        if req.cancel.is_set() or len(req.ids) + req.max_new + eng.k + 2 > eng.ctx:
                            req.error = None if req.cancel.is_set() else "context"
                            self._finish(req)
                            continue
                        eng.prefill_begin(req.sess, req.ids, self.chunk)
                        req.sess.req = req
                        self.pf.append(req)
                if self.pf and self.pf[0].cancel.is_set():
                    self._finish(self.pf.popleft())
                if self.pf:
                    req = self.pf[0]
                    t0 = pc()
                    done = eng.prefill_step(req.sess)
                    t1 = pc()
                    req.stats["pf_s"] = req.stats.get("pf_s", 0.0) + t1 - t0
                    self.stats["pf_chunks"] += 1
                    self.stats["pf_s"] += t1 - t0
                    self.trace.append((t0, t1, "pf", req.sess.pf_pos))
                    if done:
                        self.pf.popleft()
                        req.t_first = t1
                        req.stats["ttft"] = t1 - req.t_submit
                        if self._emit(req, [req.sess.y], t1):
                            self._finish(req)
                        else:
                            self.dec.append(req)
                if self.dec:
                    sess = [r.sess for r in self.dec]
                    t0 = pc()
                    spec = self.spec and len(self.dec) <= self.spec_max_b and all(r.speculative for r in self.dec)
                    out = eng.spec_round(sess) if spec else eng.plain_round(sess)
                    t1 = pc()
                    self.stats["rounds"] += 1
                    self.stats["round_rows"] += len(sess)
                    self.stats["dec_s"] += t1 - t0
                    self.trace.append((t0, t1, "dec", len(sess)))
                    for r in list(self.dec):
                        r.rounds += 1
                        if self._emit(r, out[r.sess], t1):
                            self._finish(r)
                    if prof:
                        HT["rounds"] += 1
                        HT["rows"] += len(sess)
                        HT["round"] += t1 - t0
                        HT["emit"] += pc() - t1
                        if HT["rounds"] % 100 == 0:
                            print(fires_line(eng, snap), flush=True)
        except Exception as ex:           # noqa: BLE001
            import traceback
            traceback.print_exc()
            for r in list(self.dec) + list(self.pf) + list(self.pending):
                r.error = repr(ex)
                r.done.set()
                if r.on_done is not None:
                    r.on_done()


def warmup(sch: MuScheduler, n: int, new: int = 24):
    """compile / capture what a served mix needs at the first bucket: B = 1 .. slots concurrent requests of n prompt tokens (graphs per (bucket, rows), MTP chains per slot)"""
    n = warmup_clamp(n, sch.eng.ctx, sch.eng.k, new)
    if n <= 0:
        return
    eng = sch.eng
    ids = [(i * 7919 + 13) % 50000 for i in range(n)]
    for B in range(1, eng.slots + 1):
        reqs = [sch.submit(Request(ids[i:] + ids[:i], new)) for i in range(B)]
        for r in reqs:
            r.done.wait(900)
    print(f"mu warm-up: {n} prompt tokens, 1..{eng.slots} sessions", flush=True)


def precapture(sch: MuScheduler, nbs=None, ts=None, progress=print):
    """HY4_MU_PRECAP=512,1024 (context buckets): capture every (bucket, rows) trunk graph for rows in 1..TM and every (slot, bucket) draft graph NOW, before any timing, and print
    the replay curve `MUCURVE Nb T ms` (one timed replay each; rows of one dummy session on the last slot, which a real request re-zeroes on admission). Call before the server is ready:
    the scheduler must be idle. Rows above 8 run the fallback kernels (mla_bgemv / router_gemv keep their MAXT = 8 guards, mu_slots MAX_T = 16): the curve shows their cost."""
    eng, torch = sch.eng, sch.eng.torch
    GR, MG = eng.GR, eng.MG
    TM = eng.SG.TM
    if nbs is None:
        nbs = [int(x) for x in os.environ.get("HY4_MU_PRECAP", "512").replace("+", ",").split(",") if x.strip()]
    ts = list(ts) if ts is not None else list(range(1, TM + 1))
    pc = time.perf_counter
    t00 = pc()
    s = GR.new_session(eng.slots - 1)
    for Nb in nbs:
        for T in ts:
            base = Nb - T                                   # top = Nb: the rows land in bucket Nb
            for rep in range(3):
                GR.rewind(s, 0)
                s.tokens = [0] * base
                s.n = base
                t = pc()
                GR.step_slots([(s, [1] * T)])
                torch.cuda.synchronize()
                ms = (pc() - t) * 1000
                if rep == 0:
                    cap_ms = ms
            progress(f"MUCURVE Nb={Nb} T={T} ms={ms:.1f} capture_or_first_ms={cap_ms:.0f}")
    GR.rewind(s, 0)
    nd = 0
    for slot in range(eng.slots):                           # the draft chain graph is keyed (slot, bucket): every slot may serve the first request of a cell
        s2 = GR.new_session(slot)
        GR.bind(s2)
        MG.slot = slot
        for Nb in nbs:
            if (slot, Nb) not in MG.graphs:
                MG._graph(Nb)
                nd += 1
    free = None
    try:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
    except Exception:       # noqa: BLE001
        pass
    progress(f"mu precapture: buckets {nbs} rows {ts[0]}..{ts[-1]}: {len(nbs) * len(ts)} trunk graphs, {nd} draft graphs, {pc() - t00:.1f} s, free GiB {free}")
