# draft policy for Qwen MTP speculation, installed on a Generator without engine edits.
#  (a) dynamic draft length from the drafter's own top-1 probability: a forward at depth i only if the running
#      product reach_{i-1} >= thf; token i kept only if reach_i >= thv (depth 1 always kept); depth <= maxd
#  (c) n-gram lookup: when the context suffix has an earlier match of length >= lmin, draft = its continuation
#      (up to ngk tokens); one MTP forward still runs so the drafter cache has no hole at the current position
# Rows per verify step stay <= maxd + 1 (<= 4: the range proven for the fused MoE kernel).
import os, sys
from array import array
import numpy as np
import torch

class StopDraft(Exception):
    pass

def ngram_match(ctx, nmin, K):
    n = len(ctx); suf = ctx[n - nmin:]
    best = (0, [])
    if n < nmin + 1: return best
    for j in range(n - 1, nmin - 1, -1):
        if ctx[j - nmin:j] == suf:
            L = nmin
            while L < 32 and j - L - 1 >= 0 and ctx[j - L - 1] == ctx[n - L - 1]: L += 1
            if L > best[0]: best = (L, ctx[j:j + K])
            if L >= 16: break
    return best

# ---------------------------------------------------------------------------------------------- incremental index
# ngram_match above scans the whole context every round (host time grows with depth: 7.7 ms at 128K).
# NgramIndex returns exactly the same (L, continuation) with a cost per round that does not depend on the context.
# What the old function returns, in index terms (j = end position of an earlier gram, n = context length):
#   - candidates: j in [nmin, n-1] whose gram ctx[j-nmin:j] equals the last nmin tokens;
#   - L_j = longest common suffix of ctx[:j] and ctx[:n], capped at 32 (and at j);
#   - if some candidate has L_j >= 16 the answer is the LARGEST such j (the break), else the largest L_j, ties -> largest j.
# Per order m in nmin..16 the index keeps latest[m][key] (newest end position of each m-gram) and prev[m][e] (the
# previous end position with the same key as the gram ending at e). prev[m][n] is then "the newest j <= n-1 whose
# m-gram equals the suffix", with no lookup at query time. Order 16 first, then down: the first order with a hit is
# the longest match; L is exact (extended to 32 for order 16). Rollback: truncation walks the removed positions
# newest first and restores latest[m] from prev[m] (exact inverse of the pushes).
# Keys are 62-bit double polynomial hashes; every hit is verified against the tokens and a failed check (a hash
# collision, probability ~1e-9 per hit) falls back to the old scan, so the result is exact either way.
# Memory: 16 key arrays (8 B) + 15 prev arrays (4 B) + 4 B tokens = ~190 B per token, plus the dicts of the
# newest position per distinct gram (<= 15 x ~100 B per token for text with few repeats): about 300 MB at 192K tokens.
_M1, _M2, _B1, _B2, _MASK = 2147483629, 2147483587, 1000003, 999983, (1 << 31) - 1
LMAX, LCAP = 16, 32

class NgramIndex:
    def __init__(self, nmin, lmax=LMAX):
        assert 1 <= nmin <= lmax
        self.nmin, self.lmax = nmin, lmax
        self.toks = array("i")
        self.key = [None] + [array("q", [-1]) for _ in range(lmax)]
        self.prev = [None] + [array("i", [-1]) if m >= nmin else None for m in range(1, lmax + 1)]
        self.latest = [None] + [{} if m >= nmin else None for m in range(1, lmax + 1)]
        self.fallbacks = 0
        self._hit = None

    def __len__(self):
        return len(self.toks)

    def push(self, t):
        e = len(self.toks) + 1; v = t + 1
        self.toks.append(t)
        key, prev, latest, nmin = self.key, self.prev, self.latest, self.nmin
        for m in range(1, self.lmax + 1):
            if e < m:
                key[m].append(-1)
                if m >= nmin: prev[m].append(-1)
                continue
            if m == 1: a1 = a2 = 0
            else:
                k = key[m - 1][e - 1]; a1 = k >> 31; a2 = k & _MASK
            k = (((a1 * _B1 + v) % _M1) << 31) | ((a2 * _B2 + v) % _M2)
            key[m].append(k)
            if m >= nmin:
                d = latest[m]; prev[m].append(d.get(k, -1)); d[k] = e

    def truncate(self, p):
        n = len(self.toks)
        assert 0 <= p <= n
        if p == 0 and n:
            # a whole new context (a new conversation): the walk below undoes every push one by one (11-23 ms at 8K tokens, inside the
            # time to first token) and ends in the freshly built state, so build that state directly (checked by test_ngram_index)
            lmax, nmin = self.lmax, self.nmin
            self.toks = array("i")
            self.key = [None] + [array("q", [-1]) for _ in range(lmax)]
            self.prev = [None] + [array("i", [-1]) if m >= nmin else None for m in range(1, lmax + 1)]
            self.latest = [None] + [{} if m >= nmin else None for m in range(1, lmax + 1)]
            return
        key, prev, latest = self.key, self.prev, self.latest
        for e in range(n, p, -1):
            for m in range(self.nmin, self.lmax + 1):
                k = key[m][e]
                if k >= 0:
                    pv = prev[m][e]
                    if pv >= 0: latest[m][k] = pv
                    else: del latest[m][k]
        for m in range(1, self.lmax + 1):
            del key[m][p + 1:]
            if m >= self.nmin: del prev[m][p + 1:]
        del self.toks[p:]
        if n - p > max(4096, p):
            # a dict never gives memory back after deletes: copy the survivors so a long request does not pin ~300 MB for the
            # rest of the process life when the next prompt is short
            for m in range(self.nmin, self.lmax + 1):
                latest[m] = dict(latest[m])

    def extend(self, ids):
        """Append token ids (numpy int array). Small batches push one by one, large ones are built vectorised
        (same state as pushing, checked by the proof tests)."""
        d = len(ids)
        if d < 48:
            for t in ids.tolist(): self.push(t)
            return
        a = len(self.toks)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        v = ids + 1
        epos = np.arange(a + 1, a + d + 1, dtype=np.int64)
        for m in range(1, self.lmax + 1):
            if m == 1: pk = np.zeros(d, dtype=np.int64)
            else: pk = np.frombuffer(self.key[m - 1], dtype=np.int64)[a:a + d].copy()
            h1 = (((pk >> 31) * _B1) + v) % _M1
            h2 = (((pk & _MASK) * _B2) + v) % _M2
            k = (h1 << 31) | h2
            valid = epos >= m
            k[~valid] = -1
            self.key[m].frombytes(k.tobytes())
            if m < self.nmin: continue
            dct = self.latest[m]
            pv = np.full(d, -1, dtype=np.int32)
            kv = k[valid]; ev = epos[valid]
            if len(kv):
                order = np.argsort(kv, kind="stable")
                sk = kv[order]; se = ev[order]
                first = np.empty(len(sk), dtype=bool); first[0] = True; first[1:] = sk[1:] != sk[:-1]
                pin = np.empty(len(sk), dtype=np.int64); pin[0] = -1; pin[1:] = se[:-1]
                fk = sk[first].tolist()
                pin[first] = np.array([dct.get(x, -1) for x in fk], dtype=np.int64)
                pv[se - (a + 1)] = pin
                last = np.empty(len(sk), dtype=bool); last[-1] = True; last[:-1] = first[1:]
                dct.update(zip(sk[last].tolist(), se[last].tolist()))
            self.prev[m].frombytes(pv.tobytes())
        self.toks.frombytes(ids.astype(np.int32).tobytes())

    def match(self, K):
        """Same value as ngram_match(ctx, nmin, K) on ctx = the indexed tokens."""
        toks = self.toks; n = len(toks); nmin = self.nmin
        self._hit = None
        if n < nmin + 1: return (0, [])
        prev = self.prev
        for m in range(self.lmax, nmin - 1, -1):
            j = prev[m][n]
            if j >= 0: break
        else:
            return (0, [])
        L = m
        ok = toks[j - m:j] == toks[n - m:n]
        if ok:
            if m == self.lmax:
                while L < LCAP and j - L - 1 >= 0 and toks[j - L - 1] == toks[n - L - 1]: L += 1
            else:
                ok = not (j - m - 1 >= 0 and toks[j - m - 1] == toks[n - m - 1])
        if not ok:
            self.fallbacks += 1
            return ngram_match(toks.tolist(), nmin, K)
        self._hit = (j, m)
        return (L, toks[j:j + K].tolist())

    def witness_len(self, cont):
        """After match(): how many leading tokens of its continuation the PREVIOUS occurrence of the matched gram also
        continues with. len(cont) when there is no previous occurrence (a single witness cannot be checked)."""
        if self._hit is None: return len(cont)
        j, m = self._hit; toks = self.toks
        j2 = self.prev[m][j]
        if j2 < m or toks[j2 - m:j2] != toks[j - m:j]: return len(cont)
        a = 0
        while a < len(cont) and toks[j2 + a] == cont[a]: a += 1
        return a

    def sync_array(self, ids):
        """Make the index equal to ids (numpy int array): keep the longest common prefix, roll back, add the rest."""
        have = np.frombuffer(self.toks, dtype=np.int32)
        m = min(len(ids), len(have))
        neq = np.flatnonzero(ids[:m] != have[:m])
        p = int(neq[0]) if len(neq) else m
        del have
        if p < len(self.toks): self.truncate(p)
        if len(ids) > p: self.extend(ids[p:])
        return p

def _as_np(t):
    return t.flatten().numpy().astype(np.int64, copy=False)

class SeqFollower:
    """Keeps one NgramIndex equal to a SeqTensor of token ids. The tensor changes only by append and truncate
    (job.py: pinned_ids_valid relies on the same fact); truncate/clear are wrapped on the instance to keep a low-water
    mark, so the prefix below it is known unchanged. A new tensor (next request, loop-guard restart) is matched by
    common prefix, so multi-turn reuse costs the new tokens only. The last tokens are re-checked each sync as a belt."""
    TAIL = 16
    def __init__(self, nmin):
        self.idx = NgramIndex(nmin); self.sid = None; self.resyncs = 0

    def _hook(self, sid):
        sid._ng_lw = len(sid)
        o_tr, o_cl = sid.truncate, sid.clear
        def tr(nl):
            sid._ng_lw = min(sid._ng_lw, nl); o_tr(nl)
        def cl():
            sid._ng_lw = 0; o_cl()
        sid.truncate, sid.clear = tr, cl

    def sync(self, sid):
        idx = self.idx; n = len(sid)
        if sid is not self.sid:
            if n: idx.sync_array(_as_np(sid.torch()))
            else: idx.truncate(0)
            if not hasattr(sid, "_ng_lw"): self._hook(sid)
            self.sid = sid; sid._ng_lw = n
            return idx
        p = min(sid._ng_lw, len(idx), n)
        if p < len(idx): idx.truncate(p)
        t = min(self.TAIL, len(idx))
        if t and not np.array_equal(_as_np(sid.torch_slice(len(idx) - t, len(idx))), np.frombuffer(idx.toks, dtype=np.int32)[len(idx) - t:]):
            self.resyncs += 1
            idx.sync_array(_as_np(sid.torch()))
        elif n > len(idx):
            idx.extend(_as_np(sid.torch_slice(len(idx), n)))
        sid._ng_lw = n
        return idx

# ---------------------------------------------------------------------------------------------- depth rules
# A rule decides, per drafted depth, from the drafter's max token probability p (the "reach" is the running product):
#   step(i, p) -> (keep, more): keep = this depth's token goes to the verify; more = draft one more depth.
# Greedy verification makes the generated ids independent of the rule; only the speed changes.
# ProductRule with (0.6, 0.3, maxd) is the shipped rule. CostRule / OnlineRule see a calibrated reach (see rule_from_env).
class ProductRule:
    name = "product"
    def __init__(self, thf=0.6, thv=0.3, maxd=3):
        self.thf, self.thv, self.maxd = thf, thv, maxd
        self.reach = 1.0
    def begin(self): self.reach = 1.0
    def q(self, i, p):
        """probability estimate that depths 0..i are all accepted (updates the running reach)"""
        self.reach *= p
        return self.reach
    def step(self, i, p):
        q = self.q(i, p)
        if i > 0 and q < self.thv: return False, False
        return True, not (q < self.thf or i + 1 >= self.maxd)
    def end_round(self, drafted, accepted): pass

class TierRule(ProductRule):
    """ProductRule for rows <= dsplit + 1 (identical to the shipped rule there), stricter beyond: a draft deeper than dsplit
    is kept only if its reach >= thvd, and drafting goes past depth dsplit only if the reach so far >= thfd.
    Why: one pair (thf, thv) serves two jobs. Raising thf to 0.8 to avoid unprofitable 5+ row rounds also cut the
    2..4 row rounds short, which cost 1-4 % where depth 2-4 acceptance is good (128K, chat). Verify rows cost the same at every
    context (~10.5 ms per row incl. its draft step); a row beyond the 4th pays only when its chance of being accepted is above
    ~0.5, and the drafter's reach overstates it, so the deep tier asks for reach >= 0.95 to start and >= 0.85 to keep.
    Measured on the served model: 0.9 / 0.7 lost about 1 % at 32K to deep rounds that did not pay; 0.95 / 0.85 matches the
    former 4-row rule there and keeps the gain on chat, prose, code and at 128K."""
    name = "tier"
    def __init__(self, thf=0.6, thv=0.3, maxd=7, thfd=0.95, thvd=0.85, dsplit=3):
        super().__init__(thf, thv, maxd); self.thfd, self.thvd, self.dsplit = thfd, thvd, dsplit
    def step(self, i, p):
        q = self.q(i, p); k = i + 1
        if i > 0 and q < (self.thv if k <= self.dsplit else self.thvd): return False, False
        return True, not (q < (self.thf if k < self.dsplit else self.thfd) or k >= self.maxd)

class CostRule(ProductRule):
    """Same shape, calibrated reach: q = min(1, scale * reach ** gamma). thv = cost of one verify row / ms per token
    (keep a row when its chance to add a token pays its ~row_ms), thf = the same test one depth ahead (drafting also costs)."""
    name = "cost"
    def __init__(self, thf, thv, maxd, gamma=1.0, scale=1.0):
        super().__init__(thf, thv, maxd); self.gamma, self.scale = gamma, scale
        self.raw = 1.0
    def begin(self): self.raw = 1.0
    def q(self, i, p):
        self.raw *= p
        return min(1.0, self.scale * self.raw ** self.gamma)

class OnlineRule(CostRule):
    """CostRule whose scale follows the running acceptance of the current request: ratio of accepted drafts to the
    sum of predicted per-depth reach over the last rounds (EMA, clipped), so a request that accepts more than the
    drafter claims drafts deeper and the reverse. No prompt-class knowledge."""
    name = "online"
    def __init__(self, thf, thv, maxd, gamma=1.0, alpha=0.15, lo=0.5, hi=1.6):
        super().__init__(thf, thv, maxd, gamma, 1.0)
        self.alpha, self.lo, self.hi = alpha, lo, hi
        self.pred_ema = None; self.acc_ema = None; self.pred = 0.0
    def new_request(self): self.pred_ema = None; self.acc_ema = None; self.scale = 1.0
    def begin(self): self.raw = 1.0; self.pred = 0.0
    def step(self, i, p):
        k, more = super().step(i, p)
        if k: self.pred += self.raw    # uncalibrated reach of the KEPT depths: their sum predicts the accepted count
        return k, more
    def end_round(self, drafted, accepted):
        if drafted <= 0: return
        a = self.alpha
        self.pred_ema = self.pred if self.pred_ema is None else (1 - a) * self.pred_ema + a * self.pred
        self.acc_ema = float(accepted) if self.acc_ema is None else (1 - a) * self.acc_ema + a * accepted
        if self.pred_ema > 1e-6: self.scale = min(self.hi, max(self.lo, self.acc_ema / self.pred_ema))

# measured plain-batch step ms for 1..8 rows + 3.1 ms per drafted depth + 2 ms of round overhead: cycle ms for D drafts
CYCLE_MS = [30.3 + 2.0, 41.6 + 3.1 + 2.0, 50.2 + 6.2 + 2.0, 58.6 + 9.3 + 2.0, 68.8 + 12.4 + 2.0, 75.1 + 15.5 + 2.0, 84.7 + 18.6 + 2.0, 92.9 + 21.7 + 2.0]

class TableRule(ProductRule):
    """Depth chosen BEFORE drafting from the request's own acceptance history; no probability, no per-depth host read.
    a[d] = EWMA (decay 0.75, prior 0.75) of P(draft d accepted | drafts < d accepted), updated only where observed
    (rounds stop counting at the first reject). depth = argmax_D (1 + sum_{d<=D} prod_{j<=d} a[j]) / cycle_ms[D], D in 1..cap.
    A fully accepted round is censored: the next round drafts one deeper (a probe). D = 0 (plain decode) is not used:
    skipping the drafter would leave a hole in its cache."""
    name = "table"
    needs_p = False
    def __init__(self, maxd, cycle=None, decay=0.75, prior=0.75):
        super().__init__(1.0, 0.0, maxd)
        self.cycle = list(cycle or CYCLE_MS); self.decay, self.prior = decay, prior
        self.new_request()
    def new_request(self): self.a = [self.prior] * (self.maxd + 1); self.probe = 0
    def choose(self):
        best, bd = -1.0, 1; e = 1.0; prod = 1.0
        for D in range(1, self.maxd + 1):
            prod *= self.a[D]; e += prod
            v = e / self.cycle[D]
            if v > best: best, bd = v, D
        return max(bd, self.probe)
    def end_round(self, drafted, accepted):
        if drafted <= 0: return
        for d in range(1, min(drafted, accepted + 1) + 1):
            self.a[d] = self.decay * self.a[d] + (1 - self.decay) * (1.0 if d <= accepted else 0.0)
        self.probe = drafted + 1 if accepted == drafted and drafted < self.maxd else 0

def _envf(k, d):
    v = os.environ.get(k)
    return d if v is None or v == "" else type(d)(v)

def env_defaults(ndt):
    """(thf, thv, maxd) for install(): today's 0.6 / 0.3 / ndt unless QWSPEC_THF / QWSPEC_THV / QWSPEC_MAXD are set."""
    return _envf("QWSPEC_THF", 0.6), _envf("QWSPEC_THV", 0.3), _envf("QWSPEC_MAXD", int(ndt))

def ngram_from_env(default=(2, 7, 16, 3)):
    """n-gram lookup setting for install(): (gram, K, match[, cmin]) from QWSPEC_NGRAM = "gram,K,match[,cmin]"; "off" = no
    lookup (None). cmin: see install(); 0 or absent = no second-occurrence check.
    Default (2, 7, 16, 3): the lookup replaces the drafter only on a match of 16 tokens or more and then drafts 7 tokens.
    The former (2, 3, 5) dates from the time the drafter stopped at 3 drafts. With the deep tier the drafter keeps 4-5 drafts
    on JSON and code where a 5-token match (keys, punctuation, idioms) copies the wrong values: lookup rounds accepted
    1.7 of 3 drafts there against 5.1 of 5.5 for drafter rounds, and on a plain copy 3 drafts per round is less than the
    drafter gives alone. QWSPEC_NGRAM=2,3,5 restores the former setting.
    cmin 3: on templated lists (SQL rows, log lines, table rows) a 16-token match is the constant part of a row and the
    7 copied tokens ran into the value that changes (0 to 3.9 of 7 accepted, 3-6 % slower than no lookup); cut at the point
    where the two latest rows differ, the lookup is level with "off" there and keeps its gain on copies and edits."""
    v = (os.environ.get("QWSPEC_NGRAM") or "").strip().lower()
    if v == "": return default
    if v in ("off", "0", "none"): return None
    f = [int(x) for x in v.split(",")]
    g, k, m = f[:3]
    if not (3 <= len(f) <= 4 and 1 <= g <= LMAX and k >= 1 and m >= g and min(f) >= 0): raise ValueError(f"QWSPEC_NGRAM={v}")
    return tuple(f)

def _unparked(g):
    """Active jobs minus those EXL3_SCHED_CHUNK_IL skips this step (Generator._cil_parked): a short job decoding next to a parked
    big prefill still counts as alone, so its lookup/draft policy is the solo one."""
    return [j for j in g.active_jobs if not getattr(j, "_cil_skip", False)]


def install_lookup(g, ngram, stats=None):
    """The n-gram lookup alone, for a Generator whose drafter keeps its own fixed depth (GLM): no depth rule, no read of the
    draft probabilities. ngram = (nmin, K, lmin[, cmin]) as in install(). When the context suffix (>= lmin tokens) occurred
    before, the round verifies its continuation (K tokens, cut by cmin as in install()) and runs ONE draft forward so the
    drafter cache has no hole; every other round is the Generator's own. Only with a single active request. The cache
    needs max_history >= K. Returns an uninstall function."""
    fol = SeqFollower(ngram[0]); K, lmin = ngram[1], ngram[2]; cmin = ngram[3] if len(ngram) > 3 else 0
    orig = type(g).iterate_draftmodel_mtp_gen.__get__(g)
    orig_enqueue = g.enqueue
    def enq(job):
        # prime the index at enqueue (as install()): its cold build lands in time-to-first-token, not in the first decode round
        r = orig_enqueue(job)
        if os.environ.get("QWSPEC_NGRAM_IDX", "1") != "0":
            for j in (job if isinstance(job, list) else [job]):
                if len(getattr(j, "sequences", ())) == 1: fol.sync(j.sequences[0].sequence_ids)
        return r
    g.enqueue = enq
    # Gate (QWSPEC_NGRAM_GATE=0 turns it off): per request, the mean tokens gained per lookup round and per drafter round
    # (from the sequence growth seen at the next round). A lookup round runs only while its mean is not below the drafter
    # round's, with one probe every PROBE drafter rounds so a copy section later in the answer turns it back on. Long prompts
    # hold many 16-token matches that do not continue (GLM 128K essay: lookup on 26.4 tok/s, off 30.7); copies keep it on.
    gate = os.environ.get("QWSPEC_NGRAM_GATE", "1") != "0"
    PROBE, A = 32, 0.25
    # drafter prior: 1 + 0.6 per draft token (GLM 128K essay: 0.67 accepted per draft); it is measured from the first
    # drafter round on, and lookup rounds start at the optimistic K + 1
    dr0 = lambda: 1.0 + 0.6 * max(1, int(g.num_draft_tokens))
    st = {"sid": None, "n": 0, "kind": None, "ng": float(K + 1), "dr": dr0(), "skip": 0}
    def gate_ok(sid, n):
        if sid is not st["sid"]:
            st.update(sid=sid, kind=None, ng=float(K + 1), dr=dr0(), skip=0)
        elif st["kind"] is not None and n > st["n"]:
            k = st["kind"]; st[k] = (1 - A) * st[k] + A * (n - st["n"])
        st["n"] = n
        if not gate or st["ng"] >= st["dr"] or st["skip"] >= PROBE: return True
        return False
    def wrapped(results):
        cont = None; sid = None
        act = _unparked(g)
        if len(act) == 1 and act[0].is_prefill_done() and len(act[0].sequences) == 1:
            sid = act[0].sequences[0].sequence_ids
            idx = fol.sync(sid)
            L, c = idx.match(K)
            if L >= lmin and c:
                cont = c
                if cmin > 0:
                    a = idx.witness_len(c)
                    if a < len(c): cont = c[:a] if a >= cmin else None
            if not gate_ok(sid, len(sid)):
                if cont is not None and stats is not None: stats["ng_gated"] = stats.get("ng_gated", 0) + 1
                cont = None
        if cont is None:
            if sid is not None:
                st["kind"] = "dr"; st["skip"] += 1
            return orig(results)
        st["kind"] = "ng"; st["skip"] = 0
        w = g.num_draft_tokens; g.num_draft_tokens = 1
        try: out = orig(results)
        finally: g.num_draft_tokens = w
        # the verified tokens are the copy, not the drafter's: its confidence record (one column) must not label them
        g._draft_conf_round = None
        if out is None: return None
        if stats is not None: stats["ng"] = stats.get("ng", 0) + 1; stats["ng_tokens"] = stats.get("ng_tokens", 0) + len(cont)
        return torch.tensor([cont], dtype=torch.long)
    g.iterate_draftmodel_mtp_gen = wrapped
    def uninstall():
        g.__dict__.pop("iterate_draftmodel_mtp_gen", None)
        if g.__dict__.get("enqueue") is enq: g.__dict__.pop("enqueue")
    return uninstall

def plain_lookup_from_env(rows_default=0):
    """Drafter-free n-gram lookup for a Generator WITHOUT a drafter (rtx1 14/10, rule "S2"): QWSPEC_PLAIN_LOOKUP = "lmin,K[,cmin[,rows]]"
    (unset / off = none). Returns (lmin, K, cmin, rows) or None. Measured on the 3090 (EXL3, greedy, 18 prompts x 2, text = plain up
    to near-ties): lmin 2, cmin 1, K = rows the backend verifies for ~free - 1 (15 for EXL3 dense on Ampere, 7 for a MoE pack
    whose MoE kernels change cost above 8 rows): dense x1.46 overall / x2.9 on edits, MoE x1.26 / x2.0, ahead of the upstream
    SAM defaults (x1.30 / x1.21). lmin 16 (ngram_from_env) is the right cut only when a drafter competes for the round.
    Optional 4th field rows (default rows_default = 0 = one active request only): with B >= 2 requests decoding together each one
    drafts, capped at rows // B - 1 tokens, and short drafts are padded to the longest (rtx1 sim19 / prof19: the batch keeps the
    longest copy instead of the shortest). rows = the backend's ~free verify rows (16 for EXL3 dense on Ampere). serve.py passes
    the served row_budget as rows_default so the plain-lookup cap matches the drafter-path row budget; an explicit 4th field wins.
    """
    v = (os.environ.get("QWSPEC_PLAIN_LOOKUP") or "").strip().lower()
    if v in ("", "off", "0", "none"): return None
    f = [int(x) for x in v.split(",")]
    if not (2 <= len(f) <= 4 and f[0] >= 1 and f[1] >= 1 and min(f) >= 0): raise ValueError(f"QWSPEC_PLAIN_LOOKUP={v}")
    return (f[0], f[1], f[2] if len(f) > 2 else 1, f[3] if len(f) > 3 else rows_default)

def install_plain_lookup(g, pl, stats=None):
    """Answers the Generator's n-gram slot (Generator(ngram_match_min > 0) -> iterate_ngram_gen) with the NgramIndex: the
    longest earlier suffix match (>= lmin tokens) proposes its continuation, capped at K and cut to the prefix that the
    previous occurrence of the same gram also continues with (>= cmin tokens, 0 = no cut); no match = a plain round.
    rows = 0: one active request only (batched rounds decode plain). rows > 0: every prefill-done request drafts, capped at
    rows // B - 1 for B requests, and short drafts are padded to the longest with the request's last token (filler the
    verification rejects, or accepts only where it IS the target's choice), so one request without a match no longer turns
    the lookup off for the batch (the generator's own slot trims to the shortest draft). Greedy verification: changes speed,
    not the accepted text. The cache needs max_history >= K. Returns an uninstall function."""
    lmin, K, cmin = pl[:3]; rows = pl[3] if len(pl) > 3 else 0
    fol = SeqFollower(2); fols = {}
    orig_enqueue = g.enqueue
    def enq(job):
        r = orig_enqueue(job)
        for j in (job if isinstance(job, list) else [job]):
            if len(getattr(j, "sequences", ())) == 1: (fols.setdefault(id(j), SeqFollower(2)) if rows else fol).sync(j.sequences[0].sequence_ids)
        return r
    def draft(f, sid, k):
        idx = f.sync(sid)
        L, c = idx.match(k)
        if L < lmin or not c: return []
        if cmin:
            w = idx.witness_len(c)
            if w < len(c):
                if w < cmin: return []
                c = c[:w]
        return c
    def rnd(results):
        job = _unparked(g)[0] if len(_unparked(g)) == 1 else None
        if job is None or not job.is_prefill_done() or len(job.sequences) != 1: return None
        c = draft(fol, job.sequences[0].sequence_ids, K)
        if not c: return None
        if stats is not None: stats["ng"] = stats.get("ng", 0) + 1; stats["ng_tokens"] = stats.get("ng_tokens", 0) + len(c)
        return torch.tensor([c], dtype=torch.long)
    def rnd_rows(results):
        jobs = [j for j in g.active_jobs if j.is_prefill_done()]   # the generator's verify order (iterate_ngram_gen)
        if not jobs or any(len(j.sequences) != 1 for j in jobs): return None
        k = min(K, rows // len(jobs) - 1)
        if k < 1: return None
        live = {id(j) for j in g.active_jobs}
        for d in [d for d in fols if d not in live]: del fols[d]
        fs = [fols.setdefault(id(j), SeqFollower(2)) for j in jobs]
        cs = [draft(f, j.sequences[0].sequence_ids, k) for f, j in zip(fs, jobs)]
        w = max(len(c) for c in cs)
        if not w: return None
        if stats is not None:
            stats["ng"] = stats.get("ng", 0) + 1; stats["ng_tokens"] = stats.get("ng_tokens", 0) + sum(len(c) for c in cs)
            stats["ng_pad"] = stats.get("ng_pad", 0) + sum(w - len(c) for c in cs)
        return torch.tensor([c + [f.idx.toks[-1]] * (w - len(c)) for c, f in zip(cs, fs)], dtype=torch.long)
    g.enqueue = enq
    g.iterate_ngram_gen = rnd_rows if rows else rnd
    def uninstall():
        g.__dict__.pop("iterate_ngram_gen", None)
        if g.__dict__.get("enqueue") is enq: g.__dict__.pop("enqueue")
    return uninstall

def rule_from_env(thf, thv, maxd):
    """QWSPEC_RULE = tier (default) | product (the former 0.6/0.3/3 rule when maxd=3) | tier (default thresholds below depth 4, stricter 5+ row tier: QWSPEC_THFD/THVD/DSPLIT) | cost | online | table (depth from the request's acceptance history, no probabilities); QWSPEC_GAMMA / QWSPEC_SCALE tune the calibrated reach.
    QWSPEC_FIXD = N: draft N depths with no host read and truncate once per round (rule applied on the stacked probabilities)."""
    kind = os.environ.get("QWSPEC_RULE", "tier")
    if kind == "product": r = ProductRule(thf, thv, maxd)
    elif kind == "cost": r = CostRule(thf, thv, maxd, _envf("QWSPEC_GAMMA", 1.0), _envf("QWSPEC_SCALE", 1.0))
    elif kind == "tier": r = TierRule(thf, thv, maxd, _envf("QWSPEC_THFD", 0.95), _envf("QWSPEC_THVD", 0.85), _envf("QWSPEC_DSPLIT", 3))
    elif kind == "table":
        cyc = os.environ.get("QWSPEC_CYCLE"); r = TableRule(maxd, [float(x) for x in cyc.split(",")] if cyc else None, _envf("QWSPEC_DECAY", 0.75), _envf("QWSPEC_PRIOR", 0.75))
    elif kind == "online": r = OnlineRule(thf, thv, maxd, _envf("QWSPEC_GAMMA", 1.0), _envf("QWSPEC_ALPHA", 0.15))
    else: raise ValueError(f"QWSPEC_RULE={kind}")
    return r

def apply_rule(rule, ps):
    """Offline / truncation form: rule applied to a list of per-depth max probabilities. Returns keep (0..len(ps))."""
    rule.begin(); keep = 0
    for i, p in enumerate(ps):
        k, more = rule.step(i, p)
        if k: keep = i + 1
        if not more: break
    return keep

def install(g, dm, model, thf=0.6, thv=0.3, maxd=3, ngram=None, stats=None, rule=None, fixed=None, log=None, log_depth=7):
    """ngram = (nmin, K, lmin) or (nmin, K, lmin, cmin) or None. Returns an uninstall function.
    cmin > 0: when the matched gram has a previous occurrence too, only the tokens that BOTH occurrences continue with are
      copied (a templated list repeats its constant part and changes its values: the copy stops where the two rows differ),
      and the lookup is taken only if at least cmin tokens are left, else the round is a drafter round.
    rule: a depth rule (default ProductRule(thf, thv, maxd) = the shipped rule).
    fixed = N: no host read per depth; N depths are drafted, the max probabilities stay on the device and one read per
      round feeds the rule, which then truncates (rows kept <= N + 1).
    log = list: logging arm. Drafts log_depth depths whatever the rule says, appends one dict per round
      (n, ng, ps, ids, keep) and verifies what `rule` would have kept (the stream is the baseline stream)."""
    lm = model.modules[model.logit_layer_idx]
    if rule is None: rule = ProductRule(thf, thv, maxd)
    maxd = rule.maxd                  # the rule owns the depth cap
    st = {"i": 0, "stop": False, "keep": None, "ps": [], "job": None, "n": None, "drafted": 0}
    if log is not None: fixed = None
    orig_draft = type(g).iterate_draftmodel_mtp_gen.__get__(g)
    orig_sfs = dm.sample_from_state; orig_fwd = dm.forward; orig_lmf = lm.forward
    cap = {}
    fol = SeqFollower(ngram[0]) if ngram is not None else None
    orig_enqueue = g.enqueue
    def enq(job):
        # prime the index at enqueue: its cold build (about 1 s at 192K) lands in time-to-first-token, not in decode
        r = orig_enqueue(job)
        if fol is not None and os.environ.get("QWSPEC_NGRAM_IDX", "1") != "0":
            for j in (job if isinstance(job, list) else [job]):
                if len(getattr(j, "sequences", ())) == 1: fol.sync(j.sequences[0].sequence_ids)
        return r
    if fol is not None: g.enqueue = enq
    def lmf(*a, **k):
        o = orig_lmf(*a, **k); cap["lg"] = o; return o
    def fwd(*a, **k):
        if st["stop"]: raise StopDraft()
        return orig_fwd(*a, **k)
    def sfs(state, params):
        ids = orig_sfs(state, params)
        dp = params.pop("draft_p", None)      # EXL3_DRAFT2S head: its max probability (lm.forward did not run: cap["lg"] is stale)
        if st["mode"] == "mtp" and not getattr(rule, "needs_p", True):
            return ids
        if st["mode"] == "mtp":
            if dp is not None: pt = dp.reshape(())
            else: pt = torch.softmax(cap["lg"][..., :model.config.vocab_size].float().reshape(-1, model.config.vocab_size), -1).max()
            i = st["i"]
            if fixed is not None:
                st["ps"].append(pt); st["i"] = i + 1
                if st["i"] >= fixed: st["stop"] = True
                return ids
            p = pt.item()
            if log is not None:
                st["ps"].append(p)
                if st["keep"] is None:
                    k, more = rule.step(i, p)
                    if not k: st["keep"] = i
                    elif not more: st["keep"] = i + 1
                st["i"] = i + 1
                if st["i"] >= log_depth: st["stop"] = True
                return ids
            k, more = rule.step(i, p)
            if not k:
                st["keep"] = i; raise StopDraft()
            st["i"] = i + 1
            if not more:
                st["stop"] = True; st["keep"] = st["i"]
        return ids
    win = log_depth if log is not None else (fixed if fixed is not None else maxd)
    # QWSPEC_DEPTHRULE="N:thf,thv,maxd,thfd,thvd": above N context tokens the tier rule switches to these values. Measured on the
    # RTX 3090: the full-attention verify kernel for hd 256 (qdec) serves 1-2 rows only and Triton runs 3..5
    # rows at +6 ms (128K), and 6..8 rows cost another +6 ms at 64K and +12 ms at 128K, so the deep drafts that pay at 4K-32K stop
    # paying; a shallower, stricter rule from ~48K keeps the gain on repetitive text and never loses to the former rule on prose.
    # Unset = one rule at every depth. QWSPEC_CAP_NG=1 also caps the n-gram copy length at the deep maxd.
    cap_n = None
    _c = os.environ.get("QWSPEC_DEPTHRULE", "").strip()
    if _c and fixed is None and log is None and isinstance(rule, TierRule):
        _n, _v = _c.split(":"); f = _v.split(",")
        cap_n, deep = int(_n), (float(f[0]), float(f[1]), int(f[2]), float(f[3]), float(f[4]))
        base = (rule.thf, rule.thv, rule.maxd, rule.thfd, rule.thvd); cap_ng = os.environ.get("QWSPEC_CAP_NG", "0") != "0"
    def wrapped(results):
        st.update(i=0, stop=False, keep=None, mode="mtp", ps=[])
        job = next((j for j in g.active_jobs if j.is_prefill_done()), None)
        cont = None
        n = None
        if job is not None and len(_unparked(g)) == 1:
            seq = job.sequences[0]; n = len(seq.sequence_ids)
            if st["job"] is job and st["n"] is not None and n > st["n"]:
                rule.end_round(st["drafted"], max(0, min(st["drafted"], n - st["n"] - 1)))
            elif st["job"] is not job and hasattr(rule, "new_request"):
                rule.new_request()
            st["job"] = job; st["n"] = n; st["drafted"] = 0
        if cap_n is not None and n is not None:
            rule.thf, rule.thv, rule.maxd, rule.thfd, rule.thvd = deep if n > cap_n else base
        rule.begin()                      # after end_round above: the online rule reads last round's prediction there
        if ngram is not None and job is not None and len(_unparked(g)) == 1:
            K = min(ngram[1], rule.maxd if cap_n is not None and cap_ng else maxd)
            if os.environ.get("QWSPEC_NGRAM_IDX", "1") == "0":
                L, c = ngram_match(seq.sequence_ids.torch_slice(0, n).flatten().tolist(), ngram[0], K)
            else:
                L, c = fol.sync(seq.sequence_ids).match(K)
                if os.environ.get("QWSPEC_NGRAM_SHADOW", "0") == "1":
                    ref = ngram_match(seq.sequence_ids.torch_slice(0, n).flatten().tolist(), ngram[0], K)
                    if stats is not None: stats["shadow"] = stats.get("shadow", 0) + 1
                    if ref != (L, c):
                        if stats is not None: stats["shadow_bad"] = stats.get("shadow_bad", 0) + 1
                        raise RuntimeError(f"ngram index mismatch at n={n}: old {ref} new {(L, c)}")
            if L >= ngram[2] and c:
                cont = c
                if len(ngram) > 3 and ngram[3] > 0 and os.environ.get("QWSPEC_NGRAM_IDX", "1") != "0":
                    a = fol.idx.witness_len(c)
                    if a < len(c):
                        cont = c[:a] if a >= ngram[3] else None
                        if stats is not None: stats["ng_cut" if cont else "ng_skip"] = stats.get("ng_cut" if cont else "ng_skip", 0) + 1
        if cont is not None:
            st["mode"] = "ng"; g.num_draft_tokens = 1
            try: out = orig_draft(results)
            finally: g.num_draft_tokens = win
            if out is None: return None
            if log is not None: log.append({"n": n, "ng": True, "cont": list(cont)})
            if stats is not None: stats["ng"] = stats.get("ng", 0) + 1; stats["rows"] = stats.get("rows", 0) + 1 + len(cont)
            return torch.tensor([cont], dtype=torch.long)
        w = rule.choose() if hasattr(rule, "choose") else win
        g.num_draft_tokens = w
        try:
            out = orig_draft(results)
        except StopDraft:
            out = g.draft_ids_pinned[:, :st["keep"]]
        finally:
            g.num_draft_tokens = win      # Job.prepare / reply_room reserve against this value: keep it at the deepest window
        if out is None: return None
        if fixed is not None:
            ps = torch.stack(st["ps"]).tolist()      # the one host read of the round
            out = out[:, :apply_rule(rule, ps)]
        elif log is not None:
            keep = st["keep"] if st["keep"] is not None else log_depth
            log.append({"n": n, "ng": False, "ps": list(st["ps"]), "ids": out[0, :log_depth].tolist(), "keep": keep})
            out = out[:, :keep]
        st["drafted"] = out.shape[-1]
        if stats is not None: stats["mtp"] = stats.get("mtp", 0) + 1; stats["rows"] = stats.get("rows", 0) + 1 + out.shape[-1]
        return out
    lm.forward = lmf; dm.forward = fwd; dm.sample_from_state = sfs
    g.iterate_draftmodel_mtp_gen = wrapped
    ndt0 = g.num_draft_tokens; g.num_draft_tokens = win
    def uninstall():
        g.num_draft_tokens = ndt0
        lm.forward = orig_lmf; dm.forward = orig_fwd; dm.sample_from_state = orig_sfs
        g.__dict__.pop("iterate_draftmodel_mtp_gen", None); g.__dict__.pop("enqueue", None)
    return uninstall
