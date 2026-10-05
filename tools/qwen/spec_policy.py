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
        return (L, toks[j:j + K].tolist())

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

def install(g, dm, model, thf=0.6, thv=0.3, maxd=3, ngram=None, stats=None):
    """ngram = (nmin, K, lmin) or None. Returns an uninstall function."""
    lm = model.modules[model.logit_layer_idx]
    st = {"i": 0, "reach": 1.0, "stop": False, "keep": None}
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
        if st["mode"] == "mtp":
            p = torch.softmax(cap["lg"][..., :model.config.vocab_size].float().reshape(-1, model.config.vocab_size), -1).max().item()
            i = st["i"]; st["reach"] *= p
            if i > 0 and st["reach"] < thv:
                st["keep"] = i; raise StopDraft()
            st["i"] = i + 1
            if st["reach"] < thf or st["i"] >= maxd:
                st["stop"] = True; st["keep"] = st["i"]
        return ids
    def wrapped(results):
        st.update(i=0, reach=1.0, stop=False, keep=None, mode="mtp")
        job = next((j for j in g.active_jobs if j.is_prefill_done()), None)
        cont = None
        if ngram is not None and job is not None and len(g.active_jobs) == 1:
            seq = job.sequences[0]; n = len(seq.sequence_ids); K = min(ngram[1], maxd)
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
            if L >= ngram[2] and c: cont = c
        if cont is not None:
            st["mode"] = "ng"; g.num_draft_tokens = 1
            try: out = orig_draft(results)
            finally: g.num_draft_tokens = maxd
            if out is None: return None
            if stats is not None: stats["ng"] = stats.get("ng", 0) + 1; stats["rows"] = stats.get("rows", 0) + 1 + len(cont)
            return torch.tensor([cont], dtype=torch.long)
        g.num_draft_tokens = maxd
        try:
            out = orig_draft(results)
        except StopDraft:
            out = g.draft_ids_pinned[:, :st["keep"]]
        if out is None: return None
        if stats is not None: stats["mtp"] = stats.get("mtp", 0) + 1; stats["rows"] = stats.get("rows", 0) + 1 + out.shape[-1]
        return out
    lm.forward = lmf; dm.forward = fwd; dm.sample_from_state = sfs
    g.iterate_draftmodel_mtp_gen = wrapped
    def uninstall():
        lm.forward = orig_lmf; dm.forward = orig_fwd; dm.sample_from_state = orig_sfs
        g.__dict__.pop("iterate_draftmodel_mtp_gen", None); g.__dict__.pop("enqueue", None)
    return uninstall
