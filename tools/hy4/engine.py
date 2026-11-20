"""Hy4-flash resident engine for the served path: ONE model in memory, one request at a time.

  prefill  chunked (default 8192 rows) through the prefill fast path (fused dequant GEMM MoE / dense, banded fp16 MLA, GEMM indexer scores, Engram row
           prefetch); the trunk states of every prompt row also fill the MTP draft head's cache in batches
  decode   HIP-graph trunk step + MTP draft chain graph + one T = k + 1 row verify graph per round (greedy speculative decoding: the stream equals
           plain greedy decoding of the trunk) or, per request, the plain graph step

The decode kernels (attn_eager, ref_fused, head_bf16, mla_bgemv, kv_fused, router_gemv) and the prefill kernels are two flag sets of the same
modules: `phase("prefill")` switches the decode set off, `phase("decode")` restores it. Graphs are only ever captured in the decode phase.
Every flag is read from the environment when the loader modules are imported: set the environment (serve.SHIPPED_ENV) BEFORE `Hy4Engine.load`.
"""
from __future__ import annotations

import os
import sys
import time
import types
from typing import Callable

CHUNK_DEFAULT = 8192
PRIME_ROWS = int(os.environ.get("HY4_PRIME_ROWS", "2048"))           # rows per batched MTP draft-cache fill call


WARMUP_MIN = 256        # below this many prompt tokens a warm-up compiles nothing useful: skip it
WARMUP_PAD = 16         # row alignment of the last prefill chunk (the static buffers also keep 64 spare rows, see serve.apply_env)


def warmup_clamp(n: int, ctx: int, k: int, new: int = 8) -> int:
    """Prompt length of the load warm-up: generate() accepts n + new + k + 2 <= ctx, so a requested n above ctx - (new + k + 2 + WARMUP_PAD) is clamped
    to that bound (the server would never become ready otherwise). Returns 0 (skip) when the bound is below WARMUP_MIN. Unchanged when n fits."""
    room = ctx - (new + k + 2 + WARMUP_PAD)
    if n <= room:
        return n
    if room < WARMUP_MIN:
        print(f"warm-up skipped: ctx too small (requested {n}, ctx {ctx})", flush=True)
        return 0
    print(f"warm-up clamped {n} -> {room} (ctx {ctx})", flush=True)
    return room


def _stub_package(tree: str):
    """import loader modules by path without importing the whole exllamav3 package (as the bench harnesses do)"""
    for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
        m = types.ModuleType(n)
        m.__path__ = [os.path.join(tree, "exllamav3", s)]
        sys.modules[n] = m


class Hy4Engine:
    def __init__(self, tree: str, gguf: str, ctx: int = 16384, k: int = 3, chunk: int = CHUNK_DEFAULT, prime_rows: int = 0):
        self.tree, self.gguf, self.k, self.chunk = tree, gguf, k, chunk
        self.ctx = ctx
        self.prime_rows = prime_rows              # 0 = fill the draft cache for the whole prompt; N = the last N rows only (cheaper, lower acceptance)
        self.ngram_ext = int(os.environ.get("HY4_NGRAM_EXT", "0"))        # extra verify rows from a prompt-lookup n-gram match (0 = off)
        self.ngram_min = int(os.environ.get("HY4_NGRAM_MIN", "4"))
        self.ngram_adapt = os.environ.get("HY4_NGRAM_ADAPT", "0") == "1"
        self.ngram_gate = os.environ.get("HY4_NGRAM_GATE", "0") == "1"
        self.ngram_fill = os.environ.get("HY4_NGRAM_FILL", "0") == "1"
        self.ngram_long = int(os.environ.get("HY4_NGRAM_LONG", "0"))      # >= 1: a match of at least this many tokens REPLACES the MTP draft chain (verify y + up to TM - 1 copied tokens; the draft cache is refilled from the trunk states); 0 = off
        self._tune = None           # torch.cuda.tunable module while a frozen TunableOp results file is loaded (HY4_TUNABLEOP_FILE)
        self.pipe = os.environ.get("HY4_PIPE", "1") == "1"     # start the next round's draft before the host bookkeeping of this round (host-only reorder, same tokens)
        self.policy = None
        if os.environ.get("HY4_DRAFT_POLICY", "0") == "1":
            from draft_policy import CostPolicy
            os.environ["HY4_DRAFT_CONF"] = "1"        # the draft graph appends the confidences (read at the import of mtp_head, i.e. in load())
            self.policy = CostPolicy.from_env()
        self.ready = False
        self.meta = None
        self.last_stats: dict = {}

    # ------------------------------------------------------------------------------------------ load
    def load(self, progress: Callable[[str], None] = lambda s: None):
        import torch
        self.torch = torch
        cap = int(os.environ.get("HY4_NOSYNC_CAP", "0"))
        if cap < self.ctx + 64:
            raise RuntimeError(f"HY4_NOSYNC_CAP={cap} must be set (before import) to at least ctx + 64 = {self.ctx + 64}")
        _stub_package(self.tree)
        tf = os.environ.get("HY4_TUNABLEOP_FILE", "")
        if tf and os.path.exists(tf):      # frozen TunableOp results for the prefill GEMMs (no tuning at run time; shapes missing from the file use the default kernel)
            import torch.cuda.tunable as tu
            tu.set_filename(tf, insert_device_ordinal=False)
            tu.tuning_enable(False)
            tu.enable(True)
            self._tune = tu
            try:
                ok = tu.read_file()      # load now: a validator mismatch (torch / rocBLAS / hipBLASLt version) would otherwise drop the file silently
                progress(f"TunableOp results {tf}: read={ok} entries={len(tu.get_results())}")
            except Exception as e:      # noqa: BLE001
                progress(f"TunableOp results {tf}: {type(e).__name__} {e}")
        elif tf:
            progress(f"TunableOp file missing: {tf}")
        from exllamav3.loader.gguf_collection import GGUFCollection
        from exllamav3.loader import hy4_forward as hf
        from exllamav3.loader import mla_hy4 as mh
        from exllamav3.loader import step_graph as SG
        from exllamav3.loader import mtp_head as MH
        self.hf, self.mh, self.SG, self.MH = hf, mh, SG, MH
        self._decode_flags()
        self.phase("prefill")                     # the prefill kernels are selected at layer load time too (as in the prefill harness)
        t0 = time.time()
        progress("opening the GGUF")
        g = GGUFCollection(self.gguf)
        self.g, self.meta = g, g.meta
        self.dev = torch.device("cuda")
        progress("loading layers")
        M = hf.Hy4Forward(g, device="cuda", cache_weights=True, hook=None)
        for i in range(M.n_layers):
            M._layer(i)
        torch.cuda.synchronize()
        self.M = M
        progress("graph decoder + draft head")
        self.GR = self._decoder(M)
        self.GR.prepare()                          # static K / indexer / Engram buffers at the full context size: addresses never change after this
        self.MT = MH.Hy4MTP(M)
        self.MG = self._mtp_graph()
        MH._STAGE_OV[0] = os.environ.get("HY4_SERVE_STAGE_OV", "1") == "1"     # Engram row 0 of the next verify staged during the draft chain
        torch.cuda.synchronize()
        self.load_s = time.time() - t0
        self.ready = True
        progress(f"ready: {M.n_layers} layers, {M.stats.get('resident_gib')} GiB resident experts, {self.load_s:.0f} s")

    def _decoder(self, M):                 # hooks of the multi-session engine (mu_engine.py)
        return self.SG.GraphDecoder(M)

    def _mtp_graph(self):
        return self.MH.MTPGraph(self.MT, self.k)

    def _decode_flags(self):
        """remember the decode-set switches the environment selected at import"""
        L = self.SG.__name__.rsplit(".", 1)[0]
        import importlib
        self._dec = []
        for mod, attrs in (("attn_eager", ("ENABLED", "RMS")), ("ref_fused", ("ENABLED",)), ("head_bf16", ("ENABLED",)), ("mla_bgemv", ("ENABLED",)),
                           ("kv_fused", ("ENABLED",)), ("router_gemv", ("ENABLED",)), ("pool_fused", ("ENABLED",))):
            m = importlib.import_module(f"{L}.{mod}")
            for a in attrs:
                self._dec.append((m, a, getattr(m, a)))

    def phase(self, name: str):
        """prefill: decode kernels off, prefill kernels on (the configuration the prefill numbers were measured in); decode: decode kernels restored"""
        pre = name == "prefill"
        if self._tune is not None:
            self._tune.enable(pre)          # prefill GEMMs only: the decode graph keeps the default kernels
        for m, a, v in self._dec:
            setattr(m, a, False if pre else v)
        hf, mh = self.hf, self.mh
        hf.PF_MOE = hf.PF_DENSE = hf.PF_IX = True
        hf.PF_MOE_I8 = hf.PF_MOE_REF = hf.PF_DENSE_I8 = hf.PF_ROUTER = hf.PF_DIAG = False
        mh.PF_ATTN, mh.PF_ATTN_FLASH = True, False          # (rows <= 16 never take the prefill branches)

    def warmup(self, n: int):
        """one synthetic request of n prompt tokens + 8 decode tokens: compiles the Triton prefill kernels (they would land in the first user chunk, about 0.3 s) and captures the decode graphs"""
        n = warmup_clamp(n, self.ctx, self.k)
        if n <= 0:
            return
        ids = [(i * 7919 + 13) % 50000 for i in range(n)]
        t0 = time.perf_counter()
        self.generate(ids, 8, lambda toks: False)
        spec = os.environ.get("HY4_PRECAPTURE", "0")
        if spec != "0":
            self.precapture(spec)
        if os.environ.get("HY4_WARMUP_PRIME_SPARSE", "1") == "1" and os.environ.get("HY4_LC_PRIME", "0") == "1":
            # the draft-cache fill of a long prompt (> 16K groups) runs the blocked selection + sparse gather kernels: compile them now (a request must never JIT at depth)
            old, self.SG.PRIME_IX_BLK = self.SG.PRIME_IX_BLK, 0
            try:
                self.reset()
                self.prefill(ids)
                self.torch.cuda.synchronize()
            finally:
                self.SG.PRIME_IX_BLK = old
        self.reset()
        print(f"warm-up {n} tokens: {time.perf_counter() - t0:.1f} s", flush=True)
        if os.environ.get("HY4_WARMUP_TAILS", "1") == "1":
            self.warmup_tails()

    def warmup_lengths(self) -> list[int]:
        """prompt lengths that cover every Triton specialisation class a real prompt can need (an int argument is specialised on divisible-by-16 and on == 1):
        rows aligned / unaligned, below / above the 1024-row CK threshold, a merged last chunk (chunk + up to 25 %) aligned / unaligned, and a second chunk
        with past rows > 0 (needs a tail above 25 % of a chunk). A class first met by a real request costs 2-3 s of JIT (measured)."""
        c = self.chunk
        out = [40, 100, 512, 1000, 1024, 1500, 2048 + 7, c + 1000, c + 1024]
        if c * 3 // 8 + 5 < c:
            out.append(c + c * 3 // 8 + 5)          # unaligned second chunk, past > 0
            out.append(c + c // 2)                  # aligned second chunk, past > 0
        return [n for n in out if n + 16 < self.ctx]

    def warmup_tails(self):
        t0 = time.perf_counter()
        for n in self.warmup_lengths():
            ids = [(i * 7919 + 13) % 50000 for i in range(n)]
            self.reset()
            self.prefill(ids)
            self.torch.cuda.synchronize()
        self.reset()
        print(f"warm-up tails {self.warmup_lengths()}: {time.perf_counter() - t0:.1f} s", flush=True)

    def precapture(self, spec: str = "all", progress=print):
        """build the decode graphs now instead of inside a decode round (a build = 2 eager bodies + the capture, 0.5 - 1 s of host time): spec = "all" (every context
        bucket up to the cap, T = 1 .. TM, plus every draft bucket) or "T1,T2,.." (those verify row counts; "all" buckets). Call between requests, reset() after."""
        t0 = time.perf_counter()
        ts = None if spec in ("all", "1") else [int(x) for x in spec.split(",") if x]
        built = self.GR.precapture(ts=ts)
        nb = self.MG.precapture()
        free = self.torch.cuda.mem_get_info()[0] / 2 ** 30
        progress(f"precapture: {len(built)} verify graphs ({sum(x[2] for x in built) / 1000:.1f} s), {nb} draft graphs, {time.perf_counter() - t0:.1f} s, {free:.1f} GiB free")

    # ------------------------------------------------------------------------------------------ request state
    def reset(self):
        M, GR = self.M, self.GR
        M.n, M.tokens, M._x, M._eng_pf = 0, [], None, {}
        M.kc = [None] * M.n_layers
        for ix in getattr(M, "ixf", {}).values():
            ix.n, ix.P, ix.vg = 0, 0, None
        for ix in M.ix.values():
            ix.reset()
        GR.reset()                    # zeroes the static K / indexer buffers in place (captured graphs keep their addresses)
        self.MT.reset()

    # ------------------------------------------------------------------------------------------ prefill
    def prefill(self, ids: list[int]):
        """-> (first token y, trunk state [1, D] of the last prompt row). Also fills the draft head's cache."""
        torch, M = self.torch, self.M
        n = len(ids)
        self.phase("prefill")
        M.prefetch_engram(ids, self.chunk)
        states, lg = [], None
        for s, e in self.hf.chunk_plan(0, n, self.chunk):
            lg = M.step(ids[s:e])
            states.append(M.head_state(M._x))
            M._x = None
            torch.cuda.synchronize()
        y = int(torch.argmax(lg))
        H = torch.cat(states) if len(states) > 1 else states[0]
        self._H = H
        hcur = H[-1:].clone()
        t = time.perf_counter()
        self.last_stats["prefill_trunk_s"] = t - self._t0
        self._prime_draft(ids, H)
        torch.cuda.synchronize()
        self.last_stats["prime_s"] = time.perf_counter() - t
        return y, hcur

    def _prime_draft(self, ids: list[int], H):
        """draft head cache rows for the prompt: pair (token j+1, trunk state h_j) at head position j, for j = 0 .. n-2 (the batched form of what the
        bench harness does one position at a time with MT.chain(.., k=1))"""
        torch, MT = self.torch, self.MT
        n = len(ids)
        lo = 0 if self.prime_rows <= 0 else max(0, n - 1 - self.prime_rows)
        tok = torch.tensor(ids, dtype=torch.long, device=self.dev)
        ar = MT.ar
        s = lo
        f16 = os.environ.get("HY4_PF_F16")
        os.environ["HY4_PF_F16"] = "0"      # the fp16 attention chain is tuned for the trunk's matrices; the draft layer takes the generic prefill path
        try:
            while s < n - 1:
                e = min(n - 1, s + PRIME_ROWS)
                MT.prime(tok[s + 1:e + 1], H[s:e], ar[s:e], ((e + 3) // 4) * 4)
                s = e
            self.last_stats["prime_mode"] = "batched" if lo == 0 else f"batched tail {n - 1 - lo}"
        except Exception as ex:           # noqa: BLE001 - the draft only affects speed, never the output: fall back to the last 128 rows one by one
            print(f"draft cache fill failed ({type(ex).__name__}: {ex}); falling back to a short per-row fill", flush=True)
            torch.cuda.synchronize()
            MT.reset()
            for j in range(max(0, n - 129), n - 1):
                MT.chain(tok[j + 1:j + 2], H[j:j + 1], j, 1)
            self.last_stats["prime_mode"] = "per-row tail 128 (batched failed)"
        finally:
            if f16 is None:
                os.environ.pop("HY4_PF_F16", None)
            else:
                os.environ["HY4_PF_F16"] = f16

    # ------------------------------------------------------------------------------------------ generate
    def generate(self, ids: list[int], max_new: int, on_tokens: Callable[[list[int]], bool], speculative: bool = True,
                 cancel: Callable[[], bool] = lambda: False) -> dict:
        """Run one request. on_tokens(list of new token ids) is called once per prefill (first token) / decode round and returns True to stop
        (EOS or a stop string seen by the caller). Returns the stats dict."""
        if not self.ready:
            raise RuntimeError("model not loaded")
        n = len(ids)
        if n + max_new + self.k + 2 > self.ctx:
            raise ValueError("context")
        torch, GR, MG = self.torch, self.GR, self.MG
        st = self.last_stats = {"prompt_tokens": n, "speculative": speculative, "k": self.k}
        self._t0 = t0 = time.perf_counter()
        self.reset()
        y, hcur = self.prefill(ids)
        self.phase("decode")
        t1 = time.perf_counter()
        st["prefill_s"] = t1 - t0
        out_n, rounds, hist = 0, 0, [0] * (self.SG.TM + 1)
        stop = False
        # first token: it comes from the prefill logits
        stop = on_tokens([y])
        out_n = 1
        st["ttft_s"] = time.perf_counter() - t0
        tdec = time.perf_counter()
        if speculative:
            k = self.k
            pc = time.perf_counter
            tm = [0.0, 0.0, 0.0]
            ng = None
            if self.ngram_ext > 0:
                from ngram import NGram
                ng = NGram(self.ngram_min, self.ngram_ext)
                ng.reset(list(ids) + [y])
            if self.policy is not None:
                self.policy.reset_stats()
            n_ext = n_hit = n_extacc = n_long = 0
            rate, since = 0.7, 0          # recent acceptance of the extra rows; below 0.4 the proposer pauses (a probe every 8 rounds)
            stage_ov = self.MH._STAGE_OV[0]
            cap0, caps0 = GR.stats["captures"], GR.stats["capture_s"]
            lt = [0.0]                    # time of the last draft launch (set by plan)
            td_prev = None
            busy = [0.0]                  # host time of the whole verify launch call (includes the staging that overlaps the GPU)
            gap = [0.0, 0.0]              # host time with the GPU idle: verify result known -> next draft launched, draft tokens known -> verify launched

            def plan(y_, hcur_, j_):
                """the next round's draft, started BEFORE the host bookkeeping of this round: the n-gram long-match test (a long match replaces the chain) or the draft
                launch (async: its tokens are read at the top of the round) + the Engram reads of verify row 0 (they do not depend on the drafts)"""
                skip_, cl_, dd_ = False, [], None
                if ng is not None and self.ngram_long > 0:
                    cl_, ml_ = ng.propose([], self.SG.TM - 1)
                    skip_ = bool(cl_) and ml_ >= self.ngram_long
                if not skip_:
                    dd_ = MG.run(y_, hcur_, j_)
                    lt[0] = pc()
                    if stage_ov:
                        GR.prestage(y_)
                return skip_, cl_, dd_

            planned = False
            while out_n < max_new and not stop and not cancel():
                j = GR.fw.n - 1
                ta = pc()
                if not planned:
                    skip, cl, dd = plan(y, hcur, j)
                planned = False
                if td_prev is not None and not skip:
                    gap[0] += lt[0] - td_prev
                if skip:
                    d, ext = [], cl
                    tb = pc()
                else:
                    d = dd.tolist()
                    confs = None
                    if self.MH.CONF:                      # the draft graph appended the k confidences (fixed point 2^-20)
                        confs = [c / 1048576.0 for c in d[k:]]
                        d = d[:k]
                    if self.policy is not None:
                        d = d[:self.policy.choose(confs, 1 if ng is not None else 0)]
                    tb = pc()
                    ext = []
                if ng is not None and not skip:
                    lim = min(self.ngram_ext, self.SG.TM - 1 - k)
                    if self.ngram_gate:
                        since += 1
                        lim = lim if rate >= 0.6 else (2 if rate >= 0.4 else (lim if since >= 8 else 0))
                    ext, ml = ng.propose(d, lim)
                    if self.ngram_adapt and ext:           # longer match, more rows: 2 rows at the minimum match, +2 per extra matched token
                        ext = ext[:max(1, (ml - self.ngram_min + 1) * 2)]
                toks = [y] + d + ext
                if skip:
                    n_long += 1
                elif ext:
                    n_hit += 1
                    n_ext += len(ext)
                hc0 = hcur.clone() if skip else None      # hcur views the static graph output: the replay below overwrites it
                _lg, amv, _x, hv = GR.step_rows(toks)
                tc = pc()
                amh = amv.tolist()
                td = pc()
                tm[0] += tb - ta
                tm[1] += tc - tb
                tm[2] += td - tc
                gap[1] += GR.t_launch - tb               # draft tokens known -> the verify's first graph launched (the later segments' rows are staged while the GPU runs)
                busy[0] += tc - tb
                acc = 0
                nd = len(toks) - 1
                while acc < nd and toks[acc + 1] == amh[acc]:
                    acc += 1
                kk = len(toks) - 1 - len(ext)             # drafted rows of this round (the policy may cut the chain below k)
                if skip:
                    pass
                elif self.policy is not None and confs is not None:
                    self.policy.update(confs, kk, acc)
                if not skip:
                    n_extacc += max(0, acc - kk)
                if ext and not skip:
                    since = 0
                    rate = 0.75 * rate + 0.25 * (max(0, acc - len(d)) / len(ext))
                new = toks[1:acc + 1] + [amh[acc]]        # toks[0] = y was already emitted; the verified drafts + the trunk's own next token
                GR.rewind(j + 2 + acc)
                if skip:                                  # the chain wrote no head rows: pairs (token m, state before it) for head positions j .. j + acc
                    MT = self.MT
                    N = ((j + acc + 1 + 3) // 4) * 4
                    sts = torch.cat([hc0, hv[:acc]]) if acc > 0 else hc0
                    MT.step(torch.tensor(toks[:acc + 1], dtype=torch.long, device=self.dev), sts, MT.ar[j:j + acc + 1], N)
                elif self.ngram_fill and acc >= len(d) and len(d) >= 1:          # head rows j + k .. j + acc were never written by the draft chain: true pairs (token p + 1, trunk state h_p)
                    MT = self.MT
                    N = ((j + acc + 1 + 3) // 4) * 4
                    MT.step(torch.tensor(toks[len(d):acc + 1], dtype=torch.long, device=self.dev), hv[len(d) - 1:acc], MT.ar[j + len(d):j + acc + 1], N)
                if ng is not None:
                    ng.add(new)
                y, hcur = amh[acc], hv[acc:acc + 1]
                room = max_new - out_n
                if len(new) > room:
                    new = new[:room]
                if self.pipe and out_n + len(new) < max_new:
                    skip, cl, dd = plan(y, hcur, GR.fw.n - 1)
                    planned = True
                td_prev = td
                hist[acc] += 1
                rounds += 1
                out_n += len(new)
                stop = on_tokens(new)
            st["round_ms"] = [x / max(rounds, 1) * 1000 for x in tm]
            st["gap_ms"] = [x / max(rounds, 1) * 1000 for x in gap]       # host time per round with the GPU idle: [after verify, before verify]
            st["busy_ms"] = busy[0] / max(rounds, 1) * 1000
            st["captures"] = GR.stats["captures"] - cap0
            st["capture_s"] = GR.stats["capture_s"] - caps0
            if self.policy is not None:
                st["policy"] = self.policy.stats()
            st["ngram"] = {"ext": self.ngram_ext, "min": self.ngram_min, "fill": self.ngram_fill, "hit_rounds": n_hit, "ext_rows": n_ext, "ext_accepted": n_extacc, "long": self.ngram_long, "long_rounds": n_long}
        else:
            while out_n < max_new and not stop and not cancel():
                lg = GR.step(y)
                y = int(torch.argmax(lg))
                out_n += 1
                stop = on_tokens([y])
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        st.update(decode_s=t2 - tdec, decode_tokens=out_n - 1, total_s=t2 - t0, rounds=rounds, accept_hist=hist,
                  tau=(out_n - 1) / rounds if rounds else None, generated=out_n, cancelled=bool(cancel()))
        return st
