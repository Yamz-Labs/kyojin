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
PRIME_ROWS = 2048           # rows per batched MTP draft-cache fill call


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
        self.ngram_fill = os.environ.get("HY4_NGRAM_FILL", "0") == "1"
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
        self.GR = SG.GraphDecoder(M)
        self.GR.prepare()                          # static K / indexer / Engram buffers at the full context size: addresses never change after this
        self.MT = MH.Hy4MTP(M)
        self.MG = MH.MTPGraph(self.MT, self.k)
        MH._STAGE_OV[0] = os.environ.get("HY4_SERVE_STAGE_OV", "1") == "1"     # Engram row 0 of the next verify staged during the draft chain
        torch.cuda.synchronize()
        self.load_s = time.time() - t0
        self.ready = True
        progress(f"ready: {M.n_layers} layers, {M.stats.get('resident_gib')} GiB resident experts, {self.load_s:.0f} s")

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
        for m, a, v in self._dec:
            setattr(m, a, False if pre else v)
        hf, mh = self.hf, self.mh
        hf.PF_MOE = hf.PF_DENSE = hf.PF_IX = True
        hf.PF_MOE_I8 = hf.PF_MOE_REF = hf.PF_DENSE_I8 = hf.PF_ROUTER = hf.PF_DIAG = False
        mh.PF_ATTN, mh.PF_ATTN_FLASH = True, False          # (rows <= 16 never take the prefill branches)

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
        for s in range(0, n, self.chunk):
            e = min(n, s + self.chunk)
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
                MT.step(tok[s + 1:e + 1], H[s:e], ar[s:e], ((e + 3) // 4) * 4)
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
            n_ext = n_hit = n_extacc = 0
            while out_n < max_new and not stop and not cancel():
                j = GR.fw.n - 1
                ta = pc()
                d = MG.run(y, hcur, j).tolist()
                tb = pc()
                ext = ng.propose(d, min(self.ngram_ext, self.SG.TM - 1 - k))[0] if ng is not None else []
                toks = [y] + d + ext
                if ext:
                    n_hit += 1
                    n_ext += len(ext)
                _lg, amv, _x, hv = GR.step_rows(toks)
                tc = pc()
                amh = amv.tolist()
                td = pc()
                tm[0] += tb - ta
                tm[1] += tc - tb
                tm[2] += td - tc
                acc = 0
                nd = len(toks) - 1
                while acc < nd and toks[acc + 1] == amh[acc]:
                    acc += 1
                n_extacc += max(0, acc - k)
                new = toks[1:acc + 1] + [amh[acc]]        # toks[0] = y was already emitted; the verified drafts + the trunk's own next token
                GR.rewind(j + 2 + acc)
                if self.ngram_fill and acc >= k:          # head rows j + k .. j + acc were never written by the draft chain: true pairs (token p + 1, trunk state h_p)
                    MT = self.MT
                    N = ((j + acc + 1 + 3) // 4) * 4
                    MT.step(torch.tensor(toks[k:acc + 1], dtype=torch.long, device=self.dev), hv[k - 1:acc], MT.ar[j + k:j + acc + 1], N)
                if ng is not None:
                    ng.add(new)
                y, hcur = amh[acc], hv[acc:acc + 1]
                hist[acc] += 1
                rounds += 1
                room = max_new - out_n
                if len(new) > room:
                    new = new[:room]
                out_n += len(new)
                stop = on_tokens(new)
            st["round_ms"] = [x / max(rounds, 1) * 1000 for x in tm]
            st["ngram"] = {"ext": self.ngram_ext, "min": self.ngram_min, "fill": self.ngram_fill, "hit_rounds": n_hit, "ext_rows": n_ext, "ext_accepted": n_extacc}
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
