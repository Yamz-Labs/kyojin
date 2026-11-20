"""Hy4 MTP draft head (the GGUF's next-n block blk.<n_trunk>): eager torch + two Triton expert kernels. Night 18/11.

One ordinary decoder layer of the trunk's class WITHOUT hyper-connection streams (plain residuals, sandwich norms) plus four tensors:
    x = eh_proj( cat[ enorm(embed(token j+1)), hnorm(h_j) ] )        h_j = the trunk state the output head reads (after the hc head + output_norm)
    s1 = x + attn(rms(x) * attn_norm);  s2 = s1 + rms(moe(rms(s1) * ffn_norm)) * post_ffw_norm
    y = shared_head_norm(s2);   logits = trunk output head(y);   the next chain step feeds y as `h`
(llama.cpp hy-v4-flash.cpp graph_mtp; the Mac port kyojin_mlx/hy4_mtp.py.) The layer owns a full-attention MLA with its own K cache and an indexer.
Weights are bf16 in the GGUF: dense ones are used as fp32 (draft quality only moves the acceptance rate, never the output: the trunk verifies),
the 512-expert stacks are quantised on load to int8 + per-row scale (moe_i8.py). K rows and indexer state are written BY POSITION (index_copy_),
so a rejected draft row is simply overwritten by the next round. Greedy chain only (argmax of the head logits)."""
from __future__ import annotations

import os
import time

import torch

from . import moe_i8 as MI
from . import step_graph as SG
from . import step_nosync as _ns
from .engram_rows import load_dense, rms
from .hy4_forward import _mv, head_gemm, QLinear, GG_DENSE, dense_fn
from .indexer_fused import FusedKPoolIndexer
from .mla_hy4 import Hy4MLA


def q8_0_pack(W):
    """fp32 [N, K] (K % 32 == 0, any device) -> stored Q8_0 bytes [N, K // 32, 34] uint8 (fp16 d = amax / 127, int8 q = round(w / d))"""
    N, K = W.shape
    Wb = W.reshape(N, K // 32, 32)
    d16 = (Wb.abs().amax(-1) / 127.0).half()
    ds = d16.float()
    q = torch.where(ds[..., None] > 0, torch.round(Wb / ds[..., None].clamp_min(1e-30)), torch.zeros_like(Wb)).clamp(-127, 127).to(torch.int8)
    raw = torch.empty(N, K // 32, 34, dtype=torch.uint8, device=W.device)
    raw[..., :2] = d16.contiguous().view(torch.uint8).reshape(N, K // 32, 2)
    raw[..., 2:] = q.view(torch.uint8)
    return raw


def q8_linear(W, chunk=8192):
    """fp32 [N, K] -> QLinear on Q8_0 bytes (the trunk's dense decode GEMV; the CPU test falls back to dequantised matmul)"""
    raw = torch.cat([q8_0_pack(W[a:a + chunk]) for a in range(0, W.shape[0], chunk)], 0) if W.shape[0] > chunk else q8_0_pack(W)
    f = dense_fn(8, W.shape[1])
    return QLinear(raw, 8, GG_DENSE[8], f[1] if f is not None else None)


class Hy4MTP:
    def __init__(self, fw, cap=None):
        g, dev, il = fw.g, fw.dev, fw.n_trunk
        b = f"blk.{il}."
        self.fw, self.il, self.dev, self.eps = fw, il, dev, fw.eps
        d = lambda n: load_dense(g, b + n).to(dev)
        # draft-only cheapness (HY4_MTP_FAST=1): the draft never changes the output (the trunk verifies every token), so its dense matrices are re-packed
        # as Q8_0 bytes read by the trunk's decode GEMV, its router / attention core run the fused Triton kernels, norms are one launch
        self.fast = os.environ.get("HY4_MTP_FAST", "0") == "1" and (dev.type == "cuda" or os.environ.get("HY4_MTP_FAST_CPU", "0") == "1")
        def dl(n):
            t = load_dense(g, n).to(dev)
            return q8_linear(t) if (self.fast and t.dim() == 2 and t.shape[1] % 32 == 0 and t.shape[0] >= 64) else t
        self.eh = dl(b + "nextn.eh_proj.weight")
        self.enorm, self.hnorm, self.shn = d("nextn.enorm.weight"), d("nextn.hnorm.weight"), d("nextn.shared_head_norm.weight")
        mla = Hy4MLA(g, il, lin=(lambda n: dl(n)) if self.fast else (lambda n: load_dense(g, n)))
        _mv(mla, dev, torch.float32)
        mla.post_norm, mla.sinks = mla.post_norm.to(dev), mla.sinks.to(dev)
        if self.fast and os.environ.get("HY4_DRAFT_MLA_BG", "0") == "1":         # absorb matrices wk_b / wv_b (fp32 einsum = rocBLAS SGEMM at 80-135 us each) as int8 + fp16 block scales on the trunk's batched mat-vec
            from . import mla_bgemv as _BGM
            mla.bg_kb, mla.bg_vb = _BGM.Q8Batched.from_float(mla.wk_b), _BGM.Q8Batched.from_float(mla.wv_b)
        mla.kpos_arange = True
        mla.tm = fw.tm
        self.mla = mla
        self.attn_norm, self.ffn_norm, self.post_ffw = d("attn_norm.weight"), d("ffn_norm.weight"), d("post_ffw_norm.weight")
        self.router, self.rbias = d("ffn_gate_inp.weight"), d("exp_probs_b.bias")
        self.rdown, self.rup = dl(b + "ffn_routed_down.weight"), dl(b + "ffn_routed_up.weight")
        self.shexp = [dl(b + f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down")]
        self.ex = [MI.Int8Stack(g, b + f"ffn_{n}_exps.weight", dev) for n in ("gate", "up", "down")]
        self.lim = float(fw.clamp[il])
        names = {"attn_k": "indexer.attn_k.weight", "k_norm": "indexer.k_norm.weight", "q_b": "indexer.attn_q_b.weight", "proj": "indexer.proj.weight",
                 "value": "indexer_compressor_value.weight", "gate": "indexer_compressor_gate.weight", "ape": "indexer_compressor_ape.weight",
                 "pool_norm": "indexer_compressor_norm.weight"}
        self.ix = FusedKPoolIndexer({k: d(s) for k, s in names.items()}, dev, **fw.ixc)
        # draft head: "" = the trunk's bf16 head (Triton mat-vec), "q8" = Q8_0 copy (half the bytes), "sub<N>" = the first N vocabulary rows only (draft picks among them)
        hm = os.environ.get("HY4_MTP_HEAD", "")
        self.head_q8 = self.head_sub = self.ids = None
        shared = getattr(fw, "out_q8", None)
        if shared is not None and hm in ("q8", "q8share"):        # HY4_HEAD_Q8=1: the trunk's Q8_0 head bytes ARE the draft's Q8_0 head: one resident copy
            self.head_q8 = shared
        elif hm == "q5k":          # Q5_K bytes of the bf16 head from a file (make_head_bin.py): 0.255 GB instead of 0.394 GB (Q8_0); falls back to Q8_0 when the file is missing
            from . import dense_req as _DR
            self.head_q8 = _DR.head_from_file(self, os.environ.get("HY4_MTP_HEAD_BIN", "/path/to/head_q5_k.bin"))
            if self.head_q8 is None:
                print("mtp_head: HY4_MTP_HEAD=q5k but no matching head file, using Q8_0", flush=True)
                self.head_q8 = shared if shared is not None else q8_linear(fw.out_w.float())
        elif hm == "q8":
            self.head_q8 = q8_linear(fw.out_w.float())
        elif hm.startswith("sub"):
            n = int(hm[3:])
            self.head_sub = fw.out_w[:n].contiguous()
            self.ids = torch.arange(n, device=dev)
        if shared is not None and self.head_q8 is None and self.head_sub is None:      # draft on "the trunk's head" = the shared Q8_0 bytes once the bf16 copy is gone
            self.head_q8 = shared
        st = _ns.state(fw)
        self.cap = cap or st.cap
        self.ar = st.ar if st.ar.shape[0] >= self.cap else torch.arange(self.cap, device=dev)
        self.kbuf = torch.zeros(self.cap, mla.R + mla.rope_dim, device=dev)
        SG.indexer_static_init(self.ix, self.cap)
        self.ix.gv = self.fast and os.environ.get("HY4_DRAFT_IX_GV", "0") == "1"      # indexer fp32 matrices on the mat-vec
        print(f"mtp_head: draft cfg fast={self.fast} moe_tiles={MI.GU_T}/{MI.DN_T} (flag {int(MI.TILES)}) mla_bg={getattr(mla, 'bg_kb', None) is not None and getattr(mla.bg_kb, 'draft', False)} (wk_b/wv_b int8) ix_gv={self.ix.gv}", flush=True)
        try:
            from . import attn_eager as _AE
            _AE.ensure(self.cap)
        except ImportError:
            pass
        if hasattr(fw, "free_bf16_head"):
            fw.free_bf16_head()              # HY4_HEAD_Q8=1: every consumer of the bf16 head values (draft head copies) has been built

    def reset(self):
        self.kbuf.zero_()
        self.ix.n, self.ix.P, self.ix.vg = 0, 0, None
        self.ix.pool.zero_()
        self.ix.vgb.zero_()

    def ffn(self, x):
        fw = self.fw
        if self.fast:
            from . import ref_fused as _RF
            from . import router_gemv as _RG
            lg = _RG.logits(x, self.router) if _RG.usable(x, self.router) else x @ self.router.T
            sel, w = _RF.router_select(lg, self.rbias, fw.topk, fw.w_scale)        # one launch, lowest id on ties
        else:
            p = torch.sigmoid(x @ self.router.T)
            sel = torch.topk(p + self.rbias, fw.topk, dim=-1).indices
            w = torch.gather(p, 1, sel)
            w = w / w.sum(-1, keepdim=True).clamp(min=6.103515625e-5) * fw.w_scale
        z = x @ self.rdown.T
        if x.shape[0] > 16:       # draft-cache fill of a prompt: rows grouped per expert
            lat = MI.experts_rows(self.ex[0], self.ex[1], self.ex[2], z.float().contiguous(), sel, w, self.lim)
        else:
            lat = torch.stack([MI.experts_forward(self.ex[0], self.ex[1], self.ex[2], z[r].contiguous(), sel[r].contiguous(), w[r].contiguous(), self.lim) for r in range(x.shape[0])])
        gt, up, dn = self.shexp
        sh = (torch.nn.functional.silu(x @ gt.T) * (x @ up.T)) @ dn.T
        return lat @ self.rup.T + sh

    @torch.no_grad()
    def step(self, tok, h, pos, N):
        """tok device long [T] (tokens j+1 ..), h [T, D] (trunk states h_j .., or the previous y of the chain), pos device long [T] (= j .., ascending), N rows
        attended (multiple of 4, > last position) -> y [T, D], logits of the LAST row [V]"""
        fw, mla, eps = self.fw, self.mla, self.eps
        nm = self._rmsw
        e = _ns.embed(fw, tok)
        x = torch.cat([nm(e, self.enorm), nm(h, self.hnorm)], -1) @ self.eh.T
        cur = nm(x, self.attn_norm)
        self.kbuf.index_copy_(0, pos, mla.kv(cur, pos))
        qr = nm(mla._q(cur, "wq_a") @ mla.wq_a.T, mla.q_a_norm)
        allow = SG.indexer_forward_static(self.ix, cur, qr, pos, N, self.ar)
        if self.fast:
            from . import ref_fused as _RF
            old = (_RF.ENABLED, os.environ.get("HY4_REF_MLA"))
            _RF.ENABLED = True
            os.environ["HY4_REF_MLA"] = "1"
            try:
                kqv, gated, out = mla.attend(cur, pos, self.kbuf[:N], self.ar[:N], allow)
            finally:
                _RF.ENABLED = old[0]
                if old[1] is None:
                    os.environ.pop("HY4_REF_MLA", None)
                else:
                    os.environ["HY4_REF_MLA"] = old[1]
        else:
            kqv, gated, out = mla.attend(cur, pos, self.kbuf[:N], self.ar[:N], allow)
        s1 = x + out
        f = self.ffn(nm(s1, self.ffn_norm))
        s2 = s1 + nm(f, self.post_ffw)
        y = nm(s2, self.shn)
        return y, self.logits(y[-1:].float())

    @torch.no_grad()
    def fill(self, tok, h, pos):
        """draft-cache fill of prompt rows: writes ONLY what the draft chain reads at decode (K row of every position, indexer value / gate rows, pool groups). All of it depends on the
        layer input alone (token embedding + trunk state), so it is the same state as step() leaves, without the queries, the selection, the attention and the 512-expert MoE whose outputs
        a fill discards. tok / h / pos as in step()."""
        nm = self._rmsw
        e = _ns.embed(self.fw, tok)
        x = torch.cat([nm(e, self.enorm), nm(h, self.hnorm)], -1) @ self.eh.T
        cur = nm(x, self.attn_norm)
        self.kbuf.index_copy_(0, pos, self.mla.kv(cur, pos))
        SG.indexer_keys_static(self.ix, cur, pos, self.ar)

    def prime(self, tok, h, pos, N):
        """one batched draft-cache fill call (HY4_PRIME_KVONLY=0: the full step, whose outputs are dropped)"""
        if os.environ.get("HY4_PRIME_KVONLY", "1") == "1":
            return self.fill(tok, h, pos)
        return self.step(tok, h, pos, N)

    def _rmsw(self, x, w):
        if self.fast:
            from . import attn_eager as _AE
            return _AE.rmsw(x, w, self.eps)
        return rms(x, self.eps) * w

    def logits(self, y):
        """y [1, D] fp32 -> logits [V'] of the draft's output head (V' = N rows with HY4_MTP_HEAD=sub<N>)"""
        if self.head_q8 is not None:
            return self.head_q8(y)[0]
        if self.head_sub is not None:
            from . import head_bf16 as _HB
            return _HB.gemv(y, self.head_sub)[0]
        return head_gemm(y, self.fw.out_w)[0]

    def pick(self, lg):
        """greedy draft token (device long [1]) from the draft logits"""
        t = torch.argmax(lg).reshape(1)
        return self.ids[t] if self.ids is not None else t

    def chain(self, tok_next, h, j, k):
        """greedy chain of k drafts from the pair (token j+1 = tok_next [device long [1]], trunk state h at position j): head rows j .. j+k-1 are written
        (row j is the committed pair, the others are scratch). -> list of k device long [1] tensors"""
        outs, tok = [], tok_next
        for i in range(k):
            jj = j + i
            N = ((jj + 1 + 3) // 4) * 4
            y, lg = self.step(tok, h, self.ar[jj:jj + 1], N)
            tok = self.pick(lg)
            outs.append(tok)
            h = y
        return outs


CONF = os.environ.get("HY4_DRAFT_CONF", "0") == "1"
CONF_ONE = float(1 << 20)


class MTPGraph:
    """the draft chain of k head steps as ONE HIP graph per context bucket: static inputs (token, trunk state, head position), greedy argmax feeds the next
    step on the device, output = the k drafted tokens [k] (device). No host work between the steps."""

    BUCKET = 512

    def __init__(self, mt, k):
        self.mt, self.k = mt, k
        dev = mt.dev
        D = mt.fw.D
        self.tok0 = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos0 = torch.zeros(1, dtype=torch.long, device=dev)
        self.h0 = torch.zeros(1, D, device=dev)
        self.hin = torch.zeros(2, dtype=torch.long).pin_memory()
        self.inp = torch.zeros(2, dtype=torch.long, device=dev)
        self.graphs, self.pool = {}, torch.cuda.graph_pool_handle()
        self.ev, self.ev_set = torch.cuda.Event(), False

    def body(self, N):
        mt = self.mt
        outs, tok, h = [], self.inp[0:1], self.h0
        pos = self.inp[1:2]
        confs = []
        for i in range(self.k):
            y, lg = mt.step(tok, h, pos + i if i else pos, N)
            tok = mt.pick(lg)
            outs.append(tok)
            if CONF:          # max softmax probability of the draft head, fixed point 2^-20, appended after the k tokens (cost-aware draft length)
                l32 = lg.float()
                confs.append((torch.exp(l32.max() - torch.logsumexp(l32, 0)) * CONF_ONE).long().reshape(1))
            h = y
        return torch.cat(outs + confs)

    def _graph(self, N):
        g = self.graphs.get(N)
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
        g = self.graphs[N] = (cg, out)
        return g

    def precapture(self, ns=None):
        """capture the draft graph of every context bucket now (default: every bucket up to the cap); -> number built"""
        mt = self.mt
        ns = sorted(set(ns)) if ns is not None else list(range(self.BUCKET, mt.cap + 1, self.BUCKET))
        built = 0
        for N in ns:
            if N not in self.graphs:
                self._graph(N)
                built += 1
        return built

    def run(self, tok, h, j):
        """tok: host int (the token of the pair, x_{j+1}), h: device [1, D] trunk state h_j, j: head position -> device tensor [k] of drafts"""
        mt = self.mt
        N = min(mt.cap, ((j + self.k) // self.BUCKET + 1) * self.BUCKET)
        if self.ev_set:
            self.ev.synchronize()
        self.hin[0], self.hin[1] = int(tok), int(j)
        self.inp.copy_(self.hin, non_blocking=True)
        self.h0.copy_(h)
        self.ev.record()
        self.ev_set = True
        cg, out = self._graph(N)
        cg.replay()
        return out


class MTPGraphCU(MTPGraph):
    """draft chain WITH catch-up: after a verify round that accepted `a` drafts, the head rows of the accepted positions are rewritten with their TRUE pairs
    (draft token, trunk hidden of that position) and the new pair is appended, all in the first head step (a + 1 rows); then k - 1 single-row chain steps.
    One graph per (bucket, a)."""

    def __init__(self, mt, k):
        super().__init__(mt, k)
        dev = mt.dev
        self.hs = torch.zeros(k + 1, mt.fw.D, device=dev)
        self.hin = torch.zeros(k + 2, dtype=torch.long).pin_memory()
        self.inp = torch.zeros(k + 2, dtype=torch.long, device=dev)         # [tokens (k + 1), first position]
        self.arr = torch.arange(k + 1, device=dev)

    def body(self, N, a):
        mt, k = self.mt, self.k
        tok, pos0 = self.inp[:a + 1], self.inp[k + 1:k + 2]
        y, lg = mt.step(tok, self.hs[:a + 1], pos0 + self.arr[:a + 1], N)
        t = mt.pick(lg)
        outs, h = [t], y[-1:]
        for i in range(1, k):
            y, lg = mt.step(t, h, pos0 + (a + i), N)
            t = mt.pick(lg)
            outs.append(t)
            h = y
        return torch.cat(outs)

    def _graph(self, N, a):
        g = self.graphs.get((N, a))
        if g is not None:
            return g
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.body(N, a)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg, pool=self.pool):
            out = self.body(N, a)
        g = self.graphs[(N, a)] = (cg, out)
        return g

    def run(self, toks, h_rows, pos0):
        """toks: host ints [a + 1] (accepted drafts then the new token), h_rows [a + 1, D] device (trunk hidden of the positions pos0 .. pos0 + a),
        pos0: first head position -> device [k] drafts"""
        mt, a = self.mt, len(toks) - 1
        N = min(mt.cap, ((pos0 + a + self.k) // self.BUCKET + 1) * self.BUCKET)
        if self.ev_set:
            self.ev.synchronize()
        self.hin[:a + 1] = torch.tensor([int(t) for t in toks])
        self.hin[self.k + 1] = int(pos0)
        self.inp.copy_(self.hin, non_blocking=True)
        self.hs[:a + 1].copy_(h_rows)
        self.ev.record()
        self.ev_set = True
        cg, out = self._graph(N, a)
        cg.replay()
        return out


import os
_STAGE_OV = [False]


def spec_generate(gd, mg, y, hcur, n_new, cu=False, timers=None):
    """Greedy speculative decode on the whole-step graph. gd = GraphDecoder (the trunk, context already in it), mg = MTPGraph / MTPGraphCU (k drafts per round),
    y = the trunk's greedy token after the context (host int, NOT yet fed), hcur = the trunk hidden [1, D] of the last context position, n_new = tokens wanted.
    Each round: the draft chain proposes k tokens, ONE T = k + 1 row verify replay checks them, the longest agreeing prefix plus the trunk's own next token is kept
    (the output equals plain greedy decoding of the trunk), K / indexer rows of rejected positions are overwritten by the next round.
    Returns (tokens [>= n_new, up to k more], accepted-length histogram, rounds). timers (list of 3 floats, optional) accumulates seconds in: draft chain launch +
    wait, host stage + verify launch, verify wait."""
    fw, k = gd.fw, mg.k
    out, hist, rounds = [], [0] * (k + 1), 0
    cu_toks, cu_h, cu_p = [y], hcur, fw.n - 1
    pc = time.perf_counter
    while len(out) < n_new:
        j = fw.n - 1
        ta = pc()
        dd = mg.run(cu_toks, cu_h, cu_p) if cu else mg.run(y, hcur, j)
        if _STAGE_OV[0]:
            gd.prestage(y)
        d = dd.tolist()
        tb = pc()
        toks = [y] + d
        _lg, amv, _x, hv = gd.step_rows(toks)
        tc = pc()
        amh = amv.tolist()
        td = pc()
        if timers is not None:
            timers[0] += tb - ta
            timers[1] += tc - tb
            timers[2] += td - tc
        acc = 0
        while acc < k and d[acc] == amh[acc]:
            acc += 1
        out += toks[:acc + 1]
        gd.rewind(j + 2 + acc)
        y, hcur = amh[acc], hv[acc:acc + 1]
        cu_toks, cu_h, cu_p = d[:acc] + [y], hv[:acc + 1], j + 1
        hist[acc] += 1
        rounds += 1
    return out, hist, rounds
