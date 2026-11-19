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

import torch

from . import moe_i8 as MI
from . import step_graph as SG
from . import step_nosync as _ns
from .engram_rows import load_dense, rms
from .hy4_forward import _mv, head_gemm
from .indexer_fused import FusedKPoolIndexer
from .mla_hy4 import Hy4MLA


class Hy4MTP:
    def __init__(self, fw, cap=None):
        g, dev, il = fw.g, fw.dev, fw.n_trunk
        b = f"blk.{il}."
        self.fw, self.il, self.dev, self.eps = fw, il, dev, fw.eps
        d = lambda n: load_dense(g, b + n).to(dev)
        self.eh, self.enorm, self.hnorm, self.shn = d("nextn.eh_proj.weight"), d("nextn.enorm.weight"), d("nextn.hnorm.weight"), d("nextn.shared_head_norm.weight")
        mla = Hy4MLA(g, il, lin=lambda n: load_dense(g, n))
        _mv(mla, dev, torch.float32)
        mla.post_norm, mla.sinks = mla.post_norm.to(dev), mla.sinks.to(dev)
        mla.kpos_arange = True
        mla.tm = fw.tm
        self.mla = mla
        self.attn_norm, self.ffn_norm, self.post_ffw = d("attn_norm.weight"), d("ffn_norm.weight"), d("post_ffw_norm.weight")
        self.router, self.rbias = d("ffn_gate_inp.weight"), d("exp_probs_b.bias")
        self.rdown, self.rup = d("ffn_routed_down.weight"), d("ffn_routed_up.weight")
        self.shexp = [d(f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down")]
        self.ex = [MI.Int8Stack(g, b + f"ffn_{n}_exps.weight", dev) for n in ("gate", "up", "down")]
        self.lim = float(fw.clamp[il])
        names = {"attn_k": "indexer.attn_k.weight", "k_norm": "indexer.k_norm.weight", "q_b": "indexer.attn_q_b.weight", "proj": "indexer.proj.weight",
                 "value": "indexer_compressor_value.weight", "gate": "indexer_compressor_gate.weight", "ape": "indexer_compressor_ape.weight",
                 "pool_norm": "indexer_compressor_norm.weight"}
        self.ix = FusedKPoolIndexer({k: d(s) for k, s in names.items()}, dev, **fw.ixc)
        st = _ns.state(fw)
        self.cap = cap or st.cap
        self.ar = st.ar if st.ar.shape[0] >= self.cap else torch.arange(self.cap, device=dev)
        self.kbuf = torch.zeros(self.cap, mla.R + mla.rope_dim, device=dev)
        SG.indexer_static_init(self.ix, self.cap)
        try:
            from . import attn_eager as _AE
            _AE.ensure(self.cap)
        except ImportError:
            pass

    def reset(self):
        self.kbuf.zero_()
        self.ix.n, self.ix.P, self.ix.vg = 0, 0, None
        self.ix.pool.zero_()
        self.ix.vgb.zero_()

    def ffn(self, x):
        fw = self.fw
        p = torch.sigmoid(x @ self.router.T)
        sel = torch.topk(p + self.rbias, fw.topk, dim=-1).indices
        w = torch.gather(p, 1, sel)
        w = w / w.sum(-1, keepdim=True).clamp(min=6.103515625e-5) * fw.w_scale
        z = x @ self.rdown.T
        lat = torch.stack([MI.experts_forward(self.ex[0], self.ex[1], self.ex[2], z[r].contiguous(), sel[r].contiguous(), w[r].contiguous(), self.lim) for r in range(x.shape[0])])
        gt, up, dn = self.shexp
        sh = (torch.nn.functional.silu(x @ gt.T) * (x @ up.T)) @ dn.T
        return lat @ self.rup.T + sh

    @torch.no_grad()
    def step(self, tok, h, pos, N):
        """tok device long [T] (tokens j+1 ..), h [T, D] (trunk states h_j .., or the previous y of the chain), pos device long [T] (= j .., ascending), N rows
        attended (multiple of 4, > last position) -> y [T, D], logits of the LAST row [V]"""
        fw, mla, eps = self.fw, self.mla, self.eps
        e = _ns.embed(fw, tok)
        x = torch.cat([rms(e, eps) * self.enorm, rms(h, eps) * self.hnorm], -1) @ self.eh.T
        cur = rms(x, eps) * self.attn_norm
        self.kbuf.index_copy_(0, pos, mla.kv(cur, pos))
        qr = rms(mla._q(cur, "wq_a") @ mla.wq_a.T, eps) * mla.q_a_norm
        allow = SG.indexer_forward_static(self.ix, cur, qr, pos, N, self.ar)
        kqv, gated, out = mla.attend(cur, pos, self.kbuf[:N], self.ar[:N], allow)
        s1 = x + out
        f = self.ffn(rms(s1, eps) * self.ffn_norm)
        s2 = s1 + rms(f, eps) * self.post_ffw
        y = rms(s2, eps) * self.shn
        return y, head_gemm(y[-1:].float(), fw.out_w)[0]

    def chain(self, tok_next, h, j, k):
        """greedy chain of k drafts from the pair (token j+1 = tok_next [device long [1]], trunk state h at position j): head rows j .. j+k-1 are written
        (row j is the committed pair, the others are scratch). -> list of k device long [1] tensors"""
        outs, tok = [], tok_next
        for i in range(k):
            jj = j + i
            N = ((jj + 1 + 3) // 4) * 4
            y, lg = self.step(tok, h, self.ar[jj:jj + 1], N)
            tok = torch.argmax(lg).reshape(1)
            outs.append(tok)
            h = y
        return outs


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
        for i in range(self.k):
            y, lg = mt.step(tok, h, pos + i if i else pos, N)
            tok = torch.argmax(lg).reshape(1)
            outs.append(tok)
            h = y
        return torch.cat(outs)

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
        t = torch.argmax(lg).reshape(1)
        outs, h = [t], y[-1:]
        for i in range(1, k):
            y, lg = mt.step(t, h, pos0 + (a + i), N)
            t = torch.argmax(lg).reshape(1)
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
