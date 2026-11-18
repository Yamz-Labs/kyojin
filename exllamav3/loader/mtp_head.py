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
        lat = MI.experts_forward(self.ex[0], self.ex[1], self.ex[2], z[0].contiguous(), sel[0].contiguous(), w[0].contiguous(), self.lim)
        gt, up, dn = self.shexp
        sh = (torch.nn.functional.silu(x @ gt.T) * (x @ up.T)) @ dn.T
        return lat[None] @ self.rup.T + sh

    @torch.no_grad()
    def step(self, tok, h, pos, N):
        """tok device long [1] (token j+1), h [1, D] (trunk state h_j, or the previous y of the chain), pos device long [1] (= j), N rows attended (multiple of 4,
        > j) -> y [1, D], logits [V]"""
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
        return y, head_gemm(y.float(), fw.out_w)[0]

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
