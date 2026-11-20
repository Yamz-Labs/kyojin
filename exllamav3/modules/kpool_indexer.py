"""Hy4 DSA indexer, k-pool variant, in ROW-LIST form (llama.cpp hy-v4-flash.cpp build_kpool_select, 769-905).

Per layer it keeps a pooled key per complete group of `kpool` consecutive positions. For a query token at position p it
scores the groups that ended before p, keeps the best `top_k // kpool - select_tail` groups and the tokens of p's own
group up to p (the "tail"), and returns the key ROWS (positions) of that selection, one int32 list per query, plus the
scores. No dense mask is built; the list is what the sparse attention gathers.

Maths (all fp32, CPU; the model's norms are weight-only RMSNorm):
    ik      = rms(x @ Wk^T) * k_norm                          x = attn_norm output   [n, n_embd] -> [n, d]
    value   = ik @ Wv^T ; gate = ik @ Wg^T                    [n, d] each
    pooled  = rope_tail(rms(sum_k softmax_k(gate_k + ape_k) * value_k) * pool_norm, pos = first of the group)
    iq      = rope_tail((qr @ Wq^T).reshape(n, H, d), pos = p)    qr = q_a_norm(wq_a x), the main attention's lora residual
    w       = (x @ Wp^T) / sqrt(d * H)                        [n, H]
    score   = sum_h relu(iq_h . pooled_j) * w_h               [n, n_pool]
rope_tail rotates only the trailing `rope_dim` of each d-vector, NORM pairing (x[2i], x[2i+1]), theta = pos * base^(-2i/rope_dim).
Visible groups for the token at p: group g with its last position 4g+3 < p when select_tail (the group holding p is the tail),
<= p otherwise. Ties between equal scores: ggml's CPU top-k is std::partial_sort with a '>' comparator (order of ties
unspecified); here the lowest group index wins (stable sort), the only deterministic choice.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch


def rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tail(x, pos, rope_dim, base):
    """x [..., n, (H,) d]; pos [n] int. Rotates x[..., -rope_dim:] (NORM pairs); the nope front is untouched."""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim)               # [rope_dim/2]
    ang = pos.double()[:, None] * inv[None]                                                     # [n, rope_dim/2]
    c, s = ang.cos().float(), ang.sin().float()
    shp = [x.shape[0]] + [1] * (x.dim() - 2) + [rope_dim // 2]
    c, s = c.reshape(shp), s.reshape(shp)
    t = x[..., d - rope_dim:]
    a, b = t[..., 0::2], t[..., 1::2]
    out = torch.empty_like(t)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return torch.cat([x[..., :d - rope_dim], out], -1)


def q8_act(x):
    """ggml Q8_0 activation quantisation (what the CPU mul_mat does to the activations before a Q8_0 weight dot):
    blocks of 32, d = amax/127, q = roundf(x/d) (half away from zero), d stored fp16."""
    shp = x.shape
    b = x.float().reshape(-1, 32)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    v = b * idv
    q = torch.sign(v) * torch.floor(v.abs() + 0.5)
    return (q * d.half().float()).reshape(shp)


@dataclass
class IndexerSelection:
    rows: list            # per query: int32 tensor, ascending key positions (selected groups' members + tail)
    scores: torch.Tensor  # [n, n_pool] fp32, -inf where the group is not visible to the query
    pools: list           # per query: int32 tensor, the picked group ids (descending score)
    pos: torch.Tensor     # [n] query positions


class KPoolIndexer:
    """w: dict of fp32 tensors (out, in): attn_k [d, E], k_norm [d], q_b [H*d, R], proj [H, E], value [d, d], gate [d, d],
    ape [kpool, d], pool_norm [d]. q8_weights: names whose weights are Q8_0 in the file (activation quantised when act_q8)."""

    def __init__(self, w, n_head=16, head_dim=128, top_k=2048, kpool=4, select_tail=True, rope_dim=64,
                 rope_base=1e7, eps=1e-5, act_q8=False, q8_weights=("attn_k", "q_b", "value", "gate")):
        self.w, self.H, self.d, self.top_k, self.kpool = w, n_head, head_dim, top_k, kpool
        self.tail, self.rope_dim, self.base, self.eps = int(bool(select_tail)), rope_dim, rope_base, eps
        self.act_q8, self.q8 = act_q8, set(q8_weights)
        assert top_k % kpool == 0 and top_k // kpool - self.tail >= 1 and rope_dim < head_dim
        self.n_top = top_k // kpool - self.tail
        self.reset()

    def reset(self):
        self.n = 0
        self.vg = []                      # (value, gate) per token not yet pooled: [m, 2d]
        self.pooled = torch.empty(0, self.d)

    def _mm(self, name, x):
        if self.act_q8 and name in self.q8:
            x = q8_act(x)
        return x @ self.w[name].T

    def _pool(self, vg, g0):
        """vg [m*kpool, 2d] complete groups starting at group g0 -> pooled keys [m, d]"""
        k, d = self.kpool, self.d
        v, g = vg[:, :d].reshape(-1, k, d), vg[:, d:].reshape(-1, k, d)
        p = (torch.softmax(g + self.w["ape"][None], dim=1) * v).sum(1)
        p = rms(p, self.w["pool_norm"], self.eps)
        return rope_tail(p, torch.arange(g0, g0 + p.shape[0]) * k, self.rope_dim, self.base)

    @torch.no_grad()
    def forward(self, x, qr) -> IndexerSelection:
        """x [n, E] attn_norm output, qr [n, R] q_a_norm output, for n consecutive new tokens (positions n_seen ...).
        One call = one prefill chunk or one decode step; the pooled keys persist between calls."""
        n, k, d, H = x.shape[0], self.kpool, self.d, self.H
        pos = torch.arange(self.n, self.n + n)
        ik = rms(self._mm("attn_k", x), self.w["k_norm"], self.eps)
        vg_new = torch.cat([self._mm("value", ik), self._mm("gate", ik)], -1)
        self.vg.append(vg_new)
        buf = torch.cat(self.vg, 0) if len(self.vg) > 1 else self.vg[0]
        g0 = self.pooled.shape[0]
        m = buf.shape[0] // k                                    # complete groups available (the buffer starts at group g0)
        if m:
            self.pooled = torch.cat([self.pooled, self._pool(buf[:m * k], g0)], 0)
        self.vg = [buf[m * k:]] if buf.shape[0] - m * k else []
        self.n += n
        iq = rope_tail(self._mm("q_b", qr).reshape(n, H, d), pos, self.rope_dim, self.base)
        wt = (x.float() @ self.w["proj"].T) / math.sqrt(d * H)                      # F32 tensor: no quantisation
        pk = self.pooled                                                             # [P, d]
        kq = torch.relu(torch.einsum("nhd,pd->nhp", iq, pk))
        sc = (kq * wt[:, :, None]).sum(1)                                            # [n, P]
        P = pk.shape[0]
        gend = torch.arange(P) * k + (k - 1)
        vis = (gend[None] < pos[:, None]) if self.tail else (gend[None] <= pos[:, None])
        sc = torch.where(vis, sc, torch.full_like(sc, -math.inf))
        nvis = vis.sum(1)
        order = torch.sort(-sc, dim=1, stable=True).indices if P else sc.long()      # best first, lowest index on ties
        rows, pools = [], []
        for t in range(n):
            nt = min(int(nvis[t]), self.n_top)
            g = order[t, :nt]
            p = int(pos[t])
            mem = (g[:, None] * k + torch.arange(k)[None]).reshape(-1)
            tail = torch.arange(p - (p % k), p + 1) if self.tail else torch.arange(p - ((p + 1) % k) + 1, p + 1)
            rows.append(torch.cat([mem, tail]).sort().values.int())
            pools.append(g.int())
        return IndexerSelection(rows, sc, pools, pos)
