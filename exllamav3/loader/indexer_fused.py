"""Device-resident k-pool indexer: pooled keys cached on the device, score + group top-k + attention mask built with
torch ops only and NO host sync (every size is derived from the host-side token counter).

Same maths and op order as modules/kpool_indexer.KPoolIndexer (so on one device it is bit-exact); differences:
  * pooled keys live in a preallocated [cap, d] buffer, a completed group is written in place (no torch.cat of the pool);
  * the token-position rope table (cos/sin) is built once and grown, not recomputed per step in float64;
  * selection = topk over the visible group scores -> bool group mask -> allow [T, N] (members | tail), all on device;
  * returns allow=None (host-decided, no sync) when every query sees all of its causal keys.
Generic by (n_head, head_dim, top_k, kpool, select_tail, rope_dim); nothing keyed on a model name.
Ties at the top-k boundary: torch.topk order is unspecified (the CPU reference takes the lowest group index).
"""
from __future__ import annotations
import math
import torch


def _rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class FusedKPoolIndexer:
    """w: name -> weight with `x @ w.T` support (fp32 tensor or a Q8 QLinear) for attn_k, q_b, value, gate; dense tensors
    for proj [H, E], k_norm [d], ape [kpool, d], pool_norm [d]. Small dense ones are moved to `device`."""

    def __init__(self, w, device, n_head=16, head_dim=128, top_k=2048, kpool=4, select_tail=True, rope_dim=64,
                 rope_base=1e7, eps=1e-5, cap=4096):
        self.dev = torch.device(device)
        self.w = {k: (v.to(self.dev) if isinstance(v, torch.Tensor) else v) for k, v in w.items()}
        self.H, self.d, self.top_k, self.k = n_head, head_dim, top_k, kpool
        self.tail, self.rope_dim, self.base, self.eps = int(bool(select_tail)), rope_dim, rope_base, eps
        assert top_k % kpool == 0 and top_k // kpool - self.tail >= 1 and rope_dim < head_dim
        self.n_top = top_k // kpool - self.tail
        self.n = 0                                                  # tokens seen (host int)
        self.P = 0                                                  # pooled groups stored (host int)
        self.pool = torch.zeros(max(cap // kpool, 16), head_dim, device=self.dev)
        self.vg = None                                              # [<kpool, 2d] tokens of the open group
        self._cs = None                                             # (cos, sin) [Lpos, rope_dim/2] fp32
        self.last_scores = None

    _TABS = {}      # (base, rope_dim, device) -> (cos, sin): one fp64 evaluation per process, not one per indexer layer (13 x 34 ms at 8K)

    # -- rope table, built like the reference (float64 angle, then fp32 cos / sin)
    def _table(self, upto):
        if self._cs is None or self._cs[0].shape[0] < upto:
            key = (self.base, self.rope_dim, str(self.dev))
            cs = self._TABS.get(key)
            if cs is None or cs[0].shape[0] < upto:
                L = max(upto, 2 * (cs[0].shape[0] if cs else 0), 4096)
                inv = self.base ** (-torch.arange(0, self.rope_dim, 2, dtype=torch.float64) / self.rope_dim)
                ang = torch.arange(L, dtype=torch.float64)[:, None] * inv[None]
                cs = self._TABS[key] = (ang.cos().float().to(self.dev), ang.sin().float().to(self.dev))
            self._cs = cs
        return self._cs

    def _rope(self, x, pos):
        """x [n, (H,) d], pos [n] long (device). Rotates the trailing rope_dim (NORM pairs)."""
        c, s = self._table(self._max_pos)
        c, s = c[pos], s[pos]
        sh = [x.shape[0]] + [1] * (x.dim() - 2) + [self.rope_dim // 2]
        c, s = c.reshape(sh), s.reshape(sh)
        d = x.shape[-1]
        t = x[..., d - self.rope_dim:]
        a, b = t[..., 0::2], t[..., 1::2]
        out = torch.empty_like(t)
        out[..., 0::2] = a * c - b * s
        out[..., 1::2] = a * s + b * c
        return torch.cat([x[..., :d - self.rope_dim], out], -1)

    def _pool_groups(self, vg, g0):
        k, d = self.k, self.d
        v, g = vg[:, :d].reshape(-1, k, d), vg[:, d:].reshape(-1, k, d)
        p = (torch.softmax(g + self.w["ape"][None], dim=1) * v).sum(1)
        p = _rms(p, self.w["pool_norm"], self.eps)
        return self._rope(p, (torch.arange(g0, g0 + p.shape[0], device=self.dev) * k))

    @torch.no_grad()
    def forward(self, x, qr, n_keys=None):
        """x [T, E] attn_norm output, qr [T, R] q_a_norm output (both on the device), T consecutive new tokens.
        n_keys = rows of the attention key cache after this call (default: tokens seen). -> allow [T, n_keys] bool or None."""
        T, k, d, H = x.shape[0], self.k, self.d, self.H
        start = self.n
        self._max_pos = start + T
        pos = torch.arange(start, start + T, device=self.dev)
        ik = _rms(x @ self.w["attn_k"].T, self.w["k_norm"], self.eps)
        vg_new = torch.cat([ik @ self.w["value"].T, ik @ self.w["gate"].T], -1)
        buf = vg_new if self.vg is None else torch.cat([self.vg, vg_new], 0)
        m = buf.shape[0] // k
        if m:
            if self.P + m > self.pool.shape[0]:
                big = torch.empty(max(2 * self.pool.shape[0], self.P + m), d, device=self.dev)
                big[:self.P] = self.pool[:self.P]
                self.pool = big
            self.pool[self.P:self.P + m] = self._pool_groups(buf[:m * k], self.P)
            self.P += m
        self.vg = buf[m * k:] if buf.shape[0] - m * k else None
        self.n += T
        iq = self._rope((qr @ self.w["q_b"].T).reshape(T, H, d), pos)
        wt = (x.float() @ self.w["proj"].T) / math.sqrt(d * H)
        P = self.P
        pk = self.pool[:P]
        kq = torch.relu(torch.einsum("nhd,pd->nhp", iq, pk))
        sc = (kq * wt[:, :, None]).sum(1)                                   # [T, P]
        gend = torch.arange(P, device=self.dev) * k + (k - 1)
        vis = (gend[None] < pos[:, None]) if self.tail else (gend[None] <= pos[:, None])
        sc = torch.where(vis, sc, torch.full_like(sc, -math.inf))
        self.last_scores = sc
        # host-side: does any query see fewer keys than its causal range? nvis(p) = p // k (tail) or (p + 1) // k
        last = start + T - 1
        nvis_max = last // k if self.tail else (last + 1) // k
        if nvis_max <= self.n_top:
            return None
        N = self.n if n_keys is None else n_keys
        idx = torch.topk(sc, min(self.n_top, P), dim=1).indices            # [T, n_top]
        gsel = torch.zeros(T, P, dtype=torch.bool, device=self.dev)
        gsel.scatter_(1, idx, True)
        gsel &= vis                                                         # picked -inf slots (fewer visible than n_top)
        allow = torch.zeros(T, N, dtype=torch.bool, device=self.dev)
        allow[:, :P * k] = gsel.repeat_interleave(k, dim=1)
        j = torch.arange(N, device=self.dev)[None]
        lo = (pos - (pos % k)) if self.tail else (pos - ((pos + 1) % k) + 1)
        allow |= (j >= lo[:, None]) & (j <= pos[:, None])
        return allow
