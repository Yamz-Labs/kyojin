"""HY4_ATTN_GROUPED=1: attention projections that read the SAME input row run as ONE dense GEMV launch on the stored bytes.

Generic by shape / stored type: any set of QLinear (hy4_forward.py) with equal K, stored ggml type and GEMV is concatenated along N
(rows of the stored [N, K/blk, bb] bytes; the per-row dot never depends on N, so each output row is bit-identical to the separate
launch). The originals are re-pointed to slices of the concatenated buffer (no extra resident bytes). One activation quantisation
and one launch replace n. Results are cached per input tensor (identity + version), so a later projection of the same row
(wkv_a, wgate, the indexer's repeated wq_a) is a cache hit. Rows > DENSE_MAXR (prefill) use each member's own path unchanged.
"""
from __future__ import annotations
import torch

from .hy4_forward import QLinear, DENSE_MAXR

# attribute names of the MLA object that read the layer input `cur` (wq_a, wkv_a, wgate); keyed by role, not by model
INPUT_SHARED = ("wq_a", "wkv_a", "wgate")


class _Group:
    def __init__(self, lins):
        self.sizes = [q.shape[0] for q in lins]
        self.offs = [sum(self.sizes[:i]) for i in range(len(lins))]
        self.K, self.gemv, self.stats = lins[0].shape[1], lins[0].gemv, lins[0].stats
        self.packed = torch.cat([q.packed for q in lins], 0)
        self.ref = None                       # (x, x._version, outs)

    def run(self, x):
        r = self.ref
        if r is not None and r[0] is x and r[1] == x._version:
            return r[2]
        x2 = x.reshape(-1, self.K)
        xf = x2.float().contiguous()
        out = torch.empty(xf.shape[0], sum(self.sizes), dtype=torch.float32, device=x.device)
        self.gemv(xf, self.packed, out)
        if self.stats is not None:
            self.stats["dense_gemv"] = self.stats.get("dense_gemv", 0) + 1
            self.stats["attn_grouped_calls"] = self.stats.get("attn_grouped_calls", 0) + 1
        lead = x.shape[:-1]
        outs = [out[:, o:o + n].to(x.dtype).reshape(*lead, n) for o, n in zip(self.offs, self.sizes)]
        self.ref = (x, x._version, outs)
        return outs


class _GT:
    def __init__(self, v):
        self.v = v

    def __rmatmul__(self, x):
        return self.v(x)


class GroupedView(QLinear):
    """a member of a group: same attributes as the QLinear, decode rows go through the shared launch"""

    def __init__(self, lin, group, idx, packed_view):
        super().__init__(packed_view, lin.ty, lin.spec, lin.gemv, lin.stats)
        self.group, self.idx, self.orig_call = group, idx, QLinear.__call__

    @property
    def T(self):
        return _GT(self)

    def __call__(self, x):
        if x.numel() // self.shape[1] <= DENSE_MAXR and self.gemv is not None:
            return self.group.run(x)[self.idx]
        return QLinear.__call__(self, x)


def group_linears(lins):
    """list of QLinear reading the same input -> list of GroupedView (same order), or None when they cannot share one launch"""
    if len(lins) < 2 or not all(isinstance(q, QLinear) and q.gemv is not None for q in lins):
        return None
    k = (lins[0].shape[1], lins[0].ty, lins[0].gemv)
    if any((q.shape[1], q.ty, q.gemv) != k for q in lins) or len({q.packed.device for q in lins}) != 1:
        return None
    grp = _Group(lins)
    return [GroupedView(q, grp, i, grp.packed[o:o + n]) for i, (q, o, n) in enumerate(zip(lins, grp.offs, grp.sizes))]


def install(mla, names=INPUT_SHARED):
    """regroup mla.<names> in place; returns True when grouped (originals' bytes are freed, the views share one buffer)"""
    lins = [getattr(mla, n, None) for n in names]
    views = group_linears(lins) if all(l is not None for l in lins) else None
    if views is None:
        return False
    for n, v in zip(names, views):
        setattr(mla, n, v)
    return True
