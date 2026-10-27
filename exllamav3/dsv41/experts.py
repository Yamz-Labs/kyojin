"""
Expert-provider API for the routed experts.

The model never touches routed-expert weights directly: for every MoE layer it asks a provider for
the experts the router picked. A tiered store (RAM hot set + SSD, branch feat/tier1) implements the
same four calls, so it plugs in without graph changes:

    total_experts(layer) -> int
    prefetch(layer, experts)         async hint, never blocks, may be ignored
    lookup(layer, expert) -> ExpertHandle     returns a resident expert, blocking on I/O if needed
    release(handle)                  the model is done with it (unpin)
    resident_bytes(layer) -> int     planner accounting

`layer` is the model layer index (MTP layers continue after the trunk: n_layers + j).
"""
from __future__ import annotations
import os
from typing import Iterable
import torch
import torch.nn.functional as F
from .linear import Lin
from . import ops, dec2


def _fuse_on(device) -> bool:
    try:
        from ..modules.dsv41_fn import dsv41_fuse as FZ
    except Exception:
        return False
    return FZ.available(device)


class ExpertHandle:
    """One resident routed expert (w1 = gate, w3 = up, w2 = down)."""

    def __init__(self, layer: int, expert: int, w1: Lin, w3: Lin, w2: Lin, on_release=None):
        self.layer, self.expert = layer, expert
        self.w1, self.w3, self.w2 = w1, w3, w2
        self._on_release = on_release

    def ffn(self, x: torch.Tensor, weights, limit: float) -> torch.Tensor:
        """model.py:843. weights [n,1] fp32 routing weights (or None for a shared expert)."""
        dtype = x.dtype
        if (weights is None and dtype == torch.half and x.is_cuda and os.environ.get("DSV41_FUSE_SWIGLU", "1") == "1"
                and _fuse_on(x.device)):
            gu = dec2.pair(self, "SHW", (self.w1, self.w3), x) if x.dim() == 2 and x.size(0) == 1 else None
            g, u = gu if gu is not None else (self.w1(x), self.w3(x))
            if g.dtype == torch.half and u.dtype == torch.half and g.is_contiguous() and u.is_contiguous():
                from ..modules.dsv41_fn import dsv41_fuse as FZ
                return self.w2(FZ.swiglu_h(g, u, limit))          # one launch for float/clamp/silu/mul/half (bit-identical)
            return self.w2(ops.swiglu(g, u, limit).to(dtype))
        h = ops.swiglu(self.w1(x), self.w3(x), limit)
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))

    def nbytes(self):
        return self.w1.nbytes() + self.w2.nbytes() + self.w3.nbytes()


class ExpertProvider:
    def total_experts(self, layer: int) -> int:  # pragma: no cover
        raise NotImplementedError

    def prefetch(self, layer: int, experts: Iterable[int]) -> None:
        return None

    def lookup(self, layer: int, expert: int) -> ExpertHandle:  # pragma: no cover
        raise NotImplementedError

    def release(self, handle: ExpertHandle) -> None:
        return None

    def resident_bytes(self, layer: int) -> int:
        return 0


class SourceExpertProvider(ExpertProvider):
    """Builds experts on demand from a WeightSource (official safetensors, GGUF, or a dict).
    cache=True keeps built experts resident (what a fully RAM-resident model does); cache=False drops
    them on release (streaming)."""

    def __init__(self, source, counts: dict, cache: bool = True, dtype=torch.float32, device="cpu"):
        self.src, self.counts, self.cache = source, dict(counts), cache
        self.dtype, self.device = dtype, device
        self._res = {}
        self.lookups = 0
        self.loads = 0

    def total_experts(self, layer):
        return self.counts[layer]

    def lookup(self, layer, expert):
        self.lookups += 1
        k = (layer, expert)
        h = self._res.get(k)
        if h is None:
            self.loads += 1
            h = self.src.expert(layer, expert, self.dtype, self.device)
            if self.cache:
                self._res[k] = h
        return h

    def resident_bytes(self, layer):
        return sum(h.nbytes() for (l, _), h in self._res.items() if l == layer)
