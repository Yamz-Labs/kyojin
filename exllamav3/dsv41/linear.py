"""Linear layer objects of the DS4.1 graph. The graph only ever calls `lin(x)`; how the weight is stored
(float, official fp8/fp4 block-scaled, GGUF quant) is the loader's business."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from . import quant


class Lin:
    out_features: int
    in_features: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def nbytes(self) -> int:
        return 0

    def to(self, device):
        return self


class DenseLin(Lin):
    """y = x @ w.T. Weight is held in the dtype of its site (activations are cast to it if they differ)."""

    def __init__(self, w: torch.Tensor):
        self.w = w
        self.out_features, self.in_features = w.shape

    def __call__(self, x):
        w = self.w if self.w.dtype == x.dtype else self.w.to(x.dtype)
        return F.linear(x, w)

    def nbytes(self):
        return self.w.numel() * self.w.element_size()

    def to(self, device):
        return DenseLin(self.w.to(device))


class QuantLin(Lin):
    """Official-checkpoint fp8 / fp4 linear: the activation is fake-quantised to fp8 (ue8m0 scale, block 32)
    like the reference `linear()`, the weight is decoded once to `dtype`; product in that dtype.
    `act_quant=False` skips the activation step (GGUF sources have no such training convention)."""

    def __init__(self, w_deq: torch.Tensor, act_quant: bool = True, block: int = 32):
        self.w = w_deq
        self.act_quant = act_quant
        self.block = block
        self.out_features, self.in_features = w_deq.shape

    def __call__(self, x):
        if self.act_quant:
            x = quant.act_quant_fp8(x, self.block)
        w = self.w if self.w.dtype == x.dtype else self.w.to(x.dtype)
        return F.linear(x, w)

    def nbytes(self):
        return self.w.numel() * self.w.element_size()

    def to(self, device):
        return QuantLin(self.w.to(device), self.act_quant, self.block)
