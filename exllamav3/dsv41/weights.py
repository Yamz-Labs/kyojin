"""
WeightSource: how the DS4.1 graph obtains tensors. Canonical names are the official checkpoint names
(`layers.N.attn.wq_a`, `layers.N.ffn.experts.E.w1`, `mtp.J....`, `embed`, `head`, `norm`); a source may store them any way it likes.

  tensor(name)  -> dense float tensor   (quantised storage is decoded)
  linear(name)  -> Lin
  expert(layer, e) -> ExpertHandle
"""
from __future__ import annotations
import torch
from . import quant
from .linear import Lin, DenseLin, QuantLin
from .experts import ExpertHandle


class WeightSource:
    n_layers: int = 0

    def has(self, name: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def tensor(self, name: str, dtype=torch.float32, device="cpu") -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def linear(self, name: str, dtype=torch.float32, device="cpu") -> Lin:  # pragma: no cover
        raise NotImplementedError

    def expert(self, layer: int, e: int, dtype=torch.float32, device="cpu") -> ExpertHandle:
        p = self.layer_prefix(layer)
        w = {k: self.linear(f"{p}.ffn.experts.{e}.{k}", dtype, device) for k in ("w1", "w3", "w2")}
        return ExpertHandle(layer, e, w["w1"], w["w3"], w["w2"])

    def layer_prefix(self, layer: int) -> str:
        return f"layers.{layer}" if layer < self.n_layers else f"mtp.{layer - self.n_layers}"

    def engram_store(self):  # pragma: no cover
        raise NotImplementedError


class MapWeightSource(WeightSource):
    """Official HF/inference layout behind a mapping {name: tensor} (a dict, or a lazy on-disk map):
    `X.weight` (+ `X.scale` for fp8 / fp4). A leading `model.` is accepted."""

    def __init__(self, sd, n_layers: int, engram_store=None):
        self.sd = sd
        self.n_layers = n_layers
        self._engram = engram_store

    def _key(self, name):
        for k in (name, "model." + name):
            if k in self.sd:
                return k
        return None

    def has(self, name):
        return self._key(name) is not None or self._key(name + ".weight") is not None

    def _raw(self, name):
        k = self._key(name) or self._key(name + ".weight")
        if k is None:
            raise KeyError(name)
        return self.sd[k]

    def _scale(self, name):
        k = self._key(name + ".scale")
        return None if k is None else self.sd[k]

    def tensor(self, name, dtype=torch.float32, device="cpu"):
        w = self._raw(name)
        sc = self._scale(name)
        if sc is not None:
            w = self._decode(w, sc)
            if name.endswith("wo_a"):
                # the official runtime holds wo_a as bf16 (convert.py dequantises fp8 -> bf16)
                w = w.to(torch.bfloat16)
        return w.to(dtype).to(device)

    @staticmethod
    def _decode(w, sc):
        if w.dtype == torch.float4_e2m1fn_x2 or w.dtype == torch.int8:
            return quant.dequant_fp4_rows(w, sc, 32)
        return quant.dequant_fp8_block(w, sc)

    def linear(self, name, dtype=torch.float32, device="cpu", act_quant=True):
        w = self._raw(name)
        sc = self._scale(name)
        if sc is None:
            return DenseLin(w.to(dtype).to(device))
        return QuantLin(self._decode(w, sc).to(dtype).to(device), act_quant=act_quant)

    def engram_store(self):
        return self._engram


class DictWeightSource(MapWeightSource):
    pass
