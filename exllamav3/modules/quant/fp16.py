from __future__ import annotations
import torch
from torch import nn
from ...ext import exllamav3_ext as ext
import os as _os
_f32_via_f16 = _os.environ.get("EXL3_HIP_F32OUT_VIA_F16", "1") != "0" and bool(torch.version.hip)
_f32_via_f16_min_rows = int(_os.environ.get("EXL3_HIP_F32OUT_VIA_F16_MIN_ROWS", "32"))
# gfx1151 decode: fp16-out torch.matmul at M <= 8 lands on a single-workgroup hipBLAS tile (GLM-5.3
# indexer wk/weights_proj: 108-154 us for 1 MB); ext.hgemm takes the multi-CU skinny
# split-K GEMV first (M <= 8, N <= 128) and falls back to the same GEMM otherwise
_hip = bool(torch.version.hip)
_f16_small_m_hgemm = _os.environ.get("EXL3_HIP_F16_SMALL_M_HGEMM", "1") != "0" and bool(torch.version.hip)
from ...util.tensor import to2
from ...util import first_not_none

class LinearFP16:

    quant_type: str = "fp16"

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        full_in_features: int | None = None,
        full_out_features: int | None = None,
        first_in_feature: int | None = None,
        first_out_feature: int | None = None,
        out_dtype: torch.dtype | None = None,
        key: str | None = None
    ):
        self.weight = weight
        self._pinned_store = None
        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.full_in_features = full_in_features
        self.full_out_features = full_out_features
        self.first_in_feature = first_in_feature
        self.first_out_feature = first_out_feature
        self.out_dtype = out_dtype
        self.key = key

        if self.weight.shape[0] == full_in_features and self.weight.shape[0] < in_features:
            self.weight = self.weight[first_in_feature : first_in_feature + in_features, :]
        if self.weight.shape[1] == full_out_features and self.weight.shape[1] < out_features:
            self.weight = self.weight[:, first_out_feature : first_out_feature + out_features]
            if bias is not None:
                self.bias = self.bias[..., first_out_feature : first_out_feature + out_features]

        if in_features < full_in_features or out_features < full_out_features:
            w = torch.empty(self.weight.shape, dtype = self.weight.dtype, device = self.weight.device)
            w.copy_(self.weight)
            self.weight = w

        self.bc = ext.BC_LinearFP16(
            self.weight,
            self.bias,
        )

    def unload(self):
        pass

    def get_tensors(self, key: str):
        t = {}
        t[f"{key}.weight"] = self.weight.T.contiguous()
        if self.bias is not None:
            t[f"{key}.bias"] = self.bias
        return t

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        dtype = out_dtype or self.out_dtype or torch.half
        y = torch.empty(
            (x.shape[0], self.out_features),
            dtype = dtype,
            device = x.device
        )
        weight = self.weight
        pinned = self._pinned_store
        if pinned is not None:
            # Pinned-host storage (vision towers): stage the weight into a shared VRAM scratch
            # first. cuBLAS re-reads B once per m-tile, so computing straight from the zero-copy
            # alias multiplies the PCIe traffic by ~m/128 (measured 56x wall time on a large
            # tower at m ~4k); a single async H2D pass then a normal GEMM costs one transfer
            from ...util.tensor import g_tensor_cache
            weight = g_tensor_cache.get_bucketed(x.device, pinned.numel(), pinned.dtype, "fp16_pin_stage").view(pinned.shape)
            weight.copy_(pinned, non_blocking = True)
        if (_hip and x.dtype == torch.half and dtype == torch.half and x.shape[0] >= 256
                and _os.environ.get("EXL3_DENSE_GEMM_TUNE", "1") != "0"
                and weight.is_contiguous() and x.is_contiguous()):
            # prefill: ext.hgemm's tuned rocBLAS path (see hgemm.cu dtune); torch's hipBLASLt
            # default is 1.3-1.8x slower on these small-K / small-N shapes
            ext.hgemm(x, weight, y)
        elif dtype == x.dtype and not (_f16_small_m_hgemm and x.dtype == torch.half and x.shape[0] <= 8):
            torch.matmul(x, weight, out = y)
        elif (_f32_via_f16 and dtype == torch.float and x.dtype == torch.half
              and x.shape[0] >= _f32_via_f16_min_rows):
            # gfx1151: hipblaslt's fp16-in/fp32-out kernels run at ~6 TFLOP/s vs ~34 for
            # fp16-out. Compute narrow, widen after. See exl3.py for the measurement.
            yh = torch.empty(y.shape, dtype = torch.half, device = y.device)
            torch.matmul(x, weight, out = yh)
            y.copy_(yh)
        else:
            ext.hgemm(x, weight, y)
        if self.bias is not None:
            y += self.bias
        y = y.view(out_shape)
        return y

    def get_weight_tensor(self) -> torch.Tensor:
        return self.weight

    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias

    def set_weight(self, w: torch.Tensor):
        self.weight = w.half()

    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.weight.device
        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def unswap_cpu(self):
        if self.swap_device is None:
            return
        self.weight = self.weight.to(self.swap_device)
        if self.bias is not None:
            self.bias = self.bias.to(self.swap_device)
        self.swap_device = None

    def tp_export(self, plan, producer):
        return {
            "cls": LinearFP16,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight": producer.send(self.weight),
            "bias": producer.send(self.bias),
            "out_dtype": self.out_dtype,
        }

    @staticmethod
    def tp_import_split(local_context, exported, plan, split, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_w = exported["weight"]
        id_b = exported["bias"]

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            w = consumer.recv(id_w, cuda = True, slice_dim = 1, first = first, last = last)
            b = consumer.recv(id_b, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            w = consumer.recv(id_w, cuda = True, slice_dim = 0, first = first, last = last)
            b = consumer.recv(id_b, cuda = True) if (first == 0) else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearFP16(
            in_features = in_features,
            out_features = out_features,
            weight = w,
            bias = b,
            full_in_features = in_features,
            full_out_features = out_features,
            first_in_feature = 0,
            first_out_feature = 0,
            out_dtype = exported["out_dtype"],
        )
        return module


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        return LinearFP16.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_w = exported["weight"]
        id_b = exported["bias"]

        w_ = []
        b_ = []
        in_features = 0
        out_features = 0

        for split in splits:
            assert split is not None
            split_out, first, last = split
            assert split_out
            w = consumer.recv(id_w, cuda = True, slice_dim = 1, first = first, last = last)
            b = consumer.recv(id_b, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features += last - first
            w_.append(w)
            b_.append(b)

        w = torch.cat(w_, dim = 1)
        b = torch.cat(b_, dim = 0) if b_[0] is not None else None

        module = LinearFP16(
            in_features = in_features,
            out_features = out_features,
            weight = w,
            bias = b,
            full_in_features = in_features,
            full_out_features = out_features,
            first_in_feature = 0,
            first_out_feature = 0,
            out_dtype = exported["out_dtype"],
        )
        return module


class LinearFP16_torch:

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        self.nn_linear = nn.Linear(
            in_features,
            out_features,
            bias is not None,
            device = "meta"
        )
        self.nn_linear.weight = nn.Parameter(weight, requires_grad = False)
        if bias is not None:
            self.nn_linear.bias = nn.Parameter(bias, requires_grad = False)

    def get_tensors(self, key: str):
        t = {}
        t[f"{key}.weight"] = self.nn_linear.weight.data.contiguous()
        if self.nn_linear.bias is not None:
            t[f"{key}.bias"] = self.nn_linear.bias.data.contiguous()
        return t

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        x = self.nn_linear.forward(x)
        return to2(x, out_dtype)

    def get_weight_tensor(self) -> torch.Tensor:
        return self.nn_linear.weight.data.T

    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.nn_linear.bias.data if self.nn_linear.bias is not None else None

    def set_weight(self, w):
        self.nn_linear.weight = nn.Parameter(w.T.half())
