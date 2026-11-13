"""Linear layer over third-party quantized weights, run as shipped (no requant).

A ForeignLinear holds the raw checkpoint payload (ggml Q8_0 / IQ2_XXS blocks
or ModelOpt NVFP4 nibbles + scales) on the device and dequantizes on the fly:
rows <= 16 through the fused fq_gemm_* kernels, wider batches through
fq_dequant_* + torch.matmul. Output matches dequant + matmul within fp16
tolerance (the fused path only reorders the fp32 accumulation).

Plugs into the existing dispatch: quant_type is "foreign" (never "exl3"), so
BlockSparseMLP takes the dense per-expert torch path and calls
forward(xc, params) per expert group; MLP / Mamba2 / Attention accept
prebuilt modules the same way. inner is self with bias None so the
quantized-path probes (l.inner.bias, l.inner.trellis, ...) stay dormant.

BOLT_FQ_FUSED=0 forces the unfused dequant+matmul path (A/B).
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from .module import Module
from ..ext import exllamav3_ext as ext
from ..foreign_quant.gap_formats import dequant as gap_dequant
from ..foreign_quant import triton_iq
from ..foreign_quant import triton_gap
from ..foreign_quant import triton_rows
_ROWS_TRITON = os.environ.get("BOLT_FQ_ROWS_TRITON", "1") != "0"
# Q8_0 rows 2..16 on CUDA: the Triton _rows_q8_0 kernel (tuned on Strix Halo) takes 172 us per launch on the 3090 (330 launches per step = 57 ms of an
# 87 ms N8 step on the 35B-A3B UD-Q4_K_XL, 75 % of device time); the C++ q8_1 GEMV (fq_gemv_q8_0) takes 70-80 us. HIP keeps the Triton kernel.
# BOLT_FQ_ROWS_TRITON_Q8=1 restores it on CUDA (A/B).
_ROWS_TRITON_Q8 = os.environ.get("BOLT_FQ_ROWS_TRITON_Q8", "0" if (torch.version.cuda and not torch.version.hip) else "1") != "0"

# Q4_K / Q6_K rows 2..16: the C++ multi-row q8_1 GEMV (fq_gemv_kq.cu fq_gemv_kq_mrows_kernel: weights read once for all rows, per-row result bit-identical to
# the one-row kernel) instead of the Triton rows kernel (fp16 activations, Triton speed). Default on CUDA; HIP keeps Triton until measured on gfx1151.
# BOLT_FQ_KQ_MROWS=0 restores the Triton rows kernel (A/B); default on for CUDA and HIP. Read here and in the extension (same variable).
_KQ_MROWS = os.environ.get("BOLT_FQ_KQ_MROWS", "1") != "0"
_KQ_MROWS_FORMATS = ("q4_k", "q6_k")

FUSED_ROWS_MAX = 16
_FUSED = os.environ.get("BOLT_FQ_FUSED", "1") != "0"
# Cooperative GEMV (fq_gemv.cu: shared-memory x staging, lanes split K):
# replaces fq_gemm_* wherever present. Any half/float in+out (the old path
# needed half in and always wrote half, so float-out layers like latent_out
# and the mamba projections fell back to full dequant+matmul).
# BOLT_FQ_GEMV=0 restores the fq_gemm_* path for A/B.
_GEMV = os.environ.get("BOLT_FQ_GEMV", "1") != "0"
# Row chunk of the unfused dequant+matmul path (transient fp16 bound).
_UNFUSED_ROWS = int(os.environ.get("BOLT_FQ_UNFUSED_ROWS", "8192"))
# NVFP4 E2M1 pair-table dot (fq_gemv.cuh: one shared float2 load per pair
# instead of two replicated-e2m1 loads + nibble math). BOLT_FQ_PAIR=0
# restores the direct dot for A/B.
_PAIR = os.environ.get("BOLT_FQ_PAIR", "1") != "0"

_FORMATS = ("q8_0", "iq2_xxs", "nvfp4", "q4_k", "q5_k", "q6_k", "iq4_xs",
            "q4_0", "q5_0", "q5_1", "iq4_nl", "iq3_xxs", "iq3_s", "mxfp4", "nvfp4_gguf", "iq2_xs", "iq2_s", "q3_k", "q2_k")
# GGUF blocks without a fused kernel yet: torch dequant (foreign_quant/gap_formats.py) + matmul at every width.
# BOLT_FQ_GAP_FUSED=0 turns the fused Triton decode kernel off (A/B).
_GAP_FUSED = os.environ.get("BOLT_FQ_GAP_FUSED", "1") != "0"
_GAP_FORMATS = ("q4_0", "q5_0", "q5_1", "iq4_nl", "iq3_xxs", "iq3_s", "mxfp4", "nvfp4_gguf", "iq2_xs", "iq2_s", "q3_k", "q2_k")
# GGUF K-quant formats: decode (rows < 16) via the q8_1-
# activation row kernels; prefill (rows >= 16) via the unfused fp16 dequant
# + cuBLAS GEMM path, one GEMM per call -- i.e. one GEMM per
# expert token-group in the dense MoE prefill path. MCPM_FQ_KQ_GEMM=0 forces
# the q8 row kernels for every width (A/B).
_KQ_FORMATS = ("q4_k", "q5_k", "q6_k", "iq4_xs")
# EXL3_IQ3_Q8 (default 1): IQ3_XXS decode rows run the q8_1 dp4a GEMV (fq_gemv_iq3_xxs_q8) instead of the Triton decode
_IQ3_Q8 = os.environ.get("EXL3_IQ3_Q8", "1") != "0"
_IQ_Q8_FN = {"iq3_xxs": "fq_gemv_iq3_xxs_q8", "iq2_s": "fq_gemv_iq2_s_q8", "iq3_s": "fq_gemv_iq3_s_q8"}
# BOLT_FQ_MMVQ (default 1, CUDA and HIP): IQ formats decode (1..16 rows) through the llama.cpp mmvq-style kernel (fq_mmvq.cu: ggml vec_dot_iq*_q8_1 dp4a dots,
# warp-per-row blocks, 1..8 activation columns per launch, weights read once). 0 = the previous paths (A/B).
_MMVQ_ON = os.environ.get("BOLT_FQ_MMVQ", "1") != "0"
_MMVQ_FN = {"iq2_xxs": "fq_mmvq_iq2_xxs", "iq2_xs": "fq_mmvq_iq2_xs", "iq2_s": "fq_mmvq_iq2_s", "iq3_xxs": "fq_mmvq_iq3_xxs",
            "iq3_s": "fq_mmvq_iq3_s", "iq4_nl": "fq_mmvq_iq4_nl", "iq4_xs": "fq_mmvq_iq4_xs", "q5_k": "fq_mmvq_q5_k"}
_MMVQ_QK = {"iq4_nl": 32}
_DEQ_IQ_ID = {"iq2_xs": 1, "iq2_s": 2, "iq3_xxs": 3, "iq3_s": 4, "iq4_nl": 5}   # fq_dequant_iq format ids (the others have native dequants already)
# rows served by the mmvq kernel (8 columns per launch, so R/8 passes over the weights): the old IQ path above 16 rows dequantized the whole weight in torch
# on every call (1.7 s for the 8B at ANY prompt length). Past ~64 rows the dequant + GEMM path is the better one again.
_MMVQ_MAXR = int(os.environ.get("BOLT_FQ_MMVQ_MAXR", "64"))
_KQ_BLOCK_BYTES = {"q4_k": 144, "q5_k": 176, "q6_k": 210, "iq4_xs": 136}
_KQ_GEMM_ROWS = 16
_KQ_GEMM = os.environ.get("MCPM_FQ_KQ_GEMM", "1") != "0"

# Reusable fp16 dequant scratch, one buffer per device: the KQ prefill GEMM
# path dequantizes every weight N-chunk on every forward, so a persistent
# buffer avoids an allocator round-trip per chunk. Views handed out alias
# the buffer: consume (matmul) before the next _dequant_rows call (the only
# callers do exactly that).
_DQ_SCRATCH: dict = {}


def _dq_fp16(dev: torch.device, rows: int, cols: int) -> torch.Tensor:
    key = str(dev)
    t = _DQ_SCRATCH.get(key)
    if t is None or t.numel() < rows * cols:
        t = torch.empty((rows * cols,), dtype=torch.float16, device=dev)
        _DQ_SCRATCH[key] = t
    return t[:rows * cols].view(rows, cols)


class ForeignLinear(Module):

    def __init__(
        self,
        config,
        key: str,
        in_features: int,
        out_features: int,
        format: str,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        post_scale: float = 1.0,
        softcap: float = 0.0,
    ):
        super().__init__(config, key, qmap)
        assert format in _FORMATS, f"unknown foreign format {format}"
        self.format = format
        self.in_features = in_features
        self.in_features_unpadded = in_features
        self.out_features = out_features
        self.out_features_unpadded = out_features
        self.full_in_features = in_features
        self.full_out_features = out_features
        self.first_in_feature = 0
        self.first_out_feature = 0
        self.trim_padded_out = False
        self.allow_input_padding = False
        self.transposed_load = False
        self.out_dtype = out_dtype
        self.pre_scale = 1.0
        self.post_scale = post_scale
        self.softcap = softcap
        self.quant_type = "foreign"
        # The BlockSparseMLP quantized-path probes (inner.bias / inner.trellis /
        # inner.bc / ...) only run when is_quantized, which needs quant_type ==
        # "exl3" on every expert; still, expose the attributes they would read.
        self.inner = self
        self.bias = None
        self.bc = None
        self.packed = None
        self.scales = None
        self.global_scale = None

    def can_defer_load(self):
        # Eager load: raw U8 payloads go straight to the device, no EXL3
        # deferred arena.
        return False

    def optimizer_targets(self):
        return [self.key]

    def weights_numel(self):
        return self.in_features * self.out_features

    def get_tensors(self):
        if self.packed is None:
            return {}
        d = {self.key + ".weight": self.packed}
        if self.scales is not None:
            d[self.key + ".weight_scale"] = self.scales
        return d

    def load(self, device: torch.device, **kwargs):
        self.device = device
        stc = self.config.stc
        if self.format in ("q8_0", "iq2_xxs") or self.format in _KQ_FORMATS or self.format in _GAP_FORMATS:
            # Single raw payload (KQ GGUF bytes / Q8_0 / IQ2_XXS blocks).
            self.packed = stc.get_tensor(self.key + ".weight", device, no_defer=True)
            assert self.packed.dtype == torch.uint8, f"{self.key}: expected U8 payload, got {self.packed.dtype}"
        else:
            self.packed = stc.get_tensor(self.key + ".weight", device, no_defer=True)
            scales = stc.get_tensor(self.key + ".weight_scale", device, no_defer=True)
            # E4M3 scales arrive as fp8 (stc preserves stored dtypes); the
            # kernels want the raw bytes.
            self.scales = scales.view(torch.uint8) if scales.dtype != torch.uint8 else scales
            g = stc.get_tensor(self.key + ".weight_scale_2", device, no_defer=True)
            self.global_scale = float(g.float().item())
            assert self.packed.dtype == torch.uint8 and self.scales.dtype == torch.uint8

    def unload(self):
        self.packed = None
        self.scales = None
        self.global_scale = None
        self.device = None

    def _dequant_weight(self) -> torch.Tensor:
        """Full fp16 weight [N, K] (unfused path transient; freed by caller)."""
        return self._dequant_rows(0, self.out_features)

    def _dequant_rows(self, first: int, count: int) -> torch.Tensor:
        """fp16 weight slice [count, K] for output rows [first, first+count)."""
        N = self.out_features
        assert 0 <= first and first + count <= N
        dev = self.packed.device
        if _MMVQ_ON and self.format in _DEQ_IQ_ID and self.in_features % 256 == 0 and hasattr(ext, "fq_dequant_iq"):
            # native fp16 dequant of the IQ blocks (ggml convert.cu port); the torch gap_dequant below takes ~1.7 s for the whole 8B
            out = torch.empty((count, self.in_features), dtype=torch.float16, device=dev)
            ext.fq_dequant_iq(self.packed[first:first + count].reshape(count, -1).contiguous(), out, _DEQ_IQ_ID[self.format])
            return out
        if self.format in _GAP_FORMATS:
            return gap_dequant(self.format, self.packed[first:first + count])
        if self.format in _KQ_FORMATS:
            Kb = self.in_features // 256
            bb = _KQ_BLOCK_BYTES[self.format]
            assert self.packed.shape[1] == Kb * bb, \
                f"{self.key}: packed width {self.packed.shape[1]} != K={self.in_features} ({bb}B)"
            out = _dq_fp16(dev, count, self.in_features)
            getattr(ext, f"fq_dequant_{self.format}")(
                self.packed[first:first + count].contiguous(), out)
            return out
        if self.format == "q8_0" and not hasattr(ext, "fq_dequant_q8_0"):
            return gap_dequant("q8_0", self.packed[first:first + count])
        if self.format == "q8_0":
            Kb = self.in_features // 32
            out = torch.empty((count, self.in_features), dtype=torch.float16, device=dev)
            ext.fq_dequant_q8_0(self.packed.reshape(N, Kb, 34)[first:first + count].contiguous(), out)
        elif self.format == "iq2_xxs":
            Kb = self.in_features // 256
            out = torch.empty((count, self.in_features), dtype=torch.float16, device=dev)
            ext.fq_dequant_iq2_xxs(self.packed.reshape(N, Kb, 66)[first:first + count].contiguous(), out)
        else:
            K = self.in_features
            out = torch.empty((count, K), dtype=torch.float16, device=dev)
            ext.fq_dequant_nvfp4(self.packed.reshape(N, K // 2)[first:first + count].contiguous(),
                                 self.scales.reshape(N, K // 16)[first:first + count].contiguous(),
                                 self.global_scale, out)
        return out

    def q8_fused_gemv_ok(self, flat: torch.Tensor) -> bool:
        """True when forward() serves this flat input with the plain ext.fq_gemv_q8_0 launch (the _FUSED branch below)."""
        R, K = flat.shape[0], self.in_features
        if self.format != "q8_0" or flat.device != self.packed.device or flat.shape[-1] != K:
            return False
        if _ROWS_TRITON and flat.dtype == torch.half and _ROWS_TRITON_Q8 and triton_rows.supported(self.format, R, K):
            return False
        if self.format in triton_iq.FUSED or self.format in _GAP_FORMATS or self.format in _KQ_FORMATS or self.format in _IQ_Q8_FN:
            return False
        return (_FUSED and 1 <= R <= FUSED_ROWS_MAX and flat.is_contiguous() and flat.dtype in (torch.half, torch.float)
                and hasattr(ext, "fq_gemv_q8_0") and _GEMV)

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        shape = x.shape
        K = self.in_features
        assert shape[-1] == K, f"{self.key}: width {shape[-1]} != {K}"
        flat = x.reshape(-1, K)
        R = flat.shape[0]
        dtype = out_dtype or self.out_dtype or torch.half
        dev = self.packed.device
        if flat.device != dev:
            flat = flat.to(dev)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        out: Optional[torch.Tensor] = None
        if (_MMVQ_ON and self.format in _MMVQ_FN and flat.dtype == torch.half and R <= (min(_MMVQ_MAXR, FUSED_ROWS_MAX - 1) if self.format == "q5_k" else _MMVQ_MAXR) and K % _MMVQ_QK.get(self.format, 256) == 0
                and hasattr(ext, _MMVQ_FN[self.format])):
            N = self.out_features
            out = torch.empty((R, N), dtype=dtype if dtype in (torch.half, torch.float) else torch.half, device=dev)
            dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
            qb = ext.fq_q8_scratch(dev_idx, R, K)
            ext.fq_quant_q8_1_rows(flat, qb)
            getattr(ext, _MMVQ_FN[self.format])(qb, self.packed.reshape(N, -1), out)
            if out.dtype != dtype:
                out = out.to(dtype)
        elif (_ROWS_TRITON and flat.dtype == torch.half and (_ROWS_TRITON_Q8 or self.format != "q8_0")
                and not (_KQ_MROWS and self.format in _KQ_MROWS_FORMATS)
                and triton_rows.supported(self.format, R, K)):
            # 2..16 rows (verify / multi-user): weights read once for all rows (the fq_gemv kernels re-read them per row)
            out = triton_rows.gemv(self.format, flat, self.packed.reshape(self.out_features, -1), self.scales.reshape(self.out_features, -1)
                                   if self.scales is not None else None, self.global_scale if self.global_scale is not None else 1.0)
            if out.dtype != dtype:
                out = out.to(dtype)
        elif (
            self.format in _IQ_Q8_FN and _IQ3_Q8 and R == 1 and flat.dtype == torch.half and K % 256 == 0
            and hasattr(ext, _IQ_Q8_FN[self.format])
        ):
            # IQ3_XXS / IQ2_S / IQ3_S single decode row (2+ rows stay on the Triton decode, which reads the weights once for all rows): q8_1 activations + integer dp4a dots (ggml vec_dot_iq3_xxs_q8_1 port, fq_gemv_kq.cu FMT 4)
            N = self.out_features
            out = torch.empty((R, N), dtype=dtype if dtype in (torch.half, torch.float) else torch.half, device=dev)
            dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
            qb = ext.fq_q8_scratch(dev_idx, R, K)
            ext.fq_quant_q8_1_rows(flat, qb)
            getattr(ext, _IQ_Q8_FN[self.format])(qb, self.packed.reshape(N, -1), out)
            if out.dtype != dtype:
                out = out.to(dtype)
        elif (
            self.format in triton_iq.FUSED and flat.dtype == torch.half and triton_iq.supported(self.format, R, K, self.out_features)
        ):
            # IQ3_XXS / IQ3_S / IQ2_XS / IQ2_S / Q3_K decode rows: fused Triton decode of the stored blocks (EXL3_IQ_FUSED=0: torch dequant)
            out = triton_iq.gemv(self.format, flat, self.packed.reshape(self.out_features, -1))
            if out.dtype != dtype:
                out = out.to(dtype)
        elif (
            _GAP_FUSED and self.format in _GAP_FORMATS and flat.dtype == torch.half
            and triton_gap.supported(self.format, R, K, self.out_features)
        ):
            out = triton_gap.gemv(self.format, flat, self.packed)
            if out.dtype != dtype:
                out = out.to(dtype)
        elif self.format in _KQ_FORMATS:
            # KQ path: quantize once to the cached q8 scratch, one
            # row-per-block launch. Decode widths (R < 16) always run here;
            # prefill widths (R >= 16) use the unfused fp16 GEMM below (one
            # GEMM per call = one GEMM per expert token-group in MoE).
            # Rows run in <=2048 chunks to bound the (R, N) output + q8 scratch transient
            # (full-R prefill otherwise OOMs past ~12K).
            assert flat.is_contiguous() and flat.dtype in (torch.half, torch.float)
            N = self.out_features
            bb = _KQ_BLOCK_BYTES[self.format]
            assert self.packed.shape[1] == (K // 256) * bb, \
                f"{self.key}: packed width {self.packed.shape[1]} != K={K} ({bb}B)"
            if _KQ_GEMM and R >= _KQ_GEMM_ROWS + (1 if (_KQ_MROWS and self.format in _KQ_MROWS_FORMATS) else 0):   # multi-row kernel covers R == 16 too
                xuse = flat.half() if flat.dtype != torch.half else flat
                out = torch.empty((R, N), dtype=dtype, device=dev)
                _RCH = 2048
                for _s in range(0, R, _RCH):
                    _e = min(_s + _RCH, R)
                    _xs = xuse[_s:_e]
                    if N <= _UNFUSED_ROWS:
                        w = self._dequant_rows(0, N)
                        _m = torch.matmul(_xs, w.t())
                        out[_s:_e] = _m if _m.dtype == dtype else _m.to(dtype)
                    else:
                        for a in range(0, N, _UNFUSED_ROWS):
                            c = min(_UNFUSED_ROWS, N - a)
                            w = self._dequant_rows(a, c)
                            _m = torch.matmul(_xs, w.t())
                            _o = _m if _m.dtype == dtype else _m.to(dtype)
                            out[_s:_e, a:a + c] = _o
            else:
                out = torch.empty((R, N), dtype=dtype, device=dev)
                dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
                ext_fn = getattr(ext, f"fq_gemv_{self.format}_q8")
                _CH = 2048
                for _s in range(0, R, _CH):
                    _e = min(_s + _CH, R)
                    _fl = flat[_s:_e]
                    qb = ext.fq_q8_scratch(dev_idx, _e - _s, K)
                    ext.fq_quant_q8_1_rows(_fl.half() if _fl.dtype != torch.half else _fl, qb)
                    ext_fn(qb, self.packed, out[_s:_e])
        elif (
            _FUSED and R >= 1 and R <= FUSED_ROWS_MAX and flat.is_contiguous()
            and self.format not in _GAP_FORMATS
            and flat.dtype in (torch.half, torch.float)
            and hasattr(ext, "fq_gemv_q8_0")
            and _GEMV
        ):
            N = self.out_features
            out = torch.empty((R, N), dtype=dtype, device=dev)
            if self.format == "q8_0":
                ext.fq_gemv_q8_0(flat, self.packed.reshape(N, -1, 34), out)
            elif self.format == "iq2_xxs":
                ext.fq_gemv_iq2_xxs(flat, self.packed.reshape(N, -1, 66), out)
            else:
                ext.fq_gemv_nvfp4(flat, self.packed.reshape(N, K // 2),
                                  self.scales.reshape(N, K // 16), self.global_scale, out, _PAIR)
        elif (
            _FUSED and R >= 1 and R <= FUSED_ROWS_MAX and flat.dtype == torch.half and flat.is_contiguous()
            and self.format not in _GAP_FORMATS
            and hasattr(ext, "fq_gemm_q8_0")
        ):
            N = self.out_features
            out = torch.empty((R, N), dtype=torch.float16, device=dev)
            if self.format == "q8_0":
                ext.fq_gemm_q8_0(flat, self.packed.reshape(N, -1, 34), out)
            elif self.format == "iq2_xxs":
                ext.fq_gemm_iq2_xxs(flat, self.packed.reshape(N, -1, 66), out)
            else:
                ext.fq_gemm_nvfp4(flat, self.packed.reshape(N, K // 2),
                                  self.scales.reshape(N, K // 16), self.global_scale, out)
            if out.dtype != dtype:
                out = out.to(dtype)
        else:
            # Unfused: dequant in row chunks (bounds the transient fp16) +
            # torch matmul. Single-chunk fast path: direct mm output (same
            # values bitwise, skips the empty + slice-assign copy traffic).
            N = self.out_features
            xuse = flat.half() if flat.dtype != torch.half else flat
            if N <= _UNFUSED_ROWS:
                w = self._dequant_rows(0, N)
                mmout = torch.matmul(xuse, w.t())
                out = mmout if mmout.dtype == dtype else mmout.to(dtype)
                del w
            else:
                out = torch.empty((R, N), dtype=dtype, device=dev)
                for a in range(0, N, _UNFUSED_ROWS):
                    w = self._dequant_rows(a, min(_UNFUSED_ROWS, N - a))
                    mmout = torch.matmul(xuse, w.t())
                    out[:, a:a + w.shape[0]] = mmout if mmout.dtype == dtype else mmout.to(dtype)
                    del w
        out = out.view(*shape[:-1], self.out_features)
        if self.post_scale != 1.0:
            out = out * self.post_scale
        if self.softcap != 0.0:
            ext.softcap(out, out, self.softcap)
        return out
