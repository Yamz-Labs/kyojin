"""
DeepSeek-V4.1 number formats: the fp8 / fp4 activation fake-quant steps that are part of the model
(not just storage), and decoders for the official checkpoint formats.

Everything here is plain torch, device agnostic (CPU and HIP), no extension needed.

Spec source: DeepSeek-V4.1-Flash inference/kernel.py (act_quant_kernel, fp4_quant_kernel,
fp8_gemm_kernel, fp4_gemm_kernel). Rounding conventions:
  * scale rounding (ue8m0): s = 2 ** ceil(log2(amax * f32(1/qmax))), exponent taken from the fp32 bits
    (exact, no log2 intrinsic), as `fast_round_scale` does
  * fp8 = e4m3fn, round-to-nearest-even, saturating at +-448 (clamped before the cast)
  * fp4 = e2m1, round-to-nearest-even over {0,.5,1,1.5,2,3,4,6}, clamped to +-6
"""
from __future__ import annotations
import os
import torch

F32_INV_448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)
F32_INV_6 = torch.tensor(1.0 / 6.0, dtype=torch.float32)
_DEV_CONST = {}


def _on(c: torch.Tensor, device) -> torch.Tensor:
    """Scalar constant on `device`, copied once (a per-call CPU->GPU copy is a host sync)."""
    k = (id(c), device)
    t = _DEV_CONST.get(k)
    if t is None:
        t = _DEV_CONST[k] = c.to(device)
    return t

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _round_scale_pow2(t: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(t)) for positive normal fp32 t, from the bit pattern."""
    bits = t.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    e = exp - 127 + (man != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


def fp4_round(v: torch.Tensor) -> torch.Tensor:
    """Round an fp32 tensor (already clamped to +-6) to the nearest e2m1 value, ties to even code."""
    a = v.abs()
    o = torch.zeros_like(a)
    o = torch.where(a > 0.25, 0.5, o)
    o = torch.where(a >= 0.75, 1.0, o)
    o = torch.where(a > 1.25, 1.5, o)
    o = torch.where(a >= 1.75, 2.0, o)
    o = torch.where(a > 2.5, 3.0, o)
    o = torch.where(a >= 3.5, 4.0, o)
    o = torch.where(a > 5.0, 6.0, o)
    return torch.copysign(o, v)


def _hip_fq(x: torch.Tensor, mode: int, block: int) -> bool:
    """In-place fake-quant on the HIP kernel (bit-equal to the torch path below); False when not applicable."""
    if x.dtype != torch.half or not x.is_cuda or x.shape[-1] % block or os.environ.get("DSV41_KERNELS", "1") == "0":
        return False
    if not x.is_contiguous():
        return False
    from ..modules.dsv41_fn import dsv41_hip as K
    K.fq_rows(x, mode)
    return True


def act_quant_fp8_(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """In-place fused fp8(e4m3, ue8m0 scale) quantise + dequantise along the last dim (kernel
    act_quant(..., round_scale, inplace=True)). Returns x."""
    N = x.shape[-1]
    assert N % block == 0
    if block == 32 and _hip_fq(x, 0, 32):
        return x
    xf = x.float().unflatten(-1, (N // block, block))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _round_scale_pow2(amax * _on(F32_INV_448, x.device))
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    x.copy_(q.flatten(-2).to(x.dtype))
    return x


def act_quant_fp8(x: torch.Tensor, block: int = 32):
    """Out-of-place variant: returns (dequantised activation in x.dtype)."""
    return act_quant_fp8_(x.clone(), block)


def fp4_act_quant_(x: torch.Tensor, block: int = 32, e4m3_scale: bool = False) -> torch.Tensor:
    """In-place fp4 (e2m1) fake-quant along the last dim (kernel fp4_act_quant(..., inplace=True)).
    e4m3_scale=False: ue8m0 power-of-2 scale per `block` (indexer q/k, block 32).
    e4m3_scale=True : e4m3 scale (compressed KV, block 16)."""
    N = x.shape[-1]
    assert N % block == 0
    if (block, e4m3_scale) in ((32, False), (16, True)) and _hip_fq(x, 2 if e4m3_scale else 1, block):
        return x
    xf = x.float().unflatten(-1, (N // block, block))
    amax = xf.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0 ** -126)
        s = _round_scale_pow2(amax * _on(F32_INV_6, x.device))
    q = fp4_round((xf / s).clamp(-6.0, 6.0)) * s
    x.copy_(q.flatten(-2).to(x.dtype))
    return x


# ------------------------------------------------------------------------------------------------
# checkpoint storage formats

_FP4_LUT = None


def fp4_lut(device) -> torch.Tensor:
    global _FP4_LUT
    if _FP4_LUT is None or _FP4_LUT.device != torch.device(device):
        _FP4_LUT = torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)
    return _FP4_LUT


def e8m0_to_float(s: torch.Tensor) -> torch.Tensor:
    """ue8m0 scale tensor (float8_e8m0fnu or its uint8 bits) -> fp32."""
    if s.dtype == torch.uint8:
        s = s.view(torch.float8_e8m0fnu)
    return s.float()


def dequant_fp8_block(w: torch.Tensor, scale: torch.Tensor, block: int = 0) -> torch.Tensor:
    """fp8 e4m3 weight [out,in] with one ue8m0 scale per (block x block) tile [ceil(out/b), ceil(in/b)]."""
    if w.dtype in (torch.uint8, torch.int8):
        w = w.view(torch.float8_e4m3fn)
    out_f, in_f = w.shape
    s = e8m0_to_float(scale)
    for b in ((block, 32, 128, 64, 16) if block else (32, 128, 64, 16)):    # tile size = what the scale grid implies
        if -(-out_f // b) == s.size(0) and -(-in_f // b) == s.size(1):
            bo = bi = b
            break
    else:
        raise ValueError(f"scale grid {tuple(s.shape)} does not tile weight {tuple(w.shape)}")
    s = s.repeat_interleave(bo, 0)[:out_f].repeat_interleave(bi, 1)[:, :in_f]
    return w.float() * s


def dequant_fp4_rows(w: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Packed fp4 weight [out, in/2] (2 values per byte, low nibble first, packed along in) with a
    ue8m0 scale per row per `block` input elements [out, in/block] -> fp32 [out,in]."""
    b = w.view(torch.uint8)
    lut = fp4_lut(b.device)
    lo = lut[(b & 0x0F).long()]
    hi = lut[(b >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).flatten(-2)
    s = e8m0_to_float(scale).repeat_interleave(block, -1)
    return v * s
