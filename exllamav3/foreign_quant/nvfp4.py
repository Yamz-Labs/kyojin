"""NVFP4 (ModelOpt W4A16) dequant reference, numpy-only.

Per 16-element group, NVIDIA ModelOpt's NVFP4 stores (see the Bolt
``hf_quant_config.json``: ``W4A16_NVFP4``, ``group_size 16``):

* 4-bit E2M1 payloads (OCP Microscaling E2M1, finite: ``0x7 -> +6``,
  ``0xF -> -6``, no NaN/Inf), two per byte, **low nibble = even index**
  (ModelOpt packing convention);
* one fp8 E4M3 per-group scale (OCP: ``0x38 -> 1.0``, ``0x7E -> 448``,
  NaN only at ``0x7F``/``0xFF``);
* one fp32 global scale per tensor (``weight_scale_2``).

Decoded as ``w = e2m1(q) * scale_fp8 * scale_global``.

In the Bolt safetensors, e.g. ``layers.0.mixer.in_proj``::

    weight         U8     [18560, 2048]   # [out, in/2] packed E2M1
    weight_scale   F8_E4M3 [18560, 256]  # [out, in/16] per-group scales
    weight_scale_2 F32     []             # global scale

E2M1/E4M3 codebooks follow the OCP Microscaling Formats specification and are
cross-checked against ``ml_dtypes`` (``float4_e2m1fn`` / ``float8_e4m3fn``) in
tests/test_foreign_quant_cpu.py.
"""
from __future__ import annotations

import numpy as np

NVFP4_GROUP_SIZE = 16

# OCP MX E2M1 (finite variant): index = nibble value.
E2M1_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

_E2M1 = np.array(E2M1_LUT, dtype=np.float32)


def _as_bytes(data: object) -> np.ndarray:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    return np.asarray(data, dtype=np.uint8)


def e4m3_to_float32(data: object) -> np.ndarray:
    """Decode fp8 E4M3 bytes to float32 (OCP: bias behaves as 7 for e<=14,
    e==15 is normal with exponent 8; NaN only at 0x7F/0xFF)."""
    b = np.ascontiguousarray(_as_bytes(data))
    e = ((b >> 3) & 0xF).astype(np.int32)
    m = (b & 0x7).astype(np.float32)
    is_nan = (e == 15) & (m == 7)
    is_zero = (e == 0) & (m == 0)
    is_sub = (e == 0) & (m != 0)
    # Normal (incl. e==15, m!=7): 2**(e-7) * (1 + m/8); subnormal: m * 2**-9.
    vals = np.ldexp(1.0 + m / np.float32(8.0), e - 7).astype(np.float32)
    vals = np.where(is_sub, m * np.float32(2.0 ** -9), vals)
    vals = np.where(is_zero, np.float32(0.0), vals)
    vals = np.where(is_nan, np.float32(np.nan), vals)
    sign = np.where((b >> 7) != 0, np.float32(-1.0), np.float32(1.0))
    return (vals * sign).astype(np.float32)


def unpack_e2m1(packed: object) -> np.ndarray:
    """Unpack E2M1 nibbles to float32, low nibble = even index.

    :param packed: uint8 array (..., K/2).
    :return: float32 array (..., K).
    """
    p = np.ascontiguousarray(_as_bytes(packed))
    lo = _E2M1[p & 0xF]
    hi = _E2M1[(p >> 4) & 0xF]
    out = np.empty(p.shape + (2,), dtype=np.float32)
    out[..., 0] = lo
    out[..., 1] = hi
    return out.reshape(p.shape[:-1] + (p.shape[-1] * 2,))


def dequant_nvfp4(packed: object, scales_fp8: object, global_scale: float) -> np.ndarray:
    """Dequantize a ModelOpt NVFP4 tensor to float32.

    :param packed: uint8 array (..., K/2), two E2M1 values per byte.
    :param scales_fp8: uint8 array (..., K/16), one E4M3 scale per 16 values.
    :param global_scale: fp32 ``weight_scale_2`` scalar.
    :return: float32 array (..., K).
    """
    p = np.ascontiguousarray(_as_bytes(packed))
    s = np.ascontiguousarray(_as_bytes(scales_fp8))
    if p.shape[-1] * 2 != s.shape[-1] * NVFP4_GROUP_SIZE or p.shape[:-1] != s.shape[:-1]:
        raise ValueError(f"NVFP4 shape mismatch: packed {p.shape} vs scales {s.shape}")
    vals = unpack_e2m1(p)
    gscales = np.repeat(e4m3_to_float32(s), NVFP4_GROUP_SIZE, axis=-1)
    return vals * gscales * np.float32(global_scale)
