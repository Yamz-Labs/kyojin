"""Dequantization of the checkpoint's native formats to bf16.

Rules follow the reference code (inference/model.py, kernel.py, convert.py of the model repo):

* Routed experts (and MTP experts): `weight` is int8 holding two e2m1 values per byte, the LOW
  nibble first along the input dim. `scale` is ue8m0 (float8_e8m0fnu) with shape
  [out, in // 32], one scale per 32 consecutive inputs. value = e2m1(nibble) * 2^(e - 127).
* Everything else quantized: `weight` is e4m3fn, `scale` is ue8m0 with one scale per
  block x block tile (32 x 32 in this checkpoint; the block size is inferred from the shapes).
  value = weight * scale.

Products of an e2m1 / e4m3 value and a power of two are exact in fp32 and, for these value
ranges, in bf16, so the final bf16 cast does not round.
"""

import torch

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP4_BLOCK = 32


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """int8/uint8 [out, in // 2] -> float32 [out, in] (low nibble first)."""
    assert packed.dtype in (torch.int8, torch.uint8) and packed.ndim == 2
    b = packed.view(torch.uint8).long()
    lo = FP4_TABLE[b & 0x0F]
    hi = FP4_TABLE[b >> 4]
    return torch.stack([lo, hi], dim=-1).flatten(1)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 e2m1 + ue8m0 block-32 scales -> bf16 [out, in]."""
    out_dim, half_in = packed.shape
    in_dim = half_in * 2
    assert scale.shape == (out_dim, in_dim // FP4_BLOCK), (scale.shape, packed.shape)
    x = unpack_fp4(packed).view(out_dim, in_dim // FP4_BLOCK, FP4_BLOCK)
    x = x * scale.float().unsqueeze(-1)
    return x.reshape(out_dim, in_dim).to(torch.bfloat16)


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3 + block scales -> bf16 [out, in]. Block size inferred from the scale shape."""
    assert weight.ndim == 2 and scale.ndim == 2
    out_dim, in_dim = weight.shape
    assert out_dim % scale.shape[0] == 0 and in_dim % scale.shape[1] == 0, (weight.shape, scale.shape)
    bo, bi = out_dim // scale.shape[0], in_dim // scale.shape[1]
    x = weight.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi))
    x = x * scale.float()[:, None, :, None]
    return x.flatten(2, 3).flatten(0, 1).to(torch.bfloat16)


def dequant(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Dispatch on the stored dtype. Unquantized weights are returned as bf16."""
    if scale is None:
        return weight.to(torch.bfloat16)
    if weight.dtype in (torch.int8, torch.uint8):
        return dequant_fp4(weight, scale)
    if weight.dtype == torch.float8_e4m3fn:
        return dequant_fp8(weight, scale)
    raise ValueError(f"unsupported weight dtype {weight.dtype}")
