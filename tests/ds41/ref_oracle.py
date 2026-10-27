"""Oracle for the dequant tests: the reference code of the model repo (inference/convert.py),
ported unchanged, plus a slow per-element decoder."""

import torch

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor):
    """Verbatim from the reference convert.py."""
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 32
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    offset = scale / scale_max_offset_bits
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


def ref_fp8_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The wo_a dequantization of the reference convert.py (block sizes inferred as there)."""
    out_block_size = weight.size(0) // scale.size(0)
    in_block_size = weight.size(1) // scale.size(1)
    assert (out_block_size, in_block_size) in ((32, 32), (128, 128))
    weight = (
        weight.unflatten(0, (-1, out_block_size)).unflatten(-1, (-1, in_block_size)).float()
        * scale[:, None, :, None].float()
    )
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def ref_fp4_dequant(weight_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference expert path: e2m1 -> e4m3 + fp8 block scale (reference cast), then the fp8 dequant."""
    w8, s8 = cast_e2m1fn_to_e4m3fn(weight_i8, scale)
    return ref_fp8_dequant(w8, s8)


def slow_fp4_dequant(weight_i8: torch.Tensor, scale: torch.Tensor, rows) -> torch.Tensor:
    """Per-element decode with Python floats, for a few rows only."""
    out = torch.zeros(len(rows), weight_i8.shape[1] * 2, dtype=torch.float32)
    wb = weight_i8.view(torch.uint8)
    for ri, r in enumerate(rows):
        for c in range(weight_i8.shape[1] * 2):
            byte = int(wb[r, c // 2])
            nib = (byte >> 4) if (c & 1) else (byte & 0xF)
            e = int(scale[r, c // 32].view(torch.uint8))
            out[ri, c] = float(FP4_TABLE[nib]) * (2.0 ** (e - 127))
    return out.to(torch.bfloat16)
