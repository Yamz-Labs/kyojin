"""Q8_0 dequant reference (ggml, numpy-only).

Block layout (34 bytes, groups of 32 values), after llama.cpp
``ggml/src/ggml-common.h`` (MIT License, Copyright (c) llama.cpp contributors,
https://github.com/ggml-org/llama.cpp)::

    typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;

i.e. a little-endian fp16 scale followed by 32 int8 residuals, decoded by
``dequantize_row_q8_0`` (``ggml/src/ggml-quants.c``) as ``y[j] = qs[j] * d``.

In the Bolt safetensors, Q8_0 tensors are stored as raw bytes with shape
``[..., num_blocks, 34]`` (e.g. ``lm_head`` is U8 ``[131072, 128, 34]``).
"""
from __future__ import annotations

import numpy as np

Q8_0_GROUP_SIZE = 32
Q8_0_BLOCK_BYTES = 34


def _as_bytes(data: object) -> np.ndarray:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    return np.asarray(data, dtype=np.uint8)


def _as_block_array(data: object) -> np.ndarray:
    """Raw payload as a contiguous uint8 array with trailing dim 34."""
    a = np.ascontiguousarray(_as_bytes(data))
    if a.shape[-1] != Q8_0_BLOCK_BYTES:
        raise ValueError(f"Q8_0 trailing dim must be 34, got shape {a.shape}")
    return a.reshape(-1, Q8_0_BLOCK_BYTES)


def dequant_q8_0_block(block: object) -> np.ndarray:
    """Dequantize one 34-byte Q8_0 block to float32 shape (32,)."""
    flat = _as_block_array(block)
    if flat.shape[0] != 1:
        raise ValueError(f"Expected a single block, got shape {flat.shape}")
    return dequant_q8_0(flat).reshape(Q8_0_GROUP_SIZE)


def dequant_q8_0(data: object) -> np.ndarray:
    """Dequantize Q8_0 payload(s) to float32.

    :param data: uint8 array of shape (..., 34); one 32-value group per block.
    :return: float32 array of shape (..., 32).
    """
    flat = _as_block_array(data)
    # First two bytes: little-endian fp16 scale (GGML_FP16_TO_FP32).
    scales = flat[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    # Remaining 32 bytes: signed residuals.
    quants = flat[:, 2:].copy().view(np.int8).astype(np.float32)
    out = quants * scales[:, None]
    return out.reshape(np.shape(data)[:-1] + (Q8_0_GROUP_SIZE,))
