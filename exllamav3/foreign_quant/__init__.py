"""Foreign-quant dequant reference (numpy, torch-free).

Decodes the non-EXL3 quantization formats shipped in third-party checkpoints
(NVIDIA Nemotron-3.5-Bolt: ggml IQ2_XXS / Q8_0 via ModelOpt + NVFP4 W4A16) to
float32, e.g. to stream-dequant weights for the EXL3 converter. GENERIC:
usable for any ggml/NVFP4 checkpoint, on CUDA or HIP hosts.

Layout conventions follow the sources cited in each submodule. All entry
points take raw little-endian payload bytes as uint8 numpy arrays and return
float32 arrays; no torch, no compiled extension, no GPU.
"""
from __future__ import annotations

__all__ = ["q8_0", "iq2_xxs", "nvfp4"]
