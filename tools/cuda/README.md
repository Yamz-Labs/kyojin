# Kyojin on NVIDIA (CUDA) — build and smoke test

One source tree for ROCm and CUDA. AMD-specific constructs in the Kyojin kernels go through
`exllamav3/exllamav3_ext/port.cuh`; every macro expands on ROCm to the exact text the ROCm build always used
(checked by compiling the touched files for gfx1151 before/after and diffing the assembly).

## Toolchain without a system CUDA (tested on an RTX 3090, sm_86)
NVIDIA PyPI wheels, `pip download --no-deps`, unpacked into one CUDA_HOME directory:
| wheel | sha256 |
|---|---|
| nvidia_cuda_nvcc-13.0.88 | 56fe502eb77625a12f25172caa3cdddb4e4c8ba2c8c17dba44b164761b380f03 |
| nvidia_nvvm-13.0.88 | c5f41ffeb6466944a026dfa5317d7d85355c119bbec279205d22f1869d1054e0 |
| nvidia_cuda_crt-13.0.88 | 2c8043c7c9e02492716426e9919fc78d2c5b3b2a7a768a88e952676b08aa55a4 |
| nvidia_cuda_cccl-13.0.85 | e0da7ad981f3a8aff08241b5bfc1af868742a63e2762f53a5171c492ef242649 |

CUDA_HOME layout: `bin/` (nvcc, ptxas, ...), `include/` (crt + cccl headers merged), `nvvm/`, and `lib64/` holding
unversioned links (`libcudart.so`, `libcublas.so`, `libcublasLt.so`, `libcurand.so`) to the `nvidia/cu13/lib/*.so.N`
files of the torch cu130 wheel's venv (the linker wants the unversioned names; the wheels only ship `.so.N`).
torch: 2.10 cu130 (CUDA major must match nvcc). Host compiler: gcc 13.

## Build
    tools/cuda/build.sh /path/to/cuda13 /path/to/venv-cuda/bin/python
~3 min with 10 jobs; output `exllamav3_ext*.so` in the tree root. Never point TMPDIR at /tmp on shared machines.
`EXL3_PORT_DEC=1` also compiles the Kyojin HIP decode GEMV family (`quant/exl3_dec.cu`) for CUDA; it is bound only
when `EXL3_PORT_DEC=1` is set at import too (Python enables these kernels by `hasattr`, and until a port beats the
upstream CUDA kernel it must not replace it silently). That file alone takes cicc a long time (fully unrolled decode).

## Test
    python -I tools/cuda/smoke.py <tree> <model dir> out.json 256        # 3 prompts, greedy, ttft / prefill / decode tok/s
env: `SMOKE_KV=K,V` (quantized cache bits, default fp16), `SMOKE_MTP=n` (Kyojin MTP drafter), `SMOKE_POLICY=fixed|mix`, `SMOKE_DFLASH=<DFlash2 drafter dir>`.
For lossless numbers set `EXL3_INT8_GEMV=0` (upstream's CUDA default quantizes 1-2 row GEMV activations to int8).
Kernel parity / speed of a ported kernel: `tests/port/` (e.g. `bench_dec.py`).
