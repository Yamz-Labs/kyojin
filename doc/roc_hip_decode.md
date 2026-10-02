# Experimental gfx12 ROCm decode backend

This page documents the implemented Heterogeneous-compute Interface for Portability (HIP) decode path. It accelerates EXL3 matrix-vector multiplication (GEMV) on RDNA4 and provides correctness fallbacks outside that envelope.

## What runs on gfx12

The runtime gate in `exllamav3.modules.quant.exl3.LinearEXL3` sends a layer to HIP GEMV only when:

- ROCm is active and `ext.exl3_gemv_supported(device)` is true;
- the GPU is `gfx1200` or `gfx1201`;
- the decode batch is `rows <= 16` for the HIP GEMV route, subject to the existing K/codebook constraints; rows 9..16 are only for the supported native MMODE-2 variants (K3/mul1, K4/mcg or mul1, K5/mul1, K6/mcg or mul1), while K2 and all other codebook variants retain the rows<=8 limit; unsupported combinations fall back;
- `in_features` and `out_features` are both multiples of 128;
- `K` is from 2 through 6;
- the existing codebook flags match the instantiated family (`mcg` / `mul1` when required); K5 and K6 require one of those two codebooks.

On those gfx12 devices, the real EXL3 decode kernel uses `v_wmma_f32_16x16x16_f16` with fp32 accumulation. That is the supported ROCm acceleration path here; it is not a general matrix fused multiply-add (MFMA) backend.

## What falls back

Anything outside the envelope above uses the existing reconstruct path:

- rows `> 16`
- unsupported bitrate / codebook combinations
- `EXL3_GEMV=0`
- non-gfx12 ROCm devices

The fallback is `reconstruct + hgemm`. That path is correctness-first. It is not a fused performance kernel, and it does not try to match CUDA decode throughput.

The rest of the ROCm shims in `ext_fallbacks.py` follow the same rule: they preserve behavior with PyTorch implementations for correctness, not speed.

## Why the HIP launches are split

HIP does not use the CUDA cooperative-grid decode path.

Instead it runs three stream-ordered launches:

1. input Hadamard
2. ordinary GEMV
3. output Hadamard

The earlier cooperative-launch experiment was removed after the Heterogeneous System Architecture (HSA) runtime crashed during shutdown. CUDA keeps its separate cooperative path with `grid.sync()`.

## Why LDS staging is mandatory on gfx12

The gfx12 implementation stages decoded trellis words and wave matrix multiply-accumulate (WMMA) operands through warp-private local data share (LDS).

That is not an optimization choice. On gfx12, feeding WMMA from the shuffle path caused hardware exceptions, so the HIP kernel disables that path.

## What is still deferred

Out of scope for this backend slice:

- large-throughput matrix-matrix multiplication (GEMM, `rows > 144`)
- int8 GEMV
- CDNA MFMA
- the quantizer
- CUDA graphs / `bc_*`

## Validation snapshot

Current local verification on gfx1201 includes:

- 98 GEMV matrix, routing, codebook, K5, and K6 tests per model fixture
- 88 cache / reconstruct tests
- 61 multi-head latent attention (MLA) / DeepSeek sparse attention (DSA) tests
- 34 native ROCm hyperconnection route, numerical, stream, validation, and fallback tests
- 63 grouped K3 MoE route, numerical, multirow, safety, LoRA, and fallback tests, plus opt-in full Flash and MTP oracles
- 44 native ROCm shared-gate fusion, stream, validation, and fallback tests
- 25 gfx12 batch-one router numerical, tie, stream, validation, fallback, and real-layer tests
- forced-identical-context 16-step logits oracles for mul1 and MCG models

The Qwen3.8-27B mul1 oracle reports:

- top-1 agreement: 16/16
- top-5 overlap: at least 4/5 tokens at every step
- observed max absolute logit delta: 0.488281
- worst-step mean delta: 0.046886
- overall mean delta: 0.027793
- fallback-only `-inf` positions carry <1e-6 softmax mass per step

The Qwen3.5-9B MCG oracle reports:

- top-1 agreement: 16/16
- exact top-5 agreement: 16/16 steps
- max absolute logit delta: 0.234375
- worst-step mean delta: 0.017942
- overall mean delta: 0.009152

Warmed decode measurements on gfx1201:

| Model | Trial size | K6 reconstruct median | Direct K6 median | Uplift |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 64 tokens | 14.569 tok/s | 16.250 tok/s | 11.5% |
| Qwen3.5-9B | 128 tokens | 36.522 tok/s | 47.311 tok/s | 29.5% |

Each median is three trials after a 32-token warmup, using one model, prompt, and GPU. The K6-reconstruct arm preserves direct GEMV for K2-K4 and falls back only for K6.

Reusing each assembled WMMA A fragment across adjacent N tiles provides another measured gain:

| Model | Before A reuse | With A reuse | Uplift |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | 16.315 tok/s | 17.325 tok/s | 6.2% |
| Qwen3.5-9B | 47.315 tok/s | 51.917 tok/s | 9.7% |

Specializing the m=1 reduction for its single valid output row raises Qwen3.8-27B from 17.285 to 18.115 tok/s (4.8%). Qwen3.5-9B rises from 51.880 to 54.411 tok/s (4.9%). This reduces kernel LDS use from 28,672 to 14,336 bytes and raises occupancy from two to four blocks per compute unit.

Compared with the earlier all-reconstruct baseline, the current backend measures 3.94x faster on Qwen3.8-27B (4.603 to 18.115 tok/s). Qwen3.5-9B measures 3.66x faster (14.847 to 54.411 tok/s). These cross-run ratios show the cumulative backend improvement; the tables and percentages above are controlled incremental comparisons.

Steady-state profiles after prefill show the K6 vocabulary head changing from reconstruct plus rocBLAS hgemm to one direct kernel:

| Model | Steps | K6 reconstruct + hgemm | Direct K6 | Total self GPU time change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 8 | 80.543 ms | 18.791 ms | 483.397 ms → 423.119 ms |
| Qwen3.5-9B | 12 | 97.071 ms | 21.174 ms | 292.792 ms → 217.324 ms |

A-fragment reuse reduces total self GPU time from 423.119 to 400.882 ms for the eight-step Qwen3.8 profile. The dominant K4 mul1/fp16 wide kernel falls from 163.596 to 154.410 ms. For the twelve-step Qwen3.5 profile, time falls from 217.324 to 197.155 ms. Its K4 MCG/fp32 narrow kernel falls from 67.561 to 57.072 ms.

The one-row reduction lowers the Qwen3.8 wide K4 kernel again, from 154.413 to 140.053 ms in a controlled profile. The corresponding Qwen3.5 wide K4 kernel falls from 51.207 to 42.012 ms; its narrow kernel is effectively unchanged.

## Qwen3.8-Flash-Next target

The `turboderp/Qwen3.8-Flash-Next-exl3` 3.05-bpw checkpoint runs target-only with a layer split across two gfx1201 GPUs. Its 32.64-GB PLE n-gram table stays file-backed and streams only selected rows. A balanced 27/27-GB load budget allocated approximately 28.6 and 24.4 GB on the two GPUs with a 32K cache.

A short greedy smoke produced `Paris. Paris is the capital` and executed the direct EXL3 route. Adding K5 support moves the model's high-quality shared-expert and other K5 projections off reconstruct+hgemm:

- warmed target-only decode: 6.588 to 7.687 tok/s, a 16.7% gain;
- four-token self-device profile: 875.654 to 752.568 ms, a 14.1% reduction;
- K5 reconstruction: 1,232 launches to zero;
- real shared-expert gate/up/down K5 projections match reconstruct+hgemm on both gfx1201 devices.

The existing fused GatedResidual and hyperconnection kernels are also available on wave32 ROCm devices. They replace the decode reference path's small matrix multiplications and elementwise chains with two `gr_mix` launches and one in-place `hc_apply` launch per residual site. Unsupported wave sizes retain the PyTorch path. In a controlled same-process comparison, fusion improved the 32-token median from 7.787 to 14.085 tok/s. Over four tokens, it reduced `aten::mm` from 1,552 to 392 calls and removed 7,736 HIP launches. A separate fresh-process run measured a 13.553 tok/s median.

Keeping the PLE table in RAM did not improve that fresh-process result: disk streaming measured 13.553 tok/s versus 13.410 tok/s from RAM. RAM residency also increased model load time from 17.0 to 30.9 seconds. The streamed path remains the recommended default and avoids reserving 32.64 GB of host memory.

This is a foundation result, not the final serving profile. Qwen4Exp currently uses layer split rather than tensor parallelism. MTP has not been enabled, and sparse QSA beyond the short-context dense threshold still needs ROCm qualification. The remaining short-context bottlenecks are launch count and K3 expert GEMV. Prefill and long-context performance need separate measurements before comparison with sparse QSA serving results.

## MCG compatibility

The Qwen3.5-9B fixture exposed an incorrect portable translation of the original PTX `lop3` expression. The fix restores the exact LUT `0x6a` semantics for plain and MCG procedural codebooks. A zero-state reconstruction test now checks the codebook value independently of both GEMV and the reconstruct implementation.

## Build and test

```sh
cd exllamav3
export HIPCXX=hipcc PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=4
pip install . --no-build-isolation
python -c "from exllamav3 import ext; assert ext.exllamav3_ext.exl3_gemv_supported(0)"

EXL3_TEST_MODEL=/path/to/Qwen3.8-27B-exl3 \
  python -m pytest tests/test_hip_gemv_decode.py -q

EXL3_TEST_MODEL=/path/to/Qwen3.8-27B-exl3 \
  python tests/hip_gemv_logits_oracle.py

# Repeat with an MCG fixture.
EXL3_TEST_MODEL=/path/to/Qwen3.5-9B-exl3 \
  python tests/hip_gemv_logits_oracle.py

```