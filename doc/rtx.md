# NVIDIA RTX 3090

Kyojin also runs on NVIDIA (sm_80 and up, tested on one RTX 3090 with 24 GB). It uses the same sources as the ROCm build; the CUDA path adds a quantized-KV decode attention kernel, a Flint sliding-window ring with whole-forward CUDA graphs, a grouped MoE prefill tier, an sm_86 GEMM layout and a decode GEMV policy. Build: [tools/cuda/README.md](../tools/cuda/README.md). Switches: [doc/env_vars.md](env_vars.md#nvidia-cuda-sm_80).

Machine for every table on this page: one NVIDIA RTX 3090, 24 GB. Real text, greedy, one user, KV cache quantized (Flint 6/6 bits, Qwen27 4/4 bits), tokens per second.

**Flint V1.** Kyojin 1.5.10 (measured before 2026-11-04; Kyojin 1.8.9 not re-measured yet).

| model | | 4K | 16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|---|
| Flint V1 (3.43 bpw, 17 GiB) | decode | 133.5 | 127.0 | 122.9 | 119.5 | 102.6 | 81.3 |
| | prefill | 6282 | 6956 | 5938 | 4449 | 2995 | 1807 |

**Qwen3.8-27B.** Server, vision on, MTP + n-gram. Kyojin 1.8.9, served, 2026-11-04 for 5K to 150K (1 warm-up, 1 rep; 3 reps and the median at 150K); the 250K column is Kyojin 1.5.10 (not re-measured yet).

| model (server, vision on) | | 5K | 39K | 79K | 150K | 250K (1.5.10) |
|---|---|---|---|---|---|---|
| Qwen3.8-27B (EXL3 4.0 bpw, MTP + n-gram) | decode | 74.6 | 69.7 | 71.2 | 62.5 | 49.9 |
| | prefill | 1621 | 1544 | 1353 | 1118 | 611 |

Plain decode without speculation on Qwen27 (Kyojin 1.5.10): 45.6 tok/s at 4K, 41.5 at 32K, 37.2 at 128K, 32.8 at 256K. Both models fit 256K tokens of context in 24 GB (peak 21 GiB for Flint, 20.5 GiB for Qwen27 with the vision tower).

Run at 256K:

```bash
# Flint V1: 256K with 6-bit K and V (fp16 KV fits about 131K)
python tools/flint/serve.py --model ./flint-pack -c 262144 --kv 6,6
# Qwen27: 256K, 4-bit K and V, vision and the MTP drafter on
python tools/qwen/serve.py --model ./qwen27-pack -c 262144 --k-bits 4 --v-bits 4
```

On NVIDIA both servers turn on the CUDA graph and these cache widths by default; `--kv` and `--k-bits`/`--v-bits` override them. The quantized caches are lossy: the Flint 6/6 cache measures about +9 % KL against fp16 and Qwen27 4/4 measures 1.4e-3 KL and finds the needle 3 times out of 3 at 256K (2 of 3 at 128K, the miss being a refusal wording that fp16 shares on other texts); `tools/flint/kvgate.py` and `tools/qwen/kvgate.py` rerun both checks.

Against other engines on the same card (Kyojin 1.5.10; same prompts, greedy, decode | prefill, tokens per second):

| Qwen3.8-27B | 4K | 32K | 128K | ~250K | max context |
|---|---|---|---|---|---|
| Kyojin, K4V4 (no speculation) | 44.6 \| 1130 | 43.0 \| 1074 | 37.6 \| 794 | 25.3 \| 582 | 262144 |
| ExLlamaV3 upstream, K4V4 | 41.6 \| 1109 | 38.2 \| 896 | 30.2 \| 709 | 24.7 \| 524 | 262144 |
| llama.cpp Q4_K_M, KV q4_0 | 40.5 \| 1246 | 33.9 \| 1275 | 21.6 \| 880 | 14.8 \| 628 | 262144 |
| llama.cpp Q4_K_M, KV q8_0 | 38.7 \| - | 24.1 \| 900 | - | - | about 196K |

| Flint V1 | 4K | 32K | 128K | ~250K | max context |
|---|---|---|---|---|---|
| Kyojin, KV 6/6 | 136.6 \| 4275 | 127.1 \| 4436 | 103.0 \| 2639 | 82.2 \| 1733 | 262144 |
| llama.cpp Q3_K_XL, KV f16 | 123.1 \| 4630 | 111.3 \| 4262 | 78.5 \| 2411 | - | 131072 tested |
| llama.cpp Q3_K_XL, KV q8_0 | 112.8 \| 4624 | 83.9 \| 4209 | 43.8 \| 2399 | 27.4 \| 1562 | 262144 |

Kyojin leads on decode at every depth and on the context that fits; llama.cpp prefills Qwen27 10 to 19 % faster up to 32K. These two tables come from one earlier measurement session (single user, same prompts for every engine, Kyojin without the later prefill work), so the Kyojin rows are lower than in the first two tables. Weights differ between engines (EXL3 against GGUF), so quality is not compared.


## Quickstart

Build for CUDA once ([tools/cuda/README.md](../tools/cuda/README.md)), then use the commands under "Run at 256K" above. Packs: [Flint-40B-A2.5B](https://huggingface.co/yamz-labs/Flint-40B-A2.5B-EXL3-Yamz), [Nanbeige4.3-33B](https://huggingface.co/yamz-labs/Nanbeige4.3-33B-EXL3-Yamz).
