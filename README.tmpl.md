<p align="center"><img src="assets/yamz-banner.svg" alt="Yamz" width="640"></p>

<p align="center">
Open source. Our own quantised builds of GLM and Qwen. Built on <a href="https://github.com/turboderp-org/exllamav3">ExLlamaV3</a>, with speculative decoding, for 100 GB-class models in 128 GB.</p>

<p align="center">
<a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/licence-MIT-444444"></a>
<img alt="gfx1151" src="https://img.shields.io/badge/AMD-gfx1151%20%7C%20ROCm%207-444444">
<img alt="Python" src="https://img.shields.io/badge/python-3.12-444444">
<a href="https://huggingface.co/yamz-labs"><img alt="Hugging Face" src="https://img.shields.io/badge/weights-Hugging%20Face-444444"></a>
<a href="https://x.com/YamzLabs"><img alt="X" src="https://img.shields.io/badge/news-%40YamzLabs-444444"></a>
</p>

{{TABLE}}


<p align="center"><sub>Tokens per second on one Ryzen AI Max+ 395 with 128 GB. Prefill at the context shown in each cell, speculative decode across chat, prose and code prompts. DeepSeek-V4.1-Flash is larger than memory: most experts are read from the NVMe (about 3 GB/s), so its prefill is warm/cold and its decode is plain decoding on real sessions; a tool turn of about 2K tokens waits 21 to 35 s for the first token. Details: <a href="doc/DS41.md#speed">doc/DS41.md</a>. Method, sources and full tables: <a href="doc/benchmarks.md">doc/benchmarks.md</a>.</sub></p>

<p align="center">
<img src="doc/img/prefill.svg" alt="Prefill speed against context length" width="49%">
<img src="doc/img/decode_models.svg" alt="Speculative decode speed against context length" width="49%">
</p>

{{LONG}}

## Made with our models

One prompt each, written by the model running on Kyojin and published as it came out. [Open the gallery](https://yamz-labs.github.io/gallery/).

<table align="center">
<tr><td align="center" width="33%"><a href="https://yamz-labs.github.io/gallery/#pagoda"><img src="doc/img/gallery/glm-pagoda.webp" alt="Voxel pagoda, GLM-5.3-Flash"></a><br><sub><b>Voxel pagoda</b><br>GLM-5.3-Flash</sub></td><td align="center" width="33%"><a href="https://yamz-labs.github.io/gallery/#pagoda"><img src="doc/img/gallery/qwen-pagoda.webp" alt="Voxel pagoda, Qwen3.8-Flash-Next"></a><br><sub><b>Voxel pagoda</b><br>Qwen3.8-Flash-Next</sub></td><td align="center" width="33%"><a href="https://yamz-labs.github.io/gallery/#pagoda"><img src="doc/img/gallery/mimo-pagoda.webp" alt="Voxel pagoda, MiMo-V2.6-Flash"></a><br><sub><b>Voxel pagoda</b><br>MiMo-V2.6-Flash</sub></td></tr>
</table>

**Uncensored mode.** Each model has an optional mode that makes it refuse far less. The engine applies it at load and the weights stay untouched: ready-made packs for [GLM-5.3-Flash](https://huggingface.co/yamz-labs/GLM-5.3-Flash-EXL3-Yamz-Uncensored) and [MiMo-V2.6-Flash](https://huggingface.co/yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz-Uncensored), and a preset inside the [Qwen3.8-Flash-Next pack](https://huggingface.co/yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz) that is off by default. How it works: [doc/refusal_hook.md](doc/refusal_hook.md).

These are first versions. Speed, context length and model support keep improving.

## DeepSeek-V4.1-Flash

**Experimental.** A demonstration that a 763B model runs exactly on one 128 GB machine. It is slow and not meant for daily agent use: real decode is about 5 to 11 tokens per second, and a tool turn adding about 2K tokens takes about 20 to 35 seconds before the first token (the SSD floor is about 24 s).

DeepSeek-V4.1-Flash-1017 (763B parameters, about 16B active) runs on one 128 GB machine with an NVMe: the 172 GiB EXL3 pack is larger than memory, so the engine keeps a pool of experts in RAM and reads the others from the SSD, exactly. It uses the model's own DSpark draft stages and the original Engram tables (about 190 GB, downloaded from the DeepSeek repository). Pack: [yamz-labs/DeepSeek-V4.1-Flash-EXL3-Yamz](https://huggingface.co/yamz-labs/DeepSeek-V4.1-Flash-EXL3-Yamz). Setup, SSD tier, Engram and speeds: [doc/DS41.md](doc/DS41.md).

## NVIDIA RTX 3090

Kyojin also runs on NVIDIA (sm_80 and up, tested on one RTX 3090 with 24 GB). It uses the same sources as the ROCm build; the CUDA path adds a quantized-KV decode attention kernel, a Flint sliding-window ring with whole-forward CUDA graphs, a grouped MoE prefill tier, an sm_86 GEMM layout and a decode GEMV policy. Build: [tools/cuda/README.md](tools/cuda/README.md). Switches: [doc/env_vars.md](doc/env_vars.md#nvidia-cuda-sm_80).

Measured on the 3090 with real text, greedy, one user, KV cache quantized (Flint 6/6 bits, Qwen27 4/4 bits), tokens per second:

| model | | 4K | 16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|---|
| Flint V1 (3.43 bpw, 17 GiB) | decode | 133.5 | 127.0 | 122.9 | 119.5 | 102.6 | 81.3 |
| | prefill | 6282 | 6956 | 5938 | 4449 | 2995 | 1807 |

| model (server, vision on) | | 5K | 39K | 79K | 150K | 250K |
|---|---|---|---|---|---|---|
| Qwen3.8-27B (EXL3 4.0 bpw, MTP + n-gram) | decode | 70.6 | 65.9 | 64.3 | 55.1 | 49.9 |
| | prefill | 1356 | 1143 | 977 | 787 | 611 |

Plain decode without speculation on Qwen27: 45.6 tok/s at 4K, 41.5 at 32K, 37.2 at 128K, 32.8 at 256K. Both models fit 256K tokens of context in 24 GB (peak 21 GiB for Flint, 20.5 GiB for Qwen27 with the vision tower).

Run at 256K:

```bash
# Flint V1: 256K with 6-bit K and V (fp16 KV fits about 131K)
python tools/flint/serve.py --model ./flint-pack -c 262144 --kv 6,6
# Qwen27: 256K, 4-bit K and V, vision and the MTP drafter on
python tools/qwen/serve.py --model ./qwen27-pack -c 262144 --k-bits 4 --v-bits 4
```

On NVIDIA both servers turn on the CUDA graph and these cache widths by default; `--kv` and `--k-bits`/`--v-bits` override them. The quantized caches are lossy: the Flint 6/6 cache measures about +9 % KL against fp16 and Qwen27 4/4 measures 1.4e-3 KL and finds the needle 3 times out of 3 at 256K (2 of 3 at 128K, the miss being a refusal wording that fp16 shares on other texts); `tools/flint/kvgate.py` and `tools/qwen/kvgate.py` rerun both checks.

Against other engines on the same card (same prompts, greedy, decode | prefill, tokens per second):

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

## Contributors

Thank you to everyone who tests Kyojin on their own machine and writes precise reports.

- [@felladrin](https://github.com/felladrin): pull requests for the GLM start-up, the GPU target lookup, the bench tool tests, the troubleshooting page, `--sessions` in the Qwen server and `tools/bench.sh --json`; bug reports, an independent benchmark and issue triage.
- [@MangetsuC](https://github.com/MangetsuC): the GLM image path without solver searches (first image of a new size in seconds) and the dense GEMM tune primer.
- [@nobert](https://github.com/nobert): the Prometheus `/metrics` endpoint, merged for GLM and MiMo.
- [@morrisfamily](https://github.com/morrisfamily): benchmark report on a BOSGAME Beyond-Max.
- [@ChrisC381](https://github.com/ChrisC381): reported the GLM difference between paired jobs.
- [@dturini12](https://github.com/dturini12): reported streaming without `finish_reason`; fixed and confirmed.
- [@mostlygeek](https://github.com/mostlygeek): proposal for a container setup.
- [@kamjin3086](https://github.com/kamjin3086): benchmark of Kyojin 1.6 against another engine and the report of a host freeze at start-up; both led to fixes in 1.7.

## Quickstart

Install ROCm torch and build once ([doc/install.md](doc/install.md)), then:

```bash
git clone https://github.com/Yamz-Labs/kyojin && cd kyojin
hf download yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz --local-dir ./qwen-pack
python tools/qwen/serve.py --model ./qwen-pack --port 8000 -c 131072
```

The server speaks the OpenAI API on port 8000. GLM and MiMo commands: [doc/models.md](doc/models.md).

## Read more

- [Install and first launch](doc/install.md): requirements, build, per-shell setup, first-launch tuning, [system settings for full speed](doc/install.md#system-settings-for-full-speed)
- [Models](doc/models.md): Qwen, GLM and MiMo weights, serve commands, options
- [Benchmarks](doc/benchmarks.md): how the figures are measured, where each comes from, full tables, share yours with `tools/bench.sh`
- [Troubleshooting](doc/troubleshooting.md), [refusal hook](doc/refusal_hook.md), [development](doc/development.md), [credits and licence](doc/credits.md)

Reports and pull requests are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Kernel notes: [README.strix-halo.md](README.strix-halo.md). Upstream guide: [README.upstream.md](README.upstream.md).

MIT licence. Built on [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp, with the first AMD ports by [vcruz305](https://github.com/vcruz305/exllamav3-amd) and [sdougbrown](https://github.com/sdougbrown/exllamav3): [full credits](doc/credits.md). News: [@YamzLabs on X](https://x.com/YamzLabs).
