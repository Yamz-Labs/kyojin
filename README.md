<p align="center"><img src="assets/yamz-banner.svg" alt="Yamz" width="720"></p>

# Kyojin

Kyojin is the Yamz inference engine for AMD Strix Halo, built on [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp. It adds a ROCm decode and prefill path for the AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151, unified memory) and serving for two large MoE models with multi-token prediction.

The AMD work starts from [vcruz305/exllamav3-amd](https://github.com/vcruz305/exllamav3-amd) (first gfx1151 port) and [sdougbrown/exllamav3](https://github.com/sdougbrown/exllamav3) (ROCm decode path for gfx12). Full credits are [below](#credits-and-licence).

The two models:

- GLM-5.3-Flash (`glm_moe_dsa`: MLA attention, sparse indexer, MTP)
- MiMo-V2.6-Flash (`mimo_v2`, DFlash drafter)

The CUDA paths of upstream are kept. AMD additions sit behind `USE_ROCM` and architecture guards. This repository holds the engine, the serving scripts and the benchmark harnesses. The quantisation pipeline that produced the model packs is not part of it.

Measured on one Strix Halo machine: GLM-5.3-Flash prefills at 546 to 584 tok/s (3.5K to 64K context) and decodes at 26 to 30 tok/s with MTP; MiMo-V2.6-Flash decodes at 32 to 44 tok/s with speculative decoding on (29 tok/s plain) and prefills at about 650 tok/s (numbers and sources below). Both models run in 128 GB.

Weights: [Yamz on Hugging Face](https://huggingface.co/yamz-labs) - `yamz-labs/GLM-5.3-Flash-EXL3-Yamz`, `yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz`.

## Quickstart (Strix Halo, ROCm)
Requirements: Linux (Ubuntu/Debian tested), a gfx1151 machine with 128 GB, Python 3.12, `gcc`, a ROCm 7 `libhsa-runtime64.so.1` (the PyTorch copy segfaults on gfx1151 and Ubuntu's own `libhsa-runtime64-1` is ROCm 5.7, too old; `tools/strix_halo/env.sh` finds the one inside the SDK wheel below first, then /opt/rocm, or take `EXL3_HSA_LIB=<path>`), ROCm 7.0 or newer (ROCm 6.4 has no gfx1151 code), a ROCm build of PyTorch for gfx1151, and a ROCm SDK devel tree with the `hipsparse/` and `thrust/` headers (the `rocm-sdk-devel` wheel).

```bash
git clone https://github.com/Yamz-Labs/kyojin && cd kyojin
python3 -m venv .venv && source .venv/bin/activate
# 1. ROCm torch + SDK first (AMD gfx1151 wheels; plain `pip install torch` gives a CUDA/CPU build that cannot run here):
pip install --pre torch rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
pip install -r requirements.txt
rocm-sdk init                              # expands the devel headers (about 12 GB on disk)
export EXL3_ROCM_SDK=$(rocm-sdk path --root)   # or the path of your own devel tree
source tools/strix_halo/env.sh             # run from the repository root; sets LD_PRELOAD and PYTHONPATH (torch needs it to import)
./build.sh                                 # compiles the extension for gfx1151 into the repo root (exllamav3_ext*.so)
bash tools/strix_halo/env.sh --check       # prints versions, runs a small GPU matmul (needs a free GPU)
hf download yamz-labs/GLM-5.3-Flash-EXL3-Yamz --local-dir ./glm-pack
python tools/glm/serve.py --model ./glm-pack --port 8000 -c 131072 --num-draft 2
curl http://localhost:8000/v1/models
```
In every new shell, before serving: activate the venv, run `export EXL3_ROCM_SDK=$(rocm-sdk path --root)` again, then `source tools/strix_halo/env.sh`. Without the export, `env.sh` falls back to a system ROCm that may not match the wheel (`undefined symbol: hsa_ext_image_create_v2`).
If a step fails, see [doc/troubleshooting.md](doc/troubleshooting.md).
Status: build and MiMo serving were verified from a fresh clone on a second Strix Halo machine (build in 8 to 10 minutes). Both published packs were checked there against `SHA256SUMS` and served (one chat request each); `env.sh --check` has not been run there yet. The first request after a build is slow while the kernels warm up.

First GLM launch: the server tunes its dense GEMM kernels before it opens the port. This takes about 10 minutes, and the port stays closed during that time (the log prints a progress line every 30 seconds). The next one or two launches can repeat it for the shapes still missing; later launches start fast.

GLM reasons at maximum effort by default. With a small `max_tokens`, the reasoning can use the whole budget: `content` is then empty and `finish_reason` is `length`. Send `"reasoning_effort": "low"` in the request (or start the server with `--default-reasoning-effort low`), or allow at least 1000 tokens. The server defaults to greedy sampling when a request gives no temperature; set the sampling values of the model card in your client, or with `--default-temperature` and `--default-top-p`.

GLM: the pack ships `mtp_eh_proj.st`, an unquantized MTP `eh_proj` (64 MiB). The server loads it automatically from the model folder; it raises draft acceptance and decode speed (26.7 -> 31.6 tok/s greedy on the card protocol). Details in `tools/glm/SERVE.md`.

Memory: GLM at `-c 131072` leaves about 10 GiB free on a 128 GB machine. Close browsers and other large processes first, or use a smaller `-c`.

MiMo: `hf download yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz --local-dir ./mimo-pack`, then `python tools/mimo/serve.py --model ./mimo-pack --port 8000 -c 131072`. Speculative decoding (DFlash, 4 bpw drafter, confidence-truncated drafts) is on by default: the server uses the pack's `drafter/` directory (or `$MIMO_DRAFTER`, or `--drafter <dir>`). Without a drafter it logs one line and decodes plain. Set `MIMO_SPEC=0` in the lane script (`tools/lanes/serve_mimo.sh`) or pass `--no-dflash` to `serve.py` for plain decode. Greedy output under speculation is token-identical to plain decode (exact verify arithmetic). `EXL3_MIMO_LOSSLESS=0` selects a slightly faster verify that is not token-identical: near-tied logits can flip. A loaded drafter costs 2 to 4 % prefill. Details: `tools/mimo/SERVE.md`.

Qwen3.8-Flash-Next (125B MoE, 6B active): `hf download yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz --local-dir ./qwen-pack`, then `python tools/qwen/serve.py --model ./qwen-pack --port 8000 -c 131072`. Speculative decoding is on by default and returns the same tokens as plain decoding. The pack needs about 113 GiB of the 128 GB; start to ready takes about 90 s once the pack is in the page cache; the very first start reads 95 GB from disk and can take 7 to 8 minutes. Details: `tools/qwen/SERVE.md`.

## Measured numbers
One machine: Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X, ROCm. Other GPUs are untested.

| Model / pack | Context | Prefill tok/s | Decode tok/s | Source |
|---|---|---|---|---|
| GLM-5.3-Flash, 82 GB pack, MTP 2 | 4K / 16K / 64K / 128K | 644.6 / 620.9 / 617.7 / 608.2 | 32.1 prose, 33.8 chat, 35.8 code; 38.9 chat at 128K context | run |
| GLM-5.3-Flash (99.7 GB pack), MTP 2, `-c 524288`, raw server | 3.7K / 14.2K | 611.9 / 585.0 | 27.1 default sampling, 28.1 greedy | run, single runs |
| same, through an OpenAI-style proxy | 3.7K / 14.4K | 590.0 / 589.8 | 29.1 default sampling, 28.1 greedy | run, single runs |
| GLM-5.3-Flash, 99 GB pack, MTP 2, `-c 131072` | 24K | 609.0 | 28.1 (28.12, 28.09) | run, same harness as the next row |
| GLM-5.3-Flash, 82 GB pack, hook off, today's engine, `-c 131072`, client temperature 0, mean of 3 | 3.5K / 14K | 661 / 645 | 32.0 prose, 33.6 chat, 35.7 code | run, second machine (same CPU) |
| GLM-5.3-Flash (99.7 GB), hook on + agent-lane flags, `-c 98304`, prefill mean of 3, decode mean of 6 | 3.5K / 14K / 64K | 580 / 584 / 546 | 29.0 / 30.3 / 27.6 at temperature 0; 26.0 / 28.5 / 26.4 at temperature 1.0, top-p 0.95 | run |
| MiMo-V2.6-Flash-MOPD, 105 GB pack, speculative decoding on (default), 4 bpw drafter (702 MB), `-c 32768`, client temperature 0, decode medians of 6 runs over two loads, 128 tokens | - | 32.1 prose, 34.8 chat, 44.3 code (plain: 28.9 on all three, 1.11x / 1.21x / 1.53x) | run, public tree |
| MiMo-V2.6-Flash-MOPD, 105 GB pack, plain decode (no draft), 32K window, client temperature 0 | 4.1K / 23.7K | 594 / 613 | 25.2 to 27.8 | run, single runs |

**First launch.** The engine tunes its dense GEMM kernels on the first requests and keeps the result in a cache. On a fresh install the first GLM prefills run at 200 to 240 tok/s; speed reaches the figures above within a few requests and stays there on later launches.

llama.cpp (ROCm, UD-IQ1_S 1.56 bpw, `-fa 1 -ub 2048`, no MTP), same machine: pp4096 197.9, pp16384 158.3, tg128 16.74, tg at 64K 7.19 tok/s. Coarser quant: engine and format are compared together.

Quality against the official FP8 weights (129 held-out rows): see the model cards.

## Share your numbers
Start a server from the quickstart, then run `tools/bench.sh` (standard library only, `--base` and `--model` select the server). It measures prefill on a prompt of about 3.5K tokens and decode on prose, chat and code with the prompts behind the table above, and prints one Markdown block with your hardware and versions (`--json` adds the same numbers, with every run, as a JSON block).
Paste it into a [benchmark report](https://github.com/Yamz-Labs/kyojin/issues/new?template=benchmark_report.yml). Results from other gfx1151 machines and other ROCm GPUs are the most useful contribution. See `CONTRIBUTING.md`.

## Optional refusal hook
The engine can project one fixed direction out of the residual stream at run time. It edits no weights and no quantised data. Off unless `EXL3_ABLIT_RUNTIME=/path/to/spec.json` is set or the model folder contains `uncensor_spec.json`.
- Bundled spec: if `<model_dir>/uncensor_spec.json` exists (with `uncensor_direction.st` next to it, or `uncensor_spec.safetensors`), the engine applies it at load and logs ` -- ablit runtime: spec <path> active (bundled in the model directory)`. An explicit `EXL3_ABLIT_RUNTIME=<path>` wins. `EXL3_ABLIT_RUNTIME=off` or `--no-uncensor` (GLM and MiMo servers) disables it. A missing or malformed spec stops the load with an error that names the file.
- The spec is a JSON file (`hidden`, `n_layers`, per-layer weights `attn_w[L]`, `mlp_w[L]`) plus a unit vector `r` in `spec.safetensors`.
- After each attention and MLP block of layer L the output becomes `y - w_L * r * (r . y)`.
- `EXL3_ABLIT_TORCH=1` forces the PyTorch path instead of the Triton kernel.
- A direction fitted on refusals makes the model refuse less. Whoever uses a spec is responsible for it. This repository contains no spec file. Steering presets are published separately in the Yamz presets repository (https://github.com/yamz-labs/yamz-presets).
- Code: `exllamav3/modules/ablit_runtime.py`. Test: `tests/test_ablit_runtime_cpu.py`, `tests/test_uncensor_bundled_cpu.py`.

## Tests
```bash
pip install pytest
for t in tests/test_ablit_runtime_cpu.py tests/test_uncensor_bundled_cpu.py tools/glm/test_serve.py tools/mimo/test_serve.py tools/mimo/test_toolcalls.py; do PYTHONPATH=. pytest -q $t; done
```
Run each file separately because two files share a name (`test_serve.py`). These tests need no GPU and no built extension. GPU tests need a built extension and a free GPU.

## Build on CUDA
The CUDA build is the upstream one and is unchanged. Install a CUDA 12.4 or newer build of PyTorch, then `pip install -r requirements.txt && pip install .`. `README.upstream.md` has the full upstream guide (wheels, PyPI, uv, Windows, architecture list, conversion tool, examples). `README.strix-halo.md` has the kernel notes and benchmark harnesses.

## Credits and licence
ExLlamaV3 by turboderp (MIT, `LICENSE` unchanged). ROCm decode path for gfx12 from sdougbrown/exllamav3; first gfx1151 port from vcruz305/exllamav3-amd. `exllamav3/vendor/fla` is flash-linear-attention (MIT). GLM-5.3-Flash is by Z.ai, MiMo-V2.6-Flash by Xiaomi; check each base licence before redistributing weights. Additions: MIT. This project is not affiliated with Z.ai, Xiaomi or turboderp.

## Thanks
To [@felladrin](https://github.com/felladrin) for the first community contributions: CPU tests for the bench tool,
the troubleshooting page, a faster GLM start, and field notes on running Kyojin in containers.

To the people who test Kyojin on their own machines and take the time to write precise reports:
[@felladrin](https://github.com/felladrin) (install, HIP runtime lookup, MiMo finish reason, client disconnects),
[@dturini12](https://github.com/dturini12) (streaming API), [@morrisfamily](https://github.com/morrisfamily) (benchmarks),
[@nobert](https://github.com/nobert) (metrics endpoint). Reports and pull requests are welcome.

News and benchmarks: [@YamzLabs on X](https://x.com/YamzLabs).
