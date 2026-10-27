---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
language: [en, zh]
tags: [exl3, quantized, deepseek, moe, rocm, strix-halo, gfx1151]
---
<!-- Draft. Every "{{NUM}}" cell is produced by tools/ds41/release_measure.sh (command in the cell's row of doc/DS41.md). -->
<p align="center"><img src="assets/banner.png" alt="Yamz Labs" width="100%"></p>

# DeepSeek-V4.1-Flash-1017, EXL3

**Model size:** 763B parameters (about 16B active). The parameter count in the Hugging Face sidebar is not the model's: the site counts the packed 16-bit words of the EXL3 format.

**Hardware:** sized to fit one 128 GB machine. The pack is in the EXL3 format, so it is not tied to one GPU; it has been tested only on AMD Strix Halo (Ryzen AI Max+ 395) with the Kyojin engine, and other hardware is untested.

An EXL3 quantisation of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MoE, 763B total, about 16B active, 384 routed experts per layer, 6 active) for one 128 GB unified-memory machine. It includes the model's own DSpark draft stages for speculative decoding. It runs on the [Kyojin engine, built on ExLlamaV3](https://github.com/Yamz-Labs/kyojin), not on stock ExLlamaV3. Layer mix, quantisation and packaging by Yamz. Not affiliated with DeepSeek.

## Highlights
- **Size.** {{NUM}} GB for a 763B-total MoE: one 128 GB machine holds it with the draft stages and long context. The Engram tables (FP8, about 190 GB, original files) stay on the NVMe and are read per token.
- **Bits.** {{NUM}} bpw on average. Routed experts: hot experts at 3 bit, cold experts at 2 bit, chosen from measured routing. Attention and shared experts 4 to 5 bit. DSpark stages about 4 bit.
- **Quality.** Against the unquantized model: top-1 agreement {{NUM}} %, KLD {{NUM}} (gate rows); KL top-256 {{NUM}} on the public Lucebox held-out tokens.
- **Speed.** Prefill {{NUM}} tok/s at 8K; decode {{NUM}} tok/s with DSpark at 8K. Depth table below, 8K to 1M.
- **Engine.** Kyojin runs it on gfx1151 with ROCm. OpenAI-style API, thinking, tool calls.
- **Feedback wanted.** Run it on your Strix Halo / Ryzen AI Max machine and tell us your tok/s and hardware: open an issue at https://github.com/Yamz-Labs/kyojin/issues or start a discussion on this page.

## How it was made
The weights were quantised with the Kyojin EXL3 converter from the official checkpoint (experts are FP4 in the checkpoint, the rest FP8). Bit allocation per expert came from routing statistics measured on a calibration run: experts used often get 3 bit, the others 2 bit, and the choice is made after the quantisation with the Hessian-weighted error of every candidate. {{NUM}}: calibration set, number of rows, pruned experts (if any).

## Run
```bash
git clone https://github.com/Yamz-Labs/kyojin && cd kyojin
./build.sh                                   # builds the ROCm extension for gfx1151
source tools/strix_halo/env.sh
hf download <org>/DeepSeek-V4.1-Flash-1017-EXL3-<name> --local-dir ./ds41-pack
python tools/ds41/serve.py --model ./ds41-pack --port 8000 -c 131072 --num-draft 5
```
`--num-draft 5` turns on DSpark. Exact flags, memory layout and context guidance: `doc/DS41.md` in the engine repo. These steps were run from a clean install on {{NUM}}.

## Hardware
| Machine | Status |
|---|---|
| Ryzen AI Max+ 395, 128 GB LPDDR5X, Radeon 8060S (gfx1151), ROCm | Tested. All numbers below. |
| Other AMD GPUs, NVIDIA GPUs, less than 128 GB | Not tested. The pack does not fit in less memory. |

Memory at `-c 131072`: {{NUM}} GiB in use after warm-up. Close other large processes first.

## Quality (reference = the unquantized model)
| Measure | Value |
|---|---|
| Gate rows (held-out agentic text): top-1 agreement | {{NUM}} |
| Gate rows: KLD | {{NUM}} |
| Lucebox held-out tokens: KL top-256 | {{NUM}} |
| Lucebox held-out tokens: top-1 agreement | {{NUM}} |
| Lucebox held-out tokens: perplexity (unquantized: 8.5615) | {{NUM}} |

The gate rows and the calibration rows are disjoint. {{NUM}}: the overlap check. Quality is measured against the unquantized model, not against other formats. No task scores were run.

## Speed
DSpark (5 drafts) is the server mode and returns the same tokens as plain decoding when greedy. Method: the public engine code, this pack downloaded from the hub, greedy, thinking off, server-reported prompt speed. One Ryzen AI Max+ 395, 128 GB. Tokens per second, median of 5.

| Depth | Prefill | Decode, no draft | Decode, DSpark | Accepted per step |
|---|---|---|---|---|
| 8K | {{NUM}} | {{NUM}} | {{NUM}} | {{NUM}} |
| 32K | {{NUM}} | {{NUM}} | {{NUM}} | {{NUM}} |
| 128K | {{NUM}} | {{NUM}} | {{NUM}} | {{NUM}} |
| 512K | {{NUM}} | {{NUM}} | {{NUM}} | {{NUM}} |
| 1M | {{NUM}} | {{NUM}} | {{NUM}} | {{NUM}} |

**First launch.** The server tunes and warms up before it answers; {{NUM}}: seconds on a clean install.

## Sampling
Use the base-model card values: temperature 1.0, top-p 1.0.

## Limits
- Only gfx1151 with ROCm is tested.
- Text only; image input is not supported.
- The conversion tooling is not published. The engine and these weights are.

## Licence and credits
MIT. Base model DeepSeek-V4.1-Flash-1017, MIT, licence file included as `LICENSE`. Our additions: Copyright (c) 2026 Yamz Labs, MIT. Provided as is, without warranty. Engine: built on ExLlamaV3 by turboderp (MIT). DeepSeek is a name of its owner; this release is unofficial.
