# Benchmarks

## How these figures are measured

One machine: Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X, ROCm. Other GPUs are untested.

The Qwen and MiMo figures on the front page come from a run set up like yours (GLM: see below):

- the public code at the commit named in `doc/figures.json`, built from the README install;
- an empty home directory, the published packs downloaded from the hub, the default environment (no private flags);
- greedy decoding, thinking off, the four card prompts for each of chat, prose and code, 256 tokens, median of 3 repetitions (`tools/bench.sh` runs the same prompts);
- prefill and decode by context: cold prompts of 8K and 32K tokens (Qwen also 64K, 128K, 256K), a long essay request; prefill is the engine-reported prompt speed, decode the speculative decode speed over the reply. GLM and MiMo: median of 3 requests after a warm-up request, speed from the server timing; Qwen: one request per depth, essay of 256 tokens, as `ladder.py --chat`;
- speculative decode is the default server mode and returns the same tokens as plain decode.
- the server log of each run has no `fallback`, `disabled`, `unavailable`, `not tuned` or `No module` line.

## Qwen3.8-Flash-Next and MiMo-V2.6-Flash

Tokens per second; prefill by context, speculative decode by kind of text.

{{TABLE}}

## GLM-5.3-Flash

The GLM figures were published before this page and are quoted as published (99.7 GB pack, MTP with 2 drafts, `-c 98304`, hook on with the agent-lane server flags, prefill mean of 3, decode mean of 6, temperature 0):

| Context | Prefill tok/s | Decode tok/s |
|---|---|---|
| 3.5K | 580 | 29.0 |
| 14K | 584 | 30.3 |
| 64K | 546 | 27.6 |

At temperature 1.0 and top-p 0.95 decode is 26.0 / 28.5 / 26.4 tok/s. The GLM figures were not re-run with the new-user recipe above, so they are left out of the by-text-kind decode chart (the chart needs chat, prose and code figures for the same pack).

## Where each figure comes from

- Qwen3.8-Flash-Next (V1.2) prefill (1494 / 1478 / 1457 / 1395 / 1277 tok/s at 8K / 32K / 64K / 128K / 256K) and speculative decode on the card prompts (49.8 chat, 44.5 prose, 59.1 code): runs with the recipe above, at the commit named in `doc/figures.json`. The model card on the hub still carries the earlier V1.1 figures.
- MiMo-V2.6-Flash: the same recipe and commit.
- GLM-5.3-Flash: the "Measured numbers" table of the previous README (kept below) and the Speed section of the hub card, `yamz-labs/GLM-5.3-Flash-EXL3-Yamz`.
- Charts and the front-page table are built from `doc/figures.json` with `tools/make_charts.py` and `tools/make_readme_table.py`.

Caveats, once: one machine, one OS image; prefill is the engine-reported prompt speed; contexts differ between models because each figure keeps the context at which it was published or measured. These are first versions and improvements are coming.

**First launch.** The engine tunes its dense GEMM kernels on the first requests and keeps the result in a cache. On a fresh install the first GLM prefills run at 200 to 240 tok/s; speed reaches the figures above within a few requests and stays there on later launches.

## Earlier runs (previous README, kept as published)

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

## Other engines

llama.cpp (ROCm, UD-IQ1_S 1.56 bpw, `-fa 1 -ub 2048`, no MTP), same machine: pp4096 197.9, pp16384 158.3, tg128 16.74, tg at 64K 7.19 tok/s. Coarser quant: engine and format are compared together.

Quality against the official FP8 weights (129 held-out rows): see the model cards.

## Share your numbers

Start a server from the quickstart, then run `tools/bench.sh` (standard library only, `--base` and `--model` select the server). It measures prefill on a prompt of about 3.5K tokens and decode on prose, chat and code with the prompts behind the table above, and prints one Markdown block with your hardware and versions (`--json` adds the same numbers, with every run, as a JSON block).
Paste it into a [benchmark report](https://github.com/Yamz-Labs/kyojin/issues/new?template=benchmark_report.yml). Results from other gfx1151 machines and other ROCm GPUs are the most useful contribution. See `CONTRIBUTING.md`.
