# Benchmarks

## How these figures are measured

One machine: Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X, ROCm. Other GPUs are untested.

The figures of the three models come from a run set up like yours:

- the public code at the commit named in `doc/figures.json`, built from the README install;
- an empty home directory, the published packs downloaded from the hub, the default environment (no private flags);
- greedy decoding, thinking off, the four card prompts for each of chat, prose and code, 256 tokens, median of 3 repetitions (`tools/bench.sh` runs the same prompts);
- prefill and decode by context: cold prompts of 8K to 256K tokens (MiMo: to 128K), an essay request of 256 tokens, after one discarded 8K request; prefill is the speed the server reports for the request. Decode: Qwen, like the other engines in the comparison below, is timed by the client from the first to the last streamed token; GLM and MiMo use the server's own decode counters (`/metrics`), because their streams arrive in bursts that a client clock misreads. Qwen and GLM: {{LADDER}}; MiMo: two requests per depth, the second one reported. GLM and MiMo are started with a context large enough for the longest prompt (`-c 272384` and `-c 163840`);
- speculative decode is the default server mode and returns the same tokens as plain decode.
- the server log of each run has no `fallback`, `disabled`, `unavailable`, `not tuned` or `No module` line.

## Qwen3.8-Flash-Next, GLM-5.3-Flash and MiMo-V2.6-Flash

Tokens per second; prefill by context, speculative decode by kind of text.

{{TABLE}}

![Prefill against context, three models](img/prefill.svg)

{{DECODE}}
## Where each figure comes from

- Qwen3.8-Flash-Next, GLM-5.3-Flash and MiMo-V2.6-Flash, prefill and speculative decode (by context and on the card prompts): runs with the recipe above, at the commit named in `doc/figures.json`.
- Strata and Gufo: the section "Strata and Gufo" below names the source of each figure.
- Charts and the front-page table are built from `doc/figures.json` with `tools/make_charts.py` and `tools/make_readme_table.py`.

Caveats, once: one machine, one OS image; prefill is the engine-reported prompt speed; figures are steady-state: the first request at a new prompt length is slower than the next ones. These are first versions and improvements are coming.

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
| MiMo-V2.6-Flash-MOPD, 105 GB pack, speculative decoding on (default), 4 bpw drafter (702 MB), `-c 32768`, client temperature 0, decode medians of 6 runs over two loads, 128 tokens | - | 32.1 prose, 34.8 chat, 44.3 code | run, public tree |

## Strata and Gufo

{{RIVALS}}
## Other engines

llama.cpp (ROCm, UD-IQ1_S 1.56 bpw, `-fa 1 -ub 2048`, no MTP), same machine: pp4096 197.9, pp16384 158.3, tg128 16.74, tg at 64K 7.19 tok/s. Coarser quant: engine and format are compared together.

## Share your numbers

Start a server from the quickstart, then run `tools/bench.sh` (standard library only, `--base` and `--model` select the server). It measures prefill on a prompt of about 3.5K tokens and decode on prose, chat and code with the prompts behind the table above, and prints one Markdown block with your hardware and versions (`--json` adds the same numbers, with every run, as a JSON block).
Paste it into a [benchmark report](https://github.com/Yamz-Labs/kyojin/issues/new?template=benchmark_report.yml). Results from other gfx1151 machines and other ROCm GPUs are the most useful contribution. See `CONTRIBUTING.md`.
