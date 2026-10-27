# ds41-serve — DeepSeek-V4.1-Flash-1017 OpenAI-compatible server

`tools/ds41/serve.py` is an aiohttp server in the shape of [tools/glm/serve.py](../glm/SERVE.md): the same
queue, `cache_salt`, `timings` / `usage`, request log, SSE keep-alive and start-up `/health`. Text in, text and
tool calls out (no images). The model is behind a small engine contract (`ds41_engine.py`), so the HTTP layer,
the prompt renderer and the output parser run and are tested without a GPU.

| File | Role |
|---|---|
| `serve.py` | HTTP layer, flags, sampling defaults |
| `ds41_chat.py` | prompt renderer: port of the official `encoding/encoding_dsv4.py` plus the OpenAI mapping |
| `dspark.py` | DSpark speculative-decoding controller (framework agnostic, see below) |
| `ds41_parse.py` | reply parser: thinking / answer split, DSML tool calls, streaming |
| `ds41_engine.py` | engine contract, `load_model` hook, `StubEngine` |
| `ds41_model.py` | the default loader (`ds41_model:load_model`): the port model (`exllamav3/dsv41`) as an engine, DSpark behind `--num-draft` |
| `test_ds41_*.py` | pytest suites (CPU, no weights) |
| `fetch_reference.py` | downloads the small reference files (no weights) |

## Start

```bash
python3 tools/ds41/serve.py --model <model-folder> --loader <module:function> --port 8000 -c 131072
# without a GPU, canned replies:
python3 tools/ds41/serve.py --model <model-folder> --loader stub --engine-opt replies="Hello from the stub"
```

`<model-folder>` needs `config.json` and `tokenizer.json` (and `generation_config.json` when the pack has one).

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `~/models/deepseek-v4.1-flash-1017` | model folder |
| `--model-id` | `deepseek-v4.1-flash` | id in `/v1/models` and in replies; a request naming another model gets 400 |
| `--host`, `--port` | `127.0.0.1`, `8000` | listen address |
| `-c`, `--max-ctx` | `131072` | context window in tokens |
| `--loader` | `ds41_model:load_model` | `module:function` that builds the engine; `stub` = canned replies |
| `--engine-opt KEY=VALUE` | none | extra keyword argument for the loader (JSON value or plain string), repeatable |
| `--num-draft` | `0` (off) | DSpark speculative decoding: up to N draft tokens per round, N up to 10 (above the block size of 5 the drafter chains a second pass; the extra slots are a lower-quality guess, the output stays exact). Given to the loader as `num_draft` only when above 0 |
| `--draft-auto` | off | Choose the draft length of each round from the running acceptance (1..4, at most `--num-draft`, which defaults to 4 here). Loader option `draft_auto=true`. Output unchanged |
| `--draft-conf` | none | Stop drafting at the first slot whose DSpark confidence probability is below P (0 < P < 1). Loader option `draft_conf_threshold`. Output unchanged |
| `--default-temperature` | pack file, else `1.0` | used when a request omits `temperature` |
| `--default-top-p` | pack file, else `1.0` | used when a request omits `top_p` |
| `--default-max-tokens` | all free context | reply budget (thinking included) when a request omits it |
| `--default-reasoning-effort` | `high` | `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or an integer 1..100 |
| `--default-thinking` | `on` | `on` / `off`, when a request does not choose |

Request log, SSE keep-alive and start-up progress are the shared code described in
[doc/logging.md](../../doc/logging.md) and [doc/env_vars.md](../../doc/env_vars.md).

## Pointing it at a converted pack

```bash
python3 tools/ds41/serve.py --model <pack-folder> -c 131072 --port 8000 [--num-draft 5]
```

A folder with `layers.0.safetensors` (or `quantization_config.quant_method = exl3`) is a pack: it is loaded with
`exllamav3.dsv41.pack.load` into the serving model, on the GPU in fp16 when one is visible (override with
`--engine-opt device=cpu --engine-opt dtype=float32`). Other pack options: `layers=N` (first N trunk layers, smoke
test), `moe=loop`, `prefill_chunk`, `engram_dir`, `prefix_cache=false`. Run `validate_pack.py` on the folder first.

* Warm-up: before the first request the server prefills about 4.6K tokens and decodes 16 (kernel JIT, allocator growth).
  `/health` reports the `warm-up` stage. `EXL3_SERVE_WARMUP=0` skips it.
* Prefix cache (plain decode): the state after the last prompt prefill is kept; a request whose prompt starts with
  that prompt prefills only the tail and reports it in `timings.cache_n` / `usage.prompt_tokens_details.cached_tokens`.
* Speculative hook: with `--num-draft N` the engine calls `spec_hook(model, max_seq)` (`--engine-opt
  spec_hook=module:function`, default `exllamav3.dsv41.spec:PortSpec`). It returns an object with `.target` and
  `.drafter` (the `dspark.Target` / `dspark.Drafter` protocols), or None: then plain decode runs. Without
  `--num-draft` the hook is never called and the DSpark stages are not loaded.
* Tests on the synthetic pack: `test_ds41_pack_serve.py` (HTTP, streaming, tools, thinking, timings, command line).

## Image input

A pack with `vision_config` in `config.json` and the 266 tower tensors (`vision.*`, `aligner.*`, `image_start`,
`image_newline`, `image_end`; BF16, in `global.safetensors` or listed in `model.safetensors.index.json`) takes images in
chat requests (`image_url`, `input_image`, Anthropic `image` parts; `/health` shows `"vision": true`). Options:
`--engine-opt vision=auto|true|false` (`auto`: on when the pack is complete, `true`: fail at start when it is not),
`vision_device` (default: the model device), `vision_dtype` (`bfloat16` default, `float16`), `vision_cache` (span rows kept
per recent image, default 8).

* The tower (about 1.1 GB in bf16) is read on the first image, not at start. It runs in plain torch, query blocks of 2048.
* Layout is the official one: `<start> + (image x w + newline) x h + <end>` per image, every position carries id 129264.
  Image rows replace the embeddings; image rows route with `ffn.gate.bias_vl` and carry no engram; MTP/DSpark stages see
  no image.
* The prefix cache keys image positions by a hash of the preprocessed pixels: two images of the same size never share
  cached state. A stored prompt is reused when the next prompt strictly extends it (image tokens included), as for text.
* Sources: `data:` URLs, base64, and `http(s)` URLs (fetched by the server, 30 s, 32 MiB cap; there is no address filter,
  so do not expose the server to untrusted clients). Local file paths are refused. Source images above 100 MP are refused.
* `tools/ds41/vision_gpu_check.py` checks the tower on real weights against the official one; `tools/ds41/smoke_vision.py`
  runs a served end-to-end check.

## Sampling defaults

For `temperature` and `top_p`, in order: the request, a flag, the `generation_config.json` of the model folder,
the model card values (`temperature 1.0`, `top_p 1.0`). The Hugging Face repo of the model ships no
`generation_config.json`, so without one in the pack the model card values apply. A request value of `null`
counts as omitted.

## Endpoints

* `GET /health`, `GET /v1/models`, `GET /metrics` — as on the GLM server.
* `POST /v1/chat/completions` — streaming and not. `reasoning_content`, `tool_calls`, `usage`, `timings`
  (`cache_n`, `prompt_n`, ... as llama.cpp), `cache_prompt`, `cache_salt`, `stop`, `max_tokens` /
  `max_completion_tokens` (must be integers >= 1), `n` must be 1.
* `POST /v1/completions` — verbatim prompt (text or token ids), no template, no BOS added.
* `POST /completion` — minimal llama.cpp form of the same.
* `POST /tokenize` — `{prompt}` or `{messages, tools, ...}`. With `messages` the chat prompt is rendered, BOS
  included, and `count` equals the `prompt_tokens` of the same chat request.
* `POST /apply-template` — returns the rendered prompt, generates nothing.

Finish reasons: `stop`, `length` (budget used, or a tool call cut by the budget), `tool_calls`.

## Prompt format

`ds41_chat.render_chat` ports the official encoder and is checked against it, string for string and token id for
token id (`test_ds41_chat.py`). Turn markers are `<｜User｜>`, `<｜Assistant｜>`, `<｜System｜>`; the generation
prompt ends in `<think>` (thinking) or `</think>` (chat). One BOS opens the prompt.

Handled like the reference: system, user, assistant (with `reasoning_content`), multi-turn, tools (system block
`## Tools`), tool calls (DSML), tool results (grouped into the next user turn, in call order), thinking
dropped from earlier turns (`drop_thinking`, switched off automatically when tools are present).

OpenAI layer on top:

* `developer` role is `system`. `content: null` is empty. Text-part lists are joined. Image parts get 400.
* Tools given but no system message: an empty system message is inserted to carry them.
* `tools` accepts the nested (`function: {...}`) and the flat form. `tool_choice: "none"` removes the tools from
  the prompt and turns off call parsing.
* Assistant `tool_calls[].function.arguments` may be a JSON string or an object. A `namespace` field, or a
  `ns::name` function name, renders the `ns::name` form.
* A prompt that ends on an assistant turn, or has only a system message, gets 400.

### Thinking and effort mapping

Where each setting is read, first hit wins: `chat_template_kwargs`, then the top-level body field, then the
server default.

| Request | Prompt |
|---|---|
| nothing | thinking, effort 75 (`--default-*` change this) |
| `reasoning_effort: "none"` | chat mode (no thinking), no effort line |
| `reasoning_effort: "minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` / `"max"` | thinking with effort 25 / 50 / 62 / 75 / 100 / 100 |
| `reasoning_effort: 1..100` | thinking with that effort |
| `thinking: false`, `{"type": "disabled"}`, `enable_thinking: false`, `thinking_mode: "chat"` | chat mode |
| `thinking: true`, `{"type": "enabled"}`, `enable_thinking: true`, `thinking_mode: "thinking"` | thinking |
| `reasoning: {"enabled": false}` or `{"effort": "none"}` / `{"effort": level}` (OpenRouter style, sent by Hermes) | chat mode / the level; a level the server does not know is ignored; `chat_template_kwargs` and top-level fields win over it |
| `drop_thinking: false` | keep earlier reasoning blocks in the prompt |

An explicit thinking switch beats `reasoning_effort: "none"`. `low`, `high` and `max` are the official mapping
(50, 75, 100); `minimal`, `medium` and `xhigh` are this server's choice. In thinking mode the effort line
(`Reasoning Effort: N (range 1-100, ...)`) goes at the top of the first system message. Bad values get 400.

## Reply parsing

`ds41_parse.StreamParser` reads the decoded text, text-only and incremental:

* thinking mode: text up to `</think>` is `reasoning_content`, the rest is `content`; chat mode: all `content`;
* a `<｜DSML｜ calls>` block becomes `tool_calls` (`id`, `function.name`, JSON `arguments`, `namespace` when the
  name is `ns::name`);
* streaming sends `tool_calls` deltas: the first has `index`, `id`, `type`, `name` and empty `arguments`, then
  `arguments` pieces that add up, byte for byte, to the official parser's JSON; markup is never sent as content;
* only a possible start of a marker is held back, so text streams character by character;
* a reply cut inside a call closes the JSON, and the finish reason is `length`;
* malformed markup falls back to plain content; the parser does not raise.

## Engine contract

`ds41_engine.Engine` lists what `serve.py` calls: `ctx`, `sessions`, `vocab_size`, `count_tokens`, `prompt_ids`,
`encode_ids`, `generate`, and optionally `reset_prefix_cache`. Rules for a real engine:

* `generate` yields decoded text **with special tokens kept** (`</think>`, `｜DSML｜`). A decoder that skips
  special tokens breaks reasoning and tool calls.
* `prompt_ids` tokenizes a rendered prompt with special tokens parsed and no extra BOS (the prompt has one).
* `stats` / `last_stats` get `cached_tokens`, `new_tokens`, `time_prefill`, `time_generate` at the end.
* `reset_prefix_cache` is called when `cache_salt` changes.
* With `--num-draft N` the loader gets `num_draft=N` and the engine exposes `num_draft` (the reply budget keeps room
  for the draft window). It adds `accepted_draft_tokens`, `rejected_draft_tokens`, `draft_steps` to its job stats
  (`DSparkStats.as_engine_stats()`); the server then adds those and `accepted_per_step` to `timings` and counts the
  rounds in `/metrics`. Without drafting these fields are absent. The cost fields of `as_engine_stats()`
  (`draft_len_mean`, `verify_positions`, `conf_dropped_tokens`, `conf_cut_rounds`, `tuner_k`) go to `timings` too when the
  engine sends them (`tuner_k` as `draft_tuner_k`). `conf_dropped_tokens` is the number of verify positions the
  confidence stop saved (a position is cheap next to a verify call: judge the gain with calls + c * extra positions). A loader that takes `draft_auto` builds a `dspark.DraftTuner` per request (or shares one across requests so the acceptance history carries over); one that takes
  `draft_conf_threshold` passes it as `conf_threshold` to `DSparkDecoder`.
* Sampling: `temperature` and `top_p` of the request reach the engine; build the sampler as
  `dspark.GumbelSampler(temperature, top_p=top_p)`. The server does not read `top_k` or `min_p` from requests (the
  sampler supports both).

`StubEngine` implements it over canned replies (`--loader stub`, or `StubEngine(replies)` in code).

## DSpark speculative decoding

`dspark.py` holds the draft/verify loop; the engine supplies two small Protocol objects: a `Target` (`prefill`, `verify`
a block of tokens returning logits and the hidden states of the target layers, `truncate` the cache) and a `Drafter`
(the three DSpark stages, `draft`). It has no torch import except in `GumbelSampler`.

The official `inference/` code has the DSpark forward pass but **no speculative loop**: its README says generation is
plain autoregressive and nothing calls `Transformer.forward_spec`. So the acceptance rule and the rollback are ours,
built on what `forward_spec` defines:

1. The stages take the token the target just chose followed by `dspark_block_size - 1` noise tokens
   (`dspark_noise_token_id`) and return `block_size` draft tokens, chained through the Markov head (rank
   `dspark_markov_rank`). Their attention reads a window of main-model context made from the concatenated
   hidden states of `dspark_target_layer_ids`, one window slot per position.
2. The target verifies `[anchor, d1..dK]` in one block. Greedy: accept `d_k` while it equals the argmax; the next
   argmax is the correction (or a bonus token when all matched). Output equals plain greedy, token for token.
3. Sampling: the target token of each position is drawn in order with the official Gumbel-max `sample` rule and a
   draft is accepted when it equals the draw (standard speculative sampling for a one-hot draft; same output law,
   and the same tokens as the plain loop under the same seed). Drafts that carry their laws `q` use `min(1, p/q)` with
   the `max(0, p - q)` residual instead. `GumbelSampler(top_p=, top_k=, min_p=)` truncates and renormalises the law
   before the draw (top_k, then top_p, then min_p); the draw count per emitted token is unchanged, so the seeded
   equality with the plain loop holds for truncated sampling too.
3b. The confidence head returns one raw score per slot. The official code does not say what it means; we read it as a
   logit of "this draft is accepted". With `conf_threshold=P`, drafts from the first slot with sigmoid(score) < P on are
   not verified. The score of slot i does not depend on token i, only on the tokens before it, so the cut is made
   before the draw it affects and the output (and, with draft laws, the p/q rule) stays exact. The saving is verify
   positions (`conf_dropped_tokens`); the price is the accepted tokens the cut skips. `DraftTuner` picks the length
   (1..4) of each round from the decayed per-position acceptance, maximising expected tokens per round over a cost of
   1 + 0.1 k; it reads past rounds only. Both only change cost.
4. Rollback: `truncate(n)` must leave the target cache as if only positions `0..n-1` had been run. A ring-buffer
   sliding window cannot be cut by length (a rejected position overwrites a slot an older position still needs), so
   an engine must snapshot or keep history for the verify block. The drafter needs no rollback: it is fed only
   accepted positions, and must be fed every one of them, in order.

The official layer code runs a full prefill or a single token at a later position, so the test adapter
(`dspark_refkit.py`) verifies a block as single-token steps and rolls back by restoring a snapshot and replaying.
A real engine does a native block forward.

Stats (`DSparkStats`): rounds, proposed and accepted tokens, `accepted_per_step`, `tokens_per_step`, a histogram of
draft lengths and one of accepted counts. The server reports `accepted_draft_tokens`, `rejected_draft_tokens`,
`draft_steps` and `accepted_per_step` in `timings`.

Off by default (`--num-draft 0`).

### On the port model

`ds41_model.load_model` (the default `--loader`) runs `exllamav3/dsv41/model.py::DSV41Model` from an official
safetensors folder (or one V4.1 GGUF) and implements the two Protocols in `exllamav3/dsv41/spec.py`:

* `Target.verify`: one `DSV41Model.forward_block` over `[anchor, d1..dK]` (K <= 5), logits and target-layer hidden for
  every position. Each query sees what a decode step at its position sees: the sliding-window ring as it was before
  the block plus the block's own keys, the compressed groups completed up to its own position, its own indexer top-k.
* `Target.truncate`: `DSV41State.rollback(keep)` undoes the rejected tail from the block's undo log. What a block
  overwrites that older positions still need is the window ring slots (`p % 128`) and the open group of each
  compressor (slot `p % ratio`); both are logged (old rows only) and put back. Compressed pools, indexer keys and the
  engram token history are positional and rewritten before they are read again; hyper-connection mixes never leave a
  forward, so there is nothing else to restore. No recomputation.
* `Drafter.draft`: the accepted rows seed the window rings of the three stages (`DSparkHeads.feed`, one `wkv` per
  stage), then one `forward_spec` drafts the block. The drafter is only fed accepted positions: no rollback.

```bash
python3 tools/ds41/serve.py --model <folder> --num-draft 5 --engine-opt device=cuda:0 --engine-opt dtype=bfloat16
```

Loader options (`--engine-opt`): `device`, `dtype`, `engram_dir`, `cache_experts`, `threads`, `seed`. One request at a
time, no prefix cache (every request prefills; `cached_tokens` is 0). `num_draft` above the block size, or on a model
without `mtp.*` tensors, is refused at load.

Numerics: a `[k+1, d]` GEMM and `k+1` GEMVs differ in the last bits and the graph's fp8/fp4 fake-quantisation steps
(kv, attention output, indexer, compressed latents) round to coarse grids, so block verify and decode differ by
about 1e-6 (reduced model) to 1e-2 (hidden 5120) in the logits. Greedy output with drafting therefore equals the plain
loop except where the two top logits are closer than that noise; `tests/dsv41/test_dsv41_spec.py` quantifies it.

## Tests

```bash
python3 -m pytest tools/ds41 -q
```

CPU only, no weights (`test_ds41_model.py` builds a tiny random checkpoint with `tests/dsv41/tiny_ckpt.py`). Golden tests compare against the vendored official encoder
(`testdata/reference/encoding.py`, MIT, with the official fixtures). Token-id equality needs the model's
`tokenizer.json`: set `DS41_REF_DIR` to a folder holding it, or run `python3 tools/ds41/fetch_reference.py <dir>`
(downloads only config, tokenizer and encoding files, never weights). Without it those tests are skipped and
reported as skipped.

`test_ds41_dspark_sampling.py` covers top_p / top_k / min_p, the confidence stop and the tuner: seeded equality with
the plain loop on 300 seeds, chi-square tests of the end-to-end law, and checks on the official model code.

`test_ds41_dspark.py` runs the controller on a bigram table and on a tiny random-weight model built from the
unmodified official `model.py` (vendored in `testdata/reference/inference`, MIT; CUDA kernels replaced by torch
equivalents in `dspark_refkit.py`). It checks greedy and seeded-sampling output against the official `generate.py`,
cache rollback against a fresh prefill, EOS inside a block and `max_tokens` mid-block.
