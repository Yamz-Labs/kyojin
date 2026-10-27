# qserve: OpenAI-compatible server for Qwen3.8-Flash EXL3 packs

`tools/qwen/serve.py` serves a Qwen3.8-Flash EXL3 pack (for example `Qwen3.8-Flash-Yamz`) over HTTP. It loads the target
model, the MTP drafter and the vision tower once, and keeps one Generator. Speculative decoding (MTP plus n-gram lookup,
the shipped "mix" rule) is on by default and lossless: greedy output equals plain greedy output.

## Run

Verified command (lab box, scripts/env.sh sets the ROCm runtime, `EXL3_MOE_CFG=2`, `EXL3_HIP_PREFILL_MIN_ROWS=2`;
the job ran from the repo root with the tree on `PYTHONPATH`):

    source $EXL3_ROOT
    export EXL3_ROOT=$PWD EXL3_REPO=$PWD PYTHONPATH=$PWD
    python -u tools/qwen/serve.py --model ~/models/qwen38-yamz-v1 --port 8000 -c 65536

Public tree: `source tools/strix_halo/env.sh` instead, then `export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2` (the two
values the verified runs had; the public env.sh does not set them; this variant was not run). Run from the repo root.
The server prints its effective `EXL3_*` / `MPW*` environment at start-up.

The server prints a progress line every 15 s while it loads (target, drafter, vision, warm-up). The port opens only when
everything is ready, and the last line says so:

    qserve: READY on http://127.0.0.1:8000  model=Qwen3.8-Flash-Yamz ctx=65536 sessions=1 speculative=True vision=True (start-up N s)

Quick test:

    curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "model": "Qwen3.8-Flash-Yamz", "messages": [{"role": "user", "content": "Hello"}],
      "enable_thinking": false, "max_tokens": 100}'

## Flags

| flag | default | meaning |
|---|---|---|
| `--model` | `~/models/qwen38-yamz-v1` | pack directory (needs `chat_template.jinja`, `generation_config.json`) |
| `--model-id` | `Qwen3.8-Flash-Yamz` | id in `/v1/models`; requests that name another model get 400 |
| `--host`, `--port` | `127.0.0.1`, 8000 | listen address |
| `-c`, `--ctx` | 65536 | KV cache size in tokens (prompt + answer must fit) |
| `--cache-bits` | 0 | `0` = fp16 K/V cache pages (default); `8` = packed int8 pages: 2.6 GiB less at 256K (peak 83.6 vs 86.2 GiB), but greedy output differs from the 16-bit cache; with `EXL3_ROW_INV_GROUPS=1` four parallel sessions matched solo runs, 16 of 16 (see Concurrent sessions) |
| `--ndt` | 7 | max draft tokens per speculative round (the rule drafts up to 3 on most rounds, deeper only on confident chains) |
| `--draft-policy` | `mix` | `mix` shipped rule, `mtp` fixed MTP chain, `off` plain decode (no drafter loaded) |
| `--no-vision` | off | do not load the vision tower (less memory, image input answers 400) |
| `--default-temperature/-top-p/-top-k/-min-p` | from `generation_config.json` (1.0 / 0.95 / 20) | used when a request omits them |
| `--default-reasoning-effort` | `medium` | `low`, `medium` or `xhigh` |
| `--chat-template` | `tools/qwen/chat_template.jinja` | the model's template with a `medium` effort instruction added (the model's own leaves medium empty); `<model>/chat_template.jinja` is the unchanged one |
| `--no-thinking` | off | thinking off unless a request turns it on |
| `--default-max-tokens` | free context | reply budget when a request omits `max_tokens`; unset, a reply may use all the context left |
| `--slot-save-path` | `~/cache/llama-slots` | directory for slot files |
| `--sessions` | 1 | requests decoded together; see below |

Environment: the server sets the measured Qwen configuration with `setdefault` before the engine loads (the list is
`SERVE_ENV` in `serve.py`; the start-up log prints the effective `EXL3_*` / `MPW*` environment). It covers decode and
verify (`EXL3_MOE_FUSED`, `EXL3_MOE_VALU`, `EXL3_VERIFY_*`, `EXL3_GEMV_R_DEC1`, `EXL3_MTP_FUSE_CATCHUP=0`), prefill
(`EXL3_PLE_HIP`, `EXL3_DQ_HIP`, `EXL3_GR_HIP`, `EXL3_GDN_FUSE`, `EXL3_PF_SKIP`, `EXL3_PREFILL_CHUNK=4096`,
`EXL3_PF_DEFER=0` (the MTP draft prefill runs chunk by chunk with the target prefill; `1` defers it behind the first token), `EXL3_PF_KV_ONLY_ATTN=0` (the draft prefill computes only K/V and indexer planes; `0` runs the full block again, same cache contents), `EXL3_PF_GR_FUSE`, `EXL3_MPW_GLUE`), `EXL3_MOE_CFG=2`, `EXL3_HIP_PREFILL_MIN_ROWS=2`, `EXL3_QSA_ROWINV`, `EXL3_ROLL_CKPT` and the int8 mixer with group
scales (`EXL3_GR_GS=1`; `hc_gs_sidecar.safetensors` in the model directory is picked up when present). Your own
environment wins. The engine modules read several of these when they are imported, so the server refuses to start if
`exllamav3` was imported before they were set.

## Two-stage draft head

If `draft2s.st` is in the model directory and its header matches `config.json` (vocabulary size and hidden width), qserve
uses it as the two-stage draft head and logs it at start-up. No flag needed. Output does not change (every draft is
verified); sampled jobs keep the full head. `EXL3_DRAFT2S=<path>` picks another pack; `EXL3_DRAFT2S=off` turns it off.
The pack is tied to one set of weights: download it from the same model repo.

## API

- `POST /v1/chat/completions`: stream (SSE) and non-stream, `tools` and `tool_calls`, `tool` role messages, images as
  `image_url` parts (`data:` URI or http(s) URL), `stop`, `max_tokens` / `max_completion_tokens`, `temperature`,
  `top_p`, `top_k`, `min_p`, `seed`. The chat template renders the prompt (with the model's own file, identical to the HF `apply_chat_template`).
- Thinking: ON by default, at effort `medium`; the answer then starts inside a reasoning block.
  The server splits it into `reasoning_content` and `content`. Turn it off per request with `"enable_thinking": false`
  (or `"chat_template_kwargs": {"enable_thinking": false}` or `"reasoning_effort": "none"`). Effort: `low`, `medium`,
  `xhigh` (`high` maps to `xhigh`, `minimal` to `low`).
  Give thinking room: with a small `max_tokens` the budget can end inside the thought, so `content` is empty and
  `finish_reason` is `length`.
- `finish_reason`: `stop`, `length` (cut by `max_tokens`), `tool_calls`.
  The loop guard can end an answer that degenerated into a loop. That turn reports `finish_reason` `length`,
  `choices[0].stop_reason` `loop_guard` (chat and completions, stream and JSON) and the detail in `timings.loop_guard`
  (`phase`, `kind`, `at`, `cut`). A natural end never carries `stop_reason`. Valid repetitive output (dense code, tables,
  JSON records) is not cut: in an answer the guard needs most of the tail to replay itself, or a tail that compresses to
  almost nothing. Repetitive data (a grid of zeros, identical JSON lines, a zero-padded hex dump) is cut only after 16384 tokens of
  repetition. `loop_guard: false` in the request body turns the guard off.
- Tool calls: the model writes `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`.
  They come back as OpenAI `tool_calls`; argument types follow the tool's JSON schema. In a stream, tool calls arrive
  as one delta after the text. Text after a closed call (or between two calls) is returned as `content`.
- `usage` (with `prompt_tokens_details.cached_tokens`) and `timings` (`cache_n`, `prompt_n` = tokens prefilled this turn,
  `prompt_ms`, `predicted_n`, `predicted_ms`, `predicted_per_second`, draft accept counts) on every answer; in a stream
  they are on the last chunk.
- Errors are JSON `{"error": {"message", "type", "code"}}`: 400 for bad requests (including prompt longer than the
  context), 500 for engine errors.
- `GET /v1/models`, `GET /health` (status, speculation state, vision), `POST /apply-template` (renders the prompt,
  generates nothing), `GET /slots`, `POST /slots/0?action=save|restore|erase` (llama.cpp slot files),
  `POST /completion` (llama.cpp style) and `POST /v1/completions` (OpenAI style, stream and non-stream): raw prompt, no
  template, one prompt, one choice (`n`, `best_of`, `echo` answer 400). The slot routes are CPU-tested only.
- Debug extensions (all optional): `"speculative": false` decodes plain for that request, `"cache_prompt": false`
  forgets the prompt cache first (cold run), `"return_token_ids": true` adds the generated ids. Switching
  speculative mode or `cache_prompt: false` rebuilds the Generator and so drops the prompt cache (400 with `--sessions` > 1).

- vLLM-style extensions (also used by the RigMark benchmark): `POST /tokenize` (`{"prompt"|"messages", "add_special_tokens"}` ->
  `{"tokens", "count", "max_model_len"}`); `/v1/completions` takes `prompt` as a list of token ids (`[1, 2, 3]` or `[[1, 2, 3]]`):
  the ids go to the model as sent (no template, no BOS), `usage.prompt_tokens` is their count, an id outside the vocabulary
  answers 400, and `ignore_eos: true` keeps generating through the end-of-sequence token. `usage.prompt_tokens_details.cached_tokens`
  is on every `/v1/completions` and chat answer. `cache_salt` (chat and completions, non-empty string): requests with different
  salts never share prompt cache. The server keeps one salt namespace at a time (requests without a salt are one namespace):
  when it changes, running requests finish, the prompt cache is dropped (a fresh Generator, as `cache_prompt: false` does), and the
  request starts. The same salt, or no salt at all, never drops anything, so only the first request after a change pays the rebuild.

## Diagnostics: the request log

The log and the stream keep-alive are shared code: switches, format, every event and how to read the
output accounting are documented once, in [doc/logging.md](../../doc/logging.md). This server writes the
same events as GLM except for `serve_start` and `img_cache` (GLM's image cache and its disk tier)
and `sse_ping` (this server's pinger writes no log line); `tools/test_serve_logging.py` checks the shared
part against all three servers at once. Turn it on with `KYOJIN_REQLOG_DIR=<directory>` (or `1`) and hold a silent
stream open with `KYOJIN_SSE_PING_S=<seconds>`; a run writes its own log file (`requests-<run>.log`, newest 20 kept), and both switches are in [doc/env_vars.md](../../doc/env_vars.md).

## Start-up progress on `/health`

The port answers from the first second, while the model is still loading. `GET /health` returns `200 {"status":"ok","source":"kyojin"}`
(plus the fields listed above) when the server is ready, and while it loads:

```json
503 {"status": "loading", "source": "kyojin", "message": "Target weights, 40 % (stage 1 of 5)", "progress": 0.31, "stage": "target weights", "stage_index": 1, "stage_count": 5,
     "stage_progress": 0.4, "elapsed_s": 52.1, "eta_s": null, "stage_eta_s": 78.0, "progress_basis": "stages"}
```

Every `/health` reply carries `"source": "kyojin"` (a hint for parsers). While `status` is `loading`, `message` is one plain line for a UI, built from the other fields: stage, percent, "stage x of y", and "about N s left" when an ETA exists.

Stages here: target weights, drafter weights, vision tower, engine setup, warm-up (the vision stage is absent with `--no-vision`, the drafter with `--draft-policy off`). `stage_progress` is the share of modules loaded (or kernel keys tuned) and is `null` when a
stage cannot count its work. `progress` counts finished stages plus that share, with equal weights
(`progress_basis: "stages"`); once one start has completed, its stage durations are kept in
`~/.cache/kyojin/startup-qwen.json` (`KYOJIN_HEALTH_DIR` changes the folder) and later starts weigh the stages by
them (`"history"`) and report `eta_s`. Without a history `eta_s` is `null`: no estimate is made up. `stage_eta_s` is this run's own estimate for the current stage alone (from the share loaded so far); it is `null` until 2 % of the stage is done, and is there with or without a history. A failed load
answers `500 {"status":"error","message":...}` for a few seconds before the process exits. The real API endpoints answer
`503 {"error":{"message":"Loading model",...}}` during the load, like llama.cpp; an unknown path answers `404` and a wrong method `405`, in the same JSON error shape (also after READY). Nothing changes after READY, and
the `READY` line is the same. The helper is `tools/startup_health.py`, shared by the three servers.

llama-swap polls `checkEndpoint` (default `/health`, HTTP 200 = ready) until `healthCheckTimeout` runs out, so give it
room for a first start:

```yaml
healthCheckTimeout: 300      # seconds; a first start tunes kernels
models:
  "qwen":
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /health
    cmd: python tools/qwen/serve.py --port ${PORT}
```

## Prompt cache between turns

The Generator keeps KV pages (256 tokens) and GDN recurrent checkpoints in RAM (4 GiB, about 111 MiB each). A follow-up
turn re-prefills only the part after the last checkpoint: the tail of the previous prompt (under 256 tokens) plus the
new text. `timings.prompt_n` shows how many tokens were prefilled. By default requests are served one at a time, in order (see `--sessions` below).

## Concurrent sessions

`--sessions N` decodes up to N requests together (one recurrent slot each); more requests wait in order. A request
reserves the pages for its prompt plus its reply budget when it starts, so one that does not fit next to the running ones
waits. A client that sends no `max_tokens` (Hermes, most chat UIs) asks for a reply as long as the context: with N > 1 such a
request holds only its share of the context (`-c` / N, at least 2048 tokens of reply) and the generator requeues it when the
reply goes on: the pages are released, the request re-enters the queue at the front and continues as soon as pages are free
(the full pages are found by hash, only the tail is prefilled again). The other sessions are not blocked, and a request
running alone still reaches the end of the context. A reply that fits its share is untouched; `--sessions 1` is unchanged.
`--default-max-tokens` still caps the reply when you want a hard limit. With N > 1, a request with `speculative: false` or `cache_prompt: false` answers 400 (both rebuild the Generator), and slot
save/restore/erase wait until the running requests finish.

Measured with `--ndt 3`, `-c 262144`, temperature 0, two short prompts sent together (four for `--sessions 4`):

| `--sessions` | `--cache-bits` | greedy output equal to a solo run | second stream's first token | total tok/s |
|---|---|---|---|---|
| 1 | 8 | yes | 6.55 s (waits for the first) | 44.4 |
| 2 | 8 | no (1 of 2 differs) | 1.16 s | 47.3 |
| 2 | 0 | yes | 1.17 s | 49.6 |
| 4 | 8 | no (2 of 4 differ) | 2.19 s | 51.4 |
| 4 | 0 | no (2 of 4 differ) | not measured | 54.4 (reported, another machine) |

By default the answers of parallel requests can differ from solo runs (rows above). `EXL3_ROW_INV_GROUPS=1` (opt-in) makes the
decode step row-exact: the jobs of one step are packed into forwards of at most 8 token rows, and the attention layers of a
forward with several sequences run one sequence at a time, so each row gets the bits of the same row decoded alone. With
`--ndt 7`, four cold 8K prompts sent together at `--sessions 4` gave greedy answers equal to the same prompts sent one at a
time, 16 of 16, with the 16-bit cache and with `--cache-bits 8` (without it: 5 to 7 of 16). A step with one job runs as before, so
`--sessions 1` is unchanged. Cost: about 3 % more wall time on that four-prompt load, and 8 to 10 % less aggregate decode
throughput on short prompts (four parallel sessions then decode no faster than one).

Where the opt-in is not exact:
- A request whose context grows past about 2051 tokens while it shares a step with other requests can hit one different token:
  the step where a sequence crosses the sparse-attention threshold, if the draft width of the group differs from the width the
  request would have drafted alone (measured: 7 of 8 equal on 2030-token prompts). The 8K gate prompts start past the threshold
  and never cross it. Fixing it needs a per-job draft width at the crossing step.
- While a request with images in its prompt is active, every step it shares runs as one forward (no groups), so the other
  requests lose the guarantee. Vision was off in every measurement above.

The server prints a warning at start-up with N > 1 unless `EXL3_ROW_INV_GROUPS=1`.

## Tests

    python -m pytest tools/test_startup_health.py tools/test_serve_compat.py tools/qwen/test_serve.py tools/qwen/test_guard.py   # CPU, fake engine, no GPU (template test needs the pack)
    python tools/qwen/serve_check.py http://127.0.0.1:8000 out_dir      # live server: chat, thinking, tools, stream, image,
                                                                         # 3-turn cache, plain vs speculative, warm vs cold

## Measured on the real pack (qwen38-yamz-v1, 3.47 bpw, box iGPU, `-c 65536`, `MPW_KERN=1`)

- Start-up 52 s to READY (target, drafter, vision, warm-up).
- Served decode, greedy, 160 tokens, thinking off, 4 prompts per class, 2 interleaved passes (tok/s, plain / speculative):
  chat 27.3 / 42.2, prose 27.3 / 38.7, code 27.3 / 52.2, multi 27.3 / 42.6, copy 27.2 / 54.7.
  Speculative output equals plain greedy on all 20 prompts, both passes (token ids).
- Prompt cache, 3-turn chat with a 1.9K-token system prompt: prefilled 1909, 215, 338 tokens; time to first token 2.9 s, 0.5 s, 0.9 s.
- A cached second turn does not always give the same greedy tokens as a cold run of the same prompt (4 of 5 prompts
  diverge, first difference at token 19 to 83); plain decode shows the same, and two cold runs are identical. It comes with the cache restore, not with speculation (cause not isolated further,
  see `convcache1`).
