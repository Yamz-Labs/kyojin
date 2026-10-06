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
| `--cache-bits` | 0 | `0` = fp16 K/V cache pages (default); `8` = packed int8 pages: 2.6 GiB less at 256K (peak 83.6 vs 86.2 GiB), but output is not row-invariant when several rows decode together |
| `--ndt` | 3 | max draft tokens per speculative round |
| `--draft-policy` | `mix` | `mix` shipped rule, `mtp` fixed MTP chain, `off` plain decode (no drafter loaded) |
| `--no-vision` | off | do not load the vision tower (less memory, image input answers 400) |
| `--default-temperature/-top-p/-top-k/-min-p` | from `generation_config.json` (1.0 / 0.95 / 20) | used when a request omits them |
| `--default-reasoning-effort` | template default (`xhigh`) | `low`, `medium` or `xhigh` |
| `--no-thinking` | off | thinking off unless a request turns it on |
| `--default-max-tokens` | 32768 | when a request omits `max_tokens` (always clipped to the free context) |
| `--slot-save-path` | `~/cache/llama-slots` | directory for slot files |
| `--sessions` | 1 | requests decoded together; see below |

Environment: the server sets the measured Qwen configuration with `setdefault` before the engine loads (the list is
`SERVE_ENV` in `serve.py`; the start-up log prints the effective `EXL3_*` / `MPW*` environment). It covers decode and
verify (`EXL3_MOE_FUSED`, `EXL3_MOE_VALU`, `EXL3_VERIFY_*`, `EXL3_GEMV_R_DEC1`, `EXL3_MTP_FUSE_CATCHUP=0`), prefill
(`EXL3_PLE_HIP`, `EXL3_DQ_HIP`, `EXL3_GR_HIP`, `EXL3_GDN_FUSE`, `EXL3_PF_SKIP`, `EXL3_PREFILL_CHUNK=4096`,
`EXL3_PF_DEFER=0` (the MTP draft prefill runs chunk by chunk with the target prefill; `1` defers it behind the first token), `EXL3_PF_GR_FUSE`, `EXL3_MPW_GLUE`), `EXL3_MOE_CFG=2`, `EXL3_HIP_PREFILL_MIN_ROWS=2`, `EXL3_QSA_ROWINV`, `EXL3_ROLL_CKPT` and the int8 mixer with group
scales (`EXL3_GR_GS=1`; `hc_gs_sidecar.safetensors` in the model directory is picked up when present). Your own
environment wins. The engine modules read several of these when they are imported, so the server refuses to start if
`exllamav3` was imported before they were set.

## API

- `POST /v1/chat/completions`: stream (SSE) and non-stream, `tools` and `tool_calls`, `tool` role messages, images as
  `image_url` parts (`data:` URI or http(s) URL), `stop`, `max_tokens` / `max_completion_tokens`, `temperature`,
  `top_p`, `top_k`, `min_p`, `seed`. The model's own chat template renders the prompt (identical to the HF `apply_chat_template`).
- Thinking: the template default is thinking ON at effort `xhigh`; the answer then starts inside a reasoning block.
  The server splits it into `reasoning_content` and `content`. Turn it off per request with `"enable_thinking": false`
  (or `"chat_template_kwargs": {"enable_thinking": false}` or `"reasoning_effort": "none"`). Effort: `low`, `medium`,
  `xhigh` (`high` maps to `xhigh`, `minimal` to `low`).
  Give thinking room: with a small `max_tokens` the budget can end inside the thought, so `content` is empty and
  `finish_reason` is `length`.
- `finish_reason`: `stop`, `length` (cut by `max_tokens`), `tool_calls`.
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

## Start-up progress on `/health`

The port answers from the first second, while the model is still loading. `GET /health` returns `200 {"status":"ok","source":"kyojin"}`
(plus the fields listed above) when the server is ready, and while it loads:

```json
503 {"status": "loading", "source": "kyojin", "message": "Target weights, 40 % (stage 1 of 4)", "progress": 0.31, "stage": "target weights", "stage_index": 1, "stage_count": 5,
     "stage_progress": 0.4, "elapsed_s": 52.1, "eta_s": null, "progress_basis": "stages"}
```

Every `/health` reply carries `"source": "kyojin"` (a hint for parsers). While `status` is `loading`, `message` is one plain line for a UI, built from the other fields: stage, percent, "stage x of y", and "about N s left" when an ETA exists.

Stages here: target weights, drafter weights, vision tower, engine setup, warm-up (the vision stage is absent with `--no-vision`, the drafter with `--draft-policy off`). `stage_progress` is the share of modules loaded (or kernel keys tuned) and is `null` when a
stage cannot count its work. `progress` counts finished stages plus that share, with equal weights
(`progress_basis: "stages"`); once one start has completed, its stage durations are kept in
`~/.cache/kyojin/startup-qwen.json` (`KYOJIN_HEALTH_DIR` changes the folder) and later starts weigh the stages by
them (`"history"`) and report `eta_s`. Without a history `eta_s` is `null`: no estimate is made up. A failed load
answers `500 {"status":"error","message":...}` for a few seconds before the process exits. Every other path answers
`503 {"error":{"message":"Loading model",...}}` during the load, like llama.cpp. Nothing changes after READY, and
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
reserves the pages for its prompt plus `max_tokens` when it starts, so one that does not fit next to the running ones
waits. With N > 1, a request with `speculative: false` or `cache_prompt: false` answers 400 (both rebuild the Generator), and slot
save/restore/erase wait until the running requests finish.

Measured with `--ndt 3`, `-c 262144`, temperature 0, two short prompts sent together (four for `--sessions 4`):

| `--sessions` | `--cache-bits` | greedy output equal to a solo run | second stream's first token | total tok/s |
|---|---|---|---|---|
| 1 | 8 | yes | 6.55 s (waits for the first) | 44.4 |
| 2 | 8 | no (1 of 2 differs) | 1.16 s | 47.3 |
| 2 | 0 | yes | 1.17 s | 49.6 |
| 4 | 8 | no (2 of 4 differ) | 2.19 s | 51.4 |

With N > 1, greedy output is not guaranteed equal to a solo run, for either cache width: the decode attention kernel
chooses how it splits the cache from the batch it runs with, so the last bits of the sums, and now and then a near-tie
token, can change. What was measured: the 16-bit cache (`--cache-bits 0`) stayed equal on two short prompts at 2 sessions;
the 8-bit cache did not (rows above). Two short prompts show nothing more than that.

The server prints this warning at start-up whenever N > 1, and a second one past 8 verify rows (`N x (ndt + 1)`).

## Tests

    python -m pytest tools/test_startup_health.py tools/qwen/test_serve.py tools/qwen/test_guard.py   # CPU, fake engine, no GPU (template test needs the pack)
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
