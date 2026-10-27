# glm-serve — GLM-5.3 EXL3 OpenAI-compatible server

`tools/glm/serve.py` is a single-flight aiohttp server around one resident EXL3 model
(GLM-5.3 `Glm5NextForConditionalGeneration` + its MTP head). It exposes the two
OpenAI endpoints a client needs and nothing else.

## Start

```bash
source tools/strix_halo/env.sh   # from the repository root; sets PYTHONPATH
export TMPDIR="$PWD/scratch/glm-serve"; mkdir -p "$TMPDIR"
export EXL3_MOE_UNION_V2=1 EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 \
       EXL3_BLOCK_GRAPH=1 EXL3_BLOCK_GRAPH_MLA=2
python tools/glm/serve.py \
    --model ~/models/glm53-exl3-td205 --port 18080
```

`tools/glm/serve_smoke.sh` does exactly this (honouring `GLM_SRV_ROOT`, `GLM_SRV_PORT`,
`GLM_SRV_MODEL`, default port **18080**), waits for `/v1/models`, then exercises a plain
call, a streamed call, a tool call, and two 400-token benchmark calls.

| flag | default | meaning |
|---|---|---|
| `--model` | `~/models/glm53-exl3-td205` (hardcoded `DEFAULT_MODEL`) | EXL3 pack dir; also supplies `chat_template.jinja` |
| `--model-id` | `glm-5.3-exl3` | id reported by `/v1/models` and echoed in every response |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8000` | port (smoke script uses 18080) |
| `--chat-template` | `tools/lanes/assets/glm53-template-medium.jinja` | Jinja template file. The default is the model's template with a `medium` effort level added; `<model>/chat_template.jinja` is the unchanged one |
| `--default-temperature` / `--default-top-p` | from `generation_config.json` (`1.0` / `0.95`) | used when a request omits the field (explicit null too). Greedy defaults loop on agent clients that send no sampling |
| `--default-reasoning-effort` | `medium` | template `reasoning_effort` when a request omits it: `low`, `medium`, `high` or `max` |
| `--max-history` | `1` | recurrent history slots; must equal draft tokens (1 for MTP). Each extra slot costs ~2.15 GiB |
| `-c`, `--max-ctx` | `131072` | context window in tokens, shared by all sessions |
| `--image-cache-mb` | `1024` | host memory for the decoded image embeddings, in MiB, one pool shared by all sessions; `0` = no byte limit |
| `--image-cache-items` | `32` | cap on how many pictures are held at once; `0` = no limit |
| `--image-cache-dir` | `~/.cache/kyojin/image-embeddings` | directory of the second-level picture cache; used only when `--image-cache-dir-max-gb` is set. A file is named after the pack and its preprocessing settings as well as the picture, so another model or other pixel settings never reach rows computed with the old ones |
| `--image-cache-dir-max-gb` | `0` | GiB cap on that directory; `0` = disabled. There is no unlimited setting |
| `--image-cache-dir-lock-s` | `300` | seconds a conversation keeps the room its files took before another may take it; `0` = no tenure |
| `--image-cache-format` | `fp32` | row storage: `fp32` as the vision merger emits them, `fp16` (half the bytes, bit-exact on this pack) or `q8` (8 bit with one fp16 scale per row, a quarter, 0.86 % error) |
| `--sessions` | `1` | requests decoded together through the one model (multi-user). `1` = one request at a time, others wait. With N > 1 a request without `max_tokens` holds only its share of the context (`-c` / N, at least 2048 tokens of reply) and is requeued when the reply goes on, so it does not block the other sessions |
| `--row-budget` | `5` when `--sessions` > 1 | caps the verify rows per round by lowering the draft depth when several sessions are busy; `0` = off |
| `--num-draft` | `2` | draft tokens per speculative round; `0` = plain decode, no drafter |
| `--exact-prefix-cache` | off | a request resumed from the prompt cache gives the same greedy text as its cold prefill. Costs about 5-12 % of cold prefill speed at 7-14K tokens |

Env vars the server sets for itself via `os.environ.setdefault` (so an external value
wins): `EXL3_CONV_GEMM=1`, `EXL3_DENSE_GEMM_TUNE=0`, `EXL3_MOE_UNION_V2=1`, `EXL3_MOE_CFG=2`, `EXL3_HIP_PREFILL_MIN_ROWS=2`,
`EXL3_BLOCK_GRAPH=1`, `EXL3_BLOCK_GRAPH_MLA=2`, plus the module-level
`Generator.MTP_FUSE_CATCHUP = 2`. There is no `--temperature`/seed handling beyond the
request body.

Copy lookup (`GLM_NGRAM`), sampled drafts (`EXL3_SPEC_SAMPLED`), the recurrent slot count (`EXL3_GLM_SLOTS`) and the
collector freeze (`EXL3_GC_FREEZE`) are described in [doc/env_vars.md](../../doc/env_vars.md).
The lookup code is `tools/qwen/spec_policy.py`, which the GLM server finds through its own file path, so `tools/qwen`
must ship next to `tools/glm`.

### MTP sidecar (higher draft acceptance)

The pack stores the MTP `eh_proj` at 2 bits. The pack also ships an unquantized copy,
`mtp_eh_proj.st` (BF16, 64 MiB), and the server loads it automatically: no flag, no variable.
It is a normal safetensors file with a `.st` extension, so no weight loader indexes it as
model weights. It only changes the draft; target token ids stay equal. Measured on the card
protocol (3 prompts, 400 tokens, greedy, Ryzen AI Max+ 395, MTP 2): 26.7 -> 31.6 tok/s.

Source order, one log line says which one was used:

1. `EXL3_MTP_EH_FP16=<path>` (a path that does not exist stops the load with an error; `0` turns the sidecar off)
2. `mtp_eh_proj.st` in the folder given by `--model`
3. `~/models/glm53-mtp-eh-proj-bf16.safetensors` (older location)
4. none: the server runs without it, with lower draft acceptance (one log line)

To build it yourself from the official GLM-5.3 checkpoint:

```bash
python tools/glm/mtp_eh_sidecar.py <official GLM-5.3 checkpoint dir> <model folder>/mtp_eh_proj.st
```

### Two-stage draft head (faster drafting)

If `draft2s.st` is in the model folder and its header matches `config.json` (vocabulary size and hidden width), the
server uses it as the two-stage draft head. No flag needed. It changes only how the draft is found; target token ids
stay equal. Sampled jobs keep the full head. One log line says which source was used. `EXL3_DRAFT2S=<path>` picks
another pack; `EXL3_DRAFT2S=off` turns it off. Build one with `tools/qwen/draft2s_pack.py`.

The GLM MLA decode kernels (VALU path, bitwise equal to the Triton path) are also on by default.
`EXL3_MLA_DEC_VALU=0` turns them off.

## Endpoints

### `GET /health`
`200 {"status":"ok","source":"kyojin"}` when ready; load progress before that (see Start-up progress below).

### `GET /v1/models`
One entry: `{"id": "<model-id>", "object": "model", "created": 0, "owned_by": "local"}`.

### `GET /metrics`
Prometheus text (llama.cpp `llamacpp:*` names) for scraping. Counters accumulate when a
request completes; `requests_processing`, `requests_deferred` and `kv_cache_usage_ratio`
are read live. `spec_decode_num_drafts_total` is derived from the accepted+rejected draft
token counts divided by `--num-draft` (exact except for a final MTP window truncated by
`max_tokens`). A request cancelled mid-stream is not counted in this first version.

### `POST /v1/chat/completions`

Honored:
- `messages` (required, non-empty array) — rendered through the model's own
  `chat_template.jinja` in a Jinja sandbox, with `add_generation_prompt=True`.
  Inbound `tool_calls[].function.arguments` are `json.loads`-ed when they are strings,
  because the GLM template iterates them as a dict.
- `tools` — passed to the template verbatim (no server-side validation).
- `stream` (bool) — SSE or a single JSON body.
- `max_completion_tokens` or `max_tokens` (the first wins; default: all the free context, thinking included; `--default-max-tokens` sets a limit), `temperature` and
  `top_p` (default: the pack's `generation_config.json`, 1.0 and 0.95), `stop` (string or array; also fed to EXL3 as stop conditions).
- `model` — must equal `--model-id` if present, else 400.
- `clear_thinking` — forwarded to the template as a Jinja variable.
- `loop_guard` (bool, chat and `/v1/completions`) — `false` turns the answer-phase loop guard off for this request; absent or `true`
  follows `EXL3_LOOP_GUARD` (default on). See "Loop guard" below.

Ignored (accepted, no effect): `n`, `seed`, `logprobs`, `top_k`, `presence_penalty`,
`frequency_penalty`, `response_format`, `stream_options`, `user`, `parallel_tool_calls`,
`tool_choice`.

### Images

The server loads the vision tower of the pack at start (+0.35 GB GPU memory, about 2 s) and prints `vision=True`.
An image goes in a user message as an OpenAI `image_url` part; the URL is a `data:` URI or a local file path (the
server does not fetch remote URLs). One image takes 40 to about 8,000 tokens depending on its size (a 1568x1568
page is 3,136). `--no-vision` skips the tower; an image request then gets HTTP 400, as do video and audio parts.

An image that is sent again in a later turn keeps the prompt cache **while its embedding is still held**. The
server keeps the decoded embeddings of recent pictures in host memory (GLM's merger emits fp32: 49.0 MiB for a
1568x1568 page, 3,136 tokens, 1 MB for a 224x224 crop) and re-encodes a picture it had to give up, which gives that
picture fresh pseudo token ids and breaks the paged prefix *from that picture on*. `--image-cache-mb` sizes the
cache: 1024 MiB by default, about 20 pages of 1568x1568 at 49.0 MiB each, one pool shared by all sessions
like `-c`.
Which pictures a full cache gives up is **`KYOJIN_IMAGE_CACHE_POLICY`**
([doc/env_vars.md](../../doc/env_vars.md)). The default `head` is the plain LRU this server always had, and it
drops the *earliest* pictures of a history — exactly the ones the paged prefix starts from. **A history the client
keeps re-sending should be served with `tail`**, which keeps those earliest pictures and gives up the end of the
prompt instead: 43.5 % of a 75K-token prompt reused (82.0 s for the warm turn) against 0 % (133.6 s) under the
default. The request log's `img_cache` line reports `hits`, `misses` and `first_missing`, the index the prefix
could match up to; `--image-cache-items` caps the number of pictures held (32 by default, as upstream's shared
cache; `0` = no limit), which for small crops binds before the byte budget does.

`--image-cache-format` stores those rows packed instead of as fp32 (49.0 MiB for a 1568x1568 page): `fp16` is
24.5 MiB and `q8` is 12.3 MiB. Measured per picture that is actually
re-prefilled: `q8` +0.5 ms against the 5.2 ms the fp32 path takes, i.e. 0.01 % of a 40-page cold prefill and
nothing at all on a turn the cache serves from the paged prefix. All of them are off by default. **`fp16` is bit-exact** — the
merger's fp32 output is an fp16 value widened, so nothing is rounded and the saving is free — and `q8` loses
0.863 % (rms 1.3e-4, 41 dB).

The first image of a new size costs seconds: the server runs the vision spatial-merge convolution as a GEMM
(`EXL3_CONV_GEMM=1`) and skips the dense GEMM screening (`EXL3_DENSE_GEMM_TUNE=0`). Both are server defaults
since 1.5; a value already in the environment wins. Measured on this model: start to ready 771 s -> 65 s, first
1024x1024 image 433 s -> 4 s, text prefill and decode unchanged, answers byte-identical.

`EXL3_DENSE_GEMM_TUNE=1` brings the tuner back, with up to 600 s of pre-tuning at the first start
(`EXL3_SERVE_DTUNE_PRIME_S`) and 20-45 s for each new image size. It buys about 6 % of prefill speed on a
large image and nothing on text, because the expert layers are EXL3-quantized and do not use the tuned path.
`EXL3_CONV_GEMM=0` brings the MIOpen convolution back (~0.3 s per vision token for the first image of each size).

With `EXL3_CONV_GEMM=0`, `MIOPEN_FIND_MODE=FAST` alone removes the convolution half of that cost
(a first 1024x1024 goes from 431 s to 22 s) and leaves the GEMM side, and its 600 s of startup
pre-tuning, exactly as the default has them. Measurements, the offline pre-generation alternative, and
the comparison with the default configuration: [doc/EXL3_CONV_GEMM.md](../../doc/EXL3_CONV_GEMM.md).

With `--image-cache-dir-max-gb` (off by default), the pictures the memory budget gives up are written to disk
**when they are evicted** and read back when the client sends them again. It only ever helps a conversation with
more pictures than `--image-cache-mb` holds: a session that fits writes nothing, and a longer one writes each
evicted picture once (files are keyed by content, so a picture evicted twice is not rewritten). A row is only
worth reusing while the model and the preprocessing stay the same, so a file is named after both: a short
fingerprint of the pack and its resize/canvas settings, then the content key and the storage format. Start with
another model, another pack or other pixel settings and the files already there are simply not addressed any more -
they stay until the directory cap takes them, and the request ends up encoding the pictures again. Both the
fingerprint and the fields it was built from are in the log (`image_cache_env` on `serve_start`, `env` inside the
`disk=` report of `img_cache`), so a fingerprint that changed between two runs can be compared. The directory is
`--image-cache-dir`, `~/.cache/kyojin/image-embeddings` unless changed, and anything unreadable in it is dropped
and re-encoded: the disk never becomes a reason for a request to fail. It is meant to be used with `KYOJIN_IMAGE_CACHE_POLICY=tail`: then the evicted
pictures are the end of a prompt, restoring them completes the prompt again and the paged prefix survives, which
is where the large win is (99.8 % of a 24-page prompt reused, 1.2 s against 82.0 s). Under the default `head`
policy it still saves the vision pass for the pictures it holds, but the prefix is broken at the first of them.
Note what it stores: the **rows derived from the input images stay on disk**, which is a privacy and disk-wear
consideration. Because of that its files are created `0600` whatever the process umask is, and a file left
wider by an earlier run is tightened on the next start; the directory is made `0700` only if this server has to
create it, one that already exists keeps its mode, and the shared folder above it (`~/.cache/kyojin`, or
wherever `KYOJIN_HEALTH_DIR` points) is created with the process umask — its missing parents as well — since
other processes write there too.
Two knobs bound it:
`--image-cache-dir-max-gb` caps the directory and `--image-cache-dir-lock-s` caps how often a conversation can
lose its place to another one. Each picture is written to `<name>.tmp` and moved onto its name, so a file that
is there is always complete and a process killed mid-write leaves the `.tmp` behind; the next start removes
those (keeping any younger than five minutes, which could still be another process's write) and reports how many
in `stale_tmp`, since they take room the cap does not count.

### Streaming format
`Content-Type: text/event-stream`. Every frame is a bare `data: {...}` line (no `event:`
name is emitted); frames are:

1. `{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`
2. incremental `delta.reasoning_content` while inside the think block, then
   `delta.content` for prose;
3. one frame with `delta.tool_calls` — the full parsed call list, each with `index`,
   emitted **only at the end** (the parser needs complete XML, so everything from
   `<\u200btool_call>` on is held back);
4. a final frame with an empty delta, `finish_reason` (`tool_calls` or `stop`) and
   `usage: {prompt_tokens, completion_tokens, total_tokens}`;
5. `data: [DONE]`.

If the engine fails mid-stream, the server sends a frame `{"error": {"message": ...}}`, a final frame with `finish_reason: "error"`, then `data: [DONE]`.

Every frame carries `id`, `object: chat.completion.chunk`, `created`, `model`.

Non-streaming returns `chat.completion` with `choices[0].message` holding `content`,
optional `reasoning_content`, optional `tool_calls`, plus `usage`.

### `POST /tokenize`, `POST /v1/completions`, `cache_salt` (vLLM style)
`/tokenize` takes `{"prompt" | "messages", "add_special_tokens"}` and returns `{"tokens", "count", "max_model_len"}`.
`/v1/completions` (stream and non-stream, same queue and `--sessions` as chat) takes `prompt` as text or as a list of token ids
(`[1, 2, 3]` or `[[1, 2, 3]]`). The prompt is used as sent: no chat template, no BOS, no re-encoding of ids. `usage.prompt_tokens`
is the count of ids sent, an id outside the vocabulary answers 400, `ignore_eos: true` keeps generating through EOS, `stop` and
`max_tokens` work as in chat. `usage.prompt_tokens_details.cached_tokens` is on every chat and completions answer.
`cache_salt` (chat and completions, non-empty string): requests with different salts never share prompt cache. The server keeps
one salt namespace at a time (requests without a salt are one namespace): when it changes, running requests finish, the whole
prompt cache is dropped (a fresh Generator over the same cache) and the request starts. The same salt, or no salt at all,
never drops anything. `chat_template_kwargs` (`reasoning_effort`, `clear_thinking`) is read like the top-level fields and wins.

### Loop guard
The server watches the answer (the text after `</think>`, or the whole reply when the prompt opens no think block) and ends a reply
that fell into a loop: a repeated token period, or a tail that replays itself. The thinking block itself is not watched. The guard
cancels the job and the turn reports `finish_reason` `length`, `choices[0].stop_reason` `loop_guard` and the detail in
`timings.loop_guard` (`phase`, `kind`, `at`, `cut`), so a client can tell a guard stop from a natural end. Valid repetitive output
(dense code, tables, JSON records) is not cut, and neither is repetitive data (a grid of zeros, identical JSON lines, a zero-padded
hex dump) until it has repeated for 16384 tokens. `EXL3_LOOP_GUARD=0` turns the guard off for the server; `"loop_guard": false` in a
request body turns it off for that request (a non-boolean value answers 400). It is not applied with `ignore_eos: true`.

### Thinking
GLM's generation prompt ends with `<think>`, so the model *starts inside* the block.
`opens_thinking()` detects that (`prompt.rstrip().endswith("<think>")`) and the server
prepends a synthetic `<think>` to the text it parses. That prefix is stripped again
before `usage` is computed. If a custom template does not end in `<think>`, reasoning is
simply absent. GLM tool calls are emitted by the model as `<\u200btool_call>name
<arg_key>k</arg_key><arg_value>v</arg_value></\u200btool_call>` — the zero-width space is
deliberate so plain prose never triggers the parser; arg values are JSON-parsed when they
parse, otherwise kept as strings.


## Diagnostics: the request log

The log is shared code and its switches, format, every event, the shared/GLM-only split and how to read
the output accounting are documented once, in [doc/logging.md](../../doc/logging.md). What belongs to
this server:

* **its events are the reference vocabulary** — `img_cache` and `serve_start` exist here
  because this is the server with an image cache, a disk tier and a start-up record; the other two run
  the same `Lifecycle` and `tools/test_serve_logging.py` holds all three to the same field lists;
* the start-up lines an operator sees are `serve: request log …` and `serve: SSE keep-alive …`;
* **GLM is the only server whose pinger also writes its comments to the log**, so `sse_ping` lines and
  the `sse_ping_s` field of `serve_start` appear here and not on Qwen or MiMo.

Turn it on with `KYOJIN_REQLOG_DIR=<directory>` (or `1`) — each run writes `requests-<run>.log` there, keeping the newest 20 runs; the switches are in [doc/env_vars.md](../../doc/env_vars.md), and what is written, per run, in [doc/logging.md](../../doc/logging.md).

## Start-up progress on `/health`

The port answers from the first second, while the model is still loading. `GET /health` returns `200 {"status":"ok","source":"kyojin"}`
(plus the fields listed above) when the server is ready, and while it loads:

```json
503 {"status": "loading", "source": "kyojin", "message": "Target weights, 40 % (stage 1 of 4)", "progress": 0.31, "stage": "target weights", "stage_index": 1, "stage_count": 4,
     "stage_progress": 0.4, "elapsed_s": 52.1, "eta_s": null, "stage_eta_s": 78.0, "progress_basis": "stages"}
```

Every `/health` reply carries `"source": "kyojin"` (a hint for parsers). While `status` is `loading`, `message` is one plain line for a UI, built from the other fields: stage, percent, "stage x of y", and "about N s left" when an ETA exists.

Stages here: target weights, drafter weights, warm-up, dense GEMM tuning (the last two can be switched off by `EXL3_SERVE_WARMUP=0` and `EXL3_SERVE_DTUNE_PRIME_S=0`). `stage_progress` is the share of modules loaded (or kernel keys tuned) and is `null` when a
stage cannot count its work. `progress` counts finished stages plus that share, with equal weights
(`progress_basis: "stages"`); once one start has completed, its stage durations are kept in
`~/.cache/kyojin/startup-glm.json` (`KYOJIN_HEALTH_DIR` changes the folder) and later starts weigh the stages by
them (`"history"`) and report `eta_s`. Without a history `eta_s` is `null`: no estimate is made up. `stage_eta_s` is this run's own estimate for the current stage alone (from the share loaded so far); it is `null` until 2 % of the stage is done, and is there with or without a history. GLM keeps two histories (`startup-glm-tuned.json`, `startup-glm-untuned.json`) so a start with the dense GEMM tuning already cached is never timed with the slow first start. A failed load
answers `500 {"status":"error","message":...}` for a few seconds before the process exits. The real API endpoints answer
`503 {"error":{"message":"Loading model",...}}` during the load, like llama.cpp; an unknown path answers `404` and a wrong method `405`, in the same JSON error shape (also after READY). Nothing changes after READY, and
the `READY` line is the same. The helper is `tools/startup_health.py`, shared by the three servers.

llama-swap polls `checkEndpoint` (default `/health`, HTTP 200 = ready) until `healthCheckTimeout` runs out, so give it
room for a first start:

```yaml
healthCheckTimeout: 1800      # seconds; a first start tunes kernels
models:
  "glm":
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /health
    cmd: python tools/glm/serve.py --port ${PORT}
```

## Known limits

- **One request at a time by default.** A second concurrent request waits. `--sessions N` decodes N
  requests together. `n > 1` is ignored.
- **Client disconnect leaks work.** If the HTTP client goes away mid-stream, the worker
  keeps generating into the orphaned queue and every other request stalls until that
  generation ends.
- **Context length** is min(max_position_embeddings, 131072) = 128k (cache sized 128k + 4096).
- **Speed.** ~23.8 t/s end-to-end on the lighthouse bench through this server vs ~30.8 t/s
  in the standalone MTP bench. The gap is content, not the server: mtp_round on the same prose prompt
  gives 24.6 t/s (draft acceptance 0.57) vs 28.2 on chat (0.79); server overhead is ~3%.
- No auth, no CORS, no embeddings endpoint. Bind to 127.0.0.1.
- **The image disk cache degrades with `--sessions > 1`.** With several conversations taking turns, a directory
  too small for all of them makes each request rewrite what the previous one dropped. `--image-cache-dir-lock-s`
  reduces that writing but does not remove it; a directory with room for every active conversation does. This
  needs many pictures *and* concurrent conversations at once, so it is rare in practice.

## Open issues (found while documenting, NOT fixed)

1. **(FIXED) Context length was silently 32768, not 128k.** `ResidentEngine.__init__` does
   `self.config.config_dict.get("max_position_embeddings") or 32768`. For this pack
   `max_position_embeddings` (1048576) lives under `text_config`, and `config_dict` is
   the raw JSON — the top-level lookup misses and the fallback wins. The `Cache` is
   therefore sized `32768 + 4096` instead of `min(max_position, 131072) + 4096`, so
   requests past ~32k tokens will hit cache exhaustion. Fix: read
   `text_config->max_position_embeddings` (as `Glm5NextConfig` does for every other key).
2. **`main()` imports `os` a second time** inside the function (line 324) while the module
   already imports it at top level. Harmless, dead code.
3. **`usage.completion_tokens` is re-tokenized from the parsed text**, so it counts the
   *rendered* string (reasoning + prose + XML tool call) rather than the tokens actually
   generated. Close, but not the same number as EXL3's own count.

## Tests

```bash
# unittest: no extra dependency, which is what this checkout's venv has
python -m unittest tools.glm.test_serve tools.glm.test_serve_glm_additional \
                  tools.strix_halo.test_serve_ping tools.strix_halo.test_image_cache \
                  tools.test_serve_compat tools.test_startup_health
# the same under pytest, if it is installed
python -m pytest -q tools/glm/test_serve.py tools/glm/test_serve_glm_additional.py \
                    tools/strix_halo/test_serve_ping.py tools/strix_halo/test_image_cache.py \
                    tools/test_serve_compat.py tools/test_startup_health.py
```

Model-free: a `FakeEngine` emits deterministic GLM XML in 3-character chunks so tags are
split mid-token, covering template rendering, think/tool parsing, stop strings, SSE
framing, `/v1/models`, usage, and the streamed open-`<think>` path. No GPU needed.

`tools/strix_halo/test_image_cache.py` covers the shared cache itself, including both eviction policies (the
default LRU and `KYOJIN_IMAGE_CACHE_POLICY=tail`). `tools/strix_halo/test_serve_ping.py` covers the shared
keep-alive timing (off by default, one comment per interval, retired at the first token, one pinger per
request).

`test_serve_glm_additional.py` holds the tests for what this checkout adds locally: the
request log (`KYOJIN_REQLOG_DIR`, off by default, no request text), the image cache
(`--image-cache-mb` / `--image-cache-items`, the defaults of the `--image-cache-dir*` options,
the row format, what `img_cache` reports) and the SSE keep-alive (a comment reaches the client before
the first token, and nothing at all with the switch off). It is a file of its own so upstream's
`test_serve.py` stays untouched and merges cleanly.

## Lane launcher

`tools/lanes/serve_glm.sh` runs this server with the exact flags and speed envs of our serving lane (see `tools/lanes/README.md`). The runtime uncensor hook is off unless `GLM_ABLIT` points at an edit spec.
