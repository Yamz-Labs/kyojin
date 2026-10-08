# msrv — MiMo EXL3 OpenAI server (DFlash speculation)

`tools/mimo/serve.py` serves MiMo-V2.6-Flash EXL3 over an OpenAI-compatible HTTP API, with the
DFlash drafter on the decode path (speculation is on by default).

The engine wiring is copied from `scripts/dflash-bench.py` (same `model_init.init` call, same
`-cs`/`-ndt`, same `Generator` arguments), so served throughput is engine throughput: the HTTP
layer adds < 1 t/s (measured, see "Measured" below). The HTTP/streaming layer is adapted from
`tools/glm/serve.py` in the GLM server.

## Run

    source tools/strix_halo/env.sh   # from the repository root; sets PYTHONPATH
    export EXL3_REPO=$PWD
    env \
      python tools/mimo/serve.py \
        --model ./mimo-pack --port 8000 --ctx 32768

The 4 bpw drafter is found in `<model>/drafter` (or `<model>-drafter` next to the model, or `$MIMO_DRAFTER`, or
`--drafter <dir>`). If none is found the server logs one line and decodes plain. `--no-dflash` forces plain decode.
Defaults: `--ndt 7 --dynamic-draft --draft-confidence 0.6`, no spec gate. The drafter directory carries
`draft_conf_prior.json`, an offline table that seeds the confidence threshold (`EXL3_DRAFT_PRIOR=0` skips it).
`EXL3_LAZY_DKV=0` turns off the lazy drafter KV catch-up. `EXL3_SPEC_PROF=1` records per-phase timings.

Flags: `--host --port --model --model-id --drafter --ctx --ndt`
(`--dflash`/`--no-dflash`, `--dynamic-draft`/`--no-dynamic-draft`, `--draft-confidence`, `--spec-gate`/`--no-spec-gate`, `--spec-gate-reset-per-request`,
`--default-temperature`, `--default-top-p`: used when a request omits or nulls the field; default = the pack's `generation_config.json`, 1.0 / 0.95;
`--default-reasoning-effort medium|none`: `medium` (default) adds one effort instruction as a leading system block, `none` leaves the model's own prompt;
`--chat-template`: default `tools/mimo/chat_template.jinja`, the model's template plus that `medium` level).

`tools/lanes/serve_mimo.sh` runs the lane configuration (speculation on, `MIMO_SPEC=0` for plain decode, model-card sampling).
Tool calls: a block body is JSON or `<parameter=k>v</parameter>` XML, typed by the request's tool schema; one block may hold several `<function=...>` elements; a malformed body is passed through raw; a cut-off trailing call is dropped.
Model id in requests: `MiMo-2.6-EXL3`.

`--ctx 0` (default) takes the max context from the model config, following `text_config` when the
top-level config does not carry `max_position_embeddings` — the GLM fix.

## Endpoints

- `POST /v1/chat/completions` — `stream: true` (SSE `chat.completion.chunk`) and `false`.
  `tools` / `tool_calls` / `tool` role messages, `reasoning_content` for `<think>` blocks,
  `stop` (string or list), `temperature`, `top_p`, `max_completion_tokens` or `max_tokens` (the first wins; default 32768, thinking included; `--default-max-tokens`). A reply cut by the limit reports `finish_reason: "length"`. An engine failure mid-stream ends with an `error` frame, `finish_reason: "error"` and `[DONE]`. Tool calls are emitted only
  once the XML block closes, so a client never sees half a JSON payload.
- `GET /v1/models` — one entry, `MiMo-2.6-EXL3`.
- `GET /health` — `{"status","model","generator","spec_gate"}` once ready (load progress while starting: see below); with the gate on, `spec_gate`
  carries the live counters `n_spec n_plain n_off n_shadow n_probe_fail t_plain t_spec
  tok_round`. Useful while tuning.
- `GET /metrics` — Prometheus text (llama.cpp `llamacpp:*` names) for scraping. Counters
  accumulate when a request completes; `requests_processing`, `requests_deferred` and
  `kv_cache_usage_ratio` are read live. `spec_decode_num_drafts_total` counts the engine's
  decode steps (one verification round each while speculating), an upper bound when
  `--spec-gate` falls back to plain. A request cancelled mid-stream is not counted in this
  first version.

## Start-up progress on `/health`

The port answers from the first second, while the model is still loading. `GET /health` returns `200 {"status":"ok","source":"kyojin"}`
(plus the fields listed above) when the server is ready, and while it loads:

```json
503 {"status": "loading", "source": "kyojin", "message": "Target weights, 40 % (stage 1 of 3)", "progress": 0.31, "stage": "target weights", "stage_index": 1, "stage_count": 3,
     "stage_progress": 0.4, "elapsed_s": 52.1, "eta_s": null, "stage_eta_s": 78.0, "progress_basis": "stages"}
```

Every `/health` reply carries `"source": "kyojin"` (a hint for parsers). While `status` is `loading`, `message` is one plain line for a UI, built from the other fields: stage, percent, "stage x of y", and "about N s left" when an ETA exists.

Stages here: drafter weights, target weights, engine setup (no drafter stage with `--no-dflash`). `stage_progress` is the share of modules loaded (or kernel keys tuned) and is `null` when a
stage cannot count its work. `progress` counts finished stages plus that share, with equal weights
(`progress_basis: "stages"`); once one start has completed, its stage durations are kept in
`~/.cache/kyojin/startup-mimo.json` (`KYOJIN_HEALTH_DIR` changes the folder) and later starts weigh the stages by
them (`"history"`) and report `eta_s`. Without a history `eta_s` is `null`: no estimate is made up. `stage_eta_s` is this run's own estimate for the current stage alone (from the share loaded so far); it is `null` until 2 % of the stage is done, and is there with or without a history. A failed load
answers `500 {"status":"error","message":...}` for a few seconds before the process exits. The real API endpoints answer
`503 {"error":{"message":"Loading model",...}}` during the load, like llama.cpp; an unknown path answers `404` and a wrong method `405`, in the same JSON error shape (also after READY). Nothing changes after READY, and
the `READY` line is the same. The helper is `tools/startup_health.py`, shared by the three servers.

llama-swap polls `checkEndpoint` (default `/health`, HTTP 200 = ready) until `healthCheckTimeout` runs out, so give it
room for a first start:

```yaml
healthCheckTimeout: 600      # seconds; a first start tunes kernels
models:
  "mimo":
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /health
    cmd: python tools/mimo/serve.py --port ${PORT}
```

## SpecGate across mixed requests — decision

**Kept, not reset per request** (the default). The gate holds no prompt content: only smoothed
cost measurements (`t_plain`, `t_spec`) and the probe outcome. The shadow-probe re-entry path
re-calibrates within a few plain steps when the next request's acceptance differs, which is what
makes a code request followed by a chat request safe. Verified on GPU: 6 alternating code/chat
requests in one process, gate counters and served output both sane.

`--spec-gate-reset-per-request` builds a fresh `SpecGate` per request instead. It costs the first
few steps of every request (the gate skips the first two speculative rounds and then probes), so
it is a debugging switch, not the default.

## Tests

    python tools/mimo/test_serve.py     # no GPU
    python tools/test_startup_health.py # /health load progress, no GPU
    python tools/mimo/test_toolcalls.py # tool-call parser, no GPU

No `exllamav3` import and no model: the tests drive the HTTP layer against a fake engine and cover
bad requests, unknown model, streaming vs non-streaming agreement, tool-call parsing and stop
handling, and the MiMo template render.

## GPU acceptance harness

    tools/mimo/accept.py    # 400-token code/chat prompts over HTTP, e2e + decode tok/s
    tools/mimo/parity.py    # --phase save (server up) / --phase compare (server down)
    tools/mimo/plain_greedy.py   # plain greedy reference, same prompt, no drafter
    scratch/msrv/run_accept.sh, scratch/msrv/run_arms.sh   # the exact runs below

`accept.py` reports e2e tok/s (whole request, prefill included) and decode tok/s
(`(n-1)/(e2e - TTFT)`, the same definition as `dflash-bench.py:tps()`). e2e is the honest
server-side number; decode is the one comparable with the engine tables.

## Speed with the 4 bpw drafter (default configuration)

Ryzen AI Max+ 395, 128 GB, gfx1151, 105 GB pack, `-c 32768`, client temperature 0, 128 new tokens, medians of 6 runs over two server loads (`--ndt 7 --dynamic-draft --draft-confidence 0.6`, no spec gate):

| decode tok/s | prose | chat | code |
|---|---|---|---|
| plain (`--no-dflash`) | 28.9 | 28.9 | 28.9 |
| speculation (default) | 32.1 | 34.8 | 44.3 |

A loaded drafter costs 2 to 4 % prefill. The older measurements below used a bf16 drafter and the SpecGate.

## Measured (local gfx1151, gpu-lease, 400-token prompts, 128 new tokens, greedy, n=5)

Extension rebuilt in this worktree (`scratch/msrv/build.log`, rc=0). Server:
`--ctx 2304 --dflash --spec-gate --drafter ~/models/mimo26-exl3/dflash-exl3-6b`.

| leg | code e2e | code decode | chat e2e | chat decode |
|---|---|---|---|---|
| served, DFlash ndt 7 + gate, n=5 median | 18.9 | 29.5 | 17.5 | 26.0 |
| served, `--no-dflash` control | 20.1 | 31.0 | 20.1 | 30.1 |

Per-rep code decode: 32.5 / 44.3 / 27.9 / 29.5 / 25.5. The spread is real and is the gate
alternating between speculating and falling back; a 2-rep median near 40 t/s was not a stable
number, which is why the acceptance is reported at n=5.

**`EXL3_DEC_MOE_UNION_DEV=1` is set by default in serve.py** (`setdefault`, caller wins). Without
it a speculative round costs 166 ms for 4 tokens (41.5 ms/token) against a 35.7 ms plain step, so
the gate switches speculation off entirely and the server runs at plain speed. With it the gate
stays speculative (n_plain 1-51 instead of 675) but the round still costs 121.9 ms for 2.80
tokens = 43.6 ms/token, so on this box speculation still does not pay in steady state.

The control leg is the load-bearing number: same HTTP stack, same prompts, same process, 31.0 t/s.
The server costs under 1 t/s. The brief's 42.7 t/s code figure was measured on the test box
and does not reproduce here as a steady state;
`scratch/msrv/accept.md` has every drafter/env arm that was tried.

**Greedy parity: PASS.** The server's 64 greedy tokens (DFlash + gate) are token-identical to the
plain-greedy reference for the same prompt: `first_diff null, n_diff 0` (`scratch/msrv/parity.json`).

