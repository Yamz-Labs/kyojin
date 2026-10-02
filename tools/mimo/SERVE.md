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
`--default-temperature`, `--default-top-p`: used when a request omits or nulls the field; the lane passes 1.0 / 0.95).

`tools/lanes/serve_mimo.sh` runs the lane configuration (speculation on, `MIMO_SPEC=0` for plain decode, model-card sampling).
Tool calls: a block body is JSON or `<parameter=k>v</parameter>` XML, typed by the request's tool schema; one block may hold several `<function=...>` elements; a malformed body is passed through raw; a cut-off trailing call is dropped.
Model id in requests: `MiMo-2.6-EXL3`.

`--ctx 0` (default) takes the max context from the model config, following `text_config` when the
top-level config does not carry `max_position_embeddings` — the GLM fix.

## Endpoints

- `POST /v1/chat/completions` — `stream: true` (SSE `chat.completion.chunk`) and `false`.
  `tools` / `tool_calls` / `tool` role messages, `reasoning_content` for `<think>` blocks,
  `stop` (string or list), `temperature`, `top_p`, `max_tokens`. Tool calls are emitted only
  once the XML block closes, so a client never sees half a JSON payload.
- `GET /v1/models` — one entry, `MiMo-2.6-EXL3`.
- `GET /health` — `{"status","model","generator","spec_gate"}`; with the gate on, `spec_gate`
  carries the live counters `n_spec n_plain n_off n_shadow n_probe_fail t_plain t_spec
  tok_round`. Useful while tuning.

## SpecGate across mixed requests — decision

**Kept, not reset per request** (the default). The gate holds no prompt content: only smoothed
cost measurements (`t_plain`, `t_spec`) and the probe outcome. The shadow-probe re-entry path
re-calibrates within a few plain steps when the next request's acceptance differs, which is what
makes a code request followed by a chat request safe. Verified on GPU: 6 alternating code/chat
requests in one process, gate counters and served output both sane
(`scratch/msrv/health_after.json`).

`--spec-gate-reset-per-request` builds a fresh `SpecGate` per request instead. It costs the first
few steps of every request (the gate skips the first two speculative rounds and then probes), so
it is a debugging switch, not the default.

## Tests

    python tools/mimo/test_serve.py     # no GPU
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

