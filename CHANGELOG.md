# Changelog

## 1.9.3

Hy4-flash on the Strix Halo box: faster speculative decode, a fix that gives the new MoE kernel path back to the Q3_K file, the client fixes of issue #222, and a client smoke gate that ships in the tree.

- Speed (served, speculative decode, 12K context, two prompts per class, median of 4 to 9 server boots, Strix Halo box, tok/s):

  | step | code | prose | chat | mean |
  |---|---|---|---|---|
  | 1.9.2 | 24.4 | 20.2 | 20.5 | |
  | + five decode levers | 25.7 (+5.4 %) | 22.3 (+10.7 %) | 21.5 (+4.9 %) | +7.0 % |
  | + depth gate (learn) | 26.3 (+0.9 %) | 23.7 (+6.1 %) | 22.8 (+4.6 %) | +3.9 % |
  | + row-batched MLA attention | 29.6 (+12.8 %) | 26.4 (+11.5 %) | 25.1 (+10.2 %) | +11.5 % |

  Each row is a paired A/B against the row above it. The first row pair comes from one run and the last two rows from a second run (the reference arm of that run was voided: one prose prompt spread 6.2 % over its reps), so the steps are not one continuous measurement. Greedy outputs differ from 1.9.2 at near ties only. GSM8K 20/20 on every arm, memory footprint unchanged (about 98.8 GiB GTT).
- On by default (set in `SHIPPED_ENV`; `HY4_<FLAG>=0` restores the 1.9.2 path of that lever): `HY4_DENSE_V2`, `HY4_MOE_V2`, `HY4_DRAFT_MOE_TILES`, `HY4_DRAFT_MLA_BG`, `HY4_DRAFT_IX_GV`, `HY4_SPEC2_GATE` (with `HY4_SPEC2_MODE=learn`) and `HY4_MLA_ROWS2`. Rebuild the kernels with `bash tools/hy4/install_strix.sh` after the update; without the rebuilt library `HY4_DENSE_V2` and `HY4_MOE_V2` fall back and the `FIRES` line names the reason.
- Where the new defaults step aside: the depth gate is off with `--slots > 1`, with `--num-draft` other than 3, and when `HY4_DRAFT_POLICY` is set. Row-batched MLA attention is off with `--slots > 1`, and a decode call outside its shape support runs the 1.9.2 attention path (counted in the `mla2` fires line). Setting either flag by hand in those cases is refused at load.
- Fix: on the Q3_K GGUF, 22 of the 42 MoE layers never took the V2 expert path (`stride_align`: Q3_K gate / up rows need 2-byte alignment, not 16). All layers do now.
- Server fixes of issue #222, released earlier as 1.9.2.1 and 1.9.2.2 and carried here: a refused request never stops the server, named part errors, `KYOJIN_IMAGE_PARTS`, a request without `max_tokens` gets the room left in the context window and a larger `max_tokens` is clipped instead of refused.
- Client smoke gate: `python tools/hy4/client_smoke.py --base http://localhost:8300 --ctx 16384` runs OpenAI SDK chat and completions (streamed and not), tool calls (auto, required, named, multi-turn), image / audio / file parts, a client that drops the stream, a long prompt without `max_tokens`, an oversized `max_tokens`, 4 concurrent requests and replayed agent-client requests, then a normal request after each error. 1.9.3 passes it: 172/172 checks on the built release-candidate tree, against the real model.

### Details

- Decode levers, shipped on in `SHIPPED_ENV` (`HY4_<FLAG>=0` restores 1.9.2; code default `0`), each with a fires counter printed after the load warm-up (`FIRES <name> {...}`, `loader/lever_fires.py`) and a CPU test (`tests/test_hy4_seta_levers_cpu.py`):
  `HY4_DENSE_V2` (column-tiled few-row Q8_0 dense GEMV, `gm_dense_q8`, bit-identical to the stock kernel; rebuild `exl3_gm_ext` with `tools/hy4/install_strix.sh`),
  `HY4_MOE_V2` (routed experts through the HIP code object `moe_dec.co`, built by `install_strix.sh` step 2b, `MOE_DEC_CO` overrides the path),
  `HY4_DRAFT_MOE_TILES`, `HY4_DRAFT_MLA_BG`, `HY4_DRAFT_IX_GV` (draft MTP layer: int8 expert tile retune, Q8 absorb matrices, indexer mat-vec).
- New tools: `tools/hy4/step/opbench.py` (per-op decode microbench, no server boot) and the paired A/B drivers `ab_dend.py`, `ab_exp.py`, `mtp_items.py`, `mtp_micro.py`, `mtp_arms.sh`; model and data paths come from `GGUF` / `WIKI` (no default path).
- KV footprint: `HY4_KV_FP8_MTP=1` keeps the draft layer's K cache in fp8 too (needs `HY4_KV_FP8=1`), `HY4_KV_COMPACT=1` sets fp8 trunk + draft cache + the 1024-row window ring in one switch (about 11 KB per token instead of 113 KB); all off by default. `tools/hy4/kv_bytes.py` prints the per-component table, `tools/hy4/kv_bytes_probe.py` measures it. CPU tests: `tests/test_kv_bytes.py`.
- KV footprint 2: `HY4_IX_POOL16=1` (fp16 indexer pool, -832 bytes per token) and `HY4_ROPE_POS=1` (rope cos / sin rows evaluated from the positions of a call, no tables to the full context, -1280 bytes per token), both part of `HY4_KV_COMPACT=1` now: 9220 bytes per token (9.0 KB, 9.1 KB at 1M) instead of 11332. Off by default. CPU tests (including the Triton interpreter): `tests/test_kv10.py`; GPU gate: `tools/hy4/step/kv10_driver.py`.
- Long context: opt-in fp8 e4m3 latent K cache for the full-attention layers (`HY4_KV_FP8=1`, off by default; 644 bytes per token and layer instead of 2304) and the head-blocked split-K sparse attend that reads it (`HY4_LC_HB`, `HY4_LC_HB_SPLIT`; only runs on the fp8 cache). `-c 65536` keeps the fp32 cache. Guide: `doc/hy4.md`, "Long context option". CPU tests: `tests/test_kv_fp8.py`, `tests/test_hy4_shipped_flags.py`.
- Hy4 decode graphs: context buckets can grow with the context (1/32 of the octave above 16K rows) instead of a fixed 512 rows, so a 256K prompt lists 225 buckets instead of 2049. `HY4_GRAPH_BUCKET_REL` (default 0 = fixed 512-row buckets, as in 1.9.2; 1 = growing buckets). CPU test: `tests/test_graph_bucket.py`.
- Hy4 MLA2 (`HY4_MLA_ROWS2`, shipped on in 1.9.3; `HY4_MLA_ROWS2=0` restores the 1.9.2 attention path): row-batched split-K decode attention. One pass over the K rows serves all verify rows, q is streamed in chunks, the output has 4 quarters. Tunables `HY4_MLA_ROWS2_BH` 16, `_RG` 1, `_BN` 32, `_BK` 64, `_NW` 4, `_PROG` 64, `_SPLIT` 0 (auto), `_RS` 8, `_MINB` 2. `HY4_MLA_ROWS2_PRECOMPILE=1` (default 0) compiles and loads every (T, S) specialisation at the end of load, outside any graph capture. `HY4_MLA_ROWS2_TRACE=1` prints the first sighting of each launch key and flags a JIT inside a capture. Fires counter: `HY4_MLA_ROWS2_FIRES=<file>` holds one line `FIRES mla2 <n> trunk <a> draft <b> fallback {reason: n} fallback_trunk {...} fallback_draft {...}`; an A/B arm with the flag on must show n > 0 and no fallback. Measured on the Strix Halo box (Hy4-flash): precision equal to the deployed head-blocked fp16-dot kernel (1152 calls, mean abs error 2.66e-05 for both); served hang matrix 12/12 boots clean, including an empty Triton cache. Paired served A/B in 1.9.3: mean +11.5 % decode, every prompt class above 0, GSM8K 20/20, no trunk fallback; shipped on. CPU tests (no GPU): `tools/hy4/step/mla2_precompile_cpu_test.py`, `mla2_engine_hook_test.py`, `mla2_fires_cpu_test.py`; micro `tools/hy4/step/mla2_micro.py` (`MLA_CPU=1` for a CPU smoke run).
- Hy4-flash: online speculative draft-depth gate, `HY4_SPEC2_GATE=1` (shipped on in 1.9.3, set `HY4_SPEC2_GATE=0` to turn it off), mode `HY4_SPEC2_MODE=learn` (default, the measured S7 arm O), KB 2, THETA 0.6 for the `static` mode.
  Base draft graph of 2 steps plus a confidence-gated third step. Fires counter per request: depth chosen per round and rounds where the gate changed the depth.
  The gate's tail graph uses the same bucket helper as `HY4_GRAPH_BUCKET_REL`. See `doc/env_vars.md`.
- Hy4-flash server fixes of issue #222, also released as 1.9.2.1: see the 1.9.2.1 section below (no hang after a refused request, named part errors, harmless parts accepted, `KYOJIN_IMAGE_PARTS` default `note`).
- Hy4-flash server reply budget (issue #222), also released as 1.9.2.2: see the 1.9.2.2 section below (`--default-max-tokens 0` = all the room left is the new default; a larger `max_tokens` is clipped, not a 400).

## 1.9.2.2

Hy4-flash server: a request without `max_tokens` is no longer cut at 1024 tokens, and a `max_tokens` larger than the context window can hold is clipped, not refused (issue #222: an agent client's context compaction ended with stop reason `length`).

- `--default-max-tokens 0` is the new default: a request with no `max_tokens` / `max_completion_tokens` may use all the room left in the context window (`-c` minus the prompt tokens, the draft rows and 2), like vLLM. A positive value keeps the old cap behaviour (the old default was `1024`). A negative value stops the start-up with a message.
- A `max_tokens` (or the positive default) larger than the room left is clipped to the room; the reply ends with `finish_reason: "length"` when the clipped budget is used. It was a 400 (`prompt (N tokens) + max_tokens (M) exceeds the context window`). A prompt that leaves no room for one token is still a 400, with a message that says so. Same on `/v1/chat/completions` and `/v1/completions`, streamed or not.
- New CPU tests: `tests/test_hy4_max_tokens.py`.

## 1.9.2.1

Hy4-flash server: a refused request must never stop the server, and fewer harmless content parts are refused (issue #222).

- A worker that fails while it flushes its output (or dies without its end event) no longer leaves the request handler waiting with the generation lock held: the end event is always sent, the lock is released, and the client gets a 500 (or an SSE error event followed by `[DONE]`). An unexpected exception in a request parser or the chat template is one JSON 500 and the server keeps answering. Every request check (content parts, tools, effort, `max_tokens`, `stop`, context size, JSON) runs before the lock and before the SSE response starts.
- Content parts: the 400 names the message and the part (`message 7, part 2: type 'video' is not supported`; indexes start at 0). Accepted now: `refusal` (as text), `reasoning` / `thinking` / `reasoning_content` parts of an assistant message (they become its `reasoning_content`; dropped in other roles), `redacted_thinking` (dropped), bare string parts, empty text, extra keys such as `cache_control`. Unknown types are still a 400. One table (`PART_ACTIONS` in `tools/hy4/serve_core.py`) adds a type.
- New `KYOJIN_IMAGE_PARTS=note|reject` (default `note`): each image, audio or file part of any message (user, assistant, tool) becomes one line of text such as `[an image was attached; this model cannot see images]` and the request is served, so agent clients that keep screenshots in their history keep working. `reject` is the strict 400 of 1.9.2. Any other value is `note` with one warning line. See `doc/env_vars.md`.
- New CPU tests: `tests/test_hy4_request_error.py` (every refused request, then a normal one answers, streamed or not), `tests/test_hy4_content_parts.py`.

## 1.9.2

Hy4-flash server hotfix: tool calling, a quiet exit when the client leaves, request-parser fixes, Strix install fixes, docs.

- Tool calling (it was a 400 "not supported"): `tools` and legacy `functions`, `tool_choice` `none` / `auto` / `required` / a named function (and legacy `function_call`), assistant messages with `tool_calls`, `tool` messages with `tool_call_id`, all rendered through the GGUF chat template. Replies carry `message.tool_calls` (JSON `arguments` string, `content` null when only calls, `finish_reason` `"tool_calls"`); streamed replies carry `delta.tool_calls` (index, id and name, then the arguments). Reasoning stays in `reasoning_content`. The parser reads the model's GLM-shaped tool block (`<tool_calls>`, `<tool_call>`, `<arg_key>`, `<arg_value>`, every tag with the template suffix), types the arguments by the declared schema, and leaves a cut-off or unknown block as plain content. The template has no switch for `tool_choice`: `none` drops the tools from the prompt; `required` and a named function are forced by prefilling the assistant turn (empty think block, then `<tool_calls><tool_call>` and the function name when it is named), so the model writes only the arguments (and the name for `required`) and the reply is always a call with `finish_reason` `"tool_calls"`; a named function also keeps only that tool in the prompt.
- A client that closes the stream no longer raises in `sse()` (`ClientConnectionResetError: Cannot write to closing transport`, issue #222): the generation is cancelled, one log line is printed, no traceback. Requests that are not streamed are cancelled the same way when the client leaves. An engine error after the stream began is sent as an SSE error event followed by `[DONE]` (it used to attempt a second response and leave the client hanging).
- Chat request parser: all system / developer messages are merged into one leading system message instead of a 400; a last assistant message is continued (prefill), an empty one is dropped; `response_format` `json_object` / `json_schema` is a clean 400 "not supported yet" (it was silently ignored); `n` > 1 is a clean 400; an explicit `max_completion_tokens: null` no longer hides `max_tokens`; a `stream_options` that is not an object is a 400 (was a 500); null parts, null / empty text and `input_text` / `output_text` content parts are accepted; a message shape the template cannot read is a 400, never a 500.
- `tools/strix_halo/env.sh` finds the HSA runtime through `rocm-sdk path --root` when `EXL3_ROCM_SDK` is unset, and stops with a clear error instead of letting torch segfault when there is none (`EXL3_ALLOW_NO_HSA=1` for CPU-only use). `hipsp/build.sh` and `ck_fmha/build_mla.sh` remove the old library first and fail when the build fails (the first ended with `ls` and could exit 0 with a stale `.so`).
- Docs: the README test command uses the right port per server (Hy4-flash is on 8300) and the Hy4 example answers directly (`reasoning_effort` `no_think`); README states that reasoning goes to `reasoning_content`; README version line, CHANGELOG 1.9.1 wording and the venv command (`python3.12`) are consistent across the install docs.
- New: `tools/hy4/tool_smoke.py`, a served smoke test of tool calling and client disconnect. New CPU tests: `tests/test_hy4_tools_cpu.py`, `tests/test_hy4_compat_cpu.py`, `tests/test_strix_scripts_cpu.py`.

## 1.9.1

Hy4-flash: faster prefill at every depth and faster decode on the same prompt (the 32K served decode cell reads lower than 1.9.0 because that prompt's tau and generated text differ; see the README table note), decensor preset, warm-up fix. Every lever below ships on and has its own switch: `HY4_<FLAG>=0` in the environment restores the 1.9.0 behaviour of that lever. The flag table is in `doc/env_vars.md`; numbers are in the release notes.

- Decensor preset for Hy4-flash, opt-in and off by default (`HY4_ABLIT_SPEC=on` loads the bundled preset; an explicit path loads another spec). Validated in no_think mode; in the default thinking mode some answers run long (2 of 17 hit the 4096-token cap in our check), a fix is in progress. A per-layer run-time edit of the attention and MLP / MoE outputs, no weight change. Files in `presets/hy4-uncensor/`, code in `exllamav3/loader/ablit_hy4.py`.
- Prefill: lazy layer residual, 3 stream passes per layer (`HY4_PF_HC2`); head-major K / V and Q / O operands for the CK latent-attention core (`HY4_PF_KV_HM`, `HY4_PF_QO_HM`); Q3_K gate-up kernel V2 with packed fp16 decode and aligned dword loads, bit-identical to the old kernel (`HY4_PF_Q3K_V2`; needs the rebuilt MoE extension, which `install_strix.sh` rebuilds; the old kernel runs when the library is older or K / 256 is odd).
- Decode: Q8_0 output head, one resident copy, the bf16 head is freed (`HY4_HEAD_Q8`); the shared-expert branch runs on a second stream inside the HIP graph (`HY4_GRAPH_FORK`); router mat-vec launch tuning, K block 1024 (`HY4_RG_TUNE`); block-wise absorb kernels, absorb v2 (`HY4_BG_V2`).
- `HY4_PF_ROUTER16` (fp16 router logits of prefill rows) is shipped on but inactive with the fused router (`HY4_REF_FUSED=1`, the shipped decode path): it changes nothing in the shipped configuration.
- Install: the CK latent-attention build (`install_strix.sh` step 3/3) finds the GCC install directory from `gcc` itself instead of a hard-coded GCC 13 path (`GCC_DIR` still overrides it).
- Multi-session: `serve.py --slots N` (N > 1) now starts on the shipped defaults (the long-context sparse attend, which the multi-session engine does not take, is switched off when slots > 1).
- Warm-up: the start-up warm-up prompt is clamped to the context window, so a small `-c` no longer fails the warm-up; it is skipped, with one log line, when the context is too small to warm anything useful.
- Test: `tests/test_hy4_shipped_flags.py` checks that every 1.9.1 lever is on in `SHIPPED_ENV`, reads its environment switch with the 1.9.0 behaviour as the code default, and is not overridden when the user sets it.

## 1.9.0: Hy4-flash v1

Hy4-flash (GGUF Q3_K) on one Strix Halo machine. Numbers: see the benchmark table in the release notes.

- New `tools/hy4/serve.py`: OpenAI-compatible server for Hy4-flash with prefill fast path, HIP-graph decode, MTP drafts and prompt-lookup rows. Guide: `doc/hy4.md`.
- Sparse-attention prefill: per-query gather MLA kernel in HIP/WMMA (`hipsp/`, H = 32 and H = 64), Triton fallback; blocked indexer score with linear-time exact top-k; sparse path from 2048 keys.
- Long-context decode: exact top-k sparse attend (`HY4_LC_SPARSE`), draft-cache fill with the same selection.
- Prefill: fused dequant GEMMs, WMMA expert kernels shared with the GGUF MoE path, CK latent-attention core, dense GEMMs split in pieces of 8192 rows above 8192 rows.
- Decode: static-buffer graph step, MTP head from a Q5_K copy of the output head, n-gram rows on top of the drafts, Engram rows prefetched by row.
- The server applies the frozen TunableOp results (`tools/hy4/tunableop_gfx1151.csv`) and a start-up warm-up request by default (`HY4_TUNABLEOP_FILE`, `HY4_WARMUP_TOKENS=0` to skip it), and picks up `head_q5_k.bin` next to the GGUF.
- Generic GGUF loader (`EXL3_GGUF_GENERIC=1`, off by default) and its CPU tests.
- Context: 64K tokens measured (`-c 65536` in the guide); `-c 131072` is opt-in and uses almost all of the memory of a 128 GB machine. Single user (`--slots` is experimental). Greedy decoding only.
- A missing built kernel library (repository extension, MoE extension, attention libraries) stops the server with the install step; `HY4_ALLOW_SLOW=1` keeps the old slow-path start.
- Decode levers on by default: head-batched MLA attention (`HY4_MLA_HB`), k_b / v_b and output-head products on `tl.dot` (`HY4_BG_DOT`, `HY4_HEAD_DOT`), fp16 router weights (`HY4_RG_F16`), fused K-row / q-rope kernels for verify rows (`HY4_KV_FUSED_MAXT`). Step tests in `tools/hy4/step/`.
- The MTP draft-cache fill after prefill writes only K / V and the indexer key state (`HY4_PRIME_KVONLY`, on by default).
- Prefill levers on by default: fused SwiGLU + cast and cast-add glue, finer M-split of the dense GEMMs (`HY4_PF_G_SWIGLU`, `HY4_PF_G_ADD`, `HY4_PF_MM_FINE`); Engram host side with a numpy hash, a single prefetch worker and native row reads (`HY4_ENGRAM_NPHASH`, `HY4_ENGRAM_PF_WORKERS`, `HY4_ENGRAM_NATIVE_IO`); sparse prefill attention from 2048 keys.
- Native Engram IO is built with `gcc` at first use (`~/.cache/kyojin/engram_io`). Without `gcc`, or when the build fails, the server uses the Python reader.
