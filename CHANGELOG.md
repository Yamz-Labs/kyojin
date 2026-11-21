# Changelog

## Unreleased (1.9.2.1)

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
