# Hy4-flash on Strix Halo (GGUF, one machine)

Kyojin loads the public Hy4-flash GGUF (Q3_K split, 2 files, 144 GiB) and serves it on one Strix Halo machine with ROCm. Prefill runs the sparse-attention
fast path; decode runs a HIP graph with multi-token-prediction (MTP) drafts and prompt-lookup rows. Speed numbers: see the benchmark table in the README and the release notes.

## Requirements

- A gfx1151 machine with 128 GB, Linux, ROCm 7.0 or newer, the repository built as in [install.md](install.md).
- About 150 GB of free disk for the model.

## Get the model

The GGUF is published by AngelSlim: [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF). Take the Q3_K files only. The model keeps the publisher's licence.

```bash
hf download AngelSlim/Hy4-flash-GGUF --include "Hy4-flash-Q3_K-*.gguf" --local-dir ./hy4
```

## Build the kernels (once)

```bash
bash tools/hy4/install_strix.sh
```

The script uses the repository venv (`EXL3_VENV`, default `.venv`), builds three libraries and prints `install_strix: OK`:

- `hipsp/libspmla.so`: sparse prefill attention (`SP_SO` points to another build).
- `~/.cache/exl3_gm_ext/exl3_gm_ext.so`: the MoE expert kernels (`EXL3_GM_BUILD_DIR` moves the build, `HY4_GM_EXT` points the server to it).
- `~/.cache/kyojin/ck-fmha/libckmla.so`: CK latent attention (`EXL3_CKMLA_OUT` moves the build, `EXL3_CKMLA_SO` points the server to it). The script fetches composable_kernel at the pinned commit once into `~/.cache/kyojin/ck-src`; `CK_SRC` names your own checkout.

The server stops at start when a built library is missing (the repository extension, the MoE extension or an attention library) and names the install step (`HY4_ALLOW_SLOW=1` starts on the slow path anyway). The kernel loaders stop the same way at run time.

Optional: a Q5_K copy of the output head for the MTP drafts (0.255 GB instead of 0.394 GB for Q8_0, which the server uses when no file is given). It needs the `libggml-base.so` of a llama.cpp build:

```bash
GGML_LIB=/path/to/libggml-base.so python tools/hy4/make_head_bin.py ./hy4/Hy4-flash-Q3_K-00001-of-00002.gguf ./hy4/head_q5_k.bin
```

The server finds `head_q5_k.bin` by itself when it sits next to the GGUF (`HY4_MTP_HEAD_BIN` names another place). The measured line uses this head: without the file the drafts run on the Q8_0 head, a little slower.

## Run

```bash
python tools/hy4/serve.py --model ./hy4/Hy4-flash-Q3_K-00001-of-00002.gguf --port 8300 -c 65536 --num-draft 3
curl http://127.0.0.1:8300/v1/chat/completions -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"no_think","max_tokens":64}'
```

Hy4-flash thinks first by default (`reasoning_effort` high): the reasoning is returned in `reasoning_content`, the answer in `content`, and a small `max_tokens` can end inside the reasoning with `content` empty. `"reasoning_effort": "no_think"` (or `"chat_template_kwargs": {"reasoning_effort": "no_think"}`) answers directly; otherwise leave room, for example `max_tokens` 1024.

Tool calling (1.9.2): `tools` / `tool_choice` / `tool_calls` / `tool` messages work as in the OpenAI API, streamed or not (README "Tool calling"). `python tools/hy4/tool_smoke.py --base http://127.0.0.1:8300` runs the served check. A client that closes a stream stops the generation. Not supported: `n` > 1, `response_format` `json_object` / `json_schema` (400, no constrained decoding), images and audio.

The 1.9.1 levers are listed with their rollback switches in the [env_vars.md](env_vars.md) table "Hy4-flash 1.9.1 levers". After updating from 1.9.0 run `bash tools/hy4/install_strix.sh` again (the Q3_K V2 kernel needs the rebuilt MoE extension; without it the old kernel runs).

The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, both with streaming; `/v1/models`, `/health`, `/tokenize`, `/apply-template`).
`tools/hy4/serve.py` runs itself under `tools/strix_halo/env.sh` (the HSA runtime torch needs on gfx1151) and sets the shipped kernel flags (`SHIPPED_ENV` in the file); every one can be overridden from the environment. The main ones:

| variable | shipped | effect |
|---|---|---|
| `HY4_LC_SPARSE` | 1 | exact top-k sparse attention in decode for long context |
| `HY4_PF_SPARSE`, `HY4_PF_SPARSE_MINN` | 1, 2048 | sparse prefill attention from 2048 keys |
| `HY4_PF_MM_SPLIT` | 8192 | dense prefill GEMMs of more than 8192 rows run in pieces of 8192 (`0` = off) |
| `HY4_NGRAM_EXT`, `HY4_NGRAM_MIN` | 4, 8 | prompt-lookup rows added to the MTP drafts when the last 8 tokens occurred before |
| `HY4_MTP_HEAD` | q5k | head of the MTP drafts |
| `HY4_MLA_HB` | 1 | decode attention: one kernel reads each K tile once for 16 heads (fp16 `tl.dot`); `HY4_MLA_HB_SPLIT` adds a key-split path for few-row steps |
| `HY4_KV_FP8` | 0 | opt-in fp8 e4m3 latent K cache of the full-attention layers for `-c 131072` / `-c 262144` (see "Long context option"); `HY4_LC_HB` (1) and `HY4_LC_HB_SPLIT` (-1) select the head-blocked sparse attend that reads it |
| `HY4_BG_DOT`, `HY4_HEAD_DOT` | 1 | k_b / v_b and output-head products of the verify rows on `tl.dot` (used from 5 rows for k_b / v_b, from 3 rows for the head) |
| `HY4_RG_F16` | 1 | fp16 resident router weights for the router mat-vec, fp32 accumulate |
| `HY4_KV_FUSED_MAXT` | 16 | the fused K-row / q-rope kernels take verify rows up to this count (was 1 row) |
| `HY4_PRIME_KVONLY` | 1 | the draft-cache fill at the end of prefill writes only K / V and the indexer key state (no queries, selection, attention or MoE) |
| `HY4_PF_G_SWIGLU`, `HY4_PF_G_ADD`, `HY4_PF_MM_FINE` | 1 | prefill glue: fused SwiGLU + cast, fused cast-add of the MoE layer tail, finer M-split of the dense GEMMs |
| `HY4_ENGRAM_NPHASH`, `HY4_ENGRAM_PF_WORKERS` | 1, 1 | Engram hash in numpy; one prefetch worker |
| `HY4_ENGRAM_NATIVE_IO` | 1 | Engram row reads in C threads (`pread`, no GIL) |
| `HY4_ENGRAM_FUSED` | 0 | two-launch fused Engram block in decode (opt-in) |
| `HY4_PF_HC2`, `HY4_PF_KV_HM`, `HY4_PF_QO_HM`, `HY4_PF_Q3K_V2` | 1 | 1.9.1 prefill: lazy layer residual, head-major K / V and Q / O operands of the CK core, Q3_K gate-up kernel V2 (`=0` restores 1.9.0) |
| `HY4_HEAD_Q8`, `HY4_GRAPH_FORK`, `HY4_RG_TUNE`, `HY4_BG_V2` | 1 | 1.9.1 decode: Q8_0 output head, shared-expert branch on a second graph stream, router mat-vec K block 1024, absorb v2 (`=0` restores 1.9.0) |
| `HY4_PF_ROUTER16` | 1 | fp16 router logits of prefill rows; inactive with the fused router (`HY4_REF_FUSED=1`, shipped), so no effect |
| `HY4_ABLIT_SPEC` | unset (off) | `on` loads the bundled `presets/hy4-uncensor/edit_spec.json`; a path loads that `edit_spec.json` (tensor file next to it): run-time decensor edit of the attention and MLP / MoE outputs; `0`, empty or `off` = no edit |

Native Engram IO: `exllamav3/loader/engram_io.c` is compiled with `gcc` at first use into `~/.cache/kyojin/engram_io` (`XDG_CACHE_HOME` moves it). When `gcc` is missing or the build fails, the server prints one line and uses the Python reader, which does the same reads more slowly (`HY4_ENGRAM_NATIVE_IO=0` selects it directly).

`--num-draft` is the number of MTP drafts per round (3 by default). Memory: the K buffers are sized by `-c` (about 0.1 MB per position); see Supported context for the measured totals.

The start-up warm-up prompt is clamped to the context window (and skipped with one log line when the window is too small), so a small `-c` starts normally.

## Supported context

64K tokens (`-c 65536`, the command above). Speed and passkey retrieval are measured on the release build at 4K, 16K, 32K and 64K. Longer prompts are refused with a context-length error.

Memory, measured on a 128 GB Strix Halo machine with nothing else large resident (free memory in GiB = MemAvailable, GTT = GPU memory used through the shared pool; one launch per `-c`, then 60 s idle, one large request, 60 s idle):

| `-c` | free at load (min) | free idle | free during the largest prefill | free after | GTT peak |
|---|---|---|---|---|---|
| 16384 (12K prompt) | 20.8 | 20.8 | 20.9 | 27.1 | 97.9 |
| 32768 (28K prompt) | 17.9 | 19.1 | 18.3 | 23.7 | 99.7 |
| 65536 (64K prompt, default) | 16.1 | 16.0 | 15.1 | 20.9 | 102.7 |

The free memory is steady, not a load transient: at the default `-c 65536` it stays at about 16 GiB, and 15 GiB during a 64K prefill (the page cache holds 8 to 10 GiB at ready). A smaller `-c` leaves more. Close other GPU or memory-heavy apps first.

128K (`-c 131072`) is opt-in. At load, fp32 KV, the server reached 110.4 GiB of GTT and the free memory fell to 7.8 GiB. It was not measured under a 128K prefill on the release build; if memory runs out, use a smaller `-c`.

### KV bytes per token (persistent buffers at full capacity, `python tools/hy4/kv_bytes.py`)

| component | 1.9.1 default | `HY4_KV_COMPACT=1` |
|---|---|---|
| latent K, 12 trunk full-attention layers | 27648 (fp32 2304 each) | 7728 (fp8 644 each) |
| latent K, MTP draft layer | 2304 | 644 (`HY4_KV_FP8_MTP`) |
| 30 window layers | 69120 | 0 (ring of 1024 rows, 71 MB constant, `HY4_SWA_RING`) |
| indexer pool, 13 layers (one 128-d fp32 key per 4 tokens) | 1664 | 1664 |
| indexer value / gate rows, 13 layers | 13312 | 0 (ring, 54 MB constant) |
| token history, arange, rope tables | 1296 | 1296 |
| total | 115344 B (112.6 KB) | 11332 B (11.1 KB) + 125 MB |

The default row matches the measured slope of the memory table above (98 to 118 KB per token). `HY4_KV_COMPACT=1` is a shortcut that sets `HY4_KV_FP8=1`, `HY4_KV_FP8_MTP=1` and `HY4_SWA_RING=1024` unless you set them yourself; add `--draft-fill-rows 4096` to the serve command so the draft states of a long prompt (fp32 [n, 3072], 12 KB per token) are not kept. Transients not in the table: one prefill chunk of indexer scores (`chunk x n / 4 x 4` bytes) and, on the fp8 cache, one fp16 dequantised copy of one layer's cache for the sparse prefill (1152 bytes per token, freed per layer). Left to the 10 KB mark: the indexer pool in fp16 (-832) and the rope tables (1280). Measured: `tools/hy4/kv_bytes_probe.py`.

### Long context option: fp8 KV (`HY4_KV_FP8=1`, off by default)

Default (`-c 65536`): fp32 latent K cache, the measured quality. For `-c 131072` and `-c 262144`, switch the fp8 cache on: `HY4_KV_FP8=1 python tools/hy4/serve.py ... -c 131072`. The 12 trunk full-attention layers keep their latent K in e4m3 (512 bytes) + one fp32 scale per token + bf16 rope dims: 644 bytes per token and layer instead of 2304. The MTP layer and the 30 window layers stay fp32. The sparse decode kernel decodes e4m3 itself; prefill and the other paths read a dequantised copy of the keys of the call. With the fp8 cache the head-blocked sparse attend (`HY4_LC_HB`, default 1: one key decode shared by 16 heads, split-K with `HY4_LC_HB_SPLIT=-1` auto) is on; it never runs on the fp32 cache. `HY4_LC_HB=0` returns to the one-head kernel; `HY4_KV_FP8=0` (or unset) returns to the fp32 cache.

Latent K cache, analytic (12 trunk full-attention layers in fp8 or fp32, the MTP layer in fp32, 30 window layers at 1024 rows; the 84.3 GiB of weights and 3 GiB of graph pools and activations are not included):

| `-c` | fp32 KV | fp8 KV |
|---|---|---|
| 65536 | 1.9 GiB | 0.7 GiB |
| 131072 | 3.7 GiB | 1.3 GiB |
| 262144 | 7.4 GiB | 2.5 GiB |

Estimated total memory with sparse prefill (chunk 8192 at 128K and 256K, fp32 indexer pool, full draft states): about 92 GiB at `-c 131072` and 97 GiB at `-c 262144` with fp8 KV. With fp32 KV and dense prefill the same model gives 101 and 114 GiB, and it read 100.9 GiB for the 128K row that measured 110.4 GiB of GTT: read the estimates as a lower bound. Quality, measured on the lane tree: fp8 KV + head-blocked attend, long-text NLL change -0.0025 +- 0.019 against a noise null of 0.0263. Not yet measured on this build: served speed, memory peak and passkey at 128K and 256K.

## Uncensor preset (opt-in)

Off by default: the server runs the unmodified forward. The preset bundled in `presets/hy4-uncensor/` is a run-time edit of the attention and MLP / MoE outputs and changes no weights. Switch it on:

```bash
HY4_ABLIT_SPEC=on python tools/hy4/serve.py --model ./hy4/Hy4-flash-Q3_K-00001-of-00002.gguf --port 8300 -c 65536 --num-draft 3
```

`on` (also `1`, `preset`, `true`, `yes`) loads the bundled preset, found from the repository from any working directory. `HY4_ABLIT_SPEC=<other path>/edit_spec.json` loads another spec. Unset, `0`, an empty value or `off` installs no edit. The start-up log prints `decensor preset on (HY4_ABLIT_SPEC=...)` or `decensor preset off`.

Known limit: the preset was validated in no_think mode (judge NO 2-4 of 17 on the development build). In the default thinking mode some answers run long (2 of 17 hit the 4096-token cap in our check); a fix is in progress. Details and files: [presets/hy4-uncensor](../presets/hy4-uncensor/README.md).

## Known limits

- One request at a time; other requests queue. `--slots N` (several sessions in one batched step) is experimental, see [HY4_MULTI_USER.md](HY4_MULTI_USER.md).
- Decoding is greedy: sampling parameters are accepted and ignored. Speculation returns the same tokens as plain greedy decoding.
- Context above 64K is opt-in (128K) and not measured on the release build; above 128K is not part of this release.
- The Engram tables (59 GiB) are read from disk by row, so the first requests after a start depend on storage speed.
- One model and one quantization (Q3_K). Other Hy4 quantizations are not tested.
