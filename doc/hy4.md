# Hy4-flash on Strix Halo (GGUF, one machine)

Kyojin loads the public Hy4-flash GGUF (Q3_K split, 2 files, 144 GiB) and serves it on one Strix Halo machine with ROCm. Prefill runs the sparse-attention
fast path; decode runs a HIP graph with multi-token-prediction (MTP) drafts and prompt-lookup rows. Speed numbers: see the benchmark table in the release notes.

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
     -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

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
| `HY4_BG_DOT`, `HY4_HEAD_DOT` | 1 | k_b / v_b and output-head products of the verify rows on `tl.dot` (used from 5 rows for k_b / v_b, from 3 rows for the head) |
| `HY4_RG_F16` | 1 | fp16 resident router weights for the router mat-vec, fp32 accumulate |
| `HY4_KV_FUSED_MAXT` | 16 | the fused K-row / q-rope kernels take verify rows up to this count (was 1 row) |
| `HY4_PRIME_KVONLY` | 1 | the draft-cache fill at the end of prefill writes only K / V and the indexer key state (no queries, selection, attention or MoE) |
| `HY4_PF_G_SWIGLU`, `HY4_PF_G_ADD`, `HY4_PF_MM_FINE` | 1 | prefill glue: fused SwiGLU + cast, fused cast-add of the MoE layer tail, finer M-split of the dense GEMMs |
| `HY4_ENGRAM_NPHASH`, `HY4_ENGRAM_PF_WORKERS` | 1, 1 | Engram hash in numpy; one prefetch worker |
| `HY4_ENGRAM_NATIVE_IO` | 1 | Engram row reads in C threads (`pread`, no GIL) |
| `HY4_ENGRAM_FUSED` | 0 | two-launch fused Engram block in decode (opt-in) |

Native Engram IO: `exllamav3/loader/engram_io.c` is compiled with `gcc` at first use into `~/.cache/kyojin/engram_io` (`XDG_CACHE_HOME` moves it). When `gcc` is missing or the build fails, the server prints one line and uses the Python reader, which does the same reads more slowly (`HY4_ENGRAM_NATIVE_IO=0` selects it directly).

`--num-draft` is the number of MTP drafts per round (3 by default). Memory: the K buffers are sized by `-c` (about 0.1 MB per position); see Supported context for the measured totals.

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

## Known limits

- One request at a time; other requests queue. `--slots N` (several sessions in one batched step) is experimental, see [HY4_MULTI_USER.md](HY4_MULTI_USER.md).
- Decoding is greedy: sampling parameters are accepted and ignored. Speculation returns the same tokens as plain greedy decoding.
- Context above 64K is opt-in (128K) and not measured on the release build; above 128K is not part of this release.
- The Engram tables (59 GiB) are read from disk by row, so the first requests after a start depend on storage speed.
- One model and one quantization (Q3_K). Other Hy4 quantizations are not tested.
