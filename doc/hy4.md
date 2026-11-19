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
python tools/hy4/serve.py --model ./hy4/Hy4-flash-Q3_K-00001-of-00002.gguf --port 8300 -c 131072 --num-draft 3
curl http://127.0.0.1:8300/v1/chat/completions -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, both with streaming; `/v1/models`, `/health`, `/tokenize`, `/apply-template`).
`tools/hy4/serve.py` runs itself under `tools/strix_halo/env.sh` (the HSA runtime torch needs on gfx1151) and sets the shipped kernel flags (`SHIPPED_ENV` in the file); every one can be overridden from the environment. The main ones:

| variable | shipped | effect |
|---|---|---|
| `HY4_LC_SPARSE` | 1 | exact top-k sparse attention in decode for long context |
| `HY4_PF_SPARSE`, `HY4_PF_SPARSE_MINN` | 1, 16384 | sparse prefill attention from 16K keys |
| `HY4_PF_MM_SPLIT` | 8192 | dense prefill GEMMs of more than 8192 rows run in pieces of 8192 (`0` = off) |
| `HY4_NGRAM_EXT`, `HY4_NGRAM_MIN` | 4, 8 | prompt-lookup rows added to the MTP drafts when the last 8 tokens occurred before |
| `HY4_MTP_HEAD` | q5k | head of the MTP drafts |

`--num-draft` is the number of MTP drafts per round (3 by default). Memory: the K buffers are sized by `-c` (about 0.1 MB per position).

## Supported context

128K tokens (`-c 131072`). Longer prompts are not supported in this release.

## Known limits

- One request at a time; other requests queue. `--slots N` (several sessions in one batched step) is experimental, see [HY4_MULTI_USER.md](HY4_MULTI_USER.md).
- Decoding is greedy: sampling parameters are accepted and ignored. Speculation returns the same tokens as plain greedy decoding.
- Context above 128K is not part of this release.
- The Engram tables (59 GiB) are read from disk by row, so the first requests after a start depend on storage speed.
- One model and one quantization (Q3_K). Other Hy4 quantizations are not tested.
