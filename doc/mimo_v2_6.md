# MiMo-V2.6-Flash-RL on ExLlamaV3 — build, quantize, serve

This branch (`mimo-v2.6-flash`) runs **XiaomiMiMo/MiMo-V2.6-Flash-RL** — a 309B-total /
15B-active MoE — as an EXL3 quant on a single machine with ~100 GB of GPU memory, with the
model's own shipped **DFlash** drafter for speculative decoding.

The published quant is 2.27 bpw (converter figure), 83.5 GiB of weights, wikitext-2 ppl
**5.4003** (64 × 2048), about **30 tok/s** decode on an NVIDIA DGX Spark (GB10, sm_121,
aarch64, 121.6 GiB unified memory). Most other numbers in this runbook (32K decode, DFlash
speedups, benchmarks, memory budgets) were measured on the previous 2.36 bpw / 86.2 GiB build,
which had layer 47's experts at 6 bpw; see §7. That build gave **31.5 tok/s** at 2K, **28.5**
at 32K, and **1.38–1.78×** on top from the drafter on code and reasoning. It also builds and runs on x86 (verified on H100 and
RTX PRO 6000).

---

## 1. What is on this branch

Base: upstream `turboderp-org/exllamav3` **master `6b84a21`**, plus:

| what | upstream PR | needed for |
| --- | --- | --- |
| aarch64 build guards — x86 AVX intrinsics fenced off, CPU MoE offload and the native TP CPU reduce stubbed, arm yield spin hint (`a85b1f2`, authored by *jarvis* / [vcruz305](https://github.com/vcruz305/exllamav3), cherry-picked unmodified) | — | building at all on aarch64; a no-op on x86 |
| `MiMoV2ForCausalLM`, text-only: E8M0 block scales as uint8, pluggable fused-tensor readers, optional `v_head_dim`, and the 39 sliding-window layers on `SlidingAttention`'s window ring | [#399](https://github.com/turboderp-org/exllamav3/pull/399) | the architecture |
| DFlash `(left, right)` attention window + variant switches for attention sinks, value scale and a learned mask embedding | [#396](https://github.com/turboderp-org/exllamav3/pull/396) | the drafter |
| drop the shards' page cache after load, and an `EXL3_LOAD_DEVICE` escape hatch | [#398](https://github.com/turboderp-org/exllamav3/pull/398) | loading 86 GiB on unified memory without starving the box |
| per-layer fp32 MoE intermediates (`EXL3_MIMO_FP32_MLP_LAYERS`, default off) | *not upstreamed*, MiMo-specific | an escape hatch only; layer 47's overflow is fixed by `interm_div` in #399 (§8) |

Plus, in this branch only and not in any PR: this runbook, `util/prepare_dflash_draft.py`,
`util/memguard.py`, `util/pagecache.py` and `examples/mimo_v2_6/`.

Converting needs nothing beyond #399. An earlier quant used `convert.py --module_bits` and
`--max_bad_rows` ([#397](https://github.com/turboderp-org/exllamav3/pull/397), now closed) to get
past layer 47's fp16 overflow. The real fix is `interm_div` on that layer (§8).

The upstream architecture-support issue is
[#124](https://github.com/turboderp-org/exllamav3/issues/124).

### What is not supported

* **Text only.** No vision, no audio (`vision: false`).
* **MTP drafting.** The checkpoint's `num_nextn_predict_layers: 3` head is not ported. Use
  DFlash instead — it is faster anyway.
* **Tensor parallel.** `supports_tp = False` for this architecture.
* **Batch > 1 with DFlash** is untested (single-GPU batch 1 is what was measured).

---

## 2. Quick start

```sh
# 1. code
git clone -b mimo-v2.6-flash https://github.com/benthecarman/exllamav3
git clone -b mimo-v2.6-flash https://github.com/benthecarman/tabbyAPI

# 2. environment — see §3 (aarch64/GB10) or §4 (x86)

# 3. weights: the EXL3 quant, and the original repo for its dflash/ subdirectory
hf download <the-exl3-quant-repo> --local-dir /data/hf/exl3/mimo-2.25bpw-hq
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --local-dir /data/hf/MiMo-V2.6-Flash-RL \
    --include 'dflash/*'

# 4. stage the drafter (§5)
python exllamav3/util/prepare_dflash_draft.py \
    --src /data/hf/MiMo-V2.6-Flash-RL/dflash \
    --out /data/hf/exl3/mimo-dflash-draft

# 5. serve (§6)
TABBY_DIR=$PWD/tabbyAPI DRAFT_DIR=/data/hf/exl3/mimo-dflash-draft \
  exllamav3/examples/mimo_v2_6/serve.sh /data/hf/exl3/mimo-2.25bpw-hq
```

---

## 3. Building on GB10 / DGX Spark (aarch64, sm_121)

There are no aarch64 CUDA wheels for `torch` on PyPI and no prebuilt ExLlamaV3 wheel for
sm_121, so this is a from-source, JIT-compiled install. It takes about two minutes of
compiling, once.

```sh
# --- CUDA 13.2 toolkit, laid out as a classic CUDA_HOME -----------------------
# nvcc must be >= 13.0 to accept compute_121, and its CUDA version should match the
# torch wheel's. On NixOS: github:graham33/nixos-dgx-spark#cuda gives nvcc 13.2.78
# split across ~40 store paths; symlink-farm them into one prefix with bin/, include/,
# lib/, lib64 -> lib, nvvm/. On Ubuntu/DGX OS the stock /usr/local/cuda-13.2 is fine.
export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

# --- torch: the upstream aarch64 CUDA wheel, version-matched to that nvcc -------
uv venv --python 3.13
source .venv/bin/activate
uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132
#   -> torch 2.12.1+cu132 + triton 3.7.1 + numpy. Download only, no compilation.
#   Plain PyPI has no CUDA aarch64 build, hence --index-url.

# --- GB10 is sm_121 ------------------------------------------------------------
# torch 2.12.1's own get_arch_list() stops at sm_120 and exllamav3's
# maybe_set_arch_list_env() clamps to it, so without this the extension is built for
# 12.0 and runs on GB10 only through PTX/family compatibility.
export TORCH_CUDA_ARCH_LIST=12.1a
export MAX_JOBS=20

# --- exllamav3, editable, JIT ---------------------------------------------------
export TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions
EXLLAMA_NOCOMPILE=1 uv pip install -e ./exllamav3 --no-build-isolation
#   EXLLAMA_NOCOMPILE keeps setup.py from precompiling; exllamav3/ext.py JIT-builds
#   into TORCH_EXTENSIONS_DIR on first import.

# --- build the extension now rather than on the first request -------------------
python -c "import exllamav3; from exllamav3.ext import exllamav3_ext"
#   181 translation units -> ~336 MiB .so. 1 min 46 s wall with MAX_JOBS=20.
#   Rebuilds are a ninja no-op.
```

Notes, all of them things that cost time the first time:

* **`--no-build-isolation` matters.** An isolated build environment cannot see your torch and
  will try to fetch an x86 one.
* **Do not use the `[cu13]`-style extras** if you have installed torch by hand; they pin a
  torch version and a prebuilt exllamav3 wheel.
* **`nvcc` 13.2 + gcc 15.3 needs no `-allow-unsupported-compiler`.** nvcc 13.2 supports host
  GCC up to 15.
* **flash-attn is not needed.** Upstream master does not use it; `attention_fn/dispatch.py`
  prefers its own Triton paged/varlen kernels and falls back to torch SDPA.
* **NixOS only:** triton's NVIDIA backend locates `libcuda.so.1` by shelling out to
  `/sbin/ldconfig`, which does not exist. `export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`
  fixes the first Triton kernel launch. Wheel-bundled `libcuda.so.1` / `libnvidia-ml.so.1`
  lookups need `programs.nix-ld` with the NVIDIA driver in its library set.

## 4. Building on x86

Nothing special — the normal upstream path, against this branch:

```sh
git clone -b mimo-v2.6-flash https://github.com/benthecarman/exllamav3
cd exllamav3
uv venv
uv sync --extra cu130        # or cu128 / cu129 / cu132, matching your CUDA build
```

or, into an environment that already has a suitable torch (>= 2.6, CUDA >= 12.4):

```sh
pip install --no-build-isolation -e .
```

The aarch64 guard commit is inert here (the guarded code is compiled exactly as upstream
compiles it on x86). Nothing else on the branch is architecture-specific. Verified on H100
(sm_90) and RTX PRO 6000 (sm_120).

You need a GPU (or a coherent-memory pool) that can hold **~86.2 GiB of weights + KV cache +
~2.8 GiB for the drafter** — so ~100 GB in practice, e.g. 2× A100 80GB, 2× H100, an RTX PRO
6000 96GB at short context, or a GB10/GH200-class unified-memory box.

---

## 5. Staging the DFlash drafter

The `dflash/` subdirectory of the original checkpoint **cannot be loaded as shipped**.
`util/prepare_dflash_draft.py` produces a directory that can be:

```sh
python util/prepare_dflash_draft.py \
    --src /data/hf/MiMo-V2.6-Flash-RL/dflash \
    --out /data/hf/exl3/mimo-dflash-draft
```

It is cheap — nothing is copied but the small files; the 2.94 GB tensor file is symlinked —
and it fixes four things:

1. **`config.json` is not valid JSON.** It ends with a trailing comma. The script repairs it
   in the copy. (This is a known packaging bug; there is an open discussion about it on the
   Hub repo.)
2. **`tap_shift = 0`, not the default `+1`.** ExLlamaV3's `DFlashConfig` defaults to the
   convention the original z-lab checkpoints were trained with. MiMo's own reference
   (`extract_context_feature`, `hidden_states[i+1]`) means "the output of layer i", which is
   ExLlamaV3's export index `i`. SGLang's `dflash_utils.py::build_target_layer_ids` documents
   the same thing. Getting this wrong silently halves acceptance.
3. **The learned mask embedding lives in a `mask_embedding.pt` pickle.** It is rewritten as
   `mask_embedding.safetensors` so the loader finds it. It is *not* recoverable from the
   target: `mask_token_id` 151675 is past the end of the tokenizer and the target's
   `embed_tokens[151675]` is an untrained pad row (L2 norm 2e-5, against the shipped vector's
   0.765). Without it the drafter is fed zeros for all seven mask slots and degenerates to one
   repeated token.
4. **Three declared switches the shipped reference ignores** are restated in the keys
   ExLlamaV3 reads: `attention_value_scale` (0.612, folded into `o_proj.weight_scale`),
   `attention_sink_bias` (per-head, 5 layers × 64), and `bidirectional_block` from
   `is_causal: false` — which becomes a window of `(left = 1024, right = block_size - 1 = 7)`.
   That last one is [#396](https://github.com/turboderp-org/exllamav3/pull/396): a bare int
   window normalises to `(left, 0)`, so `causal = False` was being cancelled and the drafted
   block came out causal rather than bidirectional.

All four corrections were cross-checked against **SGLang main** (the engine Xiaomi's README
recommends), which honours `partial_rotary_factor`, the value scale, the sink bias and the
shipped mask embedding exactly this way. The `dflash/dflash.py` in the checkpoint predates all
of it.

Useful flags: `--force` (rebuild in place), `--rope-full` (rotate all 128 head dims the way
the shipped `dflash.py` accidentally does — an ablation, not a fix), `--literal` (stage the
drafter the way `dflash.py` actually runs it, for parity harnesses), `--taps 0,1` (slice the
drafter down to a toy target's layer count).

The result loads in **1.3 s** and costs **2.81 GiB** resident (BF16 — no quantization needed)
plus **20.0 KiB/token** of draft KV (1.25 GiB at 64k).

---

## 6. Serving

ExLlamaV3 ships no HTTP server, so serving goes through **TabbyAPI**, driven against an
editable install of this checkout.

```sh
git clone -b mimo-v2.6-flash https://github.com/benthecarman/tabbyAPI
```

That branch carries exactly one patch on top of upstream: `backends/exllamav3/model.py`
honours **`EXL3_LOAD_DEVICE`** (§8), which is what makes an 86 GiB model load on unified
memory. Everything else is stock.

Install it **base dependencies only**:

```sh
cd tabbyAPI
uv pip install "fastapi-slim>=0.115" "pydantic>=2.11,<3" ruamel.yaml rich "uvicorn>=0.28.1" \
    "jinja2>=3.0.0" loguru "sse-starlette>=2.2.0" packaging "tokenizers>=0.21.0" numpy \
    aiofiles aiohttp async_lru huggingface_hub psutil "httptools>=0.5.0" pillow requests setuptools
uv pip install uvloop       # aarch64 only, see below
```

> **Never `pip install .[cu12]` / `.[cu13]`.** Those extras pin torch 2.9/2.11 *and* a
> prebuilt `exllamav3` wheel, which clobbers both your torch and the editable checkout. On
> aarch64 every one of those pins is gated behind `platform_machine == 'x86_64'` and would
> silently install nothing — do not rely on that.

> **aarch64 gotcha.** TabbyAPI's `pyproject.toml` marks `uvloop` as
> `platform_system == 'Linux' and platform_machine == 'x86_64'`, but `main.py` imports it
> unconditionally on non-Windows. `uv pip install uvloop` is the entire fix (0.22.1 has an
> aarch64 manylinux wheel). Nothing else about TabbyAPI needs patching.

Then either copy and edit **`examples/mimo_v2_6/tabby-config.yml`** and run
`python main.py --config <it>`, or use the wrapper:

```sh
TABBY_DIR=/path/to/tabbyAPI \
DRAFT_DIR=/data/hf/exl3/mimo-dflash-draft \
MAX_SEQ_LEN=65536 MAX_BATCH_SIZE=1 PORT=8080 \
  ./examples/mimo_v2_6/serve.sh /data/hf/exl3/mimo-2.25bpw-hq
```

`serve.sh` refuses port collisions and low memory, generates the YAML, exports
`EXL3_LOAD_DEVICE=cuda:0`, and wraps TabbyAPI in `util/memguard.py` (§9). `FOREGROUND=0`
backgrounds it. Full knob list in the script's header.

### The config that matters

| key | value | why |
| --- | --- | --- |
| `model_dir` / `model_name` | parent dir + subdirectory name | TabbyAPI's `model_dir` is a *directory of models* |
| `max_seq_len` / `cache_size` | `65536` | 1.86 GiB of KV at one slot; see §10 |
| `cache_mode` | `FP16` | `8,8` works and is coherent, but only halves the paged term — the 39 SWA layers' ring is always fp16 |
| `max_batch_size` | `1` | each extra slot costs another 175.5 MiB SWA ring + paged span + draft cache |
| `vision` | `false` | text-only port |
| `reasoning` + `<think>` / `</think>` | on | TabbyAPI splits the block into OAI `reasoning_content`, blocking and streaming |
| `prompt_template` | **unset** | so TabbyAPI reads `<model_dir>/chat_template.jinja`, which `convert.py` copies out of the source checkpoint |
| `draft_mode` | `model` + `draft_model_dir`/`draft_model_name` | the staged DFlash drafter |
| `dynamic_draft` | `true` | see below |

TabbyAPI logs, on the real model:

```
Using template "chat_template" for chat completions.
Response parsing: tool format qwen3_coder (template, tokenizer), reasoning tags <think> </think> (config)
```

Tool calling is auto-detected as **`qwen3_coder`** and verified end to end against the real
model (`finish_reason: tool_calls`, `parsed 1 tool call (qwen3_coder)`). Thinking is switched
off per request with `"chat_template_kwargs": {"enable_thinking": false}`.

### Speculative decoding: what to expect

Batch 1, greedy, 512-token cap, on the previous 2.36 bpw quant:

| prompt | no draft | DFlash, static 7 | DFlash, **dynamic** | acceptance (static → dynamic) |
| --- | --- | --- | --- | --- |
| coding | 28.16 t/s | **44.41** (1.58×) | 38.90 (1.38×) | 74.1% → 81.1% |
| reasoning | 31.29 t/s | 53.18 (1.70×) | **55.66** (1.78×) | 60.4% → 67.5% |
| prose | 31.11 t/s | 20.72 (**0.67×**) | 27.19 (0.87×) | 12.4% → 50.4% |

**`dynamic_draft: true` is the serving default**: it gives up ~12% of the coding peak for a
far better worst case and a better mean (1.35× vs 1.31×). Even so, prose is a **slowdown** —
for a pure prose workload, drop `draft_model` entirely. There is no per-request switch.

**n-gram drafting is not worth it on this model** (0.87–0.98× on all three prompts, 1.8–24.7%
acceptance): it does not repeat itself enough.

One caveat before anyone diffs outputs: greedy generation is **not** token-identical across
drafting modes on the prose prompt. n-gram differs from no-draft too, so this is fp16
tie-break sensitivity (a single divergence at a top1–top2 margin of 0.0156 ≈ two fp16 ULP,
reproduced identically with `swa_full`), not a DFlash bug.

---

## 7. Quantization (if you are making your own)

The published quant is `-b 2.25 -hq`. The converter reports "final bitrate (excluding head)
2.27", 83.5 GiB over 12 shards, ~7.3 h on one RTX PRO 6000 or ~9.2 h on one H100, peak 21.1 GiB
of GPU memory and 36.4 GiB host RSS. Per-MoE-layer cost is 537 s (RTX PRO 6000) / 684 s (H100);
the dense layer 0 is ~25 s. A 32 GB sm_120 card can do the whole job.

```sh
python convert.py -i <source> -o <out> -w <work> -b 2.25 -hq
```

No extra flags. With `interm_div` on layer 47 (§8), none of the 250 calibration rows go
non-finite. A quant made before that change is not compatible with this branch: `interm_div` is
folded into layer 47's `up_proj` weights at conversion time.

---

## 8. Environment knobs

| variable | default | what it does |
| --- | --- | --- |
| **`EXL3_LOAD_DEVICE`** | unset | e.g. `cuda:0` — take `Model._load_single` instead of `_load_autosplit`. `_load_autosplit`'s per-module `reusable = mem_get_info().free + reserved - allocated` check reads **MemFree**, not MemAvailable, so on unified memory a growing page cache makes a model that comfortably fits look unloadable. With it, the 86 GiB model loads in **21–23 s**; without it, it can be refused outright ("Insufficient VRAM in split"). Single device only; ignored under tensor parallel. |
| **`EXL3_KEEP_PAGE_CACHE`** | `0` | `1` keeps the shards' page cache after load. Off by default: `Model.load_gen()` calls `SafetensorsCollection.drop_page_cache()` at the end of every load and logs ` -- Released 86.1 GiB of shard page cache`. `stloader.cpp` reads shards with buffered `pread`, not mmap, so an 86 GiB load otherwise leaves 86 GiB of clean page cache in the *same pool* as the weights. |
| **`EXL3_MIMO_FP32_MLP_LAYERS`** | `"none"` | Escape hatch, off by default. Layers whose MoE/MLP intermediates are built with `interm_dtype = torch.float`. Resolution order: this variable, then `config.json` → `exl3_fp32_mlp_layers`, then the built-in default. Syntax: `47`, `40-47`, `0,47`, `-1`, `all`, `none`/`off`/`""`. An explicit spec naming a missing layer is a hard error; the default is filtered silently. Prints ` -- MiMoV2: fp32 MLP intermediates on layer(s) [...]` at load. Drops the fused prefill tier on those layers. |
| **`EXL3_MIMO_FP32_MLP_FUSED`** | `0` | `1` keeps the fp16 fused prefill tier (`exl3_moe`) on the overridden layers. **A/B timing only.** That kernel hard-requires fp16 gate/up intermediates (`TORCH_CHECK_DTYPE(..., kHalf)`), and at 2048 tokens × top-8 over 256 experts nearly every expert falls under its 256-row cap, so leaving it on silently reverts the fix. |
| **`EXL3_MIMO_MLP_ACT_LIMIT`** | `0` (off) | Clamp `act_fn(g)` and `u` to ±limit on the overridden layers, in every tier including the ≤8-row fused **decode** kernel — whose fp16 store does not saturate and is the one path fp32 alone cannot bound. Use ≤ 255 (`limit² ≤ 65504`); 128 was tested. |
| `-swa_full` / `swa_full=True` | off | CLI flag on the eval/example scripts, `Model.from_config(swa_full = ...)` in code: run the 39 sliding-window layers on a **full-length paged cache** instead of the fixed ring. Restores pre-ring behaviour for debugging. It costs **261.00 KiB/token instead of 27.00**, i.e. 9.2× more KV at 128k, and drops max context from 980k to 102k. Only useful for A/B. |

### Layer 47 and `interm_div`

On this checkpoint, layer 47's routed experts push `act(gate) * up` to about 84k on some number
tokens (mostly expert 208 channel 18 and expert 70 channel 1387), past the fp16 max of 65504.
`gate` and `up` on their own stay under 2.3k, and no other layer gets above 3k. The HF/SGLang
reference runs bf16 there and stays finite.

Layer 47 is built with `interm_div = 128`: `up_proj` is scaled by 1/128 at conversion and
`routed_scaling_factor` puts the 128 back in fp32. The peak becomes about 660 and the fused
kernels stay on. On the converter's 250 calibration rows the previous quant (no `interm_div`,
fp16) gave non-finite logits on 33 rows; this one gives 0.

fp32 intermediates are not a fix for this. The fused prefill tier stores fp16 regardless, and
without that tier the activation kernel clamps the product to 65504 instead of computing it.
That is why `EXL3_MIMO_FP32_MLP_LAYERS` is now off by default.

---

## 9. Unified memory: the hazard, and `util/memguard.py`

**Read this if your GPU memory and your host memory are the same bytes** (GB10 / DGX Spark,
GH200, and any other coherent-memory part).

There is no separate VRAM. A large host allocation and a large device allocation compete for
one pool, and when that pool is exhausted **the kernel does not OOM-kill anything** — it
starves, the NVIDIA driver logs `NV_ERR_NO_MEMORY`, and recovery is a physical power cycle,
not a reboot. That is not hypothetical: it happened once during this work, at 03:15, loading
this model.

The root cause and the fix are both interesting:

* `exllamav3_ext/stloader.cpp` opens each shard with `fopen(..., "rb")` and reads it with
  buffered `pread`. No mmap. So loading 86.1 GiB of shards leaves **86.1 GiB of clean page
  cache** behind, in the same physical pool the 86.1 GiB of weights now occupy. 121.6 GiB
  total, ~86 GiB claimed twice.
* Nothing in-process can see it. `torch.cuda.mem_get_info()` reports **MemFree**, not
  MemAvailable. That is also why the load was refused one level up, in `_load_autosplit`.
* Symptom before death: row times climbing 2.4 s → 3.8 s → **12.4 s** (reclaim thrash).

Three things address it, and all three are on this branch:

1. `Model.load_gen()` drops the shards' page cache at the end of every load
   (`EXL3_KEEP_PAGE_CACHE=1` opts out).
2. `EXL3_LOAD_DEVICE=cuda:0` bypasses the MemFree-based budget arithmetic entirely.
3. **`util/memguard.py`** — an external supervisor, because 1 and 2 only help a process that
   has already got far enough to run its own code:

```sh
python util/memguard.py --floor-gib 8 --fadvise-dir /data/hf/exl3/mimo-2.25bpw-hq \
    --fadvise-every 5 --label tabbyapi -- python main.py --config tabby-config.yml
```

It spawns the command in its **own process group**, polls `/proc/meminfo` every 0.5 s,
`SIGKILL`s the whole group when MemAvailable falls below the floor and says why (exit 137),
and refuses to start if it is already below the floor (exit 3). `--fadvise-dir` additionally
`posix_fadvise(DONTNEED)`s that directory's shards every 5 s, which is what holds the page
cache flat **during** the load, before any in-process hook could run — and it works for any
loader, TabbyAPI included. `examples/mimo_v2_6/serve.sh` wraps the server in it by default.

`util/pagecache.py` is the same machinery as a library and a CLI: `evict_dir()` and
`resident_bytes()` (mmap + `mincore(2)`, i.e. what `vmtouch`/`fincore` report).

With all of it in place: **shard pages resident 0.00 GiB of 86.08 GiB while serving**, row
times flat to ±0.03 s over 146 consecutive 2048-token rows, MemAvailable flat to 0.1 GiB.

If you run another inference server on the same box (SGLang, vLLM), **stop it first**. An
86 GiB model and a second engine cannot share 121 GiB, and any systemd unit you write for this
should declare `Conflicts=` against the other one.

---

## 10. Memory budget

48 layers = **9 global-attention** + **39 sliding-window** (window 128), `head_dim` 192,
4 KV heads GA / 8 SWA, `v_head_dim` 128 riding the cache zero-padded to 192. The SWA ring is
768 tokens per layer per slot.

| | paged KV / token | fixed ring / slot |
| --- | --- | --- |
| ring (default), bf16 KV | **27.00 KiB** | **175.5 MiB** |
| ring, `cache_mode: 8,8` | 13.50 KiB | 175.5 MiB (unchanged) |
| `swa_full`, bf16 KV | 261.00 KiB | — |
| **DFlash drafter, `DRAFT_RING=1`** (default) | **0** | **35.0 MiB** |
| DFlash drafter, `DRAFT_RING=0` | 20.00 KiB | — |

The drafter's own K/V used to be the second-largest per-token line item — 20.00 KiB/token, i.e.
a 74% surcharge on top of the target's 27.00 — even though all five of its layers are
`sliding_attention` with a 1024 window and none of them ever looks further back. They now run
on a per-slot ring of window + block + two pages = 1792 tokens, so **draft K/V is a constant
35.0 MiB per slot at any context**: 2,560 MiB → 35.0 MiB at `max_seq_len` 131072, 7,680 MiB →
35.0 MiB at 393216. `DRAFT_RING=0` restores the old behaviour for A/B.

Weights 86.20 GiB, reserve 10 GiB on a 121.63 GiB box ⇒ **KV budget 25.43 GiB**:

| | 1 slot | 4 slots |
| --- | --- | --- |
| ring, bf16 KV | **980,736 tok** | 240,128 |
| ring, q8 KV | 1,048,576 (capped by `max_position_embeddings`) | 480,256 |
| `swa_full`, bf16 | 102,144 | 25,344 |

32k costs 1.02 GiB at one slot; 64k costs 1.86 GiB. The ring is what makes this possible —
without it, 128k needs 32.6 GiB.

Measured at 32K context, one slot, FP16 cache, drafter attached:

| | GiB |
| --- | --- |
| MemTotal | 121.63 |
| **peak in use** | **95.06** |
| min MemAvailable seen | 26.57 |
| torch peak allocated | 86.98 |
| model load time | 21.4 s (≈ 3.9 GB/s off NVMe) |

Steady state while serving at 64k with the drafter: **MemAvailable 21.2 GiB**. Note that
TabbyAPI's host RSS oscillates ~2 GiB under sustained load (4.0 idle → 8.0 peak → 5.6 GiB) and
MemAvailable tracks it inversely; that is allocator churn, not a leak, but on a box with
~21 GiB of headroom it is most of the margin.

---

## 11. Measured quality

Measured on the previous 2.36 bpw build (not re-run on the current 2.27 bpw one, whose
wikitext-2 ppl is 5.4003), greedy (temperature 0), zero-shot, through the server, against the
full sets:

| benchmark | score |
| --- | --- |
| wikitext-2 ppl (64 × 2048) | **5.3615** (5.3174 over the whole 146-row test set) |
| HumanEval+ pass@1, n=164 | **89.63%** plus / 92.07% base |
| MBPP+ pass@1, n=378 | **77.51%** plus / 88.10% base |
| GSM8K, n=500 | **95.8%** |
| MMLU-Pro, n=500 stratified | 72.6% — a floor; 11 items truncated at a 1536-token cap |
| GPQA-Diamond, n=64 | 46.9% — a **budget artefact**: 29 of 64 hit an 8,192-token thinking cap and scored zero; **85.7%** among the 35 that finished |

Over 1,606 completed generations / 574,407 generated tokens: **0 API errors, 0 non-finite
artefacts, 0 empty code solutions**, one immediate-EOS response. Nothing traced to the layer-47
overflow.

Greedy is a deliberate deviation from the card's recommended temp 1.0 / top_p 0.95 — one
sample at temp 1.0 has enough variance to swamp a quantization delta — so these absolutes are
not the model's best-effort scores and should not be set beside anyone else's published
numbers. Note also that Xiaomi publishes no standard-benchmark numbers for MiMo-V2.6 at all
(the technical report's only results table is entirely agentic), so there is no official
baseline to subtract from.

---

## 12. Files added by this branch

| path | what |
| --- | --- |
| `doc/mimo_v2_6.md` | this file |
| `util/prepare_dflash_draft.py` | stage the shipped DFlash drafter (§5) |
| `util/memguard.py` | external memory supervisor (§9) — generic, not MiMo-specific |
| `util/pagecache.py` | page-cache eviction / residency measurement (§9) — generic |
| `examples/mimo_v2_6/serve.sh` | ready-to-copy serving wrapper |
| `examples/mimo_v2_6/tabby-config.yml` | ready-to-copy TabbyAPI config |
