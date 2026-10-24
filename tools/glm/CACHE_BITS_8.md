# GLM-5.3 `--cache-bits 8`: measured on the Strix Halo

Measurements for the packed int8 MLA latent cache added in `32951fa` ("Add `--cache-bits 8`").
Default fp16 behaviour is unchanged (verified: step 8).

## Environment and commits

| item | value |
|---|---|
| box | AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151), 128 GB GTT, 124 GB RAM |
| repo (box) | `~/engines/kyojin` (the plan said `~/kyojin`) |
| branch | `feat/glm-kv-q8` |
| feature commit | `32951fa` ("Add `--cache-bits 8`"), parent (pre-change) `18cdd3c` |
| HEAD at measurement | `9eb14ce` (`c105ba7` + the harness commit below) |
| harness commit | `9eb14ce` — `--cache-bits`/`--temp` in `tools/strix_halo/bench_mtp.py`, new `tools/glm/niah.py` |
| env | `source tools/strix_halo/env.sh` from the repo root, before every run |

**Model pack substitution.** The plan's pack `~/models/glm53-exl3-td205` is **not on the box**.
Per instruction, a symlink was created to the available GLM-5.3 EXL3 pack:

```
ln -s /data/hf/hub/models--NicSir--GLM-5.3-Flash-EXL3-2.72bpw/snapshots/db672fc5e61f1d61bdf6d641fb63538b80bb7613 \
      ~/models/glm53-exl3-td205
```

Every number below is for the **NicSir GLM-5.3-Flash-EXL3-2.72bpw**.
The MLA-relevant dims are identical across the available GLM-5.3 packs, so the cache byte accounting is unaffected;
the throughput/quality numbers are specific to this pack.
MTP `eh_proj` sidecar not present, draft acceptance is lower than a pack that ships `mtp_eh_proj.st`.
Acceptance numbers are relative to this pack.

## Step 0 — config dims

```
num_hidden_layers        45
kv_lora_rank             512
qk_rope_head_dim         0        (NoPE; the rope-key plane is empty)
index_head_dim           128
index_kpool              4
index_n_heads            32
index_topk               2048
num_nextn_predict_layers 1
layer_types              linear_attention x3 then deepseek_sparse_attention, repeated
                         -> 11 MLA/DSA layers at indices 3,7,11,...,43
indexer_types            all 45 = "full"  -> every MLA layer carries its own indexer + k-pool planes
```

## Step 1 — byte accounting

### 1.1 Analytic (per token, per MLA layer, per plane)

| plane | fp16 B/token | int8 B/token | notes |
|---|---|---|---|
| latent (packed qk + scales) | `2*512 = 1024` | `(512/32)*8*4 + (512/32)*2 = 512 + 32 = 544` | **the only quantised plane** |
| rope key | `2*0 = 0` | `0` | NoPE, plane empty |
| DSA indexer key `k_idx` | `2*idx_plane_dim = 2*256 = 512` | `512` | stays fp16 (out of scope) |
| k-pool `k_pool` | `2*(128/4) = 64` | `64` | stays fp16 (out of scope) |
| **per MLA layer** | **1600** | **1120** | −30 % (1.43x) |

`idx_plane_dim = index_head_dim*2 = 256` because `index_kpool` is set; all 11 MLA layers are
`indexer_mode == "full"`, so every one carries the indexer and k-pool planes.

Totals (main cache = 11 MLA layers, MTP draft cache = 1 layer):

| cache | fp16 B/token | int8 B/token |
|---|---|---|
| target (11 layers) | 17600 | 12320 |
| MTP draft (1 layer) | 1600 | 1120 |
| **target + draft** | **19200 (18.75 KiB)** | **13440 (13.125 KiB)** |

### 1.2 Empirical (`tools/glm/cache_bytes.py`, no weight load)

`sum(layer.storage_size())` at `max_num_tokens = ctx + 4096`:

| comp | cache_bits | layers | raw bytes | B/token |
|---|---|---|---|---|
| target | 0 | 11 | 792,985,600 | 17600.0 |
| target | 8 | 11 | 555,089,920 | 12320.0 |
| mtp | 0 | 1 | 72,089,600 | 1600.0 |
| mtp | 8 | 1 | 50,462,720 | 1120.0 |

Plane shapes at `-c 40960` (`pages = 45056/256 = 176`), identical for target and draft:

```
fp16  shape_c (176,256,1,512)  shape_r (176,256,1,0)  shape_i (176,256,256)  shape_p (176,64,128)
int8  qshape  (176,256,128) i32  sshape (176,256,16) f16  shape_r (176,256,1,0)
      shape_i (176,256,256)  shape_p (176,64,128)
```

**Reconciliation:** measured == analytic exactly. There is **no** rounding gap because every
layer is a full indexer, and `max_num_tokens` is a multiple of `PAGE_SIZE` (256), so pages are
exact. The only slack is the `+4096` tokens added to `ctx` by `serve.py`.

### 1.3 Extrapolation (linear in `max_num_tokens = ctx + 4096`; **not measured** at these sizes)

| `-c` | max_num_tokens | fp16 | int8 | saved |
|---|---|---|---|---|
| 16384 | 20480 | 375.0 MiB | 262.5 MiB | 112.5 MiB |
| 40960 | 45056 | 825.0 MiB | 577.5 MiB | 247.5 MiB |
| 131072 | 135168 | 2475.0 MiB (2.42 GiB) | 1732.5 MiB (1.69 GiB) | 742.5 MiB |
| 524288 | 528384 | 9675.0 MiB (9.45 GiB) | 6772.5 MiB (6.61 GiB) | 2902.5 MiB (2.83 GiB) |

This is a 30 % cut of the paged pool. The pool is small next to the ~110 GB of weights, so the
absolute saving only matters at very long context — and it is the only lever that reduces the
`-c 40960` working set (relevant on this 128 GB box, which OOM-crashed once during this task).

**Decision input:** the indexer key (512 B/token, 46 % of the int8 layer) and the k-pool plane
(64 B/token) stay fp16 and dominate the remaining int8 layer. Quantising them is a separate,
larger task; the 30 % here is the whole latent plane only.

## Step 2 — PPL

`tools/glm/ppl7.py` (10 x 4096 rows), run twice with its own (identical) command — it has no
cache flag:

| run | ppl | seconds |
|---|---|---|
| 1 | 2.868039 | 167.1 |
| 2 | 2.868039 | 87.4 |

**Absolute delta 0.000000, relative 0.00 %; per-row PPL lists bit-identical.** This is by
construction: `glm_base.eval_ppl` forwards with `attn_mode="flash_attn_nc"`, a cacheless path
(`exllamav3/modules/mla_attn.py:1467` asserts `cache is None`), so the protocol cannot exercise
the latent quantiser. The corpus `~/bench/ppl/wiki.test.raw` was **absent**; it was rebuilt with
the deterministic `tools/glm/make_corpus.py` (400000 chars, 82174 tokens), so this PPL is not
comparable to earlier wikitext numbers (the delta conclusion is unaffected).

## Step 3 — Retrieval (needle-in-haystack)

`tools/glm/niah.py` against the server (greedy, `temperature 0`, `-c 40960`, `--num-draft 1`,
`--no-vision`), 40083-token prompts, needle at 10 / 50 / 90 %, 3 trials each, identical seeds:

| depth | fp16 recall | int8 recall |
|---|---|---|
| 10 % | 3/3 | 3/3 |
| 50 % | 3/3 | 3/3 |
| 90 % | 3/3 | 3/3 |
| **total** | **9/9** | **9/9** |

## Step 4 — Speed and MTP

Trimmed per instruction: **no 16K**, **3 runs total per cell** (fp16/int8 order alternated by
cell). `tools/strix_halo/bench_mtp.py -n 200 -ndt 1` (MTP depth 1 = GLM's nextn depth),
one process at a time, no OOM (122+ GB free throughout).

| ctx | temp | MTP | fp16 tok/s | int8 tok/s | Δ | accept fp16/int8 |
|---|---|---|---|---|---|---|
| 4096 | greedy | off | 20.43 | 19.78 | −3.2 % | — |
| 4096 | greedy | on | 25.22 | 24.46 | −3.0 % | 85.2 / 85.2 |
| 4096 | T=1.0 | off | 20.42 | 19.77 | −3.2 % | — |
| 4096 | T=1.0 | on | 20.91 | 22.84 | * | 53.6 / 72.4 |
| 40960 | greedy | off | 20.41 | 19.77 | −3.1 % | — |
| 40960 | greedy | on | 25.16 | 24.43 | −3.0 % | 85.2 / 85.2 |
| 40960 | T=1.0 | off | 20.41 | 19.73 | −3.3 % | — |
| 40960 | T=1.0 | on | 18.78 | 21.62 | * | 38.5 / 63.9 |

`*` T=1.0 cells are stochastic (acceptance and reply length vary) and were single-sample for
int8, so their deltas are not meaningful. In the deterministic cells the intra-mode run-to-run
spread is <0.1 %, so the **≈ −3 % int8 decode penalty is real**, with **identical draft
acceptance (85.2 %)**. Context (4K vs 40K) does not change the picture.

## Step 5 — Knob sweep (triggered: int8 slower beyond noise)

16K, greedy, MTP on, int8, `EXL3_MLA_QC_BN` / `EXL3_MLA_QC_TRANS` exported per process
(medians; BN=32 and the default were run 5x, others 2x):

| BN | TRANS | tok/s | accept | n |
|---|---|---|---|---|
| (unset, tuned default) | unset | 24.46 | 85.2 | 5 |
| **32** | unset | **24.86** | **86.9** | 5 |
| 64 | unset | 24.42 | 85.2 | 2 |
| 64 | 1 | 24.45 | 85.2 | 2 |
| 64 | 0 | 23.70 | 85.2 | 2 |
| 128 | unset | — | — | `OutOfResources(131072, 65536, 'shared memory')` |
| 256 | unset | — | — | `OutOfResources(262144, 65536, 'shared memory')` |
| fp16 reference | — | 25.20 | 85.2 | 2 |

- `EXL3_MLA_QC_BN=32` is a small but **reproducible** win: +1.6 % decode (24.86 vs 24.46) and
  +1.7 pp acceptance, with every BN=32 run above every default run.
- `EXL3_MLA_QC_TRANS=0` is **worse** (−3.1 %); `TRANS=1` ≈ default.
- `BN >= 128` **fails at runtime** on gfx1151 (shared-memory OutOfResources) — not usable here.

**Engine default left unchanged.** The win is small (1.6 %), measured on one pack on one box, and
`EXL3_MLA_QC_BN` is a shared MLA kernel knob (it also governs the Qwen path). It is documented
here and in `SERVE.md` as an opt-in env setting; changing the tuned kernel default is left to a
follow-up if wanted.

## Step 6 — Behaviour checks

Server, greedy, fixed prompts; each server's `serve: cache=` line noted.

| # | check | result |
|---|---|---|
| 1 | same prompt twice, MTP on, `--sessions 1`, int8 | **identical text** |
| 2 | MTP on vs off, greedy, int8 | **diverge at token 25 of 62** ("a long series" vs "a series"); 40 % prefix |
| 3 | `--sessions 2`, two identical prompts sent together vs solo | the two batched rows are **identical to each other**, but both differ from the solo run from **char 130 of 345** |
| 4 | `--exact-prefix-cache` + int8 | cold `cache_n=0` (13.8 s) -> resumed `cache_n=4729` (3.0 s), **identical text** |
| 5a | slot same-mode round-trip (`tools/slot_check.py`, int8) | **PASS**: `n_saved=4608`, `n_read=4608`, `warm_cache_n≥4608`, `same_text=True` |
| 5b | slot cross-mode, fp16 file -> int8 server | **HTTP 400** `ValueError: slot file cache layout does not match the loaded cache` |
| 5c | slot cross-mode, int8 file -> fp16 server | **HTTP 400** (same) |

The cross-mode rejection is the existing loud guard in `slot_store.py:143-145` (the int8 MLA
layer adds `qk`+`sk` where fp16 has `k`, so the per-tensor count/dtype set differs). **No code
change was needed** for step 7.1 — it already fails loudly, in both directions.

Check 3 confirms the row-invariance caveat: with `--sessions > 1` the packed int8 pages are not
row-invariant, so batching changes the greedy text (the rows still agree with each other).

## Step 7 — Fixes

1. **Slot mode guard** — none needed; confirmed loud (6.5b/c). Documented instead.
2. **Row-invariance caveat (docs)** — the old "does not apply to GLM" wording is gone; `SERVE.md`
   now states what step 6 showed (rows agree with each other, differ from solo).
3. **`--cache-bits` help + startup log** — help now notes slot files do not cross cache modes;
   the startup line prints `serve: cache=int8 (slot files do not cross cache modes)`.
4. **Startup warning** — `--cache-bits 8` with `-sessions > 1` prints one extra stderr line.
5. **Docs + report** — `tools/glm/SERVE.md` and `doc/models.md` re-add the measured paragraph /
   line; this file is the report.

## Step 8 — Regression guard (default fp16 unchanged)

`git worktree add /tmp/kyojin-base 18cdd3c`, `cp exllamav3_ext*.so /tmp/kyojin-base/`; serve.py
at HEAD (`9eb14ce`) and at `18cdd3c`, both fp16 (default), MTP off (`--num-draft 0`), `-c 16384`,
`--no-vision`, same `--model-id`; the same `/v1/completions` request:

```
EQUAL True
HEAD: ' atomicity, consistency, and isolation. When we say a transaction is atomic, ...'
BASE: ' atomicity, consistency, and isolation. When we say a transaction is atomic, ...'
```

The fp16 path is unchanged by construction (`qkw = {}` when `cache_bits == 0`) and this is the
empirical confirmation: the response text is byte-for-byte identical.

## Validation

```bash
python -m pytest -q tests/test_mla.py tests/test_mla_dsa.py tests/test_kv_quant.py tests/test_cache8_roundtrip_cpu.py
python -m pytest -q tools/glm/test_serve.py tools/test_serve_compat.py tools/test_startup_health.py
```

Results on the box (pytest 9.1.1 installed to `/tmp/cb8/pylibs`; the venv has no pytest):

| suite | result |
|---|---|
| `tests/test_mla.py` + `tests/test_mla_dsa.py` + `tests/test_cache8_roundtrip_cpu.py` | pass |
| `tests/test_kv_quant.py` | **28 failed**, all `cache_size=32768`, all `triton.compiler.errors.CompilationError` in `_quant_cache_paged8_kernel` (`IncompatibleTypeErrorImpl('invalid operands of type pointer<int32> and triton.language.int32')`) |
| combined | 99 passed, 28 failed, 1 skipped |
| `tools/glm/test_serve.py` | 37 passed |
| `tools/test_serve_compat.py` + `tools/test_startup_health.py` | 51 passed, 4 skipped |

The 28 failures are **pre-existing and unrelated**: `32951fa` touches only
`tools/glm/serve.py` + docs (no kernel or test code), and the same case
(`test_kv_quant[8-32768-2-256-block_table_size0-cuda:0]`) fails identically at the pre-change
commit `18cdd3c` with the same Triton compile error. It is a Triton 3.7.0+rocm7.13 codegen issue
in the generic MHA paged-8-bit writer for the largest page count, not in the MLA quant path
(`--cache-bits 8` uses `mla_kv_quant_append`, which compiles and runs — the step 3/4/6 numbers
above come through it). No GLM-path regression.
