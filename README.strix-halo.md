# exllamav3 on AMD Strix Halo (gfx1151)

> **Just want to run a model?** Start with the [README](README.md) and [doc/install.md](doc/install.md). This page keeps the kernel notes of the gfx1151 port.

> **This tree adds AMD GPU functionality to ExLlamaV3 that upstream does not have.** Upstream is
> CUDA-only; the `sdougbrown/exllamav3` base adds a ROCm/HIP decode path for **gfx12 (RDNA 4)**
> only. Everything here extends that to **gfx11.5 (RDNA 3.5)** and tunes it for a 20-CU iGPU on
> unified memory. NVIDIA/CUDA paths are untouched — all changes sit behind `USE_ROCM` / arch guards.

Working tree for running exllamav3 on the AMD Ryzen AI Max+ 395 / Radeon 8060S
(RDNA 3.5, gfx1151, wave32, 128 GB unified memory) — the Framework Desktop.

Primary repo: GitHub `vcruz305/exllamav3-amd`.

Based on [sdougbrown/exllamav3](https://github.com/sdougbrown/exllamav3) branch `integration`
(gfx12 WMMA path) at `991f1a0`. Everything on top of that is in the commits after it.

## What's here

**Kernel / runtime (in `exllamav3/`):**
- **gfx1151 WMMA GEMV path** — RDNA3.5 `v_wmma_f16_16x16x16_f16` decode kernel behind the
  gfx12 dispatcher, C-fragment layout oracle-verified.
- **Int8-activation GEMV HIP port** (`exl3_gemv_int8*.cu[h]`) — `dp4a` → `v_dot4_i32_iu8`,
  register-staged B pipeline replacing `cp.async`, cooperative launch verified on gfx1151.
  Parity-proven on real tensors (argmax 1.00). **Default off on ROCm** (`EXL3_INT8_GEMV=1|2`).
- **SAD trellis decode** on gfx1151 (`v_sad_u8` gate was gfx12-only): −3–4 % kernel time.
- **CPU expert offload** port (`moe_cpu_host.py`, `block_sparse_mlp*.py`, `moe_handoff_amdgpu.cu`):
  loads and runs; measured net loss on unified memory. Default off (`-mcs 0`).
- **Build-time kernel variants** via `EXL3_HIP_DEFINES` (setup.py): `EXL3_HIP_PF_AFTER_STAGE`,
  `EXL3_MUL1_DECODE_DOT4`, `EXL3_HIP_A_RING`. All measured null (±2 %) — see below.
- Env knobs: `EXL3_MOE_CFG`, `EXL3_HIP_PREFILL_MIN_ROWS`, `EXL3_HIP_SKINNY_GEMM`, `EXL3_HIP_GR_MIX_Q8`,
  `EXL3_HIP_GR_MIX_ROWS`, `EXL3_MOE_HIP_WAIT_FINE`,
  `EXL3_INT8_GEMV_BLOCKS_PER_CU`, `EXL3_GEMV`, `EXL3_GEMV_DEBUG`, `EXL3_MOE_*_PROF`.
- Host-side decode trims (read per call; **default off**): `EXL3_HOST_LEAN` (1 = batched greedy
  verify accept + pinned draft buffers, +0.31 % served) and `EXL3_HOST_CUTS` (1 = one fused cast
  instead of cast+contiguous in the KDA conv1d decode, memoized `g_tensor_cache` bucketed views,
  `to2` dtype early-out, persistent MLA verify-row kernel `out`, persistent
  `exl3_dec_moe_union` scratch tables; only reachable together with `EXL3_HOST_LEAN=1`, and
  value-identical: greedy ids 128/128 x 3 and MTP accept unchanged). Together: served GLM-5.3
  MTP n1f2 30.53 -> 30.69 t/s (+0.54 +- 0.07 %, n=24, two loads), decode launches/round
  1362 -> 1328, aten op calls/round 5114 -> 4426, 59.8 -> 59.4 ms/round. Host-submit idle is only
  2.10 ms of a 59.8 ms round (GPU 91 % busy), so this whole lever is capped at 3.6 % even at
  -100 % launches.
- Decode MoE union knobs, R-row verify (read per call; defaults = legacy behavior):
  `EXL3_MOEDEC1_RM3` (**0 = off**; 1 = R=3 union uses RM=3 instead of RM=4; bitwise equal.
  Served R=3 union -5.8 %, round -1.08 ms, +1.46 +- 0.06 % t/s, moedec1),
  `EXL3_MOEDEC1_KBS_A` (0 = forced-6 default; 2..32 = union-only gu k-split override;
  kbs=1 invalid (LDS overflow) and rejected. 2/3 beat 6 by ~3 % at R3U21 but regroup the
  cross-block sum, so ids diverge: needs a PPL gate before serving).
- Prefill knobs (read per call/launch; defaults = fastest measured): `MPW_LDM` (1), `EXL3_KDA_FUSED_CAST` (1),
  `EXL3_RECURRENT_LAST_PAGE` (1; 0 = one-shot speed, loses the prompt-end checkpoint),
  `EXL3_PREFILL_BIG_CHUNK` (4096; chunk size while > big + chunk tokens remain, keeps every tail checkpoint; 0 = off), `EXL3_DSA_LATENT`
  (1 = chunked-QK GLM sparse-prefill kernel, 0 = one-shot `_dsa_attn_kernel`, `BH:BN:NW:DC[:KV1]` = tile sweep; KV1 = one full-width key load, slower: spills),
  `EXL3_KDA_FUSE` (2 = w/u/kg and output fused into the KDA state loop, `kda_fused_h.py`; 1 = w/u/kg only; 0 = fla chain;
  chunks < 512 tokens always use the fla chain; bitwise equal to 0),
  `EXL3_MLA_MHA` (1 = mha2 MLA prefill kernel; 0 = old, bit-exact with logs-r28), `EXL3_KDA_TUNE`
  (1 = 32:4:4:1; `32:2:4:1` = old inter_solve, bit-exact with logs-r28). Smoke reference for the
  current defaults: `logs-r35/both.json`.
- mHC prefill knobs (read per call): `EXL3_HC_1PASS_PF` (1 = one-pass hc_mix_norm ticket kernel for
  prefill rows; default 0), `EXL3_HC_FN_HALF_PF` (1 = prefill mix reads the fp16 `fn` copy;
  default 0), `EXL3_PF_HC_FUSE` (**0 = off**; 1 = `ext.hc_pf_apply_mix_norm`, one kernel for a
  sublayer's residual `hc_apply` plus the next site's `hc_mix_norm_1pass`, so the (R, 4, D) stream
  stack is written once and read once instead of twice. Prefill rows only, in the single-chunk
  regime (R > 256 and `hc_mix_num_chunks == 1`); bit-identical to the unfused pair. GLM 4K prefill
  +1.46 +- 0.48 % (n=10/arm), 16K +1.58 +- 0.22 % (n=4/arm), pffuse1).

**Harnesses (`tools/strix_halo/`):** `bench_mtp.py` (end-to-end decode), `gemv_micro.py`
(per-shape kernel timing vs. roofline), `gemv_by_m.py`, `int8_parity.py`, `moe_layer_prof.py`,
`split_prof.py`, `roofline.py`, WMMA oracles, and `env.sh` (venv + build-mode SDK env).

## Results — Qwen3.8-Flash-Next-EXL3 (3.05 bpw), decode

| config | tok/s |
|---|---|
| stock gfx12 path (reconstruct + hgemm fallback) | ~17 |
| WMMA GEMV path | 27.9 |
| + `EXL3_MOE_CFG=2` | 30.2 |
| + `EXL3_HIP_PREFILL_MIN_ROWS=2`, `-ndt 2 -dds` | **32.7 mean / 38.7 peak** |
| + int8 GEMV (`EXL3_INT8_GEMV=2`) | 31.7–31.9 (noise) |
| + CPU offload `-mcs 16/64/128` | 29.7 / 28.2 / 27.0 (loss) |
| + `EXL3_HIP_STG_PAD` (LDS bank conflict fix) | **34.9 mean / 41.1 peak** |
| + skinny fp16 GEMM for GDN b/a proj (`hgemm.cu`, replaces hipblaslt split-K) | 36.1 mean / 44.4 peak |
| + **int8 GatedResidual mixer weights** (`gr_mix_q8`, −22 % of decode bytes) | **40.0 mean / 46.3 peak** (ndt=2) · **41.3 mean / 43.7 peak** (ndt=3 dc=0.6) |

Six-prompt greedy means, 512 tokens (`prompt_sweep.py`). PPL 4.225935 unchanged through every
row. Per-round phase split (`phase_prof.py`): trunk verify forward 85–88 %, MTP head 10–12 %,
host ≈ 1 % — decode is bytes-per-forward bound. Byte budget per trunk forward after int8
mixers (`bytes_by_module.py`): routed experts 47 %, dense EXL3 GEMVs 40 %, mixers 13 %.
Every bulk kernel is now at this part's practical single-kernel streaming rate (~135 GB/s,
`read_bw_vs_size.py`): grouped-MoE prefill GEMV 131–135 GB/s on unique-expert bytes, lm_head
~200 GB/s, dense GEMVs 55–65 % of the 236 GB/s DRAM peak. `EXL3_BLOCK_GRAPH` (graph replay)
measured neutral under MTP, and stays neutral at 16 and 64 slots (the old 4-slot default was
not the limiter: 44.51 / 44.47 vs 44.72 off, bit-identical output).

PPL 4.2259, identical across all configs. Roofline 236 GB/s.

### Cost of verify width — and why single-stream stops near 47

`width_ndt_batch.sh` sweeps `-ndt` (static) and derives ms-per-forward as
`wall / (tokens − accepted)`. At `-ndt k` the verify forward carries k+1 rows:

| cfg | rows | ms/fwd | tok/fwd | tok/s |
|---|---|---|---|---|
| no MTP | 1 | 40.6 | 1.00 | 24.6 |
| ndt=1 | 2 | 51.5 | 1.94 | 37.6 |
| **ndt=2** | **3** | **59.1** | **2.75** | **47.3** |
| ndt=3 | 4 | 69.0 | 3.05 | 44.1 |
| ndt=4 | 5 | 77.9 | 3.44 | 44.1 |
| ndt=5 | 6 | 86.1 | 3.39 | 39.4 |

Linear fit: **ms/forward = 32.4 + 9.04 × rows.** Two consequences, both of which correct
earlier analysis in this file:

- **Extra verify rows are not free.** Only 47 % of a 4-row forward is row-independent; each
  added row costs 9.0 ms because routed-expert bytes scale with row count (more positions →
  more *unique* experts of 512 touched). So **tree / wide speculation is a dead end here**: a
  16-row tree costs 177 ms and would need ~7.8 accepted tokens per forward just to match
  today's 44 tok/s. No drafter delivers that, and MTP acceptance already decays to 47.8 % at
  depth 5.
- **Static `-ndt 2` is the single-stream optimum on a favourable prompt** (47.3 / 47.0
  measured twice, 87.6 % acceptance) — better than the `-ndt 3 -dds -dc 0.6` production point
  (44.1–44.7). Re-sweep the six-prompt mean before changing a default; dynamic drafting still
  wins on the hard prompts it was tuned for.

An earlier claim here that the verify forward runs "at ~95 GB/s effective, remaining gap
spread across ~1,300 launches" was arithmetically wrong: it is **78 GB/s** (4.9 GiB / 67 ms),
and 60 tok/s single-stream at 3.05 tok/fwd would need 107 GB/s — *below* the 135 GB/s
single-kernel roof, so bandwidth alone never excluded it. What actually excludes it is the
9.04 ms marginal row cost plus acceptance decay: the two trade against each other and the
product peaks at ~47.

### Aggregate throughput: 60+ tok/s is available today via batching

Because 32.4 ms of every forward is row-independent, **concurrent sequences share it**.
`batch_throughput.py` (synthetic best case: fp16 cache, raw prompts, no stop conditions, all
sequences the same length so the batch never drains; ndt=2, greedy, 512 tokens/seq):

| batch | tok/s per seq | **aggregate tok/s** | acceptance | GiB |
|---|---|---|---|---|
| 1 | 47.0 | 47.0 | 87.6 % | 54.3 |
| 3 | 23.4 | **70.3** | 86.9 % | 54.3 |
| 4 | 19.9 | **79.4** | 84.2 % | 54.3 |
| 5 | 17.2 | **86.1** | 83.2 % | 54.3 |

The `32.4 + 9.04 × rows` model predicts these within 6–10 % for batch ≤ 4 (it over-predicts
above that: per-sequence attention and page-table work grow with concurrency, which a
row-only fit does not capture). Memory is flat — the cache is the only per-sequence cost, so
a 16 k context per stream is affordable at these batch sizes.

**Realistic sustained numbers** (the recipe's `batch.sh`: 16-prompt queue so the batch stays
full, qwen35 chat template, stop conditions, Q4 cache, greedy, 512 max tokens):

| batch | rows | aggregate tok/s |
|---|---|---|
| 1 | 3 | 34.2 |
| 4 | 12 | 66.1 |
| **5** | **15** | **74.0 / 75.1** (two runs) |
| 6 | 18 | **56.1** ⚠ |
| 8 | 24 | 68.0 |
| 10 | 30 | 72.7 |
| 11 | 33 | 70.2 |

Two operational rules fall out of that table:

- **The 16-row rule.** `MOE_PREFILL_ROWS_PER_CHUNK = 16`: the grouped-MoE kernel processes
  expert rows in chunks of 16, and a batch of B sequences at `-ndt K` submits `B*(K+1)` rows
  in one forward. Choose B so `B*(K+1)` lands on or just under a multiple of 16. At ndt=2
  (3 rows/seq) **batch 5 = 15 rows = one full chunk is the peak, and batch 6 = 18 rows costs
  24 %** — a full chunk plus a nearly-empty second one. Predicted from the constant, then
  confirmed; the same dip appears in the synthetic sweep (batch 6 = 59.2 between 4 = 79.4 and
  8 = 69.4), so it is the kernel granularity and not noise. Batch 11 (33 rows) dips mildly
  for the same reason.
- **Queue deeper than batch.** Aggregate throughput needs the batch kept full. Four prompts
  at `-b 4` drains as sequences finish and yields ~51 tok/s; sixteen prompts at `-b 5`
  sustains 74–75.

**So: single stream is latency-limited at ~47 tok/s, but the GPU has ~2.2× more throughput
available and batching is how you collect it.** For a chat UI serving one person the 47
number is the one that matters; for an agent fan-out, a batch queue, or any multi-request
endpoint, 74 tok/s is real and needs no kernel work. `-cq 4` is what makes several long
contexts fit concurrently.

### Static vs dynamic draft sizing: not resolvable, keep the default

A single favourable prompt made static `-ndt 2` look like a 6 % win (47.3 vs 44.7). On the
six-prompt distribution it is not, and that gap was the cherry-pick this file warns about.
`prompt_sweep.py` now takes `NDT` / `DC` / `DDS`; `ndt_sweep_batch.sh` runs the four corners:

| config | reps | mean of means | six-prompt min |
|---|---|---|---|
| ndt=3 dynamic dc=0.6 (**production default**) | 3 | 42.25 | 38.00 |
| ndt=3 static | 3 | 42.53 | 38.59 |
| ndt=2 dynamic dc=0.4 | 1 | 41.14 | 35.44 |
| ndt=2 static | 1 | 40.09 | 34.64 |

Static ndt=3 is +0.28 tok/s over the default, but run-to-run drift *within* the default
config is 1.30 tok/s — the effect is 0.22× the noise, so it is **not resolvable** and the
default stands. `-ndt 2` static is genuinely worse (−2.2 mean, −3.4 on the worst prompt).
Methodological note for future A/Bs here: the six-prompt mean drifts >1 tok/s between
identical runs, so any claim under ~2 tok/s needs repeats before it means anything.


Rejected as speed levers (measured, not assumed): **2.05 bpw pack** — 25.7 tok/s at ndt=3
(vs 44.1 at 3.05 bpw) despite 79 % acceptance; K=2 trellis tiles this part badly, so lower
bits go the wrong way. **fp32-out-via-fp16 at decode row counts** (`MIN_ROWS=1`) — 44.0 vs
44.8; that fix only pays at prefill widths.


**m=3 GEMV kernel** (25 % of decode wall, ~30 % of roofline): counter profile showed 36 % VALU
issue, 17 % LDS, 144 GB/s — stall-bound, not issue- or bandwidth-bound. The three loop-structure
variants (B-prefetch-after-stage, DOT4 decode, A-fragment prefetch ring) are all within ±2 % of
baseline; A-ring is slightly negative (VGPR pressure). The per-slice `s_waitcnt vmcnt(0)` is
**not** the bottleneck.

**RESOLVED by `EXL3_HIP_STG_PAD`** (+10 % mean decode). The stall was an LDS bank conflict in
the WMMA staging buffer, findable by arithmetic rather than counters. `hip_mma_stg` was
`[warps][32][4]` of `__half2`; `__half2` is one dword and RDNA LDS has 32 banks of one dword
(`bank = dword_addr % 32`), so the innermost extent *is* the per-lane bank stride, and 4 divides 32:

| access | banks | conflict |
|---|---|---|
| `stg[lane][slot]`, lane 0..31 | `lane*4 % 32` → 8 banks | 4-way |
| `stg[4*(R&7)+j][s]`, R = lane&15 | `4*(R&7)*4 % 32` → {0,16} | **8-way** |

Padding the extent to 5 (`gcd(5,32)=1`) spreads stores over all 32 banks and cuts loads to 2-way.
PPL and draft acceptance unchanged, so the gain is kernel time. Any `__shared__` array of
one-dword elements here needs an innermost extent coprime with 32.

**Do not shrink `HIP_MMA_STG_WARPS`** to match `EXL3_MOE_CFG=2`'s 4 warps: the non-MoE GEMV paths
still launch 16 warps and index this buffer by warp id, so a 4-warp buffer gives NaN logits —
and `eval/ppl.py` still reports the correct 4.2259 on that broken build. Only `bench_mtp.py`
catches it.

## Build

```bash
source tools/strix_halo/env.sh build            # SDK toolchain from .venv-gfx1151
EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD" pip install -e . --no-build-isolation --no-deps
source tools/strix_halo/env.sh                  # runtime env (LD_PRELOAD Ubuntu HSA)
python tools/strix_halo/bench_mtp.py -n 128 -ndt 2 -dds -g
```

Build and bench in **separate shells** — `env.sh build` exports `ROCM_PATH`/`HIP_PATH`/
`PYTORCH_ROCM_ARCH` and a bench in the same shell fails model load with a spurious
"Insufficient VRAM".

Known transient: hipblaslt OOM at model load right after a previous process exits — sleep and retry.
