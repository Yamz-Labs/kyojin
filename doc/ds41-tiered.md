# DS4.1 decode with a RAM hot set and an NVMe cold tier

Branch `feat/tier1-ds41` = `feat/ds41` + `perf/ds41-mixedk` + the tiered store (`exllamav3/modules/expert_store.py`).
Tests are CPU only: `tests/test_expert_store_cpu.py`, `tests/test_expert_store_mixedk_cpu.py` (real `MixedKStore` over shell
linears, bytes of every picked expert checked through the K-group tables). **Nothing here ran on a GPU**: worker (the only
allowed GPU) has been offline since 16:40. No DS4.1 routing data exists yet, so every hit-rate number below is a model.

## What was ported

* **Per-expert K**: a layer is split into tier groups, one per `(K_gate_up, K_down)` (the grouping and the group-local
  numbering of `moe_mixedk.MixedKStore`). One slot pool per byte layout (size class) is shared by all layers; the nine
  pointer tables of every `KGroup` are views of the group's table buffer (`TierSet.bind_mixed`), patched in place as experts
  enter and leave slots. Kernels, expert ids and launch sequence are untouched (cache, not a numeric change).
* **Hot set from routing frequency**: `EXL3_EXPERT_WARM=freq.json` (either `{module key: [counts]}` as written by
  `EXL3_EXPERT_STATS_OUT`, or the calibration file of `tools/ds41/plan.py`, `{"layers.3": {expert: count}}`). The pool
  is split between size classes greedily by pick probability per byte (hot K=3 bytes and cold K=2 bytes compete fairly),
  then filled hottest first. `EXL3_EXPERT_PIN_FRAC=f` pins the top fraction of each pool (never evicted); the rest is
  frequency-weighted LRU (decayed use counter, half-life `EXL3_EXPERT_HALFLIFE` forwards).
* **Budget**: `EXL3_EXPERT_BUDGET_GIB=<gib>` or `auto` (free device memory and MemAvailable at the end of the load, minus
  `EXL3_EXPERT_RESERVE_GIB` (6) and the staging ring). Allocation order puts the pools after weights, KV and arenas.
  `mem_get_info` semantics on gfx1151 are not verified: check the printed `auto budget` on worker.
* **Prefetch**: cold reads start when the router output is known (host sync, O_DIRECT, 8 reader threads, pinned staging
  ring). `EXL3_EXPERT_AHEAD=1`: next-layer guess for decode rows, and for prefill chunks (>= `EXL3_EXPERT_AHEAD_ALL_ROWS`
  rows) every non-resident expert of the next layer while this layer computes (needs a ring of ~100+ entries).
* Engram tables stay on the existing O_DIRECT row gather (4 KiB random reads, 48 per token = 196 KB): no RAM budget, 78 k
  IOPS measured, negligible next to the expert stream.

## Measured on this box (`tools/tier/ssd_bench.py`, 10 GiB file, O_DIRECT, 3 rounds, box shared with other jobs)

| read size | threads | GB/s (mean +- SE) | p50 ms |
|---|---|---|---|
| 1 MiB | 8 | 3.62 +- 0.11 | 2.2 |
| 8.85 MiB (K=2 expert) | 4 / 8 | 5.15 +- 0.00 / 5.18 +- 0.01 | 7.2 / 13.6 |
| 13.3 MiB (K=3 expert) | 2 / 4 | 5.33 +- 0.02 / 5.32 +- 0.01 | 5.1 / 10.3 |
| 16 MiB | 8 | 5.41 +- 0.07 | 21 |
| 4 KiB random | 8 | 0.32 (78.5 k IOPS) | 0.06 |

One expert read alone (cold random offset, 60 reads): 8.85 MiB = 2.2 ms (4.1 GB/s), 13.3 MiB = 3.0 ms. So **5.2 GB/s
pipelined, not 6.5**, and a lone miss costs 2.2-3.0 ms against ~1.6 ms of GPU time per layer (62 ms / 40 layers).

## Memory budget (usable 118 GiB, 2.5 bpw pack: K=3 layers 0-7 and 28-39, K=2 elsewhere)

118 - 8.5 (attention, shared, embed, head, norms) - 2.7 (MTP, always resident) - 1.5 (KV, indexer keys) - 6.0 (runtime,
arenas, engram row buffers) - 0.6 (48-entry staging ring) = **98.7 GiB for 158.8 GiB of routed experts = 62 %**.
Each extra 0.1 bpw of pack = 6.4 GiB; ring of 160 entries for prefill prefetch costs 2 GiB.

## Model (`python tools/tier/model.py report`)

Resident decode 62.3 ms/token plain (16.1 tok/s, DS4-Flash Q2 measured, scaled), MTP step 100 ms / 2.25 tokens (22.5
tok/s) [from ds41/est/PERF_MODEL.md, not measured on this engine]. One NVMe = one FIFO server at 5.2 GB/s + 0.5 ms per
burst, Monte-Carlo over 40 layers.

**Per-use hit rate needed for 90 % of RAM-resident speed** (the decisive number, independent of routing skew):

| prefetch | plain | MTP |
|---|---|---|
| none | 98.85 % | 99.17 % |
| Qwen next-layer guess (53 % of misses covered, 47 % of reads useful) [measured on Qwen] | 98.97 % | 99.13 % |
| 3 layers ahead, 80 % covered | 97.9 % | 98.05 % |
| perfect, whole-token window | 95.3 % | 96.1 % |

The guess prefetch does not help because its useless reads share the one SSD (it was also slower on Qwen: read 75 s vs
40 s), and a layer window (1.6 ms) is shorter than one expert read.

Hit rate of a ranked resident set under Zipf(s) routing (same N experts in every layer) [assumption, DS4.1 not measured]:

| budget GiB (experts/layer) | s=0.8 | s=1.0 | s=1.2 | s=1.5 |
|---|---|---|---|---|
| 90 (217) | 85.2 % | 91.3 % | 95.0 % | 98.1 % |
| 100 (241) | 87.8 % | 92.9 % | 96.0 % | 98.5 % |
| 110 (265) | 90.2 % | 94.3 % | 96.8 % | 98.8 % |
| 118 (285) | 92.1 % | 95.4 % | 97.5 % | 99.1 % |

At 99 GiB: plain 7-9 tok/s (s=0.8-1.0, 45-55 % of resident), ~13.5 tok/s at s=1.5 (85 %). The smallest budget with >= 90 % is
112 GiB at s=1.5, 139 GiB at s=1.2, >= 148 GiB at s <= 1.0 (no prefetch): **only a strongly concentrated router gets there
inside 118 GiB.** The one real datum is DS4 Pro (also 384 experts): only 29.4 % of (layer, expert) cells were ever routed,
median 107 reachable experts per layer, so 160/layer covered the reachable set in 58 of 61 layers (ds41/DESIGN.md 3b).
If DS4.1 routes like that on the user's workload the hit rate is 1 - (out-of-domain leak); that leak is what the
calibration capture must measure (held-out prompts vs the calibration counts: `tools/tier/model.py hotset CAL EVAL
--budget-gib 98`).

Synthetic traces (`model.py sim`, domain-switching sessions): decayed-counter policy from a warm start beats a static set by
0-5 points when the session domain differs from the calibration mix, equals it otherwise; pinning 30 % changes nothing;
cold-start LRU converges to the decayed policy after ~half a session.

**Prefill**: a chunk touches ~every expert, so it reads all non-resident bytes (60 GiB = 12.4 s) per chunk: 49 % of resident
speed at 4096-row chunks without overlap, 97 % with next-layer prefetch of the whole non-resident set; >= 4096 rows per
chunk and a deep ring are needed.

## Levers, in order of effect on the 98.9 % requirement

1. Shrink the pack to ~2.2 bpw: 75 % resident (DESIGN 3d). 2. Second NVMe in RAID0: halves every miss (needed hit 98.9 -> 97.5 %).
3. Make cold experts K=2 only and put the hot ones at K=3 (per-expert mix): a miss is 8.5 MiB, not 12.7. 4. A real predictor
for 2-3 layers ahead (needs routing traces). 5. Overlap hit-expert kernels with the miss reads (changes the fp32 sum order,
not bitwise, not done).

## GPU validation still to do on worker

Real `exl3_moe_gfx12_k3` decode with tiered K groups (bitwise vs resident), `auto` budget on gfx1151, measured stall per
miss with the staging ring on the iGPU (unified memory may allow reading straight into slots), hit rate with real DS4.1
routing.
