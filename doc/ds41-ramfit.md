# DS4.1 RAM-fit plan: the whole routed-expert set in RAM, no NVMe on the decode path

Why: SSD tiering cannot reach 25 tok/s (tier1b, `doc/ds41-tiered.md`: 5.2 GB/s, one expert read 2.2-3.0 ms, >= 98.9 % per-use
hit rate needed for 90 % of resident speed, 7-9 tok/s at 62 % resident). So the pack must FIT: 15 360 routed experts + dense
+ KV + runtime in ~118 GiB, the engram tables (98 GB) stay on the NVMe.

Everything below is a MODEL on synthetic routing (fitted to the one real datum we have, see "Routing model") and the real K=2/K=3
quantizer errors on synthetic Hessians. Nothing was run on a GPU (no GPU was available).
`tools/ds41/ramfit.py` regenerates every table (`budget`, `curves`, `ksets`, `levers`, `plan`).

## 1. RAM budget (118 GiB usable, ctx 128K)

`ramfit.py budget --text-only --embed-nvme` (module bits attn 4, shared 4, head 6, MTP dense 4, MTP experts 2):

| item | GiB | note |
|---|---|---|
| attention (wq_a wkv wq_b wo_a wo_b, 40 L, K4) | 2.39 | 2.56 GB read per token |
| shared experts (K4) | 0.66 | 0.71 GB/token |
| head (K6) | 0.46 | 0.50 GB/token |
| MTP: dense 0.26 + routed 3 x 128 experts at K2 3.18 + copies 0.12 | 3.56 | was 6.3 GiB of experts at K4 in the old plan |
| engram wkv (raw FP8, 2 layers) | 0.29 | 0.32 GB/token, read every token, not quantized by plan.py |
| router gates, hc, other small copies | 0.35 | 0.37 GB/token |
| KV at 128K (fp16 rings + 4 shared pools + 8 indexer key planes) | 0.52 | 1M ctx: 4.2 GiB, 4.3 MB per 1K |
| runtime (context, arenas, prefill chunk activations) | 5.00 | assumption, tier1b used 6.0 incl. a staging ring |
| engram row buffers | 0.50 | rows are 4 KiB O_DIRECT reads, 48 per token = 196 KB |
| **fixed** | **13.74** | |
| embedding 1.23 + vision tower/aligner 0.90 | not resident | file-backed, one row per token / first image |
| **routed experts get** | **104.3 GiB = 112 GB = 1.65 bpw average** | all K=2 would be 127.2 GiB, all K=1 63.9 GiB |

With vision and the embedding resident the experts get 102.1 GiB (1.61 bpw). 1M context costs 3.7 GiB more (-0.06 bpw).
Engram bytes: 48 rows x 136 B = 6.5 KB/token payload, 196 KB/token of 4 KiB sector reads (78 k IOPS measured: 0.6 ms
if serial, overlappable because the hash only needs token ids), 5.9 MB/s at 30 tok/s: negligible on the decode path.
Prefill reads 98 K rows per 4096-token chunk (400 MB) per engram layer, a few seconds, overlappable.

**Consequence: with all experts resident the average is 1.65 bpw. K in {2, 3} cannot fit at all** (all-K=2 needs 127 GiB):
the fit needs K=1 for the cold experts and/or pruning.

## 2. K=1 support

* Quantizer: yes. `quantize_exl3` accepts K = 1..8 and the CUDA tile quantizer has K=1 instances (`quantize_tiles_inst_k1.cu`,
  optimized kernel on sm_86 / the 3090). On ROCm `quantize_tiles_mode` sends K=1 to the BASE kernel (RDNA path is K 2-3, optimized
  K 4-5): the iGPU may be several times slower on K=1 experts (`EXL3_QT_KERNEL=opt` selects the optimized instance, unverified for
  bit-equality). `smoke_alloc_kamel.sh` S0/S0b measure both.
* Error (CPU trellis, mul1 codebook, 16-bit state, tail-biting 256, iid N(0,1), `scratch/k1_trellis.py`, 32 tiles):
  K=1 relative MSE **0.2745 +- 0.0049**, K=2 0.0687 +- 0.0012, ratio **3.99** (R-D bound 0.25 / 0.0625). The real K=2 proxy (Hessian
  weighted, 0.0489) therefore puts K=1 at ~0.196: a K=1 expert has 4x the error energy of K=2, 16x that of K=3. Never use K=1 on a hot
  expert.
* DS4.1 decode / prefill kernels (`ds41kern`, `mixedk` trees): **K=1 is NOT supported there**. The grouped HIP MoE kernel
  `exl3_moe_gfx12_k3` accepts k2 in {4,5,6,8,10,12} (K 2, 2.5, 3, 4, 5, 6): `launch_moe_grouped_k` (`exl3_gemv.cu`), the host
  `TORCH_CHECK`, the VALU body `exl3_moe_valu.cuh` (`mv_tile`, BITS 2/3/4 only) and the prefill `exl3_moe_prefill.cu` (`MPW_K`)
  have no K=1 case, and `moe_mixedk.DECODE_K/WMMA_K` plus the `multi.K in (2, 2.5, 3, 4, 5, 6)` gates in `block_sparse_mlp.py`
  exclude it. What it takes: a `dq8_regs_1bits` helper (8 weights = 8 + 15 stream bits, one u32 plus the wrap word), the BITS == 1
  branch in `mv_tile` (NWD = 2), `launch_moe_grouped<1,...>` + `case 2` in the switch and the K check, `MPW_K(1, false)` in prefill,
  K=1 in the python tuples. The generic `exl3_gemm`/mgemm CUDA path already has K=1 instances. Pruned experts (K = 0) need the
  loader to accept an expert without tensors (skip it in `MixedKStore`, tables hold no slot; the router bias already masks it).
* Launch cost: one grouped launch set per distinct K in a layer. {1,2,3} = 3 groups per layer (~9 tiny launches each, not in the
  bytes roofline); graph capture makes it ~1 ms/token, eager ~7 ms. Another reason not to add K=4.

## 3. Routing model

One real DS4.1 datum exists (`ds41ref` CPU reference run, 2 rows x 2048 = 4096 tokens): experts hit / max tokens on one expert
= L0 366/693, L1 357/991, L2 345/2144, L3 344/2618, L4 359/1503 (L0-2 hash routing). Layer 3: 64 % of the tokens go to ONE
expert, and 40 experts got nothing. `ramfit.fit_reference` (Gumbel top-6 simulation of 4096 tokens) reproduces layers 3 and 4 with
Zipf s = 1.1 + 50 near-dead experts and s = 0.8 + 30 near-dead (simulated hit 341.5 / 357.0, max 2517 / 1506). Scenario `ref` =
per-layer s in [0.8, 1.1], 30-50 near-dead experts (inclusion probability 3e-5). `flat` (s 0.5-0.7, none dead) and `steep`
(s 1.0-1.3) bracket it. The calibration capture (262 k tokens) replaces this model per layer: the converter decides K from the
real counts (section 6), and `optimize_alloc.py --skew` prints the real skew after layer 2.

Quality proxy D (not a KLD): per layer sum_e share_e * 2 eps(K_e) = relative error energy of the routed-MoE output, mean over
layers (share = routing mass x gate^2 x output energy; eps(K): 0.196 / 0.0489 / 0.0120 for K = 1 / 2 / 3, x4.07 per bit beyond;
factor 2 = eps_w2 + 0.5 (eps_w1 + eps_w3)). A pruned expert costs share x 1.3 (its contribution is lost, plus the substitute the
router picks, beta = 0.3 assumed; the router re-normalizes over the selected top-6 by construction). Uniform K=2 = 9.8 %, uniform K=3 = 2.4 %.

## 4. Quality proxy vs RAM (scenario ref, `ramfit.py curves --text-only --embed-nvme`)

Per-layer byte budgets (what the converter does), D in % of the routed output energy, bpw averaged over all 15 360 slots, GB = routed
bytes read per token:

| experts GiB | mixed K only {1,2,3} | prune only (K=2 kept, any expert) | both | both, prune only pi < 1e-4 |
|---|---|---|---|---|
| 60 | infeasible | 24.3 % (53 % pruned) | 18.7 % | 25.8 % |
| 70 | 20.8 % | 20.6 % (45 %) | 15.1 % | 16.1 % |
| 80 | 14.6 % | 17.5 % (37 %) | 12.0 % | 12.2 % |
| 90 | 11.3 % | 14.7 % (29 %) | 9.7 % | 9.7 % |
| 100 | 9.0 % | 12.4 % (22 %) | 7.8 % | 7.8 % |
| **104.3 (fit)** | **8.2 %** (1.65 bpw) | **11.5 %** (18 % pruned) | **7.1 %** | **7.1 %** (10.5 % pruned) |
| 120 | 5.9 % | 9.8 % (6 %) | 5.1 % | 5.1 % |
| 127.2 (all K=2) | - | 9.8 % | | |

* Mixed K alone, at the fit, is 8.2 %: BETTER than uniform K=2 (9.8 %, 127 GiB) because the hot K=3 / cold K=1 split follows the
  routing mass. Pruning alone (K=2 for the survivors) costs 11.5 %: 18 % of the experts include genuinely used ones.
* Pruning only the near-dead experts (pi < 1e-4, ~10 % of the slots in this scenario) is as good as unrestricted pruning at the
  fit (7.1 % both) and is the safe version: it buys 1.1 points (8.2 -> 7.1) because the 4.4 MB of every dead K=1 expert goes to
  hot experts. If the real calibration finds no dead experts the budget is simply spent on bits (no loss, 8.2 %).
* Merging (REAP-style: fold a rare expert into its nearest survivor) is NOT modeled or implemented: it needs expert similarity
  that only the real weights can give. It can only beat K=1 + prune if the merged pair stays below the K=1 error (0.196 x share),
  which for rarely routed experts is already tiny; not worth a second conversion path before real data.
* Sensitivity (`ramfit.py ksets`, budget 103.2 GiB, D %): ref {1,2,3} 8.4 -> 7.3 with pruning; flat 13.8 (pruning irrelevant);
  steep 5.95 -> 5.2. Adding K=4 gains 0.4-0.8 points (428-711 hottest experts at K=4) but adds a 4th K group per layer; one
  network-wide multiplier (ideal allocation) beats per-layer budgets by only 0.07-0.15 points, so uniform per-layer budgets are fine.
  {1,2} without K=3 is much worse (11.7 / 14.7 / 10.8): the K=3 tier for the hottest ~12 % carries the quality.
* The flat scenario is the risk: 13.8 % (1.4x uniform K=2). Real routing decides; the converter adapts without a re-plan.

## 5. Attention bits (speed lever #1) and bytes per token

`ramfit.py levers --text-only --embed-nvme` (experts re-solved for each row; D_attn = sum of the five attention proxies per layer, in %,
D_exp as above; ceilings = bytes read per token / bandwidth, plain decode, no MTP, launch gaps not included):

| attention / shared | experts GiB | D_exp | D_attn | GB/token | tok/s @190 | @230 |
|---|---|---|---|---|---|---|
| 5 / 5 | 103.5 | 7.38 | 0.36 | 8.11 | 23.4 | 28.4 |
| **4 / 4 (plan default)** | 104.3 | 7.28 | 1.47 | **7.30** | 26.0 | **31.5** |
| big (q_b wo_a wo_b) 3, small (q_a kv) 5 | 104.8 | 7.20 | 3.74 | 6.76 | 28.1 | 34.0 |
| big 3, small 4 | 104.8 | 7.18 | 4.19 | 6.72 | 28.3 | 34.2 |
| 3 / 3 | 105.0 | 7.15 | 6.00 | 6.49 | 29.3 | 35.4 |

Each attention step 4 -> 3 on the big tensors is worth 0.55 GB/token = +2.5 tok/s @230, and frees only 0.5 GiB of RAM, so
attention bits do not change the expert budget, they change speed. The converter keeps attention K=4 primary with K=3/5
candidates (`--alt-attn 3,5`, +0.3 h) and the choice is made AFTER the conversion by `optimize_alloc.py --resident-only` and
`assemble_pack.py` against a KLD run (`eval/ds41/kld_eval.py`). `plan.py --role-bits q_b=3,wo_a=3,wo_b=3,q_a=5,kv=5` bakes a role mix
into a plan directly.

Bytes per decoded token for the recommended plan: 7.30 GB = attention 2.56 + shared 0.71 + head 0.50 + routed 2.70 (hot
experts are K=3, so the per-token routed read is ABOVE a flat 1.65 bpw: 2.70 vs 2.0) + engram wkv 0.32 + router/hc/other 0.37 +
KV scan 0.14. Ceiling 26.0 / 27.4 / 31.5 tok/s at 190 / 200 / 230 GB/s plain (34 with the attention mix above). Realistic
(eta 0.5 from GLM on the same box): ~13-16 plain; MTP raises it ~1.4-1.5x. The 25 tok/s owner target needs MTP acceptance,
attention at 3 bits and launch overhead under control; plain decode alone will not reach it.
Small remaining byte levers: engram wkv at K4 (-0.15 GB, needs a capture group), head K5 (-0.08 GB), MTP experts K1 (-1.6 GiB RAM).

## 6. Recommended plan and conversion flags

Plan (`scratch/ramfit/ramfit_plan.json`, generated by `ramfit.py plan --text-only --embed-nvme --prune`):

* routed experts: K in {1, 2, 3} chosen PER EXPERT AT QUEUE TIME from the layer's routing statistics, per-layer budget 2.77 GB
  (7.22 MB per expert on average, 103.1 GiB for the net), model histogram {K0 pruned 1608, K1 4716, K2 6942, K3 2094} (10.5 % dead,
  1.63 bpw); with no dead experts the same budget gives about {K1 7585, K2 6030, K3 1745}. Only experts routed to by < 1e-4 of the
  calibration tokens may be pruned (router bias -1e4 in the pack).
* attention K4 (+ K3/K5 candidates), shared K4 (+ K3/K5), head K6, MTP dense K4, MTP experts K2, engram tables copied raw (NVMe).
* predicted pack: experts 103.1 GiB + fixed 13.7 = **116.8 GiB resident** of 118; pack on disk ~121 GB (+ 98 GB engram tables).
* proxy D 7.3 % (ideal multiplier 7.1 %), decode 7.30 GB/token, ceiling 31.5 tok/s @230 (26.0 @190).

Conversion time: no extra quantization (no expert candidates), cost = the Ks chosen; with the NOTEBOOK model T = (384 w + 0.2016 c)/0.4286
s per layer (c = 300 s iGPU capture, w = K2-units of expert work per layer) and the model histogram: 10.2-11.5 h with pruning,
10.6-12.5 h without (r1 = 0.7..1.0, r3 = 1.0..1.3, + 0.6 h tail and dense candidates); the old plan was 13.3 h and 15.3 h with expert
candidates. r1 is unmeasured (S0/S0b), and K=1 on the iGPU base kernel may be slow: if S0b shows it, use the 3090 for K=1 or
`EXL3_QT_KERNEL=opt` on the iGPU worker.

```
python3 tools/ds41/ramfit.py plan --text-only --embed-nvme --prune --out ramfit.json      # the plan above (ctx 128K, runtime 5 GiB)
python3 tools/ds41/stream_convert.py --role coordinator --map tensor_map.json --out OUT \
   --layer-bits 0-39:2 --no-engram --mtp-queue --attn-bits 4 --shared-bits 4 --mtp-exp-bits 2 \
   --alt-attn 3,5 --alt-shared 3,5 --ramfit ramfit.json [--ramfit-smooth 0.02] [--ramfit-prune-tau 1e-4] ...
python3 tools/ds41/plan.py --map tensor_map.json --expert-bits-file alloc.json --role-bits q_b=3,wo_a=3,wo_b=3 ...   # fixed K plan
python3 tools/ds41/optimize_alloc.py --tables OUT --plan OUT/plan.json --resident-only --out alloc.json                # attention K after
```

`scratch/run_conv.sh` defaults to this (`RAMFIT=1`; `RAMFIT=0` restores the old position-based plan), `launch_kamel.sh` ships
`ramfit.json`. How the converter uses it: per layer, once the capture's routing file exists, `split.Coordinator.ramfit_layer` solves a
Lagrangian per layer (`ramfit.decide_layer`: hot experts get K=3, cold K=1, dead ones 0, filling the layer budget exactly) with
share = squared routing weights + 2 % of the mean, persists `ramfit/<unit>.json` (replayed after a restart, so experts already
quantized keep their K), rewrites the plan entries (pruned experts become action `pruned`: no job, no tensors, router bias masked at the
copy step), and `finalize` writes `quantization_config.expert_bits` (0 = pruned) and `pruned_experts`. If the capture produced
no statistics the layer falls back to equal shares (same bytes, arbitrary but deterministic K mix) and says so in the log.

The dual-candidate capture for experts is dropped under ramfit (`--alt-experts` is ignored with a log line): K is decided from the
live counts, so a measured K=3 vs K=2 pair adds nothing the budget did not already settle, and it cost +1.6-2.0 h. Dense candidates
(+0.3 h) stay: attention bits are the lever left to choose after the conversion.

## Limits

D is a first-order ranking proxy on a routing model fitted to ONE reference run (2 rows, 5 layers), K=1 error from iid Gaussian
tiles scaled by the K=2 proxy, prune cost beta = 0.3 assumed, runtime 5 GiB assumed, 118 GiB usable when the box is idle. The
engine, K=1 kernels and K=0 loader support do not exist yet (section 2). Validate with `eval/ds41/kld_eval.py` on the ref
(rows 0-15 first) and re-run `ramfit.py plan` with the measured skew before the conversion reaches layer 8.
