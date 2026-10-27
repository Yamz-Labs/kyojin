# DS4.1 bit allocation: measured at conversion time, chosen afterwards

The converter does not have to guess which module gets how many bits. It quantizes a few extra candidates while it holds
the Hessians, records the Hessian-weighted error of every candidate plus the routing statistics of the calibration run,
and `optimize_alloc.py` picks the mix afterwards (a few seconds, CPU). `assemble_pack.py` then swaps the chosen
candidates into the pack. No re-conversion, no GPU.

```
capture.py          stats/<unit>.routing.json         per layer: tokens routed to each expert, sum of squared gate weights,
                                                      mean MoE output energy (router hook, no numerics change)
stream_convert.py   cands/<unit>/e<NNN>.k<K>.*        alternative-K candidate of an expert (the plan K lives in the shard)
 split.py           cands/<unit>/lin.k<K>.*           alternative-K candidates of attention / shared linears
 quant.py           alloc/<unit>.json                 table: per expert and per K proxy errors (w1, w3, w2), bytes, where;
                                                      per linear the same; E = mean output energy tr(W H W^T)/count
optimize_alloc.py   alloc.json                        expert_bits per layer, module_bits, resident experts, predicted metrics
assemble_pack.py    the pack                          shards + index + config.json quantization_config.expert_bits
```

## What is measured per candidate

* proxy error: `tr(E H E^T) / tr(W H W^T)` of the quantizer (relative, Hessian-weighted, calibrated on the routed tokens of the
  expert); x4.07 per extra bit on synthetic Hessians, the real value is what the table records;
* E: mean output energy of the linear on its calibration inputs (so `proxy * E` is an absolute output error);
* routing count and `w2sum` (sum of squared routing weights) per expert, tokens, and the mean output energy of the MoE
  block per layer (`Y`).

## Model (optimize_alloc.py docstring has the details)

* quality `D` = sum over modules of (share of the block output energy) x (relative error), first order, errors add. Expert:
  `share = w2sum * E_w2 / (n_tokens * Y)`, error `eps_w2 + nu (eps_w1 + eps_w3)`. Attention: sum of the proxy errors of the
  role's linears, scaled by `--w-attn`. `D` ranks candidates; it is not a KLD. `--w-attn`, `--w-shared`, `--layer-weights`
  are the knobs a KLD measurement on two or three packs calibrates.
* speed: seconds per token = sum of P(read) x bytes / bandwidth. Dense modules every token from RAM. A routed expert with
  P = 1 - (1 - p_e)^rows (`--rows` 1 plain, ~5 MTP verify), from RAM if resident, else NVMe (`--bw-ssd`, `--ssd-hit`) plus RAM.
  `p_e` comes from the calibration routing counts, so hot experts cost speed per bit and cold experts only RAM / disk.
* RAM: dense + MTP + copied tensors + runtime (`--runtime-gib`) + engram row cache + the resident experts <= `--ram-gib`.
  The pack does NOT fit RAM (routed 170 GB + engram 98 GB vs ~118 GiB), so residency is part of the choice: the optimizer
  picks K and resident / streamed per expert.
* solver: multiple-choice knapsack with RAM, disk and time caps. Nested bisection on the Lagrange multipliers, then greedy
  repair and fill. Tests compare it with brute force (9 items) and with the LP relaxation (1500 items, within 1 %).

## Use

```
python tools/ds41/optimize_alloc.py --tables OUT --skew                       # routing skew of the layers captured so far
python tools/ds41/optimize_alloc.py --tables OUT --plan OUT/plan.json --sweep 14,16,18 --min-toks 16 --out OUT/alloc.json
python tools/ds41/assemble_pack.py --src OUT --alloc OUT/alloc.json --dst OUT          # in place, reversible
python tools/ds41/assemble_pack.py --src OUT --alloc OUT/alloc.json --dst PACK --link  # to another directory
```

`--dst == --src` swaps in place: the displaced tensors go to `cands/`, so the pack can be flipped back and forth with one
layer of scratch space, and the current K of every module is read from the shard headers (an interrupted run is repeated).
Asking for a K that was not measured is an error. Validate the chosen pack with `eval/ds41/kld_eval.py` on the reference.

## Candidate sets and cost

The plan K per layer (position prior) is the primary, everything else is a candidate. `--alt-experts 2,3 --alt-up-frac U
--alt-down-frac D`: in a K=2 layer the hottest U of the experts also get K=3, in a K=3 layer the coldest D also get K=2;
`--alt-attn 3,5 --alt-shared 3,5` for the dense modules. Cost relative to the plan-only conversion is (U r3 + D)/(1 + r3)
of the expert quantizations (r3 = time(K=3)/time(K=2)) plus ~60 s of iGPU per layer for the dense candidates.

Failure policy: everything here is statistics. A missing routing file, a failing energy computation or an unwritable table
never stops the conversion; the pack it builds is the plan pack and is valid on its own.
