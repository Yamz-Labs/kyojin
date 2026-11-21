# STRIX_MTP_LAYER: draft MTP layer, itemised (Strix box, 20/11, branch perf/hy4-strix-mtp off dv 7ecdfcc9a)
FINDING: the draft experts are NOT the Q3_K single-row mmvq (105 GB/s). They are int8 + per-row scale (moe_i8.py, Triton `_k_gu` / `_k_down`, 16 experts, N = K = 1536) and already read 202 GB/s (84 % of the 240 roof). Routing them through the grouped kernel was not needed.
Measured with real blk.42 tensors (64 of 512 experts, 16 distinct per step), HIP graph replay, k = 3 steps, NO head. Raw: ~/scratch/n1119/out/{mtp_micro1,mtp_prof1,mtp_arms1}.log. Tools: tools/hy4/step/{mtp_micro,mtp_items}.py, mtp_arms.sh.
| part (per step, ctx 4096, rocprofv3) | us | bytes | GB/s | note |
|---|---|---|---|---|
| fp32 rocBLAS SGEMM (7 launches: wk_b, wv_b + 5 indexer mats) | 587 | ~58 MB | 12-130 | skinny M = 1 tiles; floor ~240 us. FIXED below |
| dense Q8_0 gemv (12: eh_proj, wq_a/b, wkv_a, wo, wgate, rdown/rup, shexp x3) | 501 | 113 MB | ~225 | at roof |
| MLA attend: _sparse_mla_kernel (fp32 K cache, one head per program, 32 programs) | 391 | 9 MB | 23 | selection active at 4K (not at 1K); at 1K the head-blocked Triton kernel runs (107 us). OPEN |
| experts gate+up `_k_gu` | 366 | 76 MB | 196 | tiles retuned: 328 us / 230 GB/s |
| experts down `_k_down` | 172 | 38 MB | 219 | tiles retuned: 163 us / 232 GB/s |
| router mat-vec `_k_rg` | 47 | 6.3 MB | 134 | OPEN (fp16 router: HY4_RG_F16) |
| ~60 small kernels (rms, rope, index_copy, cat, quant) + gaps 2.5 us each | ~260 | | | OPEN: glue fusion |
| step window / round (no head) | 2448 / 7414 | 262 MB | | floor 1.09 ms / 3.3 ms; head Q5_K adds 3.3 per round (dv worker) |
Round, no head, ms (A shipped -> E all three flags): ctx 1K 5.45 -> 4.34 (-20 %), ctx 4K 7.08 -> 5.60 (-21 %).
| flag (default OFF, draft only) | 1K | 4K | step-1 rel err vs A |
|---|---|---|---|
| HY4_DRAFT_MOE_TILES=1 (expert tiles) | 5.27 | 6.90 | 4e-4 (sum order) |
| HY4_DRAFT_MLA_BG=1 (wk_b / wv_b as int8 + fp16 block scales on the trunk batched mat-vec, +7.8 MB) | 4.94 | 6.55 | 1.5e-2 (Q8 re-quant of bf16) |
| HY4_DRAFT_IX_GV=1 (draft indexer fp32 mats on the router mat-vec) | 5.05 | 6.30 | 0 |
Not applicable: HY4_LC_HB (mla branch) is the fp8-cache kernel; the draft K cache is fp32 [cap, 576]. A head-blocked kernel for the fp32 sparse attend is the next lever (-0.85 ms per round at 4K+).
Served ABBA (tau per class, draft ms, round ms): job 84_2f_mtp_ab, queued, not run yet.
