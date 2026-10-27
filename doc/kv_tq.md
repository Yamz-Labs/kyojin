# TurboQuant KV cache (`--kv-mode tq`)

Per KV head vector (`head_dim` values, a power of two) the cache stores

* `bits` per value: the index of the nearest Lloyd-Max centroid of `R x / ||R x||`, where `R` is an orthonormal Hadamard x random-sign rotation
  (`cache/tq.py`) and the codebook is the one of a coordinate of a uniform random unit vector in `R^head_dim` (`cache/tq_codebooks.py`),
* one fp16 scale per head vector: `||R x|| / ||centroid vector||`. This is the norm correction (the reconstruction has exactly the norm of
  the input) and the only side information: `bits + 16 / head_dim` bits per value (6.06 / 5.06 at head_dim 256), against 6.5 / 5.5 for the
  default per-32-scale cache.

K and V are both stored this way. The fp16 sink-4 + window-128 ring (KVWIN) is on by default for the tq modes (`EXL3_KVWIN=0` turns it off).
No QJL stage, no boundary-layer exceptions.

```
serve.py --k-bits 5 --v-bits 5 --kv-mode tq        # tq5 (5.06 b/value); 4..6 bits, head_dim 64 / 128 / 256 / 512
```

The default (`--kv-mode quant` or nothing) is unchanged. A pack that went through `tools/qualify` has `kv_cache.json` (`kv_mode`, `k_bits`,
`v_bits`); `serve.py` uses it when no KV flag is given.

## How it fits the engine

* The pool keeps `CacheLayer_quant`'s packed layout (32-value groups, power-of-two bit planes) in `qk` / `qv`; `sk` / `sv` hold one fp16 per
  head vector. `copy_page`, save / restore, the prefix disk cache and the KVWIN stash move the same four tensors.
* `R` is orthonormal, so `q . k = (R q) . (R k)` and `o = R^T (sum p (R v))`: `attn_dispatch` rotates q, k and v on the way in and the
  output on the way out, and the pool, the ring and every kernel work in the rotated domain (one fused Triton launch for q, k, v).
* Writer: `tq_triton._tq_write_kernel` (Triton, ROCm + CUDA). Decode: `qdec.hip` / `qdec.cu` with `-DTQ=1` (one scale per head vector, KVWIN ring on
  the same path); the Triton split kernel (`_qc_load_kt/v`, `TQ`) for more than 32 q rows or an odd kv-head count. On CUDA the codebook is a
  shared-memory table of code *pairs* (32-bit entries, 16 KB per side at 6 bits, 4 KB at 5), so a dequantized register is one LDS, and the
  per-vector scale is folded into the scores (K) and the probabilities (V) instead of multiplying every dequantized pair. Prefill: the
  existing staging, with `tq_dequant_window` filling the fp16 scratch (rotated domain) before the fp16 attention kernel. The BC graph
  attention declines tq layers.
  On ROCm the 5/6-bit fast path (`TQF`, `-DTQF_OFF=1` switches it off) never rebuilds fp16 values from the bit planes: for every 8 values it
  forms two index words whose bytes are the LDS byte addresses of the codebook entries (nibble plane masked and shifted, low plane spread to
  one field per byte), then reads each value with one `ds_load_u16_d16` / `_d16_hi`; the next slab's K and V are fetched while the current
  one is computed. Output is bit-identical to the generic path.
* `tools/qualify` (kv stage) measures the decode-faithful 32K KL against k8v8 of k6v6, kw6 (k6v6 + the fp16 sink-4 / window-128 ring) and
  tq6 / tq5 once per model. The default format is kw6 when its KL is <= 0.9 x k6v6's (a 256-step doc scatters 10-30 %: a tie stays on
  k6v6), written to `kv_cache.json` as `kvwin: "4:128"` (`serve.py --kvwin`, sets `EXL3_KVWIN` before the model is built). tq6 / tq5 are
  measured and reported but stay opt-in (`--kv-mode tq`); `QUALIFY_KV_TQ=1` lets them win (fewest bytes among the formats within the
  margin, a narrower one only if every wider one qualifies). `--no-kv-modes` skips the stage. The report also lists decode s/token per format.

Qualify cells, RTX 3090, one 256-step 32K doc, KL vs k8v8 (decode s/tok): Nanbeige k6v6 .00495 (.0356) | kw6 .00267 (.0361) | tq6 .00251 | tq5 .00456
-> kw6. Qwen27 k6v6 .00515 | kw6 .0478 | tq6 .0293 | tq5 .0061 -> k6v6 (one doc, noisy: the same doc gave tq6 .0125 before the kernel change).

## Measured (Qwen3.8-27B EXL3, RTX 3090, graphs on, drafts off, decode tok/s, median of 3)

| context | k6v6 | tq6 | tq5 |
|---|---|---|---|
| 4K | 46.18 | 45.28 (-1.9 %) | 45.82 (-0.8 %) |
| 32K | 44.12 | 43.05 (-2.4 %) | 42.50 (-3.7 %) |
| 120K | 38.33 | 37.55 (-2.0 %) | 37.18 (-3.0 %) |

Kernel only (one layer, 128K, qdec_split): k6v6 271 us, tq6 268 us (306.8 before the pair table and the scale folding), tq5 263 us. The
rest of the gap is the q/k/v rotation and the writer (about 12 us per layer). TQ is opt-in on CUDA. 32K KL vs k8v8 (Qwen27, two prompts):
k6v6 .00062 / .043, tq6 .00060 / .026, tq5 .00115 / .100.

Nanbeige (Llama arch, no recurrent slots): the sink/window ring now binds there too (KVWinState). 32K KL vs k8v8, p1 / p2: k6v6 (production, no
window at 6 bits) .0536 / .0156, tq6 .0319 / .0088, tq5 .0544 / .0182, **k6v6 + the same ring .0216 / .0084**. Most of tq's gain over the
production k6v6 on that model is the ring, not the codebook; k6v6 + ring is as good in KL and a little faster to decode (0.0309 vs 0.0318 s/tok).

Served A/B, same session (decode tok/s, median of 3): 4K / 32K / 120K k6v6 46.29 / 44.20 / 37.45 | kw6 46.13 / 43.75 / 38.02 | tq6 45.68 / 43.33 / 37.24
(tq6 vs kw6: -1.0 / -1.0 / -2.1 %). One 32K decode step is 27.2 ms of GPU time; tq6 adds ~0.32 ms over the same ring (q/k/v rotation 0.14, writer 0.10
more than the quant writer, qdec 0.08), the ring itself ~0.30 ms in both kw6 and tq6.
