# Hy4 multi-session decode (experimental, `--slots`)

## Where a session enters the step
- `tools/hy4/serve.py` takes an asyncio lock around `Hy4Engine.generate`: ONE request at a time. `generate` calls `reset()` (zeroes every static buffer) then prefill then the decode loop.
- `step_graph.GraphDecoder.step_rows(toks)` appends T consecutive tokens of ONE sequence to static buffers and replays a HIP graph keyed (bucket, T); T <= TM = 8.
- Per-session state in the solo engine: K rows `_ns.State.kv[layer]` [cap, 576] fp32 (every trunk layer), indexer `pool` [cap/k, d] + `vgb` [cap, 2d] (+ eager `n`, `P`, `vg`),
  `Hy4Forward.n / tokens / _x / _eng_pf`, GraphDecoder `expect_n` + Engram staging, MTP head `kbuf` + its indexer slabs, engine `y`, `hcur`, NGram, gate `rate / since`.
- Row-wise already (no session state): embedding, hyper connections, norms, dense Q8_0 GEMV, router, expert kernels (union of the rows' experts), output head.

## Design (generic: slots, not a Hy4 fork)
- Every per-session buffer becomes a SLAB of one flat tensor: K `[S*cap, W]`, indexer `vgb [S*cap, 2d]`, `pool [S*cap/k, d]`. A session's own view is a contiguous slice, so the solo code
  (eager prefill, MTP head, legacy graph) runs on a session unchanged after `MultiDecoder.bind(session)`.
- A batched step carries per row (token, position, base = slot*cap) in the static graph input (device data): graphs are keyed (bucket, rows) only, any mix of sessions reuses them.
- Ops that touch session state have slot variants in `loader/mu_slots.py`: MLA core (`mla_attend_slots`, Triton, row t reads keys base[t] + 0..pos[t]), K row write (`index_copy_` at base+pos),
  indexer update + scoring (`indexer_forward_slots`: the existing pool Triton kernel on FLAT positions with the rope table tiled per slab; scoring gathers each row's own pool slab).
  The solo ops are not edited; `hy4_forward.layer` / `mla_hy4.attend` get a 3-line `slots` branch that is dead in solo mode.
- Host side per session: Engram row ids from its own token list, n-gram proposer, gate state, MTP draft chain graph keyed (slot, bucket).
- Prefill of a new request runs eager in chunks (`MuScheduler`: one prefill chunk, then one decode round of the running sessions); the row prefetch list and `phase()` flags are global,
  so one prefill is in flight at a time (the others queue).
- Row budget: TM = 8 (mla_bgemv / router_gemv MAXT, gm R <= 8). Speculative rounds split it evenly: B = 1: 8 rows, B = 2: 4 (k = 3, no n-gram rows), B = 3-4: 2 (k = 1). Going past 8 rows needs the
  spec agent's R-row kernels (ask main).

## Memory
Slab per session ~ n_layers * cap * 2304 B (K) + 13 indexer layers * cap * 1.3 KB + MTP head: about 0.11 MB per position, 0.95 GB at cap 8704, 1.8 GB at 16K. 4 slots at 16K = 7 GB next to ~72 GiB experts.

## Tests
- `tests/test_hy4_mu_slots.py` (CPU, Triton interpreter, and MU_DEV=cuda): indexer batched rows == solo per session (allow masks, vgb rows, pooled keys; past the all-causal bound), staggered
  sessions, scrambled neighbours, sentinel slots never written (last slab, last position), MLA core reference == per-row solo reference, other slabs / future rows never read, Triton kernel == reference.
- GPU (`tools/hy4/mu_bench.py`): solo legacy reference vs batched streams (first divergence index), logits of a row alone vs inside a 2-row step, aggregate / per-session tok/s, memory, ttft + gaps.
