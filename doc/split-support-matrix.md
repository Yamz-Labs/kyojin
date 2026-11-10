# Layer-split support matrix

What the contiguous layer split (`exllamav3/device_map.py`, `exllamav3/bridge.py`,
`Model.load(device_map=...)`) can place today, per model family and per serve flavor.
Status as of the Stage 0 rebase (2026-11-10): GLM is measured on the Strix Halo / 9070 box;
Qwen and DSV4 entries are code inventories, not load-tested.

Legend — placement: **front** (mapped to the front eGPU), **main** (mapped to the main GPU),
**cpu**, **host** (never enters the map: CPU/streamed by design). Stage links point at the
roadmap stage that must finish before that row is claimable.

## Measured — GLM-5.3-Flash (`tools/glm/serve.py`, `tools/strix_halo/split_*`)

| Component | Config | Placement | Notes |
| --- | --- | --- | --- |
| text (`Model.from_config`) | shared | layers 0..N-1 front, N..44 main; endcaps per flag | measured N=4, N=6 at `-c 65536`; N=5 is the long-context setting until the gfx12 KDA kernel lands |
| mtp (`component="mtp"`) | shared | all front when `--endcaps on`, else main | separate `LayerDeviceMap` (no own trunk), reserve owned by the text map |
| vision (`component="vision"`) | shared | front when `--endcaps on`, else main | loaded with a plain `device=` argument; no bridge interaction |
| `embed_tokens` | shared | cpu (`prefer_cpu` cap) | stays host-side by design |
| KV cache (TQ 6-bit) | built per model/cache | each cache layer follows its block | measured 0.62 GiB main / 0.12 GiB front at 65K |
| bridge | shared config | installed by `_load_mapped` for all three components | one `Bridge` per config; split_active defuses graphs |

Gate status (Stage 0 replay): N=6 65K load with MTP+vision front 13.68/13.94 GiB, plain split
run-to-run deterministic; results appended in `docs-notes/split-log.md`.

Known blockers: gfx12 fused KDA kernel (separate PR), MTP cross-device acceptance/determinism,
endcaps currently moves every non-layer module to the front (GLM has no ngram/PLE/engram, so it
never showed; the universal core replaces this with an allowlist — Stage 1).

## Code inventory — Qwen3.8-Flash-Next (`tools/qwen/serve.py`)

| Component | Config | Placement intent | Stage |
| --- | --- | --- | --- |
| text | own (`Model.from_config`) | layers via SplitPlan; `.load()` with no device today → autosplit | 3 |
| dflash drafter | **separate `Config` (`dconf`)**, `attach_to` | needs its own bridge + split_active (install per config) | 3 |
| mtp | shared config, `component="mtp"` | all front (endcaps on), else main | 3 |
| vision | shared config, `component="vision"` | front when it fits | 3 |
| ngram table (`NGramEmbedding`) | inside the text model | host/swap-resident (`ensure_resident`, 24.4 GiB) — never front | 3 |
| PLE / DQ / GR / GDN HIP kernels | JIT loaders | gfx11.5-only gates; per-device arch needed; no implicit front placement | 3 |
| QSA sparse attention (quant cache pages) | cache layers | gfx1201 `dsa_topk_gfx12` illegal access — keep off the front until fixed | 3 |
| KV/GDN recurrent state | `Cache(…, max_history=ndt, max_batch_size=sessions)` | per-layer KV automatic; recurrent slot devices need an audit | 3 |

Serve env notes: CUDA-side graph preset (`EXL3_CUDA_GRAPH*`), ROCm chunked prefill
(`EXL3_SCHED_CHUNK_IL`), `EXL3_PF_GR_FUSE`/`EXL3_GR_PF_FUSE` deferred-residual keys must be
disabled for split (roadmap Stage 3).

## Code inventory — DeepSeek-V4.1-Flash (`tools/ds41` ↔ `exllamav3/dsv41`)

The DSV4 engine does **not** go through `Model.load()`: `dsv41/pack.py:load_pack` streams
`model.modules` to one device with `mod.load(dev)` and builds `DSV41Serving(..., device=dev)`.

| Component | Today | Placement intent | Stage |
| --- | --- | --- | --- |
| text blocks | per-module `mod.load(dev)` on one device | contiguous N via a SplitPlan over the same module list | 4 |
| mtp | `Model.from_config(cfg, component="mtp")` then `mod.load(dev)` | front with endcaps / main | 4 |
| engram tables | `MemoryEngramStore` / `PrefetchRowStore` (O_DIRECT, host) | stay host; `engram_inject` gathers rows at the layer that needs them | 4 |
| MoE | `GroupedMoE(b.ex_w1, ex_w3, ex_w2, dcfg, dev, …)` one device + `MixedKStore` | per-device grouped backends; no cross-device expert state | 4 |
| decode graphs | `serve_graph.DecodeGraph` per run-of-layers around engram layers | the existing per-segment split is the natural seam for Stage 6 | 4/6 |
| KV/state | `DSV41Cache` (custom pool, kv source layers + shared read) | audit device placement of pools and slots | 4 |
| vision | `VisionTower.load(vision_dir, …, vision_device)` | separate device arg; map to front/main policy | 4 |
| tiered experts (`EXL3_EXPERT_BUDGET_GIB`) | forces `DSV41_DECODE_GRAPH=0` | graphs-off for split anyway (Stage 6) | 4 |

## Candidate (best effort) — Flint, Mimo, generic `Model` trees

`tools/flint/serve.py` and `tools/mimo/serve.py` are single-`Config` engines calling
`Model.from_config(...)` + `.load()`; both already accept `device_map` after the Stage 1 core
lands (mimo loads its vision tower separately, mirroring GLM). They inherit whatever the
universal core does for text layers + endcaps; no dedicated adapter is planned (Stage 7).
