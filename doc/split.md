# Contiguous layer split (iGPU + eGPU)

*Status: Stage 1 of the universal roadmap (see `.kilo/plans/1794309410242-universal-layer-split-roadmap.md`).
The core is now policy-driven (`SplitPlan`/`ComponentPolicy`), the GLM server is the
first adapter, and the front reserve is a layout-neutral slab. This file is the
durable user/ops reference; the roadmap owns the staging, and
`docs-notes/split-log.md` (untracked) owns raw measurements.*

## What it is

One process places the first N transformer layers plus the small endcaps of a model on the
fast "front" GPU (the 15.92 GiB RX 9070 over USB4) and the rest on the "main" GPU (the 118 GiB
Strix Halo iGPU), with an asynchronous host-staged bridge on every crossing:

    src tensor --D2H--> pinned slot ring --H2D--> dst tensor

Both legs are enqueued on the two devices' own compute streams; the host never blocks. The
destination tensor is a fresh allocation, so callers may keep using it after the slot cycles.
`KYOJIN_BRIDGE=0` falls back to `to_device` (split still runs, crossings block).

## Interface

| Flag (CLI) | Env fallback | Meaning |
| --- | --- | --- |
| `--front-layers auto\|N` | `KYOJIN_FRONT_LAYERS` | first N transformer layers on the front device; 0 = off |
| `--front-device auto\|gfx1201\|cuda:N\|uuid` | `KYOJIN_FRONT_DEVICE` | front device selection (auto = first gfx12 device) |
| `--endcaps on\|off` | `KYOJIN_ENDCAPS` | move the text allowlist (norm/hc/lm_head-like) and whole MTP/vision components to the front (default on) |
| `--front-reserve-gb` | `KYOJIN_FRONT_RESERVE_GB` | VRAM held as a load-time slab on the front for workspace (default 1.2) |
| `--split-preset auto\|glm\|qwen\|ds41\|none` | `KYOJIN_SPLIT_PRESET` | per-model unsafe-env preset slot; GLM needs none today |
| `--split-graphs auto\|on\|off` | — | per-segment capture (Stage 6); split forces graphs off until then |

`KYOJIN_FRONT_LAYERS=…` without CLI flags also works in `model_init.init` paths
(examples/eval tools). The legacy `EXL3_LOAD_SPLIT` budget knob is **not** part of this
interface; combining it with the new map must error.

## Core objects (Stage 1)

- `exllamav3/device_map.py`: `SplitPlan` owns every component the process places.
  `ComponentPolicy` declares `front_eligible`, `endcaps_front`, the endcap allowlist,
  arch gate and unsafe env. `ComponentMap` resolves one component's contiguous boundary.
  Unknown non-layer modules are logged and kept on main; auxiliary components never move
  front implicitly.
- `exllamav3/bridge.py`: one bridge per *config*, installed by `SplitPlan.install_bridges`
  for the target, a separate draft config, vision and any extra config.
- `exllamav3/serve_split.py` (shared with `tools/kyojin_serve_split.py`): CLI/env parsing,
  device resolution, plan/load/report orchestration, split-safe env. GLM is the first
  adapter; the helper is pure stdlib at import so a server can apply `SPEED_ENV` first.
- `model_init.py` uses the same plan for the `KYOJIN_FRONT_LAYERS` env-only path.

## Load-time effects worth knowing

- The front reserve is an **inert device slab**, not `set_per_process_memory_fraction`.
  The Stage 0 replay showed a process-wide allocator cap makes 128K split runs bimodal;
  the slab removes bytes from the pool and is dropped at the first main module (or after
  the last component), leaving the freed block available to the workspace without
  changing the allocator's policy or the main device's numerics.
- `prepare_for_device` routes CUDA↔CUDA moves through the bridge once
  `config.split_active` is set; CPU moves keep the pinned/async path.
- Graphs: split mode declines whole-forward, lag-1, `prepare_inputs` and per-block captures
  (`split_active`). Per-segment capture is Stage 6.
- Per-component load table and a cache/recurrent placement audit print after load: every
  cache layer must sit on the device its owning layer was mapped to.
- Auto-N is a Stage 1 skeleton: it uses actual safetensors bytes for each layer, the
  cache/recurrent sizes for layers below N, the front endcap bytes and the reserve, with
  a `KYOJIN_AUTON_SAFETY_GB` margin (default 0.25) for allocator slack. Components declared
  after the target (GLM's MTP/vision) are not yet subtracted before the target load;
  Stage 5 owns cross-component/pre-load auto sizing.
- JIT HIP loaders compile per (device, defines) and launch on the tensor's device stream;
  `kda_fused_h`, `gdn_*`, `gr_mix` and `ple_hip` are gfx11.5 WMMA/CI paths and refuse other
  archs. gfx12 runs the deterministic fla chain (`EXL3_KDA_FUSE=0` forced there) — see PR C
  for the fused port. `qsa_prefill_hip` readiness is cached per device, and all audited
  dsv41/gr/ple loaders select the calling device before every launch.

## Determinism / verification policy

- Split **off** must stay byte-identical to `main` (outputs and the serve env block).
- Split **on** is judged by run-to-run determinism (plain greedy), KLD/task metrics and the
  needle probe — cross-arch (gfx1201 vs gfx1151) is not expected to be bitwise.
- `tools/strix_halo/split_eval.py` (GLM) runs the battery: `--baseline`, `--front-layers`,
  `--needle-depth`, `--num-draft`, `--perf-tokens/--perf-reps`, `--compare a.json b.json`.

## Model support

See `doc/split-support-matrix.md` (GLM measured; Qwen and DSV4 adapters staged; Flint/Mimo
best-effort inheritors of the generic `Model` path). Core placement is model-agnostic:
adapters declare components and policies, unknown non-layer modules default to main.

## Known limitations (Stage 1)

- Auto-N uses actual text bytes/cache/endcaps but not components declared after the target
  load; Stage 5 finishes cross-component, pre-load sizing. For GLM: N=6 at ≤65K, N=5 for
  long context until the gfx12 KDA kernel lands.
- MTP on the split is not yet deterministic (`--num-draft 0` remains the deterministic mode).
- No per-segment graphs (Stage 6); split runs eager and pays ~0.95× decode (GLM, MTP).
- Qwen3.8 and DSV4 adapters are Stage 3/4; their component inventories are code-level only.
