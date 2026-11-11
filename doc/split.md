# Contiguous layer split (part0 + part1 devices)

*Status: Stage 1.5 of the universal roadmap (see `.kilo/plans/1794309410242-universal-layer-split-roadmap.md`).
The core is policy-driven (`SplitPlan`/`ComponentPolicy`), devices are selected by
position rather than by architecture, the GLM server is the first adapter, and the part0
reserve is a layout-neutral slab. This file is the durable user/ops reference; the
roadmap owns the staging, and `docs-notes/split-log.md` (untracked) owns raw
measurements.*

## What it is

One process places the first N transformer layers plus the small endcaps of a model on
one GPU (`part0`) and the rest on another (`part1`), with an asynchronous host-staged
bridge on every crossing:

    src tensor --D2H--> pinned slot ring --H2D--> dst tensor

`part0` and `part1` are positions in the layer order, not CUDA indices: `part0` owns
layers `0..N-1` (plus its endcaps) and `part1` owns the remaining layers. Either physical
device may serve either part. Earlier revisions, the roadmap and
`docs-notes/split-log.md` call these `front`/`main`; the current flags and env names are
`part0`/`part1` only (`--front-*` and `KYOJIN_FRONT_*` are not accepted).

Both legs are enqueued on the two devices' own compute streams; the host never blocks. The
destination tensor is a fresh allocation, so callers may keep using it after the slot cycles.
`KYOJIN_BRIDGE=0` falls back to `to_device` (split still runs, crossings block).

Devices are not tied to a role. Either device may serve `part0` or `part1`, and the split
is not restricted to a particular architecture pair. Kernel support remains per-arch: the
split owns placement and transport, not kernel ports.

**Measured configuration.** The development box is an RX 9070 (gfx1201, 15.92 GiB) over
USB4 as `part0` and a 118 GiB Strix Halo iGPU (gfx1151) as `part1`. All numbers in
`docs-notes/split-log.md` and the support matrix come from that pair unless a note says
otherwise; they are measurements, not assumptions of the interface.

## Interface

| Flag (CLI) | Env fallback | Meaning |
| --- | --- | --- |
| `--part0-layers auto\|N` | `KYOJIN_PART0_LAYERS` | first N transformer layers on `part0`; 0 = off |
| `--part1-device auto\|cuda:N\|gfxXXXX\|uuid\|name` | `KYOJIN_PART1_DEVICE` | `part1` device selection (`auto` = `cuda:0`) |
| `--part0-device auto\|cuda:N\|gfxXXXX\|uuid\|name` | `KYOJIN_PART0_DEVICE` | `part0` device selection (`auto` = the only other device when two are visible; otherwise a selector is required) |
| `--endcaps on\|off` | `KYOJIN_ENDCAPS` | move the text allowlist (norm/hc/lm_head-like) and the whole MTP/vision component (all non-CPU modules) to `part0` (default on) |
| `--part0-reserve-gb` | `KYOJIN_PART0_RESERVE_GB` | VRAM held as a load-time slab on `part0` for workspace (default 1.2; a tuning value, size it to the card) |
| `--split-preset auto\|glm\|qwen\|ds41\|none` | `KYOJIN_SPLIT_PRESET` | per-model unsafe-env preset slot; GLM needs none today |
| `--split-graphs auto\|on\|off` | — | per-segment capture (Stage 6); split forces graphs off until then |

`cuda:N` is the torch device index **after** `HIP_VISIBLE_DEVICES`, not the `rocm-smi`
index. Explicit selectors win in either position: `part0=cuda:1/part1=cuda:0` and
`part0=cuda:0/part1=cuda:1` are both valid. A part0-only selector keeps working when it
lands on `cuda:0`: with exactly two devices, `part1=auto` takes the other device. Two
explicit selectors that resolve to the same device fail, and a one-device split fails with
a clear error. With more than two visible devices, `auto` for either part requires an
explicit selector and prints the inventory.

`KYOJIN_PART0_LAYERS=…` without CLI flags also works in `model_init.init` paths
(examples/eval tools); `KYOJIN_PART1_DEVICE` is honoured there too. The legacy
`EXL3_LOAD_SPLIT` budget knob is **not** part of this interface; combining it with the
new map must error.

## Core objects (Stage 1.5)

- `exllamav3/device_map.py`: `SplitPlan` owns every component the process places.
  `ComponentPolicy` declares `part0_eligible`, `endcaps_part0`, the endcap allowlist,
  arch gate and unsafe env. `ComponentMap` resolves one component's contiguous boundary.
  `resolve_devices(part0, part1, devices=None)` selects by position; `DeviceInfo` lets the
  selection tests run without a GPU. Unknown non-layer modules are logged and kept on
  `part1`; auxiliary components never move to `part0` implicitly.
- `exllamav3/bridge.py`: one bridge per *config*, installed by `SplitPlan.install_bridges`
  for the target, a separate draft config, vision and any extra config.
- `exllamav3/serve_split.py` (shared with `tools/kyojin_serve_split.py`): CLI/env parsing,
  device resolution, plan/load/report orchestration, split-safe env. GLM is the first
  adapter; the helper is pure stdlib at import so a server can apply `SPEED_ENV` first.
- `model_init.py` uses the same plan for the `KYOJIN_PART0_LAYERS` env-only path.

## Load-time effects worth knowing

- The `part0` reserve is an **inert device slab**, not `set_per_process_memory_fraction`.
  The Stage 0 replay showed a process-wide allocator cap makes 128K split runs bimodal;
  the slab removes bytes from the pool and is dropped at the first `part1` module (or after
  the last component), leaving the freed block available to the workspace without
  changing the allocator's policy or the `part1` device's numerics. The 1.2 GiB default is
  a tuning value for a small part0 card, not an architecture rule; Stage 5 derives it
  from the device total.
- `prepare_for_device` routes CUDA↔CUDA moves through the bridge once
  `config.split_active` is set; CPU moves keep the pinned/async path.
- Graphs: split mode declines whole-forward, lag-1, `prepare_inputs` and per-block captures
  (`split_active`). Per-segment capture is Stage 6.
- Per-component load table and a cache/recurrent placement audit print after load: every
  cache layer must sit on the device its owning layer was mapped to. The table's `devices`
  column lists the devices that actually hold tensors, not the configured part0/part1 pair;
  an empty range prints `-`. CPU-resident modules (the shared embedding) are reported
  separately, so a whole MTP/vision component with endcaps on shows `part1 -` and only
  part0 in `devices`.
- Auto-N is a Stage 1 skeleton: it uses actual safetensors bytes for each layer, the
  cache/recurrent sizes for layers below N, the part0 endcap bytes and the reserve, with
  a `KYOJIN_AUTON_SAFETY_GB` margin (default 0.25) for allocator slack. Adapter tuning
  stays in the adapter: GLM caps auto at five layers for `-c > 65536` only on a gfx12
  part0, where the deterministic split KDA chain adds a prefill workspace the byte budget
  does not yet know. Components declared after the target (GLM's MTP/vision) are not yet
  subtracted before the target load; Stage 5 owns cross-component/pre-load auto sizing.
- JIT HIP loaders compile per (device, defines) and launch on the tensor's device stream;
  `kda_fused_h`, `gdn_*`, `gr_mix` and `ple_hip` are gfx11.5 WMMA/CI paths and refuse other
  archs. In split mode, gfx12 runs the deterministic fla KDA chain (`EXL3_KDA_FUSE=0`
  forced there); other arches keep the fused path. See PR C for the fused gfx12 port.
  `qsa_prefill_hip` readiness is cached per device, and all audited dsv41/gr/ple loaders
  select the calling device before every launch.

## Determinism / verification policy

- Split **off** must stay byte-identical to `main` (outputs and the serve env block), and
  must not enumerate devices or mutate split env.
- Split **on** is judged by run-to-run determinism (plain greedy), KLD/task metrics and the
  needle probe — a mixed-arch pair is not expected to be bitwise.
- `tools/strix_halo/split_eval.py` (GLM) runs the battery: `--baseline`, `--part0-layers`,
  `--part1-device`/`--part0-device`, `--needle-depth`, `--num-draft`,
  `--perf-tokens/--perf-reps`, `--compare a.json b.json`. Its context-dependent part0-layer
  default is harness tuning exposed as `--long-ctx-part0-layers`/`--short-ctx-part0-layers`,
  not a property of the split.

## Model support

See `doc/split-support-matrix.md` (GLM measured; Qwen and DSV4 adapters staged; Flint/Mimo
best-effort inheritors of the generic `Model` path). Core placement is model-agnostic:
adapters declare components and policies, unknown non-layer modules default to `part1`.

## Known limitations (Stage 1.5)

- Auto-N uses actual text bytes/cache/endcaps but not components declared after the target
  load; Stage 5 finishes cross-component, pre-load sizing. On the measured 16 GB gfx12
  part0: N=6 at ≤65K, N=5 for long context until the gfx12 KDA kernel lands.
- MTP on the split is not yet deterministic (`--num-draft 0` remains the deterministic mode).
- No per-segment graphs (Stage 6); split runs eager and pays ~0.95× decode (GLM, MTP).
- Qwen3.8 and DSV4 adapters are Stage 3/4; their component inventories are code-level only.
- Same-arch and RDNA3/3.5 pairs are selection-tested only until such hardware is available.
  A missing kernel fallback on a new arch is a model/kernel support issue, not a split defect.
