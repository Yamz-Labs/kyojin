# Contiguous layer split (iGPU + eGPU)

*Status: Stage 0 of the universal roadmap (see `.kilo/plans/1794309410242-universal-layer-split-roadmap.md`).
This file is the durable user/ops reference; the roadmap owns the staging, and
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
| `--endcaps on\|off` | `KYOJIN_ENDCAPS` | non-layer modules (norm/head/MTP/vision) follow the front; `auto` later |
| `--front-reserve-gb` | `KYOJIN_FRONT_RESERVE_GB` | VRAM left free on the front for workspace (default 1.2) |
| `--split-preset` | `KYOJIN_SPLIT_PRESET` | per-model unsafe-env preset (Stage 3+) |
| `--split-graphs` | — | per-segment capture (Stage 6); until then split forces graphs off |

`KYOJIN_FRONT_LAYERS=…` without CLI flags also works in `model_init.init` paths
(examples/eval tools). The legacy `EXL3_LOAD_SPLIT` budget knob is **not** part of this
interface; combining it with the new map must error.

## Load-time effects worth knowing

- The front reserve caps the front card while its segment loads and is **released at the
  first main module**: on ROCm a process-wide allocator cap perturbs main-device numerics
  (map-zero control diverged from baseline 3/6 until the release was added). Never cap main.
- `prepare_for_device` routes CUDA↔CUDA moves through the bridge once
  `config.split_active` is set; CPU moves keep the pinned/async path.
- Graphs: split mode declines whole-forward, lag-1, `prepare_inputs` and per-block captures
  (`split_active`). Per-segment capture is Stage 6.
- JIT HIP loaders compile per (device, defines); `kda_fused_h` is a gfx11.5 WMMA kernel and
  refuses other archs (`vendor/fla/hip/kda_fused_h_hip.py:supported`). gfx12 runs the
  deterministic fla chain (`EXL3_KDA_FUSE=0` forced there) — see PR C for the fused port.

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

## Known limitations (Stage 0)

- Endcaps is still "all non-layer modules to the front" — replaced by an allowlist in Stage 1.
- Auto-N uses the old tier assumptions; actual-byte sizing is Stage 5. For GLM: N=6 at ≤65K,
  N=5 for long context until the gfx12 KDA kernel lands.
- MTP on the split is not yet deterministic (`--num-draft 0` remains the deterministic mode).
- No per-segment graphs (Stage 6); split runs eager and pays ~0.95× decode (GLM, MTP).
