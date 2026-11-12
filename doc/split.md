# Contiguous layer split

The layer split places the first N transformer layers of a model on one GPU (`part0`) and
the remaining layers on another (`part1`) in a single process. It is for models that do not
fit on one device, or for pairing a small fast card with a large unified-memory device.

Status: GLM-5.3-Flash is measured and supported by the GLM server. Qwen3.8-Flash-Next and
DeepSeek-V4.1-Flash adapters are staged; Flint and Mimo use the generic model path. See
[Model support](#model-support).

## How it works

    part0: layers 0..N-1 + endcaps          part1: layers N..last
        \______________________________________/
                    host-staged crossings

`part0` and `part1` are positions in the layer order, not device indices. Either physical
device can serve either part.

Every tensor that crosses the split goes through an asynchronous host-staged bridge:

    src tensor --D2H--> pinned slot ring --H2D--> dst tensor

Both legs are enqueued on the two devices' compute streams, so the host does not wait. The
destination is a fresh allocation. `KYOJIN_BRIDGE=0` disables the bridge; cross-device
moves then use `to_device` with a blocking host bounce instead of probing or using a direct
peer copy. An explicit `EXLLAMA_NO_P2P_COPY` still wins.

Measured configuration: an RX 9070 (`gfx1201`, 15.92 GiB) over USB4 as `part0` and a
118 GiB Strix Halo iGPU (`gfx1151`) as `part1`. The numbers below are measurements from
that pair, not assumptions of the interface.

## Quick start

The GLM server and the split harnesses accept the flags below. The environment-only path
is useful for wrappers that cannot pass CLI flags:

```sh
KYOJIN_PART0_LAYERS=6 \
KYOJIN_PART0_DEVICE=gfx1201 \
KYOJIN_PART1_DEVICE=gfx1151 \
  python tools/glm/serve.py --model /path/to/model
```

## Interface

| Flag | Environment fallback | Meaning |
| --- | --- | --- |
| `--part0-layers auto\|N` | `KYOJIN_PART0_LAYERS` | first N transformer layers on `part0`; 0 = split off |
| `--part1-device auto\|cuda:N\|gfxXXXX\|uuid\|name` | `KYOJIN_PART1_DEVICE` | `part1` device (`auto` = `cuda:0`) |
| `--part0-device auto\|cuda:N\|gfxXXXX\|uuid\|name` | `KYOJIN_PART0_DEVICE` | `part0` device (`auto` = the only other device when two are visible) |
| `--endcaps on\|off` | `KYOJIN_ENDCAPS` | keep text endcaps and whole MTP/vision components on `part0` (default on) |
| `--part0-reserve-gb` | `KYOJIN_PART0_RESERVE_GB` | VRAM held as a load-time slab on `part0` for workspace (default 1.2) |
| `--split-preset auto\|glm\|qwen\|ds41\|none` | `KYOJIN_SPLIT_PRESET` | per-model unsafe-environment preset; GLM needs none today |
| `--split-graphs auto\|on\|off` | - | reserved for per-segment capture; split currently runs graphs off |

`cuda:N` is the torch device index **after** `HIP_VISIBLE_DEVICES`, not the `rocm-smi`
index. Explicit selectors win in either position: `part0=cuda:1/part1=cuda:0` and
`part0=cuda:0/part1=cuda:1` are both valid. With exactly two visible devices, a part0-only
selector still works and `part1=auto` takes the other device. Two selectors that resolve to
the same device fail, and a one-device split fails with a clear error. With more than two
visible devices, `auto` requires an explicit selector and prints the inventory.

`KYOJIN_PART0_LAYERS` and `KYOJIN_PART1_DEVICE` also work in `model_init.init` paths
(examples and evaluation tools) without CLI flags. The legacy `EXL3_LOAD_SPLIT` budget knob
is not part of this interface and must not be combined with the new map.

## What gets placed

| Where | Contents |
| --- | --- |
| `part0` | first N text layers; with `--endcaps on`, the text allowlist (norm/hc/lm_head-like) and whole non-CPU MTP/vision components |
| `part1` | remaining text layers; MTP/vision when endcaps are off; unknown non-layer modules |
| CPU | the shared `embed_tokens` (`prefer_cpu`) stays host-side by design |
| cache | each KV/recurrent layer follows its owning layer; a placement audit prints after load |
| bridge | one bridge per config, covering the target, drafter and vision |

Text endcaps are selected by a class/key allowlist, not by per-model placement in the core.
“Whole component” means all non-CPU modules of that component.

## Memory and capacity

- The `part0` reserve is an inert load-time slab, not a process-wide allocator cap. It
  removes bytes from the pool while the model loads and is released at the first `part1`
  module. Size it to the part0 card.
- Split mode runs eager. Whole-forward and per-block graphs are disabled; per-segment
  capture is not implemented yet.
- Use `util/memguard.py` when loading near the limit.  It help save you from the dreaded OOM but slight adverse 
  performance effects have been noted (to be investigated further.)

## Model support

| Model | Status | Notes |
| --- | --- | --- |
| GLM-5.3-Flash | measured | text, MTP and vision; on a 16 GiB `gfx12` `part0`, N=6 at ≤65K and N=5 for long context |
| Qwen3.8-Flash-Next | staged, not load-tested | QSA/GDN/PLE components; separate dflash drafter config; QSA `gfx12` caveat below |
| DeepSeek-V4.1-Flash | staged | uses the DSV4 engine, not `Model.load` |
| Flint, Mimo | best effort | generic `Model` path; no dedicated adapter |

Kernel support is per architecture. The split owns placement and transport, not kernel
ports; a missing kernel fallback on a new architecture is a model/kernel issue, not a split
defect.

## Verification

- Split **off** must remain byte-identical to the parent branch, including outputs and the
  serve environment block. It must not enumerate devices or mutate split settings.
- Split **on** is judged by run-to-run determinism with plain greedy decoding, KLD/task
  metrics and the needle probe. A mixed-architecture pair is not expected to be bitwise.
- `tools/strix_halo/split_eval.py` runs the battery: `--baseline`, `--part0-layers`,
  `--part1-device`/`--part0-device`, `--needle-depth`, `--bench-depth`,
  `--bench-prompt passkey|summary`, `--bench-acceptance`, `--num-draft`,
  `--perf-tokens`/`--perf-reps`, `--vision-image PATH` and `--compare a.json b.json`.
  Its context-dependent part0
  default is harness tuning, not a split property; override it with `--part0-layers` or the
  `--long-ctx-part0-layers`/`--short-ctx-part0-layers` flags. `--bench-acceptance` uses
  long-form summary prompts and reports the drafted-token sample plus a 95% interval;
  the passkey prompt is too short for an acceptance comparison. `--vision-image` encodes
  one image through the split vision tower; the harness sets `EXL3_CONV_GEMM=1` as the
  safe gfx12 spatial-merge fallback.

## Known limitations

- MTP on the split is deterministic at 8K/32K/128K with N=4 in the measured
  configuration (two independent runs: 6/6 greedy prompts and identical long-prompt ids).
  The earlier N=6 128K battery used short prompts; N=6 long prefill OOMs a 16 GB part0,
  so N=4 is the long-context MTP configuration. Use `--num-draft 0` for configs outside
  this measured set.
- Per-segment graphs are not implemented. Split runs eager and pays about 0.95x decode on
  GLM with MTP.
- Auto-N uses text bytes, cache and text endcaps, but does not yet subtract components
  declared after the target load (MTP/vision). Leave headroom or set N explicitly.
- Qwen QSA sparse attention has an illegal-access report in the `gfx12` top-k path on
  `gfx1201`; keep QSA cache pages off `part0` until that kernel is fixed.
- Reverse orientation with MTP on is validated at 32K (N=42, endcaps off, MTP on
  `part1`) on the measured configuration.
- Same-architecture and RDNA3/3.5 pairs are selection-tested only until such hardware is
  available.
