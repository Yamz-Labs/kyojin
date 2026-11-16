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
| `--expandable-segments auto\|on\|off` | `KYOJIN_EXPANDABLE_SEGMENTS` | allocator mode, applied before `import exllamav3`; `auto` only turns it off for large-context unified loads |
| `--oom-guard auto\|warn\|off` | `KYOJIN_OOM_GUARD` | pre-flight per-device guard (`auto` refuses, `warn` logs, `off` disables) |
| `--oom-guard-reserve-gib` | `KYOJIN_OOM_GUARD_RESERVE_GIB` | per-device headroom, e.g. `gfx1151=4.5,gfx1201=0.8` (default 4.5 unified / 0.8 dedicated) |
| `--oom-guard-workspace-gib` | `KYOJIN_OOM_GUARD_WORKSPACE_GIB` | per-device prefill workspace margin (default 2.0 bulk / 1.5 other) |
| `--oom-guard-safety-gib` | `KYOJIN_OOM_GUARD_SAFETY_GIB` | extra per-device capacity margin |
| `--oom-guard-device-floor` | `KYOJIN_OOM_GUARD_DEVICE_FLOOR` | per-device free-memory floor, shared with the supervisor |

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

The measured 118 GiB Strix Halo iGPU has a usable ceiling of about **113.3 GiB** (the
~4.5 GiB gap is driver/allocator overhead, not free capacity). Recorded failure points on
the measured GLM pack: 128K split runs complete (peak ~110.7 GiB on the iGPU); a 512K
cache needs `expandable_segments:False` to load; 640K is killed by the supervisor before
a torch OOM; 768K hard-locked the box. Do not probe above 128K for MTP-on runs.

### Pre-flight guard

The loader estimates each device's footprint from actual bytes: layer/endcap/component
weights, KV/recurrent cache, the part0 reserve slab and a prefill workspace margin. It
compares that with the device's usable/free memory and, for unified devices, with host
`MemAvailable - 6 GiB`. `--oom-guard auto` (the split default) refuses before allocating
weights, cache or slabs, and prints a per-device table plus the chosen N/context and the
largest N/context that fits. `warn` only logs; `off` disables it. The check is read-only:
it changes no allocator setting and no split-off path.

Defaults are tuning values, not architecture rules: `4.5` GiB reserve on a unified/GTT
device, `0.8` GiB on dedicated VRAM, `2.0`/`1.5` GiB prefill workspace on the bulk/other
device (calibrated: 4.0 GiB refused measured-good 8K MTP-on/128K points and 2.0 brackets
their peaks), and a `6` GiB host floor. All accept per-device selector lists, e.g.
`--oom-guard-reserve-gib gfx1151=4.5,gfx1201=0.8`.

### Allocator mode

`--expandable-segments off` (or `EXL3_EXPANDABLE_SEGMENTS=0`, or
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`) is applied before the library import.
It recovers about **0.5 GiB** of fragmentation around a large cache and is the mode that
loaded the 512K case; it does not add capacity. The pre-flight workspace estimate is
conservative, so `auto` may still refuse a load-only 512K run that fits at 113.28 GiB
measured: pass `--oom-guard warn` (or a lower `--oom-guard-workspace-gib`) for that
measured point once the supervisor is armed. `auto` keeps expandable segments on
except for unified loads at 512K context or above whose estimate is within 8 GiB of the
usable ceiling. `on`/`off` are explicit overrides.

### External supervisor

Use `exllamav3/guard.py` (the compatibility path `util/memguard.py` execs it) when loading
near the limit. It watches host `MemAvailable`, per-device free memory (`mem_info_gtt_*`/
`mem_info_vram_*` on AMD, `nvidia-smi` on NVIDIA) and can release page cache:

```sh
python exllamav3/guard.py --floor-gib 6 --device-floor-gib 4 \
  --fadvise-dir /path/to/model -- python tools/glm/serve.py ...
```

Per-device supervisor floors use `--device-floor card1=4.5,nvidia:0=2` (or
`--device-floor-gib` for every watched device); the supervisor has no torch selector, so
`cardN`, `nvidia:N`, `amd`, `nvidia` and `default` are accepted, and it degrades to the host
floor when a per-device query is unavailable. On a host with several visible devices a
global `--device-floor-gib` is ignored (with a warning): use explicit selectors so an
unrelated GPU cannot trigger a kill. The serve scripts use a
`MEMGUARD=auto|1|0` wrapper: `auto` enables it for a unified/GTT device or a large model
and logs the chosen floors. Keep the supervisor external; an
in-process watchdog cannot be relied on while the allocator or kernel is stalled. Do not
run MTP-on probing with the retired `split_bench_ctx.py`; use `split_eval.py --bench-depth`.

## Model support

| Model | Status | Notes |
| --- | --- | --- |
| GLM-5.3-Flash | measured | text, MTP and vision; on a 16 GiB `gfx12` `part0`, N=6 fits short context (MTP off), N=4 is the 128K MTP setting, and N=5 fits 500K |
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
- `--kld-corpus PATH --kld-slices N --kld-len L --kld-out FILE` teacher-forces corpus
  slices and `--kld-compare BASE SPLIT` reports KLD both ways, top-1 agreement and PPL.
  The short battery also reports programmatic task checks. Use these for split-on quality:
  a mixed-architecture pair is not bitwise.
- `tools/strix_halo/split_sessions_smoke.py` sends N concurrent requests at a running
  server and checks that every response succeeds and that the requests overlap.
- `tools/strix_halo/split_startup_health.py` polls `/health` during load and checks the
  stage trace; run it with and without `--no-vision` and with `--num-draft 0`.

## Known limitations

- Split-on is deterministic run-to-run, but it is **not bitwise** equal to a single
  device on a mixed-architecture pair. Measured on GLM, N=4: 4/6 short greedy prompts
  identical, task checks 5/6 vs 4/6, PPL 9.374 vs 9.401 and KLD 2.08e-2 with 94.52%
  top-1 agreement over 1022 positions. The distributions and task accuracy are
  unchanged; a few near-tie argmax positions flip. Use KLD/task metrics for split-on.
- At 500K context the split fits the measured pair (`part0` 10.75 GiB, `part1` 100.2 GiB
  load-time peak). A single 118 GiB device did not have headroom for the same context
  there; size for capacity, not only speed.
- Concurrent serving is smoke-tested at 4 sessions with MTP on (4/4 responses overlap,
  no queue or slot error). A heavier mixed chat/image/tool workload is future work.
- MTP on the split is deterministic at 8K/32K/128K with N=4 in the measured
  configuration (two independent runs: 6/6 greedy prompts and identical long-prompt ids).
  The earlier N=6 128K battery used short prompts; N=6 long prefill OOMs a 16 GB part0,
  so N=4 is the 128K MTP configuration and the 500K capacity run used N=5. Use
  `--num-draft 0` for configs outside this measured set.
- Per-segment graphs are not implemented. Split runs eager; measured GLM decode is -3.4%
  at 8K and +1.3% to +8.1% at 32K/128K versus single device with MTP on.
- Auto-N subtracts the MTP drafter and vision tower when the adapter predeclares them
  before load (the GLM server does). The default 2 GiB safety covers the measured 32K
  MTP+vision warm-up (1 GiB still chose N=6 and OOMed; 1.5/2/3 GiB chose N=5), but it is
  a fixed stopgap. Stage 5 sizes the forward workspace from the context and cache so
  short context can use more part0 layers and long context reserves more;
  `KYOJIN_AUTON_SAFETY_GB` remains the override.
- Qwen QSA sparse attention has an illegal-access report in the `gfx12` top-k path on
  `gfx1201`; keep QSA cache pages off `part0` until that kernel is fixed.
- Reverse orientation with MTP on is validated at 32K (N=42, endcaps off, MTP on
  `part1`) on the measured configuration.
- Same-architecture and RDNA3/3.5 pairs are selection-tested only until such hardware is
  available.
