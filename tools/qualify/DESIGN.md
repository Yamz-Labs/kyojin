# qualify — one command per backend: converted pack -> verdict + qualify.json

`python tools/qualify/qualify.py --backend cuda|rocm --pack DIR [--ref BF16_DIR]`
(MLX later: same stage list, another `backends.py` profile). Zero hand flags: everything comes from the pack.
Builds on the existing tools: streamed BF16 reference, KV KL, max-context search, greedy sha, N=1..4 bench, the template rule of
`lane-kv-conformance.py`, `/health` start-up checks and a scripted Hermes eval suite.

## Stages (each = one resumable unit, `<out>/<stage>.json`, skipped when done and its input fingerprint is unchanged)
| # | stage | needs GPU | what | drill-0 gap |
|---|---|---|---|---|
| 1 | `fetch` | no | shards vs `model.safetensors.index.json` (size + names), license field, disk floor (`--disk-floor-gb`, default 40), optional `--hf REPO` download under the floor; `--ref` BF16 dir verified the same way | 1 |
| 2 | `meta` | no | `pack.py`: arch family, layers/heads/head_dim, KV bytes/token per bits, chat template (pack's own `chat_template.jinja`, render check with tools before system), draft path (vendor MTP head > n-gram `mix` > `off`), vision tower, serve flags | 3, 8 |
| 3 | `parity` | yes (lock) | streamed BF16 reference (per-layer, `refs/<family>.py`) on the domain docs (chat fr/en, code, tool, wiki, fr, tech), pack logits, KL / top-1 / dNLL per domain, per-layer SQNR read from the pack | 5 |
| 4 | `kv` | yes (lock) | KL of k8v8/k6v6/k4v4/k3v3 against the reference tail at 32K (real quantizer); owner rule picks the cache: smallest bits >= 6 whose KL vs k8v8 <= `--kv-kl-max` (0.02), never below Q6 | 4 |
| 5 | `maxctx` | yes (lock) | for each session count in `--sessions 1,3` (and 4 if asked): upper bound from the KV formula + measured load boundary, then bisection step 2048: start with the pack flags, startup guard, 2 distinct near-full (97 %) prompts back to back, no `out of memory` in the server log. Pure bisection in `search.py` (unit-tested with a fake probe) | 4 |
| 6 | `serve` | yes (lock, ONE load) | at the chosen KV and max `-c`: template/tool conformance (`/apply-template`), speed matrix (decode at ~0.5K, prefill 8K, curve 2K..max ctx, N=1..4 agg + TTFT), greedy sha set (3 fr x 256, run twice: self-equality) | 7 |
| 7 | `smoke` | yes (same server) | `--smoke-tasks` hermes_opt/eval tasks through the real Hermes CLI (`run_arm.sh ... --allow-local --jobs 1`); fallback: the nanb scripted tool loop (`smoke_agent.py`) when the eval suite is absent | 6 |
| 8 | `report` | no | `REPORT.md` + `qualify.json` (schema 1) | - |

Stages 6 and 7 share one server (one load per model). Stages 3-7 are grouped in GPU **jobs** by the planner; each job takes
the optional lock files of `QUALIFY_LOCKS` (in order, one holder, flock waits), checks
MemAvailable >= 4 GiB (and >= the pack's stated host peak for conversions; none here), and releases between jobs. Servers are started and
killed by PID only; readiness = `GET /health` == 200 (never a port check); the server log is scanned for OOM.

## Planner (pure, no GPU: `plan.py`)
`build_plan(meta, backend, args, state)` -> ordered list of `Stage(name, job, est_s, needs, flags)`. `--dry-run` prints it with the
derived serve command and the cost estimate, `--stage a,b` selects a subset (dependencies that are not done are listed and refused
unless `--force-deps`), `--resume` is the default (a stage with a valid done-file is skipped; `--redo STAGE` clears it).
Estimates come from `est.py` constants measured in drill 0 (e.g. 3 min per max-ctx probe on the 3090, ~100 tok/s ref stream/layer).

## Generic vs per-arch
Generic: fetch, meta (config-driven), docs, KL/top-1 maths, KV rule, search, matrix, sha, smoke, report, locks, server lifecycle.
Per arch family (small adapter in `pack.py` FAMILIES + `refs/`): which layers hold a KV cache (full attention only: dense Llama = all, hybrid GDN = every
4th, SWA/MLA/DSA = their own bytes/token formula), the streamed BF16 reference forward (`refs/llama_dense.py` exists; GDN/MoE/MLA refs
are adapters to write once per family: the stage reports `SKIPPED: no reference adapter for <family>` and the rest still runs,
the verdict says parity is unmeasured), the serve entry point (`tools/qwen/serve.py` for EXL3 Llama/Qwen, `tools/glm/...` for GLM).

## Outputs
`qualify.json`: `{schema, pack{name,sha_of_index,arch,bpw}, backend, git, stages{name:{status,wall_s,data}}, serve{cmd,kv,ctx{sessions:max_c}},
quality{by_domain{kl,top1,dnll}, kv{bits:{kl,top1}}}, speed{...}, sha[...], smoke{...}, verdict{pass, reasons[]}}`. The lanes read `serve.cmd`,
`serve.ctx`, `serve.kv` (no hand flags). Verdict gates (all configurable): mean KL <= 0.15, no domain top-1 < 0.75, KV >= Q6 kept, maxctx >= 32K for 1 session,
sha self-equal, smoke >= 80 % of the tasks, no OOM. Verdict is advisory; numbers are the deliverable.

## Not in v1 (explicit)
GDN/MoE/MLA reference adapters (only dense Llama), MLX profile, Strix real run, day-0 autotune (q2 will write into the pack cache; `serve` has a hook
`--tuning-cache DIR` reserved), automatic HF fetch of gated repos.

## Findings while building it (feed the converter / serve owners)
- `serve.py` loads the MTP drafter for every `--draft-policy` but `off`: a head-less pack (Nanbeige) must be served `off`; `pack.draft_policy` derives it from
  `mtp_num_hidden_layers`. n-gram lookup without an MTP head needs a server flag (drill-0 gap 8 stays open).
- The per-layer SQNR is only in the converter log, not in the pack: the converter should write `convert.log` into the pack (`parity` reads `<pack>/convert.log`
  or `--conv-log`).
- Servers run with `PYTHONPATH=<engine checkout>` (the built extension lives there): `--repo` is that checkout, not the qualify tree.
- `standard_cal_data` is also the quantizer's calibration source: the domain docs use late offsets (held-out in practice, not by construction). A truly
  held-out set is a v2 item.
- The Hermes smoke needs an eval suite with `lib/taskdefs.py`: point `QUALIFY_EVAL_DIR` at it.
