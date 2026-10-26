# Derived public cold128K recipe

This is a new, path-parameterized helper, not the literal private runner used for the retained measurements. Its default invocation only prints a plan. The seven standard-library tests exercise planning, prompt-preparation normalization/re-exec with a changed HOME and inert execve/tokenizer entry, exact-length synthetic fitting with an inert tokenizer, actual native/API response validation, per-arm environment, a real detached-child SIGTERM/reap path, and the actual run-leg receipt path with a tiny process and synthetic HTTP/native fixtures. No test result in this directory is a model performance result.

The quoted-tilde/private-HOME repair passed independent source review. The actual selected-venv tokenizer/template preparation generated five new synthetic prompts of exactly 131,079 rendered tokens, followed by a completed twelve-leg full-model execution of these helper bytes. A separate frozen receipt reader replayed all twelve legs/four brackets and verified selected source/native/CK inputs and weight-file stat identities after process exit. The supervisor exited normally and the parent separately observed process/cgroup/port/compute-owner/GPU-idle release before the next experiment. This public copy is byte-identical to that executed helper and test source; it is not a separate model run at the copy path.

In that public-synthetic FP16 cold128K screen, descriptive native prefill rates were 426.1108 tok/s for stock Triton and 709.2824 tok/s for sparse CK at retention 0.30 (tokens divided by median duration across eight stock and four candidate legs). Median paired gain was 66.57266%, four of four brackets won, and median identical-control spread was 0.42556%. These are this run's rates, not guaranteed reproduction targets or a transfer of historical private-prompt results. Selected API text, choosing nonempty user content before reasoning fallback, matched between the repeated stock controls in all four brackets; sparse differed in three. That comparison is not full stored decoded-text/token/logit equality or a quality grade. See [results and limitations](../../../doc/ck_cold128_results.md).

The workload is newly generated public synthetic text. It contains no retained user prompts or completions and does not reconstruct the original prompt bytes. Settings match the clean PR122 screen class: FP16 K/V, capacity 163840, prefill chunks 4096, 131079 tokens after the real chat template, 64 greedy generated tokens, no drafter, sparse retention 0.30. Retention is an approximate attention setting, not a general quality approval or a physical cache-memory saving. `reasoning_effort='none'` suppresses the medium-effort system instruction; it does not disable thinking.

## Inputs

Use an already-built Python 3.12 ROCm Kyojin environment on gfx1151. Select the checkout containing the optional CK feature, its matching official native extension, the separately built CK library and the MiMo EXL3 pack. This helper does not install, fetch or rebuild the official engine extension. Ordinary per-kernel JIT is permitted during the discarded full-depth warmup.

The example paths below are placeholders. Keep source, model, native, CK library, prompt directory, preparation directory and result directory disjoint as checked by the helper. Prompt/result roots must not already exist. The prompt preparation sibling `.public-prompts-prep` is also fresh/no-overwrite. A failed attempt is retained; choose a new prompt/output name for a retry.

```bash
PYTHON=/path/to/kyojin/.venv/bin/python
SOURCE=/path/to/kyojin
NATIVE=/path/to/matching/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
CK=/path/to/ck-cache/libck_pagedkv.so
MODEL=/path/to/mimo-exl3
PROMPTS=/path/to/new/public-prompts
RESULTS=/path/to/new/public-results

COMMON=(--source "$SOURCE" --native "$NATIVE" --ck "$CK" --model "$MODEL"
        --python "$PYTHON" --prompts "$PROMPTS" --out "$RESULTS")

# Inert plan, including all four control/candidate/control2 brackets.
python3 cold128.py "${COMMON[@]}"

# Explicit tokenizer/template preparation. Imports the selected already-built
# engine, but allocates no model weights and sends no inference request.
python3 cold128.py "${COMMON[@]}" --make-prompts

# Explicit full-model benchmark. Run detached/supervised so a client-terminal
# lifetime cannot cancel multi-minute inference. Never overlap another scored
# GPU workload, NPU execution, compilation or heavy CPU/IO activity.
python3 cold128.py "${COMMON[@]}" --execute
```

No clocks, firmware, driver, power policy, services or boot settings are changed. The caller owns the measurement window. The helper requires no other GPU/NPU compute owner and GPU idle at each launch, checks policy consistency, and refuses an occupied loopback port. It terminates and reaps only its own child, then records closed port, compute owners and GPU idle. SIGTERM unwinds cleanup; SIGKILL/power-loss cleanup is not guaranteed. Waiting heartbeats do not impose an overall benchmark deadline.

## What is measured

Each of twelve legs launches a fresh server with private writable runtime caches. It runs one full-depth unscored warmup and one distinct measured prompt, proving different first cache pages and zero native cached tokens on both requests. All three arms within a bracket use exactly the same measured prompt. Input source hashes, model metadata hashes and weight-file stat identities are captured before launch and checked between legs; weights are not bulk-hashed. This is ordinary cooperative experimental binding, not an adversarial sandbox or complete transitive toolchain authentication.

Control and control2 use stock Triton dispatch with CK absent. Candidate installs sparse CK before import and must report actual measured page-dropping deltas plus the selected CK library mapping. The historical local dense control used FullK CK instead of stock Triton; its result is a separate contrast and cannot be transferred to this run.

Primary time is the engine's native `time_prefill` in seconds, cross-checked against Chat Completions `prompt_ms`, counts and rate. It is a host-observed engine prefill interval ending before the first regular generation forward, not HTTP TTFT and not pure GPU kernel time. HTTP wall time is retained separately. Raw response bytes, full stored model text, native counters, per-request sparse counters, environment, logs, maps, cleanup and bracket results remain under RESULTS.

For each bracket:

    base = (control_s + control2_s) / 2
    gain_percent = 100 * (base / candidate_s - 1)
    spread_percent = abs(100 * (control_s / control2_s - 1))

The screen gate requires all four valid brackets, four wins, median paired gain greater than 2%, and gain greater than median identical-control spread. Descriptive rates use tokens divided by median native duration and are not the median paired gain. COMPLETE means measurement completed; inspect `screen_gate` rather than treating completion as success. Instrument/engagement/cleanup failures are separate from a valid negative screen. Exact text differences are retained; they are not a calibrated quality score, and no default-enablement or broad quality adoption follows.

## Tests

```bash
python3 -I -S -B test_cold128.py
python3 -I -S -B -O test_cold128.py
```

The test fixtures deliberately use synthetic data and cannot report model speed. Their real subprocess tests open no accelerator devices and signal only children they created.
