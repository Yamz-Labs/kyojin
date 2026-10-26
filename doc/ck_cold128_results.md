# Cold128K results and quality limits

This supplements, rather than replaces, the short `tools/bench.sh` screen already in the PR. The long-context result uses a separate cold-prefill protocol; the two benchmark classes must not be compared as if they were one measurement.

## Completed clean-PR122 cold128K screen

Two paths were compared on the same gfx1151 machine: stock current-main Triton versus the default-off CK sparse implementation in PR122. The benchmark used a strict-clean source export, not a Git checkout of the PR head. A separate bounded retained-file comparison matched all fourteen PR-changed feature files in that export to the experiment's retained source-inventory pins and the corresponding blobs of commit `3f1be68b77c965ff7a019fcb56a29752e3b7ddc0`. That establishes selected feature-file byte identity, not equality of the entire source/build/runtime closure or continuous prelaunch immutability. The frozen input contract selects the same prebuilt official engine extension and local model pack for both paths. Bound pack metadata reports EXL3 2.5-bit weights; the input contract includes the `zz-e2e-step120.safetensors` overlay. Large weights were stat-pinned, not fully content-hashed. This is not an untouched upstream checkpoint.

Recorded settings: FP16 global K/V, capacity 163,840 tokens, base prefill chunk 4,096, batch one, no drafter/vision configured, 131,079 rendered prompt tokens, 64-token greedy continuation budget, and sparse retention 0.30 rather than the feature's default 0.50. MiMo's recurrent sliding-window state is separate from the global-cache precision label. `reasoning_effort='none'` omits the medium-effort template instruction; it does not mean thinking-disabled.

The twelve recorded legs have distinct worker PIDs and private runtime-cache roots. Each completed one unscored full-depth warmup followed by one distinct measured prompt. All three legs within each bracket shared exactly the same measured prompt; prompts differed across brackets. Every measured request reported zero cached tokens. The active CK wrapper reported real per-request page dropping, not only an enabled flag or a loaded library. No separate private first-page digest is presented here, and a no-drafter configuration is not being relabeled as a retained proposal-count measurement.

| Descriptive prefill rate | Stock Triton | PR122 sparse 0.30 |
|---|---:|---:|
| Tokens divided by median native prefill duration (8 stock / 4 candidate durations) | 424.0355 tok/s | 710.4487 tok/s |

Primary statistic: four control/candidate/control2 brackets, with `base=(control_s+control2_s)/2`, `gain=100*(base/candidate_s-1)` and repeat spread `abs(100*(control_s/control2_s-1))`. The median paired gain was **67.65766%**, all four brackets were candidate wins, and median identical-control spread was **0.25554%**. This meets the predeclared finite screen gate (>2%, four wins, gain above median spread); it is not statistical equivalence, broad quality acceptance or a default change.

Per-bracket gains: 67.57782%, 68.21760%, 67.65542%, 67.65990%. Native `time_prefill` is authoritative, with API token counts, seconds/milliseconds and reported rates cross-checked. HTTP wall time was retained separately; these rates are neither HTTP TTFT nor pure GPU-kernel time. The complete producer and independent receipt/arithmetic replay cover all twelve legs. All twelve worker results record return code zero. The separately bound parent acceptance records parent return code zero and one terminal marker; a separate post-run release snapshot records the parent absent, port closed, no compute owners and GPU idle. These are retained observations, not a new or continuous host-exclusion check.

In all four brackets, the two unchanged stock controls produced equal full stored decoded text. Every sparse candidate's text differed from its control. This is not a lossless result, and the continuations were not graded for answer quality.

## Completed common-history logit diagnostic

Separate from timing, eight fresh-process model captures completed: plain, plain-repeat, CK keep-all and CK sparse for each of FP16 and packed-int8 global caches. The eight capture results record one 131,079-token prompt, capacity 163,840, chunk 4,096 and 64 continuation positions per precision. The bound, independently reviewed capture source snapshots logits before invoking the sampler, then substitutes the reference token for each non-plain arm. The stored results record the pre-sampler capture path and common-history forcing; the source checks final emitted IDs against the precision-specific plain reference. This is source-and-receipt support, not a fresh raw-tensor replay in this supplement. This is a small correlated continuation diagnostic, not corpus perplexity or 64 independent quality examples.

| Global cache | Arm versus its plain reference | Top-1 matches / 64 | Mean KL(reference ∥ candidate) | Mean reference-token NLL delta, nat |
|---|---|---:|---:|---:|
| FP16 | Plain repeat | 64 | 0 | 0 |
| FP16 | CK keep-all | 58 | 0.00876574 | +0.0172508 |
| FP16 | CK sparse 0.30 | 59 | 0.0322864 | -0.00797314 |
| Packed int8 | Plain repeat | 64 | 0 | 0 |
| Packed int8 | CK keep-all | 61 | 0.00939799 | -0.000722613 |
| Packed int8 | CK sparse 0.30 | 62 | 0.0222333 | -0.0211626 |

Both plain repeats were bit-exact over the stored valid-vocabulary logits. CK diverged even without dropping pages, so sparsity alone cannot explain the difference. That neither establishes a CK defect nor rules out additional sparse error. KL values against a common reference are not additive: subtracting keep-all KL from sparse KL would not isolate sparse-only damage. A negative mean NLL delta means the mean reference-token NLL decreased over these selected forced-history positions; individual positions can still worsen. It does not establish improved answers or general quality approval. No negligible-loss threshold has been calibrated or passed here.

Packed-int8 rows are not identical-effective-input stock/CK comparisons: the selected CK adapter attends dequantized packed past K/V plus original FP16 current-chunk K/V, while stock's selected staged prefill attends dequantized packed past and current K/V. This source-supported difference remains at keep-all; its contribution to the retained logits has not been measured.

## Historical correspondence and scope

PR122 is not the complete native implementation behind the separate historical 708.4826 tok/s result. That historical dense/sparse comparison used the same FullK binary for both arms; stock-off in this PR selects normal dispatch, not historical dense FullK. Source mapping identified historical full-width K load/LDS staging and its six-panel contraction schedule, plus QK V2 selection, as missing from PR122's native specialization (which selects V2R1 under its actual predicates). The retained historical local comparison was 517.4780 to 708.4826 tok/s, median paired gain 36.65878%; it must not be relabeled as the new 424.0355 to 710.4487 comparison. Source/binary correspondence has retained-pin and end-of-run limits.

The separate FullK successor is outside this PR head and this supplement's measurement claim. No FullK build, consumer-error, selected-function-byte identity, marginal model gain or publication approval is asserted here; its evidence and evaluation are separate.

## Completed public-helper reproduction screen

A separately derived path-parameterized [cold128 helper](../tools/strix_halo/ck_cold128/README.md) generated new public synthetic prompts without retaining private user input/completion bytes in the public files. It is not the literal runner used for the earlier private-prompt or historical measurements and does not reconstruct their prompt bytes. Its quoted-tilde/private-HOME preparation repair was independently source-reviewed. The actual repaired helper re-executed under the selected venv, generated five prompts of exactly 131,079 rendered tokens using the real tokenizer/template, and then completed its full twelve-leg model campaign. The shipped helper and test sources are byte-identical copies of that executed recipe, not separate model runs at their new repository paths.

This is a separate same-machine public-synthetic screen, not an extra four replications of the earlier prompt set. It selects the same strict-clean feature export, matching prebuilt official extension, separately built clean CK specialization and effective local EXL3 model pack. Settings remain FP16 global K/V, capacity 163,840, base chunk 4,096, batch one, no drafter, 131,079 rendered prompt tokens and 64 greedy continuation tokens, with sparse retention 0.30. The 2.5-bit pack and bundled overlay qualification above still applies. No FullK successor, NPU acceleration or speculative decoding is part of this contrast.

| Descriptive native prefill rate | Stock Triton | PR122 sparse 0.30 |
|---|---:|---:|
| Tokens divided by median duration (8 stock / 4 candidate durations) | 426.1108 tok/s | 709.2824 tok/s |

| Bracket | Paired gain | Identical-stock repeat spread |
|---|---:|---:|
|0|66.83374%|0.06658%|
|1|65.99329%|0.97356%|
|2|67.21973%|0.49490%|
|3|66.31158%|0.35621%|

Median paired gain is **66.57266%**, all four brackets are wins, and median identical-control spread is **0.42556%**. The same finite screen gate (>2%, four wins, gain above median spread) passed. Each fresh server completed a full-depth unscored warmup and one distinct measured prompt; all three legs within a bracket used the same measured prompt. Native/API timing units, token/cache accounting, actual candidate dispatch/page dropping, selected library mappings, worker exit and recorded cleanup were replayed from the retained receipts. Native prefill seconds remain authoritative rather than HTTP wall time, TTFT or pure GPU kernel time. Request settings are source/worker-argv supported; raw HTTP request bodies were not retained.

The final independent reader returned complete 12/12 legs and 4/4 brackets, matching the producer's complete fields, populations and rates. Its explicit post-exit input pass verified 451 selected source/metadata/native/CK files, 15 weight-file stat identities and recorded mapped device/inode correspondence; it did not content-hash the large weights. The original supervisor exited normally with status 0 and exactly one parent completion marker. A separate parent completion-boundary snapshot found the original process and cgroup absent, the port closed, no GPU/NPU compute owners and GPU idle. These checks are retained observations, not continuous exclusion or an adversarial sandbox. Hardware and unchanged policy match the preceding same-machine screen; no package-wattage measurement is claimed.

Repeated stock selected API text matched in all four brackets. Candidate selected text differed in three and matched in one. Here selected text means user-facing content first, falling back to reasoning only when content is empty; it is not a claim about full stored decoded completions, raw IDs or distributions. The continuations were not graded for answer quality. No broad-quality, negligible-loss, lossless, default-enablement or adoption claim follows from this timing screen.

The source-safe recipe defaults to an inert plan and requires explicit prompt preparation or model execution. It contains no retained private prompt or completion bytes; supplying the exact same local weights/overlay is the caller's responsibility. Other packs, prompts, clocks or libraries need their own measurement. The helper's seven stdlib source/planning/runtime-seam tests also passed normally and with explicit `-O`, using inert model/HTTP/tokenizer substitutions where stated; those tests do not provide model-quality evidence.
