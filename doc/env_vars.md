# Environment variables

Runtime and build-time toggles recognized by ExLlamaV3. All of these have sensible defaults;
they exist mainly for A/B testing, debugging and working around platform quirks.

Boolean-ish variables treat `0` as off and any other value as on unless noted. C++-side
variables are read once (on first use) and cached; Python-side variables are read at import
time. Either way, set them before loading a model.

## Persistent prompt cache (GLM, Qwen, DS4.1 servers)

Prompt-cache pages are written to the NVMe in the background, so a restarted server (or a session resumed the next day)
only prefills the new tokens. Off unless a directory is set.

### `EXL3_PREFIX_DISK_DIR` (default: empty = off)
Root directory of the cache. One namespace per model and build; a page is reused only when its token hash and the
model fingerprint match.

### `EXL3_PREFIX_DISK_GB` (default: `30`)
LRU byte cap for the whole root, all models together. The checkpoints of the first 8K tokens of each prompt are kept.

### `EXL3_PREFIX_DISK_FLOOR_GB` (default: `60`)
Free space that must remain on the volume after a write; writes are skipped below it.

### `EXL3_PREFIX_DISK_MB_MIN` (default: `8000`)
Write budget per minute, so long agent sessions do not hit the sustained-write slowdown of QLC drives.

## Prefill scheduling (GLM and Qwen servers, `--sessions` 2 or more)

A prompt is prefilled in chunks of `EXL3_PREFILL_CHUNK` rows, one chunk per scheduler step. By default every active request
runs its chunk in every step, so a 45K-token prompt holds back every short request that arrives next to it, and four 8K prompts
sent together all finish together. The interleave only changes when a chunk runs, never the chunks: each request's chunk
boundaries, and so its prefill math, are the ones of the default path. A single running request is not touched.

### `EXL3_SCHED_CHUNK_IL` (default: `0`)
`1` turns the interleave on. A prompt with more than `EXL3_CIL_SHORT_TOK` rows left to prefill (counted after the prompt-cache and
tail-checkpoint restore) is a big prefill; the oldest one is the leader and the other big prefills wait for it, in arrival order.
Between two leader chunks the short requests get a slice: their prefill, then one single-sequence decode step per scheduler
step, for up to `EXL3_CIL_SLICE_S`. A leader chunk never shares its step with other work. The time the leader spends waiting is
capped per leader (below), after which only its chunks run until it is done. A planner error turns the feature off for the
process with one warning. Needs `--sessions` 2 or more; the server prints one line at start-up when it is on.

### `EXL3_CIL_DEFER_FRAC` (default: `0.18`), `EXL3_CIL_MAX_DEFER_S` (default: `8`)
Most the leader may be delayed by others: `min(EXL3_CIL_MAX_DEFER_S, EXL3_CIL_DEFER_FRAC x its estimated prefill time)`. A
45K-row prefill of 36 s is delayed by at most 6.5 s, an 8K one of 6.5 s by at most 1.2 s. `EXL3_CIL_DEFER_FRAC=0` uses the
seconds cap alone. The step that crosses the limit is the only overshoot: a slice step is only taken when its estimated cost
(rows x measured seconds per row of the leader, plus `EXL3_CIL_DECODE_S` per decode step) still fits.

### `EXL3_CIL_SHORT_TOK` (default: `4096`), `EXL3_CIL_SLICE_S` (default: `0.6`)
Rows at or below which a prefill counts as short; length of one slice in seconds.

### `EXL3_CIL_SINGLE_DECODE` (default: `1`)
In a slice only one decode-ready request decodes per step (least recently served first), alone, as it would without the big
prefill. `0` lets all of them decode in one batch: the slice is cheaper for the leader, but a batched decode is not
guaranteed to give the tokens of a solo run.

### `EXL3_CIL_ROW_S` (default: `0.0007`), `EXL3_CIL_DECODE_S` (default: `0.05`)
Starting estimates in seconds of one prefill row and of one decode step; the row estimate is replaced by the measured leader
chunk times after the first chunk.

## Prompt encoding (GLM and Qwen servers)

### `KYO_TOKCACHE` (default: `8`)
Number of recent prompts whose token ids the server keeps. A client that re-sends the whole conversation, or the same long
document with another question, then encodes only the part after the last shared turn marker or line break; the ids equal a
full encode (a self-test at start-up turns the cache off where they would not). Encoding a re-sent prompt of about 90K tokens:
118 ms to 4 ms. `0` turns it off.

### `KYO_TOKCACHE_CHECK` (default: unset)
`1` also runs the full encode for every request and compares the ids. For debugging.

## Attention

### `EXL3_BC_ATTN` (default: `1`)

Graph-captured C++ decode attention. For decode steps (bsz ≤ 8, q_len ≤ 16) the whole attention
block -- q/k/v projections, fused head norm + RoPE, cache append, flash-decoding attention and
o_proj -- runs as a single C++ call, captured as one CUDA graph per (bsz, q_len) shape and
replayed with only the input/output/position/block-table pointers patched. Removes effectively
all Python host time from the attention block; the largest gains are on host-bound setups
(small or hybrid models, fast GPUs, contended CPUs). The same flag covers the equivalent path
for MLA layers (BC_MLAttention: q projections, latent projection and staging, partial RoPE,
W_UK absorption, cache append, absorbed flash-decoding, W_UV unfold and o_proj as one graph).

Module or cache configurations the path does not support (TP, headwise gates, LayerNorm or
span-heads head norms, non-EXL3 projections, compander-enabled quant cache, ...) fall back to
the regular dispatch path by design. Unexpected errors while building the path are raised, not
swallowed. Set to `0` to disable the path entirely.

### `EXL3_BC_ATTN_TRACE` (default: `0`)

Print one line per attention module/cache-layer pair when the graph-captured decode path is
built or declined (module key, device). Activation check for A/B tests: a benchmark comparing
`EXL3_BC_ATTN` settings is only meaningful if the enabled run actually built the path.

### `EXL3_BC_DSA` (default: `1`)

DeepSeek-V4 counterpart of `EXL3_BC_ATTN`: for decode steps (bsz 1, q_len ≤ 16) the whole DSA
attention step runs as one graph-captured C++ call: the batched x-side projection fan, fused
head-norm RoPE, both compressor updates (compressed/indexer entry pools and rings), the
lightning-indexer scoring and capture-safe top-k selection (long-context regime), the
flash-decoding sparse attention with fused output de-rotation, the grouped o_proj and the
sliding-window ring append. One graph per (cache layer, job slot, q_len, dense/top-k regime);
position flows through shared device scalars (one 8-byte host write per step per job) plus a
handful of patched scalar node parameters (live pool width, causal bounds), so replays never
rebuild anything. Steps that need a host-side window ring shift or rebase (page-granular, rare)
decline to the eager path for that step, as do ineligible layer configurations (non-EXL3
projections, ...) permanently. Set to `0` to force the eager path everywhere.

### `EXL3_BC_DSA_DEBUG` (default: `0`)

Raise errors encountered while building the graphed DSA path instead of silently declining to
the eager path. Diagnostic for why a configuration falls back.

### `EXL3_DSV4_NO_XFAN` (default: `0`)

Set to disable the eager DSA path's projection fans. With the fans, the x-side projections that
share the block input (q_a, wkv, and the compressor/indexer kv/gate pairs) run as a single
per-matrix-N batched MGEMM, and q_b pairs with the indexer query projection over q_res the same
way in the top-k regime: six to eight GEMV launches collapse into two. A/B switch for the
eager path only; the graphed path (`EXL3_BC_DSA`) builds its own fan and is not affected.
Layers whose projections mix quantization formats or bitrates decline the fan by themselves.

### `EXL3_QC_STAGING` (default: `1`)

How quantized K/V caches feed the attention kernels (replaces the former `EXL3_QC_ATTN`). Only
affects quantized caches.

- `0` - no staging: packed cache tensors feed the prefill and decode kernels directly, with
  dequantization fused into the kernel loads. Lowest memory: no staging scratch is ever
  allocated, and nothing extra is reserved during autosplit loading. Prefill pays for the
  in-kernel expansion (roughly 5–25% on the attention kernel depending on bitrate and GPU),
  which every kv tile repeats once per query block and sibling query head.
- `1` - prefill staging (default): prefill chunks of 256+ tokens dequantize the referenced
  cache window once into a shared fp16 scratch and run the fp16 kernel over it, putting
  quantized-cache prefill within ~1–3% of fp16. Decode stays on the direct path. The scratch is
  sized for the full cache at batch size 1 (`2 * max_num_tokens * num_kv_heads * head_dim`
  fp16 elements, shared across layers per device) and is allocated by the autosplit measuring
  pass, so the space is reserved at load time rather than discovered at the first long prefill.
  For very large caches this reservation is the tradeoff to weigh against `0` (e.g. ~4 GB at
  1M tokens with 8 kv heads of dim 128).
- `2` - full staging: legacy dequantize-then-attend path; whole cache layers are expanded into
  full-size fp16 temporaries before attention. Debug/A-B mode (same effect as the former
  `EXL3_QC_ATTN=0`); only affects decode if `EXL3_BC_ATTN` is also disabled, since the graphed
  decode path reads the packed cache directly.

### `EXL3_CK_PAGEDKV` (default: unset) -- EXPERIMENTAL

Experimental, default-off, approximate Composable Kernel sparse prefill for the MiMo
global-attention shape on gfx1151 ROCm. Skipping past pages changes attention output and
therefore later layers' K/V. This is research functionality, not a general quality-adoption
recommendation. Set to `sparse` before importing the engine to register handlers for fp16 and
8-bit KV caches when the optional library is available. The handlers own only batch-one,
causal Q/K=192, 64-query-head/4-KV-head, V=128 prefill chunks; unsupported shapes, decode,
SWA and unaligned past lengths fall back to the normal backend.

Build the required MIT-licensed CK specialization with local `git`, `cmake`, `nm`,
`/usr/bin/python3`, and an already-expanded ROCm SDK (including `lib/llvm/bin/clang{,++}`):

```bash
export EXL3_CK_SOURCE=/path/to/local/composable_kernel
export EXL3_ROCM_SDK=/path/to/already-expanded/rocm-sdk
export EXL3_CK_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/kyojin/ck-pagedkv"
tools/strix_halo/build_ck_pagedkv.sh
```

`EXL3_CK_SOURCE` is a read-only local Git donor containing commit
`6f44fdf90f69adee742ec3660b95d32963b9bd2d`. The builder ignores its current HEAD and dirty
worktree, stages that exact commit in the cache, and patches only the staged copy. It never
fetches implicitly or sources `env.sh`. `EXL3_CK_CACHE` must be disjoint from the donor (neither
may contain the other): choose a nonexistent directory on first use, or reuse a cache with a
matching builder/patch/shim/SDK/commit ownership record. Existing unowned or incompatible
caches are refused.

The output is `$EXL3_CK_CACHE/libck_pagedkv.so` with a digest-bound `receipt.json`. The default
cache is `${XDG_CACHE_HOME:-$HOME/.cache}/kyojin/ck-pagedkv`; when choosing a different cache,
set `EXL3_CK_PAGEDKV_SO="$EXL3_CK_CACHE/libck_pagedkv.so"` before importing the engine. Build
success proves source compilation/linking and export checks only, not native loading or
numerical correctness. The first implementation stages all used K and V pages from an 8-bit
cache; selected-V-only staging is not implemented yet.

A missing/unloadable optional library or missing entry point warns once and leaves the normal
handlers in place for the process. An explicit kernel rejection (`-1`) declines for that chunk;
other invalid results and execution errors are not treated as unsupported shapes. Restart after
building or replacing an unavailable library.

The 8-bit staging pair grows in 32-page steps, capped by the packed cache's physical capacity,
and is reused within each step. A width/bound change retires incompatible scratch; replacements
retire the old pair first under the engine's single-stream, fully-overwritten-scratch contract.

Sparse mode, retention and minimum length are frozen together at import time. Use a new process
to change them. Sparse page hashes, disk-prefix namespaces and saved sessions are isolated from
ordinary mode and from other sparse policies, including fallback requests in a sparse process.
Ordinary session formats 1/2 remain readable in ordinary mode. Sparse sessions use incompatible
formats 3/4 so older ordinary readers reject them too; current readers check the exact policy
before restoring any pages, stash or rootless tail. The MiMo server's `+ck-sparse` model tag is
retained as an additional separation layer.

Focused CPU tests need no Torch, Triton, native extension or CUDA/HIP toolchain. Run each directly
from the repository root to avoid package initialization:

```bash
python3 -I -S -B tests/test_ck_pagedkv_cpu.py -v
python3 -I -S -B tests/test_ck_pagedkv_fallback_cpu.py -v
python3 -I -S -B tests/test_ck_pagedkv_admission_cpu.py -v
python3 -I -S -B tests/test_ck_pagedkv_sparse_memory_cpu.py -v
python3 -I -S -B tests/test_ck_sparse_cache_cpu.py -v
python3 -I -S -B tests/test_ck_dispatch_mode_cpu.py -v
```

### `EXL3_CK_SPARSE_RETAIN` (default: `0.5`)

Nominal fraction of total pages (past plus current chunk) retained by the optional CK selector.
Current-chunk and sink pages are always kept, even if they exceed that budget; the most-recent
past pages are not forced. Read once before engine import and included in the sparse cache
identity. This is a quality/performance knob; evaluate it with the same model and context before
changing the default.

### `EXL3_CK_MIN_SK` (default: `0`)

Experimental minimum total KV length for sparse selection in the fp16 and packed8 handlers.
Calls below it use the ordinary backend. Frozen at import time and included in the sparse cache
identity. It does not enable CK for otherwise unsupported shapes.

### `EXL3_QC_PF_TWO_PASS_MIN_Q` (default: `256`)

Query-length threshold for the prefill staging pass at `EXL3_QC_STAGING=1`. Chunks shorter than
this keep the direct path, which reads less global memory (relevant for short trailing chunks
over long contexts at low cache bitrates). Tuning/testing knob.

### `EXL3_QC_PREFILL_NS` (default: `0` = measure)

Pipeline stage count for the direct quantized-cache prefill kernel. Unset/`0`, the best of
{1, 2} is measured once per (shape family, device) at first use; a nonzero value pins it,
skipping the measurement. Only relevant where the direct path still runs (`EXL3_QC_STAGING=0`,
or short chunks below the threshold above).

### `EXL3_KVWIN` (default: unset = `4:128` when k or v bits <= 5 on ROCm, off on CUDA and at 6 bits and above)

`sink:window`, e.g. `4:128`; `0` / `off` disables it, an explicit value forces it at any bit width and on any backend (the unset-switch default is ROCm-only until the CUDA served A/B clears it: `DEFAULT_ON_CUDA` in `cache/kvwin.py`). Keeps the first `sink` positions and the last `window` positions of every job in fp16
next to a quantized K/V cache (`--k-bits/--v-bits`): attention reads the quantized pool for everything in between
and a small fp16 ring (`sink + window + 16` rows per layer and job) for the rest, merged exactly by the flash-decoding
combine. Costs about 10 MiB of VRAM per job slot at `4:128` on a 16-layer, 4-kv-head, head-dim-256 model. Applies to
quantized-cache attention layers of models with recurrent state slots (Qwen3.5/3.8 hybrids); the ring is part of the
job's recurrent state, so prefix/tail checkpoints, the CPU stash and the prefix disk carry it and restores stay
bit-exact (a checkpoint written without the switch rebuilds the ring from the pool, with quantized precision for the
first `window` tokens). While it is on, the graph-captured decode attention (`EXL3_BC_ATTN`) and the whole-forward
graph (`EXL3_CUDA_GRAPH`) decline, and tensor-parallel loading is not supported. Prefill chunks attend the exact
sinks, the exact window before the chunk and all of the chunk's own rows (a superset of the per-row decode window).
With the window off (`0`, or unset at >= 6 bits) every code path is bit-identical to a build without it.

### `EXL3_MLA_PREFILL` (default: `mha`)

Prefill strategy for MLA layers: `mha` up-projects past latent tiles from the compressed cache
and attends in MHA form (~2.8× fewer FLOPs per query-past pair); `absorbed` restores the
single-kernel absorbed-form prefill for A/B testing.

### `EXL3_MLA_DEC_VALU` (default: `1` on HIP)

Decode and verify rows (up to 8) of the MLA absorb and unfold run on the VALU dot2-chain kernels. The output is
bitwise equal to the Triton path. `0` turns it off. If the extension build lacks the kernels, the Triton path is used and
one log line says so.

### `EXL3_PREFER_FA2` (default: `0`)

Put the flash-attn-2 backends ahead of the built-in Triton attention kernels in the dispatch
order. flash-attn is an optional dependency; when it is not installed, this switch is ignored
(with a warning) and the built-in kernels serve everything. The Triton kernels match or beat
FA2 across supported hardware and cover more cases (quantized caches, head dims > 256,
attention sinks); this switch exists for A/B comparison.

## EXL3 GEMM / GEMV

### `EXL3_GEMV` (default: `1`)

QTIP-style small-m fp16 GEMV path, dispatched from the main GEMM entry point when the shape
heuristic applies. `0` disables, `1` uses the measured heuristic envelope (default), `2` forces
the path wherever its hard constraints allow (testing).

### `EXL3_GEMV_SMEM` (default: `-1`)

Weight-extraction strategy inside the fp16 GEMV kernel: `-1` picks per bitrate (default), `0`
forces shuffle extraction, `1` forces shared-memory staging. Testing only.

### `EXL3_INT8_GEMV` (default: `2`)

Fused int8-activation GEMV for tensors quantized with the mul1 codebook: one cooperative launch
covering the input Hadamard, activation quantization, dp4a GEMV and output Hadamard. `2`
(default) is the plain int8 mode, `1` the error-feedback residual mode (~15–16 bit effective
activation precision, slightly slower), `0` disables the path.

Tensors quantized with other codebooks are unaffected and keep their regular kernels. When the
mode is enabled, gate/up (and other same-input) tensor pairs that the int8 path can take are
also *unfused* from the batched MGEMM when each matrix is wide enough to fill the GPU on its
own. See the two thresholds below. The graphed decode paths (BC modules) handle both the fused
and unfused configurations.

### `EXL3_INT8_GEMV_MAX_K` (default: per-arch)

Highest bitrate K the int8 GEMV path accepts; above it the regular fp16 kernel runs instead.
The default is 6 on Hopper and Blackwell and 5 elsewhere: Ampere is DRAM-bound from K = 6 up,
where the int8 path's reduced per-weight compute no longer helps (and Ada is marginal there),
but on Hopper the fp16 kernel is throughput-bound at K = 6 as well. Values up to 8 can be forced
to test the crossover on unmeasured parts; the MGEMM unfusing threshold below follows this cap
automatically.

### `EXL3_MGEMM_K_THRESHOLD` (default: per-arch), `EXL3_MGEMM_N_THRESHOLD` (default: `8192`)

Unfusing heuristics applied when the int8 GEMV mode is enabled, to mul1 tensor pairs only: keep
the fused MGEMM when the bitrate K is at or above the K threshold (the int8 path declines those
anyway), or when the matrices are narrower than the N threshold (too narrow for separate GEMV
calls to fill the GPU; batching is what restores utilization there). The K threshold defaults
to one above the int8 path's per-arch K cap (see `EXL3_INT8_GEMV_MAX_K`); setting it explicitly
pins it on every device.

### `EXL3_NO_FUSED_RECONSTRUCT` (default: `0`)

Set to disable original-basis weight reconstruction on the hgemm prefill path. Long inputs to
EXL3 linears run reconstruct-then-GEMM; by default the reconstruct kernel emits the weights in
the original basis. Both 128-point Hadamard transforms and the sign vectors are folded into
the (memory-bound) reconstruct kernel's shared-memory epilogue, so the GEMM runs on the raw
input and the standalone input/output Hadamard launches disappear (previously ~14% of
long-chunk prefill GPU time; ~+6-7% prefill throughput on DeepSeek-V4-Flash). The fused kernel
does k·n-proportional extra work while the saved Hadamard traffic scales with rows·(k+n), so it
engages at 1024+ input rows (breakeven is ~400–900 depending on shape); below that, and for the
per-expert MoE dequant path (small row counts per expert), the rotated-basis pipeline
(input Hadamard → GEMM → output Hadamard) is kept. Set to `1` to force the rotated-basis
pipeline everywhere, for A/B testing.

### `EXLLAMAV3_TUNE_CACHE` (default: platform cache dir)

Override the path of the on-disk autotune cache for the cooperative GEMM kernels (kernel shape
selection results, persisted across runs).

## Sampling

### `EXL3_FUSED_SAMPLER` (default: `1`)

Collapse eligible sampler stacks into fused kernels at sampler construction. Stacks ending in
greedy or temperature/min-P/top-K/top-P/Gumbel steps (in the orders emitted by the preset
samplers, optionally preceded by repetition/presence/frequency penalties) run as a few custom
kernels working directly in logit space, instead of the step-by-step softmax/sort pipeline.
Collapsed temperature/min-P stacks sample the same token as the uncollapsed reference for the
same seed, up to float rounding at exact ties; top-K/top-P stacks keep the same token set as
the sort-based reference (ties at the exact cutoff are all kept) but draw their Gumbel noise by
token id rather than sorted position, so individual seeds map to different samples from the
same distribution. Stacks the collapse does not recognize fall back to the step-by-step path by
design. Set to `0` to disable collapsing entirely, e.g. for A/B validation against the
reference implementation.

## CPU MoE offload

Experimental: `-mcl`/`--moe_cpu_offload` (main model) and `-dmcl`/`--draft_moe_cpu_layers`
(draft model or MTP head) run the routed experts of the first N block-sparse MoE layers on the
CPU, expert weights resident in system RAM, freeing the VRAM those layers' experts would have
used. Layer-split mode only; requires mul1-codebook experts, K ≤ 8, and uniform per-expert
biases (all or none; ineligible layers fall back to the GPU as usual). A spawned worker process
per model component (main / draft / MTP) owns its own expert weights and a job ring in pinned
shared memory; the parent's forward pass never blocks on the CPU. During prefill, hot experts
additionally stream their weights to the GPU and run there (via the fused kernel or per-expert
dequant, by size) while the CPU works the remaining tail. See `-mclt`/`-dmclt` below for 
thread configuration, and the knobs below for tuning the split.

These knobs are collected in `exllamav3/model/moe_cpu_host.py`'s `MoeCpuTuning` class (read once
from the environment at import); for a same-process sweep, mutate fields on the module-level
`TUNING` singleton before constructing a model instead of setting env vars.

### `EXL3_MOE_CPU_OFFLOAD` (default: `0`)

Fallback value for when `-mcl` is not set.

### `-mclt` / `--moe_cpu_threads`, `-dmclt` / `--draft_moe_cpu_threads` (CLI, not env)

Worker thread count, set per component via `config.infer_params.moe_cpu_threads` /
`draft_moe_cpu_threads`. Takes precedence over `EXL3_MOE_CPU_THREADS` below when set.

### `EXL3_MOE_CPU_THREADS` (default: `cpu_count // 2`)

Fallback worker thread count when the component's `-mclt`/`-dmclt` config value is not set.

### `EXL3_MOE_CPU_SLOTS` (default: `4`), `EXL3_MOE_CPU_SLOT_ROWS` (default: `64`)

Compute job-ring depth and rows per slot (the CPU-tail chunk size). Each slot holds one
in-flight chunk of the D2H-staged input, selected experts and routing weights, and the
H2D-staged fp32 output.

### `EXL3_MOE_CPU_WSLOTS` (default: `2`), `EXL3_MOE_CPU_WSLOT_MB` (default: `32`)

Depth and per-slot size of the pinned/VRAM weight-staging ring used by GPU-streamed prefill.
Each slot must be large enough to hold a batch of streamed experts' packed weights (see
`EXL3_MOE_STREAM_BATCH_EXPERTS`); if not, the batch is capped by capacity instead.

### `EXL3_MOE_CPU_STAGE_THREADS` (default: `4`)

Memcpy threads used by the worker's dedicated stager (which packs streamed experts' weights
into the pinned staging ring, concurrently with the compute pool working the CPU tail). A few
threads saturate host memcpy bandwidth; raising this mainly helps wide streamed batches on
models with many small experts (see issue trace on Qwen3.6-35B-A3B).

### `EXL3_MOE_STREAM_T` (default: per-device, bandwidth-scaled from `16`)

Minimum per-expert token-assignment count (in a prefill chunk) for an expert's weights to be
streamed to the GPU instead of computed on the CPU tail. Unset, the effective threshold scales
inversely with the measured pinned→device bandwidth (probed once per device): a chipset-attached
x4 link needs a much hotter expert to justify the weight DMA than a CPU-direct x16 one. Setting
this explicitly pins the threshold on every device and disables the bandwidth scaling.

### `EXL3_MOE_STREAM_FUSED_T` (default: `256`)

Maximum per-expert assignment count eligible for the fused `exl3_moe` GPU kernel (one launch
covers a whole batch of experts, up to three with the row tiles of `EXL3_MOE_MTILE`); above
this an expert still streams but runs through the batched reconstruct tier
(`EXL3_MOE_STREAM_BATCH_RECON`) or the per-expert reconstruct path instead. Same eligibility as
the GPU-resident fused path otherwise (mul1, silu/gelu gated or relu2 gateless, no per-expert
biases, no padded dims); ineligible layers use the reconstruct path for every streamed expert
regardless of count. The default was 512 while the alternative above it was the per-expert
loop; with the batched tier there, 128-256 measure best (Qwen3.8 4090 + 3090 split, 4k
chunks: 512 -> 256 +4%; mistral-small-4 119B full offload on the PRO 6000: +2.8%), and the
fused temp buffers (concurrency x T x (2 hidden + 2 intermediate) x 2 bytes per device) halve.

### `EXL3_MOE_STREAM_MIN_ROWS` (default: `32`)

Prefill chunk size floor below which GPU streaming never engages and every expert runs on the
CPU tail as usual (decode, at 1 row per pass, always stays under this).

### `EXL3_MOE_STREAM_BATCH_EXPERTS` (default: `24`, max `256`)

Experts packed per weight-staging batch (one stage job, one DMA, and, below
`EXL3_MOE_STREAM_FUSED_T, one fused-kernel launch). Further capped by staging-slot capacity
(`EXL3_MOE_CPU_WSLOT_MB` divided by one expert's packed byte size). The hard ceiling of 256 is
the structural size of the job descriptor's expert-id array; raising the ceiling itself costs
only a small amount of shared-memory overprovisioning, not runtime.

### `EXL3_MOE_CPU_MAX_ISA` (default: unset, auto-detect)

Caps the CPU kernel's runtime ISA detection at `scalar`, `avx2`, `bw`/`avx512bw`,
`vnni`/`avx512`, or `vbmi`, for testing a lower-tier kernel path on hardware that supports
better. The `bw` tier covers AVX-512F/BW/VL hardware without VNNI (Skylake-SP/X: 1st-gen Xeon
Scalable, Core-X), which previously fell through to `avx2`: the `vnni` dword kernel with the
AVX2 tier's vpmaddubsw/vpmaddwd accumulate, ~1.5x the `avx2` tier's cold-expert decode
throughput on a Xeon Gold 6148. The `vbmi` tier
(AVX512-VBMI byte-gather state extraction, Zen 4+ / Ice Lake+; Cascade/Cooper Lake have VNNI
without VBMI and stay on the `vnni` tier) is 15-70% faster than the dword scheme depending on
bitrate. Never upgrades past what the CPU actually supports; unrecognized values are ignored.
Read once per process (parent and worker independently), so it must be set before either is
started. Note that capping below `bw` also disables the swizzled weight layout (see
`EXL3_MOE_CPU_SWIZZLE`).

### `EXL3_MOE_CPU_SWIZZLE` (default: `1`)

Repack the CPU worker's expert trellis copies into a band-contiguous ("swizzled") layout at
load, so each GEMV band streams sequentially from DRAM instead of in short strided runs
(+45-75% cold decode GEMV throughput measured on a 7960X, reaching the sequential-read
roofline). Takes effect on every AVX-512 kernel tier: `vbmi`, whose byte-gather extraction
leaves the register headroom for the wide bands the swizzled layout wants at m > 1, `bw`
(+2-29% on Skylake-SP, where the sequential per-band k-stream beats 96-128 B strided reads)
and `vnni` (the dword kernel with the same band structure; +40% cold-expert decode measured
with the tier forced on a 7960X). The `avx2` and `scalar` tiers read the native layout. K8
tensors always stay in the native layout (they route to the dword kernel). The GPU-streaming
prefill path un-swizzles during staging, so staged bytes reaching the GPU dequant are
unaffected. Set to `0` to keep the native layout.

### `EXL3_MOE_MEMOPS` (default: `1`)

The parent enqueues its wait/publish handshake with the worker as CUDA stream memory operations
(`cuStreamWaitValue32`/`WriteValue32`, front-end executed: no SM occupancy, no per-op launch
cost) rather than the older spin-wait kernels. Set to `0` to force the kernel fallback, kept
around specifically because the memop path is not yet exercised on Windows. The kernel path's
30-second stall timeout does not apply to the memop path; a dead worker there is instead detected
by a host-side watchdog that unblocks any pending wait.

### `EXL3_MOE_STREAM_DEBUG` (default: `0`)

Print per-layer and per-batch engagement: streamed bandwidth probe result and threshold, expert
counts, streamed-vs-tail assignment split, and fused-vs-reconstruct tier split within each
streamed batch.

### `EXL3_MOE_CPU_PROF` (default: `0`)

Accumulate per-phase wall time in the CPU compute pool and report every 512 jobs. Enabled once
per worker at startup.

### `EXL3_MOE_ARENA_DEBUG` (default: `0`)

Print each hugepage-arena chunk allocation (size, running total) as the CPU worker loads expert
weights, and confirmation when the end-of-load `MADV_COLLAPSE` pass (see
`EXL3_MOE_ARENA_HUGEPAGE`) is issued. The worker copies loaded expert tensors into a small
number of large (1 GiB) anonymous mappings instead of leaving them as many separate small
(sub-2MB) allocations, confirmed via `/proc/<pid>/smaps` that the latter cannot be backed by
transparent huge pages even under system-wide THP=always, since each is its own VMA.

### `EXL3_MOE_ARENA_HUGEPAGE` (default: `1`)

Whether to attempt hugepage promotion for the arena chunks described above. This is done as a
single `MADV_COLLAPSE` (Linux 6.1+) pass over each chunk *after* all expert weights for every
offloaded layer have been loaded, deliberately not via a live `MADV_HUGEPAGE` hint during the
per-layer writes: on hosts where `/sys/kernel/mm/transparent_hugepage/defrag` is `madvise`, that
hint makes the kernel do *synchronous* compaction on first touch of a hinted region once
easily-compactable free memory runs low, which turns into multi-second stalls per offloaded
layer partway through a large model's load. The collapse pass runs on a background thread in
the worker after it has started serving: it copies the whole arena (about 4 GiB/s on a
7960X when the chunks were faulted as 4K pages, i.e. on `transparent_hugepage/enabled =
madvise` hosts, plus any compaction the kernel needs first), so it must not sit on the
startup path; the worker reads 4K pages until each chunk lands. `EXL3_MOE_ARENA_DEBUG=1`
prints how long it took. Set to `0` to skip hugepage promotion entirely.

### `EXL3_HGEMM_F16ACC` (default: auto)

GeForce parts run the fp32-accumulator tensor-core MMA at half the rate of the fp16-accumulator
form (measured 2.00x on the 3090, 4090 and 5090; 1.00x on the RTX PRO 6000). The reconstruct
(prefill) GEMMs, i.e. the dense `Linear` path above the reconstruct threshold, the per-expert
and batched MoE reconstruct tiers, run through a kernel (`hgemm_f16acc.cu`) that uses the
fp16-accumulator MMA and flushes the partial sums into fp32 every 32 elements of K, so the
accumulation across K stays fp32. `auto` runs a one-time per-device rate probe and enables the
kernel where the fp16-accumulator MMA is at least 1.5x faster; `1` forces it on, `0` forces
cuBLAS. Shapes the kernel does not cover (K not a multiple of 64, N not a multiple of 128,
unsupported strides/alignment) use cuBLAS regardless. Compute capability 12.x uses swizzled
128x128 or 128x64 tiles selected by shape, and a native mixed-precision add when folding
each 32-term FP16 partial into FP32.

### `EXL3_MOE_COOP_KSPLIT` (default: unset)

Split-k factor of the fused decode MoE kernels: `n` runs every column chunk as `n` blocks over
disjoint k ranges whose partial sums the last-arriving block adds. Measured as neutral to harmful
on every GPU here, so the default is no split. Testing knob only.

### `EXL3_MOE_COOP_KSPLIT_A` (default: unset, no split; Ampere only)

Gate/up stage of the fused decode MoE kernels at batch sizes whose (token, expert, projection,
128-column) units fit one per SM (Flint at batch 1: 48 units on 82 SMs): every unit's k loop
is one block's serial chain. `n` >= 2 (`3` is the measured choice) shares it over up to `n`
blocks (while the grid stays resident); the last-arriving block of a chunk adds the fp32 partial
rows in fixed order. The pre-activation is then no longer rounded to fp16 before the output
rotation, so results differ from the unsplit unit in that rounding (teacher-forced NLL on Flint
is lower at 4K, 32K and 64K; greedy text flips at near-ties). Unset, `0` or `1` keep the unsplit
unit, which is bit-identical to the previous kernel. Other architectures and larger batches are
not affected.

### `EXL3_MOE_COOP_WIDE` (default: unset)

Tile geometry of the fused decode MoE kernels (the bsz <= 8 path of `BlockSparseMLP`): `0` forces
the narrow tile (32 columns per block, k split 16 ways), `1` the wide one (128 columns per block,
4 x 4 warps). Unset picks per stage: wide on Ampere/Ada at every shape, on Blackwell only for
k >= 4096 or k >= 2048 with 32 or more (token, expert) slots. Testing knob only.

### `EXL3_MOE_FUSED_DET` (default: `1`), `EXL3_MOE_RECON_DET` (default: follows `EXL3_MOE_FUSED_DET`)

Bit-reproducible MoE prefill. By default the fused MoE kernel adds each expert's weighted
output into the token row with float atomics, in whatever order the expert groups finish, and
the batched reconstruct tier accumulates its padded slab with one atomic `index_add_`;
together these are the only sources of run-to-run nondeterminism on the GPU prefill path
(Qwen3.8-Flash-Next KL ~2e-2 between identical 4k-token runs, lfm2.5 ~1e-3, Qwen3-30B-A3B
~3e-4). With `EXL3_MOE_FUSED_DET=1` every assignment of the fused and batched tiers gets a
slot in one per-call fp32 scratch (`assignments x hidden`, ~320 MB per layer call on
Qwen3.8 at 4k tokens): the fused kernel stores its weighted outputs there, the batched
tier's down-projection GEMMs write straight into their slots, and one `exl3_moe_gather` per
layer sums each token's slots in k order with the routing weights. Identical runs are then
bit-identical (verified on all three models), and since the GEMM outputs are written once
and read once either way it costs nothing measurable: Qwen3.8 6198/6219 vs 6164/6178 tok/s,
lfm2.5 25.9k vs 26.2k, Qwen3-30B-A3B 7304 vs ~7100 (atomic). The remaining atomic user is
the streamed CPU-offload tier's fused-kernel call. `EXL3_MOE_RECON_DET` only matters where
the batched tier cannot write into the slot scratch (the streamed tier, or the switch off):
`1` accumulates one expert at a time, `0` with one atomic `index_add_`. The GDN/KDA
recurrent decode kernels (Qwen3.5, Qwen3.8, GLM-5.3) reduce their per-slice partial dot
products in a fixed order unconditionally (no switch, no cost), so greedy decode on those
models is reproducible as well.

### `EXL3_MOE_PINNED_ARENA` (default: `0`, experimental)

Linux only. Back the CPU worker's expert-weight arena with `memfd` chunks that the parent
process also maps and page-locks (`cudaHostRegister`), and lay each expert's gate/up/down
trellis tensors out as one contiguous block. Streamed prefill (`EXL3_MOE_STREAM_T`) then DMAs
an expert's block straight out of the arena on the copy stream instead of having the worker's
stager thread memcpy it into the pinned handoff ring first; the stager is the prefill
bottleneck on fully offloaded models (mistral-small-4 119B, 54 GiB of experts: 4k-token
prefill 700 -> 1850 tok/s on a gen5 x16 link, decode unchanged within noise). Costs: the
descriptors are passed over the worker pipe and every chunk is registered with CUDA as it
appears (~0.2 s per GiB, overlapping the load), the arena pages are shared memory
(`Shmem` in `/proc/meminfo`, counted in both processes' RSS), and shmem pages only get
transparent huge pages where `/sys/kernel/mm/transparent_hugepage/shmem_enabled` allows it
(`advise`, `within_size` or `always`; on the default `never` the CPU kernels run on 4K pages,
which cost a few percent of decode on some hosts). Not available on Windows.

### `EXL3_HOST_MEM_RESERVE_MB` (default: `2048`)

Host-memory guard for the large CPU allocations (CPU MoE expert arena chunks, the n-gram table
held in RAM with `--ngram_ram`): before each one, `MemAvailable` (from `/proc/meminfo`, or
psutil where that is unavailable) must cover the allocation plus this reserve, or the load
fails with a message naming the allocation. Linux has no allocation-time failure for anonymous
or shmem memory: an oversized arena only fails once the machine has swapped itself into a
minutes-long stall and the OOM killer picks a victim, and pinned pages cannot be reclaimed at
all. `0` disables the check.

### `EXL3_MOE_ARENA_HUGE` (default: unset)

With `EXL3_MOE_PINNED_ARENA=1`: `2m` or `1g` backs the memfd chunks with hugetlbfs pages
(`MFD_HUGETLB`) of that size instead of shmem. Requires reserved huge pages
(`vm.nr_hugepages`, or `hugepages-1048576kB` for `1g`) covering the whole arena; allocation
fails with a clear error otherwise.

### `EXL3_MOE_MTILE` (default: `1`)

Row tiles for the fused MoE prefill kernel. The kernel dequantizes each expert's weights once
per 16-row tile, so an expert holding 100 rows re-runs the whole B pipeline seven times. With
this on, experts with 17-32 rows go through a 32-row-tile instance and experts with more than 32
rows through a 64-row one, each as its own launch over its expert range (up to three launches
per layer, largest tile first; a wide instance finishes an expert's remainder with the largest
smaller tile). The wider tiles are separate kernel instances rather than a runtime switch inside
one kernel: sharing one 128-register budget between the tiers costs the 16-row path 60-70% on
Ada / Ampere. Instantiated for the mul1 codebook as N = 128 tile-shape kernels: 30-47% less
fused-kernel time at 24-128 rows per expert on every GPU generation for Qwen3.8-class shapes,
and ~20% for dims that are multiples of 256 (gemma4, Qwen3-30B), where they replace the N = 256
16-row tiling for experts above 16 rows (the N = 256 instance stays for the small-expert
launch, where it is 10-20% faster). Model level, 4k chunks: Qwen3.8 +10% on the PRO 6000, +3%
on a 4090 + 3090 split; gemma4-26B +1-4%; Qwen3-30B neutral. Other codebooks and the
all-fused fast path (no host-side counts) keep the single launch. Outputs are bit-identical to
the 16-row tiling at equal group geometry (per-launch active counts widen the groups, which
reorders the fp32 k-slice reduction to rounding level). Set to `0` for the single launch.

### `EXL3_MOE_TILE_N` (default: `0` = automatic)

`128` keeps the N = 128 tile shape for the fused kernel's 16-row launches on dims that are
multiples of 256 (which otherwise take the N = 256 instances). Measurement knob; the N = 256
instance is faster for those launches.

### `EXL3_MOE_FUSED_ROWS_WIDE` (default: `256`)

Fused-tier row capacity per expert for layers that use the wide tiles (see `EXL3_MOE_MTILE`);
`EXL3_MOE_FUSED_ROWS` still applies to every other layer. With the wide tiles the fused kernel
beats the batched reconstruct tier up to 256 rows (Qwen3.8 4k chunk on the PRO 6000: 6.87k ->
7.06k tok/s over 128 rows), at 4 x concurrency x rows x (hidden + intermediate) x 2 bytes of
static buffers per device (Qwen3.8 on a 188-SM card: +38 MB over 128 rows).

### `EXL3_MOE_BATCH_RECON` (default: `1`), `EXL3_MOE_STREAM_BATCH_RECON` (default: `1`)

Batched reconstruct tier for prefill (`exllamav3/modules/moe_batch_recon.py`): experts with
more assigned rows than the fused MoE kernel's capacity (up to the `EXL3_MOE_RECON_TILES`
budget below) are dequantized and multiplied in count-sorted groups (one pointer-table reconstruct, one
strided-batched GEMM and one Hadamard launch per projection for the whole group, rows padded
to the largest expert of the group) instead of one expert at a time, cutting the per-expert
launch count that dominates many-expert models at large chunk sizes (Qwen3.8-Flash-Next 512
experts, 8k tokens: +16% prefill). The arithmetic is the same as the per-expert path (fp16
rotated-basis weights, fp16 activations, fp32 down-projection output; models with fp32
intermediates such as Gemma 4 keep fp32 gate/up outputs and the activation kernel writes the
fp16 input of the down projection, as in the per-expert path). `EXL3_MOE_BATCH_RECON`
covers GPU-resident experts, `EXL3_MOE_STREAM_BATCH_RECON` the streamed CPU experts (the
`EXL3_MOE_STREAM_FUSED_T` overflow tier). Set to `0` to restore the per-expert loops.

### `EXL3_MOE_RECON_TILES` (default: `64`), `EXL3_MOE_RECON_BATCH` (default: `16`), `EXL3_MOE_RECON_PAD` (default: `1.1`), `EXL3_MOE_RECON_ROWS` (default: `16384`), `EXL3_MOE_RECON_MB` (default: `256`), `EXL3_MOE_RECON_MAX_ROWS` (default: unset), `EXL3_MOE_RECON_FOLDED` (default: `1`)

Tuning for the batched reconstruct tier. Batching pays while a single expert's GEMM cannot
fill the GPU: an `m x n` GEMM launches about `m/128 * n/128` output tiles, and below roughly
the SM count the strided-batched kernel is much faster (PRO 6000, `k = 2048`: `n = 768` runs
1.45x faster batched at `m = 1024` and 2.9x at `m = 256`, `n = 1792` breaks even at
`m = 1024`), while above it the batched cuBLAS kernels are 10-15% slower than the single-GEMM
ones, and the padded slabs cost traffic that grows with the rows. `EXL3_MOE_RECON_TILES` is
that tile budget: an expert stays on the per-expert path once its narrowest projection
(`min(intermediate, hidden)`) would span more tiles, i.e. above `TILES * 16384 / n_min` rows
(1365 rows for `n = 768`, 512 for `n = 2048`). Measured at 8k tokens on a PRO 6000: lfm2.5
(`n` 1792/2048) loses 15% with everything batched and 4% at a 1024-row cap, parity at 512;
Qwen3.8-Flash-Next (`n = 768`) is within 2% for any cap. `0` batches
every expert; `EXL3_MOE_RECON_MAX_ROWS` overrides the derived row cap directly. The other
knobs: experts per group; `EXL3_MOE_RECON_PAD`, the padded-to-real row ratio a group may reach
before the planner starts a new one (groups fill largest expert first, so this splits only
groups of uneven experts: on Qwen3.8-Flash-Next at 4096 tokens a limit of 1.5 pads 32% of the
tier's rows, 1.1 pads 8% and is 2% faster overall, beating a batch of 8 at 1.5; lfm2.5 gains
2-5%); the padded-row budget per group; the dequantized-weight scratch budget, which shrinks
the group for large expert shapes. `EXL3_MOE_RECON_FOLDED=1` (default) folds both
Hadamards and the sign vectors into the dequantized weights (the formulation the dense
`Linear` prefill path uses above 1024 rows): fewer launches, about 4% faster prefill on
Qwen3.8-Flash-Next, at the cost of rounding the folded weights to fp16, which roughly doubles
the tier's error against an fp32 reference of the same quantized weights (9.6e-4 vs 4.6e-4).
Against the unquantized model that difference is invisible: lfm2.5 4.10bpw over 10 x 2048
tokens scores KL 0.0902 to the HF weights folded and 0.0907 unfolded (top-1 85.2% vs 84.9%,
perplexity 40.50 vs 40.56). Note that the two modes are far apart from each other on
routing-sensitive models (Qwen3.8 KL 2e-2, lfm2.5 6e-3) while equally far from the
baseline, so mode-to-mode distance is not a fidelity measure. `0` selects the activation-side
Hadamards.

### `EXL3_MOE_CPU_START_TIMEOUT` (default: `60`)

Seconds the parent waits for the CPU worker to signal ready after every offloaded layer has
been handed over. Startup is the shared-memory attach, layer registration and thread spawn,
so the default is only a safety net against a wedged worker; raise it on very slow hosts.

### `EXL3_MOE_CPU_PIN` (default: `1`)

Pin each worker thread (and the worker's own main thread) to a distinct physical CPU core,
SMT siblings last, instead of leaving placement to the OS scheduler. On an SMT host, unpinned
placement is a real source of run-to-run throughput variance, two workers can land on the same
physical core (contending for its execution resources) on one run and not the next; measured on
a 24-core/48-thread SMT2 box, this swung matrix-decode throughput 61–105 GB/s run to run,
pinned flat at ~105 GB/s (88% of the box's measured 24-thread DRAM read bandwidth). Set to `0`
to disable, e.g. on a shared/multi-tenant host where fixed placement may fight the scheduler's
own balancing across other processes. Falls back to no pinning if the CPU topology can't be
read.

### `EXL3_MOE_HANDOFF_PROF` (default: unset)

Enable GPU/CPU handoff profiling, for debug purposes. 

## Model loading

### `EXL3_EXPANDABLE_SEGMENTS` (default: `1`)

Use expandable segments for all Torch allocations. Opt out with a value of 1 or by explicitly
setting `PYTORCH_CUDA_ALLOC_CONF`.

### `EXL3_LOAD_ARENA` (default: `1`)

Slab allocation for small weight tensors during (deferred) module loads: tensors up to 16 MB
are carved out of shared 128 MB per-device blocks (first-fit over the open blocks, so partially
filled tails are packed by later small tensors) instead of getting one CUDA caching-allocator
allocation each. MoE models with many small per-expert tensors otherwise shatter the allocator
into tens of thousands of segments with large reserved-but-unallocated overhead. Unloading a
module frees its blocks; at most one boundary block shared with a neighboring module stays
pinned. Set to `0` to fall back to per-tensor allocations.

### `EXL3_NGRAM_STREAM` (default: unset = stream from disk)

Default for `Config.infer_params.ngram_stream_from_disk`: stream an n-gram embedding table
(PLE models, e.g. Qwen3.8-Flash-Next) from disk with per-forward row gathers (run-coalesced
positioned reads into pinned staging — threaded preads on Linux, overlapped `ReadFile` at high
queue depth on Windows) instead of loading the whole table into system RAM. The quantized table
is tens of GB, and streaming costs little on SSD-class storage (decode is latency-tolerant at
~30 rows/token; prefill gathers are batched). Set to `0` to hold the table in RAM — worthwhile
only when the table lives on high-latency storage (e.g. HDD, where per-row seeks make streaming
unusable). There is no automatic RAM choice: unset always streams from disk, because host memory sampled mid-load
does not predict what the rest of the load (weights, KV cache, towers) will still take. The load log prints the
placement, the reason, the table size and MemAvailable. Also settable per load via `config.infer_params.ngram_stream_from_disk` or
`--ngram_ram` in `model_init`-based scripts.

### `EXL3_NGRAM_IO_THREADS` (default: `128`)

Upper bound on the reader threads of the n-gram table's disk gather (`EXL3_NGRAM_STREAM` on; read
once, when the first gather starts). A cold gather is bound by the queue depth the disk sees, one
blocking read per thread: 32 threads keep a consumer NVMe at about 45k random reads/s, 128 at about
85k. The pool is also capped at 8 threads per CPU thread (at least 4). Lower the value on storage
that saturates early or when the gather competes with CPU-bound work. Minimum 1.

### `EXL3_NGRAM_WARM_TOKENS` (default: `8192`)

How many prompt positions of the later part of a prompt the engine reads ahead from the n-gram
table, on the prefetch worker, behind the chunk being issued. The rows land in the page cache, so
the gather of the next chunks costs a few milliseconds instead of about 0.4 s of cold reads per
4096-token chunk. It changes no result. `0` turns the look-ahead off. It matters only for models
with an n-gram embedding table read from disk.

### `EXL3_AUTOSPLIT_WORSTCASE` (default: `1`)

The layer-split autosplit loader measures each module's transient VRAM with one forward of a
dummy state and keeps that much headroom per device. Some transients don't show in that
forward: attention/MLA decode statics (QSA/DSA families) and, for block-sparse MoE layers,
everything that depends on how the real workload routes tokens: the deterministic slot scratch
(all assignments slotted, `EXL3_MOE_FUSED_DET`), the batched-reconstruct group temporaries,
and the CPU-offload host's GPU side (per-device weight ring, fused-tier buffers, batched tier
statics, padded fp32 outputs), which the measuring forward skips entirely. With the switch on,
these modules allocate and drop an upper bound of those transients inside the measuring window
(`autosplit_extra_measure`), so the split leaves room for them. `0` restores the plain
measured forward.

### `EXL3_AUTOSPLIT_MARGIN_MB` (default: `256`)

The layer-split loader closes a device when its remaining headroom no longer covers the largest
transient measured on it. Besides the `-gs` budget, the check is made against the device
itself (free VRAM plus the allocator's reserved-but-unallocated pool): a split value at or above
the card's size otherwise plans against the several hundred MiB the CUDA context, the loaded
kernels and other processes hold, and the load passes but the first real forward fails. This
margin is added on top of the measured transient in that physical check, covering what a real
forward keeps live around a module (recurrent test states, gathered embeddings, allocator
slack). `0` disables the margin.

### `EXL3_VISION_PINNED` (default: `0`)

Default for `Config.infer_params.vision_pinned`: store the vision component's linear-layer
weights (fp16 or EXL3 trellis) in pinned host memory instead of VRAM, computing straight from
a zero-copy device alias. Trades vision-tower speed for VRAM. Set before loading the vision
component.

## Multi-GPU

### `EXLLAMA_NO_P2P_COPY` (default: unset)

Controls device-to-device tensor moves (the layer split boundary, draft/MTP heads reading the
target model's states, sparse-attention selections shared between layers). On some platforms
the driver reports peer-to-peer access that the PCIe fabric does not deliver, and a direct copy
silently yields garbage. Unset: the first move between each pair of GPUs probes it (a few
random floats there and back, checked on the host) and, if the probe fails, every later move
between that pair bounces through system memory, with a warning printed once. Set to `1`: always
bounce, no probing. Set to `0`: always copy directly, no probing.

### `EXLLAMA_MASTER_ADDR` (default: `127.0.0.1`), `EXLLAMA_MASTER_PORT` (default: auto)

Rendezvous address and port for the tensor-parallel backend. The port defaults to a free port
picked at startup.

### `EXL3_TP_NO_FWD_BARRIER` (default: `1`)

Skip the pass-start barrier in tensor-parallel forward passes. The native collectives are each
ordered by their own stage counters, so the barrier is not required for correctness; skipping it
saves one spin-kernel launch per rank per pass. Set to `0` to restore the barrier (one aligned
sync point per pass at the cost of a small amount of GPU spin time).

### `EXL3_TP_NO_FP16_WIRE` (default: `0`)

The native backend's CPU-assisted all-reduce moves fp16 payloads over an fp16 wire when the CPU
supports F16C (universal on AVX2-era hardware, probed at runtime): exactly-rounded results for
two ranks, fp16-level rounding beyond, at the same PCIe traffic as the bf16 wire. fp32 payloads
always use the bf16 wire (fp16 lacks the range for residual-stream outliers). Set to `1` to
force the bf16 wire for fp16 payloads too, e.g. for A/B comparison.

### `EXL3_TP_TRACE_WIRE` (default: `0`)

Print a line (once per process) when the fp16 all-reduce wire first activates. Activation check
for numerics A/B tests: whether the wire engages depends on the model's residual dtype, so a
comparison is only meaningful if the fp16-wire run actually used it.

### `EXL3_TP_REDUCE_THREADS` (default: number of participating ranks)

Number of threads slicing each large-payload accumulate in the native backend's CPU-reduce
helper (persistent workers, spin-parked between jobs; AVX-512 path only). The default of one
thread per participating rank covers the cases where a single thread's ~31 GB/s wire rate falls
behind: three or more ranks (multiple adds per chunk) and PCIe 5.0 links. Set to `1` to force
the single-threaded accumulate. Decode-size reduces are always single-threaded.

### `EXL3_TP_SPIN_RECV` (default: `0`)

Milliseconds each tensor-parallel child worker hot-polls its command pipe after finishing a
command before falling back to a blocking receive. A blocking receive pays scheduler wake
latency (tens to hundreds of microseconds, worse with deep C-states) at the start of every
forward pass; during decode the next command arrives within a few milliseconds, so a short spin
window (e.g. `4`) catches it with no wake cost, at the price of one busy core per rank for the
window. `0` disables the spin. Mostly useful on hosts where TP profiling shows a large stagger
between the main process and child workers reaching their first kernel launch.

## Debug

### `EXL3_NGRAM_GATHER_PROF` (default: unset)

Windows only: print per-gather statistics from the streamed n-gram table path (unique rows,
coalesced runs, reads completed synchronously vs left pending, span tasks drained by pool
workers vs the calling thread). Activation check for the overlapped-`ReadFile` gather when
validating a streamed-table model on Windows.

### `EXLLAMA_DEBUGLOG_<CATEGORY>` (default: unset)

Enables timestamped debug logging for the given category when the corresponding variable is
present in the environment. Categories are defined at the call sites (see
`exllamav3/util/debug.py`); mostly hooks for development.

## Build (JIT extension)

These only matter when the C++/CUDA extension is compiled at import time rather than installed
prebuilt.

### `CUDAHOSTCXX` (default: unset)

Host compiler passed to nvcc (`-ccbin`), for systems whose default compiler is too new for the
installed CUDA toolkit.

### `TORCH_CUDA_ARCH_LIST` (default: auto)

Standard PyTorch variable; overrides the compute architectures the extension is built for. When
unset, ExLlamaV3 derives the list from the GPUs present in the system.

## `EXL3_DSA_DEBUG_BOUNDS`

When set to `1`, the JIT DSA attention/indexer kernels compile with device-side bounds
asserts on every block-table page read and gathered pool index, and the DSA module range-
checks block-table contents on the host each forward. A violation traps at the faulting
kernel with the kernel name, source line and bad index instead of corrupting memory or
faulting asynchronously downstream. Debug tool for paged-pool issues; significant JIT
overhead (forces Triton debug mode globally), leave unset in production. AOT/BC graph
kernels are unaffected (compiled with asserts off).

## Quantization

### `EXL3_QT_OPTIMIZED`

Forces the quantizer's dense trellis specializations (`quantize_tiles_optimized.cuh`) on (`1`) or
off (`0`) regardless of the per-architecture dispatch (on for every K on sm_120; per K and codebook
on Ada and Ampere where measured faster; the original kernels elsewhere). For experiments only: the
choice also sizes the quantizer's scratch buffers.
## Kyojin servers

The switches below are Kyojin's own: the ROCm toolchain a gfx1151 machine needs, the escape
hatches in the load path, the feature switches the three servers expose, and the measured speed
defaults each server applies for you. Everything above this line comes from ExLlamaV3. Values
follow the same rule as above: `0` and an empty value are off, anything else is on unless the
entry says otherwise. Set them before you `source tools/strix_halo/env.sh` or before
`exllamav3` is imported, because several are read once at import or on first use.

### Toolchain and build

#### `EXL3_ROCM_SDK` (default: unset)

Root of the ROCm SDK devel tree, for example `export EXL3_ROCM_SDK=$(rocm-sdk path --root)`. It
supplies the `hipcc` that builds every JIT HIP kernel and the `libhsa-runtime64.so.1` that
`env.sh` preloads, and gfx1151 does not run without both. The SDK compiler builds the prefill
kernels about 20 % faster than the system ROCm compiler. Unset, the engine looks for the
`rocm-sdk` package beside the running interpreter, then `hipcc` on `PATH`, then `/opt/rocm`, and
the servers print a `WARNING` line when they land on the slower one.

#### `EXL3_ROCM_DEV_INCLUDE` (default: unset)

An include directory for the extension build and, in build mode, for the JIT `hipcc` flags. Use
it when your `hipsparse/` and `thrust/` headers are not in the tree `EXL3_ROCM_SDK` points at;
`build.sh` takes the headers from either one.

#### `EXL3_HIPCC` (default: unset)

Full path of the `hipcc` binary that builds the JIT HIP kernels. It wins over
`$EXL3_ROCM_SDK/bin/hipcc`, and with no `hipcc` anywhere the load stops and names this variable.
The resolved path goes into the kernel cache key, so a kernel built by another compiler is never
reused.

#### `EXL3_HSA_LIB` (default: unset)

Exact path of the `libhsa-runtime64.so.1` that `env.sh` puts in `LD_PRELOAD`, tried before
`$EXL3_ROCM_SDK/lib`, `/opt/rocm/lib` and the system path. Without a preload torch dies with exit
139 at the first GPU allocation, because the copy in the torch wheel segfaults on gfx1151.
`env.sh` warns when none of the four exists.

#### `EXL3_GCC_INSTALL_DIR` (default: from the local `gcc` in `env.sh`, `/usr/lib/gcc/x86_64-linux-gnu/13` in the JIT kernels)

Directory passed to `hipcc` as `--gcc-install-dir`. It is only added when the directory exists, so
a wrong value is ignored rather than fatal. Set it when `hipcc` cannot find your GCC's headers.

#### `EXL3_HIP_DEFINES` (default: unset)

Whitespace-separated preprocessor defines compiled into the extension, for example
`EXL3_HIP_DEFINES="EXL3_HIP_PF_AFTER_STAGE EXL3_MUL1_DECODE_DOT4"`. Each token becomes one `-D`
flag, so the names you pass are build-time kernel variants, not variables read at run time.

#### `EXL3_VENV` (default: `<repo>/.venv`)

The venv `env.sh` puts first on `PATH` and the one its `--check` looks for a Python in. Set it
when the ROCm torch venv is not `<repo>/.venv`; `build.sh` reads it the same way.

#### `EXL3_ROOT` (default: the checkout holding `env.sh`)

The engine root `env.sh` works from. `PYTHONPATH` is set to it, so `exllamav3` and the built
`exllamav3_ext*.so` are importable, and the default venv and SDK path are resolved under it. Set
it to serve from a checkout other than the one you sourced `env.sh` from.

#### `EXL3_BUILD` (default: unset)

Set to `1` to make `env.sh` take its build branch without the `build` argument. That branch
exports `ROCM_HOME`, `ROCM_PATH`, `HIP_PATH`, `HIPCXX`, `PYTORCH_ROCM_ARCH` and `MAX_JOBS`. Leave
it off while serving: the SDK ships its own HSA runtime, which wins over the preload and brings
the segfault back.

### Loading and memory

#### `EXL3_LOAD_DEVICE` (default: unset)

A device such as `cuda:0` that loads the model on one device instead of autosplitting it.
Autosplit sizes each module from free memory rather than available memory, so on unified memory
a growing page cache makes a model that comfortably fits look unloadable. Single device only,
ignored under tensor parallel; both lane launchers set it.

#### `EXL3_KEEP_PAGE_CACHE` (default: `0`)

`1` keeps the shards' page cache after a load. By default the engine drops it and logs how much it
released, because on unified memory the page cache and the weights are the same physical pool and
keeping it leaves the box carrying a large pack twice.

#### `EXL3_KERNEL_SEED` (default: `1`)

`0` does not unpack the pre-compiled Triton kernels shipped in `exllamav3/kernel_seed/` into the
Triton cache. With the seed a start with an empty cache compiles no kernel (GLM saved about 55 s,
spread over start-up and the first requests). A seed is tied to one Triton version and GPU
architecture; another build simply does not use it (nor does a GPU of another architecture).
The seed hits from any install prefix: Triton hashes the absolute path of its bundled device
libraries into the cache key, and the engine replaces that prefix by a placeholder in the key
(HIP only, and only when a seed matches). `build` uses the same keys but installs nothing: it is
the mode in which a seed is built from an empty cache. Rebuild with `tools/kernel_seed/` (sweep,
then build).

#### `EXL3_LOG_JIT` (default: `0`)

`1` prints one `[jit]` line per Triton kernel the server compiles or loads, with its time. Use it to
check that nothing compiles after READY.

#### `EXL3_WARM_TUNE_BUDGET_S` (default: `90`)

Seconds the start-up warm-up may spend screening dense GEMM solutions. After that the remaining
shapes run untuned (bit-exact) and are tuned on first use. `0` never defers. The shipped tune
seed makes a start on the pinned rocBLAS spend about 1 s here.

#### `EXL3_DENSE_GEMM_TUNE_FILE` (default: `$HOME/.cache/exllamav3/dense_gemm_tune.txt`)

The file that holds the tuned dense GEMM solution per shape class. Point it at a seeded file to
skip the autotune; the GLM lane copies `tools/lanes/assets/tune_seed.txt` to the path it sets.
Without it the first GLM launch spends about ten minutes tuning before the port opens.

#### `MIMO_IMAGE_MAX_PIXELS` (default: `2097152`)

Pixel cap applied to an image the MiMo server preprocesses: the tower sees `min(cap, the pack's own
limit)`, and the image is resized (aspect kept, multiples of 32) until it fits. The default is 2 097 152
pixels — 8 192 patches, 2 048 text tokens — because the pack's config would allow 8 388 608 and one tower
pass at that size needs far more attention memory than is worth it. Raise it for a pack with more
headroom, lower it to make vision cheaper; raising it past the pack's limit changes nothing.

#### `EXL3_MIMO_FP32_MLP_LAYERS` (default: `none`)

Which MiMo layers build their MoE and MLP intermediates in fp32 instead of fp16. Syntax: `47`,
`40-47`, `0,47`, `-1`, `all`, or `none` / `off` / `""`. Without it the value comes from
`exl3_fp32_mlp_layers` in `config.json`. A named layer that does not exist is an error; an
overridden layer loses the fused prefill tier.

#### `EXL3_MIMO_FP32_MLP_FUSED` (default: `0`)

`1` keeps the fused prefill tier on the layers `EXL3_MIMO_FP32_MLP_LAYERS` overrides. That kernel
needs fp16 intermediates, so leaving it on brings back the fp16 range the override was for. A/B
timing only.

#### `EXL3_MIMO_MLP_ACT_LIMIT` (default: `0`, off)

Clamp the activation and the up-projection input to plus or minus this value on the overridden
layers, in every tier, including the decode kernel whose fp16 store does not saturate. Use 255 or
less, because the square of the limit must stay under 65504.

### Feature switches

#### `EXL3_ABLIT_RUNTIME` (default: unset, the hook is off)

Path of an edit spec that projects one fixed direction out of the residual stream at run time. It
edits no weights. `off`, `0`, `false`, `no` and `none` force it off. An explicit path wins over the
`uncensor_spec.json` that a model directory may carry, and a path that does not exist stops the
load with an error naming the file. See [refusal_hook.md](refusal_hook.md).

#### `EXL3_ABLIT_TORCH` (default: `0`)

`1` uses the PyTorch projection instead of the Triton kernel. It has no effect when Triton is not
importable, since the Triton path needs it.

#### `EXL3_ROW_INV` (default: `1`)

While on, a verify forward of up to 8 rows takes the per-row exact path in the MLA attention and in
the MoE router, so each row is bit-equal to the same row decoded alone, which is two GLM jobs at
draft depth 2. Only those two modules and only up to 8 rows: `0` drops the limit to 4, lets the
faster batched paths take over, and a near-tied token can then change. Even with it on, more than
one session does not guarantee greedy output equal to a solo run: the decode attention kernel
picks how to split the cache from the batch it runs with, so a near-tie can change.
`EXL3_ROW_INV_GROUPS` closes that gap on Qwen.

#### `EXL3_ROW_INV_GROUPS` (default: `0`, off)

Qwen with `--sessions` above 1, and `EXL3_ROW_INV` on. `1` packs the jobs of a decode step, in order, into
forwards of at most 8 token rows (jobs times `1 + drafted tokens`), and inside a forward with several
sequences each attention layer runs one sequence at a time, with that sequence's own block table and cache
length. Every row then gets the bits of the same row decoded alone, and a step with one job runs as it always
did. Four cold 8K prompts at `--sessions 4`, `--ndt 7`, greedy: answers equal to the serial ones, 16 of 16
(16-bit and 8-bit cache; the same load gave 5 to 7 of 16 without it). It costs about 3 % wall time on that load
and 8 to 10 % of the aggregate decode throughput on short prompts. Not covered: a step where a sequence crosses
the sparse-attention threshold (2051 cached tokens) while the draft width of the group differs from the width the
same request would have drafted alone (any request that grows past that length while it shares a step), and
every step shared with a request that carries images (it keeps the plain one-forward path).

#### `EXL3_ROW_INV_GROUP_ROWS` (default: `8`)

With `EXL3_ROW_INV_GROUPS=1`: the most token rows one forward of a decode step may carry. `1` runs one forward per
job, which is the slowest plan and exact by construction. Values above 8 leave the range where the row-exact
kernels apply.

#### `EXL3_PLAIN_ROWS_VERIFY` (default: `0`, off; the GLM server sets `1` with `--sessions` above 1)

A decode step without drafted tokens that carries 2 to 8 rows (several sessions decoding together) is run like a verify
forward: the union MoE on the device table, the R-row GEMV, the batched router and the per-sequence attention loops
that `EXL3_ROW_INV` enables for verify forwards. Without it such a step fell to per-expert eager loops, thousands of
host launches. Needs `EXL3_ROW_INV` on. One-row steps are untouched, so a single request decodes as before. GLM-5.3-Flash
on the box (v1.7.2), 4 sessions, 4 greedy prompts: 26.1 to 40.7 tok/s aggregate. Qwen already has these paths, so it does
not change there. The text of 4 concurrent plain streams still differs from the solo run at a near-tie word (the solo step
uses the one-row kernels), as it did before the flag.

#### `QWSPEC_LOAD_ROWS` (default: `EXL3_ROW_INV_GROUP_ROWS` (`8`) with `--sessions` above 1 in the Qwen server on ROCm; on NVIDIA `16` for a dense model and `0` for a model with MoE layers; `0` elsewhere)

Qwen speculative decoding: with N requests decoding together the draft depth is capped at `R // N - 1`, so the verify rows
of the step, `N * (depth + 1)`, stay within `R` rows (no draft when that is below 1). `0` turns the cap off.
`QWSPEC_LOAD_CAPS="N:D,..."` sets the cap for chosen N. Qwen3.8-Flash-Next on the box (v1.7.2 with an equivalent cap), 4 sessions,
4 greedy prompts: 54.9 tok/s aggregate without the cap, 71.2 with it.

#### `EXL3_MTP_EH_FP16` (default: unset, off)

Path of a safetensors sidecar holding an unquantized copy of the MTP `eh_proj`, which raises
draft acceptance without changing target token ids. `0` turns it off; a path that does not exist
stops the load. The GLM server looks for the file itself in three places, listed in
[tools/glm/SERVE.md](../tools/glm/SERVE.md).

#### `EXL3_DRAFT2S` (default: `<model folder>/draft2s.st` when present and matching)

Two-stage draft head for the MTP drafters: it finds the draft argmax in two steps instead of one pass over the full
vocabulary. Every draft is still verified, so the output does not change. The GLM and Qwen servers pick up
`draft2s.st` from the model folder when its header (vocabulary size, hidden width) matches `config.json`. A path in the
variable wins; `0`, `off` or an empty value uses the full head. Sampled jobs always use the full head.

#### `EXL3_DRAFT_VOCAB` (default: unset; the Qwen server sets `98304` on NVIDIA)

Qwen3.5 MTP drafter, 1-row greedy draft steps: read only the first N rows (a multiple of 128) of the shared 248K-row head
instead of all of it. On the RTX 3090 the K5 head is 0.93 ms of a ~1.5 ms draft step; N = 98304 makes it ~0.4 ms and
costs nothing on English and code (99.3 % of tokens have an id below 98304). Every 8th step runs both heads; a language whose tokens sit
above N (French, CJK) turns the subset off by itself (misses EMA >= 0.12) and back on when the probes agree (< 0.04). Every draft is
verified, so the text is the one without the flag. `0` or unset uses the full head; sampled jobs always use the full head.

#### `EXL3_MIMO_LOSSLESS` (default: `1`)

`0` restores the fast MiMo verify that is not token-identical. With it on, speculative output
equals plain greedy output, at about 5 % chat and 12 % code tok/s.

#### `EXL3_LAZY_DKV` (default: `1`)

Lazy drafter KV catch-up: a plain 1-row step parks the target hidden rows and the next drafting
round writes them in one pass, instead of feeding the drafter cache every step. `0` turns it off.

#### `EXL3_DRAFT_PRIOR` (default: `1`)

`0` skips `draft_conf_prior.json` from the drafter directory, so the draft confidence threshold
starts from the `--draft-confidence` flag instead of the offline table that seeds it.

#### `EXL3_SPEC_GATE` (default: `0`)

Cost-model gate that skips the drafter forward and the verify whenever the measured speculative
rate would not beat plain decode. Opt-in, and only active when a draft model is loaded. The MiMo
server copies its `--spec-gate` flag into this variable.

#### `EXL3_SPEC_PROF` (default: unset)

`1` records per-phase speculative timings. Every timed phase synchronizes the device, so it is a
profiling switch rather than a tuning one.

#### `EXL3_SERVE_WARMUP` (default: `1`)

`0` skips the GLM server's warm-up request and drops that stage from `/health`. The first real
request then pays the kernel warm-up.

#### `EXL3_SERVE_DTUNE_PRIME_S` (default: `600`)

Seconds the GLM server spends pre-tuning dense GEMM shapes before it opens the port. `0` skips the
tuning and the matching `/health` stage. Read only with `EXL3_DENSE_GEMM_TUNE=1`: the GLM server sets
`EXL3_DENSE_GEMM_TUNE=0` and `EXL3_CONV_GEMM=1` by default (see [EXL3_CONV_GEMM.md](EXL3_CONV_GEMM.md)).

#### `EXL3_SERVE_EMPTY_CACHE` (default: `1`)

`1` drops the allocator's inactive slack after the GLM warm-up. `0` keeps it.

#### `EXL3_SERVE_MEMDIAG` (default: unset)

`1` prints a memory line after the GLM warm-up.

#### `EXL3_TAIL_DENSE` (default: `0` in the engine; the Qwen server sets `7`, the GLM server `0`)

Dense prompt-tail checkpoints. Besides the checkpoint on the last page of a prompt, the recurrent
state is written at each of the `N` page boundaries below it, so a follow-up prompt that shares the
cached prefix up to anywhere in its last `N` pages (the same document with another question)
resumes within one page of the divergence instead of at the previous chunk start. Costs a few
hundred MiB of host memory. The kernel side-writes at most 8 rows per chunk: the last page and the
branch row come first, the dense rows fill the free slots, deepest first. On the KDA (GLM) layers
the rows are added only with `EXL3_MIDCHUNK_CKPT=2`; the other mode splits the layer at every row
and would drop all mid-chunk checkpoints. A resume replays a different number of rows than a cold
prefill and the dense GEMM path depends on the row count, so the output of a resumed turn is not
yet bit-identical to a cold prefill (a fix is in progress). `0` is off.

#### `EXL3_TAIL_SUB` (default: `0`)

Sub-page prompt-tail checkpoints: how many of the last multiples of 64 rows of the chunk that holds
the prompt end get a checkpoint, so a prompt that diverges in its final tokens replays at most
about 64 rows instead of up to a page. Entries that do not fall on a page boundary are kept in host
memory (about 142 MiB each, at most `EXL3_TAIL_SUB_MAX`). Same numerics caveat as
`EXL3_TAIL_DENSE`. `0` is off.

#### `EXL3_TAIL_SUB_MAX` (default: `8`)

Cap on the sub-page checkpoint entries the generator keeps (oldest dropped first). `0` keeps none
and requests no kernel slot for them. Read only with `EXL3_TAIL_SUB`.

#### `EXL3_LOOP_GUARD` (default: `1`)

The answer-phase loop guard of the Qwen and GLM servers (`tools/loopguard.py`). `0` turns it off for
every request. It ends a reply that has fallen into a loop: a repeated token period, or a tail that
replays itself. The finish reason is `length` and the choice carries `stop_reason: "loop_guard"`
(with the detail in `timings.loop_guard`), so a client can tell a guard stop from a natural end.
A request can switch the guard off for itself with `"loop_guard": false` (a boolean; `true` or
absent follows this variable). The Qwen server also watches the thinking block and restarts it
from the kept thinking; the GLM server leaves a thinking block alone and watches the answer after
`</think>`. Repetitive data that is valid (a grid of zeros, identical JSON lines, a zero-padded
hex dump) is not cut before 16384 tokens of repetition.

### Speed defaults the servers set

Each server applies a measured configuration before the engine loads, so a bare `serve.py` runs
the configuration its published numbers were measured with. They are applied with
`os.environ.setdefault`, so a value in your own environment always wins and nothing is forced on
you. Set any of them to `0`, or to the value named below, to go back. The Qwen server refuses to
start if `exllamav3` was imported before they were applied, because the modules that read them
latch the value at import.

#### Qwen: `SERVE_ENV` in `tools/qwen/serve.py`

| variable | value the server sets | what it does |
|---|---|---|
| `EXL3_HOST_LEAN` | `1` | Host-side decode trims: batched verify readback and reused staging buffers. Launches, draw order and numerics are unchanged. |
| `EXL3_HOST_CUTS` | `1` | The rest of the host trims: cached staging buffers and early-outs on no-op work. Only active together with `EXL3_HOST_LEAN=1`. |
| `EXL3_MOE_FUSED` | `1` | The fused MoE kernel. Ignored on a non-ROCm build. |
| `EXL3_MOE_VALU` | `1` | The FDOT2 VALU decode matvec for K2/K4 experts instead of the WMMA body (bitwise-identical outputs). K=2 also defaults on without it; `0` forces the WMMA body everywhere. |
| `EXL3_VERIFY_ATTN_LOOP` | `1` | Runs an R-row verify attention as the exact R single-row loop, which is what keeps a verify forward bit-identical. |
| `EXL3_VERIFY_GEMV_R` | `1` | Runs the decode projections of an R-row verify as one multi-row GEMV instead of R single-row calls. |
| `EXL3_GEMV_R_DEC1` | `1` | Decodes each weight pair once per verify forward and shares it across rows, instead of decoding it again per row. Bit-identical, and it needs `EXL3_GEMV_R_RM6`. |
| `EXL3_MTP_FUSE_CATCHUP` | `0` | `0` runs a catch-up draft prefill after each accepted round, `1` folds those rows into the next draft forward, `2` does that on the union MoE path. Unset means auto. |
| `EXL3_PLE_HIP` | `1` | Hands the post-projection PLE chain to the HIP kernels instead of the torch chain. |
| `EXL3_DQ_HIP` | `1` | Deinterleaves the fused q/g tensor with a HIP kernel, for the aligned fp16 shapes it accepts. |
| `EXL3_GR_HIP` | `1` | Hands the gated-residual prefill GEMMs to HIP, for large row counts on the Qwen shape. |
| `EXL3_GDN_FUSE` | `1` | The fused HIP chain for the GDN chunk path, which is the only path that can side-write the mid-chunk state, so `EXL3_PF_NO_TAIL` needs it. |
| `EXL3_PF_SKIP` | `1` | Skips the lm_head of a prefill chunk whose output is discarded anyway, since the first token comes from the next decode round. |
| `EXL3_PREFILL_CHUNK` | `4096` | The prefill chunk size in tokens, overriding the model default. Above 65535 a forward is not supported and the load raises. |
| `EXL3_PF_DEFER` | `0` | `1` runs the draft prefill after the first token instead of chunk by chunk with the target prefill. Needs a single session and no spec gate. |
| `EXL3_PF_NO_TAIL` | `1` | Lets the last prefill chunk run to the end of the prompt instead of stopping at the last page, which saves a second pass over every expert. |
| `EXL3_PF_TAIL_MERGE` | `1024` | How many rows past the chunk size a merged last chunk may take, so a short tail is merged rather than re-read. |
| `EXL3_PF_SPLIT` | empty (off) | Comma list of offsets, in rows before the end of the prompt, at which the last prefill chunk is cut into extra passes, for example `256` or `1280,256`. Each cut lands on a page-aligned row and is stored as an exact recurrent checkpoint, so a later request that shares the prefix up to a cut resumes with the same pass shapes as a cold prefill and gets bit-identical logits. The extra passes cost a little prefill time. |
| `EXL3_PF_SPLIT_MIN` | `4096` | The shortest prompt, in tokens, that `EXL3_PF_SPLIT` applies to. |
| `EXL3_PF_SPLIT_HEAD` | `1024` | The fewest rows the first pass of a split last chunk must keep. A cut that would leave less is dropped. |
| `EXL3_PF_SPLIT_STEP` | `512` | The least spacing, in rows, between two cuts of a split last chunk. |
| `EXL3_PF_SPLIT_Q` | `256` | The grid, in rows, that split cuts are rounded down to. Rounded down to a multiple of the page size. |
| `EXL3_VERIFY_QSA_PROJ` | `0` | `1` runs the indexer q and k projection per row instead of batched. |
| `EXL3_VERIFY_QSA_ATTN` | `0` | `1` or `all` runs sparse selection and attention per row, `sel` selection only, `att` attention only. |
| `EXL3_QSA_ROWINV` | `1` | Computes the sparse-attention split plan as if the batch were a single row, so a batched verify reduces in the same order as plain decode. `0` restores the batch-dependent plan. |
| `EXL3_ROLL_CKPT` | `1` | Makes every page boundary a recurrent checkpoint candidate while decoding, so a follow-up turn resumes at the last full page of the reply. |
| `EXL3_PF_GR_FUSE` | `1` | Applies the gated-residual update inside the next site's norm pass on the prefill path. |
| `EXL3_MPW_GLUE` | `1` | The v2 form of the MoE glue in the grouped prefill kernel, for a top-k of 1 to 32. |
| `EXL3_VERIFY_ROW_SPLIT` | `2` | Lets a fused MoE call cover up to 8 rows instead of 4, split into sub-calls. Per-row arithmetic is unchanged. |
| `EXL3_MOE_R8` | `1` | Runs rows 5 to 8 as one fused launch instead of split sub-calls, for the shapes where the wide kernel exists. |
| `EXL3_GEMV_R_RM6` | `1` | Runs 5 or 6 rows in one launch of the 6-row decode GEMV instead of two chunks. Needs `EXL3_GEMV_R_DEC1=1`. |
| `EXL3_GR_GS` | `1` | Group scales for the int8 gated-residual mixer instead of one scale per row. `2` keeps both resident, for a measurement harness. |
| `EXL3_GR_GS_SIDECAR` | `<model_dir>/hc_gs_sidecar.safetensors`, set by the server | Optional group int8 codes re-derived with GPTQ-style rounding on the same group scales. Unset means plain rounding. The server uses the file only when its hash matches the published pack. |
| `EXL3_MOE_CFG` | `2` | The k-split width of the grouped-MoE decode GEMV. `0` keeps the original wide shape, `1` and `2` narrow it. Read once per process. |
| `EXL3_HIP_PREFILL_MIN_ROWS` | `2` | The row count from which the grouped prefill GEMM starts deduplicating experts. The engine default is 17, which would miss a small verify batch. |
| `MPW_KERN` | `2` | Which grouped-MoE prefill GEMM kernel to use. |

#### GLM: `SPEED_ENV` in `tools/glm/serve.py`

| variable | value the server sets | what it does |
|---|---|---|
| `EXL3_MOE_CFG` | `2` | Same switch as above; the GLM verify batch is a matvec, not a prefill, so it needs the narrow k-split. |
| `EXL3_HIP_PREFILL_MIN_ROWS` | `2` | Same switch as above. |
| `EXL3_BLOCK_GRAPH` | `1` | Captures and replays the decode block as a graph, which takes the Python host time out of the step. Off by default in the engine. |
| `EXL3_BLOCK_GRAPH_MLA` | `2` | Extends graph capture to MLA blocks. `0` leaves them eager, `1` covers the post-attention segment, `2` also the projections and norms. |
| `EXL3_HC_FN_HALF_PF` | `1` | Reads the hyper-connection weights in fp16 at any row count, not only small ones, which halves the dominant traffic of that kernel. |
| `EXL3_HC_1PASS_PF` | `1` | The one-pass prefill kernel for the hyper-connection mix and norm, above the decode row cap. |
| `EXL3_DEC_DSA_FAST` | `1` | A tuned tile and split preset for the DSA decode kernel, for 2 or more rows. A list of numbers overrides the pieces. |
| `EXL3_MIDCHUNK_CKPT` | `2` | A prefill chunk that crosses a recurrent checkpoint boundary also saves the recurrent state there, so the follow-up turn can resume from it. |
| `EXL3_PREFILL_BIG_TAIL` | `1` | Allows the big-chunk tail path. Honored only with `EXL3_MIDCHUNK_CKPT=2`. |
| `EXL3_BIG_TAIL_DENSE` | `1` | On a sparse single-session decode pass longer than the query-length cap, lets the tail attend densely instead of sparsely. |
| `EXL3_PREFILL_BIG_CHUNK_MAXPOS` | `32768` | From this position on, prefill goes back to plain chunks, which lowers peak memory at very long context. `0` is off. |
| `EXL3_MOE_UNION_V2` | `1` | The v2 union MoE table, which stays on the device instead of synchronizing to the host. Already the engine default; `0` disables it. |
| `EXL3_VERIFY_ROW_SPLIT` | `2` | Same switch as in the Qwen table: a verify round of 5 to 8 rows (copy-lookup rounds) runs as fused sub-calls. Rows 4 and below are unchanged. |
| `EXL3_MOE_R8` | `1` | Same switch as in the Qwen table: rows 5 to 8 run as one fused MoE launch. |
| `EXL3_GEMV_R_RM6` | `1` | Same switch as in the Qwen table: 5 or 6 rows run in one launch of the 6-row dense GEMV. |
| `EXL3_SPEC_SAMPLED` | `1` | See the speculative-decoding entries below. The engine default is `0`; the GLM server turns it on. |

#### MiMo: the defaults in `tools/mimo/serve.py`

| variable | value the server sets | what it does |
|---|---|---|
| `EXL3_DEC_MOE_UNION` | `1`, with `EXL3_MIMO_LOSSLESS=1` | Also the default of `EXL3_VERIFY_GEMV_R`, so the multi-row verify GEMV is available. |
| `EXL3_VERIFY_WIDE_R` | `0` | The one-launch wide verify GEMV for the lm_head. On by default in the engine; off here because it costs about 7 % tok/s on this model. |
| `EXL3_DEC_MOE_UNION_DEV` | `1` | Keeps the unique-expert table on the device through a speculative round. Without it a round costs more than a plain step and the spec gate stops speculating. |
| `EXL3_SPEC_GATE` | `1` with `--spec-gate`, else `0` | Set from the flag rather than a default. See the feature-switch entry above. |

#### MiMo lane: `tools/lanes/serve_mimo.sh`

| variable | value the lane sets | what it does |
|---|---|---|
| `EXL3_PF_MSPLIT` | `2048` | Runs big prefill GEMMs as row blocks of this size, which is bit-exact and lowers peak memory. `0` is off. |
| `EXL3_MPW2X` | `2` | Tries the grouped-MoE prefill candidate kernel. `0` is off. |

The same lane also exports `EXL3_HOST_LEAN`, `EXL3_HOST_CUTS`, `EXL3_LOAD_DEVICE` and
`EXL3_DENSE_GEMM_TUNE_FILE`, all covered above.

#### Speculative decoding and start-up options of the GLM and Qwen servers

| variable | default | what it does |
|---|---|---|
| `GLM_NGRAM` | `2,7,16,3` with one session, `off` with more | Copy lookup for the GLM server, as `gram,K,match[,cmin]`. When the last `match` tokens of the context appeared before, the next round verifies the `K` tokens that followed them, instead of the drafter's tokens. `cmin` cuts the copy where the two latest occurrences stop agreeing, and drops it if that leaves fewer than `cmin` tokens. The default copies 7 tokens after a 16-token match. It speeds up requests that copy or edit text; other requests are unchanged. It needs about 0.7 GiB of GPU memory more. It works only while one request is active. `off` (or `0`, `none`) turns it off, and so does `--num-draft 0`. A malformed value stops the server at start. With greedy decoding, a near-tie in a long answer can make the text differ from a run without lookup. |
| `QWSPEC_NGRAM` | `2,7,16,3` | The same copy lookup for the Qwen server, in the same format. `2,3,5` gives the earlier setting (a 5-token match, 3 copied tokens). `off` turns it off. |
| `QWSPEC_NGRAM_IDX` | `1` | Builds the lookup index when the request is queued, so its cost lands in the time to first token and not in the first decode round. `0` builds it lazily and also turns off the `cmin` check. |
| `EXL3_SPEC_SAMPLED` | engine `0`; the GLM server sets `1` | For requests with temperature above 0 and an MTP drafter: the drafter draws its tokens from its own distribution, and the verify keeps each one with probability min(1, p/q), or resamples from the remainder. The output has the same distribution as plain sampling. Greedy requests are untouched. `0` goes back to taking the drafter's top token. The Qwen server does not set it. |
| `EXL3_GLM_SLOTS` | `0` (one recurrent state slot per session) | Minimum number of recurrent state slots the GLM server allocates. The default sizes them to `--sessions`, which frees about 6 GiB of GPU memory at one session. `16` restores the earlier layout. |
| `EXL3_GC_FREEZE` | Qwen server `1`, GLM server `0` | Moves the loaded model and tokenizer objects out of the Python garbage collector before the first generator is built. Without it, rebuilding the generator (for example after a `cache_salt` change) pays a collector pass of about 0.5 s inside the time to first token. `0` leaves them in the collector. |

#### Image embedding cache (all three servers)

| variable | default | what it does |
|---|---|---|
| `KYOJIN_IMAGE_CACHE_POLICY` | `head` | Which pictures a full image-embedding cache gives up. The cache is one shared module (`tools/strix_halo/image_cache.py`, with `image_cache.py` symlinks in `tools/glm`, `tools/qwen` and `tools/mimo`), so this switch reaches all three servers, and it is read when a server builds its cache at start-up. `head` is the plain LRU a server always had: the picture whose last use is oldest goes first, which for a client that re-sends its whole history every turn is the earliest picture of the prompt. `tail` follows the prompt instead — a picture no remembered prompt refers to goes first, then the last pictures of the remembered prompts, and the head of the prompt is the last thing given up — which lets a history that keeps being re-sent keep the start of its prompt, so its warm turns prefill less and reach the first token sooner. Decoding is unaffected. Any other value stops the server at start-up. |

### Server state

#### Request log (all three servers)

| variable | default | what it does |
|---|---|---|
| `KYOJIN_REQLOG_DIR` | unset = off | Directory for the per-request lifecycle log, or `1`/`on`/`yes` for `~/.cache/kyojin`. A path that ends in a file name is read as that file's **directory** (`/var/log/kyojin/requests.log` means `/var/log/kyojin`), since that is what pointing it at a log means; the log file is always `requests-<run>.log` in that directory. Every run writes its own file, `requests-<run>.log` (so the default is `~/.cache/kyojin/requests-<run>.log`), and every line carries `run=<id>`, so one run can be read without filtering the others out; the oldest runs are removed at start-up, see `KYOJIN_REQLOG_KEEP` (`0`/`off`/`none`/`no`/`false` disable it). Numbers and event names only, never prompt or reply text; one line per event and a heartbeat (`alive`) while a job is queued, prefilling or decoding. The file is created `0600` whatever the umask is, and an existing file left wider is tightened on the next start; a directory this server has to create is `0700`, while one that already exists keeps its own mode; the shared folder `~/.cache/kyojin`, or wherever `KYOJIN_HEALTH_DIR` points, is created with the process umask instead, its missing parents included, because more than one process writes there. For a streaming reply the heartbeat also carries the **output accounting** (`out_events`, `out_chars`, `last_out_s`) and the request ends with one `sse_summary` line, which is what tells "the client was sent nothing" from "the client showed nothing"; the event tables, the shared/GLM-only split and how to read it are in [logging.md](logging.md). Served by all three servers from one module (`tools/strix_halo/serve_reqlog.py`, with `serve_reqlog.py` symlinks in `tools/glm`, `tools/qwen` and `tools/mimo`): the whole request lifecycle — the request table, the id on every line, the heartbeat and the route wrapper (`Lifecycle`) — is that module's, so the three servers emit one vocabulary, which `tools/test_serve_logging.py` checks against all three at once; each server's document only points back here and to [logging.md](logging.md). |
| `KYOJIN_REQLOG_HEARTBEAT_S` | `10` | Seconds between `alive` lines; `0` disables the heartbeat (the log stays on). |
| `KYOJIN_REQLOG_KEEP` | `20` | How many runs keep their log file. Checked once, at start-up: the run files of the oldest runs are removed, so the directory holds this many (this run included). It counts only files this module writes (`requests-<run>.log`); an archive or a file an earlier layout appended to is never deleted. `0` keeps every run. |

#### SSE keep-alive (all three servers)

| variable | default | what it does |
|---|---|---|
| `KYOJIN_SSE_PING_S` | `0` = off | Seconds a stream may go without a data event before the server writes one SSE comment line (`: waiting <n> s`), which keeps a client that only watches its socket from treating a long prefill — or a long decode nothing is being written for — as a dead server. The comment counts from the request's arrival, comes once per interval and only after that much silence: a stream that is producing text never gets one, and one that stops gets them again (a reply holding a `<tool_call>` back until its XML closes is decoded for minutes while the client hears nothing). The `role` chunk a client gets immediately does not count as output, so the whole prefill is covered. Shared by the three servers (`tools/strix_halo/serve_ping.py`, with `serve_ping.py` symlinks in `tools/glm`, `tools/qwen` and `tools/mimo`), read per request, reported on one start-up line. With the default nothing is written at all. Worth setting on any server that sees long prompts — an image conversation can prefill for minutes — together with a client-side timeout that is long enough; each comment is also an `sse_ping` line in the request log (`KYOJIN_REQLOG_DIR`) when that is on and the server's pinger is wired to it (GLM's is). See [logging.md](logging.md). |

#### `KYOJIN_HEALTH_DIR` (default: `~/.cache/kyojin`)

Where the three servers keep the per-model start-up timings they use to weight `/health` load
progress and report `eta_s`: `startup-qwen.json`, `startup-glm.json` and `startup-mimo.json`. Set
it to share one cache between several checkouts. The GLM server keeps two files there, one for
starts with the dense GEMM tuning already cached and one without, so a cached start is not timed
with the slow first one.

## NVIDIA (CUDA, sm_80+)

Switches of the CUDA path. Defaults are the shipped behaviour on NVIDIA; none of them changes ROCm builds. Several
are not bit-identical to the Triton / eager path on greedy output (first divergences sit on near-ties, KL within the
noise of the quantization), which is why each has an off switch for A/B work.

#### `EXL3_CUDA_GRAPH` (default `0`; `tools/qwen/serve.py` and `tools/flint/serve.py` set it to `1` on NVIDIA)

Whole-forward CUDA graph for decode steps (Qwen GDN, Flint sliding-window ring, MTP verify). `EXL3_CUDA_GRAPH_MULTI=1`
(default off; `tools/flint/serve.py --max-batch N` and `tools/qwen/serve.py --sessions N` set it to `1` on NVIDIA when N > 1) lets several jobs share one graph (B in 1..4, B*Q <= 32). `EXL3_CUDA_GRAPH_MAX` bounds the graph cache (16 with MULTI, 32 from the Qwen server with `--sessions` > 1).

#### `EXL3_QDEC` (default `1`), `EXL3_QDEC_GRAPH` (default `1`), `EXL3_QDEC_ROWS` (default `1`)

Decode attention for quantized KV caches (K/V bits 2..8): a tensor-core split kernel compiled at run time (NVRTC, no
build step) that reads the packed pages directly. `EXL3_QDEC=0` falls back to the Triton kernels, `EXL3_QDEC_GRAPH=0`
disables its CUDA-graph capture path, `EXL3_QDEC_ROWS` selects the many-row kernel for verify steps (q_len 2..8 at
head dim 256): `0` off, `1` only where the single-row kernel cannot serve, `2` always.

#### `EXL3_QWRITE_TRITON` (default `1`; ROCm)

Native Triton writer for the quantized paged cache (K/V bits 2..8, `attention_fn/qwrite_triton.py`): one launch appends the new
rows of K and V with the same arithmetic and packed plane layout as the PyTorch fallback (bit-identical words and scales,
`tests/test_qwrite.py`). The fallback needs ~70 launches per layer (decode append 402 us, 4096-row prefill chunk 5.2 ms at
K/V 6,6 per layer; native 17 us / 89 us). `0` restores the fallback.

#### `EXL3_QDEC_HIP` (default `1`; ROCm gfx11.5)

HIP counterpart of `EXL3_QDEC` for AMD (Strix Halo): a wave32 WMMA kernel (`attention_fn/qdec.hip`, compiled with hipcc at first
use and cached under `~/.cache/exllamav3`) that dequantizes the packed K/V pages straight into the WMMA operands, plus a small
combine kernel. Used for quantized-KV decode (K/V bits 2..8, head dim 128 or 256, q_len up to 6 at head dim 128 / 3 at 256) with
full attention, no softcap; windowed layers, 1-bit planes and bigger shapes keep the Triton kernels. `0` keeps the Triton kernels.
Tuning knobs for A/B work: `EXL3_QDEC_HIP_NW` (waves per CTA, default 4), `EXL3_QDEC_HIP_DW` (waves splitting the head dim),
`EXL3_QDEC_HIP_WAVES` (target resident waves for the split count), `EXL3_QDEC_HIP_SPL` (fixed split count),
`EXL3_QDEC_HIP_MAXO` (accumulator budget per lane, default 96).

#### `EXL3_SHARED_SIDE` (default `0`; ROCm)

Batch-1 MoE decode with a shared expert that is not folded into the routed launch (Flint: shared expert K=6, I=1024; the fold needs
K=4 and the routed shape): the shared expert's GEMVs, which do not depend on the routed experts, run on a side stream with their
own GEMV scratch/counters while the routed kernels run on the main one, and are joined at the shared add. Bit-identical results
(same kernels, tests/gate_kv.py KL 0), Flint 4K / 32K decode +4..5 % (tests/ab_decode.py, alternating windows in one process).
`EXL3_DEC_MIN_BLOCKS` (default 32; read per call) at 512 adds about 1.5 % on Flint (more blocks for its small GEMVs).

#### `EXL3_PREFILL_WIDE` (default `1`)

Models with sliding-window attention only and a fused MoE (Flint): prompts that fit are prefilled in one pass, in
8192-row chunks while the cache holds less than 64K tokens, 4096 beyond. `0` restores the plain 4096-row plan.

#### `EXL3_MOE_GROUPED` (default `1`), `EXL3_MOE_GROUPED_LO` (default `33`), `EXL3_MOE_GROUPED_ROWS`

Grouped MoE prefill tier: one CTA per (expert, 64-row tile, 128-column tile) with fp32 accumulation and the output
Hadamard fused with the top-k combine, for experts that receive at least `EXL3_MOE_GROUPED_LO` rows. `0` restores the
persistent expert-group kernel.

#### `EXL3_GEMV_POLICY` (default `1`), `EXL3_GEMV_ROWS`

Slice height of the int8 decode GEMV on 4/5-bit weights: taller k-slices, about two work units per SM. `0` restores the
former rule, `EXL3_GEMV_ROWS=<n>` forces a slice height.

#### `EXL3_QC_PF_MAXNREG` (default `255` for 8-bit K+V and above, else `0`)

Register cap of the quantized-KV prefill attention kernel (`0` = compiler default). Uncapped registers free 1.3 GiB of
driver local memory and let Flint run `--kv 8,8` at 256K.

#### `EXL3_TAIL_PIN` (default `1` on NVIDIA), `EXL3_TAIL_RING` (default `0`)

`EXL3_TAIL_PIN=0` makes the prompt-end recurrent checkpoint a synchronous `.cpu()` copy instead of an asynchronous copy
into pinned host memory. `EXL3_TAIL_RING=1` allows a tail restore into a partial page when the draft cache is a ring
smaller than the target cache (a DFlash2 drafter).

#### `QWSPEC_DEPTHRULE` (default unset)

`N:thf,thv,maxd,thfd,thvd`: above N context tokens the MTP draft tier rule switches to these thresholds and draft depth
(`QWSPEC_THF/THV/THFD/THVD` set the base tiers). Needed only with `EXL3_QDEC=0` or `EXL3_QDEC_ROWS=0`, where the
Triton verify makes deep drafts expensive: `96000:0.45,0.2,4,0.8,0.6`.
