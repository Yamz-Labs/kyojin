# bigconv design notes

## Layout
- `arch.py`: `ArchAdapter` (tensor roles + regexes, unit/expert enumeration, shapes, `capture_runner`, `TODO_OPS`) and the
  table-driven `TableAdapter`. Two seams only: the adapter (what the tensors are) and the source format (`scale_suffix`).
- `plan.py`, `alloc.py`, `tensor_map.py`: generic ports of tools/ds41 (plan output byte-identical on the real DS4.1 map, 12 flag variants).
- `adapters/{ds41,glm5,minimax_m3}.py`; `runners/ds41.py` wraps the existing `exllamav3.dsv41` graph (nothing re-implemented).
- `sheet.py`: resource sheet. `capture_stream.py`: memory-guarded capture.
- tools/ds41 is untouched except `common.py` (range fetch also retries http.client.HTTPException).
- `defaults.py`, `orchestra.py` (sweeper, lead, prefetch, claims, FAILED recovery, shared-GPU gate, verified steps, exit guard), `preflight.py`, `archcheck.py`
  (`bigconv check`): operational lessons of the live DS4.1 run, each with a CPU test (tests/bigconv/test_orchestra.py, test_preflight_check.py).

## Memory-guarded capture
Reference capture keeps N tokens in RAM: row states [N, hc, d], pooled MoE input X [N, d], output Y [N, d] fp32
(~116 kB/token on DS4.1: 121 GiB at 1M tokens, 235 GiB at 2M). Peak RAM grows with N.

`StreamedCapture` keeps RAM flat:
1. Row state `h` lives in one np.memmap file per row, updated in place; the cross-layer hand-off (pools, top-k) is a small torch file per row.
2. Pass A: per row, attention half, x written to the X memmap [N, d]. Dense Hessians (d x d) accumulate across rows.
3. Router runs on X chunks; idx/wts [N, k] stay in RAM (12 B/token/slot).
4. Experts one at a time: tokens gathered from X in chunks, Hessian `H += x^T x` accumulates chunk by chunk (additive),
   weighted outputs are scatter-added into the Y memmap. Shared expert runs chunked over X, then is added to Y.
5. Pass B: per row, attention half recomputed (hooks off), block output from Y, new `h` written in place.
Peak RAM = one row + one chunk + one layer of Hessian accumulators + one expert's weights + idx/wts. Disk = h + X + Y.
Cost: the attention half runs twice per row (cheaper than spilling its context).

Numerics. One chunk covering the largest expert and one router chunk is bitwise the reference (`torch.equal`, all 8 layers + head
on the tiny model). Smaller chunks change only fp32 summation order: one layer agrees to 1e-6 relative Frobenius;
over many layers a random tiny net amplifies it (router flips), checked loosely.

## Tests (CPU, no GPU)
`BIGCONV_ENGINE=<dir with exllamav3/dsv41> BIGCONV_DS41_MAP=<real DS4.1 tensor map> pytest tests/bigconv`
Missing env vars skip the engine and real-map tests explicitly.

## Map of tools/ds41 (claimdrv 79132dcf): generic vs DS4.1-bound
| file | status in bigconv |
|---|---|
| plan.py, alloc_file.py, tensor_map.py | ported (plan.py byte-identical on the real DS4.1 map; alloc schema from the adapter) |
| common.py (HTTP ranges), fetch.py | generic, reused as is |
| quant.py (quantizer back-ends), split.py (claims, coordinator/worker, spool, resume, budget) | generic logic, NOT ported or re-run on a non-DS4.1 plan |
| capture.py (SpoolSink, RoutingRecorder, HoCapture) | sink/spool generic; the forward is `StreamCalibrator` (DS4.1 graph) -> now behind `runners/ds41.py` |
| stream_convert.py, layer_pipeline.py | unit loop and capture->quantize->delete driver: DS4.1-bound by key names and units (see below) |
| dequant.py | source-format seam: DS4.1 fp4 e2m1 + ue8m0 and fp8 blocks; GLM fp8 `weight_scale_inv` is the same fp8 block form; MiniMax is bf16 (no dequant) |
| ramfit.py, make_alloc.py, optimize_alloc.py, synth_alloc.py | DS4.1 sizes and allocation rules; `bigconv sheet` replaces the size part |
| assemble_pack.py, partial_pack.py, load_test_pack.py | pack writers, DS4.1 index/config layout |
| engram_subset.py, kron_*.py, layer_err.py, evalref.py | DS4.1 specific (Engram, hyper-connection forward) or study tools |

OPEN, blocks running split/stream_convert/layer_pipeline on a GLM or MiniMax plan: those tools match DS4.1 canonical keys
(`<unit>.ffn.experts.E.w1`, `.ffn.gate.bias`, units `global`, `layers.L`, `mtp.i`). bigconv plan entries keep the source
names (`...mlp.experts.0.down_proj`). Needed: an adapter `canonical_key` + source-name map used by split/stream_convert.

Capture paths NOT covered by `StreamedCapture` (tested against the engine's `StreamCalibrator`, not tools/ds41/capture.py):
MTP stages (`mtp_begin`, `run_mtp_stage`), kron-ho (`HoCapture` monkeypatches `moe_calibrate`), the spool sink/driver wiring,
the routing recorder callback is only a hook. `min_expert_tokens` is tested on layer 0.

## Sheet validation on DS4.1 (real map, 2.3 bpw)
| quantity | sheet | measured |
|---|---|---|
| pack | 171.4 GB | 171 GB |
| spool / layer | 24.7 GB without kron-ho; 32.9 GB with the kron-ho upper bound | 24.6-24.7 (plain), 29.6 (kron-ho config: only experts with >= 2304 tokens carry it; real routing skew, not modelled) |
| capture RAM, reference code | 121 GiB at 1M tokens, 235 GiB at 2M | the 117 GB run was killed before 2M; the slope of 116 kB/token matches the logged per-token cost (within 5 %) |
| capture RAM at 1000-row runs | 7.2 and 15.0 GB | 6.9 and 14.3 GB |

Write volume of the memmap design (printed by the sheet): h is rewritten each layer, X once, Y once to (topk+1) times.
DS4.1 at 2M tokens: 246-492 GB per layer, 10-20 TB over 40 layers, footprint 246 GB. The next model trades the 117 GB RAM peak for
a disk-write cliff (QLC NVMe drives: about 140 GB sustained). Mitigation, not built: process tokens in blocks so X/Y blocks fit in RAM.
