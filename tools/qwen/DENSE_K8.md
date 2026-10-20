# Experimental dense K8 decode

Set EXL3_DENSE_K8=1 before model load to enable direct decode for five
Qwen attention projection shapes. Rebuild the extension before use.
The setting is disabled by default. The weight files do not change.

Supported input/output sizes are 2560/512, 2560/6144, 2560/10240,
2560/12288, and 6144/2560. Weights must use K8 and the mul1 codebook.
Routed MoE eligibility does not change.

## Checks

```sh
python -m pytest --noconftest -q tests/test_dense_k8_cpu.py
python tools/qwen/check_dense_k8.py --model "$MODEL_DIR" --output dense-k8-results
```

The second command requires an idle ROCm GPU and the built extension.
The output directory must not exist. The check loads selected projections,
checks buffer bounds and dispatch, reports numerical error, and measures
kernel time. It does not stop services or change model files.
A successful exit does not mean that outputs are bit-identical.
Review results.json before use.

## Earlier hardware results

The prototype ran on an EVO-X2 with 128 GB RAM and gfx1151, with a local
Qwen3.8 Flash-Next EXL3 pack labelled 6.05 bpw. These measurements preceded
the PR rebase and are not validation of this revision.

| Prompt | Before (tokens/s) | After (tokens/s) |
| --- | ---: | ---: |
| Code | 24.54 | 40.92 |
| Explanation | 19.09 | 36.50 |
| Story | 17.30 | 32.10 |

The 320 real-weight comparisons had a maximum relative L2 error of
0.0005568. Outputs were not bit-identical. Repeat GPU validation and
full-model benchmarks before merge. Use tools/bench.sh with the same
machine and command for each comparison. Do not combine these numbers
with the fused-MoE results: the test configurations differed.
