# Experimental dense K8 decode

Set EXL3_DENSE_K8=1 before model load to enable direct decode for five
Qwen attention projection shapes. Rebuild the extension before use.
The setting is disabled by default. The weight files do not change.

Supported input/output sizes are 2560/512, 2560/6144, 2560/10240,
2560/12288, and 6144/2560. Weights must use K8 and the mul1 codebook.
Routed MoE eligibility does not change.

The list is a Python-side choice, not a kernel limit: the C++ side accepts
any K8 mul1 matrix once the variable is set. It holds every K8 matrix of the
tested K6-routed / K8-shared pack that is not an expert: k_proj and v_proj
(2560/512), in_proj_z (2560/6144), in_proj_qkv (2560/10240), q_proj
(2560/12288), and o_proj / out_proj (6144/2560). The shared-expert
projections (2560/640 and 640/2560) are left out on purpose. Add a shape
only together with a real-weight run of `check_dense_k8.py`.

The variable must not change after the model is loaded. Python reads it once
per projection at load time to decide whether the direct path is eligible.
The C++ launch reads it again on every launch and refuses K8 when it is
not `1`. Setting it after the model is loaded has no effect (the projections
were already marked not eligible). Unsetting it or changing it after a load
that used it makes the next K8 launch raise an error.

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

## Measured on gfx1151 (this revision)

Real-weight check on the 6.05 bpw K6 routed / K8 shared pack, ten projections
(k_proj, v_proj, q_proj, o_proj, in_proj_z, in_proj_qkv and out_proj, from
linear-attention layers, full-attention layers and the MTP layer), rows 1 to 8, two seeds, fp16 and
fp32 output: 320 comparisons, none bit-identical, largest relative L2 error
5.57e-4 (the figure of the earlier prototype run). Kernel time against
reconstruct plus half-precision GEMM: 2.4 to 2.7 times faster for 2560/512, 3.6 to 6.2 times
for the larger shapes.

Is that error normal for the dense kernels? The same check on the direct paths
that already exist gives the same size: K4 projections (six, linear attention
layers of a 4.0 bpw pack) 5.3e-4 to 5.6e-4, K6 projections (five, the MTP layer
of that pack) 5.2e-4 to 6.3e-4. Against an fp64 evaluation of the dequantized
weights, the direct kernel is the closer one for K4 and K8 alike:

| Output | Direct kernel vs fp64 | Reconstruct + GEMM vs fp64 |
| --- | ---: | ---: |
| fp16 | 2.9e-4 to 3.0e-4 | 4.6e-4 to 4.7e-4 |
| fp32 | 2.0e-4 to 2.1e-4 | 2.8e-4 to 3.0e-4 |

(relative L2, rows 1 to 8, K8 in_proj_qkv and q_proj; the K4 projections
give the same values to two digits). The 5e-4 difference is therefore mostly
the rounding of the reference path, not an error of the new kernel.

Full model, 6.05 bpw pack, only `EXL3_DENSE_K8=1` changed, decode medians of
nine runs: 14.6 to 27.8 tokens/s at 4K context (1.90x) and 15.1 to 28.0 at 32K
(1.86x). Prefill is not routed through this path and stayed within noise
(0.95 to 1.05x). Greedy text diverged from the default on 17 of 32 prompts;
see K6K8.md for how large such rounding-level changes are in this model
(largest top-10 probability change 9.6 points on 5358 positions, RMS change of
the chosen-token log-probability 0.013, mean change -0.0004).

Registers and spills (hipcc `-Rpass-analysis=kernel-resource-usage`), K8 kernels
with the K4 to K7 kernels that existed before for comparison. The last two
numbers of `gemv_kernel_r` are the row count and a variant flag:

| Kernel | VGPRs | Spilled VGPRs | Scratch (bytes/lane) |
| --- | ---: | ---: | ---: |
| `gemv_kernel<16, 2>` (K8, 1 row) | 256 | 100 | 212 |
| `gemv_kernel<14, 2>` (K7, before) | 256 | 29 | 88 |
| `gemv_kernel<8, 2>` (K4, before) | 105 | 0 | 0 |
| `gemv_kernel_r<16, 2, 1, x>` (K8) | 256 | 73 | 152 |
| `gemv_kernel_r<16, 2, 2, 1/0>` | 197 / 252 | 0 | 0 |
| `gemv_kernel_r<16, 2, 4, 1/0>` | 245 / 256 | 0 / 104 | 0 / 228 |
| `gemv_kernel_r<16, 2, 8, 1/0>` | 256 | 184 / 270 | 480 / 700 |
| for comparison `gemv_kernel_r<8, 2, 8, 1/0>` (K4, before) | 256 | 299 / 269 | 768 / 496 |
| for comparison `gemv_kernel_r<6, 2, 8, 1/0>` (K3, before) | 256 | 288 / 280 | 624 / 460 |

K8 spills like the existing eight-row kernels and less than most of them; the
single-row kernel spills more than the K7 one, and the timings above include that.
No kernel that existed before changed its register or spill count.

Build: the extension grows from 134.66 MB to 134.92 MB (+0.27 MB, +0.2%); a full build took 493 s
against 490 s for the base (three parallel jobs, machine not idle). Compiling `exl3_dec` alone takes
249 s instead of 242 s and its object grows from 14.49 MB to 14.77 MB (+2%).

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
