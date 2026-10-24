# Experimental wide K6/K8 launches

This change requires the K6/K8 fused MoE patch. It runs five to eight rows
in one launch. Without the setting, the dispatcher splits these rows into
groups of at most four. Existing low-bit paths do not change.

Use the settings in K6K8.md. Also set EXL3_MOE_R8_K6S8=1 before startup.
The path requires native layout and the six-stage variant, without timing
or other experimental variants. Rebuild the extension before use.

## Checks

Run from the repository root:

```sh
python -m pytest --noconftest -q tests/test_moe_fused_wide_k6k8_cpu.py
```

The test file also builds `tests/moe_fused/moe8_bounds.cpp` with ASan and
UBSan in a pytest temporary directory and runs it (it is skipped when g++ is
missing). It checks that `K6S8_WIDE_SHAPES` in `moe_fused.py` is the list of
K6S8 shapes in `exl3_moe_fused_idx.h`.

On an idle ROCm GPU, with the built extension:

```sh
for shape in small real; do
  for gs in 0 1; do
    for seed in 1234 5678; do
      SHAPE="$shape" GS="$gs" SEED="$seed" python tests/moe_fused/moe8_k6k8_synth.py
    done
  done
done
```

The synthetic check requires exact equality between wide and split outputs.
It checks repeated calls, duplicate rows, batch reshape, buffer bounds,
and both top-k modes. It also measures wide and split launch times.
It does not stop services or load a model.

## Measured on gfx1151 (this revision)

Synthetic weights, `tests/moe_fused/moe8_k6k8_synth.py`, two seeds and both
group-scale settings, mean launch time. Wide and split outputs were bit-identical in all 16 runs.
Real Qwen shape (2560 wide, 512 experts, top 10):

| Expert selection | Rows | Split (ms) | Wide (ms) | Wide speedup |
| --- | ---: | ---: | ---: | ---: |
| random | 5 | 1.177 | 1.056 | 1.11x |
| random | 6 | 1.361 | 1.234 | 1.10x |
| random | 7 | 1.536 | 1.403 | 1.09x |
| random | 8 | 1.713 | 1.574 | 1.09x |
| same for all rows | 5 | 0.667 | 0.444 | 1.50x |
| same for all rows | 6 | 0.684 | 0.475 | 1.44x |
| same for all rows | 7 | 0.703 | 0.508 | 1.39x |
| same for all rows | 8 | 0.720 | 0.496 | 1.45x |

Small test shape:

| Expert selection | Rows | Split (ms) | Wide (ms) | Wide speedup |
| --- | ---: | ---: | ---: | ---: |
| random | 5 | 0.084 | 0.059 | 1.41x |
| random | 6 | 0.088 | 0.064 | 1.38x |
| random | 7 | 0.094 | 0.069 | 1.35x |
| random | 8 | 0.098 | 0.073 | 1.34x |
| same for all rows | 5 | 0.078 | 0.054 | 1.45x |
| same for all rows | 6 | 0.081 | 0.057 | 1.42x |
| same for all rows | 7 | 0.085 | 0.059 | 1.43x |
| same for all rows | 8 | 0.087 | 0.062 | 1.40x |

Wide was faster than split in all 32 configurations, so nothing needs to be
excluded by row count. When the rows pick different experts, the gain on the
real shape is 7 to 13 percent; when they share their experts it is 39 to 51 percent.

Full model (6.05 bpw pack, 4K and 32K context, fused K6/K8 enabled, only
`EXL3_MOE_R8_K6S8` added): greedy text and every top-10 log-probability of 8
prompts are bit-identical to the split path (1843 positions), so the numerical
difference to the unfused path discussed in K6K8.md is the same with and without
this change. Decode median 16.6 against 16.4 tokens/s at 4K and 17.0 against
16.6 at 32K; at this sample size that is within noise.

Build: with its prerequisite the extension grows from 134.66 MB (base) to 137.29 MB,
of which this change adds about 1.1 MB over the prerequisite alone (136.20 MB); a full
build took 517 s against 469 s for the prerequisite alone (three parallel jobs, machine not idle).

Registers and spills (hipcc `-Rpass-analysis=kernel-resource-usage`, gfx1151).
The wide launch uses the six stage kernels (`moe_fused9s_kernel`, stage 0 to 5):

| Kernel | VGPRs per stage 0 to 5 | Spilled VGPRs | Waves/SIMD |
| --- | --- | ---: | ---: |
| `Wide<QwenK6S8>` | 183, 192, 163, 190, 179, 29 | 0 in all six | 8 |
| `Wide<SmallK6S8>` | 101, 154, 41, 135, 123, 29 | 0 in all six | 9 to 16 |
| for comparison `Wide<QwenShape>` (K4/K4) | 183, 192, 163, 187, 161, 29 | 0 | 8 |

The one-kernel variants are not used by the wide path (see the comment in
`exl3_moe_fused9.cu`). For the record, `moe_fused9_kernel<Wide<QwenK6S8>>`
uses 232 VGPRs without spills, and `moe_fused9o_kernel<Wide<QwenK6S8>>` uses
192 VGPRs and spills 20 (64 bytes per lane of scratch); the same variant for
K4/K4 spills 3. No kernel that existed before this change changed its
register or spill count.

## Earlier hardware results

The prototype ran on an EVO-X2 with 128 GB RAM and gfx1151, using the local
Qwen3.8 Flash-Next EXL3 pack labelled 6.05 bpw. The full-model comparison
changed only EXL3_MOE_R8_K6S8. Other local optimizations remained enabled.
Code throughput increased about 3.6%, explanation about 1%, and story
about 0.6%. Concurrent throughput increased about 5%.
A greedy code case changed from 45.72 to 48.61 tokens/s. Tested token IDs
matched. These are historical observations from the prototype, not GPU
validation of the rebased PR. They do not resolve the numerical differences
between fused and unfused MoE described in K6K8.md.

Repeat GPU and full-model checks on this revision before merge.
Use tools/bench.sh with the same machine and command for each comparison.
