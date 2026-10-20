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
g++ -O1 -g -std=c++17 -fsanitize=address,undefined -fno-sanitize-recover=all -Iexllamav3/exllamav3_ext/quant tests/moe_fused/moe8_bounds.cpp -o /tmp/moe8-bounds
/tmp/moe8-bounds
```

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
