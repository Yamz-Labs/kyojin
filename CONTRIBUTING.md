# Contributing to Kyojin

Thank you for helping. Kyojin is the Yamz inference engine for AMD Strix Halo, built on [ExLlamaV3](https://github.com/turboderp-org/exllamav3). Issues and pull requests are welcome.

## Most wanted

- **Benchmarks from other gfx1151 machines** (Ryzen AI Max+ 395 and 390, other vendors, other RAM sizes). Run `tools/bench.sh` and post the output with the benchmark report template.
- **Other ROCm GPUs.** Other GPUs are untested. Reports of what builds, what runs and what fails are useful even when the answer is "it does not work".
- **Bug reports with logs.** Use the bug report template: hardware, kernel, ROCm version, exact command, and the server log.
- **Documentation.** Install problems you hit, missing steps, unclear wording.

## Build

Follow the Quickstart in `README.md`. In short:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --pre torch rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
pip install -r requirements.txt
rocm-sdk init
export EXL3_ROCM_SDK=$(rocm-sdk path --root)
source tools/strix_halo/env.sh      # from the repository root, in every new shell
./build.sh
```

## Run the CPU tests

These need no GPU and no built extension. Run each file separately, because two files share the name `test_serve.py`:

```bash
pip install pytest
for t in tests/test_ablit_runtime_cpu.py tests/test_uncensor_bundled_cpu.py tools/glm/test_serve.py tools/mimo/test_serve.py tools/mimo/test_toolcalls.py; do PYTHONPATH=. pytest -q $t; done
```

The whole `tests/` directory also runs without a GPU: `PYTHONPATH=. pytest tests`. Tests that need a GPU, the built extension (`import exllamav3` loads it) or Triton are skipped with the reason (`-rs` lists them); the needs are checked in `tests/hardware.py`. A new test that needs one of them takes `@pytest.mark.gpu`, `@pytest.mark.ext` or `@pytest.mark.triton`, or, when its file cannot even be imported without it, calls `require("ext")` from `hardware` before that import.

On the GPU machine, run the suite with `EXL3_TESTS_STRICT=1`: nothing is skipped for missing hardware, so a broken build fails instead of skipping.

If a test or the first import hangs, look for a stale lock file at `~/.cache/torch_extensions/*/exllamav3_ext/lock` and remove it. GPU tests under `tests/` need a built extension and a free GPU; say in the pull request whether you ran them.

## Pull requests

- Keep a pull request to one change. Explain what it fixes and how you checked it.
- For speed changes, include before and after numbers from `tools/bench.sh` on the same machine, and the command you used.
- Do not add new dependencies without a reason in the description.
- Do not commit model weights, secrets, local paths or personal data.

## Code and commit conventions

- Python: follow the style of the file you edit. Keep functions small and name things for what they do.
- Environment switches use the `EXL3_` prefix (for example `EXL3_ABLIT_RUNTIME`). Document each new switch next to the code that reads it.
- Commit subjects are short, in English, in the imperative, with no prefix and no trailing period (for example: `Fix union MoE RM=8 kernels: process rows in groups of 4`). Add a body when the reason is not obvious.
- Make normal commits. Do not rewrite history on a branch others use.

## Upstream

Kyojin is built on ExLlamaV3 by turboderp. A change that belongs to ExLlamaV3 itself (general quantisation, the CUDA path, the tokenizer, shared generator logic) should go to [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) first. Send to this repository what is specific to gfx1151, ROCm, the Strix Halo serving tools or the supported models.

## Optional GPU decode checks

These are extra manual checks for the ROCm decode path. They are not part of `pytest tests`
and each needs a built extension, a free GPU and a model that fits the card. Set
`EXL3_TEST_MODEL` to the pack and pin the device with `HIP_VISIBLE_DEVICES`.

```bash
# Run the same short prompt through the normal reconstruction path and the fast GEMV
# path, then print top-1 agreement, route counts and logit differences.
EXL3_TEST_MODEL=<model> python tests/hip_gemv_logits_oracle.py

# Longer trace, and an optional unforced run that prints tokens for loop diagnosis.
EXL3_TEST_MODEL=<model> EXL3_ORACLE_STEPS=128 python tests/hip_gemv_logits_oracle.py
EXL3_TEST_MODEL=<model> EXL3_ORACLE_FREE=1 python tests/hip_gemv_logits_oracle.py

# Eight everyday questions in two sampling modes: checks EOS, loop guard and repetition.
HIP_VISIBLE_DEVICES=1 python tools/strix_halo/coherence_suite.py -m <model>

# Times each matrix shape and a full streaming pass; --gemm compares with exl3_gemm.
HIP_VISIBLE_DEVICES=1 python tools/strix_halo/gemv_perf_ab.py -m <model> --mode both --gemm
```

For an MCG-codebook pack, point `EXL3_TEST_MODEL` at that pack; the oracle command is
the same. Say in the pull request which checks you ran.

## Licence

Contributions are under the MIT licence of the repository.
