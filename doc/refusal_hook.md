# Optional refusal hook

The engine can project one fixed direction out of the residual stream at run time. It edits no weights and no quantised data. Off unless `EXL3_ABLIT_RUNTIME=/path/to/spec.json` is set or the model folder contains `uncensor_spec.json`.
- Bundled spec: if `<model_dir>/uncensor_spec.json` exists (with `uncensor_direction.st` next to it, or `uncensor_spec.safetensors`), the engine applies it at load and logs ` -- ablit runtime: spec <path> active (bundled in the model directory)`. An explicit `EXL3_ABLIT_RUNTIME=<path>` wins. `EXL3_ABLIT_RUNTIME=off` or `--no-uncensor` (GLM and MiMo servers) disables it. A missing or malformed spec stops the load with an error that names the file.
- The spec is a JSON file (`hidden`, `n_layers`, per-layer weights `attn_w[L]`, `mlp_w[L]`) plus a unit vector `r` in `spec.safetensors`.
- After each attention and MLP block of layer L the output becomes `y - w_L * r * (r . y)`.
- `EXL3_ABLIT_TORCH=1` forces the PyTorch path instead of the Triton kernel.
- A direction fitted on refusals makes the model refuse less. Whoever uses a spec is responsible for it. This repository contains no spec file. Steering presets are published separately in the Yamz presets repository (https://github.com/yamz-labs/yamz-presets).
- Code: `exllamav3/modules/ablit_runtime.py`. Test: `tests/test_ablit_runtime_cpu.py`, `tests/test_uncensor_bundled_cpu.py`.
