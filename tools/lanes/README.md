# Lane launchers

The two launchers run the servers with the flags, sampling defaults and speed environment of our serving lanes. All paths come from environment variables or arguments.

| launcher | server | model env | notes |
|---|---|---|---|
| `serve_glm.sh` | `tools/glm/serve.py` | `GLM_MODEL` (default `~/models/GLM-5.3-Flash-EXL3-Yamz`) | MTP n=2, medium template, T 1.0 / top_p 0.95, reasoning effort medium |
| `serve_mimo.sh` | `tools/mimo/serve.py` | `MIMO_MODEL` (default `~/models/MiMo-V2.6-Flash-MOPD-EXL3-Yamz`) | DFlash speculation on by default (`MIMO_SPEC=0` for plain decode), speed envs `EXL3_HOST_LEAN`, `EXL3_HOST_CUTS`, `EXL3_PF_MSPLIT=2048`, `EXL3_MPW2X=2` on by default (an exported value wins), T 1.0 / top_p 0.95, context ceiling derived from free memory |

Common: `EXL3_VENV` (ROCm torch venv, default `<root>/.venv`), `EXL3_HSA_LIB` (HSA runtime to preload), `LANE_STATE` (scratch dir), `*_ROOT` (engine checkout with the built `exllamav3_ext*.so`).

The runtime uncensor hook (`EXL3_ABLIT_RUNTIME`) is off unless a spec is given. Three ways to enable it: (1) the model directory contains `uncensor_spec.json` (uncensored repos: applied automatically, the log prints `-- ablit runtime: spec <path> active (bundled ...)`); (2) set `GLM_ABLIT` / `MIMO_ABLIT` to an edit spec; (3) export `EXL3_ABLIT_RUNTIME=<spec>`. Precedence: explicit path, then bundled file. `EXL3_ABLIT_RUNTIME=off` or `--no-uncensor` switches it off (the launcher prints `hook=` accordingly). Presets are not part of this tree.

`assets/glm53-template-medium.jinja` is the GLM chat template of the lane. `assets/tune_seed.txt` seeds the dense-GEMM tune cache (`EXL3_DENSE_GEMM_TUNE_FILE`) so the first start skips the autotune.
