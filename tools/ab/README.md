# In-process A/B (one model load per arm-set)

An env-knob A/B no longer needs a server reload per arm.

1. Start the server with `--admin-knobs` (tools/qwen/serve.py or tools/glm/serve.py; opt-in, loopback clients only).
2. `python3 tools/ab/ab_inproc.py PORT --arm A:EXL3_GEMV_TRI=0 --arm B:EXL3_GEMV_TRI=1 --order ABBAAB --synth 4096 [--users N] [--fresh]`
   Each visit POSTs `/admin/knobs`, discards one warm-up request, runs every prompt greedy and streamed. Report: per-arm median step ms
   (one SSE chunk = one decode step), TTFT, aggregate rate, and the md5 of every reply (`md5 stable` = every visit of the arm gave the same bytes; a switch check only with `--users 1`: with concurrent users the batch composition varies from run to run and near-tie greedy tokens can flip inside one arm).
   Every visit runs under a new `cache_salt` (the server drops its prefix cache, so visits start alike); `--fresh` also warms up under another salt so nothing, not even prompt 0, comes from the cache (prefill / TTFT arms).
3. `tools/ab/run_ab.sh` wraps lock + server start + `/health` wait + ab_inproc + kill-by-PID (settings in `AB_*` env vars, see its header).
4. Engine-level profiles: `tools/depth/depth_profile.py MODEL --serve glm --arm A: --arm B:EXL3_DEC_MOE_KBS_A=9 --order ABBAAB` (one load, md5 of the logits per arm).

`POST /admin/knobs {"EXL3_X": "1", "EXL3_Y": null}` (null = default) answers `{"effective": {...every knob...}, "prefix_cache_reset": bool}`. A value that
really changed drains the requests, drops the captured CUDA/HIP/block graphs and the per-shape caches, and resets the prefix cache (disk copy included).

Adding a knob: read it with `knob("NAME", default)` / `knob_int` / `knob_on` from `exllamav3.util.knobs` at call time (never at import), list it in
`KNOBS` with the same default (tests/test_knobs.py checks both), and register a `@on_change` hook if a module keeps a derived global. C++ must `getenv` per call,
not in a `static const` initializer. What a captured graph baked is dropped by the purge; the next call recaptures.

Gate: `tools/ab/knob_md5.py --serve '<server cmd with {port}>' --knob NAME=VALUE`: toggling in a live server == a fresh server started with the value, both ways.
Tree-vs-tree release gates (two code trees) keep one load per tree: batch every prompt and N of the gate in that load.
