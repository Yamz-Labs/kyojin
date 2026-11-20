# Changelog

## Unreleased

- Decensor preset for Hy4-flash, on by default (`HY4_ABLIT_SPEC=0` switches it off; an explicit path loads another spec): a per-layer run-time edit of the attention and MLP / MoE outputs, no weight change. Files in `presets/hy4-uncensor/`, code in `exllamav3/loader/ablit_hy4.py`.

## 1.9.0: Hy4-flash v1

Hy4-flash (GGUF Q3_K) on one Strix Halo machine. Numbers: see the benchmark table in the release notes.

- New `tools/hy4/serve.py`: OpenAI-compatible server for Hy4-flash with prefill fast path, HIP-graph decode, MTP drafts and prompt-lookup rows. Guide: `doc/hy4.md`.
- Sparse-attention prefill: per-query gather MLA kernel in HIP/WMMA (`hipsp/`, H = 32 and H = 64), Triton fallback; blocked indexer score with linear-time exact top-k; sparse path from 2048 keys.
- Long-context decode: exact top-k sparse attend (`HY4_LC_SPARSE`), draft-cache fill with the same selection.
- Prefill: fused dequant GEMMs, WMMA expert kernels shared with the GGUF MoE path, CK latent-attention core, dense GEMMs split in pieces of 8192 rows above 8192 rows.
- Decode: static-buffer graph step, MTP head from a Q5_K copy of the output head, n-gram rows on top of the drafts, Engram rows prefetched by row.
- The server applies the frozen TunableOp results (`tools/hy4/tunableop_gfx1151.csv`) and a start-up warm-up request by default (`HY4_TUNABLEOP_FILE`, `HY4_WARMUP_TOKENS=0` to skip it), and picks up `head_q5_k.bin` next to the GGUF.
- Generic GGUF loader (`EXL3_GGUF_GENERIC=1`, off by default) and its CPU tests.
- Context: 64K tokens measured (`-c 65536` in the guide); `-c 131072` is opt-in and uses almost all of the memory of a 128 GB machine. Single user (`--slots` is experimental). Greedy decoding only.
- A missing built kernel library (repository extension, MoE extension, attention libraries) stops the server with the install step; `HY4_ALLOW_SLOW=1` keeps the old slow-path start.
- Decode levers on by default: head-batched MLA attention (`HY4_MLA_HB`), k_b / v_b and output-head products on `tl.dot` (`HY4_BG_DOT`, `HY4_HEAD_DOT`), fp16 router weights (`HY4_RG_F16`), fused K-row / q-rope kernels for verify rows (`HY4_KV_FUSED_MAXT`). Step tests in `tools/hy4/step/`.
- The MTP draft-cache fill after prefill writes only K / V and the indexer key state (`HY4_PRIME_KVONLY`, on by default).
- Prefill levers on by default: fused SwiGLU + cast and cast-add glue, finer M-split of the dense GEMMs (`HY4_PF_G_SWIGLU`, `HY4_PF_G_ADD`, `HY4_PF_MM_FINE`); Engram host side with a numpy hash, a single prefetch worker and native row reads (`HY4_ENGRAM_NPHASH`, `HY4_ENGRAM_PF_WORKERS`, `HY4_ENGRAM_NATIVE_IO`); sparse prefill attention from 2048 keys.
- Native Engram IO is built with `gcc` at first use (`~/.cache/kyojin/engram_io`). Without `gcc`, or when the build fails, the server uses the Python reader.
