# Changelog

## Hy4-flash v1 (draft)

Hy4-flash (GGUF Q3_K) on one Strix Halo machine. Numbers: see the benchmark table.

- New `tools/hy4/serve.py`: OpenAI-compatible server for Hy4-flash with prefill fast path, HIP-graph decode, MTP drafts and prompt-lookup rows. Guide: `doc/hy4.md`.
- Sparse-attention prefill: per-query gather MLA kernel in HIP/WMMA (`hipsp/`, H = 32 and H = 64), Triton fallback; blocked indexer score with linear-time exact top-k; sparse path from 16K keys.
- Long-context decode: exact top-k sparse attend (`HY4_LC_SPARSE`), draft-cache fill with the same selection.
- Prefill: fused dequant GEMMs, WMMA expert kernels shared with the GGUF MoE path, CK latent-attention core, dense GEMMs split in pieces of 8192 rows above 8192 rows.
- Decode: static-buffer graph step, MTP head from a Q5_K copy of the output head, n-gram rows on top of the drafts, Engram rows prefetched by row.
- Generic GGUF loader (`EXL3_GGUF_GENERIC=1`, off by default) and its CPU tests.
- Context: 128K tokens. Single user (`--slots` is experimental). Greedy decoding only.
