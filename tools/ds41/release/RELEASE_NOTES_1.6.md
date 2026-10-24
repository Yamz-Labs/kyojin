## Kyojin 1.6.0

Two things in this release: a persistent prompt cache on NVMe for every server, and an **experimental** DeepSeek-V4.1-Flash-1017 on one 128 GB Strix Halo machine.

GLM-5.3-Flash and Qwen3.8-Flash-Next speeds are unchanged against 1.5.10 (served A/B on the same machine, ABAB, one load per run): GLM prefill 745 vs 737 tok/s, decode 28.3 vs 28.3; Qwen prefill 1624 vs 1627, decode 49.8 vs 49.9 (8K and 32K prompts, median of 6).

### Persistent prompt cache (all servers)

- Set `EXL3_PREFIX_DISK_DIR` and the prompt-cache pages are written to the NVMe in the background. After a restart, or when a session resumes the next day, only the new tokens are prefilled.
- Measured on Qwen: a restarted server restores the cached prefix in about 0.1 s (14 pages) and the second turn is token-identical to the in-memory path. On GLM: 47 pages restored in 156 ms, time to first token 0.37 s instead of 1.8 s.
- One LRU cap for all models (`EXL3_PREFIX_DISK_GB`, default 30), a free-space floor and a write-rate limit. Pages are keyed by token hash and model fingerprint. Details: `doc/env_vars.md`.

### DeepSeek-V4.1-Flash-1017 (experimental)

**This is a demonstration, not a daily agent model.** 763B parameters run exactly (no pruning, no approximated routing) on one 128 GB machine, but slowly: real decode is about 5 to 11 tok/s, and a tool turn that adds about 2K tokens waits 20 to 35 s before the first token, because the experts that do not fit in RAM are read from the NVMe again. Guide, numbers and sources: `doc/DS41.md`.

- **Pack.** 172 GiB EXL3, routed experts at 2 and 3 bits (more bits on the most sensitive layers), attention and shared expert at 5 bits. On Hugging Face: yamz-labs/DeepSeek-V4.1-Flash-EXL3-Yamz. The Engram tables (FP8, about 190 GB) are the original files from deepseek-ai/DeepSeek-V4.1-Flash-1017; they are not re-hosted.
- **Quality.** Against the unquantized model, 20,470 positions: top-1 agreement 0.940, KL 0.068. The 115 GiB SAGE quant on the same run: 0.777 and 0.561.
- **SSD expert tier.** A pool of expert slots in RAM, the other experts read from the NVMe on demand (O_DIRECT, 12 readers, next-layer reads issued during compute), with scan-resistant admission so one long prompt does not flush the hot set. Greedy tokens are identical to the untiered run.
- **DSpark.** The model's own draft stages; greedy output equals plain decoding token for token.
- **Prompt cache** in RAM and on disk: an agent loop that resends its history only prefills the new tokens (follow-up turns: first token in 1 to 4 s).
- Image input, thinking (`reasoning_effort`, `thinking`), tool calls, OpenAI-style streaming API.
- `tools/ds41/release_measure.sh <pack>` prints the quality and speed tables for your machine.

### Fixes

- MoE: batches of 17 to 31 rows now run through the decode kernel in slices. The previous fallback gave wrong logits with the SSD tier (garbled answers with thinking off).
- fp32 path where fp16 overflowed on massive activations (illegal memory access).
- The OpenRouter-style `reasoning` object is read.

### Notes

- Only gfx1151 with ROCm is tested.
