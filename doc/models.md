# Models: weights, serve commands, options

Weights: [Yamz on Hugging Face](https://huggingface.co/yamz-labs). Serve commands assume the venv, `EXL3_ROCM_SDK` and `env.sh` from [install.md](install.md).

## Qwen3.8-Flash-Next

`hf download yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz --local-dir ./qwen-pack`, then `python tools/qwen/serve.py --model ./qwen-pack --port 8000 -c 131072`. Speculative decoding is on by default and returns the same tokens as plain decoding. The pack needs about 113 GiB of the 128 GB; start to ready takes about 90 s once the pack is in the page cache; the very first start reads 95 GB from disk and can take 7 to 8 minutes. Details: `tools/qwen/SERVE.md`.

## GLM-5.3-Flash

Architecture: `glm_moe_dsa` (MLA attention, sparse indexer, MTP). Pack: `yamz-labs/GLM-5.3-Flash-EXL3-Yamz`.

```bash
hf download yamz-labs/GLM-5.3-Flash-EXL3-Yamz --local-dir ./glm-pack
python tools/glm/serve.py --model ./glm-pack --port 8000 -c 131072 --num-draft 2
```

GLM reasons at maximum effort by default. With a small `max_tokens`, the reasoning can use the whole budget: `content` is then empty and `finish_reason` is `length`. Send `"reasoning_effort": "low"` in the request (or start the server with `--default-reasoning-effort low`), or allow at least 1000 tokens. The server defaults to greedy sampling when a request gives no temperature; set the sampling values of the model card in your client, or with `--default-temperature` and `--default-top-p`.

GLM: the pack ships `mtp_eh_proj.st`, an unquantized MTP `eh_proj` (64 MiB). The server loads it automatically from the model folder; it raises draft acceptance and decode speed (26.7 -> 31.6 tok/s greedy on the card protocol). Details in `tools/glm/SERVE.md`.

## MiMo-V2.6-Flash

Architecture: `mimo_v2` with a DFlash drafter. Pack: `yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz`.

`hf download yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz --local-dir ./mimo-pack`, then `python tools/mimo/serve.py --model ./mimo-pack --port 8000 -c 131072`. Speculative decoding (DFlash, 4 bpw drafter, confidence-truncated drafts) is on by default: the server uses the pack's `drafter/` directory (or `$MIMO_DRAFTER`, or `--drafter <dir>`). Without a drafter it logs one line and decodes plain. Set `MIMO_SPEC=0` in the lane script (`tools/lanes/serve_mimo.sh`) or pass `--no-dflash` to `serve.py` for plain decode. Greedy output under speculation is token-identical to plain decode (exact verify arithmetic). `EXL3_MIMO_LOSSLESS=0` selects a slightly faster verify that is not token-identical: near-tied logits can flip. A loaded drafter costs 2 to 4 % prefill. Details: `tools/mimo/SERVE.md`.

## Options shared by the servers

- `--sessions N` (Qwen server): decode up to N requests together; the default is 1. Details and measurements: `tools/qwen/SERVE.md`.
- `GET /metrics` (Qwen, GLM and MiMo servers): Prometheus text with llama.cpp-style `llamacpp:*` names, for scraping. Details: the `SERVE.md` next to each server.
- Measure your own machine with `tools/bench.sh` (`--json` adds the numbers as a JSON block); see [benchmarks.md](benchmarks.md).
