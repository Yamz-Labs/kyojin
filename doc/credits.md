# Credits and licence

ExLlamaV3 by turboderp (MIT, `LICENSE` unchanged). The AMD work starts from [vcruz305/exllamav3-amd](https://github.com/vcruz305/exllamav3-amd) (first gfx1151 port) and [sdougbrown/exllamav3](https://github.com/sdougbrown/exllamav3) (ROCm decode path for gfx12). `exllamav3/vendor/fla` is flash-linear-attention (MIT). GLM-5.3-Flash is by Z.ai, MiMo-V2.6-Flash by Xiaomi; check each base licence before redistributing weights. Additions: MIT. This project is not affiliated with Z.ai, Xiaomi or turboderp.

The CUDA paths of upstream are kept; AMD additions sit behind `USE_ROCM` and architecture guards. This repository holds the engine, the serving scripts and the benchmark harnesses; the quantisation pipeline that produced the packs is not part of it.

News and benchmarks: [@YamzLabs on X](https://x.com/YamzLabs).

## Contributors

### Code

- [@felladrin](https://github.com/felladrin): pull requests for the GLM start-up, the GPU target lookup, the bench tool tests, the troubleshooting page, `--sessions` in the Qwen server and `tools/bench.sh --json`; bug reports, an independent benchmark and issue triage.
- [@twstagg](https://github.com/twstagg): the Nix flake (dev shell, package and OCI image) and its serve fixes; a Framework Desktop benchmark.
- [@resynth](https://github.com/resynth): `--cache-bits 8` in the GLM server (int8 MLA latent cache), the gfx1201 (RDNA4) bring-up and a prefill bound fix.
- [@razethion](https://github.com/razethion): opt-in K6 and K8 fused MoE, dense K8 decode and wide launches; tuning results on an EVO-X2.
- [@MangetsuC](https://github.com/MangetsuC): the GLM image path without solver searches (first image of a new size in seconds) and the dense GEMM tune primer.
- [@mostlygeek](https://github.com/mostlygeek): the container build on each release; the Pi `developer` role request.
- [@zaxbysauce](https://github.com/zaxbysauce): opt-in CK sparse prefill for MiMo on gfx1151.
- [@thebigen01](https://github.com/thebigen01): the prefix-disk restore fix for unreachable checkpoints; reported the slow second prefill.
- [@nobert](https://github.com/nobert): the Prometheus `/metrics` endpoint, merged for GLM and MiMo.
- [@Idriss091](https://github.com/Idriss091): the whole test suite without a GPU and clearer qualify errors.
- [@JoaoFay](https://github.com/JoaoFay): escaping reserved image markers in Qwen prompts.

### Reports and benchmarks

- [@kamjin3086](https://github.com/kamjin3086): benchmark of Kyojin 1.6 against another engine and the report of a host freeze at start-up; both led to fixes in 1.7.
- [@OleksKon](https://github.com/OleksKon): reported the prompt cache stopping after an image; fixed.
- [@spawnax](https://github.com/spawnax): reported a sampling error on padded vocabulary ids; fixed.
- [@alanoo81](https://github.com/alanoo81): benchmark on an HP ZBook G1a and the K5 slow-path report.
- [@dturini12](https://github.com/dturini12): reported streaming without `finish_reason`; fixed and confirmed.
- [@ChrisC381](https://github.com/ChrisC381): reported the GLM difference between paired jobs.
- [@morrisfamily](https://github.com/morrisfamily): benchmark report on a BOSGAME Beyond-Max.
- [@xXm-mXx](https://github.com/xXm-mXx): benchmark on a Beelink GTR9 Pro.
