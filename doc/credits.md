# Credits and licence

ExLlamaV3 by turboderp (MIT, `LICENSE` unchanged). The AMD work starts from [vcruz305/exllamav3-amd](https://github.com/vcruz305/exllamav3-amd) (first gfx1151 port) and [sdougbrown/exllamav3](https://github.com/sdougbrown/exllamav3) (ROCm decode path for gfx12). `exllamav3/vendor/fla` is flash-linear-attention (MIT). GLM-5.3-Flash is by Z.ai, MiMo-V2.6-Flash by Xiaomi; check each base licence before redistributing weights. Additions: MIT. This project is not affiliated with Z.ai, Xiaomi or turboderp.

The CUDA paths of upstream are kept; AMD additions sit behind `USE_ROCM` and architecture guards. This repository holds the engine, the serving scripts and the benchmark harnesses; the quantisation pipeline that produced the packs is not part of it.

News and benchmarks: [@YamzLabs on X](https://x.com/YamzLabs).
