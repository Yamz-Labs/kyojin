# Install and first launch (Strix Halo, ROCm)

Requirements: Linux (Ubuntu/Debian tested), a gfx1151 machine with 128 GB, Python 3.12 with its headers, a C++ compiler (`g++`; on Ubuntu `sudo apt install g++ python3-dev python3-venv`, on Fedora `gcc-c++ python3.12-devel`, on Arch `gcc` and `python` have everything), a ROCm 7 `libhsa-runtime64.so.1` (the PyTorch copy segfaults on gfx1151 and Ubuntu's own `libhsa-runtime64-1` is ROCm 5.7, too old; `tools/strix_halo/env.sh` finds the one inside the SDK wheel below first, then /opt/rocm, or take `EXL3_HSA_LIB=<path>`), ROCm 7.0 or newer (ROCm 6.4 has no gfx1151 code), a ROCm build of PyTorch for gfx1151, and a ROCm SDK devel tree with the `hipsparse/` and `thrust/` headers (the `rocm-sdk-devel` wheel).

```bash
git clone https://github.com/Yamz-Labs/kyojin && cd kyojin
python3 -m venv .venv && source .venv/bin/activate
# 1. ROCm torch + SDK first (AMD gfx1151 wheels; plain `pip install torch` gives a CUDA/CPU build that cannot run here):
pip install --pre torch rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
pip install -r requirements.txt
rocm-sdk init                              # expands the devel headers (about 12 GB on disk)
export EXL3_ROCM_SDK=$(rocm-sdk path --root)   # or the path of your own devel tree
source tools/strix_halo/env.sh             # run from the repository root; sets LD_PRELOAD and PYTHONPATH (torch needs it to import)
./build.sh                                 # compiles the extension for gfx1151 into the repo root (exllamav3_ext*.so)
bash tools/strix_halo/env.sh --check       # prints versions, runs a small GPU matmul (needs a free GPU)
hf download yamz-labs/GLM-5.3-Flash-EXL3-Yamz --local-dir ./glm-pack
python tools/glm/serve.py --model ./glm-pack --port 8000 -c 131072 --num-draft 2
curl http://localhost:8000/v1/models
```
In every new shell, before serving: activate the venv, run `export EXL3_ROCM_SDK=$(rocm-sdk path --root)` again, then `source tools/strix_halo/env.sh`. The engine builds some of its GPU kernels on first use, and the server prints at start which compiler it used. The SDK compiler from the export is the fast one; the server also finds it by itself when `rocm-sdk` is installed in the same venv, but the export is the safe way. If the log shows a warning that the system compiler is in use, prefill can be about 20 % slower, so export the line above and restart. Without the export, `env.sh` falls back to a system ROCm that may not match the wheel (`undefined symbol: hsa_ext_image_create_v2`).
If a step fails, see [troubleshooting.md](troubleshooting.md).
Status: build and MiMo serving were verified from a fresh clone on a second Strix Halo machine (build in 8 to 10 minutes). Both published packs were checked there against `SHA256SUMS` and served (one chat request each); `env.sh --check` has not been run there yet. The first request after a build is slow while the kernels warm up.

First GLM launch: the server tunes its dense GEMM kernels before it opens the port. This takes about 10 minutes, and the port stays closed during that time (the log prints a progress line every 30 seconds). The next one or two launches can repeat it for the shapes still missing; later launches start fast.

Memory: GLM at `-c 131072` leaves about 10 GiB free on a 128 GB machine. Close browsers and other large processes first, or use a smaller `-c`.

Problems: [troubleshooting.md](troubleshooting.md).
