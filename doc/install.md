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
Newer ROCm: AMD also publishes a multi-arch index with more recent builds. Replacing the torch line above with the one below also installs, builds and passes the check on gfx1151 (verified on a fresh clone, Python 3.12, with the Qwen pack: same prefill and decode speed up to 32K). The published figures were measured with the line above, which stays the default; generated text can differ slightly between the two builds.
```bash
pip install --pre --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1151]==2.14.0+rocm10.1.0a20260822" "rocm[devel,device-gfx1151]==10.1.0a20260822"
```
Status: build and MiMo serving were verified from a fresh clone on a second Strix Halo machine (build in 8 to 10 minutes). Both published packs were checked there against `SHA256SUMS` and served (one chat request each); `env.sh --check` has not been run there yet. The first request after a build is slow while the kernels warm up.

First GLM launch: the server tunes its dense GEMM kernels before it opens the port. This takes about 10 minutes, and the port stays closed during that time (the log prints a progress line every 30 seconds). The next one or two launches can repeat it for the shapes still missing; later launches start fast.

Memory: GLM at `-c 131072` leaves about 10 GiB free on a 128 GB machine. Close browsers and other large processes first, or use a smaller `-c`.

## System settings for full speed

Our servers run with the three settings below. Each gain was measured on our Strix Halo machines, with the same generated text.

1. **IOMMU off.** Add `amd_iommu=off` to the kernel command line (for example in `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo update-grub` and reboot). With the IOMMU on we measured 6 % less prefill and 2.5 % less decode. Note: the NPU needs the IOMMU, so this turns the NPU off.
2. **GPU clock level `high` while serving.** It holds the GPU clock at its top level instead of letting it drop between kernels: GLM 8K prefill +5.7 %, decode +0.5 to +1.4 % (GLM and Qwen). Set it before you start the server and put it back to `auto` after, so the GPU does not stay at full clock when idle:
   ```bash
   echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
   # ... serve ...
   echo auto | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
   ```
   Do not use `profile_peak`: it pins every clock, uses the power budget and was 5 to 10 % slower.
3. **Cooling.** The chip slows itself down when it runs hot. The same machine at 96-99 °C gave 15 to 20 % less GLM prefill than when cool. Keep the air inlets and outlet free; in a closed cabinet, a small fan that moves the hot air out helps.

   **Sixunited AXB35-02 boards** (Bosgame M5, GMKtec EVO-X2, FEVM FA-EX9, Peladn YO1, NIMO, Corsair AI Workstation 300; check with `cat /sys/class/dmi/id/board_name`): the factory fan curve keeps the CPU fans at 60 % up to 83 °C and reaches 80 % only at 95 °C, so the chip sits at its thermal limit and cuts its power. The [ec-su_axb35-linux](https://github.com/cmetz/ec-su_axb35-linux) driver lets Linux set the curve. On a Bosgame M5 in Performance mode, a 3-minute GPU load went from 86-94 W, 2100 MHz and 95-100 °C to 140 W, 2850 MHz and 79-80 °C with the curve below. It is louder under load; at idle the fans still stop.
   ```bash
   git clone https://github.com/cmetz/ec-su_axb35-linux && cd ec-su_axb35-linux
   make && sudo make install && sudo modprobe ec_su_axb35
   echo ec_su_axb35 | sudo tee /etc/modules-load.d/ec_su_axb35.conf
   D=/sys/class/ec_su_axb35
   for f in 1 2; do   # CPU fans: level 1-5 = 20-100 %, thresholds in °C
     echo curve | sudo tee $D/fan$f/mode
     echo 50,60,70,78,86 | sudo tee $D/fan$f/rampup_curve
     echo 40,52,62,72,80 | sudo tee $D/fan$f/rampdown_curve
   done
   echo curve | sudo tee $D/fan3/mode   # system fan
   echo 20,50,65,75,85 | sudo tee $D/fan3/rampup_curve
   echo 0,42,58,70,80 | sudo tee $D/fan3/rampdown_curve
   ```
   The curve is lost at reboot: put the `echo` lines in a script run by a systemd oneshot service. Rebuild the module after a kernel update.

The CPU governor should be `performance` (check `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`); the GLM server is partly CPU-bound.

Problems: [troubleshooting.md](troubleshooting.md).
