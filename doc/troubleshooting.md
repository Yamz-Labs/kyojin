# Quickstart troubleshooting

Install and first-run problems on Strix Halo (gfx1151), as symptom and fix. Run the commands below from the repository root, with the venv active (`source .venv/bin/activate`). After a fix (and `./build.sh` again for the build errors), `bash tools/strix_halo/env.sh --check` should print `GPU compute: OK` and `exllamav3_ext: OK`.

## Segfault (exit code 139) at the first GPU operation, or `undefined symbol: hsa_...`

Symptoms, all from the same cause:

- The process dies with exit code 139 at the first tensor on the GPU, even `torch.zeros(4, device="cuda")`, with no Python traceback. `dmesg` shows `segfault ... in libhsa-runtime64.so`.
- `undefined symbol: hsa_ext_image_create_v2` when torch is imported.
- `undefined symbol: hsa_amd_enable_logging` when torch is imported.

Cause: the wrong `libhsa-runtime64.so.1` was loaded. The copy bundled in the torch wheel segfaults on gfx1151, Ubuntu's own `libhsa-runtime64-1` is ROCm 5.7 and too old (`hsa_amd_enable_logging`), and an older ROCm under `/opt/rocm` does not match the wheel (`hsa_ext_image_create_v2`).

In the segfault case, `torch.cuda.is_available()` returns `True` and the device is listed correctly, so they are not proof that the runtime works. The crash comes with the first allocation or copy.

Fix: run these in every new shell, from the repository root. `env.sh` then preloads the runtime from the SDK first:

```bash
source .venv/bin/activate
export EXL3_ROCM_SDK=$(rocm-sdk path --root)
source tools/strix_halo/env.sh
```

To use another copy of the runtime, set `EXL3_HSA_LIB` to its path before sourcing `env.sh`, for example `export EXL3_HSA_LIB=$(rocm-sdk path --root)/lib/libhsa-runtime64.so.1`.

If `env.sh` prints `[env.sh] WARNING: no libhsa-runtime64.so.1 found`, none of its candidates exist: set `EXL3_ROCM_SDK` or `EXL3_HSA_LIB` as above.

## Prefill is about 20 % slower than the published figures

The server prints one line at start, `[kyojin] HIP kernel compiler: ...`, and `/health` shows the same text (field `hip_compiler`). If it points to `/opt/rocm` or another system compiler and a `WARNING` line follows, the engine built its kernels with the system ROCm instead of the SDK compiler. Run `export EXL3_ROCM_SDK=$(rocm-sdk path --root)` and restart. The kernel cache is keyed by compiler version, so the new start rebuilds the kernels once; the old files stay in `~/.cache/exllamav3` and can be deleted by hand. Please include that line in a performance report.

## `HIP error: invalid device function` on every kernel

Cause: ROCm 6.4 or older. It has no code object for gfx1151.

Fix: use ROCm 7.0 or newer, for example the gfx1151 nightly wheels from the quickstart. Leave `HSA_OVERRIDE_GFX_VERSION` unset: gfx1151 is native on ROCm 7, and an override makes it pick the wrong kernels. An older install may have set it in your shell profile, so check:

```bash
echo "${HSA_OVERRIDE_GFX_VERSION:-unset}"   # must print "unset"
unset HSA_OVERRIDE_GFX_VERSION               # for this shell; also remove it from ~/.bashrc or ~/.profile
```

## The build stops: no hipsparse headers

```text
[build] EXL3_ROCM_SDK=... and EXL3_ROCM_DEV_INCLUDE have no hipsparse headers. ...
```

Cause: the build needs the `hipsparse/` and `thrust/` headers of the ROCm SDK devel tree, and `EXL3_ROCM_SDK` does not point at one. `_rocm_sdk_core` alone does not have them.

Fix: install `rocm-sdk-devel` (see the quickstart), expand its headers, and point `EXL3_ROCM_SDK` at it:

```bash
rocm-sdk init                                  # expands the devel headers, about 12 GB on disk
export EXL3_ROCM_SDK=$(rocm-sdk path --root)
./build.sh
```

With your own devel tree, set `EXL3_ROCM_DEV_INCLUDE` to an include directory that has `hipsparse/` and `thrust/` instead.

## The build stops: PyTorch is not a ROCm build

```text
[build] the venv's PyTorch is not a ROCm build (torch.version.hip is empty). ...
```

Cause: `pip install torch` (or a `requirements.txt` install that ran first) installed a CUDA or CPU build.

Fix: install torch from the gfx1151 index first, then the requirements:

```bash
pip install --pre torch rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
pip install -r requirements.txt
```

Check with `python -c "import torch; print(torch.version.hip)"`: it must print a version, not `None`.

## A test or the first import hangs with no output

Cause: a stale lock file from an interrupted JIT load. When `exllamav3` is imported and no built extension is found, torch compiles it on the fly, and it first waits on `~/.cache/torch_extensions/<version>/exllamav3_ext/lock`, silently, as long as the file is there. `./build.sh` installs through pip and does not use this lock.

Fix: make sure no other process is loading the extension, then remove the lock:

```bash
ls ~/.cache/torch_extensions/*/exllamav3_ext/lock
rm ~/.cache/torch_extensions/*/exllamav3_ext/lock
```

Then run the command that hung again. If `ls` prints `No such file or directory`, there is no stale lock, and the hang has another cause.

If `TORCH_EXTENSIONS_DIR` is set, the lock is under that directory instead.
