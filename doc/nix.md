# kyojin on Nix

Reproducible dev shell, locked Python environment, ROCm toolchain, and a
container image, all from `flake.nix` (flake-parts + uv2nix + git-hooks.nix).
For the plain virtualenv install, see [install.md](install.md). For symptoms,
see [troubleshooting.md](troubleshooting.md).

Requires Nix with flakes enabled. Budget around 30 GB of store for the full
build closure; the ROCm SDK tree alone is about 9 GB.

## Dev shell

```bash
nix develop
```

This gives Python 3.12, `uv` for lock operations only (no `uv sync` or
`uv run`; Nix owns the environment), `ruff`, gcc, ninja, the locked app
environment, the ROCm SDK with `EXL3_ROCM_SDK` exported, and pre-commit hooks
installed on entry. On a machine without a gfx1151 GPU the shell is CPU-only:
imports that touch the GPU stack hang on wrong hardware, so export
`ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES=""`
for CPU-only work. The test check below does exactly that.

## Dependencies (`uv.lock`)

`uv.lock` is committed and is the single source of truth. Change dependencies
with `uv add` / `uv remove` in the shell, run `uv lock`, and commit the lock.
`setup.py` avoids importing the top-level package at eval time so isolated
metadata builds (`uv lock --check`, PEP 517 `get_requires`) keep working.

## Building

```bash
nix build .#exllamav3      # engine wheel + gfx1151 extension, 8-10 min
nix build .#kyojin         # serve wrappers (qwen/glm/mimo)
nix build .#torch-rocm     # ROCm torch wheel
nix build .#rocm-sdk       # expanded SDK tree (headers, hipcc, libhsa)
nix build .#docker         # OCI image tarball, about 5 GB compressed
nix run .#kyojin-serve-glm -- --help
```

`nix flake check` runs the hook suite plus the CPU test subset
(`checks.kyojin-tests`, GPUs hidden, bounded runtime). `nix fmt` doubles as
the clean-tree gate: it exits non-zero when a hook fails or rewrites files.

## ROCm pins

Pinned from AMD's multi-arch index, uv-resolved `--pre` set, ROCm 10.1.
Torch tracks the line verified in doc/install.md:

| Artifact | Version |
|---|---|
| torch | 2.14.0+rocm10.1.0a20260822 |
| rocm / sdk-core / sdk-libraries / sdk-device-gfx1151 | 10.1.0a20260822 |
| rocm-sdk-devel | 10.1.0a20260822 |
| rocm (registry sdist) / bootstrap | 10.1.0a20260822 / 0.1.0 |
| triton (ROCm) | 3.8.0+git675c5987.rocm10.1.0a20260822 |

To bump: re-run the `uv pip compile` from the install guide against the
index, fetch each wheel hash with `nix-prefetch-url`, and update `url` and
`sha256` in `flake.nix` (wheel URLs stay percent-encoded, keep the `%2B`).
Then rebuild `.#torch-rocm` and `.#exllamav3`.

## Running on gfx1151 hardware

Bare metal (after `nix copy` to the machine, or with a shared store):

```bash
nix run .#kyojin-serve-qwen -- --model ./qwen-pack --port 8000 -c 32768
```

Container with podman (`docker run` works the same):

```bash
nix build .#docker -o kyojin.tar.gz
podman image load -i kyojin.tar.gz
podman run -d --name kyojin --replace \
  --device=/dev/kfd --device=/dev/dri --group-add keep-groups \
  --security-opt=seccomp=unconfined --ipc=host \
  --ulimit memlock=-1:-1 --userns=keep-id --http-proxy=false \
  -v kyojin-models:/models -p 8000:8000 \
  -e KYOJIN_FLAVOR=qwen \
  -e KYOJIN_MODEL_REPO=yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz \
  kyojin:1.5.0 -- -c 262144
curl http://localhost:8000/v1/models
```

`--userns=keep-id` keeps files the server writes to the volume (tuning
cache, slots) owned by your user. `--http-proxy=false` stops podman from
copying host proxy variables into the container, which otherwise breaks the
model download when the proxy points at localhost.

Flags worth knowing: `--userns=keep-id` keeps files the server writes to the
volume (tuning cache, slots) owned by your user. `--http-proxy=false` stops
podman from copying host proxy variables into the container, which otherwise
breaks the model download when the proxy points at localhost. `--ipc=host`
with a raised memlock limit is the usual setup for GPU inference runtimes.

The entrypoint downloads the model into the volume on first start and skips
the download when a pack is already there. Both flavor and repo are required
and fail fast; there are no silent 50 GB pulls. Tuning cache
(`~/.cache/exllamav3`) lives under `HOME`, which the entrypoint points at
`/models/home` (override with `KYOJIN_HOME`) so a fresh container does not
re-tune. First start still tunes GEMM kernels (around 10 minutes with the
port closed), same as bare metal. The servers answer `/health` from the
first second (503 while loading, 200 when ready), so it works as a
readiness check, and `/v1/models` answers once serving.

Context size is the memory lever: 262144 was verified on all three packs on
an otherwise empty 128 GB box (Qwen kept 13.9 GiB free, GLM 10.4, MiMo 6.6).
Lower it when the machine is shared.

## Test notes

`checks.kyojin-tests` runs `tests/test_*_cpu.py` (95 green). Three tests in
`test_ablit_qwen_cpu.py` need `tools/ablit/*`, which was never committed
upstream; they fail on a fresh clone regardless of runner. The
`test_hip_compiler_cpu.py` fakes use POSIX `sh` (`/bin/bash` does not exist
in sandboxes).
