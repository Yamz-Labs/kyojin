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
nix build .#exllamav3      # engine wheel + gfx1151 extension
nix build .#exllama-ext    # native extension alone (narrowed sources)
nix build .#kyojin         # serve wrappers (qwen/glm/mimo)
nix build .#torch-rocm     # ROCm torch wheel
nix build .#rocm-sdk       # expanded SDK tree (headers, hipcc, libhsa)
nix build .#docker         # OCI image tarball, about 5 GB compressed
nix run .#kyojin-serve-glm -- --help
```

The engine package is split for CI caching: `.#exllama-ext` compiles the
native extension from narrowed sources only (`setup.py`,
`exllamav3/exllamav3_ext/`, `util/arch_list.py`, plus packaging metadata),
while `.#exllamav3` builds the pure-python wheel (`EXLLAMA_NOCOMPILE`,
seconds) with the cached `.so` injected into site-packages — same module,
same top-level import path, so `exllamav3.ext` finds it via
`is_precompiled_extension_available()` exactly as before. `nix flake check`
(hook suite + CPU test subset, `checks.kyojin-tests`, GPUs hidden, bounded
runtime) therefore never compiles the extension unless native sources
changed: python/docs/tests PRs reuse the cached `.so` through
magic-nix-cache and finish in minutes, while native changes still rebuild
and test it. The serve wrappers, the venv, and the container all consume
the same package, so nothing drifts. First build compiles the extension
once (8-10 min); afterwards only native changes pay that again.
Only these paths invalidate the cached extension: `setup.py`,
`exllamav3/exllamav3_ext/`, `exllamav3/util/arch_list.py`,
`exllamav3/version.py`, `pyproject.toml`, `README.md`. Every other change
(python, tests, tools, docs) reuses the cached `.so`.

`nix fmt` doubles as the clean-tree gate: it exits non-zero when a hook
fails or rewrites files.

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

To bump: update the pins in the `rocm` extra in `pyproject.toml`, re-run
`uv lock`, and re-verify `nix build .#exllamav3`. The AMD index serves
bare links with no hashes, so the cp312/linux artifacts carry
hand-backfilled `hash = "sha256:..."` entries in `uv.lock` (convert with
`nix hash convert --hash-algo sha256 --from nix32 --to base16`); if a
re-lock drops them, Nix evaluation fails loudly asking for a sha256.
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
  kyojin:1.5.8 -- -c 262144
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
and fail fast; there are no silent 50 GB pulls. Each flavor gets its own slot
(`/models/qwen`, `/models/glm`, `/models/mimo`, overridable with
`KYOJIN_MODEL_DIR`), and a marker file records which repo filled it, so
switching repos under the same flavor errors out instead of serving a stale
pack. Tuning cache (`~/.cache/exllamav3`) lives under `HOME`, which the
entrypoint points at `/models/home` (override with `KYOJIN_HOME`) so a fresh
container does not re-tune. First start still tunes GEMM kernels (around
10 minutes with the port closed), same as bare metal. The servers answer
`/health` from the
first second (503 while loading, 200 when ready), so it works as a
readiness check, and `/v1/models` answers once serving.

Context size is the memory lever: 262144 was verified on all three packs on
an otherwise empty 128 GB box (Qwen kept 13.9 GiB free, GLM 10.4, MiMo 6.6).
Lower it when the machine is shared.

## Benchmark

`tools/bench.py` runs unmodified against any running server (stdlib only).
From a checkout the default corpus is the repo's own Markdown, so plain
`nix develop --command python3 tools/bench.py` just works when the server
is local. Against a remote box, tunnel it and bring a corpus:

```bash
ssh -L 8000:localhost:8000 <server>   # keep this open in another shell
nix run .#kyojin-bench -- --base http://localhost:8000/v1 \
  --corpus <any 8000+ word text file> --json
```

For comparable numbers across machines, freeze the input with the pinned
corpus (same revision in, same bytes out) instead of a hand-picked file:

```bash
nix run .#kyojin-bench -- --base http://localhost:8000/v1 \
  --corpus $(nix build --print-out-paths .#bench-corpus)/corpus.txt --json
```

It prints a Markdown block ready to paste into a benchmark report issue,
plus a fenced JSON block with `--json` for collecting tables.

## Test notes

`checks.kyojin-tests` runs `tests/test_*_cpu.py` (95 green). Three tests in
`test_ablit_qwen_cpu.py` need `tools/ablit/*`, which was never committed
upstream; they fail on a fresh clone regardless of runner. The
`test_hip_compiler_cpu.py` fakes use POSIX `sh` (`/bin/bash` does not exist
in sandboxes).
