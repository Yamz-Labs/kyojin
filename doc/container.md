# Running Kyojin in a container (docker / podman)

One image per model: `kyojin-qwen`, `kyojin-glm`, `kyojin-mimo`. The three share every heavy layer (14.4 GB each on disk, one copy of the layers); only the last layer differs, and it only sets which defaults the image starts with. The model is never baked in: mount it on `/models`, or mount an empty folder and the image downloads the pack from the hub. The GPU is passed in as `/dev/kfd` and `/dev/dri`. Host requirements: a gfx1151 machine (Strix Halo, 128 GB) and the `amdgpu` kernel driver. No ROCm install is needed on the host.

Status of each line below: "tested" means run on a Strix Halo machine with docker 29 (Qwen and GLM packs on 13/10/2026 with Kyojin 1.5; MiMo pack on 06/10/2026; local copies). The podman line was run on 13/10/2026 with podman 4.9 (Qwen pack). "Untested" lines are marked.

## What the image contains

ROCm torch and `rocm-sdk-devel` from AMD's gfx1151 nightly index, the Kyojin tree with its extension built by `build.sh`, and the compilers the servers need at run time. It sets `EXL3_ROCM_SDK`, `EXL3_HSA_LIB` (the SDK's HSA runtime, there is no `/opt/rocm`) and `EXL3_HIPCC` (Qwen's gated-residual and QSA prefill kernels compile with it at first start). Nothing of this has to be set by you.

## Prebuilt images

Nothing to build: the images of Kyojin 1.7 are on the GitHub registry (about 3 GB to download, 11 GB on disk).

```bash
docker pull ghcr.io/yamz-labs/kyojin-qwen:1.7     # or podman pull
docker pull ghcr.io/yamz-labs/kyojin-glm:1.7
```

Then use `ghcr.io/yamz-labs/kyojin-qwen:1.7` in place of `kyojin-qwen` (or `localhost/kyojin-qwen`) in the run lines below. The MiMo image is not published: build it with the line below.

## Build (tested)

```bash
docker build -f docker/Dockerfile --build-arg MODEL=qwen -t kyojin-qwen .
docker build -f docker/Dockerfile --build-arg MODEL=glm  -t kyojin-glm  .   # cached heavy layer: seconds
docker build -f docker/Dockerfile --build-arg MODEL=mimo -t kyojin-mimo .
```

No GPU needed. First build: 12.5 minutes measured here (32 threads; about 14 GB of wheels; the extension alone takes 7). The other two tags take under a second: every layer but the last `MODEL` one is shared. A rebuild after a Python-only edit takes 3 s (only the last code layer, ~26 MB uncompressed, is re-pushed; users re-download MBs, not ~3 GB). A native-source edit (`exllamav3/exllamav3_ext/`, `setup.py`, `build.sh`, `tools/strix_halo/env.sh` or `rebuild.sh`) or a version bump (`exllamav3/version.py`) rebuilds the 136 MB extension layer, so a release re-downloads about 160 MB. The torch and SDK versions are pinned in the Dockerfile (`TORCH_VERSION`, `ROCM_SDK_VERSION`), so two builds carry the same versions until the pins are bumped deliberately. `podman build -f docker/Dockerfile` is untested.

## Run with docker (tested)

```bash
docker run --rm --name kyojin-qwen \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --ulimit memlock=-1:-1 \
  -v /path/to/Qwen3.8-Flash-Next-EXL3-Yamz:/models:ro \
  -v kyojin-cache:/root/.cache \
  -p 8731:8731 \
  kyojin-qwen
```

The image has no `video` or `render` group by name, so docker needs the numeric ids of the host. With no `-e` at all the server starts with 131072 tokens of context, the sampling of the model card and the port 8731. The port opens when the model is loaded and warmed up; then `curl http://localhost:8731/v1/models` and `python tools/bench.py --base http://localhost:8731/v1` work from the host.

Time to ready, measured on a Strix Halo machine with nothing else on the GPU, 147456 tokens of context, Qwen without vision:

| Start | 1.6 | 1.7 |
|---|---|---|
| Qwen pack, first start, empty cache | 493 s (kernels compiled, dense GEMM table tuned, both cached) | 67 s |
| Qwen pack, second start, same cache | 84 s | 68 s |
| GLM pack, first start, empty cache | 77 s, plus 28 s of kernel compiles inside the first request | 69 s, first requests without compiles (1.7.1; 1.7.0 still compiled about 5 s) |
| GLM pack, second start, same cache | 70 s | 67 s |

The Triton kernels and the dense GEMM tune table now ship with the code (see `EXL3_KERNEL_SEED`), so an empty cache no longer costs a compile or a tuning run. A restart with warm caches still reads the whole pack from the disk, so it is bound by the disk speed (about 2 GB/s: 45 to 65 s for a 72 to 89 GB pack). The MiMo image is not part of these figures: first start 130 s (1.5).

A chat completion and a tool call (`tools` in the request, the answer carries `tool_calls`) worked on Qwen and on GLM; MiMo was loaded and answered one completion. `docker stop` returns in 1 s (Qwen), 10 s (GLM), 2 s (MiMo); the exit code is 143 (SIGTERM) for Qwen and 0 for GLM and MiMo.

### Keep the cache

The cache holds compiled HIP kernels and the dense GEMM tuning table; the Triton kernels ship with the image and are unpacked into it at start. Keep it across restarts so that kernels and tune results found at run time are not recomputed. In order: a volume on `/root/.cache` (as above); else a hidden folder `.kyojin-cache` inside `/models` if it is mounted writable; `KYOJIN_CACHE_DIR=/some/folder` forces a folder.

### No pack yet

Mount an empty writable folder on `/models`. The image downloads the pack of its model from the hub (95 to 106 GB) into it, checks the free space first, resumes after an interrupt and logs the size. `-e KYOJIN_DOWNLOAD=0` forbids it, `-e KYOJIN_HF_REPO=owner/name` picks another repository. Tested here with the packs already on disk; the download path itself is covered by the CPU tests with a fake hub and by a real download of a tiny public repository, not by a 95 GB pull.

## Run as your own user, rootless podman (tested)

Podman with `--userns=keep-id` (or docker with `--user`) runs the server as your host user, who cannot write the image's `/root`. The entrypoint then uses a container-local home and prints `HOME not writable`. Put the cache on a folder you own:

```bash
docker run --rm --name kyojin-qwen --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --ulimit memlock=-1:-1 \
  -v /path/to/pack:/models:ro -v /path/to/cache:/cache -e KYOJIN_CACHE_DIR=/cache \
  -p 8731:8731 kyojin-qwen
```

Tested with docker and `--user 1000:1000`: ready in 392 s cold, 95 s warm from that cache. The podman line below is tested with rootless podman 4.9 on the Qwen pack (ready in 566 s cold, chat completions, clean stop in 1 s):

```bash
podman run --rm --replace --name kyojin-qwen --http-proxy=false --userns=keep-id \
  --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host --ulimit memlock=-1 \
  -v /path/to/pack:/models -v /path/to/cache:/cache -e KYOJIN_CACHE_DIR=/cache \
  -p 8731:8731 localhost/kyojin-qwen
```

`--http-proxy=false` stops podman from copying the host's `http_proxy` variables into the container, where they break calls to `127.0.0.1` (the health check and the hub download). With docker, do not pass proxy variables to the container.

## Behind llama-swap

The container port must be the port llama-swap picked (`-p ${PORT}:${PORT}`), so give the server `-e PORT=${PORT}` (`--port ${PORT}` after the image name works too). Arguments after the image name are appended to the server command line and win over the defaults, as in `-c 262144` below. Shape taken from a user's working setup, with the image's own entrypoint instead of a wrapper script. The `docker` form was run by hand with these arguments (port 8099, `-c 262144`, `--model-id`, `--default-reasoning-effort medium`, `--user`); llama-swap itself and podman were not run:

```yaml
  "Kyojin-Qwen3.8-Flash-Next":
    useModelName: "Kyojin-Qwen3.8-Flash-Next"
    cmd: |
      podman run
      --rm --replace --name kyojin-qwen
      --http-proxy=false --userns=keep-id
      --device /dev/kfd --device /dev/dri --group-add keep-groups
      --ipc=host --ulimit memlock=-1
      -p 0.0.0.0:${PORT}:${PORT}
      -e PORT=${PORT}
      -v /path/to/Qwen3.8-Flash-Next-EXL3-Yamz:/models
      -v /path/to/kyojin-cache:/cache -e KYOJIN_CACHE_DIR=/cache
      localhost/kyojin-qwen
      -c 262144
      --default-reasoning-effort medium
      --model-id Kyojin-Qwen3.8-Flash-Next
    checkEndpoint: /v1/models
    cmdStop: podman stop -t 30 kyojin-qwen
    unloadTimeout: 120
    healthCheckTimeout: 900
    filters:
      setParamsByID:
        "${MODEL_ID}-Instruct":
          chat_template_kwargs:
            enable_thinking: false
```

- `healthCheckTimeout`: llama-swap waits this long for `checkEndpoint`. Qwen needs about 600 s the first time and 95 s afterwards, GLM about 95 s the first time and 65 s afterwards. Set 900 for Qwen and MiMo, 300 for GLM, and warm the cache by hand once so that later starts are short.
- `checkEndpoint: /v1/models` answers once the model is loaded.
- `cmdStop`: SIGTERM stops the server in 1 to 7 s here; `-t 30` leaves room for a slower free of the GPU.
- Context: `-c 262144` ran on the Qwen pack. GLM at 262144 is not tested. MiMo's memory is tight at 262144 on 128 GB and is not tested; its tested value is the default 131072.
- GLM and MiMo entries differ in the image name, the model id and `-c`.

## Variables (all optional; the tested runs above used the defaults)

| Variable | Default | Meaning |
|---|---|---|
| `KYOJIN_MODEL` | set by the image | `qwen`, `glm`, `mimo` |
| `MODEL_DIR` | `/models` | The pack folder, or a folder holding one pack of the family |
| `PORT` | `8731` | Listen port (also used by the health check) |
| `HOST` | `0.0.0.0` | Listen address inside the container |
| `CONTEXT` | `131072` | KV cache size in tokens (`-c`) |
| `KYOJIN_MODEL_ID` | server default | Id shown by `/v1/models` (or `--model-id`) |
| `KYOJIN_DEFAULT_TEMPERATURE`, `_TOP_P`, `_TOP_K`, `_MIN_P` | model card | Sampling when a request omits it (`TOP_K`, `MIN_P`: Qwen only) |
| `KYOJIN_REASONING_EFFORT` | server default | `low`, `medium`, `xhigh` (Qwen, GLM) |
| `KYOJIN_DRAFT_TOKENS`, `KYOJIN_DRAFT_POLICY` | server default | Speculative decoding settings |
| `KYOJIN_ENABLE_VISION`, `_THINKING`, `_UNCENSOR` | on | `0` switches the feature off (Qwen; uncensor also GLM) |
| `KYOJIN_CACHE_DIR` | see "Keep the cache" | Folder for the kernel cache |
| `KYOJIN_STOP_TIMEOUT` | `20` | Seconds after SIGTERM before a server that is still running gets SIGKILL (keep it below `docker stop -t`) |
| `KYOJIN_DOWNLOAD`, `KYOJIN_HF_REPO` | on, the model's repo | Pack download control |
| `KYOJIN_EXTRA_ARGS` | empty | Extra server arguments, shell-quoted |
| `EXL3_*` | unset | Engine variables pass through unchanged |

`docker run ... IMAGE shell` opens a shell and `... IMAGE check` runs `tools/strix_halo/env.sh --check`. An option the chosen server does not have is an error at start. A wrong-family pack on `/models` is a clear error that exits 2.

## Tests that need no GPU

`docker/test_kyojin_args.py` (60 tests, `python -m pytest docker/test_kyojin_args.py`, needs pytest on the host) and `docker/test_entrypoint.sh` (every branch of the entrypoint with a stub server; also runs inside the image: `docker run --rm --entrypoint bash IMAGE docker/test_entrypoint.sh`, the two permission tests skip as root). Both passed.

## Notes

- One container, one model, one GPU: two models do not fit in 128 GB together.
- Speed: the same request gives the same speed in the container and on the host (Qwen pack, 9.4K-token prompt, 400 generated tokens, same engine tree and flags, 6 runs each): prefill 1312 tok/s in the container and 1318 on the host, speculative decode 42.6 and 42.4 tok/s. The generated text is identical.
- `--ipc=host` and `--ulimit memlock` are the flags of the tested runs. No `seccomp=unconfined` and no `label=disable` were needed with docker. SELinux hosts (Fedora, with podman) may need `--security-opt label=disable`: untested.
