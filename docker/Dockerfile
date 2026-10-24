# syntax=docker/dockerfile:1
# Kyojin server image for AMD Strix Halo (gfx1151). Works with docker and podman.
#   docker build -f docker/Dockerfile -t kyojin-qwen .                       # default model: qwen
#   docker build -f docker/Dockerfile --build-arg MODEL=glm  -t kyojin-glm .  # same layers, other baked defaults
#   docker build -f docker/Dockerfile --build-arg MODEL=mimo -t kyojin-mimo .
# Follows the README Quickstart: ROCm torch + rocm-sdk-devel from AMD's gfx1151 index, requirements.txt,
# `rocm-sdk init`, tools/strix_halo/env.sh, build.sh. The models are NOT in the image: mount them.
#
# Layering (single stage, heaviest first so upgrades stay small):
#   A: OS + venv + pinned torch/rocm-sdk-devel + requirements.txt + `rocm-sdk init` + SDK prune.
#      Depends only on requirements.txt and the TORCH_/ROCM_ ARGs below. Python-source edits never touch it.
#   B: native extension only (setup.py/build.sh/pyproject.toml, exllamav3_ext sources, the two files
#      setup.py loads by path, build scripts). Rebuilds (~10 min) only when native sources change.
#   C: the rest of the tree (python, tools, docker/). Python-only releases change only this layer (MBs).
#   MODEL stays last: qwen/glm/mimo share every other layer.

FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
# Pinned nightly versions (same set as the tested 1.5 images): bump deliberately, then re-verify
# `docker run --rm --entrypoint python IMG -c "import torch, exllamav3_ext"`.
ARG TORCH_VERSION=2.12.0a0+rocm7.13.0a20260411
ARG ROCM_SDK_VERSION=7.14.0a20260612
ARG ROCM_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx1151/
# Single stage on purpose: a second stage would copy the ~14 GB tree again and double the disk peak.
# g++, libc6-dev, python3-dev stay: the servers compile a few HIP kernels (hipcc) and Triton kernels at start.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev gcc g++ libc6-dev make git ca-certificates \
        libnuma1 libatomic1 libgomp1 libdrm2 libdrm-amdgpu1 libelf1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/kyojin
RUN python3 -m venv .venv
ENV PATH=/opt/kyojin/.venv/bin:$PATH
# --- Layer A: heavy dependencies. Only requirements.txt and the version ARGs invalidate it. ---
# pip downloads live in a BuildKit cache mount, never in an image layer; the unpruned SDK tree is pruned
# in the same RUN. The SDK (hipcc, headers) stays, pruned of test binaries, profiler data and client samples.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --pre "torch==${TORCH_VERSION}" "rocm-sdk-devel==${ROCM_SDK_VERSION}" --index-url "${ROCM_INDEX_URL}" \
    && pip install -r requirements.txt \
    && rocm-sdk init \
    && export EXL3_ROCM_SDK=$(rocm-sdk path --root) \
    && cd "$EXL3_ROCM_SDK" && rm -rf clients share/rocprofiler-sdk share/hipblaslt share/rocdecode share/rocsparse bin/*gtest* bin/*-test bin/*-bench bin/*-gemm-tune bin/*test* \
    && rm -rf /tmp/*
ENV EXL3_ROCM_SDK=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel \
    EXL3_HIPCC=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/hipcc \
    EXL3_HSA_LIB=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1
# --- Layer B: native extension. Only native sources and build scripts invalidate it. ---
# setup.py loads exllamav3/exllamav3_ext/build_config.py and exllamav3/util/arch_list.py by file path
# (no top-level exllamav3 import) and reads the version from exllamav3/version.py; build.sh runs
# tools/strix_halo/rebuild.sh, which sources tools/strix_halo/env.sh. Nothing else is read at build time.
COPY setup.py build.sh pyproject.toml ./
COPY exllamav3/exllamav3_ext/ ./exllamav3/exllamav3_ext/
COPY exllamav3/version.py ./exllamav3/version.py
COPY exllamav3/__init__.py ./exllamav3/__init__.py
COPY exllamav3/util/arch_list.py ./exllamav3/util/arch_list.py
COPY exllamav3/util/__init__.py ./exllamav3/util/__init__.py
COPY tools/strix_halo/env.sh tools/strix_halo/rebuild.sh ./tools/strix_halo/
RUN --mount=type=cache,target=/root/.cache/pip \
    source tools/strix_halo/env.sh \
    && ./build.sh \
    && python -c "import torch, exllamav3_ext" \
    && rm -rf build .rebuild.* .sdk-deps /tmp/*
# --- Layer C: everything else (python, tools, docker/). Python-only edits land here. ---
COPY . ./
RUN python -c "import torch, exllamav3_ext"
# Last on purpose: the only layer that differs between the qwen, glm and mimo images. Defaults per model live in
# docker/kyojin_args.py (context, sampling, hub repository); MODEL only picks which set the image starts with.
ARG MODEL=qwen
ENV KYOJIN_MODEL=$MODEL MODEL_DIR=/models PORT=8731 HOME=/root
EXPOSE 8731
VOLUME ["/models"]
HEALTHCHECK --start-period=900s --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf "http://127.0.0.1:${KYOJIN_PORT:-$PORT}/health" >/dev/null || exit 1
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/kyojin/docker/entrypoint.sh"]
