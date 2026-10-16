# Kyojin server image for AMD Strix Halo (gfx1151). Works with docker and podman.
#   docker build -f docker/Dockerfile -t kyojin-qwen .                       # default model: qwen
#   docker build -f docker/Dockerfile --build-arg MODEL=glm  -t kyojin-glm .  # same layers, other baked defaults
#   docker build -f docker/Dockerfile --build-arg MODEL=mimo -t kyojin-mimo .
# Follows the README Quickstart: ROCm torch + rocm-sdk-devel from AMD's gfx1151 index, requirements.txt,
# `rocm-sdk init`, tools/strix_halo/env.sh, build.sh. The models are NOT in the image: mount them.

FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# Single stage on purpose: a second stage would copy the ~14 GB tree again and double the disk peak.
# g++, libc6-dev, python3-dev stay: the servers compile a few HIP kernels (hipcc) and Triton kernels at start.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev gcc g++ libc6-dev make git ca-certificates \
        libnuma1 libatomic1 libgomp1 libdrm2 libdrm-amdgpu1 libelf1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/kyojin
RUN python3 -m venv .venv
ENV PATH=/opt/kyojin/.venv/bin:$PATH
# Layers run from least to most often changed: dependencies, then the HIP extension, then the Python source.
# Editing Python code (or version.py, which CI stamps from the git tag) only re-runs the final COPY.

# Dependencies. One layer on purpose: the pip downloads and the unpruned SDK tree never reach an image layer.
# The SDK (hipcc, headers) stays, pruned of test binaries, profiler data and client samples.
COPY requirements.txt .
RUN pip install --pre torch rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    && pip install -r requirements.txt \
    && rocm-sdk init \
    && cd "$(rocm-sdk path --root)" && rm -rf clients share/rocprofiler-sdk share/hipblaslt share/rocdecode share/rocsparse bin/*gtest* bin/*-test bin/*-bench bin/*-gemm-tune bin/*test* \
    && rm -rf /root/.cache

# The HIP extension, built from only the files the compile reads, so it re-runs only when they change.
# rebuild.sh pip-installs the package to a scratch dir and keeps just exllamav3_ext*.so in /opt/kyojin, so the
# README and version.py that the package metadata wants are stubs here; the real ones arrive with the source below.
COPY pyproject.toml setup.py build.sh ./
COPY tools/strix_halo/env.sh tools/strix_halo/rebuild.sh tools/strix_halo/
COPY exllamav3/util/arch_list.py exllamav3/util/
COPY exllamav3/exllamav3_ext/ exllamav3/exllamav3_ext/
RUN touch README.md && printf '__version__ = "0"' > exllamav3/version.py \
    && export EXL3_ROCM_SDK=$(rocm-sdk path --root) \
    && source tools/strix_halo/env.sh \
    && ./build.sh \
    && python -c "import torch, exllamav3_ext" \
    && rm -rf build .rebuild.* .sdk-deps *.egg-info README.md exllamav3/version.py /root/.cache

# The Python source.
COPY . .
ENV EXL3_ROCM_SDK=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel \
    EXL3_HIPCC=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/hipcc \
    EXL3_HSA_LIB=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1
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
