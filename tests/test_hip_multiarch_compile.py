"""Regression guard: the extension's ROCm TUs must compile for every architecture the repo supports.

`rocm/routed_f16/routed_f16.cu` is the canary. Its kernels are gfx1151-only behind
`#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)`, and (until the fix in #<PR>) its host
wrappers launched them from *outside* that guard: any other architecture's device pass had no kernels
but still had to emit the launch stubs, so a build for gfx1201 (or any multi-arch string containing it)
failed with "use of undeclared identifier 'RoutedPadBoundsKernel'". It went unnoticed because the box
only built gfx1151 until a multi-arch build was needed (2026-11-16).

A full multi-arch wheel build takes minutes; building the one translation unit through torch's own
`cpp_extension` path (hipify + ninja, i.e. the exact pipeline `uv pip install .` uses) takes seconds and
reproduces the failure exactly.

Runs where ROCm is available and skips cleanly everywhere else (CUDA builds, CPU dev boxes, CI without
a ROCm SDK/ninja). Requires what the repo's own build requires: `torch.version.hip`, a HIP SDK for
torch's cpp_extension (`ROCM_HOME`/`EXL3_ROCM_SDK`, see tools/strix_halo/env.sh), and ninja.

    pytest tests/test_hip_multiarch_compile.py
    EXL3_TEST_ROCM_ARCHS="gfx1151 gfx1200" pytest tests/test_hip_multiarch_compile.py
    EXL3_SKIP_BUILD_SMOKE=1 pytest ...        # opt out (e.g. a ROCm box on a tight time budget)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TU = ROOT / "exllamav3" / "exllamav3_ext" / "rocm" / "routed_f16" / "routed_f16.cu"
DEFAULT_ARCHS = ("gfx1151", "gfx1201")      # the guarded arch, and the arch whose absence broke the build


def _archs() -> list[str]:
    raw = os.environ.get("EXL3_TEST_ROCM_ARCHS", "").strip()
    return raw.replace(",", " ").split() if raw else list(DEFAULT_ARCHS)


def _skip_reason() -> str | None:
    if os.environ.get("EXL3_SKIP_BUILD_SMOKE", "") == "1":
        return "EXL3_SKIP_BUILD_SMOKE=1"
    try:
        import torch
    except Exception as ex:                                  # noqa: BLE001 - torch is a test dependency
        return f"torch unavailable ({ex!r})"
    if not torch.version.hip:
        return "not a ROCm torch build (CUDA/CPU builds do not compile rocm/ TUs)"
    from torch.utils import cpp_extension
    if cpp_extension.ROCM_HOME is None:
        return ("no ROCm SDK for torch's cpp_extension (set ROCM_HOME/EXL3_ROCM_SDK, "
                "see tools/strix_halo/env.sh)")
    if shutil.which("ninja") is None:
        return "ninja not on PATH (torch.utils.cpp_extension uses it for the TU build)"
    if not TU.is_file():
        return f"missing {TU.relative_to(ROOT)}"
    return None


_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=f"build smoke skipped: {_REASON}")


@pytest.mark.parametrize("arch", _archs())
def test_tu_builds_for_arch(arch: str, tmp_path, monkeypatch):
    """Build this TU for one offload arch through torch's cpp_extension pipeline (hipify + ninja).

    `load()` compiles for the arches in PYTORCH_ROCM_ARCH; one call per arch keeps the cache keys and the
    build directories separate. A failure here is a build regression, so the compiler output is attached.
    """
    from torch.utils.cpp_extension import load

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", arch)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    build_dir = tmp_path / "build"
    try:
        load(name="tu_smoke_" + arch.replace(":", "_"), sources=[str(TU)], is_python_module=False,
             with_cuda=True, extra_cflags=["-DUSE_ROCM"], extra_cuda_cflags=[], build_directory=str(build_dir),
             verbose=False)
    except Exception as ex:                                  # noqa: BLE001 - report the compiler's message
        errs = [ln for ln in str(ex).splitlines() if "error:" in ln][:6]
        pytest.fail(f"{TU.name} does not build for {arch}:\n" + "\n".join(errs or [str(ex)[-1500:]]))
