"""What this machine can run the tests on, so that a test needing a GPU or the built extension skips instead of failing.

Three needs:
  gpu     a CUDA or ROCm device (torch.cuda.is_available())
  ext     `import exllamav3` works: it loads the native extension at import time, so it needs the extension built by
          ./build.sh (or a GPU to JIT-build it) and Triton
  triton  Triton alone (CPU tests that stub the extension but import modules that import Triton)

Test functions take the marks (@pytest.mark.gpu, .ext, .triton; tests/conftest.py skips them). A file that cannot even be
imported without them calls require("ext") or require("gpu", "ext") before the import that needs it.

EXL3_TESTS_STRICT=1 turns every such skip off: on the GPU machine, a broken build or a hidden device then fails the tests
instead of skipping them silently."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys

import torch

STRICT = os.environ.get("EXL3_TESTS_STRICT") == "1"


def has_gpu() -> bool:
    return torch.cuda.is_available()


def has_ext() -> bool:
    """Never imports exllamav3.ext itself: without a prebuilt extension that starts a JIT build (and takes its lock).
    Already in sys.modules (loaded, or a test's stub while it imports): the import goes through."""
    if "exllamav3.ext" in sys.modules:
        return True
    spec = importlib.util.find_spec("exllamav3_ext")
    prebuilt = bool(spec and spec.origin and spec.origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)))
    return (prebuilt or has_gpu()) and has_triton()


def has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


CHECKS = {"gpu": has_gpu, "ext": has_ext, "triton": has_triton}
REASONS = {"gpu": "needs a GPU", "ext": "needs the built extension (./build.sh) and Triton", "triton": "needs Triton"}


def missing(need: str) -> bool:
    return not STRICT and not CHECKS[need]()


def require(*needs: str) -> None:
    """Skip the whole file when one of `needs` is missing. Call it before the import that needs it. Without pytest (a test
    file run as a script in a venv that has none) it does nothing, and the import fails as it did before."""
    try:
        import pytest
    except ImportError:
        return
    caller = os.path.basename(sys._getframe(1).f_code.co_filename)     # -rs reports this file's line, not the caller's
    for need in needs:
        if missing(need):
            pytest.skip(f"{caller}: {REASONS[need]}", allow_module_level = True)
