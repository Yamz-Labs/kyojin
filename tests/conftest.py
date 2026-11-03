import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # tests/hardware.py, whatever the --import-mode
import hardware  # noqa: E402


def pytest_configure(config):
    for need, reason in hardware.REASONS.items():
        config.addinivalue_line("markers", f"{need}: {reason}; skipped without it (tests/hardware.py, EXL3_TESTS_STRICT=1 runs it anyway)")


def pytest_collection_modifyitems(config, items):
    for need, reason in hardware.REASONS.items():
        if hardware.missing(need):
            for item in items:
                if item.get_closest_marker(need):
                    item.add_marker(pytest.mark.skip(reason = reason))


@pytest.fixture(autouse = True)
def _restore_current_cuda_device():
    # Several test modules pin tensors to "cuda:0" but launch Triton kernels on the *current* device,
    # while others call torch.cuda.set_device() for their own device and never restore it. Reset the
    # current device after every test so the suite's result doesn't depend on collection order
    yield
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
