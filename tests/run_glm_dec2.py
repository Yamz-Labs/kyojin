#!/usr/bin/env python
"""Run tests/test_glm_dec2.py when pytest is not installed in the venv.

Same reason as tests/run_glm_gates.py: the only pytest on this box lives in
~/.local/lib/python3.12/site-packages, and putting that directory on PYTHONPATH breaks the ROCm
library resolution of the extension (ImportError: libamdhip64.so.7). Importing torch and the
extension first, then adding the user-site directory, keeps both working.

    PYTHONPATH=<repo> python tests/run_glm_dec2.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("EXL3_REPO", os.path.dirname(_HERE)))

import torch  # noqa: E402,F401
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402,F401

sys.path.append(os.path.expanduser("~/.local/lib/python3.12/site-packages"))
import pytest  # noqa: E402

sys.exit(pytest.main([os.path.join(_HERE, "test_glm_dec2.py"), "-s", "-q"] + sys.argv[1:]))
