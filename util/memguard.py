#!/usr/bin/env python3
"""Compatibility wrapper for the package guard entry point.

The supervisor lives in ``exllamav3/guard.py``.  Exec the file instead of importing
it: importing the package would initialize torch and apply allocator settings in a
process that must stay tiny and allocation-free.
"""

import os
import sys

_TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "exllamav3", "guard.py")

os.execv(sys.executable, [sys.executable, _TARGET, *sys.argv[1:]])
