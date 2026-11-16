"""Shared host-floor policy for the preflight and the external supervisor.

This module is stdlib-only: ``exllamav3/guard.py`` imports it as a script sibling so the
supervisor can start without importing torch or the package.
"""

from __future__ import annotations

import os

DEFAULT_HOST_FLOOR_GIB = 8.0


def host_floor_from_env() -> float:
    """The one shared host `MemAvailable` floor, in GiB.

    `MEMGUARD_FLOOR` is the established name, so existing lanes keep working.  The
    preflight and supervisor both read this value; they never carry separate defaults.
    """
    value = os.environ.get("MEMGUARD_FLOOR")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
    return DEFAULT_HOST_FLOOR_GIB
