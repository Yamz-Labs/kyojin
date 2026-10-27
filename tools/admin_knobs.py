"""POST /admin/knobs for tools/qwen/serve.py and tools/glm/serve.py (opt-in: --admin-knobs).

Body {"EXL3_X": "1", "EXL3_Y": null, ...} sets (or, with null, unsets) A/B knobs of exllamav3.util.knobs in the running server, waits
until no request is running, then applies them on the engine's driver thread (captured graphs and per-shape caches are dropped there)
and answers {"effective": {every knob: value}, "prefix_cache_reset": bool}. A value that really changed also resets the engine's prefix cache
(disk copies included): prompts cached under the old value must not feed the new one. GET answers the effective values. Loopback clients only, whatever --host says.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
from typing import Any, AsyncContextManager, Callable

from aiohttp import web


def _loopback(request: web.Request) -> bool:
    if request.headers.get("X-Forwarded-For") or request.headers.get("Forwarded"):
        return False                                   # behind a proxy: the peer is the proxy, not the client
    try:
        return ipaddress.ip_address(request.remote or "").is_loopback
    except ValueError:
        return False


def _text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float, str)):
        return str(v)
    raise ValueError("knob values must be strings, numbers, booleans or null")


def make_handler(engine: Any, exclusive: Callable[[], AsyncContextManager]) -> Callable:
    """exclusive(): async context manager inside which no request runs (the servers' slot-action drain)."""
    from exllamav3.util import knobs

    def err(status: int, message: str) -> web.Response:
        return web.json_response({"error": {"code": status, "message": message, "type": "invalid_request_error"}}, status=status)

    async def handler(request: web.Request) -> web.Response:
        if not _loopback(request):
            return err(403, "/admin/knobs answers loopback clients only")
        if request.method == "GET":
            return web.json_response({"effective": knobs.effective()})
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object {knob: value}")
            values = {k: _text(v) for k, v in body.items()}
            unknown = [k for k in values if k not in knobs.KNOBS]
            if unknown:
                raise ValueError("unknown knob(s): " + ", ".join(sorted(unknown)))
        except (ValueError, json.JSONDecodeError) as exc:
            return err(400, str(exc))
        async with exclusive():
            before = knobs.effective()
            if hasattr(engine, "run_exclusive"):
                effective = await engine.run_exclusive(knobs.set_knobs, values)
            else:
                effective = await asyncio.get_running_loop().run_in_executor(None, knobs.set_knobs, values)
            # a changed knob may change the bits a prefill writes: the cached prompt prefixes (and their disk copies) belong to the old value
            reset = effective != before and hasattr(engine, "reset_prefix_cache")
            if reset:
                await engine.reset_prefix_cache()
        return web.json_response({"effective": effective, "prefix_cache_reset": bool(reset)})

    return handler
