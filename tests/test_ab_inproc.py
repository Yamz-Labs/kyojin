"""tools/ab/ab_inproc.py against a fake server: switch order, warm-up per switch, per-arm md5 and medians."""
import asyncio
import importlib.util
import json
import threading
from pathlib import Path

from aiohttp import web

SPEC = importlib.util.spec_from_file_location("ab_inproc", Path(__file__).resolve().parent.parent / "tools" / "ab" / "ab_inproc.py")
ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ab)


def _fake(state):
    async def models(_):
        return web.json_response({"data": [{"id": "m"}]})

    async def knobs(r):
        state["knobs"] = {k: v for k, v in (await r.json()).items() if v is not None}
        state["log"].append(("knobs", dict(state["knobs"])))
        return web.json_response({"effective": state["knobs"]})

    async def chat(r):
        body = await r.json()
        state["log"].append(("chat", body["max_tokens"]))
        state.setdefault("salts", []).append(body.get("cache_salt"))
        word = "B" if state["knobs"].get("K") == "1" else "A"          # the knob changes the bytes
        resp = web.StreamResponse(headers = {"Content-Type": "text/event-stream"})
        await resp.prepare(r)
        for i in range(min(body["max_tokens"], 12)):
            await asyncio.sleep(0.002)
            await resp.write(b"data: " + json.dumps({"choices": [{"delta": {"content": word}}]}).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/admin/knobs", knobs)
    app.router.add_post("/v1/chat/completions", chat)
    return app


def test_interleaved_run(capsys, tmp_path):
    state = {"knobs": {}, "log": []}
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(_fake(state))
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]
    threading.Thread(target = loop.run_forever, daemon = True).start()
    out = tmp_path / "r.json"
    import sys
    sys.argv = ["ab_inproc", str(port), "--arm", "A:K=0", "--arm", "B:K=1", "--order", "ABBA", "--prompts", str(tmp_path / "p.txt"),
                "--json", str(out), "--max-tokens", "10", "--warm-tokens", "3"]
    (tmp_path / "p.txt").write_text("hello\n=====\nworld\n")
    ab.main()
    loop.call_soon_threadsafe(loop.stop)
    res = json.loads(out.read_text())
    assert res["order"] == "ABBA"
    assert res["arms"]["A"]["md5_stable"] and res["arms"]["B"]["md5_stable"]
    assert res["arms"]["A"]["md5"] != res["arms"]["B"]["md5"] and len(res["arms"]["A"]["md5"]) == 2
    log = state["log"]
    ks = [e for e in log if e[0] == "knobs"]
    assert [e[1].get("K") for e in ks[:4]] == ["0", "1", "1", "0"] and ks[-1][1] == {}           # reset to defaults at the end
    first = log.index(ks[0])
    assert log[first + 1] == ("chat", 3)                                                   # warm-up right after the switch
    assert "IDENTICAL" not in capsys.readouterr().out
    salts = state["salts"]
    assert all(salts) and len(set(salts)) == 4 and salts[0] == salts[1]               # one salt per visit, warm-up inside it
