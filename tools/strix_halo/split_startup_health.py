#!/usr/bin/env python3
"""Poll a starting server's /health and record the load-stage sequence.

The split load is the long pole, so the stage trace is the evidence that the
server's early listener reports load progress.  Start the server separately, then
run this against its port while it loads:

    python tools/glm/serve.py ... --port 8299 > /tmp/serve.log 2>&1 &
    python tools/strix_halo/split_startup_health.py --port 8299 \
        --expect "target weights,drafter weights,vision tower,warm-up" \
        --out /tmp/health-vision.json

With --no-vision the expected list is "target weights,drafter weights,warm-up".
Exits non-zero if the server never answers, never finishes, or a requested stage
never appears.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

# A dedicated direct opener: the localhost poll must not honor http_proxy/https_proxy,
# which would let an environment proxy intercept or spoof the health trace.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read_health(port: int):
    url = f"http://127.0.0.1:{port}/health"
    try:
        with _OPENER.open(url, timeout = 2) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        # The early listener answers 503 with a JSON "loading" body; urlopen raises on
        # that status, so read it explicitly instead of treating the poll as unreachable.
        body = exc.read()
        if exc.code == 503 and body:
            return exc.code, json.loads(body)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type = int, default = 8299)
    ap.add_argument("--timeout", type = float, default = 900.0)
    ap.add_argument("--interval", type = float, default = 0.25)
    ap.add_argument("--expect", default = None,
                    help = "comma-separated stages that must appear in the trace")
    ap.add_argument("--out", default = None)
    args = ap.parse_args()

    expected = [s.strip() for s in (args.expect or "").split(",") if s.strip()]
    t0 = time.perf_counter()
    trace = []
    last_stage = object()
    final = None
    while time.perf_counter() - t0 < args.timeout:
        try:
            status, body = read_health(args.port)
        except Exception as exc:                    # noqa: BLE001 - the listener may not be up yet
            if trace:
                trace.append({"t_s": round(time.perf_counter() - t0, 2),
                              "stage": None, "status": f"unreachable: {type(exc).__name__}"})
            time.sleep(args.interval)
            continue
        stage = body.get("stage")
        if stage != last_stage:
            trace.append({
                "t_s": round(time.perf_counter() - t0, 2),
                "stage": stage,
                "stage_index": body.get("stage_index"),
                "stage_count": body.get("stage_count"),
                "progress": body.get("progress"),
                "status": body.get("status"),
            })
            last_stage = stage
        if body.get("status") != "loading":
            final = body
            break
        time.sleep(args.interval)

    elapsed = round(time.perf_counter() - t0, 2)
    stages = [e["stage"] for e in trace if e.get("stage")]
    print(f"[health] {elapsed} s, stages: {stages}", flush = True)
    if final is None:
        print("[health] server did not finish loading", flush = True)
    else:
        print(f"[health] final: status={final.get('status')} stage={final.get('stage')}", flush = True)
    missing = [s for s in expected if s not in stages]
    if args.out:
        with open(args.out, "w", encoding = "utf8") as f:
            json.dump({"elapsed_s": elapsed, "stages": stages, "trace": trace,
                       "final": final, "expected": expected, "missing": missing}, f, indent = 1)
        print(f"[health] wrote {args.out}", flush = True)
    if final is None or final.get("status") != "ok":
        status = final.get("status") if final else None
        raise SystemExit(f"health trace finished with status {status!r}")
    if missing:
        raise SystemExit(f"health trace missing stage(s): {missing}")


if __name__ == "__main__":
    main()
