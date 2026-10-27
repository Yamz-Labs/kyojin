"""One arm of the cold-vs-tuned A/B: serve the pack with a FRESH tune cache, with or without the pack's shipped tuning, and time the
start-up and the first requests (where an untuned shape pays its timing) against the steady state. Run under the GPU lock by the caller.

    python ab_autotune.py {cold|tuned} PACK OUT.json [--repo DIR] [--ctx 16384] [--sessions 4] [--port 8591]

cold  : EXL3_PACK_TUNE=0, empty EXLLAMAV3_TUNE_CACHE  -> every shape is timed at first use (what a new model does today)
tuned : same empty live cache, pack tuning merged at serve start (qualify --stage autotune output)
Both arms share everything else (flags, prompts, order). The A/B shares nothing through ~/.cache (dense20: an A/B must not share one tune cache).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backends  # noqa: E402
import load  # noqa: E402
import pack as packmod  # noqa: E402
from server import oom_in, running_server  # noqa: E402


def records_in(d: Path) -> int:
    p = d / "coop_autotune_v1.bin"
    return max(0, (p.stat().st_size - 16) // 32) if p.is_file() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["cold", "tuned"])
    ap.add_argument("pack")
    ap.add_argument("out")
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--repo", default=str(HERE.parents[1]))
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--port", type=int, default=8591)
    ap.add_argument("--draft", help="override the derived --draft-policy (off = plain decode)")
    a = ap.parse_args()
    meta = packmod.read_pack(a.pack)
    prof = backends.profile(a.backend, a.repo)
    cache = Path(tempfile.mkdtemp(prefix=f"at0_{a.arm}_"))
    prof["env"] = {**prof["env"], "EXLLAMAV3_TUNE_CACHE": str(cache), "EXL3_PACK_TUNE": "0" if a.arm == "cold" else "1"}
    argv = packmod.serve_flags(meta, port=a.port, ctx=a.ctx, sessions=a.sessions, k_bits=4, v_bits=4, draft=a.draft)
    if "--no-vision" not in argv:
        argv.append("--no-vision")                           # text path only: the vision tower is another model and costs VRAM
    log = Path(a.out).with_suffix(".srv.log")
    res = dict(arm=a.arm, pack=a.pack, ctx=a.ctx, sessions=a.sessions, cmd=argv, cache_dir=str(cache), seq=[])
    t0 = time.time()
    with running_server(prof, Path(a.repo), argv, a.port, log, logf=print) as (p, ok):
        res["load_s"] = round(time.time() - t0, 1)
        res["records_after_load"] = records_in(cache)
        if not ok:
            res["error"] = "server not ready"
            Path(a.out).write_text(json.dumps(res, indent=1))
            return 1
        cl = load.Client(a.port)
        cl.calibrate(1000)                                   # also the very first request of the process (the cold arm pays tuning here)
        res["records_after_calibrate"] = records_in(cache)
        time.sleep(12)                                       # background work after the first chat request (v1 arms: a 20-row prompt right after it took 2.8-6 s on BOTH arms, 0.3 s when first)
        # requests of the shapes a real client reaches that the first request did not: odd short prompts, tails, batched decode
        # (each new row class is timed inside the request on a cold cache; replay.complete = non-streamed, total wall incl. 8 new tokens)
        import replay
        res["newshape"] = []
        for k, (ptok, conc) in enumerate([(20, 1), (28, 1), (37, 1), (3, 1), (50, 1), (90, 1), (2048 + 5, 1), (2048 + 33, 1), (48, 2), (48, 3), (48, 4)]):
            out = [None] * conc
            th = [replay.threading.Thread(target=lambda j=j: out.__setitem__(j, replay.complete(a.port, replay.token_prompt(ptok, 7000 + 10 * k + j), 8, "qual"))) for j in range(conc)]
            t1 = time.time(); [t.start() for t in th]; [t.join() for t in th]
            res["newshape"].append(dict(ptok=ptok, conc=conc, wall_s=round(time.time() - t1, 2), ok=all(o["ok"] for o in out)))
            print(res["newshape"][-1], flush=True)
        res["records_after_newshape"] = records_in(cache)
        plan = [("short128", 128, 128)] * 2 + [("p1k", 1024, 192)] * 3 + [("p4k", 4096, 128)] * 4 + [("p8k", 8192, 64)] * 2
        for i, (name, ptok, new) in enumerate(plan):
            r = cl.stream(cl.prompt(ptok, i), new)
            res["seq"].append(dict(i=i, name=name, ptok=r["ptok"], ntok=r["ntok"], ttft=r["ttft"], dec_tps=r["dec_tps"], total=r["total"], err=r["err"]))
            print(res["seq"][-1], flush=True)
        n4 = [cl.concurrent(a.sessions, 1024, 128) for _ in range(3)]
        res["n4"] = [dict(agg=x["agg_dec_tps"], per=x["dec_tps"], ttft=x["ttft"], err=x["errors"]) for x in n4]
        res["records_end"] = records_in(cache)
        res["oom"] = oom_in(log)
    by = lambda n: [s for s in res["seq"] if s["name"] == n]
    med = lambda xs: round(statistics.median(xs), 3) if xs else None
    res["summary"] = dict(
        first_request_ttft=res["seq"][0]["ttft"], first_decode_tps=res["seq"][0]["dec_tps"],
        steady_decode_tps=med([s["dec_tps"] for s in by("p4k")[1:] if s["dec_tps"]]),
        p1k_ttft_first=by("p1k")[0]["ttft"], p1k_ttft_steady=med([s["ttft"] for s in by("p1k")[1:]]),
        p4k_ttft_first=by("p4k")[0]["ttft"], p4k_ttft_steady=med([s["ttft"] for s in by("p4k")[1:]]),
        p4k_prefill_tps_steady=med([s["ptok"] / s["ttft"] for s in by("p4k")[1:] if s["ptok"]]),
        p8k_prefill_tps_steady=med([s["ptok"] / s["ttft"] for s in by("p8k")[1:] if s["ptok"]]),
        n4_agg_steady=med([x["agg"] for x in res["n4"][1:] if x["agg"]]),
        runtime_tuned_records=res["records_end"] - res["records_after_load"],
        newshape_wall_total_s=round(sum(x["wall_s"] for x in res["newshape"]), 2), newshape_wall_max_s=max(x["wall_s"] for x in res["newshape"]),
        newshape_records=res["records_after_newshape"] - res["records_after_load"],
    )
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
