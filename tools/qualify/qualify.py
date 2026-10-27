#!/usr/bin/env python3
"""qualify: one command per backend. A converted pack in, parity/KL + KV choice + max ctx + served matrix + greedy sha + Hermes smoke out.
See DESIGN.md. Examples:
  qualify.py --backend cuda --pack ~/models/nanbeige43-33b-exl3-4.0 --dry-run
  qualify.py --backend cuda --pack ... --ref ~/models/nanbeige43-33b-bf16 --stage fetch,meta
Resumable: a stage whose <out>/<stage>.json is done with the same input key is skipped (--redo STAGE to rerun)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backends  # noqa: E402
import pack as packmod  # noqa: E402
import plan as planmod  # noqa: E402
import report  # noqa: E402
import stages as st  # noqa: E402


def parse(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=sorted(backends.PROFILES))
    ap.add_argument("--pack", required=True, help="converted pack directory")
    ap.add_argument("--ref", help="BF16 reference model directory (for parity + KV KL; kept until the gate passes)")
    ap.add_argument("--out", help="state/output dir (default: <repo>/qualify_out/<pack>/<backend>)")
    ap.add_argument("--stage", help="comma list of stages to run (default: all). Stages: " + ",".join(planmod.ORDER))
    ap.add_argument("--redo", help="comma list of stages to clear and rerun")
    ap.add_argument("--force-deps", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, derived serve command and estimates; touch no GPU")
    ap.add_argument("--sessions", default="1,4", help="session counts for the max-ctx search; the largest is the served config (N=1..max matrix)")
    ap.add_argument("--kv-bits", default="8,6,4", help="symmetric KV bits measured by the kv stage (8 is the baseline)")
    ap.add_argument("--kv-kl-max", type=float, default=0.02)
    ap.add_argument("--no-kv-modes", dest="kv_modes", action="store_false", help="skip the KV format auto-select (k6v6 vs tq6 vs tq5, decode-faithful 32K KL vs k8v8) in the kv stage")
    ap.add_argument("--kvmode-steps", type=int, default=256, help="teacher-forced decode steps scored by the KV format auto-select")
    ap.add_argument("--ctx-cap", type=int, help="cap of the max-ctx search (default: the model's max_position_embeddings)")
    ap.add_argument("--disk-floor-gb", type=float, default=40)
    ap.add_argument("--hf", help="HF repo to download into --pack first (license/size/disk-floor checked)")
    ap.add_argument("--conv-log", help="converter log (per-layer SQNR); default <pack>/convert.log")
    ap.add_argument("--smoke-tasks", default="", help="hermes_opt/eval task ids (default: 4 spread over the suite)")
    ap.add_argument("--port", type=int, default=8590)
    ap.add_argument("--repo", default=str(HERE.parents[1]), help="engine checkout with the BUILT extension: PYTHONPATH + cwd of the servers (default: this checkout)")
    ap.add_argument("--gates", type=json.loads, default=None, help='JSON override of verdict gates, e.g. \'{"kl_mean": 0.1}\'')
    a = vars(ap.parse_args(argv))
    a["sessions"] = sorted({int(x) for x in a["sessions"].split(",")})
    a["kv_bits"] = [int(x) for x in a["kv_bits"].split(",")]
    a["pack"] = str(Path(a["pack"]).expanduser())
    if a["ref"]:
        a["ref"] = str(Path(a["ref"]).expanduser())
    a["out"] = str(Path(a["out"]).expanduser() if a["out"] else Path(a["repo"]) / "qualify_out" / Path(a["pack"]).name / a["backend"])
    return a


def logger(out: Path):
    def log(msg):
        line = f"{time.strftime('%F %T')} qualify {msg}"
        print(line, flush=True)
        with open(out / "qualify.log", "a") as f:
            f.write(line + "\n")
    return log


def main(argv=None) -> int:
    a = parse(argv)
    out = Path(a["out"]); out.mkdir(parents=True, exist_ok=True)
    meta = packmod.read_pack(a["pack"])
    state = planmod.load_state(out)
    only = a["stage"].split(",") if a["stage"] else None
    for s in (only or []) + (a["redo"].split(",") if a["redo"] else []):
        if s not in planmod.ORDER:
            sys.exit(f"unknown stage {s!r}")
    plan = planmod.build_plan(meta, a, state, only=only, redo=a["redo"].split(",") if a["redo"] else None, force_deps=a["force_deps"])
    cmd = [backends.profile(a["backend"], a["repo"])["python"], *packmod.serve_flags(meta, port=a["port"], ctx=0, sessions=max(a["sessions"]), k_bits=6, v_bits=6)]
    if a["dry_run"]:
        print(planmod.render_plan(plan, meta, cmd, a))
        return 0
    log = logger(out)
    prof = backends.profile(a["backend"], a["repo"])
    c = st.Ctx(meta, a, out, Path(a["repo"]), prof, log)
    c.data = {n: s["data"] for n, s in state.items() if s.get("status") == "done"}
    for s in plan:
        if s.status == "blocked":
            sys.exit(f"stage {s.name} blocked: {s.note}")
    todo = [s for s in plan if s.status == "todo"]
    results = {n: s for n, s in state.items()}
    done_jobs = set()
    t_all = time.time()
    for s in todo:
        if s.name in ("serve", "smoke"):
            if "serve" in done_jobs:
                continue
            done_jobs.add("serve")
            names = [x.name for x in todo if x.name in ("serve", "smoke")]
            runner = lambda: _split_serve(c, "serve" in names, "smoke" in names)
        else:
            runner = {"fetch": lambda: st.stage_fetch(c), "meta": lambda: st.stage_meta(c), "autotune": lambda: st.job_autotune(c), "parity": lambda: st.job_parity(c),
                      "kv": lambda: st.job_kv(c), "maxctx": lambda: st.job_maxctx(c), "report": None}[s.name]
        if s.name == "report":
            continue
        t0 = time.time(); log(f"stage {s.name} start")
        try:
            data = runner()
        except Exception as e:
            log(f"stage {s.name} FAILED after {time.time()-t0:.0f} s: {e}\n{traceback.format_exc()[-800:]}")
            planmod.save_stage(out, s.name, planmod.fingerprint(s.name, meta, a), "failed", time.time() - t0, dict(error=str(e)))
            return 1
        wall = time.time() - t0
        for name, d in (data.items() if s.name in ("serve", "smoke") else [(s.name, data)]):
            planmod.save_stage(out, name, planmod.fingerprint(name, meta, a), "done", wall, d)
            c.data[name] = d
            results[name] = dict(status="done", wall_s=round(wall, 1), data=d)
        log(f"stage {s.name} done in {wall:.0f} s")
    if True:     # the report is cheap: always rebuilt from the stage files
        results = planmod.load_state(out)
        j = report.assemble(meta, a, results, Path(a["repo"]))
        report.write(out, j)
        planmod.save_stage(out, "report", planmod.fingerprint("report", meta, a), "done", 0, dict(verdict=j["verdict"]))
        log(f"report written ({out}/REPORT.md); verdict pass={j['verdict']['pass']} reasons={j['verdict']['reasons']} unmeasured={j['verdict']['unmeasured']}; total {time.time()-t_all:.0f} s")
    return 0


def _split_serve(c, do_serve, do_smoke):
    r = st.job_serve(c, do_serve, do_smoke)
    out = {}
    if do_serve:
        out["serve"] = {**r["serve"], "cmd": r["cmd"]}
    if do_smoke:
        out["smoke"] = {**(r["smoke"] or {}), "cmd": r["cmd"]}
    return out


if __name__ == "__main__":
    sys.exit(main())
