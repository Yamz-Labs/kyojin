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
import convert_job  # noqa: E402
import pack as packmod  # noqa: E402
import plan as planmod  # noqa: E402
import report  # noqa: E402
import stages as st  # noqa: E402


def int_list(ap: argparse.ArgumentParser, value: str, flag: str) -> list[int]:
    """A comma list of positive integers; anything else is a usage error (exit 2), not a traceback."""
    try:
        out = [int(x) for x in value.split(",")]
    except ValueError:
        out = []
    if not out or min(out) < 1:
        ap.error(f"{flag} wants a comma list of positive integers, got {value!r}")
    return out


def parse(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=sorted(backends.PROFILES), help="GPU backend; picks the python whose torch matches it (QUALIFY_PYTHON overrides)")
    ap.add_argument("--pack", required=True, help="converted pack directory")
    ap.add_argument("--ref", help="BF16 reference model directory (for parity + KV KL; kept until the gate passes)")
    ap.add_argument("--out", help="state/output dir (default: <repo>/qualify_out/<pack>/<backend>)")
    ap.add_argument("--stage", help="comma list of stages to run (default: all). Stages: " + ",".join(planmod.ORDER))
    ap.add_argument("--redo", help="comma list of stages to clear and rerun")
    ap.add_argument("--force-deps", action="store_true", help="with --stage: also run the stages the selected ones need instead of stopping on them")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, derived serve command and estimates; touch no GPU")
    ap.add_argument("--sessions", default="1,4", help="session counts for the max-ctx search; the largest is the served config (N=1..max matrix)")
    ap.add_argument("--kv-bits", default="8,6,4", help="symmetric KV bits measured by the kv stage (8 is the baseline)")
    ap.add_argument("--kv-kl-max", type=float, default=0.02, help="largest KL vs the k8v8 cache for a narrower KV cache to be chosen (kv stage)")
    ap.add_argument("--no-kv-modes", dest="kv_modes", action="store_false", help="skip the KV format auto-select (k6v6 vs tq6 vs tq5, decode-faithful 32K KL vs k8v8) in the kv stage")
    ap.add_argument("--kvmode-steps", type=int, default=256, help="teacher-forced decode steps scored by the KV format auto-select")
    ap.add_argument("--drafter", help="DFlash drafter directory (default: <pack>/dflash, ~/models/drafters/*, else a HF search '<model stem> dflash')")
    ap.add_argument("--no-hf-drafter", action="store_true", help="never search / download a drafter from HF")
    ap.add_argument("--soak-s", type=float, default=360, help="seconds of mixed agent-shaped load per session count at the max-ctx result (soak stage); on any error -c steps down 8 %% and the soak reruns")
    ap.add_argument("--min-share", type=int, default=24576, help="tokens one session must get for the shipped session count (default 24,576: Hermes' live preamble is 15-30K)")
    ap.add_argument("--ctx-cap", type=int, help="cap of the max-ctx search (default: the model's max_position_embeddings)")
    ap.add_argument("--disk-floor-gb", type=float, default=40, help="free disk (GB) to keep after a download or a conversion; the fetch stage fails below it")
    ap.add_argument("--convert-from", help="BF16 checkpoint directory: convert it into --pack first when the pack is missing/incomplete (own-PID memory guard, resume, convert.log + chat template carried into the pack)")
    ap.add_argument("--bits", type=float, default=4.0, help="conversion bits per weight")
    ap.add_argument("--head-bits", type=int, default=6, help="conversion bits of the output head")
    ap.add_argument("--conv-work", help="converter work dir (default <out>/conv_work; kept until the conversion succeeds, deleted then unless --keep-work)")
    ap.add_argument("--conv-extra", default="", help="extra converter flags, e.g. '-hq'")
    ap.add_argument("--conv-pause-gib", type=float, default=6.0, help="SIGSTOP the converter when MemAvailable falls below this")
    ap.add_argument("--conv-resume-gib", type=float, default=10.0, help="SIGCONT the paused converter when MemAvailable is back to this")
    ap.add_argument("--keep-work", action="store_true", help="keep the converter work dir after a successful conversion")
    ap.add_argument("--hf", help="HF repo to download into --pack first (license/size/disk-floor checked)")
    ap.add_argument("--conv-log", help="converter log (per-layer SQNR); default <pack>/convert.log")
    ap.add_argument("--smoke-tasks", default="", help="hermes_opt/eval task ids (default: 4 spread over the suite)")
    ap.add_argument("--port", type=int, default=8590, help="first port tried for the test servers (the next free one is used)")
    ap.add_argument("--repo", default=str(HERE.parents[1]), help="engine checkout with the BUILT extension: PYTHONPATH + cwd of the servers (default: this checkout)")
    ap.add_argument("--gates", type=json.loads, default=None, help='JSON override of verdict gates, e.g. \'{"kl_mean": 0.1}\'')
    a = vars(ap.parse_args(argv))
    a["sessions"] = sorted(set(int_list(ap, a["sessions"], "--sessions")))
    a["kv_bits"] = int_list(ap, a["kv_bits"], "--kv-bits")
    a["pack"] = str(Path(a["pack"]).expanduser().resolve())
    a["repo"] = str(Path(a["repo"]).expanduser().resolve())
    if a["ref"]:
        a["ref"] = str(Path(a["ref"]).expanduser())
    a["out"] = str(Path(a["out"]).expanduser() if a["out"] else Path(a["repo"]) / "qualify_out" / Path(a["pack"]).name / a["backend"])
    return a


def read_pack_or_exit(pack: str) -> dict:
    """The pack metadata, or a one-line error naming --pack instead of a traceback."""
    p = Path(pack)
    if not p.is_dir():
        sys.exit(f"--pack {p}: no such directory")
    if not (p / "config.json").is_file():
        sys.exit(f"--pack {p}: no config.json, not a model directory")
    try:
        return packmod.read_pack(p)
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"--pack {p}: cannot read the model metadata ({type(e).__name__}: {e})")


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
    if a["convert_from"] and not convert_job.pack_complete(a["pack"]):
        if a["dry_run"]:
            print("convert stage: " + " ".join(convert_job.convert_argv(a["convert_from"], a.get("conv_work") or str(out / "conv_work"), a["pack"], a["bits"], a["head_bits"],
                                                                     False, a["conv_extra"].split())) + f"\n(needs ~{convert_job.disk_need_gb(a['convert_from']):.0f} GB; then the plan below on the new pack)")
            return 0
        _convert(a, out)
    if a["ctx_cap"] is None:
        a["ctx_cap"] = backends.PROFILES[a["backend"]].get("ctx_cap_default")
    meta = read_pack_or_exit(a["pack"])
    state = planmod.load_state(out)
    only = a["stage"].split(",") if a["stage"] else None
    for s in (only or []) + (a["redo"].split(",") if a["redo"] else []):
        if s not in planmod.ORDER:
            sys.exit(f"unknown stage {s!r}")
    plan = planmod.build_plan(meta, a, state, only=only, redo=a["redo"].split(",") if a["redo"] else None, force_deps=a["force_deps"])
    # --dry-run touches no GPU and needs no GPU torch: without one, the serve command names $QUALIFY_PYTHON instead
    prof = None if a["dry_run"] else backends.profile(a["backend"], a["repo"])
    python = prof["python"] if prof else backends.find_python(a["backend"], a["repo"], required=False)
    if python is None:
        print(f"no python with a {a['backend']} torch found: the serve command below names $QUALIFY_PYTHON, set it before a real run", file=sys.stderr)
        python = "$QUALIFY_PYTHON"
    cmd = [python, *packmod.serve_flags(meta, port=a["port"], ctx=0, sessions=max(a["sessions"]), k_bits=6, v_bits=6)]
    if a["dry_run"]:
        print(planmod.render_plan(plan, meta, cmd, a))
        return 0
    log = logger(out)
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
            runner = {"fetch": lambda: st.stage_fetch(c), "meta": lambda: st.stage_meta(c), "autotune": lambda: st.job_autotune(c), "draft": lambda: st.job_draft(c), "parity": lambda: st.job_parity(c),
                      "kv": lambda: st.job_kv(c), "maxctx": lambda: st.job_maxctx(c), "soak": lambda: st.job_soak(c), "report": None}[s.name]
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


def _convert(a, out):
    """Stage 0 (only with --convert-from and no complete pack): runs before the pack metadata exists, under the GPU locks."""
    from locks import gpu_locks
    log = logger(out)
    prof = backends.profile(a["backend"], a["repo"])
    log(f"stage convert start ({a['convert_from']} -> {a['pack']}, {a['bits']} bpw, head {a['head_bits']})")
    with gpu_locks(prof["locks"], log):
        r = convert_job.run_convert(prof, Path(a["repo"]), a, out, log)
    (out / "convert.json").write_text(json.dumps(r, indent=1))
    log(f"stage convert done in {r['wall_s']:.0f} s (paused {r['paused_s']:.0f} s, {r['pauses']} pauses, template from {r['template_from']})")


def _split_serve(c, do_serve, do_smoke):
    r = st.job_serve(c, do_serve, do_smoke)
    out = {}
    if do_serve:
        out["serve"] = {**r["serve"], "cmd": r["cmd"], "lock_hold_max_s": r.get("lock_hold_max_s")}
    if do_smoke:
        out["smoke"] = {**(r["smoke"] or {}), "cmd": r.get("smoke_cmd") or r["cmd"], "lock_hold_max_s": r.get("lock_hold_max_s")}
    return out


if __name__ == "__main__":
    sys.exit(main())
