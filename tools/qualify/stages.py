"""Stage executors. Each takes the Ctx and returns the stage's data dict (saved by the driver into <out>/<stage>.json).
GPU stages run inside job functions that hold the backend locks; every server is started and killed by PID."""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import backends
import docs as docs_mod
import load
import pack as packmod
import rules
import search
import smoke
from locks import MIN_MEM_GIB, gpu_locks
from server import oom_in, running_server


@dataclass
class Ctx:
    meta: dict
    args: dict
    out: Path
    repo: Path
    prof: dict
    log: callable = print
    data: dict = field(default_factory=dict)       # stage name -> data (done in this run or resumed)

    def py(self, script: str, *a, env_extra: dict | None = None, timeout: int = 7200) -> str:
        import os
        env = {**os.environ, **self.prof["env"], **(env_extra or {})}
        r = subprocess.run([self.prof["python"], "-u", str(Path(__file__).parent / script), *map(str, a)], env=env, capture_output=True, text=True, timeout=timeout)
        (self.out / f"{Path(script).stem}.last.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
        if r.returncode:
            raise RuntimeError(f"{script} rc={r.returncode}: {(r.stderr or r.stdout)[-600:]}")
        return r.stdout

    def need_ram(self):
        a = backends.mem_available_gib()
        if a < MIN_MEM_GIB:
            raise RuntimeError(f"MemAvailable {a:.1f} GiB < {MIN_MEM_GIB} GiB: not starting a GPU job")

    def locks(self):
        return gpu_locks(self.prof["locks"], self.log)


# ---------------------------------------------------------------- cpu stages
def stage_fetch(c: Ctx) -> dict:
    a = c.args
    d = dict(pack=c.meta["path"], shard_problems=packmod.verify_shards(c.meta["path"]), license=c.meta["license"],
             disk_free_gb=round(packmod.disk_free_gb(c.meta["path"]), 1), disk_floor_gb=a["disk_floor_gb"])
    if a.get("hf"):
        d["hf"] = fetch_hf(a["hf"], c.meta["path"], a["disk_floor_gb"])
    if a.get("ref"):
        d["ref_problems"] = packmod.verify_shards(a["ref"])
    d["ok"] = not d["shard_problems"] and not d.get("ref_problems") and d["disk_free_gb"] >= a["disk_floor_gb"]
    if not d["ok"]:
        raise RuntimeError("fetch/verify failed: " + json.dumps({k: d[k] for k in ("shard_problems", "ref_problems", "disk_free_gb") if k in d}))
    return d


def fetch_hf(repo: str, dest: str, floor_gb: float) -> dict:
    """Download under the disk floor (drill-0 gap 1: license + size + floor were checked by hand)."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(repo, files_metadata=True)
    size = sum(s.size or 0 for s in info.siblings) / 1e9
    lic = next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None)
    free = packmod.disk_free_gb(dest)
    if free - size < floor_gb:
        raise RuntimeError(f"{repo}: {size:.1f} GB would leave {free-size:.1f} GB free < floor {floor_gb} GB")
    snapshot_download(repo, local_dir=dest)
    return dict(repo=repo, size_gb=round(size, 1), license=lic)


def stage_meta(c: Ctx) -> dict:
    m = dict(c.meta)
    m["serve_cmd_template"] = packmod.serve_flags(m, port=c.args["port"], ctx=0, sessions=0, k_bits=6, v_bits=6)
    m["draft_policy"] = packmod.draft_policy(m)
    if m["problems"]:
        raise RuntimeError("pack problems: " + "; ".join(m["problems"]))
    return m


# ---------------------------------------------------------------- autotune
def _tune_cache_mod():
    """exllamav3/util/tune_cache.py by path: the cache file format needs no torch (the package import does)."""
    import importlib.util
    f = Path(__file__).resolve().parents[2] / "exllamav3/util/tune_cache.py"
    spec = importlib.util.spec_from_file_location("tune_cache", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def job_autotune(c: Ctx) -> dict:
    """Day-0 autotune by shape replay (replay.py): serve the pack on a FRESH tune cache, drive the row sweep through the real served path,
    keep the winners the engine's autotuner wrote -> <pack>/tuning/<device signature>.bin (+ .json), loaded at serve start."""
    import shutil
    import replay
    tc = _tune_cache_mod()
    c.need_ram()
    tdir = Path(c.meta["path"]) / "tuning"
    scratch = c.out / "tune_scratch"
    shutil.rmtree(scratch, ignore_errors=True); scratch.mkdir(parents=True)
    sessions = max(c.args["sessions"])
    ctx = min(c.args.get("ctx_cap") or 16384, 16384, c.meta["max_pos"])
    prof = {**c.prof, "env": {**c.prof["env"], "EXLLAMAV3_TUNE_CACHE": str(scratch), "EXL3_PACK_TUNE": "0",
                                  "EXL3_DENSE_GEMM_TUNE_FILE": str(scratch / "dense_gemm_tune.txt")}}
    k, v = chosen_kv(c)
    argv = packmod.serve_flags(c.meta, port=c.args["port"], ctx=ctx, sessions=sessions, k_bits=k, v_bits=v)
    if "--no-vision" not in argv:
        argv.append("--no-vision")                   # the tower is another model (no GEMM autotune keys of the text path) and costs VRAM
    log = c.out / "srv_autotune.log"
    with c.locks():
        sig = json.loads(c.py("autotune.py", "--signature").strip().splitlines()[-1])
        t0 = time.time()
        with running_server(prof, c.repo, argv, c.args["port"], log, logf=c.log) as (p, ok):
            if not ok:
                raise RuntimeError("autotune: server did not start: " + (log.read_text(errors="replace").strip().splitlines() or [""])[-1][:200])
            load_s = time.time() - t0
            at_load = len(tc.read_records(scratch / "coop_autotune_v1.bin"))
            sweep = replay.run_sweep(c.args["port"], sessions, ctx, log=c.log)
        if oom_in(log):
            raise RuntimeError("autotune: OOM in the server log")
    recs = tc.read_records(scratch / "coop_autotune_v1.bin")
    if not recs:
        raise RuntimeError("autotune: the replay produced no tuning records (does this build have the GEMM autotuner?)")
    tdir.mkdir(exist_ok=True)
    dense_new = _new_dense_lines(scratch / "dense_gemm_tune.txt", Path(__file__).resolve().parents[2] / "exllamav3/model/dense_gemm_tune_seed.txt")
    if dense_new:                                    # lines the shipped seed did not hold (serve start copies the seed into the scratch file)
        (tdir / f"{sig['file']}.dense.txt").write_text("\n".join(dense_new) + "\n")
    tc.write_records(tdir / f"{sig['file']}.bin", recs)
    man = dict(schema=1, signature=sig, records=len(recs), records_at_load=at_load, load_s=round(load_s, 1), wall_s=round(time.time() - t0, 1),
               sessions=sessions, ctx=ctx, sweep=sweep, cmd=argv, engine_git=_git(c.repo), pack_index_sha=c.meta.get("index_sha"))
    (tdir / f"{sig['file']}.json").write_text(json.dumps(man, indent=1))
    c.log(f"autotune: {len(recs)} records ({at_load} at load) -> {tdir / (sig['file'] + '.bin')}")
    return dict(tuning_dir=str(tdir), files=sorted(p.name for p in tdir.glob("*.bin")), records=len(recs), records_at_load=at_load,
                plan_points=len(sweep), wall_s=man["wall_s"], sweep_failed=[s for s in sweep if not s["ok"]])


def _new_dense_lines(tuned: Path, seed: Path) -> list:
    key = lambda ln: tuple(ln.split()[:6])
    have = {key(l) for l in seed.read_text().splitlines() if len(l.split()) == 7} if seed.is_file() else set()
    if not tuned.is_file():
        return []
    return [l for l in tuned.read_text().splitlines() if len(l.split()) == 7 and key(l) not in have]


def _git(repo) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:   # noqa: BLE001
        return None


# ---------------------------------------------------------------- parity / kv
SQNR = re.compile(r"Quantized: model\.layers\.(\d+)\s+bpw:\s*([\d.]+)\s+rfn:\s*([\d.]+)\s+cos:\s*([\d.]+)\s+sqnr:\s*([\d.]+)")


def parse_sqnr(text: str) -> dict:
    """Per-layer SQNR (dB) from the converter's log lines ' -- Quantized: model.layers.N  bpw: .. rfn: .. cos: .. sqnr: ..'."""
    return {int(m.group(1)): float(m.group(5)) for m in SQNR.finditer(text)}


def sqnr_summary(per_layer: dict) -> dict:
    if not per_layer:
        return dict(measured=False)
    v = list(per_layer.values())
    return dict(measured=True, layers=len(v), min=min(v), max=max(v), mean=round(statistics.mean(v), 2), worst_layer=min(per_layer, key=per_layer.get))


def aggregate_parity(res: dict) -> dict:
    tot = sum(v["tokens"] for v in res.values())
    w = lambda k: sum(v[k] * v["tokens"] for v in res.values()) / tot
    return dict(kl_mean=w("kl"), top1_mean=w("top1"), dnll_mean=w("dnll"), by_domain=res)


def run_ref(c: Ctx, docs: Path, outdir: Path, tail: int = 0):
    adapter = c.meta["ref_adapter"]
    c.py(f"refs/{adapter}.py", c.args["ref"], docs, outdir, env_extra={"TAIL": str(tail)} if tail else {})


def job_parity(c: Ctx) -> dict:
    c.need_ram()
    d = c.out / "docs_short.pt"
    if not d.exists():
        c.py("docs.py", "short", c.meta["path"], d)
    with c.locks():
        t0 = time.time()
        refdir = c.out / "ref_short"
        if not (refdir / "logits_chat_fr.pt").exists():
            run_ref(c, d, refdir)
        c.log(f"parity: reference streamed in {time.time()-t0:.0f} s")
        c.py("gpu_eval.py", "parity", c.meta["path"], d, refdir, c.out / "parity_raw.json")
    res = aggregate_parity(json.loads((c.out / "parity_raw.json").read_text()))
    log = Path(c.args["conv_log"]).expanduser() if c.args.get("conv_log") else Path(c.meta["path"]) / "convert.log"
    res["sqnr"] = sqnr_summary(parse_sqnr(log.read_text(errors="replace"))) if log.exists() else dict(measured=False, note="no conversion log (--conv-log or <pack>/convert.log)")
    return res


def job_kv(c: Ctx) -> dict:
    c.need_ram()
    d = c.out / "docs_long.pt"
    if not d.exists():
        c.py("docs.py", "long", c.meta["path"], d)
    bits = c.args["kv_bits"]
    with c.locks():
        refdir = c.out / "ref_long"
        if not (refdir / "logits_long.pt").exists():
            run_ref(c, d, refdir, tail=1024)
        c.py("gpu_eval.py", "kv", c.meta["path"], d, refdir, c.out / "kv_raw.json", *bits)
    raw = json.loads((c.out / "kv_raw.json").read_text())
    kl8 = {int(b): v["kl_vs_k8"] for b, v in raw.items() if "kl_vs_k8" in v}
    out = dict(table=raw, chosen=list(rules.choose_kv(kl8, c.args["kv_kl_max"])), kl_max=c.args["kv_kl_max"], rule="smallest symmetric bits >= Q6 with KL vs k8v8 <= kl_max, else k8v8", mode="quant")
    if c.args.get("kv_modes", True) and tuple(out["chosen"]) == (6, 6):
        out.update(job_kvmode(c, d))
    write_kv_choice(c, out)
    return out


def job_kvmode(c: Ctx, docs) -> dict:
    """KV format auto-select, ONCE per model: decode-faithful 32K KL vs k8v8 of k6v6 (quant), kw6 (k6v6 + sink/window ring), tq6, tq5 (TurboQuant + ring).
    Q6 rule (rules.choose_kv_mode): kw6 when its KL <= 0.9 x k6v6's, else k6v6; the tq formats are measured and reported, opt-in only
    (QUALIFY_KV_TQ=1 lets them win: any tq format with KL <= 0.9 x k6v6's, then the fewest bytes per value)."""
    with c.locks():
        c.py("gpu_eval.py", "kvmode", c.meta["path"], docs, c.out / "kvmode_raw.json", c.args.get("kvmode_steps", 256), timeout=14400)
    raw = json.loads((c.out / "kvmode_raw.json").read_text())
    kl = {n: v["kl_vs_k8"] for n, v in raw.items()}
    bpv = {n: v["bits_per_val"] for n, v in raw.items()}
    win = rules.choose_kv_mode(kl, bpv, allowed=rules.KV_DEFAULT_FORMATS + (("tq6", "tq5") if os.environ.get("QUALIFY_KV_TQ") == "1" else ()))
    if win == "k6v6":
        return dict(modes=raw, mode="quant")
    if win == "kw6":
        return dict(modes=raw, mode="quant", chosen=[6, 6], kvwin=rules.KV_WIN_SPEC, rule="Q6 rule: k6v6 + sink/window ring when its KL vs k8v8 <= 0.9 x k6v6's")
    b = int(win[2:])
    return dict(modes=raw, mode="tq", chosen=[b, b], q6_equivalent=True, rule="Q6 rule: tq format with KL vs k8v8 <= k6v6's, fewest bytes per value")


def write_kv_choice(c: Ctx, kv: dict) -> None:
    """The lane preset: <pack>/kv_cache.json, read by tools/qwen/serve.py when no --k-bits / --v-bits / --kv-mode is given."""
    k, v = kv["chosen"]
    doc = dict(schema=1, kv_mode=kv.get("mode", "quant"), k_bits=k, v_bits=v, kvwin=kv.get("kvwin"), engine_git=_git(c.repo), pack_index_sha=c.meta.get("index_sha"),
               kl_vs_k8=({n: x["kl_vs_k8"] for n, x in kv["modes"].items()} if kv.get("modes") else None))
    tmp = Path(c.meta["path"]) / "kv_cache.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(Path(c.meta["path"]) / "kv_cache.json")


def chosen_kv(c: Ctx) -> tuple:
    kv = c.data.get("kv")
    return tuple(kv["chosen"]) if kv else (6, 6)     # no reference: owner's Q6 minimum


def chosen_kvwin(c: Ctx) -> str | None:
    return (c.data.get("kv") or {}).get("kvwin")


def chosen_kv_mode(c: Ctx) -> str:
    return (c.data.get("kv") or {}).get("mode", "quant")


# ---------------------------------------------------------------- max ctx
def start_args(c: Ctx, ctx: int, sessions: int) -> list:
    k, v = chosen_kv(c)
    return packmod.serve_flags(c.meta, port=c.args["port"], ctx=ctx, sessions=sessions, k_bits=k, v_bits=v, kv_mode=chosen_kv_mode(c), kvwin=chosen_kvwin(c))


def probe_factory(c: Ctx, sessions: int):
    cl = load.Client(c.args["port"])
    state = {"cal": False}

    def probe(ctx: int):
        log = c.out / f"srv_maxctx_s{sessions}_c{ctx}.log"
        with running_server(c.prof, c.repo, start_args(c, ctx, sessions), c.args["port"], log, logf=c.log) as (p, ok):
            if not ok:
                tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
                return False, "no start: " + tail[0][:160]
            if not state["cal"]:
                cl.calibrate(); state["cal"] = True
            for slot in (0, 1):                                  # 2 distinct near-full prompts back to back
                r = cl.near_full(ctx, slot)
                if r["err"] or oom_in(log):
                    return False, f"near-full prompt {slot} failed: {r['err']}"
            if sessions > 1:                                     # `sessions` concurrent shares, twice (drill 0: the 2nd round OOMed)
                for rnd in range(2):
                    per = int(ctx * 0.97 / sessions) - 64
                    rr = cl.concurrent(sessions, per, 32)
                    if rr["errors"] or oom_in(log):
                        return False, f"concurrent round {rnd} failed: {rr['errors'][:1]}"
            return (not oom_in(log)), "ok"
    return probe


def job_maxctx(c: Ctx) -> dict:
    c.need_ram()
    out = {}
    k, v = chosen_kv(c)
    with c.locks():
        free = backends.gpu_free_mib(c.args["backend"])
        for s in c.args["sessions"]:
            bpt = packmod.kv_bytes_per_token(c.meta, k, v, chosen_kv_mode(c))
            free_b = (free * 2**20 if free else 24 * 2**30) - c.meta["weights_bytes"]
            hi = search.upper_bound(free_b, bpt, 2.5 * 2**30, c.args.get("ctx_cap") or c.meta["max_pos"])
            lo = min(hi, 8192)
            c.log(f"maxctx sessions={s}: KV {bpt/1024:.1f} KB/token, search [{lo}, {hi}]")
            r = search.bisect_max(lo, hi, probe_factory(c, s))
            out[str(s)] = r["best"]
            out[f"tried_s{s}"] = r["tried"]
    return out


# ---------------------------------------------------------------- serve + smoke (one server load)
def job_serve(c: Ctx, do_serve: bool, do_smoke: bool) -> dict:
    c.need_ram()
    mc = c.data.get("maxctx") or {}
    sess = max(c.args["sessions"])
    ctx = mc.get(str(sess)) or max((v for k, v in mc.items() if k.isdigit() and v), default=None)
    if not ctx:
        ctx = c.args.get("ctx_cap") or 32768
        c.log(f"serve: no max-ctx result, using {ctx}")
    sess_eff = sess if mc.get(str(sess)) else 1
    out = dict(serve={}, smoke=None)
    with c.locks():
        log = c.out / "srv_serve.log"
        argv = start_args(c, ctx, sess_eff)
        out["cmd"] = [c.prof["python"], *argv]
        with running_server(c.prof, c.repo, argv, c.args["port"], log, logf=c.log) as (p, ok):
            if not ok:
                raise RuntimeError("serve: server did not become ready; see " + str(log))
            cl = load.Client(c.args["port"], "qual")
            cl.calibrate()
            s = {}
            if do_serve:
                s["conformance"] = cl.conformance()
                s["decode_short"] = [cl.stream(cl.prompt(300, i), 256) for i in range(3)]
                s["prefill_8k"] = [cl.stream(cl.prompt(8000, i + 3), 1) for i in range(3)]
                curve, t = [], 2048
                while t <= int(ctx * 0.9):
                    r = cl.stream(cl.prompt(t, 7), 64)
                    curve.append(dict(prompt=r["ptok"], ttft=r["ttft"], prefill_tps=round(r["ptok"] / r["ttft"], 1) if r["ptok"] and r["ttft"] else None, dec_tps=r["dec_tps"], err=r["err"]))
                    t *= 4
                s["curve"] = curve
                s["agg"] = [cl.concurrent(n, min(4096, ctx // max(4, sess_eff) - 256), 128) for n in range(1, sess_eff + 1)]
                a, b = cl.sha_set(), cl.sha_set()
                s["sha"], s["sha_self_equal"] = a, [x["sha"] for x in a] == [x["sha"] for x in b]
                s["oom"] = oom_in(log)
                for r in s["decode_short"] + s["prefill_8k"]:
                    r.pop("t0", None); r.pop("first", None); r.pop("end", None)
                s["decode_tps_median"] = statistics.median(r["dec_tps"] for r in s["decode_short"] if r["dec_tps"])
                s["prefill_8k_tps"] = statistics.median(r["ptok"] / r["ttft"] for r in s["prefill_8k"] if r["ptok"] and r["ttft"])
                out["serve"] = s
            if do_smoke:
                out["smoke"] = run_smoke(c)
    return out


def pick_tasks(eval_dir: Path, n: int) -> str:
    r = subprocess.run([__import__("sys").executable, "-I", "-c",
                        f"import sys; sys.path.insert(0,'{eval_dir}/lib'); import taskdefs; ids=[t['id'] for t in taskdefs.TASKS]; "
                        f"step=max(1,len(ids)//{n}); print(','.join(ids[::step][:{n}]))"], capture_output=True, text=True)
    return r.stdout.strip()


def run_smoke(c: Ctx) -> dict:
    ev = smoke.find_eval()
    if ev:
        tasks = c.args.get("smoke_tasks") or pick_tasks(ev, 4)
        res = smoke.run_hermes_eval(ev, c.args["port"], "qual", tasks, f"qualify_{c.meta['name']}")
        if res["total"]:
            return res
        c.log("smoke: hermes eval gave no result, falling back to the scripted tool loop: " + str(res.get("error"))[:200])
    runs = [smoke.run_scripted_agent(c.args["port"], "qual", think=t) for t in (False, True)]
    return dict(kind="scripted_agent", total=len(runs), passed=sum(r["passed"] for r in runs), runs=runs, note="hermes_opt/eval not available (lib/taskdefs.py missing): fallback")
