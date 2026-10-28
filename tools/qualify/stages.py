"""Stage executors. Each takes the Ctx and returns the stage's data dict (saved by the driver into <out>/<stage>.json).
GPU stages run inside job functions that hold the backend locks; every server is started and killed by PID."""
from __future__ import annotations

import contextlib
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
import drafter as drafter_mod
import rules
import search
import smoke
from locks import LIGHT_GIB, MIN_MEM_GIB, gpu_locks
from server import oom_in, pick_port, running_server


@dataclass
class Ctx:
    meta: dict
    args: dict
    out: Path
    repo: Path
    prof: dict
    log: callable = print
    data: dict = field(default_factory=dict)       # stage name -> data (done in this run or resumed)
    holds: list = field(default_factory=list)      # (label, seconds) of every GPU-lock hold of this run

    def py(self, script: str, *a, env_extra: dict | None = None, timeout: int = 7200) -> str:
        import os
        env = {**os.environ, **self.prof["env"], **(env_extra or {})}
        r = subprocess.run([self.prof["python"], "-u", str(Path(__file__).parent / script), *map(str, a)], env=env, capture_output=True, text=True, timeout=timeout)
        (self.out / f"{Path(script).stem}.last.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
        if r.returncode:
            raise RuntimeError(f"{script} rc={r.returncode}: {(r.stderr or r.stdout)[-600:]}")
        return r.stdout

    def fresh_port(self) -> int:
        """Re-pick the server port before every start (kept in args['port']; the first value asked for stays the search base)."""
        base = self.args.setdefault("port0", self.args["port"])
        self.args["port"] = pick_port(base)
        return self.args["port"]

    def need_ram(self):
        a = backends.mem_available_gib()
        if a < MIN_MEM_GIB:
            raise RuntimeError(f"MemAvailable {a:.1f} GiB < {MIN_MEM_GIB} GiB: not starting a GPU job")

    def light_gib(self, ctx: int) -> float:
        """Upper bound of the device memory a server of this pack needs at -c ctx: weights + the k8v8 cache (the widest default) + 1 GiB of runtime."""
        return (self.meta["weights_bytes"] + packmod.kv_bytes_per_token(self.meta, 8, 8) * ctx) / 2**30 + 1.0

    @contextlib.contextmanager
    def locks(self, label: str = "", light_ctx: int | None = None):
        """GPU locks for ONE unit of work (a server load + its measurements, one KV format, one probe): every hold is timed in c.holds so the
        stage data can show that none ran long while other workers queued (lead rule 27/10: no stage holds a lock for more than ~15 min).
        light_ctx = the unit is not a timed measurement (agent-bound smoke) and its server is small (light_gib < LIGHT_GIB): it runs WITHOUT the
        exclusive GPU locks, so a 0 %-GPU step never queues another worker's cell (box-lead 27/10 08:18)."""
        light = light_ctx is not None and self.light_gib(light_ctx) < LIGHT_GIB
        if light:
            self.log(f"light step '{label}': est. {self.light_gib(light_ctx):.1f} GiB < {LIGHT_GIB:.0f} GiB, not timed: no exclusive GPU lock")
        with (contextlib.nullcontext() if light else gpu_locks(self.prof["locks"], self.log)):
            t0 = time.time()
            try:
                yield
            finally:
                self.holds.append((label, round(time.time() - t0, 1)))


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


def fetch_hf(repo: str, dest: str, floor_gb: float, revision: str = "main") -> dict:
    """Download under the disk floor (drill-0 gap 1: license + size + floor were checked by hand)."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    size = sum(s.size or 0 for s in info.siblings) / 1e9
    lic = next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None)
    free = packmod.disk_free_gb(dest)
    if free - size < floor_gb:
        raise RuntimeError(f"{repo}: {size:.1f} GB would leave {free-size:.1f} GB free < floor {floor_gb} GB")
    snapshot_download(repo, revision=revision, local_dir=dest)
    return dict(repo=repo, revision=revision, size_gb=round(size, 1), license=lic)


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
    pol, ddir = chosen_draft(c)
    c.fresh_port()
    argv = packmod.serve_flags(c.meta, port=c.args["port"], ctx=ctx, sessions=sessions, k_bits=k, v_bits=v, draft=pol, dflash_dir=ddir)
    if "--no-vision" not in argv:
        argv.append("--no-vision")                   # the tower is another model (no GEMM autotune keys of the text path) and costs VRAM
    log = c.out / "srv_autotune.log"
    with c.locks():
        sig = json.loads(c.py("autotune.py", "--signature").strip().splitlines()[-1])
        t0 = time.time()
        with running_server(prof, c.repo, argv, c.args["port"], log, logf=c.log, args=c.args) as (p, ok):
            if not ok:
                raise RuntimeError("autotune: server did not start: " + (log.read_text(errors="replace").strip().splitlines() or [""])[-1][:200])
            load_s = time.time() - t0
            at_load = len(tc.read_records(scratch / "coop_autotune_v1.bin"))
            sweep = replay.run_sweep(c.args["port"], sessions, ctx, log=c.log)
        if oom_in(log):
            raise RuntimeError("autotune: OOM in the server log")
    recs = tc.read_records(scratch / "coop_autotune_v1.bin")
    tdir.mkdir(exist_ok=True)
    dense_new = _new_dense_lines(scratch / "dense_gemm_tune.txt", Path(__file__).resolve().parents[2] / "exllamav3/model/dense_gemm_tune_seed.txt")
    if not recs and not dense_new:
        return dict(skipped=True, reason="the replay produced no tuning record on this backend/build (no coop GEMM autotuner call, no new dense-GEMM line): nothing to ship", plan_points=len(sweep),
                    sweep_failed=[x for x in sweep if not x["ok"]], wall_s=round(time.time() - t0, 1))
    if dense_new:                                    # lines the shipped seed did not hold (serve start copies the seed into the scratch file)
        (tdir / f"{sig['file']}.dense.txt").write_text("\n".join(dense_new) + "\n")
    if recs:                                         # ROCm serves most shapes through the dense path: no coop records there is normal
        tc.write_records(tdir / f"{sig['file']}.bin", recs)
    man = dict(schema=1, signature=sig, records=len(recs), records_at_load=at_load, load_s=round(load_s, 1), wall_s=round(time.time() - t0, 1),
               sessions=sessions, ctx=ctx, sweep=sweep, cmd=argv, engine_git=_git(c.repo), pack_index_sha=c.meta.get("index_sha"))
    (tdir / f"{sig['file']}.json").write_text(json.dumps(man, indent=1))
    c.log(f"autotune: {len(recs)} records ({at_load} at load) -> {tdir / (sig['file'] + '.bin')}")
    return dict(tuning_dir=str(tdir), files=sorted(p.name for p in tdir.glob(f"{sig['file']}*")), records=len(recs), records_at_load=at_load, dense_lines=len(dense_new),
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
    have_ref = bool(c.args.get("ref") and c.meta.get("ref_adapter"))
    if have_ref:
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
    else:       # no BF16 reference on disk (the pack is all there is): the pack-only kvmode measure decides, KL against the pack's own k8v8
        km = job_kvmode(c, d)
        k6 = rules.choose_kv({6: km["modes"]["k6v6"]["kl_vs_k8"]}, c.args["kv_kl_max"])
        out = dict(table={}, chosen=list(k6), kl_max=c.args["kv_kl_max"], mode="quant", rule="no BF16 reference: Q6 if kvmode KL(k6v6 vs k8v8) <= kl_max else k8v8; Q6 format by the kvmode rule")
        if k6 == (6, 6):
            out.update(km)
        else:
            out["modes"] = km["modes"]
    write_kv_choice(c, out)
    return out


def job_kvmode(c: Ctx, docs) -> dict:
    """KV format auto-select, ONCE per model: decode-faithful 32K KL vs k8v8 of k6v6 (quant), kw6 (k6v6 + sink/window ring), tq6, tq5 (TurboQuant + ring).
    Q6 rule (rules.choose_kv_mode): kw6 when its KL <= 0.9 x k6v6's; tq6 is a default candidate (tq5: QUALIFY_KV_TQ=1, tq off: QUALIFY_KV_TQ=0); the fewest bytes per value among the qualifying formats wins, else k6v6."""
    for f in (c.out / "kvmode_raw.json", c.out / "kvmode_raw.json.k8v8.pt"):
        f.unlink(missing_ok=True)
    for fmt in ("k8v8", "k6v6", "kw6", "tq6", "tq5"):       # one lock take per format (each is a model load + a 32K pass)
        with c.locks(f"kvmode {fmt}"):
            c.py("gpu_eval.py", "kvmode", c.meta["path"], docs, c.out / "kvmode_raw.json", c.args.get("kvmode_steps", 256), fmt, timeout=14400)
    raw = json.loads((c.out / "kvmode_raw.json").read_text())
    raw = {n: v for n, v in raw.items() if "kl_vs_k8" in v}       # a format the model's attention cannot hold (tq on QSA layers) is recorded as unsupported, not measured
    kl = {n: v["kl_vs_k8"] for n, v in raw.items()}
    bpv = {n: v["bits_per_val"] for n, v in raw.items()}
    dec = {n: v.get("decode_s_per_tok") for n, v in raw.items()}
    win = rules.choose_kv_mode(kl, bpv, dec_s=dec, allowed=rules.kv_allowed_formats(os.environ.get("QUALIFY_KV_TQ")))
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
    be = c.args.get("backend", "cuda")
    doc["backend"] = be
    for name in (f"kv_cache.{be}.json", "kv_cache.json"):      # per-backend file (serve.py reads it first); the plain one stays for older servers
        if name == "kv_cache.json" and (Path(c.meta["path"]) / name).exists() and json.loads((Path(c.meta["path"]) / name).read_text()).get("backend") not in (None, be):
            continue                                                         # never overwrite another backend's choice
        tmp = Path(c.meta["path"]) / (name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(Path(c.meta["path"]) / name)


def chosen_kv(c: Ctx) -> tuple:
    kv = c.data.get("kv")
    return tuple(kv["chosen"]) if kv else (6, 6)     # no reference: owner's Q6 minimum


def chosen_draft(c: Ctx) -> tuple:
    """(policy, dflash dir) the later stages serve with: the draft stage's A/B winner, else the pack's own policy (vendor MTP head or off)."""
    d = c.data.get("draft") or {}
    return d.get("policy") or packmod.draft_policy(c.meta), d.get("dflash_dir")


def chosen_kvwin(c: Ctx) -> str | None:
    return (c.data.get("kv") or {}).get("kvwin")


def chosen_kv_mode(c: Ctx) -> str:
    return (c.data.get("kv") or {}).get("mode", "quant")


# ---------------------------------------------------------------- draft path
def locate_drafter(c: Ctx) -> dict:
    """Local candidates first (--drafter, <pack>/dflash, ~/models/drafters), then one HF search by model stem (+ download under the disk floor)."""
    d, notes = drafter_mod.find_local(c.meta, c.args.get("drafter"))
    if d or c.args.get("no_hf_drafter"):
        return dict(dir=d, notes=notes, source="local" if d else None)
    repo = drafter_mod.pick_hf(drafter_mod.search_hf(c.meta))
    if not repo:
        return dict(dir=None, notes=notes + ["no HF repo named <model stem> + dflash"], source=None)
    dest = drafter_mod.DRAFTER_HOME / repo.split("/", 1)[-1]
    info = fetch_hf(repo, str(dest), c.args["disk_floor_gb"], drafter_mod.hf_revision(repo))
    bad = drafter_mod.validate(dest, c.meta)
    if bad:
        return dict(dir=None, notes=notes + [f"{repo}: " + "; ".join(bad)], source=None, hf=info)
    return dict(dir=str(dest), notes=notes, source="hf", hf=info)


def draft_variants() -> list:
    """ABBA: plain, drafted, drafted, plain (both orders, so a warm-up or thermal drift hits both policies equally)."""
    return ["plain", "dflash", "dflash", "plain"]


def speedups(runs: list) -> dict:
    """runs: [{variant, tps: {kind: [tok/s ...]}}] -> {kind: mean drafted tok/s / mean plain tok/s} (warm-up already dropped).
    A kind with no data point in some variant (a request that returned no decode rate) is left out, not fatal."""
    out = {}
    for kind in runs[0]["tps"]:
        m = {v: [x for r in runs if r["variant"] == v for x in r["tps"].get(kind, [])] for v in ("plain", "dflash")}
        if m["plain"] and m["dflash"]:
            out[kind] = round(statistics.mean(m["dflash"]) / statistics.mean(m["plain"]), 3)
    return out


def job_draft(c: Ctx) -> dict:
    base = packmod.draft_policy(c.meta)
    if base != "off" or c.meta.get("mtp_layers"):         # the pack's own MTP head is the draft path; vendor drafters are for head-less packs
        return dict(policy=base, dflash_dir=None, reason=f"pack has {c.meta['mtp_layers']} MTP layer(s): policy {base}")
    loc = locate_drafter(c)
    if not loc["dir"]:
        return dict(policy=base, dflash_dir=None, reason="no compatible drafter found", **{k: loc[k] for k in ("notes",)})
    c.need_ram()
    k, v = chosen_kv(c)
    ctx = min(12288, c.meta["max_pos"])
    runs = []
    for i, var in enumerate(draft_variants()):
        with c.locks(f"draft {var}"):
            c.fresh_port()
            argv = packmod.serve_flags(c.meta, port=c.args["port"], ctx=ctx, sessions=1, k_bits=k, v_bits=v, kv_mode=chosen_kv_mode(c), kvwin=chosen_kvwin(c),
                                       draft="dflash" if var == "dflash" else "off", dflash_dir=loc["dir"] if var == "dflash" else None)
            log = c.out / f"srv_draft_{i}_{var}.log"
            with running_server(c.prof, c.repo, argv, c.args["port"], log, logf=c.log, args=c.args) as (p, ok):
                if not ok:
                    raise RuntimeError(f"draft: {var} server did not start: " + (log.read_text(errors="replace").strip().splitlines() or [""])[-1][:200])
                cl = load.Client(c.args["port"], "qual")
                tps, errs = {}, {}
                for kind, prompts in cl.kind_prompts().items():
                    cl.stream(prompts[0], 64)                                     # warm-up (graphs, caches) discarded
                    rs = [cl.stream(pr, 256) for pr in prompts for _ in range(2)]
                    tps[kind] = [r["dec_tps"] for r in rs if r["dec_tps"]]
                    if len(tps[kind]) < len(rs):
                        errs[kind] = [r["err"] or f"ntok={r['ntok']}" for r in rs if not r["dec_tps"]][:2]
            runs.append(dict(variant=var, tps=tps, errs=errs, oom=oom_in(log)))
            c.log(f"draft: {var}: " + ", ".join(f"{k_} {statistics.mean(x):.1f}" for k_, x in tps.items() if x) + " tok/s" + (f"; unmeasured {errs}" if errs else ""))
    sp = speedups(runs)
    use, why = rules.choose_draft(sp)
    return dict(policy="dflash" if use else base, dflash_dir=loc["dir"] if use else None, drafter=loc["dir"], source=loc["source"], speedup=sp, reason=why, runs=runs,
                lock_hold_max_s=max((h for _, h in c.holds), default=None),
                note="adaptive depth (--dflash-adapt on)" if use else "drafter measured but not kept")


# ---------------------------------------------------------------- max ctx
def start_args(c: Ctx, ctx: int, sessions: int) -> list:
    c.fresh_port()
    k, v = chosen_kv(c)
    pol, ddir = chosen_draft(c)
    return packmod.serve_flags(c.meta, port=c.args["port"], ctx=ctx, sessions=sessions, k_bits=k, v_bits=v, kv_mode=chosen_kv_mode(c), kvwin=chosen_kvwin(c),
                               draft=pol, dflash_dir=ddir)


def probe_factory(c: Ctx, sessions: int):
    cl = load.Client(c.args["port"])
    state = {"cal": False}

    def probe(ctx: int):
        with c.locks(f"maxctx s{sessions} c{ctx}"):            # one lock take per probe: a probe is a load + 2 near-full prompts (+ rounds)
            return probe_locked(ctx)

    def probe_locked(ctx: int):
        log = c.out / f"srv_maxctx_s{sessions}_c{ctx}.log"
        with running_server(c.prof, c.repo, start_args(c, ctx, sessions), c.args["port"], log, logf=c.log, args=c.args) as (p, ok):
            if not ok:
                tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
                return False, "no start: " + tail[0][:160]
            cl.url = f"http://127.0.0.1:{c.args['port']}"
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
    with c.locks("maxctx free-memory read"):          # the free-memory bound is only meaningful while nobody else holds the device
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
    out["lock_hold_max_s"] = max((h for _, h in c.holds), default=None)
    return out


def served_ctx(c: Ctx) -> dict:
    """{sessions: -c} the lanes get: the soak-validated value where the soak ran, else the max-ctx search result."""
    mc = {k: v for k, v in (c.data.get("maxctx") or {}).items() if k.isdigit()}
    mc.update({k: v for k, v in ((c.data.get("soak") or {}).get("ctx") or {}).items() if v})
    return mc


def gpu_used_mib(c: Ctx):
    f = backends.gpu_free_mib(c.args["backend"])
    return None if f is None else round(f, 0)


def job_soak(c: Ctx) -> dict:
    """Mixed-load validation of each max-ctx result (drill 0 gap: synthetic probes passed, real agent load OOMed hours later). Per session count:
    serve at the search result, run `--soak-s` seconds of load.Client.soak; any error or OOM line -> -c x 0.92 (rounded to 2048), up to 4 steps.
    One lock take per attempt (a load + the soak seconds)."""
    c.need_ram()
    out = dict(ctx={}, runs={})
    mc = {k: v for k, v in (c.data.get("maxctx") or {}).items() if k.isdigit() and v}
    for s_ in c.args["sessions"]:
        ctx = mc.get(str(s_))
        if not ctx:
            continue
        for attempt in range(4):
            log = c.out / f"srv_soak_s{s_}_c{ctx}.log"
            with c.locks(f"soak s{s_} c{ctx}"):
                with running_server(c.prof, c.repo, start_args(c, ctx, s_), c.args["port"], log, logf=c.log, args=c.args) as (p, ok):
                    if not ok:
                        r = dict(ok=False, why="no start")
                    else:
                        cl = load.Client(c.args["port"], "qual"); cl.calibrate()
                        r = cl.soak(s_, ctx, c.args["soak_s"], vram=lambda: gpu_used_mib(c))
                        r["ok"] = not r["n_errors"] and not oom_in(log) and r["requests"] > 0
                        r["oom_log"] = oom_in(log)
            out["runs"][f"s{s_}_c{ctx}"] = {k: v for k, v in r.items() if k != "vram_curve"} | dict(free_mib_min=min((x for x in r.get("vram_curve", []) if x is not None), default=None))
            c.log(f"soak sessions={s_} -c {ctx}: {'ok' if r['ok'] else 'FAILED'} {out['runs'][f's{s_}_c{ctx}']}")
            if r["ok"]:
                out["ctx"][str(s_)] = ctx
                break
            ctx = int(ctx * 0.92) // 2048 * 2048
        else:
            out["ctx"][str(s_)] = None
    out["lock_hold_max_s"] = max((h for _, h in c.holds), default=None)
    return out


# ---------------------------------------------------------------- serve + smoke (one server load)
def job_serve(c: Ctx, do_serve: bool, do_smoke: bool) -> dict:
    c.need_ram()
    mc = served_ctx(c)
    sess = max(c.args["sessions"])
    ctx = mc.get(str(sess)) or max((v for k, v in mc.items() if k.isdigit() and v), default=None)
    if not ctx:
        ctx = c.args.get("ctx_cap") or 32768
        c.log(f"serve: no max-ctx result, using {ctx}")
    sess_eff = sess if mc.get(str(sess)) else 1
    import report
    ship_s, ship_c = report.served_config(c.args, mc)           # the smoke runs on the config that would ship (a real Hermes preamble is 15-30K)
    ship_s, ship_c = (ship_s, ship_c) if ship_c else (sess_eff, ctx)
    out = dict(serve={}, smoke=None)

    def on_server(label, ctx_, sess_, fn, light=False):
        """One lock take = one server load + one group of measurements (bounded hold: the lead's rule, no stage holds a GPU lock for long).
        light=True: a small model's untimed step (the Hermes smoke) takes no exclusive lock (Ctx.locks)."""
        with c.locks(label, light_ctx=ctx_ if light else None):
            log = c.out / f"srv_{label}.log"
            argv = start_args(c, ctx_, sess_)
            with running_server(c.prof, c.repo, argv, c.args["port"], log, logf=c.log, args=c.args) as (p, ok):
                if not ok:
                    raise RuntimeError(f"{label}: server did not become ready; see {log}")
                cl = load.Client(c.args["port"], "qual")
                cl.calibrate()
                return fn(cl, log), [c.prof["python"], *argv]

    if do_serve:
        def quick(cl, log):
            s = {"conformance": cl.conformance()}
            s["decode_short"] = [cl.stream(cl.prompt(300, i), 256) for i in range(3)]
            s["prefill_8k"] = [cl.fit(8000, i + 3, 1, topup=False) for i in range(3)]
            a, b = cl.sha_set(), cl.sha_set()
            s["sha"], s["sha_self_equal"] = a, [x["sha"] for x in a] == [x["sha"] for x in b]
            s["sha_repeat"] = [x["sha"] for x in b]
            s["sha_n_diff"] = sum(x["sha"] != y["sha"] for x, y in zip(a, b))
            s["oom"] = oom_in(log)
            for r in s["decode_short"] + s["prefill_8k"]:
                r.pop("t0", None); r.pop("first", None); r.pop("end", None)
            s["decode_tps_median"] = statistics.median(r["dec_tps"] for r in s["decode_short"] if r["dec_tps"])
            s["prefill_8k_tps"] = statistics.median(r["ptok"] / r["ttft"] for r in s["prefill_8k"] if r["ptok"] and r["ttft"])
            return s

        def deep(cl, log):
            curve, t = [], 2048
            while t <= int(ctx * 0.9):
                r = cl.fit(t, 7, 64, topup=False)
                curve.append(dict(prompt=r["ptok"], ttft=r["ttft"], prefill_tps=round(r["ptok"] / r["ttft"], 1) if r["ptok"] and r["ttft"] else None, dec_tps=r["dec_tps"], err=r["err"]))
                t *= 4
            return dict(curve=curve, agg=[cl.concurrent(n, min(4096, ctx // max(4, sess_eff) - 256), 128) for n in range(1, sess_eff + 1)], oom=oom_in(log))
        s, out["cmd"] = on_server("serve", ctx, sess_eff, quick)
        d, _ = on_server("serve_deep", ctx, sess_eff, deep)
        s.update(curve=d["curve"], agg=d["agg"], oom=s["oom"] or d["oom"])
        out["serve"] = s
    if do_smoke:                                        # always on the config that would ship, in its own load
        out["smoke"], out["smoke_cmd"] = on_server("smoke", ship_c, ship_s, lambda cl, log: run_smoke(c), light=True)
        if not do_serve:
            out["cmd"] = out["smoke_cmd"]
    out["lock_hold_max_s"] = max((h for _, h in c.holds), default=None)
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
        res = smoke.run_hermes_eval(ev, c.args["port"], "qual", tasks, f"qualify_{c.meta['name']}", keep=c.out)
        if res["total"]:
            return res
        c.log("smoke: hermes eval gave no result, falling back to the scripted tool loop: " + str(res.get("error"))[:200])
    runs = [smoke.run_scripted_agent(c.args["port"], "qual", think=t) for t in (False, True)]
    return dict(kind="scripted_agent", total=len(runs), passed=sum(r["passed"] for r in runs), runs=runs, note="hermes_opt/eval not available (lib/taskdefs.py missing): fallback")
