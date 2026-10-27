"""The planner: pure functions, no GPU. build_plan(meta, backend, args, state) -> ordered Stage list.

Stages are resumable: <out>/<stage>.json with {"status": "done", "key": fingerprint, "wall_s": ..., "data": {...}}.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

# stage -> (job, needs, est_s, gpu). est_s = rough wall on a 3090 for a ~30B dense pack, from drill 0 (nanb.log timings).
STAGE_TABLE = {
    "fetch":   dict(job="cpu",    needs=[],                         est=60,    gpu=False),
    "meta":    dict(job="cpu",    needs=["fetch"],                  est=5,     gpu=False),
    "autotune": dict(job="autotune", needs=["meta"],                est=180,   gpu=True),
    "parity":  dict(job="parity", needs=["meta"],                   est=900,   gpu=True),
    "kv":      dict(job="kv",     needs=["meta", "parity"],         est=900,   gpu=True),
    "maxctx":  dict(job="maxctx", needs=["meta", "kv"],             est=1500,  gpu=True),
    "serve":   dict(job="serve",  needs=["meta", "kv", "maxctx"],   est=1500,  gpu=True),
    "smoke":   dict(job="serve",  needs=["serve"],                  est=900,   gpu=True),
    "report":  dict(job="cpu",    needs=["meta"],                   est=2,     gpu=False),
}
ORDER = list(STAGE_TABLE)


@dataclass
class Stage:
    name: str
    job: str
    needs: list
    est_s: int
    gpu: bool
    status: str = "todo"        # todo | done | skip (resumed) | blocked
    note: str = ""
    cmd: list = field(default_factory=list)


RELEVANT = {"fetch": ("ref",), "meta": (), "autotune": ("sessions", "ctx_cap"), "parity": ("ref",), "kv": ("ref", "kv_kl_max", "kv_bits", "kv_modes", "kvmode_steps"), "maxctx": ("sessions", "ctx_cap"),
            "serve": ("sessions", "ctx_cap"), "smoke": ("smoke_tasks",), "report": ()}


def fingerprint(stage: str, meta: dict, args: dict) -> str:
    """Input key of a stage: pack index sha, backend + the arguments that change THIS stage's result. A done-file with another key is stale."""
    rel = {k: args.get(k) for k in ("backend",) + RELEVANT[stage]}
    return hashlib.sha1(json.dumps([stage, meta.get("index_sha"), rel], sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_state(out: Path) -> dict:
    st = {}
    for s in ORDER:
        f = out / f"{s}.json"
        if f.exists():
            try:
                st[s] = json.loads(f.read_text())
            except ValueError:
                pass
    return st


def save_stage(out: Path, name: str, key: str, status: str, wall_s: float, data: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(dict(status=status, key=key, wall_s=round(wall_s, 1), data=data, at=time.strftime("%F %T")), indent=1, default=str))


def build_plan(meta: dict, args: dict, state: dict, only: list | None = None, redo: list | None = None, force_deps: bool = False) -> list[Stage]:
    """Plan = all stages in order, marked done (valid done-file, same key), todo, or blocked (a need is neither done nor selected)."""
    redo = set(redo or [])
    plan: list[Stage] = []
    for name in ORDER:
        t = STAGE_TABLE[name]
        st = Stage(name, t["job"], list(t["needs"]), t["est"], t["gpu"])
        if name == "autotune" and args.get("backend") not in ("cuda", "rocm"):
            st.status, st.note = "skip", f"no GEMM autotuner on backend {args.get('backend')}"
        elif name == "parity" and not args.get("ref"):
            st.status, st.note = "skip", "no --ref BF16 dir: parity unmeasured"
        elif name == "parity" and not meta.get("ref_adapter"):
            st.status, st.note = "skip", f"no reference adapter for family {meta.get('family')}: parity unmeasured"
        elif name == "kv" and not args.get("ref"):
            st.status, st.note = "skip", "kv KL needs the BF16 reference tail (--ref): falling back to k6v6 (Q6 rule)"
        elif name == "kv" and not meta.get("ref_adapter"):
            st.status, st.note = "skip", f"no reference adapter for family {meta.get('family')}: falling back to k6v6"
        d = state.get(name)
        if st.status == "todo" and d and d.get("status") == "done" and d.get("key") == fingerprint(name, meta, args) and name not in redo:
            st.status, st.note = "done", f"resumed ({d.get('wall_s')} s)"
        if name == "meta" and meta.get("problems"):
            st.note = "; ".join(meta["problems"])
        plan.append(st)
    if only:
        sel = set(only)
        by = {s.name: s for s in plan}
        for st in plan:
            if st.name not in sel and st.status == "todo":
                st.status, st.note = "unselected", "not selected"
        for st in plan:
            if st.name not in sel or st.status != "todo":
                continue
            for n in st.needs:
                nd = by[n]
                if nd.status == "unselected":
                    if force_deps:
                        nd.status, nd.note = "todo", "dependency forced"
                    else:
                        st.status, st.note = "blocked", f"needs {n} (not done; add it to --stage or use --force-deps)"
                        break
    return plan


def estimate(plan: list[Stage]) -> dict:
    tot = sum(s.est_s for s in plan if s.status == "todo")
    gpu = sum(s.est_s for s in plan if s.status == "todo" and s.gpu)
    return dict(total_s=tot, gpu_s=gpu)


def render_plan(plan: list[Stage], meta: dict, serve_cmd: list, args: dict) -> str:
    L = [f"pack {meta['name']}: {meta['arch']} ({meta['family']}), {meta['layers']} layers ({meta['kv_layers']} with KV), {meta['kv_heads']} KV heads x {meta['head_dim']}, "
         f"{meta['quant']} {meta['bpw']} bpw, weights {meta['weights_bytes']/2**30:.1f} GiB, mtp layers {meta['mtp_layers']}, vision {meta['vision']}"]
    L.append(f"backend {args['backend']}, out {args['out']}")
    for p in meta["problems"]:
        L.append(f"  PROBLEM: {p}")
    L.append("stage    job     gpu  status   est     note")
    for s in plan:
        L.append(f"{s.name:<8} {s.job:<7} {'yes' if s.gpu else 'no ':<4} {s.status:<8} {s.est_s//60:>3} min  {s.note}")
    e = estimate(plan)
    L.append(f"remaining estimate: {e['total_s']//60} min total, {e['gpu_s']//60} min GPU (drill-0 constants, not a promise)")
    L.append("serve command (ctx/KV placeholders are replaced by the kv and maxctx stages): " + " ".join(serve_cmd))
    return "\n".join(L)
