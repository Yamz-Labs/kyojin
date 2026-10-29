"""REPORT.md + qualify.json from the stage data. The lanes read qualify.json: serve.cmd / serve.kv / serve.ctx need no hand flags."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import rules

SCHEMA = 1


def assemble(meta: dict, args: dict, stages: dict, repo: Path) -> dict:
    """stages: name -> {status, wall_s, data}. Builds the qualify.json document."""
    d = {n: s.get("data") for n, s in stages.items() if s.get("status") == "done"}
    ctxs = {k: v for k, v in (d.get("maxctx") or {}).items() if k.isdigit()}
    ctxs.update({k: v for k, v in ((d.get("soak") or {}).get("ctx") or {}).items()})     # the soak-validated value replaces the search result
    sv = d.get("serve") or {}
    sm = d.get("smoke")
    dr = d.get("draft") or {}
    q = dict(autotune=d.get("autotune"), parity=d.get("parity"), kv=d.get("kv"), maxctx=ctxs or None, serve=sv or None, smoke=sm)
    kv = (d.get("kv") or {}).get("chosen") or [6, 6]
    try:
        git = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        git = None
    return dict(
        schema=SCHEMA, at=time.strftime("%F %T"), git=git, backend=args["backend"],
        pack={k: meta[k] for k in ("name", "arch", "family", "layers", "kv_layers", "kv_heads", "head_dim", "quant", "bpw", "head_bits", "index_sha", "license", "mtp_layers", "vision")},
        stages={n: dict(status=s.get("status"), wall_s=s.get("wall_s")) for n, s in stages.items()},
        serve=dict(cmd=(d.get("serve") or {}).get("cmd") or (d.get("smoke") or {}).get("cmd"), kv=kv, ctx=ctxs, draft_policy=dr.get("policy") or meta.get("draft_policy"),
                   dflash_dir=dr.get("dflash_dir"), template=meta["template"], lane=lane(meta, args, ctxs, kv, d), hermes=hermes(args, ctxs)),
        draft=dr or None, soak=d.get("soak"),
        tuning={k: (d.get("autotune") or {}).get(k) for k in ("tuning_dir", "files", "records", "plan_points", "wall_s", "skipped", "reason", "dense_lines")} if d.get("autotune") else None,
        quality=dict(parity=d.get("parity"), kv=d.get("kv")),
        speed={k: sv.get(k) for k in ("decode_tps_median", "prefill_8k_tps", "curve", "agg")} if sv else None,
        sha=sv.get("sha"), smoke=sm,
        verdict={**rules.verdict(q, args.get("gates")), "pass": rules.verdict(q, args.get("gates"))["pass_"]},
    )


LANE_ENV_CUDA = {"EXL3_CUDA_GRAPH": "1"}
LANE_MAX_TOKENS = 32768        # lane recipe (rtx1/SERVE.md): a reply budget so a runaway think cannot hold a session forever


MIN_SHARE = 24576      # tokens one Hermes session needs: the live preamble (SOUL + skills + toolsets) is 15-30K before any work (duel 25/10)


def served_config(args: dict, ctxs: dict) -> tuple:
    """(sessions, -c) of the lane: the largest measured session count whose share of -c holds a Hermes session (--min-share), else a single session
    at its max -c. Drill 0 picked this by hand (3 x 24K at -c 73,728, then 2 x ... after the OOMs)."""
    share = args.get("min_share") or MIN_SHARE
    ok = [(int(k), v) for k, v in ctxs.items() if v and int(k) in args["sessions"]]
    for s, c in sorted(ok, reverse=True):
        if c // s >= share:
            return s, c
    one = [(s, c) for s, c in ok if s == 1]
    return one[0] if one else (None, None)


def lane(meta: dict, args: dict, ctxs: dict, kv: list, d: dict) -> dict | None:
    """The command to ship, with the lane-level choices that are not measured per pack (prefix disk, CUDA graphs, reply budget)."""
    sess, ctx = served_config(args, ctxs)
    if not ctx:
        return None
    import pack as packmod
    dr = d.get("draft") or {}
    kvd = d.get("kv") or {}
    argv = packmod.serve_flags(meta, port=8000, ctx=ctx, sessions=sess, k_bits=kv[0], v_bits=kv[1], draft=dr.get("policy"), dflash_dir=dr.get("dflash_dir"),
                               kv_mode=kvd.get("mode", "quant"), kvwin=kvd.get("kvwin")) + ["--default-max-tokens", str(LANE_MAX_TOKENS)]
    env = {**(LANE_ENV_CUDA if args["backend"] == "cuda" else {}), "EXL3_PREFIX_DISK_DIR": f"~/cache/prefix-{meta['name']}"}
    py = (d.get("serve") or {}).get("cmd", [None])[0] or "python"
    repo = str(args.get("repo") or ".")
    return dict(sessions=sess, ctx=ctx, env=env, argv=argv, python=py, repo=repo,
                cmd=f"cd {repo} && env PYTHONPATH={repo} " + " ".join(f"{k}={v}" for k, v in env.items()) + f" {py} -u " + " ".join(argv))


def hermes(args: dict, ctxs: dict) -> dict | None:
    """Hermes provider block for the lane (drill 0: context_length = the lane's real -c, auto-compaction BEFORE the server limit)."""
    sess, ctx = served_config(args, ctxs)
    if not ctx:
        return None
    share = ctx // sess
    return dict(model_context_length=ctx, per_session_share=share, compression_threshold=round(min(0.9, 0.8 / sess), 2),
                note="model.context_length = -c; compression.threshold = 0.8 of one session's share of -c (0.9 max); aux models stay on cloud")


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(j: dict) -> str:
    p, v = j["pack"], j["verdict"]
    L = [f"# qualify {p['name']} on {j['backend']} — {'PASS' if v['pass'] else 'NOT PASSED'}", "",
         f"{p['arch']} ({p['family']}), {p['layers']} layers ({p['kv_layers']} with KV), {p['quant']} {p['bpw']} bpw, git {j['git']}, {j['at']}", ""]
    if v["reasons"]:
        L += ["Failed gates: " + "; ".join(v["reasons"])]
    if v.get("warnings"):
        L += ["Warnings: " + "; ".join(v["warnings"])]
    if v["unmeasured"]:
        L += ["Unmeasured (not counted as pass): " + ", ".join(v["unmeasured"])]
    L += ["", "## Stages", "| stage | status | wall s |", "|---|---|---|"] + [f"| {n} | {s['status']} | {fmt(s['wall_s'], 0)} |" for n, s in j["stages"].items()]
    if j.get("tuning"):
        t = j["tuning"]
        L += ["", "## Day-0 autotune"]
        if t.get("skipped"):
            L += [f"skipped: {t.get('reason')}"]
    if j.get("tuning") and not j["tuning"].get("skipped"):
        t = j["tuning"]
        L += [f"{t['records']} GEMM tuning records + {t.get('dense_lines') or 0} dense-GEMM lines ({t['plan_points']} sweep points, {t['wall_s']} s on the device, lock wait excluded) in {t['tuning_dir']}: " + ", ".join(t["files"] or [])]
    par = j["quality"]["parity"]
    if par:
        L += ["", "## Parity vs streamed BF16", f"mean KL {par['kl_mean']:.4f}, top-1 {par['top1_mean']:.4f}, dNLL {par['dnll_mean']:+.4f}; SQNR: {par['sqnr']}", "",
              "| domain | KL | top-1 | dNLL |", "|---|---|---|---|"] + [f"| {d} | {x['kl']:.4f} | {x['top1']:.4f} | {x['dnll']:+.4f} |" for d, x in par["by_domain"].items()]
    kv = j["quality"]["kv"]
    if kv:
        L += ["", f"## KV (chosen k{kv['chosen'][0]}v{kv['chosen'][1]}; {kv['rule']}, kl_max {kv['kl_max']})"]
        if kv["table"]:
            L += ["| bits | KL vs ref | top-1 vs ref | KL vs k8v8 |", "|---|---|---|---|"]
            L += [f"| k{b}v{b} | {x['kl_ref']:.4f} | {x['top1_ref']:.4f} | {fmt(x.get('kl_vs_k8'), 4)} |" for b, x in kv["table"].items()]
        if kv.get("modes"):
            L += ["", f"KV format (decode-faithful 32K, KL vs k8v8; Q6 rule: kw6 if KL <= 0.9 x k6v6, tq opt-in): **{kv.get('mode', 'quant')}{' + ring ' + kv['kvwin'] if kv.get('kvwin') else ''}**",
                  "| format | KL vs k8v8 | top-1 vs k8v8 | bits/value | decode s/tok |", "|---|---|---|---|---|"]
            L += [f"| {n} | {x['kl_vs_k8']:.5f} | {x['top1_vs_k8']:.4f} | {x['bits_per_val']:.2f} | {fmt(x.get('decode_s_per_tok'), 4)} |" for n, x in kv["modes"].items()]
    dr = j.get("draft")
    if dr:
        L += ["", f"## Draft path: **{dr['policy']}**" + (f" ({dr['dflash_dir']})" if dr.get("dflash_dir") else ""), f"{dr.get('reason')}" + (f"; speedup drafted/plain {dr['speedup']} (ABBA, warm-up dropped)" if dr.get("speedup") else "")]
        if dr.get("notes"):
            L += ["rejected candidates: " + " | ".join(dr["notes"])]
    if j["serve"]["ctx"]:
        L += ["", "## Max context (k%dv%d, no OOM on 2 distinct near-full prompts)" % tuple(j["serve"]["kv"]), "| sessions | max -c |", "|---|---|"] + [f"| {s} | {c} |" for s, c in j["serve"]["ctx"].items()]
    sk = j.get("soak")
    if sk:
        L += ["", "## Soak (mixed agent-shaped load at the max-ctx result; -c steps down 8 % on any error)"] + [f"- {k}: {v}" for k, v in sk["runs"].items()]
    sp = j["speed"]
    if sp:
        L += ["", f"## Speed: decode {fmt(sp['decode_tps_median'],1)} tok/s, prefill 8K {fmt(sp['prefill_8k_tps'],0)} tok/s", "", "| prompt tok | ttft s | prefill tok/s | decode tok/s |", "|---|---|---|---|"]
        L += [f"| {r['prompt']} | {r['ttft']} | {fmt(r['prefill_tps'],0)} | {fmt(r['dec_tps'],1)} |" for r in sp["curve"]]
        L += ["", "| N | agg decode tok/s | per user | ttft s |", "|---|---|---|---|"] + [f"| {a['n']} | {fmt(a['agg_dec_tps'],1)} | {a['dec_tps']} | {a['ttft']} |" for a in sp["agg"]]
    if j["sha"]:
        L += ["", "Greedy sha: " + ", ".join(x["sha"] for x in j["sha"])]
    if j["smoke"]:
        L += ["", f"## Hermes smoke ({j['smoke']['kind']}): {j['smoke']['passed']}/{j['smoke']['total']} passed" + (f", mean score {j['smoke']['score_mean']}" if j['smoke'].get('score_mean') is not None else "")]
        L += [f"- {t.get('task') or t.get('id')}: score {t.get('score')}, {t.get('wall_s')} s, {t.get('tool_calls')} tool calls, {t.get('notes')}" for t in j["smoke"].get("tasks") or []]
    ln = j["serve"].get("lane")
    if ln:
        L += ["", f"## Ship this (sessions {ln['sessions']}, -c {ln['ctx']}, kv k{j['serve']['kv'][0]}v{j['serve']['kv'][1]}, draft {j['serve']['draft_policy']})", "```", ln["cmd"], "```"]
    h = j["serve"].get("hermes")
    if h:
        L += ["", f"Hermes: `model.context_length: {h['model_context_length']}`, `compression.threshold: {h['compression_threshold']}` ({h['note']})"]
    L += ["", "## Serve command as measured (qualify)", "`" + " ".join(j["serve"]["cmd"] or []) + "`"]
    return "\n".join(L) + "\n"


def write(out: Path, j: dict) -> None:
    (out / "qualify.json").write_text(json.dumps(j, indent=1, default=str))
    (out / "REPORT.md").write_text(render_md(j))
