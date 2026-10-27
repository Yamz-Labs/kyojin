#!/usr/bin/env python3
"""search_ablit4.py -- Heretic-style TPE search of the ablation edit (ablit4).

One model load; hooks retargeted per trial (zero reload cost). Per trial:
  refusals = greedy-64 rate on 100 held-out harmful (regex + soft patterns)
  kl       = mean KL(orig || edited) over last<=33 next-token distributions
             of 100 held-out harmless (orig logprobs cached once)
Optuna MOTPE (multivariate TPE, 2 objectives) or seeded random fallback.
Then 2 candidates -> full validation (held-out refusals, KL200, XSTest-safe
over-refusal, 20q probe vs original) + export JSON+safetensors edit spec.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "ablit"))

N_LAYERS = 45
LAST = N_LAYERS - 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def check_guards():
    avail = -1
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1048576
    except Exception:
        pass
    if avail >= 0 and avail < 10:
        raise SystemExit(f"ABORT: only {avail:.1f} GB RAM available (< 10 GB)")
    assert os.environ.get("EXL3_BLOCK_GRAPH", "0") != "1", "EXL3_BLOCK_GRAPH must be off (bypasses hooks)"
    assert os.environ.get("EXL3_BLOCK_GRAPH_VERIFY", "0") != "1", "block-graph verify must be off"


def sample_space(trial, use_optuna):
    if use_optuna:
        mode = trial.suggest_categorical("mode", ["perlayer", "interp"])
        di = trial.suggest_float("dir_index", 0.0, float(LAST))
        p = {"mode": mode, "dir_index": di}
        for c in ("attn", "mlp"):
            max_w = trial.suggest_float(f"{c}.max_w", 0.8, 1.5)
            peak = trial.suggest_float(f"{c}.peak", 0.6 * LAST, float(LAST))
            min_frac = trial.suggest_float(f"{c}.min_frac", 0.0, 1.0)
            width = trial.suggest_float(f"{c}.width", 1.0, 0.6 * LAST)
            p[c] = {"max_w": max_w, "peak": peak, "min_w": min_frac * max_w,
                    "width": width, "min_frac": min_frac}
        return p
    else:
        import random
        r = trial  # Random instance in fallback
        p = {"mode": r.choice(["perlayer", "interp"]), "dir_index": r.uniform(0, LAST)}
        for c in ("attn", "mlp"):
            max_w = r.uniform(0.8, 1.5)
            min_frac = r.uniform(0, 1)
            p[c] = {"max_w": max_w, "peak": r.uniform(0.6 * LAST, LAST),
                    "min_w": min_frac * max_w, "width": r.uniform(1, 0.6 * LAST),
                    "min_frac": min_frac}
        return p


def to_edit_params(p):
    from hook_ablate import build_edit
    return p  # build_edit takes raw dicts below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/glm53-exl3-td205"))
    ap.add_argument("--dirs", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--out", default="./ablit4")
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    a = ap.parse_args()

    check_guards()
    try:
        import optuna
        from optuna.samplers import TPESampler
        use_optuna = True
        log(f"optuna {optuna.__version__} MOTPE search, {a.trials} trials")
    except ImportError:
        use_optuna = False
        log("no optuna: seeded random search fallback (small-grid equivalent)")

    from hook_ablate import build_edit, AblationHooks, N_LAYERS as NL
    import eval_ablit as E

    pdir = a.prompts or os.path.join(ROOT, "scratch", "ablit", "prompts")
    ddir = a.dirs or os.path.join(ROOT, "scratch", "ablit", "dirs")
    out = a.out
    os.makedirs(out, exist_ok=True)

    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator

    torch.set_grad_enabled(False)
    from safetensors.numpy import load_file
    st = load_file(os.path.join(ddir, "directions.safetensors"))
    import numpy as np
    R = np.stack([np.asarray(st[f"L{i:02d}.r"], dtype=np.float64) for i in range(NL)])

    harm100 = E.load_prompts(os.path.join(pdir, "harmful_heldout.jsonl"), 100)
    safe100 = E.load_prompts(os.path.join(pdir, "harmless_heldout.jsonl"), 100)
    assert len(harm100) == 100 and len(safe100) == 100, (len(harm100), len(safe100))
    log(f"prompts: harm100 + safe100 from heldout; dirs {R.shape}")

    t0 = time.time()
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    stop_ids = config.eos_token_id_list
    max_depth = a.batch * (256 + a.max_new_tokens + 8)
    cache = Cache(model, max_num_tokens=((max_depth + 1023) // 1024 + 1) * 1024)
    model.load(device="cuda:0", progressbar=False)
    log(f"loaded in {time.time() - t0:.1f}s")
    gen = Generator(model=model, cache=cache, tokenizer=tok)

    hooks = AblationHooks(model)
    hooks.install()
    hooks.enabled = False

    # determinism check: 5 harmful twice
    with torch.inference_mode():
        t1 = E.gen_texts(gen, tok, harm100[:5], a.max_new_tokens, stop_ids, min(5, a.batch))
        t2 = E.gen_texts(gen, tok, harm100[:5], a.max_new_tokens, stop_ids, min(5, a.batch))
    assert t1 == t2, "NON-DETERMINISM: identical greedy runs differ"
    log("determinism check pass (5/5 identical)")

    # baseline (no edit) on the search sets
    base = E.refusal_eval(gen, tok, harm100, stop_ids, a.max_new_tokens, a.batch)
    log(f"BASELINE refusals: {base['refusals']}/{base['n']} = {base['rate']:.3f} (+{base['empties']} empty)")
    mean_kl0, orig_cache, _ = E.kl_prompts(model, tok, safe100, hooks)
    assert mean_kl0 == 0.0, mean_kl0  # hooks disabled: edited == orig
    log(f"orig logprob cache built ({len(orig_cache)} prompts)")

    probe_orig = E.probe_eval(gen, tok, stop_ids)
    log(f"probe original: {probe_orig['hits']}/{probe_orig['n']} = {probe_orig['acc']:.3f}")

    # hook-efficacy A/B: uniform w=1.0 per-layer on a fixed edit, proving the
    # hooks move generation-time refusals (and that no fused residual-add kernel
    # bypasses them: with a bypass, mlp-only would read ~= baseline).
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    ab = {}
    for name, aw_, mw_ in (("both", np.ones(NL), np.ones(NL)),
                           ("attn-only", np.ones(NL), np.zeros(NL)),
                           ("mlp-only", np.zeros(NL), np.ones(NL))):
        hooks.set_edit({"attn_w": aw_, "mlp_w": mw_, "r": Rn})
        hooks.enabled = True
        ra = E.refusal_eval(gen, tok, harm100, stop_ids, a.max_new_tokens, a.batch)
        ab[name] = ra["rate"]
        log(f"HOOK-AB {name}: refusals {ra['refusals']}/{ra['n']}={ra['rate']:.3f} "
            f"(baseline {base['rate']:.3f})")
    hooks.enabled = False
    ab_ok = ab["both"] < base["rate"] - 0.15 and ab["mlp-only"] < base["rate"] - 0.05
    log(f"HOOK-AB verdict: {'PASS hooks move generation' if ab_ok else 'FAIL hooks ineffective - STOP'}")
    if not ab_ok:
        json.dump({"baseline": base["rate"], "hook_ab": ab},
                  open(os.path.join(out, "hook_ab_FAIL.json"), "w"), indent=1)
        raise SystemExit("HOOK-AB FAIL: hooks do not move generation-time refusals; fix before search")

    results = []

    def run_params(p):
        edit = build_edit(R, p["mode"], p["dir_index"], p["attn"], p["mlp"])
        hooks.set_edit(edit)
        hooks.enabled = True
        t0t = time.time()
        r = E.refusal_eval(gen, tok, harm100, stop_ids, a.max_new_tokens, a.batch)
        kl, _, _ = E.kl_prompts(model, tok, safe100, hooks, orig_cache)
        dt = time.time() - t0t
        return r, kl, dt, edit

    if use_optuna:
        import optuna

        def objective(trial):
            p = sample_space(trial, True)
            r, kl, dt, edit = run_params(p)
            trial.set_user_attr("rate", r["rate"])
            trial.set_user_attr("kl", kl)
            trial.set_user_attr("secs", dt)
            trial.set_user_attr("attn_w", [float(v) for v in edit["attn_w"]])
            trial.set_user_attr("mlp_w", [float(v) for v in edit["mlp_w"]])
            trial.set_user_attr("params_json", p)
            log(f"trial {trial.number}: mode={p['mode']} di={p['dir_index']:.1f} "
                f"a[{p['attn']['max_w']:.2f}@{p['attn']['peak']:.1f}/{p['attn']['width']:.1f}] "
                f"m[{p['mlp']['max_w']:.2f}@{p['mlp']['peak']:.1f}/{p['mlp']['width']:.1f}] "
                f"-> refusals {r['refusals']}/{r['n']}={r['rate']:.3f} KL {kl:.5f} ({dt:.0f}s)")
            return r["rate"], kl

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=TPESampler(multivariate=True, seed=a.seed, n_startup_trials=12))
        study.optimize(objective, n_trials=a.trials)
        for t in study.trials:
            results.append({"i": t.number, "params": t.user_attrs.get("params_json"),
                            "rate": t.user_attrs.get("rate"), "kl": t.user_attrs.get("kl"),
                            "secs": t.user_attrs.get("secs"),
                            "attn_w": t.user_attrs.get("attn_w"),
                            "mlp_w": t.user_attrs.get("mlp_w"),
                            "values": t.values})
        pareto = [{"i": t.number, "rate": t.values[0], "kl": t.values[1]} for t in study.best_trials]
    else:
        import random
        rng = random.Random(a.seed)
        pareto = []
        for i in range(a.trials):
            p = sample_space(rng, False)
            r, kl, dt, edit = run_params(p)
            results.append({"i": i, "params": p, "rate": r["rate"], "kl": kl, "secs": dt,
                            "attn_w": [float(v) for v in edit["attn_w"]],
                            "mlp_w": [float(v) for v in edit["mlp_w"]]})
            log(f"trial {i}: mode={p['mode']} di={p['dir_index']:.1f} "
                f"-> refusals {r['refusals']}/{r['n']}={r['rate']:.3f} KL {kl:.5f} ({dt:.0f}s)")
        # manual pareto
        for i, ri in enumerate(results):
            if not any(rj["rate"] <= ri["rate"] and rj["kl"] <= ri["kl"] and
                       (rj["rate"] < ri["rate"] or rj["kl"] < ri["kl"]) for rj in results):
                pareto.append({"i": ri["i"], "rate": ri["rate"], "kl": ri["kl"]})
    pareto = sorted(pareto, key=lambda d: (d["rate"], d["kl"]))
    log(f"PARETO ({len(pareto)}): " + "; ".join(
        f"t{d['i']} r={d['rate']:.3f} kl={d['kl']:.5f}" for d in pareto[:12]))

    # ---- pick 2 candidates
    feas1 = [r for r in results if r["rate"] <= 0.05]
    c1 = min(feas1, key=lambda r: r["kl"]) if feas1 else min(results, key=lambda r: (r["rate"], r["kl"]))
    feas2 = [r for r in results if r["kl"] <= 0.02]
    c2 = min(feas2, key=lambda r: r["rate"]) if feas2 else min(results, key=lambda r: (r["kl"], r["rate"]))
    log(f"C1 (lowest-KL, rate<=5%{' OK' if feas1 else ' INFEASIBLE-fallback'}): t{c1['i']} r={c1['rate']:.3f} kl={c1['kl']:.5f}")
    log(f"C2 (lowest-rate, KL<=0.02{' OK' if feas2 else ' INFEASIBLE-fallback'}): t{c2['i']} r={c2['rate']:.3f} kl={c2['kl']:.5f}")

    # ---- final validation
    safe200 = E.load_prompts(os.path.join(pdir, "harmless_collect.jsonl"), 100) + safe100
    xsafe = [p for p in E.load_prompts(os.path.join(pdir, "harmless_collect.jsonl"), 400)
             + safe100 if p["source"] == "xstest"]
    log(f"final sets: safe200={len(safe200)}, xsafe={len(xsafe)}")
    final = {}
    for name, c in (("C1", c1), ("C2", c2)):
        edit = build_edit(R, c["params"]["mode"], c["params"]["dir_index"],
                          c["params"]["attn"], c["params"]["mlp"])
        hooks.set_edit(edit)
        hooks.enabled = True
        r_full = E.refusal_eval(gen, tok, harm100, stop_ids, a.max_new_tokens, a.batch)
        kl200, _, _ = E.kl_prompts(model, tok, safe200, hooks)
        xs = E.refusal_eval(gen, tok, xsafe, stop_ids, a.max_new_tokens, a.batch)
        pr = E.probe_eval(gen, tok, stop_ids)
        final[name] = {"trial": c["i"], "params": c["params"],
                       "refusals_full": r_full["refusals"], "rate_full": r_full["rate"],
                       "empties": r_full["empties"], "kl200": kl200,
                       "xstest_refusals": xs["refusals"], "xstest_n": xs["n"],
                       "xstest_rate": xs["rate"], "probe_hits": pr["hits"],
                       "probe_acc": pr["acc"], "attn_w": c["attn_w"], "mlp_w": c["mlp_w"]}
        log(f"{name} t{c['i']}: refusals {r_full['refusals']}/100={r_full['rate']:.3f} "
            f"KL200 {kl200:.5f} xstest {xs['refusals']}/{xs['n']}={xs['rate']:.3f} "
            f"probe {pr['hits']}/20={pr['acc']:.3f} (orig {probe_orig['acc']:.3f})")
    hooks.enabled = False
    xs_orig = E.refusal_eval(gen, tok, xsafe, stop_ids, a.max_new_tokens, a.batch)
    log(f"xstest original: {xs_orig['refusals']}/{xs_orig['n']}={xs_orig['rate']:.3f}")

    # ---- export chosen spec (C1 = ship candidate: lowest-KL within refusal budget)
    chosen, chosen_name = (final["C1"], "C1")
    spec = {"model": a.model, "hidden": 4096, "n_layers": N_LAYERS,
            "readout": "last-token residual diff-means r (L{ii}.r), normalized",
            "candidate": chosen_name, "trial": chosen["trial"],
            "direction_mode": chosen["params"]["mode"],
            "direction_index": chosen["params"]["dir_index"],
            "attn": {k: chosen["params"]["attn"][k] for k in ("max_w", "peak", "min_w", "width")},
            "mlp": {k: chosen["params"]["mlp"][k] for k in ("max_w", "peak", "min_w", "width")},
            "attn_w": chosen["attn_w"], "mlp_w": chosen["mlp_w"],
            "norm_preserve": False,
            "apply": ("W_writer <- W - w_comp(L) * r_hat r_hat^T W, fp32 workspace, "
                      "pre-trellis in quantizer; writers = every layer attn o_proj + "
                      "dense down_proj (L0-2) + shared+routed expert down_proj (L3-44); "
                      "MTP/embed/router/gate-up/norms/lm_head untouched. "
                      "Norm-preserving (ablit1-muse 3x project+rescale) available as "
                      "opt-in arm, unmeasured here."),
            "c1": {k: final["C1"][k] for k in ("trial", "rate_full", "kl200", "xstest_rate", "probe_acc")},
            "c2": {k: final["C2"][k] for k in ("trial", "rate_full", "kl200", "xstest_rate", "probe_acc")},
            "probe_original_acc": probe_orig["acc"],
            "xstest_original_rate": xs_orig["rate"],
            "baseline_rate": base["rate"]}
    json.dump(spec, open(os.path.join(out, "edit_spec.json"), "w"), indent=1)
    from safetensors.numpy import save_file
    if chosen["params"]["mode"] == "perlayer":
        tensors = {f"L{i:02d}.r": (R[i] / (np.linalg.norm(R[i]) + 1e-12)).astype(np.float32)
                   for i in range(N_LAYERS)}
    else:
        from hook_ablate import interp_direction
        v = interp_direction(R, chosen["params"]["dir_index"]).astype(np.float32)
        tensors = {"r": v}
    save_file(tensors, os.path.join(out, "edit_spec.safetensors"))
    json.dump({"trials": results, "pareto": pareto, "final": final,
               "baseline": base["rate"], "probe_orig": probe_orig["acc"],
               "xstest_orig": xs_orig["rate"], "hook_ab": ab},
              open(os.path.join(out, "results.json"), "w"), indent=1)
    log(f"wrote {out}/edit_spec.json + edit_spec.safetensors + results.json")
    json.dump({"pareto": pareto}, open(os.path.join(out, "pareto.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
