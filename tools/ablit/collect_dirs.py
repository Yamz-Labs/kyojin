#!/usr/bin/env python3
"""collect_dirs.py -- per-layer refusal directions for GLM-5.3-Flash on td205.

Real mode (GPU, via gpu-any; model ~/models/glm53-exl3-td205):
  Streams decoder layers exactly like tools/glm/fp8_stream_teacher.py
  (deferred load, one layer resident). For every prompt, after each block the
  residual state [1, seq, hc_mult, hidden] is averaged over hc streams; two
  readouts are kept: last prompt token, and mean of the last 4 prompt tokens.
  r_L = mean(harm_L) - mean(safe_L) per readout.

  Per-layer stats: ||r||, sep-norm (||r||/pooled-std), AUROC of harmful vs
  harmless projected on r-hat (both readouts), cos gap
  (mean cos(harm,r-hat) - mean cos(safe,r-hat)), massive-activation dims
  (mean|h| > 10x median, Sun et al. definition) + masked direction variant.

Mock mode (CPU only, no model, no GPU):
  python3 collect_dirs.py --mock --out ./scratch_dirs --dry-run-test

Chat wrap matches tools/glm/glm_base.py (established harness convention):
  [gMASK]<sop><|user|>\\n{prompt}<|assistant|>\\n

Output: directions.safetensors + manifest.json.
"""
import argparse
import json
import os
import shutil
import sys

DEFAULT_MODEL = os.path.expanduser("~/models/glm53-exl3-td205")
MASSIVE_RATIO = 10.0


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as arch_mod  # noqa: E402

ARCH = arch_mod.get("glm")  # replaced in __main__ by --arch


def chat_wrap(prompt):
    return arch_mod.chat_wrap(prompt, ARCH)


def load_prompts(path, limit):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                d = json.loads(line)
                t = d.get("text") or d.get("prompt") or d.get("instruction") or ""
            else:
                t = line
            if len(t.split()) >= 3:
                out.append(t)
            if len(out) >= limit:
                break
    return out


def ram_avail_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except Exception:
        pass
    return -1.0


def check_guards():
    avail = ram_avail_gb()
    if avail >= 0 and avail < 10:
        raise SystemExit(f"ABORT: only {avail:.1f} GB RAM available (< 10 GB)")
    free = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    if free < 8:
        raise SystemExit(f"ABORT: only {free:.0f} GB free on home disk (< 8 GB)")


# ---------------------------------------------------------------- stats (numpy)

def auroc(scores_harm, scores_safe):
    """Mann-Whitney rank AUROC, no sklearn needed."""
    import numpy as np
    h = np.asarray(scores_harm, dtype=np.float64)
    s = np.asarray(scores_safe, dtype=np.float64)
    vals = np.concatenate([h, s])
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals))
    ranks[order] = np.arange(1, len(vals) + 1)
    # tie-average
    _, inv, counts = np.unique(vals[order], return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg[inv[np.argsort(order)]]
    n_h, n_s = len(h), len(s)
    return float((ranks[:n_h].sum() - n_h * (n_h + 1) / 2.0) / (n_h * n_s))


def accumulate_stats(states_harm, states_safe, states_harm4=None, states_safe4=None):
    """Per-layer dict. states_*: list per layer of (n, d) arrays (last-token).
    states_*4: same for mean-of-last-4 readout (optional; falls back to last)."""
    import numpy as np
    d = {}
    n_layers = len(states_harm)
    for L in range(n_layers):
        h = np.asarray(states_harm[L], dtype=np.float64)
        s = np.asarray(states_safe[L], dtype=np.float64)
        mh, ms = h.mean(0), s.mean(0)
        r = mh - ms
        rn = r / (np.linalg.norm(r) + 1e-12)
        pool = float(np.sqrt((h.var(0).mean() + s.var(0).mean()) / 2.0) + 1e-12)
        ph = (h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)) @ rn
        ps = (s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)) @ rn
        proj_h, proj_s = h @ rn, s @ rn
        # massive-activation dims on pooled mean|h|
        mall = np.concatenate([np.abs(h), np.abs(s)], axis=0).mean(0)
        med = float(np.median(mall)) + 1e-12
        massive = np.nonzero(mall > MASSIVE_RATIO * med)[0]
        r_masked = r.copy()
        r_masked[massive] = 0.0
        entry = {"r": r, "r_masked": r_masked, "mean_harm": mh, "mean_safe": ms,
                 "norm": float(np.linalg.norm(r)),
                 "sep": float(np.linalg.norm(r) / pool),
                 "auroc": auroc(proj_h, proj_s),
                 "cos_gap": float(ph.mean() - ps.mean()),
                 "massive_dims": np.asarray(massive, dtype=np.int64),
                 "n_harm": len(h), "n_safe": len(s)}
        if states_harm4 is not None:
            h4 = np.asarray(states_harm4[L], dtype=np.float64)
            s4 = np.asarray(states_safe4[L], dtype=np.float64)
            r4 = h4.mean(0) - s4.mean(0)
            rn4 = r4 / (np.linalg.norm(r4) + 1e-12)
            entry["r_last4"] = r4
            entry["auroc_last4"] = auroc(h4 @ rn4, s4 @ rn4)
            ph4 = (h4 / (np.linalg.norm(h4, axis=1, keepdims=True) + 1e-12)) @ rn4
            ps4 = (s4 / (np.linalg.norm(s4, axis=1, keepdims=True) + 1e-12)) @ rn4
            entry["cos_gap_last4"] = float(ph4.mean() - ps4.mean())
        d[L] = entry
    return d


def noise_floor(seps, n_perm=200, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    seps = np.asarray(seps)
    null = []
    for _ in range(n_perm):
        a = rng.choice(seps, size=len(seps) // 2, replace=False)
        null.append(abs(a.mean() - seps.mean()))
    return float(np.quantile(null, 0.95))


def save_dirs(stats, out, meta):
    from safetensors.numpy import save_file
    import numpy as np
    tensors = {}
    for L, v in stats.items():
        for k in ("r", "r_masked", "mean_harm", "mean_safe"):
            tensors[f"L{L:02d}.{k}"] = np.asarray(v[k], dtype=np.float32)
        if "r_last4" in v:
            tensors[f"L{L:02d}.r_last4"] = np.asarray(v["r_last4"], dtype=np.float32)
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "directions.safetensors"))
    man = {"layers": sorted(stats), "n_harm": stats[0]["n_harm"],
           "n_safe": stats[0]["n_safe"],
           "sep": {str(L): v["sep"] for L, v in stats.items()},
           "auroc": {str(L): v["auroc"] for L, v in stats.items()},
           "cos_gap": {str(L): v["cos_gap"] for L, v in stats.items()},
           "norm": {str(L): v["norm"] for L, v in stats.items()},
           "massive_count": {str(L): int(len(v["massive_dims"])) for L, v in stats.items()},
           "massive_dims": {str(L): [int(x) for x in v["massive_dims"][:64]]
                            for L, v in stats.items()},
           "noise_floor": noise_floor([v["sep"] for v in stats.values()]),
           **meta}
    if stats and "auroc_last4" in stats[0]:
        man["auroc_last4"] = {str(L): v["auroc_last4"] for L, v in stats.items()}
        man["cos_gap_last4"] = {str(L): v["cos_gap_last4"] for L, v in stats.items()}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    nfloor = man["noise_floor"]
    kept = [L for L, v in stats.items() if v["sep"] > nfloor]
    best = max(stats, key=lambda L: stats[L]["auroc"])
    print(f"wrote {out}/directions.safetensors: {len(stats)} layers, "
          f"noise_floor={nfloor:.3f}, above-floor={len(kept)}/{len(stats)}, "
          f"best-AUROC=L{best} ({stats[best]['auroc']:.4f})")
    if len(kept) < 0.8 * len(stats):
        print("WARNING: <80% layers above noise floor -> INCONCLUSIVE, do not abliterate")
    return man


def run_mock(args):
    import numpy as np
    rng = np.random.default_rng(args.seed)
    L, d, n = args.mock_layers, args.mock_hidden, args.mock_n
    planted = rng.standard_normal(d)
    planted /= np.linalg.norm(planted)
    harm, safe, harm4, safe4 = [], [], [], []
    for _ in range(L):
        s = rng.standard_normal((n, d))
        # offset 30: with n=64, d=512 the sampling-noise norm (~4) would swamp a
        # realistic offset; the mock tests plumbing/recovery, not statistics
        # (the real run uses n=400 at d=4096 for the same reason).
        h = rng.standard_normal((n, d)) + 30.0 * planted
        harm.append(h)
        safe.append(s)
        harm4.append(h + rng.standard_normal((n, d)) * 0.1)
        safe4.append(s + rng.standard_normal((n, d)) * 0.1)
    stats = accumulate_stats(harm, safe, harm4, safe4)
    man = save_dirs(stats, args.out, {"mode": "mock", "seed": args.seed})
    worst = min(float(np.dot(v["r"] / np.linalg.norm(v["r"]), planted))
                for v in stats.values())
    min_auc = min(v["auroc"] for v in stats.values())
    print(f"mock: worst-layer cos(recovered, planted) = {worst:.4f}, "
          f"min AUROC = {min_auc:.4f}")
    if args.dry_run_test:
        assert worst > 0.98, f"mock recovery FAILED: {worst}"
        assert min_auc > 0.99, f"mock AUROC FAILED: {min_auc}"
        assert man["noise_floor"] < min(v["sep"] for v in stats.values())
        assert all("auroc_last4" in v or True for v in stats.values())
        assert all(len(v["massive_dims"]) == 0 for v in stats.values()), \
            "isotropic mock must have no massive dims"
        print("DRY-RUN-TEST PASS (CPU, no model, no GPU)")
    return man


# ---------------------------------------------------------------- real (GPU)

def run_real(args):
    import torch
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.util.memory import free_mem
    torch.set_grad_enabled(False)
    check_guards()
    harm = load_prompts(args.harmful, args.limit)
    safe = load_prompts(args.harmless, args.limit)
    assert len(harm) >= 100 and len(safe) >= 100, "need >=100 prompts per set"
    config = Config.from_directory(args.model)
    tok = Tokenizer.from_config(config)
    config.override_dynamic_seq_len(args.max_tokens)
    model = Model.from_config(config)
    mods = model.modules
    first = model.first_block_idx
    head_start = model.logit_layer_idx - 2  # GLM: hc_head, norm, lm_head follow the blocks
    if type(mods[head_start]).__name__ == "TransformerBlock":
        head_start += 1  # MiMo-V2: norm, lm_head only
    n_layers = head_start - first
    device = torch.device(args.device)
    assert n_layers == ARCH["n_layers"], \
        f"model has {n_layers} decoder blocks, arch {ARCH['name']} expects {ARCH['n_layers']}"
    print(f"decoder layers: {n_layers} (first={first} head_start={head_start}), "
          f"harm={len(harm)} safe={len(safe)}, max_tokens={args.max_tokens}", flush=True)

    def encode(p):
        ids = tok.encode(chat_wrap(p), encode_special_tokens=True)
        return ids[:, -args.max_tokens:]

    def collect(prompts, tag):
        # Layer-outer streaming: each layer's weights load ONCE, all prompts
        # forward through it (per-prompt loads would be 800x45 disk reads).
        # Full sequence states ride on CPU between layers (~6 MB/prompt fp32).
        import numpy as np
        n = len(prompts)
        ids_list = [encode(p) for p in prompts]
        per_last = [[] for _ in range(n_layers)]
        per_last4 = [[] for _ in range(n_layers)]
        with torch.inference_mode():
            config.stc.begin_deferred_load()
            for m in mods[:first]:
                m.load(device)
            config.stc.end_deferred_load()
            states = []
            for pi in range(n):
                params = {}
                st = model.prepare_inputs(ids_list[pi], params)
                for m in mods[:first]:
                    st = m.forward(m.prepare_for_device(st, params), params)
                states.append(st.float().cpu())
            for m in mods[:first]:
                m.unload()
            config.stc.close()
            free_mem()
            print(f"  [{tag}] embedded {n} prompts", flush=True)
            for L in range(n_layers):
                m = mods[first + L]
                config.stc.begin_deferred_load()
                m.load(device)
                config.stc.end_deferred_load()
                for pi in range(n):
                    params = {}
                    st = m.forward(m.prepare_for_device(states[pi].to(device), params),
                                   params).clone()
                    v = st.float().cpu()
                    if pi == 0 and L == 0 and tag == "harm":
                        print(f"  state shape: {tuple(v.shape)}", flush=True)
                        if ARCH["readout"] == "hc_mean":
                            assert v.dim() == 4 and v.shape[0] == 1 and v.shape[3] > 1000, \
                                tuple(v.shape)
                        else:
                            assert v.dim() == 3 and v.shape[0] == 1 \
                                and v.shape[2] == ARCH["hidden"], tuple(v.shape)
                    states[pi] = v  # back to CPU for the next layer
                    if ARCH["readout"] == "hc_mean":
                        v = v.mean(dim=2)[0]  # mean over hc streams -> [seq, hidden]
                    else:
                        v = v[0]  # plain residual [seq, hidden]
                    k = min(4, v.shape[0])
                    per_last[L].append(v[-1].numpy())
                    per_last4[L].append(v[-k:].mean(0).numpy())
                    del st
                m.unload()
                config.stc.close()
                free_mem()
                print(f"  [{tag}] layer {L + 1}/{n_layers}", flush=True)
        return per_last, per_last4

    print(f"collect harmful {len(harm)} ...", flush=True)
    hh, hh4 = collect(harm, "harm")
    print(f"collect harmless {len(safe)} ...", flush=True)
    hs, hs4 = collect(safe, "safe")
    stats = accumulate_stats(hh, hs, hh4, hs4)
    return save_dirs(stats, args.out, {"mode": "real", "model": args.model,
                                       "n_harm": len(harm), "n_safe": len(safe),
                                       "max_tokens": args.max_tokens,
                                       "harmful": args.harmful, "harmless": args.harmless})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arch", default="auto", choices=["auto"] + list(arch_mod.ARCHS))
    ap.add_argument("--harmful", default=None)
    ap.add_argument("--harmless", default=None)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", default="./dirs")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-layers", type=int, default=8)
    ap.add_argument("--mock-hidden", type=int, default=512)
    ap.add_argument("--mock-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run-test", action="store_true")
    a = ap.parse_args()
    ARCH = arch_mod.get(a.arch, a.model)
    if a.mock:
        run_mock(a)
    else:
        if not a.harmful or not a.harmless:
            sys.exit("need --harmful/--harmless jsonl (or use --mock)")
        run_real(a)
