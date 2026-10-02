# Decode-path A/B (ported from glm-base scratch/r38_ab.py, r39): one load, arms = env sets
# (graphs purged per switch, so call-time knobs are re-captured), prompts shared across arms and
# interleaved. Per depth x rep (rep < R38_REPS): greedy NDEC-token decode (t/s + ids/logits vs
# arm 0). At depth TF_DEPTH, for rep < TF_REPS: teacher-forced NNLL-token continuation
# (decode-path NLL, KL and argmax flips vs arm 0).
# Usage: python tools/glm/dec_tf_ab.py <out.json> '<arms json>' [depths csv]
import gc, json, math, os, statistics, sys
out = sys.argv[1]
_argv = list(sys.argv)
sys.argv = [sys.argv[0], "fast", "-o", out, "--prefill-depths", "4096", "--ppl-rows", "1"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import glm_base as G

ARMS = json.loads(_argv[2])
DEPTHS = [int(x) for x in (_argv[3] if len(_argv) > 3 else "4096,32768").split(",")]
KNOBS = sorted({k for v in ARMS.values() for k in v})
REPS, NDEC, NNLL = int(os.environ.get("R38_REPS", 3)), int(os.environ.get("R38_NDEC", 128)), int(os.environ.get("R38_NNLL", 128))
TF_DEPTH, TF_REPS = int(os.environ.get("TF_DEPTH", DEPTHS[0])), int(os.environ.get("TF_REPS", REPS))
NREP = max(REPS, TF_REPS)


def main():
    torch.set_grad_enabled(False)
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.modules import block_graph as BG
    args = G.args
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=max(DEPTHS) + 2048)
    model.load(device="cuda:0", progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    from exllamav3.modules.attention_fn.bc_mla import BCMLA

    R = {"arms": ARMS, "smoke": {}, "decode": {}, "nll": {}, "ids_eq": {}, "logit_md": {}, "tf_kl": {}, "tf_flips": {}}

    def save():
        with open(out + ".tmp", "w") as f:
            json.dump(R, f, indent=1)
        os.replace(out + ".tmp", out)

    def set_arm(a):
        torch.cuda.synchronize()
        for k in KNOBS: os.environ.pop(k, None)
        for k, v in ARMS[a].items(): os.environ[k] = str(v)
        BG.purge()
        mods, seen, st = [], set(), list(model.modules)
        while st:
            m = st.pop()
            if id(m) in seen: continue
            seen.add(id(m)); mods.append(m)
            st.extend(getattr(m, "modules", None) or [])
        ctrls = [v for m in mods for k, v in (getattr(m, "dispatch_cache", None) or {}).items()
                 if isinstance(k, tuple) and k and k[0] == "bcm" and v]
        for c in ctrls: c.configured.clear()
        print(f"[init] arm {a}: walked {len(mods)} modules, cleared {len(ctrls)} BCMLA", flush=True)

    cids = G.corpus_ids(tok, NREP * sum(DEPTHS) + NREP * len(DEPTHS) * (NNLL + 64) + 4096)
    cur = [0]

    def nxt(n):
        s = cids[:, cur[0]: cur[0] + n].contiguous()
        cur[0] += n + 37
        return s

    def cmp(ref, r):
        bi, bl = ref
        n = min(len(bi), len(r["ids"]))
        eq = torch.equal(bi[:n], r["ids"][:n]) and len(bi) == len(r["ids"])
        m = min(bl.shape[0], r["logits"].shape[0])
        la, lb = bl[:m].float(), r["logits"][:m].float()
        fin = torch.isfinite(la) & torch.isfinite(lb)
        md = (la[fin] - lb[fin]).abs().max().item() if fin.any() else 0.0
        return eq, (md if torch.equal(torch.isfinite(la), torch.isfinite(lb)) else float("inf"))

    A0 = next(iter(ARMS))
    print(f"RULE pass iff paired dNLL <= +0.1 % and dNLL <= 2*SE and dPPL <= +0.1 % (vs {A0}); "
          f"combined arm must pass too. arms={ARMS}", flush=True)
    # short chat smoke (dense regime) per arm
    base = {}
    for a in ARMS:
        set_arm(a)
        G.run_job(gen, nxt(2200), 8, stop=False)
        ent = []
        for i, p in enumerate(G.PROMPTS):
            r = G.run_job(gen, G.chat(tok, p), args.smoke_tokens, stop=False, return_logits=True)
            if a == A0: base[i] = (r["ids"], r["logits"]); ent.append([True, 0.0])
            else: ent.append(list(cmp(base[i], r)))
        R["smoke"][a] = ent
        print(f"[smoke {a}] " + " ".join(f"eq={e[0]} md={e[1]:.3g}" for e in ent), flush=True)
        save()

    for rep in range(NREP):
        for d in DEPTHS:
            do_dec, do_tf = rep < REPS, d == TF_DEPTH and rep < TF_REPS
            if not (do_dec or do_tf): continue
            prompt = nxt(d)
            cont = nxt(NNLL)
            ref = None
            for a in ARMS:
                set_arm(a)
                G.run_job(gen, prompt[:, :2100], 4, stop=False)          # regime-1 graph capture
                tps, eq, md, nll, kl, fl = float("nan"), None, float("nan"), float("nan"), float("nan"), -1
                if do_dec:
                    r = G.run_job(gen, prompt, NDEC, stop=False, return_logits=True)
                    tps = (r["ntok"] - 1) / (r["total"] - r["ttft"])
                    R["decode"].setdefault(f"{a}@{d}", []).append(tps)
                    if a == A0: ref = (r["ids"], r["logits"]); eq, md = True, 0.0
                    else: eq, md = cmp(ref, r)
                    R["ids_eq"].setdefault(f"{a}@{d}", []).append(eq)
                    R["logit_md"].setdefault(f"{a}@{d}", []).append(md)
                    del r
                if not do_tf:
                    print(f"[{a}@{d}] rep{rep}: {tps:.3f} t/s ids_eq={eq} md={md:.3g}", flush=True)
                    save(); continue
                f = G.run_job(gen, prompt, NNLL, stop=False, return_logits=True, constrain=cont[0])
                lg = f["logits"][:NNLL].float()
                nll = -torch.log_softmax(lg, -1).gather(-1, cont[0, :lg.shape[0], None].to(lg.device)).mean().item()
                R["nll"].setdefault(f"{a}@{d}", []).append(nll)
                lp = torch.log_softmax(lg, -1).cpu()
                if a == A0: tref = lp; kl, fl = 0.0, 0
                else:
                    kl = sum(torch.nn.functional.kl_div(lp[i:i + 256], tref[i:i + 256], log_target = True,
                             reduction = "sum").item() for i in range(0, lp.shape[0], 256)) / lp.shape[0]
                    fl = int((lp.argmax(-1) != tref.argmax(-1)).sum())
                R["tf_kl"].setdefault(f"{a}@{d}", []).append(kl)
                R["tf_flips"].setdefault(f"{a}@{d}", []).append(fl)
                del lp, lg
                print(f"[{a}@{d}] rep{rep}: {tps:.3f} t/s ids_eq={eq} md={md:.3g} nll={nll:.5f} kl={kl:.3g} flips={fl}", flush=True)
                save()
    R["median"] = {k: statistics.median(v) for k, v in R["decode"].items()}
    R["ms_per_tok"] = {k: 1000.0 / v for k, v in R["median"].items()}
    R["ppl_dec"] = {k: math.exp(statistics.mean(v)) for k, v in R["nll"].items()}
    k0 = f"{A0}@{TF_DEPTH}"
    R["tf_paired"] = {}
    for a in ARMS:
        k = f"{a}@{TF_DEPTH}"
        if a == A0 or k not in R["nll"]: continue
        dl = [(x - y) / y * 100.0 for x, y in zip(R["nll"][k], R["nll"][k0])]
        n = len(dl)
        se = statistics.stdev(dl) / math.sqrt(n) if n > 1 else float("nan")
        kls, fls = R["tf_kl"][k], R["tf_flips"][k]
        kse = statistics.stdev(kls) / math.sqrt(n) if n > 1 else float("nan")
        R["tf_paired"][a] = {"n": n, "dnll_pct": statistics.mean(dl), "se_pct": se,
                             "kl": statistics.mean(kls), "kl_se": kse, "flips": statistics.mean(fls),
                             "dppl_pct": (math.exp(statistics.mean(R["nll"][k]) - statistics.mean(R["nll"][k0])) - 1) * 100.0}
        print(f"RESULT tf {a} vs {A0}: dNLL {statistics.mean(dl):+.3f} +- {se:.3f} % KL {statistics.mean(kls):.5f} "
              f"+- {kse:.5f} flips {statistics.mean(fls):.1f}/{NNLL} dPPL {R['tf_paired'][a]['dppl_pct']:+.3f} % n={n}", flush=True)
    save()
    for k in R["median"]:
        print(f"RESULT {k:14s} {R['median'][k]:.3f} t/s {R['ms_per_tok'][k]:.2f} ms/tok ppl_dec {R['ppl_dec'].get(k, float('nan')):.5f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
