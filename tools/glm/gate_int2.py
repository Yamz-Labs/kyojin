#!/usr/bin/env python
"""glm-int gate (int2): combined tree (glm-next 164e90b + kda-dec 6c40939 + P0 tools) in ONE GLM load.
Arms: int = committed defaults; nodec = EXL3_KDA_DEC_CAT=0 EXL3_KDA_DEC_LR_BATCHED=0 (toggled in-process via
GDN.KDA_KNOBS). Per arm: smoke ids vs mla-prefill logs-r35/both.json (copy in tools/glm/data/r35_both.json);
speed reps interleaved (alternating order): 4K prefill + 128-token decode in one job, 16K prefill; unique
wiki.train.raw prompts (as glm-next scratch/next3). PPL 10 x 4096 cacheless on int. Paths follow $HOME (local).
Usage: gate_int2.py <out.json>"""
import json, os, statistics, sys, time
import torch

H = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
SPEED_CORPUS = f"{H}/bench/ppl/wikitext-2-raw/wiki.train.raw"
OUT = sys.argv[1]
sys.argv = ["glm_base", "fast", "-o", OUT + ".glm_base.json", "-m", f"{H}/models/glm53-exl3-td205",
            "--corpus", f"{H}/bench/ppl/wiki.test.raw"]
sys.path.insert(0, HERE)
import glm_base as G

ARMS = {"int": {"dec_cat": True, "lr_batched": True}, "nodec": {"dec_cat": False, "lr_batched": False}}
PPL_ARMS = [a for a in os.environ.get("GATE_PPL_ARMS", "int").split(",") if a in ARMS]
PPL_ROWS = int(os.environ.get("GATE_PPL_ROWS", "10"))
PPL_LEN = 4096
REPS = int(os.environ.get("GATE_REPS", "3"))
DEC_N = 128
REF35 = os.path.join(HERE, "data/r35_both.json")
RES = {"arms": ARMS, "ppl_rows": PPL_ROWS, "ppl_len": PPL_LEN, "res": {a: {} for a in ARMS}}


def save():
    with open(OUT + ".tmp", "w") as f:
        json.dump(RES, f, indent=1)
    os.replace(OUT + ".tmp", OUT)


def prefix(ref, got):
    return next((k for k, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))


@torch.inference_mode()
def main():
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.modules import block_graph as BG, gated_delta_net as GDN

    def set_arm(a):
        torch.cuda.synchronize()
        for k, v in ARMS[a].items():
            GDN.KDA_KNOBS[k] = v
        BG.purge()

    t0 = time.time()
    from exllamav3 import ext as E
    from exllamav3.ext import exllamav3_ext as X
    kn = {"kda." + k: v for k, v in GDN.KDA_KNOBS.items() if isinstance(v, (bool, int, float, str))}
    kn.update({"rocm." + k: v for k, v in E.ROCM_KNOBS.items()})
    kn.update({"bg." + k: v for k, v in BG.BG_KNOBS.items()})
    kn["dt_f32"] = GDN._KDA_DT_F32
    kn["has_skinny_cat"] = hasattr(X, "skinny_cat")
    kn.update({k: os.environ.get(k) for k in sorted(os.environ) if k.startswith("EXL3_")})
    RES["knobs_at_start"] = kn
    print("[knobs] " + " ".join(f"{k}={v}" for k, v in kn.items()), flush=True)
    config = Config.from_directory(G.args.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=20480)
    model.load(device="cuda:0", progressbar=False)
    RES["load_s"] = time.time() - t0
    print(f"loaded {RES['load_s']:.0f}s", flush=True)
    G.gtt_guard("after_load")
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    ref35 = json.load(open(REF35))["smoke"]

    for a in ARMS:
        set_arm(a)
        ts = time.time()
        sm = []
        for i, p in enumerate(G.PROMPTS):
            r = G.run_job(gen, G.chat(tok, p), 96, stop=False)
            got = r["ids"].tolist()
            e = {"text": r["text"][:200], "ids": got, "r35_prefix": prefix(ref35[i]["ids"], got)}
            if a != "int":
                e["int_prefix"] = prefix(RES["res"]["int"]["smoke"][i]["ids"], got)
            sm.append(e)
        RES["res"][a]["smoke"] = sm
        RES["res"][a]["smoke_s"] = time.time() - ts
        print(f"[smoke {a}] r35 " + "/".join(str(e["r35_prefix"]) for e in sm) +
              ("" if a == "int" else " vs int " + "/".join(str(e["int_prefix"]) for e in sm)) +
              f" ({RES['res'][a]['smoke_s']:.0f}s)", flush=True)
        save()

    bad = [a for a in ARMS if min(e["r35_prefix"] for e in RES["res"][a]["smoke"]) < 96]
    if bad and os.environ.get("GATE_FORCE") == "1":  # stale r35 ref: log, go on to speed/PPL
        print("[smoke] FAIL " + ",".join(bad) + " (GATE_FORCE=1, continuing)", flush=True)
        bad = []
    if bad:  # same-load smoke-only bisect by knob, then stop (no speed/PPL on a broken tree)
        BIS = {"cat_only": ({"dec_cat": True, "lr_batched": False}, {}, {}),
               "no_gnorm_f16g": ({}, {"gnorm_f16g": False}, {}),
               "no_f16gates": ({}, {}, {"EXL3_KDA_F16_GATES": "0", "EXL3_KDA_SMALL_FIRST": "0"}),
               "no_all": ({"dec_cat": False, "lr_batched": False}, {"gnorm_f16g": False},
                          {"EXL3_KDA_F16_GATES": "0", "EXL3_KDA_SMALL_FIRST": "0"})}
        for b, (kk, rk, env) in BIS.items():
            torch.cuda.synchronize()
            for k, v in {**ARMS["int"], **kk}.items():
                GDN.KDA_KNOBS[k] = v
            E.ROCM_KNOBS["gnorm_f16g"] = rk.get("gnorm_f16g", True)
            for k in ("EXL3_KDA_F16_GATES", "EXL3_KDA_SMALL_FIRST"):
                os.environ.pop(k, None)
            os.environ.update(env)
            BG.purge()
            pf = [prefix(ref35[i]["ids"], G.run_job(gen, G.chat(tok, p), 96, stop=False)["ids"].tolist())
                  for i, p in enumerate(G.PROMPTS)]
            RES.setdefault("bisect", {})[b] = pf
            print(f"[smoke bisect {b}] r35 " + "/".join(map(str, pf)), flush=True)
            save()
        print("DONE smoke FAIL " + ",".join(bad), flush=True)
        return

    cids = tok.encode(open(SPEED_CORPUS, encoding="utf-8").read()[:3_000_000], add_bos=False)
    cur = [1000]

    def nxt(n):
        assert cur[0] + n <= cids.shape[-1], ("corpus exhausted", cur[0], n)
        s = cids[:, cur[0]: cur[0] + n].contiguous()
        cur[0] += n + 37
        return s

    names = list(ARMS)
    for a in names:
        set_arm(a)
        G.run_job(gen, nxt(2048), 16, stop=False)
    ts = time.time()
    for i in range(REPS):
        for a in (names if i % 2 == 0 else names[::-1]):
            set_arm(a)
            G.run_job(gen, nxt(512), 8, stop=False)
            R = RES["res"][a]
            r = G.run_job(gen, nxt(4096), DEC_N, stop=False)
            R.setdefault("prefill4k", []).append(4096 / r["ttft"])
            R.setdefault("decode4k", []).append((r["ntok"] - 1) / (r["total"] - r["ttft"]))
            r = G.run_job(gen, nxt(16384), 1, stop=False)
            R.setdefault("prefill16k", []).append(16384 / r["ttft"])
            print(f"[speed {a}] rep{i}: pf4k {R['prefill4k'][-1]:.1f} dec4k {R['decode4k'][-1]:.3f} "
                  f"pf16k {R['prefill16k'][-1]:.1f}", flush=True)
            save()
    RES["speed_s"] = time.time() - ts
    for a in names:
        R = RES["res"][a]
        for m in ("prefill4k", "decode4k", "prefill16k"):
            R[m + "_median"] = statistics.median(R[m])
    save()
    for a in PPL_ARMS:
        set_arm(a)
        p = G.eval_ppl(model, tok, PPL_ROWS, PPL_LEN, a)
        RES["res"][a]["ppl"] = p
        print(f"[ppl {a}] {p['ppl']:.6f} ({p['seconds']:.0f}s)", flush=True)
        save()
    G.gtt_guard("after_ppl")
    RES["wall_s"] = time.time() - t0
    save()
    for a in names:
        R = RES["res"][a]
        print(f"SUMMARY {a}: ppl {R.get('ppl', {}).get('ppl', 0):.6f} pf4k {R['prefill4k_median']:.1f} "
              f"pf16k {R['prefill16k_median']:.1f} dec4k {R['decode4k_median']:.3f}", flush=True)
    print(f"DONE wall {RES['wall_s']:.0f}s", flush=True)


main()
